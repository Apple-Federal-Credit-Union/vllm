# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-cache hits must not confirm another client's prompt.

Prefix caching is on by default. A block hash includes ``cache_salt`` only
when the request supplies one, and the salt is not mixed with any
server-side caller identity. These tests run that hash and lookup, then the
serving code that copies the hit into a public usage object.

An unsalted request must not report a positive cached-token count for a
prefix another unsalted request left in the cache. A request that repeats a
caller-supplied salt may still report that hit.

Streaming responses call ``responses_full_generator`` for the terminal usage
object, so that path covers them. Chat, completion, and token-in/token-out
streaming copy the count in their own generators; those are covered when
prompt token details are enabled. Cohere and Anthropic serving read the chat
usage object and are not called here.
"""

import json
from unittest.mock import Mock

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.entrypoints.generate.base.protocol import (
    RequestResponseMetadata,
    StreamOptions,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.decisions.protocol import DecisionRequest
from vllm.entrypoints.openai.decisions.serving import OpenAIServingDecisions
from vllm.entrypoints.openai.decisions.strategies import (
    DecisionLimits,
    NextTokenStrategy,
    ReadContext,
    ReadStrategy,
)
from vllm.entrypoints.openai.responses.context import (
    HarmonyContext,
    ParsableContext,
    SimpleContext,
    TurnMetrics,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.openai.responses.serving import OpenAIServingResponses
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.inputs import tokens_input
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.request import Request

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def _block_size() -> int:
    # ``None`` means the default, applied during construction.
    return CacheConfig().block_size


def _processor() -> InputProcessor:
    model_config = Mock(
        max_model_len=8192,
        max_logprobs=20,
        logits_processors=None,
        is_diffusion=False,
        return_sampling_mask=False,
        supports_multimodal_inputs=False,
        enable_trace_replay=False,
    )
    model_config.get_vocab_size.return_value = 32000
    model_config.try_get_generation_config.return_value = {}
    config = Mock(
        model_config=model_config,
        speculative_config=None,
        structured_outputs_config=None,
        lora_config=None,
        cache_config=Mock(kv_sharing_fast_prefill=False),
        scheduler_config=Mock(),
        observability_config=Mock(),
        diffusion_config=None,
        reasoning_config=None,
        watermark_config=None,
        use_v2_model_runner=False,
    )
    config._check_supports_watermarking.return_value = False
    config.parallel_config = Mock(
        data_parallel_size=1,
        data_parallel_size_local=1,
        local_engines_only=False,
    )
    renderer = Mock(tokenizer=None)
    renderer.get_eos_token_id.return_value = None
    return InputProcessor(config, renderer, mm_registry=Mock())


def _manager(block_size: int) -> KVCacheManager:
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    return KVCacheManager(
        KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=block_size,
        scheduler_block_size=block_size,
    )


def _prompt(n_blocks: int, tail: int, block_size: int) -> list[int]:
    token_ids: list[int] = []
    for block in range(n_blocks):
        token_ids.extend([1_000 + block] * block_size)
    token_ids.append(tail)
    return token_ids


def _engine_request(
    processor: InputProcessor,
    request_id: str,
    token_ids: list[int],
    cache_salt: str | None,
):
    engine_input = tokens_input(token_ids, cache_salt=cache_salt)
    if cache_salt is None:
        assert "cache_salt" not in engine_input
    request = processor.process_inputs(
        request_id,
        engine_input,
        SamplingParams(max_tokens=1, temperature=0.0, detokenize=False),
        ("generate",),
    )
    assert request.cache_salt == cache_salt
    assert request.prompt_token_ids == token_ids
    # The engine assigns the internal id before the output processor runs.
    InputProcessor.assign_request_id(request)
    return request


def _scheduler_request(engine_request, block_size: int) -> Request:
    return Request.from_engine_core_request(
        engine_request,
        get_request_block_hasher(block_size, sha256),
    )


def _fill_cache(
    manager: KVCacheManager,
    processor: InputProcessor,
    token_ids: list[int],
    cache_salt: str | None,
    block_size: int,
) -> None:
    """Store full blocks the way a finished request leaves them cached."""
    engine_request = _engine_request(processor, "victim", token_ids, cache_salt)
    request = _scheduler_request(engine_request, block_size)
    computed, num_cached, _ = manager.get_computed_blocks(request)
    assert num_cached == 0
    allocated = manager.allocate_slots(
        request,
        len(token_ids),
        len(computed.blocks[0]) * block_size,
        computed,
    )
    assert allocated is not None
    request.num_computed_tokens += len(token_ids)
    manager.free(request)


def _probe(
    manager: KVCacheManager,
    processor: InputProcessor,
    token_ids: list[int],
    cache_salt: str | None,
    block_size: int,
) -> RequestOutput:
    """Look up the prefix and publish it the way the scheduler and output
    processor do: ``get_computed_blocks`` into ``PrefillStats``, then
    ``OutputProcessor``."""
    engine_request = _engine_request(processor, "attacker", token_ids, cache_salt)
    request = _scheduler_request(engine_request, block_size)
    _computed, num_cached, _boundary = manager.get_computed_blocks(request)
    stats = PrefillStats()
    stats.set(
        num_prompt_tokens=request.num_prompt_tokens,
        num_local_cached_tokens=num_cached,
        num_external_cached_tokens=0,
    )
    processor_out = OutputProcessor(tokenizer=None, log_stats=False)
    processor_out.add_request(engine_request, prompt="")
    processed = processor_out.process_outputs(
        [
            EngineCoreOutput(
                request_id=engine_request.request_id,
                new_token_ids=[7],
                finish_reason=FinishReason.STOP,
                prefill_stats=stats,
            )
        ]
    )
    (request_output,) = processed.request_outputs
    assert request_output.num_cached_tokens == num_cached
    return request_output


def _cached_output(
    victim_tokens: list[int],
    victim_salt: str | None,
    attacker_tokens: list[int],
    attacker_salt: str | None,
) -> RequestOutput:
    init_none_hash(sha256)
    block_size = _block_size()
    processor = _processor()
    manager = _manager(block_size)
    _fill_cache(manager, processor, victim_tokens, victim_salt, block_size)
    return _probe(manager, processor, attacker_tokens, attacker_salt, block_size)


def _disclosed(usage) -> int:
    if usage is None:
        return 0
    details = getattr(usage, "input_tokens_details", None)
    if details is None:
        details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    cached = details.cached_tokens
    return 0 if cached is None else int(cached)


def _responses_serving() -> OpenAIServingResponses:
    engine = Mock()
    model_config = Mock()
    model_config.max_model_len = 8192
    model_config.hf_config.model_type = "test"
    model_config.hf_text_config = Mock()
    model_config.get_diff_sampling_param.return_value = {}
    engine.model_config = model_config
    engine.renderer = Mock()
    engine.renderer.get_tokenizer.return_value = Mock()
    return OpenAIServingResponses(
        engine_client=engine,
        models=Mock(),
        online_renderer=Mock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


async def _responses_usage(
    request_output: RequestOutput,
    cache_salt: str | None,
    *,
    context_kind: str,
):
    serving = _responses_serving()
    request = ResponsesRequest(
        model="test-model",
        input="guessed prefix",
        cache_salt=cache_salt,
        stream=False,
        store=False,
    )

    async def _empty():
        if False:
            yield None

    if context_kind == "simple":
        context: SimpleContext | ParsableContext | HarmonyContext = SimpleContext()
        context.append_output(request_output)
    elif context_kind == "parsable":
        context = ParsableContext(
            response_messages=[],
            tokenizer=Mock(),
            parser_cls=None,
            request=request,
            available_tools=[],
            chat_template=None,
            chat_template_content_format="auto",
        )
        context.append_output(request_output)
    elif context_kind == "harmony":
        serving.use_harmony = True
        harmony = HarmonyContext.__new__(HarmonyContext)
        harmony._messages = []
        harmony.num_init_messages = 0
        harmony.streamed_items_by_message = []
        harmony.num_tool_output_tokens = 0
        harmony.finish_reason = None
        harmony.num_prompt_tokens = 0
        harmony.num_output_tokens = 0
        harmony.num_cached_tokens = 0
        harmony.num_cache_creation_tokens = 0
        harmony.num_reasoning_tokens = 0
        harmony.all_turn_metrics = []
        harmony.current_turn_metrics = TurnMetrics()
        harmony.is_first_turn = True
        harmony.request_metrics = None
        harmony.request_metrics_cover_all_generation_turns = True
        harmony.kv_transfer_params = None
        harmony.ec_transfer_params = None
        harmony._update_prefill_token_usage(request_output)
        context = harmony
    else:
        raise AssertionError(context_kind)

    response = await serving.responses_full_generator(
        request=request,
        sampling_params=SamplingParams(max_tokens=1),
        result_generator=_empty(),
        context=context,
        model_name="test-model",
        tokenizer=Mock(),
        request_metadata=RequestResponseMetadata(request_id="attacker"),
    )
    return response.usage


class _ProbeStrategy(ReadStrategy):
    def __init__(self, context: ReadContext, request_output_for):
        super().__init__(context)
        self._request_output_for = request_output_for

    def limits(self) -> DecisionLimits:
        return DecisionLimits(max_questions=8, max_options=2)

    async def read(
        self,
        questions,
        instructions,
        state,
        *,
        request_id,
        chat_template_kwargs,
        lora_request,
        priority,
        cache_salt=None,
        trace_headers=None,
        media=None,
    ):
        request_output = self._request_output_for(cache_salt)
        return [
            NextTokenStrategy._question_read(
                request_output,
                [[10], [11]],
                [0.0, 0.0],
            )
        ]


def _decisions_serving(request_output_for) -> OpenAIServingDecisions:
    engine = Mock()
    engine.errored = False
    engine.model_config = Mock()
    models = Mock()
    models.is_base_model.return_value = True
    models.lora_requests = {}
    models.model_name.return_value = "test-model"
    context = ReadContext(
        engine_client=engine,
        online_renderer=Mock(),
        chat_template=None,
        chat_template_content_format="auto",
        default_chat_template_kwargs={},
    )
    return OpenAIServingDecisions(
        models,
        _ProbeStrategy(context, request_output_for),
    )


async def _decisions_usage(request_output: RequestOutput, cache_salt: str | None):
    def _output_for(salt: str | None) -> RequestOutput:
        assert salt == cache_salt
        return request_output

    serving = _decisions_serving(_output_for)
    request = DecisionRequest(
        model="test-model",
        input="guessed prefix",
        questions=[{"type": "predicate", "instructions": "Is this the prompt?"}],
        cache_salt=cache_salt,
    )
    response = await serving.create_decisions(request)
    return response.usage


def _chat_serving(enable_prompt_tokens_details: bool) -> OpenAIServingChat:
    serving = OpenAIServingChat.__new__(OpenAIServingChat)
    serving.response_role = "assistant"
    serving.parser_cls = None
    serving.enable_auto_tools = False
    serving._include_reasoning_tokens_details = False
    serving.enable_prompt_tokens_details = enable_prompt_tokens_details
    serving.enable_log_outputs = False
    serving.enable_log_deltas = False
    serving.enable_force_include_usage = False
    serving.request_logger = None
    serving.system_fingerprint = None
    serving.enable_per_request_metrics = False
    return serving


def _usage_from_sse(chunks: list[str]):
    usage = None
    for chunk in chunks:
        if not chunk.startswith("data: ") or chunk.startswith("data: [DONE]"):
            continue
        body = json.loads(chunk.removeprefix("data: ").strip())
        if body.get("usage"):
            usage = body["usage"]
    return usage


async def _chat_stream_usage(
    request_output: RequestOutput,
    cache_salt: str | None,
    *,
    enable_prompt_tokens_details: bool,
):
    serving = _chat_serving(enable_prompt_tokens_details)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "guessed prefix"}],
        max_tokens=1,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
        cache_salt=cache_salt,
    )

    async def _one():
        yield request_output

    usage = None
    async for chunk in serving.chat_completion_stream_generator_raw(
        request,
        _one(),
        "attacker",
        "test-model",
        [],
        Mock(),
        RequestResponseMetadata(request_id="attacker"),
    ):
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
    return usage


async def _completion_stream_usage(
    request_output: RequestOutput,
    cache_salt: str | None,
    *,
    enable_prompt_tokens_details: bool,
):
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = enable_prompt_tokens_details
    serving.enable_log_outputs = False
    serving.request_logger = None
    serving.enable_per_request_metrics = False
    serving.enable_force_include_usage = False
    serving.system_fingerprint = None
    serving.return_tokens_as_token_ids = False
    request = CompletionRequest(
        model="test-model",
        prompt="guessed prefix",
        max_tokens=1,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
        cache_salt=cache_salt,
    )

    async def _one():
        yield 0, request_output

    chunks = [
        chunk
        async for chunk in serving.completion_stream_generator(
            request,
            [tokens_input(list(request_output.prompt_token_ids or [1]))],
            _one(),
            "attacker",
            0,
            "test-model",
            1,
            None,
            RequestResponseMetadata(request_id="attacker"),
        )
    ]
    body = _usage_from_sse(chunks)
    if body is None:
        return None
    return UsageInfo.model_validate(body)


async def _chat_usage(
    request_output: RequestOutput,
    cache_salt: str | None,
    *,
    enable_prompt_tokens_details: bool,
):
    serving = _chat_serving(enable_prompt_tokens_details)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "guessed prefix"}],
        max_tokens=1,
        cache_salt=cache_salt,
    )

    async def _one():
        yield request_output

    response = await serving.chat_completion_full_generator(
        request,
        _one(),
        "attacker",
        "test-model",
        [],
        Mock(),
        RequestResponseMetadata(request_id="attacker"),
    )
    return response.usage


def _completion_usage(
    request_output: RequestOutput,
    cache_salt: str | None,
    *,
    enable_prompt_tokens_details: bool,
):
    serving = OpenAIServingCompletion.__new__(OpenAIServingCompletion)
    serving.enable_prompt_tokens_details = enable_prompt_tokens_details
    serving.enable_log_outputs = False
    serving.request_logger = None
    serving.enable_per_request_metrics = False
    serving.enable_force_include_usage = False
    serving.system_fingerprint = None
    serving.return_tokens_as_token_ids = False
    request = CompletionRequest(
        model="test-model",
        prompt="guessed prefix",
        max_tokens=1,
        cache_salt=cache_salt,
    )
    response = serving.request_output_to_completion_response(
        [request_output],
        request,
        "attacker",
        0,
        "test-model",
        None,
        RequestResponseMetadata(request_id="attacker"),
    )
    return response.usage


def _generate_usage(
    request_output: RequestOutput,
    cache_salt: str | None,
    *,
    enable_prompt_tokens_details: bool,
):
    serving = ServingTokens.__new__(ServingTokens)
    serving.enable_prompt_tokens_details = enable_prompt_tokens_details
    serving.enable_log_outputs = False
    serving.request_logger = None
    request = GenerateRequest(
        token_ids=list(request_output.prompt_token_ids or [1]),
        sampling_params=SamplingParams(max_tokens=1, n=1),
        cache_salt=cache_salt,
    )
    metadata = RequestResponseMetadata(request_id="attacker")
    fields = serving._build_response_fields(
        request,
        request_output,
        "attacker",
        "test-model",
        0,
        1,
        metadata,
    )
    return fields["usage"]


async def _generate_stream_usage(
    request_output: RequestOutput,
    cache_salt: str | None,
    *,
    enable_prompt_tokens_details: bool,
):
    serving = ServingTokens.__new__(ServingTokens)
    serving.enable_prompt_tokens_details = enable_prompt_tokens_details
    serving.enable_log_outputs = False
    serving.request_logger = None
    serving.return_tokens_as_token_ids = False
    request = GenerateRequest(
        token_ids=list(request_output.prompt_token_ids or [1]),
        sampling_params=SamplingParams(max_tokens=1, n=1),
        cache_salt=cache_salt,
        stream=True,
        stream_options=StreamOptions(include_usage=True),
    )
    chunks = [
        chunk
        async for chunk in serving.serve_tokens_stream_generator(
            request,
            _single_output(request_output),
            "attacker",
            "test-model",
            RequestResponseMetadata(request_id="attacker"),
        )
    ]
    body = _usage_from_sse(chunks)
    if body is None:
        return None
    return UsageInfo.model_validate(body)


async def _single_output(request_output: RequestOutput):
    yield request_output


def test_prefix_caching_is_on_and_salt_is_optional():
    cache = CacheConfig()
    assert cache.enable_prefix_caching is True
    assert cache.block_size == 16
    assert ResponsesRequest(model="m", input="x").cache_salt is None
    assert (
        DecisionRequest(
            model="m",
            input="x",
            questions=[{"type": "predicate", "instructions": "q"}],
        ).cache_salt
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("context_kind", ["simple", "parsable", "harmony"])
async def test_unsalted_responses_do_not_confirm_another_prefix(context_kind: str):
    """A second client who omits cache_salt must not learn that the first
    client's prefix is still cached."""
    block = _block_size()
    victim = _prompt(2, tail=7, block_size=block)
    attacker = _prompt(2, tail=8, block_size=block)
    request_output = _cached_output(victim, None, attacker, None)
    usage = await _responses_usage(request_output, None, context_kind=context_kind)
    assert _disclosed(usage) == 0, (
        f"{context_kind} usage confirmed an unsalted prefix "
        f"(engine cached {request_output.num_cached_tokens} tokens)"
    )


@pytest.mark.asyncio
async def test_unsalted_responses_do_not_confirm_a_one_block_prefix():
    block = _block_size()
    victim = _prompt(2, tail=7, block_size=block)
    # Same first block, different continuation. One full block is enough.
    attacker = [1_000] * block + [2_001] * block + [8]
    request_output = _cached_output(victim, None, attacker, None)
    usage = await _responses_usage(request_output, None, context_kind="simple")
    assert _disclosed(usage) == 0, (
        "one-block unsalted prefix was confirmed "
        f"(engine cached {request_output.num_cached_tokens} tokens)"
    )


@pytest.mark.asyncio
async def test_unsalted_decisions_do_not_confirm_another_prefix():
    block = _block_size()
    victim = _prompt(2, tail=7, block_size=block)
    attacker = _prompt(2, tail=8, block_size=block)
    request_output = _cached_output(victim, None, attacker, None)
    usage = await _decisions_usage(request_output, None)
    assert _disclosed(usage) == 0, (
        "decisions usage confirmed an unsalted prefix "
        f"(engine cached {request_output.num_cached_tokens} tokens)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "publish",
    [
        _chat_usage,
        _chat_stream_usage,
        _completion_usage,
        _completion_stream_usage,
        _generate_usage,
        _generate_stream_usage,
    ],
    ids=[
        "chat",
        "chat_stream",
        "completion",
        "completion_stream",
        "generate",
        "generate_stream",
    ],
)
async def test_unsalted_prompt_details_do_not_confirm_another_prefix(publish):
    """Chat, completions, and token-in/token-out report the hit only when the
    server enables prompt-token details. With that flag on, an unsalted
    caller still must not see another client's prefix."""
    block = _block_size()
    victim = _prompt(2, tail=7, block_size=block)
    attacker = _prompt(2, tail=8, block_size=block)
    request_output = _cached_output(victim, None, attacker, None)
    usage = publish(
        request_output,
        None,
        enable_prompt_tokens_details=True,
    )
    if hasattr(usage, "__await__"):
        usage = await usage
    assert _disclosed(usage) == 0, (
        f"{publish.__name__} confirmed an unsalted prefix "
        f"(engine cached {request_output.num_cached_tokens} tokens)"
    )


@pytest.mark.asyncio
async def test_matching_cache_salt_still_reports_the_prefix_hit():
    block = _block_size()
    victim = _prompt(2, tail=7, block_size=block)
    attacker = _prompt(2, tail=8, block_size=block)
    salt = "caller-supplied-salt"
    request_output = _cached_output(victim, salt, attacker, salt)
    responses_usage = await _responses_usage(
        request_output, salt, context_kind="simple"
    )
    decisions_usage = await _decisions_usage(request_output, salt)
    chat_usage = await _chat_usage(
        request_output, salt, enable_prompt_tokens_details=True
    )
    assert request_output.num_cached_tokens == 2 * block
    assert _disclosed(responses_usage) == 2 * block
    assert _disclosed(decisions_usage) == 2 * block
    assert _disclosed(chat_usage) == 2 * block


@pytest.mark.asyncio
async def test_different_cache_salt_reports_no_hit():
    block = _block_size()
    victim = _prompt(2, tail=7, block_size=block)
    attacker = _prompt(2, tail=8, block_size=block)
    request_output = _cached_output(victim, "salt-a", attacker, "salt-b")
    usage = await _responses_usage(request_output, "salt-b", context_kind="simple")
    assert request_output.num_cached_tokens == 0
    assert _disclosed(usage) == 0


@pytest.mark.asyncio
async def test_prefix_shorter_than_one_block_is_not_confirmed():
    block = _block_size()
    victim = _prompt(2, tail=7, block_size=block)
    attacker = victim[: block - 1]
    request_output = _cached_output(victim, None, attacker, None)
    usage = await _responses_usage(request_output, None, context_kind="simple")
    assert request_output.num_cached_tokens == 0
    assert _disclosed(usage) == 0
