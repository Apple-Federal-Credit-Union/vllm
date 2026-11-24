# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Malformed kv_transfer_params.disagg_spec must not kill the engine.

LMCacheConnectorV1 forwards scheduler updates to the installed LMCache
adapter. That adapter indexes disagg_spec with no type or key checks, and
schedule() does not catch the resulting KeyError or TypeError. These tests
are the expected behavior: every public request model rejects a bad
disagg_spec, and a request that already carries one still schedules.
"""

from __future__ import annotations

import pytest
import torch

from vllm.config import (
    CacheConfig,
    KVTransferConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.cohere.protocol import CohereChatV2Request
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.fault_tolerance.engine_core_sentinel import fault_tolerant_wrapper
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

pytestmark = pytest.mark.cpu_test

_VALID_DISAGG_SPEC = {
    "receiver_host": "10.0.0.8",
    "receiver_init_port": 5555,
    "receiver_alloc_port": 5556,
    "req_id": "downstream-1",
}

# Each value is client JSON that reaches update_state_after_alloc and raises.
_MALFORMED_DISAGG_SPECS = [
    {},
    "prefill",
    None,
    [],
    {"receiver_init_port": 5555, "receiver_alloc_port": 5556, "req_id": "r"},
    {
        "receiver_host": 10,
        "receiver_init_port": 5555,
        "receiver_alloc_port": 5556,
        "req_id": "r",
    },
    {
        "receiver_host": "10.0.0.8",
        "receiver_init_port": "5555",
        "receiver_alloc_port": 5556,
        "req_id": "r",
    },
    {
        "receiver_host": "10.0.0.8",
        "receiver_init_port": 70000,
        "receiver_alloc_port": 5556,
        "req_id": "r",
    },
    {
        "receiver_host": "10.0.0.8",
        "receiver_init_port": 0,
        "receiver_alloc_port": 5556,
        "req_id": "r",
    },
    {
        "receiver_host": "10.0.0.8",
        "receiver_init_port": 5555,
        "receiver_alloc_port": 5556,
        "req_id": 12,
    },
    {
        "receiver_host": "",
        "receiver_init_port": 5555,
        "receiver_alloc_port": 5556,
        "req_id": "r",
    },
]


def _chat_body(disagg_spec):
    params = {"lmcache.tag.user": "alice"}
    if disagg_spec is not _OMIT:
        params = {"disagg_spec": disagg_spec}
    return {
        "model": "facebook/opt-125m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "kv_transfer_params": params,
    }


_OMIT = object()


def _payloads(disagg_spec):
    kv = {"disagg_spec": disagg_spec}
    return [
        ChatCompletionRequest,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "kv_transfer_params": kv,
        },
        CompletionRequest,
        {"model": "m", "prompt": "hi", "kv_transfer_params": kv},
        ResponsesRequest,
        {"model": "m", "input": "hi", "kv_transfer_params": kv},
        AnthropicMessagesRequest,
        {
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
            "kv_transfer_params": kv,
        },
        CohereChatV2Request,
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "kv_transfer_params": kv,
        },
        GenerateRequest,
        {
            "token_ids": [1, 2, 3],
            "sampling_params": {"max_tokens": 4},
            "kv_transfer_params": kv,
        },
    ]


def _pairs(disagg_spec):
    items = _payloads(disagg_spec)
    return list(zip(items[0::2], items[1::2], strict=True))


@pytest.mark.parametrize(
    ("model_cls", "body"),
    _pairs({}),
    ids=[
        "chat",
        "completion",
        "responses",
        "anthropic",
        "cohere",
        "generate",
    ],
)
@pytest.mark.parametrize("disagg_spec", _MALFORMED_DISAGG_SPECS)
def test_public_apis_reject_malformed_disagg_spec(model_cls, body, disagg_spec):
    body = dict(body)
    body["kv_transfer_params"] = {"disagg_spec": disagg_spec}
    with pytest.raises(VLLMValidationError, match="disagg_spec"):
        model_cls.model_validate(body)


@pytest.mark.parametrize(
    ("model_cls", "body"),
    _pairs(_VALID_DISAGG_SPEC),
    ids=[
        "chat",
        "completion",
        "responses",
        "anthropic",
        "cohere",
        "generate",
    ],
)
def test_public_apis_accept_complete_disagg_spec(model_cls, body):
    request = model_cls.model_validate(body)
    assert request.kv_transfer_params["disagg_spec"] == _VALID_DISAGG_SPEC


def test_chat_completion_forwards_valid_disagg_spec_to_sampling_params():
    request = ChatCompletionRequest.model_validate(_chat_body(_VALID_DISAGG_SPEC))
    sampling = request.to_sampling_params(4, {})
    assert sampling.extra_args is not None
    assert sampling.extra_args["kv_transfer_params"]["disagg_spec"] == (
        _VALID_DISAGG_SPEC
    )


def test_completion_forwards_valid_disagg_spec_to_sampling_params():
    request = CompletionRequest.model_validate(
        {
            "model": "m",
            "prompt": "hi",
            "kv_transfer_params": {"disagg_spec": _VALID_DISAGG_SPEC},
        }
    )
    sampling = request.to_sampling_params(4, {})
    assert sampling.extra_args["kv_transfer_params"]["disagg_spec"]["req_id"] == (
        "downstream-1"
    )


def test_responses_forwards_valid_disagg_spec_to_sampling_params():
    request = ResponsesRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "kv_transfer_params": {"disagg_spec": _VALID_DISAGG_SPEC},
        }
    )
    sampling = request.to_sampling_params(4, {})
    forwarded = sampling.extra_args["kv_transfer_params"]["disagg_spec"]
    assert forwarded["receiver_host"] == "10.0.0.8"


def test_other_kv_transfer_params_stay_accepted():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "kv_transfer_params": {"lmcache.tag.user": "alice"},
        }
    )
    sampling = request.to_sampling_params(4, {})
    assert sampling.extra_args["kv_transfer_params"] == {"lmcache.tag.user": "alice"}


def _engine_request(request_id: str, disagg_spec) -> Request:
    extra = None
    if disagg_spec is not _OMIT:
        extra = {"kv_transfer_params": {"disagg_spec": disagg_spec}}
    sampling = SamplingParams(max_tokens=4, extra_args=extra)
    return Request(
        request_id=request_id,
        prompt_token_ids=[11, 22, 33, 44],
        sampling_params=sampling,
        pooling_params=None,
    )


def _scheduler(kv_role: str = "kv_both") -> Scheduler:
    init_none_hash(sha256)
    model_config = ModelConfig(
        model="facebook/opt-125m",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
        max_model_len=128,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        max_model_len=128,
        enable_chunked_prefill=True,
        watermark=0.0,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        kv_transfer_config=KVTransferConfig(
            kv_connector="LMCacheConnectorV1",
            kv_role=kv_role,
        ),
    )
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    cache_config.num_gpu_blocks = 100
    register_all_kvcache_specs(vllm_config)
    return Scheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=16,
        log_stats=False,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )


class _BusyLoop:
    """Engine-core object the fault-tolerance wrapper drives.

    schedule() is the real scheduler. enable_fault_tolerance matches the
    ParallelConfig default, so an exception from schedule() leaves the loop.
    """

    enable_fault_tolerance = False

    def __init__(self, scheduler: Scheduler, follow_up: Request):
        self.scheduler = scheduler
        self.follow_up = follow_up
        self.outputs = []

    def run(self):
        self.outputs.append(self.scheduler.schedule())
        self.scheduler.add_request(self.follow_up)
        self.outputs.append(self.scheduler.schedule())
        # The real busy loop repeats until shutdown. Stop after one pass so
        # a surviving scheduler is observable. A scheduling exception is
        # re-raised by the wrapper instead of reaching this exit.
        raise SystemExit(0)


@pytest.mark.parametrize("kv_role", ["kv_both", "kv_producer"])
@pytest.mark.parametrize("disagg_spec", [{}, "prefill", None])
def test_malformed_disagg_spec_does_not_kill_the_scheduler(kv_role, disagg_spec):
    scheduler = _scheduler(kv_role)
    assert scheduler.connector is not None
    assert type(scheduler.connector).__name__ == "LMCacheConnectorV1"
    scheduler.add_request(_engine_request("bad", disagg_spec))
    follow_up = _engine_request("good", _OMIT)
    loop = _BusyLoop(scheduler, follow_up)
    # Same wrapper EngineCoreProc.run_busy_loop uses. Fault tolerance is off
    # by default, so a KeyError or TypeError from schedule() leaves the loop
    # and the process entry marks the engine core dead.
    with pytest.raises(SystemExit):
        fault_tolerant_wrapper(_BusyLoop.run)(loop)
    assert "good" in loop.outputs[1].num_scheduled_tokens


def test_valid_disagg_spec_is_scheduled():
    scheduler = _scheduler()
    scheduler.add_request(_engine_request("ok", _VALID_DISAGG_SPEC))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["ok"] > 0


def test_request_without_disagg_spec_is_scheduled():
    scheduler = _scheduler()
    scheduler.add_request(_engine_request("plain", _OMIT))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["plain"] > 0
