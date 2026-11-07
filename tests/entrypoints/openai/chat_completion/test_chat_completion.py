# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

# any model with a chat template defined in tokenizer_config should work here
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--enable-per-request-metrics",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_json_schema(client: openai.AsyncOpenAI, model_name: str) -> None:
    invalid_json_schema = {
        "$defs": {
            "CarType": {
                "enum": ["sedan", "SUV", "Truck", "Coupe"],
                "title": "CarType",
                "type": "string",
            }
        },
        "properties": {
            "brand": {"title": "Brand", "type": "string"},
            "model": {"title": "Model", "type": "string"},
            "car_type": {"$ref": "#/$defs/CarType"},
            "foo": "bar",
        },
        "required": ["brand", "model", "car_type"],
        "title": "CarDescription",
        "type": "object",
    }
    prompt = (
        "Generate a JSON with the brand, model and car_type of"
        "the most iconic car from the 90's"
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"json": invalid_json_schema}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_regex(client: openai.AsyncOpenAI, model_name: str):
    prompt = (
        "Generate an email address for Alan Turing, who works in Enigma."
        "End in .com and new line. Example result:"
        "alan.turing@enigma.com\n"
    )

    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"regex": r"[.*"}, "stop": ["\n"]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_grammar(client: openai.AsyncOpenAI, model_name: str):
    invalid_simplified_sql_grammar = """
        root ::= select_statementinvalidsyntax

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    prompt = (
        "Generate an SQL query to show the 'username' and 'email'"
        "from the 'users' table."
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={
                "structured_outputs": {"grammar": invalid_simplified_sql_grammar}
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_empty_grammar(client: openai.AsyncOpenAI, model_name: str) -> None:
    prompt = "Say hello"
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"grammar": ""}},
        )


@pytest.mark.asyncio
async def test_kv_transfer_prompt_token_ids_do_not_replace_messages(
    client: openai.AsyncOpenAI,
):
    """Ids in kv_transfer_params must not replace the templated messages."""
    forwarded = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello, how are you today?"}],
        max_completion_tokens=1,
        extra_body={"return_token_ids": True},
    )
    forwarded_ids = forwarded.prompt_token_ids
    assert forwarded_ids

    messages = [{"role": "user", "content": "unrelated decode-side text"}]
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=1,
        extra_body={"return_token_ids": True},
    )
    assert baseline.prompt_token_ids != forwarded_ids

    decode = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=1,
        extra_body={
            "kv_transfer_params": {"prompt_token_ids": forwarded_ids},
            "return_token_ids": True,
        },
    )
    assert decode.prompt_token_ids == baseline.prompt_token_ids


def _assert_complete_timing_metrics(metrics: object) -> None:
    assert isinstance(metrics, dict)
    timing_fields = {
        "time_to_first_token_ms",
        "generation_time_ms",
        "queue_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
    }
    assert timing_fields <= metrics.keys()
    assert all(metrics[field] >= 0 for field in timing_fields)


@pytest.mark.asyncio
async def test_per_request_metrics_non_streaming(client: openai.AsyncOpenAI):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Count to four."}],
        max_completion_tokens=4,
        temperature=0,
        extra_body={"min_tokens": 4},
    )

    _assert_complete_timing_metrics(response.model_extra.get("metrics"))


@pytest.mark.asyncio
async def test_per_request_metrics_on_final_usage_chunk(client: openai.AsyncOpenAI):
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello."}],
        max_completion_tokens=1,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"min_tokens": 1},
    )

    chunks = [chunk async for chunk in stream]
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.usage is None
        assert "metrics" not in (chunk.model_extra or {})

    usage_chunk = chunks[-1]
    assert usage_chunk.usage is not None
    assert usage_chunk.choices == []
    metrics = (usage_chunk.model_extra or {}).get("metrics")
    assert isinstance(metrics, dict)
    assert metrics["time_to_first_token_ms"] >= 0
    assert metrics["generation_time_ms"] >= 0
    assert metrics["queue_time_ms"] >= 0
    assert metrics["tokens_per_second"] >= 0
    assert "mean_itl_ms" not in metrics
