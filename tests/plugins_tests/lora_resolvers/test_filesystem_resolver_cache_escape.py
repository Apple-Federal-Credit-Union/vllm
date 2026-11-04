# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache confinement for the filesystem LoRA resolver.

These tests fail while a request model name can resolve to an adapter outside
the configured cache directory. They pass once absolute names, parent-directory
segments, and symlink escapes are refused before the adapter is opened, and
once the path that is loaded is still the directory that was authorized.
"""

import asyncio
import errno
import json
import os
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import vllm.lora.lora_model as lora_model_module
import vllm.plugins as plugins
from vllm.config.lora import LoRAConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.pooling.base.serving import PoolingBaseServing
from vllm.entrypoints.pooling.embed.protocol import EmbeddingCompletionRequest
from vllm.entrypoints.serve.engine.serving import BaseServing
from vllm.exceptions import VLLMNotFoundError
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolverRegistry
from vllm.lora.utils import get_adapter_absolute_path
from vllm.plugins.lora_resolvers.filesystem_resolver import (
    FilesystemResolver,
    register_filesystem_resolver,
)
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

BASE_MODEL = "org/base-model"
SERVED_NAME = "served-alias"
INSIDE_ALPHA = 8
OUTSIDE_ALPHA = 64
INSIDE_MARKER = 0.25
OUTSIDE_MARKER = 0.75
LORA_A = "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
LORA_B = "base_model.model.layers.0.self_attn.q_proj.lora_B.weight"
MODULE_NAME = "layers.0.self_attn.q_proj"

ESCAPE_KINDS = [
    "absolute",
    "absolute_dotdot",
    "parent",
    "parent_through_subdir",
    "symlink",
    "dot_parent",
    "relative_cache",
]


@dataclass
class Layout:
    root: Path
    cache: Path
    outside: Path
    inside: Path


@dataclass
class LoadedAdapter:
    lora_path: str
    lora_alpha: int
    marker: float
    request: LoRARequest


@dataclass
class _ModelConfig:
    model: str = BASE_MODEL
    max_model_len: int = 32


@pytest.fixture(autouse=True)
def _restore_resolver_registry():
    saved = dict(LoRAResolverRegistry.resolvers)
    plugins_loaded = plugins.plugins_loaded
    LoRAResolverRegistry.resolvers.clear()
    yield
    LoRAResolverRegistry.resolvers.clear()
    LoRAResolverRegistry.resolvers.update(saved)
    plugins.plugins_loaded = plugins_loaded


@pytest.fixture
def runtime_lora_updates(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")
    monkeypatch.delenv("VLLM_SKIP_MODEL_NAME_VALIDATION", raising=False)


def _write_adapter(directory: Path, alpha: int, marker: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": BASE_MODEL,
        "r": 8,
        "lora_alpha": alpha,
        "target_modules": ["q_proj"],
        "bias": "none",
    }
    (directory / "adapter_config.json").write_text(json.dumps(config))
    save_file(
        {
            LORA_A: torch.full((8, 4), marker),
            LORA_B: torch.full((4, 8), marker),
        },
        str(directory / "adapter_model.safetensors"),
    )


def _layout(tmp_path: Path) -> Layout:
    root = tmp_path
    cache = root / "adapters"
    outside = root / "adapters_evil"
    inside = cache / "good"
    cache.mkdir()
    (cache / "subdir").mkdir()
    _write_adapter(inside, INSIDE_ALPHA, INSIDE_MARKER)
    _write_adapter(cache / "team" / "adapter", INSIDE_ALPHA, INSIDE_MARKER)
    _write_adapter(outside, OUTSIDE_ALPHA, OUTSIDE_MARKER)
    os.symlink(inside, cache / "inside_link")
    os.symlink(outside, cache / "outside_link")
    return Layout(root=root, cache=cache, outside=outside, inside=inside)


def _escape_name(kind: str, layout: Layout) -> str:
    outside = layout.outside.name
    if kind == "absolute":
        return str(layout.outside)
    if kind == "absolute_dotdot":
        return str(layout.cache / ".." / outside)
    if kind == "parent":
        return f"../{outside}"
    if kind == "parent_through_subdir":
        return f"subdir/../../{outside}"
    if kind == "symlink":
        return "outside_link"
    if kind == "dot_parent":
        return f"./../{outside}"
    if kind == "relative_cache":
        return str(layout.outside)
    raise AssertionError(kind)


def _resolver_for(kind: str, layout: Layout) -> FilesystemResolver:
    if kind == "relative_cache":
        return FilesystemResolver(os.path.relpath(layout.cache))
    return FilesystemResolver(str(layout.cache))


def _is_within(path: str, root: str) -> bool:
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(path)
    try:
        return os.path.commonpath([root_real, candidate]) == root_real
    except ValueError:
        return False


def _allow_cpu_load_without_pin() -> None:
    """Host memory pinning is unavailable when this process has no accelerator.

    ``LoRAModel.from_local_checkpoint`` reads the checkpoint before it pins
    tensors. Pinning is skipped on the CPU platform; if this process resolved
    another platform, turn that placement step off so the read still runs.
    """
    if not lora_model_module.PIN_MEMORY:
        return
    try:
        torch.empty(1).pin_memory()
    except RuntimeError:
        lora_model_module.PIN_MEMORY = False


def _transport_like_engine(request: LoRARequest) -> LoRARequest:
    """Replay the API-server utility call and the worker RPC pickle."""

    class _Add:
        def add_lora(self, lora_request: LoRARequest) -> bool:
            return True

    encoded = MsgpackEncoder().encode((0, 1, "add_lora", (request,)))
    _client_idx, _call_id, method, args = MsgpackDecoder().decode(encoded)
    assert method == "add_lora"
    converted = EngineCoreProc._convert_msgspec_args(_Add().add_lora, args)
    transported = converted[0]
    assert isinstance(transported, LoRARequest)
    assert transported.lora_path == request.lora_path
    pickled = pickle.loads(pickle.dumps(transported, protocol=pickle.HIGHEST_PROTOCOL))
    assert isinstance(pickled, LoRARequest)
    assert pickled.lora_path == transported.lora_path
    return pickled


def _load_like_worker(request: LoRARequest) -> LoadedAdapter:
    """Open the adapter the way ``WorkerLoRAManager._load_adapter`` does."""
    _allow_cpu_load_without_pin()
    lora_path = get_adapter_absolute_path(request.lora_path)
    peft_helper = PEFTHelper.from_local_dir(lora_path, max_position_embeddings=None)
    lora_config = LoRAConfig()
    lora_config.lora_dtype = torch.float32
    peft_helper.validate_legal(lora_config)
    model = LoRAModel.from_local_checkpoint(
        lora_path,
        {"q_proj"},
        peft_helper=peft_helper,
        lora_model_id=request.lora_int_id,
        device="cpu",
        dtype=torch.float32,
    )
    marker = float(model.loras[MODULE_NAME].lora_a.flatten()[0])
    return LoadedAdapter(
        lora_path=lora_path,
        lora_alpha=peft_helper.lora_alpha,
        marker=marker,
        request=request,
    )


def _load_transported(request: LoRARequest) -> LoadedAdapter:
    return _load_like_worker(_transport_like_engine(request))


class _Engine:
    def __init__(self) -> None:
        self.model_config = _ModelConfig()
        self.renderer = None
        self.input_processor = None
        self.loaded: list[LoadedAdapter] = []

    async def add_lora(self, lora_request: LoRARequest) -> bool:
        self.loaded.append(_load_transported(lora_request))
        return True


class _PoolingProbe(PoolingBaseServing):
    def _build_response(self, ctx):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def get_io_processor(self, request):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def _serving(
    cache: Path,
) -> tuple[_Engine, OpenAIServingModels, BaseServing, _PoolingProbe]:
    LoRAResolverRegistry.register_resolver(
        "Filesystem Resolver", FilesystemResolver(str(cache))
    )
    engine = _Engine()
    models = OpenAIServingModels(
        engine,  # type: ignore[arg-type]
        [BaseModelPath(name=SERVED_NAME, model_path=BASE_MODEL)],
    )
    base = BaseServing(models, engine.model_config)  # type: ignore[arg-type]
    pooling = _PoolingProbe.__new__(_PoolingProbe)
    pooling.models = models
    return engine, models, base, pooling


def _completion(model: str) -> CompletionRequest:
    return CompletionRequest(model=model, prompt="hello")


def _chat(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model, messages=[{"role": "user", "content": "hello"}]
    )


def _embedding(model: str) -> EmbeddingCompletionRequest:
    return EmbeddingCompletionRequest(model=model, input="hello")


def _adapter_problem(loaded: LoadedAdapter, cache: Path) -> str | None:
    if not _is_within(loaded.lora_path, str(cache)):
        return f"loaded path {loaded.lora_path} is outside {cache}"
    if loaded.marker == OUTSIDE_MARKER or loaded.lora_alpha == OUTSIDE_ALPHA:
        return (
            f"loaded outside adapter alpha={loaded.lora_alpha} "
            f"marker={loaded.marker} from {loaded.lora_path}"
        )
    return None


def _followup(serving: BaseServing, request) -> LoRARequest | None:
    try:
        return serving._maybe_get_adapters(request)
    except VLLMNotFoundError:
        return None


async def _serve(cache: Path, model_name: str) -> list[str]:
    engine, models, base, pooling = _serving(cache)
    completion = _completion(model_name)
    chat = _chat(model_name)
    embedding = _embedding(model_name)
    assert completion.model == model_name
    assert chat.model == model_name
    assert embedding.model == model_name
    problems: list[str] = []
    await base._check_model(completion)
    await base._check_model(chat)
    await pooling._check_model(embedding)
    for loaded in engine.loaded:
        problem = _adapter_problem(loaded, cache)
        if problem:
            problems.append(problem)
    stored = models.lora_requests.get(model_name)
    if stored is not None and not _is_within(stored.lora_path, str(cache)):
        problems.append(f"stored LoRA path {stored.lora_path} is outside {cache}")
    for request in (completion, chat, embedding):
        follow = _followup(base if request is not embedding else pooling, request)
        if follow is not None and not _is_within(follow.lora_path, str(cache)):
            problems.append(f"later request for {model_name} uses {follow.lora_path}")
    return problems


def _run_blocking(fn, fifo: Path, timeout: float = 10.0):
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    opened = False
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        try:
            fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != errno.ENXIO:
                raise
        else:
            opened = True
            os.write(fd, b"{}")
            os.close(fd)
        time.sleep(0.01)
    thread.join(timeout=2)
    return box, opened, thread.is_alive()


@pytest.mark.parametrize("name", ["good", "team/adapter", "inside_link"])
def test_in_cache_adapter_is_loaded(
    tmp_path: Path, runtime_lora_updates, name: str
) -> None:
    layout = _layout(tmp_path)
    request = asyncio.run(
        FilesystemResolver(str(layout.cache)).resolve_lora(BASE_MODEL, name)
    )
    assert request is not None
    loaded = _load_transported(request)
    assert _is_within(loaded.lora_path, str(layout.cache))
    assert loaded.lora_alpha == INSIDE_ALPHA
    assert loaded.marker == INSIDE_MARKER
    problems = asyncio.run(_serve(layout.cache, name))
    assert problems == []


@pytest.mark.parametrize("kind", ESCAPE_KINDS)
def test_model_name_outside_cache_is_not_loaded(
    tmp_path: Path, runtime_lora_updates, kind: str
) -> None:
    layout = _layout(tmp_path)
    name = _escape_name(kind, layout)
    request = asyncio.run(_resolver_for(kind, layout).resolve_lora(BASE_MODEL, name))
    problems: list[str] = []
    if request is not None:
        problem = _adapter_problem(_load_transported(request), layout.cache)
        if problem:
            problems.append(problem)
    problems.extend(asyncio.run(_serve(layout.cache, name)))
    assert problems == [], problems


def test_absolute_name_inside_cache_is_rejected(
    tmp_path: Path, runtime_lora_updates
) -> None:
    layout = _layout(tmp_path)
    name = str(layout.inside)
    request = asyncio.run(
        FilesystemResolver(str(layout.cache)).resolve_lora(BASE_MODEL, name)
    )
    problems = (
        [] if request is None else [f"resolved absolute path {request.lora_path}"]
    )
    problems.extend(asyncio.run(_serve(layout.cache, name)))
    assert problems == [], problems


def test_parent_segment_inside_cache_is_rejected(
    tmp_path: Path, runtime_lora_updates
) -> None:
    layout = _layout(tmp_path)
    name = "subdir/../good"
    request = asyncio.run(
        FilesystemResolver(str(layout.cache)).resolve_lora(BASE_MODEL, name)
    )
    problems = [] if request is None else [f"resolved parent path {request.lora_path}"]
    problems.extend(asyncio.run(_serve(layout.cache, name)))
    assert problems == [], problems


def test_symlink_retarget_between_resolve_and_load(
    tmp_path: Path, runtime_lora_updates
) -> None:
    layout = _layout(tmp_path)
    alias = layout.cache / "alias"
    os.symlink(layout.inside, alias)
    request = asyncio.run(
        FilesystemResolver(str(layout.cache)).resolve_lora(BASE_MODEL, "alias")
    )
    escaped = layout.root / "retargeted"
    escaped.mkdir()
    fifo = escaped / "adapter_config.json"
    os.mkfifo(fifo)
    os.remove(alias)
    os.symlink(escaped, alias)
    if request is None:
        return
    box, opened, hung = _run_blocking(lambda: _load_transported(request), fifo)
    assert not opened, "adapter open followed the retargeted symlink"
    assert not hung
    if "error" in box:
        return
    loaded = box["value"]
    assert _is_within(loaded.lora_path, str(layout.cache))
    assert loaded.lora_alpha == INSIDE_ALPHA
    assert loaded.marker == INSIDE_MARKER


@pytest.mark.parametrize("kind", ESCAPE_KINDS)
def test_model_name_outside_cache_is_not_opened(
    tmp_path: Path, runtime_lora_updates, kind: str
) -> None:
    cache = tmp_path / "adapters"
    outside = tmp_path / "adapters_evil"
    cache.mkdir()
    (cache / "subdir").mkdir()
    outside.mkdir()
    fifo = outside / "adapter_config.json"
    os.mkfifo(fifo)
    os.symlink(outside, cache / "outside_link")
    before = sorted(p.name for p in outside.iterdir())
    layout = Layout(root=tmp_path, cache=cache, outside=outside, inside=cache / "good")
    name = _escape_name(kind, layout)

    def resolve_and_serve():
        return asyncio.run(_resolve_and_serve(cache, name, kind, layout))

    box, opened, hung = _run_blocking(resolve_and_serve, fifo)
    after = sorted(p.name for p in outside.iterdir())
    assert after == before
    assert not opened, f"{kind} opened {fifo}"
    assert not hung, box
    if "error" in box:
        raise box["error"]
    direct, problems = box["value"]
    if direct is not None:
        assert _is_within(direct.lora_path, str(cache))
    assert problems == []


async def _resolve_and_serve(
    cache: Path, name: str, kind: str, layout: Layout
) -> tuple[LoRARequest | None, list[str]]:
    direct = await _resolver_for(kind, layout).resolve_lora(BASE_MODEL, name)
    problems = await _serve(cache, name)
    return direct, problems


def test_filesystem_resolver_registers_without_plugin_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "adapters"
    cache.mkdir()
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    monkeypatch.setenv("VLLM_LORA_RESOLVER_CACHE_DIR", str(cache))
    register_filesystem_resolver()
    registered = LoRAResolverRegistry.get_resolver("Filesystem Resolver")
    assert isinstance(registered, FilesystemResolver)
    assert os.path.realpath(registered.lora_cache_dir) == os.path.realpath(cache)

    LoRAResolverRegistry.resolvers.clear()
    plugins.plugins_loaded = False
    plugins.load_general_plugins()
    loaded = LoRAResolverRegistry.get_resolver("Filesystem Resolver")
    assert isinstance(loaded, FilesystemResolver)
    assert os.path.realpath(loaded.lora_cache_dir) == os.path.realpath(cache)
