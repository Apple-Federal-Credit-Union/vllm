# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Encoder-cache paths must stay inside the configured storage root.

These tests fail while ``ECExampleConnector`` joins a caller-supplied
multimodal identifier onto ``shared_storage_path`` and creates or reads
``encoder_cache.safetensors`` there. A fix has to keep every save, load,
and existence check inside that root, including when the check and the
filesystem call are not the same path string. Creating or deleting a file
outside the root is a failure even if the call then raises.
"""

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import pytest
import safetensors.torch
import torch
from PIL import Image

import vllm.platforms as platforms
from vllm.config import DeviceConfig, VllmConfig
from vllm.config.ec_transfer import ECTransferConfig
from vllm.config.lora import LoRAConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnector,
    ECExampleConnectorMetadata,
    MMMeta,
)
from vllm.distributed.ec_transfer.ec_transfer_state import (
    ensure_ec_transfer_initialized,
    ensure_ec_transfer_shutdown,
    get_ec_transfer,
)
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.platforms.cpu import CpuPlatform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request
from vllm.v1.worker.ec_connector_model_runner_mixin import (
    ECConnectorModelRunnerMixin,
)
from vllm.v1.worker.gpu.ec_connector import ActiveECConnector

_IDENTIFIER_KINDS = (
    "parent",
    "dot_parent",
    "nested_parent",
    "storage_parent",
    "absolute",
)
# ``..`` names the storage directory's parent, which already exists, so a
# missing load cannot create a new directory there. The save and planted-load
# tests cover that identifier.
_NEW_DIRECTORY_KINDS = tuple(
    kind for kind in _IDENTIFIER_KINDS if kind != "storage_parent"
)


class _IdentifierState:
    """Only the attributes ``InputProcessor._get_mm_identifier`` reads."""

    def __init__(self, lora_config: LoRAConfig | None) -> None:
        self.lora_config = lora_config


@pytest.fixture(autouse=True)
def _use_cpu_platform() -> Iterator[None]:
    """``start_load_caches`` passes ``current_platform.device_type`` through.

    An unspecified platform uses an empty device string, and safetensors
    rejects that before the tensor is stored. CPU is the device this build
    can actually load onto.
    """
    previous = platforms.current_platform
    if previous.device_type != "cpu":
        platforms.current_platform = CpuPlatform()
    yield
    platforms.current_platform = previous


@pytest.fixture
def storage(tmp_path: Path) -> Path:
    path = tmp_path / "storage"
    path.mkdir()
    return path


@pytest.fixture
def vllm_config(storage: Path) -> VllmConfig:
    return VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role="ec_both",
            ec_connector_extra_config={"shared_storage_path": str(storage)},
        ),
    )


@pytest.fixture
def connector(vllm_config: VllmConfig) -> ECExampleConnector:
    return ECExampleConnector(vllm_config, ECConnectorRole.WORKER)


@pytest.fixture
def scheduler_connector(vllm_config: VllmConfig) -> ECExampleConnector:
    return ECExampleConnector(vllm_config, ECConnectorRole.SCHEDULER)


@pytest.fixture
def worker_transfer(vllm_config: VllmConfig) -> Iterator[ECExampleConnector]:
    ensure_ec_transfer_shutdown()
    ensure_ec_transfer_initialized(vllm_config)
    try:
        connector = get_ec_transfer()
        assert isinstance(connector, ECExampleConnector)
        yield connector
    finally:
        ensure_ec_transfer_shutdown()


def _run(action: Callable[[], None]) -> None:
    try:
        action()
    except Exception:
        return


def _resolved_tree(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    paths = {root.resolve()}
    paths.update(path.resolve() for path in root.rglob("*"))
    return paths


def _outside(storage: Path, paths: set[Path]) -> list[Path]:
    storage_resolved = storage.resolve()
    outside = []
    for path in paths:
        try:
            path.relative_to(storage_resolved)
        except ValueError:
            outside.append(path)
    return outside


def _assert_storage_boundary_unchanged(
    storage: Path, sandbox: Path, before: set[Path]
) -> None:
    after = _resolved_tree(sandbox)
    created = _outside(storage, after - before)
    removed = _outside(storage, before - after)
    assert not created, f"paths created outside {storage.resolve()}: {created}"
    assert not removed, f"paths removed outside {storage.resolve()}: {removed}"


def _identifier(kind: str, sandbox: Path) -> str:
    if kind == "parent":
        return "../outside/item"
    if kind == "dot_parent":
        return "./../outside/item"
    if kind == "nested_parent":
        return "nested/../../outside/item"
    if kind == "storage_parent":
        return ".."
    if kind == "absolute":
        return str(sandbox / "absolute_target")
    raise AssertionError(kind)


def _cache_tensor() -> torch.Tensor:
    return torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)


def _planted_tensor() -> torch.Tensor:
    return torch.tensor([[9.0, 8.0, 7.0]], dtype=torch.float32)


def _plant_outside_cache(storage: Path, identifier: str) -> None:
    filename = os.path.join(
        os.path.join(str(storage), identifier), "encoder_cache.safetensors"
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    safetensors.torch.save_file({"ec_cache": _planted_tensor()}, filename)


def _loaded_planted(cache: dict[str, torch.Tensor], identifier: str) -> bool:
    loaded = cache.get(identifier)
    return loaded is not None and torch.equal(loaded, _planted_tensor())


def test_safe_identifier_round_trips_inside_storage(
    connector: ECExampleConnector, storage: Path, tmp_path: Path
) -> None:
    """A normal identifier is saved, found, and loaded under the storage root."""
    identifier = "mm-item-1"
    tensor = _cache_tensor()
    before = _resolved_tree(tmp_path)

    connector.save_caches({identifier: tensor}, identifier)
    assert connector.has_cache_item(identifier)

    metadata = ECExampleConnectorMetadata()
    metadata.add_mm_data(MMMeta.make_meta(identifier, 1))
    connector.bind_connector_metadata(metadata)
    loaded: dict[str, torch.Tensor] = {}
    connector.start_load_caches(loaded)

    assert torch.equal(loaded[identifier], tensor)
    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize("identifier_kind", _IDENTIFIER_KINDS)
def test_save_caches_does_not_write_outside_storage(
    connector: ECExampleConnector,
    storage: Path,
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    """``save_caches`` must not create the escaped directory or tensor file."""
    assert connector.is_producer
    identifier = _identifier(identifier_kind, tmp_path)
    before = _resolved_tree(tmp_path)

    _run(lambda: connector.save_caches({identifier: _cache_tensor()}, identifier))

    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize("identifier_kind", _IDENTIFIER_KINDS)
def test_start_load_caches_does_not_read_or_create_outside_storage(
    connector: ECExampleConnector,
    storage: Path,
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    """A consumer must not load a planted file, or mkdir, outside storage."""
    identifier = _identifier(identifier_kind, tmp_path)
    _plant_outside_cache(storage, identifier)
    before = _resolved_tree(tmp_path)
    metadata = ECExampleConnectorMetadata()
    metadata.add_mm_data(MMMeta.make_meta(identifier, 4))
    connector.bind_connector_metadata(metadata)
    cache: dict[str, torch.Tensor] = {}

    _run(lambda: connector.start_load_caches(cache))

    assert not _loaded_planted(cache, identifier)
    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize("identifier_kind", _NEW_DIRECTORY_KINDS)
def test_missing_load_does_not_create_escaped_directory(
    connector: ECExampleConnector,
    storage: Path,
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    """Refusing a missing cache must happen before any directory is created."""
    identifier = _identifier(identifier_kind, tmp_path)
    before = _resolved_tree(tmp_path)
    metadata = ECExampleConnectorMetadata()
    metadata.add_mm_data(MMMeta.make_meta(identifier, 4))
    connector.bind_connector_metadata(metadata)

    _run(lambda: connector.start_load_caches({}))

    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize("identifier_kind", _IDENTIFIER_KINDS)
def test_has_cache_item_ignores_files_outside_storage(
    scheduler_connector: ECExampleConnector,
    storage: Path,
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    """An outside file is not a cache hit, and the check creates nothing."""
    identifier = _identifier(identifier_kind, tmp_path)
    _plant_outside_cache(storage, identifier)
    before = _resolved_tree(tmp_path)

    assert scheduler_connector.has_cache_item(identifier) is False
    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize("identifier_kind", _IDENTIFIER_KINDS)
def test_update_state_after_alloc_does_not_schedule_outside_file(
    scheduler_connector: ECExampleConnector,
    storage: Path,
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    """The scheduler must not queue a load of a file outside storage."""
    assert scheduler_connector.is_consumer
    identifier = _identifier(identifier_kind, tmp_path)
    _plant_outside_cache(storage, identifier)
    before = _resolved_tree(tmp_path)
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier=identifier,
                mm_position=PlaceholderRange(offset=0, length=4),
            )
        ],
    )

    scheduler_connector.update_state_after_alloc(request, 0)
    metadata = scheduler_connector.build_connector_meta(SchedulerOutput.make_empty())

    assert isinstance(metadata, ECExampleConnectorMetadata)
    assert all(item.mm_hash != identifier for item in metadata.mm_datas)
    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize(
    ("lora_config", "lora_request"),
    [
        (None, None),
        (
            LoRAConfig(enable_tower_connector_lora=False),
            LoRARequest("adapter", 1, "/tmp/adapter-placeholder"),
        ),
    ],
)
def test_empty_processor_kwargs_keep_uuid_off_the_filesystem(
    connector: ECExampleConnector,
    storage: Path,
    tmp_path: Path,
    lora_config: LoRAConfig | None,
    lora_request: LoRARequest | None,
) -> None:
    """Chat uuid with empty media and processor kwargs must not be a path.

    ``ProcessorInputs.get_mm_hashes`` keeps that uuid when both factor maps
    are empty, and ``InputProcessor._get_mm_identifier`` returns it unchanged
    unless tower-connector LoRA is enabled. Saving that identifier must not
    leave the storage root.
    """
    uuid = "../outside/from-chat-uuid"
    image = Image.new("RGB", (8, 8), color=(1, 2, 3))
    mm_data_items = MultiModalDataParser().parse_mm_data({"image": [image]})
    hashes = ProcessorInputs(
        prompt=[],
        mm_data_items=mm_data_items,
        mm_uuid_items={"image": [uuid]},
        hf_processor_mm_kwargs={},
        media_io_kwargs={},
    ).get_mm_hashes("example-multimodal", "blake3")
    identifier = InputProcessor._get_mm_identifier(
        cast(InputProcessor, _IdentifierState(lora_config)),
        hashes["image"][0],
        lora_request,
    )
    before = _resolved_tree(tmp_path)

    _run(lambda: connector.save_caches({identifier: _cache_tensor()}, identifier))

    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize("identifier_kind", _IDENTIFIER_KINDS)
def test_gpu_worker_save_does_not_escape_storage(
    worker_transfer: ECExampleConnector,
    vllm_config: VllmConfig,
    storage: Path,
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    """New encoder-cache keys saved after a GPU step stay inside storage."""
    assert worker_transfer.is_producer
    identifier = _identifier(identifier_kind, tmp_path)
    before = _resolved_tree(tmp_path)
    encoder_cache: dict[str, torch.Tensor] = {}
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.ec_connector_metadata = ECExampleConnectorMetadata()
    worker = ActiveECConnector(vllm_config, encoder_cache)

    def _step() -> None:
        with worker.maybe_get_output(scheduler_output):
            encoder_cache[identifier] = _cache_tensor()

    _run(_step)

    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize("identifier_kind", _IDENTIFIER_KINDS)
def test_model_runner_mixin_save_does_not_escape(
    worker_transfer: ECExampleConnector,
    storage: Path,
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    """``maybe_save_ec_to_connector`` must not write outside storage."""
    assert worker_transfer.is_producer
    identifier = _identifier(identifier_kind, tmp_path)
    before = _resolved_tree(tmp_path)

    _run(
        lambda: ECConnectorModelRunnerMixin.maybe_save_ec_to_connector(
            {identifier: _cache_tensor()}, identifier
        )
    )

    _assert_storage_boundary_unchanged(storage, tmp_path, before)


@pytest.mark.parametrize("identifier_kind", _IDENTIFIER_KINDS)
def test_model_runner_mixin_load_does_not_escape(
    worker_transfer: ECExampleConnector,
    storage: Path,
    tmp_path: Path,
    identifier_kind: str,
) -> None:
    """The V1 runner load path must not import a file from outside storage."""
    assert worker_transfer.is_consumer
    identifier = _identifier(identifier_kind, tmp_path)
    _plant_outside_cache(storage, identifier)
    before = _resolved_tree(tmp_path)
    metadata = ECExampleConnectorMetadata()
    metadata.add_mm_data(MMMeta.make_meta(identifier, 4))
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.ec_connector_metadata = metadata
    loaded: dict[str, torch.Tensor] = {}

    def _load() -> None:
        with ECConnectorModelRunnerMixin._get_ec_connector_output(
            scheduler_output, loaded
        ):
            return

    _run(_load)

    assert not _loaded_planted(loaded, identifier)
    _assert_storage_boundary_unchanged(storage, tmp_path, before)
