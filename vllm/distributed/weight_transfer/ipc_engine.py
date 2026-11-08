# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC-based weight transfer engine using CUDA IPC for communication."""

import binascii
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

import pybase64 as base64
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    PackedBufferImporter,
    packed_ipc_consumer,
    packed_ipc_producer,
)

# Positional args of `rebuild_cuda_tensor`, as produced by `reduce_tensor`.
_IPC_ARG_FIELDS = (
    "tensor_cls",
    "tensor_size",
    "tensor_stride",
    "tensor_offset",
    "storage_cls",
    "dtype",
    "storage_device",
    "storage_handle",
    "storage_size_bytes",
    "storage_offset_bytes",
    "requires_grad",
    "ref_counter_handle",
    "ref_counter_offset",
    "event_handle",
    "event_sync_required",
)
_IPC_INT_FIELDS = (
    "tensor_offset",
    "storage_device",
    "storage_size_bytes",
    "storage_offset_bytes",
    "ref_counter_offset",
)
_IPC_BOOL_FIELDS = ("requires_grad", "event_sync_required")
_IPC_BYTES_FIELDS = ("storage_handle", "ref_counter_handle", "event_handle")
_IPC_TENSOR_CLASSES: dict[str, type] = {
    "Tensor": torch.Tensor,
    "Parameter": torch.nn.Parameter,
}
_IPC_STORAGE_CLASSES: dict[str, type] = {
    "TypedStorage": torch.storage.TypedStorage,
    "UntypedStorage": torch.UntypedStorage,
}
_TORCH_DTYPES: dict[str, torch.dtype] = {
    str(dtype).removeprefix("torch."): dtype
    for dtype in vars(torch).values()
    if isinstance(dtype, torch.dtype)
}


def _class_key(cls: type, allowed: dict[str, type]) -> str:
    for key, allowed_cls in allowed.items():
        if cls is allowed_cls:
            return key
    raise ValueError(f"Unsupported class in IPC handle: {cls!r}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def ipc_handle_args_to_json(args: Sequence[Any]) -> dict[str, Any]:
    """Encode CUDA `reduce_tensor` rebuild args as JSON-native fields."""
    fields = dict(zip(_IPC_ARG_FIELDS, args, strict=True))
    fields["tensor_cls"] = _class_key(fields["tensor_cls"], _IPC_TENSOR_CLASSES)
    fields["storage_cls"] = _class_key(fields["storage_cls"], _IPC_STORAGE_CLASSES)
    fields["dtype"] = str(fields["dtype"]).removeprefix("torch.")
    fields["tensor_size"] = list(fields["tensor_size"])
    fields["tensor_stride"] = list(fields["tensor_stride"])
    for name in _IPC_BYTES_FIELDS:
        if fields[name] is not None:
            fields[name] = base64.b64encode(fields[name]).decode("ascii")
    return fields


def ipc_handle_args_from_json(obj: dict[str, Any]) -> tuple:
    """Validate `ipc_handle_args_to_json` output and rebuild the args tuple."""
    if set(obj) != set(_IPC_ARG_FIELDS):
        raise ValueError(
            f"IPC handle must have exactly the fields {list(_IPC_ARG_FIELDS)}, "
            f"got {sorted(obj)}"
        )
    fields = dict(obj)
    for name, allowed in (
        ("tensor_cls", _IPC_TENSOR_CLASSES),
        ("storage_cls", _IPC_STORAGE_CLASSES),
    ):
        key = fields[name]
        if not isinstance(key, str) or key not in allowed:
            raise ValueError(f"IPC handle `{name}` must be one of {list(allowed)}")
        fields[name] = allowed[key]
    dtype = fields["dtype"]
    if not isinstance(dtype, str) or dtype not in _TORCH_DTYPES:
        raise ValueError(f"IPC handle `dtype` is not a torch dtype: {dtype!r}")
    fields["dtype"] = _TORCH_DTYPES[dtype]
    for name in ("tensor_size", "tensor_stride"):
        dims = fields[name]
        if not isinstance(dims, list) or not all(_is_int(d) for d in dims):
            raise ValueError(f"IPC handle `{name}` must be a list of ints")
    fields["tensor_size"] = torch.Size(fields["tensor_size"])
    fields["tensor_stride"] = tuple(fields["tensor_stride"])
    for name in _IPC_INT_FIELDS:
        if not _is_int(fields[name]):
            raise ValueError(f"IPC handle `{name}` must be an int")
    for name in _IPC_BOOL_FIELDS:
        if not isinstance(fields[name], bool):
            raise ValueError(f"IPC handle `{name}` must be a bool")
    for name in _IPC_BYTES_FIELDS:
        value = fields[name]
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"IPC handle `{name}` must be a base64 string")
        try:
            fields[name] = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"IPC handle `{name}` is not valid base64") from e
    return tuple(fields[name] for name in _IPC_ARG_FIELDS)


def ipc_handles_to_json(
    ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
) -> list[dict[str, dict[str, Any]]] | dict[str, dict[str, Any]]:
    """Encode `ipc_handles` for JSON transports such as HTTP."""

    def encode(handle: dict[str, tuple]) -> dict[str, dict[str, Any]]:
        return {uuid: ipc_handle_args_to_json(args) for uuid, args in handle.items()}

    if isinstance(ipc_handles, list):
        return [encode(handle) for handle in ipc_handles]
    return encode(ipc_handles)


def _decode_ipc_handle(handle: Any) -> dict[str, tuple]:
    if not isinstance(handle, dict):
        raise ValueError("Each IPC handle must map GPU UUIDs to rebuild args")
    return {
        gpu_uuid: ipc_handle_args_from_json(args) if isinstance(args, dict) else args
        for gpu_uuid, args in handle.items()
    }


@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for IPC weight transfer. No rendezvous needed.

    `packed` is a must-agree wire param: the trainer ships it here at the init
    handshake so the worker decodes with the same setting the trainer encoded
    with. The consumer rebuilds from the IPC handle + `tensor_sizes`, so it does
    not need the buffer size (producer-only)."""

    packed: bool = False


@dataclass
class IPCTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for IPC weight transfer. No rendezvous needed;
    `rank` (from `TrainerInitInfo`) identifies this trainer process — rank 0
    ships the merged IPC handles. All ranks still join the handle all-gather.

    `packed` / `packed_buffer_size_bytes` are the transfer's wire params. The
    trainer propagates `packed` to the worker at `trainer_init` so the two sides
    cannot disagree. `backend` is the factory dispatch key."""

    backend: ClassVar[str] = "ipc"
    packed: bool = False
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES


@dataclass
class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the IPC weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, Any]] | dict[str, Any] | None = None
    """IPC handles mapping physical GPU UUID to rebuild_cuda_tensor args.
    For non-packed mode: list of per-parameter handle dicts.
    For packed mode: single handle dict for the packed buffer.
    Over JSON transports each args tuple is sent in the typed form produced by
    `ipc_handle_args_to_json` and decoded here."""
    tensor_sizes: list[int] | None = None
    """Per-parameter sizes in bytes within the packed buffer.
    Required when packed=True, unused otherwise."""

    def __post_init__(self):
        if self.ipc_handles is None:
            raise ValueError("`ipc_handles` must be provided")
        if isinstance(self.ipc_handles, list):
            self.ipc_handles = [_decode_ipc_handle(h) for h in self.ipc_handles]
        else:
            self.ipc_handles = _decode_ipc_handle(self.ipc_handles)
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )
        # Unpacked transfers carry a per-parameter handle dict (a list); packed
        # transfers carry a single dict for the whole buffer, so the list check
        # only applies to the list case.
        if isinstance(self.ipc_handles, list) and len(self.ipc_handles) != num_params:
            raise ValueError(
                f"`ipc_handles` should be of the same size as `names`: "
                f"got {len(self.ipc_handles)} and {len(self.names)}"
            )


class IPCWeightTransferEngine(
    WeightTransferEngine[IPCWeightTransferInitInfo, IPCWeightTransferUpdateInfo]
):
    """Weight transfer engine using CUDA IPC between trainer and workers.

    This implementation uses CUDA IPC to transfer weights from the trainer (rank 0)
    to all inference workers in a process group. IPC handles are used to share
    memory between processes on the same node.
    """

    # Define backend-specific dataclass types
    init_info_cls = IPCWeightTransferInitInfo
    update_info_cls = IPCWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        # Set from the trainer-supplied init info at the handshake; defaults are
        # only for the (unreachable) receive-before-init case.
        self.packed = False
        # Shared across all chunks of every packed transfer this engine
        # receives; see PackedBufferImporter for the refcount contract.
        self._packed_importer = PackedBufferImporter()

    def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None:
        """Initialize the weight transfer mechanism. No data-plane rendezvous is
        needed for IPC; this just records the trainer-supplied wire params so the
        worker decodes exactly as the trainer encoded.

        Args:
            init_info: IPC initialization info (carries `packed`).

        """
        self.packed = init_info.packed

    def start_weight_update(self) -> None:
        """Initialize layerwise reloading for the incoming checkpoint weights."""
        from vllm.model_executor.model_loader.reload import (
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        """Finalize layerwise reloading after all weights have been received."""
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
        )

        finalize_layerwise_reload(self.model, self.model_config)
        # Every reduce_tensor call is a fresh export with its own refcount
        # slot, so releasing once per update always balances this update's
        # export and lets the trainer reclaim its staging buffer. Callers
        # that skip finish are still covered by the replace-on-next-export
        # path inside the importer.
        self._packed_importer.close()

    def receive_weights(self, update_info: IPCWeightTransferUpdateInfo) -> None:
        """Receive weights from the trainer via CUDA IPC handles and load them.

        Whether the transfer is packed is read from `self.packed`, set at the
        init handshake from the trainer's init info, so it is guaranteed to
        match how the trainer encoded.

        Args:
            update_info: IPC update info containing parameter names, dtypes, shapes,
                        and IPC handles. Each IPC handle is a mapping between physical
                        GPU UUID and the rebuild_cuda_tensor args tuple.

        """
        # Use the worker's assigned device rather than the ambient current
        # device: the receive path is no longer wrapped in
        # `with torch.device(self.device)` by the caller, so the current device
        # is not guaranteed to match self.device. The IPC tensors must be
        # rebuilt on the device the model lives on.
        device_index = self.device.index

        if self.packed:
            if update_info.tensor_sizes is None:
                raise ValueError("`tensor_sizes` is required when packed=True")
            assert isinstance(update_info.ipc_handles, dict)
            weights = packed_ipc_consumer(
                ipc_handle=update_info.ipc_handles,
                names=update_info.names,
                shapes=update_info.shapes,
                dtype_names=update_info.dtype_names,
                tensor_sizes=update_info.tensor_sizes,
                device_index=device_index,
                importer=self._packed_importer,
            )
        else:
            assert isinstance(update_info.ipc_handles, list)
            weights = []
            for name, ipc_handle in zip(
                update_info.names,
                update_info.ipc_handles,
            ):
                props = torch.cuda.get_device_properties(device_index)
                physical_gpu_id = str(props.uuid)

                if physical_gpu_id not in ipc_handle:
                    raise ValueError(
                        f"IPC handle not found for GPU UUID "
                        f"{physical_gpu_id}. "
                        f"Available UUIDs: {list(ipc_handle.keys())}"
                    )

                args = ipc_handle[physical_gpu_id]
                list_args = list(args)
                # Index 6 of the args from reduce_tensor is the device_index.
                # We need to overwrite it with the receiver's device index.
                list_args[6] = device_index
                weight = rebuild_cuda_tensor(*list_args)
                weights.append((name, weight))

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        with disable_mtp_completeness_check():
            self.model.load_weights(weights)

    def shutdown(self) -> None:
        self._packed_importer.close()


class IPCTrainerWeightTransferEngine(TrainerWeightTransferEngine[IPCTrainerInitInfo]):
    """Trainer-side CUDA IPC weight transfer engine.

    Called on every trainer rank. For multi-rank (e.g. FSDP) trainers all ranks
    drive the source's `iter_full()` (materializing each tensor) and contribute
    to the IPC-handle all-gather, so every rank must own every parameter; only
    the sender (rank 0) ships the merged handles to the inference side. IPC
    transfer is straight-line (no concurrent broadcast like NCCL):
    `update_weights` *is* the transfer, and it rides the client, so it no-ops
    on non-senders.

    `packed` / `packed_buffer_size_bytes` come from `IPCTrainerInitInfo`; the
    sender propagates `packed` to the worker at `trainer_init`.
    """

    init_info_cls = IPCTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        packed: bool = False,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.device_index = torch.accelerator.current_device_index()
        self.gpu_uuid = str(torch.cuda.get_device_properties(self.device_index).uuid)

    @classmethod
    def trainer_init(
        cls,
        init_info: IPCTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("IPC trainer weight transfer requires a WeightSource.")
        if not source.implements_iter_full():
            raise TypeError(
                f"IPC sends full tensors, but {type(source).__name__} does not "
                "implement iter_full()."
            )
        # Each rank exports its own copies for its colocated worker, merged by
        # position, so every rank must hold every parameter.
        if len(source.owned_names(init_info.rank)) != len(source.metadata):
            raise ValueError(
                f"rank {init_info.rank} does not own every parameter, but IPC "
                "needs every trainer rank to own every parameter."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
        )
        # IPC needs no data-plane rendezvous. The sender ships the must-agree
        # `packed` flag so the worker decodes exactly as this trainer encodes.
        if engine.is_sender:
            engine.client.init_weight_transfer_engine({"packed": init_info.packed})
        return engine

    def send_weights(self) -> None:
        assert self.source is not None
        source = self.source
        if self.is_sender:
            self.client.start_weight_update()
        # Unpacked returns strong refs to its IPC-shared copies; they must stay
        # alive across `finish_weight_update` too.
        weight_refs = self._send(source)
        if self.is_sender:
            self.client.finish_weight_update()
        self._post_send_sync()
        del weight_refs

    # ---- data plane (runs on all ranks; only the sender ships) ----

    def _send(self, source: WeightSource) -> list[torch.Tensor] | None:
        if self.packed:
            self._send_packed(source)
            return None
        return self._send_unpacked(source)

    def _all_gather_and_merge_handles(
        self,
        handles: list[dict[str, tuple]],
    ) -> list[dict[str, tuple]]:
        """All-gather and merge IPC handle dicts across ranks in one call.

        Each rank contributes a list of {gpu_uuid: ipc_args} dicts (one
        per parameter or one per chunk). A single all_gather_object
        collects every rank's full list, then the sender merges per-index so
        each dict maps every GPU UUID to its args.

        The all-gather runs over the *default* process group; this assumes the
        default group is exactly the set of colocated trainer ranks and that the
        sender is a member. No-op (returns handles unchanged) when no distributed
        group exists.
        """
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return handles

        world_size = torch.distributed.get_world_size()
        gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, handles)
        torch.distributed.barrier()
        torch.accelerator.synchronize()

        if self.is_sender:
            merged: list[dict[str, tuple]] = []
            for param_idx in range(len(handles)):
                m: dict[str, tuple] = {}
                for rank_handles in gathered:
                    if rank_handles is not None:
                        m.update(rank_handles[param_idx])
                merged.append(m)
            return merged
        return [{} for _ in handles]

    @staticmethod
    def _post_send_sync() -> None:
        """Barrier + ipc_collect after a send; no-op if single-GPU."""
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        torch.cuda.ipc_collect()

    def _send_unpacked(self, source: WeightSource) -> list[torch.Tensor]:
        """Iterate the source, build one IPC handle per param, all-gather the
        handles across ranks, and (sender) ship them in one update call.

        Returns the strong refs to every contiguous copy. reduce_tensor's args
        do NOT keep storage alive, and non-contiguous inputs allocate fresh
        storage in .contiguous(); the caller must keep these alive until the
        post-send barrier (past `finish`) so the consumer's IPC views stay valid.
        """
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles: list[dict[str, tuple]] = []
        weight_refs: list[torch.Tensor] = []

        for name, tensor in source.iter_full():
            # FSDP shards were gathered by the source; ensure contiguity here.
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

            weight = tensor.detach().contiguous()
            weight_refs.append(weight)
            _, ipc_args = reduce_tensor(weight)
            ipc_handles.append({self.gpu_uuid: ipc_args})

        ipc_handles = self._all_gather_and_merge_handles(ipc_handles)
        self._do_send(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            ipc_handles=ipc_handles,
        )
        return weight_refs

    def _send_packed(self, source: WeightSource) -> None:
        """Send weights in bounded-memory chunks (packed mode)."""
        for chunk in packed_ipc_producer(
            iterator=source.iter_full(),
            gpu_uuid=self.gpu_uuid,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=self.packed_buffer_size_bytes,
        ):
            ipc_handle = self._all_gather_and_merge_handles([chunk.ipc_handle])[0]
            self._do_send(
                names=chunk.names,
                dtype_names=chunk.dtype_names,
                shapes=chunk.shapes,
                ipc_handles=ipc_handle,
                tensor_sizes=chunk.tensor_sizes,
            )
            # Per-chunk barrier: the producer reuses a single IPC buffer across
            # chunks, but only the sender waits for the consumers (via _do_send).
            # Without syncing every rank here, non-sender ranks race ahead and
            # overwrite their buffer while their colocated worker is still
            # reading the current chunk, silently corrupting the transfer.
            self._post_send_sync()

    def _do_send(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
        tensor_sizes: list[int] | None = None,
    ) -> None:
        """Build one update payload and ship it via the client. Only the sender
        ships (non-sender ranks already contributed to the handle all-gather).

        Emits raw `ipc_handles`; transports that cannot carry them natively
        (HTTP/JSON) encode them with `ipc_handles_to_json` in their client (see
        `HTTPVLLMWeightSyncClient`).
        """
        if not self.is_sender:
            return
        update_fields: dict[str, Any] = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles": ipc_handles,
        }
        if tensor_sizes is not None:
            update_fields["tensor_sizes"] = tensor_sizes

        update_info = IPCWeightTransferUpdateInfo(**update_fields)
        self.client.update_weights(asdict(update_info))
