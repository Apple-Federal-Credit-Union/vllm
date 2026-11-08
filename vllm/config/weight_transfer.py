# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class WeightTransferConfig:
    """Configuration for weight transfer during RL training."""

    backend: Literal["nccl", "ipc", "sparse_nccl", "sharded_rdt", "nccl_m2n"] | str = (
        "nccl"
    )
    """The backend to use for weight transfer. Validated against the
    `WeightTransferEngineFactory` registry at engine creation time.
    """

    master_address: str | None = None
    """Address of the trainer's TCPStore for the `nccl`, `sparse_nccl` and
    `nccl_m2n` backends. Workers only join this store; the `master_address` in
    `init_weight_transfer_engine` must match it. Required for TCPStore
    rendezvous."""

    master_port: int | None = Field(default=None, ge=1, le=65535)
    """Port of the trainer's TCPStore. See `master_address`."""
