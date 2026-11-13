# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings Base Class."""

import math

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.custom_op import CustomOp

from .common import ApplyRotaryEmb

# Positions are materialized as float32, which stops representing every
# integer at 2**24; no real context window comes close to it.
MAX_COS_SIN_CACHE_POSITIONS = 1 << 24


def get_max_model_len() -> int | None:
    """The largest token position + 1 the engine can feed, if configured."""
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return None
    max_model_len = getattr(vllm_config.model_config, "max_model_len", None)
    if max_model_len is None:
        return None
    # Drafts built under their own ModelConfig are fed the target's positions.
    spec_config = getattr(vllm_config, "speculative_config", None)
    target_config = getattr(spec_config, "target_model_config", None)
    target_max_model_len = getattr(target_config, "max_model_len", None)
    if target_max_model_len is not None:
        max_model_len = max(max_model_len, target_max_model_len)
    return max_model_len


def get_cos_sin_cache_len(num_positions: float, max_cache_len: int | None) -> int:
    """Number of cos/sin cache rows to build for `num_positions` positions.

    `num_positions` comes from the checkpoint (e.g. `max_position_embeddings`
    times a rope scaling factor), so it is validated before anything is
    allocated, then capped at `max_cache_len` since rows past the largest
    position the engine can feed are never read.

    Raises:
        ValueError: If `num_positions` is not in
            `(0, MAX_COS_SIN_CACHE_POSITIONS)`.

    """
    if not 0 < num_positions < MAX_COS_SIN_CACHE_POSITIONS:
        raise ValueError(
            f"RoPE needs {num_positions} cos/sin cache positions, derived from "
            "the model config's max_position_embeddings and rope scaling "
            f"factor; expected a value in (0, {MAX_COS_SIN_CACHE_POSITIONS})."
        )
    num_rows = math.ceil(num_positions)
    if max_cache_len is not None:
        num_rows = min(num_rows, max_cache_len)
    return num_rows


# --8<-- [start:rotary_embedding]
@CustomOp.register("rotary_embedding")
class RotaryEmbeddingBase(CustomOp):
    """Original rotary positional embedding."""

    # --8<-- [end:rotary_embedding]

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
        bound_cache_by_max_model_len: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        # Captured once: some subclasses rebuild the cache outside of a
        # set_current_vllm_config() context and must keep the same length.
        self.max_model_len = (
            get_max_model_len() if bound_cache_by_max_model_len else None
        )
        # TODO(mgoin): disabled for now due to failures
        # Flashinfer only supports head_size=64, 128, 256, 512.
        # https://github.com/flashinfer-ai/flashinfer/blob/ebfd655efe830048dba5d582aaa61d61d1cf9a87/include/flashinfer/utils.cuh#L174-L202
        # self.use_flashinfer = (self.enabled()
        #                        and dtype in (torch.float16, torch.bfloat16)
        #                        and current_platform.is_cuda()
        #                        and has_flashinfer()
        #                        and self.head_size in [64, 128, 256, 512])

        # Check if use_flashinfer is already set
        if not hasattr(self, "use_flashinfer"):
            self.use_flashinfer = False

        self.use_aiter = (
            self.enabled() and rocm_aiter_ops.is_triton_rotary_embed_enabled()
        )
        if self.use_aiter:
            self.rocm_aiter_triton_rotary_embedding = (
                rocm_aiter_ops.get_triton_rotary_embedding_op()
            )

        if init_cache:
            cache = self._compute_cos_sin_cache()
            if not self.use_flashinfer:
                cache = cache.to(dtype)
            self.cos_sin_cache: torch.Tensor
            self.register_buffer("cos_sin_cache", cache, persistent=False)

            # Reuse a precomputed bf16 cache for the AITER compile path.
            if self.use_aiter and cache.dtype != torch.bfloat16:
                self.cos_sin_cache_bf16: torch.Tensor | None
                self.register_buffer(
                    "cos_sin_cache_bf16",
                    cache.to(torch.bfloat16),
                    persistent=False,
                )
            else:
                self.cos_sin_cache_bf16 = None

        self.apply_rotary_emb = ApplyRotaryEmb(
            is_neox_style=self.is_neox_style,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _max_cache_len(self) -> int | None:
        """Upper bound on the positions this embedding is indexed with."""
        return self.max_model_len

    def _cos_sin_cache_len(self, num_positions: float) -> int:
        return get_cos_sin_cache_len(num_positions, self._max_cache_len())

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(
            self._cos_sin_cache_len(self.max_position_embeddings), dtype=torch.float
        )

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> torch.Tensor:
        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
        # is expensive, so avoid calling it if possible
        cos_sin_cache = self.cos_sin_cache
        if (
            cos_sin_cache.device == query.device
            and self.cos_sin_cache.dtype == query.dtype
        ):
            return cos_sin_cache

        # Reuse precomputed bf16 cache in the AITER compile path.
        if (
            self.use_aiter
            and torch.compiler.is_compiling()
            and query.dtype == torch.bfloat16
        ):
            cache_bf16 = getattr(self, "cos_sin_cache_bf16", None)
            if cache_bf16 is not None and cache_bf16.device == query.device:
                return cache_bf16

        cos_sin_cache = cos_sin_cache.to(query.device, dtype=query.dtype)
        # Avoid mutating buffers during torch.compile (cudagraph) tracing.
        if torch.compiler.is_compiling():
            return cos_sin_cache

        self.cos_sin_cache = cos_sin_cache
        return cos_sin_cache

    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[:seqlen]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return cos, sin


class RotaryEmbedding(RotaryEmbeddingBase):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
        bound_cache_by_max_model_len: bool = True,
    ) -> None:
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
            init_cache=init_cache,
            bound_cache_by_max_model_len=bound_cache_by_max_model_len,
        )

    @staticmethod
    def forward_static(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A PyTorch-native implementation of forward()."""
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, head_size)
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        query_rot = ApplyRotaryEmb.forward_static(
            query_rot,
            cos,
            sin,
            is_neox_style,
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, head_size)
            key_rot = key[..., :rotary_dim]
            key_pass = key[..., rotary_dim:]
            key_rot = ApplyRotaryEmb.forward_static(
                key_rot,
                cos,
                sin,
                is_neox_style,
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """A PyTorch-native implementation of forward()."""
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        return self.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
        )

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_flashinfer:
            torch.ops.vllm.flashinfer_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
            return query, key

        from vllm import _custom_ops as ops

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)

        # ops.rotary_embedding() is an in-place operation
        # that updates the query and key tensors.
        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_aiter:
            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            self.rocm_aiter_triton_rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
            return query, key
        return self.forward_cuda(positions, query, key)

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._match_cos_sin_cache_dtype(query)
        # ops.rotary_embedding() is an in-place operation
        # that updates the query and key tensors.
        if key is None:
            return self.forward_native(positions, query, key)
        else:
            from vllm import _custom_ops as ops

            cos_sin_cache = self._match_cos_sin_cache_dtype(query)
            ops.rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                cos_sin_cache,
                self.is_neox_style,
            )
        return query, key

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from vllm import _custom_ops as ops

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)

        # ops.rotary_embedding() is an in-place operation
        # that updates the query and key tensors.
        ops.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s
