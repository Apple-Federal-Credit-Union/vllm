# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import regex as re

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

if TYPE_CHECKING:
    import torch

    from .inc import INCConfig

logger = init_logger(__name__)

_REGEX_SPECIAL_CHARS = frozenset(r"*+?^$()[]{}|\\")
_MAX_PATTERN_LENGTH = 1024
# The regex module unrolls counted repeats when compiling, so their bounds
# multiply into compile time and memory.
_COUNTED_REPEAT = re.compile(r"\{\s*(\d*)\s*(?:,\s*(\d*)\s*)?\}")
_MAX_COUNTED_REPEAT_PRODUCT = 1000
_PATTERN_SEARCH_TIMEOUT_S = 0.1
_PATTERN_BUDGET_S = 5.0


def _exceeds_counted_repeat_limit(pattern: str) -> bool:
    product = 1
    for low, high in _COUNTED_REPEAT.findall(pattern):
        product *= max(int(low or 0), int(high or 0), 1)
        if product > _MAX_COUNTED_REPEAT_PRODUCT:
            return True
    return False


class _ExtraConfigPatterns:
    """Regex keys of one extra_config, compiled once.

    Keys come from the checkpoint, so their size is capped, every search has a
    timeout, and compiling and searching share one time budget. A key that
    times out is dropped; exhausting the budget aborts loading.
    """

    def __init__(self, extra_config: dict[str, Any]) -> None:
        self._patterns: dict[str, re.Pattern] = {}
        self._budget_s = _PATTERN_BUDGET_S
        for key in extra_config:
            if not isinstance(key, str) or not any(
                c in _REGEX_SPECIAL_CHARS for c in key
            ):
                continue
            if len(key) > _MAX_PATTERN_LENGTH or _exceeds_counted_repeat_limit(key):
                logger.warning(
                    "Ignoring INC extra_config regex key: it is longer than %d "
                    "characters or its counted repeats exceed %d.",
                    _MAX_PATTERN_LENGTH,
                    _MAX_COUNTED_REPEAT_PRODUCT,
                )
                continue
            start = time.perf_counter()
            with contextlib.suppress(re.error, RecursionError, MemoryError):
                self._patterns[key] = re.compile(key)
            self._charge(time.perf_counter() - start)

    def _charge(self, elapsed_s: float) -> None:
        self._budget_s -= elapsed_s
        if self._budget_s <= 0:
            raise ValueError(
                "Compiling and matching INC extra_config regex keys exceeded "
                f"{_PATTERN_BUDGET_S}s. Use exact layer names or simpler "
                "patterns in extra_config."
            )

    def first_match(self, name: str) -> str | None:
        for key, pattern in list(self._patterns.items()):
            timeout = min(_PATTERN_SEARCH_TIMEOUT_S, self._budget_s)
            start = time.perf_counter()
            try:
                matched = pattern.search(name, timeout=timeout) is not None
            except TimeoutError:
                matched = False
                del self._patterns[key]
                logger.warning(
                    "Ignoring INC extra_config key %r: matching it against "
                    "%r exceeded %.2fs.",
                    key,
                    name,
                    timeout,
                )
            self._charge(time.perf_counter() - start)
            if matched:
                return key
        return None


@dataclass(frozen=True)
class INCLayerConfig:
    bits: int
    group_size: int | tuple[int, int]
    sym: bool
    packing_format: str
    backend: str
    data_type: str
    quantized: bool

    @property
    def is_gptq(self) -> bool:
        return "gptq" in self.packing_format or "gptq" in self.backend

    @property
    def is_awq(self) -> bool:
        return "awq" in self.packing_format or "awq" in self.backend

    @property
    def is_wna16_int(self) -> bool:
        return self.data_type == "int" and self.quantized

    @property
    def is_mxfp4(self) -> bool:
        return "mx_fp" in self.data_type and self.bits == 4

    @property
    def is_mxfp8(self) -> bool:
        return "mx_fp" in self.data_type and self.bits == 8

    @property
    def is_fp8_block(self) -> bool:
        return (
            self.data_type == "fp"
            and self.bits == 8
            and self.sym
            and self.quantized
            and self.packing_format == "auto_round:fp8"
            and isinstance(self.group_size, tuple)
            and len(self.group_size) == 2
        )


class INCConfigParser:
    def __init__(self, config: "INCConfig") -> None:
        self._config = config
        self._patterns_source: dict[str, Any] | None = None
        self._patterns = _ExtraConfigPatterns({})

    def _extra_config_patterns(self) -> _ExtraConfigPatterns:
        # apply_vllm_mapper replaces extra_config after construction.
        extra_config = self._config.extra_config
        if extra_config is not self._patterns_source:
            self._patterns = _ExtraConfigPatterns(extra_config or {})
            self._patterns_source = extra_config
        return self._patterns

    def resolve(self, layer: "torch.nn.Module", layer_name: str) -> INCLayerConfig:
        bits, group_size, sym = self._resolve_raw(layer, layer_name)
        return INCLayerConfig(
            bits=bits,
            group_size=group_size,
            sym=sym,
            packing_format=self._config.packing_format,
            backend=self._config.backend,
            data_type=self._config.data_type,
            quantized=bits < 16,
        )

    def get_layer_config(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int | tuple[int, int], bool]:
        layer_config = self.resolve(layer, layer_name)
        return layer_config.bits, layer_config.group_size, layer_config.sym

    def _resolve_raw(
        self, layer: "torch.nn.Module", layer_name: str
    ) -> tuple[int, int | tuple[int, int], bool]:
        patterns = self._extra_config_patterns()

        def is_explicitly_configured(name: str) -> bool:
            """Return True if *name* has an explicit entry in extra_config,
            either via exact key match or via a regex pattern key."""
            if not self._config.extra_config:
                return False
            if name in self._config.extra_config:
                return True
            return patterns.first_match(name) is not None

        def get_config(
            name: str, quantized: bool = True
        ) -> tuple[int, int | tuple[int, int], bool]:
            if not self._config.extra_config:
                return (
                    self._config.weight_bits if quantized else 16,
                    self._config.group_size if quantized else -1,
                    self._config.sym if quantized else True,
                )

            if name in self._config.extra_config:
                cfg = self._config.extra_config[name]
                return (
                    cfg.get("bits", self._config.weight_bits if quantized else 16),
                    self._normalize_group_size(
                        cfg.get(
                            "group_size",
                            self._config.group_size if quantized else -1,
                        )
                    ),
                    cfg.get("sym", self._config.sym if quantized else True),
                )

            pattern = patterns.first_match(name)
            if pattern is not None:
                cfg = self._config.extra_config[pattern]
                return (
                    cfg.get("bits", self._config.weight_bits if quantized else 16),
                    self._normalize_group_size(
                        cfg.get(
                            "group_size",
                            self._config.group_size if quantized else -1,
                        )
                    ),
                    cfg.get("sym", self._config.sym if quantized else True),
                )

            return (
                self._config.weight_bits if quantized else 16,
                self._config.group_size if quantized else -1,
                self._config.sym if quantized else True,
            )

        if self._config.extra_config and layer_name in self._config.extra_config:
            return get_config(layer_name)

        # Suffix match: handle cases where extra_config keys use short names
        # (e.g. "lm_head") but the layer_name is fully qualified
        # (e.g. "model.language_model.lm_head") due to model nesting.
        if self._config.extra_config:
            for cfg_key in self._config.extra_config:
                if layer_name.endswith(f".{cfg_key}"):
                    return get_config(cfg_key)

        quantized = not isinstance(layer, ParallelLMHead)
        if self._config.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name)
                for name in self._config.block_name_to_quantize
            )

        if self._config.extra_config and "fusedmoe" in layer.__class__.__name__.lower():
            moe_configs = [
                get_config(name, quantized)
                for name in self._config.extra_config
                if name.startswith(layer_name)
            ]
            if moe_configs:
                if len(set(moe_configs)) == 1:
                    return moe_configs[0]
                raise ValueError(
                    f"Fused MoE layer '{layer_name}' requires "
                    f"consistent quant config for all sub-layers"
                )

        if self._config.extra_config:
            for fusion_key, sub_keys in self._config.packed_modules_mapping.items():
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                    sub_names = [
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                    ]
                    # Only trigger if at least one sub_name is explicitly
                    # configured in extra_config (via exact match or regex).
                    # This prevents false matches when a short fusion_key
                    # (e.g. "qkv") is merely a substring of a longer layer
                    # name (e.g. "in_proj_qkvz") and none of the generated
                    # sub_names are actually configured.
                    if not any(is_explicitly_configured(n) for n in sub_names):
                        continue
                    sub_configs = [get_config(name, quantized) for name in sub_names]
                    if len(set(sub_configs)) == 1:
                        return sub_configs[0]
                    raise ValueError(
                        f"Fused module '{layer_name}' requires "
                        f"consistent quant config for {sub_names}"
                    )

        return get_config(layer_name, quantized)

    @staticmethod
    def _normalize_group_size(
        group_size: int | list[int] | tuple[int, int],
    ) -> int | tuple[int, int]:
        """Normalize INC group_size into either an int or a 2-D int tuple."""
        if isinstance(group_size, (list, tuple)):
            if len(group_size) != 2 or not all(
                isinstance(value, int) for value in group_size
            ):
                raise ValueError(
                    "INC block-wise FP8 requires group_size to be a 2-D "
                    f"integer sequence, but found {group_size!r}."
                )
            return (group_size[0], group_size[1])

        if not isinstance(group_size, int):
            raise ValueError(
                "INC group_size must be an int or a 2-D integer sequence, "
                f"but found {type(group_size).__name__}."
            )

        return group_size
