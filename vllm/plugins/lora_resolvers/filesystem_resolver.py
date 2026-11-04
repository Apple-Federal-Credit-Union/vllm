# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import stat
from typing import Any

import vllm.envs as envs
from vllm.lora.adapter_path import open_adapter_dir, resolve_adapter_dir
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry


class FilesystemResolver(LoRAResolver):
    def __init__(self, lora_cache_dir: str):
        self.lora_cache_dir = os.path.abspath(lora_cache_dir)

    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None:
        """Resolves `lora_name` to an adapter directory inside the cache
        directory. Names that are absolute, contain parent-directory segments,
        or resolve through symlinks to a directory outside the cache are not
        resolved.
        """
        cache_dir = os.path.realpath(self.lora_cache_dir)
        lora_path = resolve_adapter_dir(cache_dir, lora_name)
        if lora_path is None:
            return None
        try:
            dir_fd = open_adapter_dir(cache_dir, lora_path)
        except (OSError, ValueError):
            return None
        try:
            adapter_config = _read_adapter_config(dir_fd)
        finally:
            os.close(dir_fd)
        if not _is_lora_for_base_model(adapter_config, base_model_name):
            return None
        return LoRARequest(
            lora_name=lora_name,
            lora_int_id=abs(hash(lora_name)),
            lora_path=lora_path,
            lora_path_root=cache_dir,
        )

    async def _get_lora_req_from_path(
        self, lora_name: str, lora_path: str, base_model_name: str
    ) -> LoRARequest | None:
        """Builds a LoraRequest pointing to the lora path if it's a valid
        LoRA adapter and has a matching base_model_name.
        """
        if os.path.exists(lora_path):
            adapter_config_path = os.path.join(lora_path, "adapter_config.json")

            if os.path.exists(adapter_config_path):
                with open(adapter_config_path) as file:
                    adapter_config = json.load(file)
                if _is_lora_for_base_model(adapter_config, base_model_name):
                    lora_request = LoRARequest(
                        lora_name=lora_name,
                        lora_int_id=abs(hash(lora_name)),
                        lora_path=lora_path,
                    )
                    return lora_request
        return None


def _read_adapter_config(dir_fd: int) -> Any:
    try:
        fd = os.open(
            "adapter_config.json",
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=dir_fd,
        )
    except OSError:
        return None
    with os.fdopen(fd) as file:
        if not stat.S_ISREG(os.fstat(file.fileno()).st_mode):
            return None
        try:
            return json.load(file)
        except ValueError:
            return None


def _is_lora_for_base_model(adapter_config: Any, base_model_name: str) -> bool:
    return (
        isinstance(adapter_config, dict)
        and adapter_config.get("peft_type") == "LORA"
        and adapter_config.get("base_model_name_or_path") == base_model_name
    )


def register_filesystem_resolver():
    """Register the filesystem LoRA Resolver with vLLM."""
    lora_cache_dir = envs.VLLM_LORA_RESOLVER_CACHE_DIR
    if lora_cache_dir:
        if not os.path.exists(lora_cache_dir) or not os.path.isdir(lora_cache_dir):
            raise ValueError(
                "VLLM_LORA_RESOLVER_CACHE_DIR must be set to a valid directory \
                for Filesystem Resolver plugin to function"
            )
        fs_resolver = FilesystemResolver(lora_cache_dir)
        LoRAResolverRegistry.register_resolver("Filesystem Resolver", fs_resolver)

    return
