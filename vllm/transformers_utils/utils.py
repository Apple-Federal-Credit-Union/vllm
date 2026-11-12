# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import struct
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Same limit as `MAX_HEADER_SIZE` in the safetensors library.
SAFETENSORS_MAX_HEADER_SIZE = 100_000_000
_SAFETENSORS_LENGTH_PREFIX_SIZE = 8


def is_s3(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("s3://")


def is_gcs(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("gs://")


def is_azure(model_or_path: str) -> bool:
    return model_or_path.lower().startswith("az://")


def is_cloud_storage(model_or_path: str) -> bool:
    return is_s3(model_or_path) or is_gcs(model_or_path) or is_azure(model_or_path)


def without_trust_remote_code(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs without trust_remote_code without modifying original dict."""
    if "trust_remote_code" not in kwargs:
        return kwargs
    return {k: v for k, v in kwargs.items() if k != "trust_remote_code"}


def modelscope_list_repo_files(
    repo_id: str,
    revision: str | None = None,
    token: str | bool | None = None,
) -> list[str]:
    """List files in a modelscope repo."""
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token)

    # same as huggingface_hub.list_repo_files
    return [
        file["Path"]
        for file in api.get_model_files(
            model_id=repo_id, revision=revision, recursive=True
        )
    ]


def _maybe_json_dict(path: str | PathLike) -> dict[str, str]:
    with open(path) as f:
        try:
            return json.loads(f.read())
        except Exception:
            return dict[str, str]()


def _maybe_space_split_dict(path: str | PathLike) -> dict[str, str]:
    parsed_dict = dict[str, str]()
    with open(path) as f:
        for line in f.readlines():
            try:
                model_name, redirect_name = line.strip().split()
                parsed_dict[model_name] = redirect_name
            except Exception:
                pass
    return parsed_dict


@cache
def maybe_model_redirect(model: str) -> str:
    """Use model_redirect to redirect the model name to a local folder.

    Args:
        model: hf model name

    Returns:
        maybe redirect to a local folder

    """
    model_redirect_path = envs.VLLM_MODEL_REDIRECT_PATH

    if not model_redirect_path:
        return model

    if not Path(model_redirect_path).exists():
        return model

    redirect_dict = _maybe_json_dict(model_redirect_path) or _maybe_space_split_dict(
        model_redirect_path
    )
    if redirect_model := redirect_dict.get(model):
        logger.info("model redirect: [ %s ] -> [ %s ]", model, redirect_model)
        return redirect_model

    return model


def read_safetensors_header(f: BinaryIO) -> bytes:
    """Read the JSON header of a safetensors file opened in binary mode.

    The declared length is checked against the size of the open file and
    the safetensors header limit before anything is read, so a corrupt or
    hostile length prefix cannot force a large allocation.
    """
    prefix = f.read(_SAFETENSORS_LENGTH_PREFIX_SIZE)
    if len(prefix) != _SAFETENSORS_LENGTH_PREFIX_SIZE:
        raise ValueError("Invalid safetensors header: file is too short")
    (header_len,) = struct.unpack("<Q", prefix)
    file_size = os.fstat(f.fileno()).st_size
    max_header_len = min(
        SAFETENSORS_MAX_HEADER_SIZE, file_size - _SAFETENSORS_LENGTH_PREFIX_SIZE
    )
    if header_len > max_header_len:
        raise ValueError(
            f"Invalid safetensors header: declared length {header_len} does not "
            f"fit in a {file_size}-byte file or exceeds the "
            f"{SAFETENSORS_MAX_HEADER_SIZE}-byte limit"
        )
    header = f.read(header_len)
    if len(header) != header_len:
        raise ValueError("Invalid safetensors header: file is truncated")
    return header


def parse_safetensors_file_metadata(path: str | PathLike) -> dict[str, Any]:
    with open(path, "rb") as f:
        return json.loads(read_safetensors_header(f).decode("utf-8"))


def convert_model_repo_to_path(model_repo: str) -> str:
    """When VLLM_USE_MODELSCOPE is True convert a model
    repository string to a Path str."""
    if not envs.VLLM_USE_MODELSCOPE or Path(model_repo).exists():
        return model_repo
    from modelscope.utils.file_utils import get_model_cache_root

    return os.path.join(get_model_cache_root(), model_repo)
