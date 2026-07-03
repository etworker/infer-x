"""Shared utility functions used across the codebase."""

from __future__ import annotations

import re
from typing import Any

from .models import BackendType, DefaultConfig


def guess_family(name: str) -> str | None:
    """Guess model family from filename."""
    name_lower = name.lower()
    for family in ["qwen", "gemma", "llama", "mistral", "phi", "deepseek", "yi", "baichuan"]:
        if family in name_lower:
            return family
    return None


def guess_quantization(name: str) -> str | None:
    """Guess quantization from filename."""
    m = re.search(r"(Q[0-9]+_[A-Z0-9]+|F16|F32|BF16|IQ[0-9]+_[A-Z0-9]+)", name, re.IGNORECASE)
    return m.group(1).upper() if m else None


# Maps BackendType -> config attribute name for binary paths
BINARY_PATH_ATTRS: dict[BackendType, str] = {
    BackendType.llamacpp: "llama_server_bin",
    BackendType.vllm: "vllm_server_bin",
    BackendType.sglang: "sglang_server_bin",
    BackendType.tgi: "tgi_bin",
    BackendType.ollama: "ollama_bin",
    BackendType.tensorrt_llm: "tensorrt_llm_bin",
    BackendType.lmdeploy: "lmdeploy_bin",
    BackendType.openvino: "openvino_bin",
}


def get_binary_path(backend_type: BackendType, config: DefaultConfig) -> str:
    """Get the configured binary path for a backend."""
    attr = BINARY_PATH_ATTRS.get(backend_type)
    if attr:
        return getattr(config, attr, "")
    return ""


def get_server_paths(config: DefaultConfig) -> dict[str, str]:
    """Get a dict mapping backend ID -> binary path for all backends."""
    return {bt.value: get_binary_path(bt, config) for bt in BackendType}
