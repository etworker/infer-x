"""Shared utility functions used across the codebase."""

from __future__ import annotations

import re
from typing import Any

from .models import DefaultConfig
from .backends.registry import registry


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


def get_binary_path(backend_type: Any, config: DefaultConfig) -> str:
    """Get the configured binary path for a backend."""
    attr = registry.get_binary_config_attr(backend_type)
    if attr:
        return getattr(config, attr, "")
    return ""


def get_server_paths(config: DefaultConfig) -> dict[str, str]:
    """Get a dict mapping backend ID -> binary path for all backends."""
    return {
        bt.value: get_binary_path(bt, config)
        for bt in registry.backends
    }
