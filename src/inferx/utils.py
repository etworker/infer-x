"""Shared utility functions used across the codebase."""

from __future__ import annotations

import re
from typing import Any


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


def get_binary_path(backend_type: Any, config: Any) -> str:
    """Get the configured binary path for a backend."""
    from .backends.base import get_backend
    backend = get_backend(backend_type)
    attr = getattr(backend, "binary_config_attr", None)
    if attr:
        return getattr(config, attr, "")
    return ""


def get_server_paths(config: Any) -> dict[str, str]:
    """Get a dict mapping backend ID -> binary path for all backends."""
    from .backends.base import get_all_backends_status
    backends = get_all_backends_status()
    return {
        b["id"]: get_binary_path(b["id"], config)
        for b in backends
    }
