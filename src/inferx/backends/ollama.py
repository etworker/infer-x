"""Ollama inference backend."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .base import Backend, register_backend


@register_backend
class OllamaBackend(Backend):
    """Ollama inference backend."""
    backend_id = "ollama"
    backend_name = "Ollama"
    description = "User-friendly local LLM runner"
    model_types = ["ollama"]
    check_type = "binary"
    binary_config_attr = "ollama_bin"

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        cmd = ["ollama", "serve"]
        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {}

    @classmethod
    def is_installed(cls) -> bool:
        return shutil.which("ollama") is not None

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        return []
