"""Ollama inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend


class OllamaBackend(Backend):
    """Ollama inference backend."""

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

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        return []
