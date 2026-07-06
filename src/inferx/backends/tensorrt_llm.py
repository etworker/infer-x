"""NVIDIA TensorRT-LLM backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend, register_backend


@register_backend
class TensorRTLLMBackend(Backend):
    """NVIDIA TensorRT-LLM backend."""
    backend_id = "tensorrt_llm"
    backend_name = "TensorRT-LLM"
    description = "NVIDIA optimized inference for maximum performance"
    model_types = ["tensorrt_engine"]
    check_type = "python_module"
    binary_config_attr = "tensorrt_llm_bin"

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        binary = params.get("binary", "python -m tensorrt_llm.commands.tritonserver")
        cmd = binary.split()
        cmd.extend(["--model-repository", str(model_path), "--port", str(port)])
        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {}

    @classmethod
    def is_installed(cls) -> bool:
        try:
            import importlib
            importlib.import_module("tensorrt_llm")
            return True
        except ImportError:
            return False

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        return []
