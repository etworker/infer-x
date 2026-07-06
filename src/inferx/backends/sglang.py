"""SGLang inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._utils import discover_hf_models
from .base import Backend, register_backend


@register_backend
class SGLangBackend(Backend):
    """SGLang inference backend."""
    backend_id = "sglang"
    backend_name = "SGLang"
    description = "High-performance LLM framework with RadixAttention"
    model_types = ["huggingface", "safetensors"]
    check_type = "python_module"
    binary_config_attr = "sglang_server_bin"

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        binary = params.get("binary", "python -m sglang.launch_server")
        cmd = binary.split()
        cmd.extend(["--model-path", str(model_path), "--host", host, "--port", str(port)])

        if params.get("tensor_parallel_size") and params["tensor_parallel_size"] > 1:
            cmd.extend(["--tp", str(params["tensor_parallel_size"])])
        if params.get("gpu_memory_utilization") and params["gpu_memory_utilization"] < 1.0:
            cmd.extend(["--mem-fraction-static", str(params["gpu_memory_utilization"])])

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {}

    @classmethod
    def is_installed(cls) -> bool:
        try:
            import importlib
            importlib.import_module("sglang")
            return True
        except ImportError:
            return False

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        return discover_hf_models(model_dir, self._guess_family, self._guess_quantization)
