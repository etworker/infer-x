"""vLLM inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._utils import discover_hf_models
from .base import Backend, register_backend


@register_backend
class VLLMBackend(Backend):
    """vLLM inference backend."""
    backend_id = "vllm"
    backend_name = "vLLM"
    description = "High-performance LLM serving with PagedAttention"
    model_types = ["huggingface", "safetensors"]
    check_type = "python_module"
    binary_config_attr = "vllm_server_bin"

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        binary = params.get("binary", "python -m vllm.entrypoints.openai.api_server")
        cmd = binary.split()
        cmd.extend(["--model", str(model_path), "--host", host, "--port", str(port)])

        if params.get("tensor_parallel_size") and params["tensor_parallel_size"] > 1:
            cmd.extend(["--tensor-parallel-size", str(params["tensor_parallel_size"])])
        if params.get("max_model_len"):
            cmd.extend(["--max-model-len", str(params["max_model_len"])])
        if params.get("gpu_memory_utilization") and params["gpu_memory_utilization"] < 1.0:
            cmd.extend(["--gpu-memory-utilization", str(params["gpu_memory_utilization"])])
        if params.get("dtype") and params["dtype"] != "auto":
            cmd.extend(["--dtype", params["dtype"]])
        if params.get("quantization"):
            cmd.extend(["--quantization", params["quantization"]])

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {}

    @classmethod
    def is_installed(cls) -> bool:
        try:
            import importlib
            importlib.import_module("vllm")
            return True
        except ImportError:
            return False

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        return discover_hf_models(model_dir, self._guess_family, self._guess_quantization)
