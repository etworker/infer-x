"""vLLM inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._utils import resolve_binary
from .base import Backend
from ..models import BackendType
from .registry import register_backend


@register_backend(BackendType.vllm)
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
        cmd = resolve_binary(params.get("binary", "python -m vllm.entrypoints.openai.api_server"))

        cmd.extend([
            "--model", str(model_path),
            "--host", host,
            "--port", str(port),
        ])

        from ._utils import add_flag
        add_flag(cmd, params, "max_model_len", "--max-model-len")
        add_flag(cmd, params, "max_num_seqs", "--max-num-seqs")
        add_flag(cmd, params, "max_num_batched_tokens", "--max-num-batched-tokens")
        add_flag(cmd, params, "quantization", "--quantization")
        add_flag(cmd, params, "chat_template", "--chat-template")
        add_flag(cmd, params, "seed", "--seed")
        add_flag(cmd, params, "max_context_len_to_capture", "--max-context-len-to-capture")
        add_flag(cmd, params, "trust_remote_code", "--trust-remote-code")
        add_flag(cmd, params, "disable_log_requests", "--disable-log-requests")
        add_flag(cmd, params, "enforce_eager", "--enforce-eager")
        if params.get("tensor_parallel_size") and params["tensor_parallel_size"] > 1:
            cmd.extend(["--tensor-parallel-size", str(params["tensor_parallel_size"])])
        if params.get("pipeline_parallel_size") and params["pipeline_parallel_size"] > 1:
            cmd.extend(["--pipeline-parallel-size", str(params["pipeline_parallel_size"])])
        if params.get("gpu_memory_utilization") and params["gpu_memory_utilization"] < 1.0:
            cmd.extend(["--gpu-memory-utilization", str(params["gpu_memory_utilization"])])
        if params.get("dtype") and params["dtype"] != "auto":
            cmd.extend(["--dtype", params["dtype"]])

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
        """Discover HuggingFace model directories."""
        models = []
        if not model_dir.exists():
            return models

        for p in sorted(model_dir.iterdir()):
            if not p.is_dir():
                continue

            config_file = p / "config.json"
            has_safetensors = any(p.glob("*.safetensors"))
            has_bin = any(p.glob("*.bin"))

            if config_file.exists() and (has_safetensors or has_bin):
                total_size = 0
                for f in p.glob("*.safetensors"):
                    total_size += f.stat().st_size
                for f in p.glob("*.bin"):
                    total_size += f.stat().st_size

                models.append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(total_size / (1024 * 1024), 1) if total_size > 0 else 0,
                    "family": self._guess_family(p.name),
                    "quantization": self._guess_quantization(p.name),
                })

        return models
