"""NVIDIA TensorRT-LLM backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._utils import resolve_binary
from .base import Backend
from ..models import BackendType
from .registry import register_backend


@register_backend(BackendType.tensorrt_llm)
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
        cmd = resolve_binary(params.get("binary", "python -m tensorrt_llm.commands.tritonserver"))

        cmd.extend([
            "--model_repo", str(model_path),
            "--http-port", str(port),
        ])

        if params.get("trt_max_batch_size"):
            cmd.extend(["--max-batch-size", str(params["trt_max_batch_size"])])
        if params.get("trt_max_input_len"):
            cmd.extend(["--max-input-len", str(params["trt_max_input_len"])])
        if params.get("trt_max_output_len"):
            cmd.extend(["--max-output-len", str(params["trt_max_output_len"])])
        if params.get("trt_max_seq_len"):
            cmd.extend(["--max-seq-len", str(params["trt_max_seq_len"])])
        if params.get("trt_dtype"):
            cmd.extend(["--dtype", params["trt_dtype"]])

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str, host: str = "localhost", port: int = 8080) -> dict[str, str]:
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
        models = []
        if not model_dir.exists():
            return models
        for p in sorted(model_dir.iterdir()):
            if p.is_dir():
                if any(p.glob("*.engine")) or (p / "config.json").exists():
                    total_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    models.append({
                        "name": p.name,
                        "path": str(p),
                        "size_mb": round(total_size / (1024 * 1024), 1) if total_size > 0 else 0,
                        "family": self._guess_family(p.name),
                        "quantization": None,
                    })
        return models
