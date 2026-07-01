"""NVIDIA TensorRT-LLM backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import Backend


class TensorRTLLMBackend(Backend):
    """NVIDIA TensorRT-LLM backend."""

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: Dict[str, Any],
        extra_args: List[str],
    ) -> List[str]:
        binary = params.get("binary", "python -m tensorrt_llm.commands.tritonserver")

        if binary.startswith("python"):
            cmd = binary.split()
        else:
            cmd = [binary]

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

    def get_env(self, binary_path: str) -> Dict[str, str]:
        return {}

    def get_model_paths(self, model_dir: Path) -> List[Dict[str, Any]]:
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
