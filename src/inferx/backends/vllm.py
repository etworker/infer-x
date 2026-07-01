"""vLLM inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .base import Backend


class VLLMBackend(Backend):
    """vLLM inference backend."""

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: Dict[str, Any],
        extra_args: List[str],
    ) -> List[str]:
        binary = params.get("binary", "python -m vllm.entrypoints.openai.api_server")

        if binary.startswith("python"):
            cmd = binary.split()
        else:
            cmd = [binary]

        cmd.extend([
            "--model", str(model_path),
            "--host", host,
            "--port", str(port),
            "--log-file", str(log_file),
        ])

        if params.get("tensor_parallel_size") and params["tensor_parallel_size"] > 1:
            cmd.extend(["--tensor-parallel-size", str(params["tensor_parallel_size"])])
        if params.get("pipeline_parallel_size") and params["pipeline_parallel_size"] > 1:
            cmd.extend(["--pipeline-parallel-size", str(params["pipeline_parallel_size"])])
        if params.get("max_model_len"):
            cmd.extend(["--max-model-len", str(params["max_model_len"])])
        if params.get("gpu_memory_utilization") and params["gpu_memory_utilization"] < 1.0:
            cmd.extend(["--gpu-memory-utilization", str(params["gpu_memory_utilization"])])
        if params.get("max_num_seqs"):
            cmd.extend(["--max-num-seqs", str(params["max_num_seqs"])])
        if params.get("max_num_batched_tokens"):
            cmd.extend(["--max-num-batched-tokens", str(params["max_num_batched_tokens"])])
        if params.get("dtype") and params["dtype"] != "auto":
            cmd.extend(["--dtype", params["dtype"]])
        if params.get("quantization"):
            cmd.extend(["--quantization", params["quantization"]])
        if params.get("trust_remote_code"):
            cmd.append("--trust-remote-code")
        if params.get("chat_template"):
            cmd.extend(["--chat-template", params["chat_template"]])
        if params.get("seed") is not None:
            cmd.extend(["--seed", str(params["seed"])])
        if params.get("disable_log_requests"):
            cmd.append("--disable-log-requests")
        if params.get("enforce_eager"):
            cmd.append("--enforce-eager")
        if params.get("max_context_len_to_capture"):
            cmd.extend(["--max-context-len-to-capture", str(params["max_context_len_to_capture"])])

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> Dict[str, str]:
        return {}

    def get_model_paths(self, model_dir: Path) -> List[Dict[str, Any]]:
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
