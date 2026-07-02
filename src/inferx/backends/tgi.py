"""HuggingFace Text Generation Inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend


class TGIBackend(Backend):
    """HuggingFace Text Generation Inference backend."""

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        binary = params.get("binary", "text-generation-launcher")
        model_id = params.get("tgi_model_id") or model_path

        cmd = [binary, "--model-id", str(model_id)]

        if host and host != "0.0.0.0":
            cmd.extend(["--hostname", host])
        cmd.extend(["--port", str(port)])

        if params.get("tgi_max_batch_prefill_tokens"):
            cmd.extend(["--max-batch-prefill-tokens", str(params["tgi_max_batch_prefill_tokens"])])
        if params.get("tgi_max_batch_total_tokens"):
            cmd.extend(["--max-batch-total-tokens", str(params["tgi_max_batch_total_tokens"])])
        if params.get("tgi_max_concurrent_requests"):
            cmd.extend(["--max-concurrent-requests", str(params["tgi_max_concurrent_requests"])])
        if params.get("tgi_max_input_length"):
            cmd.extend(["--max-input-length", str(params["tgi_max_input_length"])])
        if params.get("tgi_max_total_tokens"):
            cmd.extend(["--max-total-tokens", str(params["tgi_max_total_tokens"])])
        if params.get("tgi_sharded"):
            cmd.append("--sharded")
        if params.get("tgi_num_shard"):
            cmd.extend(["--num-shard", str(params["tgi_num_shard"])])
        if params.get("tgi_quantize"):
            cmd.extend(["--quantize", params["tgi_quantize"]])
        if params.get("tgi_dtype") and params["tgi_dtype"] != "auto":
            cmd.extend(["--dtype", params["tgi_dtype"]])
        if params.get("tgi_cuda_flash_attention") is False:
            cmd.append("--no-cuda-flash-attention")
        if params.get("tgi_disable_grammar"):
            cmd.append("--disable-grammar")

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {}

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        models = []
        if not model_dir.exists():
            return models
        for p in sorted(model_dir.iterdir()):
            if p.is_dir():
                config_file = p / "config.json"
                has_safetensors = any(p.glob("*.safetensors"))
                if config_file.exists() and has_safetensors:
                    total_size = sum(f.stat().st_size for f in p.glob("*.safetensors"))
                    models.append({
                        "name": p.name,
                        "path": str(p),
                        "size_mb": round(total_size / (1024 * 1024), 1) if total_size > 0 else 0,
                        "family": self._guess_family(p.name),
                        "quantization": None,
                    })
        return models
