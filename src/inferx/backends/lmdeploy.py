"""LMDeploy inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._utils import resolve_binary
from .base import Backend
from ..models import BackendType
from .registry import register_backend


@register_backend(BackendType.lmdeploy)
class LMDeployBackend(Backend):
    """LMDeploy inference backend."""
    backend_id = "lmdeploy"
    backend_name = "LMDeploy"
    description = "Shanghai AI Lab inference, optimized for Chinese models"
    model_types = ["huggingface", "safetensors"]
    check_type = "python_module"
    binary_config_attr = "lmdeploy_bin"

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        cmd = resolve_binary(params.get("binary", "lmdeploy serve api_server"))

        cmd.extend([str(model_path), "--server-port", str(port)])

        if host and host != "0.0.0.0":
            cmd.extend(["--server-name", host])
        if params.get("lmdeploy_tp") and params["lmdeploy_tp"] > 1:
            cmd.extend(["--tp", str(params["lmdeploy_tp"])])
        if params.get("lmdeploy_session_len"):
            cmd.extend(["--session-len", str(params["lmdeploy_session_len"])])
        if params.get("lmdeploy_max_batch_size"):
            cmd.extend(["--max-batch-size", str(params["lmdeploy_max_batch_size"])])
        if params.get("lmdeploy_cache_max_entry_count"):
            cmd.extend(["--cache-max-entry-count", str(params["lmdeploy_cache_max_entry_count"])])
        if params.get("lmdeploy_quant_policy"):
            cmd.extend(["--quant-policy", str(params["lmdeploy_quant_policy"])])
        if params.get("lmdeploy_rope_scaling_factor") and params["lmdeploy_rope_scaling_factor"] > 0:
            cmd.extend(["--rope-scaling-factor", str(params["lmdeploy_rope_scaling_factor"])])
        if params.get("lmdeploy_turbomind_tp") and params["lmdeploy_turbomind_tp"] > 1:
            cmd.extend(["--turbomind-tp", str(params["lmdeploy_turbomind_tp"])])

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {}

    @classmethod
    def is_installed(cls) -> bool:
        try:
            import importlib
            importlib.import_module("lmdeploy")
            return True
        except ImportError:
            return False

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
