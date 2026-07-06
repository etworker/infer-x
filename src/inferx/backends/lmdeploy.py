"""LMDeploy inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._utils import discover_hf_models
from .base import Backend, register_backend


@register_backend
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
        cmd = params.get("binary", "lmdeploy serve api_server").split()
        cmd.extend([str(model_path), "--server-port", str(port)])

        if host and host != "0.0.0.0":
            cmd.extend(["--server-name", host])

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
        return discover_hf_models(model_dir, self._guess_family, self._guess_quantization)
