"""Intel OpenVINO Model Server backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend, register_backend


@register_backend
class OpenVINOBackend(Backend):
    """Intel OpenVINO Model Server backend."""
    backend_id = "openvino"
    backend_name = "OpenVINO"
    description = "Intel optimized inference for Intel hardware"
    model_types = ["openvino"]
    check_type = "python_module"
    binary_config_attr = "openvino_bin"

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        binary = params.get("binary", "ovms")
        cmd = [binary, "--model_name", params.get("alias", "model"),
               "--model_path", str(model_path), "--rest_port", str(port)]
        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {}

    @classmethod
    def is_installed(cls) -> bool:
        try:
            import importlib
            importlib.import_module("openvino")
            return True
        except ImportError:
            return False

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        return []
