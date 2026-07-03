"""Intel OpenVINO Model Server backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend
from ..models import BackendType
from .registry import register_backend


@register_backend(BackendType.openvino)
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

        cmd = [
            binary,
            "--model_name", params.get("ov_model_name", "model"),
            "--model_path", str(model_path),
            "--rest_port", str(port),
        ]

        if params.get("ov_batch_size"):
            cmd.extend(["--batch_size", str(params["ov_batch_size"])])
        if params.get("ov_max_model_len"):
            cmd.extend(["--max_model_len", str(params["ov_max_model_len"])])
        if params.get("ov_nireq"):
            cmd.extend(["--nireq", str(params["ov_nireq"])])
        if params.get("ov_plugin_config"):
            cmd.extend(["--plugin_config", params["ov_plugin_config"]])
        if params.get("ov_model_section"):
            cmd.extend(["--config", params["ov_model_section"]])

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
        models = []
        if not model_dir.exists():
            return models
        for p in sorted(model_dir.iterdir()):
            if p.is_dir():
                if any(p.glob("*.xml")) or any(p.glob("*.bin")):
                    models.append({
                        "name": p.name,
                        "path": str(p),
                        "size_mb": 0,
                        "family": self._guess_family(p.name),
                        "quantization": None,
                    })
        return models
