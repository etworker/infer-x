"""HuggingFace Text Generation Inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._utils import discover_hf_models
from .base import Backend, register_backend


@register_backend
class TGIBackend(Backend):
    """HuggingFace Text Generation Inference backend."""
    backend_id = "tgi"
    backend_name = "TGI"
    description = "HuggingFace Text Generation Inference, production-grade"
    model_types = ["huggingface", "safetensors"]
    check_type = "docker_image"
    binary_config_attr = "tgi_bin"

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
        cmd = [binary, "--model-id", str(model_path)]

        if host and host != "0.0.0.0":
            cmd.extend(["--hostname", host])
        cmd.extend(["--port", str(port)])

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {}

    @classmethod
    def is_installed(cls) -> bool:
        import shutil
        return shutil.which("text-generation-launcher") is not None

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        return discover_hf_models(model_dir, self._guess_family, self._guess_quantization)
