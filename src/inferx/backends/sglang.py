"""SGLang inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend


class SGLangBackend(Backend):
    """SGLang inference backend."""

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        binary = params.get("binary", "python -m sglang.launch_server")

        # Prepend environment variable to command

        if binary.startswith("python"):
            cmd = [binary.split()[0], "-m", binary.split()[-1]]
        else:
            cmd = [binary]

        cmd.extend([
            "--model-path", str(model_path),
            "--host", host,
            "--port", str(port),
        ])

        if params.get("tp") and params["tp"] > 1:
            cmd.extend(["--tp", str(params["tp"])])
        if params.get("mem_fraction_static") and params["mem_fraction_static"] < 1.0:
            cmd.extend(["--mem-fraction-static", str(params["mem_fraction_static"])])
        if params.get("chat_template"):
            cmd.extend(["--chat-template", params["chat_template"]])
        if params.get("nnodes") and params["nnodes"] > 1:
            cmd.extend(["--nnodes", str(params["nnodes"])])

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {
            "SGLANG_KERNEL_DISABLE_JIT": "1",
            "SGL_KERNEL_DISABLE_JIT": "1",
        }

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
