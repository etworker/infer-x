"""llama.cpp inference backend."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .base import Backend, register_backend


@register_backend
class LlamaCppBackend(Backend):
    """llama.cpp inference backend."""
    backend_id = "llamacpp"
    backend_name = "llama.cpp"
    description = "Local GGUF model inference, lightweight"
    model_types = ["gguf"]
    check_type = "binary"
    binary_config_attr = "llama_server_bin"

    def build_command(
        self,
        model_path: str,
        port: int,
        host: str,
        log_file: str,
        params: dict[str, Any],
        extra_args: list[str],
    ) -> list[str]:
        binary = params.get("binary", "llama-server")
        cmd = [
            binary,
            "-m", str(model_path),
            "--host", host,
            "--port", str(port),
            "-c", str(params.get("ctx_size", 4096)),
            "-ngl", str(params.get("n_gpu_layers", "auto")),
            "-b", str(params.get("batch_size", 2048)),
            "-np", str(params.get("n_parallel", 1)),
            "--log-file", str(log_file),
        ]
        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {"LD_LIBRARY_PATH": str(Path(binary_path).parent)}

    @classmethod
    def is_installed(cls) -> bool:
        if shutil.which("llama-server"):
            return True
        env_bin = os.environ.get("LLAMA_SERVER_BIN")
        if env_bin and Path(env_bin).exists():
            return True
        home = Path.home()
        return any(
            p.exists() for p in [
                home / "llama.cpp" / "build" / "bin" / "llama-server",
                Path("/usr/local/bin/llama-server"),
                Path("/usr/bin/llama-server"),
            ]
        )

    def get_model_paths(self, model_dir: Path) -> list[dict[str, Any]]:
        models = []
        if not model_dir.exists():
            return models

        for p in sorted(model_dir.iterdir()):
            if p.suffix == ".gguf" and p.is_file():
                models.append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 1),
                    "family": self._guess_family(p.name),
                    "quantization": self._guess_quantization(p.name),
                })
            elif p.is_dir():
                ggufs = list(p.glob("*.gguf"))
                if ggufs:
                    main = ggufs[0]
                    models.append({
                        "name": f"{p.name}/{main.name}",
                        "path": str(main),
                        "size_mb": round(main.stat().st_size / (1024 * 1024), 1),
                        "family": self._guess_family(main.name),
                        "quantization": self._guess_quantization(main.name),
                    })

        return models
