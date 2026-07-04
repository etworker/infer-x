"""llama.cpp inference backend."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ._utils import add_flag
from .base import Backend
from ..models import BackendType
from .registry import register_backend


@register_backend(BackendType.llamacpp)
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
        ctx_size = params.get("ctx_size", 4096)
        n_gpu_layers = params.get("n_gpu_layers", "auto")
        batch_size = params.get("batch_size", 2048)
        n_parallel = params.get("n_parallel", 1)
        threads = params.get("threads")
        flash_attn = params.get("flash_attn")
        sleep_idle = params.get("sleep_idle_seconds")
        alias = params.get("alias")
        mlock = params.get("mlock")
        no_mmap = params.get("no_mmap")
        numa = params.get("numa")
        cont_batching = params.get("cont_batching")

        cmd = [
            binary,
            "-m", str(model_path),
            "--host", host,
            "--port", str(port),
            "-c", str(ctx_size),
            "-ngl", str(n_gpu_layers),
            "-b", str(batch_size),
            "-np", str(n_parallel),
            "--log-file", str(log_file),
        ]

        add_flag(cmd, params, "threads", "-t")
        add_flag(cmd, params, "alias", "-a")
        add_flag(cmd, params, "mlock", "--mlock")
        add_flag(cmd, params, "no_mmap", "--no-mmap")
        add_flag(cmd, params, "numa", "--numa")
        add_flag(cmd, params, "cont_batching", "--cont-batching")
        if flash_attn and flash_attn != "none":
            if flash_attn is True:
                cmd.append("-fa")
            else:
                cmd.extend(["-fa", str(flash_attn)])
        if sleep_idle is not None and sleep_idle > 0:
            cmd.extend(["--sleep-idle-seconds", str(sleep_idle)])

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str, host: str = "localhost", port: int = 8080) -> dict[str, str]:
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
