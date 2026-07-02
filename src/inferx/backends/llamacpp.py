"""llama.cpp inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend


class LlamaCppBackend(Backend):
    """llama.cpp inference backend."""

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

        if threads is not None:
            cmd.extend(["-t", str(threads)])
        if flash_attn and flash_attn != "none":
            cmd.extend(["-fa", str(flash_attn)])
        if sleep_idle is not None and sleep_idle > 0:
            cmd.extend(["--sleep-idle-seconds", str(sleep_idle)])
        if alias:
            cmd.extend(["-a", alias])
        if mlock:
            cmd.append("--mlock")
        if no_mmap:
            cmd.append("--no-mmap")
        if numa:
            cmd.extend(["--numa", str(numa)])
        if cont_batching:
            cmd.append("--cont-batching")

        cmd.extend(extra_args)
        return cmd

    def get_env(self, binary_path: str) -> dict[str, str]:
        return {"LD_LIBRARY_PATH": str(Path(binary_path).parent)}

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
