"""Process lifecycle: start, kill, stop — no health monitoring or logging."""

from __future__ import annotations

import asyncio
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .backends import get_backend
from .config import ConfigManager
from .logging import logger
from .models import (
    BackendType,
    InstanceInfo,
    InstanceStartRequest,
    InstanceStatus,
)
from .params import resolve_params
from .utils import get_binary_path


@dataclass
class InstanceProcess:
    info: InstanceInfo
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    log_file: Path | None = field(default=None, repr=False)
    _health_task: asyncio.Task | None = field(default=None, repr=False)
    _restart_task: asyncio.Task | None = field(default=None, repr=False)


class ProcessLifecycle:
    """Responsible for starting, killing, and stopping subprocesses."""

    def __init__(self, config: ConfigManager, logs_dir: Path) -> None:
        self._config = config
        self._logs_dir = logs_dir

    async def start(
        self,
        req: InstanceStartRequest,
        port: int,
        host: str,
        extra_args: list[str],
        preset=None,
    ) -> InstanceProcess:
        cfg = self._config.config
        backend_type = req.backend or cfg.default_backend
        backend = get_backend(backend_type)

        from .backends import check_backend_installed
        if not check_backend_installed(backend_type):
            raise RuntimeError(f"Backend '{backend_type.value}' is not installed.")

        binary = get_binary_path(backend_type, cfg)

        model_path = Path(cfg.model_dir).expanduser() / req.model
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {req.model} (looked in {Path(cfg.model_dir).expanduser()})"
            )

        inst_id = f"inst-{uuid.uuid4().hex[:8]}"
        log_path = self._logs_dir / f"{inst_id}.log"

        params = {"binary": binary, "log_file": str(log_path)}
        params.update(resolve_params(backend_type, req, preset, cfg))
        if req.alias:
            params["alias"] = req.alias

        cmd = backend.build_command(
            model_path=str(model_path),
            port=port,
            host=host,
            log_file=str(log_path),
            params=params,
            extra_args=extra_args,
        )

        logger.info("Starting instance {} (backend={}): {}", inst_id, backend_type.value, " ".join(cmd))

        env = os.environ.copy()
        env.update(backend.get_env(binary, host, port))

        try:
            stderr_path = self._logs_dir / f"{inst_id}.stderr"
            stderr_file = open(stderr_path, "w")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=stderr_file,
                    start_new_session=True,
                )
            except Exception:
                stderr_file.close()
                raise
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Failed to start {backend_type.value}: binary not found. "
                f"Command: {' '.join(cmd)}. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start {backend_type.value}: {e}")

        info = InstanceInfo(
            id=inst_id,
            model=req.model,
            backend=backend_type,
            status=InstanceStatus.starting,
            port=port,
            host=host,
            pid=proc.pid,
            ctx_size=params.get("ctx_size", 4096),
            n_gpu_layers=params.get("n_gpu_layers", "auto"),
            n_parallel=params.get("n_parallel", 1),
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            extra_args=extra_args,
        )

        return InstanceProcess(info=info, process=proc, log_file=log_path)

    async def kill(self, inst: InstanceProcess) -> None:
        logger.debug("killing process pid={} for instance {}", inst.process.pid if inst.process else None, inst.info.id)
        if inst.process and inst.process.returncode is None:
            try:
                os.killpg(os.getpgid(inst.process.pid), signal.SIGTERM)
                await asyncio.wait_for(inst.process.wait(), timeout=self._config.config.shutdown_timeout_seconds)
            except (asyncio.TimeoutError, ProcessLookupError, OSError):
                try:
                    os.killpg(os.getpgid(inst.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            except Exception:
                pass

    async def stop(self, inst: InstanceProcess) -> None:
        inst.info.status = InstanceStatus.stopping
        if inst._health_task:
            inst._health_task.cancel()
        await self.kill(inst)
