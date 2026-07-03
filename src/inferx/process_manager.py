"""Process lifecycle: startup, kill, health monitoring."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .backends import get_backend
from .config import ConfigManager
from .logging import logger
from .models import (
    BackendType,
    InstanceInfo,
    InstanceStartRequest,
    InstanceStatus,
)
from .monitor import ResourceMonitor
from .params import resolve_params
from .utils import get_binary_path


@dataclass
class InstanceProcess:
    info: InstanceInfo
    process: subprocess.Popen | None = field(default=None, repr=False)
    log_file: Path | None = field(default=None, repr=False)
    _health_task: asyncio.Task | None = field(default=None, repr=False)
    _restart_task: asyncio.Task | None = field(default=None, repr=False)


class ProcessManager:
    def __init__(self, config: ConfigManager, monitor: ResourceMonitor, logs_dir: Path):
        self._config = config
        self._monitor = monitor
        self._logs_dir = logs_dir
        self._instances: dict[str, InstanceProcess] = {}

    @property
    def instances(self) -> dict[str, InstanceProcess]:
        return self._instances

    async def start(
        self,
        req: InstanceStartRequest,
        port: int,
        host: str,
        extra_args: list[str],
        preset=None,
    ) -> InstanceInfo:
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
        env.update(backend.get_env(binary))

        try:
            stderr_file = self._logs_dir / f"{inst_id}.stderr"
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=open(stderr_file, "w"),
                start_new_session=True,
            )
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

        inst = InstanceProcess(info=info, process=proc, log_file=log_path)
        self._instances[inst_id] = inst
        inst._health_task = asyncio.create_task(self._wait_and_monitor(inst_id))

        return info

    async def _wait_and_monitor(self, inst_id: str) -> None:
        inst = self._instances.get(inst_id)
        if not inst:
            return
        cfg = self._config.config
        url = f"http://{inst.info.host}:{inst.info.port}"

        logger.debug("wait_and_monitor started for {} (pid={})", inst_id, inst.info.pid)

        max_iters = cfg.startup_timeout_seconds
        for i in range(max_iters):
            await asyncio.sleep(1)
            alive = self._monitor.is_process_alive(inst.info.pid)
            if not alive:
                logger.warning("{} process died during startup", inst_id)
                inst.info.status = InstanceStatus.error
                return
            try:
                async with httpx.AsyncClient(timeout=cfg.health_check_timeout) as c:
                    r = await c.get(f"{url}/health")
                    if r.status_code == 200:
                        logger.debug("{} health OK at iter {}", inst_id, i)
                        inst.info.status = InstanceStatus.running
                        break
            except Exception:
                continue
        else:
            logger.warning("{} health check exhausted {}s, forcing running", inst_id, max_iters)
            inst.info.status = InstanceStatus.running

        logger.debug("{} entering health loop, status={}", inst_id, inst.info.status)

        while inst.info.status == InstanceStatus.running:
            await asyncio.sleep(cfg.health_check_interval)
            alive = self._monitor.is_process_alive(inst.info.pid)
            if not alive:
                inst.info.status = InstanceStatus.error
                logger.warning("Instance {} process died", inst_id)
                if cfg.auto_restart and inst.info.restart_count < cfg.auto_restart_max_retries:
                    inst.info.restart_count += 1
                    await asyncio.sleep(cfg.auto_restart_delay)
                    await self._auto_restart(inst_id)
                break
            ram = self._monitor.get_process_memory_mb(inst.info.pid)
            inst.info.ram_usage_mb = round(ram, 1) if ram else None

    async def _auto_restart(self, inst_id: str) -> None:
        inst = self._instances.get(inst_id)
        if not inst:
            return
        old_req = InstanceStartRequest(
            model=inst.info.model,
            backend=inst.info.backend,
            port=inst.info.port,
            ctx_size=inst.info.ctx_size,
            n_gpu_layers=inst.info.n_gpu_layers,
            n_parallel=inst.info.n_parallel,
            extra_args=inst.info.extra_args,
        )
        self.kill(inst)
        del self._instances[inst_id]
        try:
            from .port_manager import PortManager
            pm = PortManager(self._config)
            pm.acquire(inst.info.port)
            await self.start(old_req, port=inst.info.port, host=inst.info.host, extra_args=inst.info.extra_args)
            logger.info("Instance {} restarted successfully", inst_id)
        except Exception as e:
            logger.error("Failed to restart instance {}: {}", inst_id, e)

    def kill(self, inst: InstanceProcess) -> None:
        logger.debug("killing process pid={} for instance {}", inst.process.pid if inst.process else None, inst.info.id)
        if inst.process and inst.process.poll() is None:
            try:
                os.killpg(os.getpgid(inst.process.pid), signal.SIGTERM)
                inst.process.wait(timeout=self._config.config.shutdown_timeout_seconds)
            except (subprocess.TimeoutExpired, ProcessLookupError, OSError):
                try:
                    os.killpg(os.getpgid(inst.process.pid), signal.SIGKILL)
                except Exception:
                    pass
            except Exception:
                pass

    def stop(self, inst_id: str) -> bool:
        inst = self._instances.get(inst_id)
        if not inst:
            return False
        inst.info.status = InstanceStatus.stopping
        if inst._health_task:
            inst._health_task.cancel()
        self.kill(inst)
        del self._instances[inst_id]
        return True

    def get_info(self, inst_id: str) -> InstanceInfo | None:
        inst = self._instances.get(inst_id)
        if inst:
            if inst.process and inst.info.pid:
                alive = self._monitor.is_process_alive(inst.info.pid)
                if not alive and inst.info.status == InstanceStatus.running:
                    inst.info.status = InstanceStatus.error
                ram = self._monitor.get_process_memory_mb(inst.info.pid)
                inst.info.ram_usage_mb = round(ram, 1) if ram else None
            return inst.info
        return None

    def get_logs(self, inst_id: str, lines: int = 100) -> list[str]:
        inst = self._instances.get(inst_id)
        if not inst or not inst.log_file or not inst.log_file.exists():
            return []
        try:
            with open(inst.log_file, encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            return [line.rstrip("\n") for line in all_lines[-lines:]]
        except Exception:
            return []

    def list_all(self) -> list[InstanceInfo]:
        return [inst.info for inst in self._instances.values()]
