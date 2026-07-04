"""Health monitoring: startup wait, periodic health polling, auto-restart."""

from __future__ import annotations

import asyncio

import httpx

from .config import ConfigManager
from .logging import logger
from .models import InstanceStatus
from .models import InstanceStartRequest
from .monitor import ResourceMonitor
from .port_manager import PortManager
from .process_lifecycle import InstanceProcess, ProcessLifecycle


class HealthChecker:
    """Monitors instance health and triggers auto-restart on failure."""

    def __init__(
        self,
        config: ConfigManager,
        monitor: ResourceMonitor,
        lifecycle: ProcessLifecycle,
        port_manager: PortManager,
    ) -> None:
        self._config = config
        self._monitor = monitor
        self._lifecycle = lifecycle
        self._port_mgr = port_manager

    async def wait_and_monitor(self, inst: InstanceProcess) -> None:
        cfg = self._config.config
        url = f"http://{inst.info.host}:{inst.info.port}"

        logger.debug("wait_and_monitor started for {} (pid={})", inst.info.id, inst.info.pid)

        async with httpx.AsyncClient(timeout=httpx.Timeout(cfg.health_check_timeout)) as client:
            max_iters = cfg.startup_timeout_seconds
            for i in range(max_iters):
                await asyncio.sleep(1)
                alive = self._monitor.is_process_alive(inst.info.pid)
                if not alive:
                    logger.warning("{} process died during startup", inst.info.id)
                    inst.info.status = InstanceStatus.error
                    return
                try:
                    r = await client.get(f"{url}/health")
                    if r.status_code == 200:
                        logger.debug("{} health OK at iter {}", inst.info.id, i)
                        inst.info.status = InstanceStatus.running
                        break
                except Exception:
                    continue
            else:
                logger.warning("{} health check exhausted {}s, setting error", inst.info.id, max_iters)
                inst.info.status = InstanceStatus.error

            logger.debug("{} entering health loop, status={}", inst.info.id, inst.info.status)

            while inst.info.status == InstanceStatus.running:
                await asyncio.sleep(cfg.health_check_interval)
                alive = self._monitor.is_process_alive(inst.info.pid)
                if not alive:
                    inst.info.status = InstanceStatus.error
                    logger.warning("Instance {} process died", inst.info.id)
                    if cfg.auto_restart and inst.info.restart_count < cfg.auto_restart_max_retries:
                        inst.info.restart_count += 1
                        await asyncio.sleep(cfg.auto_restart_delay)
                        await self._auto_restart(inst)
                    break
                ram = self._monitor.get_process_memory_mb(inst.info.pid)
                inst.info.ram_usage_mb = round(ram, 1) if ram else None

    async def _auto_restart(self, inst: InstanceProcess) -> None:
        old_req = InstanceStartRequest(
            model=inst.info.model,
            backend=inst.info.backend,
            port=inst.info.port,
            ctx_size=inst.info.ctx_size,
            n_gpu_layers=inst.info.n_gpu_layers,
            n_parallel=inst.info.n_parallel,
            extra_args=inst.info.extra_args,
        )
        await self._lifecycle.kill(inst)

        try:
            self._port_mgr.acquire(inst.info.port)
            new_inst = await self._lifecycle.start(
                old_req, port=inst.info.port, host=inst.info.host, extra_args=inst.info.extra_args
            )
            inst.process = new_inst.process
            inst.info.pid = new_inst.info.pid
            inst.info.status = InstanceStatus.starting
            inst.info.restart_count = inst.info.restart_count
            inst.log_file = new_inst.log_file
            # Start health monitoring on the new process
            inst._health_task = asyncio.create_task(self.wait_and_monitor(inst))
            logger.info("Instance {} restarted successfully", inst.info.id)
        except Exception as e:
            logger.error("Failed to restart instance {}: {}", inst.info.id, e)
