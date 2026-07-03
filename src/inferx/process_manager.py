"""ProcessManager — thin orchestrator over ProcessLifecycle, HealthChecker, LogReader."""

from __future__ import annotations

import asyncio
from pathlib import Path

from .config import ConfigManager
from .health_checker import HealthChecker
from .logging import logger
from .log_reader import LogReader
from .models import InstanceInfo, InstanceStartRequest, InstanceStatus
from .monitor import ResourceMonitor
from .port_manager import PortManager
from .process_lifecycle import ProcessLifecycle, InstanceProcess


class ProcessManager:
    """Orchestrates process lifecycle, health monitoring, and log retrieval."""

    def __init__(
        self,
        config: ConfigManager,
        monitor: ResourceMonitor,
        logs_dir: Path,
        port_manager: PortManager | None = None,
    ) -> None:
        self._lifecycle = ProcessLifecycle(config, logs_dir)
        self._log_reader = LogReader(logs_dir)
        self._health_checker = HealthChecker(config, monitor, self._lifecycle, port_manager)
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
        inst = await self._lifecycle.start(req, port, host, extra_args, preset)
        inst._health_task = asyncio.create_task(self._health_checker.wait_and_monitor(inst))
        self._instances[inst.info.id] = inst
        return inst.info

    async def kill(self, inst: InstanceProcess) -> None:
        await self._lifecycle.kill(inst)

    async def stop(self, inst_id: str) -> bool:
        inst = self._instances.get(inst_id)
        if not inst:
            return False
        await self._lifecycle.stop(inst)
        del self._instances[inst_id]
        return True

    def get_info(self, inst_id: str) -> InstanceInfo | None:
        inst = self._instances.get(inst_id)
        if inst:
            if inst.process is not None and inst.info.pid:
                alive = self._health_checker._monitor.is_process_alive(inst.info.pid)
                if not alive and inst.info.status == InstanceStatus.running:
                    inst.info.status = InstanceStatus.error
                ram = self._health_checker._monitor.get_process_memory_mb(inst.info.pid)
                inst.info.ram_usage_mb = round(ram, 1) if ram else None
            return inst.info
        return None

    def get_logs(self, inst_id: str, lines: int = 100) -> list[str]:
        inst = self._instances.get(inst_id)
        log_file = inst.log_file if inst else None
        return self._log_reader.get_logs(inst_id, log_file, lines)

    def get_error_logs(self, inst_id: str, lines: int = 50) -> list[str]:
        return self._log_reader.get_error_logs(inst_id, lines)

    def list_all(self) -> list[InstanceInfo]:
        return [inst.info for inst in self._instances.values()]
