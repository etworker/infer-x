"""GPU and system resource monitoring."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

from models import GPUInfo, SystemInfo


class ResourceMonitor:
    def __init__(self):
        self._instance_gpu_usage: Dict[int, int] = {}  # instance_pid -> gpu_mem_mb

    # ---- GPU ----------------------------------------------------------------

    def get_gpus(self) -> List[GPUInfo]:
        if not _NVML_AVAILABLE:
            return []
        gpus = []
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                util_pct = float(util.gpu)
            except Exception:
                util_pct = None
            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    total_memory_mb=mem.total // (1024 * 1024),
                    used_memory_mb=mem.used // (1024 * 1024),
                    free_memory_mb=mem.free // (1024 * 1024),
                    utilization_pct=util_pct,
                )
            )
        return gpus

    def get_total_gpu_free_mb(self) -> int:
        gpus = self.get_gpus()
        if not gpus:
            return 0
        return min(g.free_memory_mb for g in gpus)

    # ---- RAM ----------------------------------------------------------------

    def get_ram(self) -> Dict[str, int]:
        vm = psutil.virtual_memory()
        return {
            "total_mb": vm.total // (1024 * 1024),
            "used_mb": vm.used // (1024 * 1024),
            "available_mb": vm.available // (1024 * 1024),
        }

    # ---- CPU ----------------------------------------------------------------

    def get_cpu_info(self) -> Dict[str, Any]:
        return {
            "count": psutil.cpu_count(logical=True),
            "percent": psutil.cpu_percent(interval=0.1),
        }

    # ---- combined -----------------------------------------------------------

    def get_system_info(self, llama_server_path: str) -> SystemInfo:
        gpus = self.get_gpus()
        ram = self.get_ram()
        cpu = self.get_cpu_info()
        return SystemInfo(
            gpus=gpus,
            total_ram_mb=ram["total_mb"],
            used_ram_mb=ram["used_mb"],
            available_ram_mb=ram["available_mb"],
            cpu_count=cpu["count"],
            cpu_percent=cpu["percent"],
            llama_server_path=llama_server_path,
        )

    # ---- per-instance tracking ----------------------------------------------

    def update_instance_gpu(self, pid: int, gpu_mb: int) -> None:
        self._instance_gpu_usage[pid] = gpu_mb

    def remove_instance_gpu(self, pid: int) -> None:
        self._instance_gpu_usage.pop(pid, None)

    def get_instance_gpu(self, pid: int) -> Optional[int]:
        return self._instance_gpu_usage.get(pid)

    # ---- process helpers ----------------------------------------------------

    @staticmethod
    def get_process_memory_mb(pid: int) -> Optional[float]:
        try:
            p = psutil.Process(pid)
            mem = p.memory_info()
            return mem.rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    @staticmethod
    def is_process_alive(pid: int) -> bool:
        try:
            p = psutil.Process(pid)
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False


from typing import Any  # noqa: E402  (needed for get_cpu_info return type)
