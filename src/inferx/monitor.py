"""GPU and system resource monitoring."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

from .models import GPUInfo, SystemInfo


_BACKEND_PATTERNS: list[tuple[re.Pattern, str, list[tuple[str, str, type]]]] = [
    (re.compile(r"llama-server"), "llamacpp", [
        ("-m", "model", str), ("--port", "port", int),
    ]),
    (re.compile(r"python.*vllm"), "vllm", [
        ("--model", "model", str), ("--port", "port", int),
    ]),
    (re.compile(r"python.*sglang"), "sglang", [
        ("--model-path", "model", str), ("--port", "port", int),
    ]),
    (re.compile(r"text-generation-launcher"), "tgi", [
        ("--model-id", "model", str),
    ]),
    (re.compile(r"ollama serve"), "ollama", []),
    (re.compile(r"lmdeploy"), "lmdeploy", [
        ("--server-port", "port", int),
    ]),
    (re.compile(r"ovms"), "openvino", []),
]


class ResourceMonitor:
    def get_gpus(self) -> list[GPUInfo]:
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

    def get_ram(self) -> dict[str, int]:
        vm = psutil.virtual_memory()
        return {
            "total_mb": vm.total // (1024 * 1024),
            "used_mb": vm.used // (1024 * 1024),
            "available_mb": vm.available // (1024 * 1024),
        }

    def get_cpu_info(self) -> dict[str, Any]:
        return {
            "count": psutil.cpu_count(logical=True),
            "percent": psutil.cpu_percent(interval=0.1),
        }

    def get_system_info(self, server_paths: dict[str, str]) -> SystemInfo:
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
            server_paths=server_paths,
        )

    @staticmethod
    def get_process_memory_mb(pid: int) -> float | None:
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

    def detect_gpu_processes(self) -> list[dict[str, Any]]:
        """Scan GPU processes and identify potential inference backends."""
        if not _NVML_AVAILABLE:
            return []

        results: list[dict[str, Any]] = []
        seen_pids: set[int] = set()

        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception:
            return results

        for i in range(count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode()
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                continue

            for proc in procs:
                pid = proc.pid
                if pid in seen_pids:
                    continue
                seen_pids.add(pid)

                gpu_mem = proc.usedGpuMemory // (1024 * 1024) if hasattr(proc, "usedGpuMemory") else 0
                results.append(self._analyze_process(pid, gpu_mem, gpu_name, i))

        return results

    @staticmethod
    def _analyze_process(pid: int, gpu_mem_mb: int, gpu_name: str, gpu_index: int) -> dict[str, Any]:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_text().replace("\0", " ").strip()
        except Exception:
            cmdline = ""

        result: dict[str, Any] = {
            "pid": pid,
            "discovered": True,
            "gpu_memory_mb": gpu_mem_mb,
            "gpu_index": gpu_index,
            "gpu_name": gpu_name,
            "cmdline": cmdline,
            "backend": None,
            "model": None,
            "port": None,
        }

        if not cmdline:
            return result

        for pattern, backend_name, flags in _BACKEND_PATTERNS:
            if pattern.search(cmdline):
                result["backend"] = backend_name
                for flag, key, cast in flags:
                    m = re.search(rf"{re.escape(flag)}\s+(\S+)", cmdline)
                    if m:
                        try:
                            result[key] = cast(m.group(1))
                        except (ValueError, TypeError):
                            pass
                break

        if result["model"]:
            result["model"] = Path(result["model"]).name

        return result
