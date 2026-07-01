"""Core instance manager: process lifecycle, health checks, auto-restart."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import psutil

from config import ConfigManager
from downloader import ModelDownloader
from monitor import ResourceMonitor
from models import (
    InstanceInfo,
    InstanceStartRequest,
    InstanceStatus,
    Preset,
)

logger = logging.getLogger("llamacpp_manager")


@dataclass
class InstanceProcess:
    info: InstanceInfo
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    log_file: Optional[Path] = field(default=None, repr=False)
    _health_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _restart_task: Optional[asyncio.Task] = field(default=None, repr=False)


class InstanceManager:
    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager
        self._monitor = ResourceMonitor()
        self._downloader = ModelDownloader(
            model_dir=config_manager.config.model_dir,
            hf_mirror_url=config_manager.config.hf_mirror_url,
            max_concurrent=config_manager.config.download_max_concurrent,
        )
        self._instances: Dict[str, InstanceProcess] = {}
        self._ports_in_use: set = set()
        self._logs_dir = Path(__file__).parent / "logs"
        self._logs_dir.mkdir(exist_ok=True)

    @property
    def monitor(self) -> ResourceMonitor:
        return self._monitor

    @property
    def downloader(self) -> ModelDownloader:
        return self._downloader

    # ---- port management ----------------------------------------------------

    def _find_available_port(self) -> int:
        cfg = self._config.config
        for port in range(cfg.port_range_start, cfg.port_range_end + 1):
            if port in self._ports_in_use:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available ports in configured range")

    def _release_port(self, port: int) -> None:
        self._ports_in_use.discard(port)

    # ---- model discovery ----------------------------------------------------

    def list_models(self) -> List[Dict[str, Any]]:
        model_dir = Path(self._config.config.model_dir)
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

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        model_dir = Path(self._config.config.model_dir)
        target = model_dir / name
        if not target.exists():
            for p in model_dir.rglob(name):
                if p.suffix == ".gguf":
                    target = p
                    break
        if not target.exists():
            return None
        return {
            "name": name,
            "path": str(target),
            "size_mb": round(target.stat().st_size / (1024 * 1024), 1),
            "family": self._guess_family(target.name),
            "quantization": self._guess_quantization(target.name),
        }

    def delete_model(self, name: str) -> bool:
        model_dir = Path(self._config.config.model_dir)
        target = model_dir / name
        if target.exists() and target.is_file():
            target.unlink()
            return True
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
            return True
        return False

    @staticmethod
    def _guess_family(name: str) -> Optional[str]:
        name_lower = name.lower()
        for family in ["qwen", "gemma", "llama", "mistral", "phi", "deepseek", "yi", "baichuan"]:
            if family in name_lower:
                return family
        return None

    @staticmethod
    def _guess_quantization(name: str) -> Optional[str]:
        import re
        m = re.search(r"(Q[0-9]+_[A-Z0-9]+|F16|F32|BF16|IQ[0-9]+_[A-Z0-9]+)", name, re.IGNORECASE)
        return m.group(1).upper() if m else None

    # ---- instance lifecycle -------------------------------------------------

    async def start_instance(self, req: InstanceStartRequest) -> InstanceInfo:
        cfg = self._config.config

        if len(self._instances) >= cfg.max_instances:
            raise RuntimeError(f"Max instances ({cfg.max_instances}) reached")

        model_path = Path(cfg.model_dir) / req.model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {req.model}")

        port = req.port or self._find_available_port()
        if port in self._ports_in_use and (req.port is None or port != req.port):
            raise RuntimeError(f"Port {port} already in use")
        self._ports_in_use.add(port)

        # resolve preset defaults
        preset = None
        if req.preset:
            preset = self._config.get_preset(req.preset)

        ctx_size = req.ctx_size or (preset.ctx_size if preset and preset.ctx_size else cfg.default_ctx_size)
        n_gpu_layers = req.n_gpu_layers or (preset.n_gpu_layers if preset and preset.n_gpu_layers else cfg.default_n_gpu_layers)
        threads = req.threads or (preset.threads if preset and preset.threads else cfg.default_threads)
        batch_size = req.batch_size or (preset.batch_size if preset and preset.batch_size else cfg.default_batch_size)
        n_parallel = req.n_parallel or (preset.n_parallel if preset and preset.n_parallel else cfg.default_n_parallel)
        flash_attn = req.flash_attn or (preset.flash_attn if preset and preset.flash_attn else cfg.default_flash_attn)
        host = req.host or cfg.default_host
        sleep_idle = req.sleep_idle_seconds if req.sleep_idle_seconds is not None else (
            preset.sleep_idle_seconds if preset and preset.sleep_idle_seconds is not None else cfg.default_sleep_idle_seconds
        )
        extra_args = req.extra_args or (preset.extra_args if preset else [])

        inst_id = f"inst-{uuid.uuid4().hex[:8]}"
        log_path = self._logs_dir / f"{inst_id}.log"

        # build command
        cmd = [
            cfg.llama_server_bin,
            "-m", str(model_path),
            "--host", host,
            "--port", str(port),
            "-c", str(ctx_size),
            "-ngl", str(n_gpu_layers),
            "-b", str(batch_size),
            "-np", str(n_parallel),
            "--log-file", str(log_path),
        ]
        if threads is not None:
            cmd.extend(["-t", str(threads)])
        if flash_attn and flash_attn != "none":
            cmd.extend(["-fa", str(flash_attn)])
        if sleep_idle and sleep_idle > 0:
            cmd.extend(["--sleep-idle-seconds", str(sleep_idle)])
        if req.alias:
            cmd.extend(["-a", req.alias])
        cmd.extend(extra_args)

        logger.info("Starting instance %s: %s", inst_id, " ".join(cmd))

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(Path(cfg.llama_server_bin).parent)

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            self._release_port(port)
            raise RuntimeError(f"Failed to start llama-server: {e}")

        info = InstanceInfo(
            id=inst_id,
            model=req.model,
            status=InstanceStatus.starting,
            port=port,
            host=host,
            pid=proc.pid,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            n_parallel=n_parallel,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            extra_args=extra_args,
        )

        inst = InstanceProcess(info=info, process=proc, log_file=log_path)
        self._instances[inst_id] = inst

        # background: wait for ready + health check
        inst._health_task = asyncio.create_task(self._wait_and_monitor(inst_id))

        return info

    async def _wait_and_monitor(self, inst_id: str) -> None:
        inst = self._instances.get(inst_id)
        if not inst:
            return
        cfg = self._config.config
        url = f"http://{inst.info.host}:{inst.info.port}"

        # wait for server to be ready
        for _ in range(60):
            await asyncio.sleep(1)
            if not self._monitor.is_process_alive(inst.info.pid):
                inst.info.status = InstanceStatus.error
                return
            try:
                async with httpx.AsyncClient(timeout=2) as c:
                    r = await c.get(f"{url}/health")
                    if r.status_code == 200:
                        inst.info.status = InstanceStatus.running
                        break
            except Exception:
                continue
        else:
            inst.info.status = InstanceStatus.running

        # periodic health check
        while inst.info.status == InstanceStatus.running:
            await asyncio.sleep(cfg.health_check_interval)
            if not self._monitor.is_process_alive(inst.info.pid):
                inst.info.status = InstanceStatus.error
                logger.warning("Instance %s process died", inst_id)
                if cfg.auto_restart and inst.info.restart_count < cfg.auto_restart_max_retries:
                    inst.info.restart_count += 1
                    await asyncio.sleep(cfg.auto_restart_delay)
                    await self._restart_instance(inst_id)
                break
            # update memory usage
            ram = self._monitor.get_process_memory_mb(inst.info.pid)
            inst.info.ram_usage_mb = round(ram, 1) if ram else None

    async def _restart_instance(self, inst_id: str) -> None:
        inst = self._instances.get(inst_id)
        if not inst:
            return
        old_req = InstanceStartRequest(
            model=inst.info.model,
            port=inst.info.port,
            ctx_size=inst.info.ctx_size,
            n_gpu_layers=inst.info.n_gpu_layers,
            n_parallel=inst.info.n_parallel,
            extra_args=inst.info.extra_args,
        )
        # stop old process
        self._kill_process(inst)
        self._release_port(inst.info.port)
        del self._instances[inst_id]
        # restart
        try:
            await self.start_instance(old_req)
            logger.info("Instance %s restarted successfully", inst_id)
        except Exception as e:
            logger.error("Failed to restart instance %s: %s", inst_id, e)

    def _kill_process(self, inst: InstanceProcess) -> None:
        if inst.process and inst.process.poll() is None:
            try:
                inst.process.send_signal(signal.SIGTERM)
                inst.process.wait(timeout=10)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                inst.process.kill()
            except Exception:
                pass

    async def stop_instance(self, inst_id: str) -> bool:
        inst = self._instances.get(inst_id)
        if not inst:
            return False
        inst.info.status = InstanceStatus.stopping
        if inst._health_task:
            inst._health_task.cancel()
        self._kill_process(inst)
        self._release_port(inst.info.port)
        del self._instances[inst_id]
        return True

    async def restart_instance(self, inst_id: str) -> InstanceInfo:
        inst = self._instances.get(inst_id)
        if not inst:
            raise KeyError(f"Instance {inst_id} not found")
        old_req = InstanceStartRequest(
            model=inst.info.model,
            port=inst.info.port,
            ctx_size=inst.info.ctx_size,
            n_gpu_layers=inst.info.n_gpu_layers,
            n_parallel=inst.info.n_parallel,
            extra_args=inst.info.extra_args,
        )
        self._kill_process(inst)
        self._release_port(inst.info.port)
        del self._instances[inst_id]
        return await self.start_instance(old_req)

    def list_instances(self) -> List[InstanceInfo]:
        return [inst.info for inst in self._instances.values()]

    def get_instance(self, inst_id: str) -> Optional[InstanceInfo]:
        inst = self._instances.get(inst_id)
        if inst:
            # refresh stats
            if inst.process and inst.info.pid:
                alive = self._monitor.is_process_alive(inst.info.pid)
                if not alive and inst.info.status == InstanceStatus.running:
                    inst.info.status = InstanceStatus.error
                ram = self._monitor.get_process_memory_mb(inst.info.pid)
                inst.info.ram_usage_mb = round(ram, 1) if ram else None
            return inst.info
        return None

    def get_instance_logs(self, inst_id: str, lines: int = 100) -> List[str]:
        inst = self._instances.get(inst_id)
        if not inst or not inst.log_file or not inst.log_file.exists():
            return []
        try:
            with open(inst.log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            return [l.rstrip("\n") for l in all_lines[-lines:]]
        except Exception:
            return []

    # ---- shutdown -----------------------------------------------------------

    async def shutdown(self) -> None:
        inst_ids = list(self._instances.keys())
        for inst_id in inst_ids:
            await self.stop_instance(inst_id)
        logger.info("All instances stopped")
