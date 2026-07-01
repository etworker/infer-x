"""FastAPI route definitions."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config import ConfigManager
from manager import InstanceManager
from models import (
    ConfigUpdate,
    DefaultConfig,
    DownloadRequest,
    DownloadProgress,
    HealthResponse,
    InstanceInfo,
    InstanceList,
    InstanceLogs,
    InstanceStartRequest,
    ModelInfo,
    Preset,
    SystemInfo,
)

router = APIRouter(prefix="/api")

# These are set during app startup in main.py
_config: Optional[ConfigManager] = None
_manager: Optional[InstanceManager] = None


def init_routes(config: ConfigManager, manager: InstanceManager) -> None:
    global _config, _manager
    _config = config
    _manager = manager


def _mgr() -> InstanceManager:
    assert _manager is not None
    return _manager


def _cfg() -> ConfigManager:
    assert _config is not None
    return _config


# ===========================================================================
# System
# ===========================================================================

@router.get("/system/info", response_model=SystemInfo)
async def system_info():
    return _mgr().monitor.get_system_info(_cfg().config.llama_server_bin)


@router.get("/system/health", response_model=HealthResponse)
async def system_health():
    instances = _mgr().list_instances()
    running = sum(1 for i in instances if i.status.value == "running")
    return HealthResponse(
        status="ok",
        instances_running=running,
        instances_total=len(instances),
    )


@router.get("/system/config", response_model=DefaultConfig)
async def get_config():
    return _cfg().config


@router.put("/system/config", response_model=DefaultConfig)
async def update_config(body: ConfigUpdate):
    data = body.model_dump(exclude_unset=True)
    return _cfg().update_config(**data)


# ===========================================================================
# Models
# ===========================================================================

@router.get("/models")
async def list_models():
    return _mgr().list_models()


@router.get("/models/{name:path}/info")
async def model_info(name: str):
    info = _mgr().get_model_info(name)
    if not info:
        raise HTTPException(404, f"Model not found: {name}")
    return info


@router.get("/models/online")
async def online_models():
    instances = _mgr().list_instances()
    return [i for i in instances if i.status.value in ("running", "starting")]


@router.post("/models/download", response_model=DownloadProgress)
async def download_model(body: DownloadRequest):
    if body.source.value in ("hf", "hf_mirror") and not body.repo:
        raise HTTPException(400, "repo is required for hf/hf_mirror source")
    if body.source.value == "ms" and not body.repo:
        raise HTTPException(400, "repo is required for modelscope source")
    if body.source.value == "url" and not body.url:
        raise HTTPException(400, "url is required for url source")
    return await _mgr().downloader.start_download(body)


@router.get("/models/download/status")
async def download_status():
    return _mgr().downloader.tasks


@router.get("/models/download/{task_id}")
async def download_task_status(task_id: str):
    task = _mgr().downloader.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Download task not found: {task_id}")
    return task


@router.delete("/models/{name:path}")
async def delete_model(name: str):
    ok = _mgr().delete_model(name)
    if not ok:
        raise HTTPException(404, f"Model not found: {name}")
    return {"success": True}


# ===========================================================================
# Instances
# ===========================================================================

@router.post("/instances", response_model=InstanceInfo)
async def start_instance(body: InstanceStartRequest):
    try:
        return await _mgr().start_instance(body)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.get("/instances", response_model=InstanceList)
async def list_instances():
    instances = _mgr().list_instances()
    return InstanceList(instances=instances, total=len(instances))


@router.get("/instances/{inst_id}", response_model=InstanceInfo)
async def get_instance(inst_id: str):
    info = _mgr().get_instance(inst_id)
    if not info:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    return info


@router.delete("/instances/{inst_id}")
async def stop_instance(inst_id: str):
    ok = await _mgr().stop_instance(inst_id)
    if not ok:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    return {"success": True}


@router.post("/instances/{inst_id}/restart", response_model=InstanceInfo)
async def restart_instance(inst_id: str):
    try:
        return await _mgr().restart_instance(inst_id)
    except KeyError:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    except RuntimeError as e:
        raise HTTPException(400, str(e))


@router.get("/instances/{inst_id}/logs")
async def instance_logs(inst_id: str, lines: int = Query(default=100, ge=1, le=10000)):
    logs = _mgr().get_instance_logs(inst_id, lines)
    return InstanceLogs(instance_id=inst_id, logs=[], total_lines=len(logs))


# ===========================================================================
# Presets
# ===========================================================================

@router.get("/presets")
async def list_presets():
    return _cfg().list_presets()


@router.post("/presets", response_model=Preset)
async def create_preset(body: Preset):
    return _cfg().save_preset(body)


@router.put("/presets/{name}", response_model=Preset)
async def update_preset(name: str, body: Preset):
    body.name = name
    return _cfg().save_preset(body)


@router.delete("/presets/{name}")
async def delete_preset(name: str):
    ok = _cfg().delete_preset(name)
    if not ok:
        raise HTTPException(404, f"Preset not found: {name}")
    return {"success": True}
