"""System-related routes: info, health, backends, config, version, stats, GPUs."""

from __future__ import annotations

import sys
from datetime import datetime

from fastapi import APIRouter

from ..models import ConfigUpdate, DefaultConfig, HealthResponse, SystemInfo
from ..utils import get_server_paths
from . import get_config, get_manager

router = APIRouter()


@router.get("/system/info", response_model=SystemInfo)
async def system_info():
    cfg = get_config().config
    server_paths = get_server_paths(cfg)
    return get_manager().monitor.get_system_info(server_paths)


@router.get("/system/health", response_model=HealthResponse)
async def system_health():
    instances = get_manager().list_instances()
    running = sum(1 for i in instances if i.status.value == "running")
    return HealthResponse(
        status="ok",
        instances_running=running,
        instances_total=len(instances),
    )


@router.get("/system/backends")
async def list_backends():
    """List all supported inference backends with installation status."""
    from ..backends import get_all_backends_status
    backends = get_all_backends_status()
    return {
        "backends": backends,
        "default": get_config().config.default_backend.value,
    }


@router.get("/system/config", response_model=DefaultConfig)
async def get_system_config():
    return get_config().config


@router.put("/system/config", response_model=DefaultConfig)
async def update_config(body: ConfigUpdate):
    data = body.model_dump(exclude_unset=True)
    return get_config().update_config(**data)


@router.get("/system/version")
async def system_version():
    """Get manager version and build info."""
    return {
        "version": "1.0.0",
        "name": "Inference Server Manager",
        "build_date": datetime.now().strftime("%Y-%m-%d"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


@router.get("/system/stats")
async def system_stats():
    """Get system usage statistics."""
    mgr = get_manager()
    instances = mgr.list_instances()
    by_backend = {}
    by_status = {}
    for inst in instances:
        backend = inst.backend.value
        status = inst.status.value
        by_backend[backend] = by_backend.get(backend, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1

    return {
        "total_instances": len(instances),
        "by_backend": by_backend,
        "by_status": by_status,
        "total_models": len(mgr.list_models()),
        "total_presets": len(get_config().list_presets()),
    }


@router.get("/system/gpus")
async def system_gpus():
    """Get detailed GPU information."""
    gpus = get_manager().monitor.get_gpus()
    return {
        "gpus": [g.model_dump() for g in gpus],
        "count": len(gpus),
    }


@router.get("/system/discover")
async def discover_processes():
    """Discover external GPU processes that may be inference backends."""
    procs = get_manager().monitor.detect_gpu_processes()
    return {"processes": procs}


@router.get("/system/export")
async def export_config():
    """Export current configuration as JSON."""
    config = get_config()
    return {
        "config": config.config.model_dump(),
        "presets": {name: p.model_dump() for name, p in config.list_presets().items()},
        "exported_at": datetime.now().isoformat(),
    }


from pydantic import BaseModel


class ImportRequest(BaseModel):
    config: dict | None = None
    presets: dict | None = None
    overwrite: bool = False


@router.post("/system/import")
async def import_config(body: ImportRequest):
    """Import configuration from JSON."""
    config = get_config()
    imported = {"config": False, "presets": 0}

    if body.config:
        config.update_config(**body.config)
        imported["config"] = True

    if body.presets:
        from ..models import Preset
        for name, preset_data in body.presets.items():
            preset_data["name"] = name
            preset = Preset(**preset_data)
            config.save_preset(preset)
            imported["presets"] += 1

    return imported
