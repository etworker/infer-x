"""FastAPI route definitions."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel

from .config import ConfigManager
from .manager import InstanceManager
from .models import (
    ConfigUpdate,
    DefaultConfig,
    DownloadProgress,
    DownloadRequest,
    HealthResponse,
    InstanceInfo,
    InstanceList,
    InstanceLogs,
    InstanceStartRequest,
    Preset,
    SystemInfo,
)
from .monitoring import AlertManager, AlertRule, AuditLogger, UsageTracker

router = APIRouter(prefix="/api")

# These are set during app startup in main.py
_config: ConfigManager | None = None
_manager: InstanceManager | None = None
_alert_manager: AlertManager | None = None
_usage_tracker: UsageTracker | None = None
_audit_logger: AuditLogger | None = None


def init_routes(config: ConfigManager, manager: InstanceManager) -> None:
    global _config, _manager, _alert_manager, _usage_tracker, _audit_logger
    _config = config
    _manager = manager

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    _alert_manager = AlertManager(data_dir)
    _usage_tracker = UsageTracker(data_dir)
    _audit_logger = AuditLogger(data_dir)

    # Initialize benchmark manager with manager
    _bench().set_manager(manager)


def _mgr() -> InstanceManager:
    assert _manager is not None
    return _manager


def _cfg() -> ConfigManager:
    assert _config is not None
    return _config


def _alerts() -> AlertManager:
    assert _alert_manager is not None
    return _alert_manager


def _usage() -> UsageTracker:
    assert _usage_tracker is not None
    return _usage_tracker


def _audit() -> AuditLogger:
    assert _audit_logger is not None
    return _audit_logger


# ===========================================================================
# System
# ===========================================================================

@router.get("/system/info", response_model=SystemInfo)
async def system_info():
    cfg = _cfg().config
    server_paths = {
        "llamacpp": cfg.llama_server_bin,
        "vllm": cfg.vllm_server_bin,
        "sglang": cfg.sglang_server_bin,
        "tgi": cfg.tgi_bin,
        "ollama": cfg.ollama_bin,
        "tensorrt_llm": cfg.tensorrt_llm_bin,
        "lmdeploy": cfg.lmdeploy_bin,
        "openvino": cfg.openvino_bin,
    }
    return _mgr().monitor.get_system_info(server_paths)


@router.get("/system/health", response_model=HealthResponse)
async def system_health():
    instances = _mgr().list_instances()
    running = sum(1 for i in instances if i.status.value == "running")
    return HealthResponse(
        status="ok",
        instances_running=running,
        instances_total=len(instances),
    )


@router.get("/system/backends")
async def list_backends():
    """List all supported inference backends with installation status."""
    from .backends import get_all_backends_status
    backends = get_all_backends_status()
    return {
        "backends": backends,
        "default": _cfg().config.default_backend.value,
    }


@router.get("/system/config", response_model=DefaultConfig)
async def get_config():
    return _cfg().config


@router.put("/system/config", response_model=DefaultConfig)
async def update_config(body: ConfigUpdate):
    data = body.model_dump(exclude_unset=True)
    return _cfg().update_config(**data)


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
    instances = _mgr().list_instances()
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
        "total_models": len(_mgr().list_models()),
        "total_presets": len(_cfg().list_presets()),
    }


@router.get("/system/gpus")
async def system_gpus():
    """Get detailed GPU information."""
    gpus = _mgr().monitor.get_gpus()
    return {
        "gpus": [g.model_dump() for g in gpus],
        "count": len(gpus),
    }


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


@router.post("/models/download/safetensors")
async def auto_download_safetensors(source: str = "auto"):
    """Auto-download safetensor versions for all gguf models.

    Args:
        source: "hf" for HuggingFace only, "ms" for ModelScope only,
                "auto" for try HF first, fallback to ModelScope
    """
    models = _mgr().list_models()
    gguf_models = [m["name"] for m in models if m["name"].endswith(".gguf")]

    if not gguf_models:
        return {"message": "No GGUF models found", "results": []}

    results = await _mgr().downloader.auto_download_safetensors(gguf_models, source=source)
    return {"message": f"Processed {len(results)} models", "results": results}


@router.get("/models/download/status")
async def download_status():
    return _mgr().downloader.tasks


@router.get("/models/download/{task_id}")
async def download_task_status(task_id: str):
    task = _mgr().downloader.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Download task not found: {task_id}")
    return task


@router.delete("/models/download/{task_id}")
async def cancel_download(task_id: str):
    """Cancel a download task."""
    ok = _mgr().downloader.cancel_task(task_id)
    if not ok:
        raise HTTPException(404, f"Download task not found: {task_id}")
    return {"success": True}


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
        raise HTTPException(404, detail={"error": "model_not_found", "message": str(e)})
    except RuntimeError as e:
        raise HTTPException(400, detail={"error": "startup_failed", "message": str(e)})
    except Exception as e:
        raise HTTPException(500, detail={"error": "internal_error", "message": str(e)})


@router.get("/instances", response_model=InstanceList)
async def list_instances(
    backend: str | None = Query(None, description="Filter by backend"),
    status: str | None = Query(None, description="Filter by status"),
):
    instances = _mgr().list_instances()
    if backend:
        instances = [i for i in instances if i.backend.value == backend]
    if status:
        instances = [i for i in instances if i.status.value == status]
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


@router.get("/instances/{inst_id}/logs/raw")
async def instance_logs_raw(inst_id: str, lines: int = Query(default=100)):
    """Get raw log lines as text."""
    logs = _mgr().get_instance_logs(inst_id, lines)
    return StreamingResponse(
        iter(["\n".join(logs)]),
        media_type="text/plain",
    )


@router.get("/instances/{inst_id}/error")
async def instance_error(inst_id: str):
    """Get error logs for a failed instance."""
    inst = _mgr().get_instance(inst_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {inst_id}")

    # Read stderr log if exists
    stderr_path = _mgr()._logs_dir / f"{inst_id}.stderr"
    error_logs = []
    if stderr_path.exists():
        try:
            with open(stderr_path, encoding="utf-8", errors="replace") as f:
                error_logs = f.readlines()[-50:]  # Last 50 lines
        except Exception:
            pass

    return {
        "instance_id": inst_id,
        "status": inst.status.value,
        "error_logs": [line.rstrip("\n") for line in error_logs],
    }


# ===========================================================================
# Batch Operations
# ===========================================================================

class BatchStopRequest(BaseModel):
    instance_ids: list[str]


class BatchStartRequest(BaseModel):
    requests: list[InstanceStartRequest]


@router.post("/instances/batch/start")
async def batch_start_instances(body: BatchStartRequest):
    """Start multiple instances at once."""
    results = []
    for req in body.requests:
        try:
            info = await _mgr().start_instance(req)
            results.append({"success": True, "instance_id": info.id, "model": req.model})
        except Exception as e:
            results.append({"success": False, "model": req.model, "error": str(e)})
    return {"results": results}


@router.post("/instances/batch/stop")
async def batch_stop_instances(body: BatchStopRequest):
    """Stop multiple instances at once."""
    results = []
    for inst_id in body.instance_ids:
        try:
            ok = await _mgr().stop_instance(inst_id)
            results.append({"success": ok, "instance_id": inst_id})
        except Exception as e:
            results.append({"success": False, "instance_id": inst_id, "error": str(e)})
    return {"results": results}


@router.post("/instances/batch/restart")
async def batch_restart_instances(body: BatchStopRequest):
    """Restart multiple instances at once."""
    results = []
    for inst_id in body.instance_ids:
        try:
            info = await _mgr().restart_instance(inst_id)
            results.append({"success": True, "instance_id": info.id})
        except Exception as e:
            results.append({"success": False, "instance_id": inst_id, "error": str(e)})
    return {"results": results}


@router.post("/instances/stop-all")
async def stop_all_instances():
    """Stop all running instances."""
    instances = _mgr().list_instances()
    results = []
    for inst in instances:
        try:
            ok = await _mgr().stop_instance(inst.id)
            results.append({"success": ok, "instance_id": inst.id})
        except Exception as e:
            results.append({"success": False, "instance_id": inst.id, "error": str(e)})
    return {"stopped": len([r for r in results if r["success"]]), "results": results}


# ===========================================================================
# Proxy (Forward requests to backend instances)
# ===========================================================================


@router.api_route("/proxy/{inst_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_instance(inst_id: str, path: str, body: dict | None = None):
    """Proxy request to an instance backend (e.g., /api/proxy/inst-abc/v1/chat/completions)."""
    inst = _mgr().get_instance(inst_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    if inst.status.value != "running":
        raise HTTPException(400, f"Instance is not running (status: {inst.status.value})")

    url = f"http://{inst.host}:{inst.port}/{path}"
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            if body:
                resp = await client.post(url, json=body)
            else:
                resp = await client.get(url)
            return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "Backend request timed out")
    except Exception as e:
        raise HTTPException(502, f"Backend request failed: {str(e)}")


# ===========================================================================
# Export/Import
# ===========================================================================

@router.get("/system/export")
async def export_config():
    """Export current configuration as JSON."""
    config = _cfg().config
    presets = _cfg().list_presets()
    return {
        "config": config.model_dump(),
        "presets": {name: p.model_dump() for name, p in presets.items()},
        "exported_at": datetime.now().isoformat(),
    }


class ImportRequest(BaseModel):
    config: dict[str, Any] | None = None
    presets: dict[str, Any] | None = None
    overwrite: bool = False


@router.post("/system/import")
async def import_config(body: ImportRequest):
    """Import configuration from JSON."""
    imported = {"config": False, "presets": 0}

    if body.config:
        _cfg().update_config(**body.config)
        imported["config"] = True

    if body.presets:
        for name, preset_data in body.presets.items():
            preset_data["name"] = name
            preset = Preset(**preset_data)
            _cfg().save_preset(preset)
            imported["presets"] += 1

    return imported


# ===========================================================================
# Tags (Instance labeling)
# ===========================================================================

@router.post("/instances/{inst_id}/tags")
async def add_instance_tags(inst_id: str, tags: dict[str, str]):
    """Add tags to an instance."""
    inst = _mgr().get_instance(inst_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {inst_id}")

    # Store tags in instance (we'd need to add this to InstanceInfo)
    # For now, return the tags
    return {"instance_id": inst_id, "tags": tags}


@router.get("/instances/{inst_id}/tags")
async def get_instance_tags(inst_id: str):
    """Get tags for an instance."""
    inst = _mgr().get_instance(inst_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    # Return empty tags for now
    return {"instance_id": inst_id, "tags": {}}


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


@router.post("/presets/{name}/clone")
async def clone_preset(name: str, new_name: str):
    """Clone a preset with a new name."""
    preset = _cfg().get_preset(name)
    if not preset:
        raise HTTPException(404, f"Preset not found: {name}")

    new_preset = preset.model_copy(update={"name": new_name})
    return _cfg().save_preset(new_preset)


# ===========================================================================
# Monitoring & Alerts
# ===========================================================================

class CreateAlertRuleRequest(BaseModel):
    name: str
    enabled: bool = True
    metric: str
    condition: str
    threshold: float
    duration_seconds: int = 60
    cooldown_seconds: int = 300
    notify_channels: list[str] = ["log"]
    message_template: str = ""


@router.get("/alerts/rules")
async def list_alert_rules():
    """List all alert rules."""
    return {"rules": [r.model_dump() for r in _alerts().list_rules()]}


@router.post("/alerts/rules")
async def create_alert_rule(body: CreateAlertRuleRequest):
    """Create a new alert rule."""
    import uuid
    rule = AlertRule(
        id=f"rule-{uuid.uuid4().hex[:8]}",
        **body.model_dump()
    )
    return _alerts().create_rule(rule).model_dump()


@router.put("/alerts/rules/{rule_id}")
async def update_alert_rule(rule_id: str, body: CreateAlertRuleRequest):
    """Update an alert rule."""
    rule = _alerts().update_rule(rule_id, body.model_dump())
    if not rule:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    return rule.model_dump()


@router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str):
    """Delete an alert rule."""
    ok = _alerts().delete_rule(rule_id)
    if not ok:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    return {"success": True}


@router.get("/alerts")
async def list_alerts(status: str | None = None):
    """List active/resolved alerts."""
    return {"alerts": [a.model_dump() for a in _alerts().list_alerts(status)]}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    ok = _alerts().acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(404, f"Alert not found: {alert_id}")
    return {"success": True}


@router.get("/alerts/check")
async def check_alerts():
    """Manually trigger alert check with current system metrics."""
    gpus = _mgr().monitor.get_gpus()
    new_alerts = []

    for gpu in gpus:
        mem_pct = (gpu.used_memory_mb / gpu.total_memory_mb * 100) if gpu.total_memory_mb > 0 else 0
        alerts = _alerts().check_metric("gpu_memory_pct", mem_pct)
        new_alerts.extend(alerts)

        if gpu.utilization_pct is not None:
            alerts = _alerts().check_metric("gpu_utilization", gpu.utilization_pct)
            new_alerts.extend(alerts)

    # Check instance count
    instances = _mgr().list_instances()
    running = sum(1 for i in instances if i.status.value == "running")
    alerts = _alerts().check_metric("instance_count", running)
    new_alerts.extend(alerts)

    return {
        "checked_at": datetime.now().isoformat(),
        "new_alerts": [a.model_dump() for a in new_alerts],
    }


# ===========================================================================
# Usage Statistics
# ===========================================================================

@router.get("/stats/overview")
async def usage_stats_overview():
    """Get overall usage statistics."""
    return _usage().get_overall_stats().model_dump()


@router.get("/stats/models")
async def usage_stats_models():
    """Get per-model usage statistics."""
    return {"models": [m.model_dump() for m in _usage().get_model_stats()]}


@router.get("/stats/hourly")
async def usage_stats_hourly(days: int = Query(default=7, ge=1, le=30)):
    """Get hourly request counts."""
    return {"hourly": _usage().get_hourly_stats(days)}


# ===========================================================================
# Audit Log
# ===========================================================================

class AuditLogQuery(BaseModel):
    action: str | None = None
    target_type: str | None = None
    limit: int = 100
    offset: int = 0


@router.get("/audit")
async def list_audit_logs(
    action: str | None = None,
    target_type: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List audit log entries."""
    entries = _audit().list_entries(action, target_type, limit, offset)
    return {
        "entries": [e.model_dump() for e in entries],
        "total": len(_audit()._entries),
    }


@router.get("/audit/stats")
async def audit_stats():
    """Get audit log statistics."""
    return _audit().get_stats()


@router.get("/audit/{entry_id}")
async def get_audit_entry(entry_id: str):
    """Get a specific audit entry."""
    entry = _audit().get_entry(entry_id)
    if not entry:
        raise HTTPException(404, f"Audit entry not found: {entry_id}")
    return entry.model_dump()


# ===========================================================================
# Benchmark
# ===========================================================================

_benchmark_manager = None


def _bench():
    global _benchmark_manager
    if _benchmark_manager is None:
        from .benchmark import BenchmarkManager
        data_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks"
        _benchmark_manager = BenchmarkManager(data_dir)
        if _manager:
            _benchmark_manager.set_manager(_manager)
    return _benchmark_manager


# Reports
@router.get("/benchmark/reports")
async def list_reports():
    return {"reports": _bench().list_reports()}

@router.get("/benchmark/reports/{report_id}")
async def get_report(report_id: str):
    report = _bench().get_report(report_id)
    if not report:
        raise HTTPException(404, "Report not found")
    return report.model_dump()

@router.delete("/benchmark/reports/{report_id}")
async def delete_report(report_id: str):
    ok = _bench().delete_report(report_id)
    if not ok:
        raise HTTPException(404, "Report not found")
    return {"success": True}

# Batches
class BatchRequest(BaseModel):
    models: list[str]
    backends: list[str]
    num_iterations: int = 3
    timeout_seconds: int = 120

@router.get("/benchmark/batches")
async def list_batches():
    return {"batches": _bench().list_batches()}

@router.get("/benchmark/batches/{batch_id}")
async def get_batch(batch_id: str):
    progress = _bench().get_batch_progress(batch_id)
    if not progress:
        raise HTTPException(404, "Batch not found")
    return progress.model_dump()

@router.post("/benchmark/batches")
async def start_batch(body: BatchRequest):
    from .benchmark import BatchBenchmarkConfig
    config = BatchBenchmarkConfig(
        models=body.models,
        backends=body.backends,
        num_iterations=body.num_iterations,
        timeout_seconds=body.timeout_seconds,
    )
    batch_id = await _bench().start_batch(config)
    return {"batch_id": batch_id, "status": "started"}

@router.post("/benchmark/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    ok = _bench().cancel_batch(batch_id)
    if not ok:
        raise HTTPException(404, "Batch not found")
    return {"success": True}

@router.delete("/benchmark/batches/{batch_id}")
async def delete_batch(batch_id: str):
    ok = _bench().delete_batch(batch_id)
    if not ok:
        raise HTTPException(404, "Batch not found")
    return {"success": True}

# GPU info
@router.get("/benchmark/gpu")
async def gpu_info():
    from .benchmark import GPUMonitor
    return GPUMonitor().get_gpu_info()
