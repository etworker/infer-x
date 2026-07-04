"""Instance-related routes: CRUD, batch operations, logs, proxy."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx

from ..models import InstanceInfo, InstanceList, InstanceLogs, InstanceStartRequest, LogEntry
from . import get_audit_logger, get_manager

router = APIRouter()


# --- Single instance operations ---

@router.post("/instances", response_model=InstanceInfo)
async def start_instance(body: InstanceStartRequest):
    audit = get_audit_logger()
    try:
        info = await get_manager().start_instance(body)
        audit.log("instance.start", "instance", info.id, details={"model": body.model, "backend": str(body.backend)})
        return info
    except FileNotFoundError as e:
        audit.log("instance.start", "instance", body.model, success=False, error_message=str(e))
        raise HTTPException(404, detail={"error": "model_not_found", "message": str(e)})
    except RuntimeError as e:
        audit.log("instance.start", "instance", body.model, success=False, error_message=str(e))
        raise HTTPException(400, detail={"error": "startup_failed", "message": str(e)})
    except Exception as e:
        audit.log("instance.start", "instance", body.model, success=False, error_message=str(e))
        raise HTTPException(500, detail={"error": "internal_error", "message": str(e)})


@router.get("/instances", response_model=InstanceList)
async def list_instances(
    backend: str | None = Query(None, description="Filter by backend"),
    status: str | None = Query(None, description="Filter by status"),
):
    instances = get_manager().list_instances()
    if backend:
        instances = [i for i in instances if i.backend.value == backend]
    if status:
        instances = [i for i in instances if i.status.value == status]
    return InstanceList(instances=instances, total=len(instances))


@router.get("/instances/{inst_id}", response_model=InstanceInfo)
async def get_instance(inst_id: str):
    info = get_manager().get_instance(inst_id)
    if not info:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    return info


@router.delete("/instances/{inst_id}")
async def stop_instance(inst_id: str):
    ok = await get_manager().stop_instance(inst_id)
    audit = get_audit_logger()
    if not ok:
        audit.log("instance.stop", "instance", inst_id, success=False, error_message="not found")
        raise HTTPException(404, f"Instance not found: {inst_id}")
    audit.log("instance.stop", "instance", inst_id)
    return {"success": True}


@router.post("/instances/{inst_id}/restart", response_model=InstanceInfo)
async def restart_instance(inst_id: str):
    try:
        return await get_manager().restart_instance(inst_id)
    except KeyError:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    except RuntimeError as e:
        raise HTTPException(400, str(e))


# --- Logs ---

@router.get("/instances/{inst_id}/logs")
async def instance_logs(inst_id: str, lines: int = Query(default=100, ge=1, le=10000)):
    logs = get_manager().get_instance_logs(inst_id, lines)
    log_entries = [LogEntry(timestamp="", level="INFO", message=line) for line in logs]
    return InstanceLogs(instance_id=inst_id, logs=log_entries, total_lines=len(logs))


@router.get("/instances/{inst_id}/logs/raw")
async def instance_logs_raw(inst_id: str, lines: int = Query(default=100)):
    """Get raw log lines as text."""
    logs = get_manager().get_instance_logs(inst_id, lines)
    return StreamingResponse(
        iter(["\n".join(logs)]),
        media_type="text/plain",
    )


@router.get("/instances/{inst_id}/error")
async def instance_error(inst_id: str):
    """Get error logs for a failed instance."""
    mgr = get_manager()
    inst = mgr.get_instance(inst_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {inst_id}")

    stderr_path = mgr._logs_dir / f"{inst_id}.stderr"
    error_logs = []
    if stderr_path.exists():
        try:
            with open(stderr_path, encoding="utf-8", errors="replace") as f:
                error_logs = f.readlines()[-50:]
        except Exception:
            pass

    return {
        "instance_id": inst_id,
        "status": inst.status.value,
        "error_logs": [line.rstrip("\n") for line in error_logs],
    }


# --- Tags ---

@router.post("/instances/{inst_id}/tags")
async def add_instance_tags(inst_id: str, tags: dict[str, str]):
    mgr = get_manager()
    inst = mgr.get_instance(inst_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    # 直接更新实例的 tags（需要通过 process_manager 修改）
    proc_inst = mgr._proc_mgr.instances.get(inst_id)
    if proc_inst:
        proc_inst.info.tags.update(tags)
    return {"instance_id": inst_id, "tags": inst.tags if inst else tags}


@router.get("/instances/{inst_id}/tags")
async def get_instance_tags(inst_id: str):
    mgr = get_manager()
    inst = mgr.get_instance(inst_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    return {"instance_id": inst_id, "tags": inst.tags}


# --- Batch operations ---

class BatchStopRequest(BaseModel):
    instance_ids: list[str]


class BatchStartRequest(BaseModel):
    requests: list[InstanceStartRequest]


@router.post("/instances/batch/start")
async def batch_start_instances(body: BatchStartRequest):
    """Start multiple instances at once."""
    mgr = get_manager()
    results = []
    for req in body.requests:
        try:
            info = await mgr.start_instance(req)
            results.append({"success": True, "instance_id": info.id, "model": req.model})
        except Exception as e:
            results.append({"success": False, "model": req.model, "error": str(e)})
    return {"results": results}


@router.post("/instances/batch/stop")
async def batch_stop_instances(body: BatchStopRequest):
    """Stop multiple instances at once."""
    mgr = get_manager()
    results = []
    for inst_id in body.instance_ids:
        try:
            ok = await mgr.stop_instance(inst_id)
            results.append({"success": ok, "instance_id": inst_id})
        except Exception as e:
            results.append({"success": False, "instance_id": inst_id, "error": str(e)})
    return {"results": results}


@router.post("/instances/batch/restart")
async def batch_restart_instances(body: BatchStopRequest):
    """Restart multiple instances at once."""
    mgr = get_manager()
    results = []
    for inst_id in body.instance_ids:
        try:
            info = await mgr.restart_instance(inst_id)
            results.append({"success": True, "instance_id": info.id})
        except Exception as e:
            results.append({"success": False, "instance_id": inst_id, "error": str(e)})
    return {"results": results}


@router.post("/instances/stop-all")
async def stop_all_instances():
    """Stop all running instances."""
    mgr = get_manager()
    instances = mgr.list_instances()
    results = []
    for inst in instances:
        try:
            ok = await mgr.stop_instance(inst.id)
            results.append({"success": ok, "instance_id": inst.id})
        except Exception as e:
            results.append({"success": False, "instance_id": inst.id, "error": str(e)})
    return {"stopped": len([r for r in results if r["success"]]), "results": results}


# --- Proxy ---

@router.api_route("/proxy/{inst_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_to_instance(inst_id: str, path: str, request: Request):
    """Proxy request to an instance backend."""
    mgr = get_manager()
    inst = mgr.get_instance(inst_id)
    if not inst:
        raise HTTPException(404, f"Instance not found: {inst_id}")
    if inst.status.value != "running":
        raise HTTPException(400, f"Instance is not running (status: {inst.status.value})")

    url = f"http://{inst.host}:{inst.port}/{path}"
    try:
        client = mgr.http_client
        body = None
        if request.method in ("POST", "PUT"):
            body = await request.json()
        if request.method == "GET":
            resp = await client.get(url, timeout=120)
        elif request.method == "POST":
            resp = await client.post(url, json=body, timeout=120)
        elif request.method == "PUT":
            resp = await client.put(url, json=body, timeout=120)
        elif request.method == "DELETE":
            resp = await client.delete(url, timeout=120)
        return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(504, "Backend request timed out")
    except Exception as e:
        raise HTTPException(502, f"Backend request failed: {str(e)}")
