"""Model-related routes: listing, info, download, delete."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..models import DownloadProgress, DownloadRequest
from . import audit_log, get_manager

router = APIRouter()


@router.get("/models")
async def list_models():
    return get_manager().list_models()


@router.get("/models/{name:path}/info")
async def model_info(name: str):
    info = get_manager().get_model_info(name)
    if not info:
        raise HTTPException(404, f"Model not found: {name}")
    return info


@router.get("/models/online")
async def online_models():
    instances = get_manager().list_instances()
    return [i for i in instances if i.status.value in ("running", "starting")]


@router.post("/models/download", response_model=DownloadProgress)
async def download_model(body: DownloadRequest):
    if body.source.value in ("hf", "hf_mirror") and not body.repo:
        raise HTTPException(400, "repo is required for hf/hf_mirror source")
    if body.source.value == "ms" and not body.repo:
        raise HTTPException(400, "repo is required for modelscope source")
    if body.source.value == "url" and not body.url:
        raise HTTPException(400, "url is required for url source")
    task = await get_manager().downloader.start_download(body)
    audit_log("model.download", "model", task.task_id, source=body.source.value, repo=body.repo or body.url)
    return task


@router.post("/models/download/safetensors")
async def auto_download_safetensors(source: str = "auto"):
    """Auto-download safetensor versions for all gguf models."""
    models = get_manager().list_models()
    gguf_models = [m["name"] for m in models if m["name"].endswith(".gguf")]

    if not gguf_models:
        return {"message": "No GGUF models found", "results": []}

    results = await get_manager().downloader.auto_download_safetensors(gguf_models, source=source)
    return {"message": f"Processed {len(results)} models", "results": results}


@router.get("/models/download/status")
async def download_status():
    return get_manager().downloader.tasks


@router.get("/models/download/{task_id}")
async def download_task_status(task_id: str):
    task = get_manager().downloader.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Download task not found: {task_id}")
    return task


@router.delete("/models/download/{task_id}")
async def cancel_download(task_id: str):
    """Cancel a download task."""
    ok = get_manager().downloader.cancel_task(task_id)
    if not ok:
        raise HTTPException(404, f"Download task not found: {task_id}")
    return {"success": True}


@router.delete("/models/{name:path}")
async def delete_model(name: str):
    ok = get_manager().delete_model(name)
    if not ok:
        raise HTTPException(404, f"Model not found: {name}")
    return {"success": True}
