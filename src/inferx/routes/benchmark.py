"""Benchmark routes: reports, batches, GPU info."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import get_manager

router = APIRouter()

# Lazy-initialized benchmark manager
_benchmark_manager = None


def _bench():
    global _benchmark_manager
    if _benchmark_manager is None:
        from ..benchmark import BenchmarkManager
        data_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks"
        _benchmark_manager = BenchmarkManager(data_dir)
        mgr = get_manager()
        if mgr:
            _benchmark_manager.set_manager(mgr)
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
    from ..benchmark import BatchBenchmarkConfig
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
    from ..monitor import ResourceMonitor
    monitor = ResourceMonitor()
    gpus = monitor.get_gpus()
    if not gpus:
        return {"available": False}
    return {
        "available": True,
        "count": len(gpus),
        "gpus": [{"index": g.index, "name": g.name, "total_mb": g.total_memory_mb, "used_mb": g.used_memory_mb} for g in gpus],
    }
