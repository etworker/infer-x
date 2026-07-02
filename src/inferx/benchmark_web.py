"""Benchmark web interface support with batch testing and progress tracking."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .benchmark import (
    BenchmarkConfig,
    BenchmarkManager,
    BatchBenchmarkReport,
)


class BatchBenchmarkConfig(BaseModel):
    """Configuration for batch benchmark testing."""
    models: list[str]
    backends: list[str]
    scenarios: list[dict[str, Any]] | None = None
    num_iterations: int = 3
    warmup_iterations: int = 1
    timeout_seconds: int = 120


class BenchmarkProgress(BaseModel):
    """Progress tracking for benchmark runs."""
    batch_id: str
    status: str  # pending, running, completed, failed
    total_tasks: int = 0
    completed_tasks: int = 0
    current_task: str | None = None
    current_model: str | None = None
    current_backend: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []


class BenchmarkWebManager:
    """Manages benchmark execution with web interface support."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._executor = BenchmarkManager(data_dir)
        self._active_batches: dict[str, BenchmarkProgress] = {}

    def set_manager(self, manager):
        """Set the instance manager for starting/stopping instances."""
        self._executor.set_manager(manager)
        self._batch_tasks: dict[str, asyncio.Task] = {}

    def list_reports(self) -> list[dict[str, Any]]:
        """List all benchmark reports."""
        return self._executor.list_reports()

    def get_report(self, report_id: str) -> BatchBenchmarkReport | None:
        """Get a specific benchmark report."""
        return self._executor.get_report(report_id)

    def delete_report(self, report_id: str) -> bool:
        """Delete a benchmark report."""
        return self._executor.delete_report(report_id)

    def list_batches(self) -> list[dict[str, Any]]:
        """List all batch benchmark runs."""
        batches = []
        for batch_file in self._data_dir.glob("batch_*.json"):
            try:
                with open(batch_file) as f:
                    data = json.load(f)
                batches.append({
                    "batch_id": data.get("batch_id"),
                    "status": data.get("status"),
                    "total_tasks": data.get("total_tasks"),
                    "completed_tasks": data.get("completed_tasks"),
                    "started_at": data.get("started_at"),
                    "completed_at": data.get("completed_at"),
                    "models": data.get("models", []),
                    "backends": data.get("backends", []),
                })
            except Exception:
                continue
        return sorted(batches, key=lambda x: x.get("started_at", ""), reverse=True)

    def get_batch_progress(self, batch_id: str) -> BenchmarkProgress | None:
        """Get progress of a batch benchmark run."""
        # Check active batches first
        if batch_id in self._active_batches:
            return self._active_batches[batch_id]

        # Load from file
        batch_file = self._data_dir / f"batch_{batch_id}.json"
        if batch_file.exists():
            with open(batch_file) as f:
                data = json.load(f)
            return BenchmarkProgress(**data)

        return None

    async def start_batch(self, config: BatchBenchmarkConfig) -> str:
        """Start a batch benchmark run."""
        batch_id = uuid.uuid4().hex[:8]

        # Generate all combinations (Cartesian product)
        tasks = []
        for model in config.models:
            for backend in config.backends:
                tasks.append({"model": model, "backend": backend})

        progress = BenchmarkProgress(
            batch_id=batch_id,
            status="pending",
            total_tasks=len(tasks),
            completed_tasks=0,
            models=config.models,
            backends=config.backends,
            started_at=datetime.now().isoformat(),
        )

        self._active_batches[batch_id] = progress
        self._save_batch(progress)

        # Start background task
        task = asyncio.create_task(
            self._run_batch(batch_id, config, tasks)
        )
        self._batch_tasks[batch_id] = task

        return batch_id

    async def _run_batch(
        self,
        batch_id: str,
        config: BatchBenchmarkConfig,
        tasks: list[dict[str, str]],
    ):
        """Run batch benchmark in background."""
        progress = self._active_batches.get(batch_id)
        if not progress:
            return

        progress.status = "running"

        for i, task_info in enumerate(tasks):
            model = task_info["model"]
            backend = task_info["backend"]

            progress.current_model = model
            progress.current_backend = backend
            progress.current_task = f"{backend}/{model}"
            self._save_batch(progress)

            try:
                # Run benchmark for this model/backend combination
                bench_config = BenchmarkConfig(
                    backend=backend,
                    model=model,
                    host="localhost",
                    port=8080,
                    num_iterations=config.num_iterations,
                    warmup_iterations=config.warmup_iterations,
                    timeout_seconds=config.timeout_seconds,
                )

                if config.scenarios:
                    bench_config.scenarios = config.scenarios

                report = await self._executor.run_benchmark(bench_config)

                progress.results.append({
                    "model": model,
                    "backend": backend,
                    "report_id": report.id,
                    "avg_tokens_per_second": report.avg_tokens_per_second,
                    "avg_time_to_first_token_ms": report.avg_time_to_first_token_ms,
                    "max_gpu_memory_used_mb": report.max_gpu_memory_used_mb,
                })

            except Exception as e:
                progress.errors.append({
                    "model": model,
                    "backend": backend,
                    "error": str(e),
                })

            progress.completed_tasks = i + 1
            self._save_batch(progress)

        progress.status = "completed"
        progress.completed_at = datetime.now().isoformat()
        self._save_batch(progress)

    def _save_batch(self, progress: BenchmarkProgress):
        """Save batch progress to file."""
        batch_file = self._data_dir / f"batch_{progress.batch_id}.json"
        with open(batch_file, "w") as f:
            json.dump(progress.model_dump(), f, indent=2, ensure_ascii=False)

    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch benchmark."""
        if batch_id in self._batch_tasks:
            task = self._batch_tasks[batch_id]
            if not task.done():
                task.cancel()

        progress = self._active_batches.get(batch_id)
        if progress:
            progress.status = "cancelled"
            self._save_batch(progress)
            return True

        return False

    def delete_batch(self, batch_id: str) -> bool:
        """Delete a batch benchmark record."""
        batch_file = self._data_dir / f"batch_{batch_id}.json"
        if batch_file.exists():
            batch_file.unlink()
            self._active_batches.pop(batch_id, None)
            self._batch_tasks.pop(batch_id, None)
            return True
        return False
