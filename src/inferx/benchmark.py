"""Inference benchmark module - batch testing with summary reports."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .models import BackendType, InstanceStartRequest, InstanceStatus

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""
    backend: str
    model: str
    host: str = "localhost"
    port: int = 8080
    scenarios: list[dict[str, Any]] = Field(default_factory=lambda: [
        {"name": "short_prompt", "prompt": "Hello", "max_tokens": 50},
        {"name": "medium_prompt", "prompt": "Explain quantum computing.", "max_tokens": 100},
        {"name": "long_prompt", "prompt": "Write a detailed analysis of AI.", "max_tokens": 200},
    ])
    num_iterations: int = 3
    timeout_seconds: int = 120


class BenchmarkResult(BaseModel):
    """Result of a single benchmark run."""
    scenario: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_used_mb: float = 0.0
    success: bool = True
    error: str | None = None


class SingleBenchmarkResult(BaseModel):
    """Result for one model+backend combination."""
    model: str
    backend: str
    results: list[BenchmarkResult] = []
    avg_tokens_per_second: float = 0.0
    avg_ttft_ms: float = 0.0
    max_gpu_memory_mb: float = 0.0
    success: bool = True
    error: str | None = None


class BatchBenchmarkReport(BaseModel):
    """Complete batch benchmark report with all results."""
    id: str
    timestamp: str
    config: dict[str, Any]  # {models, backends, iterations, scenarios}
    results: list[SingleBenchmarkReport] = []
    duration_seconds: float = 0.0
    # Summary
    best_tokens_per_second: dict[str, Any] | None = None
    best_ttft: dict[str, Any] | None = None
    lowest_memory: dict[str, Any] | None = None


class SingleBenchmarkReport(BaseModel):
    """Report for a single model+backend combination."""
    model: str
    backend: str
    scenario_results: list[dict[str, Any]] = []
    avg_tokens_per_second: float = 0.0
    avg_ttft_ms: float = 0.0
    max_gpu_memory_mb: float = 0.0
    total_time_seconds: float = 0.0
    success: bool = True
    error: str | None = None


class BatchBenchmarkConfig(BaseModel):
    """Configuration for batch benchmark testing."""
    models: list[str]
    backends: list[str]
    num_iterations: int = 3
    timeout_seconds: int = 120


class BatchBenchmarkProgress(BaseModel):
    """Progress tracking for batch benchmark."""
    batch_id: str
    status: str  # pending, running, completed, failed
    total_tasks: int = 0
    completed_tasks: int = 0
    current_model: str | None = None
    current_backend: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    report_id: str | None = None


# ---------------------------------------------------------------------------
# Benchmark Manager
# ---------------------------------------------------------------------------

class BenchmarkManager:
    """Manages benchmark execution with batch testing and summary reports."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._manager = None  # InstanceManager
        self._active_batches: dict[str, BatchBenchmarkProgress] = {}
        self._batch_tasks: dict[str, asyncio.Task] = {}

    def set_manager(self, manager):
        self._manager = manager

    # ---- Reports ----

    def list_reports(self) -> list[dict[str, Any]]:
        reports = []
        for f in self._data_dir.glob("report_*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                reports.append({
                    "id": data["id"],
                    "timestamp": data["timestamp"],
                    "models": data.get("config", {}).get("models", []),
                    "backends": data.get("config", {}).get("backends", []),
                    "total_results": len(data.get("results", [])),
                    "duration_seconds": data.get("duration_seconds", 0),
                })
            except Exception:
                continue
        return sorted(reports, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_report(self, report_id: str) -> BatchBenchmarkReport | None:
        f = self._data_dir / f"report_{report_id}.json"
        if f.exists():
            with open(f) as fh:
                data = json.load(fh)
            return BatchBenchmarkReport(**data)
        return None

    def delete_report(self, report_id: str) -> bool:
        f = self._data_dir / f"report_{report_id}.json"
        if f.exists():
            f.unlink()
            return True
        return False

    # ---- Batches ----

    def list_batches(self) -> list[dict[str, Any]]:
        batches = []
        for f in self._data_dir.glob("batch_*.json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                batches.append({
                    "batch_id": data["batch_id"],
                    "status": data["status"],
                    "total_tasks": data.get("total_tasks"),
                    "completed_tasks": data.get("completed_tasks"),
                    "current_model": data.get("current_model"),
                    "current_backend": data.get("current_backend"),
                    "started_at": data.get("started_at"),
                    "completed_at": data.get("completed_at"),
                    "report_id": data.get("report_id"),
                })
            except Exception:
                continue
        # Also include active batches
        for bid, progress in self._active_batches.items():
            if not any(b["batch_id"] == bid for b in batches):
                batches.append(progress.model_dump())
        return sorted(batches, key=lambda x: x.get("started_at", ""), reverse=True)

    def get_batch_progress(self, batch_id: str) -> BatchBenchmarkProgress | None:
        if batch_id in self._active_batches:
            return self._active_batches[batch_id]
        f = self._data_dir / f"batch_{batch_id}.json"
        if f.exists():
            with open(f) as fh:
                data = json.load(fh)
            return BatchBenchmarkProgress(**data)
        return None

    async def start_batch(self, config: BatchBenchmarkConfig) -> str:
        batch_id = uuid.uuid4().hex[:8]

        # Filter models based on backend compatibility
        tasks = []
        for model in config.models:
            for backend in config.backends:
                # Skip .gguf models for non-llamacpp backends
                if backend != "llamacpp" and model.endswith(".gguf"):
                    continue
                # Skip HuggingFace models for llamacpp
                if backend == "llamacpp" and not model.endswith(".gguf"):
                    continue
                tasks.append({"model": model, "backend": backend})

        progress = BatchBenchmarkProgress(
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

        task = asyncio.create_task(self._run_batch(batch_id, config, tasks))
        self._batch_tasks[batch_id] = task
        return batch_id

    async def _run_batch(self, batch_id: str, config: BatchBenchmarkConfig, tasks: list[dict]):
        progress = self._active_batches.get(batch_id)
        if not progress:
            return

        progress.status = "running"
        results = []
        start_time = time.time()

        for i, task_info in enumerate(tasks):
            model = task_info["model"]
            backend = task_info["backend"]

            progress.current_model = model
            progress.current_backend = backend
            self._save_batch(progress)

            try:
                result = await self._run_single(model, backend, config)
                results.append(result)
            except Exception as e:
                results.append(SingleBenchmarkReport(
                    model=model, backend=backend, success=False, error=str(e)
                ))

            progress.completed_tasks = i + 1
            self._save_batch(progress)

        # Generate summary report
        report_id = f"report-{uuid.uuid4().hex[:8]}"
        report = self._generate_report(report_id, config, results, time.time() - start_time)
        self._save_report(report)

        progress.status = "completed"
        progress.completed_at = datetime.now().isoformat()
        progress.report_id = report_id
        self._save_batch(progress)

    async def _run_single(self, model: str, backend: str, config: BatchBenchmarkConfig) -> SingleBenchmarkReport:
        """Run benchmark for a single model+backend."""
        report = SingleBenchmarkReport(model=model, backend=backend)

        instance_id = None
        try:
            # Start instance via manager
            if self._manager:
                inst = await self._manager.start_instance(
                    InstanceStartRequest(
                        model=model,
                        backend=BackendType(backend),
                    )
                )
                instance_id = inst.id

                # Wait for ready
                for _ in range(120):
                    await asyncio.sleep(1)
                    inst = self._manager.get_instance(instance_id)
                    if inst and inst.status == InstanceStatus.running:
                        break
                    if inst and inst.status == InstanceStatus.error:
                        raise RuntimeError("Instance failed to start")

            # Run scenarios
            scenarios = [
                {"name": "short", "prompt": "Hello", "max_tokens": 50},
                {"name": "medium", "prompt": "Explain AI.", "max_tokens": 100},
                {"name": "long", "prompt": "Write a detailed analysis.", "max_tokens": 200},
            ]

            port = 8080
            if self._manager:
                inst = self._manager.get_instance(instance_id)
                if inst:
                    port = inst.port

            for scenario in scenarios:
                for _ in range(config.num_iterations):
                    result = await self._run_scenario(backend, model, port, scenario, config.timeout_seconds)
                    report.scenario_results.append(result.model_dump())

            # Calculate averages
            successful = [r for r in report.scenario_results if r.get("success")]
            if successful:
                report.avg_tokens_per_second = sum(r["tokens_per_second"] for r in successful) / len(successful)
                report.avg_ttft_ms = sum(r["time_to_first_token_ms"] for r in successful) / len(successful)
                report.max_gpu_memory_mb = max(r["gpu_memory_used_mb"] for r in successful)
                report.total_time_seconds = sum(r["total_time_ms"] for r in successful) / 1000

            report.success = len(successful) > 0
            if not report.success and report.scenario_results:
                report.error = report.scenario_results[0].get("error", "All scenarios failed")

        except Exception as e:
            report.success = False
            report.error = str(e)

        finally:
            if self._manager and instance_id:
                try:
                    await self._manager.stop_instance(instance_id)
                except Exception:
                    pass

        return report

    async def _run_scenario(self, backend: str, model: str, port: int, scenario: dict, timeout: int = 120) -> BenchmarkResult:
        result = BenchmarkResult(scenario=scenario["name"])
        try:
            url = f"http://localhost:{port}"
            payload = self._build_request(backend, model, scenario)
            # Add stream=true for TTFT measurement on supported backends
            supports_stream = backend in ("vllm", "sglang", "tgi")
            if supports_stream:
                payload["stream"] = True

            async with httpx.AsyncClient(timeout=timeout) as client:
                if backend in ("vllm", "sglang", "tgi"):
                    endpoint = f"{url}/v1/chat/completions"
                elif backend == "ollama":
                    endpoint = f"{url}/api/generate"
                else:
                    endpoint = f"{url}/completion"

                start = time.time()
                first_token_time = None

                if supports_stream:
                    # Streaming request to measure TTFT
                    async with client.stream("POST", endpoint, json=payload) as resp:
                        if resp.status_code == 200:
                            async for line in resp.aiter_lines():
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    first_token_time = time.time()
                                    break
                            # Read remaining to get final stats
                            result.total_time_ms = (time.time() - start) * 1000
                            if first_token_time:
                                result.time_to_first_token_ms = (first_token_time - start) * 1000
                            # Parse from last chunks for token counts
                            result.success = True
                        else:
                            result.success = False
                            result.error = f"HTTP {resp.status_code}"
                else:
                    # Non-streaming request
                    response = await client.post(endpoint, json=payload)
                    end = time.time()
                    if response.status_code == 200:
                        data = response.json()
                        parsed = self._parse_response(backend, data)
                        result.prompt_tokens = parsed.get("prompt_tokens", 0)
                        result.completion_tokens = parsed.get("completion_tokens", 0)
                        result.total_time_ms = (end - start) * 1000
                        if result.total_time_ms > 0 and result.completion_tokens > 0:
                            result.tokens_per_second = (result.completion_tokens * 1000) / result.total_time_ms
                        result.success = True
                    else:
                        result.success = False
                        result.error = f"HTTP {response.status_code}"

            # Get GPU memory usage via ResourceMonitor
            from .monitor import ResourceMonitor
            gpus = ResourceMonitor().get_gpus()
            if gpus:
                result.gpu_memory_used_mb = float(gpus[0].used_memory_mb)

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _build_request(self, backend: str, model: str, scenario: dict) -> dict:
        prompt = scenario.get("prompt", "")
        max_tokens = scenario.get("max_tokens", 100)
        if backend in ("vllm", "sglang", "tgi"):
            return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        elif backend == "ollama":
            return {"model": model, "prompt": prompt, "options": {"num_predict": max_tokens}}
        else:
            return {"prompt": prompt, "n_predict": max_tokens}

    def _parse_response(self, backend: str, data: dict) -> dict:
        if backend in ("vllm", "sglang", "tgi"):
            choices = data.get("choices", [{}])
            if choices:
                return {
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                }
        elif backend == "ollama":
            return {"prompt_tokens": data.get("prompt_eval_count", 0), "completion_tokens": data.get("eval_count", 0)}
        else:
            return {"prompt_tokens": data.get("tokens_evaluated", 0), "completion_tokens": data.get("tokens_predicted", 0)}
        return {"prompt_tokens": 0, "completion_tokens": 0}

    def _generate_report(self, report_id: str, config: BatchBenchmarkConfig, results: list[SingleBenchmarkReport], duration: float) -> BatchBenchmarkReport:
        successful = [r for r in results if r.success and r.avg_tokens_per_second > 0]

        # Find best performers
        best_tps = None
        best_ttft = None
        lowest_mem = None

        if successful:
            best_tps_item = max(successful, key=lambda x: x.avg_tokens_per_second)
            best_tps = {"model": best_tps_item.model, "backend": best_tps_item.backend, "value": best_tps_item.avg_tokens_per_second}

            best_ttft_item = min(successful, key=lambda x: x.avg_ttft_ms)
            best_ttft = {"model": best_ttft_item.model, "backend": best_ttft_item.backend, "value": best_ttft_item.avg_ttft_ms}

            lowest_mem_item = min(successful, key=lambda x: x.max_gpu_memory_mb)
            lowest_mem = {"model": lowest_mem_item.model, "backend": lowest_mem_item.backend, "value": lowest_mem_item.max_gpu_memory_mb}

        return BatchBenchmarkReport(
            id=report_id,
            timestamp=datetime.now().isoformat(),
            config={"models": config.models, "backends": config.backends, "iterations": config.num_iterations},
            results=results,
            duration_seconds=duration,
            best_tokens_per_second=best_tps,
            best_ttft=best_ttft,
            lowest_memory=lowest_mem,
        )

    def _save_batch(self, progress: BatchBenchmarkProgress):
        f = self._data_dir / f"batch_{progress.batch_id}.json"
        with open(f, "w") as fh:
            json.dump(progress.model_dump(), fh, indent=2, ensure_ascii=False)

    def _save_report(self, report: BatchBenchmarkReport):
        f = self._data_dir / f"report_{report.id}.json"
        with open(f, "w") as fh:
            json.dump(report.model_dump(), fh, indent=2, ensure_ascii=False)

    def cancel_batch(self, batch_id: str) -> bool:
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
        f = self._data_dir / f"batch_{batch_id}.json"
        if f.exists():
            f.unlink()
            self._active_batches.pop(batch_id, None)
            self._batch_tasks.pop(batch_id, None)
            return True
        return False
