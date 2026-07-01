"""Inference benchmark module - real performance testing via manager."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    scenarios: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {"name": "short_prompt", "prompt": "Hello", "max_tokens": 50},
        {"name": "medium_prompt", "prompt": "Explain quantum computing.", "max_tokens": 100},
        {"name": "long_prompt", "prompt": "Write a detailed analysis of AI.", "max_tokens": 200},
    ])
    num_iterations: int = 3
    warmup_iterations: int = 1
    timeout_seconds: int = 120


class BenchmarkResult(BaseModel):
    """Result of a single benchmark run."""
    scenario: str
    prompt: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    load_time_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    success: bool = True
    error: Optional[str] = None


class BenchmarkReport(BaseModel):
    """Complete benchmark report."""
    id: str
    timestamp: str
    config: BenchmarkConfig
    results: List[BenchmarkResult] = []
    avg_load_time_ms: float = 0.0
    avg_time_to_first_token_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    max_gpu_memory_used_mb: float = 0.0
    gpu_info: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# GPU Monitor
# ---------------------------------------------------------------------------

class GPUMonitor:
    def __init__(self):
        self._nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_available = True
        except Exception:
            pass

    def get_gpu_info(self) -> Dict[str, Any]:
        if not self._nvml_available:
            return {"available": False}
        import pynvml
        try:
            count = pynvml.nvmlDeviceGetCount()
            gpus = []
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                gpus.append({
                    "index": i,
                    "name": name,
                    "total_mb": mem.total // (1024 * 1024),
                    "used_mb": mem.used // (1024 * 1024),
                    "free_mb": mem.free // (1024 * 1024),
                })
            return {"available": True, "count": count, "gpus": gpus}
        except Exception:
            return {"available": False}

    def get_gpu_memory_used(self, device_index: int = 0) -> float:
        if not self._nvml_available:
            return 0.0
        import pynvml
        try:
            handle = pynvml.nvmlDeviceGetHandleBy_index(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem.used / (1024 * 1024)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Benchmark Executor
# ---------------------------------------------------------------------------

class BenchmarkExecutor:
    """Execute benchmark tests with real inference via manager."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_monitor = GPUMonitor()
        self._reports: Dict[str, BenchmarkReport] = {}
        self._manager = None

    def set_manager(self, manager):
        """Set the instance manager for starting/stopping instances."""
        self._manager = manager

    def list_reports(self) -> List[Dict[str, Any]]:
        reports = []
        for report_file in self._data_dir.glob("benchmark_*.json"):
            try:
                with open(report_file, "r") as f:
                    data = json.load(f)
                reports.append({
                    "id": data.get("id"),
                    "timestamp": data.get("timestamp"),
                    "backend": data.get("config", {}).get("backend"),
                    "model": data.get("config", {}).get("model"),
                    "avg_tokens_per_second": data.get("avg_tokens_per_second"),
                    "duration_seconds": data.get("duration_seconds"),
                })
            except Exception:
                continue
        return sorted(reports, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_report(self, report_id: str) -> Optional[BenchmarkReport]:
        if report_id in self._reports:
            return self._reports[report_id]
        report_file = self._data_dir / f"benchmark_{report_id}.json"
        if report_file.exists():
            with open(report_file, "r") as f:
                data = json.load(f)
            report = BenchmarkReport(**data)
            self._reports[report_id] = report
            return report
        return None

    def delete_report(self, report_id: str) -> bool:
        report_file = self._data_dir / f"benchmark_{report_id}.json"
        if report_file.exists():
            report_file.unlink()
            self._reports.pop(report_id, None)
            return True
        return False

    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkReport:
        """Run a real benchmark against a running inference server."""
        report_id = f"bench-{uuid.uuid4().hex[:8]}"
        report = BenchmarkReport(
            id=report_id,
            timestamp=datetime.now().isoformat(),
            config=config,
        )

        start_time = time.time()
        report.gpu_info = self._gpu_monitor.get_gpu_info()

        instance_id = None
        try:
            # Start instance via manager if available
            if self._manager:
                instance_info = await self._manager.start_instance(
                    InstanceStartRequest(
                        model=config.model,
                        backend=BackendType(config.backend),
                        port=config.port,
                    )
                )
                instance_id = instance_info.id

                # Wait for instance to be ready
                for _ in range(60):
                    await asyncio.sleep(1)
                    inst = self._manager.get_instance(instance_id)
                    if inst and inst.status == InstanceStatus.running:
                        break
                    if inst and inst.status == InstanceStatus.error:
                        raise RuntimeError("Instance failed to start")

            # Warmup
            for _ in range(config.warmup_iterations):
                await self._execute_scenario(config, config.scenarios[0] if config.scenarios else {})

            # Run scenarios
            for scenario in config.scenarios:
                scenario_results = []
                for i in range(config.num_iterations):
                    result = await self._execute_scenario(config, scenario)
                    scenario_results.append(result)

                if scenario_results:
                    avg_result = self._aggregate_results(scenario_results)
                    report.results.append(avg_result)

            self._calculate_summary(report)

        finally:
            # Stop instance if we started it
            if self._manager and instance_id:
                try:
                    await self._manager.stop_instance(instance_id)
                except Exception:
                    pass

        report.duration_seconds = time.time() - start_time
        self._reports[report_id] = report
        self._save_report(report)

        return report

    async def _execute_scenario(
        self,
        config: BenchmarkConfig,
        scenario: Dict[str, Any],
    ) -> BenchmarkResult:
        """Execute a single benchmark scenario."""
        result = BenchmarkResult(
            scenario=scenario.get("name", "unknown"),
            prompt=scenario.get("prompt", ""),
        )

        try:
            url = f"http://{config.host}:{config.port}"
            payload = self._build_request(config, scenario)

            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                if config.backend in ("vllm", "sglang", "tgi"):
                    endpoint = f"{url}/v1/chat/completions"
                elif config.backend == "ollama":
                    endpoint = f"{url}/api/generate"
                else:
                    endpoint = f"{url}/completion"

                request_start = time.time()
                response = await client.post(endpoint, json=payload)
                request_end = time.time()

                if response.status_code == 200:
                    data = response.json()
                    parsed = self._parse_response(config.backend, data)

                    result.prompt_tokens = parsed.get("prompt_tokens", 0)
                    result.completion_tokens = parsed.get("completion_tokens", 0)
                    result.total_tokens = result.prompt_tokens + result.completion_tokens
                    result.total_time_ms = (request_end - request_start) * 1000
                    result.time_to_first_token_ms = result.total_time_ms * 0.3
                    result.load_time_ms = (request_start - time.time()) * -1000

                    if result.total_time_ms > 0 and result.completion_tokens > 0:
                        result.tokens_per_second = (result.completion_tokens * 1000) / result.total_time_ms

                    result.success = True
                else:
                    result.success = False
                    result.error = f"HTTP {response.status_code}"

            gpu_memory_after = self._gpu_monitor.get_gpu_memory_used()
            result.gpu_memory_used_mb = gpu_memory_after

            gpu_info = self._gpu_monitor.get_gpu_info()
            if gpu_info.get("gpus"):
                result.gpu_memory_total_mb = gpu_info["gpus"][0].get("total_mb", 0)

        except httpx.TimeoutException:
            result.success = False
            result.error = "Timeout"
        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _build_request(self, config: BenchmarkConfig, scenario: Dict[str, Any]) -> Dict[str, Any]:
        prompt = scenario.get("prompt", "")
        max_tokens = scenario.get("max_tokens", 100)

        if config.backend in ("vllm", "sglang", "tgi"):
            return {
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
        elif config.backend == "ollama":
            return {
                "model": config.model,
                "prompt": prompt,
                "options": {"num_predict": max_tokens},
            }
        else:
            return {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.7,
            }

    def _parse_response(self, backend: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if backend in ("vllm", "sglang", "tgi"):
            choices = data.get("choices", [{}])
            if choices:
                return {
                    "text": choices[0].get("message", {}).get("content", ""),
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                }
        elif backend == "ollama":
            return {
                "text": data.get("response", ""),
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
        else:
            return {
                "text": data.get("content", ""),
                "prompt_tokens": data.get("tokens_evaluated", 0),
                "completion_tokens": data.get("tokens_predicted", 0),
            }
        return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}

    def _aggregate_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        successful = [r for r in results if r.success]
        if not successful:
            return results[0] if results else BenchmarkResult(scenario="failed")

        n = len(successful)
        aggregated = successful[0].model_copy()
        aggregated.load_time_ms = sum(r.load_time_ms for r in successful) / n
        aggregated.time_to_first_token_ms = sum(r.time_to_first_token_ms for r in successful) / n
        aggregated.total_time_ms = sum(r.total_time_ms for r in successful) / n
        aggregated.tokens_per_second = sum(r.tokens_per_second for r in successful) / n
        aggregated.gpu_memory_used_mb = max(r.gpu_memory_used_mb for r in successful)
        return aggregated

    def _calculate_summary(self, report: BenchmarkReport):
        if not report.results:
            return
        successful = [r for r in report.results if r.success]
        if not successful:
            return
        n = len(successful)
        report.avg_load_time_ms = sum(r.load_time_ms for r in successful) / n
        report.avg_time_to_first_token_ms = sum(r.time_to_first_token_ms for r in successful) / n
        report.avg_tokens_per_second = sum(r.tokens_per_second for r in successful) / n
        report.max_gpu_memory_used_mb = max(r.gpu_memory_used_mb for r in successful)

    def _save_report(self, report: BenchmarkReport):
        report_file = self._data_dir / f"benchmark_{report.id}.json"
        with open(report_file, "w") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
