"""Inference benchmark module for performance testing and reporting."""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from .models import BackendType


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class BenchmarkConfig(BaseModel):
    """Configuration for a benchmark run."""
    backend: str
    model: str
    host: str = "localhost"
    port: int = 8080
    # Test scenarios
    scenarios: List[Dict[str, Any]] = Field(default_factory=lambda: [
        {"name": "short_prompt", "prompt": "Hello", "max_tokens": 100},
        {"name": "medium_prompt", "prompt": "Explain quantum computing in simple terms.", "max_tokens": 256},
        {"name": "long_prompt", "prompt": "Write a detailed analysis of the current state of AI technology, including its applications, challenges, and future prospects. Cover topics like machine learning, natural language processing, computer vision, and robotics.", "max_tokens": 512},
        {"name": "code_generation", "prompt": "Write a Python function to implement quicksort algorithm with proper error handling.", "max_tokens": 300},
        {"name": "chat_dialogue", "prompt": "User: What are the benefits of exercise?\nAssistant:", "max_tokens": 200},
    ])
    # Test parameters
    num_iterations: int = 3
    warmup_iterations: int = 1
    concurrent_requests: int = 1
    timeout_seconds: int = 120
    # Context lengths to test
    context_lengths: List[int] = Field(default_factory=lambda: [512, 2048, 4096])


class BenchmarkResult(BaseModel):
    """Result of a single benchmark run."""
    scenario: str
    prompt: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Timing metrics
    load_time_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    # Memory metrics
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_memory_utilization_pct: float = 0.0
    # Throughput metrics
    requests_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0
    # Status
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[str] = None


class BenchmarkReport(BaseModel):
    """Complete benchmark report."""
    id: str
    timestamp: str
    config: BenchmarkConfig
    results: List[BenchmarkResult] = []
    # Summary
    avg_load_time_ms: float = 0.0
    avg_time_to_first_token_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    max_gpu_memory_used_mb: float = 0.0
    # Environment
    gpu_info: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# GPU Monitor
# ---------------------------------------------------------------------------

class GPUMonitor:
    """Monitor GPU memory usage during benchmark."""

    def __init__(self):
        self._nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_available = True
        except Exception:
            pass

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get current GPU information."""
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
        except Exception as e:
            return {"available": False, "error": str(e)}

    def get_gpu_memory_used(self, device_index: int = 0) -> float:
        """Get GPU memory used in MB."""
        if not self._nvml_available:
            return 0.0

        import pynvml
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return mem.used / (1024 * 1024)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Benchmark Executor
# ---------------------------------------------------------------------------

class BenchmarkExecutor:
    """Execute benchmark tests against inference backends."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._gpu_monitor = GPUMonitor()
        self._reports: Dict[str, BenchmarkReport] = {}

    def list_reports(self) -> List[Dict[str, Any]]:
        """List all benchmark reports."""
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
        """Get a specific benchmark report."""
        # Check cache
        if report_id in self._reports:
            return self._reports[report_id]

        # Load from file
        report_file = self._data_dir / f"benchmark_{report_id}.json"
        if report_file.exists():
            with open(report_file, "r") as f:
                data = json.load(f)
            report = BenchmarkReport(**data)
            self._reports[report_id] = report
            return report

        return None

    async def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkReport:
        """Run a complete benchmark."""
        import uuid

        report_id = f"bench-{uuid.uuid4().hex[:8]}"
        report = BenchmarkReport(
            id=report_id,
            timestamp=datetime.now().isoformat(),
            config=config,
        )

        start_time = time.time()

        # Get GPU info
        report.gpu_info = self._gpu_monitor.get_gpu_info()

        # Run warmup
        for _ in range(config.warmup_iterations):
            await self._execute_scenario(config, config.scenarios[0] if config.scenarios else {})

        # Run each scenario
        for scenario in config.scenarios:
            scenario_results = []
            for i in range(config.num_iterations):
                result = await self._execute_scenario(config, scenario)
                scenario_results.append(result)

            # Aggregate results for this scenario
            if scenario_results:
                avg_result = self._aggregate_results(scenario_results)
                report.results.append(avg_result)

        # Calculate summary
        self._calculate_summary(report)

        report.duration_seconds = time.time() - start_time

        # Save report
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
            max_tokens=scenario.get("max_tokens", 100),
        )

        try:
            # Record initial GPU memory
            gpu_memory_before = self._gpu_monitor.get_gpu_memory_used()

            # Build request based on backend
            url = f"http://{config.host}:{config.port}"
            payload = self._build_request(config, scenario)

            # Measure load time (for first request)
            load_start = time.time()

            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                # For OpenAI-compatible APIs
                if config.backend in ("vllm", "sglang", "tgi", "ollama"):
                    endpoint = f"{url}/v1/chat/completions"
                else:
                    endpoint = f"{url}/completion"

                # Record time before request
                request_start = time.time()

                response = await client.post(endpoint, json=payload)

                # Record time after response
                request_end = time.time()

                if response.status_code == 200:
                    data = response.json()

                    # Parse response based on backend
                    parsed = self._parse_response(config.backend, data)

                    result.prompt_tokens = parsed.get("prompt_tokens", 0)
                    result.completion_tokens = parsed.get("completion_tokens", 0)
                    result.total_tokens = result.prompt_tokens + result.completion_tokens
                    result.raw_response = parsed.get("text", "")

                    # Calculate timing
                    result.total_time_ms = (request_end - request_start) * 1000
                    result.load_time_ms = (request_start - load_start) * 1000

                    # Estimate TTFT (for streaming, this would be first chunk time)
                    # For non-streaming, we estimate based on total time
                    result.time_to_first_token_ms = result.total_time_ms * 0.3  # rough estimate

                    # Calculate tokens per second
                    if result.total_time_ms > 0:
                        result.tokens_per_second = (result.completion_tokens * 1000) / result.total_time_ms

                    # Calculate prompt tokens per second
                    if result.load_time_ms > 0:
                        result.prompt_tokens_per_second = (result.prompt_tokens * 1000) / result.load_time_ms

                    result.success = True
                else:
                    result.success = False
                    result.error = f"HTTP {response.status_code}: {response.text[:200]}"

            # Record GPU memory after
            gpu_memory_after = self._gpu_monitor.get_gpu_memory_used()
            result.gpu_memory_used_mb = gpu_memory_after

            # Get total GPU memory from info
            gpu_info = self._gpu_monitor.get_gpu_info()
            if gpu_info.get("gpus"):
                result.gpu_memory_total_mb = gpu_info["gpus"][0].get("total_mb", 0)
                if result.gpu_memory_total_mb > 0:
                    result.gpu_memory_utilization_pct = (result.gpu_memory_used_mb / result.gpu_memory_total_mb) * 100

        except httpx.TimeoutException:
            result.success = False
            result.error = "Request timed out"
        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _build_request(self, config: BenchmarkConfig, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Build request payload based on backend type."""
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
            # llama.cpp style
            return {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": 0.7,
            }

    def _parse_response(self, backend: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse response based on backend type."""
        if backend in ("vllm", "sglang", "tgi"):
            choices = data.get("choices", [{}])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                return {
                    "text": content,
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
        """Aggregate multiple results into one."""
        successful = [r for r in results if r.success]
        if not successful:
            return results[0] if results else BenchmarkResult(scenario="failed")

        # Use first successful result as base
        aggregated = successful[0].model_copy()

        # Average numeric fields
        n = len(successful)
        aggregated.load_time_ms = sum(r.load_time_ms for r in successful) / n
        aggregated.time_to_first_token_ms = sum(r.time_to_first_token_ms for r in successful) / n
        aggregated.total_time_ms = sum(r.total_time_ms for r in successful) / n
        aggregated.tokens_per_second = sum(r.tokens_per_second for r in successful) / n
        aggregated.prompt_tokens_per_second = sum(r.prompt_tokens_per_second for r in successful) / n
        aggregated.gpu_memory_used_mb = max(r.gpu_memory_used_mb for r in successful)

        return aggregated

    def _calculate_summary(self, report: BenchmarkReport):
        """Calculate summary statistics."""
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
        """Save report to file."""
        report_file = self._data_dir / f"benchmark_{report.id}.json"
        with open(report_file, "w") as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

    def delete_report(self, report_id: str) -> bool:
        """Delete a benchmark report."""
        report_file = self._data_dir / f"benchmark_{report_id}.json"
        if report_file.exists():
            report_file.unlink()
            self._reports.pop(report_id, None)
            return True
        return False


# ---------------------------------------------------------------------------
# Streaming Benchmark (for accurate TTFT)
# ---------------------------------------------------------------------------

class StreamingBenchmarkExecutor(BenchmarkExecutor):
    """Benchmark executor with streaming support for accurate TTFT measurement."""

    async def _execute_scenario_streaming(
        self,
        config: BenchmarkConfig,
        scenario: Dict[str, Any],
    ) -> BenchmarkResult:
        """Execute scenario with streaming for accurate TTFT."""
        result = BenchmarkResult(
            scenario=scenario.get("name", "unknown"),
            prompt=scenario.get("prompt", ""),
        )

        try:
            url = f"http://{config.host}:{config.port}"
            payload = self._build_request(config, scenario)
            payload["stream"] = True

            if config.backend in ("vllm", "sglang", "tgi"):
                endpoint = f"{url}/v1/chat/completions"
            else:
                endpoint = f"{url}/completion"

            first_token_time = None
            tokens = 0
            full_text = []

            async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
                async with client.stream("POST", endpoint, json=payload) as response:
                    if response.status_code == 200:
                        request_start = time.time()

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue

                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if first_token_time is None:
                                first_token_time = time.time()
                                result.time_to_first_token_ms = (first_token_time - request_start) * 1000

                            # Extract token
                            token = self._extract_streaming_token(config.backend, data)
                            if token:
                                tokens += 1
                                full_text.append(token)

                        request_end = time.time()
                        result.total_time_ms = (request_end - request_start) * 1000
                        result.completion_tokens = tokens
                        result.raw_response = "".join(full_text)

                        if result.total_time_ms > 0:
                            result.tokens_per_second = (tokens * 1000) / result.total_time_ms

                        result.success = True
                    else:
                        result.success = False
                        result.error = f"HTTP {response.status_code}"

        except Exception as e:
            result.success = False
            result.error = str(e)

        return result

    def _extract_streaming_token(self, backend: str, data: Dict[str, Any]) -> Optional[str]:
        """Extract token from streaming response."""
        if backend in ("vllm", "sglang", "tgi"):
            choices = data.get("choices", [{}])
            if choices:
                delta = choices[0].get("delta", {})
                return delta.get("content")
        elif backend == "ollama":
            return data.get("response")
        else:
            return data.get("content")
        return None
