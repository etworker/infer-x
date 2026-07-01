"""Tests for benchmark module."""

import pytest
from pathlib import Path
from inferx.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkExecutor,
    GPUMonitor,
)


class TestBenchmarkConfig:
    def test_default_config(self):
        config = BenchmarkConfig(backend="vllm", model="test-model")
        assert config.backend == "vllm"
        assert config.model == "test-model"
        assert len(config.scenarios) > 0
        assert config.num_iterations == 3

    def test_custom_scenarios(self):
        scenarios = [
            {"name": "test", "prompt": "Hello", "max_tokens": 50}
        ]
        config = BenchmarkConfig(
            backend="llamacpp",
            model="model.gguf",
            scenarios=scenarios,
        )
        assert len(config.scenarios) == 1
        assert config.scenarios[0]["name"] == "test"


class TestBenchmarkResult:
    def test_result_creation(self):
        result = BenchmarkResult(
            scenario="test",
            prompt="Hello",
            prompt_tokens=10,
            completion_tokens=50,
            total_time_ms=1000,
            tokens_per_second=50,
        )
        assert result.success is True
        assert result.tokens_per_second == 50

    def test_failed_result(self):
        result = BenchmarkResult(
            scenario="test",
            prompt="Hello",
            success=False,
            error="Connection refused",
        )
        assert result.success is False
        assert result.error == "Connection refused"


class TestBenchmarkReport:
    def test_report_creation(self):
        report = BenchmarkReport(
            id="bench-123",
            timestamp="2026-07-01T12:00:00",
            config=BenchmarkConfig(backend="vllm", model="test"),
        )
        assert report.id == "bench-123"
        assert len(report.results) == 0


class TestBenchmarkExecutor:
    def test_executor_init(self, tmp_path):
        executor = BenchmarkExecutor(tmp_path)
        assert executor._data_dir == tmp_path

    def test_list_reports_empty(self, tmp_path):
        executor = BenchmarkExecutor(tmp_path)
        reports = executor.list_reports()
        assert len(reports) == 0

    def test_delete_nonexistent_report(self, tmp_path):
        executor = BenchmarkExecutor(tmp_path)
        result = executor.delete_report("nonexistent")
        assert result is False


class TestGPUMonitor:
    def test_get_gpu_info(self):
        monitor = GPUMonitor()
        info = monitor.get_gpu_info()
        assert "available" in info

    def test_get_gpu_memory(self):
        monitor = GPUMonitor()
        memory = monitor.get_gpu_memory_used()
        assert isinstance(memory, float)
