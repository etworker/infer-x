"""Comprehensive tests for benchmark module."""

import pytest
from pathlib import Path
from inferx.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    BatchBenchmarkReport,
    BenchmarkManager,
    GPUMonitor,
    SingleBenchmarkResult,
)


class TestBenchmarkConfig:
    def test_default_config(self):
        config = BenchmarkConfig(backend="vllm", model="test-model")
        assert config.backend == "vllm"
        assert config.model == "test-model"
        assert len(config.scenarios) == 3
        assert config.num_iterations == 3
        assert config.timeout_seconds == 120

    def test_custom_scenarios(self):
        scenarios = [{"name": "test", "prompt": "Hello", "max_tokens": 50}]
        config = BenchmarkConfig(backend="llamacpp", model="m.gguf", scenarios=scenarios)
        assert len(config.scenarios) == 1
        assert config.scenarios[0]["name"] == "test"

    def test_custom_iterations(self):
        config = BenchmarkConfig(backend="vllm", model="m", num_iterations=10)
        assert config.num_iterations == 10

    def test_custom_timeout(self):
        config = BenchmarkConfig(backend="vllm", model="m", timeout_seconds=300)
        assert config.timeout_seconds == 300


class TestBenchmarkResult:
    def test_successful_result(self):
        result = BenchmarkResult(
            scenario="test", prompt_tokens=10, completion_tokens=50,
            total_time_ms=1000, tokens_per_second=50,
        )
        assert result.success is True
        assert result.tokens_per_second == 50
        assert result.time_to_first_token_ms == 0.0

    def test_failed_result(self):
        result = BenchmarkResult(
            scenario="test", success=False, error="Connection refused",
        )
        assert result.success is False
        assert result.error == "Connection refused"

    def test_result_with_ttft(self):
        result = BenchmarkResult(
            scenario="test", time_to_first_token_ms=150.5, tokens_per_second=30,
        )
        assert result.time_to_first_token_ms == 150.5

    def test_result_with_gpu_memory(self):
        result = BenchmarkResult(
            scenario="test", gpu_memory_used_mb=8000.0,
        )
        assert result.gpu_memory_used_mb == 8000.0


class TestSingleBenchmarkResult:
    def test_creation(self):
        result = SingleBenchmarkResult(model="m", backend="vllm")
        assert result.model == "m"
        assert result.results == []
        assert result.avg_tokens_per_second == 0.0
        assert result.success is True

    def test_with_results(self):
        result = SingleBenchmarkResult(
            model="m", backend="vllm",
            avg_tokens_per_second=45.0, avg_ttft_ms=200.0,
            max_gpu_memory_mb=12000.0,
        )
        assert result.avg_tokens_per_second == 45.0
        assert result.max_gpu_memory_mb == 12000.0


class TestBatchBenchmarkReport:
    def test_creation(self):
        report = BatchBenchmarkReport(
            id="bench-123", timestamp="2026-07-01T12:00:00",
            config={"models": ["test"], "backends": ["vllm"], "iterations": 3},
        )
        assert report.id == "bench-123"
        assert len(report.results) == 0
        assert report.duration_seconds == 0.0

    def test_with_summary(self):
        report = BatchBenchmarkReport(
            id="b1", timestamp="2026-01-01T00:00:00", config={},
            best_tokens_per_second={"value": 50.0, "backend": "vllm", "model": "m"},
        )
        assert report.best_tokens_per_second["value"] == 50.0


class TestBenchmarkManager:
    def test_init(self, tmp_path):
        manager = BenchmarkManager(tmp_path)
        assert manager._data_dir == tmp_path

    def test_list_reports_empty(self, tmp_path):
        manager = BenchmarkManager(tmp_path)
        reports = manager.list_reports()
        assert len(reports) == 0

    def test_list_batches_empty(self, tmp_path):
        manager = BenchmarkManager(tmp_path)
        batches = manager.list_batches()
        assert len(batches) == 0

    def test_delete_nonexistent_report(self, tmp_path):
        manager = BenchmarkManager(tmp_path)
        assert manager.delete_report("nonexistent") is False

    def test_delete_nonexistent_batch(self, tmp_path):
        manager = BenchmarkManager(tmp_path)
        assert manager.delete_batch("nonexistent") is False

    def test_get_nonexistent_report(self, tmp_path):
        manager = BenchmarkManager(tmp_path)
        assert manager.get_report("nonexistent") is None

    def test_get_nonexistent_batch(self, tmp_path):
        manager = BenchmarkManager(tmp_path)
        assert manager.get_batch_progress("nonexistent") is None


class TestGPUMonitor:
    def test_get_gpu_info(self):
        monitor = GPUMonitor()
        info = monitor.get_gpu_info()
        assert "available" in info

    def test_get_gpu_memory(self):
        monitor = GPUMonitor()
        memory = monitor.get_gpu_memory_used()
        assert isinstance(memory, float)
        assert memory >= 0
