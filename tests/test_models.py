"""Comprehensive tests for data models."""

import pytest
from inferx.models import (
    BackendType,
    DefaultConfig,
    DownloadProgress,
    DownloadRequest,
    DownloadSource,
    DownloadStatus,
    GPUInfo,
    HealthResponse,
    InstanceInfo,
    InstanceList,
    InstanceLogs,
    InstanceStartRequest,
    InstanceStatus,
    LogEntry,
    ModelFileInfo,
    Preset,
    SystemInfo,
    ConfigUpdate,
)


class TestBackendType:
    def test_all_values(self):
        expected = ["llamacpp", "vllm", "sglang", "tgi", "ollama", "tensorrt_llm", "lmdeploy", "openvino"]
        for val in expected:
            assert BackendType(val).value == val

    def test_count(self):
        assert len(BackendType) == 8

    def test_string_behavior(self):
        assert BackendType.llamacpp == "llamacpp"
        assert BackendType.vllm != "llamacpp"

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            BackendType("nonexistent")


class TestInstanceStatus:
    def test_all_values(self):
        expected = ["starting", "running", "stopping", "stopped", "error"]
        for val in expected:
            assert InstanceStatus(val).value == val

    def test_count(self):
        assert len(InstanceStatus) == 5


class TestDownloadSource:
    def test_all_values(self):
        assert DownloadSource.huggingface.value == "hf"
        assert DownloadSource.modelscope.value == "ms"
        assert DownloadSource.hf_mirror.value == "hf_mirror"
        assert DownloadSource.url.value == "url"


class TestDownloadStatus:
    def test_all_values(self):
        expected = ["pending", "downloading", "completed", "failed"]
        for val in expected:
            assert DownloadStatus(val).value == val


class TestGPUInfo:
    def test_creation(self):
        gpu = GPUInfo(
            index=0, name="NVIDIA L4",
            total_memory_mb=23034, used_memory_mb=468,
            free_memory_mb=22566, utilization_pct=0.0,
        )
        assert gpu.index == 0
        assert gpu.name == "NVIDIA L4"
        assert gpu.total_memory_mb == 23034

    def test_optional_utilization(self):
        gpu = GPUInfo(
            index=0, name="Test",
            total_memory_mb=1000, used_memory_mb=0,
            free_memory_mb=1000,
        )
        assert gpu.utilization_pct is None


class TestSystemInfo:
    def test_creation(self):
        info = SystemInfo(
            gpus=[],
            total_ram_mb=16000,
            used_ram_mb=4000,
            available_ram_mb=12000,
            cpu_count=4,
            cpu_percent=25.0,
            server_paths={"llamacpp": "/usr/bin/llama-server"},
        )
        assert info.total_ram_mb == 16000
        assert len(info.gpus) == 0
        assert info.server_paths["llamacpp"] == "/usr/bin/llama-server"


class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse()
        assert h.status == "ok"
        assert h.instances_running == 0
        assert h.instances_total == 0

    def test_custom(self):
        h = HealthResponse(status="ok", instances_running=3, instances_total=5)
        assert h.instances_running == 3


class TestDefaultConfig:
    def test_default_values(self):
        config = DefaultConfig()
        assert config.default_backend == BackendType.llamacpp
        assert config.port_range_start == 8080
        assert config.port_range_end == 8180
        assert config.max_instances == 4
        assert config.default_host == "0.0.0.0"
        assert config.health_check_interval == 10
        assert config.auto_restart is True

    def test_custom_values(self):
        config = DefaultConfig(
            model_dir="/custom/path",
            default_backend=BackendType.vllm,
            port_range_start=9000,
            port_range_end=9100,
            max_instances=8,
        )
        assert config.model_dir == "/custom/path"
        assert config.default_backend == BackendType.vllm
        assert config.port_range_start == 9000
        assert config.port_range_end == 9100
        assert config.max_instances == 8

    def test_all_backend_binaries(self):
        config = DefaultConfig()
        assert config.llama_server_bin
        assert config.vllm_server_bin
        assert config.sglang_server_bin
        assert config.tgi_bin
        assert config.ollama_bin
        assert config.tensorrt_llm_bin
        assert config.lmdeploy_bin
        assert config.openvino_bin

    def test_serialization_roundtrip(self):
        config = DefaultConfig(default_backend=BackendType.sglang, port_range_start=7000)
        data = config.model_dump()
        restored = DefaultConfig(**data)
        assert restored.default_backend == BackendType.sglang
        assert restored.port_range_start == 7000


class TestConfigUpdate:
    def test_partial_update(self):
        update = ConfigUpdate(default_backend=BackendType.vllm)
        assert update.default_backend == BackendType.vllm
        assert update.model_dir is None
        assert update.port_range_start is None

    def test_all_none_by_default(self):
        update = ConfigUpdate()
        for field in update.model_fields:
            assert getattr(update, field) is None


class TestInstanceStartRequest:
    def test_minimal(self):
        req = InstanceStartRequest(model="test-model")
        assert req.model == "test-model"
        assert req.backend is None
        assert req.port is None
        assert req.host is None
        assert req.extra_args == []

    def test_with_backend(self):
        req = InstanceStartRequest(model="m", backend=BackendType.vllm, port=8080)
        assert req.backend == BackendType.vllm
        assert req.port == 8080

    def test_llama_params(self):
        req = InstanceStartRequest(
            model="m", ctx_size=8192, n_gpu_layers="32",
            threads=8, batch_size=4096, n_parallel=2,
        )
        assert req.ctx_size == 8192
        assert req.n_gpu_layers == "32"
        assert req.n_parallel == 2

    def test_vllm_params(self):
        req = InstanceStartRequest(
            model="m", backend=BackendType.vllm,
            tensor_parallel_size=2, max_model_len=4096,
            gpu_memory_utilization=0.8,
        )
        assert req.tensor_parallel_size == 2
        assert req.gpu_memory_utilization == 0.8

    def test_sglang_params(self):
        req = InstanceStartRequest(
            model="m", backend=BackendType.sglang,
            tp=2, mem_fraction_static=0.7,
        )
        assert req.tp == 2
        assert req.mem_fraction_static == 0.7


class TestInstanceInfo:
    def test_creation(self):
        info = InstanceInfo(
            id="inst-1", model="test.gguf",
            status=InstanceStatus.running, port=8080, host="0.0.0.0",
        )
        assert info.id == "inst-1"
        assert info.status == InstanceStatus.running
        assert info.backend == BackendType.llamacpp
        assert info.ctx_size == 4096
        assert info.restart_count == 0

    def test_optional_fields(self):
        info = InstanceInfo(
            id="x", model="m",
            status=InstanceStatus.running, port=8080, host="0.0.0.0",
            pid=12345, gpu_memory_mb=8000, ram_usage_mb=2000,
        )
        assert info.pid == 12345
        assert info.gpu_memory_mb == 8000


class TestInstanceList:
    def test_empty(self):
        lst = InstanceList(instances=[], total=0)
        assert len(lst.instances) == 0
        assert lst.total == 0


class TestPreset:
    def test_creation(self):
        p = Preset(name="fast", backend=BackendType.vllm, ctx_size=2048)
        assert p.name == "fast"
        assert p.ctx_size == 2048
        assert p.extra_args == []

    def test_serialization(self):
        p = Preset(name="test", backend=BackendType.llamacpp)
        data = p.model_dump()
        assert data["name"] == "test"
        restored = Preset(**data)
        assert restored.name == "test"


class TestModelFileInfo:
    def test_creation(self):
        m = ModelFileInfo(name="model.gguf", path="/tmp/m.gguf", size_mb=4000)
        assert m.name == "model.gguf"
        assert m.quantization is None


class TestLogEntry:
    def test_creation(self):
        entry = LogEntry(timestamp="2026-01-01T00:00:00", level="INFO", message="started")
        assert entry.level == "INFO"


class TestInstanceLogs:
    def test_creation(self):
        logs = InstanceLogs(instance_id="i1", logs=[], total_lines=0)
        assert logs.instance_id == "i1"


class TestDownloadRequest:
    def test_hf_request(self):
        req = DownloadRequest(source=DownloadSource.huggingface, repo="user/repo")
        assert req.repo == "user/repo"

    def test_url_request(self):
        req = DownloadRequest(source=DownloadSource.url, url="https://example.com/model.bin")
        assert req.url == "https://example.com/model.bin"


class TestDownloadProgress:
    def test_creation(self):
        dp = DownloadProgress(
            task_id="t1", source="hf", status=DownloadStatus.pending,
        )
        assert dp.progress_pct == 0.0
        assert dp.downloaded_bytes == 0
