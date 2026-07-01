"""Tests for data models."""

import pytest
from inferx.models import (
    BackendType,
    DefaultConfig,
    InstanceStartRequest,
    InstanceStatus,
    Preset,
)


class TestBackendType:
    def test_all_values(self):
        assert BackendType.llamacpp.value == "llamacpp"
        assert BackendType.vllm.value == "vllm"
        assert BackendType.sglang.value == "sglang"
        assert BackendType.tgi.value == "tgi"
        assert BackendType.ollama.value == "ollama"
        assert BackendType.tensorrt_llm.value == "tensorrt_llm"
        assert BackendType.lmdeploy.value == "lmdeploy"
        assert BackendType.openvino.value == "openvino"


class TestDefaultConfig:
    def test_default_values(self):
        config = DefaultConfig()
        assert config.default_backend == BackendType.llamacpp
        assert config.port_range_start == 8080
        assert config.port_range_end == 8180
        assert config.max_instances == 4
        assert config.default_host == "0.0.0.0"

    def test_custom_values(self):
        config = DefaultConfig(
            model_dir="/custom/path",
            default_backend=BackendType.vllm,
            port_range_start=9000,
        )
        assert config.model_dir == "/custom/path"
        assert config.default_backend == BackendType.vllm
        assert config.port_range_start == 9000


class TestInstanceStartRequest:
    def test_minimal_request(self):
        req = InstanceStartRequest(model="test-model")
        assert req.model == "test-model"
        assert req.backend is None
        assert req.port is None

    def test_with_backend(self):
        req = InstanceStartRequest(model="test-model", backend=BackendType.vllm)
        assert req.backend == BackendType.vllm


class TestPreset:
    def test_preset_creation(self):
        preset = Preset(
            name="test-preset",
            description="Test preset",
            backend=BackendType.llamacpp,
            ctx_size=8192,
        )
        assert preset.name == "test-preset"
        assert preset.ctx_size == 8192
