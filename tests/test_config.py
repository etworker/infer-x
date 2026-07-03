"""Comprehensive tests for configuration management."""

import pytest
from pathlib import Path
from inferx.config import ConfigManager
from inferx.models import BackendType, DefaultConfig, Preset


class TestConfigManager:
    def test_default_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        assert manager.config.default_backend == BackendType.llamacpp
        assert manager.config.port_range_start == 8080
        assert manager.config.port_range_end == 8180
        assert manager.config.max_instances == 4

    def test_config_file_created(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        ConfigManager(str(config_file))
        assert config_file.exists()

    def test_update_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.update_config(default_backend=BackendType.vllm)
        assert manager.config.default_backend == BackendType.vllm

    def test_update_persists(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.update_config(port_range_start=9000)
        # Reload
        manager2 = ConfigManager(str(config_file))
        assert manager2.config.port_range_start == 9000

    def test_update_multiple_fields(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.update_config(
            default_backend=BackendType.sglang,
            port_range_start=7000,
            port_range_end=7100,
            max_instances=8,
        )
        assert manager.config.default_backend == BackendType.sglang
        assert manager.config.port_range_start == 7000
        assert manager.config.port_range_end == 7100
        assert manager.config.max_instances == 8

    def test_update_none_values_ignored(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        original_backend = manager.config.default_backend
        manager.update_config(default_backend=None)
        assert manager.config.default_backend == original_backend

    def test_update_invalid_field_ignored(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        # Should not raise
        manager.update_config(nonexistent_field="value")

    def test_config_property(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        assert isinstance(manager.config, DefaultConfig)


class TestPresetOperations:
    def test_create_preset(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        preset = Preset(name="test", description="Test", backend=BackendType.llamacpp, ctx_size=4096)
        manager.save_preset(preset)
        assert manager.get_preset("test") is not None

    def test_create_multiple_presets(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.save_preset(Preset(name="fast", backend=BackendType.vllm))
        manager.save_preset(Preset(name="quality", backend=BackendType.llamacpp, ctx_size=16384))
        presets = manager.list_presets()
        assert len(presets) == 2
        assert "fast" in presets
        assert "quality" in presets

    def test_get_preset(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        preset = Preset(name="test", ctx_size=8192, n_gpu_layers="32")
        manager.save_preset(preset)
        retrieved = manager.get_preset("test")
        assert retrieved is not None
        assert retrieved.ctx_size == 8192
        assert retrieved.n_gpu_layers == "32"

    def test_get_nonexistent_preset(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        assert manager.get_preset("nonexistent") is None

    def test_delete_preset(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.save_preset(Preset(name="to-delete"))
        assert manager.delete_preset("to-delete") is True
        assert manager.get_preset("to-delete") is None

    def test_delete_nonexistent_preset(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        assert manager.delete_preset("nonexistent") is False

    def test_update_preset(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.save_preset(Preset(name="test", ctx_size=4096))
        manager.save_preset(Preset(name="test", ctx_size=8192))
        retrieved = manager.get_preset("test")
        assert retrieved.ctx_size == 8192

    def test_list_presets_empty(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        presets = manager.list_presets()
        assert len(presets) == 0

    def test_list_presets_returns_copy(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.save_preset(Preset(name="a"))
        presets = manager.list_presets()
        presets["injected"] = Preset(name="injected")
        assert manager.get_preset("injected") is None


class TestConfigPersistence:
    def test_presets_persist(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.save_preset(Preset(name="persist-test", ctx_size=2048))
        # Reload
        manager2 = ConfigManager(str(config_file))
        retrieved = manager2.get_preset("persist-test")
        assert retrieved is not None
        assert retrieved.ctx_size == 2048

    def test_config_persists_across_reloads(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.update_config(max_instances=16)
        manager2 = ConfigManager(str(config_file))
        assert manager2.config.max_instances == 16

    def test_existing_config_loaded(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        # Create with specific values
        manager1 = ConfigManager(str(config_file))
        manager1.update_config(port_range_start=5000)
        # Load existing
        manager2 = ConfigManager(str(config_file))
        assert manager2.config.port_range_start == 5000

    def test_delete_preset_persists(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        manager.save_preset(Preset(name="temp"))
        manager.delete_preset("temp")
        manager2 = ConfigManager(str(config_file))
        assert manager2.get_preset("temp") is None
