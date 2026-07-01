"""Tests for configuration management."""

import pytest
import tempfile
from pathlib import Path
from inferx.config import ConfigManager
from inferx.models import BackendType, Preset


class TestConfigManager:
    def test_default_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        
        assert manager.config.default_backend == BackendType.llamacpp
        assert manager.config.port_range_start == 8080

    def test_update_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        
        manager.update_config(default_backend=BackendType.vllm)
        assert manager.config.default_backend == BackendType.vllm

    def test_preset_operations(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        manager = ConfigManager(str(config_file))
        
        # Create preset
        preset = Preset(
            name="test-preset",
            description="Test",
            backend=BackendType.llamacpp,
            ctx_size=4096,
        )
        manager.save_preset(preset)
        
        # Get preset
        retrieved = manager.get_preset("test-preset")
        assert retrieved is not None
        assert retrieved.ctx_size == 4096
        
        # List presets
        presets = manager.list_presets()
        assert "test-preset" in presets
        
        # Delete preset
        assert manager.delete_preset("test-preset")
        assert manager.get_preset("test-preset") is None
