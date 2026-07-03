"""Comprehensive tests for API endpoints."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from inferx.main import app
from inferx.config import ConfigManager
from inferx.manager import InstanceManager
from inferx.router import init_routes


@pytest.fixture(scope="module")
def setup_app():
    """Initialize app for testing."""
    config = ConfigManager()
    manager = InstanceManager(config)
    init_routes(config, manager)
    return app


@pytest.fixture
def client(setup_app):
    return TestClient(setup_app)


# ---------------------------------------------------------------------------
# System endpoints
# ---------------------------------------------------------------------------

class TestSystemHealth:
    def test_health(self, client):
        r = client.get("/api/system/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "instances_running" in data
        assert "instances_total" in data

    def test_health_values(self, client):
        r = client.get("/api/system/health")
        data = r.json()
        assert isinstance(data["instances_running"], int)
        assert isinstance(data["instances_total"], int)


class TestSystemVersion:
    def test_version(self, client):
        r = client.get("/api/system/version")
        assert r.status_code == 200
        data = r.json()
        assert "version" in data
        assert "name" in data
        assert "python_version" in data
        assert "build_date" in data


class TestSystemInfo:
    def test_info(self, client):
        r = client.get("/api/system/info")
        assert r.status_code == 200
        data = r.json()
        assert "gpus" in data
        assert "total_ram_mb" in data
        assert "cpu_count" in data
        assert "cpu_percent" in data
        assert "server_paths" in data

    def test_info_gpu_fields(self, client):
        r = client.get("/api/system/info")
        data = r.json()
        if data["gpus"]:
            gpu = data["gpus"][0]
            assert "index" in gpu
            assert "name" in gpu
            assert "total_memory_mb" in gpu
            assert "used_memory_mb" in gpu


class TestSystemBackends:
    def test_backends(self, client):
        r = client.get("/api/system/backends")
        assert r.status_code == 200
        data = r.json()
        assert "backends" in data
        assert "default" in data
        assert len(data["backends"]) == 8

    def test_backend_fields(self, client):
        r = client.get("/api/system/backends")
        data = r.json()
        for b in data["backends"]:
            assert "id" in b
            assert "name" in b
            assert "installed" in b
            assert isinstance(b["installed"], bool)


class TestSystemConfig:
    def test_get_config(self, client):
        r = client.get("/api/system/config")
        assert r.status_code == 200
        data = r.json()
        assert "model_dir" in data
        assert "default_backend" in data
        assert "port_range_start" in data

    def test_update_config(self, client):
        r = client.put("/api/system/config", json={"port_range_start": 9500})
        assert r.status_code == 200
        data = r.json()
        assert data["port_range_start"] == 9500
        # Reset
        client.put("/api/system/config", json={"port_range_start": 8080})


class TestSystemStats:
    def test_stats(self, client):
        r = client.get("/api/system/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_instances" in data
        assert "by_backend" in data
        assert "by_status" in data
        assert "total_models" in data
        assert "total_presets" in data


class TestSystemGPUs:
    def test_gpus(self, client):
        r = client.get("/api/system/gpus")
        assert r.status_code == 200
        data = r.json()
        assert "gpus" in data
        assert "count" in data


# ---------------------------------------------------------------------------
# Instance endpoints
# ---------------------------------------------------------------------------

class TestListInstances:
    def test_list(self, client):
        r = client.get("/api/instances")
        assert r.status_code == 200
        data = r.json()
        assert "instances" in data
        assert "total" in data

    def test_filter_by_backend(self, client):
        r = client.get("/api/instances?backend=llamacpp")
        assert r.status_code == 200

    def test_filter_by_status(self, client):
        r = client.get("/api/instances?status=running")
        assert r.status_code == 200


class TestGetInstance:
    def test_nonexistent(self, client):
        r = client.get("/api/instances/nonexistent")
        assert r.status_code == 404


class TestStopInstance:
    def test_nonexistent(self, client):
        r = client.delete("/api/instances/nonexistent")
        assert r.status_code == 404


class TestRestartInstance:
    def test_nonexistent(self, client):
        r = client.post("/api/instances/nonexistent/restart")
        assert r.status_code == 404


class TestInstanceLogs:
    def test_nonexistent(self, client):
        r = client.get("/api/instances/nonexistent/logs")
        # API returns 200 with empty logs for nonexistent instances
        assert r.status_code == 200


class TestInstanceError:
    def test_nonexistent(self, client):
        r = client.get("/api/instances/nonexistent/error")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Preset endpoints
# ---------------------------------------------------------------------------

class TestPresets:
    def test_list(self, client):
        r = client.get("/api/presets")
        assert r.status_code == 200

    def test_create_delete(self, client):
        r = client.post("/api/presets", json={
            "name": "api-test-preset", "description": "Test", "backend": "llamacpp",
        })
        assert r.status_code == 200
        r = client.delete("/api/presets/api-test-preset")
        assert r.status_code == 200

    def test_update(self, client):
        client.post("/api/presets", json={"name": "upd-test", "ctx_size": 2048})
        r = client.put("/api/presets/upd-test", json={"name": "upd-test", "ctx_size": 8192})
        assert r.status_code == 200
        client.delete("/api/presets/upd-test")

    def test_clone(self, client):
        client.post("/api/presets", json={"name": "clone-src"})
        r = client.post("/api/presets/clone-src/clone?new_name=clone-dst")
        assert r.status_code == 200
        client.delete("/api/presets/clone-src")
        client.delete("/api/presets/clone-dst")

    def test_delete_nonexistent(self, client):
        r = client.delete("/api/presets/nonexistent")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Alert endpoints
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_list_rules(self, client):
        r = client.get("/api/alerts/rules")
        assert r.status_code == 200
        assert "rules" in r.json()

    def test_create_rule(self, client):
        r = client.post("/api/alerts/rules", json={
            "name": "test-rule", "metric": "gpu_memory_pct",
            "condition": "gt", "threshold": 90.0,
        })
        assert r.status_code == 200
        rule_id = r.json()["id"]
        # Cleanup
        client.delete(f"/api/alerts/rules/{rule_id}")

    def test_list_alerts(self, client):
        r = client.get("/api/alerts")
        assert r.status_code == 200
        assert "alerts" in r.json()

    def test_check_alerts(self, client):
        r = client.get("/api/alerts/check")
        assert r.status_code == 200
        assert "new_alerts" in r.json()


# ---------------------------------------------------------------------------
# Audit endpoints
# ---------------------------------------------------------------------------

class TestAudit:
    def test_list(self, client):
        r = client.get("/api/audit")
        assert r.status_code == 200
        data = r.json()
        assert "entries" in data
        assert "total" in data

    def test_stats(self, client):
        r = client.get("/api/audit/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_entries" in data


# ---------------------------------------------------------------------------
# Export/Import
# ---------------------------------------------------------------------------

class TestExportImport:
    def test_export(self, client):
        r = client.get("/api/system/export")
        assert r.status_code == 200
        data = r.json()
        assert "config" in data
        assert "presets" in data
        assert "exported_at" in data


# ---------------------------------------------------------------------------
# Usage stats
# ---------------------------------------------------------------------------

class TestUsageStats:
    def test_overview(self, client):
        r = client.get("/api/stats/overview")
        assert r.status_code == 200

    def test_models(self, client):
        r = client.get("/api/stats/models")
        assert r.status_code == 200

    def test_hourly(self, client):
        r = client.get("/api/stats/hourly")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Benchmark endpoints
# ---------------------------------------------------------------------------

class TestBenchmarkEndpoints:
    def test_list_reports(self, client):
        r = client.get("/api/benchmark/reports")
        assert r.status_code == 200

    def test_list_batches(self, client):
        r = client.get("/api/benchmark/batches")
        assert r.status_code == 200

    def test_get_report_nonexistent(self, client):
        r = client.get("/api/benchmark/reports/nonexistent")
        assert r.status_code == 404

    def test_get_batch_nonexistent(self, client):
        r = client.get("/api/benchmark/batches/nonexistent")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

class TestStaticFiles:
    def test_index(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_docs(self, client):
        r = client.get("/docs")
        assert r.status_code == 200
