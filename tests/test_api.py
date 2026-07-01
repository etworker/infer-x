"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from infer_helper.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestSystemEndpoints:
    def test_system_health(self, client):
        response = client.get("/api/system/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "instances_running" in data

    def test_system_version(self, client):
        response = client.get("/api/system/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "name" in data

    def test_list_backends(self, client):
        response = client.get("/api/system/backends")
        assert response.status_code == 200
        data = response.json()
        assert "backends" in data
        assert len(data["backends"]) > 0


class TestInstanceEndpoints:
    def test_list_instances(self, client):
        response = client.get("/api/instances")
        assert response.status_code == 200
        data = response.json()
        assert "instances" in data
        assert "total" in data

    def test_list_instances_with_filter(self, client):
        response = client.get("/api/instances?backend=llamacpp")
        assert response.status_code == 200

    def test_get_nonexistent_instance(self, client):
        response = client.get("/api/instances/nonexistent")
        assert response.status_code == 404


class TestPresetEndpoints:
    def test_list_presets(self, client):
        response = client.get("/api/presets")
        assert response.status_code == 200

    def test_create_and_delete_preset(self, client):
        # Create
        response = client.post("/api/presets", json={
            "name": "test-api-preset",
            "description": "Test",
            "backend": "llamacpp",
        })
        assert response.status_code == 200
        
        # Delete
        response = client.delete("/api/presets/test-api-preset")
        assert response.status_code == 200
