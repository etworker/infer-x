"""Comprehensive tests for monitoring, alerts, usage stats, and audit logging."""

import time
import pytest
from pathlib import Path
from inferx.monitoring import (
    AlertManager,
    AlertRule,
    UsageTracker,
    AuditLogger,
)


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class TestAlertRule:
    def test_creation(self):
        rule = AlertRule(
            id="rule-1", name="High GPU", metric="gpu_memory_pct",
            condition="gt", threshold=90.0,
        )
        assert rule.id == "rule-1"
        assert rule.enabled is True
        assert rule.duration_seconds == 60
        assert rule.cooldown_seconds == 300

    def test_disabled_rule(self):
        rule = AlertRule(
            id="r", name="r", enabled=False,
            metric="cpu", condition="gt", threshold=80,
        )
        assert rule.enabled is False


class TestAlertManager:
    def test_init(self, tmp_path):
        mgr = AlertManager(tmp_path)
        assert len(mgr.list_rules()) == 0
        assert len(mgr.list_alerts()) == 0

    def test_create_rule(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(
            id="r1", name="GPU High", metric="gpu_memory_pct",
            condition="gt", threshold=90.0,
        )
        created = mgr.create_rule(rule)
        assert created.id == "r1"
        assert len(mgr.list_rules()) == 1

    def test_get_rule(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(id="r1", name="test", metric="cpu", condition="gt", threshold=80)
        mgr.create_rule(rule)
        assert mgr.get_rule("r1") is not None
        assert mgr.get_rule("nonexistent") is None

    def test_update_rule(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(id="r1", name="old", metric="cpu", condition="gt", threshold=80)
        mgr.create_rule(rule)
        updated = mgr.update_rule("r1", {"name": "new", "threshold": 95})
        assert updated.name == "new"
        assert updated.threshold == 95

    def test_update_nonexistent_rule(self, tmp_path):
        mgr = AlertManager(tmp_path)
        assert mgr.update_rule("nonexistent", {"name": "x"}) is None

    def test_delete_rule(self, tmp_path):
        mgr = AlertManager(tmp_path)
        mgr.create_rule(AlertRule(id="r1", name="x", metric="cpu", condition="gt", threshold=80))
        assert mgr.delete_rule("r1") is True
        assert mgr.get_rule("r1") is None

    def test_delete_nonexistent_rule(self, tmp_path):
        mgr = AlertManager(tmp_path)
        assert mgr.delete_rule("nonexistent") is False

    def test_rule_persistence(self, tmp_path):
        mgr = AlertManager(tmp_path)
        mgr.create_rule(AlertRule(id="r1", name="persist", metric="cpu", condition="gt", threshold=80))
        mgr2 = AlertManager(tmp_path)
        assert mgr2.get_rule("r1") is not None
        assert mgr2.get_rule("r1").name == "persist"

    def test_check_metric_triggers_alert(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(
            id="r1", name="GPU alert", metric="gpu_memory_pct",
            condition="gt", threshold=80.0, duration_seconds=0, cooldown_seconds=0,
        )
        mgr.create_rule(rule)
        alerts = mgr.check_metric("gpu_memory_pct", 95.0)
        assert len(alerts) == 1
        assert alerts[0].status == "firing"
        assert alerts[0].current_value == 95.0

    def test_check_metric_no_trigger(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(
            id="r1", name="GPU alert", metric="gpu_memory_pct",
            condition="gt", threshold=80.0,
        )
        mgr.create_rule(rule)
        alerts = mgr.check_metric("gpu_memory_pct", 50.0)
        assert len(alerts) == 0

    def test_check_metric_wrong_metric(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(
            id="r1", name="GPU", metric="gpu_memory_pct",
            condition="gt", threshold=80.0,
        )
        mgr.create_rule(rule)
        alerts = mgr.check_metric("cpu_percent", 95.0)
        assert len(alerts) == 0

    def test_check_disabled_rule(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(
            id="r1", name="Disabled", enabled=False,
            metric="gpu_memory_pct", condition="gt", threshold=80.0,
        )
        mgr.create_rule(rule)
        alerts = mgr.check_metric("gpu_memory_pct", 95.0)
        assert len(alerts) == 0

    def test_evaluate_conditions(self, tmp_path):
        mgr = AlertManager(tmp_path)
        # gt
        assert mgr._evaluate_condition(95, "gt", 80) is True
        assert mgr._evaluate_condition(75, "gt", 80) is False
        # lt
        assert mgr._evaluate_condition(50, "lt", 80) is True
        assert mgr._evaluate_condition(90, "lt", 80) is False
        # gte
        assert mgr._evaluate_condition(80, "gte", 80) is True
        assert mgr._evaluate_condition(79, "gte", 80) is False
        # lte
        assert mgr._evaluate_condition(80, "lte", 80) is True
        assert mgr._evaluate_condition(81, "lte", 80) is False
        # eq
        assert mgr._evaluate_condition(80, "eq", 80) is True
        assert mgr._evaluate_condition(81, "eq", 80) is False
        # unknown
        assert mgr._evaluate_condition(80, "unknown", 80) is False

    def test_cooldown_prevents_duplicate(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(
            id="r1", name="alert", metric="x", condition="gt",
            threshold=80.0, duration_seconds=0, cooldown_seconds=9999,
        )
        mgr.create_rule(rule)
        alerts1 = mgr.check_metric("x", 95.0)
        assert len(alerts1) == 1
        # Second check within cooldown should not fire
        alerts2 = mgr.check_metric("x", 95.0)
        assert len(alerts2) == 0

    def test_acknowledge_alert(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(
            id="r1", name="a", metric="x", condition="gt",
            threshold=80.0, duration_seconds=0, cooldown_seconds=0,
        )
        mgr.create_rule(rule)
        alerts = mgr.check_metric("x", 95.0)
        assert len(alerts) == 1
        alert_id = alerts[0].id
        assert mgr.acknowledge_alert(alert_id) is True
        # Check acknowledged
        fired = mgr.list_alerts("firing")
        assert any(a.id == alert_id and a.acknowledged for a in fired)

    def test_acknowledge_nonexistent(self, tmp_path):
        mgr = AlertManager(tmp_path)
        assert mgr.acknowledge_alert("nonexistent") is False

    def test_list_alerts_filtered(self, tmp_path):
        mgr = AlertManager(tmp_path)
        rule = AlertRule(
            id="r1", name="a", metric="x", condition="gt",
            threshold=80.0, duration_seconds=0, cooldown_seconds=0,
        )
        mgr.create_rule(rule)
        mgr.check_metric("x", 95.0)
        assert len(mgr.list_alerts("firing")) >= 1
        assert len(mgr.list_alerts("resolved")) == 0
        assert len(mgr.list_alerts()) >= 1


# ---------------------------------------------------------------------------
# UsageTracker
# ---------------------------------------------------------------------------

class TestUsageTracker:
    def test_init(self, tmp_path):
        tracker = UsageTracker(tmp_path)
        stats = tracker.get_overall_stats()
        assert stats.total_requests == 0

    def test_record_request(self, tmp_path):
        tracker = UsageTracker(tmp_path)
        tracker.record_request("model-a", "vllm", 100.0, tokens_in=10, tokens_out=20)
        stats = tracker.get_overall_stats()
        assert stats.total_requests == 1
        assert stats.total_tokens_in == 10
        assert stats.total_tokens_out == 20

    def test_record_multiple_requests(self, tmp_path):
        tracker = UsageTracker(tmp_path)
        for i in range(5):
            tracker.record_request("model-a", "vllm", float(i * 10))
        stats = tracker.get_overall_stats()
        assert stats.total_requests == 5
        assert stats.avg_latency_ms == 20.0  # (0+10+20+30+40)/5

    def test_model_stats(self, tmp_path):
        tracker = UsageTracker(tmp_path)
        tracker.record_request("m1", "vllm", 100, tokens_in=10, tokens_out=20)
        tracker.record_request("m1", "vllm", 200, tokens_in=15, tokens_out=30)
        tracker.record_request("m2", "llamacpp", 50)
        models = tracker.get_model_stats()
        assert len(models) == 2
        m1 = [m for m in models if m.model_name == "m1"][0]
        assert m1.total_requests == 2
        assert m1.total_tokens_in == 25

    def test_latency_stats(self, tmp_path):
        tracker = UsageTracker(tmp_path)
        for i in range(100):
            tracker.record_request("m", "vllm", float(i))
        stats = tracker.get_overall_stats()
        assert stats.p50_latency_ms == 50.0
        assert stats.max_latency_ms == 99.0

    def test_hourly_stats(self, tmp_path):
        tracker = UsageTracker(tmp_path)
        tracker.record_request("m", "vllm", 100)
        hourly = tracker.get_hourly_stats(days=1)
        assert len(hourly) >= 1

    def test_persistence(self, tmp_path):
        tracker = UsageTracker(tmp_path)
        tracker.record_request("m", "vllm", 100)
        tracker._save_stats()
        tracker2 = UsageTracker(tmp_path)
        stats = tracker2.get_overall_stats()
        assert stats.total_requests == 1


# ---------------------------------------------------------------------------
# AuditLogger
# ---------------------------------------------------------------------------

class TestAuditLogger:
    def test_init(self, tmp_path):
        logger = AuditLogger(tmp_path)
        assert len(logger.list_entries()) == 0

    def test_log_entry(self, tmp_path):
        logger = AuditLogger(tmp_path)
        entry = logger.log(
            action="instance.start", target_type="instance",
            target_id="inst-1", details={"model": "test"},
        )
        assert entry.id.startswith("audit-")
        assert entry.action == "instance.start"
        assert entry.success is True

    def test_log_with_error(self, tmp_path):
        logger = AuditLogger(tmp_path)
        entry = logger.log(
            action="instance.stop", target_type="instance",
            target_id="i1", success=False, error_message="Process not found",
        )
        assert entry.success is False
        assert entry.error_message == "Process not found"

    def test_list_entries(self, tmp_path):
        logger = AuditLogger(tmp_path)
        logger.log(action="a1", target_type="instance", target_id="i1")
        logger.log(action="a2", target_type="model", target_id="m1")
        entries = logger.list_entries()
        assert len(entries) == 2

    def test_list_entries_filtered_by_action(self, tmp_path):
        logger = AuditLogger(tmp_path)
        logger.log(action="instance.start", target_type="instance", target_id="i1")
        logger.log(action="model.download", target_type="model", target_id="m1")
        entries = logger.list_entries(action="instance")
        assert len(entries) == 1

    def test_list_entries_filtered_by_target(self, tmp_path):
        logger = AuditLogger(tmp_path)
        logger.log(action="instance.start", target_type="instance", target_id="i1")
        logger.log(action="model.download", target_type="model", target_id="m1")
        entries = logger.list_entries(target_type="model")
        assert len(entries) == 1

    def test_list_entries_pagination(self, tmp_path):
        logger = AuditLogger(tmp_path)
        for i in range(10):
            logger.log(action="test", target_type="t", target_id=f"i{i}")
        entries = logger.list_entries(limit=3, offset=0)
        assert len(entries) == 3
        entries = logger.list_entries(limit=3, offset=7)
        assert len(entries) == 3

    def test_list_entries_reverse_order(self, tmp_path):
        logger = AuditLogger(tmp_path)
        e1 = logger.log(action="first", target_type="t", target_id="1")
        e2 = logger.log(action="second", target_type="t", target_id="2")
        entries = logger.list_entries()
        # Most recent first
        assert entries[0].id == e2.id

    def test_get_entry(self, tmp_path):
        logger = AuditLogger(tmp_path)
        entry = logger.log(action="test", target_type="t", target_id="1")
        found = logger.get_entry(entry.id)
        assert found is not None
        assert found.id == entry.id

    def test_get_nonexistent_entry(self, tmp_path):
        logger = AuditLogger(tmp_path)
        assert logger.get_entry("nonexistent") is None

    def test_stats(self, tmp_path):
        logger = AuditLogger(tmp_path)
        logger.log(action="instance.start", target_type="instance", target_id="i1")
        logger.log(action="instance.stop", target_type="instance", target_id="i1")
        logger.log(action="model.download", target_type="model", target_id="m1")
        stats = logger.get_stats()
        assert stats["total_entries"] == 3
        assert stats["by_action"]["instance.start"] == 1
        assert stats["by_target_type"]["instance"] == 2

    def test_persistence(self, tmp_path):
        logger = AuditLogger(tmp_path)
        logger.log(action="test", target_type="t", target_id="1")
        logger2 = AuditLogger(tmp_path)
        assert len(logger2.list_entries()) == 1

    def test_ip_address_recorded(self, tmp_path):
        logger = AuditLogger(tmp_path)
        entry = logger.log(
            action="test", target_type="t", target_id="1",
            ip_address="192.168.1.1",
        )
        assert entry.ip_address == "192.168.1.1"

    def test_actor_recorded(self, tmp_path):
        logger = AuditLogger(tmp_path)
        entry = logger.log(
            action="test", target_type="t", target_id="1", actor="user",
        )
        assert entry.actor == "user"
