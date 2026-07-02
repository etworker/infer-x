"""Monitoring, alerts, usage statistics, and audit logging."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Alert Rules & Alerts
# ---------------------------------------------------------------------------

class AlertRule(BaseModel):
    """Configuration for a monitoring alert rule."""
    id: str
    name: str
    enabled: bool = True
    metric: str  # gpu_memory_pct, gpu_utilization, cpu_percent, ram_percent, instance_count
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    duration_seconds: int = 60  # How long condition must be true
    cooldown_seconds: int = 300  # Minimum time between alerts
    notify_channels: list[str] = Field(default_factory=lambda: ["log"])
    message_template: str = ""


class Alert(BaseModel):
    """An active or resolved alert."""
    id: str
    rule_id: str
    rule_name: str
    status: str  # firing, resolved
    metric: str
    current_value: float
    threshold: float
    message: str
    fired_at: str
    resolved_at: str | None = None
    acknowledged: bool = False


class AlertManager:
    """Manages alert rules and active alerts."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._rules: dict[str, AlertRule] = {}
        self._alerts: list[Alert] = []
        self._alert_timestamps: dict[str, float] = {}  # rule_id -> last_alert_time
        self._condition_start: dict[str, float] = {}  # rule_id -> condition_start_time
        self._load_rules()

    def _load_rules(self):
        rules_file = self._data_dir / "alert_rules.json"
        if rules_file.exists():
            try:
                with open(rules_file) as f:
                    data = json.load(f)
                for rule_data in data.get("rules", []):
                    rule = AlertRule(**rule_data)
                    self._rules[rule.id] = rule
            except Exception:
                pass

    def _save_rules(self):
        rules_file = self._data_dir / "alert_rules.json"
        rules_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"rules": [r.model_dump() for r in self._rules.values()]}
        with open(rules_file, "w") as f:
            json.dump(data, f, indent=2)

    def list_rules(self) -> list[AlertRule]:
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> AlertRule | None:
        return self._rules.get(rule_id)

    def create_rule(self, rule: AlertRule) -> AlertRule:
        self._rules[rule.id] = rule
        self._save_rules()
        return rule

    def update_rule(self, rule_id: str, updates: dict[str, Any]) -> AlertRule | None:
        rule = self._rules.get(rule_id)
        if not rule:
            return None
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        self._save_rules()
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        if rule_id in self._rules:
            del self._rules[rule_id]
            self._save_rules()
            return True
        return False

    def list_alerts(self, status: str | None = None) -> list[Alert]:
        if status:
            return [a for a in self._alerts if a.status == status]
        return list(self._alerts)

    def acknowledge_alert(self, alert_id: str) -> bool:
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def check_metric(self, metric_name: str, current_value: float) -> list[Alert]:
        """Check if any rules are triggered by the current metric value."""
        import uuid
        new_alerts = []
        now = time.time()

        for rule in self._rules.values():
            if not rule.enabled or rule.metric != metric_name:
                continue

            condition_met = self._evaluate_condition(
                current_value, rule.condition, rule.threshold
            )

            if condition_met:
                # Track how long condition has been true
                start = self._condition_start.get(rule.id, now)
                self._condition_start[rule.id] = start

                if now - start >= rule.duration_seconds:
                    # Check cooldown
                    last_alert = self._alert_timestamps.get(rule.id, 0)
                    if now - last_alert >= rule.cooldown_seconds:
                        alert = Alert(
                            id=f"alert-{uuid.uuid4().hex[:8]}",
                            rule_id=rule.id,
                            rule_name=rule.name,
                            status="firing",
                            metric=rule.metric,
                            current_value=current_value,
                            threshold=rule.threshold,
                            message=rule.message_template or f"{rule.metric} is {current_value} (threshold: {rule.threshold})",
                            fired_at=datetime.now().isoformat(),
                        )
                        self._alerts.append(alert)
                        self._alert_timestamps[rule.id] = now
                        new_alerts.append(alert)
            else:
                # Condition no longer met, reset tracking
                self._condition_start.pop(rule.id, None)

        return new_alerts

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return value == threshold
        return False


# ---------------------------------------------------------------------------
# Usage Statistics
# ---------------------------------------------------------------------------

class RequestStats(BaseModel):
    """Statistics for a single request or aggregated stats."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0


class ModelStats(BaseModel):
    """Per-model usage statistics."""
    model_name: str
    backend: str
    total_requests: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    avg_latency_ms: float = 0.0
    last_used: str | None = None
    first_used: str | None = None


class UsageTracker:
    """Tracks API usage statistics."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir
        self._request_log: list[dict[str, Any]] = []
        self._model_stats: dict[str, ModelStats] = {}
        self._latencies: list[float] = []
        self._hourly_counts: dict[str, int] = defaultdict(int)  # "YYYY-MM-DD-HH" -> count
        self._load_stats()

    def _load_stats(self):
        stats_file = self._data_dir / "usage_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    data = json.load(f)
                self._hourly_counts = defaultdict(int, data.get("hourly_counts", {}))
                for ms_data in data.get("model_stats", []):
                    ms = ModelStats(**ms_data)
                    self._model_stats[ms.model_name] = ms
            except Exception:
                pass

    def _save_stats(self):
        stats_file = self._data_dir / "usage_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_stats": [ms.model_dump() for ms in self._model_stats.values()],
            "hourly_counts": dict(self._hourly_counts),
        }
        with open(stats_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_request(
        self,
        model: str,
        backend: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        success: bool = True,
    ):
        """Record a single request."""
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d-%H")
        self._hourly_counts[hour_key] += 1
        self._latencies.append(latency_ms)

        # Keep only last 10000 latencies
        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-10000:]

        # Update model stats
        key = f"{model}:{backend}"
        if key not in self._model_stats:
            self._model_stats[key] = ModelStats(
                model_name=model,
                backend=backend,
                first_used=now.isoformat(),
            )
        ms = self._model_stats[key]
        ms.total_requests += 1
        ms.total_tokens_in += tokens_in
        ms.total_tokens_out += tokens_out
        ms.last_used = now.isoformat()

        # Update latency (running average)
        if ms.total_requests > 1:
            ms.avg_latency_ms = (
                (ms.avg_latency_ms * (ms.total_requests - 1) + latency_ms)
                / ms.total_requests
            )
        else:
            ms.avg_latency_ms = latency_ms

        # Save periodically
        if ms.total_requests % 10 == 0:
            self._save_stats()

    def get_overall_stats(self) -> RequestStats:
        """Get overall usage statistics."""
        total = sum(ms.total_requests for ms in self._model_stats.values())
        tokens_in = sum(ms.total_tokens_in for ms in self._model_stats.values())
        tokens_out = sum(ms.total_tokens_out for ms in self._model_stats.values())

        latencies = sorted(self._latencies) if self._latencies else [0]
        n = len(latencies)

        return RequestStats(
            total_requests=total,
            total_tokens_in=tokens_in,
            total_tokens_out=tokens_out,
            avg_latency_ms=sum(latencies) / n if n else 0,
            p50_latency_ms=latencies[n // 2] if n else 0,
            p95_latency_ms=latencies[int(n * 0.95)] if n else 0,
            p99_latency_ms=latencies[int(n * 0.99)] if n else 0,
            max_latency_ms=max(latencies) if latencies else 0,
        )

    def get_model_stats(self) -> list[ModelStats]:
        """Get per-model usage statistics."""
        return list(self._model_stats.values())

    def get_hourly_stats(self, days: int = 7) -> dict[str, int]:
        """Get hourly request counts for the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d-%H")
        return {k: v for k, v in self._hourly_counts.items() if k >= cutoff}


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """Single audit log entry."""
    id: str
    timestamp: str
    action: str  # instance.start, instance.stop, config.update, model.download, etc.
    actor: str = "system"  # user, system, api
    target_type: str  # instance, model, config, preset
    target_id: str
    details: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_message: str | None = None
    ip_address: str | None = None


class AuditLogger:
    """Logs all system operations for audit trail."""

    def __init__(self, data_dir: Path, max_entries: int = 10000):
        self._data_dir = data_dir
        self._entries: list[AuditEntry] = []
        self._max_entries = max_entries
        self._load_entries()

    def _load_entries(self):
        log_file = self._data_dir / "audit_log.json"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    data = json.load(f)
                for entry_data in data.get("entries", []):
                    self._entries.append(AuditEntry(**entry_data))
            except Exception:
                pass

    def _save_entries(self):
        log_file = self._data_dir / "audit_log.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Keep only recent entries
        entries_to_save = self._entries[-self._max_entries:]
        data = {"entries": [e.model_dump() for e in entries_to_save]}
        with open(log_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def log(
        self,
        action: str,
        target_type: str,
        target_id: str,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
        actor: str = "api",
        ip_address: str | None = None,
    ) -> AuditEntry:
        """Create an audit log entry."""
        import uuid
        entry = AuditEntry(
            id=f"audit-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            action=action,
            actor=actor,
            target_type=target_type,
            target_id=target_id,
            details=details or {},
            success=success,
            error_message=error_message,
            ip_address=ip_address,
        )
        self._entries.append(entry)

        # Trim if needed
        if len(self._entries) > self._max_entries * 1.1:
            self._entries = self._entries[-self._max_entries:]

        self._save_entries()
        return entry

    def list_entries(
        self,
        action: str | None = None,
        target_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """List audit entries with optional filters."""
        entries = self._entries

        if action:
            entries = [e for e in entries if e.action.startswith(action)]
        if target_type:
            entries = [e for e in entries if e.target_type == target_type]

        # Return most recent first
        entries = list(reversed(entries))
        return entries[offset:offset + limit]

    def get_entry(self, entry_id: str) -> AuditEntry | None:
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics."""
        by_action = defaultdict(int)
        by_target = defaultdict(int)
        for entry in self._entries:
            by_action[entry.action] += 1
            by_target[entry.target_type] += 1

        return {
            "total_entries": len(self._entries),
            "by_action": dict(by_action),
            "by_target_type": dict(by_target),
            "oldest_entry": self._entries[0].timestamp if self._entries else None,
            "newest_entry": self._entries[-1].timestamp if self._entries else None,
        }
