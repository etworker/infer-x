"""Alert rules and alert management."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AlertRule(BaseModel):
    """Configuration for a monitoring alert rule."""
    id: str
    name: str
    enabled: bool = True
    metric: str
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    duration_seconds: int = 60
    cooldown_seconds: int = 300
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
        self._alert_timestamps: dict[str, float] = {}
        self._condition_start: dict[str, float] = {}
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
        new_alerts = []
        now = time.time()

        for rule in self._rules.values():
            if not rule.enabled or rule.metric != metric_name:
                continue

            condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)

            if condition_met:
                start = self._condition_start.get(rule.id, now)
                self._condition_start[rule.id] = start

                if now - start >= rule.duration_seconds:
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
                self._condition_start.pop(rule.id, None)

        return new_alerts

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        ops = {"gt": lambda a, b: a > b, "lt": lambda a, b: a < b,
               "gte": lambda a, b: a >= b, "lte": lambda a, b: a <= b,
               "eq": lambda a, b: a == b}
        return ops.get(condition, lambda a, b: False)(value, threshold)
