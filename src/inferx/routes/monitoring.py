"""Monitoring routes: alerts, usage stats, audit log."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..monitoring import AlertRule
from . import get_alert_manager, get_audit_logger, get_manager, get_usage_tracker

router = APIRouter()


# --- Alert Rules ---

class CreateAlertRuleRequest(BaseModel):
    name: str
    enabled: bool = True
    metric: str
    condition: str
    threshold: float
    duration_seconds: int = 60
    cooldown_seconds: int = 300
    notify_channels: list[str] = ["log"]
    message_template: str = ""


@router.get("/alerts/rules")
async def list_alert_rules():
    """List all alert rules."""
    return {"rules": [r.model_dump() for r in get_alert_manager().list_rules()]}


@router.post("/alerts/rules")
async def create_alert_rule(body: CreateAlertRuleRequest):
    """Create a new alert rule."""
    import uuid
    rule = AlertRule(
        id=f"rule-{uuid.uuid4().hex[:8]}",
        **body.model_dump()
    )
    return get_alert_manager().create_rule(rule).model_dump()


@router.put("/alerts/rules/{rule_id}")
async def update_alert_rule(rule_id: str, body: CreateAlertRuleRequest):
    """Update an alert rule."""
    rule = get_alert_manager().update_rule(rule_id, body.model_dump())
    if not rule:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    return rule.model_dump()


@router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str):
    """Delete an alert rule."""
    ok = get_alert_manager().delete_rule(rule_id)
    if not ok:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    return {"success": True}


@router.get("/alerts")
async def list_alerts(status: str | None = None):
    """List active/resolved alerts."""
    return {"alerts": [a.model_dump() for a in get_alert_manager().list_alerts(status)]}


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    ok = get_alert_manager().acknowledge_alert(alert_id)
    if not ok:
        raise HTTPException(404, f"Alert not found: {alert_id}")
    return {"success": True}


@router.get("/alerts/check")
async def check_alerts():
    """Manually trigger alert check with current system metrics."""
    mgr = get_manager()
    alerts_mgr = get_alert_manager()
    gpus = mgr.monitor.get_gpus()
    new_alerts = []

    for gpu in gpus:
        mem_pct = (gpu.used_memory_mb / gpu.total_memory_mb * 100) if gpu.total_memory_mb > 0 else 0
        alerts = alerts_mgr.check_metric("gpu_memory_pct", mem_pct)
        new_alerts.extend(alerts)

        if gpu.utilization_pct is not None:
            alerts = alerts_mgr.check_metric("gpu_utilization", gpu.utilization_pct)
            new_alerts.extend(alerts)

    instances = mgr.list_instances()
    running = sum(1 for i in instances if i.status.value == "running")
    alerts = alerts_mgr.check_metric("instance_count", running)
    new_alerts.extend(alerts)

    return {
        "checked_at": datetime.now().isoformat(),
        "new_alerts": [a.model_dump() for a in new_alerts],
    }


# --- Usage Statistics ---

@router.get("/stats/overview")
async def usage_stats_overview():
    """Get overall usage statistics."""
    return get_usage_tracker().get_overall_stats().model_dump()


@router.get("/stats/models")
async def usage_stats_models():
    """Get per-model usage statistics."""
    return {"models": [m.model_dump() for m in get_usage_tracker().get_model_stats()]}


@router.get("/stats/hourly")
async def usage_stats_hourly(days: int = Query(default=7, ge=1, le=30)):
    """Get hourly request counts."""
    return {"hourly": get_usage_tracker().get_hourly_stats(days)}


# --- Audit Log ---

@router.get("/audit")
async def list_audit_logs(
    action: str | None = None,
    target_type: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    """List audit log entries."""
    audit = get_audit_logger()
    entries = audit.list_entries(action, target_type, limit, offset)
    return {
        "entries": [e.model_dump() for e in entries],
        "total": audit.total_entries,
    }


@router.get("/audit/stats")
async def audit_stats():
    """Get audit log statistics."""
    return get_audit_logger().get_stats()


@router.get("/audit/{entry_id}")
async def get_audit_entry(entry_id: str):
    """Get a specific audit entry."""
    entry = get_audit_logger().get_entry(entry_id)
    if not entry:
        raise HTTPException(404, f"Audit entry not found: {entry_id}")
    return entry.model_dump()
