"""Audit logging for system operations."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """Single audit log entry."""
    id: str
    timestamp: str
    action: str
    actor: str = "system"
    target_type: str
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
        entry = AuditEntry(
            id=f"audit-{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            action=action, actor=actor, target_type=target_type,
            target_id=target_id, details=details or {},
            success=success, error_message=error_message, ip_address=ip_address,
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries * 1.1:
            self._entries = self._entries[-self._max_entries:]
        self._save_entries()
        return entry

    def list_entries(
        self, action: str | None = None, target_type: str | None = None,
        limit: int = 100, offset: int = 0,
    ) -> list[AuditEntry]:
        entries = self._entries
        if action:
            entries = [e for e in entries if e.action.startswith(action)]
        if target_type:
            entries = [e for e in entries if e.target_type == target_type]
        entries = list(reversed(entries))
        return entries[offset:offset + limit]

    def get_entry(self, entry_id: str) -> AuditEntry | None:
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_stats(self) -> dict[str, Any]:
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
