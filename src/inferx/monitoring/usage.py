"""Usage statistics tracking."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel


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
        self._model_stats: dict[str, ModelStats] = {}
        self._latencies: list[float] = []
        self._hourly_counts: dict[str, int] = defaultdict(int)
        self._successful_requests: int = 0
        self._failed_requests: int = 0
        self._load_stats()

    def _load_stats(self):
        stats_file = self._data_dir / "usage_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    data = json.load(f)
                self._hourly_counts = defaultdict(int, data.get("hourly_counts", {}))
                self._successful_requests = data.get("successful_requests", 0)
                self._failed_requests = data.get("failed_requests", 0)
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
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
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
        now = datetime.now()
        hour_key = now.strftime("%Y-%m-%d-%H")
        self._hourly_counts[hour_key] += 1
        self._latencies.append(latency_ms)

        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-10000:]

        # 清理超过 30 天的小时计数，防止无限增长
        cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d-%H")
        stale_keys = [k for k in self._hourly_counts if k < cutoff]
        for k in stale_keys:
            del self._hourly_counts[k]

        if success:
            self._successful_requests += 1
        else:
            self._failed_requests += 1

        key = f"{model}:{backend}"
        if key not in self._model_stats:
            self._model_stats[key] = ModelStats(
                model_name=model, backend=backend, first_used=now.isoformat(),
            )
        ms = self._model_stats[key]
        ms.total_requests += 1
        ms.total_tokens_in += tokens_in
        ms.total_tokens_out += tokens_out
        ms.last_used = now.isoformat()

        if ms.total_requests > 1:
            ms.avg_latency_ms = (ms.avg_latency_ms * (ms.total_requests - 1) + latency_ms) / ms.total_requests
        else:
            ms.avg_latency_ms = latency_ms

        if ms.total_requests % 10 == 0:
            self._save_stats()

    def get_overall_stats(self) -> RequestStats:
        total = sum(ms.total_requests for ms in self._model_stats.values())
        tokens_in = sum(ms.total_tokens_in for ms in self._model_stats.values())
        tokens_out = sum(ms.total_tokens_out for ms in self._model_stats.values())
        latencies = sorted(self._latencies) if self._latencies else [0]
        n = len(latencies)
        return RequestStats(
            total_requests=total,
            successful_requests=self._successful_requests,
            failed_requests=self._failed_requests,
            total_tokens_in=tokens_in, total_tokens_out=tokens_out,
            avg_latency_ms=sum(latencies) / n if n else 0,
            p50_latency_ms=latencies[n // 2] if n else 0,
            p95_latency_ms=latencies[int(n * 0.95)] if n else 0,
            p99_latency_ms=latencies[int(n * 0.99)] if n else 0,
            max_latency_ms=max(latencies) if latencies else 0,
        )

    def get_model_stats(self) -> list[ModelStats]:
        return list(self._model_stats.values())

    def get_hourly_stats(self, days: int = 7) -> dict[str, int]:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d-%H")
        return {k: v for k, v in self._hourly_counts.items() if k >= cutoff}
