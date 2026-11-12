"""Run traces and in-process metrics (counters + per-tool latency)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Event:
    run_id: str
    type: str
    at: float
    step: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


class Tracer:
    def __init__(self, max_events_per_run: int = 1000) -> None:
        self._events: dict[str, list[Event]] = defaultdict(list)
        self._lock = threading.Lock()
        self.max_events = max_events_per_run

    def emit(self, run_id: str, type: str, step: str | None = None, **data: Any) -> None:
        with self._lock:
            ev = self._events[run_id]
            if len(ev) < self.max_events:
                ev.append(Event(run_id, type, time.time(), step, data))

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(e) for e in self._events.get(run_id, [])]


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: dict[str, int] = defaultdict(int)
        self._latency: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, by: int = 1) -> None:
        with self._lock:
            self.counters[name] += by

    def observe(self, tool: str, ms: float) -> None:
        with self._lock:
            lat = self._latency[tool]
            lat.append(ms)
            if len(lat) > 1000:
                del lat[:-1000]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            tools = {}
            for name, lat in self._latency.items():
                s = sorted(lat)
                tools[name] = {
                    "calls": len(s),
                    "avg_ms": round(sum(s) / len(s), 2),
                    "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
                    "max_ms": round(s[-1], 2),
                }
            return {"counters": dict(self.counters), "tools": tools}
