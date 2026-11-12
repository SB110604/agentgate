"""Short-term conversation memory and persistent long-term memory (SQLite, keyword-ranked recall)."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from has in is it of on or the to was were with".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


class ConversationMemory:
    """A bounded window of recent messages (oldest messages drop out first)."""

    def __init__(self, max_messages: int = 20) -> None:
        self._messages: deque[dict[str, str]] = deque(maxlen=max_messages)

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})

    def messages(self) -> list[dict[str, str]]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


@dataclass
class MemoryItem:
    key: str
    value: Any
    tags: list[str]
    created_at: float
    score: float = 0.0


class LongTermMemory:
    """Key-value memory persisted in SQLite. ``recall`` ranks items by keyword overlap with the query (TF weighting)."""

    def __init__(self, path: str = ":memory:") -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, value TEXT NOT NULL, tags TEXT NOT NULL, "
            "text TEXT NOT NULL, created_at REAL NOT NULL)"
        )

    def remember(self, key: str, value: Any, tags: list[str] | None = None) -> None:
        text = f"{key} {json.dumps(value, default=str)} {' '.join(tags or [])}"
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO memory VALUES (?, ?, ?, ?, ?)",
                (key, json.dumps(value, default=str), json.dumps(tags or []), text, time.time()),
            )

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM memory WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def forget(self, key: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM memory WHERE key = ?", (key,)).rowcount > 0

    def recall(self, query: str, k: int = 5, tag: str | None = None) -> list[MemoryItem]:
        q = Counter(tokens(query))
        with self._lock:
            rows = self._db.execute("SELECT key, value, tags, text, created_at FROM memory").fetchall()
        out: list[MemoryItem] = []
        for key, value, tags, text, created in rows:
            tag_list = json.loads(tags)
            if tag and tag not in tag_list:
                continue
            doc = Counter(tokens(text))
            score = sum(min(c, doc[w]) for w, c in q.items()) / (1 + len(q))
            if score > 0:
                out.append(MemoryItem(key, json.loads(value), tag_list, created, round(score, 4)))
        out.sort(key=lambda m: (-m.score, -m.created_at))
        return out[:k]

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM memory").fetchone()[0])
