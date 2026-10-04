"""Semantic response cache for the gateway.

Two lookups, cheapest first:

1. Exact match on a hash of (scope, full prompt). No embedding call, no false
   positives. Catches the common case of the same ticket or question repeated.
2. Embedding similarity above a threshold, within the same scope. Catches
   rephrasings, at the price of one embedding call per miss and a nonzero
   chance of returning an answer to a *different* question. That risk is why
   the threshold is explicit and why run_finops.py measures it instead of assuming.

Every entry is stored under a *scope* (tenant, plus a hash of the system
prompt). A cache that ignores scope is a cross-tenant data leak waiting to
happen, so the scope is part of the key and lookups never cross it.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from requisite.core.interfaces import ChatResponse, Message


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def prompt_text(messages: Sequence[Message]) -> str:
    """Flatten a conversation into the text that gets hashed and embedded."""
    return "\n".join(f"{getattr(m.role, 'value', m.role)}: {m.content}" for m in messages)


def make_scope(tenant: str, messages: Sequence[Message]) -> str:
    """Scope = tenant + hash of the system prompt(s), so different instructions never share entries."""
    system = "\n".join(m.content for m in messages if getattr(m.role, "value", m.role) == "system")
    return f"{tenant}:{hashlib.sha256(system.encode()).hexdigest()[:12]}"


@dataclass
class CacheHit:
    response: ChatResponse
    kind: str          # "exact" or "semantic"
    similarity: float  # 1.0 for exact


@dataclass
class _Entry:
    scope: str
    key: str
    embedding: Optional[list[float]]
    response: ChatResponse
    stored_at: float


@dataclass
class SemanticCache:
    embed: Callable[[str], list[float]]
    threshold: float = 0.92
    ttl_s: float = 3600.0
    max_entries: int = 512
    clock: Callable[[], float] = time.monotonic
    hits_exact: int = 0
    hits_semantic: int = 0
    misses: int = 0
    evictions: int = 0
    _entries: "OrderedDict[str, _Entry]" = field(default_factory=OrderedDict)

    def _key(self, scope: str, text: str) -> str:
        return hashlib.sha256(f"{scope}\x00{text}".encode()).hexdigest()

    def _expired(self, e: _Entry) -> bool:
        return self.clock() - e.stored_at > self.ttl_s

    def lookup(self, scope: str, text: str) -> Optional[CacheHit]:
        key = self._key(scope, text)
        entry = self._entries.get(key)
        if entry is not None and not self._expired(entry):
            self._entries.move_to_end(key)
            self.hits_exact += 1
            return CacheHit(entry.response, "exact", 1.0)

        scoped = [e for e in self._entries.values() if e.scope == scope and not self._expired(e)]
        if scoped:
            q = self.embed(text)
            best = max(scoped, key=lambda e: _cosine(q, e.embedding or []))
            sim = _cosine(q, best.embedding or [])
            if sim >= self.threshold:
                self._entries.move_to_end(best.key)
                self.hits_semantic += 1
                return CacheHit(best.response, "semantic", sim)
        self.misses += 1
        return None

    def store(self, scope: str, text: str, response: ChatResponse) -> None:
        key = self._key(scope, text)
        # The embedding is computed at store time so a later lookup costs one
        # embedding call (the query), not one per cached entry.
        self._entries[key] = _Entry(scope, key, self.embed(text), response, self.clock())
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1

    def stats(self) -> dict:
        total = self.hits_exact + self.hits_semantic + self.misses
        return {
            "entries": len(self._entries),
            "hits_exact": self.hits_exact,
            "hits_semantic": self.hits_semantic,
            "misses": self.misses,
            "hit_rate": round((self.hits_exact + self.hits_semantic) / total, 3) if total else 0.0,
        }
