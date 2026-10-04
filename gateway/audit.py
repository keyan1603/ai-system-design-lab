"""Audit trail and metrics for gateway traffic.

One `AuditRecord` per request, whether it succeeded, hit the cache, was
blocked by the budget, or exhausted every route. Prompts are NOT stored by
default, only a hash and a length, because an audit log that copies every
prompt becomes the most sensitive data store in the system. Opt in with
`log_content=True` for non-sensitive environments.
"""

from __future__ import annotations

import json
import statistics
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AuditRecord:
    request_id: str
    tenant: str
    tier: str
    outcome: str                 # ok | cache_exact | cache_semantic | blocked_budget | exhausted
    route: Optional[str]         # route that served it, None if nobody did
    model: Optional[str]
    attempts: list = field(default_factory=list)   # [{"route": ..., "ok": bool, "error": str|None}]
    prompt_sha256: str = ""
    prompt_chars: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    saved_tokens: int = 0         # tokens a cache hit avoided spending
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cache_similarity: Optional[float] = None
    prompt: Optional[str] = None  # only populated when log_content=True
    correlation_id: str = ""      # from the request context, ties gateway calls to one ticket or question


class AuditLog:
    def __init__(self, path: Optional[str] = None, log_content: bool = False) -> None:
        self.records: list[AuditRecord] = []
        self.log_content = log_content
        self._path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, rec: AuditRecord) -> None:
        if not self.log_content:
            rec.prompt = None
        with self._lock:
            self.records.append(rec)
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")

    def summary(self) -> dict:
        recs = self.records
        if not recs:
            return {"requests": 0}
        by_outcome: dict[str, int] = {}
        by_route: dict[str, int] = {}
        for r in recs:
            by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
            if r.route:
                by_route[r.route] = by_route.get(r.route, 0) + 1
        lat = [r.latency_ms for r in recs if r.outcome == "ok"]
        return {
            "requests": len(recs),
            "by_outcome": by_outcome,
            "by_route": by_route,
            "total_cost_usd": round(sum(r.cost_usd for r in recs), 6),
            "input_tokens": sum(r.input_tokens for r in recs),
            "output_tokens": sum(r.output_tokens for r in recs),
            "tokens_saved_by_cache": sum(r.saved_tokens for r in recs),
            "p50_latency_ms_uncached": round(statistics.median(lat), 1) if lat else None,
            "fallbacks": sum(1 for r in recs if len(r.attempts) > 1),
        }
