"""Circuit breaker used by the gateway to stop sending traffic to a failing route.

Requisite's providers already retry transient errors a couple of times
(`BaseProvider._call_with_retries`), which protects a single request. A breaker
protects the *fleet*: once a route has failed repeatedly, the gateway stops
paying a timeout on every request and skips straight to the next route, then
probes the failed route again after a cool-down.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

CLOSED = "closed"          # healthy, traffic flows
OPEN = "open"              # failing, traffic is skipped
HALF_OPEN = "half_open"    # cool-down elapsed, exactly one probe request allowed


@dataclass
class CircuitBreaker:
    """Three-state breaker with an injectable clock so tests never sleep."""

    failure_threshold: int = 3
    reset_timeout_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _state: str = field(default=CLOSED, init=False)
    _probe_in_flight: bool = field(default=False, init=False)

    @property
    def state(self) -> str:
        # An OPEN breaker becomes HALF_OPEN lazily, the first time anyone looks
        # after the cool-down, so no background timer is needed.
        if self._state == OPEN and self.clock() - self._opened_at >= self.reset_timeout_s:
            self._state = HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def allow(self) -> bool:
        """True if a request may be sent to this route right now."""
        state = self.state
        if state == CLOSED:
            return True
        if state == HALF_OPEN and not self._probe_in_flight:
            # Only one probe at a time, otherwise a recovering route gets a
            # thundering herd the moment its cool-down ends.
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._state = CLOSED
        self._probe_in_flight = False

    def record_failure(self) -> None:
        self._probe_in_flight = False
        if self._state == HALF_OPEN:
            # The probe failed: straight back to OPEN with a fresh cool-down.
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = OPEN
        self._opened_at = self.clock()
