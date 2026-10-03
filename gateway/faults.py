"""Fault injection: wrap any real provider so an outage can be switched on and off.

The wrapped provider is still the real one; the injector only decides whether
a call reaches it. That makes failover and circuit-breaker behavior testable
against real models without waiting for a real outage. Any result produced
through this wrapper in a write-up must be labelled as an *injected* fault.
"""

from __future__ import annotations

import time
from typing import Any

from requisite.core.exceptions import ProviderException
from requisite.providers.base import BaseProvider


class FaultInjector(BaseProvider):
    def __init__(self, inner: BaseProvider, label: str | None = None) -> None:
        super().__init__(api_key="fault-injector", model=inner.model)
        self.inner = inner
        self.label = label or inner.name
        self.mode = "up"          # up | down | slow
        self.delay_s = 0.0
        self.injected_failures = 0

    name = property(lambda self: self.inner.name)
    model = property(lambda self: self.inner.model)

    def set(self, mode: str, delay_s: float = 0.0) -> None:
        self.mode, self.delay_s = mode, delay_s

    def _gate(self) -> None:
        if self.mode == "down":
            self.injected_failures += 1
            raise ProviderException(f"injected outage on {self.label}", provider=self.label)
        if self.mode == "slow":
            time.sleep(self.delay_s)

    def validate_config(self) -> None:
        self.inner.validate_config()

    def chat(self, messages, **kw: Any):
        self._gate()
        return self.inner.chat(messages, **kw)

    async def achat(self, messages, **kw: Any):
        self._gate()
        return await self.inner.achat(messages, **kw)

    def stream(self, messages, **kw: Any):
        self._gate()
        return self.inner.stream(messages, **kw)

    def astream(self, messages, **kw: Any):
        self._gate()
        return self.inner.astream(messages, **kw)
