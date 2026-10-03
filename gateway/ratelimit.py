"""RateLimitedProvider: put a Requisite RateLimiter directly in front of one real API.

A limiter belongs next to the upstream quota it protects, not at the top of the
gateway. Placed at the top, every failover attempt (including attempts that
never reach a real API) would burn quota budget, and a route with no quota,
like a local model, would be throttled by a limit that has nothing to do with
it. Routes that share one API key share one limiter instance; a local route
gets none.
"""

from __future__ import annotations

from typing import Any

from requisite.core.rate_limiter import RateLimiter
from requisite.providers.base import BaseProvider


class RateLimitedProvider(BaseProvider):
    def __init__(self, inner: BaseProvider, limiter: RateLimiter) -> None:
        super().__init__(api_key="rate-limited", model=inner.model)
        self.inner, self.limiter = inner, limiter

    name = property(lambda self: self.inner.name)
    model = property(lambda self: self.inner.model)

    def validate_config(self) -> None:
        self.inner.validate_config()

    def chat(self, messages, **kw: Any):
        self.limiter.acquire()
        return self.inner.chat(messages, **kw)

    async def achat(self, messages, **kw: Any):
        await self.limiter.aacquire()
        return await self.inner.achat(messages, **kw)

    def stream(self, messages, **kw: Any):
        self.limiter.acquire()
        return self.inner.stream(messages, **kw)

    def astream(self, messages, **kw: Any):
        # Acquire lazily on first iteration so the async generator stays a generator.
        async def _gen():
            await self.limiter.aacquire()
            async for chunk in self.inner.astream(messages, **kw):
                yield chunk
        return _gen()
