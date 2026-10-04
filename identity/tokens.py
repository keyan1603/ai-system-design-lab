"""Lab-grade signed identity tokens and delegation (token exchange).

NOT a production design. A real deployment uses OAuth 2.1 / OIDC with an
authorization server, asymmetric keys and the security schemes the A2A Agent
Card declares. This module keeps the same *shape* so the architecture can be
exercised end to end without an identity provider:

  * a user logs in and receives a token for the entry service (audience);
  * a service that needs another service on the user's behalf EXCHANGES that
    token for one aimed at the other service (a new audience), as in OAuth 2.0
    token exchange (RFC 8693);
  * the exchanged token's groups are the INTERSECTION of the user's groups and
    the calling service's own ceiling, so a service can never lend more access
    than it holds itself (no confused deputy);
  * tokens are short-lived, audience-bound and HMAC-signed, and a verifier
    rejects anything malformed, forged, expired or aimed at someone else.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional


class AuthError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Claims:
    sub: str                      # the human (or principal) the access belongs to
    groups: tuple
    aud: str                      # the service this token is for
    exp: float
    act: Optional[str] = None     # the service acting on the subject's behalf, if delegated
    jti: str = ""


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue(secret: bytes, sub: str, groups, aud: str, ttl_s: float = 300.0, act: Optional[str] = None,
          now: Optional[float] = None) -> str:
    payload = {"sub": sub, "groups": sorted(groups), "aud": aud, "act": act, "jti": uuid.uuid4().hex[:12],
               "exp": (now if now is not None else time.time()) + ttl_s}
    body = _b64(json.dumps(payload, sort_keys=True).encode())
    sig = _b64(hmac.new(secret, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify(secret: bytes, token: str, aud: str, now: Optional[float] = None) -> Claims:
    try:
        body, sig = token.split(".")
        payload = json.loads(_unb64(body))
    except Exception as exc:  # noqa: BLE001 - any parse failure is just "malformed"
        raise AuthError("malformed") from exc
    expected = _b64(hmac.new(secret, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        raise AuthError("bad_signature")
    if payload.get("aud") != aud:
        raise AuthError("wrong_audience")
    if (now if now is not None else time.time()) >= float(payload["exp"]):
        raise AuthError("expired")
    return Claims(payload["sub"], tuple(payload["groups"]), payload["aud"], float(payload["exp"]),
                  payload.get("act"), payload.get("jti", ""))


class IdentityProvider:
    """Issues user tokens and performs token exchange against per-service ceilings."""

    def __init__(self, secret: bytes, directory: dict, service_ceilings: dict, ttl_s: float = 300.0) -> None:
        self._secret, self._directory, self._ceilings, self.ttl_s = secret, directory, service_ceilings, ttl_s

    @property
    def secret(self) -> bytes:
        return self._secret

    def login(self, username: str, audience: str, now: Optional[float] = None) -> str:
        user = self._directory[username]
        return issue(self._secret, user.name, user.groups, audience, self.ttl_s, now=now)

    def exchange(self, token: str, calling_service: str, target_audience: str, now: Optional[float] = None) -> str:
        """Trade a token addressed to `calling_service` for one addressed to `target_audience`.

        Effective groups = the user's groups intersected with the calling
        service's ceiling. A service that was never registered gets an empty
        ceiling, hence no access (default deny).
        """
        claims = verify(self._secret, token, aud=calling_service, now=now)
        ceiling = set(self._ceilings.get(calling_service, ()))
        effective = sorted(set(claims.groups) & ceiling)
        return issue(self._secret, claims.sub, effective, target_audience, self.ttl_s, act=calling_service, now=now)
