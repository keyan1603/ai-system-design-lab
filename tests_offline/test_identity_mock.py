import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from identity.tokens import AuthError, IdentityProvider, issue, verify
from knowledge.corpus import USERS

SECRET = b"unit-test-secret"
IDP = IdentityProvider(SECRET, USERS, {"ticket-agent": {"employee", "support"}}, ttl_s=60)


def test_roundtrip_and_claims():
    t = issue(SECRET, "sam", ["support", "employee"], "knowledge-agent", ttl_s=60, now=1000)
    c = verify(SECRET, t, "knowledge-agent", now=1001)
    assert c.sub == "sam" and c.groups == ("employee", "support") and c.act is None


@pytest.mark.parametrize("mutate,reason", [
    (lambda t: t[:-4] + "AAAA", "bad_signature"),
    (lambda t: "garbage", "malformed"),
    (lambda t: t.split(".")[0] + ".", "bad_signature"),
])
def test_tampered_or_malformed_tokens_are_rejected(mutate, reason):
    t = issue(SECRET, "sam", ["support"], "knowledge-agent", now=1000)
    with pytest.raises(AuthError) as e:
        verify(SECRET, mutate(t), "knowledge-agent", now=1001)
    assert e.value.reason == reason


def test_expired_and_wrong_audience_and_wrong_key():
    t = issue(SECRET, "sam", ["support"], "knowledge-agent", ttl_s=10, now=1000)
    with pytest.raises(AuthError, match="expired"):
        verify(SECRET, t, "knowledge-agent", now=1010)
    with pytest.raises(AuthError, match="wrong_audience"):
        verify(SECRET, t, "ticket-agent", now=1001)
    with pytest.raises(AuthError, match="bad_signature"):
        verify(b"another-secret", t, "knowledge-agent", now=1001)


def test_payload_edit_with_original_signature_is_rejected():
    import base64, json
    t = issue(SECRET, "dave", ["contractor"], "knowledge-agent", now=1000)
    body, sig = t.split(".")
    payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    payload["groups"] = ["exec"]
    forged = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True).encode()).decode().rstrip("=") + "." + sig
    with pytest.raises(AuthError, match="bad_signature"):
        verify(SECRET, forged, "knowledge-agent", now=1001)


def test_exchange_intersects_user_groups_with_service_ceiling():
    user_token = IDP.login("erin", "ticket-agent", now=1000)             # erin: employee + exec
    ex = IDP.exchange(user_token, "ticket-agent", "knowledge-agent", now=1001)
    c = verify(SECRET, ex, "knowledge-agent", now=1002)
    assert c.sub == "erin" and c.act == "ticket-agent"
    assert c.groups == ("employee",)                                      # exec is NOT lent by the ticket agent


def test_exchange_for_support_user_keeps_support():
    ex = IDP.exchange(IDP.login("sam", "ticket-agent", now=1000), "ticket-agent", "knowledge-agent", now=1001)
    assert set(verify(SECRET, ex, "knowledge-agent", now=1002).groups) == {"employee", "support"}


def test_unregistered_service_has_no_ceiling_and_gets_no_access():
    tok = IDP.login("sam", "mystery-agent", now=1000)
    ex = IDP.exchange(tok, "mystery-agent", "knowledge-agent", now=1001)
    assert verify(SECRET, ex, "knowledge-agent", now=1002).groups == ()


def test_exchange_rejects_a_token_not_addressed_to_the_calling_service():
    wrong = IDP.login("sam", "someone-else", now=1000)
    with pytest.raises(AuthError, match="wrong_audience"):
        IDP.exchange(wrong, "ticket-agent", "knowledge-agent", now=1001)
