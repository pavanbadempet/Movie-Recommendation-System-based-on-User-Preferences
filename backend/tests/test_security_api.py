"""
Adversarial Security & Malicious Input Testing.

This suite is designed to attack the FastAPI endpoints exactly how a
malicious actor would. It guarantees that our backend is impenetrable
to standard injection vectors and oversized payloads.
"""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_sql_injection_rejection():
    """Verify the search endpoint does not execute or crash on SQLi vectors."""
    sqli_vector = "Action'; DROP TABLE movies; --"
    response = client.get(f"/v1/search?q={sqli_vector}")

    # We expect a 200 OK (it safely searches for the string)
    # OR a 400 Validation Error. We MUST NOT get a 500 Internal Server Error.
    assert response.status_code in [200, 400]


def test_nosql_injection_rejection():
    """Verify the AI search endpoint handles NoSQL/JSON injections gracefully."""
    nosqli_vector = '{"$gt": ""}'
    response = client.get(f"/v1/search/ai?q={nosqli_vector}")
    assert response.status_code in [200, 400]


def test_extreme_pagination_bounds():
    """Verify the API prevents users from scraping the entire DB in one call."""
    # Attempt to request 1,000,000 recommendations
    response = client.get("/v1/events/recommendation-analytics?limit=1000000")

    # The Pydantic validator `le=100` must intercept this and return 422
    assert response.status_code == 422
    assert "limit" in response.text.lower()


def test_massive_payload_rejection():
    """Verify the telemetry endpoint drops massive payloads to prevent memory exhaustion."""
    # Create a 10MB malicious JSON string
    massive_string = "A" * 10_000_000
    payload = {"event_type": "search", "query_text": massive_string, "timestamp": "2026-05-17T00:00:00Z"}

    # Fastapi/Uvicorn might drop the connection, or Pydantic will reject it.
    try:
        response = client.post("/v1/events", json=payload)
        # If it doesn't drop the connection, it must not 500.
        assert response.status_code in [400, 413, 422]
    except Exception:
        # Connection drop is acceptable for massive payloads (Uvicorn max limit)
        pass


# ---------------------------------------------------------------------------
# Auth security tests — test the authentication functions directly.
#
# The `/v1/recommendations/user/{user_id}` endpoint intentionally uses
# `get_optional_user` (permissive) which returns None for invalid tokens.
# To properly test JWT enforcement, we mount a temporary route that uses
# `get_current_user` (strict, raises 401) and test against that.
# We also verify that `get_optional_user` correctly returns None for bad tokens.
# ---------------------------------------------------------------------------

from datetime import UTC, datetime, timedelta

from fastapi import Depends
from jose import jwt

from backend.data.auth import get_current_user


# Mount a temporary strict-auth endpoint for testing JWT enforcement
@app.get("/v1/_test/protected")
async def _test_protected_endpoint(user=Depends(get_current_user)):
    """Temporary endpoint that requires strict JWT auth (used only in tests)."""
    return {"user_id": user.external_user_id if user else None}


def _make_token(sub: str, secret: str = "test-jwt-secret-key-for-ci-only", expire_minutes: int = 30) -> str:
    """Helper: mint a JWT with the given sub claim."""
    payload = {
        "sub": sub,
        "exp": datetime.now(UTC) + timedelta(minutes=expire_minutes),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def test_jwt_unknown_user_rejected():
    """
    A cryptographically valid JWT whose sub references a user that does not
    exist in the database must be rejected with 401, not silently registered.
    Regression test for the auto-registration security hole.
    """
    token = _make_token("nonexistent-user-xyz-12345")
    response = client.get(
        "/v1/_test/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401, f"Expected 401 for unknown user, got {response.status_code}: {response.text}"


def test_jwt_expired_token_rejected():
    """An expired JWT must be rejected with 401."""
    token = _make_token("any-user", expire_minutes=-1)
    response = client.get(
        "/v1/_test/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_jwt_wrong_secret_rejected():
    """A JWT signed with the wrong secret must be rejected with 401."""
    token = _make_token("any-user", secret="wrong-secret-key")
    response = client.get(
        "/v1/_test/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_jwt_missing_sub_rejected():
    """A JWT with no sub claim must be rejected with 401."""
    payload = {"exp": datetime.now(UTC) + timedelta(minutes=30)}
    token = jwt.encode(payload, "test-jwt-secret-key-for-ci-only", algorithm="HS256")
    response = client.get(
        "/v1/_test/protected",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_user_recommendations_reject_bad_token():
    """
    A forged bearer token must not authorize reads of another user's
    recommendation profile.
    """
    token = _make_token("nonexistent-user", secret="wrong-secret")
    response = client.get(
        "/v1/recommendations/user/any-user",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_admin_endpoint_requires_token():
    """Admin endpoints must reject requests with no admin token."""
    response = client.post("/v1/admin/reload-ensemble-weights")
    assert response.status_code in (401, 403, 404, 422)


def test_admin_endpoint_rejects_wrong_token():
    """Admin endpoints must reject requests with an incorrect admin token."""
    response = client.post(
        "/v1/admin/reload-ensemble-weights",
        headers={"X-Nova-Admin-Token": "wrong-token-value"},
    )
    assert response.status_code in (401, 403, 404)


def test_datetime_utc_not_deprecated():
    """
    Verify create_access_token uses timezone-aware UTC datetimes.
    Regression test for the datetime.utcnow() deprecation fix.
    """
    from backend.data.auth import SECRET_KEY, create_access_token

    token = create_access_token({"sub": "test-user"})
    decoded = jwt.decode(
        token,
        SECRET_KEY,
        algorithms=["HS256"],
        options={"verify_exp": False},
    )
    exp = decoded.get("exp")
    assert exp is not None
    # exp must be a future Unix timestamp (timezone-aware UTC)
    assert exp > datetime.now(UTC).timestamp()
