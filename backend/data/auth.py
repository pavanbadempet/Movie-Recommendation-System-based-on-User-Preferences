"""
Authentication and Authorization Module.

Handles JWT User Authentication for UI clients, and API Key verification
for B2B Multi-Tenant integrations. Ensures strong cryptographic security via PostgreSQL.
Replaces the legacy environment-variable based API keys.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hmac
import logging
from typing import Any

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from backend.data.database import APIKey, User, get_db

logger = logging.getLogger(__name__)

# Security Settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()
if not SECRET_KEY:
    logger.warning(
        "WARNING: JWT_SECRET_KEY env var is not set. Falling back to a default key for preview mode. Please configure JWT_SECRET_KEY in production settings!"
    )
    SECRET_KEY = "demo-fallback-secret-key-do-not-use-in-production-12345678"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

DEFAULT_TENANT_ID = os.getenv("NOVA_TENANT_ID", "demo-media-co")
DEFAULT_CATALOG_ID = os.getenv("NOVA_CATALOG_ID", "tmdb-movies")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")
api_key_header = APIKeyHeader(name="X-Nova-API-Key", auto_error=False)


@dataclass(frozen=True)
class TenantContext:
    """Resolved customer/catalog context for a request."""

    tenant_id: str
    catalog_id: str
    plan: str = "demo"
    authenticated: bool = False
    api_key_label: str | None = None


def _configured_static_api_keys() -> dict[str, TenantContext]:
    """Parse legacy static API keys from NOVA_API_KEYS.

    Format: key:tenant_id:catalog_id:plan,another-key:tenant:catalog:plan
    """
    entries: dict[str, TenantContext] = {}
    raw = os.getenv("NOVA_API_KEYS", "").strip()
    if not raw:
        return entries

    for item in raw.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 4 or not parts[0]:
            logger.warning("Ignoring malformed NOVA_API_KEYS entry.")
            continue
        key, tenant_id, catalog_id, plan = parts[:4]
        entries[key] = TenantContext(
            tenant_id=tenant_id,
            catalog_id=catalog_id,
            plan=plan,
            authenticated=True,
            api_key_label="static",
        )
    return entries


# -----------------------------------------------------------------------------
# PASSWORD CRYPTOGRAPHY
# -----------------------------------------------------------------------------


def verify_password(plain_password: str, hashed_password: str) -> bool:
    import bcrypt
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception:
            return False


def get_password_hash(password: str) -> str:
    import bcrypt
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


# -----------------------------------------------------------------------------
# JWT TOKENS (B2C / Web UI)
# -----------------------------------------------------------------------------


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Verifies the JWT and loads the user from PostgreSQL."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.external_user_id == user_id).first()
    if user is None:
        # Token is cryptographically valid but references a user that doesn't
        # exist in the database. Reject rather than silently create — silent
        # creation would allow any forged sub claim to gain access.
        logger.warning("JWT references unknown user_id=%s — rejecting", user_id)
        raise credentials_exception

    return user


_optional_oauth2 = OAuth2PasswordBearer(tokenUrl="/v1/auth/token", auto_error=False)


async def get_optional_user(
    token: str | None = Depends(_optional_oauth2),
    db: Session = Depends(get_db),
):
    """Like get_current_user but returns None instead of 401 when no token is present."""
    if token is None:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str | None = payload.get("sub")
        if user_id is None:
            return None
    except JWTError:
        return None
    return db.query(User).filter(User.external_user_id == user_id).first()


# -----------------------------------------------------------------------------
# B2B MULTI-TENANT API KEYS
# -----------------------------------------------------------------------------


def resolve_admin_token(
    x_nova_admin_token: str | None = Header(default=None, alias="X-Nova-Admin-Token"),
) -> None:
    expected_token = os.getenv("NOVA_ADMIN_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(status_code=404, detail="Admin operations are disabled")
    if not x_nova_admin_token or not hmac.compare_digest(expected_token, x_nova_admin_token):
        raise HTTPException(status_code=401, detail="Invalid X-Nova-Admin-Token")


def resolve_tenant_context(
    x_nova_api_key: str | None = Header(default=None, alias="X-Nova-API-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    x_catalog_id: str | None = Header(default=None, alias="X-Catalog-ID"),
    db: Session = Depends(get_db),
) -> TenantContext:
    """
    Resolve request tenant context directly from PostgreSQL `dim_api_key`
    using bcrypt validation.
    """
    static_keys = _configured_static_api_keys()
    if static_keys:
        if not x_nova_api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Nova-API-Key")

        for expected_key, static_context in static_keys.items():
            if hmac.compare_digest(expected_key, x_nova_api_key):
                return TenantContext(
                    tenant_id=static_context.tenant_id,
                    catalog_id=x_catalog_id or static_context.catalog_id,
                    plan=static_context.plan,
                    authenticated=True,
                    api_key_label=static_context.api_key_label,
                )

        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-Nova-API-Key")

    if not x_nova_api_key:
        return TenantContext(
            tenant_id=x_tenant_id or DEFAULT_TENANT_ID,
            catalog_id=x_catalog_id or DEFAULT_CATALOG_ID,
            plan="demo",
            authenticated=False,
        )

    prefix = x_nova_api_key[:10]
    potential_keys = db.query(APIKey).filter(APIKey.key_prefix == prefix, ~APIKey.is_revoked).all()

    is_valid = False
    active_tenant = None

    for key_record in potential_keys:
        if verify_password(x_nova_api_key, key_record.api_key_hash):
            is_valid = True
            active_tenant = key_record.tenant
            break

    if not is_valid or not active_tenant:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-Nova-API-Key")

    if not active_tenant.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant account suspended.")

    return TenantContext(
        tenant_id=str(active_tenant.tenant_id),
        catalog_id=x_catalog_id or DEFAULT_CATALOG_ID,
        plan=active_tenant.plan_tier,
        authenticated=True,
        api_key_label=active_tenant.company_name,
    )


def require_authenticated_tenant_context(
    context: TenantContext,
    operation: str = "this operation",
) -> None:
    """Require a tenant context proven by an API key.

    `resolve_tenant_context` intentionally supports public demo reads by
    returning an unauthenticated default/header-selected context when no API
    key is supplied. Tenant-scoped writes and billing actions must call this
    helper before using that context for protected state.
    """
    if not context.authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Valid X-Nova-API-Key required for {operation}.",
        )


def enforce_payload_context(
    payload: Any,
    context: TenantContext,
) -> None:
    require_authenticated_tenant_context(context, "event ingestion")
    payload_tenant = getattr(payload, "tenant_id", None)
    payload_catalog = getattr(payload, "catalog_id", None)
    if payload_tenant and payload_tenant != context.tenant_id:
        raise HTTPException(status_code=403, detail="tenant_id does not match API key context")
    if payload_catalog and payload_catalog != context.catalog_id:
        raise HTTPException(status_code=403, detail="catalog_id does not match API key context")
