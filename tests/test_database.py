"""
Unit tests for the PostgreSQL database connection, ORM models, and Auth.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.data.auth import get_password_hash, verify_password
from backend.data.database import APIKey, Base, Tenant, User, UserEvent

# Use an in-memory SQLite database for fast unit testing without needing the Docker container
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_tenant_creation(db):
    """Test that a SaaS tenant can be created successfully."""
    tenant = Tenant(company_name="Netflix Clone Corp", plan_tier="enterprise")
    db.add(tenant)
    db.commit()

    saved_tenant = db.query(Tenant).filter_by(company_name="Netflix Clone Corp").first()
    assert saved_tenant is not None
    assert saved_tenant.plan_tier == "enterprise"
    assert saved_tenant.is_active is True


def test_api_key_hashing_and_verification(db):
    """Test the bcrypt cryptographic hash verification for Multi-Tenant API Keys."""
    tenant = Tenant(company_name="Hulu Clone Corp", plan_tier="pro")
    db.add(tenant)
    db.commit()

    raw_api_key = "nova_live_9a8b7c6d5e4f3g2h"
    hashed_key = get_password_hash(raw_api_key)

    api_key_record = APIKey(
        tenant_id=tenant.tenant_id, api_key_hash=hashed_key, key_prefix=raw_api_key[:10], rate_limit_rpm=100
    )
    db.add(api_key_record)
    db.commit()

    # Simulate an incoming request
    db_key = db.query(APIKey).filter_by(key_prefix="nova_live_").first()
    assert db_key is not None
    assert verify_password(raw_api_key, db_key.api_key_hash) is True
    assert verify_password("nova_live_WRONG_KEY", db_key.api_key_hash) is False


def test_user_event_persistence(db):
    """Test that analytical events are durably saved to the Star Schema."""
    tenant = Tenant(company_name="HBO Clone Corp", plan_tier="free")
    db.add(tenant)
    db.commit()

    # Create user
    user = User(tenant_id=tenant.tenant_id, external_user_id="user_12345")
    db.add(user)
    db.commit()

    # Record an event
    event = UserEvent(
        tenant_id=tenant.tenant_id, user_sk=user.user_sk, event_type="rating", event_value=4.5, context_device="iOS"
    )
    db.add(event)
    db.commit()

    # Query back
    saved_event = db.query(UserEvent).filter_by(event_type="rating").first()
    assert saved_event is not None
    assert saved_event.event_value == 4.5
    assert saved_event.context_device == "iOS"
    assert saved_event.user_sk == user.user_sk


def test_transaction_rollback(db):
    """Test that failed transactions are safely rolled back to prevent corruption."""
    tenant = Tenant(company_name="Rollback Corp")
    db.add(tenant)
    db.commit()

    try:
        # Intentionally violate NOT NULL constraint (external_user_id is missing)
        invalid_user = User(tenant_id=tenant.tenant_id)
        db.add(invalid_user)
        db.commit()
    except Exception:
        db.rollback()

    users = db.query(User).all()
    assert len(users) == 0  # Transaction successfully rolled back
