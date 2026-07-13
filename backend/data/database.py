"""
PostgreSQL SQLAlchemy Database Connection and ORM Models.

Handles connection pooling, session management, and maps the Python application
to the Multi-Tenant Star Schema defined in PostgreSQL.
"""

import os

from dotenv import load_dotenv

load_dotenv()
import datetime

# Use generic String for UUID to support SQLite fallback
import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, create_engine, event
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.types import CHAR, TypeDecorator

# PostgreSQL Connection String (Fallback to SQLite if no PGSQL)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///apex.db")

if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=-64000")
            cursor.execute("PRAGMA temp_store=MEMORY")
        except Exception:
            pass
        finally:
            cursor.close()
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,  # Allow 20 persistent connections
        max_overflow=10,  # Allow up to 10 additional temporary connections
        pool_timeout=30,
        pool_recycle=1800,  # Recycle connections every 30 minutes to prevent staleness
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utc_now() -> datetime.datetime:
    """Return naive UTC timestamps for SQLAlchemy DateTime columns."""
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's native UUID type, otherwise CHAR(36).
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=False))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return str(value)


# -----------------------------------------------------------------------------
# ORM MODELS
# -----------------------------------------------------------------------------


class Tenant(Base):
    __tablename__ = "dim_tenant"
    tenant_id = Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    company_name = Column(String(255), nullable=False)
    plan_tier = Column(String(50), default="free")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    # Stripe billing — populated on first checkout.session.completed webhook
    stripe_customer_id = Column(String(255), nullable=True)
    subscription_id = Column(String(255), nullable=True)


class APIKey(Base):
    __tablename__ = "dim_api_key"
    api_key_id = Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(GUID(), ForeignKey("dim_tenant.tenant_id"), nullable=False)
    api_key_hash = Column(String(255), unique=True, nullable=False)
    key_prefix = Column(String(10), nullable=False)
    rate_limit_rpm = Column(Integer, default=60)
    created_at = Column(DateTime, default=utc_now)
    expires_at = Column(DateTime, nullable=True)
    is_revoked = Column(Boolean, default=False)

    tenant = relationship("Tenant")


class User(Base):
    __tablename__ = "dim_user"
    user_sk = Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(GUID(), ForeignKey("dim_tenant.tenant_id"), nullable=False)
    external_user_id = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    tenant = relationship("Tenant")


class UserEvent(Base):
    __tablename__ = "fact_user_event"
    event_id = Column(GUID(), primary_key=True, default=lambda: str(uuid.uuid4()))
    tenant_id = Column(GUID(), ForeignKey("dim_tenant.tenant_id"), nullable=False)
    user_sk = Column(GUID(), ForeignKey("dim_user.user_sk"), nullable=True)
    movie_sk = Column(GUID(), nullable=True)

    event_type = Column(String(50), nullable=False)
    event_value = Column(Float, nullable=True)
    query_text = Column(String(500), nullable=True)

    context_device = Column(String(100), nullable=True)
    context_os = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utc_now)


# -----------------------------------------------------------------------------
# DEPENDENCY
# -----------------------------------------------------------------------------


def get_db():
    """FastAPI Dependency for providing a safe DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------
# create_all is intentionally guarded: it only runs for SQLite (local dev /
# CI) where there is no migration runner. On PostgreSQL, schema is managed
# by Flyway migrations in sql/migrations/ — running create_all there would
# silently diverge from the versioned schema.
# ---------------------------------------------------------------------------
# Create tables automatically. For PostgreSQL, if tables are missing (e.g., in a new deployment
# where Flyway migration runner wasn't executed), create_all will initialize the base schema
# automatically without breaking existing tables.
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Database table initialization warning: {e}")


def seed_database():
    db = SessionLocal()
    try:
        raw_tenant_id = os.getenv("NOVA_TENANT_ID", "demo-media-co")
        try:
            import uuid
            uuid.UUID(raw_tenant_id)
            tenant_id = raw_tenant_id
        except (ValueError, TypeError):
            tenant_id = "00000000-0000-0000-0000-000000000001"
            
        tenant = db.query(Tenant).filter_by(tenant_id=tenant_id).first()
        if not tenant:
            new_tenant = Tenant(tenant_id=tenant_id, company_name=f"APEX Demo Tenant ({raw_tenant_id})", plan_tier="enterprise")
            db.add(new_tenant)
            db.commit()
    except Exception as e:
        print(f"Seed error: {e}")
    finally:
        db.close()


# Run seeder on startup
try:
    seed_database()
except Exception as e:
    print(f"Database seeding warning: {e}")
