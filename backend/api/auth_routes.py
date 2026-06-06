"""
Authentication routes for B2C web UI flows.
"""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.data.auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_password_hash,
    verify_password,
)
from backend.data.database import Tenant, User, get_db

router = APIRouter(tags=["Authentication"])


class RegisterRequest(BaseModel):
    username: str
    password: str


@router.post("/v1/auth/register")
def register_user(req: RegisterRequest, db: Session = Depends(get_db)):
    username = req.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    tenant = db.query(Tenant).filter_by(company_name="B2C Web App").first()
    if not tenant:
        tenant = Tenant(company_name="B2C Web App", plan_tier="free")
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

    existing = db.query(User).filter_by(external_user_id=username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")

    user = User(
        tenant_id=tenant.tenant_id,
        external_user_id=username,
        password_hash=get_password_hash(req.password),
    )
    db.add(user)
    db.commit()
    return {"msg": "User created successfully"}


@router.post("/v1/auth/token")
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter_by(external_user_id=form_data.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.password_hash or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.external_user_id}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}
