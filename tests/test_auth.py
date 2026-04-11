"""
Tests for authentication flow: register, login (OTP), password policy.
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient


# ── Registration ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_first_user_is_admin(client: AsyncClient):
    """First registered user should get admin role."""
    resp = await client.post("/api/auth/register", json={
        "username": "admin_test",
        "email": "admin@test.local",
        "password": "TestPass1!",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["username"] == "admin_test"
    assert data["role"] == "admin"


@pytest.mark.asyncio
async def test_register_second_user_is_regular(client: AsyncClient):
    """Subsequent users should get 'user' role."""
    resp = await client.post("/api/auth/register", json={
        "username": "user_test",
        "email": "user@test.local",
        "password": "TestPass1!",
    })
    assert resp.status_code == 200
    assert resp.json()["role"] == "user"


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient):
    """Registering with an existing username should fail."""
    resp = await client.post("/api/auth/register", json={
        "username": "admin_test",
        "email": "other@test.local",
        "password": "TestPass1!",
    })
    assert resp.status_code == 400
    assert "already registered" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    """Registering with an existing email should fail."""
    resp = await client.post("/api/auth/register", json={
        "username": "another_user",
        "email": "admin@test.local",
        "password": "TestPass1!",
    })
    assert resp.status_code == 400
    assert "already registered" in resp.json()["detail"].lower()


# ── Password policy ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_weak_password_rejected(client: AsyncClient):
    """Weak passwords should be rejected with 422."""
    resp = await client.post("/api/auth/register", json={
        "username": "weak_user",
        "email": "weak@test.local",
        "password": "short",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_register_no_special_char(client: AsyncClient):
    resp = await client.post("/api/auth/register", json={
        "username": "nospecial",
        "email": "nospecial@test.local",
        "password": "TestPass1",
    })
    assert resp.status_code == 422
    assert "special" in resp.json()["detail"].lower() or "caracter" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_password_rules_endpoint(client: AsyncClient):
    resp = await client.get("/api/auth/password-rules")
    assert resp.status_code == 200
    data = resp.json()
    assert data["min_length"] == 8
    assert len(data["rules"]) >= 4


# ── Login (OTP flow) ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient):
    resp = await client.post("/api/auth/login", data={
        "username": "admin_test",
        "password": "WrongPass1!",
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user(client: AsyncClient):
    resp = await client.post("/api/auth/login", data={
        "username": "ghost_user",
        "password": "TestPass1!",
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_correct_returns_otp_required(client: AsyncClient):
    """Correct password should trigger OTP step, not return token directly."""
    resp = await client.post("/api/auth/login", data={
        "username": "admin_test",
        "password": "TestPass1!",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["otp_required"] is True
    assert "email_hint" in data
    assert "***" in data["email_hint"]


# ── OTP verification ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_otp_wrong_code(client: AsyncClient):
    resp = await client.post("/api/auth/verify-otp", json={
        "username": "admin_test",
        "code": "000000",
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_verify_otp_correct_code(client: AsyncClient, session):
    """Full flow: login → get OTP from DB → verify → get JWT."""
    from backend.database import OTPCode
    from sqlalchemy import select

    # Step 1: Login to trigger OTP generation
    resp = await client.post("/api/auth/login", data={
        "username": "admin_test",
        "password": "TestPass1!",
    })
    assert resp.status_code == 200

    # Step 2: Read OTP from database directly (dev shortcut)
    result = await session.execute(
        select(OTPCode).where(OTPCode.is_used == 0).order_by(OTPCode.created_at.desc())
    )
    otp = result.scalar_one()
    code = otp.code

    # Step 3: Verify OTP → get JWT
    resp = await client.post("/api/auth/verify-otp", json={
        "username": "admin_test",
        "code": code,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


# ── Protected endpoint ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_me_without_token(client: AsyncClient):
    resp = await client.get("/api/auth/me")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_me_with_valid_token(client: AsyncClient, session):
    """Full auth flow → access /me."""
    from backend.database import OTPCode
    from sqlalchemy import select

    # Login
    await client.post("/api/auth/login", data={
        "username": "admin_test",
        "password": "TestPass1!",
    })

    # Get OTP
    result = await session.execute(
        select(OTPCode).where(OTPCode.is_used == 0).order_by(OTPCode.created_at.desc())
    )
    otp = result.scalar_one()

    # Verify OTP → get token
    resp = await client.post("/api/auth/verify-otp", json={
        "username": "admin_test",
        "code": otp.code,
    })
    token = resp.json()["access_token"]

    # Access /me
    resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "admin_test"
    assert resp.json()["role"] == "admin"
