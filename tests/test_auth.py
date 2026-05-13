"""
Tests for authentication flow: register, login, password policy.
"""

import pytest
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


# ── Login ───────────────────────────────────────────────────────────────────

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
async def test_login_admin_gets_direct_token(client: AsyncClient):
    """Admin login should return JWT directly."""
    resp = await client.post("/api/auth/login", data={
        "username": "admin_test",
        "password": "TestPass1!",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_regular_user_gets_direct_token(client: AsyncClient):
    """Regular user login should also return JWT directly."""
    resp = await client.post("/api/auth/login", data={
        "username": "user_test",
        "password": "TestPass1!",
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
async def test_me_with_valid_token(client: AsyncClient):
    """Admin gets direct token → access /me."""
    # Admin login returns JWT directly
    resp = await client.post("/api/auth/login", data={
        "username": "admin_test",
        "password": "TestPass1!",
    })
    token = resp.json()["access_token"]

    # Access /me
    resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "admin_test"
    assert resp.json()["role"] == "admin"
