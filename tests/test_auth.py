"""
Teste pentru fluxul de autentificare: înregistrare, login, politica de parolă.
"""

import pytest
from httpx import AsyncClient


# ── Înregistrare ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_first_user_is_admin(client: AsyncClient):
    """Primul utilizator înregistrat trebuie să primească rolul admin."""
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
    """Următorii utilizatori trebuie să primească rolul 'user'."""
    resp = await client.post("/api/auth/register", json={
        "username": "user_test",
        "email": "user@test.local",
        "password": "TestPass1!",
    })
    assert resp.status_code == 200
    assert resp.json()["role"] == "user"


@pytest.mark.asyncio
async def test_register_duplicate_username(client: AsyncClient):
    """Înregistrarea cu un username existent trebuie să eșueze."""
    resp = await client.post("/api/auth/register", json={
        "username": "admin_test",
        "email": "other@test.local",
        "password": "TestPass1!",
    })
    assert resp.status_code == 400
    assert "utilizator" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient):
    """Înregistrarea cu un email existent trebuie să eșueze."""
    resp = await client.post("/api/auth/register", json={
        "username": "another_user",
        "email": "admin@test.local",
        "password": "TestPass1!",
    })
    assert resp.status_code == 400
    assert "email" in resp.json()["detail"].lower()


# ── Politica de parolă ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_register_weak_password_rejected(client: AsyncClient):
    """Parolele slabe trebuie respinse cu 422."""
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


# ── Login ─────────────────────────────────────────────────────────────────────

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
    """Login-ul de admin trebuie să întoarcă direct un JWT."""
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
    """Login-ul unui user normal trebuie să întoarcă tot direct un JWT."""
    resp = await client.post("/api/auth/login", data={
        "username": "user_test",
        "password": "TestPass1!",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


# ── Endpoint protejat ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_me_without_token(client: AsyncClient):
    resp = await client.get("/api/auth/me")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_me_with_valid_token(client: AsyncClient):
    """Adminul primește token direct → accesează /me."""
    # Login-ul de admin întoarce direct JWT-ul
    resp = await client.post("/api/auth/login", data={
        "username": "admin_test",
        "password": "TestPass1!",
    })
    token = resp.json()["access_token"]

    # Accesează /me
    resp = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.json()["username"] == "admin_test"
    assert resp.json()["role"] == "admin"
