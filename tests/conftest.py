"""
Configurare pytest — fixtures comune pentru toate testele.
Folosește o bază de date SQLite izolată, în memorie, per sesiune de test.
"""

import asyncio
import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Suprascrie setările ÎNAINTE de a importa modulele din backend
os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough-for-hs256-validation"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"  # în memorie

from backend import database as db
from backend.main import app


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Creează un engine de DB nou, în memorie, pentru sesiunea de test."""
    _engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with _engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)
    yield _engine
    await _engine.dispose()


@pytest_asyncio.fixture()
async def session(engine):
    """Oferă o sesiune de DB tranzacțională, cu rollback după fiecare test."""
    _session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with _session_maker() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture()
async def client(engine):
    """Client HTTP async de test, cu sesiunea de DB suprascrisă prin dependență."""
    _session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db():
        async with _session_maker() as session:
            yield session

    app.dependency_overrides[db.get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
