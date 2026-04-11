"""
Pytest configuration — shared fixtures for all tests.
Uses an isolated in-memory SQLite database per test session.
"""

import asyncio
import os
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Override settings BEFORE importing backend modules
os.environ["SECRET_KEY"] = "test-secret-key-that-is-long-enough-for-hs256-validation"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite://"  # in-memory

from backend import database as db
from backend.main import app


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def engine():
    """Create a fresh in-memory database engine for the test session."""
    _engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with _engine.begin() as conn:
        await conn.run_sync(db.Base.metadata.create_all)
    yield _engine
    await _engine.dispose()


@pytest_asyncio.fixture()
async def session(engine):
    """Provide a transactional database session that rolls back after each test."""
    _session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with _session_maker() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture()
async def client(engine):
    """Async HTTP test client with dependency-overridden DB session."""
    _session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_get_db():
        async with _session_maker() as session:
            yield session

    app.dependency_overrides[db.get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
