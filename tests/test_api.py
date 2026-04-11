"""
Tests for core API endpoints: sessions, stats, export.
These tests don't require ML models loaded (they test DB-driven endpoints).
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_homepage_returns_html(client: AsyncClient):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_sessions_list_empty(client: AsyncClient):
    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_sessions_detail_not_found(client: AsyncClient):
    resp = await client.get("/api/sessions/99999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stats_endpoint(client: AsyncClient):
    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_sessions" in data or "timeline" in data


@pytest.mark.asyncio
async def test_leaderboard(client: AsyncClient):
    resp = await client.get("/api/leaderboard")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_map_reports(client: AsyncClient):
    resp = await client.get("/api/map/reports")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_zones(client: AsyncClient):
    resp = await client.get("/api/zones")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_export_csv_no_auth(client: AsyncClient):
    """CSV export should work without auth (public endpoint)."""
    resp = await client.get("/api/export/csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_video_sessions_list(client: AsyncClient):
    resp = await client.get("/api/video/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data


@pytest.mark.asyncio
async def test_detect_no_file(client: AsyncClient):
    """POST /api/detect without auth should fail with 401."""
    resp = await client.post("/api/detect")
    assert resp.status_code in (401, 422)
