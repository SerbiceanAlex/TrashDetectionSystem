"""
Tests for the current B2B monitoring API surface.

These tests intentionally avoid ML inference and focus on DB/API behavior:
app shell, dashboard, locations, reports, video sessions, and incident listing.
"""

import pytest
from httpx import AsyncClient
from types import SimpleNamespace

from backend.auth_router import get_current_active_user
from backend.main import app


def _override_admin() -> None:
    """Use a synthetic admin user without creating DB rows."""
    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=1,
        username="api_admin",
        email="api_admin@test.local",
        role="admin",
        organization_id=1,
    )


@pytest.mark.asyncio
async def test_root_redirects_to_app(client: AsyncClient):
    resp = await client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/app"


@pytest.mark.asyncio
async def test_app_shell_returns_html(client: AsyncClient):
    resp = await client.get("/app")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_littering_events_list_empty_for_admin(client: AsyncClient):
    _override_admin()
    resp = await client.get("/api/littering/events")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_littering_event_detail_not_found_for_admin(client: AsyncClient):
    _override_admin()
    resp = await client.get("/api/littering/events/99999")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_dashboard_b2b_requires_auth(client: AsyncClient):
    resp = await client.get("/api/dashboard/b2b")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_dashboard_b2b_empty_for_admin(client: AsyncClient):
    _override_admin()
    resp = await client.get("/api/dashboard/b2b")
    assert resp.status_code == 200
    data = resp.json()
    assert data["incidents_today"] == 0
    assert data["pending_review"] == 0
    assert isinstance(data["trend_30d"], list)


@pytest.mark.asyncio
async def test_locations_list_empty_for_admin(client: AsyncClient):
    _override_admin()
    resp = await client.get("/api/locations")
    assert resp.status_code == 200
    assert resp.json() == {"locations": []}


@pytest.mark.asyncio
async def test_reports_stats_empty_for_admin(client: AsyncClient):
    _override_admin()
    resp = await client.get("/api/reports/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_incidents"] == 0
    assert data["pending"] == 0
    assert isinstance(data["hourly_distribution"], list)


@pytest.mark.asyncio
async def test_reports_export_csv_requires_auth(client: AsyncClient):
    resp = await client.get("/api/reports/export?format=csv")
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_reports_export_csv_for_admin(client: AsyncClient):
    _override_admin()
    resp = await client.get("/api/reports/export?format=csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_video_sessions_list_for_admin(client: AsyncClient):
    _override_admin()
    resp = await client.get("/api/video/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_detect_no_file(client: AsyncClient):
    resp = await client.post("/api/detect")
    assert resp.status_code in (401, 422)
