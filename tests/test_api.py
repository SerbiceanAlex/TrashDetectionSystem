"""
Tests for the current B2B monitoring API surface.

These tests intentionally avoid ML inference and focus on DB/API behavior:
app shell, dashboard, locations, reports, video sessions, and incident listing.
"""

import pytest
from httpx import AsyncClient
from types import SimpleNamespace
from uuid import uuid4

from backend import auth
from backend import database as db
from backend.auth_router import get_current_active_user
from backend.main import ANNOTATED_DIR, LITTERING_DIR, VIDEOS_DIR, app


def _override_admin() -> None:
    """Use a synthetic admin user without creating DB rows."""
    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=1,
        username="api_admin",
        email="api_admin@test.local",
        role="admin",
        organization_id=1,
    )


def _override_operator() -> None:
    """Use a synthetic operator user without creating DB rows."""
    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=2,
        username="api_operator",
        email="api_operator@test.local",
        role="user",
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
async def test_system_info_is_public_and_uses_production_model(client: AsyncClient):
    resp = await client.get("/api/system/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models"]["detector"]["weights"] == "models/detector/production/best.pt"
    assert data["models"]["detector"]["name"]
    assert "mAP50" in data["models"]["detector"]["metrics"]
    assert data["models"]["classifier"]["weights"] == "models/classify/B2/best.pt"
    assert data["runtime"]["littering_file_retention_days"] >= 1


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
async def test_littering_media_is_authenticated_and_owner_scoped(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Audit org", plan="pro")
    admin = db.User(username="media_admin", email="media_admin@test.local", hashed_password="x", role="admin", organization=org)
    owner = db.User(username="media_owner", email="media_owner@test.local", hashed_password="x", role="user", organization=org)
    outsider = db.User(username="media_outsider", email="media_outsider@test.local", hashed_password="x", role="user", organization=org)
    session.add_all([org, admin, owner, outsider])
    await session.flush()

    suffix = uuid4().hex
    thumb_name = f"pytest_{suffix}_thumb.jpg"
    clip_name = f"pytest_{suffix}.mp4"
    thumb_path = LITTERING_DIR / thumb_name
    clip_path = LITTERING_DIR / clip_name
    thumb_path.write_bytes(b"\xff\xd8\xff\xd9")
    clip_path.write_bytes(b"pytest clip")

    event = db.LitteringEvent(
        material="plastic",
        det_score=0.91,
        status="pending",
        reporter_id=owner.id,
        organization_id=org.id,
        thumbnail_path=thumb_name,
        clip_path=clip_name,
    )
    session.add(event)
    await session.flush()

    admin_token = auth.create_access_token({"username": admin.username, "role": admin.role, "id": admin.id})
    owner_token = auth.create_access_token({"username": owner.username, "role": owner.role, "id": owner.id})
    outsider_token = auth.create_access_token({"username": outsider.username, "role": outsider.role, "id": outsider.id})

    try:
        direct = await client.get(f"/littering/{thumb_name}")
        assert direct.status_code == 404
        direct_video = await client.get(f"/videos/{clip_name}")
        assert direct_video.status_code == 404

        no_auth = await client.get(f"/api/littering/events/{event.id}/thumbnail")
        assert no_auth.status_code == 401

        admin_resp = await client.get(
            f"/api/littering/events/{event.id}/thumbnail",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert admin_resp.status_code == 200
        assert "image/jpeg" in admin_resp.headers.get("content-type", "")

        owner_resp = await client.get(f"/api/littering/events/{event.id}/clip?token={owner_token}")
        assert owner_resp.status_code == 200
        assert "video/mp4" in owner_resp.headers.get("content-type", "")

        outsider_resp = await client.get(f"/api/littering/events/{event.id}/thumbnail?token={outsider_token}")
        assert outsider_resp.status_code == 403
    finally:
        thumb_path.unlink(missing_ok=True)
        clip_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_admin_storage_breaks_littering_evidence_down_by_status(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Storage audit org", plan="pro")
    session.add(org)
    await session.flush()
    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=1,
        username="storage_admin",
        email="storage_admin@test.local",
        role="admin",
        organization_id=org.id,
    )

    suffix = uuid4().hex
    pending_clip_name = f"pytest_{suffix}_pending.mp4"
    pending_thumb_name = f"pytest_{suffix}_pending.jpg"
    reviewed_clip_name = f"pytest_{suffix}_reviewed.mp4"
    pending_clip = LITTERING_DIR / pending_clip_name
    pending_thumb = LITTERING_DIR / pending_thumb_name
    reviewed_clip = LITTERING_DIR / reviewed_clip_name
    pending_clip_bytes = b"pending evidence clip"
    pending_thumb_bytes = b"pending evidence thumbnail"
    reviewed_clip_bytes = b"reviewed evidence clip"
    pending_clip.write_bytes(pending_clip_bytes)
    pending_thumb.write_bytes(pending_thumb_bytes)
    reviewed_clip.write_bytes(reviewed_clip_bytes)

    pending_bytes = len(pending_clip_bytes) + len(pending_thumb_bytes)
    reviewed_bytes = len(reviewed_clip_bytes)
    session.add_all(
        [
            db.LitteringEvent(
                material="plastic",
                status="pending",
                organization_id=org.id,
                clip_path=pending_clip_name,
                thumbnail_path=pending_thumb_name,
            ),
            db.LitteringEvent(
                material="metal",
                status="reviewed",
                organization_id=org.id,
                clip_path=reviewed_clip_name,
            ),
        ]
    )
    await session.flush()

    try:
        resp = await client.get("/api/admin/storage")
        assert resp.status_code == 200
        data = resp.json()

        assert data["evidence_by_status"]["pending"]["bytes"] == pending_bytes
        assert data["evidence_by_status"]["pending"]["files"] == 2
        assert data["evidence_by_status"]["reviewed"]["bytes"] == reviewed_bytes
        assert data["evidence_by_status"]["reviewed"]["files"] == 1
        assert data["evidence_tracked_bytes"] == pending_bytes + reviewed_bytes
        assert data["evidence_tracked_files"] == 3
        assert data["evidence_bytes"] >= data["evidence_tracked_bytes"]
    finally:
        pending_clip.unlink(missing_ok=True)
        pending_thumb.unlink(missing_ok=True)
        reviewed_clip.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_generated_image_media_is_authenticated(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Detection media org", plan="pro")
    admin = db.User(username="detect_admin", email="detect_admin@test.local", hashed_password="x", role="admin", organization=org)
    owner = db.User(username="detect_owner", email="detect_owner@test.local", hashed_password="x", role="user", organization=org)
    outsider = db.User(username="detect_outsider", email="detect_outsider@test.local", hashed_password="x", role="user", organization=org)
    session.add_all([org, admin, owner, outsider])
    await session.flush()

    suffix = uuid4().hex
    ann_path = ANNOTATED_DIR / f"pytest_{suffix}_annotated.jpg"
    # annotated/ se creează acum lazy (la prima scanare reală); testul își
    # pregătește singur folderul fixture.
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    ann_path.write_bytes(b"\xff\xd8\xff\xd9")
    det_session = db.DetectionSession(
        filename="pytest.jpg",
        annotated_path=str(ann_path),
        total_objects=1,
        inference_ms=12.5,
        organization_id=org.id,
        reporter_id=owner.id,
    )
    session.add(det_session)
    await session.flush()

    admin_token = auth.create_access_token({"username": admin.username, "role": admin.role, "id": admin.id})
    owner_token = auth.create_access_token({"username": owner.username, "role": owner.role, "id": owner.id})
    outsider_token = auth.create_access_token({"username": outsider.username, "role": outsider.role, "id": outsider.id})

    try:
        direct = await client.get(f"/annotated/{ann_path.name}")
        assert direct.status_code == 404

        no_auth = await client.get(f"/api/detect/sessions/{det_session.id}/annotated")
        assert no_auth.status_code == 401

        owner_resp = await client.get(f"/api/detect/sessions/{det_session.id}/annotated?token={owner_token}")
        assert owner_resp.status_code == 200
        assert "image/jpeg" in owner_resp.headers.get("content-type", "")

        admin_resp = await client.get(
            f"/api/detect/sessions/{det_session.id}/annotated",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert admin_resp.status_code == 200

        outsider_resp = await client.get(f"/api/detect/sessions/{det_session.id}/annotated?token={outsider_token}")
        assert outsider_resp.status_code == 403
    finally:
        ann_path.unlink(missing_ok=True)


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
async def test_location_mutations_are_admin_only(client: AsyncClient):
    _override_operator()

    create_resp = await client.post("/api/locations", json={"name": "Operator location"})
    assert create_resp.status_code == 403

    update_resp = await client.patch("/api/locations/1", json={"name": "Renamed"})
    assert update_resp.status_code == 403

    delete_resp = await client.delete("/api/locations/1")
    assert delete_resp.status_code == 403

    rtsp_resp = await client.post("/api/locations/test-rtsp", json={"rtsp_url": "rtsp://127.0.0.1/test"})
    assert rtsp_resp.status_code == 403


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
async def test_video_upload_requires_auth(client: AsyncClient):
    resp = await client.post(
        "/api/video/upload",
        files={"file": ("clip.mp4", b"not-a-real-video", "video/mp4")},
    )
    assert resp.status_code in (401, 403)


@pytest.mark.asyncio
async def test_video_upload_rejects_files_over_video_limit(client: AsyncClient, monkeypatch):
    _override_admin()
    monkeypatch.setattr("backend.main.VIDEO_MAX_UPLOAD_BYTES", 4)
    resp = await client.post(
        "/api/video/upload",
        files={"file": ("clip.mp4", b"12345", "video/mp4")},
    )
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_detect_no_file(client: AsyncClient):
    resp = await client.post("/api/detect")
    assert resp.status_code in (401, 422)
