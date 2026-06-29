"""
Teste pentru suprafața API curentă de monitorizare B2B.

Aceste teste evită intenționat inferența ML și se concentrează pe comportamentul DB/API:
shell-ul aplicației, dashboard, export rapoarte, sesiuni video și listarea incidentelor.
"""

import pytest
from datetime import datetime, timedelta, timezone
from httpx import AsyncClient
from types import SimpleNamespace
from uuid import uuid4

from backend import auth
from backend import database as db
from backend.auth_router import get_current_active_user
from backend.main import LITTERING_DIR, app


def _override_admin() -> None:
    """Folosește un admin sintetic, fără a crea rânduri în DB."""
    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=1,
        username="api_admin",
        email="api_admin@test.local",
        role="admin",
        organization_id=1,
    )


def _override_operator() -> None:
    """Folosește un operator sintetic, fără a crea rânduri în DB."""
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
async def test_delete_littering_event_is_idempotent_for_admin(client: AsyncClient):
    _override_admin()
    resp = await client.delete("/api/littering/events/99999")
    assert resp.status_code == 200
    assert "deja șters" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_littering_media_is_authenticated_and_owner_scoped(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Audit org")
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

    org = db.Organization(name="Storage audit org")
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
async def test_admin_users_include_incident_counters(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Users stats org")
    created_at = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    latest_incident_at = datetime(2026, 1, 3, 12, 30, tzinfo=timezone.utc)
    admin = db.User(username="users_stats_admin", email="users_stats_admin@test.local", hashed_password="x", role="admin", organization=org, created_at=created_at)
    reporter = db.User(username="users_stats_reporter", email="users_stats_reporter@test.local", hashed_password="x", role="user", organization=org, created_at=created_at)
    idle = db.User(username="users_stats_idle", email="users_stats_idle@test.local", hashed_password="x", role="user", organization=org, created_at=created_at)
    session.add_all([org, admin, reporter, idle])
    await session.flush()

    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=admin.id,
        username=admin.username,
        email=admin.email,
        role="admin",
        organization_id=org.id,
    )

    session.add_all(
        [
            db.LitteringEvent(status="pending", reporter_id=reporter.id, organization_id=org.id, detected_at=latest_incident_at),
            db.LitteringEvent(status="reviewed", reporter_id=reporter.id, organization_id=org.id, detected_at=latest_incident_at - timedelta(hours=1)),
            db.LitteringEvent(status="dismissed", reporter_id=reporter.id, organization_id=org.id, detected_at=latest_incident_at - timedelta(days=1)),
        ]
    )
    await session.flush()

    resp = await client.get("/api/admin/users")
    assert resp.status_code == 200
    users = {item["username"]: item for item in resp.json()}

    assert users["users_stats_reporter"]["total_reports"] == 3
    assert users["users_stats_reporter"]["pending_reports"] == 1
    assert users["users_stats_reporter"]["confirmed_reports"] == 1
    assert users["users_stats_reporter"]["last_activity_at"].startswith("2026-01-03T12:30")

    assert users["users_stats_idle"]["total_reports"] == 0
    assert users["users_stats_idle"]["pending_reports"] == 0
    assert users["users_stats_idle"]["confirmed_reports"] == 0
    assert users["users_stats_idle"]["last_activity_at"].startswith("2026-01-01T09:00")


@pytest.mark.asyncio
async def test_admin_can_edit_user_profile_and_demote_other_admin(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Users edit org")
    admin = db.User(username="users_edit_admin", email="users_edit_admin@test.local", hashed_password="x", role="admin", organization=org)
    member = db.User(username="users_edit_member", email="users_edit_member@test.local", hashed_password="x", role="user", organization=org)
    session.add_all([org, admin, member])
    await session.flush()

    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=admin.id,
        username=admin.username,
        email=admin.email,
        role="admin",
        organization_id=org.id,
    )

    other_admin = None
    try:
        resp = await client.patch(
            f"/api/admin/users/{member.id}",
            json={"username": "users_edit_member_renamed", "email": "renamed_member@test.local"},
        )
        assert resp.status_code == 200
        assert resp.json()["username"] == "users_edit_member_renamed"
        assert resp.json()["email"] == "renamed_member@test.local"

        resp = await client.patch(f"/api/admin/users/{admin.id}", json={"role": "user"})
        assert resp.status_code == 400
        resp = await client.delete(f"/api/admin/users/{admin.id}")
        assert resp.status_code == 400

        other_admin = db.User(username="users_edit_admin2", email="users_edit_admin2@test.local", hashed_password="x", role="admin", organization=org)
        session.add(other_admin)
        await session.flush()
        resp = await client.patch(f"/api/admin/users/{other_admin.id}", json={"role": "user"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "user"
    finally:
        for obj in [other_admin, member, admin, org]:
            if obj is not None:
                await session.delete(obj)
        await session.commit()


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
async def test_clear_video_sessions_removes_finished_history_only(client: AsyncClient, session, tmp_path):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session
    org = db.Organization(name="Video cleanup org")
    admin = db.User(username="video_cleanup_admin", email="video_cleanup_admin@test.local", hashed_password="x", role="admin", organization=org)
    session.add_all([org, admin])
    await session.flush()

    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=admin.id,
        username=admin.username,
        email=admin.email,
        role="admin",
        organization_id=org.id,
    )

    finished_file = tmp_path / "finished.mp4"
    running_file = tmp_path / "running.mp4"
    finished_file.write_bytes(b"finished")
    running_file.write_bytes(b"running")
    finished = db.VideoSession(
        source_type="upload",
        filename="finished.mp4",
        status="completed",
        video_path=str(finished_file),
        user_id=admin.id,
        organization_id=org.id,
    )
    running = db.VideoSession(
        source_type="upload",
        filename="running.mp4",
        status="running",
        video_path=str(running_file),
        user_id=admin.id,
        organization_id=org.id,
    )
    session.add_all([finished, running])
    await session.flush()

    try:
        resp = await client.delete("/api/video/sessions")
        assert resp.status_code == 200
        assert not finished_file.exists()
        assert running_file.exists()
        assert await db.get_video_session_by_id(session, finished.id) is None
        assert await db.get_video_session_by_id(session, running.id) is not None
    finally:
        remaining = await db.get_video_session_by_id(session, running.id)
        if remaining is not None:
            await session.delete(remaining)
        await session.delete(admin)
        await session.delete(org)
        await session.commit()


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


