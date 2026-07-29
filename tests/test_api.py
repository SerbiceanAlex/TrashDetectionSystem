"""
Teste pentru suprafața API curentă de monitorizare B2B.

Aceste teste evită intenționat inferența ML și se concentrează pe comportamentul DB/API:
shell-ul aplicației, dashboard, export rapoarte, sesiuni video și listarea incidentelor.
"""

import csv

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
async def test_admin_can_move_incident_between_all_review_states(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Status workflow org")
    admin = db.User(username="status_admin", email="status_admin@test.local", hashed_password="x", role="admin", organization=org)
    session.add_all([org, admin])
    await session.flush()
    event = db.LitteringEvent(material="plastic", det_score=0.86, status="pending", organization_id=org.id)
    session.add(event)
    await session.flush()

    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=admin.id,
        username=admin.username,
        email=admin.email,
        role="admin",
        organization_id=org.id,
    )

    try:
        for status in ["reviewed", "forwarded", "dismissed", "pending"]:
            resp = await client.patch(f"/api/littering/events/{event.id}/status", json={"status": status})
            assert resp.status_code == 200
            assert resp.json()["status"] == status

        await session.refresh(event)
        assert event.status == "pending"
        assert event.reviewed_by is None
        assert event.reviewed_at is None
        assert event.forwarded_at is None
    finally:
        await session.delete(event)
        await session.delete(admin)
        await session.delete(org)
        await session.commit()


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
async def test_operator_can_manage_own_incident_metadata_only(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="User own incident management org")
    operator = db.User(
        username="own_operator",
        email="own_operator@test.local",
        hashed_password="x",
        role="user",
        organization=org,
    )
    other_operator = db.User(
        username="other_operator",
        email="other_operator@test.local",
        hashed_password="x",
        role="user",
        organization=org,
    )
    session.add_all([org, operator, other_operator])
    await session.flush()

    event = db.LitteringEvent(
        material="metal",
        det_score=0.74,
        status="pending",
        reporter_id=operator.id,
        notes="admin note",
        organization_id=org.id,
    )
    other_event = db.LitteringEvent(
        material="glass",
        det_score=0.67,
        status="pending",
        reporter_id=other_operator.id,
        notes="private note",
        organization_id=org.id,
    )
    session.add_all([event, other_event])
    await session.flush()

    app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
        id=operator.id,
        username=operator.username,
        email=operator.email,
        role="user",
        organization_id=org.id,
    )

    try:
        view_resp = await client.get(f"/api/littering/events/{event.id}")
        assert view_resp.status_code == 200
        assert view_resp.json()["notes"] == "admin note"

        material_resp = await client.patch(
            f"/api/littering/events/{event.id}/material",
            json={"material": "plastic"},
        )
        assert material_resp.status_code == 200
        assert material_resp.json()["material"] == "plastic"

        status_resp = await client.patch(
            f"/api/littering/events/{event.id}/status",
            json={"status": "reviewed"},
        )
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "reviewed"

        note_resp = await client.patch(
            f"/api/littering/events/{event.id}/notes",
            json={"notes": "user edited note"},
        )
        assert note_resp.status_code == 200
        assert note_resp.json()["notes"] == "admin note"
        assert note_resp.json()["user_notes"] == "user edited note"

        await session.refresh(event)
        assert event.material == "plastic"
        assert event.status == "reviewed"
        assert event.notes == "admin note"
        assert event.user_notes == "user edited note"

        for endpoint, payload in [
            ("material", {"material": "paper"}),
            ("status", {"status": "reviewed"}),
            ("notes", {"notes": "changed"}),
        ]:
            blocked = await client.patch(f"/api/littering/events/{other_event.id}/{endpoint}", json=payload)
            assert blocked.status_code == 403

        await session.refresh(other_event)
        assert other_event.material == "glass"
        assert other_event.status == "pending"
        assert other_event.notes == "private note"
        assert other_event.user_notes is None
    finally:
        for obj in [event, other_event, operator, other_operator, org]:
            await session.delete(obj)
        await session.commit()


@pytest.mark.asyncio
async def test_notifications_are_scoped_per_user(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Notifications scoped org")
    admin = db.User(username="notif_admin", email="notif_admin@test.local", hashed_password="x", role="admin", organization=org)
    operator = db.User(username="notif_operator", email="notif_operator@test.local", hashed_password="x", role="user", organization=org)
    session.add_all([org, admin, operator])
    await session.flush()
    admin_notif = db.Notification(user_id=admin.id, message="admin only", category="incident")
    operator_notif = db.Notification(user_id=operator.id, message="operator only", category="incident")
    session.add_all([admin_notif, operator_notif])
    await session.commit()

    try:
        app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
            id=operator.id,
            username=operator.username,
            email=operator.email,
            role="user",
            organization_id=org.id,
        )
        operator_resp = await client.get("/api/me/notifications")
        assert operator_resp.status_code == 200
        assert [n["message"] for n in operator_resp.json()["notifications"]] == ["operator only"]

        await client.post("/api/me/notifications/read-all")
        await session.refresh(admin_notif)
        await session.refresh(operator_notif)
        assert operator_notif.is_read == 1
        assert admin_notif.is_read == 0

        app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
            id=admin.id,
            username=admin.username,
            email=admin.email,
            role="admin",
            organization_id=org.id,
        )
        admin_resp = await client.get("/api/me/notifications")
        assert admin_resp.status_code == 200
        assert [n["message"] for n in admin_resp.json()["notifications"]] == ["admin only"]
    finally:
        for obj in [admin_notif, operator_notif, admin, operator, org]:
            await session.delete(obj)
        await session.commit()


@pytest.mark.asyncio
async def test_admin_can_set_global_and_user_detection_sensitivity(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="Sensitivity org")
    admin = db.User(username="sensitivity_admin", email="sensitivity_admin@test.local", hashed_password="x", role="admin", organization=org)
    operator = db.User(username="sensitivity_operator", email="sensitivity_operator@test.local", hashed_password="x", role="user", organization=org)
    session.add_all([org, admin, operator])
    await session.flush()

    try:
        app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
            id=admin.id,
            username=admin.username,
            email=admin.email,
            role="admin",
            organization_id=org.id,
        )
        global_resp = await client.patch(
            "/api/admin/detection-settings/global",
            json={"det_conf": 0.22, "person_conf": 0.31, "analysis_fps": 45},
        )
        assert global_resp.status_code == 200
        assert global_resp.json()["global"]["analysis_fps"] == 45

        user_resp = await client.patch(
            f"/api/admin/detection-settings/users/{operator.id}",
            json={"det_conf": 0.18, "person_conf": None, "analysis_fps": 30},
        )
        assert user_resp.status_code == 200
        assert user_resp.json()["effective"]["det_conf"] == 0.18
        assert user_resp.json()["effective"]["person_conf"] == 0.31
        assert user_resp.json()["effective"]["analysis_fps"] == 30

        app.dependency_overrides[get_current_active_user] = lambda: SimpleNamespace(
            id=operator.id,
            username=operator.username,
            email=operator.email,
            role="user",
            organization_id=org.id,
        )
        own_resp = await client.get("/api/me/detection-settings")
        assert own_resp.status_code == 200
        assert own_resp.json()["effective"] == {
            "det_conf": 0.18,
            "person_conf": 0.31,
            "analysis_fps": 30,
        }
    finally:
        for obj in [admin, operator, org]:
            await session.delete(obj)
        await session.commit()


@pytest.mark.asyncio
async def test_admin_users_export_uses_sequential_numbers(client: AsyncClient, session):
    async def _override_get_db_same_session():
        yield session

    app.dependency_overrides[db.get_db] = _override_get_db_same_session

    org = db.Organization(name="CSV users org")
    admin = db.User(
        id=901,
        username="csv_admin",
        email="csv_admin@test.local",
        hashed_password="x",
        role="admin",
        organization=org,
        created_at=datetime(2026, 4, 8, 10, 25, tzinfo=timezone.utc),
    )
    operator = db.User(
        id=912,
        username="csv_operator",
        email="csv_operator@test.local",
        hashed_password="x",
        role="user",
        organization=org,
        created_at=datetime(2026, 5, 28, 11, 54, tzinfo=timezone.utc),
    )
    operator_new = db.User(
        id=914,
        username="csv_operator_new",
        email="csv_operator_new@test.local",
        hashed_password="x",
        role="user",
        organization=org,
        created_at=datetime(2026, 6, 27, 18, 55, tzinfo=timezone.utc),
    )
    session.add_all([org, admin, operator, operator_new])
    await session.flush()

    token = auth.create_access_token({"username": admin.username, "role": admin.role, "id": admin.id})
    resp = await client.get("/api/admin/export/users", params={"token": token})

    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")
    rows = list(csv.reader(resp.text.splitlines()))
    assert rows[0][:3] == ["Nr.", "Utilizator", "Email"]
    assert [row[0] for row in rows[1:]] == ["1", "2", "3"]
    assert [row[1] for row in rows[1:]] == ["csv_admin", "csv_operator", "csv_operator_new"]


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


