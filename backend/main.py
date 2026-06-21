"""
FastAPI application — Trash Detection System web interface.

Start with:
    .venv\\Scripts\\uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

from typing import Annotated, Optional
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, delete as sa_delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend import database as db
from backend import inference as infer
from backend import schemas
from backend.auth_router import router as auth_router, get_current_active_user, get_current_user_optional, oauth2_scheme
from backend.auth import decode_access_token
from backend.config import settings
from backend.storage_retention import cleanup_littering_evidence, storage_cleanup_loop
from backend import video as vid
from backend.billing_router import router as billing_router

STATIC_DIR = settings.REPO_ROOT / "frontend" / "static"
TEMPLATES_DIR = settings.REPO_ROOT / "frontend" / "templates"

UPLOADS_DIR = settings.uploads_dir
ANNOTATED_DIR = settings.annotated_dir
VIDEOS_DIR = settings.videos_dir
LITTERING_DIR = settings.littering_dir

for runtime_dir in settings.runtime_dirs:
    runtime_dir.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = settings.max_upload_bytes
VIDEO_MAX_UPLOAD_BYTES = settings.video_max_upload_bytes

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _resolve_littering_evidence_path(stored_path: str | None) -> Path | None:
    """Resolve a DB evidence path while keeping access inside LITTERING_DIR."""
    if not stored_path:
        return None

    root = LITTERING_DIR.resolve()
    raw = Path(str(stored_path))
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(root / raw)
        candidates.append(settings.REPO_ROOT / raw)
        if LITTERING_DIR.name in raw.parts:
            idx = len(raw.parts) - 1 - list(reversed(raw.parts)).index(LITTERING_DIR.name)
            suffix_parts = raw.parts[idx + 1:]
            if suffix_parts:
                candidates.append(root / Path(*suffix_parts))

    allowed: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        resolved_key = str(resolved)
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        allowed.append(resolved)

    if not allowed:
        logger.warning("Ignoring evidence path outside littering dir: %s", stored_path)
        return None
    return next((path for path in allowed if path.exists()), allowed[0])


# ── Lifespan: load models + create DB tables on startup ──────────────────────

async def _migrate_schema():
    """Add new columns to existing tables (SQLite ALTER TABLE).
    
    Safe to run repeatedly — each ALTER is wrapped in try/except
    so it's a no-op if the column already exists.
    """
    alter_statements = [
        # DetectionSession table — lifecycle fields
        "ALTER TABLE detection_sessions ADD COLUMN status VARCHAR(20) DEFAULT 'pending'",
        "ALTER TABLE detection_sessions ADD COLUMN cluster_id INTEGER REFERENCES detection_sessions(id)",
        "ALTER TABLE detection_sessions ADD COLUMN claimed_by INTEGER REFERENCES users(id)",
        "ALTER TABLE detection_sessions ADD COLUMN claimed_at DATETIME",
        "ALTER TABLE detection_sessions ADD COLUMN cleaned_image_path TEXT",
        "ALTER TABLE detection_sessions ADD COLUMN cleaned_at DATETIME",
        "ALTER TABLE detection_sessions ADD COLUMN expires_at DATETIME",
        "ALTER TABLE detection_sessions ADD COLUMN verification_score REAL DEFAULT 0.0",
        "ALTER TABLE detection_sessions ADD COLUMN user_note TEXT",
        # DetectionRecord — impact metrics
        "ALTER TABLE detection_records ADD COLUMN estimated_weight_kg REAL DEFAULT 0.0",
        # LitteringEvent — distance-based evidence fields (v2 state machine)
        "ALTER TABLE littering_events ADD COLUMN incident_uid VARCHAR(36)",
        "ALTER TABLE littering_events ADD COLUMN owner_person_id INTEGER",
        "ALTER TABLE littering_events ADD COLUMN distance_at_abandonment REAL",
        "ALTER TABLE littering_events ADD COLUMN detection_method VARCHAR(32) DEFAULT 'zone'",
        "ALTER TABLE littering_events ADD COLUMN reporter_id INTEGER REFERENCES users(id)",
        # Organization multi-tenant
        "ALTER TABLE users ADD COLUMN organization_id INTEGER REFERENCES organizations(id)",
        "ALTER TABLE monitored_locations ADD COLUMN organization_id INTEGER REFERENCES organizations(id)",
        "ALTER TABLE littering_events ADD COLUMN organization_id INTEGER REFERENCES organizations(id)",
        # Video + detection session isolation
        "ALTER TABLE video_sessions ADD COLUMN littering_count INTEGER DEFAULT 0",
        "ALTER TABLE video_sessions ADD COLUMN user_id INTEGER REFERENCES users(id)",
        "ALTER TABLE video_sessions ADD COLUMN organization_id INTEGER REFERENCES organizations(id)",
        "ALTER TABLE detection_sessions ADD COLUMN organization_id INTEGER REFERENCES organizations(id)",
    ]
    async with db.engine.begin() as conn:
        for stmt in alter_statements:
            try:
                await conn.execute(db.sa_text(stmt))
            except Exception:
                pass  # Column already exists — expected on subsequent runs

    # Migrate legacy data: is_resolved=1 → status='cleaned'
    async with db.engine.begin() as conn:
        await conn.execute(
            db.sa_text(
                "UPDATE detection_sessions SET status = 'cleaned' "
                "WHERE is_resolved = 1 AND (status IS NULL OR status = 'pending')"
            )
        )

    # Ensure default org exists and assign all legacy rows to it
    async with db.AsyncSessionLocal() as session:
        await db.get_or_create_default_org(session)
    async with db.engine.begin() as conn:
        for tbl in ("users", "monitored_locations", "littering_events",
                    "video_sessions", "detection_sessions"):
            try:
                await conn.execute(db.sa_text(
                    f"UPDATE {tbl} SET organization_id = 1 WHERE organization_id IS NULL"
                ))
            except Exception:
                pass

    print("[migration] Schema migration complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.create_tables()
    await _migrate_schema()
    infer.load_models()
    await asyncio.to_thread(vid.prewarm_monitor_inference)

    cleanup_task: asyncio.Task | None = None
    if settings.STORAGE_CLEANUP_ENABLED:
        try:
            summary = await cleanup_littering_evidence()
            logger.info("Startup storage cleanup complete: %s", summary)
        except Exception:
            logger.exception("Startup storage cleanup failed")
        cleanup_task = asyncio.create_task(storage_cleanup_loop())

    try:
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="Trash Detection System",
    description="Two-stage YOLO-based trash detection and material classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# Generated media files are served through authenticated API endpoints below,
# not as public static directories.

# Include Routers
app.include_router(auth_router)
app.include_router(billing_router)

# Serve the frontend SPA
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")

@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/js/") or request.url.path.startswith("/static/css/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    # Bypass ngrok browser warning page automatically
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


# ── Organization dependency ───────────────────────────────────────────────────

async def get_current_org(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
) -> db.Organization:
    """Return the Organization for the current user (creates default if needed)."""
    if current_user.organization_id:
        org = await db.get_org_by_id(session, current_user.organization_id)
        if org:
            return org
    return await db.get_or_create_default_org(session)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_files(original_bytes: bytes, annotated_bytes: bytes, stem: str):
    """Write original + annotated images to disk (runs as a background task)."""
    # Creează folderele doar la prima scanare de imagine (lazy), ca să nu
    # apară goale în proiect dacă nu s-au folosit niciodată.
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOADS_DIR / f"{stem}.jpg").write_bytes(original_bytes)
    (ANNOTATED_DIR / f"{stem}_annotated.jpg").write_bytes(annotated_bytes)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def landing():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app")


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context={}
    )


@app.get("/api/system/info", summary="Public system/model metadata")
async def system_info():
    """Return deploy-safe runtime metadata used by the frontend system panel."""
    manifest_path = settings.detector_path.parent / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Could not read detector manifest: %s", manifest_path)

    metrics = manifest.get("test_metrics") or {}

    return {
        "app": {
            "name": "TrashDet",
            "version": app.version,
            "base_url": settings.APP_BASE_URL,
        },
        "models": {
            "detector": {
                "name": manifest.get("active_model", "Detector final"),
                "architecture": manifest.get("architecture", "yolov8s"),
                "weights": settings.DETECTOR_WEIGHTS,
                "available": settings.detector_path.exists(),
                "manifest_available": bool(manifest),
                "dataset": manifest.get("dataset", ""),
                "dataset_split": manifest.get("dataset_split", ""),
                "imgsz": manifest.get("imgsz", settings.LIVE_IMGSZ),
                "sha256": manifest.get("sha256", ""),
                "metrics": {
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "mAP50": metrics.get("mAP50"),
                    "mAP50_95": metrics.get("mAP50_95"),
                },
            },
            "classifier": {
                "name": "B2 material classifier",
                "architecture": "YOLOv8n-cls",
                "weights": settings.CLASSIFIER_WEIGHTS,
                "available": settings.classifier_path.exists(),
            },
            "person_detector": {
                "name": "YOLOv8n COCO person detector",
                "architecture": "YOLOv8n",
                "weights": settings.PERSON_DETECTOR_WEIGHTS,
                "available": settings.person_detector_path.exists(),
            },
        },
        "runtime": {
            "live_imgsz": settings.LIVE_IMGSZ,
            "default_det_conf": settings.DEFAULT_DET_CONF,
            "monitor_min_det_conf": settings.MONITOR_MIN_DET_CONF,
            "monitor_target_fps": settings.MONITOR_TARGET_FPS,
            "monitor_camera_width": settings.MONITOR_CAMERA_WIDTH,
            "monitor_camera_height": settings.MONITOR_CAMERA_HEIGHT,
            "monitor_capture_max_dim": settings.MONITOR_CAPTURE_MAX_DIM,
            "monitor_jpeg_quality": settings.MONITOR_JPEG_QUALITY,
            "monitor_trash_imgsz": settings.MONITOR_TRASH_IMGSZ,
            "monitor_person_imgsz": settings.MONITOR_PERSON_IMGSZ,
            "max_upload_mb": settings.MAX_UPLOAD_MB,
            "video_max_upload_mb": settings.VIDEO_MAX_UPLOAD_MB,
            "littering_file_retention_days": settings.LITTERING_FILE_RETENTION_DAYS,
            "storage_cleanup_enabled": settings.STORAGE_CLEANUP_ENABLED,
        },
    }


@app.post("/api/detect", response_model=schemas.DetectResponse, summary="Upload image and run detection")
async def detect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    det_conf: float = Query(default=settings.DEFAULT_DET_CONF, ge=0.05, le=0.95, description="Detector confidence threshold"),
    latitude: float = Query(default=None, description="GPS latitude"),
    longitude: float = Query(default=None, description="GPS longitude"),
    user_note: str = Query(default=None, description="User note/description for the report"),
    session: AsyncSession = Depends(db.get_db),
    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,
):
    """
    Upload a JPEG/PNG image, run the two-stage pipeline, store results in DB,
    and return the annotated image URL + detection JSON.
    """
    # Optional auth links detections to the current user/organization.
    current_user = None
    if token:
        try:
            payload = decode_access_token(token)
            if payload and "username" in payload:
                res = await session.execute(select(db.User).where(db.User.username == payload["username"]))
                current_user = res.scalar_one_or_none()
        except Exception:
            pass
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")

    final_lat = latitude
    final_lng = longitude
    address   = None
    gps_src   = "browser" if latitude else None

    # Run inference
    try:
        detections, annotated_bytes, elapsed_ms = infer.run_pipeline(image_bytes, det_conf=det_conf)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Unique stem for saved files
    stem = uuid.uuid4().hex

    # Persist session row
    det_session = db.DetectionSession(
        filename=file.filename or "upload.jpg",
        image_path=str(UPLOADS_DIR / f"{stem}.jpg"),
        annotated_path=str(ANNOTATED_DIR / f"{stem}_annotated.jpg"),
        total_objects=len(detections),
        inference_ms=round(elapsed_ms, 2),
        latitude=final_lat,
        longitude=final_lng,
        address=address,
        gps_source=gps_src,
        organization_id=current_user.organization_id if current_user else None,
        reporter_id=current_user.id if current_user else None,
        user_note=user_note.strip()[:500] if user_note else None
    )
    session.add(det_session)
    await session.flush()  # get the auto-generated id

    # Persist individual detection records
    records = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        rec = db.DetectionRecord(
            session_id=det_session.id,
            material=det["material_name"],
            det_score=round(det["det_score"], 4),
            cls_score=round(det["material_score"], 4),
            box_x1=x1,
            box_y1=y1,
            box_x2=x2,
            box_y2=y2,
        )
        session.add(rec)
        records.append(rec)

    await session.commit()
    await session.refresh(det_session)
    for rec in records:
        await session.refresh(rec)

    # Save image files in background (non-blocking)
    background_tasks.add_task(_save_files, image_bytes, annotated_bytes, stem)

    return schemas.DetectResponse(
        session_id=det_session.id,
        filename=det_session.filename,
        total_objects=det_session.total_objects,
        inference_ms=det_session.inference_ms,
        annotated_url=f"/api/detect/sessions/{det_session.id}/annotated",
        detections=[schemas.DetectionRecordOut.model_validate(r) for r in records],
        latitude=final_lat,
        longitude=final_lng,
        address=address,
        gps_source=gps_src,
        reporter_id=current_user.id if current_user else None
    )


# ── Video endpoints ─────────────────────────────────────────────────────────

@app.websocket("/ws/video/live")
async def ws_video_live(
    websocket: WebSocket,
    det_conf: float = settings.DEFAULT_DET_CONF,
):
    """WebSocket for live webcam video: browser sends JPEG frames, server
    returns annotated frames + stats JSON."""
    async with db.AsyncSessionLocal() as session:
        await vid.handle_live_ws(websocket, det_conf, session)


@app.websocket("/ws/video/monitor")
async def ws_video_monitor(
    websocket: WebSocket,
    det_conf: float = Query(default=settings.MONITOR_MIN_DET_CONF, ge=0.10, le=0.95),
    person_conf: float = Query(default=0.20, ge=0.10, le=0.95),
    analysis_fps: float = Query(default=float(settings.MONITOR_TARGET_FPS), ge=5.0, le=120.0),
    lat: Optional[float] = Query(default=None),
    lng: Optional[float] = Query(default=None),
    token: Optional[str] = Query(default=None),
):
    """
    WebSocket for littering-event detection (monitor mode).
    Browser sends JPEG frames; server runs trash tracker + person detector,
    fires alert JSON when a littering event is detected.
    """
    async with db.AsyncSessionLocal() as session:
        current_user = None
        if token:
            try:
                payload = decode_access_token(token)
                username = payload.get("username") if payload else None
                if username:
                    current_user = (
                        await session.execute(select(db.User).where(db.User.username == username))
                    ).scalar_one_or_none()
            except Exception:
                current_user = None
        await vid.handle_monitor_ws(
            websocket,
            det_conf,
            person_conf,
            analysis_fps,
            lat,
            lng,
            session,
            user_id=current_user.id if current_user else None,
            organization_id=(current_user.organization_id or 1) if current_user else 1,
        )


# ── Littering Events REST ─────────────────────────────────────────────────────

@app.get(
    "/api/littering/events",
    response_model=schemas.LitteringEventsPage,
    summary="List littering events scoped to the current user role",
)
async def list_littering_events(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None, description="Filter by status: pending/reviewed/forwarded/dismissed"),
    material: Optional[str] = Query(default=None),
    reporter_id: Optional[int] = Query(default=None, description="Admin-only filter by reporter/user id"),
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    org_id = current_user.organization_id or 1
    reporter_filter = reporter_id if current_user.role == "admin" else current_user.id
    items, total = await db.list_littering_events(
        session,
        skip=skip,
        limit=limit,
        status=status,
        material=material,
        org_id=org_id,
        reporter_id=reporter_filter,
    )
    return schemas.LitteringEventsPage(total=total, skip=skip, limit=limit, items=items)


def _same_org(user: db.User, event: db.LitteringEvent) -> bool:
    return (event.organization_id or 1) == (user.organization_id or 1)


def _can_view_littering_event(user: db.User, event: db.LitteringEvent) -> bool:
    if user.role == "admin":
        return _same_org(user, event)
    return _same_org(user, event) and event.reporter_id == user.id


def _same_video_org(user: db.User, video_session: db.VideoSession) -> bool:
    return (video_session.organization_id or 1) == (user.organization_id or 1)


def _can_view_video_session(user: db.User, video_session: db.VideoSession) -> bool:
    if user.role == "admin":
        return _same_video_org(user, video_session)
    return _same_video_org(user, video_session) and video_session.user_id == user.id


def _same_detection_org(user: db.User, detection_session: db.DetectionSession) -> bool:
    return (detection_session.organization_id or 1) == (user.organization_id or 1)


def _can_view_detection_session(user: db.User, detection_session: db.DetectionSession) -> bool:
    if user.role == "admin":
        return _same_detection_org(user, detection_session)
    return _same_detection_org(user, detection_session) and detection_session.reporter_id == user.id


async def _user_from_bearer_or_query(
    request: Request,
    session: AsyncSession,
    token: Optional[str] = None,
) -> db.User | None:
    raw_token = token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        raw_token = auth_header[7:]
    if not raw_token:
        return None
    try:
        payload = decode_access_token(raw_token)
        username = payload.get("username") if payload else None
        if not username:
            return None
        result = await session.execute(select(db.User).where(db.User.username == username))
        return result.scalar_one_or_none()
    except Exception:
        return None


@app.get(
    "/api/detect/sessions/{session_id}/annotated",
    summary="Download annotated image for a detection session",
)
async def download_detection_annotated_image(
    session_id: int,
    request: Request,
    token: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(db.get_db),
):
    current_user = await _user_from_bearer_or_query(request, session, token)
    if current_user is None:
        raise HTTPException(status_code=401, detail="Autentificare necesară.")

    det_session = await session.get(db.DetectionSession, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiune de detecție negăsită.")
    if not _can_view_detection_session(current_user, det_session):
        raise HTTPException(status_code=403, detail="Acces restricționat la această detecție.")
    if not det_session.annotated_path:
        raise HTTPException(status_code=404, detail="Imaginea adnotată nu este disponibilă.")

    p = Path(det_session.annotated_path)
    if not p.exists():
        raise HTTPException(status_code=410, detail="Imaginea adnotată a fost ștearsă.")

    return FileResponse(p, media_type="image/jpeg", filename=p.name)


@app.get(
    "/api/littering/events/{event_id}",
    response_model=schemas.LitteringEventOut,
    summary="Get littering event by ID",
)
async def get_littering_event(
    event_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    evt = await db.get_littering_event_by_id(session, event_id)
    if evt is None:
        raise HTTPException(status_code=404, detail="Eveniment negăsit.")
    if not _can_view_littering_event(current_user, evt):
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    return evt


@app.patch(
    "/api/littering/events/{event_id}/status",
    response_model=schemas.LitteringEventOut,
    summary="[Admin] Update littering event status",
)
async def update_littering_event_status(
    event_id: int,
    body: schemas.LitteringEventStatusUpdate,
    background_tasks: BackgroundTasks,
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    evt_current = await db.get_littering_event_by_id(session, event_id)
    if evt_current is None:
        raise HTTPException(status_code=404, detail="Eveniment negăsit.")
    if not _same_org(current_user, evt_current):
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    allowed_statuses = {"reviewed", "forwarded", "dismissed"}
    if body.status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Status invalid. Valori permise: {', '.join(sorted(allowed_statuses))}"
        )
    evt = await db.update_littering_event_status(
        session, event_id,
        status=body.status,
        reviewed_by=current_user.id,
        notes=body.notes,
    )
    if evt is None:
        raise HTTPException(status_code=404, detail="Eveniment negăsit.")

    # In the thesis build, "forwarded" is used as a local archive/prepared status.
    # No outbound email is sent from the application.
    if body.status == "forwarded":
        logger.info("Incident #%d arhivat local; livrarea prin email este dezactivată.", evt.id)

    return evt


@app.patch(
    "/api/littering/events/{event_id}/notes",
    response_model=schemas.LitteringEventOut,
    summary="[Admin] Update notes on a littering event",
)
async def update_littering_event_notes(
    event_id: int,
    body: schemas.LitteringEventNotesUpdate,
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    evt = await db.get_littering_event_by_id(session, event_id)
    if evt is None:
        raise HTTPException(status_code=404, detail="Eveniment negăsit.")
    if not _can_view_littering_event(current_user, evt):
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    evt.notes = body.notes
    await session.commit()
    await session.refresh(evt)
    return evt


@app.get(
    "/api/littering/events/{event_id}/clip",
    summary="Download clip for a littering event",
)
async def download_littering_clip(
    event_id: int,
    request: Request,
    token: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(db.get_db),
):
    current_user = await _user_from_bearer_or_query(request, session, token)
    if current_user is None:
        raise HTTPException(status_code=401, detail="Autentificare necesară.")
    evt = await db.get_littering_event_by_id(session, event_id)
    if evt is None or not evt.clip_path:
        raise HTTPException(status_code=404, detail="Clip indisponibil.")
    if not _can_view_littering_event(current_user, evt):
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    full_path = _resolve_littering_evidence_path(evt.clip_path)
    if full_path is None or not full_path.exists():
        raise HTTPException(status_code=404, detail="Fișier clip negăsit pe disk.")
    return FileResponse(
        str(full_path),
        media_type="video/mp4",
        filename=f"littering_event_{event_id}.mp4",
    )


@app.get(
    "/api/littering/events/{event_id}/thumbnail",
    summary="Get thumbnail for a littering event",
)
async def get_littering_thumbnail(
    event_id: int,
    request: Request,
    token: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(db.get_db),
):
    current_user = await _user_from_bearer_or_query(request, session, token)
    if current_user is None:
        raise HTTPException(status_code=401, detail="Autentificare necesară.")
    evt = await db.get_littering_event_by_id(session, event_id)
    if evt is None or not evt.thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail indisponibil.")
    if not _can_view_littering_event(current_user, evt):
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    full_path = _resolve_littering_evidence_path(evt.thumbnail_path)
    if full_path is None or not full_path.exists():
        raise HTTPException(status_code=404, detail="Fișier thumbnail negăsit pe disk.")
    return FileResponse(str(full_path), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# B2B Dashboard / Locations / Reports — endpoint-uri pentru produsul B2B
# ---------------------------------------------------------------------------

@app.get("/api/dashboard/b2b", summary="B2B dashboard — KPI + trend + recent incidents")
async def get_dashboard_b2b(
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    """KPI-uri și statistici pentru dashboard-ul B2B (incidente, trend, locații)."""
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)

    LE = db.LitteringEvent
    ML = db.MonitoredLocation
    org_id = current_user.organization_id or 1 if current_user else 1
    org_cond = db.or_(LE.organization_id == org_id, LE.organization_id.is_(None))
    role_cond = org_cond if (current_user and current_user.role == "admin") else db.and_(org_cond, LE.reporter_id == current_user.id)

    async def count_where(*conds):
        q = db.select(db.func.count()).select_from(LE).where(role_cond)
        for c in conds:
            q = q.where(c)
        return (await session.execute(q)).scalar_one() or 0

    incidents_today = await count_where(LE.detected_at >= today_start)
    incidents_week = await count_where(LE.detected_at >= week_start)
    incidents_month = await count_where(LE.detected_at >= month_start)
    pending_review = await count_where(LE.status == "pending")
    forwarded = await count_where(LE.status == "forwarded")

    # Active locations — scoped by org
    try:
        ml_cond = db.or_(ML.organization_id == org_id, ML.organization_id.is_(None))
        active_locations = (
            await session.execute(
                db.select(db.func.count()).select_from(ML)
                .where(ML.is_active == 1).where(ml_cond)
            )
        ).scalar_one() or 0
    except Exception:
        active_locations = 0

    # Recent incidents (last 5) — scoped by org
    rec = (
        await session.execute(
            db.select(LE)
            .options(selectinload(LE.reporter))
            .where(role_cond)
            .order_by(LE.detected_at.desc())
            .limit(5)
        )
    ).scalars().all()
    recent_incidents = [
        {
            "id": e.id,
            "detected_at": e.detected_at.isoformat() if e.detected_at else None,
            "material": e.material,
            "status": e.status,
            "det_score": e.det_score or 0,
            "thumbnail_path": e.thumbnail_path,
            "clip_path": e.clip_path,
            "reporter_id": e.reporter_id,
            "reporter_username": e.reporter_username,
        }
        for e in rec
    ]

    # Material distribution (last 30 days) — org-scoped
    mat_q = await session.execute(
        db.select(LE.material, db.func.count())
        .where(role_cond, LE.detected_at >= month_start)
        .group_by(LE.material)
        .order_by(db.func.count().desc())
    )
    material_distribution = [
        {"material": m or "unknown", "count": c} for m, c in mat_q.all()
    ]

    # Trend last 30 days — org-scoped
    trend_q = await session.execute(
        db.select(db.func.date(LE.detected_at), db.func.count())
        .where(role_cond, LE.detected_at >= month_start)
        .group_by(db.func.date(LE.detected_at))
        .order_by(db.func.date(LE.detected_at))
    )
    trend_map = {str(d): c for d, c in trend_q.all()}
    trend_30d = []
    for i in range(30):
        day = (today_start - timedelta(days=29 - i)).date().isoformat()
        trend_30d.append({"day": day, "count": trend_map.get(day, 0)})

    # Hourly distribution — org-scoped
    hour_q = await session.execute(
        db.select(db.func.strftime('%H', LE.detected_at), db.func.count())
        .where(role_cond)
        .group_by(db.func.strftime('%H', LE.detected_at))
    )
    hour_map = {int(h): c for h, c in hour_q.all() if h is not None}
    hourly_distribution = [hour_map.get(h, 0) for h in range(24)]

    # Resolution rate (reviewed + forwarded) / total — org-scoped
    resolved = await count_where(LE.status.in_(["reviewed", "forwarded", "dismissed"]))
    total_all = await count_where()
    resolution_rate = round(resolved / max(total_all, 1) * 100)

    return {
        "incidents_today": incidents_today,
        "incidents_week": incidents_week,
        "incidents_month": incidents_month,
        "pending_review": pending_review,
        "forwarded": forwarded,
        "active_locations": active_locations,
        "recent_incidents": recent_incidents,
        "material_distribution": material_distribution,
        "trend_30d": trend_30d,
        "hourly_distribution": hourly_distribution,
        "resolution_rate": resolution_rate,
        "total_all_time": total_all,
    }


@app.post("/api/locations/test-rtsp", summary="Test RTSP URL reachability (socket check)")
async def test_rtsp(
    body: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
):
    import socket, urllib.parse
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restrictionat.")

    url = (body.get("rtsp_url") or "").strip()
    if not url:
        return {"ok": False, "message": "URL gol"}
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname
        port = parsed.port or 554
        if not host:
            return {"ok": False, "message": "URL invalid — host lipsă"}
        sock = socket.create_connection((host, port), timeout=4)
        sock.close()
        return {"ok": True, "message": f"Conexiune reușită la {host}:{port}"}
    except socket.timeout:
        return {"ok": False, "message": "Timeout — camera nu răspunde în 4s"}
    except OSError as e:
        return {"ok": False, "message": f"Eroare rețea: {e}"}


@app.get("/api/locations", summary="List monitored locations (B2B)")
async def list_locations(
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    ML = db.MonitoredLocation
    LE = db.LitteringEvent
    org_id = current_user.organization_id or 1
    locs = (await session.execute(
        db.select(ML)
        .where(db.or_(ML.organization_id == org_id, ML.organization_id.is_(None)))
        .order_by(ML.created_at.desc())
    )).scalars().all()

    out = []
    for loc in locs:
        # incidente associate (by gps proximity ~ 100m radius — temporar simplificat)
        incidents_count = 0
        pending_count = 0
        last_event_at = None
        if loc.latitude and loc.longitude:
            # raza ~ 0.001 deg ≈ 100m
            r = 0.001
            cnt_q = await session.execute(
                db.select(db.func.count()).select_from(LE).where(
                    LE.latitude.between(loc.latitude - r, loc.latitude + r),
                    LE.longitude.between(loc.longitude - r, loc.longitude + r),
                )
            )
            incidents_count = cnt_q.scalar_one() or 0
            pend_q = await session.execute(
                db.select(db.func.count()).select_from(LE).where(
                    LE.latitude.between(loc.latitude - r, loc.latitude + r),
                    LE.longitude.between(loc.longitude - r, loc.longitude + r),
                    LE.status == "pending",
                )
            )
            pending_count = pend_q.scalar_one() or 0
            last_q = await session.execute(
                db.select(db.func.max(LE.detected_at)).where(
                    LE.latitude.between(loc.latitude - r, loc.latitude + r),
                    LE.longitude.between(loc.longitude - r, loc.longitude + r),
                )
            )
            last_event_at = last_q.scalar_one()

        out.append({
            "id": loc.id,
            "name": loc.name,
            "address": loc.address,
            "lat": loc.latitude,
            "lng": loc.longitude,
            "rtsp_url": loc.rtsp_url,
            "alert_email": loc.alert_email,
            "is_active": bool(loc.is_active),
            "created_at": loc.created_at.isoformat() if loc.created_at else None,
            "incidents_count": incidents_count,
            "pending_count": pending_count,
            "last_event_at": last_event_at.isoformat() if last_event_at else None,
        })
    return {"locations": out}


@app.post("/api/locations", summary="Create monitored location (B2B)")
async def create_location(
    payload: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restrictionat.")

    ML = db.MonitoredLocation
    loc = ML(
        name=payload.get("name", "").strip(),
        address=payload.get("address"),
        latitude=payload.get("lat"),
        longitude=payload.get("lng"),
        rtsp_url=payload.get("rtsp_url"),
        alert_email=payload.get("alert_email"),
        is_active=1 if payload.get("is_active", True) else 0,
        created_by=current_user.id if current_user else None,
        organization_id=current_user.organization_id or 1 if current_user else 1,
    )
    if not loc.name:
        raise HTTPException(status_code=400, detail="Numele locației este obligatoriu.")
    session.add(loc)
    await session.commit()
    await session.refresh(loc)
    return {"id": loc.id, "name": loc.name}


@app.delete(
    "/api/littering/events/{event_id}",
    response_model=schemas.DetailResponse,
    summary="[Admin] Delete a littering event completely and its associated files",
)
async def delete_littering_event(
    event_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat.")

    event = await session.get(db.LitteringEvent, event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Incident not found")
    if not _same_org(current_user, event):
        raise HTTPException(status_code=403, detail="Acces restricționat.")

    for rel_path in (event.clip_path, event.thumbnail_path):
        candidate = _resolve_littering_evidence_path(rel_path)
        if candidate is None:
            continue
        if candidate.exists() and candidate.is_file():
            candidate.unlink()

    await session.delete(event)
    await session.commit()
    return schemas.DetailResponse(detail="Incident șters definitiv și stocarea eliberată.")


@app.patch("/api/locations/{loc_id}", summary="Update monitored location (B2B)")
async def update_location(
    loc_id: int,
    payload: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restrictionat.")

    ML = db.MonitoredLocation
    loc = (await session.execute(db.select(ML).where(ML.id == loc_id))).scalar_one_or_none()
    if not loc:
        raise HTTPException(status_code=404, detail="Locație inexistentă.")
    if (loc.organization_id or 1) != (current_user.organization_id or 1):
        raise HTTPException(status_code=403, detail="Acces restrictionat.")
    if "name" in payload and payload["name"]: loc.name = payload["name"]
    if "address" in payload: loc.address = payload["address"]
    if "lat" in payload: loc.latitude = payload["lat"]
    if "lng" in payload: loc.longitude = payload["lng"]
    if "rtsp_url" in payload: loc.rtsp_url = payload["rtsp_url"]
    if "alert_email" in payload: loc.alert_email = payload["alert_email"]
    if "is_active" in payload: loc.is_active = 1 if payload["is_active"] else 0
    await session.commit()
    return {"id": loc.id, "is_active": bool(loc.is_active)}


@app.delete("/api/locations/{loc_id}", summary="Delete monitored location (B2B)")
async def delete_location(
    loc_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restrictionat.")

    ML = db.MonitoredLocation
    loc = (await session.execute(db.select(ML).where(ML.id == loc_id))).scalar_one_or_none()
    if not loc:
        raise HTTPException(status_code=404, detail="Locație inexistentă.")
    if (loc.organization_id or 1) != (current_user.organization_id or 1):
        raise HTTPException(status_code=403, detail="Acces restrictionat.")
    await session.delete(loc)
    await session.commit()
    return {"deleted": True}


@app.get("/api/reports/stats", summary="Reports stats (B2B export)")
async def reports_stats(
    period: str = Query(default="week"),
    from_: Optional[str] = Query(default=None, alias="from"),
    to: Optional[str] = Query(default=None),
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "today":
        start = today_start; end = now
    elif period == "week":
        start = today_start - timedelta(days=7); end = now
    elif period == "month":
        start = today_start - timedelta(days=30); end = now
    elif period == "year":
        start = today_start - timedelta(days=365); end = now
    elif period == "custom" and from_ and to:
        start = datetime.fromisoformat(from_).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(to).replace(tzinfo=timezone.utc) + timedelta(days=1)
    else:
        start = today_start - timedelta(days=7); end = now

    LE = db.LitteringEvent
    org_id_r = current_user.organization_id or 1 if current_user else 1
    org_cond_r = db.or_(LE.organization_id == org_id_r, LE.organization_id.is_(None))
    role_cond_r = org_cond_r if (current_user and current_user.role == "admin") else db.and_(org_cond_r, LE.reporter_id == current_user.id)
    cond = db.and_(LE.detected_at >= start, LE.detected_at <= end, role_cond_r)

    total_incidents = (await session.execute(db.select(db.func.count()).select_from(LE).where(cond))).scalar_one() or 0
    pending = (await session.execute(db.select(db.func.count()).select_from(LE).where(cond, LE.status == "pending"))).scalar_one() or 0
    forwarded = (await session.execute(db.select(db.func.count()).select_from(LE).where(cond, LE.status == "forwarded"))).scalar_one() or 0

    # Hourly distribution (24 buckets)
    hourly_q = await session.execute(
        db.select(db.func.strftime("%H", LE.detected_at), db.func.count())
        .where(cond).group_by(db.func.strftime("%H", LE.detected_at))
    )
    hourly_distribution = [0] * 24
    for h_str, c in hourly_q.all():
        try:
            hourly_distribution[int(h_str)] = c
        except Exception: pass

    # Material distribution
    mat_q = await session.execute(
        db.select(LE.material, db.func.count()).where(cond).group_by(LE.material).order_by(db.func.count().desc())
    )
    material_distribution = [{"material": m or "unknown", "count": c} for m, c in mat_q.all()]

    # Active locations count
    try:
        ML = db.MonitoredLocation
        locations_active = (await session.execute(db.select(db.func.count()).select_from(ML).where(ML.is_active == 1))).scalar_one() or 0
    except Exception:
        locations_active = 0

    return {
        "period": period,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "total_incidents": total_incidents,
        "pending": pending,
        "forwarded": forwarded,
        "locations_active": locations_active,
        "hourly_distribution": hourly_distribution,
        "material_distribution": material_distribution,
        "top_locations": [],  # to be populated when location-incident linkage exists
    }


@app.get("/api/reports/export", summary="Export report (CSV / PDF / ZIP)")
async def reports_export(
    period: str = Query(default="week"),
    format: str = Query(default="csv"),
    from_: Optional[str] = Query(default=None, alias="from"),
    to: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    material: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(db.get_db),
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
):
    """Export incidents as CSV, PDF, or ZIP."""
    from datetime import datetime, timedelta, timezone
    import csv, io
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if period == "today":
        start = today_start; end = now
    elif period == "week":
        start = today_start - timedelta(days=7); end = now
    elif period == "month":
        start = today_start - timedelta(days=30); end = now
    elif period == "year":
        start = today_start - timedelta(days=365); end = now
    elif period == "all":
        start = datetime(1970, 1, 1, tzinfo=timezone.utc); end = now
    elif period == "custom" and from_ and to:
        start = datetime.fromisoformat(from_).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(to).replace(tzinfo=timezone.utc) + timedelta(days=1)
    else:
        start = today_start - timedelta(days=7); end = now

    LE = db.LitteringEvent
    org_id = current_user.organization_id or 1 if current_user else 1
    org_cond = db.or_(LE.organization_id == org_id, LE.organization_id.is_(None))
    role_cond = org_cond if (current_user and current_user.role == "admin") else db.and_(org_cond, LE.reporter_id == current_user.id)
    query = db.select(LE).options(selectinload(LE.reporter)).where(LE.detected_at >= start, LE.detected_at <= end, role_cond)
    if status:
        query = query.where(LE.status == status)
    if material:
        query = query.where(LE.material == material)
    rows = (await session.execute(query.order_by(LE.detected_at.desc()))).scalars().all()

    # ── CSV ──────────────────────────────────────────────────────────────
    if format == "csv":
        status_ro = {
            "pending": "În așteptare",
            "reviewed": "Confirmat",
            "forwarded": "Arhivat",
            "dismissed": "Fals pozitiv",
        }
        material_ro = {
            "plastic": "Plastic",
            "paper": "Hârtie",
            "glass": "Sticlă",
            "metal": "Metal",
            "other": "Altele",
            "unknown": "Necunoscut",
        }

        def _csv_value(value, default="N/A"):
            if value is None:
                return default
            if isinstance(value, str) and not value.strip():
                return default
            return value

        def _location_source(event):
            if event.latitude is not None and event.longitude is not None:
                return "GPS browser/cameră"
            if event.address:
                return "Adresă configurată manual"
            return "Nespecificată"

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([
            "ID",
            "Data detectare",
            "Material",
            "Confidenta (%)",
            "Status",
            "Utilizator",
            "Sursa locatie",
            "Latitudine (optional)",
            "Longitudine (optional)",
            "Adresa / camera",
            "Metoda detectie",
            "Persoane",
            "Hash SHA-256",
            "Clip",
            "Note",
        ])
        for e in rows:
            w.writerow([
                e.id,
                e.detected_at.strftime("%d.%m.%Y %H:%M:%S") if e.detected_at else "",
                material_ro.get(e.material or "unknown", e.material or "Necunoscut"),
                round((e.det_score or 0) * 100, 1),
                status_ro.get(e.status or "", e.status or "N/A"),
                e.reporter_username or (f"utilizator #{e.reporter_id}" if e.reporter_id else "legacy / neatribuit"),
                _location_source(e),
                _csv_value(e.latitude),
                _csv_value(e.longitude),
                _csv_value(e.address),
                _csv_value(e.detection_method, "zone"),
                _csv_value(e.person_count),
                _csv_value(e.image_hash),
                _csv_value(e.clip_path),
                _csv_value(e.notes, ""),
            ])
        return PlainTextResponse(
            "\ufeff" + buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="raport_trashdet_{period}.csv"'},
        )

    # ── PDF ──────────────────────────────────────────────────────────────
    if format == "pdf":
        try:
            import os
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                            Paragraph, Spacer)
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont

            # Înregistrăm Arial cu suport Unicode complet (diacritice românești)
            _FONT_NORMAL = "Helvetica"
            _FONT_BOLD   = "Helvetica-Bold"
            _win_fonts = r"C:\Windows\Fonts"
            if os.path.exists(os.path.join(_win_fonts, "arial.ttf")):
                try:
                    pdfmetrics.registerFont(TTFont("Arial",     os.path.join(_win_fonts, "arial.ttf")))
                    pdfmetrics.registerFont(TTFont("Arial-Bold",os.path.join(_win_fonts, "arialbd.ttf")))
                    _FONT_NORMAL, _FONT_BOLD = "Arial", "Arial-Bold"
                except Exception:
                    pass  # fallback la Helvetica cu diacritice stripped

            def _txt(s: str) -> str:
                """Strip diacritice dacă fontul nu are suport Unicode."""
                if _FONT_NORMAL == "Arial":
                    return str(s) if s else ""
                # Helvetica fallback — înlocuim diacritice
                for ro, en in [("ă","a"),("â","a"),("î","i"),("ș","s"),("ț","t"),
                                ("Ă","A"),("Â","A"),("Î","I"),("Ș","S"),("Ț","T")]:
                    s = str(s).replace(ro, en)
                return s

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4,
                                    leftMargin=2*cm, rightMargin=2*cm,
                                    topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story = []

            # Titlu
            title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                         fontName=_FONT_BOLD, fontSize=18,
                                         textColor=colors.HexColor("#059669"),
                                         alignment=TA_CENTER, spaceAfter=4)
            story.append(Paragraph(_txt("TrashDet — Raport Incidente"), title_style))

            sub_style = ParagraphStyle("sub", parent=styles["Normal"],
                                       fontName=_FONT_NORMAL,
                                       fontSize=10, textColor=colors.HexColor("#6b7280"),
                                       alignment=TA_CENTER, spaceAfter=6)
            period_labels = {"today": "Astazi", "week": "Ultimele 7 zile",
                             "month": "Ultimele 30 zile", "year": "Ultimul an", "custom": "Perioada personalizata"}
            story.append(Paragraph(
                _txt(f"Perioada: {period_labels.get(period, period)} · "
                     f"Generat: {now.strftime('%d.%m.%Y %H:%M UTC')} · "
                     f"Administrator: {current_user.username}"), sub_style))
            story.append(Spacer(1, 0.4*cm))

            # Rezumat
            pending = sum(1 for e in rows if e.status == "pending")
            reviewed = sum(1 for e in rows if e.status == "reviewed")
            forwarded = sum(1 for e in rows if e.status == "forwarded")
            dismissed = sum(1 for e in rows if e.status == "dismissed")

            summary_data = [
                [_txt(h) for h in ["Total incidente", "Asteptare", "Confirmate", "Arhivate", "Fals pozitiv"]],
                [str(len(rows)), str(pending), str(reviewed), str(forwarded), str(dismissed)],
            ]
            summary_table = Table(summary_data, colWidths=[3.4*cm]*5)
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#059669")),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTNAME",   (0,0), (-1,0), _FONT_BOLD),
                ("FONTSIZE",   (0,0), (-1,0), 9),
                ("FONTNAME",   (0,1), (-1,1), _FONT_BOLD),
                ("FONTSIZE",   (0,1), (-1,1), 16),
                ("ALIGN",      (0,0), (-1,-1), "CENTER"),
                ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0,1), (-1,1), [colors.HexColor("#f0fdf4")]),
                ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#a7f3d0")),
                ("TOPPADDING",  (0,0), (-1,-1), 6),
                ("BOTTOMPADDING",(0,0), (-1,-1), 6),
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.6*cm))

            # Tabel incidente
            if rows:
                header = [_txt(h) for h in ["#", "Data", "Material", "Conf.", "Status", "Adresa / GPS"]]
                table_data = [header]
                status_ro = {"pending": _txt("Asteptare"), "reviewed": _txt("Confirmat"),
                             "forwarded": _txt("Arhivat"), "dismissed": _txt("Fals pozitiv")}
                for e in rows:
                    loc = _txt(e.address) or (f"{e.latitude:.4f}, {e.longitude:.4f}"
                                               if e.latitude else "-")
                    table_data.append([
                        str(e.id),
                        e.detected_at.strftime("%d.%m.%Y\n%H:%M") if e.detected_at else "-",
                        _txt((e.material or "-").capitalize()),
                        f"{(e.det_score or 0)*100:.0f}%",
                        status_ro.get(e.status or "", e.status or "-"),
                        loc[:40] + ("..." if loc and len(loc) > 40 else ""),
                    ])

                col_w = [1.2*cm, 2.4*cm, 2.2*cm, 1.4*cm, 2.2*cm, 7.4*cm]
                inc_table = Table(table_data, colWidths=col_w, repeatRows=1)
                inc_table.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#065f46")),
                    ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
                    ("FONTNAME",      (0,0), (-1,0), _FONT_BOLD),
                    ("FONTNAME",      (0,1), (-1,-1), _FONT_NORMAL),
                    ("FONTSIZE",      (0,0), (-1,0), 8),
                    ("FONTSIZE",      (0,1), (-1,-1), 8),
                    ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                    ("ALIGN",         (5,1), (5,-1), "LEFT"),
                    ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f9fafb")]),
                    ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#e5e7eb")),
                    ("TOPPADDING",    (0,0),(-1,-1), 5),
                    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
                ]))
                story.append(inc_table)
            else:
                story.append(Paragraph(_txt("Niciun incident in perioada selectata."), styles["Normal"]))

            story.append(Spacer(1, 0.8*cm))
            footer_style = ParagraphStyle("footer", parent=styles["Normal"],
                                          fontName=_FONT_NORMAL,
                                          fontSize=8, textColor=colors.HexColor("#9ca3af"),
                                          alignment=TA_CENTER)
            story.append(Paragraph(
                _txt("Document generat automat de TrashDet · Sistem de detectie ilegala a deseurilor · "
                     "dovezile sunt stocate local conform perioadei de retentie configurate"), footer_style))

            doc.build(story)
            buf.seek(0)
            return Response(
                content=buf.getvalue(), media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="raport_trashdet_{period}.pdf"'},
            )
        except ImportError:
            raise HTTPException(status_code=500, detail="reportlab nu este instalat pe server.")

    # ── ZIP cu clipuri ────────────────────────────────────────────────────
    if format == "zip":
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # metadata JSON
            import json
            meta = [{"id": e.id, "detected_at": e.detected_at.isoformat() if e.detected_at else None,
                     "material": e.material, "status": e.status, "det_score": e.det_score,
                     "image_hash": e.image_hash, "address": e.address} for e in rows]
            zf.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))

            # clipuri + thumbnail-uri
            for e in rows:
                if e.clip_path:
                    fp = LITTERING_DIR / e.clip_path
                    if fp.exists():
                        zf.write(fp, f"clips/{fp.name}")
                if e.thumbnail_path:
                    tp = LITTERING_DIR / e.thumbnail_path
                    if tp.exists():
                        zf.write(tp, f"thumbnails/{tp.name}")

        buf.seek(0)
        return Response(
            content=buf.getvalue(), media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="evidence_trashdet_{period}.zip"'},
        )

    raise HTTPException(status_code=400, detail=f"Format '{format}' necunoscut. Valori valide: csv, pdf, zip.")


@app.post("/api/video/upload", response_model=schemas.VideoUploadResponse,
          summary="Upload a video file for offline processing")
async def upload_video(
    file: UploadFile = File(...),
    det_conf: float = Query(default=settings.DEFAULT_DET_CONF, ge=0.05, le=0.95),
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    allowed = {
        "video/mp4", "video/mpeg", "video/x-msvideo", "video/quicktime",
        "video/x-matroska", "video/webm", "video/avi",
    }
    ct = file.content_type or ""
    fname = file.filename or "upload.mp4"
    # Also accept by extension if mime unknown
    ext = Path(fname).suffix.lower()
    if ct not in allowed and ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {ct}")

    stem = uuid.uuid4().hex
    save_path = VIDEOS_DIR / f"{stem}{ext or '.mp4'}"

    # Write to disk in 1 MB chunks — avoids loading the entire file into RAM
    chunk_size = 1024 * 1024
    video_empty = True
    written_bytes = 0
    over_video_limit = False
    with open(save_path, "wb") as out_f:
        while True:
            chunk = await asyncio.to_thread(file.file.read, chunk_size)
            if not chunk:
                break
            written_bytes += len(chunk)
            if written_bytes > VIDEO_MAX_UPLOAD_BYTES:
                over_video_limit = True
                break
            out_f.write(chunk)
            video_empty = False
    if over_video_limit:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"Video too large. Maximum size is {settings.VIDEO_MAX_UPLOAD_MB} MB.",
        )
    if video_empty:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    vs = await db.create_video_session(session, source_type="upload", filename=fname)
    vs.video_path = str(save_path)
    vs.user_id = current_user.id
    vs.organization_id = current_user.organization_id or 1
    await session.commit()

    # Process in background — fire-and-forget with error logging
    task = asyncio.create_task(vid.process_uploaded_video(save_path, det_conf, vs.id))
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)

    return schemas.VideoUploadResponse(
        session_id=vs.id,
        status="processing",
        message=f"Video '{fname}' is being processed. Check /api/video/sessions/{vs.id} for status.",
    )


@app.get("/api/video/sessions", response_model=schemas.VideoSessionsPage,
         summary="List video sessions")
async def list_video_sessions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    # Admin sees all org sessions; regular user sees only own sessions
    user_id_filter = None if (current_user and current_user.role == "admin") else (current_user.id if current_user else None)
    org_id_filter = current_user.organization_id or 1 if current_user else None
    items, total = await db.get_video_sessions_paginated(session, skip, limit, org_id=org_id_filter, user_id=user_id_filter)
    return schemas.VideoSessionsPage(total=total, skip=skip, limit=limit, items=items)


@app.get("/api/video/sessions/{session_id}", response_model=schemas.VideoSessionOut,
         summary="Get video session details")
async def get_video_session(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    vs = await db.get_video_session_by_id(session, session_id)
    if vs is None:
        raise HTTPException(status_code=404, detail="Video session not found.")
    if not _can_view_video_session(current_user, vs):
        raise HTTPException(status_code=403, detail="Acces restricționat la această sesiune video.")
    return vs


# ── ADMIN endpoints ───────────────────────────────────────────────────────────

@app.get("/api/admin/users", summary="[Admin] List all users with stats")
async def admin_list_users(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    org_id = current_user.organization_id or 1
    result = await session.execute(
        select(
            db.User,
            func.count(db.LitteringEvent.id).label("total_reports")
        )
        .outerjoin(db.LitteringEvent, db.LitteringEvent.reporter_id == db.User.id)
        .where(db.or_(db.User.organization_id == org_id, db.User.organization_id.is_(None)))
        .group_by(db.User.id)
        .order_by(db.User.points.desc())
    )
    rows = result.all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "points": u.points,
            "total_reports": total,
            "created_at": u.created_at.isoformat() if u.created_at else None,
        }
        for u, total in rows
    ]


@app.patch("/api/admin/users/{user_id}", summary="[Admin] Update user role or points")
async def admin_update_user(
    user_id: int,
    body: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Nu poți modifica propriul cont.")
    result = await session.execute(select(db.User).where(db.User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilizatorul nu a fost găsit.")
    allowed_roles = {"user", "admin"}
    if "role" in body and body["role"] in allowed_roles:
        user.role = body["role"]
    if "points" in body and isinstance(body["points"], int):
        user.points = max(0, body["points"])
    await session.commit()
    await session.refresh(user)
    return {"id": user.id, "username": user.username, "role": user.role, "points": user.points}


@app.post("/api/admin/users/invite", summary="[Admin] Invite a new user to the organization")
async def admin_invite_user(
    body: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    """Create a new user in the same org as the admin. Returns generated temp password."""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii pot invita.")

    import secrets, string
    from backend import auth as auth_mod

    username = (body.get("username") or "").strip()
    email = (body.get("email") or "").strip()
    role = (body.get("role") or "user").strip()
    if role not in ("user", "admin"):
        role = "user"
    if not username or not email:
        raise HTTPException(status_code=422, detail="Numele de utilizator și emailul sunt obligatorii.")

    # Check duplicates
    existing = await session.execute(
        select(db.User).where(db.or_(db.User.username == username, db.User.email == email))
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Numele de utilizator sau emailul este deja folosit.")

    # Generate temp password (12 chars, mixed)
    alphabet = string.ascii_letters + string.digits + "!@#$"
    temp_password = ''.join(secrets.choice(alphabet) for _ in range(12))
    # Ensure password meets policy: at least one upper, lower, digit, special
    temp_password = temp_password[:8] + "Aa1!"

    org_id = current_user.organization_id or 1
    new_user = db.User(
        username=username,
        email=email,
        hashed_password=auth_mod.get_password_hash(temp_password),
        role=role,
        organization_id=org_id,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    return {
        "id": new_user.id,
        "username": new_user.username,
        "email": new_user.email,
        "role": new_user.role,
        "dev_password": temp_password,
        "message": "Utilizator creat. Trimite parola temporară manual.",
    }


@app.delete("/api/admin/users/{user_id}", response_model=schemas.DetailResponse, summary="[Admin] Delete a user account")
async def admin_delete_user(
    user_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Nu poți șterge propriul cont.")
    result = await session.execute(select(db.User).where(db.User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="Utilizatorul nu a fost găsit.")
    await session.delete(user)
    await session.commit()
    return {"detail": f"Utilizatorul '{user.username}' a fost șters."}


@app.get("/api/admin/stats", response_model=schemas.AdminStats, summary="[Admin] Global platform stats")
async def admin_stats(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    org_id = current_user.organization_id or 1
    org_event_cond = db.or_(db.LitteringEvent.organization_id == org_id, db.LitteringEvent.organization_id.is_(None))
    org_user_cond = db.or_(db.User.organization_id == org_id, db.User.organization_id.is_(None))

    user_count = await session.scalar(select(func.count(db.User.id)).where(org_user_cond))
    total_incidents = await session.scalar(
        select(func.count(db.LitteringEvent.id)).where(org_event_cond)
    )
    resolved_count = await session.scalar(
        select(func.count(db.LitteringEvent.id)).where(
            org_event_cond,
            db.LitteringEvent.status.in_(["reviewed", "forwarded", "dismissed"]),
        )
    )

    return {
        "total_users": user_count,
        "total_sessions": total_incidents or 0,
        "total_objects": total_incidents or 0,
        "resolved_reports": resolved_count,
        "avg_inference_ms": 0.0,
    }



# ── Admin: Charts data (registrations per month, reports per day, materials) ──

@app.get("/api/admin/charts", summary="[Admin] Chart data for admin dashboard")
async def admin_charts(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    
    from datetime import timedelta

    org_id = current_user.organization_id or 1
    event_cond = db.or_(db.LitteringEvent.organization_id == org_id, db.LitteringEvent.organization_id.is_(None))
    user_cond = db.or_(db.User.organization_id == org_id, db.User.organization_id.is_(None))
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    thirty_days_ago = today - timedelta(days=29)

    # Incidents per day (last 30 days)
    reports_per_day = await session.execute(
        select(
            func.date(db.LitteringEvent.detected_at).label("day"),
            func.count(db.LitteringEvent.id).label("count"),
        )
        .where(event_cond, db.LitteringEvent.detected_at >= thirty_days_ago)
        .group_by(func.date(db.LitteringEvent.detected_at))
        .order_by(func.date(db.LitteringEvent.detected_at))
    )
    day_map = {str(r.day): r.count for r in reports_per_day}
    reports_timeline = [
        {
            "day": (today - timedelta(days=29 - i)).date().isoformat(),
            "count": day_map.get((today - timedelta(days=29 - i)).date().isoformat(), 0),
        }
        for i in range(30)
    ]

    # Users per month (all time)
    users_per_month = await session.execute(
        select(
            func.strftime("%Y-%m", db.User.created_at).label("month"),
            func.count(db.User.id).label("count"),
        )
        .where(user_cond)
        .group_by(func.strftime("%Y-%m", db.User.created_at))
        .order_by(func.strftime("%Y-%m", db.User.created_at))
    )
    users_timeline = [{"month": r.month, "count": r.count} for r in users_per_month]

    # Materials distribution (all time)
    materials = await session.execute(
        select(
            db.LitteringEvent.material,
            func.count(db.LitteringEvent.id).label("count"),
        )
        .where(event_cond)
        .group_by(db.LitteringEvent.material)
        .order_by(func.count(db.LitteringEvent.id).desc())
    )
    material_dist = [{"material": r.material or "unknown", "count": r.count} for r in materials]

    # Resolution rate
    total_reports = await session.scalar(
        select(func.count(db.LitteringEvent.id)).where(event_cond)
    )
    resolved_reports = await session.scalar(
        select(func.count(db.LitteringEvent.id)).where(
            event_cond,
            db.LitteringEvent.status.in_(["reviewed", "forwarded", "dismissed"]),
        )
    )

    return {
        "reports_timeline": reports_timeline,
        "users_timeline": users_timeline,
        "material_distribution": material_dist,
        "resolution_rate": {
            "resolved": resolved_reports or 0,
            "unresolved": (total_reports or 0) - (resolved_reports or 0),
        },
    }


# ── Admin: Export users CSV ───────────────────────────────────────────────────

@app.get("/api/admin/export/users", summary="[Admin] Export users as CSV")
async def admin_export_users_csv(
    request: Request,
    session: AsyncSession = Depends(db.get_db),
    token: Optional[str] = Query(default=None),
):
    # Accept token from Authorization header or query param (for download links)
    user = None
    auth_header = request.headers.get("Authorization", "")
    raw_token = token
    if auth_header.startswith("Bearer "):
        raw_token = auth_header[7:]
    if raw_token:
        try:
            payload = decode_access_token(raw_token)
            username = payload.get("username")
            if username:
                result = await session.execute(select(db.User).where(db.User.username == username))
                user = result.scalar_one_or_none()
        except Exception:
            pass
    if not user or user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    import csv
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["ID", "Utilizator", "Email", "Rol", "Puncte", "Incidente", "Creat"])
    result = await session.execute(
        select(db.User, func.count(db.LitteringEvent.id).label("total_reports"))
        .outerjoin(db.LitteringEvent, db.LitteringEvent.reporter_id == db.User.id)
        .group_by(db.User.id)
        .order_by(db.User.id)
    )
    for u, total in result:
        writer.writerow([u.id, u.username, u.email, u.role, u.points, total,
                         u.created_at.strftime("%Y-%m-%d %H:%M") if u.created_at else ""])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=users_export.csv"},
    )


# ── Notifications ─────────────────────────────────────────────────────────────

@app.get("/api/me/notifications", response_model=schemas.NotificationsResponse, summary="Get notifications for the current user")
async def get_notifications(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
    limit: int = Query(default=20, ge=1, le=100),
):
    rows = await session.execute(
        select(db.Notification)
        .where(db.Notification.user_id == current_user.id)
        .order_by(db.Notification.created_at.desc())
        .limit(limit)
    )
    notifications = rows.scalars().all()
    unread = sum(1 for n in notifications if n.is_read == 0)
    return {
        "unread": unread,
        "notifications": [
            {
                "id": n.id,
                "message": n.message,
                "category": n.category,
                "session_id": n.session_id,
                "is_read": n.is_read,
                "created_at": n.created_at.isoformat() if n.created_at else "",
            }
            for n in notifications
        ],
    }


@app.post("/api/me/notifications/{notification_id}/read", response_model=schemas.OkResponse, summary="Mark a notification as read")
async def mark_notification_read(
    notification_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    row = await session.execute(
        select(db.Notification)
        .where(db.Notification.id == notification_id)
        .where(db.Notification.user_id == current_user.id)
    )
    notif = row.scalar_one_or_none()
    if notif is None:
        raise HTTPException(status_code=404, detail="Notificarea nu a fost găsită.")
    notif.is_read = 1
    await session.commit()
    return {"ok": True}


@app.post("/api/me/notifications/read-all", response_model=schemas.OkResponse, summary="Mark all notifications as read")
async def mark_all_notifications_read(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    await session.execute(
        update(db.Notification)
        .where(db.Notification.user_id == current_user.id)
        .where(db.Notification.is_read == 0)
        .values(is_read=1)
    )
    await session.commit()
    return {"ok": True}


@app.delete("/api/video/sessions/{session_id}", response_model=schemas.DetailResponse, summary="Delete a video session and files")
async def delete_video_session(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    vs = await db.get_video_session_by_id(session, session_id)
    if vs is None:
        raise HTTPException(status_code=404, detail="Video session not found.")
    # Admin poate șterge orice sesiune din org; userul poate șterge doar ale lui
    is_owner = vs.user_id == current_user.id
    is_admin = current_user.role == "admin"
    same_org = (vs.organization_id or 1) == (current_user.organization_id or 1)
    if not ((is_admin and same_org) or is_owner):
        raise HTTPException(status_code=403, detail="Nu poți șterge o sesiune care nu îți aparține.")

    for path_str in (vs.video_path, vs.annotated_video_path):
        if path_str:
            p = Path(path_str)
            if p.exists():
                p.unlink()

    await session.delete(vs)
    await session.commit()
    return {"detail": f"Video session {session_id} deleted."}


@app.get("/api/video/sessions/{session_id}/download",
         summary="Download the annotated video file")
async def download_annotated_video(
    session_id: int,
    request: Request,
    token: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(db.get_db),
):
    current_user = await _user_from_bearer_or_query(request, session, token)
    if current_user is None:
        raise HTTPException(status_code=401, detail="Autentificare necesară.")
    vs = await db.get_video_session_by_id(session, session_id)
    if vs is None:
        raise HTTPException(status_code=404, detail="Video session not found.")
    if not _can_view_video_session(current_user, vs):
        raise HTTPException(status_code=403, detail="Acces restricționat la această sesiune video.")
    if not vs.annotated_video_path:
        raise HTTPException(status_code=404, detail="Annotated video not yet available.")

    p = Path(vs.annotated_video_path)
    if not p.exists():
        raise HTTPException(status_code=410, detail="Annotated video file was deleted.")

    return FileResponse(p, media_type="video/mp4", filename=p.name)


# ---------------------------------------------------------------------------
# Phase B: File Management
# ---------------------------------------------------------------------------

# ── B4: Admin storage stats ──────────────────────────────────────────────────

@app.get("/api/admin/storage", summary="[Admin] Disk storage stats")
async def admin_storage_stats(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")

    def _dir_stats(d: Path):
        if not d.exists():
            return {"count": 0, "bytes": 0, "size_mb": 0}
        files = [p for p in d.rglob("*") if p.is_file()]
        total = sum(f.stat().st_size for f in files)
        return {"count": len(files), "bytes": total, "size_mb": round(total / 1024 / 1024, 2)}

    def _empty_status_stats():
        return {"events": 0, "files": 0, "bytes": 0, "size_mb": 0}

    evidence = _dir_stats(LITTERING_DIR)
    uploads = _dir_stats(UPLOADS_DIR)
    annotated = _dir_stats(ANNOTATED_DIR)
    videos = _dir_stats(VIDEOS_DIR)
    status_keys = ("pending", "reviewed", "forwarded", "dismissed", "other")
    evidence_by_status = {status: _empty_status_stats() for status in status_keys}
    seen_by_status: dict[str, set[str]] = {status: set() for status in status_keys}

    org_id = current_user.organization_id or 1
    result = await session.execute(
        select(db.LitteringEvent).where(
            db.or_(
                db.LitteringEvent.organization_id == org_id,
                db.LitteringEvent.organization_id.is_(None),
            )
        )
    )
    for event in result.scalars().all():
        status = event.status if event.status in evidence_by_status else "other"
        status_stats = evidence_by_status[status]
        status_stats["events"] += 1
        for relative_path in (event.clip_path, event.thumbnail_path):
            candidate = _resolve_littering_evidence_path(relative_path)
            if candidate is None or not candidate.exists() or not candidate.is_file():
                continue
            resolved = str(candidate)
            if resolved in seen_by_status[status]:
                continue
            seen_by_status[status].add(resolved)
            status_stats["files"] += 1
            status_stats["bytes"] += candidate.stat().st_size

    tracked_evidence_bytes = 0
    tracked_evidence_files = 0
    for status_stats in evidence_by_status.values():
        status_stats["size_mb"] = round(status_stats["bytes"] / 1024 / 1024, 2)
        tracked_evidence_bytes += status_stats["bytes"]
        tracked_evidence_files += status_stats["files"]

    return {
        "uploads": uploads,
        "annotated": annotated,
        "videos": videos,
        "evidence": evidence,
        "evidence_by_status": evidence_by_status,
        "uploads_bytes": uploads["bytes"],
        "annotated_bytes": annotated["bytes"],
        "videos_bytes": videos["bytes"],
        "evidence_bytes": evidence["bytes"],
        "evidence_tracked_bytes": tracked_evidence_bytes,
        "evidence_tracked_files": tracked_evidence_files,
        "evidence_orphan_bytes": max(evidence["bytes"] - tracked_evidence_bytes, 0),
    }


# ── Organization endpoints ────────────────────────────────────────────────────

@app.get("/api/me/organization", summary="Current user's organization info + trial status")
async def get_my_organization(
    org: Annotated[db.Organization, Depends(get_current_org)],
):
    from datetime import timezone as _tz
    now = datetime.now(_tz.utc)
    trial_days_left = None
    trial_expired = False
    if org.plan == "trial" and org.trial_ends_at:
        delta = org.trial_ends_at - now
        trial_days_left = max(0, delta.days)
        trial_expired = delta.total_seconds() <= 0
    return {
        "id": org.id,
        "name": org.name,
        "plan": org.plan,
        "trial_ends_at": org.trial_ends_at.isoformat() if org.trial_ends_at else None,
        "trial_days_left": trial_days_left,
        "trial_expired": trial_expired,
        "subscription_active": org.subscription_active,
        "max_cameras": org.max_cameras,
        "max_incidents_month": org.max_incidents_month,
    }


@app.patch("/api/admin/organization", summary="[Admin] Update organization name or plan")
async def update_organization(
    body: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    org: Annotated[db.Organization, Depends(get_current_org)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    if "name" in body and body["name"].strip():
        org.name = body["name"].strip()
    if "plan" in body and body["plan"] in ("trial", "starter", "pro", "enterprise"):
        org.plan = body["plan"]
    await session.commit()
    await session.refresh(org)
    return {"id": org.id, "name": org.name, "plan": org.plan}
