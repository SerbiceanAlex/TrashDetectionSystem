"""
Aplicația FastAPI — interfața web a sistemului de detecție a gunoiului.

Pornire:
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
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend import database as db
from backend import inference as infer
from backend import schemas
from backend.auth_router import router as auth_router, get_current_active_user
from backend.auth import decode_access_token
from backend.config import settings
from backend.storage_retention import cleanup_littering_evidence, storage_cleanup_loop
from backend import video as vid

STATIC_DIR = settings.REPO_ROOT / "frontend" / "static"
TEMPLATES_DIR = settings.REPO_ROOT / "frontend" / "templates"

VIDEOS_DIR = settings.videos_dir
LITTERING_DIR = settings.littering_dir

for runtime_dir in settings.runtime_dirs:
    runtime_dir.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = settings.max_upload_bytes
VIDEO_MAX_UPLOAD_BYTES = settings.video_max_upload_bytes

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _resolve_littering_evidence_path(stored_path: str | None) -> Path | None:
    """Rezolvă o cale de dovadă din DB, păstrând accesul doar în LITTERING_DIR."""
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
        logger.warning("Ignor calea de dovadă din afara directorului de incidente: %s", stored_path)
        return None
    return next((path for path in allowed if path.exists()), allowed[0])


# ── Ciclul de viață: încarcă modelele + creează tabelele la pornire ──────────

async def _migrate_schema():
    """Adaugă coloane noi în tabelele existente (ALTER TABLE pe SQLite).

    Sigur de rulat de mai multe ori — fiecare ALTER e prins în try/except,
    deci nu face nimic dacă coloana există deja.
    """
    alter_statements = [
        # LitteringEvent — câmpuri de dovadă pe distanță (mașina de stări v2)
        "ALTER TABLE littering_events ADD COLUMN incident_uid VARCHAR(36)",
        "ALTER TABLE littering_events ADD COLUMN owner_person_id INTEGER",
        "ALTER TABLE littering_events ADD COLUMN distance_at_abandonment REAL",
        "ALTER TABLE littering_events ADD COLUMN detection_method VARCHAR(32) DEFAULT 'zone'",
        "ALTER TABLE littering_events ADD COLUMN source VARCHAR(16) DEFAULT 'live'",
        "ALTER TABLE littering_events ADD COLUMN reporter_id INTEGER REFERENCES users(id)",
        # Izolare pe organizație
        "ALTER TABLE users ADD COLUMN organization_id INTEGER REFERENCES organizations(id)",
        "ALTER TABLE littering_events ADD COLUMN organization_id INTEGER REFERENCES organizations(id)",
        # Izolarea sesiunilor video
        "ALTER TABLE video_sessions ADD COLUMN littering_count INTEGER DEFAULT 0",
        "ALTER TABLE video_sessions ADD COLUMN user_id INTEGER REFERENCES users(id)",
        "ALTER TABLE video_sessions ADD COLUMN organization_id INTEGER REFERENCES organizations(id)",
        # Notificări — incidentul vizat (pentru deschidere la click)
        "ALTER TABLE notifications ADD COLUMN event_id INTEGER REFERENCES littering_events(id)",
    ]
    async with db.engine.begin() as conn:
        for stmt in alter_statements:
            try:
                await conn.execute(db.sa_text(stmt))
            except Exception:
                pass  # coloana există deja — normal la rulările următoare

    # Asigură organizația implicită și atribuie rândurile vechi ei
    async with db.AsyncSessionLocal() as session:
        await db.get_or_create_default_org(session)
    async with db.engine.begin() as conn:
        for tbl in ("users", "littering_events", "video_sessions"):
            try:
                await conn.execute(db.sa_text(
                    f"UPDATE {tbl} SET organization_id = 1 WHERE organization_id IS NULL"
                ))
            except Exception:
                pass

    print("[migration] Migrarea schemei e completă.")


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
            logger.info("Curățarea stocării la pornire s-a încheiat: %s", summary)
        except Exception:
            logger.exception("Curățarea stocării la pornire a eșuat")
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
    description="API pentru detecția deșeurilor și clasificarea materialelor cu YOLO în două etape",
    version="1.0.0",
    lifespan=lifespan,
)

# Fișierele media generate sunt servite prin endpoint-uri autentificate,
# nu prin directoare statice publice.

# Înregistrează routerele aplicației.
app.include_router(auth_router)

# Servește aplicația frontend (SPA)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static")

@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/js/") or request.url.path.startswith("/static/css/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    # Sare automat peste pagina de avertizare ngrok din browser
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response


# ── Organization dependency ───────────────────────────────────────────────────

async def get_current_org(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
) -> db.Organization:
    """Întoarce organizația utilizatorului curent (o creează pe cea implicită dacă lipsește)."""
    if current_user.organization_id:
        org = await db.get_org_by_id(session, current_user.organization_id)
        if org:
            return org
    return await db.get_or_create_default_org(session)


# ── Helpers ───────────────────────────────────────────────────────────────────


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


@app.get("/api/system/info", summary="Metadate publice despre sistem/model")
async def system_info():
    """Întoarce metadate de runtime (sigure public) pentru panoul de sistem din frontend."""
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
            "monitor_person_conf": settings.MONITOR_PERSON_CONF,
            "monitor_target_fps": settings.MONITOR_TARGET_FPS,
            "monitor_logic_fps": settings.MONITOR_LOGIC_FPS,
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


# ── Endpoint-uri video ───────────────────────────────────────────────────────

# Singurul flux video live folosit de interfață este monitorul de incidente
# (/ws/video/monitor): detecție + logică temporală de abandonare.

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
    WebSocket pentru detecția incidentelor de aruncare (modul monitor).
    Browserul trimite cadre JPEG; serverul rulează tracker-ul de gunoi +
    detectorul de persoane și trimite o alertă JSON când apare un incident.
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
            settings.MONITOR_MIN_DET_CONF,
            settings.MONITOR_PERSON_CONF,
            analysis_fps,
            lat,
            lng,
            session,
            user_id=current_user.id if current_user else None,
            organization_id=(current_user.organization_id or 1) if current_user else 1,
        )


# ── REST pentru incidente ─────────────────────────────────────────────────────

@app.get(
    "/api/littering/events",
    response_model=schemas.LitteringEventsPage,
    summary="Listează incidentele după rolul utilizatorului curent",
)
async def list_littering_events(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None, description="Filtrare după status: pending/reviewed/forwarded/dismissed"),
    material: Optional[str] = Query(default=None),
    reporter_id: Optional[int] = Query(default=None, description="Filtru disponibil doar pentru admin, după ID-ul raportorului"),
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
    "/api/littering/events/{event_id}",
    response_model=schemas.LitteringEventOut,
    summary="Obține un incident după ID",
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


@app.delete(
    "/api/littering/events/{event_id}",
    response_model=schemas.DetailResponse,
    summary="[Admin] Șterge un incident și dovezile lui",
)
async def delete_littering_event(
    event_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    evt = await db.get_littering_event_by_id(session, event_id)
    if evt is None:
        raise HTTPException(status_code=404, detail="Eveniment negăsit.")
    if not _same_org(current_user, evt):
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    # Șterge dovezile de pe disc (clip + miniatură) înainte de rândul din DB.
    for rel in (evt.clip_path, evt.thumbnail_path):
        if rel:
            try:
                (LITTERING_DIR / rel).unlink(missing_ok=True)
            except Exception:
                logger.warning("Nu am putut șterge fișierul de dovadă: %s", rel)
    await session.delete(evt)
    await session.commit()
    logger.info("Incident #%d șters de %s.", event_id, current_user.username)
    return schemas.DetailResponse(detail="Incident șters.")


@app.patch(
    "/api/littering/events/{event_id}/status",
    response_model=schemas.LitteringEventOut,
    summary="[Admin] Actualizează statusul unui incident",
)
async def update_littering_event_status(
    event_id: int,
    body: schemas.LitteringEventStatusUpdate,
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
            detail=f"Status nevalid. Valori permise: {', '.join(sorted(allowed_statuses))}"
        )
    evt = await db.update_littering_event_status(
        session, event_id,
        status=body.status,
        reviewed_by=current_user.id,
        notes=body.notes,
    )
    if evt is None:
        raise HTTPException(status_code=404, detail="Eveniment negăsit.")

    # În varianta de licență, "forwarded" e un status local de arhivă/pregătit.
    # Aplicația nu trimite niciun email în exterior.
    if body.status == "forwarded":
        logger.info("Incident #%d arhivat local; livrarea prin email este dezactivată.", evt.id)

    return evt


@app.patch(
    "/api/littering/events/{event_id}/notes",
    response_model=schemas.LitteringEventOut,
    summary="[Admin] Actualizează notele unui incident",
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


@app.patch(
    "/api/littering/events/{event_id}/material",
    response_model=schemas.LitteringEventOut,
    summary="[Admin] Corectează materialul estimat al unui incident",
)
async def update_littering_event_material(
    event_id: int,
    body: schemas.LitteringEventMaterialUpdate,
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat.")
    evt = await db.get_littering_event_by_id(session, event_id)
    if evt is None:
        raise HTTPException(status_code=404, detail="Eveniment negăsit.")
    if not _same_org(current_user, evt):
        raise HTTPException(status_code=403, detail="Acces restricționat.")

    material = (body.material or "").strip().lower()
    allowed_materials = {"unknown", "plastic", "paper", "glass", "metal", "other"}
    if material not in allowed_materials:
        raise HTTPException(
            status_code=400,
            detail=f"Material nevalid. Valori permise: {', '.join(sorted(allowed_materials))}"
        )

    evt.material = material
    await session.commit()
    await session.refresh(evt)
    return evt


@app.get(
    "/api/littering/events/{event_id}/clip",
    summary="Descarcă clipul unui incident",
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
    summary="Obține miniatura unui incident",
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
# Dashboard B2B / locații / rapoarte — endpoint-uri pentru produsul B2B
# ---------------------------------------------------------------------------

@app.get("/api/dashboard/b2b", summary="Dashboard B2B — KPI + trend + incidente recente")
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

    # Incidente recente (ultimele 5) — limitate la organizație
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

    # Distribuția materialelor (ultimele 30 zile) — pe organizație
    mat_q = await session.execute(
        db.select(LE.material, db.func.count())
        .where(role_cond, LE.detected_at >= month_start)
        .group_by(LE.material)
        .order_by(db.func.count().desc())
    )
    material_distribution = [
        {"material": m or "unknown", "count": c} for m, c in mat_q.all()
    ]

    # Tendința pe ultimele 30 zile — pe organizație
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

    # Distribuția pe ore — pe organizație
    hour_q = await session.execute(
        db.select(db.func.strftime('%H', LE.detected_at), db.func.count())
        .where(role_cond)
        .group_by(db.func.strftime('%H', LE.detected_at))
    )
    hour_map = {int(h): c for h, c in hour_q.all() if h is not None}
    hourly_distribution = [hour_map.get(h, 0) for h in range(24)]

    # Rata de rezolvare (reviewed + forwarded) / total — pe organizație
    resolved = await count_where(LE.status.in_(["reviewed", "forwarded", "dismissed"]))
    total_all = await count_where()
    resolution_rate = round(resolved / max(total_all, 1) * 100)

    return {
        "incidents_today": incidents_today,
        "incidents_week": incidents_week,
        "incidents_month": incidents_month,
        "pending_review": pending_review,
        "forwarded": forwarded,
        "recent_incidents": recent_incidents,
        "material_distribution": material_distribution,
        "trend_30d": trend_30d,
        "hourly_distribution": hourly_distribution,
        "resolution_rate": resolution_rate,
        "total_all_time": total_all,
    }


@app.get("/api/reports/export", summary="Exportă raport (CSV / PDF / ZIP)")
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
    """Exportă incidentele ca CSV, PDF sau ZIP."""
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
            from reportlab.lib.enums import TA_CENTER
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
                    pass  # variantă de rezervă: Helvetica cu diacritice eliminate

            def _txt(s: str) -> str:
                """Elimină diacriticele dacă fontul nu are suport Unicode."""
                if _FONT_NORMAL == "Arial":
                    return str(s) if s else ""
                # Variantă de rezervă Helvetica: înlocuim diacriticele.
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
            # metadate JSON
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
          summary="Încarcă un fișier video pentru procesare offline")
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
    # Acceptă și după extensie dacă MIME-ul lipsește sau este necunoscut.
    ext = Path(fname).suffix.lower()
    if ct not in allowed and ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail=f"Tip video neacceptat: {ct}")

    stem = uuid.uuid4().hex
    save_path = VIDEOS_DIR / f"{stem}{ext or '.mp4'}"

    # Scrie pe disc în bucăți de 1 MB — evită încărcarea întregului fișier în RAM
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
            detail=f"Video prea mare. Dimensiunea maximă este {settings.VIDEO_MAX_UPLOAD_MB} MB.",
        )
    if video_empty:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Fișierul încărcat este gol.")

    vs = await db.create_video_session(session, source_type="upload", filename=fname)
    vs.video_path = str(save_path)
    vs.user_id = current_user.id
    vs.organization_id = current_user.organization_id or 1
    await session.commit()

    # Procesează în fundal, cu logarea erorilor.
    task = asyncio.create_task(vid.process_uploaded_video(save_path, det_conf, vs.id))
    task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)

    return schemas.VideoUploadResponse(
        session_id=vs.id,
        status="processing",
        message=f"Video-ul '{fname}' se procesează. Verifică /api/video/sessions/{vs.id} pentru status.",
    )


@app.get("/api/video/sessions", response_model=schemas.VideoSessionsPage,
         summary="Listează sesiunile video")
async def list_video_sessions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    # Adminul vede toate sesiunile organizației; userul normal doar pe ale lui
    user_id_filter = None if (current_user and current_user.role == "admin") else (current_user.id if current_user else None)
    org_id_filter = current_user.organization_id or 1 if current_user else None
    items, total = await db.get_video_sessions_paginated(session, skip, limit, org_id=org_id_filter, user_id=user_id_filter)
    return schemas.VideoSessionsPage(total=total, skip=skip, limit=limit, items=items)


@app.get("/api/video/sessions/{session_id}", response_model=schemas.VideoSessionOut,
         summary="Detaliile unei sesiuni video")
async def get_video_session(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    vs = await db.get_video_session_by_id(session, session_id)
    if vs is None:
        raise HTTPException(status_code=404, detail="Sesiunea video nu a fost găsită.")
    if not _can_view_video_session(current_user, vs):
        raise HTTPException(status_code=403, detail="Acces restricționat la această sesiune video.")
    return vs


# ── Endpoint-uri ADMIN ───────────────────────────────────────────────────────

@app.get("/api/admin/users", summary="[Admin] Listează toți utilizatorii cu statistici")
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
            func.count(db.LitteringEvent.id).label("total_reports"),
            func.coalesce(
                func.sum(case((db.LitteringEvent.status == "reviewed", 1), else_=0)), 0
            ).label("confirmed_reports"),
            func.coalesce(
                func.sum(case((db.LitteringEvent.status == "pending", 1), else_=0)), 0
            ).label("pending_reports"),
            func.max(db.LitteringEvent.detected_at).label("last_incident_at"),
        )
        .outerjoin(db.LitteringEvent, db.LitteringEvent.reporter_id == db.User.id)
        .where(db.or_(db.User.organization_id == org_id, db.User.organization_id.is_(None)))
        .group_by(db.User.id)
        .order_by(db.User.points.desc(), db.User.created_at.asc(), db.User.id.asc())
    )
    rows = result.all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role,
            "points": u.points,
            "total_reports": int(total or 0),
            "confirmed_reports": int(confirmed or 0),
            "pending_reports": int(pending or 0),
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_activity_at": (last_incident_at or u.created_at).isoformat() if (last_incident_at or u.created_at) else None,
        }
        for u, total, confirmed, pending, last_incident_at in rows
    ]


@app.patch("/api/admin/users/{user_id}", summary="[Admin] Actualizează rolul sau punctele unui utilizator")
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
    if (user.organization_id or 1) != (current_user.organization_id or 1):
        raise HTTPException(status_code=403, detail="Utilizatorul aparține altei organizații.")

    if "username" in body:
        username = str(body.get("username") or "").strip()
        if not username:
            raise HTTPException(status_code=422, detail="Numele de utilizator este obligatoriu.")
        duplicate = await session.scalar(
            select(db.User).where(db.User.username == username, db.User.id != user_id)
        )
        if duplicate is not None:
            raise HTTPException(status_code=400, detail="Numele de utilizator este deja folosit.")
        user.username = username

    if "email" in body:
        email = str(body.get("email") or "").strip()
        if not email:
            raise HTTPException(status_code=422, detail="Emailul este obligatoriu.")
        duplicate = await session.scalar(
            select(db.User).where(db.User.email == email, db.User.id != user_id)
        )
        if duplicate is not None:
            raise HTTPException(status_code=400, detail="Emailul este deja folosit.")
        user.email = email

    allowed_roles = {"user", "admin"}
    if "role" in body and body["role"] in allowed_roles:
        user.role = body["role"]
    if "points" in body and isinstance(body["points"], int):
        user.points = max(0, body["points"])
    await session.commit()
    await session.refresh(user)
    return {"id": user.id, "username": user.username, "email": user.email, "role": user.role, "points": user.points}


@app.post("/api/admin/users/invite", summary="[Admin] Invită un utilizator nou în organizație")
async def admin_invite_user(
    body: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    """Creează un utilizator nou în organizația adminului. Întoarce parola temporară generată."""
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

    # Verifică duplicatele
    existing = await session.execute(
        select(db.User).where(db.or_(db.User.username == username, db.User.email == email))
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Numele de utilizator sau emailul este deja folosit.")

    # Generează o parolă temporară de 14 caractere, complet aleatoare, garantând
    # politica (literă mare/mică, cifră, simbol) fără un sufix fix previzibil.
    # Reîncearcă până când conține toate clasele cerute — fără pattern ghicibil.
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    while True:
        temp_password = ''.join(secrets.choice(alphabet) for _ in range(14))
        if not auth_mod.validate_password(temp_password):
            break

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


@app.delete("/api/admin/users/{user_id}", response_model=schemas.DetailResponse, summary="[Admin] Șterge un cont de utilizator")
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
    if (user.organization_id or 1) != (current_user.organization_id or 1):
        raise HTTPException(status_code=403, detail="Utilizatorul aparține altei organizații.")
    await session.delete(user)
    await session.commit()
    return {"detail": f"Utilizatorul '{user.username}' a fost șters."}


@app.get("/api/admin/stats", response_model=schemas.AdminStats, summary="[Admin] Statistici globale ale platformei")
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


# ── Admin: date pentru grafice (înregistrări/lună, rapoarte/zi, materiale) ──

@app.get("/api/admin/charts", summary="[Admin] Date pentru graficele din dashboard")
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

    # Incidente pe zi (ultimele 30 zile)
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

    # Utilizatori pe lună (din totdeauna)
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

    # Distribuția materialelor (din totdeauna)
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

    # Rata de rezolvare
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


# ── Admin: export utilizatori în CSV ─────────────────────────────────────────

@app.get("/api/admin/export/users", summary="[Admin] Exportă utilizatorii în CSV")
async def admin_export_users_csv(
    request: Request,
    session: AsyncSession = Depends(db.get_db),
    token: Optional[str] = Query(default=None),
):
    # Acceptă token din antetul Authorization sau din query (pentru linkuri de descărcare)
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
    org_id = user.organization_id or 1
    result = await session.execute(
        select(db.User, func.count(db.LitteringEvent.id).label("total_reports"))
        .outerjoin(db.LitteringEvent, db.LitteringEvent.reporter_id == db.User.id)
        .where(db.or_(db.User.organization_id == org_id, db.User.organization_id.is_(None)))
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


# ── Notificări ────────────────────────────────────────────────────────────────

@app.get("/api/me/notifications", response_model=schemas.NotificationsResponse, summary="Obține notificările utilizatorului curent")
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
                "event_id": n.event_id,
                "is_read": n.is_read,
                "created_at": n.created_at.isoformat() if n.created_at else "",
            }
            for n in notifications
        ],
    }


@app.post("/api/me/notifications/{notification_id}/read", response_model=schemas.OkResponse, summary="Marchează o notificare ca citită")
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


@app.post("/api/me/notifications/read-all", response_model=schemas.OkResponse, summary="Marchează toate notificările ca citite")
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


def _delete_video_session_files(vs: db.VideoSession) -> None:
    for path_str in (vs.video_path, vs.annotated_video_path):
        if path_str:
            p = Path(path_str)
            if p.exists():
                p.unlink()


@app.delete("/api/video/sessions", response_model=schemas.DetailResponse, summary="Curăță istoricul procesărilor video")
async def clear_video_sessions(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    org_id = current_user.organization_id or 1
    query = select(db.VideoSession).where(db.VideoSession.status.notin_(["running", "processing"]))
    if current_user.role == "admin":
        query = query.where(db.or_(db.VideoSession.organization_id == org_id, db.VideoSession.organization_id.is_(None)))
    else:
        query = query.where(db.VideoSession.user_id == current_user.id)

    rows = (await session.execute(query)).scalars().all()
    for vs in rows:
        _delete_video_session_files(vs)
        await session.delete(vs)
    await session.commit()
    return {"detail": f"Istoricul video a fost curățat: {len(rows)} sesiune(i) șterse."}


@app.delete("/api/video/sessions/{session_id}", response_model=schemas.DetailResponse, summary="Șterge o sesiune video și fișierele ei")
async def delete_video_session(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    vs = await db.get_video_session_by_id(session, session_id)
    if vs is None:
        raise HTTPException(status_code=404, detail="Sesiunea video nu a fost găsită.")
    # Admin poate șterge orice sesiune din org; userul poate șterge doar ale lui
    is_owner = vs.user_id == current_user.id
    is_admin = current_user.role == "admin"
    same_org = (vs.organization_id or 1) == (current_user.organization_id or 1)
    if not ((is_admin and same_org) or is_owner):
        raise HTTPException(status_code=403, detail="Nu poți șterge o sesiune care nu îți aparține.")

    _delete_video_session_files(vs)
    await session.delete(vs)
    await session.commit()
    return {"detail": f"Sesiunea video {session_id} a fost ștearsă."}


@app.get("/api/video/sessions/{session_id}/download",
         summary="Descarcă fișierul video adnotat")
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
        raise HTTPException(status_code=404, detail="Sesiunea video nu a fost găsită.")
    if not _can_view_video_session(current_user, vs):
        raise HTTPException(status_code=403, detail="Acces restricționat la această sesiune video.")
    if not vs.annotated_video_path:
        raise HTTPException(status_code=404, detail="Video-ul adnotat nu este încă disponibil.")

    p = Path(vs.annotated_video_path)
    if not p.exists():
        raise HTTPException(status_code=410, detail="Fișierul video adnotat a fost șters.")

    return FileResponse(p, media_type="video/mp4", filename=p.name)


# ---------------------------------------------------------------------------
# Faza B: gestionarea fișierelor
# ---------------------------------------------------------------------------

# ── B4: statistici de stocare (admin) ────────────────────────────────────────

@app.get("/api/admin/storage", summary="[Admin] Statistici de stocare pe disc")
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
        "videos": videos,
        "evidence": evidence,
        "evidence_by_status": evidence_by_status,
        "videos_bytes": videos["bytes"],
        "evidence_bytes": evidence["bytes"],
        "evidence_tracked_bytes": tracked_evidence_bytes,
        "evidence_tracked_files": tracked_evidence_files,
        "evidence_orphan_bytes": max(evidence["bytes"] - tracked_evidence_bytes, 0),
    }


# ── Endpoint-uri organizație ──────────────────────────────────────────────────

@app.get("/api/me/organization", summary="Informații despre organizația utilizatorului curent")
async def get_my_organization(
    org: Annotated[db.Organization, Depends(get_current_org)],
):
    return {"id": org.id, "name": org.name}


@app.patch("/api/admin/organization", summary="[Admin] Actualizează numele organizației")
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
    await session.commit()
    await session.refresh(org)
    return {"id": org.id, "name": org.name}
