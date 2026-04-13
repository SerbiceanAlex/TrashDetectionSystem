"""
FastAPI application — Trash Detection System web interface.

Start with:
    .venv\\Scripts\\uvicorn backend.main:app --reload --port 8000
"""

import asyncio
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Annotated, Optional
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import case, delete as sa_delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend import database as db
from backend import geo
from backend import inference as infer
from backend import schemas
from backend.auth_router import router as auth_router, get_current_active_user, oauth2_scheme
from backend.auth import decode_access_token
from backend.config import settings
from backend import video as vid
from backend import ecoscore as eco
from backend import notifications as notif

APP_DIR = Path(__file__).parent
UPLOADS_DIR = APP_DIR / "uploads"
ANNOTATED_DIR = APP_DIR / "annotated"
VIDEOS_DIR = APP_DIR / "videos"
STATIC_DIR = settings.REPO_ROOT / "frontend" / "static"
TEMPLATES_DIR = settings.REPO_ROOT / "frontend" / "templates"

UPLOADS_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

MAX_UPLOAD_BYTES = settings.max_upload_bytes

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Lifespan: load models + create DB tables on startup ──────────────────────

async def _migrate_schema():
    """Add new columns to existing tables (SQLite ALTER TABLE).
    
    Safe to run repeatedly — each ALTER is wrapped in try/except
    so it's a no-op if the column already exists.
    """
    alter_statements = [
        # User table — new community fields
        "ALTER TABLE users ADD COLUMN eco_score INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN rank VARCHAR(32) DEFAULT 'Novice'",
        "ALTER TABLE users ADD COLUMN streak_days INTEGER DEFAULT 0",
        "ALTER TABLE users ADD COLUMN last_active_date DATE",
        "ALTER TABLE users ADD COLUMN anonymous_reports BOOLEAN DEFAULT 0",
        "ALTER TABLE users ADD COLUMN hide_exact_location BOOLEAN DEFAULT 0",
        "ALTER TABLE users ADD COLUMN trust_weight REAL DEFAULT 1.0",
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
        # User table — Phase A/C fields
        "ALTER TABLE users ADD COLUMN avatar_path TEXT",
        "ALTER TABLE users ADD COLUMN onboarding_done BOOLEAN DEFAULT 0",
        "ALTER TABLE users ADD COLUMN authority_area_lat REAL",
        "ALTER TABLE users ADD COLUMN authority_area_lng REAL",
        "ALTER TABLE users ADD COLUMN authority_area_radius_km REAL DEFAULT 10.0",
        # DetectionRecord — impact metrics
        "ALTER TABLE detection_records ADD COLUMN estimated_weight_kg REAL DEFAULT 0.0",
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
    # Sync existing points → eco_score for users that haven't been migrated
    async with db.engine.begin() as conn:
        await conn.execute(
            db.sa_text(
                "UPDATE users SET eco_score = points "
                "WHERE eco_score = 0 AND points > 0"
            )
        )

    print("[migration] Schema migration complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.create_tables()
    await _migrate_schema()
    infer.load_models()

    # Background task: streak motivation + auto-expire pending reports
    async def _background_checks():
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                async with db.AsyncSessionLocal() as session:
                    # Motivation notifications for users about to lose streak
                    from datetime import date
                    users = await session.execute(
                        select(db.User).where(db.User.streak_days >= 3)
                    )
                    for user in users.scalars():
                        await notif.notify_motivation(session, user)

                    # Auto-expire old pending reports with no votes
                    pending = await session.execute(
                        select(db.DetectionSession)
                        .where(db.DetectionSession.status == "pending")
                        .where(db.DetectionSession.upload_time <= datetime.now(timezone.utc) - __import__('datetime').timedelta(hours=72))
                    )
                    for det_s in pending.scalars():
                        await eco.check_auto_expire(session, det_s)
                        await eco.check_auto_verify(session, det_s)

                    await session.commit()
            except asyncio.CancelledError:
                break
            except Exception:
                pass  # Don't crash background task

    bg_task = asyncio.create_task(_background_checks())
    yield
    bg_task.cancel()


app = FastAPI(
    title="Trash Detection System",
    description="Two-stage YOLO-based trash detection and material classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve annotated images at /annotated/<filename>
app.mount("/annotated", StaticFiles(directory=str(ANNOTATED_DIR)), name="annotated")

# Serve annotated videos at /videos/<filename>
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")

# Include Routers
app.include_router(auth_router)

# Serve the frontend SPA
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_files(original_bytes: bytes, annotated_bytes: bytes, stem: str):
    """Write original + annotated images to disk (runs as a background task)."""
    (UPLOADS_DIR / f"{stem}.jpg").write_bytes(original_bytes)
    (ANNOTATED_DIR / f"{stem}_annotated.jpg").write_bytes(annotated_bytes)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="base.html",
        context={}
    )


@app.get("/public/map", response_class=HTMLResponse, include_in_schema=False,
         summary="Public map page — shareable, no auth required")
async def public_map_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="public_map.html",
        context={}
    )


@app.post("/api/detect", response_model=schemas.DetectResponse, summary="Upload image and run detection")
async def detect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    det_conf: float = Query(default=0.50, ge=0.05, le=0.95, description="Detector confidence threshold"),
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
    # Optional User Auth for points
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

    # ── Determine location (EXIF GPS first, browser GPS fallback) ────────────
    location = await geo.get_image_location(
        image_bytes,
        fallback_lat=latitude,
        fallback_lng=longitude,
    )
    final_lat = location["latitude"]
    final_lng = location["longitude"]
    address   = location["address"]
    gps_src   = location["gps_source"]

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
        reporter_id=current_user.id if current_user else None,
        user_note=user_note.strip()[:500] if user_note else None
    )
    if current_user:
        base_pts = eco.POINTS_REPORT_PER_OBJECT * len(detections)
        actual_pts, ranked_up, new_rank = await eco.award_ecoscore(session, current_user, base_pts)
        # Smart notifications for rank up and streak milestones
        if ranked_up:
            await notif.notify_rank_up(session, current_user.id, new_rank)
        await notif.notify_streak_milestone(session, current_user.id, current_user.streak_days)
    session.add(det_session)
    await session.flush()  # get the auto-generated id

    # Proximity clustering — if nearby pending report exists, link to it
    if final_lat and final_lng:
        nearby = await eco.find_nearby_pending(session, final_lat, final_lng, exclude_id=det_session.id)
        if nearby:
            det_session.cluster_id = nearby.id

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
        annotated_url=f"/annotated/{stem}_annotated.jpg",
        detections=[schemas.DetectionRecordOut.model_validate(r) for r in records],
        latitude=final_lat,
        longitude=final_lng,
        address=address,
        gps_source=gps_src,
        reporter_id=current_user.id if current_user else None
    )


@app.get("/api/sessions", response_model=schemas.SessionsPage, summary="List all detection sessions")
async def list_sessions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    q: str = Query(default=None, description="Search by filename"),
    material: str = Query(default=None, description="Filter by material type"),
    min_objects: int = Query(default=None, ge=0, description="Min objects detected"),
    session: AsyncSession = Depends(db.get_db),
):
    items, total = await db.search_sessions(session, skip, limit, q, material, min_objects)
    return schemas.SessionsPage(total=total, skip=skip, limit=limit, items=items)


@app.get("/api/sessions/{session_id}", response_model=schemas.DetectionSessionDetail, summary="Get session details")
async def get_session(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    from sqlalchemy.orm import selectinload
    result = await session.execute(
        select(db.DetectionSession)
        .where(db.DetectionSession.id == session_id)
        .options(selectinload(db.DetectionSession.records))
    )
    det_session = result.scalar_one_or_none()
    if det_session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return det_session


@app.delete("/api/sessions/{session_id}", summary="Delete a session and its saved images")
async def delete_session(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Numai administratorii pot șterge raportări.")
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Delete saved files if they exist
    for path_str in (det_session.image_path, det_session.annotated_path):
        if path_str:
            p = Path(path_str)
            if p.exists():
                p.unlink()

    await session.delete(det_session)
    await session.commit()
    return {"detail": f"Session {session_id} deleted."}


@app.get("/api/stats", response_model=schemas.GlobalStats, summary="Aggregate statistics")
async def global_stats(session: AsyncSession = Depends(db.get_db)):
    total_s, total_o, avg_ms = await db.get_global_stats(session)
    materials = await db.get_material_stats(session)
    timeline = await db.get_timeline_stats(session)
    mpd = await db.get_material_per_day_stats(session)

    return schemas.GlobalStats(
        total_sessions=total_s,
        total_objects=total_o,
        avg_inference_ms=round(avg_ms, 1),
        material_distribution=[
            schemas.MaterialStat(material=row.material, count=row.cnt)
            for row in materials
        ],
        timeline=[
            schemas.TimelinePoint(day=row.day, total=int(row.total or 0))
            for row in timeline
        ],
        material_per_day=[
            {"day": row.day, "material": row.material, "count": row.cnt}
            for row in mpd
        ],
    )


@app.get("/api/export/csv", summary="Download all detections as CSV")
async def export_csv(
    resolved: Optional[int] = Query(default=None, ge=0, le=1, description="Filter: 0=unresolved, 1=resolved, omit=all"),
    material: Optional[str] = Query(default=None, description="Filter by material"),
    session: AsyncSession = Depends(db.get_db),
):
    import csv
    import io

    q = (
        select(db.DetectionSession, db.DetectionRecord, db.User)
        .outerjoin(db.DetectionRecord, db.DetectionRecord.session_id == db.DetectionSession.id)
        .outerjoin(db.User, db.User.id == db.DetectionSession.reporter_id)
        .order_by(db.DetectionSession.upload_time.desc())
    )
    if resolved is not None:
        q = q.where(db.DetectionSession.is_resolved == resolved)
    if material:
        q = q.where(db.DetectionRecord.material == material.lower())

    rows = (await session.execute(q)).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "session_id", "filename", "upload_time", "inference_ms",
        "latitude", "longitude", "address", "gps_source",
        "is_resolved", "resolved_at", "reporter",
        "material", "det_score", "cls_score",
        "box_x1", "box_y1", "box_x2", "box_y2",
    ])
    for s, r, u in rows:
        writer.writerow([
            s.id, s.filename,
            s.upload_time.isoformat() if s.upload_time else '',
            s.inference_ms,
            s.latitude, s.longitude, s.address or '', s.gps_source or '',
            s.is_resolved, s.resolved_at.isoformat() if s.resolved_at else '',
            u.username if u else '',
            r.material if r else '', r.det_score if r else '',
            r.cls_score if r else '',
            r.box_x1 if r else '', r.box_y1 if r else '',
            r.box_x2 if r else '', r.box_y2 if r else '',
        ])

    content = output.getvalue()
    from fastapi.responses import Response
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=trashdet_export.csv"},
    )


# ── GeoJSON export ───────────────────────────────────────────────────────────

@app.get("/api/export/geojson", summary="Download geolocated reports as GeoJSON")
async def export_geojson(
    resolved: Optional[int] = Query(default=None, ge=0, le=1),
    material: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    session: AsyncSession = Depends(db.get_db),
):
    """Export all geolocated reports as a standard GeoJSON FeatureCollection.
    Compatible with QGIS, ArcGIS, geojson.io, and municipal GIS tools."""
    import json

    q = (
        select(db.DetectionSession, db.User)
        .outerjoin(db.User, db.User.id == db.DetectionSession.reporter_id)
        .where(db.DetectionSession.latitude.isnot(None))
        .where(db.DetectionSession.longitude.isnot(None))
        .order_by(db.DetectionSession.upload_time.desc())
    )
    if resolved is not None:
        q = q.where(db.DetectionSession.is_resolved == resolved)
    if status:
        q = q.where(db.DetectionSession.status == status)
    if material:
        from sqlalchemy import exists as sa_exists
        q = q.where(
            sa_exists(
                select(db.DetectionRecord.id)
                .where(db.DetectionRecord.session_id == db.DetectionSession.id)
                .where(db.DetectionRecord.material == material.lower())
                .correlate(db.DetectionSession)
            )
        )

    rows = (await session.execute(q)).all()

    # Build materials list per session
    all_session_ids = [s.id for s, u in rows]
    materials_map = {}
    if all_session_ids:
        mat_q = await session.execute(
            select(db.DetectionRecord.session_id, db.DetectionRecord.material)
            .where(db.DetectionRecord.session_id.in_(all_session_ids))
        )
        for sid, mat in mat_q.all():
            materials_map.setdefault(sid, []).append(mat)

    features = []
    for s, u in rows:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [s.longitude, s.latitude],
            },
            "properties": {
                "id": s.id,
                "upload_time": s.upload_time.isoformat() if s.upload_time else None,
                "status": s.status,
                "total_objects": s.total_objects,
                "materials": materials_map.get(s.id, []),
                "address": s.address or "",
                "reporter": u.username if u and not u.anonymous_reports else "anonim",
                "verification_score": s.verification_score,
                "image_url": f"/annotated/{Path(s.annotated_path).name}" if s.annotated_path else None,
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    return Response(
        content=json.dumps(geojson, ensure_ascii=False),
        media_type="application/geo+json; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=trashdet_export.geojson"},
    )


# ── Rerun detection on saved image with new confidence ───────────────────────

@app.post("/api/sessions/{session_id}/rerun", response_model=schemas.DetectResponse,
          summary="Re-run detection on a saved image with a new confidence threshold")
async def rerun_detection(
    session_id: int,
    background_tasks: BackgroundTasks,
    det_conf: float = Query(default=0.50, ge=0.05, le=0.95),
    session: AsyncSession = Depends(db.get_db),
    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,
):
    if token is None:
        raise HTTPException(status_code=401, detail="Autentificare necesară pentru a rerula detecția.")
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    img_path = Path(det_session.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=410,
                            detail="Original image has been deleted from disk.")

    image_bytes = img_path.read_bytes()

    try:
        detections, annotated_bytes, elapsed_ms = infer.run_pipeline(
            image_bytes, det_conf=det_conf
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Delete old records
    await session.execute(
        sa_delete(db.DetectionRecord).where(db.DetectionRecord.session_id == session_id)
    )

    # Update session stats
    det_session.total_objects = len(detections)
    det_session.inference_ms = round(elapsed_ms, 2)

    # New annotated stem (reuse original stem)
    stem = img_path.stem  # e.g. "<uuid>"
    ann_path = ANNOTATED_DIR / f"{stem}_annotated.jpg"

    records = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        rec = db.DetectionRecord(
            session_id=session_id,
            material=det["material_name"],
            det_score=round(det["det_score"], 4),
            cls_score=round(det["material_score"], 4),
            box_x1=x1, box_y1=y1, box_x2=x2, box_y2=y2,
        )
        session.add(rec)
        records.append(rec)

    await session.commit()
    await session.refresh(det_session)
    for rec in records:
        await session.refresh(rec)

    # Overwrite annotated image
    background_tasks.add_task(ann_path.write_bytes, annotated_bytes)

    return schemas.DetectResponse(
        session_id=det_session.id,
        filename=det_session.filename,
        total_objects=det_session.total_objects,
        inference_ms=det_session.inference_ms,
        annotated_url=f"/annotated/{stem}_annotated.jpg",
        detections=[schemas.DetectionRecordOut.model_validate(r) for r in records],
    )


# ── Batch detect — multiple images ──────────────────────────────────────────

from typing import List

@app.post("/api/detect/batch", response_model=schemas.BatchDetectResponse,
          summary="Upload multiple images for detection")
async def detect_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    det_conf: float = Query(default=0.50, ge=0.05, le=0.95),
    session: AsyncSession = Depends(db.get_db),
    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,
):
    # Optional User Auth for points
    current_user = None
    if token:
        try:
            payload = decode_access_token(token)
            if payload and "username" in payload:
                res = await session.execute(select(db.User).where(db.User.username == payload["username"]))
                current_user = res.scalar_one_or_none()
        except Exception:
            pass
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files per batch.")

    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    results = []
    total_objects = 0
    total_ms = 0.0

    for file in files:
        if file.content_type not in allowed:
            continue  # skip non-image files silently

        image_bytes = await file.read()
        if not image_bytes:
            continue
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            continue  # skip oversized files silently in batch

        try:
            detections, annotated_bytes, elapsed_ms = infer.run_pipeline(
                image_bytes, det_conf=det_conf
            )
        except ValueError:
            continue

        stem = uuid.uuid4().hex
        det_session = db.DetectionSession(
            filename=file.filename or "upload.jpg",
            image_path=str(UPLOADS_DIR / f"{stem}.jpg"),
            annotated_path=str(ANNOTATED_DIR / f"{stem}_annotated.jpg"),
            total_objects=len(detections),
            inference_ms=round(elapsed_ms, 2),
            reporter_id=current_user.id if current_user else None,
        )
        if current_user:
            current_user.points += 10
        session.add(det_session)
        await session.flush()

        records = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            rec = db.DetectionRecord(
                session_id=det_session.id,
                material=det["material_name"],
                det_score=round(det["det_score"], 4),
                cls_score=round(det["material_score"], 4),
                box_x1=x1, box_y1=y1, box_x2=x2, box_y2=y2,
            )
            session.add(rec)
            records.append(rec)

        await session.flush()
        for rec in records:
            await session.refresh(rec)

        background_tasks.add_task(_save_files, image_bytes, annotated_bytes, stem)

        results.append(schemas.DetectResponse(
            session_id=det_session.id,
            filename=det_session.filename,
            total_objects=det_session.total_objects,
            inference_ms=det_session.inference_ms,
            annotated_url=f"/annotated/{stem}_annotated.jpg",
            detections=[schemas.DetectionRecordOut.model_validate(r) for r in records],
        ))
        total_objects += len(detections)
        total_ms += elapsed_ms

    await session.commit()

    return schemas.BatchDetectResponse(
        results=results,
        total_files=len(results),
        total_objects=total_objects,
        total_ms=round(total_ms, 1),
    )


# ── Serve original uploaded image ────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/original", summary="Get the original uploaded image")
async def get_original_image(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    img_path = Path(det_session.image_path)
    if not img_path.exists():
        raise HTTPException(status_code=410, detail="Original image deleted from disk.")

    return FileResponse(img_path, media_type="image/jpeg")


# ── Report export (printable HTML → save-as-PDF via browser) ────────────────

@app.get("/api/export/report", summary="Download a printable HTML report (open in browser → Print → Save as PDF)")
async def export_report(session: AsyncSession = Depends(db.get_db)):
    total_s, total_o, avg_ms = await db.get_global_stats(session)
    materials = await db.get_material_stats(session)
    timeline = await db.get_timeline_stats(session)

    # Printable HTML report — open in browser and use Ctrl+P → Save as PDF
    mat_rows = ""
    for row in materials:
        pct = (row.cnt / total_o * 100) if total_o > 0 else 0
        mat_rows += f"<tr><td style='padding:6px 12px'>{row.material}</td><td style='padding:6px 12px;text-align:right'>{row.cnt}</td><td style='padding:6px 12px;text-align:right'>{pct:.1f}%</td></tr>"

    tl_rows = ""
    for row in timeline:
        tl_rows += f"<tr><td style='padding:6px 12px'>{row.day}</td><td style='padding:6px 12px;text-align:right'>{int(row.total or 0)}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='UTF-8'/>
<title>Raport Trash Detection System</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #1f2937; }}
  h1 {{ color: #16a34a; border-bottom: 3px solid #16a34a; padding-bottom: 8px; }}
  h2 {{ color: #374151; margin-top: 32px; }}
  .card {{ display: inline-block; background: #f9fafb; border-radius: 12px; padding: 16px 28px; margin: 8px 8px 8px 0; text-align: center; }}
  .card .num {{ font-size: 2em; font-weight: 700; color: #16a34a; }}
  .card .label {{ font-size: 0.85em; color: #6b7280; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; padding: 8px 12px; text-align: left; font-size: 0.8em; text-transform: uppercase; color: #6b7280; }}
  .footer {{ margin-top: 40px; font-size: 0.75em; color: #9ca3af; border-top: 1px solid #e5e7eb; padding-top: 12px; }}
  @media print {{ body {{ margin: 0; }} }}
</style>
</head><body>
<h1>🗑️ Raport — Trash Detection System</h1>
<p style='color:#6b7280'>Generat automat · {__import__('datetime').datetime.now().strftime('%d.%m.%Y %H:%M')}</p>

<div>
  <div class='card'><div class='num'>{total_s}</div><div class='label'>Sesiuni</div></div>
  <div class='card'><div class='num'>{total_o}</div><div class='label'>Obiecte detectate</div></div>
  <div class='card'><div class='num'>{avg_ms:.1f} ms</div><div class='label'>Timp mediu inferență</div></div>
</div>

<h2>Distribuție materiale</h2>
<table>
  <thead><tr><th>Material</th><th style='text-align:right'>Detecții</th><th style='text-align:right'>Procent</th></tr></thead>
  <tbody>{mat_rows}</tbody>
</table>

<h2>Obiecte pe zi</h2>
<table>
  <thead><tr><th>Data</th><th style='text-align:right'>Obiecte</th></tr></thead>
  <tbody>{tl_rows}</tbody>
</table>

<div class='footer'>
  Trash Detection System · YOLOv8 · FastAPI · SQLite<br>
  Proiect licență — Detectarea automată a deșeurilor în zone urbane
</div>
</body></html>"""

    return Response(
        content=html,
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Disposition": "attachment; filename=raport_trash_detection.html",
        },
    )


# ── Map endpoints ──────────────────────────────────────────────────────────

@app.get("/api/map/reports", response_model=list[schemas.MapReport],
         summary="Get geolocated detection sessions for map display")
async def map_reports(
    limit: int = Query(default=500, ge=1, le=2000),
    resolved: Optional[int] = Query(default=None, ge=0, le=1, description="0=unresolved, 1=resolved, omit=all"),
    material: Optional[str] = Query(default=None, description="Filter by material (plastic/paper/glass/metal/other)"),
    session: AsyncSession = Depends(db.get_db),
):
    items = await db.get_geolocated_sessions(session, limit, resolved=resolved, material=material)
    return items


@app.get("/api/zones", response_model=list[schemas.ZoneStats],
         summary="Get aggregated zone contamination stats for EcoAlert map")
async def get_zones(
    grid_size: float = Query(default=0.002, ge=0.0005, le=0.05,
                             description="Grid cell size in degrees (~200m default)"),
    session: AsyncSession = Depends(db.get_db),
):
    """
    Returns grid cells with aggregated trash levels for the community heatmap.
    Each cell represents ~200m x 200m. Severity: 0=clean 1=low 2=medium 3=high.
    """
    zones = await db.get_zone_stats(session, grid_size=grid_size)
    return [schemas.ZoneStats(**z) for z in zones]


@app.get("/api/nearby", response_model=list[schemas.NearbyReport],
         summary="Get reports near a GPS coordinate")
async def get_nearby(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lng: float = Query(..., ge=-180, le=180, description="Longitude"),
    radius_km: float = Query(default=1.0, ge=0.1, le=50.0, description="Search radius in km"),
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(db.get_db),
):
    """
    Returns geolocated reports within radius_km of the given coordinates.
    Useful for "what's near me" feature on mobile.
    """
    items = await db.get_nearby_reports(session, lat, lng, radius_km, limit)
    return items


# ── Video endpoints ─────────────────────────────────────────────────────────

@app.websocket("/ws/video/live")
async def ws_video_live(
    websocket: WebSocket,
    det_conf: float = 0.50,
):
    """WebSocket for live webcam video: browser sends JPEG frames, server
    returns annotated frames + stats JSON."""
    async with db.AsyncSessionLocal() as session:
        await vid.handle_live_ws(websocket, det_conf, session)


@app.post("/api/video/upload", response_model=schemas.VideoUploadResponse,
          summary="Upload a video file for offline processing")
async def upload_video(
    file: UploadFile = File(...),
    det_conf: float = Query(default=0.50, ge=0.05, le=0.95),
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
    with open(save_path, "wb") as out_f:
        while True:
            chunk = await asyncio.to_thread(file.file.read, chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            video_empty = False
    if video_empty:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    vs = await db.create_video_session(session, source_type="upload", filename=fname)
    vs.video_path = str(save_path)
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
    session: AsyncSession = Depends(db.get_db),
):
    items, total = await db.get_video_sessions_paginated(session, skip, limit)
    return schemas.VideoSessionsPage(total=total, skip=skip, limit=limit, items=items)


@app.get("/api/video/sessions/{session_id}", response_model=schemas.VideoSessionOut,
         summary="Get video session details")
async def get_video_session(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    vs = await db.get_video_session_by_id(session, session_id)
    if vs is None:
        raise HTTPException(status_code=404, detail="Video session not found.")
    return vs


# ── ADMIN endpoints ───────────────────────────────────────────────────────────

@app.get("/api/admin/users", summary="[Admin] List all users with stats")
async def admin_list_users(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    result = await session.execute(
        select(
            db.User,
            func.count(db.DetectionSession.id).label("total_reports")
        )
        .outerjoin(db.DetectionSession, db.DetectionSession.reporter_id == db.User.id)
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
            "eco_score": u.eco_score or 0,
            "rank": u.rank or "Novice",
            "streak_days": u.streak_days or 0,
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


@app.post("/api/sessions/{session_id}/resolve", summary="[Admin] Mark session as resolved/cleaned")
async def resolve_session(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")
    det_session.is_resolved = 1 if det_session.is_resolved == 0 else 0
    det_session.resolved_at = datetime.now(timezone.utc) if det_session.is_resolved == 1 else None
    det_session.resolver_id = current_user.id if det_session.is_resolved == 1 else None
    if det_session.is_resolved == 1 and det_session.reporter_id:
        # +5 bonus points to reporter when their report is cleaned
        rep_r = await session.execute(select(db.User).where(db.User.id == det_session.reporter_id))
        reporter = rep_r.scalar_one_or_none()
        if reporter:
            reporter.points += 5
            # create in-app notification for reporter
            notif = db.Notification(
                user_id=reporter.id,
                message=f"Raportul tău #{session_id} a fost marcat ca rezolvat! +5 puncte.",
                category="resolved",
                session_id=session_id,
            )
            session.add(notif)
    await session.commit()
    return {"session_id": session_id, "is_resolved": det_session.is_resolved}


# ── Community Voting & Report Lifecycle ──────────────────────────────────────

@app.post("/api/sessions/{session_id}/vote", summary="Vote confirm or fake on a report")
async def vote_on_session(
    session_id: int,
    body: schemas.VoteRequest,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if body.vote_type not in ("confirm", "fake"):
        raise HTTPException(status_code=400, detail="vote_type trebuie să fie 'confirm' sau 'fake'.")

    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")

    if det_session.status not in ("pending", "verified"):
        raise HTTPException(status_code=400, detail=f"Nu se poate vota pe un raport cu status '{det_session.status}'.")

    if det_session.reporter_id == current_user.id:
        raise HTTPException(status_code=400, detail="Nu poți vota pe propriul raport.")

    # Check if already voted
    existing = await session.execute(
        select(db.CommunityVote)
        .where(db.CommunityVote.session_id == session_id)
        .where(db.CommunityVote.user_id == current_user.id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Ai votat deja pe acest raport.")

    weight = current_user.trust_weight or 1.0
    vote = db.CommunityVote(
        user_id=current_user.id,
        session_id=session_id,
        vote_type=body.vote_type,
        weight=weight,
    )
    session.add(vote)

    # Update verification_score
    if body.vote_type == "confirm":
        det_session.verification_score = (det_session.verification_score or 0.0) + weight
    else:
        det_session.verification_score = (det_session.verification_score or 0.0) - weight

    # Award EcoScore to voter
    voter_pts, voter_ranked_up, voter_new_rank = await eco.award_ecoscore(session, current_user, eco.POINTS_VOTE)
    if voter_ranked_up:
        await notif.notify_rank_up(session, current_user.id, voter_new_rank)
    await notif.notify_streak_milestone(session, current_user.id, current_user.streak_days)

    # Notify reporter about the vote
    if det_session.reporter_id and det_session.reporter_id != current_user.id:
        await notif.notify_vote_on_report(session, det_session.reporter_id, session_id, body.vote_type)

    # Check auto-verify
    verified = await eco.check_auto_verify(session, det_session)

    # Notify reporter on verification
    if verified and det_session.reporter_id:
        reporter_user = (await session.execute(select(db.User).where(db.User.id == det_session.reporter_id))).scalar_one()
        pts_awarded, rep_ranked_up, rep_new_rank = await eco.award_ecoscore(
            session, reporter_user, eco.POINTS_REPORT_VERIFIED,
        )
        await notif.notify_report_verified(session, det_session.reporter_id, session_id, pts_awarded)
        if rep_ranked_up:
            await notif.notify_rank_up(session, det_session.reporter_id, rep_new_rank)

    # Check fake threshold — if score drops to negative threshold
    if not verified and det_session.verification_score <= -(await eco.get_verification_threshold(session)):
        det_session.status = "fake"
        if det_session.reporter_id:
            await notif.notify_report_rejected(session, det_session.reporter_id, session_id)

    await session.commit()

    # Return vote summary
    votes_result = await session.execute(
        select(
            func.sum(case((db.CommunityVote.vote_type == "confirm", 1), else_=0)).label("confirms"),
            func.sum(case((db.CommunityVote.vote_type == "fake", 1), else_=0)).label("fakes"),
            func.sum(case((db.CommunityVote.vote_type == "confirm", db.CommunityVote.weight), else_=0.0)).label("wc"),
            func.sum(case((db.CommunityVote.vote_type == "fake", db.CommunityVote.weight), else_=0.0)).label("wf"),
        ).where(db.CommunityVote.session_id == session_id)
    )
    row = votes_result.one()
    return {
        "confirms": row.confirms or 0,
        "fakes": row.fakes or 0,
        "total_weight_confirm": round(row.wc or 0, 2),
        "total_weight_fake": round(row.wf or 0, 2),
        "user_vote": body.vote_type,
        "status": det_session.status,
    }


@app.get("/api/sessions/{session_id}/votes", response_model=schemas.VoteSummary, summary="Get vote summary for a session")
async def get_vote_summary(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,
):
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")

    votes_result = await session.execute(
        select(
            func.sum(case((db.CommunityVote.vote_type == "confirm", 1), else_=0)).label("confirms"),
            func.sum(case((db.CommunityVote.vote_type == "fake", 1), else_=0)).label("fakes"),
            func.sum(case((db.CommunityVote.vote_type == "confirm", db.CommunityVote.weight), else_=0.0)).label("wc"),
            func.sum(case((db.CommunityVote.vote_type == "fake", db.CommunityVote.weight), else_=0.0)).label("wf"),
        ).where(db.CommunityVote.session_id == session_id)
    )
    row = votes_result.one()

    user_vote = None
    if token:
        try:
            payload = decode_access_token(token)
            if payload:
                uv = await session.execute(
                    select(db.CommunityVote.vote_type)
                    .where(db.CommunityVote.session_id == session_id)
                    .where(db.CommunityVote.user_id == payload.get("id"))
                )
                uv_row = uv.scalar_one_or_none()
                if uv_row:
                    user_vote = uv_row
        except Exception:
            pass

    return {
        "confirms": row.confirms or 0,
        "fakes": row.fakes or 0,
        "total_weight_confirm": round(row.wc or 0, 2),
        "total_weight_fake": round(row.wf or 0, 2),
        "user_vote": user_vote,
    }


@app.post("/api/sessions/{session_id}/claim", summary="Claim a verified report for cleanup")
async def claim_session(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")

    if det_session.status != "verified":
        raise HTTPException(status_code=400, detail=f"Doar rapoartele verificate pot fi revendicate. Status actual: '{det_session.status}'.")

    # Check rank — minimum Ranger required
    user_rank = current_user.rank or "Novice"
    rank_names = [r["name"] for r in eco.RANKS]
    if rank_names.index(user_rank) < rank_names.index("Ranger"):
        raise HTTPException(status_code=403, detail="Ai nevoie de rangul Ranger sau mai mare pentru a revendica curățări.")

    if det_session.claimed_by and det_session.claimed_by != current_user.id:
        raise HTTPException(status_code=409, detail="Acest raport a fost deja revendicat de altcineva.")

    det_session.claimed_by = current_user.id
    det_session.claimed_at = datetime.now(timezone.utc)
    det_session.status = "in_progress"

    # Notify reporter
    if det_session.reporter_id and det_session.reporter_id != current_user.id:
        await notif.send_notification(
            session, det_session.reporter_id,
            f"🧹 Cineva a revendicat curățarea raportului tău #{session_id}!",
            "info", session_id=session_id,
        )

    await session.commit()
    return {"session_id": session_id, "status": det_session.status, "claimed_by": current_user.id}


CLEANED_DIR = APP_DIR / "cleaned"
CLEANED_DIR.mkdir(exist_ok=True)


@app.post("/api/sessions/{session_id}/clean", summary="Upload cleanup proof photo")
async def clean_session(
    session_id: int,
    file: UploadFile = File(...),
    current_user: Annotated[db.User, Depends(get_current_active_user)] = None,
    session: AsyncSession = Depends(db.get_db),
):
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")

    if det_session.status != "in_progress":
        raise HTTPException(status_code=400, detail=f"Doar rapoartele 'in_progress' pot fi marcate ca curățate. Status actual: '{det_session.status}'.")

    if det_session.claimed_by != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar persoana care a revendicat raportul poate submite dovada.")

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Tip de fișier nesuportat: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Fișier prea mare.")

    stem = uuid.uuid4().hex
    clean_path = CLEANED_DIR / f"{stem}_clean.jpg"
    clean_path.write_bytes(image_bytes)

    det_session.cleaned_image_path = str(clean_path)
    det_session.cleaned_at = datetime.now(timezone.utc)
    det_session.status = "cleaned"
    det_session.is_resolved = 1
    det_session.resolved_at = datetime.now(timezone.utc)
    det_session.resolver_id = current_user.id

    # Award EcoScore for cleanup
    pts_awarded, ranked_up, new_rank = await eco.award_ecoscore(session, current_user, eco.POINTS_CLEANUP)
    if ranked_up:
        await notif.notify_rank_up(session, current_user.id, new_rank)
    await notif.notify_streak_milestone(session, current_user.id, current_user.streak_days)

    # Notify reporter
    if det_session.reporter_id and det_session.reporter_id != current_user.id:
        await notif.notify_report_cleaned(session, det_session.reporter_id, session_id)

    await session.commit()
    return {
        "session_id": session_id,
        "status": "cleaned",
        "cleaned_image_url": f"/cleaned/{stem}_clean.jpg",
        "eco_score_awarded": pts_awarded,
    }


# Serve cleaned proof images
app.mount("/cleaned", StaticFiles(directory=str(CLEANED_DIR)), name="cleaned")


@app.get("/api/sessions/{session_id}/clean-image", summary="Get the cleanup proof image")
async def get_clean_image(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None or not det_session.cleaned_image_path:
        raise HTTPException(status_code=404, detail="Imaginea de curățare nu a fost găsită.")
    path = Path(det_session.cleaned_image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fișierul nu mai există.")
    return FileResponse(path, media_type="image/jpeg")


# ── Community Feed & Profile ─────────────────────────────────────────────────

@app.get("/api/community/feed", response_model=list[schemas.CommunityFeedItem], summary="Community activity feed")
async def community_feed(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=50),
    session: AsyncSession = Depends(db.get_db),
):
    # Recent sessions with meaningful status changes
    result = await session.execute(
        select(db.DetectionSession, db.User.username)
        .outerjoin(db.User, db.DetectionSession.reporter_id == db.User.id)
        .where(db.DetectionSession.latitude.is_not(None))
        .order_by(db.DetectionSession.upload_time.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = result.all()

    # Build set of anonymous reporter IDs
    reporter_ids = [det_s.reporter_id for det_s, _ in rows if det_s.reporter_id]
    anon_ids = set()
    hide_loc_ids = set()
    if reporter_ids:
        anon_result = await session.execute(
            select(db.User.id, db.User.anonymous_reports, db.User.hide_exact_location)
            .where(db.User.id.in_(reporter_ids))
        )
        for uid, anon, hide in anon_result.all():
            if anon:
                anon_ids.add(uid)
            if hide:
                hide_loc_ids.add(uid)

    feed = []
    for det_s, username in rows:
        event_type = "report"
        if det_s.status == "cleaned":
            event_type = "cleaned"
        elif det_s.status == "verified":
            event_type = "verified"

        # Privacy: hide username if user opted for anonymous reports
        display_name = username
        if det_s.reporter_id and det_s.reporter_id in anon_ids:
            display_name = None

        # Privacy: round GPS if user opted for hidden exact location
        lat = det_s.latitude
        lng = det_s.longitude
        if det_s.reporter_id and det_s.reporter_id in hide_loc_ids:
            lat = round(lat, 2) if lat else None   # ~1.1km precision
            lng = round(lng, 2) if lng else None

        feed.append({
            "event_type": event_type,
            "session_id": det_s.id,
            "username": display_name,
            "timestamp": det_s.upload_time,
            "total_objects": det_s.total_objects,
            "status": det_s.status,
            "latitude": lat,
            "longitude": lng,
        })
    return feed


@app.get("/api/me/profile", response_model=schemas.ProfileOut, summary="Current user's full profile")
async def my_profile(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    # Count reports
    total_reports = await session.scalar(
        select(func.count(db.DetectionSession.id))
        .where(db.DetectionSession.reporter_id == current_user.id)
    ) or 0
    total_objects = await session.scalar(
        select(func.coalesce(func.sum(db.DetectionSession.total_objects), 0))
        .where(db.DetectionSession.reporter_id == current_user.id)
    ) or 0
    verified_reports = await session.scalar(
        select(func.count(db.DetectionSession.id))
        .where(db.DetectionSession.reporter_id == current_user.id)
        .where(db.DetectionSession.status == "verified")
    ) or 0
    cleaned_reports = await session.scalar(
        select(func.count(db.DetectionSession.id))
        .where(db.DetectionSession.reporter_id == current_user.id)
        .where(db.DetectionSession.status == "cleaned")
    ) or 0
    total_votes = await session.scalar(
        select(func.count(db.CommunityVote.id))
        .where(db.CommunityVote.user_id == current_user.id)
    ) or 0

    return {
        "id": current_user.id,
        "username": current_user.username,
        "role": current_user.role,
        "eco_score": current_user.eco_score or 0,
        "rank": current_user.rank or "Novice",
        "streak_days": current_user.streak_days or 0,
        "trust_weight": current_user.trust_weight or 1.0,
        "total_reports": total_reports,
        "total_objects": total_objects,
        "verified_reports": verified_reports,
        "cleaned_reports": cleaned_reports,
        "total_votes": total_votes,
        "anonymous_reports": current_user.anonymous_reports or False,
        "hide_exact_location": current_user.hide_exact_location or False,
        "onboarding_done": current_user.onboarding_done if hasattr(current_user, 'onboarding_done') else False,
        "avatar_url": f"/avatars/{Path(current_user.avatar_path).name}" if getattr(current_user, 'avatar_path', None) else None,
        "created_at": current_user.created_at,
    }


@app.patch("/api/me/settings", summary="Update privacy settings")
async def update_my_settings(
    body: schemas.PrivacySettings,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if body.anonymous_reports is not None:
        current_user.anonymous_reports = body.anonymous_reports
    if body.hide_exact_location is not None:
        current_user.hide_exact_location = body.hide_exact_location
    await session.commit()
    return {"ok": True, "anonymous_reports": current_user.anonymous_reports, "hide_exact_location": current_user.hide_exact_location}


@app.post("/api/records/{record_id}/suggest-material", summary="Suggest material correction")
async def suggest_material(
    record_id: int,
    body: schemas.MaterialSuggestionRequest,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    # Minimum Guardian rank
    user_rank = current_user.rank or "Novice"
    rank_names = [r["name"] for r in eco.RANKS]
    if rank_names.index(user_rank) < rank_names.index("Guardian"):
        raise HTTPException(status_code=403, detail="Ai nevoie de rangul Guardian pentru sugestii de material.")

    valid_materials = {"plastic", "glass", "metal", "paper", "other"}
    if body.suggested_material.lower() not in valid_materials:
        raise HTTPException(status_code=400, detail=f"Material invalid. Alege din: {', '.join(valid_materials)}")

    record = await session.execute(
        select(db.DetectionRecord).where(db.DetectionRecord.id == record_id)
    )
    record = record.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="Detecția nu a fost găsită.")

    existing = await session.execute(
        select(db.MaterialSuggestion)
        .where(db.MaterialSuggestion.record_id == record_id)
        .where(db.MaterialSuggestion.user_id == current_user.id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Ai sugerat deja un material pentru această detecție.")

    suggestion = db.MaterialSuggestion(
        record_id=record_id,
        user_id=current_user.id,
        suggested_material=body.suggested_material.lower(),
    )
    session.add(suggestion)

    # If 3+ suggestions for same material, auto-correct
    count_result = await session.execute(
        select(func.count(db.MaterialSuggestion.id))
        .where(db.MaterialSuggestion.record_id == record_id)
        .where(db.MaterialSuggestion.suggested_material == body.suggested_material.lower())
    )
    same_count = (count_result.scalar_one() or 0) + 1  # +1 for current
    if same_count >= 3 and record.material != body.suggested_material.lower():
        record.material = body.suggested_material.lower()
        # Award points to all suggesters
        suggesters = await session.execute(
            select(db.MaterialSuggestion.user_id)
            .where(db.MaterialSuggestion.record_id == record_id)
            .where(db.MaterialSuggestion.suggested_material == body.suggested_material.lower())
        )
        for (uid,) in suggesters.all():
            if uid != current_user.id:
                u_result = await session.execute(select(db.User).where(db.User.id == uid))
                u = u_result.scalar_one_or_none()
                if u:
                    await eco.award_ecoscore(session, u, eco.POINTS_MATERIAL_CORRECTION)
        await eco.award_ecoscore(session, current_user, eco.POINTS_MATERIAL_CORRECTION)

    await session.commit()
    return {"ok": True, "suggested_material": body.suggested_material.lower(), "auto_corrected": same_count >= 3}


@app.get("/api/ranks", response_model=list[schemas.RankInfo], summary="Get rank definitions")
async def get_ranks():
    return [
        {
            "name": r["name"],
            "min_score": r["min"],
            "max_score": r["max"],
            "trust_weight": r["trust"],
            "benefits": r["benefits"],
        }
        for r in eco.RANKS
    ]


@app.get("/api/leaderboard", response_model=list[schemas.LeaderboardEntry], summary="Top users by community points")
async def leaderboard(
    limit: int = Query(default=10, ge=1, le=50),
    session: AsyncSession = Depends(db.get_db),
):
    result = await session.execute(
        select(
            db.User,
            func.count(db.DetectionSession.id).label("total_reports")
        )
        .outerjoin(db.DetectionSession, db.DetectionSession.reporter_id == db.User.id)
        .group_by(db.User.id)
        .order_by(db.User.eco_score.desc())
        .limit(limit)
    )
    rows = result.all()
    return [
        {
            "rank": i + 1,
            "username": u.username,
            "role": u.role,
            "points": u.points,
            "eco_score": u.eco_score or 0,
            "user_rank": u.rank or "Novice",
            "streak_days": u.streak_days or 0,
            "total_reports": total,
        }
        for i, (u, total) in enumerate(rows)
    ]


@app.get("/api/admin/stats", response_model=schemas.AdminStats, summary="[Admin] Global platform stats")
async def admin_stats(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    user_count = await session.scalar(select(func.count(db.User.id)))
    resolved_count = await session.scalar(
        select(func.count(db.DetectionSession.id)).where(db.DetectionSession.is_resolved == 1)
    )
    total_s, total_o, avg_ms = await db.get_global_stats(session)

    # Community status counts
    total_votes = await session.scalar(select(func.count(db.CommunityVote.id))) or 0
    pending_count = await session.scalar(
        select(func.count(db.DetectionSession.id)).where(db.DetectionSession.status == "pending")
    ) or 0
    verified_count = await session.scalar(
        select(func.count(db.DetectionSession.id)).where(db.DetectionSession.status == "verified")
    ) or 0
    cleaned_count = await session.scalar(
        select(func.count(db.DetectionSession.id)).where(db.DetectionSession.status == "cleaned")
    ) or 0
    fake_count = await session.scalar(
        select(func.count(db.DetectionSession.id)).where(db.DetectionSession.status == "fake")
    ) or 0
    in_progress_count = await session.scalar(
        select(func.count(db.DetectionSession.id)).where(db.DetectionSession.status == "in_progress")
    ) or 0

    return {
        "total_users": user_count,
        "total_sessions": total_s,
        "total_objects": total_o,
        "resolved_reports": resolved_count,
        "avg_inference_ms": round(avg_ms, 1),
        "total_votes": total_votes,
        "pending_reports": pending_count,
        "verified_reports": verified_count,
        "cleaned_reports": cleaned_count,
        "fake_reports": fake_count,
        "in_progress_reports": in_progress_count,
    }


# ── Admin: Reports management ────────────────────────────────────────────────

@app.get("/api/admin/reports", summary="[Admin] List all detection reports with filters")
async def admin_list_reports(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    status: Optional[str] = Query(default=None, description="Filter: all|pending|verified|in_progress|cleaned|fake|resolved|unresolved"),
    search: Optional[str] = Query(default=None, description="Search by filename or address"),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    
    q = select(db.DetectionSession, db.User.username.label("reporter_name")).outerjoin(
        db.User, db.DetectionSession.reporter_id == db.User.id
    )
    count_q = select(func.count(db.DetectionSession.id))

    if status == "resolved":
        q = q.where(db.DetectionSession.is_resolved == 1)
        count_q = count_q.where(db.DetectionSession.is_resolved == 1)
    elif status == "unresolved":
        q = q.where(db.DetectionSession.is_resolved == 0)
        count_q = count_q.where(db.DetectionSession.is_resolved == 0)
    elif status and status not in ("all", None):
        q = q.where(db.DetectionSession.status == status)
        count_q = count_q.where(db.DetectionSession.status == status)

    if search:
        pattern = f"%{search}%"
        q = q.where(
            (db.DetectionSession.filename.ilike(pattern)) |
            (db.DetectionSession.address.ilike(pattern))
        )
        count_q = count_q.where(
            (db.DetectionSession.filename.ilike(pattern)) |
            (db.DetectionSession.address.ilike(pattern))
        )

    total = await session.scalar(count_q)
    rows = await session.execute(
        q.order_by(db.DetectionSession.upload_time.desc()).offset(skip).limit(limit)
    )
    items = []
    for det, reporter_name in rows:
        items.append({
            "id": det.id,
            "filename": det.filename,
            "upload_time": det.upload_time.isoformat() if det.upload_time else None,
            "total_objects": det.total_objects,
            "inference_ms": det.inference_ms,
            "address": det.address,
            "is_resolved": det.is_resolved,
            "resolved_at": det.resolved_at.isoformat() if det.resolved_at else None,
            "reporter_name": reporter_name,
            "has_gps": det.latitude is not None,
            "status": det.status or "pending",
            "verification_score": det.verification_score or 0.0,
        })
    return {"total": total or 0, "skip": skip, "limit": limit, "items": items}


# ── Admin: Broadcast notification ─────────────────────────────────────────────

@app.post("/api/admin/broadcast", summary="[Admin] Send notification to all users")
async def admin_broadcast(
    body: dict,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    message = (body.get("message") or "").strip()
    if not message or len(message) > 500:
        raise HTTPException(status_code=422, detail="Mesajul trebuie să aibă între 1 și 500 caractere.")
    
    users_result = await session.execute(select(db.User.id))
    user_ids = [uid for (uid,) in users_result]
    for uid in user_ids:
        session.add(db.Notification(
            user_id=uid,
            message=message,
            category="info",
        ))
    await session.commit()
    return {"ok": True, "sent_to": len(user_ids)}


# ── Admin: Recent activity feed ───────────────────────────────────────────────

@app.get("/api/admin/activity", summary="[Admin] Recent platform activity feed")
async def admin_activity(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
    limit: int = Query(default=15, ge=1, le=50),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    
    # Recent reports
    reports_q = await session.execute(
        select(db.DetectionSession, db.User.username.label("reporter_name"))
        .outerjoin(db.User, db.DetectionSession.reporter_id == db.User.id)
        .order_by(db.DetectionSession.upload_time.desc())
        .limit(limit)
    )
    activities = []
    for det, reporter_name in reports_q:
        activities.append({
            "type": "report",
            "time": det.upload_time.isoformat() if det.upload_time else "",
            "user": reporter_name or "Anonim",
            "detail": f"{det.total_objects} obiecte în {det.filename}",
            "session_id": det.id,
        })

    # Recent registrations
    users_q = await session.execute(
        select(db.User).order_by(db.User.created_at.desc()).limit(limit)
    )
    for u in users_q.scalars():
        activities.append({
            "type": "register",
            "time": u.created_at.isoformat() if u.created_at else "",
            "user": u.username,
            "detail": f"Cont nou ({u.role})",
            "session_id": None,
        })

    # Sort combined by time descending, take top N
    activities.sort(key=lambda a: a["time"], reverse=True)
    return activities[:limit]


# ── Admin: Charts data (registrations per month, reports per day, materials) ──

@app.get("/api/admin/charts", summary="[Admin] Chart data for admin dashboard")
async def admin_charts(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Acces restricționat — doar pentru administratori.")
    
    from datetime import timedelta

    # Reports per day (last 30 days)
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=29)
    reports_per_day = await session.execute(
        select(
            func.date(db.DetectionSession.upload_time).label("day"),
            func.count(db.DetectionSession.id).label("count"),
        )
        .where(db.DetectionSession.upload_time >= thirty_days_ago)
        .group_by(func.date(db.DetectionSession.upload_time))
        .order_by(func.date(db.DetectionSession.upload_time))
    )
    reports_timeline = [{"day": str(r.day), "count": r.count} for r in reports_per_day]

    # Users per month (all time)
    users_per_month = await session.execute(
        select(
            func.strftime("%Y-%m", db.User.created_at).label("month"),
            func.count(db.User.id).label("count"),
        )
        .group_by(func.strftime("%Y-%m", db.User.created_at))
        .order_by(func.strftime("%Y-%m", db.User.created_at))
    )
    users_timeline = [{"month": r.month, "count": r.count} for r in users_per_month]

    # Materials distribution (all time)
    materials = await session.execute(
        select(
            db.DetectionRecord.material,
            func.count(db.DetectionRecord.id).label("count"),
        )
        .group_by(db.DetectionRecord.material)
        .order_by(func.count(db.DetectionRecord.id).desc())
    )
    material_dist = [{"material": r.material, "count": r.count} for r in materials]

    # Resolution rate
    total_reports = await session.scalar(select(func.count(db.DetectionSession.id)))
    resolved_reports = await session.scalar(
        select(func.count(db.DetectionSession.id)).where(db.DetectionSession.is_resolved == 1)
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
    writer.writerow(["ID", "Username", "Email", "Role", "Points", "Reports", "Created"])
    result = await session.execute(
        select(db.User, func.count(db.DetectionSession.id).label("total_reports"))
        .outerjoin(db.DetectionSession, db.DetectionSession.reporter_id == db.User.id)
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


@app.get("/api/me/stats", response_model=schemas.PersonalStats, summary="Personal stats for the logged-in user")
async def my_stats(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    """Returns personal stats: total reports, objects detected, resolved count, points, weekly activity."""
    from datetime import timedelta

    # Total personal sessions
    total_sessions = await session.scalar(
        select(func.count(db.DetectionSession.id))
        .where(db.DetectionSession.reporter_id == current_user.id)
    )
    # Total objects
    total_objects = await session.scalar(
        select(func.coalesce(func.sum(db.DetectionSession.total_objects), 0))
        .where(db.DetectionSession.reporter_id == current_user.id)
    )
    # Resolved by user
    resolved_count = await session.scalar(
        select(func.count(db.DetectionSession.id))
        .where(db.DetectionSession.reporter_id == current_user.id)
        .where(db.DetectionSession.is_resolved == 1)
    )
    # Weekly activity: last 7 days, reports per day
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=6)
    rows = await session.execute(
        select(
            func.date(db.DetectionSession.upload_time).label("day"),
            func.count(db.DetectionSession.id).label("reports"),
            func.coalesce(func.sum(db.DetectionSession.total_objects), 0).label("objects"),
        )
        .where(db.DetectionSession.reporter_id == current_user.id)
        .where(db.DetectionSession.upload_time >= seven_days_ago)
        .group_by(func.date(db.DetectionSession.upload_time))
        .order_by(func.date(db.DetectionSession.upload_time))
    )
    weekly = [{"day": r.day, "reports": r.reports, "objects": int(r.objects)} for r in rows]

    return {
        "username": current_user.username,
        "role": current_user.role,
        "points": current_user.points,
        "eco_score": current_user.eco_score or 0,
        "rank": current_user.rank or "Novice",
        "streak_days": current_user.streak_days or 0,
        "trust_weight": current_user.trust_weight or 1.0,
        "total_sessions": total_sessions or 0,
        "total_objects": total_objects or 0,
        "resolved_count": resolved_count or 0,
        "weekly_activity": weekly,
    }


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
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Numai administratorii pot șterge sesiuni video.")
    vs = await db.get_video_session_by_id(session, session_id)
    if vs is None:
        raise HTTPException(status_code=404, detail="Video session not found.")

    for path_str in (vs.video_path, vs.annotated_video_path):
        if path_str:
            p = Path(path_str)
            if p.exists():
                p.unlink()

    await session.delete(vs)
    await session.commit()
    return {"detail": f"Video session {session_id} deleted."}


# ── Comments on reports ──────────────────────────────────────────────────────

@app.get("/api/sessions/{session_id}/comments", response_model=list[schemas.CommentOut],
         summary="Get comments for a report")
async def get_comments(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")
    result = await session.execute(
        select(db.Comment, db.User.username)
        .join(db.User, db.Comment.user_id == db.User.id)
        .where(db.Comment.session_id == session_id)
        .order_by(db.Comment.created_at.asc())
    )
    return [
        {
            "id": c.id,
            "session_id": c.session_id,
            "user_id": c.user_id,
            "username": username,
            "text": c.text,
            "created_at": c.created_at.isoformat() if c.created_at else "",
        }
        for c, username in result.all()
    ]


@app.post("/api/sessions/{session_id}/comments", response_model=schemas.CommentOut,
          summary="Add a comment to a report")
async def add_comment(
    session_id: int,
    body: schemas.CommentCreate,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    text = body.text.strip()
    if not text or len(text) > 500:
        raise HTTPException(status_code=422, detail="Comentariul trebuie să aibă între 1 și 500 caractere.")

    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")

    comment = db.Comment(
        session_id=session_id,
        user_id=current_user.id,
        text=text,
    )
    session.add(comment)
    await session.commit()
    await session.refresh(comment)

    # Notify report owner about new comment
    if det_session.reporter_id and det_session.reporter_id != current_user.id:
        await notif.send_notification(
            session, det_session.reporter_id,
            f"💬 {current_user.username} a comentat pe raportul tău #{session_id}",
            "info", session_id=session_id,
        )
        await session.commit()

    return {
        "id": comment.id,
        "session_id": comment.session_id,
        "user_id": comment.user_id,
        "username": current_user.username,
        "text": comment.text,
        "created_at": comment.created_at.isoformat() if comment.created_at else "",
    }


@app.delete("/api/comments/{comment_id}", response_model=schemas.OkResponse,
            summary="Delete a comment (own or admin)")
async def delete_comment(
    comment_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    result = await session.execute(
        select(db.Comment).where(db.Comment.id == comment_id)
    )
    comment = result.scalar_one_or_none()
    if comment is None:
        raise HTTPException(status_code=404, detail="Comentariul nu a fost găsit.")
    if comment.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Nu poți șterge acest comentariu.")
    await session.delete(comment)
    await session.commit()
    return {"ok": True}


# ── User note on a report ────────────────────────────────────────────────────

@app.patch("/api/sessions/{session_id}/note", summary="Add or update user note on a report")
async def update_session_note(
    session_id: int,
    body: schemas.UserNoteUpdate,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")
    if det_session.reporter_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar reporterul sau adminul poate edita nota.")
    note_text = body.user_note.strip()[:500]
    det_session.user_note = note_text if note_text else None
    await session.commit()
    return {"ok": True, "user_note": det_session.user_note}


@app.get("/api/video/sessions/{session_id}/download",
         summary="Download the annotated video file")
async def download_annotated_video(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    vs = await db.get_video_session_by_id(session, session_id)
    if vs is None:
        raise HTTPException(status_code=404, detail="Video session not found.")
    if not vs.annotated_video_path:
        raise HTTPException(status_code=404, detail="Annotated video not yet available.")

    p = Path(vs.annotated_video_path)
    if not p.exists():
        raise HTTPException(status_code=410, detail="Annotated video file was deleted.")

    return FileResponse(p, media_type="video/mp4", filename=p.name)


# ══════════════════════════════════════════════════════════════════════════════
# Phase A: Authority Integration Pipeline
# ══════════════════════════════════════════════════════════════════════════════

# ── A3: Authority contacts CRUD ──────────────────────────────────────────────

@app.get("/api/admin/authorities", response_model=list[schemas.AuthorityContactOut],
         summary="[Admin] List authority contacts")
async def admin_list_authorities(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    rows = await session.execute(
        select(db.AuthorityContact).order_by(db.AuthorityContact.created_at.desc())
    )
    return rows.scalars().all()


@app.post("/api/admin/authorities", response_model=schemas.AuthorityContactOut,
          summary="[Admin] Add authority contact")
async def admin_add_authority(
    body: schemas.AuthorityContactCreate,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    contact = db.AuthorityContact(
        name=body.name.strip()[:200],
        email=body.email.strip()[:200],
        area_description=body.area_description.strip()[:500] if body.area_description else "",
        created_by=current_user.id,
    )
    session.add(contact)
    await session.commit()
    await session.refresh(contact)
    return contact


@app.delete("/api/admin/authorities/{authority_id}",
            summary="[Admin] Delete authority contact")
async def admin_delete_authority(
    authority_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    result = await session.execute(
        select(db.AuthorityContact).where(db.AuthorityContact.id == authority_id)
    )
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact negăsit.")
    await session.delete(contact)
    await session.commit()
    return {"ok": True}


# ── A3: Forward report to authority via email ────────────────────────────────

@app.post("/api/admin/forward/{session_id}",
          summary="[Admin] Forward a report to an authority via email")
async def admin_forward_report(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    authority_id: int = Query(...),
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")

    det_session = await db.get_session_by_id(session, session_id)
    if not det_session:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")

    result = await session.execute(
        select(db.AuthorityContact).where(db.AuthorityContact.id == authority_id)
    )
    authority = result.scalar_one_or_none()
    if not authority:
        raise HTTPException(status_code=404, detail="Contact autoritate negăsit.")

    # Build records list
    recs = await session.execute(
        select(db.DetectionRecord).where(db.DetectionRecord.session_id == session_id)
    )
    materials = [r.material for r in recs.scalars()]

    map_url = (
        f"https://www.google.com/maps?q={det_session.latitude},{det_session.longitude}"
        if det_session.latitude and det_session.longitude else "N/A"
    )

    # Build HTML email
    html_body = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto">
      <h2 style="color:#059669">🗑️ TrashDet — Raport deșeuri #{session_id}</h2>
      <table style="border-collapse:collapse;width:100%;font-size:14px">
        <tr><td style="padding:8px;border-bottom:1px solid #e5e7eb;font-weight:bold">Adresă</td>
            <td style="padding:8px;border-bottom:1px solid #e5e7eb">{det_session.address or 'Necunoscută'}</td></tr>
        <tr><td style="padding:8px;border-bottom:1px solid #e5e7eb;font-weight:bold">Coordonate</td>
            <td style="padding:8px;border-bottom:1px solid #e5e7eb"><a href="{map_url}">{det_session.latitude}, {det_session.longitude}</a></td></tr>
        <tr><td style="padding:8px;border-bottom:1px solid #e5e7eb;font-weight:bold">Obiecte detectate</td>
            <td style="padding:8px;border-bottom:1px solid #e5e7eb">{det_session.total_objects}</td></tr>
        <tr><td style="padding:8px;border-bottom:1px solid #e5e7eb;font-weight:bold">Materiale</td>
            <td style="padding:8px;border-bottom:1px solid #e5e7eb">{', '.join(materials) if materials else 'N/A'}</td></tr>
        <tr><td style="padding:8px;border-bottom:1px solid #e5e7eb;font-weight:bold">Status</td>
            <td style="padding:8px;border-bottom:1px solid #e5e7eb">{det_session.status}</td></tr>
        <tr><td style="padding:8px;border-bottom:1px solid #e5e7eb;font-weight:bold">Data raportării</td>
            <td style="padding:8px;border-bottom:1px solid #e5e7eb">{det_session.upload_time.strftime('%d.%m.%Y %H:%M') if det_session.upload_time else 'N/A'}</td></tr>
      </table>
      {f'<p style="margin-top:12px"><strong>Notă reporter:</strong> {det_session.user_note}</p>' if det_session.user_note else ''}
      <p style="margin-top:16px;font-size:12px;color:#6b7280">
        Raport generat automat de platforma TrashDet. Imaginile pot fi descărcate din aplicație.
      </p>
    </div>
    """

    # Send email using the existing SMTP setup
    if settings.SMTP_HOST:
        try:
            import aiosmtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[TrashDet] Raport deșeuri #{session_id} — {det_session.address or 'Locație necunoscută'}"
            msg["From"] = settings.SMTP_FROM
            msg["To"] = authority.email
            msg.attach(MIMEText(html_body, "html"))

            await aiosmtplib.send(
                msg,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USER,
                password=settings.SMTP_PASS,
                use_tls=True,
            )
            return {"ok": True, "message": f"Email trimis la {authority.email}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Eroare trimitere email: {str(e)}")
    else:
        # Dev mode — log the email
        print(f"\n[EMAIL-FORWARD] To: {authority.email}")
        print(f"Subject: Raport #{session_id}")
        print(f"Body: {det_session.address}, {len(materials)} materiale\n")
        return {"ok": True, "message": f"[DEV] Email simulat către {authority.email}"}


# ── A4: Webhook CRUD + fire ──────────────────────────────────────────────────

@app.get("/api/admin/webhooks", response_model=list[schemas.WebhookOut],
         summary="[Admin] List webhook configs")
async def admin_list_webhooks(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    rows = await session.execute(
        select(db.WebhookConfig).order_by(db.WebhookConfig.created_at.desc())
    )
    return rows.scalars().all()


@app.post("/api/admin/webhooks", response_model=schemas.WebhookOut,
          summary="[Admin] Create webhook config")
async def admin_create_webhook(
    body: schemas.WebhookCreate,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    import secrets as _secrets
    wh = db.WebhookConfig(
        url=body.url.strip()[:500],
        secret=body.secret.strip()[:128] if body.secret else _secrets.token_hex(16),
        events=body.events.strip()[:200] if body.events else "verified",
        active=body.active,
        created_by=current_user.id,
    )
    session.add(wh)
    await session.commit()
    await session.refresh(wh)
    return wh


@app.patch("/api/admin/webhooks/{webhook_id}", response_model=schemas.WebhookOut,
           summary="[Admin] Update webhook config")
async def admin_update_webhook(
    webhook_id: int,
    body: schemas.WebhookUpdate,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    result = await session.execute(
        select(db.WebhookConfig).where(db.WebhookConfig.id == webhook_id)
    )
    wh = result.scalar_one_or_none()
    if not wh:
        raise HTTPException(status_code=404, detail="Webhook negăsit.")
    if body.url is not None:
        wh.url = body.url.strip()[:500]
    if body.events is not None:
        wh.events = body.events.strip()[:200]
    if body.active is not None:
        wh.active = body.active
    await session.commit()
    await session.refresh(wh)
    return wh


@app.delete("/api/admin/webhooks/{webhook_id}",
            summary="[Admin] Delete webhook config")
async def admin_delete_webhook(
    webhook_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    result = await session.execute(
        select(db.WebhookConfig).where(db.WebhookConfig.id == webhook_id)
    )
    wh = result.scalar_one_or_none()
    if not wh:
        raise HTTPException(status_code=404, detail="Webhook negăsit.")
    await session.delete(wh)
    await session.commit()
    return {"ok": True}


@app.post("/api/admin/webhooks/{webhook_id}/test",
          summary="[Admin] Send test event to webhook")
async def admin_test_webhook(
    webhook_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")
    result = await session.execute(
        select(db.WebhookConfig).where(db.WebhookConfig.id == webhook_id)
    )
    wh = result.scalar_one_or_none()
    if not wh:
        raise HTTPException(status_code=404, detail="Webhook negăsit.")

    import httpx, hmac, hashlib, json as _json
    payload = {"event": "test", "message": "Webhook test from TrashDet", "timestamp": datetime.now(timezone.utc).isoformat()}
    body_bytes = _json.dumps(payload).encode()
    sig = hmac.new(wh.secret.encode(), body_bytes, hashlib.sha256).hexdigest()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                wh.url,
                content=body_bytes,
                headers={"Content-Type": "application/json", "X-TrashDet-Signature": sig},
            )
        return {"ok": True, "status_code": resp.status_code, "response": resp.text[:500]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


async def fire_webhooks(db_session: AsyncSession, event: str, payload: dict):
    """Fire all active webhooks for a given event. Called from verification/cleanup logic."""
    import httpx, hmac, hashlib, json as _json
    rows = await db_session.execute(
        select(db.WebhookConfig).where(db.WebhookConfig.active == True)
    )
    for wh in rows.scalars():
        if event not in wh.events.split(","):
            continue
        body_bytes = _json.dumps(payload).encode()
        sig = hmac.new(wh.secret.encode(), body_bytes, hashlib.sha256).hexdigest()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    wh.url,
                    content=body_bytes,
                    headers={"Content-Type": "application/json", "X-TrashDet-Signature": sig},
                )
        except Exception:
            pass  # Don't fail main operation on webhook error


# ── A5: Authority role endpoints ─────────────────────────────────────────────

@app.get("/api/authority/reports", summary="[Authority] Reports in assigned area")
async def authority_reports(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
    status_filter: Optional[str] = Query(default=None, alias="status"),
):
    if current_user.role != "authority":
        raise HTTPException(status_code=403, detail="Acces doar pentru autorități.")
    if not current_user.authority_area_lat or not current_user.authority_area_lng:
        raise HTTPException(status_code=400, detail="Zona de acoperire nu este configurată.")

    from math import radians, cos
    lat = current_user.authority_area_lat
    lng = current_user.authority_area_lng
    radius_km = current_user.authority_area_radius_km or 10.0
    # Rough bounding box (1 degree lat ≈ 111km)
    d_lat = radius_km / 111.0
    d_lng = radius_km / (111.0 * max(cos(radians(lat)), 0.01))

    q = (
        select(db.DetectionSession)
        .where(db.DetectionSession.latitude.between(lat - d_lat, lat + d_lat))
        .where(db.DetectionSession.longitude.between(lng - d_lng, lng + d_lng))
        .order_by(db.DetectionSession.upload_time.desc())
        .limit(200)
    )
    if status_filter:
        q = q.where(db.DetectionSession.status == status_filter)

    rows = await session.execute(q)
    sessions = rows.scalars().all()
    return [
        {
            "id": s.id,
            "filename": s.filename,
            "upload_time": s.upload_time.isoformat() if s.upload_time else None,
            "total_objects": s.total_objects,
            "status": s.status,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "address": s.address,
            "annotated_path": f"/annotated/{Path(s.annotated_path).name}" if s.annotated_path else None,
        }
        for s in sessions
    ]


@app.post("/api/authority/reports/{session_id}/acknowledge",
          summary="[Authority] Acknowledge a report")
async def authority_acknowledge(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "authority":
        raise HTTPException(status_code=403, detail="Acces doar pentru autorități.")
    det = await db.get_session_by_id(session, session_id)
    if not det:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")
    det.status = "acknowledged"
    await session.commit()
    return {"ok": True, "status": "acknowledged"}


@app.post("/api/authority/reports/{session_id}/schedule",
          summary="[Authority] Schedule cleanup for a report")
async def authority_schedule(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    if current_user.role != "authority":
        raise HTTPException(status_code=403, detail="Acces doar pentru autorități.")
    det = await db.get_session_by_id(session, session_id)
    if not det:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")
    det.status = "scheduled"
    await session.commit()
    return {"ok": True, "status": "scheduled"}


# ══════════════════════════════════════════════════════════════════════════════
# Phase B: File Management
# ══════════════════════════════════════════════════════════════════════════════

# ── B4: Admin storage stats ──────────────────────────────────────────────────

@app.get("/api/admin/storage", summary="[Admin] Disk storage stats")
async def admin_storage_stats(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Doar administratorii.")

    def _dir_stats(d: Path):
        files = list(d.glob("*"))
        total = sum(f.stat().st_size for f in files if f.is_file())
        return {"count": len(files), "size_mb": round(total / 1024 / 1024, 2)}

    CLEANED_DIR = APP_DIR / "cleaned"
    THUMBNAILS_DIR = APP_DIR / "thumbnails"
    AVATARS_DIR = APP_DIR / "avatars"
    return {
        "uploads": _dir_stats(UPLOADS_DIR),
        "annotated": _dir_stats(ANNOTATED_DIR),
        "cleaned": _dir_stats(CLEANED_DIR) if CLEANED_DIR.exists() else {"count": 0, "size_mb": 0},
        "videos": _dir_stats(VIDEOS_DIR),
        "thumbnails": _dir_stats(THUMBNAILS_DIR) if THUMBNAILS_DIR.exists() else {"count": 0, "size_mb": 0},
        "avatars": _dir_stats(AVATARS_DIR) if AVATARS_DIR.exists() else {"count": 0, "size_mb": 0},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase C: Remaining Features
# ══════════════════════════════════════════════════════════════════════════════

# ── C1: Onboarding ───────────────────────────────────────────────────────────

@app.post("/api/me/onboarding-done", summary="Mark onboarding as complete")
async def complete_onboarding(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    current_user.onboarding_done = True
    await session.commit()
    return {"ok": True}


# ── C2: Campaigns ────────────────────────────────────────────────────────────

@app.get("/api/campaigns", response_model=list[schemas.CampaignOut],
         summary="List active + recent campaigns")
async def list_campaigns(
    session: AsyncSession = Depends(db.get_db),
):
    rows = await session.execute(
        select(db.Campaign).order_by(db.Campaign.end_date.desc()).limit(50)
    )
    campaigns = rows.scalars().all()
    result = []
    for c in campaigns:
        # Count participants
        p_count = await session.scalar(
            select(func.count(db.CampaignParticipant.id))
            .where(db.CampaignParticipant.campaign_id == c.id)
        )
        # Count reports in campaign area + date range
        r_count_q = (
            select(func.count(db.DetectionSession.id))
            .where(db.DetectionSession.upload_time >= c.start_date)
            .where(db.DetectionSession.upload_time <= c.end_date)
        )
        if c.area_lat and c.area_lng:
            from math import radians, cos
            d_lat = c.area_radius_km / 111.0
            d_lng = c.area_radius_km / (111.0 * max(cos(radians(c.area_lat)), 0.01))
            r_count_q = r_count_q.where(
                db.DetectionSession.latitude.between(c.area_lat - d_lat, c.area_lat + d_lat)
            ).where(
                db.DetectionSession.longitude.between(c.area_lng - d_lng, c.area_lng + d_lng)
            )
        r_count = await session.scalar(r_count_q) or 0

        # Get creator username
        creator = await session.execute(
            select(db.User.username).where(db.User.id == c.created_by)
        )
        creator_name = creator.scalar_one_or_none() or "?"

        result.append(schemas.CampaignOut(
            id=c.id, title=c.title, description=c.description,
            target_reports=c.target_reports, start_date=c.start_date,
            end_date=c.end_date, area_lat=c.area_lat, area_lng=c.area_lng,
            area_radius_km=c.area_radius_km, created_by=c.created_by,
            created_at=c.created_at, participant_count=p_count,
            report_count=r_count, creator_username=creator_name,
        ))
    return result


@app.post("/api/campaigns", response_model=schemas.CampaignOut,
          summary="Create a campaign (admin or Champion+)")
async def create_campaign(
    body: schemas.CampaignCreate,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    # Admin or Champion+ rank
    if current_user.role != "admin" and current_user.eco_score < 1000:
        raise HTTPException(status_code=403, detail="Trebuie rang Champion+ sau admin pentru a crea campanii.")
    campaign = db.Campaign(
        title=body.title.strip()[:200],
        description=body.description.strip()[:1000] if body.description else "",
        target_reports=max(1, min(body.target_reports, 10000)),
        start_date=body.start_date,
        end_date=body.end_date,
        area_lat=body.area_lat,
        area_lng=body.area_lng,
        area_radius_km=body.area_radius_km,
        created_by=current_user.id,
    )
    session.add(campaign)
    await session.commit()
    await session.refresh(campaign)
    return schemas.CampaignOut(
        id=campaign.id, title=campaign.title, description=campaign.description,
        target_reports=campaign.target_reports, start_date=campaign.start_date,
        end_date=campaign.end_date, area_lat=campaign.area_lat, area_lng=campaign.area_lng,
        area_radius_km=campaign.area_radius_km, created_by=campaign.created_by,
        created_at=campaign.created_at, participant_count=0, report_count=0,
        creator_username=current_user.username,
    )


@app.post("/api/campaigns/{campaign_id}/join", summary="Join a campaign")
async def join_campaign(
    campaign_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    session: AsyncSession = Depends(db.get_db),
):
    campaign = await session.get(db.Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campania nu a fost găsită.")
    # Check if already joined
    existing = await session.execute(
        select(db.CampaignParticipant)
        .where(db.CampaignParticipant.campaign_id == campaign_id)
        .where(db.CampaignParticipant.user_id == current_user.id)
    )
    if existing.scalar_one_or_none():
        return {"ok": True, "message": "Ești deja înscris."}
    participant = db.CampaignParticipant(
        campaign_id=campaign_id, user_id=current_user.id,
    )
    session.add(participant)
    await session.commit()
    return {"ok": True, "message": "Te-ai înscris în campanie!"}


@app.get("/api/campaigns/{campaign_id}/leaderboard",
         response_model=list[schemas.CampaignLeaderboardEntry],
         summary="Campaign leaderboard")
async def campaign_leaderboard(
    campaign_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    campaign = await session.get(db.Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campania nu a fost găsită.")

    from math import radians, cos
    # Reports in campaign area + date range, grouped by reporter
    q = (
        select(
            db.User.username,
            func.count(db.DetectionSession.id).label("report_count"),
            func.coalesce(func.sum(db.DetectionSession.total_objects), 0).label("objects"),
        )
        .join(db.DetectionSession, db.DetectionSession.reporter_id == db.User.id)
        .where(db.DetectionSession.upload_time >= campaign.start_date)
        .where(db.DetectionSession.upload_time <= campaign.end_date)
    )
    if campaign.area_lat and campaign.area_lng:
        d_lat = campaign.area_radius_km / 111.0
        d_lng = campaign.area_radius_km / (111.0 * max(cos(radians(campaign.area_lat)), 0.01))
        q = q.where(
            db.DetectionSession.latitude.between(campaign.area_lat - d_lat, campaign.area_lat + d_lat)
        ).where(
            db.DetectionSession.longitude.between(campaign.area_lng - d_lng, campaign.area_lng + d_lng)
        )
    q = q.group_by(db.User.id).order_by(func.count(db.DetectionSession.id).desc()).limit(20)

    rows = await session.execute(q)
    return [
        schemas.CampaignLeaderboardEntry(
            username=r.username, report_count=r.report_count, objects_detected=int(r.objects)
        )
        for r in rows
    ]


# ── C3: Avatar upload ────────────────────────────────────────────────────────

AVATARS_DIR = APP_DIR / "avatars"
AVATARS_DIR.mkdir(exist_ok=True)

app.mount("/avatars", StaticFiles(directory=str(AVATARS_DIR)), name="avatars")


@app.post("/api/me/avatar", summary="Upload profile avatar")
async def upload_avatar(
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    file: UploadFile = File(...),
    session: AsyncSession = Depends(db.get_db),
):
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:  # 5MB max
        raise HTTPException(status_code=413, detail="Avatar prea mare (max 5MB).")

    from PIL import Image
    import io

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=422, detail="Fișierul nu este o imagine validă.")

    # Resize to 200x200 square crop
    img = img.convert("RGB")
    size = min(img.size)
    left = (img.width - size) // 2
    top = (img.height - size) // 2
    img = img.crop((left, top, left + size, top + size))
    img = img.resize((200, 200), Image.LANCZOS)

    filename = f"{current_user.id}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = AVATARS_DIR / filename
    img.save(filepath, "JPEG", quality=85)

    # Delete old avatar if exists
    if current_user.avatar_path:
        old = Path(current_user.avatar_path)
        if old.exists():
            old.unlink()

    current_user.avatar_path = str(filepath)
    await session.commit()
    return {"ok": True, "avatar_url": f"/avatars/{filename}"}


# ── C4: Report photo gallery ────────────────────────────────────────────────

PHOTOS_DIR = APP_DIR / "photos"
PHOTOS_DIR.mkdir(exist_ok=True)

app.mount("/photos", StaticFiles(directory=str(PHOTOS_DIR)), name="photos")


@app.get("/api/sessions/{session_id}/photos", response_model=list[schemas.ReportPhotoOut],
         summary="Get all photos for a session")
async def get_session_photos(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    rows = await session.execute(
        select(db.ReportPhoto)
        .where(db.ReportPhoto.session_id == session_id)
        .order_by(db.ReportPhoto.created_at)
    )
    photos = rows.scalars().all()
    return [
        schemas.ReportPhotoOut(
            id=p.id, session_id=p.session_id, user_id=p.user_id,
            image_path=f"/photos/{Path(p.image_path).name}" if p.image_path else "",
            caption=p.caption, photo_type=p.photo_type,
            created_at=p.created_at,
        )
        for p in photos
    ]


@app.post("/api/sessions/{session_id}/photos", response_model=schemas.ReportPhotoOut,
          summary="Add a photo to a session (max 5)")
async def add_session_photo(
    session_id: int,
    current_user: Annotated[db.User, Depends(get_current_active_user)],
    file: UploadFile = File(...),
    caption: str = Query(default="", max_length=200),
    photo_type: str = Query(default="additional"),
    session: AsyncSession = Depends(db.get_db),
):
    det = await db.get_session_by_id(session, session_id)
    if not det:
        raise HTTPException(status_code=404, detail="Sesiunea nu a fost găsită.")

    # Max 5 photos per session
    count = await session.scalar(
        select(func.count(db.ReportPhoto.id))
        .where(db.ReportPhoto.session_id == session_id)
    )
    if count >= 5:
        raise HTTPException(status_code=400, detail="Maxim 5 fotografii per raport.")

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Fișier prea mare.")

    filename = f"{session_id}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = PHOTOS_DIR / filename

    from PIL import Image
    import io
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        img.thumbnail((1920, 1920))
        img.save(filepath, "JPEG", quality=85)
    except Exception:
        raise HTTPException(status_code=422, detail="Fișierul nu este o imagine validă.")

    photo = db.ReportPhoto(
        session_id=session_id,
        user_id=current_user.id,
        image_path=str(filepath),
        caption=caption.strip()[:200] if caption else "",
        photo_type=photo_type if photo_type in ("additional", "cleanup") else "additional",
    )
    session.add(photo)
    await session.commit()
    await session.refresh(photo)
    return schemas.ReportPhotoOut(
        id=photo.id, session_id=photo.session_id, user_id=photo.user_id,
        image_path=f"/photos/{filename}", caption=photo.caption,
        photo_type=photo.photo_type, created_at=photo.created_at,
    )


# ── C5: Impact metrics ──────────────────────────────────────────────────────

@app.get("/api/impact", summary="Global impact metrics (weight + CO2)")
async def impact_metrics(
    session: AsyncSession = Depends(db.get_db),
):
    # Count materials from all sessions
    rows = await session.execute(
        select(db.DetectionRecord.material, func.count(db.DetectionRecord.id).label("cnt"))
        .group_by(db.DetectionRecord.material)
    )
    total_weight = 0.0
    total_co2 = 0.0
    material_impact = []
    for mat, cnt in rows:
        w = eco.WEIGHT_PER_ITEM.get(mat, 0.02) * cnt
        c = eco.CO2_PER_KG.get(mat, 1.0) * w
        total_weight += w
        total_co2 += c
        material_impact.append({"material": mat, "count": cnt, "weight_kg": round(w, 2), "co2_kg": round(c, 2)})

    return {
        "total_weight_kg": round(total_weight, 2),
        "total_co2_saved_kg": round(total_co2, 2),
        "material_breakdown": material_impact,
    }
