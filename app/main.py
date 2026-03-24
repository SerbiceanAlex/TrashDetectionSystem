"""
FastAPI application — Trash Detection System web interface.

Start with:
    .venv\\Scripts\\uvicorn app.main:app --reload --port 8000
"""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from app import database as db
from app import inference as infer
from app import schemas

APP_DIR = Path(__file__).parent
UPLOADS_DIR = APP_DIR / "uploads"
ANNOTATED_DIR = APP_DIR / "annotated"
STATIC_DIR = APP_DIR / "static"

UPLOADS_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)


# ── Lifespan: load models + create DB tables on startup ──────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.create_tables()
    infer.load_models()
    yield


app = FastAPI(
    title="Trash Detection System",
    description="Two-stage YOLO-based trash detection and material classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve annotated images at /annotated/<filename>
app.mount("/annotated", StaticFiles(directory=str(ANNOTATED_DIR)), name="annotated")

# Serve the frontend SPA
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_files(original_bytes: bytes, annotated_bytes: bytes, stem: str):
    """Write original + annotated images to disk (runs as a background task)."""
    (UPLOADS_DIR / f"{stem}.jpg").write_bytes(original_bytes)
    (ANNOTATED_DIR / f"{stem}_annotated.jpg").write_bytes(annotated_bytes)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/detect", response_model=schemas.DetectResponse, summary="Upload image and run detection")
async def detect(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    det_conf: float = Query(default=0.25, ge=0.05, le=0.95, description="Detector confidence threshold"),
    session: AsyncSession = Depends(db.get_db),
):
    """
    Upload a JPEG/PNG image, run the two-stage pipeline, store results in DB,
    and return the annotated image URL + detection JSON.
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

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
        annotated_url=f"/annotated/{stem}_annotated.jpg",
        detections=[schemas.DetectionRecordOut.model_validate(r) for r in records],
    )


@app.get("/api/sessions", response_model=schemas.SessionsPage, summary="List all detection sessions")
async def list_sessions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(db.get_db),
):
    total = await db.count_sessions(session)
    items = await db.get_sessions_paginated(session, skip, limit)
    return schemas.SessionsPage(total=total, skip=skip, limit=limit, items=items)


@app.get("/api/sessions/{session_id}", response_model=schemas.DetectionSessionDetail, summary="Get session details")
async def get_session(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
    det_session = await db.get_session_by_id(session, session_id)
    if det_session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Eagerly load records via relationship
    from sqlalchemy import select
    result = await session.execute(
        select(db.DetectionRecord).where(db.DetectionRecord.session_id == session_id)
    )
    records = result.scalars().all()
    det_session.records = list(records)
    return det_session


@app.delete("/api/sessions/{session_id}", summary="Delete a session and its saved images")
async def delete_session(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
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
    )


@app.get("/api/export/csv", summary="Download all detections as CSV")
async def export_csv(session: AsyncSession = Depends(db.get_db)):
    from sqlalchemy import select
    import csv
    import io

    result = await session.execute(
        select(db.DetectionSession, db.DetectionRecord)
        .join(db.DetectionRecord, db.DetectionRecord.session_id == db.DetectionSession.id)
        .order_by(db.DetectionSession.upload_time.desc())
    )
    rows = result.all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "session_id", "filename", "upload_time", "inference_ms",
        "material", "det_score", "cls_score",
        "box_x1", "box_y1", "box_x2", "box_y2",
    ])
    for s, r in rows:
        writer.writerow([
            s.id, s.filename, s.upload_time.isoformat(), s.inference_ms,
            r.material, r.det_score, r.cls_score,
            r.box_x1, r.box_y1, r.box_x2, r.box_y2,
        ])

    content = output.getvalue()
    from fastapi.responses import Response
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=detections.csv"},
    )
