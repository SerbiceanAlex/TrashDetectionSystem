"""
FastAPI application — Trash Detection System web interface.

Start with:
    .venv\\Scripts\\uvicorn app.main:app --reload --port 8000
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession

from app import database as db
from app import inference as infer
from app import schemas
from app import video as vid

APP_DIR = Path(__file__).parent
UPLOADS_DIR = APP_DIR / "uploads"
ANNOTATED_DIR = APP_DIR / "annotated"
VIDEOS_DIR = APP_DIR / "videos"
STATIC_DIR = APP_DIR / "static"

UPLOADS_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)


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

# Serve annotated videos at /videos/<filename>
app.mount("/videos", StaticFiles(directory=str(VIDEOS_DIR)), name="videos")

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
    det_conf: float = Query(default=0.30, ge=0.05, le=0.95, description="Detector confidence threshold"),
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


# ── Rerun detection on saved image with new confidence ───────────────────────

@app.post("/api/sessions/{session_id}/rerun", response_model=schemas.DetectResponse,
          summary="Re-run detection on a saved image with a new confidence threshold")
async def rerun_detection(
    session_id: int,
    background_tasks: BackgroundTasks,
    det_conf: float = Query(default=0.30, ge=0.05, le=0.95),
    session: AsyncSession = Depends(db.get_db),
):
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
    from sqlalchemy import delete as sa_delete
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
    det_conf: float = Query(default=0.30, ge=0.05, le=0.95),
    session: AsyncSession = Depends(db.get_db),
):
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
        )
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


# ── PDF export ────────────────────────────────────────────────────────────────

@app.get("/api/export/pdf", summary="Download a PDF report with statistics")
async def export_pdf(session: AsyncSession = Depends(db.get_db)):
    total_s, total_o, avg_ms = await db.get_global_stats(session)
    materials = await db.get_material_stats(session)
    timeline = await db.get_timeline_stats(session)

    # Build a simple HTML report to serve as downloadable HTML
    # (avoids heavy wkhtmltopdf/weasyprint dependency, users can print-to-PDF)
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
        media_type="text/html",
        headers={
            "Content-Disposition": "attachment; filename=raport_trash_detection.html",
        },
    )


# ── Video endpoints ─────────────────────────────────────────────────────────

@app.websocket("/ws/video/live")
async def ws_video_live(
    websocket: WebSocket,
    det_conf: float = 0.30,
):
    """WebSocket for live webcam video: browser sends JPEG frames, server
    returns annotated frames + stats JSON."""
    async with db.AsyncSessionLocal() as session:
        await vid.handle_live_ws(websocket, det_conf, session)


@app.post("/api/video/upload", response_model=schemas.VideoUploadResponse,
          summary="Upload a video file for offline processing")
async def upload_video(
    file: UploadFile = File(...),
    det_conf: float = Query(default=0.30, ge=0.05, le=0.95),
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


@app.delete("/api/video/sessions/{session_id}", summary="Delete a video session and files")
async def delete_video_session(
    session_id: int,
    session: AsyncSession = Depends(db.get_db),
):
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
