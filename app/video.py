"""
Video processing module — WebSocket live handler + uploaded-video processor.
"""

import asyncio
import base64
import json
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from app import database as db
from app import inference as infer

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent
VIDEOS_DIR = APP_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)


# ── WebSocket: live webcam stream ──────────────────────────────────────────

async def handle_live_ws(websocket: WebSocket, det_conf: float, session: AsyncSession):
    """
    Receive JPEG frames from the browser webcam, run inference,
    send back the annotated frame + stats as JSON.
    """
    await websocket.accept()

    vs = await db.create_video_session(session, source_type="webcam", filename="webcam_live")
    session_id = vs.id

    material_counts: Counter = Counter()
    total_objects = 0
    total_frames = 0
    total_ms = 0.0
    t_start = time.time()

    try:
        while True:
            # Receive JPEG bytes from browser
            data = await websocket.receive_bytes()

            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            detections, annotated, elapsed_ms = infer.run_pipeline_frame(
                frame, det_conf=det_conf
            )

            total_frames += 1
            total_ms += elapsed_ms
            frame_objects = len(detections)
            total_objects += frame_objects

            for det in detections:
                material_counts[det["material_name"]] += 1

            # Encode annotated frame to JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64_frame = base64.b64encode(buf.tobytes()).decode("ascii")

            avg_fps = total_frames / max(time.time() - t_start, 0.001)

            payload = json.dumps({
                "frame": b64_frame,
                "total_objects": frame_objects,
                "fps": round(avg_fps, 1),
                "elapsed_ms": round(elapsed_ms, 1),
                "material_counts": dict(material_counts),
                "detections": [
                    {
                        "material": d["material_name"],
                        "det_score": round(d["det_score"], 3),
                        "cls_score": round(d["material_score"], 3),
                        "box": d["box"],
                    }
                    for d in detections
                ],
            })

            await websocket.send_text(payload)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for video session %d", session_id)
    except Exception:
        logger.exception("Unexpected error in live video WebSocket (session %d)", session_id)
    finally:
        duration = time.time() - t_start
        avg_fps = total_frames / max(duration, 0.001)
        avg_ms = total_ms / max(total_frames, 1)

        await db.finish_video_session(
            session,
            session_id,
            total_frames=total_frames,
            total_objects=total_objects,
            avg_fps=avg_fps,
            avg_inference_ms=avg_ms,
            duration_sec=duration,
            materials_summary=json.dumps(dict(material_counts)),
        )


# ── Process uploaded video file (runs in background) ──────────────────────

def _process_video_sync(
    file_path: Path,
    det_conf: float,
    progress_callback=None,
) -> dict:
    """
    Synchronous video processing — runs in a thread to avoid blocking the
    async event loop.  Returns a dict with aggregated stats.
    Calls progress_callback(frames_processed, total_frames_expected) every 30 frames.
    """
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return {"error": "cannot_open"}

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    scale = min(1.0, 1920 / max(w, h))
    out_w, out_h = int(w * scale), int(h * scale)

    out_path = VIDEOS_DIR / f"{file_path.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (out_w, out_h))

    material_counts: Counter = Counter()
    total_objects = 0
    total_frames = 0
    total_ms = 0.0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections, annotated, elapsed_ms = infer.run_pipeline_frame(
                frame, det_conf=det_conf
            )

            ah, aw = annotated.shape[:2]
            if (aw, ah) != (out_w, out_h):
                annotated = cv2.resize(annotated, (out_w, out_h))

            writer.write(annotated)

            total_frames += 1
            total_ms += elapsed_ms
            total_objects += len(detections)

            for det in detections:
                material_counts[det["material_name"]] += 1

            # Report progress every 30 frames
            if progress_callback and total_frames % 30 == 0:
                progress_callback(total_frames, total_frames_expected)
    finally:
        cap.release()
        writer.release()

    duration = time.time() - t_start
    avg_fps = total_frames / max(duration, 0.001)
    avg_ms = total_ms / max(total_frames, 1)

    return {
        "total_frames": total_frames,
        "total_frames_expected": total_frames_expected,
        "total_objects": total_objects,
        "avg_fps": avg_fps,
        "avg_inference_ms": avg_ms,
        "duration_sec": duration,
        "materials_summary": json.dumps(dict(material_counts)),
        "annotated_video_path": str(out_path),
    }


async def process_uploaded_video(
    file_path: Path,
    det_conf: float,
    session_id: int,
):
    """
    Read a video file frame-by-frame, run inference, write annotated video,
    and update the DB session when done.  Heavy CV work runs in a thread via
    asyncio.to_thread() so the event loop stays responsive.
    Progress is reported to the DB every 30 frames.
    """
    loop = asyncio.get_event_loop()

    async def _write_progress(frames: int, total: int):
        async with db.AsyncSessionLocal() as s:
            await db.update_video_progress(s, session_id, frames, total)

    def progress_callback(frames: int, total: int):
        asyncio.run_coroutine_threadsafe(_write_progress(frames, total), loop)

    try:
        result = await asyncio.to_thread(
            _process_video_sync, file_path, det_conf, progress_callback
        )

        if result.get("error") == "cannot_open":
            async with db.AsyncSessionLocal() as session:
                await db.finish_video_session(
                    session, session_id,
                    total_frames=0, total_objects=0, avg_fps=0, avg_inference_ms=0,
                    duration_sec=0, materials_summary="{}", status="failed",
                )
            return

        async with db.AsyncSessionLocal() as session:
            await db.finish_video_session(
                session,
                session_id,
                total_frames=result["total_frames"],
                total_objects=result["total_objects"],
                avg_fps=result["avg_fps"],
                avg_inference_ms=result["avg_inference_ms"],
                duration_sec=result["duration_sec"],
                materials_summary=result["materials_summary"],
                annotated_video_path=result["annotated_video_path"],
            )

    except Exception as exc:
        import traceback
        traceback.print_exc()
        try:
            async with db.AsyncSessionLocal() as session:
                await db.finish_video_session(
                    session, session_id,
                    total_frames=0, total_objects=0, avg_fps=0, avg_inference_ms=0,
                    duration_sec=0, materials_summary="{}", status="failed",
                )
        except Exception:
            traceback.print_exc()
