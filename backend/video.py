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

from backend import database as db
from backend import inference as infer
from backend.auth import send_incident_alert

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

            # Run inference — frame only (no annotated image needed, client draws boxes)
            detections, _annotated, elapsed_ms = infer.run_pipeline_frame(
                frame, det_conf=det_conf
            )

            total_frames += 1
            total_ms += elapsed_ms
            frame_objects = len(detections)
            total_objects += frame_objects

            for det in detections:
                material_counts[det["material_name"]] += 1

            avg_fps = total_frames / max(time.time() - t_start, 0.001)

            # Send ONLY lightweight detection data (no base64 image) — ~1KB vs ~100KB
            # Client-side canvas overlay draws the boxes on the live video feed
            payload = json.dumps({
                "total_objects": frame_objects,
                "fps": round(avg_fps, 1),
                "elapsed_ms": round(elapsed_ms, 1),
                "material_counts": dict(material_counts),
                # Normalised box coords [x1,y1,x2,y2] in pixels of the 640px downscaled frame
                "detections": [
                    {
                        "material": d["material_name"],
                        "score": round(d["det_score"], 2),
                        "box": d["box"],           # [x1, y1, x2, y2] pixels
                        "frame_w": frame.shape[1], # actual processed width
                        "frame_h": frame.shape[0], # actual processed height
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
    Synchronous video processing with integrated LitteringDetector.

    Runs in a thread (asyncio.to_thread) to avoid blocking the event loop.
    For each frame:
      1. Tracks trash objects with a fresh per-video ByteTrack YOLO instance.
      2. Detects persons with the shared person detector.
      3. Applies the same temporal smoothing used by the live WebSocket monitor.
      4. Feeds the LitteringDetector state machine — collects LitteringEvent
         objects without touching the DB (DB writes happen in the async caller).
      5. Draws an annotated frame with state overlay and writes it to the output video.

    Returns a dict with aggregated stats + collected littering_events list.
    """
    from ultralytics import YOLO
    from backend.littering_detector import LitteringDetector
    from backend.config import settings
    import torch

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

    # ── Per-video fresh tracker + state machine ──────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = YOLO(str(settings.detector_path))
    tracker.to(device)
    # Ensure person detector is loaded (lazy singleton may not be initialized in thread)
    infer.load_models()
    detector = LitteringDetector(fps=fps_in, monitor_seconds=8.0, pre_event_seconds=4.0)

    # Temporal smoothing (mirrors handle_monitor_ws logic)
    _PERSON_CONFIRM = 2
    _PERSON_CLEAR   = 8
    _person_streak  = 0
    _person_stable  = False
    _trash_tracks: dict = {}

    # Collected events — DB writes happen in the async caller after the thread finishes
    collected_events: list = []          # list of (LitteringEvent, frame_timestamp_sec)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()

            # Stage 1 — trash tracking (ByteTrack, per-video instance)
            results = tracker.track(
                frame, conf=det_conf, imgsz=_TRASH_TRACK_IMGSZ, verbose=False,
                persist=True, tracker="bytetrack.yaml",
            )
            boxes = results[0].boxes
            trash_dets: list = []
            if boxes is not None and boxes.xyxy is not None:
                xyxy_list = boxes.xyxy.tolist()
                conf_list = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
                id_list = (
                    [int(x) for x in boxes.id.tolist()]
                    if (hasattr(boxes, "id") and boxes.id is not None)
                    else list(range(len(xyxy_list)))
                )
                h_f, w_f = frame.shape[:2]
                for xyxy, det_score, track_id in zip(xyxy_list, conf_list, id_list):
                    x1 = max(0, int(xyxy[0])); y1 = max(0, int(xyxy[1]))
                    x2 = min(w_f, int(xyxy[2])); y2 = min(h_f, int(xyxy[3]))
                    if _valid_trash_box((x1, y1, x2, y2), w_f, h_f):
                        trash_dets.append({
                            "track_id": track_id,
                            "box": (x1, y1, x2, y2),
                            "det_score": float(det_score),
                            "material_name": "unknown",
                        })

            # Stage 2 — person detection
            person_boxes = infer.detect_persons(frame, conf=0.20, imgsz=1280)

            # Temporal smoothing
            if person_boxes:
                _person_streak = min(_person_streak + 1, _PERSON_CONFIRM)
            else:
                _person_streak = max(_person_streak - 1, -_PERSON_CLEAR)
            if _person_streak >= _PERSON_CONFIRM:
                _person_stable = True
            elif _person_streak <= -_PERSON_CLEAR:
                _person_stable = False
            smoothed_person_boxes = person_boxes if _person_stable else []

            # Suppress trash bboxes that are body false-positives
            if person_boxes:
                person_filter_boxes = [_shrink_box(pb, _PERSON_FILTER_SHRINK) for pb in person_boxes]
                trash_dets = [
                    d for d in trash_dets
                    if not _should_suppress_overlapped_trash(d["box"], person_filter_boxes)
                ]

            # Track-level stabilizer
            current_ids: set = set()
            for d in trash_dets:
                tid = d["track_id"]
                current_ids.add(tid)
                st = _trash_tracks.get(tid)
                if st is None:
                    _trash_tracks[tid] = {"seen": 1, "miss": 0, "det": d}
                else:
                    st["seen"] = min(st["seen"] + 1, 9999)
                    st["miss"] = 0
                    st["det"] = d
            for tid in list(_trash_tracks):
                if tid not in current_ids:
                    _trash_tracks[tid]["miss"] += 1
                    if _trash_tracks[tid]["miss"] > _TRASH_GRACE_MISSES:
                        del _trash_tracks[tid]

            stable_trash = [
                st["det"] for st in _trash_tracks.values()
                if st["seen"] >= _TRASH_STABLE_SEEN and st["miss"] == 0
            ]
            display_trash = [
                st["det"] for st in _trash_tracks.values()
                if st["seen"] >= _TRASH_STABLE_SEEN and st["miss"] <= _TRASH_GRACE_MISSES
            ]

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_frames += 1
            total_ms += elapsed_ms
            total_objects += len(trash_dets)
            for d in trash_dets:
                material_counts[d["material_name"]] += 1

            # Stage 3 — state machine
            event = detector.update(frame, stable_trash, smoothed_person_boxes)
            if event is not None:
                ts_sec = total_frames / fps_in
                event.clip_frames = list(detector._frame_buffer)   # pre-event frames
                collected_events.append((event, ts_sec))

            # Draw annotated frame
            annotated = frame.copy()
            for pb in smoothed_person_boxes:
                cv2.rectangle(annotated, (pb[0], pb[1]), (pb[2], pb[3]), (255, 165, 0), 2)
            for d in display_trash:
                b = d["box"]
                cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            state_color = (0, 200, 0) if detector.current_state == "CLEAR" else (0, 165, 255) if detector.current_state == "MONITORING" else (0, 0, 255)
            cv2.putText(annotated, detector.current_state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)
            if event is not None:
                cv2.putText(annotated, "ALERT!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            if annotated.shape[1] != out_w or annotated.shape[0] != out_h:
                annotated = cv2.resize(annotated, (out_w, out_h))
            writer.write(annotated)

            if progress_callback and total_frames % 30 == 0:
                progress_callback(total_frames, total_frames_expected)

    finally:
        cap.release()
        writer.release()

    # Re-encode to H.264 for browser playback
    out_path = _reencode_h264(out_path)

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
        "littering_events": collected_events,    # list of (LitteringEvent, ts_sec)
        "littering_count": len(collected_events),
        "fps": fps_in,
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

        # ── Save littering events detected in the uploaded video ─────────────
        for event, ts_sec in result.get("littering_events", []):
            try:
                # Save thumbnail to disk (blocking I/O → offload to thread)
                thumb_rel = None
                if event.thumbnail is not None:
                    # Use a temporary id placeholder; real id comes from DB flush
                    import tempfile, uuid
                    tmp_id = int(uuid.uuid4().int % 1_000_000)
                    thumb_rel = await asyncio.to_thread(
                        _save_thumbnail, event.thumbnail, tmp_id
                    )

                async with db.AsyncSessionLocal() as s:
                    db_event = await db.create_littering_event(
                        s,
                        material=event.material,
                        det_score=event.det_score,
                        person_present=event.person_present,
                        person_count=1,
                        thumbnail_path=thumb_rel,
                        detection_method="zone",
                    )

                # Save clip frames
                clip_rel = None
                if event.clip_frames:
                    clip_rel = await asyncio.to_thread(
                        _save_clip, event.clip_frames, result.get("fps", 25.0) or 25.0, db_event.id
                    )

                # Rename thumbnail + update DB with paths
                final_thumb = thumb_rel
                if thumb_rel:
                    old_path = LITTERING_DIR / thumb_rel
                    new_name = f"event_{db_event.id:06d}_thumb.jpg"
                    try:
                        old_path.rename(LITTERING_DIR / new_name)
                        final_thumb = new_name
                    except Exception:
                        pass

                if clip_rel or final_thumb != thumb_rel:
                    async with db.AsyncSessionLocal() as s:
                        ev = await db.get_littering_event_by_id(s, db_event.id)
                        if ev:
                            if final_thumb != thumb_rel:
                                ev.thumbnail_path = final_thumb
                            if clip_rel:
                                ev.clip_path = clip_rel
                            await s.commit()

                logger.info(
                    "Littering event #%d saved from uploaded video (t=%.1fs, material=%s)",
                    db_event.id, ts_sec, event.material,
                )
            except Exception:
                logger.exception("Failed to save littering event from uploaded video")

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


# ── WebSocket: littering monitor mode ──────────────────────────────────────

# Directory where event clips and thumbnails are stored
LITTERING_DIR = APP_DIR / "littering"
LITTERING_DIR.mkdir(exist_ok=True)


def _reencode_h264(src: Path) -> Path:
    """Re-encode mp4v → H.264 using imageio-ffmpeg for browser playback."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        dst = src.with_suffix(".h264.mp4")
        import subprocess
        result = subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(src), "-vcodec", "libx264",
             "-preset", "fast", "-crf", "23", "-movflags", "+faststart",
             "-acodec", "copy", str(dst)],
            capture_output=True, timeout=120
        )
        if result.returncode == 0 and dst.exists():
            src.unlink(missing_ok=True)
            dst.rename(src)
        return src
    except Exception as e:
        logger.warning("H.264 re-encode failed (mp4v kept): %s", e)
        return src


def _save_clip(frames: list, fps: float, event_id: int) -> str | None:
    """
    Encode a list of BGR numpy frames as an mp4 clip (H.264 for browser).
    Returns the filename or None on failure.
    """
    if not frames:
        return None
    try:
        h, w = frames[0].shape[:2]
        out_path = LITTERING_DIR / f"event_{event_id:06d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for f in frames:
            fh, fw = f.shape[:2]
            if (fw, fh) != (w, h):
                f = cv2.resize(f, (w, h))
            writer.write(f)
        writer.release()
        _reencode_h264(out_path)
        return out_path.name
    except Exception:
        logger.exception("Failed to save littering event clip")
        return None


def _save_thumbnail(thumbnail_frame, event_id: int) -> str | None:
    """Save the annotated thumbnail jpg. Returns filename or None."""
    if thumbnail_frame is None:
        return None
    try:
        out_path = LITTERING_DIR / f"event_{event_id:06d}_thumb.jpg"
        cv2.imwrite(str(out_path), thumbnail_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return out_path.name   # just the filename, e.g. "event_000001_thumb.jpg"
    except Exception:
        logger.exception("Failed to save littering event thumbnail")
        return None


def _sha256_frame(frame) -> str:
    """Return SHA-256 hex of a numpy frame (chain-of-custody hash)."""
    import hashlib
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return hashlib.sha256(buf.tobytes()).hexdigest()


def _iou_overlap(tb, pb) -> float:
    """Fraction of trash box area that overlaps with person box (0..1)."""
    ix1 = max(tb[0], pb[0]); iy1 = max(tb[1], pb[1])
    ix2 = min(tb[2], pb[2]); iy2 = min(tb[3], pb[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    trash_area = max((tb[2] - tb[0]) * (tb[3] - tb[1]), 1)
    return inter / trash_area


_OVERLAP_THRESH = 0.35  # overlap over this may be body false-positive (adaptive)
_MIN_TRASH_AREA_FRAC = 0.00015  # ignore tiny noise boxes
_MAX_TRASH_AREA_FRAC = 0.18     # ignore huge background regions (e.g. bed/floor)
_TRASH_TRACK_IMGSZ = 416         # better small-object recall vs 320 with moderate latency
_PERSON_FILTER_SHRINK = 0.72     # shrink person boxes for overlap filtering only
_HANDHELD_MAX_PERSON_RATIO = 0.12  # keep small objects overlapping a person (in hand)
_TRASH_STABLE_SEEN = 4             # require 4 consecutive detections — reduces duplicate/ghost boxes
_TRASH_GRACE_MISSES = 4            # keep last box for a few missed frames (visual stability)


def _valid_trash_box(box: tuple[int, int, int, int], frame_w: int, frame_h: int) -> bool:
    """
    Basic geometry sanity filter for trash detections.

    Keeps small/medium object-sized boxes and rejects detections that are too
    tiny (noise) or too large for plausible litter (background furniture/bed).
    """
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 0)
    bh = max(y2 - y1, 0)
    if bw <= 0 or bh <= 0 or frame_w <= 0 or frame_h <= 0:
        return False

    frame_area = max(frame_w * frame_h, 1)
    frac = (bw * bh) / frame_area
    if frac < _MIN_TRASH_AREA_FRAC:
        return False
    if frac > _MAX_TRASH_AREA_FRAC:
        return False

    return True


def _box_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def _should_suppress_overlapped_trash(
    trash_box: tuple[int, int, int, int],
    person_boxes: list[tuple[int, int, int, int]],
) -> bool:
    """
    Suppress likely body false-positives while keeping handheld litter visible.

    If trash heavily overlaps a person but is still small relative to that person,
    keep it (typical bag/wrapper in hand). Suppress only medium/large overlaps.
    """
    if not person_boxes:
        return False

    best_overlap = 0.0
    best_person_box = None
    for pb in person_boxes:
        ov = _iou_overlap(trash_box, pb)
        if ov > best_overlap:
            best_overlap = ov
            best_person_box = pb

    if best_overlap <= _OVERLAP_THRESH or best_person_box is None:
        return False

    person_area = max(_box_area(best_person_box), 1)
    ratio_vs_person = _box_area(trash_box) / person_area

    # Keep small overlapping objects (likely held in hand).
    if ratio_vs_person <= _HANDHELD_MAX_PERSON_RATIO:
        return False

    return True


def _shrink_box(box: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    """Return a box scaled around center; used to reduce over-suppression near people."""
    x1, y1, x2, y2 = box
    if factor <= 0 or factor >= 1:
        return box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    hw = (x2 - x1) * factor / 2.0
    hh = (y2 - y1) * factor / 2.0
    nx1 = int(round(cx - hw))
    ny1 = int(round(cy - hh))
    nx2 = int(round(cx + hw))
    ny2 = int(round(cy + hh))
    if nx2 <= nx1 or ny2 <= ny1:
        return box
    return (nx1, ny1, nx2, ny2)


async def _send_location_alert(session, event_id, material, detected_at, address, lat, lng):
    """Găsește locația monitorizată cea mai apropiată și trimite alertă email."""
    try:
        ML = db.MonitoredLocation
        locs = (await session.execute(
            db.select(ML).where(ML.is_active == 1, ML.alert_email.isnot(None))
        )).scalars().all()

        if not locs:
            return

        # Dacă avem GPS, găsim locația cea mai apropiată
        target_email = None
        if lat and lng:
            min_dist = float('inf')
            for loc in locs:
                if loc.latitude and loc.longitude:
                    dist = ((loc.latitude - lat) ** 2 + (loc.longitude - lng) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        target_email = loc.alert_email
        if not target_email:
            target_email = locs[0].alert_email  # fallback: primul

        if target_email:
            await send_incident_alert(target_email, event_id, material, detected_at, address)
    except Exception as e:
        logger.warning("Alert email failed: %s", e)


async def handle_monitor_ws(
    websocket: WebSocket,
    det_conf: float,
    person_conf: float,
    latitude: float | None,
    longitude: float | None,
    session: AsyncSession,
):
    """
    WebSocket endpoint for littering detection (monitor mode).

    Protocol:
      Client → server: JPEG frame bytes (same as live mode)
      Server → client: JSON payload, one of:
        {"type": "frame", "state": ..., "persons": N, "trash": N, "fps": ..., "ms": ...}
        {"type": "alert", "event_id": ..., "material": ..., "det_score": ...,
         "thumbnail_url": ..., "detected_at": ...}

    Each WebSocket session gets its own YOLO tracker instance (fresh, no
    state bleed from concurrent sessions).
    """
    from ultralytics import YOLO
    from backend.littering_detector import LitteringDetector
    from backend.config import settings
    from datetime import timezone
    import torch

    await websocket.accept()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fresh tracker in a thread — avoids blocking the async event loop
    # (YOLO model loading is synchronous and takes ~100-500ms)
    def _load_tracker():
        t = YOLO(str(settings.detector_path))
        t.to(device)
        return t

    tracker = await asyncio.to_thread(_load_tracker)

    det_conf   = max(det_conf, 0.35)    # floor — minimum 0.35 to avoid false positives on non-trash items
    detector = LitteringDetector(fps=25.0, monitor_seconds=10.0, pre_event_seconds=5.0, zone_expand=0.35)

    # Temporal smoothing counters — require N consecutive frames to confirm/clear
    _PERSON_CONFIRM = 2   # frames needed to count person as "present"
    _PERSON_CLEAR   = 8   # frames needed to count person as "gone" (~0.27s @30fps)
    _person_streak  = 0   # >0 = seen consecutively, <0 = absent consecutively
    _person_stable  = False  # last stable person state
    _trash_tracks: dict[int, dict] = {}

    total_frames = 0
    total_ms = 0.0
    t_start = time.time()

    resolved_address: str | None = None

    try:
        while True:
            data = await websocket.receive_bytes()

            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()

            # Stage 1: trash tracking (per-session fresh tracker)
            results = tracker.track(
                frame, conf=det_conf, imgsz=_TRASH_TRACK_IMGSZ, verbose=False,
                persist=True, tracker="bytetrack.yaml"
            )
            boxes = results[0].boxes
            trash_dets: list[dict] = []
            if boxes is not None and boxes.xyxy is not None:
                xyxy_list = boxes.xyxy.tolist()
                conf_list = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
                id_list = (
                    [int(x) for x in boxes.id.tolist()]
                    if (hasattr(boxes, "id") and boxes.id is not None)
                    else list(range(len(xyxy_list)))
                )
                h_f, w_f = frame.shape[:2]
                for xyxy, det_score, track_id in zip(xyxy_list, conf_list, id_list):
                    x1 = max(0, int(xyxy[0])); y1 = max(0, int(xyxy[1]))
                    x2 = min(w_f, int(xyxy[2])); y2 = min(h_f, int(xyxy[3]))
                    if _valid_trash_box((x1, y1, x2, y2), w_f, h_f):
                        trash_dets.append({
                            "track_id": track_id,
                            "box": (x1, y1, x2, y2),
                            "det_score": float(det_score),
                            "material_name": "unknown",  # classify only on event
                        })

            # Stage 2: person detection — imgsz=1280 + conf=0.20 to capture small/distant persons in CCTV footage
            person_boxes = infer.detect_persons(
                frame,
                conf=max(person_conf, 0.25),
                imgsz=1280,
            )

            # Temporal smoothing: avoid ghost persons / single-frame flicker
            if len(person_boxes) > 0:
                _person_streak = min(_person_streak + 1, _PERSON_CONFIRM)
            else:
                _person_streak = max(_person_streak - 1, -_PERSON_CLEAR)

            if _person_streak >= _PERSON_CONFIRM:
                _person_stable = True
            elif _person_streak <= -_PERSON_CLEAR:
                _person_stable = False
            # else: keep previous state (hysteresis)

            smoothed_person_boxes = person_boxes if _person_stable else []

            # ── Filter false positives: remove trash boxes that overlap
            #    significantly with a person box (body/face detected as trash)
            if person_boxes:  # use raw boxes for filtering (before smoothing)
                person_filter_boxes = [
                    _shrink_box(pb, _PERSON_FILTER_SHRINK) for pb in person_boxes
                ]
                trash_dets = [
                    d for d in trash_dets
                    if not _should_suppress_overlapped_trash(d["box"], person_filter_boxes)
                ]

            # Track-level stabilizer: avoid flicker when object is briefly missed.
            current_ids = set()
            for d in trash_dets:
                tid = d["track_id"]
                current_ids.add(tid)
                st = _trash_tracks.get(tid)
                if st is None:
                    _trash_tracks[tid] = {"seen": 1, "miss": 0, "det": d}
                else:
                    st["seen"] = min(st["seen"] + 1, 9999)
                    st["miss"] = 0
                    st["det"] = d

            for tid in list(_trash_tracks.keys()):
                if tid in current_ids:
                    continue
                st = _trash_tracks[tid]
                st["miss"] += 1
                if st["miss"] > _TRASH_GRACE_MISSES:
                    del _trash_tracks[tid]

            detector_trash_dets = [
                st["det"]
                for st in _trash_tracks.values()
                if st["seen"] >= _TRASH_STABLE_SEEN and st["miss"] == 0
            ]
            display_trash_dets = [
                st["det"]
                for st in _trash_tracks.values()
                if st["seen"] >= _TRASH_STABLE_SEEN and st["miss"] <= _TRASH_GRACE_MISSES
            ]

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            total_frames += 1
            total_ms += elapsed_ms
            avg_fps = total_frames / max(time.time() - t_start, 0.001)

            # Stage 3: state machine update
            event = detector.update(frame, detector_trash_dets, smoothed_person_boxes)

            # ── Event detected ────────────────────────────────────────────
            if event is not None:
                # Classify the material of the trigger object via full pipeline
                try:
                    tx1, ty1, tx2, ty2 = event.trash_box
                    crop = frame[ty1:ty2, tx1:tx2]
                    if crop.size > 0:
                        from src.detect_two_stage import classify_crop
                        from backend.inference import _classifier, _cls_names
                        mat_name, mat_score = classify_crop(
                            _classifier, crop, 224, _cls_names
                        )
                        event.material   = mat_name
                        event.det_score  = mat_score
                except Exception:
                    pass  # keep "unknown"

                # Apply face blur before storing evidence
                blurred_frame = infer.blur_face_regions(frame, person_boxes)

                # Hash of blurred frame for chain of custody
                img_hash = _sha256_frame(blurred_frame)

                # Save evidence to DB first (need ID for filenames)
                db_event = await db.create_littering_event(
                    session,
                    material=event.material,
                    det_score=event.det_score,
                    person_present=event.person_present,
                    person_count=max(len(smoothed_person_boxes), 1),
                    latitude=latitude,
                    longitude=longitude,
                    address=resolved_address,
                    image_hash=img_hash,
                    incident_uid=event.incident_uid,
                    owner_person_id=event.owner_person_id,
                    distance_at_abandonment=event.distance_at_abandonment,
                    detection_method=event.detection_method,
                )
                event_id = db_event.id

                # Save clip (pre+post frames from state machine buffer)
                clip_rel = None
                if event.clip_frames:
                    clip_rel = await asyncio.to_thread(
                        _save_clip, event.clip_frames, detector.fps, event_id
                    )

                # Save thumbnail
                thumb_rel = None
                if event.thumbnail is not None:
                    # Overlay blur on thumbnail too
                    thumb_blurred = infer.blur_face_regions(event.thumbnail, [])
                    thumb_rel = await asyncio.to_thread(
                        _save_thumbnail, thumb_blurred, event_id
                    )

                # Update DB with file paths
                if clip_rel or thumb_rel:
                    await db.update_littering_event_status(
                        session, event_id, status="pending"
                    )
                    evt_obj = await db.get_littering_event_by_id(session, event_id)
                    if evt_obj:
                        if clip_rel:
                            evt_obj.clip_path = clip_rel
                        if thumb_rel:
                            evt_obj.thumbnail_path = thumb_rel
                        await session.commit()

                logger.info(
                    "Littering event #%d detected — material=%s score=%.2f",
                    event_id, event.material, event.det_score
                )

                # Send alert to browser
                await websocket.send_text(json.dumps({
                    "type": "alert",
                    "event_id": event_id,
                    "material": event.material,
                    "det_score": round(event.det_score, 3),
                    "thumbnail_url": f"/littering/event_{event_id:06d}_thumb.jpg" if thumb_rel else None,
                    "detected_at": db_event.detected_at.isoformat(),
                    "address": resolved_address,
                }))

                # Send instant email alert to nearest location's alert_email
                asyncio.create_task(_send_location_alert(
                    session, event_id, event.material,
                    db_event.detected_at.isoformat(), resolved_address,
                    latitude, longitude
                ))

            else:
                # ── Normal status frame ───────────────────────────────────
                await websocket.send_text(json.dumps({
                    "type": "frame",
                    "state": detector.current_state,
                    "monitor_progress": round(detector.monitoring_progress, 2),
                    "persons": len(smoothed_person_boxes),
                    "trash": len(display_trash_dets),
                    "fps": round(avg_fps, 1),
                    "ms": round(elapsed_ms, 1),
                    "person_boxes": [list(b) for b in smoothed_person_boxes],
                    "trash_boxes": [
                        {"box": list(d["box"]), "track_id": d["track_id"]}
                        for d in display_trash_dets
                    ],
                    "frame_w": frame.shape[1],
                    "frame_h": frame.shape[0],
                    # Send last person zones when monitoring — lets UI draw the target area
                    "last_person_zones": [
                        [z.x1, z.y1, z.x2, z.y2]
                        for z in detector._person_zones
                    ] if detector.current_state == "MONITORING" else [],
                }))

    except WebSocketDisconnect:
        logger.info("Monitor WebSocket disconnected")
    except Exception:
        logger.exception("Unexpected error in monitor WebSocket")
    finally:
        detector.reset()
