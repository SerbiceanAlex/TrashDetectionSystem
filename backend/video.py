"""
Video processing module — WebSocket live handler + uploaded-video processor.
"""

import asyncio
import json
import logging
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from backend import database as db
from backend import inference as infer
from backend.config import settings

logger = logging.getLogger(__name__)

VIDEOS_DIR = settings.videos_dir
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


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
                    if not _should_suppress_overlapped_trash(d["box"], person_filter_boxes, d["det_score"])
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

        # Flush any pending events at the end of the video
        final_event = detector.finalize()
        if final_event is not None:
            collected_events.append((final_event, total_frames / fps_in))

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
                    total_frames=0, total_objects=0, littering_count=0, avg_fps=0, avg_inference_ms=0,
                    duration_sec=0, materials_summary="{}", status="failed",
                )
            return

        saved_event_count = 0

        # ── Save littering events detected in the uploaded video ─────────────
        async with db.AsyncSessionLocal() as owner_session:
            owner_vs = await db.get_video_session_by_id(owner_session, session_id)
            owner_user_id = owner_vs.user_id if owner_vs else None
            owner_org_id = (owner_vs.organization_id if owner_vs else None) or 1

        for event, ts_sec in result.get("littering_events", []):
            try:
                # Save thumbnail to disk (blocking I/O → offload to thread)
                thumb_rel = None
                if event.thumbnail is not None:
                    # Use a temporary id until the real id is created by the DB flush.
                    import uuid
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
                        reporter_id=owner_user_id,
                        organization_id=owner_org_id,
                    )

                # Save clip frames
                clip_rel = None
                if event.clip_frames:
                    clip_rel = await asyncio.to_thread(
                        _save_clip, event.clip_frames, result.get("fps", 25.0) or 25.0, db_event.id, event.trash_box
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
                saved_event_count += 1
            except Exception:
                logger.exception("Failed to save littering event from uploaded video")

        async with db.AsyncSessionLocal() as session:
            await db.finish_video_session(
                session,
                session_id,
                total_frames=result["total_frames"],
                total_objects=result["total_objects"],
                littering_count=saved_event_count,
                avg_fps=result["avg_fps"],
                avg_inference_ms=result["avg_inference_ms"],
                duration_sec=result["duration_sec"],
                materials_summary=result["materials_summary"],
                annotated_video_path=result["annotated_video_path"],
            )

    except Exception:
        import traceback
        traceback.print_exc()
        try:
            async with db.AsyncSessionLocal() as session:
                await db.finish_video_session(
                    session, session_id,
                    total_frames=0, total_objects=0, littering_count=0, avg_fps=0, avg_inference_ms=0,
                    duration_sec=0, materials_summary="{}", status="failed",
                )
        except Exception:
            traceback.print_exc()


# ── WebSocket: littering monitor mode ──────────────────────────────────────

# Directory where event clips and thumbnails are stored
LITTERING_DIR = settings.littering_dir
LITTERING_DIR.mkdir(parents=True, exist_ok=True)


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


def _save_clip(frames: list, fps: float, event_id: int, trigger_box: tuple[int, int, int, int] | None = None) -> str | None:
    """
    Encode a list of BGR numpy frames as an mp4 clip (H.264 for browser).
    Returns the filename or None on failure.

    Frames are annotated with:
      - GREEN bbox + label for trash detections (run via detector)
      - ORANGE bbox for the trigger trash region that caused the incident
      - BLUE bbox + label for person detections (yolov8n)
    NO face blur — visual evidence intentionally shows the perpetrator.
    """
    if not frames:
        return None
    try:
        h, w = frames[0].shape[:2]
        out_path = LITTERING_DIR / f"event_{event_id:06d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        for f_raw in frames:
            f = f_raw.copy()
            fh, fw = f.shape[:2]
            if (fw, fh) != (w, h):
                f = cv2.resize(f, (w, h))

            cv2.putText(
                f, f"TrashDet incident #{event_id:06d}", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA,
            )
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


# Box de deșeu care are peste atât din aria sa în interiorul siluetei (micșorate)
# a unei persoane este candidat de fals pozitiv pe corp (ochi, umăr, haine).
_OVERLAP_THRESH = 0.30
# ...dar dacă încrederea detecției e cel puțin atât, e un obiect real ținut în
# mână (cutie/sticlă/ambalaj) și se păstrează. Sub prag = fals pe corp → suprimat.
_OVERLAP_KEEP_CONF = 0.45
_OVERLAP_TINY_PERSON_RATIO = 0.018
_MIN_TRASH_AREA_FRAC = 0.00015  # ignore tiny noise boxes
_MAX_TRASH_AREA_FRAC = 0.18     # ignore huge background regions (e.g. bed/floor)
_TRASH_TRACK_IMGSZ = settings.MONITOR_TRASH_IMGSZ
_PERSON_FILTER_SHRINK = 0.72     # shrink person boxes for overlap filtering only
_TRASH_STABLE_SEEN = 4             # require 4 consecutive detections — reduces duplicate/ghost boxes
_TRASH_GRACE_MISSES = 4            # keep last box for a few missed frames (visual stability)
_MONITOR_PREWARMED = False


def prewarm_monitor_inference() -> None:
    """
    Warm up the monitor-specific inference path once during backend startup.

    The first YOLO/ByteTrack call pays a one-time CUDA/kernel initialization
    cost. Doing it at startup makes the first Monitor click feel ready instead
    of waiting several seconds for the first boxes/FPS.
    """
    global _MONITOR_PREWARMED
    if _MONITOR_PREWARMED:
        return

    try:
        import torch
        from ultralytics import YOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dummy = np.zeros((384, 640, 3), dtype=np.uint8)

        tracker = YOLO(str(settings.detector_path))
        tracker.to(device)
        tracker.track(
            dummy,
            conf=max(settings.MONITOR_MIN_DET_CONF, 0.35),
            imgsz=_TRASH_TRACK_IMGSZ,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml",
            device=device,
        )

        infer.detect_persons(dummy, conf=0.25, imgsz=settings.MONITOR_PERSON_IMGSZ)
        _MONITOR_PREWARMED = True
        logger.info("Monitor inference prewarmed on %s", device)
    except Exception:
        logger.exception("Monitor inference prewarm failed")


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


def _should_suppress_overlapped_trash(
    trash_box: tuple[int, int, int, int],
    person_boxes: list[tuple[int, int, int, int]],
    det_score: float = 1.0,
) -> bool:
    """
    Suprimă falsele detecții de „deșeu" care cad pe corpul persoanei (ochi,
    umăr, haine), DAR păstrează obiectele reale ținute în mână peste corp.

    Discriminatorul e ÎNCREDEREA, nu doar suprapunerea: un obiect real (cutie,
    sticlă, ambalaj) are încredere mare (≥ _OVERLAP_KEEP_CONF), pe când falsele
    pe piele/haine au încredere mică. Astfel:
      • obiect peste persoană + încredere MARE → păstrat (gunoi ținut în mână);
      • obiect peste persoană + încredere MICĂ → suprimat (fals pe corp).
    Obiectele care NU se suprapun cu persoana sunt mereu păstrate.
    """
    if not person_boxes:
        return False

    best_overlap = 0.0
    best_person_box = None
    for pb in person_boxes:
        overlap = _iou_overlap(trash_box, pb)
        if overlap > best_overlap:
            best_overlap = overlap
            best_person_box = pb
    if best_overlap <= _OVERLAP_THRESH:
        return False  # nu se suprapune destul → obiect real lângă/lângă persoană
    # Se suprapune cu persoana: suprimă doar dacă încrederea e mică (fals pe corp).
    if best_person_box is not None:
        tx1, ty1, tx2, ty2 = trash_box
        px1, py1, px2, py2 = best_person_box
        trash_area = max((tx2 - tx1) * (ty2 - ty1), 1)
        person_area = max((px2 - px1) * (py2 - py1), 1)
        if trash_area / person_area < _OVERLAP_TINY_PERSON_RATIO:
            return True

    return det_score < _OVERLAP_KEEP_CONF


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


async def handle_monitor_ws(
    websocket: WebSocket,
    det_conf: float,
    person_conf: float,
    analysis_fps: float,
    latitude: float | None,
    longitude: float | None,
    session: AsyncSession,
    user_id: int | None = None,
    organization_id: int | None = None,
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

    det_conf = settings.MONITOR_MIN_DET_CONF
    person_conf = settings.MONITOR_PERSON_CONF
    analysis_fps = max(5.0, min(float(analysis_fps or settings.MONITOR_TARGET_FPS), 120.0))
    logic_fps = max(5.0, min(float(settings.MONITOR_LOGIC_FPS), 30.0))
    detector = LitteringDetector(
        fps=logic_fps,
        monitor_seconds=10.0,
        # Pre-roll mai lung: evenimentul se declanșează la ABANDONARE (la câteva
        # secunde după ce obiectul e lăsat), deci clipul trebuie să înceapă
        # destul de devreme cât să prindă apropierea + aruncarea, nu doar urmarea.
        pre_event_seconds=7.0,
        zone_expand=0.35,
    )

    # Temporal smoothing counters — require N consecutive frames to confirm/clear
    _PERSON_CONFIRM = max(2, int(round(0.15 * logic_fps)))
    _PERSON_CLEAR   = max(4, int(round(0.45 * logic_fps)))
    _TRASH_DETECT_STRIDE = 2 if analysis_fps >= 20 else 1
    _PERSON_DETECT_STRIDE = 3 if analysis_fps >= 20 else 2
    _person_streak  = 0   # >0 = seen consecutively, <0 = absent consecutively
    _person_stable  = False  # last stable person state
    _trash_tracks: dict[int, dict] = {}

    total_frames = 0
    total_ms = 0.0
    t_start = time.time()
    recent_frame_times: deque[float] = deque(maxlen=max(8, int(round(analysis_fps * 2))))
    _cached_trash_dets: list[dict] = []
    _cached_person_boxes: list = []   # reuse between frames for person skip

    resolved_address: str | None = None
    # Clipul de dovadă se scrie DUPĂ fereastra post-incident (~3s): la momentul
    # alertei, event.clip_frames conține doar cadrele pre-incident, iar masina
    # de stări adaugă cadrele post-incident în iterațiile următoare.
    _pending_clip: tuple | None = None   # (event, event_id)
    _clip_tasks: set[asyncio.Task] = set()

    async def _flush_pending_clip(evt, evt_id: int) -> None:
        clip_rel = await asyncio.to_thread(
            _save_clip, evt.clip_frames, detector.fps, evt_id, evt.trash_box
        )
        if clip_rel:
            # Sesiune proaspătă: flush-ul poate rula și la închiderea conexiunii,
            # când sesiunea injectată a WebSocket-ului nu mai e utilizabilă.
            async with db.AsyncSessionLocal() as s:
                evt_obj = await db.get_littering_event_by_id(s, evt_id)
                if evt_obj:
                    evt_obj.clip_path = clip_rel
                    await s.commit()

    def _schedule_clip_flush(evt, evt_id: int) -> None:
        task = asyncio.create_task(_flush_pending_clip(evt, evt_id))
        _clip_tasks.add(task)
        def _done(done_task: asyncio.Task) -> None:
            _clip_tasks.discard(done_task)
            try:
                done_task.result()
            except Exception:
                logger.exception("Failed to export evidence clip in background")
        task.add_done_callback(_done)

    # Throttle pentru logica temporală: `detector.update()` trebuie chemat la o
    # rată FIXĂ (logic_fps), nu la rata reală de cadre (care variază 15→55 între
    # telefon și laptop). Altfel pragurile în secunde (pre-roll, fereastra de
    # confirmare, abandonare) ies în timp real diferite pe fiecare dispozitiv.
    # Cadrele YOLO + preview rămân la rata reală (FPS afișat = real, fluid).
    _logic_interval = 1.0 / logic_fps
    _last_logic_wall = 0.0

    try:
        while True:
            data = await websocket.receive_bytes()

            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()

            # Stage 1: trash tracking (per-session fresh tracker).
            # For the live monitor, running YOLO every frame is overkill on a
            # 4GB laptop GPU. On the stable FPS profile, every second frame
            # keeps latency low while freeing enough GPU time for smoother UI.
            if total_frames % _TRASH_DETECT_STRIDE == 0:
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
                _cached_trash_dets = trash_dets
            else:
                trash_dets = [dict(d) for d in _cached_trash_dets]

            # Stage 2: person detection — run periodically and cache result between.
            # Every 3rd frame is enough for stable person presence while keeping
            # the detector workload smooth on a laptop GPU.
            # Person detection is intentionally lighter in live monitor mode.
            if total_frames % _PERSON_DETECT_STRIDE == 0:
                _cached_person_boxes = infer.detect_persons(
                    frame, conf=max(person_conf, 0.25), imgsz=settings.MONITOR_PERSON_IMGSZ,
                )
            person_boxes = _cached_person_boxes

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
                    if not _should_suppress_overlapped_trash(d["box"], person_filter_boxes, d["det_score"])
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
            now_wall = time.time()
            recent_frame_times.append(now_wall)
            if len(recent_frame_times) >= 2:
                avg_fps = (len(recent_frame_times) - 1) / max(
                    recent_frame_times[-1] - recent_frame_times[0],
                    0.001,
                )
            else:
                avg_fps = total_frames / max(now_wall - t_start, 0.001)
            display_fps = min(avg_fps, analysis_fps)

            # Stage 3: state machine update — limitat la logic_fps (wall-clock)
            # ca timpii în secunde să fie corecți și consistenți între dispozitive.
            # NU sărim restul buclei: clientul așteaptă un răspuns per cadru
            # (single-flight), iar overlay-ul/preview-ul rămân la rata reală.
            event = None
            if now_wall - _last_logic_wall >= _logic_interval:
                _last_logic_wall = now_wall
                event = detector.update(frame, detector_trash_dets, smoothed_person_boxes)

                # Fereastra post-incident s-a umplut — scrie clipul complet (pre+post)
                if _pending_clip is not None and not detector._capture_post:
                    evt_to_save, evt_id_to_save = _pending_clip
                    _pending_clip = None
                    _schedule_clip_flush(evt_to_save, evt_id_to_save)

            # ── Event detected ────────────────────────────────────────────
            if event is not None:
                # Classify the material of the trigger object via full pipeline
                try:
                    tx1, ty1, tx2, ty2 = event.trash_box
                    crop = frame[ty1:ty2, tx1:tx2]
                    if crop.size > 0:
                        from backend.ml.two_stage import classify_crop
                        from backend.inference import _classifier, _cls_names
                        mat_name, mat_score = classify_crop(
                            _classifier, crop, 224, _cls_names
                        )
                        event.material   = mat_name
                        event.det_score  = mat_score
                except Exception:
                    pass  # keep "unknown"

                # No face blur — visual evidence intentionally identifies the perpetrator
                img_hash = _sha256_frame(frame)

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
                    reporter_id=user_id,
                    organization_id=organization_id or 1,
                )
                event_id = db_event.id

                # Un incident anterior încă așteaptă fereastra post — scrie-l
                # acum cu cadrele disponibile, altfel clipul lui s-ar pierde.
                if _pending_clip is not None:
                    prev_evt, prev_id = _pending_clip
                    _pending_clip = None
                    _schedule_clip_flush(prev_evt, prev_id)

                # Clipul se salvează după fereastra post-incident (vezi _pending_clip)
                _pending_clip = (event, event_id)

                # Salvează imediat un clip cu cadrele pre-incident, ca incidentul
                # să nu rămână temporar doar imagine. După fereastra post-incident,
                # același fișier este rescris cu dovada completă pre+post.
                if event.clip_frames:
                    _schedule_clip_flush(event, event_id)

                # Save thumbnail
                thumb_rel = None
                if event.thumbnail is not None:
                    thumb_rel = await asyncio.to_thread(
                        _save_thumbnail, event.thumbnail, event_id
                    )

                # Update DB with file paths
                if thumb_rel:
                    await db.update_littering_event_status(
                        session, event_id, status="pending"
                    )
                    evt_obj = await db.get_littering_event_by_id(session, event_id)
                    if evt_obj:
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
                    "thumbnail_url": f"/api/littering/events/{event_id}/thumbnail" if thumb_rel else None,
                    "detected_at": db_event.detected_at.isoformat(),
                    "address": resolved_address,
                }))

            else:
                # ── Normal status frame ───────────────────────────────────
                await websocket.send_text(json.dumps({
                    "type": "frame",
                    "state": detector.current_state,
                    "monitor_progress": round(detector.monitoring_progress, 2),
                    "persons": len(smoothed_person_boxes),
                    "trash": len(display_trash_dets),
                    "fps": round(display_fps, 1),
                    "analysis_fps_target": round(analysis_fps, 1),
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
        # Conexiunea s-a închis înainte de finalul ferestrei post-incident —
        # salvează clipul cu cadrele disponibile, ca dovada să nu se piardă.
        if _pending_clip is not None:
            evt_to_save, evt_id_to_save = _pending_clip
            try:
                await _flush_pending_clip(evt_to_save, evt_id_to_save)
            except Exception:
                logger.exception("Failed to flush pending evidence clip on close")
        detector.reset()
