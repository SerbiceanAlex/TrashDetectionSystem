"""
Inference wrapper — loads YOLO models once at startup and exposes run_pipeline().
Uses src.detect_two_stage for the actual detection + classification logic.
"""

import time
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import settings

# Lazy-loaded singletons (populated on first load_models() call)
_detector        = None
_classifier      = None
_cls_names: dict[int, str] = {}

# Person detector — yolov8n.pt (COCO pretrained, class 0 = person)
# Loaded lazily on first call to load_models(); never None after that.
_person_detector = None

# Serialise model calls — YOLO/PyTorch is not thread-safe when sharing weights
_inference_lock = threading.Lock()
# Person detector gets its own lock so trash + person can potentially
# be called from different threads without cross-contamination.
_person_lock = threading.Lock()

# Auto-detect best device: CUDA GPU > Apple MPS > CPU
import torch  # noqa: E402
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    """Load YOLO models into memory on the best available device (GPU if present)."""
    global _detector, _classifier, _cls_names, _person_detector
    if _detector is None:
        _detector = YOLO(str(settings.detector_path))
        _detector.to(_DEVICE)
    if _classifier is None:
        _classifier = YOLO(str(settings.classifier_path))
        _classifier.to(_DEVICE)
        raw = getattr(_classifier, "names", {})
        if isinstance(raw, dict):
            _cls_names = {int(k): str(v) for k, v in raw.items()}
        elif isinstance(raw, list):
            _cls_names = {i: str(v) for i, v in enumerate(raw)}
    if _person_detector is None:
        person_pt = settings.REPO_ROOT / "yolov8n.pt"
        _person_detector = YOLO(str(person_pt))
        _person_detector.to(_DEVICE)
    import logging
    logging.getLogger(__name__).info("Models loaded on device: %s", _DEVICE)


def _resize_if_needed(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= settings.MAX_IMAGE_DIM:
        return frame
    scale = settings.MAX_IMAGE_DIM / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def run_pipeline(
    image_bytes: bytes,
    det_conf: float = 0.25,
    det_imgsz: int = 640,
    cls_imgsz: int = 224,
) -> tuple[list[dict], bytes, float]:
    """
    Run the two-stage pipeline on raw image bytes.

    Returns:
        detections  — list of dicts from detect_and_classify()
        annotated   — JPEG bytes of the annotated image
        elapsed_ms  — inference time in milliseconds
    """
    import sys
    sys.path.insert(0, str(settings.REPO_ROOT))
    from src.detect_two_stage import detect_and_classify, draw_detections

    # Decode bytes → numpy BGR frame
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode image — unsupported format or corrupted file.")

    frame = _resize_if_needed(frame)

    t0 = time.perf_counter()
    with _inference_lock:
        detections = detect_and_classify(
            frame, _detector, _classifier, det_conf, det_imgsz, cls_imgsz, _cls_names
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    fps = 1000.0 / max(elapsed_ms, 1e-3)
    annotated = draw_detections(frame, detections, fps=fps, max_labels=5, line_width=2)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_bytes = buf.tobytes()

    return detections, annotated_bytes, elapsed_ms


def run_pipeline_frame(
    frame: np.ndarray,
    det_conf: float = 0.25,
    det_imgsz: int = settings.LIVE_IMGSZ,   # 320 by default for live — faster on GPU
    cls_imgsz: int = 224,
) -> tuple[list[dict], np.ndarray, float]:
    """
    Run the two-stage pipeline on a numpy BGR frame (optimised for video —
    skips the JPEG encode/decode round-trip used by run_pipeline()).

    Returns:
        detections  — list of dicts from detect_and_classify()
        annotated   — numpy BGR annotated frame
        elapsed_ms  — inference time in milliseconds
    """
    import sys
    sys.path.insert(0, str(settings.REPO_ROOT))
    from src.detect_two_stage import detect_and_classify, draw_detections

    frame = _resize_if_needed(frame)

    t0 = time.perf_counter()
    with _inference_lock:
        detections = detect_and_classify(
            frame, _detector, _classifier, det_conf, det_imgsz, cls_imgsz, _cls_names
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    fps = 1000.0 / max(elapsed_ms, 1e-3)
    annotated = draw_detections(frame, detections, fps=fps, max_labels=5, line_width=2)

    return detections, annotated, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# Littering Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_persons(frame: np.ndarray, conf: float = 0.40) -> list[dict]:
    """
    Detect persons in a frame using the COCO-pretrained yolov8n model.

    Returns:
        List of dicts: {"box": (x1,y1,x2,y2), "conf": float}
        (Only COCO class 0 — person — is returned.)
    """
    if _person_detector is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    with _person_lock:
        results = _person_detector.predict(
            frame, conf=conf, classes=[0], imgsz=640, verbose=False
        )

    persons: list[dict] = []
    boxes = results[0].boxes if results else None
    if boxes is None:
        return persons

    h, w = frame.shape[:2]
    for xyxy, score in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
        x1 = max(int(xyxy[0]), 0)
        y1 = max(int(xyxy[1]), 0)
        x2 = min(int(xyxy[2]), w)
        y2 = min(int(xyxy[3]), h)
        if x2 > x1 and y2 > y1:
            persons.append({"box": (x1, y1, x2, y2), "conf": float(score)})

    return persons


def blur_face_regions(frame: np.ndarray, persons: list[dict],
                      head_fraction: float = 0.25) -> np.ndarray:
    """
    Apply Gaussian blur to the top head_fraction of each person bounding box.

    This is a privacy-by-design measure (GDPR): faces are anonymised before
    any frame is stored or transmitted.

    Args:
        frame:         BGR numpy frame to blur (modified IN-PLACE on a copy).
        persons:       List of person dicts from detect_persons().
        head_fraction: Fraction of bbox height treated as face region (default 25%).

    Returns:
        New frame with face regions blurred.
    """
    out = frame.copy()
    for p in persons:
        x1, y1, x2, y2 = p["box"]
        face_y2 = y1 + max(1, int((y2 - y1) * head_fraction))
        face_y2 = min(face_y2, y2)
        region = out[y1:face_y2, x1:x2]
        if region.size > 0:
            # Kernel size must be odd; scale with region size for consistent blur
            ksize = max(21, (((x2 - x1) // 4) | 1))  # ensure odd
            out[y1:face_y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), 0)
    return out


def run_pipeline_track(
    frame: np.ndarray,
    det_conf: float = 0.25,
    det_imgsz: int = settings.LIVE_IMGSZ,
    cls_imgsz: int = 224,
    person_conf: float = 0.40,
) -> tuple[list[dict], list[dict], np.ndarray, float]:
    """
    Run the two-stage trash pipeline WITH ByteTrack object tracking,
    PLUS person detection in the same call.

    Each trash detection dict now includes a 'track_id' field (int or None).

    Returns:
        trash_dets  — list of detection dicts with track_id
        persons     — list of person dicts {box, conf}
        annotated   — numpy BGR frame with annotations (faces blurred)
        elapsed_ms  — total inference time in milliseconds
    """
    import sys as _sys
    _sys.path.insert(0, str(settings.REPO_ROOT))
    from src.detect_two_stage import clamp_box, classify_crop, draw_detections

    frame = _resize_if_needed(frame)
    h, w = frame.shape[:2]

    t0 = time.perf_counter()

    # ── Stage 1: Trash detection with ByteTrack ───────────────────────────
    with _inference_lock:
        track_results = _detector.track(
            frame,
            conf=det_conf,
            imgsz=det_imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

    trash_dets: list[dict] = []
    tboxes = track_results[0].boxes if track_results else None
    if tboxes is not None and tboxes.xyxy is not None:
        xyxy_list  = tboxes.xyxy.tolist()
        conf_list  = tboxes.conf.tolist() if tboxes.conf is not None else [0.0] * len(xyxy_list)
        id_list    = tboxes.id.int().tolist() if tboxes.id is not None else [None] * len(xyxy_list)

        for idx, (xyxy, det_score, track_id) in enumerate(zip(xyxy_list, conf_list, id_list)):
            left, top, right, bottom = clamp_box(*xyxy, w, h)
            if right <= left or bottom <= top:
                continue
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                continue
            material_name, material_score = classify_crop(_classifier, crop, cls_imgsz, _cls_names)
            trash_dets.append({
                "index":          idx,
                "track_id":       int(track_id) if track_id is not None else None,
                "box":            (left, top, right, bottom),
                "det_score":      float(det_score),
                "material_name":  material_name,
                "material_score": material_score,
            })

    # ── Stage 2: Person detection ─────────────────────────────────────────
    persons = detect_persons(frame, conf=person_conf)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # ── Annotate: blur faces first, then draw trash boxes ─────────────────
    annotated = blur_face_regions(frame, persons)
    fps = 1000.0 / max(elapsed_ms, 1e-3)
    annotated = draw_detections(annotated, trash_dets, fps=fps, max_labels=5, line_width=2)

    # Draw person boxes (blue, semi-transparent label)
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (255, 100, 0), 2)
        cv2.putText(annotated, f"person {p['conf']:.2f}",
                    (px1, py1 - 8 if py1 > 20 else py1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2, cv2.LINE_AA)

    return trash_dets, persons, annotated, elapsed_ms
