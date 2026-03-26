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

# ── Paths to the best trained weights ────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
DETECTOR_PT = REPO_ROOT / "runs/detect/parks-trash-A3-final/weights/best.pt"
CLASSIFIER_PT = REPO_ROOT / "runs/classify/parks-cls-B2/weights/best.pt"

# Lazy-loaded singletons (populated on first load_models() call)
_detector = None
_classifier = None
_cls_names: dict[int, str] = {}

# Serialise model calls — YOLO/PyTorch is not thread-safe when sharing weights
_inference_lock = threading.Lock()

# Max image dimension before resize (prevents OOM on huge images)
MAX_DIM = 1920


def load_models():
    """Load YOLO models into memory (called once at app startup)."""
    global _detector, _classifier, _cls_names
    if _detector is None:
        _detector = YOLO(str(DETECTOR_PT))
    if _classifier is None:
        _classifier = YOLO(str(CLASSIFIER_PT))
        raw = getattr(_classifier, "names", {})
        if isinstance(raw, dict):
            _cls_names = {int(k): str(v) for k, v in raw.items()}
        elif isinstance(raw, list):
            _cls_names = {i: str(v) for i, v in enumerate(raw)}


def _resize_if_needed(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= MAX_DIM:
        return frame
    scale = MAX_DIM / max(h, w)
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
    sys.path.insert(0, str(REPO_ROOT))
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
    det_imgsz: int = 640,
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
    sys.path.insert(0, str(REPO_ROOT))
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
