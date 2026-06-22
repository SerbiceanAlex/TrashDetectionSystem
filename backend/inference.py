"""
Inference wrapper — loads YOLO models once at startup and exposes run_pipeline().
Uses backend.ml.two_stage for the actual detection + classification logic.

Models loaded:
  _detector    — custom YOLOv8s trash detector (single class: trash)
  _classifier  — custom YOLOv8n-cls material classifier (5 classes)
  _person_det  — pretrained yolov8n (COCO) used only for class 0 = person

Funcții expuse:
  load_models()         — încarcă o singură dată modelele YOLO la pornire.
  run_pipeline()        — pipeline pe bytes de imagine (scanare foto).
  detect_persons()      — rulează person_det pe un cadru, întoarce bbox-urile.
"""

import time
import threading

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import settings

# ── Singleton models (populated on first load_models() call) ─────────────────
_detector   = None
_classifier = None
_person_det = None
_cls_names: dict[int, str] = {}

# Serialise model calls — YOLO/PyTorch is not thread-safe when sharing weights
_inference_lock = threading.Lock()
_person_lock    = threading.Lock()

# Auto-detect best device: CUDA GPU > CPU
import torch  # noqa: E402
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    """Load all YOLO models into memory on the best available device."""
    import logging
    log = logging.getLogger(__name__)
    global _detector, _classifier, _cls_names, _person_det

    if _detector is None:
        _detector = YOLO(str(settings.detector_path))
        _detector.to(_DEVICE)
        log.info("Detector loaded: %s", settings.detector_path)

    if _classifier is None:
        cls_path = settings.classifier_path
        if cls_path.exists():
            _classifier = YOLO(str(cls_path))
            _classifier.to(_DEVICE)
            raw = getattr(_classifier, "names", {})
            if isinstance(raw, dict):
                _cls_names = {int(k): str(v) for k, v in raw.items()}
            elif isinstance(raw, list):
                _cls_names = {i: str(v) for i, v in enumerate(raw)}
            log.info("Classifier loaded: %s", cls_path)
        else:
            # Classifier weights missing — material classification disabled
            # Detection and littering monitoring still work fully
            log.warning("Classifier not found at %s — material will show as 'unknown'", cls_path)
            _cls_names = {0: "unknown"}

    if _person_det is None:
        person_pt = settings.person_detector_path
        _person_det = YOLO(str(person_pt))
        _person_det.to(_DEVICE)
        log.info("Person detector loaded: %s", person_pt)

    log.info("All models loaded on device: %s", _DEVICE)


def _resize_if_needed(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= settings.MAX_IMAGE_DIM:
        return frame
    scale = settings.MAX_IMAGE_DIM / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def run_pipeline(
    image_bytes: bytes,
    det_conf: float = settings.DEFAULT_DET_CONF,
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
    from backend.ml.two_stage import detect_and_classify, draw_detections

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


# ─────────────────────────────────────────────────────────────────────────────
# Littering Detection helpers
# ─────────────────────────────────────────────────────────────────────────────

# Minimum person bbox size — filters out partial-body detections (arm/leg
# visible as person leaves the frame edge), which would falsely spike the count.
_MIN_PERSON_W = 15   # pixels — supports distant persons in CCTV footage
_MIN_PERSON_H = 20   # pixels — supports small persons in overhead CCTV cameras


def detect_persons(
    frame: np.ndarray,
    conf: float = 0.40,
    imgsz: int = 640,
) -> list[tuple[int, int, int, int]]:
    """
    Run person detector (yolov8n, COCO class 0) on a frame.

    Returns:
        list of (x1, y1, x2, y2) integer bounding boxes for each person found.
        Partial-body detections (limb only) are filtered out by minimum-size
        threshold so the counter does not spike as a person exits the frame.
    """
    frame = _resize_if_needed(frame)
    h, w = frame.shape[:2]
    with _person_lock:
        results = _person_det.predict(
            frame, conf=conf, imgsz=imgsz, verbose=False,
            classes=[0], iou=0.5,
        )
    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None:
        return []
    persons = []
    for xyxy in boxes.xyxy.tolist():
        x1 = max(0, int(xyxy[0]))
        y1 = max(0, int(xyxy[1]))
        x2 = min(w, int(xyxy[2]))
        y2 = min(h, int(xyxy[3]))
        bw = x2 - x1
        bh = y2 - y1
        # Skip tiny / partial-body detections at frame edges
        if bw < _MIN_PERSON_W or bh < _MIN_PERSON_H:
            continue
        persons.append((x1, y1, x2, y2))
    return persons
