"""
Inference wrapper — loads YOLO models once at startup and exposes run_pipeline().
Uses src.detect_two_stage for the actual detection + classification logic.

Models loaded:
  _detector    — custom YOLOv8s trash detector (single class: trash)
  _classifier  — custom YOLOv8n-cls material classifier (5 classes)
  _person_det  — pretrained yolov8n (COCO) used only for class 0 = person

Littering detection helpers:
  run_pipeline_track()  — same as run_pipeline_frame() but uses model.track()
                          so each trash bbox gets a persistent track_id.
  detect_persons()      — runs person_det on a frame, returns person bboxes.
  blur_face_regions()   — applies GaussianBlur to top-25% of each person bbox
                          (GDPR privacy-by-design — face anonymisation).
"""

import time
import threading
from pathlib import Path

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
        person_pt = settings.REPO_ROOT / "yolov8n.pt"
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

def run_pipeline_track(
    frame: np.ndarray,
    det_conf: float = 0.25,
    det_imgsz: int = settings.LIVE_IMGSZ,
    cls_imgsz: int = 224,
) -> tuple[list[dict], np.ndarray, float]:
    """
    Same as run_pipeline_frame() but uses model.track() with ByteTrack so
    each detected trash bbox gets a persistent integer track_id.

    Returns:
        detections  — list of dicts; each dict has 'track_id' in addition to
                      the standard detect_and_classify keys.
        annotated   — numpy BGR annotated frame
        elapsed_ms  — inference time in milliseconds

    NOTE: persist=True keeps ByteTrack state between calls — call this
    function consecutively on frames of the SAME video stream.  Each new
    video/WebSocket session should use a fresh YOLO instance (see
    video.py:handle_monitor_ws) to avoid state bleed between sessions.
    """
    import sys as _sys
    _sys.path.insert(0, str(settings.REPO_ROOT))
    from src.detect_two_stage import draw_detections, clamp_box, classify_crop

    frame = _resize_if_needed(frame)
    h, w = frame.shape[:2]

    t0 = time.perf_counter()
    with _inference_lock:
        results = _detector.track(
            frame,
            conf=det_conf,
            imgsz=det_imgsz,
            verbose=False,
            persist=True,
            tracker="bytetrack.yaml",
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    boxes = results[0].boxes
    detections: list[dict] = []

    if boxes is not None and boxes.xyxy is not None:
        xyxy_list  = boxes.xyxy.tolist()
        conf_list  = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
        id_list    = (
            [int(x) for x in boxes.id.tolist()]
            if (hasattr(boxes, "id") and boxes.id is not None)
            else list(range(len(xyxy_list)))
        )

        for idx, (xyxy, det_score, track_id) in enumerate(zip(xyxy_list, conf_list, id_list)):
            x1, y1, x2, y2 = clamp_box(*xyxy, w, h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            with _inference_lock:
                mat_name, mat_score = classify_crop(_classifier, crop, cls_imgsz, _cls_names)
            detections.append({
                "index":         idx,
                "track_id":      track_id,
                "box":           (x1, y1, x2, y2),
                "det_score":     float(det_score),
                "material_name": mat_name,
                "material_score": mat_score,
            })

    fps = 1000.0 / max(elapsed_ms, 1e-3)
    annotated = draw_detections(frame, detections, fps=fps, max_labels=5, line_width=2)
    return detections, annotated, elapsed_ms


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


def blur_face_regions(
    frame: np.ndarray,
    person_boxes: list[tuple[int, int, int, int]],
    face_fraction: float = 0.25,
    kernel_size: int = 51,
) -> np.ndarray:
    """
    Apply Gaussian blur to the top `face_fraction` of each person bounding box.

    This is a privacy-by-design measure (GDPR Art. 25) — faces are anonymised
    before the frame is stored or transmitted.  The decision to issue a fine
    remains with a human officer who reviews the evidence packet.

    Args:
        frame:         BGR numpy array (will be copied internally).
        person_boxes:  list of (x1, y1, x2, y2) person bboxes.
        face_fraction: fraction of the bbox height treated as the face region.
        kernel_size:   GaussianBlur kernel (must be odd, larger = stronger blur).

    Returns:
        BGR frame with face regions blurred (original frame is NOT modified).
    """
    out = frame.copy()
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # ensure odd
    for (x1, y1, x2, y2) in person_boxes:
        face_h = max(1, int((y2 - y1) * face_fraction))
        fy2 = min(y1 + face_h, out.shape[0])
        region = out[y1:fy2, x1:x2]
        if region.size == 0:
            continue
        out[y1:fy2, x1:x2] = cv2.GaussianBlur(region, (k, k), 0)
    return out


def draw_persons_overlay(
    frame: np.ndarray,
    person_boxes: list[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (255, 165, 0),
) -> np.ndarray:
    """
    Draw person bounding boxes (orange) on a frame.
    Used for the monitor WebSocket feed so the operator can see tracked persons.
    """
    out = frame
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, "person", (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out
