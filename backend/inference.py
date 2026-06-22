"""
Strat de inferență — încarcă o singură dată modelele YOLO la pornire și expune
funcțiile de detecție. Logica propriu-zisă detecție + clasificare e în
backend.ml.two_stage.

Modele încărcate:
  _detector    — detector de deșeuri YOLOv8s antrenat (o clasă: trash)
  _classifier  — clasificator de material YOLOv8n-cls antrenat (5 clase)
  _person_det  — yolov8n preantrenat (COCO), folosit doar pentru clasa 0 = persoană

Funcții expuse:
  load_models()         — încarcă o singură dată modelele YOLO la pornire.
  detect_persons()      — rulează person_det pe un cadru, întoarce casetele.
"""

import threading

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import settings

# ── Modele unice, populate la primul apel load_models() ──────────────────────
_detector   = None
_classifier = None
_person_det = None
_cls_names: dict[int, str] = {}

# Serializează apelurile la detectorul de persoane (YOLO/PyTorch nu e sigur pe
# fire de execuție când aceleași ponderi sunt partajate).
_person_lock    = threading.Lock()

# Alege automat cel mai bun dispozitiv: GPU CUDA dacă există, altfel CPU
import torch  # noqa: E402
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    """Încarcă toate modelele YOLO în memorie, pe cel mai bun dispozitiv disponibil."""
    import logging
    log = logging.getLogger(__name__)
    global _detector, _classifier, _cls_names, _person_det

    if _detector is None:
        _detector = YOLO(str(settings.detector_path))
        _detector.to(_DEVICE)
        log.info("Detector încărcat: %s", settings.detector_path)

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
            log.info("Clasificator încărcat: %s", cls_path)
        else:
            # Lipsesc ponderile clasificatorului — clasificarea materialului e oprită.
            # Detecția și monitorizarea abandonării funcționează în continuare complet.
            log.warning("Clasificatorul nu a fost găsit la %s — materialul va apărea ca 'unknown'", cls_path)
            _cls_names = {0: "unknown"}

    if _person_det is None:
        person_pt = settings.person_detector_path
        _person_det = YOLO(str(person_pt))
        _person_det.to(_DEVICE)
        log.info("Detector de persoane încărcat: %s", person_pt)

    log.info("Toate modelele încărcate pe dispozitivul: %s", _DEVICE)


def _resize_if_needed(frame: np.ndarray) -> np.ndarray:
    """Micșorează cadrul dacă latura maximă depășește MAX_IMAGE_DIM (altfel îl lasă neschimbat)."""
    h, w = frame.shape[:2]
    if max(h, w) <= settings.MAX_IMAGE_DIM:
        return frame
    scale = settings.MAX_IMAGE_DIM / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────────────
# Funcții ajutătoare pentru detecția persoanelor
# ─────────────────────────────────────────────────────────────────────────────

# Dimensiunea minimă a casetei de persoană — elimină detecțiile de corp parțial
# (un braț/picior vizibil la marginea cadrului), care altfel ar umfla fals numărul.
_MIN_PERSON_W = 15   # px — permite persoane la distanță în filmări CCTV
_MIN_PERSON_H = 20   # px — permite persoane mici în camere CCTV de sus


def detect_persons(
    frame: np.ndarray,
    conf: float = 0.40,
    imgsz: int = 640,
) -> list[tuple[int, int, int, int]]:
    """
    Rulează detectorul de persoane (yolov8n, clasa 0 COCO) pe un cadru.

    Întoarce lista de casete (x1, y1, x2, y2) pentru fiecare persoană găsită.
    Detecțiile de corp parțial (doar un membru) sunt filtrate prin pragul de
    mărime minimă, ca numărul să nu sară când o persoană iese din cadru.
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
        # Sari peste detecțiile mici / de corp parțial de la marginea cadrului
        if bw < _MIN_PERSON_W or bh < _MIN_PERSON_H:
            continue
        persons.append((x1, y1, x2, y2))
    return persons
