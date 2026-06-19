"""
Exporta cadre-cheie pentru licenta: secventa completa a unui incident detectat.

Ruleaza PIPELINE-UL DE PRODUCTIE din backend/video.py (_process_video_sync):
detector YOLO + ByteTrack, filtrul geometric _valid_trash_box, suprimarea
suprapunerilor cu persoane, stabilizatorul de track-uri si netezirea temporala
a persoanelor — apoi LitteringDetector. Salveaza o figura cu trei panouri:

  (a) PERSOANA PREZENTA  — persoana si obiectul vizibile in cadru;
  (b) MONITORIZARE       — persoana a plecat, zona ei este urmarita;
  (c) INCIDENT CONFIRMAT — obiect nou detectat in zona => alerta.

Exemple:
    .venv\\Scripts\\python.exe scripts\\presentation\\export_thesis_event_frames.py
    .venv\\Scripts\\python.exe scripts\\presentation\\export_thesis_event_frames.py ^
        --video datasets\\test_videos\\littering_cctv_2024.mp4 --frame-skip 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.config import settings
from backend.littering_detector import DetectorState, LitteringDetector
from backend import inference as infer
from backend.video import (
    _PERSON_FILTER_SHRINK,
    _TRASH_GRACE_MISSES,
    _TRASH_STABLE_SEEN,
    _TRASH_TRACK_IMGSZ,
    _should_suppress_overlapped_trash,
    _shrink_box,
    _valid_trash_box,
)

COL_TRASH = (0, 60, 255)      # rosu (BGR)
COL_PERSON = (0, 165, 255)    # portocaliu
COL_ZONE = (0, 200, 255)      # galben-portocaliu
COL_ALERT = (0, 0, 255)
COL_BANNER_BG = (25, 25, 25)

DEFAULT_OUT = REPO / "thesis" / "figuri" / "imagini" / "secventa_incident.png"
PANELS_DIR = REPO / "outputs" / "thesis_figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export cadre-cheie incident pentru licenta")
    p.add_argument("--video", default=str(REPO / "datasets" / "test_videos" / "littering_cctv_2024.mp4"))
    p.add_argument("--conf", type=float, default=settings.DEFAULT_DET_CONF)
    p.add_argument("--person-conf", type=float, default=0.40)
    p.add_argument("--imgsz", type=int, default=settings.LIVE_IMGSZ)
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    return p.parse_args()


def annotate(
    frame: np.ndarray,
    trash_dets: list[dict],
    person_boxes: list[tuple[int, int, int, int]],
    detector: LitteringDetector,
    state: DetectorState,
) -> np.ndarray:
    """Deseneaza persoanele, obiectele si zona monitorizata (fara banner)."""
    out = frame.copy()
    h, w = out.shape[:2]

    if state == DetectorState.MONITORING and detector._person_zones:
        for zone in detector._person_zones:
            x1, y1, x2, y2 = zone.expanded(detector.zone_expand)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            shade = out.copy()
            cv2.rectangle(shade, (x1, y1), (x2, y2), COL_ZONE, -1)
            cv2.addWeighted(shade, 0.15, out, 0.85, 0, out)
            cv2.rectangle(out, (x1, y1), (x2, y2), COL_ZONE, 2)
            cv2.putText(out, "zona monitorizata", (x1 + 4, max(18, y1 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_ZONE, 2, cv2.LINE_AA)

    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), COL_PERSON, 2)
        cv2.putText(out, "persoana", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_PERSON, 2, cv2.LINE_AA)

    for det in trash_dets:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), COL_TRASH, 2)
        cv2.putText(out, f"deseu {det['det_score']:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TRASH, 2, cv2.LINE_AA)

    return out


def add_banner(img: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    """Adauga o bara de titlu sus, proportionala cu latimea cadrului."""
    h, w = img.shape[:2]
    bar_h = max(34, h // 14)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.6, w / 1100.0)
    out = np.vstack([np.full((bar_h, w, 3), COL_BANNER_BG, dtype=np.uint8), img])
    cv2.putText(out, label, (10, int(bar_h * 0.72)), font, scale, color, 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = REPO / video_path
    if not video_path.exists():
        raise SystemExit(f"Clip inexistent: {video_path}")

    detector_model = YOLO(str(settings.detector_path))
    infer.load_models()  # detectorul de persoane partajat, ca in productie

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Nu pot deschide videoclipul: {video_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    effective_fps = fps_src / max(args.frame_skip, 1)
    detector = LitteringDetector(
        fps=effective_fps,
        monitor_seconds=8.0,
        pre_event_seconds=4.0,
    )

    if hasattr(detector_model, "predictor"):
        detector_model.predictor = None

    # Netezirea temporala + stabilizatorul de track-uri (identic cu
    # backend/video.py::_process_video_sync)
    _PERSON_CONFIRM = 2
    _PERSON_CLEAR = 8
    person_streak = 0
    person_stable = False
    trash_tracks: dict = {}

    # Cadrele-cheie retinute pe parcurs
    panel_person: tuple[np.ndarray, float] | None = None
    panel_monitor: tuple[np.ndarray, float] | None = None
    panel_alert: tuple[np.ndarray, float] | None = None

    frame_idx = 0
    t0 = time.time()
    print(f"Procesez {video_path.name} (fps={fps_src:.1f}, device={device})...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(args.frame_skip, 1) != 0:
            frame_idx += 1
            continue

        t_sec = frame_idx / fps_src
        h_f, w_f = frame.shape[:2]

        # Etapa 1 — tracking deseuri (ByteTrack) + filtru geometric
        results = detector_model.track(
            frame, conf=args.conf, imgsz=_TRASH_TRACK_IMGSZ,
            persist=True, tracker="bytetrack.yaml", verbose=False, device=device,
        )
        trash_dets: list[dict] = []
        boxes = results[0].boxes if results else None
        if boxes is not None and boxes.xyxy is not None:
            xyxy_list = boxes.xyxy.tolist()
            conf_list = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
            id_list = (
                [int(x) for x in boxes.id.tolist()]
                if getattr(boxes, "id", None) is not None
                else list(range(len(xyxy_list)))
            )
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

        # Etapa 2 — detectie persoane + netezire temporala
        person_boxes = infer.detect_persons(frame, conf=0.20, imgsz=1280)
        if person_boxes:
            person_streak = min(person_streak + 1, _PERSON_CONFIRM)
        else:
            person_streak = max(person_streak - 1, -_PERSON_CLEAR)
        if person_streak >= _PERSON_CONFIRM:
            person_stable = True
        elif person_streak <= -_PERSON_CLEAR:
            person_stable = False
        smoothed_person_boxes = person_boxes if person_stable else []

        # Suprimarea deseurilor suprapuse cu persoane (fals-pozitive pe corp)
        if person_boxes:
            person_filter_boxes = [_shrink_box(pb, _PERSON_FILTER_SHRINK) for pb in person_boxes]
            trash_dets = [
                d for d in trash_dets
                if not _should_suppress_overlapped_trash(d["box"], person_filter_boxes)
            ]

        # Stabilizatorul de track-uri: 4 detectii consecutive inainte de utilizare
        current_ids: set = set()
        for d in trash_dets:
            tid = d["track_id"]
            current_ids.add(tid)
            st = trash_tracks.get(tid)
            if st is None:
                trash_tracks[tid] = {"seen": 1, "miss": 0, "det": d}
            else:
                st["seen"] = min(st["seen"] + 1, 9999)
                st["miss"] = 0
                st["det"] = d
        for tid in list(trash_tracks):
            if tid not in current_ids:
                trash_tracks[tid]["miss"] += 1
                if trash_tracks[tid]["miss"] > _TRASH_GRACE_MISSES:
                    del trash_tracks[tid]

        stable_trash = [
            st["det"] for st in trash_tracks.values()
            if st["seen"] >= _TRASH_STABLE_SEEN and st["miss"] == 0
        ]

        # Etapa 3 — masina de stari
        state_before = detector.state
        event = detector.update(frame, stable_trash, smoothed_person_boxes)

        # Panou (a): ultima imagine clara cu persoana in cadru
        if state_before == DetectorState.PERSON_PRESENT and smoothed_person_boxes:
            panel_person = (annotate(frame, stable_trash, smoothed_person_boxes, detector, state_before), t_sec)

        # Panou (b): zona monitorizata dupa plecarea persoanei
        if detector.state == DetectorState.MONITORING and not smoothed_person_boxes:
            panel_monitor = (annotate(frame, stable_trash, smoothed_person_boxes, detector, DetectorState.MONITORING), t_sec)

        # Panou (c): momentul alertei — obiectul declansator evidentiat
        if event is not None:
            alert_img = annotate(frame, stable_trash, smoothed_person_boxes, detector, state_before)
            tx1, ty1, tx2, ty2 = event.trash_box
            cv2.rectangle(alert_img, (tx1, ty1), (tx2, ty2), COL_ALERT, 3)
            cv2.putText(alert_img, "OBIECT NOU", (tx1, max(20, ty1 - 10)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, COL_ALERT, 2, cv2.LINE_AA)
            panel_alert = (alert_img, t_sec)
            print(f"  Incident detectat la t={t_sec:.1f}s (metoda={event.detection_method})")
            break

        frame_idx += 1

    cap.release()
    print(f"Procesare terminata in {time.time() - t0:.1f}s")

    if panel_alert is None:
        raise SystemExit("Niciun incident detectat in clip — alege alt clip de test.")
    if panel_person is None or panel_monitor is None:
        raise SystemExit("Lipsesc cadrele intermediare (persoana/monitorizare).")

    # Salveaza panourile individuale
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    labeled = [
        (add_banner(panel_person[0], f"(a) Persoana prezenta cu obiectul  t={panel_person[1]:.1f}s", COL_PERSON), "a_persoana"),
        (add_banner(panel_monitor[0], f"(b) Monitorizarea zonei dupa plecare  t={panel_monitor[1]:.1f}s", COL_ZONE), "b_monitorizare"),
        (add_banner(panel_alert[0], f"(c) Incident confirmat: obiect nou in zona  t={panel_alert[1]:.1f}s", COL_ALERT), "c_alerta"),
    ]
    for img, suffix in labeled:
        out_p = PANELS_DIR / f"incident_{stem}_{suffix}.png"
        cv2.imwrite(str(out_p), img)
        print(f"  scris {out_p.relative_to(REPO)}")

    # Compune figura finala: (a)+(b) sus, (c) jos pe toata latimea
    a_img, b_img, c_img = labeled[0][0], labeled[1][0], labeled[2][0]
    gap = 6
    top_h = min(a_img.shape[0], b_img.shape[0])

    def resize_h(img: np.ndarray, h: int) -> np.ndarray:
        return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))

    a_r, b_r = resize_h(a_img, top_h), resize_h(b_img, top_h)
    top = np.hstack([a_r, np.full((top_h, gap, 3), 255, np.uint8), b_r])
    c_r = cv2.resize(c_img, (top.shape[1], int(c_img.shape[0] * top.shape[1] / c_img.shape[1])))
    figure = np.vstack([top, np.full((gap, top.shape[1], 3), 255, np.uint8), c_r])

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), figure)
    print(f"Figura finala: {out_path}")


if __name__ == "__main__":
    main()
