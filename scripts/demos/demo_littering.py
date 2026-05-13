"""
Standalone visual demo for illegal littering detection.

Runs on a video file or on a live camera and shows:
  - trash detections with persistent track IDs
  - person detections
  - detector state: CLEAR / PERSON_PRESENT / MONITORING
  - alert flash when LitteringDetector emits an event

Examples:
    .venv\\Scripts\\python.exe scripts\\demos\\demo_littering.py --video datasets\\test_videos\\clip.mp4
    .venv\\Scripts\\python.exe scripts\\demos\\demo_littering.py --camera 0
    .venv\\Scripts\\python.exe scripts\\demos\\demo_littering.py --camera http://192.168.x.x:4747/video
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

from backend.littering_detector import DetectorState, LitteringDetector


DEFAULT_DETECTOR = REPO / "models" / "detector" / "A4-8010" / "best.pt"
DEFAULT_PERSON = REPO / "yolov8n.pt"

COL_TRASH = (0, 60, 255)
COL_PERSON = (255, 140, 0)
COL_CLEAR = (180, 180, 180)
COL_PERSON_STATE = (50, 200, 50)
COL_MONITOR = (0, 200, 255)
COL_ALERT = (0, 0, 255)

STATE_COLORS = {
    DetectorState.CLEAR: COL_CLEAR,
    DetectorState.PERSON_PRESENT: COL_PERSON_STATE,
    DetectorState.MONITORING: COL_MONITOR,
}

STATE_LABELS = {
    DetectorState.CLEAR: "CLEAR",
    DetectorState.PERSON_PRESENT: "PERSOANA DETECTATA",
    DetectorState.MONITORING: "MONITORIZARE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo detectie aruncare ilegala")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str, help="Cale catre fisier video")
    source.add_argument("--camera", type=str, help="Index camera (0, 1) sau URL stream")
    parser.add_argument("--save", action="store_true", help="Salveaza output in outputs/demo_output.mp4")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR, help="Cale catre detectorul YOLO")
    parser.add_argument("--person-model", type=Path, default=DEFAULT_PERSON, help="Cale catre yolov8n.pt")
    parser.add_argument("--det-conf", type=float, default=0.30, help="Prag confidenta trash")
    parser.add_argument("--person-conf", type=float, default=0.40, help="Prag confidenta persoana")
    parser.add_argument("--imgsz", type=int, default=640, help="Dimensiune inferenta YOLO")
    parser.add_argument("--device", default=None, help="Ex: 0 pentru GPU, cpu pentru CPU; implicit auto")
    return parser.parse_args()


def yolo_kwargs(args: argparse.Namespace) -> dict:
    return {"device": args.device} if args.device is not None else {}


def clamp_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int] | None:
    left = max(0, int(round(x1)))
    top = max(0, int(round(y1)))
    right = min(w, int(round(x2)))
    bottom = min(h, int(round(y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def detect_trash(model: YOLO, frame: np.ndarray, args: argparse.Namespace) -> list[dict]:
    h, w = frame.shape[:2]
    results = model.track(
        frame,
        conf=args.det_conf,
        imgsz=args.imgsz,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        **yolo_kwargs(args),
    )
    boxes = results[0].boxes if results else None
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy_list = boxes.xyxy.tolist()
    conf_list = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
    id_list = (
        [int(x) for x in boxes.id.tolist()]
        if getattr(boxes, "id", None) is not None
        else list(range(len(xyxy_list)))
    )

    detections: list[dict] = []
    for idx, (xyxy, score, track_id) in enumerate(zip(xyxy_list, conf_list, id_list)):
        box = clamp_box(*xyxy, w, h)
        if box is None:
            continue
        detections.append({
            "index": idx,
            "track_id": int(track_id),
            "box": box,
            "det_score": float(score),
            "material_name": "unknown",
            "material_score": 0.0,
        })
    return detections


def detect_persons(model: YOLO, frame: np.ndarray, args: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    results = model.predict(
        frame,
        conf=args.person_conf,
        classes=[0],
        imgsz=args.imgsz,
        verbose=False,
        **yolo_kwargs(args),
    )
    boxes = results[0].boxes if results else None
    if boxes is None or boxes.xyxy is None:
        return []

    persons = []
    for xyxy in boxes.xyxy.tolist():
        box = clamp_box(*xyxy, w, h)
        if box is not None:
            persons.append(box)
    return persons


def draw_overlay(
    frame: np.ndarray,
    state: DetectorState,
    detector: LitteringDetector,
    trash_dets: list[dict],
    person_boxes: list[tuple[int, int, int, int]],
    fps_actual: float,
    event_count: int,
    alert_flash: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    if alert_flash:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), COL_ALERT, -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for det in trash_dets:
        x1, y1, x2, y2 = det["box"]
        score = det.get("det_score", 0.0)
        track_id = det.get("track_id")
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_TRASH, 2)
        cv2.putText(
            frame,
            f"trash {score:.2f} #{track_id}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COL_TRASH,
            2,
            cv2.LINE_AA,
        )

    for x1, y1, x2, y2 in person_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_PERSON, 2)
        cv2.putText(
            frame,
            "persoana",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COL_PERSON,
            2,
            cv2.LINE_AA,
        )

    if state == DetectorState.MONITORING and detector._person_zones:
        for zone in detector._person_zones:
            zone_box = zone.expanded(detector.zone_expand)
            x1, y1, x2, y2 = zone_box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            zone_overlay = frame.copy()
            cv2.rectangle(zone_overlay, (x1, y1), (x2, y2), COL_MONITOR, -1)
            cv2.addWeighted(zone_overlay, 0.14, frame, 0.86, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COL_MONITOR, 2)

    panel_h = 74
    cv2.rectangle(frame, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    color = COL_ALERT if alert_flash else STATE_COLORS.get(state, COL_CLEAR)
    label = "ARUNCARE DETECTATA!" if alert_flash else STATE_LABELS.get(state, state.name)
    cv2.putText(frame, label, (12, h - 42), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"FPS: {fps_actual:.0f} | Alerte: {event_count} | Trash: {len(trash_dets)} | Persoane: {len(person_boxes)}",
        (12, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    cv2.circle(frame, (w - 30, 30), 18, color, -1)
    cv2.circle(frame, (w - 30, 30), 18, (255, 255, 255), 2)
    return frame


def run_demo(args: argparse.Namespace) -> int:
    source = args.video if args.video else args.camera
    assert source is not None

    if not args.detector.exists():
        print(f"EROARE: detectorul nu exista: {args.detector}")
        return 2
    if not args.person_model.exists():
        print(f"EROARE: modelul de persoane nu exista: {args.person_model}")
        return 2

    print("\nTrashDetection - Demo Littering")
    print(f"Sursa    : {source}")
    print(f"Detector : {args.detector}")
    print("Taste    : Q/ESC=iesire, SPACE=pauza, S=screenshot\n")

    print("Incarc modele...", end="", flush=True)
    detector_model = YOLO(str(args.detector))
    person_model = YOLO(str(args.person_model))
    print(" OK")

    cap_source: int | str = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"EROARE: nu pot deschide sursa: {source}")
        return 2

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    littering = LitteringDetector(fps=fps_src, monitor_seconds=10.0, pre_event_seconds=5.0)

    writer = None
    if args.save:
        out_path = REPO / "outputs" / "demo_output.mp4"
        out_path.parent.mkdir(exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (width, height))
        print(f"Salvez output la: {out_path}")

    window_name = "TrashDetection - Demo Littering"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(width, 1280), min(height, 720))

    event_count = 0
    frame_n = 0
    paused = False
    alert_until_frame = -1
    fps_display = 0.0
    t_prev = time.time()
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("s"):
                shot = REPO / "outputs" / f"screenshot_{frame_n:05d}.jpg"
                shot.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(shot), frame)
                print(f"Screenshot salvat: {shot}")

            if paused:
                cv2.imshow(window_name, frame)
                continue

            ok, frame = cap.read()
            if not ok:
                final_event = littering.finalize()
                if final_event is not None:
                    event_count += 1
                    print(f"[ALERT] Eveniment finalizat la sfarsitul clipului #{event_count}")
                print("Video terminat.")
                break

            frame_n += 1
            trash_dets = detect_trash(detector_model, frame, args)
            person_boxes = detect_persons(person_model, frame, args)

            event = littering.update(frame, trash_dets, person_boxes)
            if event is not None:
                event_count += 1
                alert_until_frame = frame_n + int(2.5 * fps_src)
                print(
                    f"[ALERT] Aruncare detectata #{event_count} "
                    f"la frame {frame_n} ({event.detection_method})"
                )

            now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            alert_flash = frame_n <= alert_until_frame
            frame = draw_overlay(
                frame,
                littering.state,
                littering,
                trash_dets,
                person_boxes,
                fps_display,
                event_count,
                alert_flash,
            )

            if total > 0:
                progress = int(width * frame_n / total)
                cv2.rectangle(frame, (0, height - 76), (progress, height - 72), (100, 100, 255), -1)

            if writer is not None:
                writer.write(frame)

            cv2.imshow(window_name, frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\nRezultat: {event_count} evenimente detectate")
    print(f"Frame-uri procesate: {frame_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_demo(parse_args()))
