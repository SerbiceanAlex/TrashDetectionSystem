"""
evaluate_littering.py — Harness offline de evaluare pentru detecția actului de aruncare.

Rulează pipeline-ul complet `/ws/video/monitor` offline pe clipuri video, compară rezultatele
cu ground truth-ul din `datasets/littering_eval/annotations.json` și raportează
precision / recall / F1 / latency median.

Reutilizează EXACT aceleași constante și helpers ca `backend/video.py:handle_monitor_ws` —
fără drift între evaluare și live pipeline.

Rulare:
    .venv/Scripts/python.exe scripts/evaluate_littering.py
    .venv/Scripts/python.exe scripts/evaluate_littering.py --det-conf 0.35 --person-conf 0.40

Output:
    datasets/littering_eval/baseline_results.csv
    Sumar pe stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Reuse live-pipeline helpers + constants — critical pentru paritate evaluare↔producție
from backend import inference as infer  # noqa: E402
from backend.littering_detector import LitteringDetector  # noqa: E402
from backend.video import (  # noqa: E402
    _PERSON_FILTER_SHRINK,
    _TRASH_GRACE_MISSES,
    _TRASH_STABLE_SEEN,
    _TRASH_TRACK_IMGSZ,
    _shrink_box,
    _should_suppress_overlapped_trash,
    _valid_trash_box,
)

# Aceleași praguri de hysteresis ca handle_monitor_ws
_PERSON_CONFIRM = 2
_PERSON_CLEAR = 8


def evaluate_clip(
    clip_path: Path,
    fps: float = 25.0,
    det_conf: float = 0.40,
    person_conf: float = 0.40,
) -> tuple[list[int], int]:
    """
    Rulează pipeline-ul complet pe un clip. Returnează:
      - lista de frame_idx unde s-a declanșat o alertă
      - total_frames procesate
    """
    from ultralytics import YOLO
    import torch
    from backend.config import settings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = YOLO(str(settings.detector_path))
    tracker.to(device)

    detector = LitteringDetector(fps=fps, monitor_seconds=10.0, pre_event_seconds=5.0)

    person_streak = 0
    person_stable = False
    trash_tracks: dict[int, dict] = {}

    alerts: list[int] = []
    frame_idx = 0

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 nu poate deschide {clip_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # ── Stage 1: trash tracking
            results = tracker.track(
                frame,
                conf=det_conf,
                imgsz=_TRASH_TRACK_IMGSZ,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
            )
            boxes = results[0].boxes
            trash_dets: list[dict] = []
            if boxes is not None and boxes.xyxy is not None:
                xyxy_list = boxes.xyxy.tolist()
                conf_list = (
                    boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
                )
                id_list = (
                    [int(x) for x in boxes.id.tolist()]
                    if (hasattr(boxes, "id") and boxes.id is not None)
                    else list(range(len(xyxy_list)))
                )
                h_f, w_f = frame.shape[:2]
                for xyxy, dscore, tid in zip(xyxy_list, conf_list, id_list):
                    x1 = max(0, int(xyxy[0]))
                    y1 = max(0, int(xyxy[1]))
                    x2 = min(w_f, int(xyxy[2]))
                    y2 = min(h_f, int(xyxy[3]))
                    if _valid_trash_box((x1, y1, x2, y2), w_f, h_f):
                        trash_dets.append(
                            {
                                "track_id": tid,
                                "box": (x1, y1, x2, y2),
                                "det_score": float(dscore),
                                "material_name": "unknown",
                            }
                        )

            # ── Stage 2: person detection
            person_boxes = infer.detect_persons(frame, conf=max(person_conf, 0.28), imgsz=640)

            # Temporal smoothing (hysteresis)
            if len(person_boxes) > 0:
                person_streak = min(person_streak + 1, _PERSON_CONFIRM)
            else:
                person_streak = max(person_streak - 1, -_PERSON_CLEAR)
            if person_streak >= _PERSON_CONFIRM:
                person_stable = True
            elif person_streak <= -_PERSON_CLEAR:
                person_stable = False

            smoothed_person_boxes = person_boxes if person_stable else []

            # Body-overlap suppression (păstrează handheld mic)
            if person_boxes:
                pfboxes = [_shrink_box(pb, _PERSON_FILTER_SHRINK) for pb in person_boxes]
                trash_dets = [
                    d
                    for d in trash_dets
                    if not _should_suppress_overlapped_trash(d["box"], pfboxes)
                ]

            # Track-level stabilizer
            current_ids: set[int] = set()
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

            for tid in list(trash_tracks.keys()):
                if tid in current_ids:
                    continue
                st = trash_tracks[tid]
                st["miss"] += 1
                if st["miss"] > _TRASH_GRACE_MISSES:
                    del trash_tracks[tid]

            detector_trash_dets = [
                st["det"]
                for st in trash_tracks.values()
                if st["seen"] >= _TRASH_STABLE_SEEN and st["miss"] == 0
            ]

            # ── Stage 3: state machine
            event = detector.update(frame, detector_trash_dets, smoothed_person_boxes)
            if event is not None:
                alerts.append(frame_idx)

    finally:
        cap.release()
        detector.reset()

    return alerts, frame_idx


def classify_result(
    label_gt: str,
    alerts: list[int],
    onset_gt: int | None,
    tol_before: int,
    tol_after: int,
) -> str:
    """
    Returnează:
      "TP" — clip etichetat TP și sistemul a alertat în toleranța onset-ului
      "FN" — clip TP dar sistemul nu a alertat (sau în afara toleranței)
      "FP" — clip etichetat FP și sistemul a alertat (confirmă FP)
      "TN" — clip FP și sistemul NU a alertat (corect respins)
    """
    if label_gt == "TP":
        if not alerts:
            return "FN"
        if onset_gt is None:
            return "TP"
        for a in alerts:
            if onset_gt - tol_before <= a <= onset_gt + tol_after:
                return "TP"
        return "FN"
    return "FP" if alerts else "TN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        default=str(REPO_ROOT / "scripts/littering_annotations.json"),
        help="Cale catre annotations.json",
    )
    parser.add_argument(
        "--out-csv",
        default=str(REPO_ROOT / "results/littering/baseline_results.csv"),
        help="Cale CSV output",
    )
    parser.add_argument("--det-conf", type=float, default=0.40)
    parser.add_argument("--person-conf", type=float, default=0.40)
    parser.add_argument("--tol-before", type=int, default=30, help="Frames înainte de onset tolerate")
    parser.add_argument("--tol-after", type=int, default=120, help="Frames după onset tolerate")
    args = parser.parse_args()

    ann_path = Path(args.annotations).resolve()
    with ann_path.open(encoding="utf-8") as f:
        ann = json.load(f)

    print("[INFO] Încarc modelele YOLO...")
    t_load = time.perf_counter()
    infer.load_models()
    print(f"[INFO] Modele încărcate în {time.perf_counter() - t_load:.1f}s\n")

    ann_dir = ann_path.parent
    results: list[dict] = []

    for clip_ann in ann["clips"]:
        clip_path = (ann_dir / clip_ann["path"]).resolve()
        if not clip_path.exists():
            print(f"[WARN] Clip lipsă: {clip_path}")
            continue

        print(
            f"[CLIP] {clip_ann['clip_id']:22s}  label={clip_ann['label']}  "
            f"scene={clip_ann.get('scene', '')[:40]}"
        )
        t0 = time.perf_counter()
        alerts, total_frames = evaluate_clip(
            clip_path,
            fps=clip_ann.get("fps", 25.0),
            det_conf=args.det_conf,
            person_conf=args.person_conf,
        )
        elapsed = time.perf_counter() - t0

        outcome = classify_result(
            clip_ann["label"],
            alerts,
            clip_ann.get("onset_frame_gt"),
            tol_before=args.tol_before,
            tol_after=args.tol_after,
        )

        latency = None
        if outcome == "TP" and clip_ann.get("onset_frame_gt") is not None and alerts:
            # Prima alertă în toleranță
            for a in alerts:
                if clip_ann["onset_frame_gt"] - args.tol_before <= a <= clip_ann["onset_frame_gt"] + args.tol_after:
                    latency = a - clip_ann["onset_frame_gt"]
                    break

        print(
            f"       → {outcome:3s}  alerts={alerts}  frames={total_frames}  "
            f"eval={elapsed:.1f}s"
        )

        results.append(
            {
                "clip_id": clip_ann["clip_id"],
                "label_gt": clip_ann["label"],
                "scene": clip_ann.get("scene", ""),
                "trash_type": clip_ann.get("trash_type", ""),
                "onset_gt": clip_ann.get("onset_frame_gt"),
                "first_alert_frame": alerts[0] if alerts else None,
                "n_alerts": len(alerts),
                "total_frames": total_frames,
                "latency_frames": latency,
                "outcome": outcome,
                "eval_time_s": round(elapsed, 2),
            }
        )

    if not results:
        print("[ERR] Niciun clip procesat.")
        return 1

    # ── CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ── Sumar
    tp = sum(1 for r in results if r["outcome"] == "TP")
    fp = sum(1 for r in results if r["outcome"] == "FP")
    fn = sum(1 for r in results if r["outcome"] == "FN")
    tn = sum(1 for r in results if r["outcome"] == "TN")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    latencies = [r["latency_frames"] for r in results if r["latency_frames"] is not None]
    med_lat = statistics.median(latencies) if latencies else None

    print()
    print("=" * 60)
    print("  SUMAR EVALUARE LITTERING")
    print("=" * 60)
    print(f"  Parametri: det_conf={args.det_conf} person_conf={args.person_conf} "
          f"tol=±[{args.tol_before},{args.tol_after}]")
    print(f"  Total clipuri:  {len(results)}")
    print(f"  Confuzion matrix:")
    print(f"    TP (prins corect):        {tp}")
    print(f"    FN (ratat aruncare):      {fn}")
    print(f"    FP (alertă falsă):        {fp}")
    print(f"    TN (respins corect):      {tn}")
    print(f"  Precision:    {precision:.3f}")
    print(f"  Recall:       {recall:.3f}")
    print(f"  F1:           {f1:.3f}")
    if med_lat is not None:
        print(f"  Latency (median): {med_lat:.0f} frames ({med_lat / 25:.2f}s @25fps)")
    print(f"\n  CSV salvat: {out_csv}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
