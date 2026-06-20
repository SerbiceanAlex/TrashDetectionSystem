"""
Evaluate the final video littering pipeline on multiple test clips.

This is the batch version of scripts/smoke/pipeline_e2e_smoke.py. It keeps the
same production pipeline, but writes thesis-friendly CSV/JSON summaries.

Examples:
    .venv\\Scripts\\python.exe scripts\\evaluation\\evaluate_video_events.py --clips all --frame-skip 2

    .venv\\Scripts\\python.exe scripts\\evaluation\\evaluate_video_events.py ^
        --clips littering_cctv_2024.mp4,dumping_neighbor_00001.mp4,cctv_parking_away_00001.mp4

Optional manifest format (CSV):
    clip,expected_event
    littering_cctv_2024.mp4,positive
    cctv_parking_away_00001.mp4,negative

expected_event can be: positive, negative, unknown.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.config import settings
from backend.littering_detector import LitteringDetector


VIDEOS_DIR = REPO / "datasets" / "test_videos"
DEFAULT_OUT = REPO / "results" / "video_events" / "final_video_event_eval"
VALID_EXPECTED = {"positive", "negative", "unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch video event evaluation")
    parser.add_argument(
        "--clips",
        default="all",
        help="all, or comma-separated clip names from datasets/test_videos",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional CSV with columns: clip,expected_event",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output prefix. Writes .csv and .json",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=settings.DEFAULT_DET_CONF,
        help=f"Trash detector confidence threshold (default: {settings.DEFAULT_DET_CONF})",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.40,
        help="Person detector confidence threshold (default: 0.40)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=settings.LIVE_IMGSZ,
        help=f"Inference image size (default from settings: {settings.LIVE_IMGSZ})",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame. Use 2 for faster diagnostic runs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many processed frames per clip. 0 means full clip.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device, e.g. 0 or cpu (default: 0)",
    )
    return parser.parse_args()


def list_available_clips() -> list[Path]:
    return sorted(VIDEOS_DIR.glob("*.mp4"))


def parse_clip_selection(raw: str) -> list[Path]:
    if raw.strip().lower() == "all":
        return list_available_clips()

    selected: list[Path] = []
    for name in raw.split(","):
        clean = name.strip()
        if not clean:
            continue
        path = VIDEOS_DIR / clean
        if not path.exists():
            raise FileNotFoundError(f"Clip not found: {path}")
        selected.append(path)
    return selected


def load_manifest(path_raw: str) -> dict[str, str]:
    if not path_raw:
        return {}

    path = Path(path_raw)
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"clip", "expected_event"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            clip = (row.get("clip") or "").strip()
            expected = (row.get("expected_event") or "unknown").strip().lower()
            if not clip:
                continue
            if expected not in VALID_EXPECTED:
                raise ValueError(
                    f"Invalid expected_event for {clip}: {expected}. "
                    f"Use one of {sorted(VALID_EXPECTED)}"
                )
            labels[clip] = expected
    return labels


def outcome(expected: str, events_detected: int) -> str:
    predicted_positive = events_detected > 0
    if expected == "unknown":
        return "unknown"
    if expected == "positive" and predicted_positive:
        return "TP"
    if expected == "positive" and not predicted_positive:
        return "FN"
    if expected == "negative" and predicted_positive:
        return "FP"
    if expected == "negative" and not predicted_positive:
        return "TN"
    return "unknown"


def percent_state(state_counts: Counter[str], state: str, total: int) -> float:
    return round(100.0 * state_counts.get(state, 0) / max(total, 1), 2)


def run_clip(
    clip_path: Path,
    detector_model: YOLO,
    person_model: YOLO,
    *,
    conf: float,
    person_conf: float,
    imgsz: int,
    frame_skip: int,
    max_frames: int,
    device: str,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps_src if fps_src else 0.0

    effective_fps = fps_src / max(frame_skip, 1)
    event_detector = LitteringDetector(
        fps=effective_fps,
        monitor_seconds=10.0,
        pre_event_seconds=5.0,
        person_conf=person_conf,
    )

    state_counts: Counter[str] = Counter()
    transitions: Counter[str] = Counter()
    events: list[dict[str, Any]] = []

    frame_idx = 0
    processed = 0
    previous_state: str | None = None
    started = time.time()

    # Match the original single-clip smoke test: each clip starts with a fresh
    # Ultralytics predictor/tracker, then ByteTrack persists across frames.
    if hasattr(detector_model, "predictor"):
        detector_model.predictor = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        trash_results = detector_model.track(
            frame,
            conf=conf,
            imgsz=imgsz,
            persist=True,
            verbose=False,
            device=device,
        )
        trash_dets: list[dict[str, Any]] = []
        if trash_results and trash_results[0].boxes is not None:
            for i, box in enumerate(trash_results[0].boxes):
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                score = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else i
                trash_dets.append(
                    {
                        "track_id": tid,
                        "box": (x1, y1, x2, y2),
                        "det_score": score,
                        "material_name": "unknown",
                        "material_score": 0.0,
                    }
                )

        person_results = person_model(
            frame,
            conf=person_conf,
            classes=[0],
            imgsz=imgsz,
            verbose=False,
            device=device,
        )
        person_boxes: list[tuple[int, int, int, int]] = []
        if person_results and person_results[0].boxes is not None:
            for box in person_results[0].boxes:
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                person_boxes.append((x1, y1, x2, y2))

        event = event_detector.update(frame, trash_dets, person_boxes)
        if event is not None:
            events.append(
                {
                    "frame_idx": frame_idx,
                    "time_sec": round(frame_idx / fps_src, 2),
                    "method": event.detection_method,
                    "trash_count": len(trash_dets),
                    "person_count": len(person_boxes),
                    "det_score": round(float(event.det_score), 4),
                }
            )

        state = event_detector.state.name
        state_counts[state] += 1
        if previous_state and previous_state != state:
            transitions[f"{previous_state}->{state}"] += 1
        previous_state = state

        processed += 1
        frame_idx += 1
        if max_frames and processed >= max_frames:
            break

    cap.release()

    # Flush orice candidat aflat în fereastra de confirmare la final de clip,
    # exact ca procesarea de upload din producție (_process_video_sync).
    final_event = event_detector.finalize()
    if final_event is not None:
        events.append({
            "frame_idx": frame_idx,
            "time_sec": round(frame_idx / fps_src, 2),
            "method": final_event.detection_method,
            "trash_count": -1,
            "person_count": -1,
            "det_score": round(float(final_event.det_score), 4),
        })

    elapsed = time.time() - started
    fps_runtime = processed / elapsed if elapsed > 0 else 0.0
    first_event_time = events[0]["time_sec"] if events else ""

    return {
        "clip": clip_path.name,
        "video_fps": round(float(fps_src), 2),
        "effective_fps": round(float(effective_fps), 2),
        "duration_sec": round(float(duration_sec), 2),
        "total_frames": total_frames,
        "frames_processed": processed,
        "frame_skip": frame_skip,
        "conf": conf,
        "person_conf": person_conf,
        "imgsz": imgsz,
        "events_detected": len(events),
        "first_event_time_sec": first_event_time,
        "runtime_sec": round(elapsed, 2),
        "runtime_fps": round(fps_runtime, 2),
        "state_CLEAR_pct": percent_state(state_counts, "CLEAR", processed),
        "state_PERSON_PRESENT_pct": percent_state(state_counts, "PERSON_PRESENT", processed),
        "state_MONITORING_pct": percent_state(state_counts, "MONITORING", processed),
        "transitions_count": sum(transitions.values()),
        "transitions": dict(transitions),
        "events": events,
    }


def write_outputs(rows: list[dict[str, Any]], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")

    csv_columns = [
        "clip",
        "expected_event",
        "outcome",
        "events_detected",
        "first_event_time_sec",
        "frames_processed",
        "video_fps",
        "effective_fps",
        "duration_sec",
        "runtime_fps",
        "conf",
        "person_conf",
        "imgsz",
        "frame_skip",
        "state_CLEAR_pct",
        "state_PERSON_PRESENT_pct",
        "state_MONITORING_pct",
        "transitions_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in csv_columns})

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== Batch Summary ===")
    print(f"{'clip':<34} {'exp':<8} {'out':<4} {'evt':>3} {'first':>7} {'fps':>7}")
    print("-" * 72)
    for row in rows:
        first = row.get("first_event_time_sec", "")
        first_str = f"{first}s" if first != "" else "-"
        print(
            f"{row['clip']:<34} "
            f"{row.get('expected_event', 'unknown'):<8} "
            f"{row.get('outcome', 'unknown'):<4} "
            f"{row['events_detected']:>3} "
            f"{first_str:>7} "
            f"{row['runtime_fps']:>7.1f}"
        )

    known = [r for r in rows if r.get("outcome") in {"TP", "FP", "TN", "FN"}]
    if known:
        counts = Counter(r["outcome"] for r in known)
        tp = counts.get("TP", 0)
        fp = counts.get("FP", 0)
        tn = counts.get("TN", 0)
        fn = counts.get("FN", 0)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print("\nKnown-label clip metrics:")
        print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"  precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    else:
        print("\nNo known expected_event labels were provided; outcome metrics are skipped.")


def main() -> int:
    args = parse_args()
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be >= 1")

    clips = parse_clip_selection(args.clips)
    if not clips:
        print("No clips selected.")
        return 1

    expected_by_clip = load_manifest(args.manifest)
    out_prefix = Path(args.out)
    if not out_prefix.is_absolute():
        out_prefix = REPO / out_prefix

    detector_path = settings.detector_path
    person_path = settings.person_detector_path
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector not found: {detector_path}")
    if not person_path.exists():
        raise FileNotFoundError(f"Person model not found: {person_path}")

    print("=== Video Event Evaluation ===")
    print(f"Detector : {detector_path}")
    print(f"Person   : {person_path}")
    print(f"Clips    : {len(clips)}")
    print(f"conf     : {args.conf}")
    print(f"imgsz    : {args.imgsz}")
    print(f"skip     : {args.frame_skip}")
    print()

    detector_model = YOLO(str(detector_path))
    person_model = YOLO(str(person_path))

    rows: list[dict[str, Any]] = []
    started = time.time()
    for idx, clip in enumerate(clips, start=1):
        print(f"[{idx}/{len(clips)}] {clip.name}")
        try:
            row = run_clip(
                clip,
                detector_model,
                person_model,
                conf=args.conf,
                person_conf=args.person_conf,
                imgsz=args.imgsz,
                frame_skip=args.frame_skip,
                max_frames=args.max_frames,
                device=args.device,
            )
        except Exception as exc:
            row = {
                "clip": clip.name,
                "error": str(exc),
                "events_detected": "",
                "first_event_time_sec": "",
                "frames_processed": 0,
                "runtime_fps": 0.0,
            }
            print(f"  ERROR: {exc}")

        expected = expected_by_clip.get(clip.name, "unknown")
        row["expected_event"] = expected
        row["outcome"] = outcome(expected, int(row["events_detected"] or 0))
        rows.append(row)

        if row.get("error"):
            print("  -> failed")
        else:
            print(
                f"  -> events={row['events_detected']} "
                f"first={row['first_event_time_sec'] or '-'} "
                f"speed={row['runtime_fps']:.1f} FPS"
            )

    elapsed = time.time() - started
    write_outputs(rows, out_prefix)
    print_summary(rows)
    print(f"\nTotal runtime: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
