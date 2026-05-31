"""
pipeline_e2e_smoke.py
=====================
End-to-end smoke test for the full littering detection pipeline.

Runs the actual production pipeline (detector + person detection + state machine)
on real test clips and reports:
  - Number of frames processed
  - State machine transitions (CLEAR -> PERSON -> MONITORING -> EVENT)
  - Final number of detected littering events
  - Average FPS

Use this to validate that:
  1. The model loads from the configured path
  2. The state machine correctly transitions
  3. Events are generated on positive clips
  4. False positives are minimal on negative clips

Usage:
    .venv\\Scripts\\python.exe scripts\\smoke\\pipeline_e2e_smoke.py
    .venv\\Scripts\\python.exe scripts\\smoke\\pipeline_e2e_smoke.py --clip dumping_neighbor_00001.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.config import settings
from backend.littering_detector import LitteringDetector, DetectorState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E pipeline smoke test")
    parser.add_argument(
        "--clip",
        default="littering_cctv_2024.mp4",
        help="Clip name from datasets/test_videos/",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Detector confidence threshold (default: 0.30)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=settings.LIVE_IMGSZ,
        help=f"Detector image size (default from settings: {settings.LIVE_IMGSZ})",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)",
    )
    return parser.parse_args()


def run_smoke(clip_path: Path, conf: float, imgsz: int, frame_skip: int) -> dict:
    print(f"\n=== E2E Smoke Test ===")
    print(f"Clip      : {clip_path.name}")
    print(f"Detector  : {settings.DETECTOR_WEIGHTS}")
    print(f"imgsz     : {imgsz}")
    print(f"conf      : {conf}")
    print(f"frame_skip: {frame_skip}")
    print()

    # Load models
    detector_path = settings.detector_path
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector not found: {detector_path}")

    print("Loading detector...")
    detector_model = YOLO(str(detector_path))
    person_path = settings.person_detector_path
    print(f"Loading person detector: {person_path}")
    person_model = YOLO(str(person_path))

    # Open video
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps_src:.1f}, total frames: {total_frames}\n")

    # Initialize state machine
    detector = LitteringDetector(fps=fps_src, monitor_seconds=10.0, pre_event_seconds=5.0)

    # Stats
    state_history: list[str] = []
    state_counts: Counter[str] = Counter()
    events: list[dict] = []
    frame_idx = 0
    processed = 0
    started = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Detect trash with tracking
        trash_results = detector_model.track(
            frame, conf=conf, imgsz=imgsz,
            persist=True, verbose=False, device="0",
        )
        trash_dets = []
        if trash_results and trash_results[0].boxes is not None:
            for box in trash_results[0].boxes:
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                score = float(box.conf[0])
                tid = int(box.id[0]) if box.id is not None else None
                trash_dets.append({
                    "track_id": int(tid) if tid is not None else len(trash_dets),
                    "box": (x1, y1, x2, y2),
                    "det_score": score,
                    "material_name": "unknown",
                    "material_score": 0.0,
                })

        # Detect persons (yolov8n COCO class 0)
        person_results = person_model(
            frame, conf=0.40, classes=[0],
            imgsz=imgsz, verbose=False, device="0",
        )
        person_boxes = []
        if person_results and person_results[0].boxes is not None:
            for box in person_results[0].boxes:
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                person_boxes.append((x1, y1, x2, y2))

        # Update state machine
        event = detector.update(frame, trash_dets, person_boxes)
        if event:
            events.append({
                "frame_idx": frame_idx,
                "time_sec": round(frame_idx / fps_src, 2),
                "trash_count": len(trash_dets),
                "person_count": len(person_boxes),
            })

        # Track state
        state_str = detector.state.name
        state_history.append(state_str)
        state_counts[state_str] += 1

        frame_idx += 1
        processed += 1

        # Progress log every 100 frames
        if processed % 100 == 0:
            elapsed = time.time() - started
            fps_real = processed / elapsed if elapsed > 0 else 0
            print(f"  frame {frame_idx}/{total_frames}  "
                  f"state={state_str}  "
                  f"events={len(events)}  "
                  f"speed={fps_real:.1f} FPS")

    cap.release()
    elapsed_total = time.time() - started
    fps_avg = processed / elapsed_total if elapsed_total > 0 else 0

    print(f"\n=== Results ===")
    print(f"Frames processed   : {processed}")
    print(f"Total time         : {elapsed_total:.1f}s")
    print(f"Average FPS        : {fps_avg:.1f}")
    print(f"\nState distribution:")
    for state, count in state_counts.most_common():
        pct = 100 * count / max(processed, 1)
        print(f"  {state:<20} {count:>5} frames ({pct:>5.1f}%)")

    print(f"\nLittering events detected: {len(events)}")
    for i, evt in enumerate(events, 1):
        print(f"  Event {i}: frame {evt['frame_idx']} "
              f"@ {evt['time_sec']}s "
              f"(trash={evt['trash_count']}, persons={evt['person_count']})")

    # Transition analysis
    print(f"\nState transitions:")
    transitions = Counter()
    for prev, curr in zip(state_history, state_history[1:]):
        if prev != curr:
            transitions[f"{prev} -> {curr}"] += 1
    if transitions:
        for trans, count in transitions.most_common():
            print(f"  {trans:<40} {count}x")
    else:
        print("  (no transitions — state remained unchanged)")

    return {
        "clip": clip_path.name,
        "frames_processed": processed,
        "events_detected": len(events),
        "fps_avg": round(fps_avg, 1),
        "state_counts": dict(state_counts),
        "transitions": dict(transitions),
    }


def main():
    args = parse_args()
    clip_path = REPO / "datasets" / "test_videos" / args.clip

    if not clip_path.exists():
        print(f"ERROR: Clip not found: {clip_path}")
        print(f"Available clips:")
        videos_dir = REPO / "datasets" / "test_videos"
        for v in sorted(videos_dir.glob("*.mp4"))[:10]:
            print(f"  {v.name}")
        sys.exit(1)

    try:
        result = run_smoke(clip_path, args.conf, args.imgsz, args.frame_skip)
        print("\n[OK] Smoke test passed.")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
