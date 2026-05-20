"""
diagnose_pipeline.py
====================
Empirical pipeline diagnostic for the production trash detector.

Tests the active production detector at different detector image sizes and
confidence thresholds on the same video clips. The goal is to separate three issues:

1. live pipeline configuration, especially low imgsz;
2. detector checkpoint quality;
3. confidence threshold behavior.

Default values follow the diagnostic plan:
  - imgsz: 320, 640, 960
  - conf: 0.15, 0.25, 0.35
  - positive littering-like clips from datasets/test_videos

Examples:
    .venv\\Scripts\\python.exe scripts\\diagnose_pipeline.py
    .venv\\Scripts\\python.exe scripts\\diagnose_pipeline.py --quick
    .venv\\Scripts\\python.exe scripts\\diagnose_pipeline.py --frame-skip 10 --max-frames 120
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MODELS = {
    "production": REPO / "models" / "detector" / "production" / "best.pt",
}

TEST_CLIPS = [
    "littering_cctv_2024.mp4",
    "dumping_neighbor_00001.mp4",
    "illegal_dumping_cctv.mp4",
    "litter_cctv_drop_00001.mp4",
    "man_throws_street_00001.mp4",
    "littering_car_toss.mp4",
    "person_litter_full_00001.mp4",
]

VIDEOS_DIR = REPO / "datasets" / "test_videos"
OUT_DIR = REPO / "results" / "diagnostic"


def parse_csv_numbers(raw: str, cast):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    if not values:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose detector model/imgsz/conf behavior")
    parser.add_argument(
        "--imgsz",
        default="320,640,960",
        help="Comma-separated detector image sizes, default: 320,640,960",
    )
    parser.add_argument(
        "--conf",
        default="0.15,0.25,0.35",
        help="Comma-separated confidence thresholds, default: 0.15,0.25,0.35",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process one frame every N frames, default: 5",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum processed frames per clip per config. 0 = no limit.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: frame-skip=15 and max-frames=120 unless explicitly overridden.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, e.g. 0 or cpu. Default: auto.",
    )
    parser.add_argument(
        "--clips",
        default=",".join(TEST_CLIPS),
        help="Comma-separated clip names from datasets/test_videos.",
    )
    parser.add_argument(
        "--out-prefix",
        default="pipeline_diagnostic",
        help="Output filename prefix inside results/diagnostic.",
    )
    return parser.parse_args()


def available_device(raw_device: str | None) -> str:
    if raw_device is not None:
        return raw_device
    return "0" if torch.cuda.is_available() else "cpu"


def video_metadata(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "total_frames": total_frames,
    }


def box_area_ratio(box, width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    area = max(0.0, float(x2 - x1) * float(y2 - y1))
    return area / max(1.0, float(width * height))


def test_config(
    model: YOLO,
    video_path: Path,
    imgsz: int,
    conf: float,
    frame_skip: int,
    max_frames: int,
    device: str,
) -> dict | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    meta = video_metadata(video_path)
    width = int(meta["width"])
    height = int(meta["height"])

    processed = 0
    frames_with_det = 0
    total_dets = 0
    confidence_scores: list[float] = []
    area_ratios: list[float] = []
    small_dets = 0
    medium_dets = 0
    large_dets = 0
    frame_idx = 0
    started = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        results = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        processed += 1

        boxes = results[0].boxes if results and results[0].boxes is not None else None
        if boxes is not None and len(boxes) > 0:
            frames_with_det += 1
            xyxy_list = boxes.xyxy.tolist() if boxes.xyxy is not None else []
            conf_list = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
            total_dets += len(xyxy_list)

            for xyxy, score in zip(xyxy_list, conf_list):
                score = float(score)
                confidence_scores.append(score)
                area = box_area_ratio(xyxy, width, height)
                area_ratios.append(area)
                if area < 0.005:
                    small_dets += 1
                elif area < 0.02:
                    medium_dets += 1
                else:
                    large_dets += 1

        frame_idx += 1
        if max_frames > 0 and processed >= max_frames:
            break

    cap.release()
    elapsed = time.time() - started
    return {
        **meta,
        "frames_processed": processed,
        "frames_with_det": frames_with_det,
        "detection_rate": round(frames_with_det / max(processed, 1), 4),
        "total_dets": total_dets,
        "avg_det_per_frame": round(total_dets / max(processed, 1), 4),
        "avg_confidence": round(float(np.mean(confidence_scores)) if confidence_scores else 0.0, 4),
        "max_confidence": round(float(np.max(confidence_scores)) if confidence_scores else 0.0, 4),
        "avg_area_ratio": round(float(np.mean(area_ratios)) if area_ratios else 0.0, 6),
        "small_dets_lt_0_5pct": small_dets,
        "medium_dets_0_5_to_2pct": medium_dets,
        "large_dets_gt_2pct": large_dets,
        "elapsed_sec": round(elapsed, 2),
    }


def write_outputs(rows: list[dict], nested: dict, out_prefix: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"{out_prefix}.json"
    csv_path = OUT_DIR / f"{out_prefix}.csv"
    json_path.write_text(json.dumps(nested, indent=2, ensure_ascii=False), encoding="utf-8")

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def print_summary(rows: list[dict], model_names: list[str], imgsz_values: list[int], conf_values: list[float]) -> None:
    print("\n" + "=" * 76)
    print("SUMAR - detection_rate mediu pe clipuri")
    print("=" * 76)
    print(f"{'Config':<22}", end="")
    for model_name in model_names:
        print(f"{model_name:>14}", end="")
    print()
    print("-" * 76)

    for imgsz in imgsz_values:
        for conf in conf_values:
            print(f"imgsz={imgsz}, conf={conf:<4}", end="")
            for model_name in model_names:
                values = [
                    row["detection_rate"]
                    for row in rows
                    if row["model"] == model_name and row["imgsz"] == imgsz and row["conf"] == conf
                ]
                avg = mean(values) if values else 0.0
                print(f"{avg:>14.3f}", end="")
            print()

    print("\nInterpretare rapida:")
    print("- daca detection_rate creste clar de la 320 la 640/960, live imgsz=320 este blocaj real;")
    print("- foloseste acest diagnostic doar pentru calibrarea imgsz/conf a modelului de productie;")
    print("- daca conf=0.15 creste mult dar poate produce fals pozitive, trebuie validare pe clipuri negative.")


def main() -> int:
    args = parse_args()
    if args.quick:
        if args.frame_skip == 5:
            args.frame_skip = 15
        if args.max_frames == 0:
            args.max_frames = 120

    imgsz_values = parse_csv_numbers(args.imgsz, int)
    conf_values = parse_csv_numbers(args.conf, float)
    clip_names = [clip.strip() for clip in args.clips.split(",") if clip.strip()]
    device = available_device(args.device)

    available_clips = [clip for clip in clip_names if (VIDEOS_DIR / clip).exists()]
    available_models = {name: path for name, path in MODELS.items() if path.exists()}

    print("Diagnostic pipeline detector")
    print(f"Device: {device}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Max frames per clip/config: {args.max_frames or 'all'}")
    print(f"Clipuri disponibile: {len(available_clips)}/{len(clip_names)}")
    for clip in available_clips:
        print(f"  - {clip}")
    missing_clips = sorted(set(clip_names) - set(available_clips))
    for clip in missing_clips:
        print(f"  ! lipseste: {clip}")

    print(f"Modele disponibile: {', '.join(available_models) if available_models else 'niciunul'}")
    if not available_clips or not available_models:
        return 1

    rows: list[dict] = []
    nested: dict = {}
    total_tests = len(available_models) * len(imgsz_values) * len(conf_values) * len(available_clips)
    current = 0

    for model_name, model_path in available_models.items():
        print("\n" + "=" * 76)
        print(f"MODEL: {model_name} ({model_path})")
        print("=" * 76)
        model = YOLO(str(model_path))
        nested[model_name] = {}

        for imgsz in imgsz_values:
            for conf in conf_values:
                config_key = f"imgsz={imgsz}_conf={conf}"
                nested[model_name][config_key] = {}
                print(f"\nConfig: imgsz={imgsz}, conf={conf}")

                for clip in available_clips:
                    current += 1
                    stats = test_config(
                        model,
                        VIDEOS_DIR / clip,
                        imgsz,
                        conf,
                        args.frame_skip,
                        args.max_frames,
                        device,
                    )
                    nested[model_name][config_key][clip] = stats
                    if stats is None:
                        print(f"  [{current}/{total_tests}] {clip:<35} ERROR")
                        continue

                    row = {
                        "model": model_name,
                        "clip": clip,
                        "imgsz": imgsz,
                        "conf": conf,
                        **stats,
                    }
                    rows.append(row)
                    print(
                        f"  [{current:03d}/{total_tests}] {clip[:35]:<35} "
                        f"rate={stats['detection_rate']:.3f} "
                        f"dets/frame={stats['avg_det_per_frame']:.2f} "
                        f"avg_conf={stats['avg_confidence']:.2f} "
                        f"small={stats['small_dets_lt_0_5pct']}"
                    )

    json_path, csv_path = write_outputs(rows, nested, args.out_prefix)
    print(f"\nSalvat JSON: {json_path}")
    print(f"Salvat CSV : {csv_path}")
    print_summary(rows, list(available_models), imgsz_values, conf_values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
