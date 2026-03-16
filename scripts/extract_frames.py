"""extract_frames.py – Extract frames from videos for dataset building.

Usage
-----
python scripts/extract_frames.py \\
    --videos-dir data/videos \\
    --frames-dir datasets/raw/frames \\
    --every-seconds 1.0 \\
    --max-frames 300 \\
    --overwrite

For each video file the frames are written to:
    <frames-dir>/<video_stem>/frame_<NNNNNN>.jpg

A manifest CSV is written to <frames-dir>/manifest.csv with columns:
    video_id, frame_index, timestamp_s, path
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frames from videos at a fixed time interval."
    )
    p.add_argument(
        "--videos-dir",
        default="data/videos",
        help="Directory containing input video files (default: data/videos)",
    )
    p.add_argument(
        "--frames-dir",
        default="datasets/raw/frames",
        help="Root output directory for extracted frames (default: datasets/raw/frames)",
    )
    p.add_argument(
        "--every-seconds",
        type=float,
        default=1.0,
        help="Extract one frame every N seconds (default: 1.0)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to extract per video (default: unlimited)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract frames even if the output directory already exists",
    )
    return p.parse_args()


def extract_video(
    video_path: Path,
    out_dir: Path,
    every_seconds: float,
    max_frames: int | None,
    overwrite: bool,
) -> list[dict]:
    """Extract frames from a single video file.

    Returns a list of metadata dicts (one per saved frame).
    """
    video_id = video_path.stem

    if out_dir.exists() and not overwrite:
        existing = sorted(out_dir.glob("frame_*.jpg"))
        if existing:
            print(f"  [SKIP] {video_id}: {len(existing)} frames already exist (use --overwrite)")
            return []

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}; skipping.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(fps * every_seconds))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    records: list[dict] = []
    frame_pos = 0   # position in the video stream
    saved_count = 0       # number of frames written

    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_pos % frame_interval == 0:
            filename = f"frame_{saved_count:06d}.jpg"
            out_path = out_dir / filename
            cv2.imwrite(str(out_path), frame)

            records.append(
                {
                    "video_id": video_id,
                    "frame_index": saved_count,
                    "timestamp_s": round(frame_pos / fps, 3),
                    "path": str(out_path),
                }
            )
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_pos += 1

    cap.release()
    elapsed = time.time() - t0
    est_total = f"/{total_frames}" if total_frames > 0 else ""
    print(
        f"  [OK]   {video_id}: {saved_count} frames extracted "
        f"(video frames{est_total}, {elapsed:.1f}s)"
    )
    return records


def save_manifest(records: list[dict], frames_dir: Path) -> None:
    manifest_path = frames_dir / "manifest.csv"
    fieldnames = ["video_id", "frame_index", "timestamp_s", "path"]
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[MANIFEST] {len(records)} rows → {manifest_path}")


def main() -> None:
    args = parse_args()

    videos_dir = Path(args.videos_dir)
    frames_dir = Path(args.frames_dir)

    if not videos_dir.exists():
        raise SystemExit(f"[ERROR] Videos directory not found: {videos_dir}")

    video_files = sorted(
        p for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        raise SystemExit(
            f"[ERROR] No video files found in {videos_dir} "
            f"(supported extensions: {', '.join(sorted(VIDEO_EXTENSIONS))})"
        )

    print(f"[INFO] Found {len(video_files)} video(s) in {videos_dir}")
    print(f"[INFO] Extracting one frame every {args.every_seconds}s → {frames_dir}\n")

    all_records: list[dict] = []
    for vp in video_files:
        out_subdir = frames_dir / vp.stem
        records = extract_video(
            video_path=vp,
            out_dir=out_subdir,
            every_seconds=args.every_seconds,
            max_frames=args.max_frames,
            overwrite=args.overwrite,
        )
        all_records.extend(records)

    if all_records:
        save_manifest(all_records, frames_dir)
    else:
        print("[INFO] No new frames extracted; manifest not updated.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
