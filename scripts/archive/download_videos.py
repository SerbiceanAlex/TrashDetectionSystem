"""Download videos from YouTube and extract frames for annotation.

Uses yt-dlp (must be installed: pip install yt-dlp) to download videos,
then extracts frames at a configurable interval.

Usage:
    python scripts/download_videos.py --urls "URL1" "URL2" --every 2.0
    python scripts/download_videos.py --url-file video_urls.txt --every 1.5 --stage

Suggested search queries for park trash videos:
    - "park cleanup timelapse"
    - "litter picking volunteer"
    - "trash in park drone"
    - "beach cleanup gopro"
    - "street garbage walk"
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Download videos and extract frames for annotation")
    p.add_argument("--urls", nargs="+", default=[], help="Video URLs to download")
    p.add_argument("--url-file", default=None, help="Text file with one URL per line")
    p.add_argument("--videos-dir", default="datasets/raw/videos", help="Directory to save downloaded videos")
    p.add_argument("--frames-dir", default="datasets/raw/frames", help="Directory to save extracted frames")
    p.add_argument("--every", type=float, default=2.0, help="Extract one frame every N seconds")
    p.add_argument("--max-duration", default="600", help="Max video duration to download (seconds, or yt-dlp format like '10:00')")
    p.add_argument("--resolution", default="720", help="Max video resolution (e.g., 720, 1080)")
    p.add_argument("--stage", action="store_true", help="Also stage frames into datasets/parks_detect/images/all")
    p.add_argument("--skip-download", action="store_true", help="Skip download, only extract frames from existing videos")
    p.add_argument("--prefix", default=None, help="Custom prefix for video filenames (default: auto from index)")
    return p.parse_args()


def sanitize_filename(name: str) -> str:
    """Remove problematic characters from a filename."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name[:80]


def download_video(url: str, videos_dir: Path, index: int, prefix: str | None, resolution: str, max_duration: str) -> Path | None:
    """Download a single video using yt-dlp."""
    videos_dir.mkdir(parents=True, exist_ok=True)

    tag = prefix or f"video_{index:03d}"

    # Use yt-dlp to download
    output_template = str(videos_dir / f"{tag}_%(title).40s.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "-f", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]",
        "--merge-output-format", "mp4",
        "--match-filter", f"duration<={max_duration}",
        "-o", output_template,
        "--no-overwrites",
        url,
    ]

    print(f"\n[INFO] Downloading: {url}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"[ERROR] yt-dlp failed for {url}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return None
    except FileNotFoundError:
        print("[ERROR] yt-dlp not found. Install it: pip install yt-dlp")
        return None
    except subprocess.TimeoutExpired:
        print(f"[ERROR] Download timed out for {url}")
        return None

    # Find the downloaded file
    matches = sorted(videos_dir.glob(f"{tag}_*.*"))
    mp4_matches = [m for m in matches if m.suffix.lower() == ".mp4"]
    if mp4_matches:
        print(f"[INFO] Saved: {mp4_matches[-1].name}")
        return mp4_matches[-1]

    if matches:
        print(f"[INFO] Saved: {matches[-1].name}")
        return matches[-1]

    print(f"[WARNING] Could not locate downloaded file for {url}")
    return None


def extract_frames(video_path: Path, frames_dir: Path, every_seconds: float) -> int:
    """Extract frames from a video at the given interval."""
    output_dir = frames_dir / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[ERROR] Invalid FPS for {video_path}")
        cap.release()
        return 0

    frame_interval = max(int(round(fps * every_seconds)), 1)
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"{video_path.stem}_frame_{saved + 1:06d}.jpg"
            if cv2.imwrite(str(out_path), frame):
                saved += 1
        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {saved} frames from {video_path.name} → {output_dir}")
    return saved


def stage_frames(frames_dir: Path, dest_dir: Path):
    """Copy/link extracted frames into the annotation pool."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    for img in sorted(frames_dir.rglob("*")):
        if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
            dst = dest_dir / img.name
            if not dst.exists():
                try:
                    import os
                    os.link(str(img), str(dst))
                except OSError:
                    import shutil
                    shutil.copy2(img, dst)
                staged += 1
    print(f"[INFO] Staged {staged} new frames into {dest_dir}")


def main():
    args = parse_args()

    videos_dir = Path(args.videos_dir)
    frames_dir = Path(args.frames_dir)

    # Collect URLs
    urls = list(args.urls)
    if args.url_file:
        url_file = Path(args.url_file)
        if url_file.exists():
            with open(url_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urls.append(line)

    # Download phase
    downloaded = []
    if not args.skip_download:
        if not urls:
            print("[INFO] No URLs provided. Use --urls or --url-file to specify videos to download.")
            print("[INFO] Proceeding to extract frames from existing videos only.")
        else:
            for i, url in enumerate(urls, 1):
                video_path = download_video(url, videos_dir, i, args.prefix, args.resolution, args.max_duration)
                if video_path:
                    downloaded.append(video_path)
            print(f"\n[INFO] Downloaded {len(downloaded)}/{len(urls)} videos.")

    # Extract frames from ALL videos in the directory
    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        print(f"[WARNING] No .mp4 files found in {videos_dir}")
        return

    total_frames = 0
    for video_path in video_files:
        check_dir = frames_dir / video_path.stem
        if check_dir.exists() and any(check_dir.iterdir()):
            existing = sum(1 for f in check_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            print(f"[INFO] Frames already exist for {video_path.name} ({existing} frames), skipping extraction.")
            total_frames += existing
            continue
        total_frames += extract_frames(video_path, frames_dir, args.every)

    print(f"\n[INFO] Total frames available: {total_frames}")

    # Stage into annotation pool
    if args.stage:
        stage_frames(frames_dir, Path("datasets/parks_detect/images/all"))

    print("\n[DONE] Video processing complete.")
    print("[NEXT] Steps:")
    print("  1. Review frames and delete duplicates/bad quality")
    print("  2. Auto-label: python scripts/auto_label_coco_to_trash.py")
    print("  3. Review labels in Label Studio / CVAT")
    print("  4. Split: python scripts/split_yolo_detection_dataset.py --clear")


if __name__ == "__main__":
    main()
