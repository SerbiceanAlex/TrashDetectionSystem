"""
Extrage cadre eșantionate dintr-un folder de videouri, pentru adnotare ulterioară.

Folosit pentru surse video open-source (ex. Stipra sau MIVIA-IWDD) după ce sunt
descărcate local. NU creează etichete YOLO — cadrele rezultate sunt menite să fie
revizuite și adnotate manual înainte de a fi folosite la antrenare.

Exemplu:
    .venv\\Scripts\\python.exe scripts\\data\\extract_video_frames.py ^
        --videos datasets\\raw\\stipra\\videos ^
        --out datasets\\raw\\stipra\\frames ^
        --sample-fps 1 ^
        --max-frames-per-video 30
"""

from __future__ import annotations

import sys
# Diacriticele românești se afișează corect indiferent de codarea consolei.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import csv
import hashlib
from pathlib import Path

import cv2


REPO = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrage cadre eșantionate din videouri locale")
    parser.add_argument("--videos", required=True, help="Fișier video sau folder de intrare")
    parser.add_argument("--out", required=True, help="Folderul de ieșire pentru cadre")
    parser.add_argument("--sample-fps", type=float, default=1.0, help="Câte cadre se salvează pe secundă")
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help="Număr maxim de cadre salvate per video. 0 = fără limită.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Caută recursiv în folderul de intrare.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="Calitatea JPEG a cadrelor salvate.",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Prefixul numelui fișierelor de cadre extrase.",
    )
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    """Absolutizează o cale relativă față de rădăcina proiectului."""
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    return path


def collect_videos(path: Path, recursive: bool) -> list[Path]:
    """Adună fișierele video dintr-un folder (sau un singur fișier), sortate."""
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    if not path.exists():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        item
        for item in path.glob(pattern)
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    )


def safe_video_id(video_path: Path) -> str:
    """Creează un identificator unic și sigur pentru nume de fișier (stem + hash)."""
    digest = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:8]
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in video_path.stem)
    return f"{stem}_{digest}"


def extract_video(
    video_path: Path,
    out_dir: Path,
    sample_fps: float,
    max_frames: int,
    jpeg_quality: int,
    prefix: str,
) -> list[dict]:
    """Extrage cadre dintr-un video la rata cerută; întoarce rândurile de manifest."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ATENȚIE] Nu pot deschide videoul: {video_path}")
        return []

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Pasul de eșantionare: din câte în câte cadre salvăm, ca să obținem sample_fps
    stride = 1
    if source_fps > 0 and sample_fps > 0:
        stride = max(1, int(round(source_fps / sample_fps)))

    rows: list[dict] = []
    video_id = safe_video_id(video_path)
    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            filename = f"{prefix}_{video_id}_f{frame_idx:06d}.jpg"
            out_path = out_dir / filename
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            timestamp_sec = frame_idx / source_fps if source_fps > 0 else 0.0
            rows.append(
                {
                    "frame_path": str(out_path.relative_to(REPO)),
                    "video_path": str(video_path.relative_to(REPO)) if video_path.is_relative_to(REPO) else str(video_path),
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "source_fps": round(source_fps, 3),
                    "total_frames": total_frames,
                }
            )
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return rows


def main() -> int:
    args = parse_args()
    videos_path = resolve_path(args.videos)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(videos_path, args.recursive)
    print(f"Videouri găsite: {len(videos)}")
    if not videos:
        return 1

    manifest_rows: list[dict] = []
    for idx, video_path in enumerate(videos, 1):
        rows = extract_video(
            video_path=video_path,
            out_dir=out_dir,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames_per_video,
            jpeg_quality=args.jpeg_quality,
            prefix=args.prefix,
        )
        manifest_rows.extend(rows)
        print(f"[{idx}/{len(videos)}] {video_path.name}: salvate {len(rows)} cadre")

    manifest_path = out_dir / "manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    print(f"Total cadre salvate: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
