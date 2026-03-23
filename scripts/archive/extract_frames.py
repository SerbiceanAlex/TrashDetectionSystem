from pathlib import Path

import cv2


FRAME_EVERY_SECONDS = 2.0


def resolve_raw_root():
    candidates = [Path("raw"), Path("datasets/raw")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def extract_frames_from_video(video_path: Path, output_dir: Path, every_seconds: float = 1.0):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[ERROR] Invalid FPS for video: {video_path}")
        cap.release()
        return

    frame_interval = max(int(round(fps * every_seconds)), 1)

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_name = f"frame_{saved_idx + 1:06d}.jpg"
            frame_path = output_dir / frame_name

            success = cv2.imwrite(str(frame_path), frame)
            if success:
                saved_idx += 1
            else:
                print(f"[WARNING] Failed to save frame {frame_idx} from {video_path.name}")

        frame_idx += 1

    cap.release()
    print(f"[INFO] Extracted {saved_idx} frames from {video_path.name} -> {output_dir}")


def main():
    raw_root = resolve_raw_root()
    videos_dir = raw_root / "videos"
    frames_dir = raw_root / "frames"

    frames_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        print(f"[WARNING] No .mp4 files found in {videos_dir}")
        return

    for video_path in video_files:
        output_subdir = frames_dir / video_path.stem
        extract_frames_from_video(video_path, output_subdir, FRAME_EVERY_SECONDS)


if __name__ == "__main__":
    main()