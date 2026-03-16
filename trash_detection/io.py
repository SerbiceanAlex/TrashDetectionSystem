"""trash_detection/io.py – Path discovery utilities.

All functions accept pathlib.Path objects (or str, coerced internally) and
return lists of pathlib.Path for portability across operating systems.
"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".avi", ".mov", ".mkv"})


def find_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return sorted list of image paths found inside *directory*.

    Parameters
    ----------
    directory:
        Root directory to search.
    recursive:
        When True, descend into sub-directories.
    """
    root = Path(directory)
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_frames(frames_dir: str | Path) -> list[Path]:
    """Return all frame images nested under *frames_dir*.

    Expected layout::

        frames_dir/
            <video_id>/
                frame_000000.jpg
                frame_000001.jpg
                ...

    Returns a flat, sorted list of image paths across all video subdirectories.
    """
    return find_images(frames_dir, recursive=True)


def find_videos(directory: str | Path) -> list[Path]:
    """Return sorted list of video file paths found inside *directory*."""
    root = Path(directory)
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def list_video_ids(frames_dir: str | Path) -> list[str]:
    """Return sorted list of video IDs (sub-directory names) in *frames_dir*.

    Each immediate child directory of *frames_dir* is treated as a video ID.
    Empty directories or non-directory entries are ignored.
    """
    root = Path(frames_dir)
    return sorted(
        p.name for p in root.iterdir() if p.is_dir()
    )
