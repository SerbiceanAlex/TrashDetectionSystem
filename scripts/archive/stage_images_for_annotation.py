import argparse
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Stage images into datasets/parks_detect/images/all for annotation")
    parser.add_argument("--source", required=True, help="Source directory containing images to annotate")
    parser.add_argument(
        "--dest",
        default="datasets/parks_detect/images/all",
        help="Destination annotation pool directory",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using hardlinks")
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Search recursively under source and stage all matching images into one flat folder",
    )
    return parser.parse_args()


def iter_images(source_dir: Path, flatten: bool):
    if flatten:
        return sorted(path for path in source_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def link_or_copy(src: Path, dst: Path, copy_files: bool):
    if dst.exists():
        return False
    if copy_files:
        shutil.copy2(src, dst)
        return True
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return True


def unique_destination(dest_dir: Path, image_path: Path):
    candidate = dest_dir / image_path.name
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = dest_dir / f"{image_path.stem}_{suffix}{image_path.suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1


def main():
    args = parse_args()
    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    images = iter_images(source_dir, args.flatten)
    if not images:
        raise SystemExit(f"No images found in {source_dir}")

    for image_path in images:
        dst_path = unique_destination(dest_dir, image_path)
        if link_or_copy(image_path, dst_path, args.copy):
            staged += 1

    print(f"[DONE] Staged {staged} images into {dest_dir}")


if __name__ == "__main__":
    main()