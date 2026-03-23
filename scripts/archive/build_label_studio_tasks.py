import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a Label Studio task JSON from an image folder")
    parser.add_argument(
        "--images-dir",
        default="datasets/parks_detect/images/all",
        help="Directory containing images for annotation",
    )
    parser.add_argument(
        "--out",
        default="annotation/label_studio_tasks.json",
        help="Output JSON file for Label Studio task import",
    )
    parser.add_argument(
        "--mode",
        choices=("local-files", "file-url"),
        default="local-files",
        help="Task URL mode for Label Studio import",
    )
    parser.add_argument(
        "--local-files-base",
        default=".",
        help="Base directory configured in Label Studio Local Files storage",
    )
    return parser.parse_args()


def image_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def build_local_files_url(image_path: Path, base_dir: Path):
    relative_path = image_path.resolve().relative_to(base_dir.resolve())
    return "/data/local-files/?d=" + relative_path.as_posix()


def build_file_url(image_path: Path):
    return image_path.resolve().as_uri()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    output_path = Path(args.out)
    local_files_base = Path(args.local_files_base)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images = image_files(images_dir)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    tasks = []
    for image_path in images:
        if args.mode == "local-files":
            try:
                image_url = build_local_files_url(image_path, local_files_base)
            except ValueError as error:
                raise SystemExit(
                    f"Image '{image_path}' is not inside local-files base '{local_files_base}'. "
                    "Set --local-files-base to a parent directory shared with Label Studio."
                ) from error
        else:
            image_url = build_file_url(image_path)

        tasks.append(
            {
                "data": {
                    "image": image_url,
                    "image_name": image_path.name,
                }
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote {len(tasks)} Label Studio tasks to {output_path}")
    print(f"[INFO] Mode: {args.mode}")
    if args.mode == "local-files":
        print(f"[INFO] Local files base: {local_files_base.resolve()}")


if __name__ == "__main__":
    main()