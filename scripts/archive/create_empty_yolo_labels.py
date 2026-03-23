import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create empty YOLO label files for images that intentionally contain no objects"
    )
    parser.add_argument("--images-dir", required=True, help="Directory containing source images")
    parser.add_argument("--labels-dir", required=True, help="Directory where YOLO txt labels should exist")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Only create labels that do not already exist",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    labels_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists() and args.only_missing:
            skipped += 1
            continue

        label_path.write_text("", encoding="utf-8")
        created += 1

    print(f"[DONE] Created {created} empty label files in {labels_dir}")
    if skipped:
        print(f"[INFO] Skipped {skipped} existing label files")


if __name__ == "__main__":
    main()