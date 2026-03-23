import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Report annotation progress for images/all and labels/all")
    parser.add_argument("--images-dir", default="datasets/parks_detect/images/all", help="Directory with source images")
    parser.add_argument("--labels-dir", default="datasets/parks_detect/labels/all", help="Directory with YOLO labels")
    return parser.parse_args()


def image_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    images = image_files(images_dir)
    total_images = len(images)
    labeled_images = 0
    empty_labels = 0
    missing_labels = 0

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels += 1
            continue

        contents = label_path.read_text(encoding="utf-8").strip()
        if contents:
            labeled_images += 1
        else:
            empty_labels += 1

    done_images = labeled_images + empty_labels
    progress_pct = (done_images / total_images * 100.0) if total_images else 0.0

    print(f"[INFO] Images directory: {images_dir}")
    print(f"[INFO] Labels directory: {labels_dir}")
    print(f"[INFO] Total images: {total_images}")
    print(f"[INFO] Labeled images: {labeled_images}")
    print(f"[INFO] Empty-label images: {empty_labels}")
    print(f"[INFO] Missing labels: {missing_labels}")
    print(f"[INFO] Annotation progress: {progress_pct:.1f}%")

    if missing_labels:
        print("[WARN] Images still missing a label file:")
        for image_path in images[: min(total_images, 10)]:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                print(f" - {image_path.name}")


if __name__ == "__main__":
    main()