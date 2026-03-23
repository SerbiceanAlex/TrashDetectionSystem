import argparse
from pathlib import Path


DEFAULT_CLASSES = ["glass", "metal", "other", "paper", "plastic"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Report basic statistics for an image classification dataset")
    parser.add_argument("--data", default="datasets/parks_cls", help="Classification dataset root")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Expected class folder names",
    )
    return parser.parse_args()


def count_images(directory: Path):
    if not directory.exists():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    args = parse_args()
    data_root = Path(args.data)
    if not data_root.exists():
        raise FileNotFoundError(f"Classification dataset root not found: {data_root}")

    overall_total = 0
    for split in ("train", "val", "test"):
        split_dir = data_root / split
        print(f"[INFO] Split: {split}")
        split_total = 0
        missing_classes = []
        for class_name in args.classes:
            class_dir = split_dir / class_name
            image_count = count_images(class_dir)
            if not class_dir.exists():
                missing_classes.append(class_name)
            print(f" - {class_name}: {image_count}")
            split_total += image_count

        print(f" - total: {split_total}")
        if missing_classes:
            print(f" - missing class folders: {', '.join(missing_classes)}")
        overall_total += split_total

    print(f"[INFO] Overall images: {overall_total}")


if __name__ == "__main__":
    main()