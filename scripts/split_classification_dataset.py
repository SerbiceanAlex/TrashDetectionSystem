import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_CLASSES = ["glass", "metal", "other", "paper", "plastic"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a labeled crop pool into train/val/test for material classification"
    )
    parser.add_argument(
        "--source-root",
        default="datasets/parks_cls_pool",
        help="Source root containing one folder per class with labeled crop images",
    )
    parser.add_argument(
        "--out-root",
        default="datasets/parks_cls",
        help="Output classification dataset root with train/val/test splits",
    )
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class folder names to split",
    )
    parser.add_argument(
        "--manifest",
        default="datasets/parks_cls_unsorted/crops_manifest.csv",
        help="Optional crop manifest from export_yolo_crops.py used for grouping by source image",
    )
    parser.add_argument(
        "--group-by",
        choices=("crop", "source-image"),
        default="source-image",
        help="Keep crops from the same source image in the same split to reduce leakage",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them")
    parser.add_argument("--clear", action="store_true", help="Clear existing train/val/test output folders before writing")
    return parser.parse_args()


def iter_images(directory: Path):
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        return {}

    mapping = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            crop_path = Path(row["crop_path"]).name
            source_image = Path(row["source_image"]).stem
            mapping[crop_path] = source_image
    return mapping


def ensure_output_dirs(out_root: Path, classes):
    for split in ("train", "val", "test"):
        for class_name in classes:
            (out_root / split / class_name).mkdir(parents=True, exist_ok=True)


def clear_output_dirs(out_root: Path):
    for split in ("train", "val", "test"):
        split_dir = out_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)


def assign_groups(group_keys, val_ratio: float, test_ratio: float, seed: int):
    shuffled = list(group_keys)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = int(round(total * test_ratio))
    val_count = int(round(total * val_ratio))
    if total >= 3:
        if test_ratio > 0 and test_count == 0:
            test_count = 1
        if val_ratio > 0 and val_count == 0:
            val_count = 1
        if test_count + val_count >= total:
            overflow = test_count + val_count - total + 1
            while overflow > 0 and test_count > 0:
                test_count -= 1
                overflow -= 1
            while overflow > 0 and val_count > 0:
                val_count -= 1
                overflow -= 1

    split_map = {}
    for index, group_key in enumerate(shuffled):
        if index < test_count:
            split_map[group_key] = "test"
        elif index < test_count + val_count:
            split_map[group_key] = "val"
        else:
            split_map[group_key] = "train"
    return split_map


def transfer_file(src: Path, dst: Path, copy_files: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main():
    args = parse_args()
    if args.val < 0 or args.test < 0 or args.val + args.test >= 1:
        raise ValueError("--val and --test must be >= 0 and sum to less than 1")

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    manifest_map = load_manifest(Path(args.manifest)) if args.group_by == "source-image" else {}

    grouped_items = defaultdict(list)
    missing_classes = []
    for class_name in args.classes:
        class_dir = source_root / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            continue

        for image_path in iter_images(class_dir):
            if args.group_by == "source-image":
                group_key = manifest_map.get(image_path.name, image_path.stem)
            else:
                group_key = image_path.stem
            grouped_items[(class_name, group_key)].append(image_path)

    if not grouped_items:
        raise SystemExit(f"No labeled crop images found under {source_root}")

    if args.clear:
        clear_output_dirs(out_root)
    ensure_output_dirs(out_root, args.classes)

    grouped_by_class = defaultdict(list)
    for class_name, group_key in grouped_items:
        grouped_by_class[class_name].append(group_key)

    stats = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}
    for class_name in args.classes:
        group_keys = grouped_by_class.get(class_name, [])
        if not group_keys:
            continue
        split_map = assign_groups(group_keys, args.val, args.test, args.seed)
        for group_key in group_keys:
            split = split_map[group_key]
            for src_image in grouped_items[(class_name, group_key)]:
                dst_image = out_root / split / class_name / src_image.name
                transfer_file(src_image, dst_image, args.copy)
                stats[split][class_name] += 1

    for split in ("train", "val", "test"):
        print(f"[INFO] Split: {split}")
        for class_name in args.classes:
            print(f" - {class_name}: {stats[split][class_name]}")

    if missing_classes:
        print(f"[WARN] Missing source class folders: {', '.join(missing_classes)}")


if __name__ == "__main__":
    main()