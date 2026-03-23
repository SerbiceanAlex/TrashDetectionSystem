import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split YOLO detection data from images/all and labels/all into train/val/test"
    )
    parser.add_argument("--dataset-root", default="datasets/parks_detect", help="YOLO dataset root")
    parser.add_argument("--source-images", default=None, help="Override source images directory")
    parser.add_argument("--source-labels", default=None, help="Override source labels directory")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--group-by",
        choices=("prefix", "image"),
        default="prefix",
        help="Keep related frames together by prefix or split per-image",
    )
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using hardlinks")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing train/val/test files before generating a new split",
    )
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help="Treat images without a label file as valid negatives and create empty labels",
    )
    return parser.parse_args()


def image_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def group_key_for(stem: str, mode: str):
    if mode == "image":
        return stem

    marker = "_frame_"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


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


def ensure_split_dirs(dataset_root: Path):
    for split in ("train", "val", "test"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_split_dirs(dataset_root: Path):
    for split in ("train", "val", "test"):
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if image_dir.exists():
            shutil.rmtree(image_dir)
        if label_dir.exists():
            shutil.rmtree(label_dir)


def link_or_copy(src: Path, dst: Path, copy_files: bool):
    if dst.exists():
        dst.unlink()

    if copy_files:
        shutil.copy2(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    args = parse_args()
    if args.val < 0 or args.test < 0 or args.val + args.test >= 1:
        raise ValueError("--val and --test must be >= 0 and sum to less than 1")

    dataset_root = Path(args.dataset_root)
    source_images = Path(args.source_images) if args.source_images else dataset_root / "images" / "all"
    source_labels = Path(args.source_labels) if args.source_labels else dataset_root / "labels" / "all"

    if not source_images.exists():
        raise FileNotFoundError(f"Source images directory not found: {source_images}")

    if not source_labels.exists() and not args.allow_empty_labels:
        raise FileNotFoundError(
            f"Source labels directory not found: {source_labels}. Use --allow-empty-labels only if unlabeled negatives are intentional."
        )

    images = image_files(source_images)
    if not images:
        raise SystemExit(f"No source images found in {source_images}")

    grouped_images = defaultdict(list)
    skipped_missing_labels = []

    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"
        if not label_path.exists() and not args.allow_empty_labels:
            skipped_missing_labels.append(image_path)
            continue

        group_key = group_key_for(image_path.stem, args.group_by)
        grouped_images[group_key].append((image_path, label_path))

    if not grouped_images:
        raise SystemExit("No eligible images found for splitting. Add label files or use --allow-empty-labels.")

    if args.clear:
        clear_split_dirs(dataset_root)
    ensure_split_dirs(dataset_root)

    split_map = assign_groups(grouped_images.keys(), args.val, args.test, args.seed)
    stats = {"train": 0, "val": 0, "test": 0}

    for group_key, items in grouped_images.items():
        split = split_map[group_key]
        for image_path, label_path in items:
            dst_image = dataset_root / "images" / split / image_path.name
            dst_label = dataset_root / "labels" / split / f"{image_path.stem}.txt"

            link_or_copy(image_path, dst_image, args.copy)
            if label_path.exists():
                link_or_copy(label_path, dst_label, args.copy)
            else:
                dst_label.write_text("", encoding="utf-8")
            stats[split] += 1

    print(f"[INFO] Source images: {source_images}")
    print(f"[INFO] Source labels: {source_labels}")
    print(f"[INFO] Grouping mode: {args.group_by}")
    print(f"[INFO] Groups assigned: {len(grouped_images)}")
    print(f"[INFO] Images written -> train: {stats['train']}, val: {stats['val']}, test: {stats['test']}")
    if skipped_missing_labels:
        print(f"[WARN] Skipped {len(skipped_missing_labels)} images without labels")
        for image_path in skipped_missing_labels[:10]:
            print(f" - {image_path}")
        if len(skipped_missing_labels) > 10:
            print(" - ...")


if __name__ == "__main__":
    main()