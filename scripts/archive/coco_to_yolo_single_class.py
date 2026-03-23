import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to a single-class YOLO trash dataset")
    parser.add_argument("--coco", required=True, help="Path to COCO annotations JSON")
    parser.add_argument("--images-root", required=True, help="Folder containing source images")
    parser.add_argument("--out", required=True, help="Output YOLO dataset root")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of using hardlinks")
    parser.add_argument(
        "--group-by-prefix",
        action="store_true",
        help="Keep related frames from the same source sequence in the same split",
    )
    parser.add_argument(
        "--include-categories",
        nargs="+",
        default=None,
        help="Optional COCO category names to keep. When omitted, all annotated categories are collapsed into trash.",
    )
    return parser.parse_args()


def yolo_line_from_bbox(bbox, img_w, img_h):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_w
    y_center = (y_min + height / 2) / img_h
    width_norm = width / img_w
    height_norm = height / img_h
    return f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"


def group_key_for(file_name: str, use_prefix: bool):
    stem = Path(file_name).stem
    if not use_prefix:
        return stem
    marker = "_frame_"
    if marker in stem:
        return stem.split(marker, 1)[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def find_image_file(images_root: Path, file_name: str):
    direct_path = images_root / file_name
    if direct_path.exists():
        return direct_path

    base_name = Path(file_name).name
    matches = list(images_root.rglob(base_name))
    if matches:
        return matches[0]
    return None


def assign_splits(group_keys, val_ratio: float, test_ratio: float, seed: int):
    group_keys = list(group_keys)
    random.Random(seed).shuffle(group_keys)

    total = len(group_keys)
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
    for index, group_key in enumerate(group_keys):
        if index < test_count:
            split_map[group_key] = "test"
        elif index < test_count + val_count:
            split_map[group_key] = "val"
        else:
            split_map[group_key] = "train"
    return split_map


def link_or_copy(src: Path, dst: Path, copy_files: bool):
    if copy_files:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yaml(out_root: Path):
    yaml_path = out_root / f"{out_root.name}.yaml"
    lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        "  0: trash",
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.val < 0 or args.test < 0 or args.val + args.test >= 1:
        raise ValueError("--val and --test must be >= 0 and sum to less than 1")

    coco_path = Path(args.coco)
    images_root = Path(args.images_root)
    out_root = Path(args.out)

    with coco_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    categories = {category["id"]: category["name"] for category in coco.get("categories", [])}
    images = {image["id"]: image for image in coco.get("images", [])}
    annotations_by_image = defaultdict(list)
    missing_images = []
    ignored_categories = set()
    included_categories = {name.strip() for name in args.include_categories} if args.include_categories else None

    for annotation in coco.get("annotations", []):
        if annotation.get("iscrowd", 0) == 1:
            continue
        category_name = categories.get(annotation["category_id"])
        if included_categories is not None and category_name not in included_categories:
            ignored_categories.add(category_name)
            continue
        annotations_by_image[annotation["image_id"]].append(annotation)

    groups = defaultdict(list)
    for image_id, image_info in images.items():
        src_image = find_image_file(images_root, image_info["file_name"])
        if src_image is None:
            missing_images.append(image_info["file_name"])
            continue
        group_key = group_key_for(image_info["file_name"], args.group_by_prefix)
        groups[group_key].append((image_id, image_info, src_image))

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_map = assign_splits(groups.keys(), args.val, args.test, args.seed)
    written_images = 0
    written_labels = 0

    for group_key, items in groups.items():
        split = split_map[group_key]
        for image_id, image_info, src_image in items:
            dst_image = out_root / "images" / split / src_image.name
            dst_label = out_root / "labels" / split / f"{src_image.stem}.txt"

            width = int(image_info["width"])
            height = int(image_info["height"])
            yolo_lines = []
            for annotation in annotations_by_image.get(image_id, []):
                bbox = annotation.get("bbox")
                if not bbox or bbox[2] <= 1 or bbox[3] <= 1:
                    continue
                yolo_lines.append(yolo_line_from_bbox(bbox, width, height))

            link_or_copy(src_image, dst_image, args.copy)
            dst_label.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            written_images += 1
            written_labels += 1

    write_yaml(out_root)

    print(f"[DONE] Wrote {written_images} images and {written_labels} labels into {out_root}")
    if included_categories is not None:
        print(f"[INFO] Included COCO categories collapsed into trash: {sorted(included_categories)}")
    else:
        print("[INFO] Collapsed all annotated COCO categories into single class: trash")
    if ignored_categories:
        print(f"[WARN] Ignored categories excluded by --include-categories: {sorted(ignored_categories)}")
    if missing_images:
        print(f"[WARN] Missing {len(missing_images)} images referenced by COCO annotations")
        for image_name in missing_images[:10]:
            print(f" - {image_name}")
        if len(missing_images) > 10:
            print(" - ...")


if __name__ == "__main__":
    main()