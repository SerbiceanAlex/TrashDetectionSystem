import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_line_from_bbox(bbox, img_w, img_h, class_id=0):
    # COCO bbox = [x_min, y_min, width, height] in pixels
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def find_image_file(images_root: Path, file_name: str):
    # COCO file_name may be "xxx.jpg" or include subfolders; try direct, then basename search.
    p = images_root / file_name
    if p.exists():
        return p

    # try by basename anywhere under images_root (slower but robust)
    base = Path(file_name).name
    for ext in ("", ".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = images_root / base
        if ext and candidate.suffix.lower() != ext:
            candidate = candidate.with_suffix(ext)
        if candidate.exists():
            return candidate

    matches = list(images_root.rglob(base))
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="Path to COCO annotations.json")
    ap.add_argument("--images-root", required=True, help="Folder where images are stored")
    ap.add_argument("--out", required=True, help="Output YOLO dataset root (will create images/labels split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--copy", action="store_true", help="Copy images instead of hardlink (safer on OneDrive)")
    args = ap.parse_args()

    coco_path = Path(args.coco)
    images_root = Path(args.images_root)
    out_root = Path(args.out)

    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build image dict and annotation index
    images = {img["id"]: img for img in coco["images"]}
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        ann_by_img[ann["image_id"]].append(ann)

    img_ids = list(images.keys())
    random.Random(args.seed).shuffle(img_ids)

    n = len(img_ids)
    n_test = int(n * args.test)
    n_val = int(n * args.val)

    splits = {
        "test": img_ids[:n_test],
        "val": img_ids[n_test : n_test + n_val],
        "train": img_ids[n_test + n_val :],
    }

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    missing = 0
    written = 0

    for split, ids in splits.items():
        for img_id in ids:
            img = images[img_id]
            file_name = img["file_name"]
            w = int(img["width"])
            h = int(img["height"])

            src_img = find_image_file(images_root, file_name)
            if src_img is None or not src_img.exists():
                missing += 1
                continue

            dst_img = out_root / "images" / split / src_img.name
            dst_lbl = out_root / "labels" / split / (dst_img.stem + ".txt")

            # write labels
            lines = []
            for ann in ann_by_img.get(img_id, []):
                bbox = ann.get("bbox")
                if not bbox:
                    continue
                # skip invalid
                if bbox[2] <= 1 or bbox[3] <= 1:
                    continue
                lines.append(yolo_line_from_bbox(bbox, w, h, class_id=0))

            dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            # copy/link image
            if dst_img.exists():
                continue

            if args.copy:
                shutil.copy2(src_img, dst_img)
            else:
                try:
                    os.link(src_img, dst_img)  # hardlink, fast + no extra disk
                except OSError:
                    shutil.copy2(src_img, dst_img)

            written += 1

    print(f"[DONE] Wrote {written} images with labels into {out_root}")
    if missing:
        print(f"[WARN] Missing {missing} images referenced in COCO json (not found under {images_root})")


if __name__ == "__main__":
    main()