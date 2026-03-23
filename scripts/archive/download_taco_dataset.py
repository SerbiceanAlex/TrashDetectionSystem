"""Download and convert the TACO dataset to YOLO single-class (trash) format.

TACO (Trash Annotations in Context) provides COCO-format annotations of
litter in the wild. This script:
  1. Clones the TACO repo (annotations + download script)
  2. Downloads all images via the official TACO downloader
  3. Converts COCO annotations → YOLO format (all categories → class 0 'trash')
  4. Copies images + labels into the project's dataset structure

Usage:
    python scripts/download_taco_dataset.py
    python scripts/download_taco_dataset.py --out-dir datasets/taco_yolo --min-area 400
    python scripts/download_taco_dataset.py --skip-download   # if images already downloaded
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Download TACO and convert to YOLO single-class format")
    p.add_argument("--taco-dir", default="datasets/raw/taco", help="Where to clone/store TACO repo")
    p.add_argument("--out-dir", default="datasets/taco_yolo", help="Output directory for YOLO images+labels")
    p.add_argument("--skip-download", action="store_true", help="Skip image download (use if already downloaded)")
    p.add_argument("--min-area", type=float, default=200, help="Min bbox area in pixels to keep (filters tiny fragments)")
    p.add_argument("--unofficial", action="store_true", help="Also include unofficial (unreviewed) annotations")
    return p.parse_args()


def clone_taco(taco_dir: Path):
    """Clone or update the TACO repository."""
    taco_dir = taco_dir.resolve()
    if (taco_dir / "data").exists():
        print(f"[INFO] TACO repo already exists at {taco_dir}")
        return

    taco_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] Cloning TACO repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/pedropro/TACO.git", str(taco_dir)],
        check=True,
    )
    print("[INFO] TACO repository cloned.")


def download_taco_images(taco_dir: Path, unofficial: bool = False):
    """Run the TACO download script to fetch images."""
    taco_dir = taco_dir.resolve()
    download_script = taco_dir / "download.py"
    if not download_script.exists():
        raise FileNotFoundError(f"TACO download.py not found at {download_script}")

    print("[INFO] Downloading TACO images (this may take a while)...")
    cmd = [sys.executable, str(download_script)]
    if unofficial:
        cmd += ["--dataset_path", str(taco_dir / "data" / "annotations_unofficial.json")]
    subprocess.run(cmd, cwd=str(taco_dir), check=True)
    print("[INFO] TACO images downloaded.")


def load_coco_annotations(taco_dir: Path, unofficial: bool = False):
    """Load COCO-format annotations from TACO."""
    ann_file = taco_dir / "data" / "annotations.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations not found: {ann_file}")

    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if unofficial:
        unofficial_file = taco_dir / "data" / "annotations_unofficial.json"
        if unofficial_file.exists():
            with open(unofficial_file, "r", encoding="utf-8") as f:
                unofficial_data = json.load(f)
            # Merge: offset image/annotation IDs to avoid conflicts
            max_img_id = max(img["id"] for img in data["images"]) + 1
            max_ann_id = max(ann["id"] for ann in data["annotations"]) + 1
            id_map = {}
            for img in unofficial_data["images"]:
                old_id = img["id"]
                img["id"] = old_id + max_img_id
                id_map[old_id] = img["id"]
                data["images"].append(img)
            for ann in unofficial_data["annotations"]:
                ann["id"] = ann["id"] + max_ann_id
                ann["image_id"] = id_map.get(ann["image_id"], ann["image_id"])
                data["annotations"].append(ann)
            print(f"[INFO] Merged {len(unofficial_data['images'])} unofficial images.")

    return data


def coco_to_yolo_single_class(coco_data: dict, min_area: float):
    """Convert COCO annotations to YOLO single-class format.

    Returns dict: {image_filename: [list of 'cls cx cy w h' strings]}
    """
    # Build image lookup: id → {file_name, width, height}
    images = {}
    for img in coco_data["images"]:
        images[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # Group annotations by image
    labels = {}
    skipped_tiny = 0
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in images:
            continue

        img_info = images[img_id]
        bbox = ann.get("bbox")  # COCO format: [x, y, width, height] in pixels
        if not bbox or len(bbox) != 4:
            continue

        x, y, bw, bh = bbox
        area = bw * bh
        if area < min_area:
            skipped_tiny += 1
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]
        if img_w <= 0 or img_h <= 0:
            continue

        # Convert to YOLO normalized xywh
        cx = (x + bw / 2) / img_w
        cy = (y + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))

        fname = img_info["file_name"]
        if fname not in labels:
            labels[fname] = []
        labels[fname].append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    if skipped_tiny:
        print(f"[INFO] Skipped {skipped_tiny} annotations smaller than {min_area}px² area.")

    # Also include images with no annotations (negatives)
    all_files = {img["file_name"] for img in images.values()}
    for fname in all_files:
        if fname not in labels:
            labels[fname] = []

    return labels


def find_taco_image(taco_dir: Path, file_name: str):
    """Locate a TACO image file, checking common subdirectories."""
    candidates = [
        taco_dir / "data" / file_name,
        taco_dir / file_name,
        taco_dir / "data" / Path(file_name).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def export_yolo_dataset(taco_dir: Path, out_dir: Path, labels: dict):
    """Copy images and write YOLO label files."""
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    total_boxes = 0

    for file_name, yolo_lines in labels.items():
        src_img = find_taco_image(taco_dir, file_name)
        if src_img is None:
            missing += 1
            continue

        # Normalize extension and name
        stem = Path(file_name).stem
        suffix = src_img.suffix.lower()
        # Use a clean name (replace subdir separators)
        clean_name = file_name.replace("/", "_").replace("\\", "_")
        clean_stem = Path(clean_name).stem

        dst_img = images_dir / f"{clean_stem}{suffix}"
        dst_lbl = labels_dir / f"{clean_stem}.txt"

        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))
            if yolo_lines:
                f.write("\n")

        copied += 1
        total_boxes += len(yolo_lines)

    return copied, missing, total_boxes


def main():
    args = parse_args()
    taco_dir = Path(args.taco_dir)
    out_dir = Path(args.out_dir)

    # Step 1: Clone
    clone_taco(taco_dir)

    # Step 2: Download images
    if not args.skip_download:
        download_taco_images(taco_dir, args.unofficial)
    else:
        print("[INFO] Skipping image download (--skip-download).")

    # Step 3: Load and convert annotations
    print("[INFO] Loading COCO annotations...")
    coco_data = load_coco_annotations(taco_dir, args.unofficial)
    n_images = len(coco_data["images"])
    n_anns = len(coco_data["annotations"])
    n_cats = len(coco_data.get("categories", []))
    print(f"[INFO] TACO dataset: {n_images} images, {n_anns} annotations, {n_cats} categories")
    print(f"[INFO] All {n_cats} categories will be mapped → class 0 (trash)")

    labels = coco_to_yolo_single_class(coco_data, args.min_area)
    labeled_count = sum(1 for v in labels.values() if v)
    print(f"[INFO] Images with at least 1 box: {labeled_count}/{len(labels)}")

    # Step 4: Export
    print(f"[INFO] Exporting YOLO dataset to {out_dir}...")
    copied, missing, total_boxes = export_yolo_dataset(taco_dir, out_dir, labels)

    print()
    print(f"[DONE] TACO → YOLO conversion complete!")
    print(f"  Images exported: {copied}")
    print(f"  Images not found: {missing}")
    print(f"  Total bounding boxes: {total_boxes}")
    print(f"  Output: {out_dir}")
    print()
    print("[NEXT] Merge into your park dataset:")
    print(f"  python scripts/merge_yolo_datasets.py --sources {out_dir} datasets/parks_detect --dest datasets/parks_detect_merged")


if __name__ == "__main__":
    main()
