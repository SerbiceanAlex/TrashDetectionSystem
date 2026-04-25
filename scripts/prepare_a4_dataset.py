"""
prepare_a4_dataset.py
=====================
Prepara dataset-ul pentru Experiment A4 (detector extins):

  Source 1: datasets/parks_detect_full/  (deja YOLO, 1544 imagini)
             Baza experimentului A3-final — pastrare completa.

  Source 2: datasets/raw/taco/data/      (COCO format, 1500 imagini)
             60 categorii diverse de gunoi → remapate la clasa 0 (trash).

  Output: datasets/parks_detect_A4/
          Structura identica cu parks_detect_full (train/val/test split).
          dataset.yaml actualizat pentru antrenarea YOLOv8.

Split TACO: 70% train / 15% val / 15% test (seed fix pentru reproductibilitate).

Rulare:
    .venv/Scripts/python.exe scripts/prepare_a4_dataset.py
    (nu necesita GPU, dureaza ~30s)
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

MAX_DIM = 1280  # cap all images at 1280px max dimension before saving

REPO_ROOT = Path(__file__).resolve().parents[1]
PARKS_SRC = REPO_ROOT / "datasets" / "parks_detect_full"
TACO_IMG_DIR = REPO_ROOT / "datasets" / "raw" / "taco" / "data"
TACO_ANN    = REPO_ROOT / "datasets" / "raw" / "taco" / "data" / "annotations.json"
OUT_DIR     = REPO_ROOT / "datasets" / "parks_detect_A4"

SPLIT_SEED   = 42
TACO_TRAIN   = 0.70
TACO_VAL     = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Copy parks_detect_full as base
# ─────────────────────────────────────────────────────────────────────────────

def _save_resized(src: Path, dst: Path) -> bool:
    """Copy image to dst, resizing to MAX_DIM if larger. Returns True on success."""
    img = cv2.imread(str(src))
    if img is None:
        return False
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def copy_parks(out_dir: Path) -> dict[str, int]:
    counts = {}
    for split in ["train", "val", "test"]:
        src_imgs = PARKS_SRC / "images" / split
        src_lbls = PARKS_SRC / "labels" / split
        dst_imgs = out_dir / "images" / split
        dst_lbls = out_dir / "labels" / split
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        n = 0
        for img in src_imgs.glob("*"):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = src_lbls / img.with_suffix(".txt").name
            if not lbl.exists():
                continue
            dst_img = dst_imgs / (img.stem + ".jpg")
            if not _save_resized(img, dst_img):
                continue
            shutil.copy2(lbl, dst_lbls / lbl.name)
            n += 1
        counts[split] = n
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Convert TACO COCO → YOLO single-class
# ─────────────────────────────────────────────────────────────────────────────

def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert COCO [x, y, w, h] → YOLO [cx, cy, w, h] normalised."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, nw)),
        max(0.0, min(1.0, nh)),
    )


def convert_taco(out_dir: Path) -> dict[str, int]:
    if not TACO_ANN.exists():
        print(f"  WARN: TACO annotations.json nu exista la {TACO_ANN}")
        return {}

    with open(TACO_ANN, encoding="utf-8") as f:
        coco = json.load(f)

    # Build image_id → image info
    img_info: dict[int, dict] = {img["id"]: img for img in coco["images"]}

    # Build image_id → list of YOLO annotation lines (all class 0)
    img_to_anns: dict[int, list[str]] = {}
    skipped = 0
    for ann in coco["annotations"]:
        iid    = ann["image_id"]
        info   = img_info.get(iid)
        if info is None:
            continue
        w_img  = info["width"]
        h_img  = info["height"]
        bbox   = ann.get("bbox")
        if not bbox or w_img <= 0 or h_img <= 0:
            skipped += 1
            continue
        # Skip annotations with zero area
        if bbox[2] <= 0 or bbox[3] <= 0:
            skipped += 1
            continue
        cx, cy, nw, nh = coco_bbox_to_yolo(bbox, w_img, h_img)
        if nw < 0.001 or nh < 0.001:
            skipped += 1
            continue
        line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        img_to_anns.setdefault(iid, []).append(line)

    if skipped:
        print(f"  Sarit {skipped} adnotari cu bbox invalid.")

    # Only keep images that have at least one valid annotation
    valid_ids = [iid for iid, anns in img_to_anns.items() if anns]
    print(f"  {len(valid_ids)} imagini TACO cu adnotari valide (din {len(img_info)})")

    # Split: train / val / test
    random.seed(SPLIT_SEED)
    shuffled = valid_ids.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * TACO_TRAIN)
    n_val   = int(n * TACO_VAL)
    splits = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train: n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }

    counts: dict[str, int] = {}
    for split, ids in splits.items():
        dst_imgs = out_dir / "images" / split
        dst_lbls = out_dir / "labels" / split
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        n_copied = 0
        for iid in ids:
            info     = img_info[iid]
            # TACO stores relative filenames like "batch_1/000001.jpg"
            rel_name = info["file_name"]
            src_img  = TACO_IMG_DIR / rel_name
            if not src_img.exists():
                continue

            # Use unique filename to avoid collision with parks images
            safe_name = rel_name.replace("/", "_").replace("\\", "_")
            stem      = Path(safe_name).stem
            dst_img   = dst_imgs / f"taco_{stem}.jpg"
            if not _save_resized(src_img, dst_img):
                continue

            dst_lbl = dst_lbls / f"taco_{stem}.txt"
            dst_lbl.write_text("\n".join(img_to_anns[iid]) + "\n")
            n_copied += 1

        counts[split] = n_copied

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Write dataset.yaml
# ─────────────────────────────────────────────────────────────────────────────

def write_yaml(out_dir: Path):
    yaml_content = f"""# Experiment A4 — Extended trash detector dataset
# Parks (A3-final base) + TACO (diverse trash categories, remapped to single class)
# Generated by scripts/prepare_a4_dataset.py

path: {out_dir.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: trash
"""
    (out_dir / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"  dataset.yaml scris in {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if OUT_DIR.exists():
        print(f"Output dir exists, curatam: {OUT_DIR}")
        # Defensive cleanup — skip files that are locked
        for item in OUT_DIR.rglob("*"):
            if item.is_file():
                try:
                    item.unlink()
                except (PermissionError, OSError):
                    pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  Prepare A4 Dataset")
    print("=" * 55)

    print("\n[1/3] Copiez parks_detect_full...")
    parks_counts = copy_parks(OUT_DIR)
    for split, n in parks_counts.items():
        print(f"  parks {split:6s}: {n:4d} imagini")

    print("\n[2/3] Convertesc TACO COCO -> YOLO single-class...")
    taco_counts = convert_taco(OUT_DIR)
    for split, n in taco_counts.items():
        print(f"  taco  {split:6s}: {n:4d} imagini")

    print("\n[3/3] Scriu dataset.yaml...")
    write_yaml(OUT_DIR)

    print("\n" + "=" * 55)
    print("  SUMAR FINAL — dataset A4")
    print("=" * 55)
    total = 0
    for split in ["train", "val", "test"]:
        n_imgs = len(list((OUT_DIR / "images" / split).glob("*")))
        n_lbls = len(list((OUT_DIR / "labels" / split).glob("*.txt")))
        p_c    = parks_counts.get(split, 0)
        t_c    = taco_counts.get(split, 0)
        total += n_imgs
        print(f"  {split:6s}: {n_imgs:4d} imgs  ({p_c} parks + {t_c} taco)  labels={n_lbls}")
    print(f"  TOTAL : {total:4d} imagini")
    print(f"\n  Dataset gata: {OUT_DIR}")
    print("\nUrmatorul pas — antrenare A4:")
    print(f"  yolo detect train \\")
    print(f"    model=runs/detect/parks-trash-A3-final/weights/best.pt \\")
    print(f"    data={OUT_DIR}/dataset.yaml \\")
    print(f"    epochs=50 imgsz=640 project=runs/detect name=parks-trash-A4 \\")
    print(f"    freeze=10 batch=16 patience=15")


if __name__ == "__main__":
    main()
