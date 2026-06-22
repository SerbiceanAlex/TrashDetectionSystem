"""
Filtrează un dataset YOLO multi-clasă într-unul cu o singură clasă (trash).

Util pentru dataseturile Roboflow Universe, unde etichetele pot include clase
ajutătoare (ex. human/person). Scriptul păstrează doar clasele de tip deșeu, le
mapează la clasa 0 și scrie fișiere de etichetă goale pentru imaginile devenite
exemple negative.

Exemple:
    .venv\\Scripts\\python.exe scripts\\data\\filter_yolo_classes.py ^
        --src datasets\\raw\\roboflow\\litter_detection ^
        --out datasets\\annotations\\roboflow_litter_detection_trash_only ^
        --keep waste,litter,trash,garbage ^
        --overwrite
"""

from __future__ import annotations

import sys
# Diacriticele românești se afișează corect indiferent de codarea consolei.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import shutil
from pathlib import Path

import cv2
import yaml


REPO = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filtrează clasele YOLO într-o singură clasă (trash)")
    parser.add_argument("--src", required=True, help="Folderul datasetului YOLO sursă")
    parser.add_argument("--out", required=True, help="Folderul datasetului YOLO de ieșire")
    parser.add_argument(
        "--keep",
        default="trash,waste,litter,garbage",
        help="Cuvinte-cheie de clase de păstrat, separate prin virgulă (insensibil la majuscule).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Șterge mai întâi ieșirea existentă")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    """Absolutizează o cale relativă față de rădăcina proiectului."""
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    return path


def load_names(src_dir: Path) -> dict[int, str]:
    """Citește numele claselor din data.yaml / dataset.yaml."""
    yaml_path = None
    for candidate in ["data.yaml", "dataset.yaml"]:
        path = src_dir / candidate
        if path.exists():
            yaml_path = path
            break
    if yaml_path is None:
        raise SystemExit(f"Nu am găsit data.yaml sau dataset.yaml în {src_dir}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    raise SystemExit(f"Format de nume neacceptat în {yaml_path}")


def discover_split_dirs(src_dir: Path) -> dict[str, tuple[Path, Path]]:
    """Detectează folderele de split (train/val/test) în diversele layout-uri YOLO."""
    split_dirs: dict[str, tuple[Path, Path]] = {}

    # Roboflow exportă de obicei train/images + train/labels.
    for alias, split in SPLIT_ALIASES.items():
        img_dir = src_dir / alias / "images"
        lbl_dir = src_dir / alias / "labels"
        if img_dir.exists():
            split_dirs[split] = (img_dir, lbl_dir)

    # Layout-ul Ultralytics poate fi și images/train + labels/train.
    for split in ["train", "val", "test"]:
        img_dir = src_dir / "images" / split
        lbl_dir = src_dir / "labels" / split
        if img_dir.exists():
            split_dirs[split] = (img_dir, lbl_dir)

    # Layout fără split: images + labels.
    if not split_dirs and (src_dir / "images").exists():
        split_dirs["train"] = (src_dir / "images", src_dir / "labels")

    if not split_dirs:
        raise SystemExit(f"Nu am găsit foldere de imagini YOLO în {src_dir}")
    return split_dirs


def list_images(img_dir: Path) -> list[Path]:
    """Întoarce imaginile dintr-un folder, sortate."""
    return sorted(item for item in img_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def copy_image(src: Path, dst: Path) -> bool:
    """Copiază (re-encodează) o imagine; False dacă nu poate fi citită."""
    image = cv2.imread(str(src))
    if image is None:
        return False
    cv2.imwrite(str(dst), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def filter_label(label_path: Path, keep_ids: set[int]) -> tuple[list[str], int]:
    """Păstrează doar casetele claselor dorite, remapate la clasa 0; întoarce (linii, eliminate)."""
    if not label_path.exists():
        return [], 0

    kept: list[str] = []
    dropped = 0
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id not in keep_ids:
            dropped += 1
            continue
        parts[0] = "0"
        kept.append(" ".join(parts[:5]))
    return kept, dropped


def write_dataset_yaml(out_dir: Path) -> None:
    """Scrie dataset.yaml pentru datasetul rezultat (o singură clasă: trash)."""
    content = f"""# Dataset filtrat, o singură clasă (trash)

path: {out_dir.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: trash
"""
    (out_dir / "dataset.yaml").write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    src_dir = resolve_path(args.src)
    out_dir = resolve_path(args.out)
    keep_tokens = {token.strip().lower() for token in args.keep.split(",") if token.strip()}

    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Ieșirea există: {out_dir}. Adaugă --overwrite pentru a reconstrui.")
        shutil.rmtree(out_dir)

    names = load_names(src_dir)
    keep_ids = {
        class_id
        for class_id, name in names.items()
        if any(token in name.lower() for token in keep_tokens)
    }
    if not keep_ids:
        raise SystemExit(f"Nicio clasă nu se potrivește cu {sorted(keep_tokens)} în names={names}")

    print(f"Clase sursă: {names}")
    print(f"Păstrez id-urile: {sorted(keep_ids)} -> {[names[i] for i in sorted(keep_ids)]}")

    split_dirs = discover_split_dirs(src_dir)
    total_images = 0
    total_positive = 0
    total_negative = 0
    total_kept_boxes = 0
    total_dropped_boxes = 0

    for split, (img_dir, lbl_dir) in split_dirs.items():
        dst_img_dir = out_dir / "images" / split
        dst_lbl_dir = out_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = list_images(img_dir)
        split_positive = 0
        split_negative = 0
        split_boxes = 0
        split_dropped = 0

        for image_path in images:
            dst_image = dst_img_dir / f"{image_path.stem}.jpg"
            dst_label = dst_lbl_dir / f"{image_path.stem}.txt"
            if not copy_image(image_path, dst_image):
                continue
            kept_lines, dropped = filter_label(lbl_dir / f"{image_path.stem}.txt", keep_ids)
            dst_label.write_text(("\n".join(kept_lines) + "\n") if kept_lines else "", encoding="utf-8")

            split_boxes += len(kept_lines)
            split_dropped += dropped
            if kept_lines:
                split_positive += 1
            else:
                split_negative += 1

        total_images += split_positive + split_negative
        total_positive += split_positive
        total_negative += split_negative
        total_kept_boxes += split_boxes
        total_dropped_boxes += split_dropped
        print(
            f"{split:5s}: images={split_positive + split_negative:5d} "
            f"positive={split_positive:5d} negative={split_negative:5d} "
            f"kept_boxes={split_boxes:5d} dropped_boxes={split_dropped:5d}"
        )

    write_dataset_yaml(out_dir)
    print("\nSumar")
    print(f"images={total_images}, positive={total_positive}, negative={total_negative}")
    print(f"kept_boxes={total_kept_boxes}, dropped_non_target_boxes={total_dropped_boxes}")
    print(f"Dataset YAML: {out_dir / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
