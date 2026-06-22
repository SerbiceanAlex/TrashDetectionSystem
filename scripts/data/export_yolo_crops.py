"""
Exportă decupaje (crops) de obiecte dintr-un dataset YOLO de detecție.

Fiecare bounding box dintr-un fișier de etichete YOLO este decupat din imaginea
sursă și salvat ca imagine separată, plus un manifest CSV care descrie fiecare
decupaj. Decupajele sunt apoi etichetate manual pe materiale și folosite la
antrenarea clasificatorului de material.
"""

import argparse
import csv
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exportă decupaje de obiecte dintr-un dataset YOLO, pentru clasificarea ulterioară a materialului"
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Folderul cu imaginile sursă",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Folderul cu fișiere de etichete YOLO corespunzătoare imaginilor",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/material_crops_unsorted/all",
        help="Folderul de ieșire unde se scriu decupajele neetichetate",
    )
    parser.add_argument(
        "--manifest",
        default="datasets/material_crops_unsorted/crops_manifest.csv",
        help="Manifestul CSV care descrie fiecare decupaj generat",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Margine suplimentară în jurul fiecărui decupaj, ca fracțiune din lățime și înălțime",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Sare peste imaginile al căror fișier de etichete YOLO e gol",
    )
    return parser.parse_args()


def iter_images(directory: Path):
    """Întoarce, sortate, imaginile dintr-un folder (după extensiile acceptate)."""
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_yolo_line(line: str):
    """Parsează o linie YOLO (class_id x_center y_center width height, normalizate)."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Linie YOLO invalidă, se așteptau 5 valori: {line}")
    class_id = int(parts[0])
    x_center, y_center, width, height = (float(value) for value in parts[1:])
    return class_id, x_center, y_center, width, height


def yolo_to_xyxy(x_center: float, y_center: float, width: float, height: float, image_width: int, image_height: int):
    """Transformă o casetă YOLO normalizată în coordonate de pixeli (x1, y1, x2, y2)."""
    x1 = (x_center - width / 2.0) * image_width
    y1 = (y_center - height / 2.0) * image_height
    x2 = (x_center + width / 2.0) * image_width
    y2 = (y_center + height / 2.0) * image_height
    return x1, y1, x2, y2


def expand_box(x1: float, y1: float, x2: float, y2: float, margin: float, image_width: int, image_height: int):
    """Extinde caseta cu o margine procentuală, fără a ieși din imagine."""
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * margin
    pad_y = box_height * margin

    left = max(int(round(x1 - pad_x)), 0)
    top = max(int(round(y1 - pad_y)), 0)
    right = min(int(round(x2 + pad_x)), image_width)
    bottom = min(int(round(y2 + pad_y)), image_height)
    return left, top, right, bottom


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    if not images_dir.exists():
        raise FileNotFoundError(f"Folderul de imagini nu există: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Folderul de etichete nu există: {labels_dir}")
    if args.margin < 0:
        raise ValueError("--margin trebuie să fie >= 0")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    images = iter_images(images_dir)
    if not images:
        raise SystemExit(f"Nicio imagine găsită în {images_dir}")

    written = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "crop_path",
                "source_image",
                "label_path",
                "box_index",
                "class_id",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                skipped += 1
                continue

            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines and args.skip_empty:
                skipped += 1
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                skipped += 1
                continue

            image_height, image_width = image.shape[:2]
            for index, line in enumerate(lines):
                class_id, x_center, y_center, width, height = parse_yolo_line(line)
                x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, width, height, image_width, image_height)
                left, top, right, bottom = expand_box(x1, y1, x2, y2, args.margin, image_width, image_height)
                if right <= left or bottom <= top:
                    continue

                crop = image[top:bottom, left:right]
                crop_name = f"{image_path.stem}_crop_{index:03d}{image_path.suffix}"
                crop_path = out_dir / crop_name
                if not cv2.imwrite(str(crop_path), crop):
                    continue

                writer.writerow(
                    [
                        crop_path.as_posix(),
                        image_path.as_posix(),
                        label_path.as_posix(),
                        index,
                        class_id,
                        left,
                        top,
                        right,
                        bottom,
                    ]
                )
                written += 1

    print(f"[GATA] Scrise {written} decupaje în {out_dir}")
    print(f"[INFO] Manifest salvat în: {manifest_path}")
    if skipped:
        print(f"[INFO] Sărite {skipped} imagini (etichete lipsă, goale sau fișiere necitibile)")


if __name__ == "__main__":
    main()
