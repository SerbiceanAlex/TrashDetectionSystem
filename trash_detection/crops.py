"""trash_detection/crops.py – Crop generation utilities for the classification stage.

Crops are generated from detection bounding boxes (YOLO format labels) and
saved as individual JPEG files.  A metadata manifest CSV is produced for use
in the classification training notebook.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from trash_detection.io import IMAGE_EXTENSIONS
from trash_detection.yolo import read_yolo_label, yolo_to_bbox


def generate_crop(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding: float = 0.1,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Crop a bounding-box region from *image* with optional padding.

    Parameters
    ----------
    image:
        BGR image array (H, W, 3).
    x1, y1, x2, y2:
        Bounding-box corners in pixel coordinates.
    padding:
        Fractional padding added on each side, relative to the bbox dimension.
        E.g. 0.1 adds 10 % of width/height on each side.
    target_size:
        ``(width, height)`` to which the crop is resized.

    Returns
    -------
    Resized crop as a numpy array (target_h, target_w, 3).
    """
    img_h, img_w = image.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(img_w, x2 + pad_x)
    y2p = min(img_h, y2 + pad_y)

    crop = image[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        # Bounding box is degenerate (zero area); return a blank crop
        print(
            f"  [WARN] Degenerate bounding box ({x1},{y1},{x2},{y2}) – "
            "returning blank crop. Check your annotations."
        )
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    return cv2.resize(crop, target_size)


def save_crops_from_yolo_labels(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    class_name: str = "trash",
    padding: float = 0.1,
    target_size: tuple[int, int] = (224, 224),
) -> list[dict]:
    """Generate and save crops for every annotated bounding box.

    For each image in *images_dir* with a corresponding label file in
    *labels_dir*, every bounding box is cropped, padded, resized, and saved
    under *output_dir/<class_name>/*.

    Parameters
    ----------
    images_dir:
        Directory containing source images.
    labels_dir:
        Directory containing YOLO .txt label files (same stems as images).
    output_dir:
        Root output directory.  Crops land in ``output_dir/<class_name>/``.
    class_name:
        Sub-directory name used for saving crops (the assumed label).
    padding:
        Fractional bbox padding passed to :func:`generate_crop`.
    target_size:
        Resize target ``(width, height)`` for saved crops.

    Returns
    -------
    List of metadata dicts with keys:
    ``source_image``, ``label_path``, ``crop_path``, ``class_name``,
    ``bbox_x1``, ``bbox_y1``, ``bbox_x2``, ``bbox_y2``.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_class_dir = Path(output_dir) / class_name
    out_class_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    crop_idx = 0

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        annotations = read_yolo_label(label_path)
        if not annotations:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] Cannot read image: {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        for ann in annotations:
            x1, y1, x2, y2 = yolo_to_bbox(
                ann["cx"], ann["cy"], ann["w"], ann["h"], img_w, img_h
            )
            crop = generate_crop(image, x1, y1, x2, y2, padding, target_size)
            crop_filename = f"{img_path.stem}_crop_{crop_idx:05d}.jpg"
            crop_path = out_class_dir / crop_filename
            cv2.imwrite(str(crop_path), crop)

            records.append(
                {
                    "source_image": str(img_path),
                    "label_path": str(label_path),
                    "crop_path": str(crop_path),
                    "class_name": class_name,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                }
            )
            crop_idx += 1

    return records


def build_crops_manifest(records: list[dict], csv_path: str | Path) -> Path:
    """Save crop metadata *records* to a CSV file.

    Parameters
    ----------
    records:
        List of dicts as returned by :func:`save_crops_from_yolo_labels`.
    csv_path:
        Destination CSV path.

    Returns
    -------
    Path to the written CSV file.
    """
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        out.write_text("", encoding="utf-8")
        return out

    fieldnames = list(records[0].keys())
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return out
