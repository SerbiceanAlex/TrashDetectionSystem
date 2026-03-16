"""trash_detection/yolo.py – YOLO dataset utilities.

Covers dataset YAML generation, coordinate conversion between pixel bounding
boxes and YOLO's normalised (cx, cy, w, h) format, and label-file I/O.
"""

from __future__ import annotations

from pathlib import Path


def write_dataset_yaml(
    path: str | Path,
    dataset_root: str | Path,
    nc: int,
    names: list[str],
    train_rel: str = "images/train",
    val_rel: str = "images/val",
    test_rel: str = "images/test",
) -> Path:
    """Write a YOLO-compatible dataset YAML file.

    Parameters
    ----------
    path:
        Destination path for the YAML file (e.g. ``trash_yolo.yaml``).
    dataset_root:
        Absolute path to the dataset root directory (written as ``path:``).
    nc:
        Number of classes.
    names:
        List of class name strings, length must equal *nc*.
    train_rel, val_rel, test_rel:
        Relative paths (from *dataset_root*) to the images sub-directories.

    Returns
    -------
    Path to the written YAML file.
    """
    if len(names) != nc:
        raise ValueError(
            f"Number of class names ({len(names)}) must equal the nc parameter ({nc}). "
            f"Please provide exactly {nc} class name(s)."
        )

    names_yaml = "\n".join(f"  - {n}" for n in names)
    content = (
        f"path: {Path(dataset_root).resolve()}\n"
        f"train: {train_rel}\n"
        f"val:   {val_rel}\n"
        f"test:  {test_rel}\n"
        f"\nnc: {nc}\n"
        f"names:\n{names_yaml}\n"
    )

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return out


def annotation_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    class_id: int = 0,
) -> str:
    """Convert a pixel bounding box to a YOLO annotation line.

    Parameters
    ----------
    x1, y1, x2, y2:
        Bounding-box corners in pixel coordinates (top-left, bottom-right).
    img_w, img_h:
        Image dimensions in pixels.
    class_id:
        Integer class index.

    Returns
    -------
    YOLO annotation string: ``"<class_id> <cx> <cy> <w> <h>"``
    """
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def read_yolo_label(path: str | Path) -> list[dict]:
    """Parse a YOLO .txt label file.

    Parameters
    ----------
    path:
        Path to the label file.

    Returns
    -------
    List of dicts, each containing:
    ``{"class_id": int, "cx": float, "cy": float, "w": float, "h": float}``

    Empty files return an empty list (no annotations).
    """
    label_path = Path(path)
    if not label_path.exists():
        return []

    records: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        records.append(
            {
                "class_id": int(parts[0]),
                "cx": float(parts[1]),
                "cy": float(parts[2]),
                "w": float(parts[3]),
                "h": float(parts[4]),
            }
        )
    return records


def yolo_to_bbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Convert YOLO normalised coordinates to pixel bounding box.

    Parameters
    ----------
    cx, cy, w, h:
        Normalised centre and size values in [0, 1].
    img_w, img_h:
        Image dimensions in pixels.

    Returns
    -------
    ``(x1, y1, x2, y2)`` in integer pixel coordinates.
    """
    x1 = int((cx - w / 2.0) * img_w)
    y1 = int((cy - h / 2.0) * img_h)
    x2 = int((cx + w / 2.0) * img_w)
    y2 = int((cy + h / 2.0) * img_h)
    return x1, y1, x2, y2
