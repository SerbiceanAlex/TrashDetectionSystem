"""trash_detection/viz.py – Visualization utilities.

Helpers for drawing bounding boxes on images, plotting class distributions,
and displaying image grids using matplotlib.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def draw_bbox(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str = "",
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a single bounding box on *image* (in-place) and return it.

    Parameters
    ----------
    image:
        BGR image array.
    x1, y1, x2, y2:
        Bounding-box corners in pixels.
    label:
        Optional text label drawn above the box.
    color:
        BGR colour tuple.
    thickness:
        Rectangle line thickness in pixels.
    """
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        bg_y1 = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(image, (x1, bg_y1), (x1 + text_w, y1), color, cv2.FILLED)
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def draw_yolo_annotations(
    image: np.ndarray,
    yolo_labels: list[dict],
    class_names: Optional[list[str]] = None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw all YOLO annotations on *image* and return the annotated copy.

    Parameters
    ----------
    image:
        BGR image array.
    yolo_labels:
        List of dicts as returned by :func:`trash_detection.yolo.read_yolo_label`.
    class_names:
        Optional list of class name strings indexed by ``class_id``.
    color:
        Default BGR colour for all boxes.
    """
    from trash_detection.yolo import yolo_to_bbox

    img_h, img_w = image.shape[:2]
    out = image.copy()

    for ann in yolo_labels:
        x1, y1, x2, y2 = yolo_to_bbox(
            ann["cx"], ann["cy"], ann["w"], ann["h"], img_w, img_h
        )
        cid = ann["class_id"]
        label = class_names[cid] if class_names and cid < len(class_names) else str(cid)
        draw_bbox(out, x1, y1, x2, y2, label=label, color=color)

    return out


def plot_class_distribution(
    class_counts: dict[str, int],
    title: str = "Class Distribution",
) -> None:
    """Show a bar chart of class counts using matplotlib.

    Parameters
    ----------
    class_counts:
        Mapping from class name to count.
    title:
        Chart title.
    """
    import matplotlib.pyplot as plt

    names = list(class_counts.keys())
    counts = [class_counts[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
    bars = ax.bar(names, counts, color="steelblue", edgecolor="white")
    ax.bar_label(bars, padding=3)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    fig.tight_layout()
    plt.show()


def show_image_grid(
    images: list[np.ndarray],
    titles: Optional[list[str]] = None,
    cols: int = 4,
) -> None:
    """Display a grid of BGR images using matplotlib.

    Parameters
    ----------
    images:
        List of BGR numpy arrays (uint8).
    titles:
        Optional list of titles, one per image.
    cols:
        Number of columns in the grid.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    if n == 0:
        print("[viz] No images to display.")
        return

    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)  # flatten for uniform indexing

    for i, ax in enumerate(axes):
        if i < n:
            # Convert BGR → RGB for matplotlib
            rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=9)
            ax.axis("off")
        else:
            ax.axis("off")

    fig.tight_layout()
    plt.show()
