"""Auto-label images using a pretrained YOLOv8 model (COCO weights).

Maps relevant COCO classes to a single 'trash' class (0) and writes
YOLO-format label files.  Useful for generating quick pseudo-labels
that can be manually reviewed and corrected.

COCO classes mapped → trash:
  bottle(39), wine glass(40), cup(41), fork(42), knife(43), spoon(44),
  bowl(45), banana(46), apple(47), sandwich(48), orange(49), broccoli(50),
  carrot(51), hot dog(52), pizza(53), donut(54), cake(55),
  handbag(26), backpack(24), suitcase(28), umbrella(25),
  vase(75), scissors(76), toothbrush(79), book(73), cell phone(67)

Usage:
    python scripts/auto_label_coco_to_trash.py
    python scripts/auto_label_coco_to_trash.py --images-dir datasets/parks_detect/images/all --conf 0.20
    python scripts/auto_label_coco_to_trash.py --model yolov8s.pt --conf 0.15 --preview
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

# COCO class indices that are plausible trash / litter items
TRASH_COCO_IDS = {
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    40,  # wine glass
    41,  # cup
    42,  # fork
    43,  # knife
    44,  # spoon
    45,  # bowl
    46,  # banana
    47,  # apple
    48,  # sandwich
    49,  # orange
    50,  # broccoli
    51,  # carrot
    52,  # hot dog
    53,  # pizza
    54,  # donut
    55,  # cake
    67,  # cell phone
    73,  # book
    75,  # vase
    76,  # scissors
    79,  # toothbrush
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Auto-label images using COCO pretrained YOLO → single-class trash")
    p.add_argument("--images-dir", default="datasets/parks_detect/images/all", help="Directory with images")
    p.add_argument("--labels-dir", default=None, help="Output labels directory (default: sibling labels/all)")
    p.add_argument("--model", default="yolov8n.pt", help="Pretrained YOLO model path")
    p.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing label files")
    p.add_argument("--preview", action="store_true", help="Print detections without writing labels")
    return p.parse_args()


def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if args.labels_dir:
        labels_dir = Path(args.labels_dir)
    else:
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_files:
        raise SystemExit(f"No images found in {images_dir}")

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    total_images = len(image_files)
    labeled_count = 0
    total_boxes = 0
    skipped_existing = 0

    print(f"[INFO] Processing {total_images} images (conf={args.conf}, iou={args.iou})...\n")

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if label_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            verbose=False,
        )

        lines = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                if cls_id not in TRASH_COCO_IDS:
                    continue
                # normalized xywh
                xywhn = boxes.xywhn[i]
                cx, cy, w, h = xywhn[0].item(), xywhn[1].item(), xywhn[2].item(), xywhn[3].item()
                conf = boxes.conf[i].item()
                coco_name = model.names[cls_id]
                lines.append((cx, cy, w, h, conf, coco_name))

        if args.preview:
            if lines:
                print(f"  {img_path.name}: {len(lines)} detections")
                for cx, cy, w, h, conf, name in lines:
                    print(f"    {name} ({conf:.2f})  xywh=[{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}]")
            else:
                print(f"  {img_path.name}: 0 detections")
            continue

        # Write YOLO label (class 0 = trash)
        with open(label_path, "w", encoding="utf-8") as f:
            for cx, cy, w, h, conf, name in lines:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        n_boxes = len(lines)
        total_boxes += n_boxes
        if n_boxes > 0:
            labeled_count += 1
        print(f"  {img_path.name} → {n_boxes} boxes {'(empty negative)' if n_boxes == 0 else ''}")

    print()
    if args.preview:
        print("[INFO] Preview mode — no files written.")
    else:
        print(f"[INFO] Done. Labels written to: {labels_dir}")
        print(f"[INFO] Images with detections: {labeled_count}/{total_images}")
        print(f"[INFO] Total bounding boxes: {total_boxes}")
        if skipped_existing:
            print(f"[INFO] Skipped (existing labels): {skipped_existing}")
    print()
    print("[IMPORTANT] These are pseudo-labels from COCO pretrained weights.")
    print("            Review and correct them manually before final training!")


if __name__ == "__main__":
    main()
