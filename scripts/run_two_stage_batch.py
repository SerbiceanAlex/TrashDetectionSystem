import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detect_two_stage import classifier_names, detect_and_classify, draw_detections
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run two-stage trash detection and material classification on a directory of images"
    )
    parser.add_argument("--source-dir", required=True, help="Directory containing input images")
    parser.add_argument(
        "--detector",
        default="runs/detect/parks-trash-yolov8/weights/best.pt",
        help="YOLO detector checkpoint for the single-class trash detector",
    )
    parser.add_argument(
        "--classifier",
        default="runs/classify/parks-trash-material-cls/weights/best.pt",
        help="YOLO classification checkpoint for material classification",
    )
    parser.add_argument("--det-conf", type=float, default=0.25, help="Detector confidence threshold")
    parser.add_argument("--det-imgsz", type=int, default=640, help="Detector inference image size")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="Classifier inference image size")
    parser.add_argument("--max-labels", type=int, default=5, help="Number of class counts shown in overlay")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")
    parser.add_argument("--save-images", action="store_true", help="Save annotated images to the output directory")
    parser.add_argument("--device", default=None, help="Device to use: cpu, 0, 0,1, ...")
    parser.add_argument("--out-dir", default="outputs/two_stage_batch", help="Output directory for artifacts")
    return parser.parse_args()


def iter_images(directory: Path):
    if not directory.exists():
        raise FileNotFoundError(f"Source directory not found: {directory}")
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_models(detector_path: Path, classifier_path: Path):
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {detector_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")
    return YOLO(str(detector_path)), YOLO(str(classifier_path))


def main():
    args = parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    annotated_dir = out_dir / "annotated"
    csv_path = out_dir / "detections.csv"

    images = iter_images(source_dir)
    if not images:
        raise SystemExit(f"No images found in {source_dir}")

    detector, classifier = load_models(Path(args.detector), Path(args.classifier))
    class_names = classifier_names(classifier)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_images:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_path",
                "detection_index",
                "x1",
                "y1",
                "x2",
                "y2",
                "det_score",
                "material_name",
                "material_score",
            ]
        )

        processed = 0
        total_detections = 0
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            detections = detect_and_classify(
                frame,
                detector,
                classifier,
                args.det_conf,
                args.det_imgsz,
                args.cls_imgsz,
                class_names,
            )
            total_detections += len(detections)
            processed += 1

            for detection in detections:
                left, top, right, bottom = detection["box"]
                writer.writerow(
                    [
                        image_path.as_posix(),
                        detection["index"],
                        left,
                        top,
                        right,
                        bottom,
                        detection["det_score"],
                        detection["material_name"],
                        detection["material_score"],
                    ]
                )

            if args.save_images:
                annotated = draw_detections(frame, detections, fps=0.0, max_labels=args.max_labels, line_width=args.line_width)
                cv2.imwrite(str(annotated_dir / image_path.name), annotated)

    print(f"[INFO] Processed images: {processed}")
    print(f"[INFO] Total detections: {total_detections}")
    print(f"[INFO] Detection CSV: {csv_path}")
    if args.save_images:
        print(f"[INFO] Annotated images directory: {annotated_dir}")


if __name__ == "__main__":
    main()