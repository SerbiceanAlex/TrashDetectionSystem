import argparse
import os
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser("Two-stage trash detection and material classification")
    parser.add_argument("--source", required=True, help="0 for webcam, or path to an image or video file")
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
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--save", action="store_true", help="Save annotated image or video to outputs/")
    parser.add_argument("--max-labels", type=int, default=5, help="Number of class counts shown in overlay")
    parser.add_argument("--line-width", type=int, default=2, help="Bounding box line width")
    return parser.parse_args()


def is_image_source(source):
    return Path(source).suffix.lower() in IMAGE_EXTENSIONS


def is_webcam_source(source):
    return str(source).isdigit()


def classifier_names(model):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def clamp_box(x1, y1, x2, y2, width, height):
    left = max(int(round(x1)), 0)
    top = max(int(round(y1)), 0)
    right = min(int(round(x2)), width)
    bottom = min(int(round(y2)), height)
    return left, top, right, bottom


def classify_crop(classifier, crop, imgsz, class_names):
    result = classifier.predict(crop, imgsz=imgsz, verbose=False)[0]
    probs = getattr(result, "probs", None)
    if probs is None:
        return "unknown", 0.0

    top_index = int(probs.top1)
    top_conf = float(probs.top1conf.item() if hasattr(probs.top1conf, "item") else probs.top1conf)
    return class_names.get(top_index, str(top_index)), top_conf


def detect_and_classify(frame, detector, classifier, det_conf, det_imgsz, cls_imgsz, class_names):
    result = detector.predict(frame, conf=det_conf, imgsz=det_imgsz, verbose=False)[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []

    detections = []
    frame_height, frame_width = frame.shape[:2]
    xyxy_list = boxes.xyxy.tolist()
    conf_list = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)

    for index, (xyxy, det_score) in enumerate(zip(xyxy_list, conf_list)):
        left, top, right, bottom = clamp_box(*xyxy, frame_width, frame_height)
        if right <= left or bottom <= top:
            continue

        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue

        material_name, material_score = classify_crop(classifier, crop, cls_imgsz, class_names)
        detections.append(
            {
                "index": index,
                "box": (left, top, right, bottom),
                "det_score": float(det_score),
                "material_name": material_name,
                "material_score": material_score,
            }
        )

    return detections


def draw_detections(frame, detections, fps, max_labels, line_width):
    annotated = frame.copy()
    counts = Counter(detection["material_name"] for detection in detections)

    for detection in detections:
        left, top, right, bottom = detection["box"]
        material_name = detection["material_name"]
        material_score = detection["material_score"]
        det_score = detection["det_score"]
        label = f"trash | {material_name} {material_score:.2f} | det {det_score:.2f}"

        cv2.rectangle(annotated, (left, top), (right, bottom), (0, 220, 0), line_width)
        text_y = top - 10 if top > 25 else top + 25
        cv2.putText(
            annotated,
            label,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Trash objects: {len(detections)}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    y = 95
    for class_name, count in counts.most_common(max_labels):
        cv2.putText(
            annotated,
            f"{class_name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return annotated


def ensure_output_path(source, is_image):
    os.makedirs("outputs", exist_ok=True)
    timestamp = int(time.time())
    source_name = Path(str(source)).stem if not is_webcam_source(source) else "webcam"
    suffix = ".jpg" if is_image else ".mp4"
    return Path("outputs") / f"two_stage_{source_name}_{timestamp}{suffix}"


def run_on_image(source_path, detector, classifier, args, class_names):
    frame = cv2.imread(str(source_path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {source_path}")

    start = time.time()
    detections = detect_and_classify(
        frame,
        detector,
        classifier,
        args.det_conf,
        args.det_imgsz,
        args.cls_imgsz,
        class_names,
    )
    fps = 1.0 / max(time.time() - start, 1e-6)
    annotated = draw_detections(frame, detections, fps, args.max_labels, args.line_width)

    if args.save:
        out_path = ensure_output_path(source_path, is_image=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[INFO] Saved annotated image to: {out_path.resolve()}")

    if args.show:
        cv2.imshow("TrashDetectionSystem - Two-Stage Inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(source, detector, classifier, args, class_names):
    source_value = int(source) if is_webcam_source(source) else source
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    if args.save:
        out_path = ensure_output_path(source, is_image=False)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
        print(f"[INFO] Saving annotated video to: {out_path.resolve()}")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detect_and_classify(
            frame,
            detector,
            classifier,
            args.det_conf,
            args.det_imgsz,
            args.cls_imgsz,
            class_names,
        )
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        annotated = draw_detections(frame, detections, fps, args.max_labels, args.line_width)

        if writer is not None:
            writer.write(annotated)

        if args.show:
            cv2.imshow("TrashDetectionSystem - Two-Stage Inference", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    detector_path = Path(args.detector)
    classifier_path = Path(args.classifier)
    if not detector_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {detector_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")

    detector = YOLO(str(detector_path))
    classifier = YOLO(str(classifier_path))
    class_names = classifier_names(classifier)

    if is_image_source(args.source):
        run_on_image(Path(args.source), detector, classifier, args, class_names)
    else:
        run_on_video(args.source, detector, classifier, args, class_names)


if __name__ == "__main__":
    main()