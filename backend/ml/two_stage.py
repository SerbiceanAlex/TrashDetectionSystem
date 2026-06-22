"""
Two-stage trash pipeline: single-class YOLO detector + material classifier.

Library functions used across the backend:
  - detect_and_classify(): detect trash boxes, then classify each crop's material
  - classify_crop():        classify a single crop's material
  - draw_detections():      annotate a frame with boxes + material labels
  - classifier_names():     normalize a YOLO model's class names to {index: name}
"""

from collections import Counter

import cv2


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
    if classifier is None:
        return "unknown", 0.0
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
