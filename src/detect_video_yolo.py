import argparse
import os
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser("YOLOv8 video detection (baseline)")
    p.add_argument("--source", required=True, help="0 for webcam, or path to a video file")
    p.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model path or name (e.g. yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--show", action="store_true", help="Show preview window")
    p.add_argument("--save", action="store_true", help="Save annotated output video to outputs/")
    p.add_argument("--max-labels", type=int, default=5, help="Number of per-frame class counts to overlay")
    return p.parse_args()


def frame_class_counts(result, class_names):
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return Counter()

    counts = Counter()
    for class_id in boxes.cls.tolist():
        class_index = int(class_id)
        counts[class_names.get(class_index, str(class_index))] += 1
    return counts


def draw_overlay(frame, fps, counts, max_labels):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    y = 80
    for class_name, count in counts.most_common(max_labels):
        cv2.putText(
            frame,
            f"{class_name}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28


def main():
    args = parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source

    model = YOLO(args.model)
    class_names = model.names if isinstance(model.names, dict) else {i: name for i, name in enumerate(model.names)}

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    writer = None
    if args.save:
        os.makedirs("outputs", exist_ok=True)
        out_path = Path("outputs") / f"annotated_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (w, h))
        print(f"[INFO] Saving to: {out_path.resolve()}")

    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)
        annotated = results[0].plot()
        counts = frame_class_counts(results[0], class_names)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        draw_overlay(annotated, fps, counts, args.max_labels)

        if writer is not None:
            writer.write(annotated)

        if args.show:
            cv2.imshow("TrashDetectionSystem - YOLOv8 baseline (press q)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()