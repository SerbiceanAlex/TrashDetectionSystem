"""
Standalone smoke test for the littering-detection stack.

This file is intentionally safe to import, so pytest collection does not execute
YOLO model loading. Run it directly when you want an end-to-end local check:

    .venv\\Scripts\\python.exe scripts\\smoke\\littering_system_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

__test__ = False

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PASS = "[  OK  ]"
FAIL = "[ FAIL ]"
INFO = "[ INFO ]"


def ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def fail(errors: list[str], msg: str, detail: str = "") -> None:
    print(f"{FAIL} {msg}" + (f": {detail}" if detail else ""))
    errors.append(msg)


def info(msg: str) -> None:
    print(f"{INFO} {msg}")


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def find_test_image() -> Path | None:
    candidates = [
        REPO_ROOT / "datasets/raw/images/park01_download_img_001.jpg",
        REPO_ROOT / "datasets/parks_detect_A4/images/test",
        REPO_ROOT / "datasets/parks_detect_full/images/val",
        REPO_ROOT / "datasets/parks_detect/images/val",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            images = sorted(candidate.glob("*.jpg")) + sorted(candidate.glob("*.png"))
            if images:
                return images[0]
    return None


def main() -> int:
    errors: list[str] = []

    model_files = {
        "Trash detector": REPO_ROOT / "models/detector/production/best.pt",
        "Material classifier": REPO_ROOT / "models/classify/B2/best.pt",
        "Person detector": REPO_ROOT / "models/pretrained/yolov8n.pt",
    }

    section("TEST 1 - Model files")
    for name, path in model_files.items():
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            ok(f"{name}: {path.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
        else:
            fail(errors, f"{name} missing", str(path))

    if errors:
        section("SUMMARY")
        print(f"{len(errors)} required file(s) missing.")
        return 1

    section("TEST 2 - Load YOLO models")
    try:
        import torch
        from ultralytics import YOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"
        info(f"Device: {device}")

        t0 = time.perf_counter()
        trash_det = YOLO(str(model_files["Trash detector"]))
        ok(f"Trash detector loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")

        t0 = time.perf_counter()
        classifier = YOLO(str(model_files["Material classifier"]))
        ok(f"Material classifier loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")

        raw = getattr(classifier, "names", {})
        cls_names = {int(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {
            i: str(v) for i, v in enumerate(raw)
        }
        info(f"Classifier classes: {list(cls_names.values())}")

        t0 = time.perf_counter()
        person_det = YOLO(str(model_files["Person detector"]))
        ok(f"Person detector loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")

        names = getattr(person_det, "names", {})
        person_name = names.get(0) if isinstance(names, dict) else names[0]
        if person_name == "person":
            ok("COCO class 0 is person")
        else:
            fail(errors, "COCO class 0 is not person", str(person_name))
    except Exception as exc:
        fail(errors, "Failed to load models", str(exc))
        return 1

    section("TEST 3 - Inference on a real image")
    test_image = find_test_image()
    frame = None
    if test_image is None:
        fail(errors, "No test image found in datasets")
    else:
        ok(f"Test image: {test_image.relative_to(REPO_ROOT)}")
        frame = cv2.imread(str(test_image))
        if frame is None:
            fail(errors, "cv2.imread returned None", str(test_image))
        else:
            ok(f"Image read: {frame.shape[1]}x{frame.shape[0]} px")
            t0 = time.perf_counter()
            results = trash_det.predict(frame, conf=0.25, imgsz=640, verbose=False)
            elapsed = (time.perf_counter() - t0) * 1000
            boxes = results[0].boxes
            n_trash = len(boxes) if boxes is not None else 0
            ok(f"Trash predict: {n_trash} objects in {elapsed:.0f} ms")

            if n_trash > 0 and boxes.xyxy is not None:
                x1, y1, x2, y2 = [max(0, int(v)) for v in boxes.xyxy[0].tolist()]
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    cls_result = classifier.predict(crop, imgsz=224, verbose=False)[0]
                    probs = getattr(cls_result, "probs", None)
                    if probs is not None:
                        top_idx = int(probs.top1)
                        top_conf = float(probs.top1conf)
                        ok(f"Material crop 0: {cls_names.get(top_idx, top_idx)} ({top_conf:.3f})")

    section("TEST 4 - ByteTrack")
    if frame is None:
        info("Skipped because no frame was available")
    else:
        track_ids_seen: set[int] = set()
        for _ in range(3):
            results = trash_det.track(
                frame,
                conf=0.25,
                imgsz=640,
                verbose=False,
                persist=True,
                tracker="bytetrack.yaml",
            )
            boxes = results[0].boxes
            if boxes is not None and getattr(boxes, "id", None) is not None:
                track_ids_seen.update(int(x) for x in boxes.id.tolist())
        if track_ids_seen:
            ok(f"Track IDs generated: {sorted(track_ids_seen)}")
        else:
            info("No track IDs on this still image; normal when there are no detections")

    section("TEST 5 - Person detection")
    if frame is None:
        info("Skipped because no frame was available")
    else:
        t0 = time.perf_counter()
        results = person_det.predict(frame, conf=0.40, imgsz=640, verbose=False, classes=[0])
        elapsed = (time.perf_counter() - t0) * 1000
        boxes = results[0].boxes
        n_persons = len(boxes) if boxes is not None else 0
        ok(f"Person predict: {n_persons} persons in {elapsed:.0f} ms")

    section("TEST 6 - Face blur helper")
    blur_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blur_frame[50:300, 200:350] = 180
    person_box = (200, 50, 350, 300)
    x1, y1, x2, y2 = person_box
    face_h = int((y2 - y1) * 0.25)
    face_region = blur_frame[y1:y1 + face_h, x1:x2].copy()
    blur_frame[y1:y1 + face_h, x1:x2] = cv2.GaussianBlur(face_region, (51, 51), 0)
    ok(f"Gaussian blur ran; region mean={float(blur_frame[y1:y1 + face_h, x1:x2].mean()):.1f}")

    section("TEST 7 - Tracker configs")
    try:
        import ultralytics

        ul_root = Path(ultralytics.__file__).parent
        bytetrack_cfg = ul_root / "cfg" / "trackers" / "bytetrack.yaml"
        if bytetrack_cfg.exists():
            ok(f"bytetrack.yaml found: {bytetrack_cfg}")
        else:
            fail(errors, "bytetrack.yaml missing", str(bytetrack_cfg))
    except Exception as exc:
        fail(errors, "Failed to check tracker config", str(exc))

    section("SUMMARY")
    if not errors:
        print("All smoke checks passed.")
        return 0

    print(f"{len(errors)} check(s) failed:")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
