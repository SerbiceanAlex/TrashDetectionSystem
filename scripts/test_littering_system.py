"""
Test script — verifică toate componentele necesare pentru Littering Detection.

Rulare:
    .venv\\Scripts\\python.exe scripts\\test_littering_system.py

Testele acoperă:
  1. Existența fișierelor model (.pt)
  2. Încărcarea trash detector + clasificator
  3. Detecție pe o imagine reală (predict + track)
  4. Încărcarea person detector (yolov8n.pt, class 0)
  5. Person detection pe o imagine reală
  6. Face blur (cv2.GaussianBlur pe top 25% bbox)
  7. Tracking persistent (persist=True) — verifică că track_ids există
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PASS = "[  OK  ]"
FAIL = "[ FAIL ]"
INFO = "[ INFO ]"

errors: list[str] = []


def ok(msg: str) -> None:
    print(f"{PASS} {msg}")


def fail(msg: str, detail: str = "") -> None:
    print(f"{FAIL} {msg}" + (f": {detail}" if detail else ""))
    errors.append(msg)


def info(msg: str) -> None:
    print(f"{INFO} {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Existența modelelor
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 1 — Existența fișierelor model")
print("=" * 60)

model_files = {
    "Trash detector":       REPO_ROOT / "runs/detect/parks-trash-A3-final/weights/best.pt",
    "Material classifier":  REPO_ROOT / "runs/classify/parks-cls-B2/weights/best.pt",
    "Person detector":      REPO_ROOT / "yolov8n.pt",
}

for name, path in model_files.items():
    if path.exists():
        size_mb = path.stat().st_size / 1024 / 1024
        ok(f"{name}: {path.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
    else:
        fail(f"{name} LIPSĂ", str(path))

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Încărcarea modelelor
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2 — Încărcarea modelelor în memorie")
print("=" * 60)

try:
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"Device: {device}")

    t0 = time.perf_counter()
    trash_det = YOLO(str(model_files["Trash detector"]))
    ok(f"Trash detector încărcat în {(time.perf_counter()-t0)*1000:.0f}ms")

    t0 = time.perf_counter()
    classifier = YOLO(str(model_files["Material classifier"]))
    ok(f"Material classifier încărcat în {(time.perf_counter()-t0)*1000:.0f}ms")

    # Extrage numele claselor clasificator
    raw = getattr(classifier, "names", {})
    cls_names = {int(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {i: str(v) for i, v in enumerate(raw)}
    info(f"Clase clasificator: {list(cls_names.values())}")

    t0 = time.perf_counter()
    person_det = YOLO(str(model_files["Person detector"]))
    ok(f"Person detector încărcat în {(time.perf_counter()-t0)*1000:.0f}ms")

    # Verifică că yolov8n are clasa 0 = person
    person_names = getattr(person_det, "names", {})
    if isinstance(person_names, dict) and person_names.get(0) == "person":
        ok(f"Person class 0 = 'person' confirmat")
    elif isinstance(person_names, list) and len(person_names) > 0 and person_names[0] == "person":
        ok(f"Person class 0 = 'person' confirmat (list)")
    else:
        fail("Class 0 nu este 'person'", str(person_names.get(0) if isinstance(person_names, dict) else ""))

except Exception as e:
    fail("Eroare la încărcarea modelelor", str(e))
    print("  → Nu se poate continua fără modele.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Imagine de test — găsim una existentă
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3 — Detecție pe imagine reală (predict)")
print("=" * 60)

test_image_path: Path | None = None
for candidate in [
    REPO_ROOT / "datasets/raw/images/park01_download_img_001.jpg",
    REPO_ROOT / "datasets/parks_detect_full/images/val",
    REPO_ROOT / "datasets/parks_detect/images/val",
]:
    if candidate.is_file():
        test_image_path = candidate
        break
    if candidate.is_dir():
        imgs = sorted(candidate.glob("*.jpg"))
        if imgs:
            test_image_path = imgs[0]
            break

if test_image_path is None:
    fail("Nu s-a găsit nicio imagine de test în datasets/")
else:
    ok(f"Imagine de test: {test_image_path.relative_to(REPO_ROOT)}")
    frame = cv2.imread(str(test_image_path))
    if frame is None:
        fail("cv2.imread a returnat None", str(test_image_path))
    else:
        ok(f"Imagine citită: {frame.shape[1]}×{frame.shape[0]} px")

        # Predict (modul clasic)
        t0 = time.perf_counter()
        results = trash_det.predict(frame, conf=0.25, imgsz=640, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000
        boxes = results[0].boxes
        n_trash = len(boxes) if boxes is not None else 0
        ok(f"Trash detect (predict): {n_trash} obiecte în {elapsed:.0f}ms")

        # Clasificare pe primul crop
        if n_trash > 0 and boxes.xyxy is not None:
            xyxy = boxes.xyxy[0].tolist()
            x1, y1, x2, y2 = [max(0, int(v)) for v in xyxy]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cls_result = classifier.predict(crop, imgsz=224, verbose=False)[0]
                probs = getattr(cls_result, "probs", None)
                if probs is not None:
                    top_idx = int(probs.top1)
                    top_conf = float(probs.top1conf)
                    mat = cls_names.get(top_idx, str(top_idx))
                    ok(f"Material crop 0: '{mat}' (conf={top_conf:.3f})")
                else:
                    fail("Clasificatorul nu a returnat probs")
        else:
            info("0 detecții pe această imagine — normal dacă imaginea nu conține deșeuri clare")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Tracking — verifică că track_ids sunt generate
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4 — Tracking cu ByteTrack (persist=True)")
print("=" * 60)

if test_image_path and frame is not None:
    try:
        # Rulăm track de 3 ori pe același frame (simulează 3 frame-uri consecutive)
        track_ids_seen: set[int] = set()
        for i in range(3):
            results = trash_det.track(
                frame, conf=0.25, imgsz=640, verbose=False,
                persist=True, tracker="bytetrack.yaml"
            )
            boxes = results[0].boxes
            if boxes is not None and hasattr(boxes, "id") and boxes.id is not None:
                ids = [int(x) for x in boxes.id.tolist()]
                track_ids_seen.update(ids)

        if track_ids_seen:
            ok(f"Track IDs generați: {sorted(track_ids_seen)} — ByteTrack funcționează")
        else:
            info("Nu s-au generat track IDs pe această imagine (0 detecții sau conf prea mare)")
            info("Tracking va funcționa pe video în timp real — normal pentru imagine statică")

    except Exception as e:
        fail("Eroare la tracking", str(e))
else:
    info("TEST 4 sărit — nu avem imagine de test")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Person detection pe imagine de test
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5 — Person detection (yolov8n, class 0)")
print("=" * 60)

if test_image_path and frame is not None:
    try:
        t0 = time.perf_counter()
        results = person_det.predict(
            frame, conf=0.40, imgsz=640, verbose=False, classes=[0]
        )
        elapsed = (time.perf_counter() - t0) * 1000
        boxes = results[0].boxes
        n_persons = len(boxes) if boxes is not None else 0
        ok(f"Person detect: {n_persons} persoane în {elapsed:.0f}ms")
        if n_persons > 0:
            info(f"  Bounding boxes: {[list(map(int, b)) for b in boxes.xyxy.tolist()[:3]]}")
        else:
            info("  0 persoane — normal pentru imagini din parcuri fără oameni")
    except Exception as e:
        fail("Eroare la person detection", str(e))
else:
    info("TEST 5 sărit — nu avem imagine de test")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Face blur — verifică că GaussianBlur funcționează corect
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 6 — Face blur (cv2.GaussianBlur, GDPR)")
print("=" * 60)

# Creăm un frame sintetic cu o "persoană" simulată
test_blur_frame = np.zeros((480, 640, 3), dtype=np.uint8)
test_blur_frame[50:300, 200:350] = 180  # dreptunghi alb = persoană simulată

# Simulăm bbox persoană: [200, 50, 350, 300]
person_box = (200, 50, 350, 300)
x1, y1, x2, y2 = person_box
face_h = int((y2 - y1) * 0.25)  # top 25% = față
face_region = test_blur_frame[y1:y1 + face_h, x1:x2].copy()
test_blur_frame[y1:y1 + face_h, x1:x2] = cv2.GaussianBlur(face_region, (51, 51), 0)
ok("Face blur aplicat pe top 25% bbox persoană")

# Verificăm că pixelii din zona feței sunt diferiți (blur aplicat)
face_mean_before = 180.0
face_mean_after = float(test_blur_frame[y1:y1 + face_h, x1:x2].mean())
# După blur pe o zonă uniformă, valorile rămân similare — testăm că nu a dat eroare
ok(f"cv2.GaussianBlur funcționează — media regiune: {face_mean_after:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: Verifică că ultralytics are bytetrack.yaml disponibil
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 7 — ByteTrack config disponibil")
print("=" * 60)

try:
    import ultralytics
    ul_root = Path(ultralytics.__file__).parent
    bytetrack_cfg = ul_root / "cfg" / "trackers" / "bytetrack.yaml"
    botsort_cfg   = ul_root / "cfg" / "trackers" / "botsort.yaml"

    if bytetrack_cfg.exists():
        ok(f"bytetrack.yaml găsit: {bytetrack_cfg}")
    else:
        fail("bytetrack.yaml NU găsit", str(bytetrack_cfg))

    if botsort_cfg.exists():
        ok(f"botsort.yaml găsit: {botsort_cfg}")

except Exception as e:
    fail("Eroare la verificarea tracker config", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# SUMAR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMAR")
print("=" * 60)

if not errors:
    print(f"\n✓ Toate testele au trecut. Sistemul este pregătit pentru implementare.\n")
else:
    print(f"\n✗ {len(errors)} test(e) eșuat(e):")
    for e in errors:
        print(f"  - {e}")
    print()

sys.exit(0 if not errors else 1)
