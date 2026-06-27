"""
Clasificator MIXT (bun pe AMBELE): crop-uri reale proprii (domeniul tău) +
TOT TrashNet (multe obiecte variate) + augmentare de distanță (downscale/blur).

Scop: să NU uite obiectele generale (TrashNet) DAR să meargă și pe obiectele tale
de cameră. ~50% TrashNet + ~50% crop-uri proprii pe fiecare clasă.

Test: TrashNet test (general) ȘI crop-uri reale held-out (domeniu), clean + departe.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "datasets" / "real_material"
TRASHNET = REPO / "datasets" / "trashnet_cls"
DST = REPO / "datasets" / "material_mixed"
PER_SOURCE = 220       # ~câte pe clasă din FIECARE sursă (TrashNet + real) → ~440/clasă
SEED = 42
random.seed(SEED); np.random.seed(SEED)
CLASSES = ["glass", "plastic", "paper", "metal"]


def _motion_blur(img, k):
    ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
    if random.random() < 0.5: ker = ker.T
    return cv2.filter2D(img, -1, ker)


def degrade_distance(img):
    """Simulează obiect mic/la distanță: downscale agresiv + blur + artefacte."""
    h, w = img.shape[:2]
    f = random.uniform(0.12, 0.45)
    s = cv2.resize(img, (max(6, int(w * f)), max(6, int(h * f))), interpolation=cv2.INTER_AREA)
    img = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    r = random.random()
    if r < 0.5: img = cv2.GaussianBlur(img, (random.choice([3, 5]),) * 2, 0)
    elif r < 0.65: img = _motion_blur(img, random.choice([5, 7]))
    if random.random() < 0.7:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=random.uniform(-20, 20))
    if random.random() < 0.6:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(35, 70)])
        if ok: img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def _imgs(d: Path):
    return [f for f in sorted(d.glob("*")) if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]


def _fill(out: Path, files, n, degrade_frac):
    """Scrie n imagini din `files` (oversample cu repetiție), cu degradare pe o fracțiune."""
    out.mkdir(parents=True, exist_ok=True)
    if not files:
        return 0
    i = 0
    # întâi originalele (până la n), apoi oversample
    pool = list(files)
    while i < n:
        f = pool[i] if i < len(pool) else random.choice(files)
        img = cv2.imread(str(f))
        if img is None:
            i += 1; continue
        if random.random() < degrade_frac:
            img = degrade_distance(img)
        cv2.imwrite(str(out / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92]); i += 1
    return i


def build():
    if DST.exists():
        shutil.rmtree(DST)
    # TRAIN: pe fiecare clasă reală — TrashNet (PER_SOURCE) + real (PER_SOURCE)
    for cls in CLASSES:
        tn = _imgs(TRASHNET / "train" / cls)
        rl = _imgs(REAL / "train" / cls)
        out = DST / "train" / cls; out.mkdir(parents=True, exist_ok=True)
        # TrashNet: ~30% degradate (robustețe distanță generală)
        _fill(out, random.sample(tn, min(len(tn), PER_SOURCE)) if len(tn) >= PER_SOURCE else tn, PER_SOURCE, 0.30)
        # Real: ~55% degradate (domeniul tău, și aproape și departe)
        out2 = DST / "train" / cls
        # adăugăm real peste, cu prefix diferit
        if rl:
            i = len(list(out2.glob("*")))
            j = 0
            while j < PER_SOURCE:
                f = rl[j] if j < len(rl) else random.choice(rl)
                img = cv2.imread(str(f))
                if img is None: j += 1; continue
                if random.random() < 0.55: img = degrade_distance(img)
                cv2.imwrite(str(out2 / f"r{i+j:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92]); j += 1
    # other: doar TrashNet
    tn = _imgs(TRASHNET / "train" / "other")
    _fill(DST / "train" / "other", tn, PER_SOURCE, 0.30)

    # TEST/VAL: TrashNet test (general) + real held-out (clean + far)
    for split in ("test", "val"):
        for cls in CLASSES:
            out = DST / split / cls; out.mkdir(parents=True, exist_ok=True)
            k = 0
            for f in _imgs(TRASHNET / "test" / cls):           # general (clean)
                img = cv2.imread(str(f))
                if img is not None: cv2.imwrite(str(out / f"tn_{k:05d}.jpg"), img); k += 1
            for f in _imgs(REAL / "test" / cls):               # domeniu (clean + departe)
                img = cv2.imread(str(f))
                if img is None: continue
                cv2.imwrite(str(out / f"rl_{k:05d}.jpg"), img); k += 1
                cv2.imwrite(str(out / f"rl_{k:05d}_far.jpg"), degrade_distance(img)); k += 1
        out = DST / split / "other"; out.mkdir(parents=True, exist_ok=True)
        k = 0
        for f in _imgs(TRASHNET / "test" / "other"):
            img = cv2.imread(str(f))
            if img is not None: cv2.imwrite(str(out / f"tn_{k:05d}.jpg"), img); k += 1

    print("Set MIXT construit:")
    for split in ("train", "test"):
        for d in sorted((DST / split).iterdir()):
            print(f"  {split}/{d.name}: {len(list(d.glob('*')))}")


def main():
    build()
    # Pornire curată (ImageNet), nu de la modelul supra-învățat. Setul mixt
    # (TrashNet + real) îl face bun pe AMBELE.
    m = YOLO("models/pretrained/yolov8n-cls.pt")
    m.train(
        data=str(DST), epochs=90, imgsz=224, batch=32, device=0,
        project=str(REPO / "runs" / "classify"), name="material-mixed-full",
        seed=SEED, patience=25, verbose=True, workers=0, lr0=0.001,
        degrees=12.0, translate=0.1, scale=0.5, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4, auto_augment="randaugment",
    )
    print("\n=== TEST combinat (TrashNet + real, clean + departe) ===")
    mt = m.val(split="test")
    print("top1:", getattr(mt, "top1", "na"))


if __name__ == "__main__":
    main()
