"""
Extrage decupaje (crops) de obiecte din clipurile filmate de utilizator,
folosind detectorul de deșeuri. Fiecare clip = un material (din numele
fișierului), deci etichetarea e automată.

Rezultat: datasets/real_material/{train,test}/<clasa>/  (split 80/20)
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import random
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from backend.config import settings  # noqa: E402

DOWNLOADS = Path(r"C:\Users\sandu\Downloads")
CLIPS = {  # fisier -> clasa
    "sticla.MOV":  "glass",
    "plastic.MOV": "plastic",
    "hartie.MOV":  "paper",
    "metal.MOV":   "metal",
}
DST = REPO / "datasets" / "real_material"
EVERY_N = 4          # ia un cadru din N (clip ~25s @30fps -> ~180 cadre procesate)
TEST_FRAC = 0.20
random.seed(42)


def main():
    import shutil
    if DST.exists():
        shutil.rmtree(DST)
    det = YOLO(str(settings.detector_path))

    for fname, cls in CLIPS.items():
        p = DOWNLOADS / fname
        if not p.exists():
            print(f"  ! lipsă: {p}")
            continue
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        crops = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % EVERY_N == 0:
                r = det.predict(frame, conf=0.25, imgsz=896, verbose=False)[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    h, w = frame.shape[:2]
                    confs = r.boxes.conf.tolist()
                    boxes = r.boxes.xyxy.tolist()
                    # pastreaza DOAR detectia cu cea mai mare incredere care e in
                    # treimea superioara/medie a cadrului (obiectul TINUT in mana),
                    # nu zona de jos (masa, caietul) -> elimina etichetarea gresita
                    best = None; best_c = -1.0
                    for (x1, y1, x2, y2), cf in zip(boxes, confs):
                        cy = (y1 + y2) / 2.0
                        if cy > 0.60 * h:   # zona de jos (masa) -> ignora
                            continue
                        if cf > best_c:
                            best_c = cf; best = (int(x1), int(y1), int(x2), int(y2))
                    if best is not None:
                        x1, y1, x2, y2 = best
                        px = int((x2 - x1) * 0.2); py = int((y2 - y1) * 0.2)
                        cx1 = max(0, x1 - px); cy1 = max(0, y1 - py)
                        cx2 = min(w, x2 + px); cy2 = min(h, y2 + py)
                        c = frame[cy1:cy2, cx1:cx2]
                        if c.size > 0 and (cx2 - cx1) > 40 and (cy2 - cy1) > 40:
                            crops.append(c)
            idx += 1
        cap.release()

        random.shuffle(crops)
        n_test = max(8, int(len(crops) * TEST_FRAC))
        test, train = crops[:n_test], crops[n_test:]
        for split, items in (("train", train), ("test", test)):
            d = DST / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i, c in enumerate(items):
                cv2.imwrite(str(d / f"{cls}_{i:04d}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  {cls:<8} crop-uri: {len(crops):<4} -> train {len(train)}, test {len(test)}")

    print(f"\nGata: {DST}")


if __name__ == "__main__":
    main()
