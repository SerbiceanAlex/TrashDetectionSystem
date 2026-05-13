"""
Retrain Experiment A3-final — parks_detect_full, YOLOv8s, 150 epochs.

Reproduces the original A3-final training for weight availability
and fair comparison with A4 (parks+TACO, 50 epochs).

Original config confirmed from results/detector/A3-final-test.json:
  model:  yolov8s
  imgsz:  640
  amp:    False
  mAP50:  0.443  Precision: 0.623  Recall: 0.406

Run:
    .venv/Scripts/python.exe scripts/training/train_a3_retrain.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from ultralytics import YOLO

model = YOLO(str(REPO / "notebooks/training/yolov8s.pt"))

results = model.train(
    data        = str(REPO / "datasets/parks_detect_A3_resized/dataset.yaml"),
    epochs      = 150,
    imgsz       = 640,
    batch       = 8,
    patience    = 30,
    project     = str(REPO / "runs/detect"),
    name        = "parks-trash-A3-retrain",
    device      = 0,
    workers     = 0,
    amp         = False,
    cache       = False,
    seed        = 42,
    exist_ok    = True,
)

rd = results.results_dict
print("\n=== A3-RETRAIN RESULTS ===")
print(f"mAP50:     {rd.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"mAP50-95:  {rd.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
print(f"Precision: {rd.get('metrics/precision(B)', 'N/A'):.4f}")
print(f"Recall:    {rd.get('metrics/recall(B)', 'N/A'):.4f}")
print("Original A3-final: mAP50=0.4432  P=0.6233  R=0.4059")
print("==========================")
sys.exit(0)
