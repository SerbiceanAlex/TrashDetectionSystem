"""Train Experiment A4 — parks + TACO dataset on YOLOv8s."""
from ultralytics import YOLO
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]

model = YOLO(str(REPO / "notebooks/training/yolov8s.pt"))
results = model.train(
    data=str(REPO / "datasets/parks_detect_A4/dataset.yaml"),
    epochs=50,
    imgsz=640,
    batch=8,
    patience=15,
    project=str(REPO / "runs/detect"),
    name="parks-trash-A4",
    device=0,
    workers=0,
    amp=False,
    cache=False,
    exist_ok=True,
)

rd = results.results_dict
print("\n=== A4 TRAINING RESULTS ===")
print(f"mAP50:     {rd.get('metrics/mAP50(B)', 'N/A')}")
print(f"mAP50-95:  {rd.get('metrics/mAP50-95(B)', 'N/A')}")
print(f"Precision: {rd.get('metrics/precision(B)', 'N/A')}")
print(f"Recall:    {rd.get('metrics/recall(B)', 'N/A')}")
print("===========================")
sys.exit(0)
