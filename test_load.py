"""Minimal test: just load models."""
import sys
print("start", flush=True)
from ultralytics import YOLO
print("ultralytics imported", flush=True)
m = YOLO("runs/detect/parks-trash-A3-final/weights/best.pt")
print("detector loaded", flush=True)
