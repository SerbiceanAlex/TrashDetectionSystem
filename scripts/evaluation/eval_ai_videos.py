"""
Evalueaza videourile generate cu AI (datasets/ai_videos) prin acelasi pipeline
de productie ca evaluarea din teza. Pentru fiecare clip: cate incidente s-au
detectat, la ce secunda, si daca rezultatul corespunde etichetei asteptate.

Rulare:
    .venv\\Scripts\\python.exe scripts\\evaluation\\eval_ai_videos.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.config import settings
from scripts.evaluation.evaluate_video_events import run_clip, outcome

AI_DIR = REPO / "datasets" / "ai_videos"

# Etichete: toate sunt acte de aruncare (pozitive), exceptie obiectul preexistent.
EXPECTED = {
    "ai_preexisting_litter": "negative",  # obiect deja pe jos, fara act de aruncare
}


def main() -> None:
    device = "0" if torch.cuda.is_available() else "cpu"
    det = YOLO(str(settings.detector_path))
    person = YOLO(str(settings.person_detector_path))

    clips = sorted(AI_DIR.glob("*.mp4"))
    print(f"Evaluez {len(clips)} clipuri pe device={device}, "
          f"conf={settings.DEFAULT_DET_CONF}, imgsz={settings.MONITOR_TRASH_IMGSZ}\n")

    tp = fp = tn = fn = 0
    rows = []
    for clip in clips:
        if hasattr(det, "predictor"):
            det.predictor = None
        res = run_clip(
            clip, det, person,
            conf=settings.DEFAULT_DET_CONF, person_conf=0.40,
            imgsz=settings.MONITOR_TRASH_IMGSZ, frame_skip=2, max_frames=0, device=device,
        )
        exp = EXPECTED.get(clip.stem, "positive")
        ev = res["events_detected"]
        o = outcome(exp, ev)
        tp += o == "TP"; fp += o == "FP"; tn += o == "TN"; fn += o == "FN"
        first = res["first_event_time_sec"]
        rows.append((clip.stem, exp, ev, first, o, res["runtime_fps"]))
        mark = {"TP": "OK ", "TN": "OK ", "FP": "FALS+", "FN": "RATAT"}[o]
        print(f"[{mark}] {clip.stem:26s} astept={exp:8s} incidente={ev} "
              f"prima={first or '-':>6} ({res['runtime_fps']:.0f} FPS)")

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"\nTP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Precizie={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")


if __name__ == "__main__":
    main()
