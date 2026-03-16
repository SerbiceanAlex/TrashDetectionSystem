# TrashDetectionSystem

A university thesis project implementing a **two-stage waste detection and
material classification** system for park and green urban-area scenes.

---

## Architecture Overview

```
Video / Image
      │
      ▼
┌─────────────────────┐
│  Stage 1: Detection │   YOLOv8  →  1 class ("trash")
│  (locate all waste) │   Outputs bounding boxes
└─────────────────────┘
      │  bounding-box crops
      ▼
┌──────────────────────────┐
│ Stage 2: Classification  │   EfficientNetV2-S  →  5 classes
│ (identify material type) │   plastic / glass / metal / paper / other
└──────────────────────────┘
      │  labelled detections
      ▼
  Annotated output video / image
```

The two-stage design lets us train the detector on large unlabelled-material
datasets (TACO, UAVVaste) while training the classifier on a smaller, manually
relabelled crop dataset.

---

## Domain

Parks, green areas, and urban outdoor scenes captured with handheld cameras
and UAVs.  The system is designed to be robust to:
- Varying lighting conditions (shadows, direct sunlight)
- Partially occluded or crumpled waste
- Small objects at distances typical of park surveillance

---

## Repository Structure

```
TrashDetectionSystem/
├── datasets/
│   ├── raw/
│   │   ├── images/         ← standalone labelled images
│   │   ├── frames/         ← frames extracted from videos
│   │   │   └── <video_id>/
│   │   └── metadata/       ← data documentation (committed to git)
│   ├── detection/          ← YOLO-format dataset (images + labels)
│   │   ├── images/{train,val,test}/
│   │   └── labels/{train,val,test}/
│   ├── processed/
│   │   └── classification/
│   │       └── crops/      ← material-labelled crops
│   │           ├── plastic/
│   │           ├── glass/
│   │           ├── metal/
│   │           ├── paper/
│   │           └── other/
│   └── splits/
│       └── detection_split.json
├── models/
│   └── classification/     ← saved classifier checkpoints
├── notebooks/
│   ├── 01_data_overview_and_splits.ipynb
│   ├── 02_detection_training_yolov8.ipynb
│   ├── 03_crop_generation_for_classification.ipynb
│   ├── 04_classification_training.ipynb
│   └── 05_inference_on_videos.ipynb
├── scripts/
│   ├── extract_frames.py
│   ├── train.py
│   ├── merge_yolo_datasets.py
│   └── coco_to_yolo_single_class.py
├── src/
│   ├── detect_video_yolo.py
│   └── video_preview.py
├── trash_detection/        ← shared Python package
│   ├── __init__.py
│   ├── io.py
│   ├── splits.py
│   ├── yolo.py
│   ├── crops.py
│   └── viz.py
├── trash_yolo.yaml         ← YOLO dataset config
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- (Optional) CUDA-capable GPU for training

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Additional packages for classification training:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn seaborn pandas
```

---

## Quick Start

### 1 · Extract frames from videos

```bash
python scripts/extract_frames.py \
    --videos-dir data/videos \
    --frames-dir datasets/raw/frames \
    --every-seconds 1.0 \
    --max-frames 300
```

### 2 · Label images with LabelImg

```bash
pip install labelImg
labelImg datasets/detection/images/train
```

Select **YOLO** format, use `trash` as the only label.

### 3 · Generate the dataset YAML

```python
from trash_detection.yolo import write_dataset_yaml
write_dataset_yaml("trash_yolo.yaml", "datasets/detection", nc=1, names=["trash"])
```

### 4 · Train the detector

```bash
yolo detect train \
    model=yolov8n.pt \
    data=trash_yolo.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16
```

### 5 · Run inference on a video

```bash
python src/detect_video_yolo.py \
    --source data/videos/sample.mp4 \
    --model runs/detect/trash_detector_v1/weights/best.pt \
    --save
```

---

## Pipeline Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_overview_and_splits.ipynb` | EDA, frame counts, group-aware train/val/test split |
| `02_detection_training_yolov8.ipynb` | Labelling workflow, YAML config, YOLOv8 training & evaluation |
| `03_crop_generation_for_classification.ipynb` | Crop extraction from GT boxes, manifest CSV |
| `04_classification_training.ipynb` | EfficientNetV2-S training with class weights, confusion matrix |
| `05_inference_on_videos.ipynb` | End-to-end two-stage inference, annotated video export |

---

## CLI Tools

| Script | Purpose |
|--------|---------|
| `scripts/extract_frames.py` | Extract frames from videos at a fixed time interval |
| `scripts/train.py` | Wrapper for YOLO training |
| `scripts/merge_yolo_datasets.py` | Merge multiple YOLO datasets into one |
| `scripts/coco_to_yolo_single_class.py` | Convert COCO annotations to YOLO single-class format |
| `src/detect_video_yolo.py` | Run YOLOv8 detection on a live camera or video file |
| `src/video_preview.py` | Preview a video file |

---

## Evaluation Protocol

### Detection (Stage 1)

| Metric | Tool |
|--------|------|
| mAP@50 | `yolo val` |
| mAP@50-95 | `yolo val` |
| Precision / Recall | `yolo val` |
| Inference FPS | `src/detect_video_yolo.py` |

### Classification (Stage 2)

| Metric | Notebook |
|--------|----------|
| Accuracy | `04_classification_training.ipynb` |
| Macro-F1 | `04_classification_training.ipynb` |
| Per-class F1 | `04_classification_training.ipynb` |
| Confusion matrix | `04_classification_training.ipynb` |

### Leakage prevention

All train/val/test splits are made **by video ID**, never by individual frame.
See `trash_detection/splits.py` and `datasets/raw/metadata/DATA_SOURCES.md`.

---

## License

Academic use only.  See thesis documentation for dataset licensing details.
