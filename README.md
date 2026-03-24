# Trash Detection System — Sistem Two-Stage de Detecție a Deșeurilor

Sistem two-stage de detecție și clasificare a deșeurilor în spații verzi urbane (parcuri), implementat cu YOLOv8.

**Lucrare de licență — Universitatea Politehnica București, 2026.**

---

## Arhitectură

```
Imagine/Video → [Stage 1: Detector YOLO] → bounding boxes (clasa: trash)
                                                    ↓
                                    [Stage 2: Clasificator YOLO]
                                                    ↓
                            material: glass / metal / paper / plastic / other
```

- **Stage 1** (`src/detect_two_stage.py`): YOLOv8s, imgsz=640, clasă unică `trash`, antrenat pe dataset Parks adnotat manual din videoclipuri de parc
- **Stage 2** (`src/detect_two_stage.py`): YOLOv8n-cls, imgsz=224, antrenat pe TrashNet + crops extrase din dataset-ul de parcuri

---

## Rezultate Finale

### Detector — Experiment A

| Experiment | Model | imgsz | Precision | Recall | **mAP50** | mAP50-95 |
|------------|-------|-------|-----------|--------|-----------|----------|
| A22 (baseline) | YOLOv8n | 416 | 0.707 | 0.286 | 0.393 | 0.281 |
| **A3-final** ✅ best | **YOLOv8s** | **640** | **0.623** | **0.406** | **0.443** | **0.321** |

### Clasificator — Experiment B

| Experiment | Model | Dataset | **Acc Top-1 (val)** |
|------------|-------|---------|---------------------|
| **B2** ✅ best | YOLOv8n-cls | TrashNet + parks crops | **95.2%** |

### Pipeline End-to-End — Experiment C

| Experiment | Detector | Clasificator | Imagini cu detecții | **Rată detecție** | Total detecții | Viteză |
|------------|----------|--------------|---------------------|-------------------|----------------|--------|
| C1 (baseline) | A22 | B2 | 9 / 225 | 4.0% | 10 | 21.4 ms/img |
| **C2** ✅ best | **A3-final** | **B2** | **219 / 225** | **97.3%** | **739** | 60.7 ms/img |

**Distribuție materiale detectate (C2, 739 detecții):**

| Material | Nr. detecții | Procent |
|----------|-------------|---------|
| paper    | 276 | 37.3% |
| metal    | 267 | 36.1% |
| glass    | 94  | 12.7% |
| plastic  | 77  | 10.4% |
| other    | 25  | 3.4%  |

---

## Modele antrenate

| Model | Cale | Rol |
|-------|------|-----|
| **Detector A3-final** | `runs/detect/parks-trash-A3-final/weights/best.pt` | Stage 1 — detectează obiecte trash |
| **Clasificator B2** | `runs/classify/parks-cls-B2/weights/best.pt` | Stage 2 — clasifică materialul |

---

## Notebook-uri

Întreaga pipeline este documentată și reproductibilă prin notebook-uri Jupyter:

| Notebook | Scop |
|----------|------|
| `notebooks/data/01_data_preparation.ipynb` | Pregătire dataset detecție (split train/val/test) |
| `notebooks/data/02_classification_data.ipynb` | Pregătire dataset clasificare (crops + split) |
| `notebooks/data/03_annotate_parks_crops.ipynb` | Export crops pentru adnotare |
| `notebooks/training/01_train_detector.ipynb` | Antrenare detectori — Experiment A |
| `notebooks/training/02_train_classifier.ipynb` | Antrenare clasificatori — Experiment B |
| `notebooks/evaluation/03_inference_demo.ipynb` | Demo vizual two-stage pe test set (227 imagini) |
| `notebooks/evaluation/04_pipeline_C1_C2.ipynb` | Pipeline end-to-end C1 vs C2 |
| `notebooks/evaluation/05_thesis_figures.ipynb` | Generare figuri și tabele pentru teză |

---

## Structura proiectului

```
TrashDetectionSystem/
├── src/
│   ├── detect_two_stage.py      # Pipeline two-stage + funcții draw_detections
│   └── detect_video_yolo.py     # Inferență video baseline (un singur detector)
├── scripts/
│   ├── train.py                 # Antrenare detector
│   ├── train_classifier.py      # Antrenare clasificator
│   ├── evaluate_detector.py     # Evaluare detector pe test set
│   ├── evaluate_classifier.py   # Evaluare clasificator pe test set
│   ├── export_yolo_crops.py     # Export crops din detecții pentru clasificare
│   ├── split_yolo_detection_dataset.py   # Split all→train/val/test (detecție)
│   ├── split_classification_dataset.py   # Split all→train/val/test (clasificare)
│   ├── merge_classification_datasets.py  # Merge TrashNet + parks crops
│   ├── run_two_stage_batch.py   # Rulare pipeline two-stage pe un folder
│   ├── validate_yolo_dataset.py # Validare format dataset YOLO
│   ├── report_yolo_dataset_stats.py         # Statistici dataset detecție
│   └── report_classification_dataset_stats.py # Statistici dataset clasificare
├── notebooks/
│   ├── data/                    # Pregătire date
│   ├── training/                # Antrenare modele + greutăți de bază .pt
│   └── evaluation/              # Evaluare, demo, figuri teză
├── datasets/
│   ├── parks_detect_full/       # Dataset detecție (train/val/test, adnotat manual)
│   ├── parks_cls/               # Dataset clasificare (train/val/test)
│   ├── mixed_cls/               # Dataset clasificare extins (TrashNet + parks)
│   └── trashnet_cls/            # TrashNet original
├── results/
│   ├── detector/                # JSON metrici A22, A3-final
│   └── pipeline/                # JSON/CSV sumare C1, C2
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Utilizare

### Inferență pe video/webcam

```bash
# Detector simplu (un singur model)
python -m src.detect_video_yolo --source 0 --model runs/detect/parks-trash-A3-final/weights/best.pt --show

# Pipeline two-stage complet (detector + clasificator material)
python -m src.detect_two_stage --source path/to/video.mp4 \
    --detector runs/detect/parks-trash-A3-final/weights/best.pt \
    --classifier runs/classify/parks-cls-B2/weights/best.pt \
    --show --save
```

### Rulare two-stage pe un folder de imagini

```bash
python scripts/run_two_stage_batch.py \
    --source-dir datasets/parks_detect_full/images/test \
    --detector runs/detect/parks-trash-A3-final/weights/best.pt \
    --classifier runs/classify/parks-cls-B2/weights/best.pt \
    --save-images \
    --out-dir outputs/demo
```

### Antrenare modele

```bash
# Detector (Stage 1)
python scripts/train.py --model notebooks/training/yolov8s.pt \
    --data datasets/parks_detect_full/dataset.yaml \
    --epochs 100 --imgsz 640 --batch 16 --device 0

# Clasificator (Stage 2)
python scripts/train_classifier.py --model notebooks/training/yolov8n-cls.pt \
    --data datasets/parks_cls \
    --epochs 50 --imgsz 224 --batch 32 --device 0
```

### Evaluare

```bash
# Evaluare detector
python scripts/evaluate_detector.py \
    --model runs/detect/parks-trash-A3-final/weights/best.pt \
    --data datasets/parks_detect_full/dataset.yaml \
    --split test

# Evaluare clasificator
python scripts/evaluate_classifier.py \
    --model runs/classify/parks-cls-B2/weights/best.pt \
    --data datasets/parks_cls \
    --split test
```

### Pregătire dataset

```bash
# Split dataset detecție (din all/ → train/val/test)
python scripts/split_yolo_detection_dataset.py --clear

# Split dataset clasificare
python scripts/split_classification_dataset.py \
    --source-root datasets/parks_cls_unsorted \
    --out-root datasets/parks_cls --clear

# Validare format YOLO
python scripts/validate_yolo_dataset.py

# Statistici dataset
python scripts/report_yolo_dataset_stats.py
python scripts/report_classification_dataset_stats.py
```