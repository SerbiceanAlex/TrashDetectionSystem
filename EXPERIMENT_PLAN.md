# Experiment Plan — Trash Detection System (Licență)

## 1. Data Budget

### Stage 1 — Detection
| Set       | Target images | Purpose                     |
|-----------|--------------|------------------------------|
| Train     | 150–200      | Model fitting                |
| Val       |  30–50       | Epoch selection, early stop  |
| Test      |  30–50       | Final reported metrics       |
| **Total** | **200–300**  |                              |

Current pool: **48 images**, 0 annotated.

Action plan:
1. Annotate the 48 existing images first (quick baseline).
2. Record more park videos or download additional park scenes.
3. Run `extract_frames.py` → `stage_images_for_annotation.py` to grow the pool.
4. Target ≥ 200 annotated images before final experiments.

### Stage 2 — Classification
| Set       | Target crops / class | Notes                          |
|-----------|---------------------|--------------------------------|
| Train     | 80–150              | Augmented by Ultralytics       |
| Val       | 15–30               | Per class                      |
| Test      | 15–30               | Per class                      |

Sources:
- Crops from park detector labels (`export_yolo_crops.py`).
- TrashNet public dataset (≈2500 real images, 6 classes — map cardboard → paper, trash → other).
- Manual web images if a class is underrepresented.

---

## 2. Experiments

### Experiment A — Detection Baseline
| ID  | Model     | imgsz | epochs | Data              | Goal                        |
|-----|-----------|-------|--------|-------------------|-----------------------------|
| A1  | yolov8n   | 640   | 100    | 48-image quick set | Sanity check, pipeline test |
| A2  | yolov8n   | 640   | 150    | Full dataset       | Main baseline               |
| A3  | yolov8s   | 640   | 150    | Full dataset       | Bigger model comparison     |

Report: precision, recall, mAP@50, mAP@50-95, inference speed (ms/img).

### Experiment B — Classification Baseline
| ID  | Model         | imgsz | epochs | Data           | Goal                      |
|-----|--------------|-------|--------|----------------|---------------------------|
| B1  | yolov8n-cls  | 224   | 50     | Park crops only | Domain-specific baseline  |
| B2  | yolov8n-cls  | 224   | 50     | TrashNet only   | Public data baseline      |
| B3  | yolov8n-cls  | 224   | 80     | Mixed (park + TrashNet) | Best expected result |

Report: accuracy, macro-precision, macro-recall, macro-F1, confusion matrix.

### Experiment C — End-to-End
| ID  | Detector | Classifier | Purpose                                      |
|-----|----------|------------|----------------------------------------------|
| C1  | A2 best  | B3 best    | Full pipeline on test images                 |
| C2  | A3 best  | B3 best    | Does a larger detector improve end-to-end?   |

Report: per-image qualitative results, total correct/incorrect, failure categories.

### Experiment D — Ablations (optional, strengthens thesis)
| ID  | Variable             | Values tested        | Purpose                         |
|-----|----------------------|----------------------|---------------------------------|
| D1  | Detection imgsz      | 416, 640, 800       | Resolution vs. mAP trade-off   |
| D2  | Classifier data mix  | 0%, 50%, 100% park  | Domain gap analysis             |
| D3  | Confidence threshold | 0.15, 0.25, 0.40    | Precision–recall sweet spot     |

---

## 3. Thesis Deliverables

### Tables
| #  | Content                                        | Source experiment |
|----|------------------------------------------------|-------------------|
| T1 | Detection metrics (P, R, mAP50, mAP50-95)     | A2, A3            |
| T2 | Classification metrics (Acc, P, R, F1)         | B1, B2, B3        |
| T3 | End-to-end summary                             | C1, C2            |
| T4 | Ablation results (if done)                     | D1–D3             |

### Figures
| #  | Content                                        |
|----|------------------------------------------------|
| F1 | System architecture diagram (two-stage flow)   |
| F2 | Training curves (loss, mAP over epochs)        |
| F3 | Confusion matrix for classifier (B3)           |
| F4 | Sample predictions — correct detections        |
| F5 | Sample predictions — failure cases             |
| F6 | Dataset class distribution bar chart            |

### Discussion topics
- Why two-stage over single-stage multiclass detection?
- Public data (TrashNet) vs. domain-specific data — gap analysis.
- Small dataset strategies: transfer learning, augmentation.
- Failure modes: occlusion, small objects, similar materials.
- Practical deployment considerations (speed, edge devices).

---

## 4. Execution Sequence

```text
Phase 0  ──  Quick sanity (now)
  ├─ Annotate 10–15 images in Label Studio / CVAT
  ├─ Run A1 (48-image quick baseline, even partial labels)
  └─ Verify entire pipeline works end-to-end

Phase 1  ──  Data collection & annotation
  ├─ Grow image pool to ≥ 200
  ├─ Complete all bounding-box annotations
  ├─ Run split + validate + stats
  └─ Checkpoint: detection dataset locked

Phase 2  ──  Detection experiments
  ├─ Train A2, A3
  ├─ Evaluate on test split
  ├─ Export crops with best detector
  └─ Checkpoint: crops ready for classification

Phase 3  ──  Classification preparation
  ├─ Sort crops into material folders
  ├─ (Optional) Add TrashNet data
  ├─ Split classification dataset
  └─ Checkpoint: classification dataset locked

Phase 4  ──  Classification experiments
  ├─ Train B1, B2, B3
  ├─ Evaluate on test split
  └─ Checkpoint: classifier selected

Phase 5  ──  End-to-end & write-up
  ├─ Run C1, C2
  ├─ Collect failure examples
  ├─ (Optional) Run ablations D1–D3
  └─ Fill thesis tables and figures
```

---

## 5. Quick-start Command Reference

### Phase 0 — annotate & sanity-check

```powershell
# Start Label Studio for annotation
.\scripts\start_label_studio.ps1

# Check annotation progress
.venv\Scripts\python.exe scripts\report_annotation_progress.py

# Quick split + validate (no training)
.venv\Scripts\python.exe scripts\run_detection_pipeline.py --skip-train

# Quick training (sanity check)
.venv\Scripts\python.exe scripts\train.py --model yolov8n.pt --epochs 30 --imgsz 640 --batch 8
```

### Phase 2 — detection experiments

```powershell
# Full baseline A2
.venv\Scripts\python.exe scripts\train.py --model yolov8n.pt --epochs 150 --imgsz 640 --batch 16 --name parks-trash-A2

# Bigger model A3
.venv\Scripts\python.exe scripts\train.py --model yolov8s.pt --epochs 150 --imgsz 640 --batch 16 --name parks-trash-A3

# Evaluate
.venv\Scripts\python.exe scripts\evaluate_detector.py --model runs\detect\parks-trash-A2\weights\best.pt --data datasets\parks_detect\parks_detect.yaml --split test
```

### Phase 3 — crop export & sort

```powershell
# Export crops from best detector labels
.venv\Scripts\python.exe scripts\export_yolo_crops.py --images-dir datasets\parks_detect\images\all --labels-dir datasets\parks_detect\labels\all --out-dir datasets\parks_cls_pool\unsorted

# After manual sorting into class folders:
.venv\Scripts\python.exe scripts\split_classification_dataset.py --source-root datasets\parks_cls_pool --out-root datasets\parks_cls --clear
```

### Phase 4 — classification experiments

```powershell
# B1 park crops only
.venv\Scripts\python.exe scripts\train_classifier.py --data datasets\parks_cls --epochs 50 --imgsz 224 --name parks-cls-B1

# Evaluate
.venv\Scripts\python.exe scripts\evaluate_classifier.py --model runs\classify\parks-cls-B1\weights\best.pt --data datasets\parks_cls --split test
```

### Phase 5 — end-to-end

```powershell
.venv\Scripts\python.exe scripts\run_two_stage_batch.py --source-dir datasets\parks_detect\images\test --detector runs\detect\parks-trash-A2\weights\best.pt --classifier runs\classify\parks-cls-B3\weights\best.pt --save-images
```
