# Trash Detection System

Sistem two-stage de detecție și clasificare a deșeurilor în spații verzi urbane (parcuri), implementat cu YOLOv8.

Lucrare de licență — Universitatea Politehnica București, 2026.

---

## Arhitectură

```
Imagine → [Stage 1: Detector YOLO] → bounding boxes (trash)
                                           ↓
                               [Stage 2: Clasificator YOLO]
                                           ↓
                               material: glass / metal / paper / plastic / other
```

**Stage 1 — Detecție** (`src/detect_two_stage.py`):
- Model: YOLOv8s, imgsz=640, clasă unică `trash`
- Dataset: `parks_detect_full` (adnotat manual din videoclipuri de parc)

**Stage 2 — Clasificare** (`src/detect_two_stage.py`):
- Model: YOLOv8n-cls, imgsz=224
- Dataset: `parks_cls` (TrashNet + crops din parcuri)

---

## Rezultate Finale

### Detector — Experiment A (detecție trash)

| Experiment | Model | imgsz | Precision | Recall | **mAP50** | mAP50-95 |
|------------|-------|-------|-----------|--------|-----------|----------|
| A22 (baseline) | YOLOv8n | 416 | 0.707 | 0.286 | 0.393 | 0.281 |
| **A3-final** ✅ | **YOLOv8s** | **640** | **0.623** | **0.406** | **0.443** | **0.321** |

### Clasificator — Experiment B (clasificare material)

| Experiment | Model | Dataset | **Acc Top-1 (val)** |
|------------|-------|---------|---------------------|
| **B2** ✅ | YOLOv8n-cls | TrashNet + parks crops | **95.2%** |

### Pipeline End-to-End — Experiment C

| Experiment | Detector | Clasif. | Imagini cu detecții | **Rată detecție** | Total detecții | Viteză |
|------------|----------|---------|---------------------|-------------------|----------------|--------|
| C1 (baseline) | A22 | B2 | 9 / 227 | 3.96% | 10 | 26.3 ms/img |
| **C2** ✅ | **A3-final** | **B2** | **219 / 227** | **96.5%** | **739** | 74.7 ms/img |

**Distribuție materiale detectate (C2, 739 detecții):**
- paper: 276 (37.3%)
- metal: 267 (36.1%)
- glass: 94 (12.7%)
- plastic: 77 (10.4%)
- other: 25 (3.4%)

---

## Notebook-uri

Toată pipeline-ul este documentat în notebook-uri Jupyter:

| Notebook | Scop |
|----------|------|
| `notebooks/training/01_train_detector.ipynb` | Antrenare detectori (Exp A) |
| `notebooks/training/02_train_classifier.ipynb` | Antrenare clasificatori (Exp B) |
| `notebooks/evaluation/03_inference_demo.ipynb` | Demo vizual two-stage pe test set |
| `notebooks/evaluation/04_pipeline_C1_C2.ipynb` | Pipeline end-to-end C1 vs C2 |
| `notebooks/evaluation/05_thesis_figures.ipynb` | Generare figuri și tabele pentru teză |

---

## Modele antrenate (cele mai bune)

| Model | Fișier | Rol |
|-------|--------|-----|
| Detector A3-final | `runs/detect/parks-trash-A3-final/weights/best.pt` | Stage 1 — detecție trash |
| Clasificator B2 | `runs/classify/parks-cls-B2/weights/best.pt` | Stage 2 — clasificare material |

---

## Setup
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Recommended implementation flow
1. Put raw videos in `raw/videos` or `datasets/raw/videos`.
2. Extract frames with `python scripts/extract_frames.py`.
3. Stage images for annotation with `python scripts/stage_images_for_annotation.py --source path\\to\\images --flatten` if needed.
4. Annotate frames in YOLO format under `datasets/parks_detect/images/all` and `datasets/parks_detect/labels/all` using one class: `trash`.
5. Check progress with `python scripts/report_annotation_progress.py`.
6. Create the split with `python scripts/split_yolo_detection_dataset.py --clear`.
7. Validate the dataset with `python scripts/validate_yolo_dataset.py`.
8. Inspect class balance with `python scripts/report_yolo_dataset_stats.py`.
9. Train the detector with `python scripts/train.py --model yolov8n.pt --val`.
10. Run inference on video with `python -m src.detect_video_yolo --source path\\to\\video.mp4 --model runs/detect/parks-trash-yolov8/weights/best.pt --show`.

One-command flow after annotation:
- `python scripts/run_detection_pipeline.py --skip-train`
- `python scripts/run_detection_pipeline.py --model yolov8n.pt --train-val`

Full research pipeline:
- `python scripts/run_full_pipeline.py --stop-after-crops`
- `python scripts/run_full_pipeline.py --skip-detection --skip-crops`

Alternative source path if you annotate in COCO format:
1. Export annotations as COCO JSON.
2. Convert them with `python scripts/coco_to_yolo_single_class.py --coco path\\to\\annotations.json --images-root path\\to\\images --out datasets/parks_detect`.
3. Validate with `python scripts/validate_yolo_dataset.py`.

See `ANNOTATION_GUIDE.md` for the exact labeling rules and annotation workflow.
See `annotation/ANNOTATION_QUICKSTART.md` for the practical setup in Label Studio or CVAT.
See `annotation/STAGE2_CHECKLIST.md` for the later crop-generation and classification workflow.
See `annotation/EVALUATION_PROTOCOL.md` for the reporting protocol.

To prepare a Label Studio import file from the current image pool:
- `python scripts/build_label_studio_tasks.py --images-dir datasets/parks_detect/images/all --out annotation/label_studio_tasks.json --mode local-files --local-files-base .`

To start Label Studio locally for this project:
- `./scripts/start_label_studio.ps1`

## Dataset layout for detection
```text
datasets/
	parks_detect/
		parks_detect.yaml
		images/
			train/
			val/
			test/
		labels/
			train/
			val/
			test/
```

Each image must have a matching `.txt` file with YOLO annotations:
```text
class_id x_center y_center width height
```

Recommended annotation source layout before split:
```text
datasets/
	parks_detect/
		images/
			all/
		labels/
			all/
```

Useful annotation helpers:
- `python scripts/stage_images_for_annotation.py --source path\\to\\images --flatten`
- `python scripts/report_annotation_progress.py`

Example class mapping from `datasets/parks_detect/parks_detect.yaml`:
```yaml
names:
	0: trash
```

## Training
Quick baseline:
```bash
python scripts/train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --val
```

Better accuracy baseline:
```bash
python scripts/train.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16 --val
```

Useful options:
- `--device cpu` or `--device 0`
- `--cache`
- `--patience 20`
- `--name custom-run-name`

## Validation
Check the dataset before training:
```bash
python scripts/validate_yolo_dataset.py
```

Inspect dataset balance before training:
```bash
python scripts/report_yolo_dataset_stats.py
```

At this stage, class balance means balance between positive images, negative images, and object counts for the single `trash` class.

Evaluate the trained detector on the held-out split with:

```bash
python scripts/evaluate_detector.py --model runs/detect/parks-trash-yolov8/weights/best.pt --data datasets/parks_detect/parks_detect.yaml --split test
```

This writes a metrics JSON under `runs/detect_eval/`.

## Split preparation
Create train, val and test from the annotated `all` folders:
```bash
python scripts/split_yolo_detection_dataset.py --clear
```

Run the full post-annotation pipeline in one command:
```bash
python scripts/run_detection_pipeline.py --model yolov8n.pt --train-val
```

Useful options:
- `--group-by prefix` keeps frames from the same video sequence in the same split
- `--group-by image` splits each image independently
- `--allow-empty-labels` includes unlabeled images as negative examples
- `--copy` copies files instead of creating hardlinks

Create empty labels only for images that intentionally contain no trash:
```bash
python scripts/create_empty_yolo_labels.py --images-dir datasets/parks_detect/images/all --labels-dir datasets/parks_detect/labels/all --only-missing
```

## Inference
Run webcam or video inference with the trained detector:
```bash
python -m src.detect_video_yolo --source 0 --model runs/detect/parks-trash-yolov8/weights/best.pt --show
# or
python -m src.detect_video_yolo --source path\to\video.mp4 --model runs/detect/parks-trash-yolov8/weights/best.pt --show --save
```

Run the full two-stage pipeline after both models are trained:

```bash
python -m src.detect_two_stage --source path\to\video.mp4 --detector runs/detect/parks-trash-yolov8/weights/best.pt --classifier runs/classify/parks-trash-material-cls/weights/best.pt --show --save
```

The same script also works on a single image:

```bash
python -m src.detect_two_stage --source path\to\image.jpg --detector runs/detect/parks-trash-yolov8/weights/best.pt --classifier runs/classify/parks-trash-material-cls/weights/best.pt --show --save
```

Run the full two-stage pipeline on a directory of test images and export a CSV of detections:

```bash
python scripts/run_two_stage_batch.py --source-dir path\to\images --detector runs/detect/parks-trash-yolov8/weights/best.pt --classifier runs/classify/parks-trash-material-cls/weights/best.pt --save-images
```

Notes:
- `--source 0` uses the default webcam.
- Press `q` to quit the preview window.
- The detector only becomes useful after the dataset is annotated and trained.
- The video inference overlay also shows FPS and per-frame counts for detected trash instances.

## Classification dataset
Expected folder layout for stage-2 material classification:

```text
datasets/
	parks_cls/
		train/
			glass/
			metal/
			other/
			paper/
			plastic/
		val/
			glass/
			metal/
			other/
			paper/
			plastic/
		test/
			glass/
			metal/
			other/
			paper/
			plastic/
```

Report classifier dataset balance with:

```bash
python scripts/report_classification_dataset_stats.py
```

If you first sort labeled crops into a pool like `datasets/parks_cls_pool/<class>/`, create train/val/test with:

```bash
python scripts/split_classification_dataset.py --source-root datasets/parks_cls_pool --out-root datasets/parks_cls --clear
```

Train the material classifier with:

```bash
python scripts/train_classifier.py --model yolov8n-cls.pt --data datasets/parks_cls --val-split val
```

The classifier should be trained only after the crop dataset has been reviewed and organized by class.

Run the full stage-2 pipeline in one command:

```bash
python scripts/run_classification_pipeline.py --data datasets/parks_cls --model yolov8n-cls.pt --train-val --eval-split test
```

Evaluate the classifier on the held-out split with:

```bash
python scripts/evaluate_classifier.py --model runs/classify/parks-trash-material-cls/weights/best.pt --data datasets/parks_cls --split test
```

This writes a prediction CSV, a confusion matrix CSV, and a summary JSON under `runs/classify_eval/`.