# Stage 1 Checklist

## Objective
Build the first usable single-class detection baseline for park litter images.

Current classes:
- trash

## Current status
- Staged images in `datasets/parks_detect/images/all`: 48
- Label files in `datasets/parks_detect/labels/all`: 0
- Train/val/test split: empty
- Detector training: not started

## Stage 1 target
Minimum acceptable target for the first baseline:
- annotate all 48 current images
- extend the pool toward 200 to 300 annotated images
- keep labels consistent with `ANNOTATION_GUIDE.md`
- create a valid train/val/test split
- train one YOLOv8n single-class baseline

## Exact workflow
1. Annotate images from `datasets/parks_detect/images/all`.
2. Save one YOLO label file per image in `datasets/parks_detect/labels/all`.
3. Run progress check:

```powershell
.venv\Scripts\python.exe scripts\report_annotation_progress.py
```

4. Build the split and validate the dataset:

```powershell
.venv\Scripts\python.exe scripts\run_detection_pipeline.py --skip-train
```

5. Train the first baseline detector:

```powershell
.venv\Scripts\python.exe scripts\train.py --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --val
```

6. Run inference on a test image or video after training.

## Annotation rules
- draw one bounding box per visible trash object
- every valid waste object is labeled `trash`
- do not label tiny ambiguous fragments
- do not merge multiple objects into one box

## Acceptance criteria
Stage 1 is complete only when:
- every staged image has a matching `.txt` label file
- dataset validation passes without errors
- object coverage is at least minimally usable
- one baseline detector has been trained successfully
- you can run inference and inspect sample predictions

## What not to do yet
- do not start the separate classifier yet
- do not generate crop datasets yet
- do not optimize architecture variants yet
- do not claim final performance from the first baseline