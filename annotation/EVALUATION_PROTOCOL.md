# Evaluation Protocol

## Goal
Evaluate the project at three levels:
- detector performance
- classifier performance
- end-to-end two-stage behavior

## Detector evaluation
Use the held-out detection split and report at minimum:
- precision
- recall
- mAP50
- mAP50-95
- number of test images

Command:

```powershell
.venv\Scripts\python.exe scripts\evaluate_detector.py --model runs\detect\parks-trash-yolov8\weights\best.pt --data datasets\parks_detect\parks_detect.yaml --split test
```

## Classifier evaluation
Use the held-out classification split and report at minimum:
- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix

Command:

```powershell
.venv\Scripts\python.exe scripts\evaluate_classifier.py --model runs\classify\parks-trash-material-cls\weights\best.pt --data datasets\parks_cls --split test
```

## End-to-end analysis
For the final pipeline, evaluate qualitatively and quantitatively when possible:
- missed detections
- wrong material predictions on correct detections
- detector-localization errors that degrade classification
- difficult scenes: grass, bushes, benches, occlusion, small objects, clutter

Useful batch command for folder-level inspection:

```powershell
.venv\Scripts\python.exe scripts\run_two_stage_batch.py --source-dir path\to\images --detector runs\detect\parks-trash-yolov8\weights\best.pt --classifier runs\classify\parks-trash-material-cls\weights\best.pt --save-images
```

## Minimum reporting package for the thesis
- one table for detector metrics
- one table for classifier metrics
- one confusion matrix for classifier results
- one error-analysis section with representative failures
- one short discussion of domain shift between public data and park-specific data