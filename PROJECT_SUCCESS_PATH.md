# Project Success Path

This project is now centered on the final detector:

- Active production detector: `models/detector/production/best.pt`
- Final training run: `runs/detect/parks-trash-final`
- Final dataset: `datasets/parks_detect_final`
- Final test metrics: `results/detector/parks-trash-final-test.json`

## Final Detector Metrics

| Metric | Value |
|---|---:|
| Precision | 0.9234 |
| Recall | 0.8229 |
| F1 | 0.8703 |
| mAP50 | 0.9102 |
| mAP50-95 | 0.6834 |

## Thesis Narrative

The thesis should focus on the successful path:

1. Baseline object detector.
2. Dataset expansion and observed noise issues.
3. A7 dataset filtering and 80/10/10 split.
4. Final A7 extended training.
5. Final evaluation and integration in the application.

Old experiments were moved out of the active tree to reduce confusion.

## Active Items

Keep these in the active project:

- `models/detector/production/`
- `models/classify/B2/`
- `models/pretrained/yolov8n.pt`
- `datasets/parks_detect_final/`
- `runs/detect/parks-trash-final/`
- `notebooks/training/01_train_classifier.ipynb`
- `notebooks/training/02_train_detector_final.ipynb`
- `notebooks/evaluation/01_evaluate_detector.ipynb`
- `notebooks/evaluation/04_inference_demo.ipynb`

The only detector checkpoint used by the application is
`models/detector/production/best.pt`.

## Archived Cleanup

Moved on 2026-05-20:

- Old generated datasets: `datasets/archive/cleanup_2026-05-20/`
- Historical baseline dataset was moved under `datasets/archive/cleanup_2026-05-20/`
- Temporary validation runs: `runs/archive/validation_tmp_2026-05-20/`

Historical detector runs were removed from the active tree; the retained run is
`runs/detect/parks-trash-final`. Archived material is retained only as recovery
material until the thesis is defended.
