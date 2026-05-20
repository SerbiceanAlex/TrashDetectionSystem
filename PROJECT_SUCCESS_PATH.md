# Project Success Path

This project is now centered on the final detector:

- Active production detector: `models/detector/production/best.pt`
- Final training run: `runs/detect/parks-trash-A7-best-ext-i640`
- Final dataset: `datasets/parks_detect_A7_parks_focused_801010`
- Final test metrics: `results/detector/parks-trash-A7-best-ext-i640-test.json`

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
- `models/detector/archive/A4-ext/`
- `models/detector/archive/A4-8010/`
- `datasets/parks_detect_A4/`
- `datasets/parks_detect_A7_parks_focused_801010/`
- `runs/detect/parks-trash-A7-best-ext-i640/`
- `notebooks/training/06_train_detector_A7_parks_focused.ipynb`
- `notebooks/training/07_train_detector_A7_best_extended.ipynb`

## Archived Cleanup

Moved on 2026-05-20:

- Old runs: `runs/archive/detect/cleanup_2026-05-20/`
- Old generated datasets: `datasets/archive/cleanup_2026-05-20/`

These are retained only as recovery material until the thesis is defended.
