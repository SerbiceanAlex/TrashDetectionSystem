# Data Sources & Organisation

Documentation for the TrashDetectionSystem dataset pipeline.  
Keep this file up to date whenever new data is added or the labelling workflow changes.

---

## Data Organisation Structure

```
datasets/
├── raw/
│   ├── images/            ← standalone images (not from video)
│   ├── frames/            ← frames extracted from source videos
│   │   ├── <video_id_1>/
│   │   │   ├── frame_000000.jpg
│   │   │   ├── frame_000001.jpg
│   │   │   └── ...
│   │   └── <video_id_2>/
│   └── metadata/          ← this directory; committed to git
├── detection/             ← final YOLO-format detection dataset
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
├── processed/
│   └── classification/
│       └── crops/         ← material-labelled crops for classifier training
│           ├── plastic/
│           ├── glass/
│           ├── metal/
│           ├── paper/
│           └── other/
└── splits/
    └── detection_split.json
```

---

## Video Sources and Frame Extraction

Frames are extracted from source videos using `scripts/extract_frames.py`.

### Recommended settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--every-seconds` | 1.0 s | Enough visual diversity without excessive redundancy |
| `--max-frames` | 300 | Cap per video to keep dataset balanced |

### Naming convention

Each video gets its own sub-directory named after the video file stem:

```
datasets/raw/frames/<video_stem>/frame_<NNNNNN>.jpg
```

The zero-padded 6-digit index guarantees deterministic ordering and no naming
gaps when `--max-frames` is used.

A `manifest.csv` is written to `datasets/raw/frames/manifest.csv` with columns:
`video_id`, `frame_index`, `timestamp_s`, `path`.

---

## Leakage Prevention for Video Frames

### Why this matters

Consecutive video frames are visually nearly identical.  If frames from the
same video appear in both the training and validation sets, the model learns to
"recognise" the specific park scene rather than general trash features.  This
inflates validation metrics and leads to poor real-world performance.

### How we prevent it

The `make_detection_split` function in `trash_detection/splits.py` performs a
**group-aware split**:

1. All frames from a given video (sharing the same `video_id`) are placed in
   the *same* split.
2. Groups (video IDs) are shuffled with a fixed seed (42) before partitioning.
3. The default ratio is **70 % train / 15 % val / 15 % test** at the
   **group** level, not the frame level.

The resulting split is saved to `datasets/splits/detection_split.json` and
verified for zero overlap in notebook `01_data_overview_and_splits.ipynb`.

### Implication for dataset size

With a small number of videos (e.g. 5–10), group-aware splitting can produce
uneven splits.  Collect at least **10–15 distinct videos** to ensure robust
group-level partitioning.

---

## Labelling Workflow – Detection Stage

Labels are created in **YOLO format** (class-normalised bounding boxes).

### Recommended tool: LabelImg

```bash
pip install labelImg
labelImg datasets/detection/images/train
```

- Select format: **YOLO**
- Use a single class: `trash` (class index 0)
- Labels are saved as `.txt` files alongside the images

### Alternative: Roboflow

Export from Roboflow using the **YOLOv8** format.  Run
`scripts/coco_to_yolo_single_class.py` if exporting from a COCO-format source.

### Label format

One row per bounding box:

```
<class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
```

All values normalised to [0, 1] relative to image width and height.

### Quality guidelines

- Label **all** visible trash, including partially occluded items.
- Use tight bounding boxes (minimal background).
- For small or distant objects (< 10 × 10 px), skip if the object is
  indistinguishable from the background.

---

## Classification Labels

After detection crops are generated (notebook 03), each crop must be
manually moved into the correct material sub-folder:

| Class | Sub-folder | Examples |
|-------|-----------|---------|
| Plastic | `crops/plastic/` | PET bottles, bags, food wrappers, straws |
| Glass | `crops/glass/` | Bottles, jars, broken glass shards |
| Metal | `crops/metal/` | Aluminium cans, foil, wire, bottle caps |
| Paper | `crops/paper/` | Cardboard boxes, newspapers, paper cups |
| Other | `crops/other/` | Cigarette butts, organic waste, unknown |

### Minimum recommended samples

| Class | Minimum |
|-------|---------|
| Plastic | 300 |
| Glass | 200 |
| Metal | 200 |
| Paper | 200 |
| Other | 200 |

Plastic is expected to be the most frequent class; use class-weighted loss
during training (see notebook 04) to handle imbalance.

---

## Evaluation Protocol

### Detection (Stage 1)

Evaluated with `yolo val` on the held-out test split:

- **mAP@50** – primary metric
- **mAP@50-95** – stricter metric for precise localisation
- **Precision / Recall** at default confidence threshold (0.25)

### Classification (Stage 2)

Evaluated on the validation portion of crops:

- **Macro-F1** – primary metric (unaffected by class imbalance)
- **Per-class F1** – to identify weak classes
- **Confusion matrix** – to spot systematic misclassifications

### End-to-end evaluation

Run notebook 05 on a held-out video and manually count:

- Correct detections with correct material label (true positives)
- Missed items (false negatives)
- Wrong material label on correctly detected items
