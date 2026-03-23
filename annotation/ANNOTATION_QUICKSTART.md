# Annotation Quickstart

## Class set
Use exactly these labels:

- `trash`

They must stay identical across all tools and exports.

## If you use Label Studio
1. Start Label Studio locally:

```powershell
./scripts/start_label_studio.ps1
```

2. Create a new project.
3. Open `annotation/label_studio_config.xml` and paste its content into the labeling config.
4. Build import tasks:

```bash
python scripts/build_label_studio_tasks.py --images-dir datasets/parks_detect/images/all --out annotation/label_studio_tasks.json --mode local-files --local-files-base .
```

5. In Label Studio, configure Local Files storage to the project root if needed.
6. Import `annotation/label_studio_tasks.json`.
7. Annotate bounding boxes for each visible waste object with the single label `trash`.
8. Export in COCO format if possible.
9. Convert with:

```bash
python scripts/coco_to_yolo_single_class.py --coco path\\to\\result.json --images-root datasets/parks_detect/images/all --out datasets/parks_detect --group-by-prefix
```

## If you use CVAT
1. Create a detection task.
2. Add the labels from `annotation/classes.txt`.
3. Upload images from `datasets/parks_detect/images/all`.
4. Annotate rectangles only.
5. Export as COCO 1.0.
6. Convert with:

```bash
python scripts/coco_to_yolo_single_class.py --coco path\\to\\annotations.json --images-root datasets/parks_detect/images/all --out datasets/parks_detect --group-by-prefix
```

## If you annotate directly in YOLO format
1. Keep images in `datasets/parks_detect/images/all`.
2. Save one `.txt` file per image in `datasets/parks_detect/labels/all`.
3. Use class IDs:

```text
0 trash
```

4. Then run:

```bash
python scripts/run_detection_pipeline.py --skip-train
```

## Next command after first annotations
When you have at least a small batch annotated, run:

```bash
python scripts/run_detection_pipeline.py --model yolov8n.pt --train-val
```