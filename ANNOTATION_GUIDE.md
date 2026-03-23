# Annotation Guide

## Goal
Create consistent YOLO annotations for stage-1 public-space waste detection with one class:

- `0 = trash`

## Recommended tool
Use CVAT or Label Studio.

Practical setup files are available under `annotation/`.

Recommended export formats:
- YOLO detection if you want to fill `datasets/parks_detect/images/all` and `datasets/parks_detect/labels/all` directly
- COCO JSON if you want to convert with `scripts/coco_to_yolo_single_class.py`

## Folder flow
If you annotate directly in YOLO format:

1. Put images in `datasets/parks_detect/images/all`, or stage them with `python scripts/stage_images_for_annotation.py --source path\\to\\images --flatten`
2. Put matching `.txt` labels in `datasets/parks_detect/labels/all`
3. Check progress with `python scripts/report_annotation_progress.py`
4. Run `python scripts/split_yolo_detection_dataset.py --clear`
5. Run `python scripts/validate_yolo_dataset.py`
6. Run `python scripts/report_yolo_dataset_stats.py`
7. Start training with `python scripts/train.py --model yolov8n.pt --val`
7. Or run everything after annotation with `python scripts/run_detection_pipeline.py --model yolov8n.pt --train-val`

If you annotate in COCO format:

1. Export one `annotations.json`
2. Run `python scripts/coco_to_yolo_single_class.py --coco path\\to\\annotations.json --images-root path\\to\\images --out datasets/parks_detect --group-by-prefix`
3. Run `python scripts/validate_yolo_dataset.py`
4. Run `python scripts/report_yolo_dataset_stats.py`
5. Start training with `python scripts/train.py --model yolov8n.pt --val`

## Labeling rules
- Draw one bounding box per visible waste object.
- Use tight boxes around the object, without too much background.
- If an object is heavily occluded but still clearly identifiable, annotate the visible region.
- Ignore tiny ambiguous objects that cannot be localized reliably.
- Do not merge multiple nearby objects into one box.
- At stage 1, do not encode material in the detector label. Every valid waste object is `trash`.

## What counts as trash
Include bottles, cans, wrappers, bags, paper waste, broken glass, cartons, mixed packaging, and similar abandoned waste objects.

Do not create different detector classes for material. That distinction belongs to the second-stage classifier.

## Negative examples
If an image intentionally contains no trash, keep the image and create an empty label file:

`python scripts/create_empty_yolo_labels.py --images-dir datasets/parks_detect/images/all --labels-dir datasets/parks_detect/labels/all --only-missing`

Only do this for real negative images. Do not create empty labels for images that still need annotation.

## Quality checklist
- Every image has one matching `.txt` file.
- No class IDs outside `0`.
- Similar waste objects are labeled consistently across the dataset.
- Frames from the same video sequence remain grouped together when you create the split.

## Minimum target before first training
- At least 200 to 500 annotated images as a first baseline
- At least a few hundred labeled trash objects if possible
- Some negative images without trash
- Mix of easy and difficult scenes: different light, distance, occlusion, and background clutter