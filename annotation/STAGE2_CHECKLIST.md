# Stage 2 Checklist

## Objective
Build the classification dataset and baseline classifier for material prediction on detected trash crops.

Final classes:
- glass
- metal
- other
- paper
- plastic

## Input sources
- public classification datasets such as TrashNet
- crops generated from the park detector dataset
- crops extracted later from detector predictions on park images and videos

## Dataset structure
The classifier dataset should use this folder layout:

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

## Supporting utility
To generate unlabeled crops from YOLO detection labels, use:

```powershell
.venv\Scripts\python.exe scripts\export_yolo_crops.py --images-dir datasets\parks_detect\images\all --labels-dir datasets\parks_detect\labels\all
```

This creates an unsorted crop pool and a manifest file. The crops must then be reviewed and placed into the correct material classes.

After you sort reviewed crops into a pool like `datasets/parks_cls_pool/<class>/`, create the train/val/test split with:

```powershell
.venv\Scripts\python.exe scripts\split_classification_dataset.py --source-root datasets\parks_cls_pool --out-root datasets\parks_cls --clear
```

## Rules
- the detector decides whether an object is trash
- the classifier decides the apparent material class
- ambiguous crops should go to `other`
- do not let the same near-duplicate crop appear in both train and test
- preserve the park domain as much as possible when building your own crop set

## Acceptance criteria
Stage 2 is ready to train when:
- each split folder contains all five classes
- class imbalance is documented
- train, val, and test are separated correctly
- the crop quality is visually checked
- ambiguous or low-quality crops are filtered out

## Training commands
Inspect class balance:

```powershell
.venv\Scripts\python.exe scripts\report_classification_dataset_stats.py
```

Train the classifier baseline:

```powershell
.venv\Scripts\python.exe scripts\train_classifier.py --model yolov8n-cls.pt --data datasets\parks_cls --val-split val --epochs 100 --imgsz 224 --batch 32
```

Evaluate the classifier on the held-out split:

```powershell
.venv\Scripts\python.exe scripts\evaluate_classifier.py --model runs\classify\parks-trash-material-cls\weights\best.pt --data datasets\parks_cls --split test
```

Or run the entire stage-2 workflow in one command:

```powershell
.venv\Scripts\python.exe scripts\run_classification_pipeline.py --data datasets\parks_cls --model yolov8n-cls.pt --train-val --eval-split test
```

Review at minimum:
- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix
- per-image prediction errors