import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full two-stage research pipeline: detector stage, crop export, and classifier stage"
    )
    parser.add_argument("--skip-detection", action="store_true", help="Skip the detector split/train/eval stage")
    parser.add_argument("--skip-crops", action="store_true", help="Skip exporting detection crops for stage 2")
    parser.add_argument("--skip-classification", action="store_true", help="Skip the classifier split/train/eval stage")
    parser.add_argument(
        "--stop-after-crops",
        action="store_true",
        help="Stop after crop export so labeled crop sorting can be done manually before stage 2",
    )

    parser.add_argument("--detect-data", default="datasets/parks_detect/parks_detect.yaml", help="Detection dataset YAML path")
    parser.add_argument("--detect-root", default="datasets/parks_detect", help="Detection dataset root")
    parser.add_argument("--detect-model", default="yolov8n.pt", help="Base detector checkpoint")
    parser.add_argument("--detect-epochs", type=int, default=100, help="Detector training epochs")
    parser.add_argument("--detect-imgsz", type=int, default=640, help="Detector training image size")
    parser.add_argument("--detect-batch", type=int, default=16, help="Detector training batch size")
    parser.add_argument("--detect-device", default=None, help="Detector training device")
    parser.add_argument("--detect-workers", type=int, default=8, help="Detector dataloader workers")
    parser.add_argument("--detect-patience", type=int, default=20, help="Detector early stopping patience")
    parser.add_argument("--detect-project", default="runs/detect", help="Detector output directory")
    parser.add_argument("--detect-name", default="parks-trash-yolov8", help="Detector run name")
    parser.add_argument("--detect-seed", type=int, default=42, help="Detector random seed")
    parser.add_argument("--detect-val", type=float, default=0.15, help="Detection validation split ratio")
    parser.add_argument("--detect-test", type=float, default=0.15, help="Detection test split ratio")
    parser.add_argument("--detect-group-by", choices=("prefix", "image"), default="prefix", help="Detection split grouping mode")
    parser.add_argument("--detect-train-val", action="store_true", help="Run detector validation after training")
    parser.add_argument("--detect-save-json", action="store_true", help="Save detector JSON metrics when supported")

    parser.add_argument("--crop-images", default="datasets/parks_detect/images/all", help="Source images for crop export")
    parser.add_argument("--crop-labels", default="datasets/parks_detect/labels/all", help="Source labels for crop export")
    parser.add_argument("--crop-out", default="datasets/parks_cls_unsorted/all", help="Output directory for exported crops")
    parser.add_argument("--crop-manifest", default="datasets/parks_cls_unsorted/crops_manifest.csv", help="Crop manifest CSV path")
    parser.add_argument("--crop-margin", type=float, default=0.05, help="Relative crop padding margin")
    parser.add_argument("--crop-skip-empty", action="store_true", help="Skip images with empty detector labels during crop export")

    parser.add_argument("--cls-pool", default="datasets/parks_cls_pool", help="Reviewed crop pool root for classification")
    parser.add_argument("--cls-data", default="datasets/parks_cls", help="Classification dataset root")
    parser.add_argument("--cls-model", default="yolov8n-cls.pt", help="Base classifier checkpoint")
    parser.add_argument("--cls-epochs", type=int, default=100, help="Classifier training epochs")
    parser.add_argument("--cls-imgsz", type=int, default=224, help="Classifier training image size")
    parser.add_argument("--cls-batch", type=int, default=32, help="Classifier batch size")
    parser.add_argument("--cls-device", default=None, help="Classifier training device")
    parser.add_argument("--cls-workers", type=int, default=8, help="Classifier dataloader workers")
    parser.add_argument("--cls-patience", type=int, default=20, help="Classifier early stopping patience")
    parser.add_argument("--cls-project", default="runs/classify", help="Classifier output directory")
    parser.add_argument("--cls-name", default="parks-trash-material-cls", help="Classifier run name")
    parser.add_argument("--cls-seed", type=int, default=42, help="Classifier random seed")
    parser.add_argument("--cls-val", type=float, default=0.15, help="Classification validation split ratio")
    parser.add_argument("--cls-test", type=float, default=0.15, help="Classification test split ratio")
    parser.add_argument("--cls-group-by", choices=("crop", "source-image"), default="source-image", help="Classification split grouping mode")
    parser.add_argument("--cls-train-val", action="store_true", help="Run classifier validation after training")
    parser.add_argument("--cls-eval-split", choices=("train", "val", "test"), default="test", help="Held-out split for classifier evaluation")
    parser.add_argument("--cls-eval-project", default="runs/classify_eval", help="Classifier evaluation output directory")
    parser.add_argument("--cls-eval-name", default="parks-trash-material-cls", help="Classifier evaluation run name")
    return parser.parse_args()


def run_step(title, command):
    print(f"[STEP] {title}")
    print(f"[CMD] {' '.join(command)}")
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main():
    args = parse_args()
    python_executable = sys.executable

    if not args.skip_detection:
        detect_cmd = [
            python_executable,
            "scripts/run_detection_pipeline.py",
            "--data",
            args.detect_data,
            "--dataset-root",
            args.detect_root,
            "--val",
            str(args.detect_val),
            "--test",
            str(args.detect_test),
            "--seed",
            str(args.detect_seed),
            "--group-by",
            args.detect_group_by,
            "--model",
            args.detect_model,
            "--epochs",
            str(args.detect_epochs),
            "--imgsz",
            str(args.detect_imgsz),
            "--batch",
            str(args.detect_batch),
            "--workers",
            str(args.detect_workers),
            "--patience",
            str(args.detect_patience),
            "--project",
            args.detect_project,
            "--name",
            args.detect_name,
        ]
        if args.detect_device:
            detect_cmd.extend(["--device", args.detect_device])
        if args.detect_train_val:
            detect_cmd.append("--train-val")
        if args.detect_save_json:
            detect_cmd.append("--save-json")
        run_step("Run detector stage", detect_cmd)

        detector_best = f"{args.detect_project}/{args.detect_name}/weights/best.pt"
        detect_eval_cmd = [
            python_executable,
            "scripts/evaluate_detector.py",
            "--model",
            detector_best,
            "--data",
            args.detect_data,
            "--split",
            "test",
            "--imgsz",
            str(args.detect_imgsz),
            "--batch",
            str(args.detect_batch),
            "--project",
            "runs/detect_eval",
            "--name",
            args.detect_name,
        ]
        if args.detect_device:
            detect_eval_cmd.extend(["--device", args.detect_device])
        run_step("Evaluate detector stage", detect_eval_cmd)

    if not args.skip_crops:
        crop_cmd = [
            python_executable,
            "scripts/export_yolo_crops.py",
            "--images-dir",
            args.crop_images,
            "--labels-dir",
            args.crop_labels,
            "--out-dir",
            args.crop_out,
            "--manifest",
            args.crop_manifest,
            "--margin",
            str(args.crop_margin),
        ]
        if args.crop_skip_empty:
            crop_cmd.append("--skip-empty")
        run_step("Export detection crops", crop_cmd)

    if args.stop_after_crops:
        print("[DONE] Pipeline stopped after crop export so reviewed crop labeling can be completed manually")
        return

    if not args.skip_classification:
        split_cls_cmd = [
            python_executable,
            "scripts/split_classification_dataset.py",
            "--source-root",
            args.cls_pool,
            "--out-root",
            args.cls_data,
            "--val",
            str(args.cls_val),
            "--test",
            str(args.cls_test),
            "--seed",
            str(args.cls_seed),
            "--manifest",
            args.crop_manifest,
            "--group-by",
            args.cls_group_by,
            "--clear",
        ]
        run_step("Split classification dataset", split_cls_cmd)

        cls_cmd = [
            python_executable,
            "scripts/run_classification_pipeline.py",
            "--data",
            args.cls_data,
            "--model",
            args.cls_model,
            "--epochs",
            str(args.cls_epochs),
            "--imgsz",
            str(args.cls_imgsz),
            "--batch",
            str(args.cls_batch),
            "--workers",
            str(args.cls_workers),
            "--patience",
            str(args.cls_patience),
            "--project",
            args.cls_project,
            "--name",
            args.cls_name,
            "--seed",
            str(args.cls_seed),
            "--val-split",
            "val",
            "--eval-split",
            args.cls_eval_split,
            "--eval-project",
            args.cls_eval_project,
            "--eval-name",
            args.cls_eval_name,
        ]
        if args.cls_device:
            cls_cmd.extend(["--device", args.cls_device])
        if args.cls_train_val:
            cls_cmd.append("--train-val")
        run_step("Run classifier stage", cls_cmd)

    print("[DONE] Full two-stage pipeline completed")


if __name__ == "__main__":
    main()