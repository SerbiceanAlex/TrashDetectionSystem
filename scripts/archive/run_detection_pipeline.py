import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the post-annotation pipeline: split, validate, report stats, and optionally train"
    )
    parser.add_argument("--data", default="datasets/parks_detect/parks_detect.yaml", help="YOLO dataset YAML path")
    parser.add_argument("--dataset-root", default="datasets/parks_detect", help="YOLO dataset root")
    parser.add_argument("--source-images", default=None, help="Override source images directory for split")
    parser.add_argument("--source-labels", default=None, help="Override source labels directory for split")
    parser.add_argument("--val", type=float, default=0.15, help="Validation ratio for split")
    parser.add_argument("--test", type=float, default=0.15, help="Test ratio for split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--group-by", choices=("prefix", "image"), default="prefix", help="Split grouping mode")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of using hardlinks during split")
    parser.add_argument("--allow-empty-labels", action="store_true", help="Include unlabeled negatives during split")
    parser.add_argument("--no-clear", action="store_true", help="Do not clear existing train/val/test files before split")
    parser.add_argument("--skip-train", action="store_true", help="Stop after split, validation, and stats")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--device", default=None, help="Training device")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--project", default="runs/detect", help="Training output directory")
    parser.add_argument("--name", default="parks-trash-yolov8", help="Training run name")
    parser.add_argument("--optimizer", default="auto", help="Optimizer passed to Ultralytics")
    parser.add_argument("--cache", action="store_true", help="Enable image cache during training")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine LR schedule")
    parser.add_argument("--save-json", action="store_true", help="Save JSON metrics when supported")
    parser.add_argument("--train-val", action="store_true", help="Run Ultralytics validation after training")
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

    split_cmd = [
        python_executable,
        "scripts/split_yolo_detection_dataset.py",
        "--dataset-root",
        args.dataset_root,
        "--val",
        str(args.val),
        "--test",
        str(args.test),
        "--seed",
        str(args.seed),
        "--group-by",
        args.group_by,
    ]
    if args.source_images:
        split_cmd.extend(["--source-images", args.source_images])
    if args.source_labels:
        split_cmd.extend(["--source-labels", args.source_labels])
    if args.copy:
        split_cmd.append("--copy")
    if args.allow_empty_labels:
        split_cmd.append("--allow-empty-labels")
    if not args.no_clear:
        split_cmd.append("--clear")

    validate_cmd = [python_executable, "scripts/validate_yolo_dataset.py", "--data", args.data]
    stats_cmd = [python_executable, "scripts/report_yolo_dataset_stats.py", "--data", args.data]

    run_step("Create train/val/test split", split_cmd)
    run_step("Validate YOLO dataset", validate_cmd)
    run_step("Report dataset statistics", stats_cmd)

    if args.skip_train:
        print("[DONE] Pipeline finished without training")
        return

    train_cmd = [
        python_executable,
        "scripts/train.py",
        "--model",
        args.model,
        "--data",
        args.data,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--patience",
        str(args.patience),
        "--project",
        args.project,
        "--name",
        args.name,
        "--optimizer",
        args.optimizer,
    ]
    if args.device:
        train_cmd.extend(["--device", args.device])
    if args.cache:
        train_cmd.append("--cache")
    if args.resume:
        train_cmd.append("--resume")
    if args.cos_lr:
        train_cmd.append("--cos-lr")
    if args.save_json:
        train_cmd.append("--save-json")
    if args.train_val:
        train_cmd.append("--val")

    run_step("Train YOLO detector", train_cmd)
    print("[DONE] Full detection pipeline completed")


if __name__ == "__main__":
    main()