import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the stage-2 classification pipeline: report stats, train classifier, and evaluate"
    )
    parser.add_argument("--data", default="datasets/parks_cls", help="Classification dataset root")
    parser.add_argument("--skip-train", action="store_true", help="Stop after reporting dataset statistics")
    parser.add_argument("--skip-eval", action="store_true", help="Train the classifier but skip held-out evaluation")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Base classification checkpoint")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Training image size")
    parser.add_argument("--batch", type=int, default=32, help="Training batch size")
    parser.add_argument("--device", default=None, help="Training device")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--project", default="runs/classify", help="Training output directory")
    parser.add_argument("--name", default="parks-trash-material-cls", help="Training run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cache", action="store_true", help="Enable image cache during training")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    parser.add_argument("--train-val", action="store_true", help="Run validation after training")
    parser.add_argument("--val-split", choices=("val", "test"), default="val", help="Split used during training validation")
    parser.add_argument("--eval-split", choices=("train", "val", "test"), default="test", help="Held-out split used for final evaluation")
    parser.add_argument("--eval-project", default="runs/classify_eval", help="Evaluation output directory")
    parser.add_argument("--eval-name", default="parks-trash-material-cls", help="Evaluation run name")
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

    stats_cmd = [python_executable, "scripts/report_classification_dataset_stats.py", "--data", args.data]
    run_step("Report classification dataset statistics", stats_cmd)

    if args.skip_train:
        print("[DONE] Classification pipeline finished without training")
        return

    train_cmd = [
        python_executable,
        "scripts/train_classifier.py",
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
        "--seed",
        str(args.seed),
        "--val-split",
        args.val_split,
    ]
    if args.device:
        train_cmd.extend(["--device", args.device])
    if args.cache:
        train_cmd.append("--cache")
    if args.resume:
        train_cmd.append("--resume")
    if args.train_val:
        train_cmd.append("--val")

    run_step("Train classifier", train_cmd)

    if args.skip_eval:
        print("[DONE] Classification pipeline finished without held-out evaluation")
        return

    best_model = f"{args.project}/{args.name}/weights/best.pt"
    eval_cmd = [
        python_executable,
        "scripts/evaluate_classifier.py",
        "--model",
        best_model,
        "--data",
        args.data,
        "--split",
        args.eval_split,
        "--imgsz",
        str(args.imgsz),
        "--project",
        args.eval_project,
        "--name",
        args.eval_name,
    ]
    if args.device:
        eval_cmd.extend(["--device", args.device])

    run_step("Evaluate classifier", eval_cmd)
    print("[DONE] Full classification pipeline completed")


if __name__ == "__main__":
    main()