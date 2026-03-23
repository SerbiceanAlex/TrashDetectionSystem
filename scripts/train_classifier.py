import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "parks_cls"
DEFAULT_PROJECT_DIR = REPO_ROOT / "runs" / "classify"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 image classifier for trash material recognition")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Base classification checkpoint")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Classification dataset root")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Training image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default=None, help="Device to use: cpu, 0, 0,1, ...")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT_DIR), help="Directory for training runs")
    parser.add_argument("--name", default="parks-trash-material-cls", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training")
    parser.add_argument("--resume", action="store_true", help="Resume the latest interrupted run")
    parser.add_argument("--val", action="store_true", help="Run validation after training")
    parser.add_argument(
        "--val-split",
        choices=("val", "test"),
        default="val",
        help="Dataset split to use for classifier validation during or after training",
    )
    return parser.parse_args()


def validate_args(args):
    data_root = Path(args.data)
    if not data_root.exists():
        raise FileNotFoundError(f"Classification dataset root not found: {data_root}")

    train_dir = data_root / "train"
    eval_dir = data_root / args.val_split
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing training split directory: {train_dir}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Missing evaluation split directory: {eval_dir}")

    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be > 0")
    if args.batch == 0:
        raise ValueError("--batch cannot be 0")
    if args.workers < 0:
        raise ValueError("--workers must be >= 0")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")


def main():
    args = parse_args()
    validate_args(args)

    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "seed": args.seed,
        "cache": args.cache,
        "resume": args.resume,
    }
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    if args.val:
        model.val(data=args.data, split=args.val_split, imgsz=args.imgsz, batch=args.batch, device=args.device)

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"[DONE] Training artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()