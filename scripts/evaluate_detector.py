import argparse
import json
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "detect_eval"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained trash detector on a YOLO dataset split")
    parser.add_argument("--model", required=True, help="Detector checkpoint path")
    parser.add_argument("--data", default="datasets/parks_detect/parks_detect.yaml", help="YOLO dataset YAML path")
    parser.add_argument("--split", choices=("val", "test"), default="test", help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold")
    parser.add_argument("--device", default=None, help="Device to use: cpu, 0, 0,1, ...")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers (0=safe on Windows)")
    parser.add_argument("--project", default=str(DEFAULT_OUTPUT_DIR), help="Directory for evaluation artifacts")
    parser.add_argument("--name", default="parks-trash-yolov8", help="Evaluation run name")
    parser.add_argument("--save-json", action="store_true", help="Save COCO-style JSON when supported")
    return parser.parse_args()


def to_builtin(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def main():
    args = parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    if not model_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(str(model_path))
    val_kwargs = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "project": args.project,
        "name": args.name,
        "save_json": args.save_json,
        "workers": args.workers,
        "plots": False,
        "verbose": False,
    }
    if args.device:
        val_kwargs["device"] = args.device

    results = model.val(**val_kwargs)
    metrics = to_builtin(getattr(results, "results_dict", {}))

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name)) / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": args.split,
        "model": str(model_path),
        "data": str(data_path),
        "metrics": metrics,
        "map50": to_builtin(getattr(getattr(results, "box", None), "map50", None)),
        "map50_95": to_builtin(getattr(getattr(results, "box", None), "map", None)),
        "precision": to_builtin(getattr(getattr(results, "box", None), "mp", None)),
        "recall": to_builtin(getattr(getattr(results, "box", None), "mr", None)),
    }

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Split: {args.split}")
    if summary["precision"] is not None:
        print(f"[INFO] Precision: {summary['precision']:.4f}")
    if summary["recall"] is not None:
        print(f"[INFO] Recall: {summary['recall']:.4f}")
    if summary["map50"] is not None:
        print(f"[INFO] mAP50: {summary['map50']:.4f}")
    if summary["map50_95"] is not None:
        print(f"[INFO] mAP50-95: {summary['map50_95']:.4f}")
    print(f"[INFO] Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()