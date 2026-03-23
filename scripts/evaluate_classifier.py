import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES = ["glass", "metal", "other", "paper", "plastic"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "classify_eval"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained trash material classifier on a dataset split")
    parser.add_argument("--model", required=True, help="Classifier checkpoint path")
    parser.add_argument("--data", default="datasets/parks_cls", help="Classification dataset root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test", help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size")
    parser.add_argument("--device", default=None, help="Device to use: cpu, 0, 0,1, ...")
    parser.add_argument("--workers", type=int, default=0, help="Number of dataloader workers (0=safe on Windows)")
    parser.add_argument("--project", default=str(DEFAULT_OUTPUT_DIR), help="Directory for evaluation artifacts")
    parser.add_argument("--name", default="parks-trash-material-cls", help="Evaluation run name")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Expected class folder names in evaluation order",
    )
    return parser.parse_args()


def iter_images(directory: Path):
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def dataset_items(split_dir: Path, class_names):
    items = []
    for class_name in class_names:
        class_dir = split_dir / class_name
        for image_path in iter_images(class_dir):
            items.append((image_path, class_name))
    return items


def classifier_names(model):
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def predict_label(model, image_path: Path, imgsz: int, device, class_name_map):
    kwargs = {"imgsz": imgsz, "verbose": False}
    if device:
        kwargs["device"] = device
    result = model.predict(str(image_path), **kwargs)[0]
    probs = getattr(result, "probs", None)
    if probs is None:
        return "unknown", 0.0

    top_index = int(probs.top1)
    top_conf = float(probs.top1conf.item() if hasattr(probs.top1conf, "item") else probs.top1conf)
    return class_name_map.get(top_index, str(top_index)), top_conf


def main():
    args = parse_args()

    model_path = Path(args.model)
    data_root = Path(args.data)
    split_dir = data_root / args.split
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {model_path}")
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {split_dir}")

    items = dataset_items(split_dir, args.classes)
    if not items:
        raise SystemExit(f"No evaluation images found in {split_dir}")

    model = YOLO(str(model_path))
    class_name_map = classifier_names(model)

    y_true = []
    y_pred = []
    rows = []

    for image_path, true_label in items:
        pred_label, pred_conf = predict_label(model, image_path, args.imgsz, args.device, class_name_map)
        y_true.append(true_label)
        y_pred.append(pred_label)
        rows.append(
            {
                "image_path": image_path.as_posix(),
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_conf": pred_conf,
                "correct": pred_label == true_label,
            }
        )

    labels = list(args.classes)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    output_dir = Path(args.project) / args.name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = output_dir / "predictions.csv"
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "true_label", "pred_label", "pred_conf", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    confusion_csv = output_dir / "confusion_matrix.csv"
    with confusion_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *labels])
        for true_label, counts in zip(labels, matrix.tolist()):
            writer.writerow([true_label, *counts])

    summary = {
        "split": args.split,
        "num_images": len(items),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": {
            label: {
                "precision": float(cls_precision),
                "recall": float(cls_recall),
                "f1": float(cls_f1),
                "support": int(cls_support),
            }
            for label, cls_precision, cls_recall, cls_f1, cls_support in zip(labels, precision, recall, f1, support)
        },
    }

    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Images: {len(items)}")
    print(f"[INFO] Accuracy: {accuracy:.4f}")
    print(f"[INFO] Macro precision: {macro_precision:.4f}")
    print(f"[INFO] Macro recall: {macro_recall:.4f}")
    print(f"[INFO] Macro F1: {macro_f1:.4f}")
    print(f"[INFO] Predictions CSV: {predictions_csv}")
    print(f"[INFO] Confusion matrix CSV: {confusion_csv}")
    print(f"[INFO] Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()