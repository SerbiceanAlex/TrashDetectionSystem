"""
Antrenează clasificatorul de material (YOLOv8-cls) pentru recunoașterea tipului
de deșeu din decupajele de obiecte.
"""

import sys
# Diacriticele românești se afișează corect indiferent de codarea consolei.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "trashnet_cls"
DEFAULT_PROJECT_DIR = REPO_ROOT / "runs" / "classify"


def parse_args():
    parser = argparse.ArgumentParser(description="Antrenează un clasificator YOLOv8 pentru materialul deșeurilor")
    parser.add_argument("--model", default="models/pretrained/yolov8n-cls.pt", help="Checkpoint de bază pentru clasificare")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Rădăcina datasetului de clasificare")
    parser.add_argument("--epochs", type=int, default=100, help="Numărul de epoci de antrenare")
    parser.add_argument("--imgsz", type=int, default=224, help="Dimensiunea imaginii la antrenare")
    parser.add_argument("--batch", type=int, default=32, help="Dimensiunea batch-ului")
    parser.add_argument("--device", default=None, help="Dispozitivul: cpu, 0, 0,1, ...")
    parser.add_argument("--workers", type=int, default=8, help="Numărul de workeri pentru dataloader")
    parser.add_argument("--patience", type=int, default=20, help="Răbdarea pentru early stopping")
    parser.add_argument("--project", default=str(DEFAULT_PROJECT_DIR), help="Folderul pentru rulările de antrenare")
    parser.add_argument("--name", default="trashnet-material-cls", help="Numele rulării")
    parser.add_argument("--seed", type=int, default=42, help="Seed-ul aleator")
    parser.add_argument("--cache", action="store_true", help="Pune imaginile în cache pentru antrenare mai rapidă")
    parser.add_argument("--resume", action="store_true", help="Reia ultima rulare întreruptă")
    parser.add_argument("--val", action="store_true", help="Rulează validarea după antrenare")
    parser.add_argument(
        "--val-split",
        choices=("val", "test"),
        default="val",
        help="Split-ul folosit la validarea clasificatorului (în timpul sau după antrenare)",
    )
    return parser.parse_args()


def validate_args(args):
    """Verifică existența split-urilor de dataset și corectitudinea hiperparametrilor."""
    data_root = Path(args.data)
    if not data_root.exists():
        raise FileNotFoundError(f"Rădăcina datasetului de clasificare nu există: {data_root}")

    train_dir = data_root / "train"
    eval_dir = data_root / args.val_split
    if not train_dir.exists():
        raise FileNotFoundError(f"Lipsește folderul de split de antrenare: {train_dir}")
    if not eval_dir.exists():
        raise FileNotFoundError(f"Lipsește folderul de split de evaluare: {eval_dir}")

    if args.epochs <= 0:
        raise ValueError("--epochs trebuie să fie > 0")
    if args.imgsz <= 0:
        raise ValueError("--imgsz trebuie să fie > 0")
    if args.batch == 0:
        raise ValueError("--batch nu poate fi 0")
    if args.workers < 0:
        raise ValueError("--workers trebuie să fie >= 0")
    if args.patience < 0:
        raise ValueError("--patience trebuie să fie >= 0")


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
        print(f"[GATA] Artefactele de antrenare salvate în: {save_dir}")


if __name__ == "__main__":
    main()
