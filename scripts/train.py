import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_DATASET = Path("datasets/parks_detect/parks_detect.yaml")


def parse_args():
	parser = argparse.ArgumentParser(description="Train a YOLOv8 detector for public-space trash detection")
	parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model checkpoint")
	parser.add_argument("--data", default=str(DEFAULT_DATASET), help="YOLO dataset YAML path")
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
	parser.add_argument("--batch", type=int, default=16, help="Batch size")
	parser.add_argument("--device", default=None, help="Device to use: cpu, 0, 0,1, ...")
	parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
	parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
	parser.add_argument("--project", default=None, help="Directory for training runs (default: Ultralytics auto)")
	parser.add_argument("--name", default="parks-trash-yolov8", help="Run name")
	parser.add_argument("--optimizer", default="auto", help="Optimizer passed to Ultralytics")
	parser.add_argument("--cache", action="store_true", help="Cache images for faster training")
	parser.add_argument("--resume", action="store_true", help="Resume the latest interrupted run")
	parser.add_argument("--cos-lr", action="store_true", help="Enable cosine learning rate schedule")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--val", action="store_true", help="Run validation after training")
	parser.add_argument("--save-json", action="store_true", help="Save COCO-style metrics json when supported")
	return parser.parse_args()


def validate_args(args):
	data_path = Path(args.data)
	if not data_path.exists():
		raise FileNotFoundError(
			f"Dataset YAML not found: {data_path}. Create labels and split images before training."
		)

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
		"name": args.name,
		"optimizer": args.optimizer,
		"cache": args.cache,
		"resume": args.resume,
		"cos_lr": args.cos_lr,
		"seed": args.seed,
		"save_json": args.save_json,
	}
	if args.device:
		train_kwargs["device"] = args.device
	if args.project:
		train_kwargs["project"] = args.project

	results = model.train(**train_kwargs)

	if args.val:
		model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=args.device)

	save_dir = getattr(results, "save_dir", None)
	if save_dir:
		print(f"[DONE] Training artifacts saved to: {save_dir}")


if __name__ == "__main__":
	main()
