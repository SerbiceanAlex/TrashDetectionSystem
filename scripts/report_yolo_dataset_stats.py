import argparse
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Report basic statistics for a YOLO detection dataset")
    parser.add_argument("--data", default="datasets/parks_detect/parks_detect.yaml", help="YOLO dataset YAML path")
    return parser.parse_args()


def parse_names(raw_names):
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    raise ValueError("'names' must be a mapping or a list")


def resolve_dataset_root(config_path: Path, raw_path: str):
    configured_path = Path(raw_path)
    if configured_path.is_absolute():
        return configured_path

    candidates = [configured_path, Path.cwd() / configured_path, config_path.parent / configured_path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (Path.cwd() / configured_path).resolve()


def resolve_split_dir(root: Path, split_value: str):
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return root / split_path


def image_files(directory: Path):
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def count_labels(label_path: Path):
    object_count = 0
    class_counts = Counter()
    if not label_path.exists():
        return object_count, class_counts

    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if not parts:
            continue
        class_id = int(parts[0])
        class_counts[class_id] += 1
        object_count += 1
    return object_count, class_counts


def main():
    args = parse_args()
    config_path = Path(args.data)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    dataset_root = resolve_dataset_root(config_path, data["path"])
    names = parse_names(data["names"])

    overall_images = 0
    overall_empty = 0
    overall_objects = 0
    overall_class_counts = Counter()

    for split_name in ("train", "val", "test"):
        split_value = data.get(split_name)
        if not split_value:
            continue

        image_dir = resolve_split_dir(dataset_root, split_value)
        label_dir = dataset_root / "labels" / Path(split_value).name
        images = image_files(image_dir)
        split_images = len(images)
        split_empty = 0
        split_objects = 0
        split_class_counts = Counter()

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            object_count, class_counts = count_labels(label_path)
            if object_count == 0:
                split_empty += 1
            split_objects += object_count
            split_class_counts.update(class_counts)

        overall_images += split_images
        overall_empty += split_empty
        overall_objects += split_objects
        overall_class_counts.update(split_class_counts)

        print(f"[INFO] Split: {split_name}")
        print(f" - images: {split_images}")
        print(f" - empty images: {split_empty}")
        print(f" - labeled objects: {split_objects}")
        for class_id, class_name in names.items():
            print(f" - {class_name}: {split_class_counts.get(class_id, 0)}")

    print("[INFO] Overall")
    print(f" - images: {overall_images}")
    print(f" - empty images: {overall_empty}")
    print(f" - labeled objects: {overall_objects}")
    for class_id, class_name in names.items():
        print(f" - {class_name}: {overall_class_counts.get(class_id, 0)}")


if __name__ == "__main__":
    main()