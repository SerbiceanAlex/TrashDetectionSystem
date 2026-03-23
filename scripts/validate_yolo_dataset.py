import argparse
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a YOLO detection dataset")
    parser.add_argument(
        "--data",
        default="datasets/parks_detect/parks_detect.yaml",
        help="Path to the YOLO dataset YAML file",
    )
    return parser.parse_args()


def load_dataset_config(config_path: Path):
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must define a mapping")

    required_keys = {"path", "train", "val", "names"}
    missing = sorted(required_keys - set(data))
    if missing:
        raise ValueError(f"Dataset YAML is missing keys: {', '.join(missing)}")

    return data


def resolve_split_dir(root: Path, split_value: str):
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path
    return root / split_path


def resolve_dataset_root(config_path: Path, raw_path: str):
    configured_path = Path(raw_path)
    if configured_path.is_absolute():
        return configured_path

    candidates = [
        configured_path,
        Path.cwd() / configured_path,
        config_path.parent / configured_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (Path.cwd() / configured_path).resolve()


def parse_class_map(raw_names):
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    raise ValueError("'names' must be a mapping or a list")


def validate_label_file(label_path: Path, class_count: int):
    problems = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{label_path}: line {line_number} should have 5 values")
            continue

        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            problems.append(f"{label_path}: line {line_number} contains non-numeric values")
            continue

        if class_id < 0 or class_id >= class_count:
            problems.append(f"{label_path}: line {line_number} has invalid class id {class_id}")

        x_center, y_center, width, height = coords
        if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
            problems.append(f"{label_path}: line {line_number} has center outside [0, 1]")
        if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            problems.append(f"{label_path}: line {line_number} has invalid width/height")

    return problems


def collect_images(directory: Path):
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_split(dataset_root: Path, split_name: str, split_value: str, class_count: int):
    image_dir = resolve_split_dir(dataset_root, split_value)
    label_dir = dataset_root / "labels" / Path(split_value).name

    problems = []
    images = collect_images(image_dir)
    if not image_dir.exists():
        problems.append(f"Missing image directory for split '{split_name}': {image_dir}")
        return problems, 0

    if not images:
        problems.append(f"Split '{split_name}' has no images: {image_dir}")

    if not label_dir.exists():
        problems.append(f"Missing label directory for split '{split_name}': {label_dir}")
        return problems, len(images)

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            problems.append(f"Missing label file for image: {image_path}")
            continue

        problems.extend(validate_label_file(label_path, class_count))

    label_files = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    for label_path in label_files:
        if label_path.stem not in image_stems:
            problems.append(f"Label without matching image: {label_path}")

    return problems, len(images)


def main():
    args = parse_args()
    config_path = Path(args.data)
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {config_path}")

    data = load_dataset_config(config_path)
    dataset_root = resolve_dataset_root(config_path, data["path"])

    class_map = parse_class_map(data["names"])
    class_count = len(class_map)

    total_images = 0
    all_problems = []
    for split_name in ("train", "val", "test"):
        split_value = data.get(split_name)
        if not split_value:
            continue

        problems, image_count = validate_split(dataset_root, split_name, split_value, class_count)
        total_images += image_count
        all_problems.extend(problems)

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Classes ({class_count}): {class_map}")
    print(f"[INFO] Total indexed images: {total_images}")

    if all_problems:
        print("[FAIL] Dataset validation found issues:")
        for problem in all_problems:
            print(f" - {problem}")
        raise SystemExit(1)

    print("[OK] Dataset structure is valid for YOLO detection training")


if __name__ == "__main__":
    main()