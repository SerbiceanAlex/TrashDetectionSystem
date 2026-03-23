"""Merge multiple classification datasets into one combined dataset.

Expects each source dataset to have the structure:
  <dataset>/
    train/  class_a/ class_b/ ...
    val/    ...
    test/   ...

All sources must share the same class names.

Usage:
  python scripts/merge_classification_datasets.py \\
      --datasets datasets/trashnet_cls datasets/parks_cls \\
      --out-dir datasets/mixed_cls
"""

import argparse
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description="Merge classification datasets")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Two or more dataset root directories to merge")
    p.add_argument("--out-dir", required=True,
                   help="Output merged dataset root")
    return p.parse_args()


def get_classes(dataset_root: Path) -> set[str]:
    train_dir = dataset_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train/ in {dataset_root}")
    return {d.name for d in train_dir.iterdir() if d.is_dir()}


def copy_split(src_root: Path, dst_root: Path, split: str, classes: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {c: 0 for c in classes}
    src_split = src_root / split
    if not src_split.exists():
        return counts
    for cls in classes:
        src_cls = src_split / cls
        if not src_cls.exists():
            continue
        dst_cls = dst_root / split / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        imgs = list(src_cls.glob("*.jpg")) + list(src_cls.glob("*.png")) + list(src_cls.glob("*.jpeg"))
        for img in imgs:
            # Prefix filename with source dataset name to avoid collisions
            prefix = src_root.name
            dst_path = dst_cls / f"{prefix}_{img.name}"
            shutil.copy2(img, dst_path)
            counts[cls] += 1
    return counts


def main():
    args = parse_args()

    sources = [Path(d) for d in args.datasets]
    out_dir = Path(args.out_dir)

    # Validate sources
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"Dataset not found: {src}")

    # Find union of all classes
    all_classes: set[str] = set()
    for src in sources:
        classes = get_classes(src)
        all_classes.update(classes)
    all_classes_sorted = sorted(all_classes)
    print(f"Classes: {all_classes_sorted}")

    # Prepare output
    if out_dir.exists():
        print(f"Removing existing output: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # Merge
    total: dict[str, dict[str, int]] = {c: {s: 0 for s in SPLITS} for c in all_classes_sorted}

    for src in sources:
        print(f"\nMerging from: {src}")
        for split in SPLITS:
            counts = copy_split(src, out_dir, split, all_classes_sorted)
            for cls, n in counts.items():
                total[cls][split] += n
                if n > 0:
                    print(f"  [{split:5s}] {cls}: +{n}")

    # Print summary
    print(f"\nMerged dataset: {out_dir}")
    header = f"  {'Class':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    grand = {"train": 0, "val": 0, "test": 0}
    for cls in all_classes_sorted:
        tr = total[cls]["train"]
        va = total[cls]["val"]
        te = total[cls]["test"]
        print(f"  {cls:<12} {tr:>6} {va:>6} {te:>6} {tr+va+te:>7}")
        grand["train"] += tr
        grand["val"] += va
        grand["test"] += te
    print("  " + "-" * (len(header) - 2))
    t = grand["train"] + grand["val"] + grand["test"]
    print(f"  {'TOTAL':<12} {grand['train']:>6} {grand['val']:>6} {grand['test']:>6} {t:>7}")

    print("\nNext step — Train classifier on mixed data (Experiment B3):")
    print(f"  .venv\\Scripts\\python.exe scripts\\train_classifier.py --data {out_dir} --epochs 80 --batch 32 --workers 0 --name mixed-cls-b3")


if __name__ == "__main__":
    main()
