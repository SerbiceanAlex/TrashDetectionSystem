"""Build a comprehensive detection dataset by combining multiple sources.

Orchestrates:
  1. TACO public dataset (download + convert)
  2. Park-specific images (existing + new videos)
  3. Split each source independently
  4. Merge into a single training-ready dataset

Usage:
    python scripts/build_full_dataset.py --skip-taco-download   # if TACO already downloaded
    python scripts/build_full_dataset.py --taco-only             # just TACO conversion
    python scripts/build_full_dataset.py                         # full pipeline

The merged dataset lands in datasets/parks_detect_full/ with a matching YAML.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


PYTHON = sys.executable


def run(cmd: list[str], check: bool = True, **kwargs):
    """Run a subprocess and print the command."""
    print(f"\n> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def parse_args():
    p = argparse.ArgumentParser(description="Build the full merged detection dataset")
    p.add_argument("--taco-dir", default="datasets/raw/taco", help="TACO repo location")
    p.add_argument("--taco-yolo", default="datasets/taco_yolo", help="TACO in YOLO format")
    p.add_argument("--park-dataset", default="datasets/parks_detect", help="Park-specific YOLO dataset root")
    p.add_argument("--merged-out", default="datasets/parks_detect_full", help="Final merged dataset root")
    p.add_argument("--skip-taco-download", action="store_true", help="Skip TACO download (already cloned)")
    p.add_argument("--taco-only", action="store_true", help="Only process TACO, skip park data")
    p.add_argument("--val", type=float, default=0.15, help="Validation split ratio")
    p.add_argument("--test", type=float, default=0.15, help="Test split ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    return p.parse_args()


def step_taco(args):
    """Download and convert TACO → YOLO."""
    print("\n" + "=" * 60)
    print("STEP 1: TACO dataset → YOLO single-class")
    print("=" * 60)

    cmd = [
        PYTHON, "scripts/download_taco_dataset.py",
        "--taco-dir", args.taco_dir,
        "--out-dir", args.taco_yolo,
    ]
    if args.skip_taco_download:
        cmd.append("--skip-download")

    run(cmd)


def step_split_taco(args):
    """Split the TACO YOLO dataset into train/val/test."""
    print("\n" + "=" * 60)
    print("STEP 2: Split TACO dataset")
    print("=" * 60)

    taco_yolo = Path(args.taco_yolo)
    images_dir = taco_yolo / "images"
    labels_dir = taco_yolo / "labels"

    # If TACO comes as flat images/labels (no splits), create an "all" structure
    # and then split
    if not (images_dir / "all").exists():
        # Already flat → treat images/ and labels/ as "all"
        all_img = taco_yolo / "images_all_tmp"
        all_lbl = taco_yolo / "labels_all_tmp"

        if images_dir.exists():
            images_dir.rename(all_img)
        if labels_dir.exists():
            labels_dir.rename(all_lbl)

        (images_dir / "all").mkdir(parents=True, exist_ok=True)
        (labels_dir / "all").mkdir(parents=True, exist_ok=True)

        # Move files into all/
        for f in all_img.iterdir():
            if f.is_file():
                shutil.move(str(f), str(images_dir / "all" / f.name))
        for f in all_lbl.iterdir():
            if f.is_file():
                shutil.move(str(f), str(labels_dir / "all" / f.name))

        # Cleanup temp dirs
        if all_img.exists():
            shutil.rmtree(all_img)
        if all_lbl.exists():
            shutil.rmtree(all_lbl)

    run([
        PYTHON, "scripts/split_yolo_detection_dataset.py",
        "--dataset-root", args.taco_yolo,
        "--val", str(args.val),
        "--test", str(args.test),
        "--seed", str(args.seed),
        "--group-by", "image",
        "--clear",
        "--allow-empty-labels",
    ])


def step_split_park(args):
    """Ensure park dataset is split."""
    print("\n" + "=" * 60)
    print("STEP 3: Split park dataset")
    print("=" * 60)

    run([
        PYTHON, "scripts/split_yolo_detection_dataset.py",
        "--dataset-root", args.park_dataset,
        "--val", str(args.val),
        "--test", str(args.test),
        "--seed", str(args.seed),
        "--clear",
        "--allow-empty-labels",
    ])


def step_merge(args):
    """Merge TACO + park into final dataset."""
    print("\n" + "=" * 60)
    print("STEP 4: Merge datasets")
    print("=" * 60)

    sources = [args.taco_yolo, args.park_dataset]
    names = ["taco", "park"]

    if args.taco_only:
        sources = [args.taco_yolo]
        names = ["taco"]

    run([
        PYTHON, "scripts/merge_yolo_datasets.py",
        "--sources", *sources,
        "--names", *names,
        "--out", args.merged_out,
    ])


def step_create_yaml(args):
    """Create a YOLO dataset YAML for the merged dataset."""
    print("\n" + "=" * 60)
    print("STEP 5: Create dataset YAML")
    print("=" * 60)

    merged = Path(args.merged_out)
    yaml_path = merged / "dataset.yaml"

    yaml_content = f"""path: {merged.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: trash
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"[INFO] Dataset YAML written to: {yaml_path}")


def step_stats(args):
    """Report stats for the merged dataset."""
    print("\n" + "=" * 60)
    print("STEP 6: Dataset statistics")
    print("=" * 60)

    run([
        PYTHON, "scripts/report_yolo_dataset_stats.py",
        "--data", str(Path(args.merged_out) / "dataset.yaml"),
    ], check=False)


def main():
    args = parse_args()

    step_taco(args)
    step_split_taco(args)

    if not args.taco_only:
        step_split_park(args)

    step_merge(args)
    step_create_yaml(args)
    step_stats(args)

    print("\n" + "=" * 60)
    print("[DONE] Full dataset built!")
    print("=" * 60)
    merged = Path(args.merged_out)
    print(f"  Dataset root: {merged}")
    print(f"  YAML config:  {merged / 'dataset.yaml'}")
    print()
    print("[NEXT] Train with the merged dataset:")
    print(f"  {PYTHON} scripts/train.py --data {merged / 'dataset.yaml'} --epochs 150 --imgsz 640 --batch 16")


if __name__ == "__main__":
    main()
