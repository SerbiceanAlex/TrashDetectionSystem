"""Download and prepare the TrashNet dataset for Stage 2 classification.

TrashNet (Stanford) — 2527 images, 6 classes:
  glass (501), paper (594), cardboard (403), plastic (482), metal (410), trash (137)

Mapping to our 5-class system:
  glass      → glass
  metal      → metal
  paper      → paper
  cardboard  → paper   (merged)
  plastic    → plastic
  trash      → other

Output structure (mirrors parks_cls):
  datasets/trashnet_cls/
    train/  glass/ metal/ paper/ plastic/ other/
    val/    ...
    test/   ...

Usage:
  python scripts/download_trashnet.py
  python scripts/download_trashnet.py --out-dir datasets/trashnet_cls --split 0.8 0.1 0.1
"""

import argparse
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# Direct download URL (Hugging Face mirror — no auth required)
TRASHNET_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
)
ZIP_NAME = "trashnet-dataset-resized.zip"

# Map TrashNet classes → our 5 classes
CLASS_MAP = {
    "glass": "glass",
    "metal": "metal",
    "paper": "paper",
    "cardboard": "paper",   # merged into paper
    "plastic": "plastic",
    "trash": "other",
}

OUR_CLASSES = ["glass", "metal", "other", "paper", "plastic"]


def parse_args():
    p = argparse.ArgumentParser(description="Download and split TrashNet for Stage 2 classifier")
    p.add_argument("--out-dir", default="datasets/trashnet_cls",
                   help="Output dataset root (default: datasets/trashnet_cls)")
    p.add_argument("--cache-dir", default="datasets/raw",
                   help="Directory to cache the downloaded zip")
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Train/val/test split ratios (must sum to 1.0)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip download if zip already exists in cache-dir")
    return p.parse_args()


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file, showing a simple progress bar."""
    print(f"Downloading {url}")
    print(f"  → {dest}")

    def _hook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(100, count * block_size * 100 // total_size)
        bar = "#" * (pct // 2)
        print(f"\r  [{bar:<50}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)
    print()  # newline after progress


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """Extract zip and return the top-level extracted folder."""
    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the common prefix (top-level folder inside zip)
        names = zf.namelist()
        top = Path(names[0]).parts[0] if names else "dataset-resized"
        zf.extractall(extract_to)
    extracted = extract_to / top
    print(f"  Extracted to: {extracted}")
    return extracted


def collect_images(src_root: Path) -> dict[str, list[Path]]:
    """
    Walk src_root for known TrashNet class folders and collect image paths.
    Returns {our_class: [image_paths]}.
    """
    collected: dict[str, list[Path]] = {c: [] for c in OUR_CLASSES}
    found_any = False

    for trashnet_cls, our_cls in CLASS_MAP.items():
        src_dir = src_root / trashnet_cls
        if not src_dir.exists():
            # Try nested: some zips have dataset-resized/dataset-resized/
            src_dir = src_root / "dataset-resized" / trashnet_cls
        if not src_dir.exists():
            print(f"  [WARN] Class folder not found: {src_dir}")
            continue
        imgs = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        print(f"  {trashnet_cls:12s} ({len(imgs):4d} images) → {our_cls}")
        collected[our_cls].extend(imgs)
        found_any = True

    if not found_any:
        raise FileNotFoundError(
            f"No TrashNet class folders found under {src_root}. "
            "Try re-extracting the zip or check the --cache-dir."
        )
    return collected


def split_and_copy(
    collected: dict[str, list[Path]],
    out_dir: Path,
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, dict[str, int]]:
    """Split each class list and copy files to train/val/test folders."""
    rng = random.Random(seed)
    stats: dict[str, dict[str, int]] = {}

    for cls, paths in collected.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits = {
            "train": paths[:n_train],
            "val": paths[n_train: n_train + n_val],
            "test": paths[n_train + n_val:],
        }
        stats[cls] = {}
        for split_name, split_paths in splits.items():
            dest_dir = out_dir / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                shutil.copy2(src, dest_dir / src.name)
            stats[cls][split_name] = len(split_paths)

    return stats


def print_stats(stats: dict[str, dict[str, int]]) -> None:
    header = f"  {'Class':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    totals = {"train": 0, "val": 0, "test": 0}
    for cls in OUR_CLASSES:
        row = stats.get(cls, {})
        tr, va, te = row.get("train", 0), row.get("val", 0), row.get("test", 0)
        print(f"  {cls:<12} {tr:>6} {va:>6} {te:>6} {tr+va+te:>7}")
        totals["train"] += tr
        totals["val"] += va
        totals["test"] += te
    print("  " + "-" * (len(header) - 2))
    t = totals["train"] + totals["val"] + totals["test"]
    print(f"  {'TOTAL':<12} {totals['train']:>6} {totals['val']:>6} {totals['test']:>6} {t:>7}")


def main():
    args = parse_args()

    train_r, val_r, test_r = args.split
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        print("ERROR: --split ratios must sum to 1.0")
        sys.exit(1)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / ZIP_NAME

    # ── 1. Download ──────────────────────────────────────────────────────────
    if zip_path.exists() and args.skip_download:
        print(f"Skipping download — using cached {zip_path}")
    elif zip_path.exists():
        print(f"Zip already exists at {zip_path}, skipping download.")
        print("  (use --skip-download to suppress this message)")
    else:
        download_with_progress(TRASHNET_URL, zip_path)

    # ── 2. Extract ────────────────────────────────────────────────────────────
    extract_dir = cache_dir / "trashnet_extracted"
    if extract_dir.exists():
        print(f"Extract folder already exists: {extract_dir}, reusing.")
    else:
        src_root = extract_zip(zip_path, extract_dir)
    # Handle nested extraction
    src_root = extract_dir
    if (extract_dir / "dataset-resized").exists():
        src_root = extract_dir / "dataset-resized"

    # ── 3. Collect images ────────────────────────────────────────────────────
    print("\nCollecting images per class:")
    collected = collect_images(src_root)
    total = sum(len(v) for v in collected.values())
    print(f"  Total images collected: {total}")

    # ── 4. Split & copy ───────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"\nOutput dir exists ({out_dir}), overwriting...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print(f"\nSplitting {train_r:.0%}/{val_r:.0%}/{test_r:.0%} → {out_dir}")
    stats = split_and_copy(collected, out_dir, (train_r, val_r, test_r), args.seed)

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print(f"\nDataset ready: {out_dir}")
    print_stats(stats)

    print("\nNext steps:")
    print(f"  # Train on TrashNet only (Experiment B2):")
    print(f"  .venv\\Scripts\\python.exe scripts\\train_classifier.py --data {out_dir} --epochs 50 --batch 32 --workers 0 --name trashnet-cls-b2")
    print(f"")
    print(f"  # Then merge with park crops for B3:")
    print(f"  .venv\\Scripts\\python.exe scripts\\merge_classification_datasets.py --datasets {out_dir} datasets/parks_cls --out-dir datasets/mixed_cls")


if __name__ == "__main__":
    main()
