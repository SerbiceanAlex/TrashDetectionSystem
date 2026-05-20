"""
Build parks_detect_A7_parks_focused dataset from A6 sources, filtered.

Strategy:
- parks_detect:       TOATE (gold standard, scene exterioare publice cu gunoi)
- illegal_dumping:    TOATE (mic dar primordial — persoana cu gunoi, scene CCTV)
- person_trash:       DOAR cu eticheta (skip 35% negative samples — scene goale fara gunoi)
- litter_aiengineer:  SKIP COMPLET (label quality slabe, 87% tiny + fake boxes confirmate vizual)

Resultat estimat: ~7,500 imagini (vs 17,795 din A6) — cu 56% mai putin, dar 100% relevant.

Usage:
    .venv\\Scripts\\python.exe scripts\\data\\build_a7_parks_focused.py --overwrite
    .venv\\Scripts\\python.exe scripts\\data\\build_a7_parks_focused.py --split-mode preserve --out datasets/parks_detect_A7_parks_focused
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")

# Source prefix → policy
# "all"        = copy everything (with or without labels)
# "labeled"    = copy only images that have non-empty label file
# "skip"       = skip entirely
POLICIES = {
    "a6_parks_detect":      "labeled",
    "a6_illegal_dumping":   "labeled",
    "a6_person_trash":      "labeled",
    "a6_litter_aiengineer": "skip",
}


@dataclass(frozen=True)
class Item:
    source: str
    original_split: str
    image: Path
    label: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build A7 parks-focused YOLO dataset")
    parser.add_argument(
        "--src",
        default="datasets/archive/cleanup_2026-05-20/parks_detect_A6_curated_20k",
        help="Source A6 dataset, archived after final A7 cleanup",
    )
    parser.add_argument(
        "--out",
        default="datasets/parks_detect_A7_parks_focused_801010",
        help="Output dataset directory",
    )
    parser.add_argument(
        "--split-mode",
        choices=("801010", "preserve"),
        default="801010",
        help="801010 creates a stratified 80/10/10 split by source; preserve keeps source splits.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for 80/10/10 split")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory before building")
    return parser.parse_args()


def has_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    try:
        content = label_path.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    return bool(content)


def get_policy(filename: str) -> str:
    for prefix, policy in POLICIES.items():
        if filename.startswith(prefix):
            return policy
    return "unknown"


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def collect_items(src: Path, counters) -> list[Item]:
    items: list[Item] = []

    for split in SPLITS:
        src_img_dir = src / "images" / split
        src_lbl_dir = src / "labels" / split

        if not src_img_dir.exists():
            print(f"[skip] {split}: src dir missing")
            continue

        for img_path in src_img_dir.glob("*.jpg"):
            policy = get_policy(img_path.name)
            source = next((p for p in POLICIES if img_path.name.startswith(p)), "unknown")

            counters[source]["total"] += 1
            counters[source][f"total_{split}"] += 1

            if policy == "skip":
                counters[source]["skipped"] += 1
                continue

            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            labeled = has_label(lbl_path)

            if policy == "labeled" and not labeled:
                counters[source]["dropped_no_label"] += 1
                continue

            items.append(Item(source=source, original_split=split, image=img_path, label=lbl_path))
            counters[source]["selected"] += 1

        print(f"[done] split={split}")

    return items


def assign_splits(items: list[Item], split_mode: str, seed: int) -> dict[str, list[Item]]:
    assigned = {split: [] for split in SPLITS}

    if split_mode == "preserve":
        for item in items:
            assigned[item.original_split].append(item)
        return assigned

    grouped = defaultdict(list)
    for item in items:
        grouped[item.source].append(item)

    rng = random.Random(seed)
    for source, source_items in sorted(grouped.items()):
        shuffled = list(source_items)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        assigned["train"].extend(shuffled[:n_train])
        assigned["val"].extend(shuffled[n_train:n_train + n_val])
        assigned["test"].extend(shuffled[n_train + n_val:])

    for split in SPLITS:
        assigned[split].sort(key=lambda item: item.image.name)
    return assigned


def copy_items(assigned: dict[str, list[Item]], out: Path, counters) -> None:
    for split, items in assigned.items():
        out_img_dir = out / "images" / split
        out_lbl_dir = out / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        used_names: set[str] = set()

        for idx, item in enumerate(items):
            # Resplitting can move files from original train/val/test into the
            # same target split. Some Roboflow exports reuse the same filename
            # across original splits, so we write a deterministic unique name.
            base_stem = f"{item.original_split}_{item.image.stem}"
            out_stem = base_stem
            suffix = 1
            while f"{out_stem}{item.image.suffix.lower()}" in used_names:
                suffix += 1
                out_stem = f"{base_stem}_{suffix}"
            out_img_name = f"{out_stem}{item.image.suffix.lower()}"
            used_names.add(out_img_name)

            shutil.copy2(item.image, out_img_dir / out_img_name)
            shutil.copy2(item.label, out_lbl_dir / f"{out_stem}.txt")
            counters[item.source]["copied_with_label"] += 1
            counters[item.source][f"copied_{split}"] += 1


def main():
    args = parse_args()
    src = resolve(args.src)
    out = resolve(args.out)

    if out.exists():
        if not args.overwrite:
            print(f"[warn] {out} already exists. Use --overwrite or choose another --out.")
            return 1
        shutil.rmtree(out)

    print(f"Source: {src}")
    print(f"Target: {out}")
    print(f"Split mode: {args.split_mode}\n")

    counters = defaultdict(lambda: defaultdict(int))
    items = collect_items(src, counters)
    assigned = assign_splits(items, args.split_mode, args.seed)
    copy_items(assigned, out, counters)

    # Write dataset.yaml (same format as A6)
    yaml_content = f"""# parks_detect_A7_parks_focused
# Built from parks_detect_A6_curated_20k with filtering:
# - parks_detect:       TOATE (gold standard, TACO-derived outdoor public spaces)
# - illegal_dumping:    TOATE (mic dar primordial pentru littering act)
# - person_trash:       DOAR labeled (35% negatives dropped)
# - litter_aiengineer:  SKIP (label quality slabe)
# Split: {args.split_mode}

path: {out.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: 1
names:
  0: trash
"""
    (out / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    # Curation report
    report = {
        "source": str(src),
        "target": str(out),
        "policies": POLICIES,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "split_counts": {split: len(assigned[split]) for split in SPLITS},
        "stats_per_source": {k: dict(v) for k, v in counters.items()},
    }
    (out / "curation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Source':<25} {'Total':>7} {'Copied':>7} {'Skipped':>8} {'No-label':>10}")
    print("-" * 70)
    grand_total = 0
    grand_copied = 0
    for source, stats in counters.items():
        total = stats.get("total", 0)
        copied = stats.get("copied_with_label", 0) + stats.get("copied_negative", 0)
        skipped = stats.get("skipped", 0)
        no_label = stats.get("dropped_no_label", 0)
        grand_total += total
        grand_copied += copied
        print(f"{source:<25} {total:>7} {copied:>7} {skipped:>8} {no_label:>10}")
    print("-" * 70)
    print(f"{'TOTAL':<25} {grand_total:>7} {grand_copied:>7}")
    print()
    print("Split counts:")
    for split in SPLITS:
        pct = (len(assigned[split]) / max(len(items), 1)) * 100
        print(f"  {split:<5}: {len(assigned[split]):>5} ({pct:>5.1f}%)")
    print()
    print(f"dataset.yaml      : {out / 'dataset.yaml'}")
    print(f"curation_report   : {out / 'curation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
