"""trash_detection/splits.py – Group-aware train/val/test split utilities.

Video frames must be split *by video*, not by individual frame, to prevent
data leakage between train and val/test sets (frames from the same video are
visually very similar).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def make_detection_split(
    items: list,
    groups: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list]:
    """Split *items* into train/val/test keeping all items from the same group together.

    Parameters
    ----------
    items:
        Arbitrary list of items (e.g. file paths as strings or Path objects).
    groups:
        Parallel list of group labels – one per item.  Items that share the
        same group label will always end up in the same split.
    train_ratio, val_ratio, test_ratio:
        Desired fractional sizes; must sum to 1.0 (tolerance 1e-6).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys "train", "val", "test", each containing a list of items.
    """
    if len(items) != len(groups):
        raise ValueError(
            f"Length mismatch: 'items' has {len(items)} elements but "
            f"'groups' has {len(groups)} elements. Both must have the same length."
        )

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must equal 1.0 (got {total})"
        )

    # Collect items per group
    group_to_items: dict[str, list] = defaultdict(list)
    for item, group in zip(items, groups):
        group_to_items[group].append(item)

    unique_groups = sorted(group_to_items.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    # test gets the remainder so rounding errors don't lose groups
    n_test = n - n_train - n_val

    train_groups = unique_groups[:n_train]
    val_groups = unique_groups[n_train : n_train + n_val]
    test_groups = unique_groups[n_train + n_val : n_train + n_val + n_test]

    split: dict[str, list] = {"train": [], "val": [], "test": []}
    for g in train_groups:
        split["train"].extend(group_to_items[g])
    for g in val_groups:
        split["val"].extend(group_to_items[g])
    for g in test_groups:
        split["test"].extend(group_to_items[g])

    return split


def save_split(split_dict: dict[str, list], path: str | Path) -> None:
    """Persist a split dictionary to a JSON file.

    Path objects are converted to strings for JSON serialisation.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serialisable = {
        split: [str(item) for item in items]
        for split, items in split_dict.items()
    }
    out_path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


def load_split(path: str | Path) -> dict[str, list[str]]:
    """Load a split dictionary previously saved with :func:`save_split`.

    Returns a dict with "train", "val", "test" keys mapping to lists of strings.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))
