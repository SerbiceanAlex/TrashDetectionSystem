"""
resplit_a4_dataset.py
=====================
Re-imparte dataset-ul parks_detect_A4 la 80/10/10 (train/val/test).

Ia TOATE cele 2307 imagini din splitul curent (62.9/18.6/18.5),
le amesteca cu seed=42 si le re-distribuie la proportia standard ML.

Rezultat:
    Train : 1845 imagini  (80.0%)
    Val   :  231 imagini  (10.0%)
    Test  :  231 imagini  (10.0%)

Rulare:
    python scripts/resplit_a4_dataset.py
"""

import random
import shutil
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parents[1]
A4_DIR     = REPO_ROOT / "datasets" / "parks_detect_A4"
SEED       = 42
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# TEST_FRAC = 1.0 - TRAIN_FRAC - VAL_FRAC = 0.10

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    backup = REPO_ROOT / "datasets" / "parks_detect_A4_backup_6318"
    # ── 1. Colecteaza toate perechile din backup (daca exista) sau A4_DIR ────
    src_root = backup if backup.exists() else A4_DIR
    print(f"Sursa: {src_root}")
    all_pairs = []
    for split in SPLITS:
        img_dir = src_root / "images" / split
        lbl_dir = src_root / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                print(f"  SKIP (no label): {img_path.name}")
                continue
            all_pairs.append((img_path, lbl_path))

    total = len(all_pairs)
    print(f"Total perechi (img, label): {total}")

    # ── 2. Shuffle reproducibil ─────────────────────────────────────────────
    random.seed(SEED)
    random.shuffle(all_pairs)

    # ── 3. Calculeaza dimensiunile ───────────────────────────────────────────
    n_train = int(total * TRAIN_FRAC)
    n_val   = int(total * VAL_FRAC)
    n_test  = total - n_train - n_val

    split_pairs = {
        "train": all_pairs[:n_train],
        "val":   all_pairs[n_train: n_train + n_val],
        "test":  all_pairs[n_train + n_val:],
    }

    print(f"Nou split 80/10/10:")
    print(f"  Train: {n_train} ({100*n_train/total:.1f}%)")
    print(f"  Val  : {n_val}   ({100*n_val/total:.1f}%)")
    print(f"  Test : {n_test}  ({100*n_test/total:.1f}%)")

    # ── 4. Backup structura curenta ─────────────────────────────────────────
    backup = REPO_ROOT / "datasets" / "parks_detect_A4_backup_6318"
    if backup.exists():
        print(f"\nBackup exista deja: {backup}")
    else:
        print(f"\nCreez backup: {backup}")
        shutil.copytree(A4_DIR, backup)
        print("Backup creat.")

    # ── 5. Curata directoarele target ────────────────────────────────────────
    for split in SPLITS:
        for sub in ["images", "labels"]:
            d = A4_DIR / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    # ── 6. Copiaza in noile locatii (cu prefix original split pentru unicitate) ─
    for new_split, pairs in split_pairs.items():
        img_dst = A4_DIR / "images" / new_split
        lbl_dst = A4_DIR / "labels" / new_split
        copied = 0
        used_names = set()
        for img_src, lbl_src in pairs:
            name = img_src.name
            # Daca exista coliziune de nume, adauga prefix unic
            if name in used_names:
                orig_split = img_src.parent.parent.name  # train/val/test din backup
                name = f"{orig_split}_{name}"
                stem = img_src.stem
                lbl_name = f"{orig_split}_{stem}.txt"
            else:
                lbl_name = lbl_src.name
            used_names.add(name)
            shutil.copy2(img_src, img_dst / name)
            shutil.copy2(lbl_src, lbl_dst / lbl_name)
            copied += 1
        print(f"  {new_split}: {copied} perechi copiate")

    # ── 7. Verifica ─────────────────────────────────────────────────────────
    print("\nVerificare finala:")
    for split in SPLITS:
        n_imgs = sum(1 for p in (A4_DIR / "images" / split).iterdir() if p.suffix.lower() in IMAGE_EXTS)
        n_lbls = len(list((A4_DIR / "labels" / split).glob("*.txt")))
        print(f"  {split}: {n_imgs} imgs, {n_lbls} labels {'OK' if n_imgs == n_lbls else 'MISMATCH'}")

    print("\nDataset re-split la 80/10/10. Acum re-antreneaza A4:")
    print("  Deschide notebooks/training/01b_train_detector_A4.ipynb")
    print("  Ruleaza celulele de antrenare (Section 8 sau A4 training cell)")


if __name__ == "__main__":
    main()
