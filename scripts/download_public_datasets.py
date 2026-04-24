"""
download_public_datasets.py
============================
Descarcă și prepară dataset-uri publice de detecție a gunoiului de pe Roboflow.
Toate clasele sunt re-mapate la clasa unică 'trash' (single-class, ca A3-final).

Dataset-uri descărcate:
  1. "Cigarette Butt Detection" — obiecte mici (mucuri de tigara), esential pentru Faza 2
  2. "Trash Detection" (roboflow universe) — gunoi urban variat
  3. "Garbage Detection" — diverse tipuri de deseuri

Output: datasets/public_trash/ cu structura YOLO (images/ + labels/)

Rulare:
    .venv/Scripts/python.exe scripts/download_public_datasets.py

Nota: necesita un API key Roboflow GRATUIT (https://roboflow.com → sign up → Settings → API Key)
      sau descarca manual de pe https://universe.roboflow.com si pune in datasets/public_trash/
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "datasets" / "public_trash"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def remap_labels_to_single_class(dataset_dir: Path, out_dir: Path) -> int:
    """
    Copiaza toate imaginile si labelele unui dataset YOLO in out_dir,
    re-mapand orice clasa la clasa 0 (trash single-class).
    Returneaza numarul de imagini copiate.
    """
    copied = 0
    for split in ["train", "valid", "test"]:
        src_imgs  = dataset_dir / split / "images"
        src_lbls  = dataset_dir / split / "labels"
        dst_imgs  = out_dir / split / "images"
        dst_lbls  = out_dir / split / "labels"
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        if not src_imgs.exists():
            continue

        for img_file in src_imgs.glob("*"):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            lbl_file = src_lbls / img_file.with_suffix(".txt").name
            if not lbl_file.exists():
                continue

            # Remap label: orice clasa -> 0
            lines = lbl_file.read_text().strip().splitlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    new_lines.append("0 " + " ".join(parts[1:]))
            if not new_lines:
                continue

            # Copy image
            dst_img = dst_imgs / img_file.name
            shutil.copy2(img_file, dst_img)

            # Write remapped label
            dst_lbl = dst_lbls / lbl_file.name
            dst_lbl.write_text("\n".join(new_lines) + "\n")

            copied += 1

    return copied


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    out_name: str,
) -> Path | None:
    """Descarca un dataset Roboflow in format YOLOv8 si returneaza calea."""
    try:
        from roboflow import Roboflow

        rf       = Roboflow(api_key=api_key)
        project_ = rf.workspace(workspace).project(project)
        dataset  = project_.version(version).download("yolov8", location=str(REPO_ROOT / "datasets" / "raw_rf" / out_name))
        print(f"  Descarcat: {dataset.location}")
        return Path(dataset.location)
    except Exception as e:
        print(f"  EROARE la {workspace}/{project} v{version}: {e}")
        return None


def main() -> int:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("ROBOFLOW_API_KEY nu e setat.")
        print("")
        print("1. Du-te la https://roboflow.com si inregistreaza-te (cont gratuit).")
        print("2. Mergi la Settings -> API Key -> copiaza cheia.")
        print("3. Ruleaza:")
        print("   set ROBOFLOW_API_KEY=cheia_ta")
        print("   .venv\\Scripts\\python.exe scripts\\download_public_datasets.py")
        print("")
        print("Alternativ: descarca manual de pe https://universe.roboflow.com")
        print("si pune structura YOLO (images/+labels/) in datasets/public_trash/train/")
        print("")
        print("Dataset-uri recomandate:")
        print("  - universe.roboflow.com/cigarette-butts/cigarette-butt-detection")
        print("  - universe.roboflow.com/trash-detection-cwwf3/trash-detection-recycling")
        print("  - universe.roboflow.com/litter-detection/litter-detect")
        return 1

    print("Descarcam dataset-uri publice de pe Roboflow...")
    print(f"Output: {OUT_DIR}\n")

    # Lista dataset-uri: (workspace, project, version, out_name, descriere)
    datasets_to_download = [
        # Cigars / mucuri de tigara — obiect mic, esential
        ("cigarette-butts", "cigarette-butt-detection", 1, "cigarette_butts",
         "Cigarette Butt Detection — obiecte mici, mucuri tigara"),
        # Gunoi urban general
        ("trash-detection-cwwf3", "trash-detection-recycling", 3, "trash_recycling",
         "Trash Detection Recycling — gunoi urban variat"),
        # Gunoi in mediu exterior
        ("litter-detection", "litter-detect", 1, "litter_outdoor",
         "Litter Detection outdoor"),
    ]

    merged_count = 0

    for ws, proj, ver, out_name, desc in datasets_to_download:
        print(f"\n[{out_name}] {desc}")
        raw_path = download_roboflow_dataset(ws, proj, ver, api_key, out_name)
        if raw_path is None:
            print(f"  Sarit (eroare la descarcare).")
            continue

        print(f"  Remap clase -> clasa 0 (trash)...")
        n = remap_labels_to_single_class(raw_path, OUT_DIR)
        merged_count += n
        print(f"  {n} imagini adaugate in {OUT_DIR}")

    # Statistici finale
    print(f"\n{'='*55}")
    print("  SUMAR")
    print(f"{'='*55}")
    total = 0
    for split in ["train", "valid", "test"]:
        n = len(list((OUT_DIR / split / "images").glob("*"))) if (OUT_DIR / split / "images").exists() else 0
        print(f"  {split:6s}: {n:5d} imagini")
        total += n
    print(f"  TOTAL : {total:5d} imagini")
    print(f"\nDataset gata in: {OUT_DIR}")
    print("Urmatorul pas: notebooks/data/04_merge_public_datasets.ipynb")

    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
