import argparse
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_split(src_root: Path, dst_root: Path, prefix: str, split: str):
    src_img_dir = src_root / "images" / split
    src_lbl_dir = src_root / "labels" / split

    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0

    for img_path in src_img_dir.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue

        new_stem = f"{prefix}_{img_path.stem}"
        dst_img_path = dst_img_dir / f"{new_stem}{img_path.suffix}"
        src_lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        dst_lbl_path = dst_lbl_dir / f"{new_stem}.txt"

        shutil.copy2(img_path, dst_img_path)

        if src_lbl_path.exists():
            shutil.copy2(src_lbl_path, dst_lbl_path)
        else:
            dst_lbl_path.write_text("", encoding="utf-8")

        copied += 1

    print(f"[{prefix}] {split}: copied {copied} images")


def main():
    ap = argparse.ArgumentParser(description="Merge multiple YOLO datasets into one final dataset")
    ap.add_argument("--sources", nargs="+", required=True, help="List of YOLO dataset roots")
    ap.add_argument("--names", nargs="+", required=True, help="Prefixes for each dataset source")
    ap.add_argument("--out", required=True, help="Output merged YOLO dataset root")
    args = ap.parse_args()

    if len(args.sources) != len(args.names):
        raise ValueError("The number of --sources must match the number of --names")

    out_root = Path(args.out)

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for src, name in zip(args.sources, args.names):
        src_root = Path(src)
        for split in ("train", "val", "test"):
            copy_split(src_root, out_root, name, split)

    print(f"[DONE] Merged datasets into: {out_root}")


if __name__ == "__main__":
    main()