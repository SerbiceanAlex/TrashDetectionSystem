"""
Resetează datele generate la rulare și golește tabelele de runtime din DB.
Păstrează conturile de utilizator.

Utilizare:
    python -m scripts.maintenance.reset_data
"""

import sys
# Diacriticele românești se afișează corect indiferent de codarea consolei.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Foldere vechi (din versiuni anterioare) curățate defensiv dacă mai există.
LEGACY_DIRS_TO_CLEAN = [
    ROOT / "backend" / "uploads",
    ROOT / "backend" / "annotated",
    ROOT / "backend" / "videos",
    ROOT / "backend" / "littering",
]


async def reset_database():
    """Golește tabelele de runtime (incidente, sesiuni, notificări), păstrând conturile."""
    from backend.database import engine, sa_text

    tables = ["video_sessions", "littering_events", "notifications"]
    async with engine.begin() as conn:
        for table in tables:
            try:
                await conn.execute(sa_text(f"DELETE FROM {table}"))
                print(f"  Golit tabelul: {table}")
            except Exception:
                pass  # tabelul poate să nu existe încă


def reset_files():
    """Șterge fișierele generate din folderele de runtime (și cele vechi)."""
    from backend.config import settings

    total_deleted = 0
    dirs_to_clean = [*settings.runtime_dirs, *LEGACY_DIRS_TO_CLEAN]
    for d in dirs_to_clean:
        if not d.exists():
            continue
        count = 0
        for f in d.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
            elif f.is_dir() and f.name != "__pycache__":
                shutil.rmtree(f)
                count += 1
        total_deleted += count
        print(f"  Șterse {count} elemente din {d.name}/")
    print(f"  Total fișiere șterse: {total_deleted}")


def main():
    print("=== Resetare date TrashDet ===")
    print()
    print("[1/2] Curăț folderele de fișiere...")
    reset_files()
    print()
    print("[2/2] Golesc tabelele din baza de date...")
    asyncio.run(reset_database())
    print()
    print("Gata! Toate datele de detecție au fost șterse.")
    print("Conturile de utilizator au fost păstrate.")


if __name__ == "__main__":
    main()
