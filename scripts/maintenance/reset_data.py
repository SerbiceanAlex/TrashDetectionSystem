"""
Reset generated detection data and clear runtime database rows.
Keeps user accounts intact.

Usage:
    python -m scripts.maintenance.reset_data
"""

import asyncio
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIRS_TO_CLEAN = [
    ROOT / "backend" / "uploads",
    ROOT / "backend" / "annotated",
    ROOT / "backend" / "cleaned",
    ROOT / "backend" / "videos",
    ROOT / "backend" / "littering",
    ROOT / "backend" / "thumbnails",
    ROOT / "backend" / "avatars",
]


async def reset_database():
    from backend.database import engine, sa_text

    tables = [
        "detection_records",
        "video_sessions",
        "detection_sessions",
        "littering_events",
    ]
    async with engine.begin() as conn:
        for table in tables:
            try:
                await conn.execute(sa_text(f"DELETE FROM {table}"))
                print(f"  Cleared table: {table}")
            except Exception:
                pass  # Table may not exist yet

    # Reset notification counters but keep user accounts
    async with engine.begin() as conn:
        try:
            await conn.execute(sa_text("DELETE FROM notifications"))
            print("  Cleared table: notifications")
        except Exception:
            pass


def reset_files():
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
        print(f"  Deleted {count} items from {d.name}/")
    print(f"  Total files deleted: {total_deleted}")


def main():
    print("=== TrashDet Data Reset ===")
    print()
    print("[1/2] Cleaning file directories...")
    reset_files()
    print()
    print("[2/2] Clearing database tables...")
    asyncio.run(reset_database())
    print()
    print("Done! All detection data has been cleared.")
    print("User accounts have been preserved.")


if __name__ == "__main__":
    main()
