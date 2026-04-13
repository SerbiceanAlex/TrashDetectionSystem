"""
Reset all detection data — clear uploads, annotated, cleaned, videos directories
and delete all session/record/vote/comment rows from the database.
Keeps user accounts intact.

Usage:
    python -m scripts.reset_data
"""

import asyncio
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
DIRS_TO_CLEAN = [
    ROOT / "backend" / "uploads",
    ROOT / "backend" / "annotated",
    ROOT / "backend" / "cleaned",
    ROOT / "backend" / "videos",
    ROOT / "backend" / "thumbnails",
    ROOT / "backend" / "avatars",
]


async def reset_database():
    from backend.database import engine, sa_text

    tables = [
        "comments",
        "material_suggestions",
        "community_votes",
        "campaign_participants",
        "report_photos",
        "detection_records",
        "video_sessions",
        "detection_sessions",
        "webhook_configs",
        "authority_contacts",
        "campaigns",
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
    total_deleted = 0
    for d in DIRS_TO_CLEAN:
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
