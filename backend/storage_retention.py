"""
Local storage retention for littering evidence files.

The database keeps incident metadata, but video clips and thumbnails are local
files. This module deletes old evidence files after a configurable number of
days so a live deployment can run for weeks without unbounded disk growth.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select

from backend import database as db
from backend.config import settings

logger = logging.getLogger(__name__)

LITTERING_DIR = settings.REPO_ROOT / "backend" / "littering"


def _safe_littering_path(relative_path: str | None) -> Path | None:
    if not relative_path:
        return None

    root = LITTERING_DIR.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        logger.warning("Refusing to delete path outside littering dir: %s", candidate)
        return None
    return candidate


def _append_retention_note(existing: str | None, retention_days: int) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    note = (
        f"[system] Evidence files auto-deleted after "
        f"{retention_days} days retention at {stamp}."
    )
    if not existing:
        return note
    return f"{existing}\n{note}"


async def cleanup_littering_evidence(retention_days: int | None = None) -> dict[str, Any]:
    """
    Delete old littering clips/thumbnails and keep the DB incident rows.

    Returns a small summary dict that is safe to log or expose in a maintenance
    endpoint. If retention_days <= 0, the cleanup is treated as disabled.
    """
    days = settings.LITTERING_FILE_RETENTION_DAYS if retention_days is None else retention_days
    if days <= 0:
        return {
            "enabled": False,
            "retention_days": days,
            "events_scanned": 0,
            "files_deleted": 0,
            "bytes_deleted": 0,
        }

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    files_deleted = 0
    bytes_deleted = 0
    events_scanned = 0
    events_updated = 0

    async with db.AsyncSessionLocal() as session:
        result = await session.execute(
            select(db.LitteringEvent).where(
                db.LitteringEvent.detected_at < cutoff,
                or_(
                    db.LitteringEvent.clip_path.is_not(None),
                    db.LitteringEvent.thumbnail_path.is_not(None),
                ),
            )
        )
        events = result.scalars().all()
        events_scanned = len(events)

        for event in events:
            changed = False
            for attr in ("clip_path", "thumbnail_path"):
                relative = getattr(event, attr)
                full_path = _safe_littering_path(relative)
                if full_path is not None:
                    if full_path.exists():
                        try:
                            size = full_path.stat().st_size
                            full_path.unlink()
                            files_deleted += 1
                            bytes_deleted += size
                        except OSError:
                            logger.exception("Failed to delete retention file: %s", full_path)
                            continue
                    setattr(event, attr, None)
                    changed = True

            if changed:
                event.notes = _append_retention_note(event.notes, days)
                events_updated += 1

        if events_updated:
            await session.commit()

    summary = {
        "enabled": True,
        "retention_days": days,
        "cutoff": cutoff.isoformat(timespec="seconds"),
        "events_scanned": events_scanned,
        "events_updated": events_updated,
        "files_deleted": files_deleted,
        "bytes_deleted": bytes_deleted,
        "mb_deleted": round(bytes_deleted / (1024 * 1024), 3),
    }
    logger.info("Storage retention cleanup: %s", summary)
    return summary


async def storage_cleanup_loop() -> None:
    """Run cleanup once per configured interval until the app shuts down."""
    interval_hours = max(settings.STORAGE_CLEANUP_INTERVAL_HOURS, 1)
    interval_sec = interval_hours * 3600
    while True:
        await asyncio.sleep(interval_sec)
        try:
            await cleanup_littering_evidence()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Storage retention cleanup failed")
