"""
Retenție locală pentru fișierele de dovadă ale incidentelor.

Baza de date păstrează metadatele incidentelor, iar clipurile și miniaturile
sunt fișiere locale. Modulul șterge dovezile mai vechi decât perioada
configurată, astfel încât aplicația să nu ocupe spațiu nelimitat pe disc.
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

LITTERING_DIR = settings.littering_dir


def _safe_littering_path(relative_path: str | None) -> Path | None:
    """Rezolvă o cale salvată în DB, permițând ștergerea doar din LITTERING_DIR."""
    if not relative_path:
        return None

    root = LITTERING_DIR.resolve()
    raw = Path(str(relative_path))
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(root / raw)
        candidates.append(settings.REPO_ROOT / raw)
        if LITTERING_DIR.name in raw.parts:
            idx = len(raw.parts) - 1 - list(reversed(raw.parts)).index(LITTERING_DIR.name)
            suffix_parts = raw.parts[idx + 1:]
            if suffix_parts:
                candidates.append(root / Path(*suffix_parts))

    allowed: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        resolved_key = str(resolved)
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        allowed.append(resolved)

    if not allowed:
        logger.warning("Refuz ștergerea unei căi din afara directorului de dovezi: %s", relative_path)
        return None
    return next((path for path in allowed if path.exists()), allowed[0])


def _append_retention_note(existing: str | None, retention_days: int) -> str:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    note = (
        f"[sistem] Fișierele de dovadă au fost șterse automat după "
        f"{retention_days} zile de retenție la {stamp}."
    )
    if not existing:
        return note
    return f"{existing}\n{note}"


async def cleanup_littering_evidence(retention_days: int | None = None) -> dict[str, Any]:
    """
    Șterge clipurile/miniaturile vechi și păstrează rândurile incidentelor în DB.

    Întoarce un rezumat sigur pentru loguri sau endpoint-uri de mentenanță.
    Dacă retention_days <= 0, curățarea este considerată dezactivată.
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
                            logger.exception("Nu am putut șterge fișierul expirat: %s", full_path)
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
    logger.info("Curățare retenție stocare: %s", summary)
    return summary


async def storage_cleanup_loop() -> None:
    """Rulează curățarea periodică până la oprirea aplicației."""
    interval_hours = max(settings.STORAGE_CLEANUP_INTERVAL_HOURS, 1)
    interval_sec = interval_hours * 3600
    while True:
        await asyncio.sleep(interval_sec)
        try:
            await cleanup_littering_evidence()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Curățarea retenției de stocare a eșuat")
