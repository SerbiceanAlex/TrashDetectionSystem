"""
Smart notification system — 8 notification types with anti-spam.

Categories:
  verified  — "Raportul tău a fost verificat de comunitate!"
  rejected  — "Raportul tău a fost marcat ca fals"
  vote      — "Cineva a votat pe raportul tău"
  rank_up   — "Felicitări! Ai avansat la rangul Guardian!"
  streak    — "Streak de 7 zile! Multiplicator ×1.2"
  nearby    — "Gunoi raportat la 200m de tine"
  cleaned   — "Zona raportată de tine a fost curățată!"
  motivation— "Nu ai fost activ de 2 zile. Streak-ul se pierde mâine!"
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import Notification, User
from backend.ecoscore import get_streak_multiplier, RANKS


# ── Anti-spam: max notifications per user per day ────────────────────────────

MAX_NOTIFS_PER_DAY = 10

async def _count_today_notifs(session: AsyncSession, user_id: int) -> int:
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await session.scalar(
        select(func.count(Notification.id))
        .where(Notification.user_id == user_id)
        .where(Notification.created_at >= today_start)
    )
    return result or 0


async def _can_notify(session: AsyncSession, user_id: int) -> bool:
    count = await _count_today_notifs(session, user_id)
    return count < MAX_NOTIFS_PER_DAY


async def _has_recent_notif(session: AsyncSession, user_id: int, category: str,
                            session_id: int | None = None, hours: int = 1) -> bool:
    """Check if a similar notification was sent recently (dedup)."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    q = (
        select(func.count(Notification.id))
        .where(Notification.user_id == user_id)
        .where(Notification.category == category)
        .where(Notification.created_at >= cutoff)
    )
    if session_id is not None:
        q = q.where(Notification.session_id == session_id)
    count = await session.scalar(q)
    return (count or 0) > 0


async def send_notification(
    session: AsyncSession,
    user_id: int,
    message: str,
    category: str = "info",
    session_id: int | None = None,
    deduplicate_hours: int = 1,
) -> bool:
    """Send a notification with anti-spam and dedup checks.
    Returns True if notification was actually sent."""
    if not await _can_notify(session, user_id):
        return False
    if await _has_recent_notif(session, user_id, category, session_id, deduplicate_hours):
        return False
    session.add(Notification(
        user_id=user_id,
        message=message,
        category=category,
        session_id=session_id,
    ))
    return True


# ── Smart notification triggers ──────────────────────────────────────────────

async def notify_rank_up(session: AsyncSession, user_id: int, new_rank: str):
    """Notify user they've ranked up."""
    rank_emojis = {
        "Scout": "🔭", "Guardian": "🛡️", "Ranger": "🌲",
        "Champion": "🏆", "Legend": "🌟",
    }
    emoji = rank_emojis.get(new_rank, "🎉")
    # Find rank benefits
    benefits = ""
    for r in RANKS:
        if r["name"] == new_rank:
            benefits = ", ".join(r["benefits"])
            break
    msg = f"{emoji} Felicitări! Ai avansat la rangul {new_rank}! Noi beneficii: {benefits}"
    await send_notification(session, user_id, msg, "rank_up", deduplicate_hours=24)


async def notify_streak_milestone(session: AsyncSession, user_id: int, streak_days: int):
    """Notify user on streak milestones (4, 7, 14, 21, 30 days)."""
    milestones = {4: "🔥", 7: "🔥🔥", 14: "🔥🔥", 21: "🔥🔥🔥", 30: "🔥🔥🔥"}
    if streak_days not in milestones:
        return
    multiplier = get_streak_multiplier(streak_days)
    emoji = milestones[streak_days]
    msg = f"{emoji} Streak de {streak_days} zile! Multiplicator activ: ×{multiplier}"
    await send_notification(session, user_id, msg, "streak", deduplicate_hours=24)


async def notify_vote_on_report(session: AsyncSession, reporter_id: int, session_id: int, vote_type: str):
    """Notify reporter that someone voted on their report."""
    if vote_type == "confirm":
        msg = f"✅ Cineva a confirmat raportul tău #{session_id}!"
    else:
        msg = f"🚩 Cineva a contestat raportul tău #{session_id}."
    await send_notification(session, reporter_id, msg, "vote", session_id=session_id, deduplicate_hours=1)


async def notify_nearby_report(session: AsyncSession, user_id: int, session_id: int, distance_m: int):
    """Notify user about a new report near their location."""
    msg = f"📍 Gunoi raportat la ~{distance_m}m de locația ta! Raport #{session_id}"
    await send_notification(session, user_id, msg, "nearby", session_id=session_id, deduplicate_hours=6)


async def notify_report_cleaned(session: AsyncSession, reporter_id: int, session_id: int):
    """Notify reporter that their reported area was cleaned."""
    msg = f"🎉 Zona raportată de tine (#{session_id}) a fost curățată!"
    await send_notification(session, reporter_id, msg, "cleaned", session_id=session_id, deduplicate_hours=24)


async def notify_report_verified(session: AsyncSession, reporter_id: int, session_id: int, pts: int):
    """Notify reporter that their report was verified."""
    msg = f"✅ Raportul tău #{session_id} a fost verificat de comunitate! +{pts} EcoScore"
    await send_notification(session, reporter_id, msg, "verified", session_id=session_id, deduplicate_hours=24)


async def notify_report_rejected(session: AsyncSession, reporter_id: int, session_id: int):
    """Notify reporter that their report was marked as fake."""
    msg = f"🚫 Raportul tău #{session_id} a fost marcat ca fals de comunitate."
    await send_notification(session, reporter_id, msg, "rejected", session_id=session_id, deduplicate_hours=24)


async def notify_motivation(session: AsyncSession, user: User):
    """Send motivation notification if user is about to lose their streak.
    Called from a background check (not real-time)."""
    if not user.last_active_date or not user.streak_days or user.streak_days < 3:
        return
    from datetime import date
    days_since = (date.today() - user.last_active_date).days
    if days_since == 2:  # About to lose streak tomorrow
        msg = (
            f"⚠️ Nu ai fost activ de 2 zile! Streak-ul tău de {user.streak_days} zile "
            f"(×{get_streak_multiplier(user.streak_days)}) se pierde mâine!"
        )
        await send_notification(session, user.id, msg, "motivation", deduplicate_hours=24)
