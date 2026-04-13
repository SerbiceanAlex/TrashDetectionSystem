"""
EcoScore engine — ranks, streaks, trust weights, verification thresholds.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


def _utc(dt: datetime) -> datetime:
    """Ensure datetime is offset-aware (UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

from backend.database import CommunityVote, DetectionSession, User

# ── Rank definitions ──────────────────────────────────────────────────────────

RANKS = [
    {"name": "Novice",   "min": 0,    "max": 49,   "trust": 1.0, "benefits": ["Poate raporta și vota"]},
    {"name": "Scout",    "min": 50,   "max": 199,  "trust": 1.0, "benefits": ["Nearby alerts"]},
    {"name": "Guardian", "min": 200,  "max": 499,  "trust": 1.2, "benefits": ["Sugestii material", "Vot ×1.2"]},
    {"name": "Ranger",   "min": 500,  "max": 999,  "trust": 1.5, "benefits": ["Claim cleanup", "Vot ×1.5"]},
    {"name": "Champion", "min": 1000, "max": 2499, "trust": 1.8, "benefits": ["Achievements vizibile", "Vot ×1.8"]},
    {"name": "Legend",   "min": 2500, "max": None, "trust": 2.0, "benefits": ["Badge special", "Vot ×2.0"]},
]

# ── Streak multiplier tiers ──────────────────────────────────────────────────

STREAK_TIERS = [
    (30,  2.0),
    (22,  1.8),
    (15,  1.6),
    (8,   1.4),
    (4,   1.2),
    (0,   1.0),
]

# ── Point values ─────────────────────────────────────────────────────────────

POINTS_REPORT_PER_OBJECT = 10
POINTS_VOTE = 2
POINTS_REPORT_VERIFIED = 15
POINTS_CLEANUP = 25
POINTS_MATERIAL_CORRECTION = 5
POINTS_CAMPAIGN_REPORT = 15

# ── Impact metrics — estimated weight per detected item (kg) ─────────────────

WEIGHT_PER_ITEM = {
    "plastic":  0.025,   # avg PET bottle
    "glass":    0.350,   # avg glass bottle
    "metal":    0.015,   # avg aluminium can
    "paper":    0.005,   # avg paper scrap
    "other":    0.020,   # misc small item
}

# ── CO2 equivalents per kg of material recycled vs landfill (kg CO2) ─────────

CO2_PER_KG = {
    "plastic":  6.0,
    "glass":    0.87,
    "metal":    1.83,
    "paper":    1.1,
    "other":    1.0,
}


def get_rank_for_score(eco_score: int) -> str:
    for r in reversed(RANKS):
        if eco_score >= r["min"]:
            return r["name"]
    return "Novice"


def get_trust_weight_for_rank(rank_name: str) -> float:
    for r in RANKS:
        if r["name"] == rank_name:
            return r["trust"]
    return 1.0


def get_streak_multiplier(streak_days: int) -> float:
    for threshold, mult in STREAK_TIERS:
        if streak_days >= threshold:
            return mult
    return 1.0


def calculate_points(base_points: int, streak_days: int) -> int:
    """Apply streak multiplier to base points."""
    multiplier = get_streak_multiplier(streak_days)
    return int(base_points * multiplier)


async def update_user_streak(user: User, now: Optional[date] = None):
    """Update streak based on last_active_date. Call BEFORE awarding points."""
    today = now or date.today()
    if user.last_active_date is None:
        user.streak_days = 1
        user.last_active_date = today
        return

    delta = (today - user.last_active_date).days
    if delta == 0:
        # Already active today
        return
    elif delta == 1:
        # Consecutive day
        user.streak_days += 1
    elif delta <= 2:
        # Grace: 48h inactivity doesn't break streak but doesn't add
        pass
    else:
        # Streak broken
        user.streak_days = 1

    user.last_active_date = today


async def award_ecoscore(db_session: AsyncSession, user: User, base_points: int):
    """Award EcoScore points with streak multiplier, update rank and trust_weight."""
    await update_user_streak(user)
    actual_points = calculate_points(base_points, user.streak_days)
    user.eco_score += actual_points
    user.points += actual_points  # keep legacy points in sync

    new_rank = get_rank_for_score(user.eco_score)
    old_rank = user.rank or "Novice"
    user.rank = new_rank
    user.trust_weight = get_trust_weight_for_rank(new_rank)

    return actual_points, old_rank != new_rank, new_rank


async def get_verification_threshold(db_session: AsyncSession) -> float:
    """Dynamic threshold based on active users in last 30 days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    result = await db_session.execute(
        select(func.count(func.distinct(DetectionSession.reporter_id)))
        .where(DetectionSession.upload_time >= cutoff)
        .where(DetectionSession.reporter_id.is_not(None))
    )
    active_users = result.scalar_one() or 0

    if active_users < 10:
        return 2.0
    if active_users < 50:
        return 3.0
    if active_users < 200:
        return 5.0
    return 8.0


async def check_auto_verify(db_session: AsyncSession, session: DetectionSession):
    """Check if a session should be auto-verified based on votes."""
    if session.status != "pending":
        return False

    threshold = await get_verification_threshold(db_session)

    if session.verification_score >= threshold:
        session.status = "verified"
        return True

    # Auto-verify fallback: 72h with ≥1 confirm and 0 fake
    if session.upload_time:
        age = datetime.now(timezone.utc) - _utc(session.upload_time)
        if age >= timedelta(hours=72):
            result = await db_session.execute(
                select(
                    func.sum(func.iif(CommunityVote.vote_type == "confirm", 1, 0)).label("confirms"),
                    func.sum(func.iif(CommunityVote.vote_type == "fake", 1, 0)).label("fakes"),
                ).where(CommunityVote.session_id == session.id)
            )
            row = result.one()
            confirms = row.confirms or 0
            fakes = row.fakes or 0
            if confirms >= 1 and fakes == 0:
                session.status = "verified"
                return True

    return False


async def check_auto_expire(db_session: AsyncSession, session: DetectionSession):
    """Expire pending sessions with no activity after 72h."""
    if session.status != "pending":
        return False

    if session.upload_time:
        age = datetime.now(timezone.utc) - _utc(session.upload_time)
        if age >= timedelta(hours=72) and session.verification_score == 0.0:
            # No votes at all → expire
            result = await db_session.execute(
                select(func.count(CommunityVote.id))
                .where(CommunityVote.session_id == session.id)
            )
            vote_count = result.scalar_one() or 0
            if vote_count == 0:
                session.status = "expired"
                return True

    return False


def is_nearby(lat1: float, lon1: float, lat2: float, lon2: float, threshold_m: float = 50.0) -> bool:
    """Quick Haversine approximation — good enough for <1km distances."""
    import math
    R = 6371000  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    distance = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return distance <= threshold_m


async def find_nearby_pending(db_session: AsyncSession, lat: float, lon: float, exclude_id: int = 0):
    """Find a pending session within 50m to cluster with."""
    # Simple bounding box pre-filter (~0.0005° ≈ 55m at equator)
    delta = 0.0005
    result = await db_session.execute(
        select(DetectionSession)
        .where(DetectionSession.status == "pending")
        .where(DetectionSession.latitude.between(lat - delta, lat + delta))
        .where(DetectionSession.longitude.between(lon - delta, lon + delta))
        .where(DetectionSession.id != exclude_id)
        .where(DetectionSession.cluster_id.is_(None))
        .order_by(DetectionSession.upload_time.asc())
        .limit(1)
    )
    candidate = result.scalar_one_or_none()
    if candidate and is_nearby(lat, lon, candidate.latitude, candidate.longitude):
        return candidate
    return None
