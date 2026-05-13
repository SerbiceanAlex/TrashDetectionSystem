"""
SQLAlchemy 2.0 async database layer.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    and_,
    func,
    or_,
    select,
    text as sa_text,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

from backend.config import settings

engine = create_async_engine(settings.db_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Organization(Base):
    """Client organization / tenant."""

    __tablename__ = "organizations"

    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String(200), nullable=False)
    plan         = Column(String(20), default="trial")  # trial/starter/pro/enterprise
    trial_ends_at = Column(DateTime, nullable=True)
    subscription_active = Column(Boolean, default=True)
    stripe_customer_id   = Column(String(100), nullable=True)
    max_cameras          = Column(Integer, default=1)
    max_incidents_month  = Column(Integer, default=500)
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    users = relationship("User", back_populates="organization")


class User(Base):
    """Platform users."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), default="user") # 'user' or 'admin'
    points = Column(Integer, default=0)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    organization = relationship("Organization", back_populates="users")
    reports = relationship("DetectionSession", foreign_keys="[DetectionSession.reporter_id]", back_populates="reporter")
    resolutions = relationship("DetectionSession", foreign_keys="[DetectionSession.resolver_id]", back_populates="resolver")


class DetectionSession(Base):
    """One row per uploaded image."""

    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    image_path = Column(Text, nullable=True)       # original saved
    annotated_path = Column(Text, nullable=True)   # annotated saved
    total_objects = Column(Integer, default=0)
    inference_ms = Column(Float, default=0.0)
    latitude = Column(Float, nullable=True)        # GPS coordinates
    longitude = Column(Float, nullable=True)
    address = Column(Text, nullable=True)          # reverse-geocoded address
    gps_source = Column(String(16), nullable=True) # 'exif' | 'browser' | 'manual'
    is_resolved = Column(Integer, default=0)       # 0=dirty, 1=cleaned
    resolved_at = Column(DateTime, nullable=True)

    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    reporter_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    resolver_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    reporter = relationship("User", foreign_keys=[reporter_id], back_populates="reports")
    resolver = relationship("User", foreign_keys=[resolver_id], back_populates="resolutions")

    records = relationship(
        "DetectionRecord", back_populates="session", cascade="all, delete-orphan"
    )


class DetectionRecord(Base):
    """One row per detected object (bounding box)."""

    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=False)
    material = Column(String(64), nullable=False)
    det_score = Column(Float, nullable=False)
    cls_score = Column(Float, nullable=False)
    box_x1 = Column(Integer, nullable=False)
    box_y1 = Column(Integer, nullable=False)
    box_x2 = Column(Integer, nullable=False)
    box_y2 = Column(Integer, nullable=False)
    estimated_weight_kg = Column(Float, default=0.0)

    session = relationship("DetectionSession", back_populates="records")




class VideoSession(Base):
    """One row per video stream / upload."""

    __tablename__ = "video_sessions"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(16), nullable=False)       # "webcam" or "upload"
    filename = Column(String(255), nullable=True)
    start_time = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_sec = Column(Float, default=0.0)
    total_frames = Column(Integer, default=0)
    total_objects = Column(Integer, default=0)
    avg_fps = Column(Float, default=0.0)
    avg_inference_ms = Column(Float, default=0.0)
    materials_summary = Column(Text, nullable=True)         # JSON string
    video_path = Column(Text, nullable=True)                # original upload
    annotated_video_path = Column(Text, nullable=True)      # annotated output
    status = Column(String(16), default="running")          # running / completed / failed
    frames_processed = Column(Integer, default=0)           # progress tracking
    total_frames_expected = Column(Integer, default=0)      # total frames in source video
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)


class Notification(Base):
    """In-app notification for a user."""

    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    message = Column(Text, nullable=False)
    category = Column(String(32), default="info")   # 'incident' | 'review' | 'info'
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=True)
    is_read = Column(Integer, default=0)             # 0=unread, 1=read
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", foreign_keys=[user_id])
    session = relationship("DetectionSession", foreign_keys=[session_id])


class AuthorityContact(Base):
    """External authority/municipality contact for report forwarding."""

    __tablename__ = "authority_contacts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), nullable=False)
    area_description = Column(Text, nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class WebhookConfig(Base):
    """Webhook endpoint config — fires on report lifecycle events."""

    __tablename__ = "webhook_configs"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(Text, nullable=False)
    secret = Column(String(128), nullable=False)
    events = Column(Text, default="verified")
    active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))




class LitteringEvent(Base):
    """
    Records a detected illegal-dumping event.

    Populated by the /ws/video/monitor WebSocket endpoint when the
    LitteringDetector state machine fires an event (new trash object appears
    in the zone recently vacated by a tracked person).

    Lifecycle (status field):
        pending   → event detected, awaiting admin review
        reviewed  → admin has opened the record
        forwarded → evidence packet emailed to authority contact
        dismissed → admin marked as false positive
    """

    __tablename__ = "littering_events"

    id              = Column(Integer, primary_key=True, index=True)

    # ── When / where ────────────────────────────────────────────────────────
    detected_at     = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    latitude        = Column(Float, nullable=True)
    longitude       = Column(Float, nullable=True)
    address         = Column(Text, nullable=True)

    # ── What was detected ────────────────────────────────────────────────────
    material        = Column(String(32), nullable=False, default="unknown")
    det_score       = Column(Float, default=0.0)   # trash detector confidence
    person_present  = Column(Integer, default=1)   # 1 = yes, 0 = no
    person_count    = Column(Integer, default=1)

    # ── Privacy ──────────────────────────────────────────────────────────────
    face_blurred    = Column(Integer, default=1)   # always 1 — blur applied before storage

    # ── Evidence files ───────────────────────────────────────────────────────
    clip_path       = Column(Text, nullable=True)       # relative path to .mp4 clip
    thumbnail_path  = Column(Text, nullable=True)       # relative path to thumbnail .jpg
    image_hash      = Column(String(64), nullable=True) # SHA-256 of thumbnail (chain of custody)

    # ── Distance-based evidence (v2 state machine) ───────────────────────────
    incident_uid            = Column(String(36), nullable=True, index=True)  # UUID chain-of-custody
    owner_person_id         = Column(Integer, nullable=True)   # ByteTrack ID al persoanei
    distance_at_abandonment = Column(Float, nullable=True)     # distanța estimată în metri la momentul ABANDONED
    detection_method        = Column(String(32), default="zone")  # "zone" | "distance"

    # ── Workflow ─────────────────────────────────────────────────────────────
    status          = Column(String(16), default="pending", index=True)
    # "pending" | "reviewed" | "forwarded" | "dismissed"
    reviewed_by     = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at     = Column(DateTime, nullable=True)
    forwarded_at    = Column(DateTime, nullable=True)
    notes           = Column(Text, nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)

    # ── Relationships ────────────────────────────────────────────────────────
    reviewer = relationship("User", foreign_keys=[reviewed_by])


class MonitoredLocation(Base):
    """
    Locație fizică monitorizată (parcare mall, campus, etc.) cu cameră IP/RTSP.
    """
    __tablename__ = "monitored_locations"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String(128), nullable=False)
    address     = Column(Text, nullable=True)
    latitude    = Column(Float, nullable=True)
    longitude   = Column(Float, nullable=True)
    rtsp_url    = Column(Text, nullable=True)
    alert_email = Column(String(128), nullable=True)
    is_active   = Column(Integer, default=1)
    created_at  = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    created_by  = Column(Integer, ForeignKey("users.id"), nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)

    creator = relationship("User", foreign_keys=[created_by])


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ── Organization helpers ──────────────────────────────────────────────────────

async def get_org_by_id(db: AsyncSession, org_id: int) -> "Organization | None":
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    return result.scalar_one_or_none()


async def get_or_create_default_org(db: AsyncSession) -> "Organization":
    """Return org id=1, creating it if it doesn't exist."""
    org = await get_org_by_id(db, 1)
    if org is None:
        from datetime import timedelta
        org = Organization(
            id=1,
            name="Default Organization",
            plan="trial",
            trial_ends_at=datetime.now(timezone.utc) + timedelta(days=14),
        )
        db.add(org)
        await db.commit()
        await db.refresh(org)
    return org


async def create_organization(db: AsyncSession, name: str) -> "Organization":
    from datetime import timedelta
    org = Organization(
        name=name,
        plan="trial",
        trial_ends_at=datetime.now(timezone.utc) + timedelta(days=14),
    )
    db.add(org)
    await db.commit()
    await db.refresh(org)
    return org


# ── Query helpers ────────────────────────────────────────────────────────────


async def get_global_stats(db: AsyncSession):
    """Returns (total_sessions, total_objects, avg_inference_ms)."""
    result = await db.execute(
        select(
            func.count(DetectionSession.id),
            func.coalesce(func.sum(DetectionSession.total_objects), 0),
            func.coalesce(func.avg(DetectionSession.inference_ms), 0.0),
        )
    )
    return result.one()




# ── Video session helpers ──────────────────────────────────────────────────

async def create_video_session(db: AsyncSession, source_type: str, filename: str | None = None) -> VideoSession:
    vs = VideoSession(source_type=source_type, filename=filename)
    db.add(vs)
    await db.flush()
    await db.refresh(vs)
    return vs


async def finish_video_session(
    db: AsyncSession,
    session_id: int,
    *,
    total_frames: int,
    total_objects: int,
    avg_fps: float,
    avg_inference_ms: float,
    duration_sec: float,
    materials_summary: str,
    annotated_video_path: str | None = None,
    status: str = "completed",
):
    vs = (await db.execute(select(VideoSession).where(VideoSession.id == session_id))).scalar_one_or_none()
    if vs is None:
        return
    vs.end_time = datetime.now(timezone.utc)
    vs.total_frames = total_frames
    vs.total_objects = total_objects
    vs.avg_fps = round(avg_fps, 1)
    vs.avg_inference_ms = round(avg_inference_ms, 1)
    vs.duration_sec = round(duration_sec, 1)
    vs.materials_summary = materials_summary
    vs.status = status
    vs.frames_processed = total_frames  # mark as 100% done
    if annotated_video_path:
        vs.annotated_video_path = annotated_video_path
    await db.commit()


async def update_video_progress(db: AsyncSession, session_id: int, frames_processed: int, total_frames_expected: int = 0):
    vs = (await db.execute(select(VideoSession).where(VideoSession.id == session_id))).scalar_one_or_none()
    if vs is None:
        return
    vs.frames_processed = frames_processed
    if total_frames_expected:
        vs.total_frames_expected = total_frames_expected
    await db.commit()


async def get_video_session_by_id(db: AsyncSession, session_id: int):
    result = await db.execute(select(VideoSession).where(VideoSession.id == session_id))
    return result.scalar_one_or_none()


# ── LitteringEvent CRUD ───────────────────────────────────────────────────────

async def create_littering_event(
    db: AsyncSession,
    *,
    material: str,
    det_score: float,
    person_present: bool = True,
    person_count: int = 1,
    latitude: float | None = None,
    longitude: float | None = None,
    address: str | None = None,
    clip_path: str | None = None,
    thumbnail_path: str | None = None,
    image_hash: str | None = None,
    incident_uid: str | None = None,
    owner_person_id: int | None = None,
    distance_at_abandonment: float | None = None,
    detection_method: str = "zone",
) -> "LitteringEvent":
    evt = LitteringEvent(
        material=material,
        det_score=round(det_score, 4),
        person_present=1 if person_present else 0,
        person_count=person_count,
        latitude=latitude,
        longitude=longitude,
        address=address,
        clip_path=clip_path,
        thumbnail_path=thumbnail_path,
        image_hash=image_hash,
        face_blurred=1,
        status="pending",
        incident_uid=incident_uid,
        owner_person_id=owner_person_id,
        distance_at_abandonment=round(distance_at_abandonment, 3) if distance_at_abandonment is not None else None,
        detection_method=detection_method,
    )
    db.add(evt)
    await db.commit()
    await db.refresh(evt)
    return evt


async def list_littering_events(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 20,
    status: str | None = None,
    material: str | None = None,
    org_id: int | None = None,
) -> tuple[list["LitteringEvent"], int]:
    q = select(LitteringEvent).order_by(LitteringEvent.detected_at.desc())
    if org_id is not None:
        q = q.where(or_(LitteringEvent.organization_id == org_id, LitteringEvent.organization_id.is_(None)))
    if status:
        q = q.where(LitteringEvent.status == status)
    if material:
        q = q.where(LitteringEvent.material == material)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    items = (await db.execute(q.offset(skip).limit(limit))).scalars().all()
    return list(items), total


async def get_littering_event_by_id(db: AsyncSession, event_id: int) -> "LitteringEvent | None":
    result = await db.execute(select(LitteringEvent).where(LitteringEvent.id == event_id))
    return result.scalar_one_or_none()


async def update_littering_event_status(
    db: AsyncSession,
    event_id: int,
    status: str,
    reviewed_by: int | None = None,
    notes: str | None = None,
) -> "LitteringEvent | None":
    evt = await get_littering_event_by_id(db, event_id)
    if evt is None:
        return None
    evt.status = status
    if reviewed_by is not None:
        evt.reviewed_by = reviewed_by
        evt.reviewed_at = datetime.now(timezone.utc)
    if status == "forwarded":
        evt.forwarded_at = datetime.now(timezone.utc)
    if notes is not None:
        evt.notes = notes
    await db.commit()
    await db.refresh(evt)
    return evt


async def get_video_sessions_paginated(
    db: AsyncSession, skip: int, limit: int,
    org_id: int | None = None, user_id: int | None = None,
):
    q = select(VideoSession).order_by(VideoSession.start_time.desc())
    if org_id is not None:
        q = q.where(or_(VideoSession.organization_id == org_id, VideoSession.organization_id.is_(None)))
    if user_id is not None:
        q = q.where(VideoSession.user_id == user_id)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    items = (await db.execute(q.offset(skip).limit(limit))).scalars().all()
    return items, total
