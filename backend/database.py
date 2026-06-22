"""
Stratul de bază de date (SQLAlchemy 2.0, async).

Conține: modelele (tabelele) ORM, motorul + fabrica de sesiuni, și funcțiile
helper de acces la date (CRUD) folosite de rutele din main.py.

Notă: `and_`, `or_`, `func`, `select`, `text` sunt importate aici și folosite
și prin `db.<nume>` în main.py (re-export), ca rutele să nu importe direct din
sqlalchemy. De aceea unele apar „neutilizate" pentru linter, dar sunt necesare.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    and_,       # noqa: F401 — re-export pentru main.py (db.and_)
    func,
    or_,
    select,
    text as sa_text,   # noqa: F401 — re-export pentru main.py (db.sa_text)
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship, selectinload

from backend.config import settings

engine = create_async_engine(settings.db_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Organization(Base):
    """Organizația (tenant) care grupează utilizatorii și incidentele lor."""

    __tablename__ = "organizations"

    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String(200), nullable=False)
    created_at   = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    users = relationship("User", back_populates="organization")


class User(Base):
    """Utilizatorii platformei (rol 'admin' sau 'user'), legați de o organizație."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), default="user")  # 'user' sau 'admin'
    points = Column(Integer, default=0)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relații
    organization = relationship("Organization", back_populates="users")


class VideoSession(Base):
    """Un rând per video procesat (upload sau flux), cu statistici și progres."""

    __tablename__ = "video_sessions"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(16), nullable=False)       # "webcam" sau "upload"
    filename = Column(String(255), nullable=True)
    start_time = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_sec = Column(Float, default=0.0)
    total_frames = Column(Integer, default=0)
    total_objects = Column(Integer, default=0)
    littering_count = Column(Integer, default=0)
    avg_fps = Column(Float, default=0.0)
    avg_inference_ms = Column(Float, default=0.0)
    materials_summary = Column(Text, nullable=True)         # JSON ca text
    video_path = Column(Text, nullable=True)                # uploadul original
    annotated_video_path = Column(Text, nullable=True)      # videoul adnotat
    status = Column(String(16), default="running")          # running / completed / failed
    frames_processed = Column(Integer, default=0)           # urmărirea progresului
    total_frames_expected = Column(Integer, default=0)      # total cadre în sursă
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)


class Notification(Base):
    """Notificare în aplicație pentru un utilizator (incident, review, info)."""

    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    message = Column(Text, nullable=False)
    category = Column(String(32), default="info")   # 'incident' | 'reviewed' | 'info'
    event_id = Column(Integer, ForeignKey("littering_events.id"), nullable=True)  # incidentul vizat (pt. click)
    is_read = Column(Integer, default=0)             # 0=necitit, 1=citit
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", foreign_keys=[user_id])


class LitteringEvent(Base):
    """
    Records a detected illegal-dumping event.

    Populated by the /ws/video/monitor WebSocket endpoint when the
    LitteringDetector state machine fires an event (new trash object appears
    in the zone recently vacated by a tracked person).

    Lifecycle (status field):
        pending   → event detected, awaiting admin review
        reviewed  → admin has opened the record
        forwarded → evidence archived locally for reporting
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
    face_blurred    = Column(Integer, default=0)   # 0 = evidence is stored unblurred for review

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
    reporter_id     = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    reviewed_at     = Column(DateTime, nullable=True)
    forwarded_at    = Column(DateTime, nullable=True)
    notes           = Column(Text, nullable=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=True, index=True)

    # ── Relationships ────────────────────────────────────────────────────────
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    reporter = relationship("User", foreign_keys=[reporter_id])

    @property
    def reporter_username(self) -> str | None:
        return self.reporter.username if self.reporter else None


async def create_tables():
    """Creează tabelele lipsă în baza de date (rulat la pornire)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependență FastAPI: oferă o sesiune async de DB pe durata unei cereri."""
    async with AsyncSessionLocal() as session:
        yield session


# ── Organization helpers ──────────────────────────────────────────────────────

async def get_org_by_id(db: AsyncSession, org_id: int) -> "Organization | None":
    """Întoarce organizația după id (sau None dacă nu există)."""
    result = await db.execute(select(Organization).where(Organization.id == org_id))
    return result.scalar_one_or_none()


async def get_or_create_default_org(db: AsyncSession) -> "Organization":
    """Return org id=1, creating it if it doesn't exist."""
    org = await get_org_by_id(db, 1)
    if org is None:
        org = Organization(id=1, name="Default Organization")
        db.add(org)
        await db.commit()
        await db.refresh(org)
    return org


async def create_organization(db: AsyncSession, name: str) -> "Organization":
    """Creează o organizație nouă (pentru primul utilizator = admin propriu)."""
    org = Organization(name=name)
    db.add(org)
    await db.commit()
    await db.refresh(org)
    return org


# ── Video session helpers ──────────────────────────────────────────────────

async def create_video_session(db: AsyncSession, source_type: str, filename: str | None = None) -> VideoSession:
    """Deschide o sesiune video (la începutul procesării unui upload/flux)."""
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
    littering_count: int,
    avg_fps: float,
    avg_inference_ms: float,
    duration_sec: float,
    materials_summary: str,
    annotated_video_path: str | None = None,
    status: str = "completed",
):
    """Închide o sesiune video: salvează statisticile finale și marchează 100%."""
    vs = (await db.execute(select(VideoSession).where(VideoSession.id == session_id))).scalar_one_or_none()
    if vs is None:
        return
    vs.end_time = datetime.now(timezone.utc)
    vs.total_frames = total_frames
    vs.total_objects = total_objects
    vs.littering_count = littering_count
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
    """Actualizează progresul procesării unui video (bara de progres din UI)."""
    vs = (await db.execute(select(VideoSession).where(VideoSession.id == session_id))).scalar_one_or_none()
    if vs is None:
        return
    vs.frames_processed = frames_processed
    if total_frames_expected:
        vs.total_frames_expected = total_frames_expected
    await db.commit()


async def get_video_session_by_id(db: AsyncSession, session_id: int):
    """Întoarce o sesiune video după id (sau None)."""
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
    reporter_id: int | None = None,
    organization_id: int | None = None,
) -> "LitteringEvent":
    """Salvează un incident de aruncare detectat (apelat când se declanșează alerta)."""
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
        face_blurred=0,
        status="pending",
        incident_uid=incident_uid,
        owner_person_id=owner_person_id,
        distance_at_abandonment=round(distance_at_abandonment, 3) if distance_at_abandonment is not None else None,
        detection_method=detection_method,
        reporter_id=reporter_id,
        organization_id=organization_id,
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
    reporter_id: int | None = None,
) -> tuple[list["LitteringEvent"], int]:
    """
    Listează incidentele paginat, filtrabile după organizație, raportor, status
    și material. Întoarce (lista_pagina, total). Folosită de lista din UI.
    """
    q = select(LitteringEvent).options(selectinload(LitteringEvent.reporter)).order_by(LitteringEvent.detected_at.desc())
    if org_id is not None:
        q = q.where(or_(LitteringEvent.organization_id == org_id, LitteringEvent.organization_id.is_(None)))
    if reporter_id is not None:
        q = q.where(LitteringEvent.reporter_id == reporter_id)
    if status:
        q = q.where(LitteringEvent.status == status)
    if material:
        q = q.where(LitteringEvent.material == material)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    items = (await db.execute(q.offset(skip).limit(limit))).scalars().all()
    return list(items), total


async def get_littering_event_by_id(db: AsyncSession, event_id: int) -> "LitteringEvent | None":
    """Întoarce un incident după id, cu raportorul încărcat (sau None)."""
    result = await db.execute(
        select(LitteringEvent)
        .options(selectinload(LitteringEvent.reporter))
        .where(LitteringEvent.id == event_id)
    )
    return result.scalar_one_or_none()


async def update_littering_event_status(
    db: AsyncSession,
    event_id: int,
    status: str,
    reviewed_by: int | None = None,
    notes: str | None = None,
) -> "LitteringEvent | None":
    """
    Schimbă statusul unui incident (pending/reviewed/forwarded/dismissed) și
    setează cine/când l-a verificat. Folosită la validarea de către admin.
    """
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
    """Listează sesiunile video paginat, filtrabile pe organizație/utilizator."""
    q = select(VideoSession).order_by(VideoSession.start_time.desc())
    if org_id is not None:
        q = q.where(or_(VideoSession.organization_id == org_id, VideoSession.organization_id.is_(None)))
    if user_id is not None:
        q = q.where(VideoSession.user_id == user_id)
    total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
    items = (await db.execute(q.offset(skip).limit(limit))).scalars().all()
    return items, total


# ── Notificări ─────────────────────────────────────────────────────────────────

async def create_notification(
    db: AsyncSession, user_id: int, message: str,
    category: str = "info", event_id: int | None = None,
) -> "Notification | None":
    """
    Creează o notificare în aplicație pentru un utilizator (ex. la un incident nou).
    `event_id` leagă notificarea de un incident, ca să poată fi deschis la click.
    """
    if not user_id:
        return None
    notif = Notification(user_id=user_id, message=message, category=category, event_id=event_id)
    db.add(notif)
    await db.commit()
    await db.refresh(notif)
    return notif
