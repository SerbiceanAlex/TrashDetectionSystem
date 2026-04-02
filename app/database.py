"""
SQLAlchemy 2.0 async database layer.
Database file: app/trash_detection.db
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, relationship

DB_PATH = Path(__file__).parent / "trash_detection.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    """Platform users."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), default="user") # 'user' or 'admin'
    points = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    reports = relationship("DetectionSession", foreign_keys="[DetectionSession.reporter_id]", back_populates="reporter")
    resolutions = relationship("DetectionSession", foreign_keys="[DetectionSession.resolver_id]", back_populates="resolver")


class DetectionSession(Base):
    """One row per uploaded image."""

    __tablename__ = "detection_sessions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
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

    session = relationship("DetectionSession", back_populates="records")


class VideoSession(Base):
    """One row per video stream / upload."""

    __tablename__ = "video_sessions"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(16), nullable=False)       # "webcam" or "upload"
    filename = Column(String(255), nullable=True)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
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


class Notification(Base):
    """In-app notification for a user."""

    __tablename__ = "notifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    message = Column(Text, nullable=False)
    category = Column(String(32), default="info")   # 'resolved' | 'info' | 'badge'
    session_id = Column(Integer, ForeignKey("detection_sessions.id"), nullable=True)
    is_read = Column(Integer, default=0)             # 0=unread, 1=read
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", foreign_keys=[user_id])
    session = relationship("DetectionSession", foreign_keys=[session_id])


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ── Query helpers ────────────────────────────────────────────────────────────

async def get_sessions_paginated(db: AsyncSession, skip: int, limit: int):
    result = await db.execute(
        select(DetectionSession)
        .order_by(DetectionSession.upload_time.desc())
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


async def count_sessions(db: AsyncSession) -> int:
    result = await db.execute(select(func.count()).select_from(DetectionSession))
    return result.scalar_one()


async def get_session_by_id(db: AsyncSession, session_id: int):
    result = await db.execute(
        select(DetectionSession).where(DetectionSession.id == session_id)
    )
    return result.scalar_one_or_none()


async def get_material_stats(db: AsyncSession):
    """Returns list of (material, count) ordered by count desc."""
    result = await db.execute(
        select(DetectionRecord.material, func.count(DetectionRecord.id).label("cnt"))
        .group_by(DetectionRecord.material)
        .order_by(func.count(DetectionRecord.id).desc())
    )
    return result.all()


async def get_timeline_stats(db: AsyncSession, days: int = 30):
    """Returns list of (date_str, count) for last N days."""
    result = await db.execute(
        select(
            func.strftime("%Y-%m-%d", DetectionSession.upload_time).label("day"),
            func.sum(DetectionSession.total_objects).label("total"),
        )
        .group_by(func.strftime("%Y-%m-%d", DetectionSession.upload_time))
        .order_by(func.strftime("%Y-%m-%d", DetectionSession.upload_time))
    )
    return result.all()


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


async def get_material_per_day_stats(db: AsyncSession):
    """Returns list of (day, material, count) for stacked bar chart."""
    result = await db.execute(
        select(
            func.strftime("%Y-%m-%d", DetectionSession.upload_time).label("day"),
            DetectionRecord.material,
            func.count(DetectionRecord.id).label("cnt"),
        )
        .join(DetectionSession, DetectionRecord.session_id == DetectionSession.id)
        .group_by(
            func.strftime("%Y-%m-%d", DetectionSession.upload_time),
            DetectionRecord.material,
        )
        .order_by(func.strftime("%Y-%m-%d", DetectionSession.upload_time))
    )
    return result.all()


async def search_sessions(
    db: AsyncSession,
    skip: int,
    limit: int,
    q: str | None = None,
    material: str | None = None,
    min_objects: int | None = None,
):
    """Paginated + filtered sessions query."""
    from sqlalchemy import distinct

    stmt = select(DetectionSession).order_by(DetectionSession.upload_time.desc())

    if q:
        stmt = stmt.where(DetectionSession.filename.ilike(f"%{q}%"))
    if min_objects is not None:
        stmt = stmt.where(DetectionSession.total_objects >= min_objects)
    if material:
        # sessions that contain at least one record with this material
        sub = select(distinct(DetectionRecord.session_id)).where(
            DetectionRecord.material.ilike(material)
        )
        stmt = stmt.where(DetectionSession.id.in_(sub))

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    items = (await db.execute(stmt.offset(skip).limit(limit))).scalars().all()
    return items, total


# ── Map helpers ──────────────────────────────────────────────────────────

async def get_geolocated_sessions(
    db: AsyncSession,
    limit: int = 500,
    resolved: int | None = None,   # None=all, 0=unresolved, 1=resolved
    material: str | None = None,   # filter to sessions containing this material
):
    """Return sessions that have GPS coordinates, for map display."""
    from sqlalchemy import exists as sa_exists

    q = (
        select(DetectionSession)
        .where(DetectionSession.latitude.isnot(None))
        .where(DetectionSession.longitude.isnot(None))
    )
    if resolved is not None:
        q = q.where(DetectionSession.is_resolved == resolved)
    if material:
        q = q.where(
            sa_exists(
                select(DetectionRecord.id)
                .where(DetectionRecord.session_id == DetectionSession.id)
                .where(DetectionRecord.material == material.lower())
                .correlate(DetectionSession)
            )
        )
    q = q.order_by(DetectionSession.upload_time.desc()).limit(limit)
    result = await db.execute(q)
    return result.scalars().all()


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
    vs.end_time = datetime.utcnow()
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


async def get_video_sessions_paginated(db: AsyncSession, skip: int, limit: int):
    result = await db.execute(
        select(VideoSession)
        .order_by(VideoSession.start_time.desc())
        .offset(skip)
        .limit(limit)
    )
    items = result.scalars().all()
    total = (await db.execute(select(func.count()).select_from(VideoSession))).scalar_one()
    return items, total


# ── Zone / community helpers ──────────────────────────────────────────────

async def get_zone_stats(db: AsyncSession, grid_size: float = 0.002):
    """
    Aggregate geolocated sessions into grid cells (~200m x ~200m).
    Returns list of zone dicts with centroid lat/lng, total_reports,
    total_objects, dominant_material, and severity score.
    grid_size ≈ 0.002 degrees ≈ 200m at mid-latitudes.
    """
    import math

    result = await db.execute(
        select(
            DetectionSession.latitude,
            DetectionSession.longitude,
            DetectionSession.total_objects,
            DetectionSession.upload_time,
        )
        .where(DetectionSession.latitude.isnot(None))
        .where(DetectionSession.longitude.isnot(None))
    )
    rows = result.all()

    # Also load material breakdown per session
    mat_result = await db.execute(
        select(
            DetectionRecord.session_id,
            DetectionRecord.material,
            func.count(DetectionRecord.id).label("cnt"),
        )
        .join(DetectionSession, DetectionSession.id == DetectionRecord.session_id)
        .where(DetectionSession.latitude.isnot(None))
        .group_by(DetectionRecord.session_id, DetectionRecord.material)
    )
    mat_rows = mat_result.all()

    # Build session-id → materials map  (session_id → {material: count})
    session_materials: dict = {}
    for m in mat_rows:
        session_materials.setdefault(m.session_id, {})[m.material] = m.cnt

    # Load session ids alongside rows
    id_result = await db.execute(
        select(DetectionSession.id, DetectionSession.latitude, DetectionSession.longitude)
        .where(DetectionSession.latitude.isnot(None))
    )
    id_rows = {(r.latitude, r.longitude): r.id for r in id_result.all()}

    # Snap each session to a grid cell
    cells: dict = {}
    for row in rows:
        lat_cell = math.floor(row.latitude / grid_size) * grid_size
        lng_cell = math.floor(row.longitude / grid_size) * grid_size
        key = (round(lat_cell, 6), round(lng_cell, 6))

        if key not in cells:
            cells[key] = {
                "lat": round(lat_cell + grid_size / 2, 6),
                "lng": round(lng_cell + grid_size / 2, 6),
                "total_reports": 0,
                "total_objects": 0,
                "last_scan": None,
                "materials": {},
            }
        c = cells[key]
        c["total_reports"] += 1
        c["total_objects"] += row.total_objects or 0
        if c["last_scan"] is None or (row.upload_time and row.upload_time > c["last_scan"]):
            c["last_scan"] = row.upload_time

        # Aggregate materials
        sess_id = id_rows.get((row.latitude, row.longitude))
        if sess_id and sess_id in session_materials:
            for mat, cnt in session_materials[sess_id].items():
                c["materials"][mat] = c["materials"].get(mat, 0) + cnt

    # Build output list with severity
    zones = []
    for cell in cells.values():
        obj = cell["total_objects"]
        # Severity: 0=clean, 1=low, 2=medium, 3=high
        if obj == 0:
            severity = 0
        elif obj < 5:
            severity = 1
        elif obj < 15:
            severity = 2
        else:
            severity = 3

        dominant = max(cell["materials"], key=cell["materials"].get) if cell["materials"] else None
        zones.append({
            "lat": cell["lat"],
            "lng": cell["lng"],
            "total_reports": cell["total_reports"],
            "total_objects": cell["total_objects"],
            "severity": severity,
            "dominant_material": dominant,
            "materials": cell["materials"],
            "last_scan": cell["last_scan"].isoformat() if cell["last_scan"] else None,
        })

    return zones


async def get_nearby_reports(db: AsyncSession, lat: float, lng: float, radius_km: float = 1.0, limit: int = 50):
    """
    Return geolocated sessions within radius_km of (lat, lng).
    Uses a simple bounding-box pre-filter then Haversine distance.
    """
    import math

    # 1 degree lat ≈ 111 km
    delta_lat = radius_km / 111.0
    delta_lng = radius_km / (111.0 * math.cos(math.radians(lat)))

    result = await db.execute(
        select(DetectionSession)
        .where(DetectionSession.latitude.isnot(None))
        .where(DetectionSession.latitude.between(lat - delta_lat, lat + delta_lat))
        .where(DetectionSession.longitude.between(lng - delta_lng, lng + delta_lng))
        .order_by(DetectionSession.upload_time.desc())
        .limit(limit * 2)  # fetch more for Haversine filtering
    )
    candidates = result.scalars().all()

    def haversine(lat1, lng1, lat2, lng2) -> float:
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    nearby = [r for r in candidates if haversine(lat, lng, r.latitude, r.longitude) <= radius_km]
    return nearby[:limit]
