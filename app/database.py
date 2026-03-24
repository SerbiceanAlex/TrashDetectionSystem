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
