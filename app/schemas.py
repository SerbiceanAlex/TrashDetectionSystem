"""Pydantic v2 schemas for request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ── Detection record (single bounding box) ──────────────────────────────────

class DetectionRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    material: str
    det_score: float
    cls_score: float
    box_x1: int
    box_y1: int
"""Pydantic v2 schemas for request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ── Detection record (single bounding box) ──────────────────────────────────

class DetectionRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    material: str
    det_score: float
    cls_score: float
    box_x1: int
    box_y1: int
    box_x2: int
    box_y2: int


# ── Session (one uploaded image) ────────────────────────────────────────────

class DetectionSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    upload_time: datetime
    total_objects: int
    inference_ms: float
    annotated_path: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    gps_source: Optional[str] = None


class DetectionSessionDetail(DetectionSessionOut):
    records: list[DetectionRecordOut] = []


# ── Detect endpoint response ─────────────────────────────────────────────────

class DetectResponse(BaseModel):
    session_id: int
    filename: str
    total_objects: int
    inference_ms: float
    annotated_url: str
    detections: list[DetectionRecordOut]
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    gps_source: Optional[str] = None


# ── Stats ────────────────────────────────────────────────────────────────────

class MaterialStat(BaseModel):
    material: str
    count: int


class TimelinePoint(BaseModel):
    day: str
    total: int


class GlobalStats(BaseModel):
    total_sessions: int
    total_objects: int
    avg_inference_ms: float
    material_distribution: list[MaterialStat]
    timeline: list[TimelinePoint]
    material_per_day: list[dict] = []   # [{day, material, count}, ...]


# ── Sessions list (paginated) ────────────────────────────────────────────────

class SessionsPage(BaseModel):
    total: int
    skip: int
    limit: int
    items: list[DetectionSessionOut]


# ── Batch detect response ────────────────────────────────────────────────────

class BatchDetectResponse(BaseModel):
    results: list[DetectResponse]
    total_files: int
    total_objects: int
    total_ms: float


# ── Video session schemas ───────────────────────────────────────────────────

class VideoSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source_type: str
    filename: Optional[str] = None
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_sec: float = 0.0
    total_frames: int = 0
    total_objects: int = 0
    avg_fps: float = 0.0
    avg_inference_ms: float = 0.0
    materials_summary: Optional[str] = None
    annotated_video_path: Optional[str] = None
    status: str = "running"
    frames_processed: Optional[int] = 0
    total_frames_expected: Optional[int] = 0


class VideoSessionsPage(BaseModel):
    total: int
    skip: int
    limit: int
    items: list[VideoSessionOut]


class VideoUploadResponse(BaseModel):
    session_id: int
    status: str
    message: str


# ── Map schemas ──────────────────────────────────────────────────────────

class MapReport(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    upload_time: datetime
    total_objects: int
    inference_ms: float
    latitude: float
    longitude: float
    annotated_path: Optional[str] = None
    address: Optional[str] = None
    gps_source: Optional[str] = None


# ── Zone / community schemas ───────────────────────────────────────

class ZoneStats(BaseModel):
    """Aggregated grid cell for the community map."""
    lat: float
    lng: float
    total_reports: int
    total_objects: int
    severity: int             # 0=clean, 1=low, 2=medium, 3=high
    dominant_material: Optional[str] = None
    materials: dict = {}
    last_scan: Optional[str] = None


class NearbyReport(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    upload_time: datetime
    total_objects: int
    inference_ms: float
    latitude: float
    longitude: float
    annotated_path: Optional[str] = None
    address: Optional[str] = None
    gps_source: Optional[str] = None
