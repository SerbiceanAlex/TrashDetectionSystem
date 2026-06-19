"""Pydantic v2 schemas used by the current TrashDet API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# Auth and users

class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    role: str
    points: int
    organization_id: Optional[int] = None
    created_at: datetime


# Image detection

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
    estimated_weight_kg: float = 0.0


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
    is_resolved: int = 0
    resolved_at: Optional[datetime] = None
    reporter_id: Optional[int] = None
    resolver_id: Optional[int] = None
    status: str = "pending"
    verification_score: float = 0.0
    claimed_by: Optional[int] = None
    claimed_at: Optional[datetime] = None
    cleaned_at: Optional[datetime] = None
    user_note: Optional[str] = None


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
    is_resolved: int = 0
    resolved_at: Optional[datetime] = None
    reporter_id: Optional[int] = None
    resolver_id: Optional[int] = None
    status: str = "pending"
    verification_score: float = 0.0
    claimed_by: Optional[int] = None
    claimed_at: Optional[datetime] = None
    cleaned_at: Optional[datetime] = None
    user_note: Optional[str] = None


class SessionsPage(BaseModel):
    total: int
    skip: int
    limit: int
    items: list[DetectionSessionOut]


# Video processing

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
    littering_count: int = 0
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


# Admin, notifications and generic responses

class AdminStats(BaseModel):
    total_users: int
    total_sessions: int
    total_objects: int
    resolved_reports: int
    avg_inference_ms: float


class NotificationOut(BaseModel):
    id: int
    message: str
    category: Optional[str] = None
    session_id: Optional[int] = None
    is_read: int
    created_at: str


class NotificationsResponse(BaseModel):
    unread: int
    notifications: list[NotificationOut]


class OkResponse(BaseModel):
    ok: bool


class DetailResponse(BaseModel):
    detail: str


# Littering incidents

class LitteringEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    detected_at: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    material: str
    det_score: float
    person_present: int = 1
    person_count: int = 1
    face_blurred: int = 0
    clip_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    image_hash: Optional[str] = None
    status: str = "pending"
    reporter_id: Optional[int] = None
    reporter_username: Optional[str] = None
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None
    forwarded_at: Optional[datetime] = None
    notes: Optional[str] = None
    incident_uid: Optional[str] = None
    owner_person_id: Optional[int] = None
    distance_at_abandonment: Optional[float] = None
    detection_method: str = "zone"


class LitteringEventsPage(BaseModel):
    total: int
    skip: int
    limit: int
    items: list[LitteringEventOut]


class LitteringEventStatusUpdate(BaseModel):
    status: str
    notes: Optional[str] = None


class LitteringEventNotesUpdate(BaseModel):
    notes: str
