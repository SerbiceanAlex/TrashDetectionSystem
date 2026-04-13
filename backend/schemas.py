"""Pydantic v2 schemas for request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# ── Auth & Users ─────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: str
    role: str
    points: int
    eco_score: int = 0
    rank: str = "Novice"
    streak_days: int = 0
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    id: Optional[int] = None


# ── OTP (Two-Factor) ────────────────────────────────────────────────────────

class OTPRequired(BaseModel):
    """Returned when password is correct but OTP verification is needed."""
    otp_required: bool = True
    email_hint: str          # masked email, e.g. "a***@gmail.com"
    message: str = "Cod de verificare trimis pe email"

class OTPVerify(BaseModel):
    """Client sends this to verify the OTP code."""
    username: str
    code: str

class PasswordErrors(BaseModel):
    """Returned when password doesn't meet policy."""
    errors: list[str]

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
    estimated_weight_kg: float = 0.0


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
    is_resolved: int = 0
    status: str = "pending"


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
    is_resolved: int = 0
    status: str = "pending"


# ── Leaderboard ───────────────────────────────────────────────────────────────

class LeaderboardEntry(BaseModel):
    rank: int
    username: str
    role: str
    points: int
    eco_score: int = 0
    user_rank: str = "Novice"
    streak_days: int = 0
    total_reports: int


# ── Admin stats ───────────────────────────────────────────────────────────────

class AdminStats(BaseModel):
    total_users: int
    total_sessions: int
    total_objects: int
    resolved_reports: int
    avg_inference_ms: float
    total_votes: int = 0
    pending_reports: int = 0
    verified_reports: int = 0
    cleaned_reports: int = 0
    fake_reports: int = 0
    in_progress_reports: int = 0


# ── Personal stats ────────────────────────────────────────────────────────────

class WeeklyPoint(BaseModel):
    day: str
    reports: int
    objects: int


class PersonalStats(BaseModel):
    username: str
    role: str
    points: int
    eco_score: int = 0
    rank: str = "Novice"
    streak_days: int = 0
    trust_weight: float = 1.0
    total_sessions: int
    total_objects: int
    resolved_count: int
    weekly_activity: list[WeeklyPoint]


# ── Notifications ─────────────────────────────────────────────────────────────

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


# ── Generic responses ─────────────────────────────────────────────────────────

class OkResponse(BaseModel):
    ok: bool


class DetailResponse(BaseModel):
    detail: str


# ── Community / EcoScore ──────────────────────────────────────────────────────

class VoteRequest(BaseModel):
    vote_type: str  # 'confirm' or 'fake'

class VoteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    session_id: int
    vote_type: str
    weight: float
    created_at: datetime

class VoteSummary(BaseModel):
    confirms: int
    fakes: int
    total_weight_confirm: float
    total_weight_fake: float
    user_vote: Optional[str] = None  # user's own vote if any

class MaterialSuggestionRequest(BaseModel):
    suggested_material: str

class MaterialSuggestionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    record_id: int
    user_id: int
    suggested_material: str
    created_at: datetime

class ProfileOut(BaseModel):
    id: int
    username: str
    role: str
    eco_score: int
    rank: str
    streak_days: int
    trust_weight: float
    total_reports: int
    total_objects: int
    verified_reports: int
    cleaned_reports: int
    total_votes: int
    anonymous_reports: bool
    hide_exact_location: bool
    onboarding_done: bool = False
    avatar_url: Optional[str] = None
    created_at: datetime

class PrivacySettings(BaseModel):
    anonymous_reports: Optional[bool] = None
    hide_exact_location: Optional[bool] = None

class CommunityFeedItem(BaseModel):
    event_type: str  # 'report', 'verified', 'cleaned', 'vote'
    session_id: int
    username: Optional[str] = None
    timestamp: datetime
    total_objects: Optional[int] = None
    status: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class RankInfo(BaseModel):
    name: str
    min_score: int
    max_score: Optional[int] = None
    trust_weight: float
    benefits: list[str]


# ── Comments ──────────────────────────────────────────────────────────────────

class CommentCreate(BaseModel):
    text: str

class CommentOut(BaseModel):
    id: int
    session_id: int
    user_id: int
    username: str
    text: str
    created_at: str

class UserNoteUpdate(BaseModel):
    user_note: str


# ── Authority contacts ────────────────────────────────────────────────────────

class AuthorityContactCreate(BaseModel):
    name: str
    email: str
    area_description: str = ""

class AuthorityContactOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
    area_description: Optional[str] = None
    created_by: int
    created_at: datetime


# ── Webhooks ──────────────────────────────────────────────────────────────────

class WebhookCreate(BaseModel):
    url: str
    secret: str = ""
    events: str = "verified"
    active: bool = True

class WebhookUpdate(BaseModel):
    url: Optional[str] = None
    events: Optional[str] = None
    active: Optional[bool] = None

class WebhookOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    url: str
    secret: str
    events: str
    active: bool
    created_by: int
    created_at: datetime


# ── Campaigns ─────────────────────────────────────────────────────────────────

class CampaignCreate(BaseModel):
    title: str
    description: str = ""
    target_reports: int = 50
    start_date: datetime
    end_date: datetime
    area_lat: Optional[float] = None
    area_lng: Optional[float] = None
    area_radius_km: float = 5.0

class CampaignOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str
    description: Optional[str] = None
    target_reports: int
    start_date: datetime
    end_date: datetime
    area_lat: Optional[float] = None
    area_lng: Optional[float] = None
    area_radius_km: float
    created_by: int
    created_at: datetime
    participant_count: int = 0
    report_count: int = 0
    creator_username: str = ""

class CampaignLeaderboardEntry(BaseModel):
    username: str
    report_count: int
    objects_detected: int


# ── Report photos ─────────────────────────────────────────────────────────────

class ReportPhotoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    session_id: int
    user_id: int
    image_path: str
    caption: Optional[str] = None
    photo_type: str
    created_at: datetime


# ── Impact metrics ────────────────────────────────────────────────────────────

class ImpactMetrics(BaseModel):
    total_weight_kg: float
    co2_saved_kg: float
    items_collected: int


# ── Storage stats (admin) ────────────────────────────────────────────────────

class StorageStats(BaseModel):
    uploads_count: int
    uploads_size_mb: float
    annotated_count: int
    annotated_size_mb: float
    cleaned_count: int
    cleaned_size_mb: float
    thumbnails_count: int
    thumbnails_size_mb: float
    avatars_count: int
    avatars_size_mb: float
    videos_count: int
    videos_size_mb: float
    total_size_mb: float
