"""
Centralized configuration — loaded once from environment variables / .env file.

Usage:
    from backend.config import settings
    print(settings.SECRET_KEY)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    REPO_ROOT: Path = Path(__file__).parent.parent
    DETECTOR_WEIGHTS: str = "models/detector/production/best.pt"
    CLASSIFIER_WEIGHTS: str = "models/classify/B2/best.pt"
    PERSON_DETECTOR_WEIGHTS: str = "models/pretrained/yolov8n.pt"

    # ── JWT / Auth ───────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE-ME-generate-with-python-c-import-secrets-secrets.token_hex(32)"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # ── SMTP (optional: incident alerts and authority forwarding) ────────────
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASS: str = ""
    SMTP_FROM: str = "noreply@trashdet.local"
    ENABLE_INCIDENT_EMAILS: bool = False
    ENABLE_AUTHORITY_EMAILS: bool = False

    # ── Rate limiting ────────────────────────────────────────────────────────
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15

    # ── Stripe (empty = dev mode, checkout activates plans locally) ─────────
    STRIPE_SECRET_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    STRIPE_PRICE_STARTER: str = ""   # price_... from Stripe dashboard
    STRIPE_PRICE_PRO: str = ""       # price_... from Stripe dashboard
    APP_BASE_URL: str = "http://localhost:8000"

    # ── Upload limits ────────────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 20

    # ── File retention (days before auto-cleanup) ────────────────────────────
    RETENTION_DAYS_FAKE: int = 30
    RETENTION_DAYS_EXPIRED: int = 60
    RETENTION_DAYS_CLEANED: int = 365
    STORAGE_CLEANUP_ENABLED: bool = True
    STORAGE_CLEANUP_INTERVAL_HOURS: int = 24
    LITTERING_FILE_RETENTION_DAYS: int = 30

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = ""  # computed in property if empty

    # ── Inference ────────────────────────────────────────────────────────────
    MAX_IMAGE_DIM: int = 1920
    LIVE_IMGSZ: int = 640
    DEFAULT_DET_CONF: float = 0.30
    MONITOR_MIN_DET_CONF: float = 0.35
    MONITOR_TARGET_FPS: int = 25
    MONITOR_CAPTURE_MAX_DIM: int = 512
    MONITOR_JPEG_QUALITY: float = 0.72
    MONITOR_TRASH_IMGSZ: int = 512
    MONITOR_PERSON_IMGSZ: int = 512

    @property
    def detector_path(self) -> Path:
        return self.REPO_ROOT / self.DETECTOR_WEIGHTS

    @property
    def classifier_path(self) -> Path:
        return self.REPO_ROOT / self.CLASSIFIER_WEIGHTS

    @property
    def person_detector_path(self) -> Path:
        return self.REPO_ROOT / self.PERSON_DETECTOR_WEIGHTS

    @property
    def db_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        db_path = self.REPO_ROOT / "backend" / "trash_detection.db"
        return f"sqlite+aiosqlite:///{db_path}"

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()
