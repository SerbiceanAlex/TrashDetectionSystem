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
    DETECTOR_WEIGHTS: str = "runs/detect/parks-trash-A3-final/weights/best.pt"
    CLASSIFIER_WEIGHTS: str = "runs/classify/parks-cls-B2/weights/best.pt"

    # ── JWT / Auth ───────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE-ME-generate-with-python-c-import-secrets-secrets.token_hex(32)"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # ── OTP ──────────────────────────────────────────────────────────────────
    OTP_LENGTH: int = 6
    OTP_EXPIRE_MINUTES: int = 5

    # ── SMTP (empty = dev mode, prints OTP to console) ───────────────────────
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASS: str = ""
    SMTP_FROM: str = "noreply@trashdet.local"

    # ── Rate limiting ────────────────────────────────────────────────────────
    MAX_LOGIN_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 15

    # ── Upload limits ────────────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 20

    # ── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = ""  # computed in property if empty

    # ── Inference ────────────────────────────────────────────────────────────
    MAX_IMAGE_DIM: int = 1920
    LIVE_IMGSZ: int = 320

    @property
    def detector_path(self) -> Path:
        return self.REPO_ROOT / self.DETECTOR_WEIGHTS

    @property
    def classifier_path(self) -> Path:
        return self.REPO_ROOT / self.CLASSIFIER_WEIGHTS

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
