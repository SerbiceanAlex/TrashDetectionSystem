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
    STORAGE_ROOT: str = "data/runtime"
    DETECTOR_WEIGHTS: str = "models/detector/production/best.pt"
    CLASSIFIER_WEIGHTS: str = "models/classify/B2/best.pt"
    PERSON_DETECTOR_WEIGHTS: str = "models/pretrained/yolov8n.pt"

    # ── JWT / Auth ───────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE-ME-generate-with-python-c-import-secrets-secrets.token_hex(32)"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

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
    VIDEO_MAX_UPLOAD_MB: int = 200

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
    # Țintă realistă: fiecare cadru face encode pe telefon → WiFi → YOLO → răspuns.
    # 15 FPS de analiză e sustenabil pe RTX 3050 cu telefon prin LAN; camera
    # rulează oricum la 30fps nativ pentru un preview fluid.
    MONITOR_TARGET_FPS: int = 15
    MONITOR_CAMERA_WIDTH: int = 1280
    MONITOR_CAMERA_HEIGHT: int = 720
    # Capturarea trimisă spre AI și dimensiunea de inferență urcate la 768:
    # benchmark pe RTX 3050 arată +~1 ms cost de detector (server tot ~42 FPS),
    # dar recall sensibil mai bun pe obiecte mici la 1–3 m, principala limită
    # observată în testele reale. Capturarea și inferența trebuie să crească
    # împreună, altfel detectorul ar mări un JPEG de 640 fără detaliu nou.
    MONITOR_CAPTURE_MAX_DIM: int = 768
    MONITOR_JPEG_QUALITY: float = 0.75
    MONITOR_TRASH_IMGSZ: int = 768
    MONITOR_PERSON_IMGSZ: int = 416

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
        db_path = self.REPO_ROOT / "data" / "trash_detection.db"
        return f"sqlite+aiosqlite:///{db_path}"

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def video_max_upload_bytes(self) -> int:
        return self.VIDEO_MAX_UPLOAD_MB * 1024 * 1024

    @property
    def storage_root(self) -> Path:
        root = Path(self.STORAGE_ROOT)
        return root if root.is_absolute() else self.REPO_ROOT / root

    @property
    def uploads_dir(self) -> Path:
        return self.storage_root / "uploads"

    @property
    def annotated_dir(self) -> Path:
        return self.storage_root / "annotated"

    @property
    def videos_dir(self) -> Path:
        return self.storage_root / "videos"

    @property
    def littering_dir(self) -> Path:
        return self.storage_root / "littering"

    @property
    def runtime_dirs(self) -> list[Path]:
        # Doar folderele scrise efectiv în mod curent. uploads/ și annotated/
        # se creează lazy la prima scanare de imagine (vezi main.py), iar
        # cleaned/ și thumbnails/ au fost scoase (nu erau folosite — dovezile
        # incidentelor merg în littering/).
        return [
            self.videos_dir,
            self.littering_dir,
        ]


settings = Settings()
