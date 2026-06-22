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

    # ── App base URL (printed by start_https, shown in system info) ─────────
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
    MONITOR_MIN_DET_CONF: float = 0.25
    MONITOR_PERSON_CONF: float = 0.25
    # Țintă 120 = plafon larg, NU throttle. Lecție din teste: o țintă mică
    # (ex. 60) se sincronizează prost cu refresh-ul ecranului și gâtuiește
    # artificial (~46); o țintă mult peste capacitatea reală lasă bucla să
    # ruleze la ceiling-ul fizic. Real obținut: ~55 FPS pe laptop la 768px
    # (encode JPEG = bariera), ~15 pe telefon prin WiFi. Contorul afișează
    # rata reală, nu ținta.
    MONITOR_TARGET_FPS: int = 120
    MONITOR_LOGIC_FPS: int = 25
    MONITOR_CAMERA_WIDTH: int = 1280
    MONITOR_CAMERA_HEIGHT: int = 720
    # Captură și inferență la 768 — prioritate pe CALITATEA detecției (recall
    # bun pe obiecte mici la 1–3 m), decizia autorului. Pe laptop encode-ul de
    # 768 limitează la ~26-27 FPS, suficient pentru un act de câteva secunde;
    # afișajul FPS e stabilizat ca să nu pâlpâie. Evaluarea pe 19 clipuri AI
    # confirmă imgsz 768 ca optim (recall 0,81 vs 0,69 la 1024).
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
