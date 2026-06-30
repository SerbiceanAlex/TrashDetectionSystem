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

    # ── Căi ──────────────────────────────────────────────────────────────────
    REPO_ROOT: Path = Path(__file__).parent.parent      # rădăcina proiectului
    STORAGE_ROOT: str = "data/runtime"                  # unde se scriu fișierele generate
    DETECTOR_WEIGHTS: str = "models/detector/production/best.pt"   # YOLOv8 deșeuri (antrenat)
    CLASSIFIER_WEIGHTS: str = "models/classify/B2/best.pt"         # clasificator material final adaptat pe domeniul real
    PERSON_DETECTOR_WEIGHTS: str = "models/pretrained/yolov8n.pt"  # detector persoane (COCO)

    # ── JWT / Autentificare ──────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE-ME-generate-with-python-c-import-secrets-secrets.token_hex(32)"
    ALGORITHM: str = "HS256"                            # algoritm de semnare a token-ului
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7      # valabilitate token: 7 zile

    # ── Rate limiting la login (anti brute-force) ────────────────────────────
    MAX_LOGIN_ATTEMPTS: int = 5                         # eșuări permise înainte de blocare
    LOGIN_LOCKOUT_MINUTES: int = 15                     # durata blocării temporare

    # ── URL-ul aplicației (afișat de start_https, în system info) ────────────
    APP_BASE_URL: str = "http://localhost:8000"

    # ── Limite de upload ─────────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 20                             # imagini
    VIDEO_MAX_UPLOAD_MB: int = 200                      # videouri

    # ── Curățare automată a probelor (fișierele incidentelor) ────────────────
    STORAGE_CLEANUP_ENABLED: bool = True          # pornește bucla de curățare în fundal
    STORAGE_CLEANUP_INTERVAL_HOURS: int = 24      # cât de des rulează curățarea
    LITTERING_FILE_RETENTION_DAYS: int = 30       # vechimea după care se șterg clip/thumbnail

    # ── Bază de date ─────────────────────────────────────────────────────────
    DATABASE_URL: str = ""  # gol → se calculează SQLite local în proprietatea db_url

    # ── Inferență (parametri model) ──────────────────────────────────────────
    MAX_IMAGE_DIM: int = 1920          # latura maximă la care se redimensionează un cadru
    LIVE_IMGSZ: int = 640              # imgsz implicit pentru scanarea de imagini
    DEFAULT_DET_CONF: float = 0.30     # prag de încredere implicit (upload/scanare)
    MONITOR_MIN_DET_CONF: float = 0.15 # prag MINIM pe monitorul live (prinde și obiecte la distanță)
    MONITOR_PERSON_CONF: float = 0.25  # prag pentru detecția persoanelor pe monitor
    # Țintă 120 = plafon larg, NU throttle. Lecție din teste: o țintă mică
    # (ex. 60) se sincronizează prost cu refresh-ul ecranului și gâtuiește
    # artificial (~46); o țintă mult peste capacitatea reală lasă bucla să
    # ruleze la ceiling-ul fizic. Real obținut: ~55 FPS pe laptop la 768px
    # (encode JPEG = bariera), ~15 pe telefon prin WiFi. Contorul afișează
    # rata reală, nu ținta.
    MONITOR_TARGET_FPS: int = 120
    # Rata FIXĂ la care rulează logica temporală (mașina de stări). Independentă
    # de rata reală de cadre, ca timpii în secunde (pre-roll, confirmare,
    # abandonare) să fie corecți și la fel pe orice dispozitiv. Vezi video.py.
    MONITOR_LOGIC_FPS: int = 25
    MONITOR_CAMERA_WIDTH: int = 1280   # rezoluția cerută camerei (orientativ)
    MONITOR_CAMERA_HEIGHT: int = 720
    # Captură și inferență la ~900px: păstrează mai mult detaliu pentru obiecte
    # mici la distanță, fără costul mare al unei rezoluții full-HD.
    MONITOR_CAPTURE_MAX_DIM: int = 896
    MONITOR_JPEG_QUALITY: float = 0.75
    MONITOR_TRASH_IMGSZ: int = 896
    MONITOR_PERSON_IMGSZ: int = 416

    # ── Valori derivate (căi absolute + conversii) ───────────────────────────
    # Toate transformă setările de mai sus în căi/valori gata de folosit.

    @property
    def detector_path(self) -> Path:
        """Calea absolută către modelul de detecție a deșeurilor."""
        return self.REPO_ROOT / self.DETECTOR_WEIGHTS

    @property
    def trash_tracker_cfg(self) -> str:
        """Profilul ByteTrack reglat pentru deșeuri (continuitate/anti-churn)."""
        return str(self.REPO_ROOT / "backend" / "trackers" / "bytetrack_trash.yaml")

    @property
    def classifier_path(self) -> Path:
        return self.REPO_ROOT / self.CLASSIFIER_WEIGHTS

    @property
    def person_detector_path(self) -> Path:
        return self.REPO_ROOT / self.PERSON_DETECTOR_WEIGHTS

    @property
    def db_url(self) -> str:
        """URL-ul bazei de date: DATABASE_URL dacă e setat, altfel SQLite local."""
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
        """Folderul rădăcină pentru fișierele generate (absolutizat față de proiect)."""
        root = Path(self.STORAGE_ROOT)
        return root if root.is_absolute() else self.REPO_ROOT / root

    @property
    def videos_dir(self) -> Path:
        return self.storage_root / "videos"

    @property
    def littering_dir(self) -> Path:
        return self.storage_root / "littering"

    @property
    def runtime_dirs(self) -> list[Path]:
        """Folderele create la pornire pentru fișierele generate de aplicație."""
        # videos/ = videourile încărcate + adnotate; littering/ = dovezile
        # incidentelor (clip + miniatură). Atât se scrie efectiv.
        return [
            self.videos_dir,
            self.littering_dir,
        ]


settings = Settings()
