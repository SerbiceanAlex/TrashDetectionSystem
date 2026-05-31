from datetime import datetime, timedelta, timezone
from typing import Optional
import re
import logging

import bcrypt
import jwt

from backend.config import settings

logger = logging.getLogger(__name__)

# In-memory rate-limit store  {username: (fail_count, locked_until)}
_login_attempts: dict[str, tuple[int, datetime]] = {}


# ── Password policy ──────────────────────────────────────────────────────────

PASSWORD_MIN_LENGTH = 8
PASSWORD_RULES = [
    (r"[A-Z]", "cel puțin o literă mare (A-Z)"),
    (r"[a-z]", "cel puțin o literă mică (a-z)"),
    (r"[0-9]", "cel puțin o cifră (0-9)"),
    (r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?`~]", "cel puțin un caracter special (!@#$%^&*...)"),
]


def validate_password(password: str) -> list[str]:
    """Return list of error messages. Empty list → password is valid."""
    errors = []
    if len(password) < PASSWORD_MIN_LENGTH:
        errors.append(f"minim {PASSWORD_MIN_LENGTH} caractere")
    for pattern, msg in PASSWORD_RULES:
        if not re.search(pattern, password):
            errors.append(msg)
    return errors


# ── Hashing ──────────────────────────────────────────────────────────────────

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# ── JWT ──────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    try:
        decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return decoded
    except jwt.PyJWTError:
        return {}


# ── Email sending ────────────────────────────────────────────────────────────


async def send_incident_alert(to_email: str, event_id: int, material: str, detected_at: str, address: str = "") -> bool:
    """Trimite alertă email instant la detectarea unui incident."""
    if not settings.ENABLE_INCIDENT_EMAILS:
        logger.info("Email incident dezactivat; alerta #%s către %s nu a fost trimisă.", event_id, to_email)
        return False

    subject = f"[TrashDet] Incident detectat #{event_id} — {material}"
    body = (
        f"Incident de aruncare ilegală detectat automat.\n\n"
        f"ID incident: #{event_id}\n"
        f"Material: {material}\n"
        f"Data/Ora: {detected_at}\n"
        f"Locație: {address or 'fără adresă GPS'}\n\n"
        f"Vizualizează și gestionează incidentul în panoul de administrare:\n"
        f"{settings.APP_BASE_URL}/app\n\n"
        f"— TrashDet Monitoring System"
    )

    if settings.SMTP_HOST and settings.SMTP_USER:
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = settings.SMTP_FROM
            msg["To"] = to_email
            await aiosmtplib.send(
                msg,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USER,
                password=settings.SMTP_PASS,
                start_tls=True,
            )
            logger.info(f"Alertă incident #{event_id} trimisă la {to_email}")
            return True
        except Exception as e:
            logger.error(f"Eroare alertă incident: {e}")

    # Dev mode: log in console
    logger.info(
        f"\n{'='*50}\n"
        f"  [ALERTĂ INCIDENT] #{event_id} — {material}\n"
        f"  Destinatar: {to_email}\n"
        f"  Locație: {address or 'N/A'}\n"
        f"{'='*50}"
    )
    return True


async def send_forward_email(
    to_email: str,
    authority_name: str,
    event_id: int,
    material: str,
    detected_at: str,
    image_hash: str,
    address: str = "",
    notes: str = "",
    admin_username: str = "admin",
) -> bool:
    """Trimite email de transmitere a dovezii unui incident la autoritate."""
    if not settings.ENABLE_AUTHORITY_EMAILS:
        logger.info("Email autoritate dezactivat; incidentul #%s nu a fost trimis către %s.", event_id, to_email)
        return False

    subject = f"[TrashDet] Incident #{event_id} — Dovadă aruncare ilegală"
    body = (
        f"Stimate/Stimată reprezentant {authority_name},\n\n"
        f"Sistemul TrashDet a detectat și documentat un act de aruncare ilegală "
        f"a deșeurilor în spații publice. Vă transmitem dovezile pentru acțiune în consecință.\n\n"
        f"{'─'*40}\n"
        f"  ID incident:   #{event_id}\n"
        f"  Material:      {material}\n"
        f"  Data/Ora:      {detected_at}\n"
        f"  Locație:       {address or 'coordonate GPS disponibile în sistem'}\n"
        f"  Hash dovadă:   {image_hash or 'N/A'} (SHA-256, GDPR Art. 25)\n"
    )
    if notes:
        body += f"  Note admin:    {notes}\n"
    body += (
        f"{'─'*40}\n\n"
        f"Clipul video (cu fețele anonimizate conform GDPR Art. 25) și "
        f"thumbnail-ul sunt disponibile în panoul administrativ TrashDet.\n\n"
        f"Transmis de: {admin_username} · TrashDet Monitoring System\n"
        f"{'─'*40}"
    )

    if settings.SMTP_HOST and settings.SMTP_USER:
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = settings.SMTP_FROM
            msg["To"] = to_email
            await aiosmtplib.send(
                msg,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USER,
                password=settings.SMTP_PASS,
                start_tls=True,
            )
            logger.info(f"Email forward incident #{event_id} → {authority_name} <{to_email}>")
            return True
        except Exception as e:
            logger.error(f"Eroare trimitere email forward #{event_id}: {e}")

    # Dev mode: log detaliat în consolă
    logger.info(
        f"\n{'='*55}\n"
        f"  [DEV — FORWARD INCIDENT] Incident #{event_id}\n"
        f"  Autoritate: {authority_name} <{to_email}>\n"
        f"  Material: {material} | Locație: {address or 'N/A'}\n"
        f"  Hash SHA-256: {image_hash or 'N/A'}\n"
        f"{'='*55}"
    )
    return True


# ── Rate limiting ────────────────────────────────────────────────────────────

def check_rate_limit(username: str) -> tuple[bool, int]:
    """
    Returns (is_locked, remaining_seconds).
    is_locked=True means the user must wait.
    """
    now = datetime.now(timezone.utc)
    entry = _login_attempts.get(username)
    if entry is None:
        return False, 0
    fail_count, locked_until = entry
    if locked_until and now < locked_until:
        remaining = int((locked_until - now).total_seconds())
        return True, remaining
    if locked_until and now >= locked_until:
        # Lockout expired, reset
        _login_attempts.pop(username, None)
        return False, 0
    return False, 0


def record_failed_login(username: str):
    """Record a failed login attempt. Lock after MAX_LOGIN_ATTEMPTS."""
    now = datetime.now(timezone.utc)
    entry = _login_attempts.get(username)
    if entry is None:
        _login_attempts[username] = (1, None)
    else:
        fail_count, _ = entry
        fail_count += 1
        if fail_count >= settings.MAX_LOGIN_ATTEMPTS:
            locked_until = now + timedelta(minutes=settings.LOGIN_LOCKOUT_MINUTES)
            _login_attempts[username] = (fail_count, locked_until)
        else:
            _login_attempts[username] = (fail_count, None)


def reset_login_attempts(username: str):
    """Reset failed attempts after successful login."""
    _login_attempts.pop(username, None)
