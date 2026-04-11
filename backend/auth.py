from datetime import datetime, timedelta, timezone
from typing import Optional
import re
import secrets
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


# ── OTP generation ───────────────────────────────────────────────────────────

def generate_otp() -> str:
    """Generate a cryptographically secure 6-digit OTP code."""
    return "".join(secrets.choice("0123456789") for _ in range(settings.OTP_LENGTH))


# ── Email sending ────────────────────────────────────────────────────────────

async def send_otp_email(to_email: str, otp_code: str, username: str) -> bool:
    """Send OTP code via email. Falls back to console print in dev mode."""
    subject = f"TrashDet – Cod de verificare: {otp_code}"
    body = (
        f"Salut {username},\n\n"
        f"Codul tău de verificare este: {otp_code}\n\n"
        f"Codul expiră în {settings.OTP_EXPIRE_MINUTES} minute.\n"
        f"Dacă nu ai solicitat acest cod, ignoră mesajul.\n\n"
        f"─ TrashDet"
    )

    # If SMTP is configured, send real email
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
            logger.info(f"OTP email trimis la {to_email}")
            return True
        except Exception as e:
            logger.error(f"Eroare trimitere email: {e}")
            # Fall through to console print
    
    # Dev mode: print to console
    logger.info(
        f"\n{'='*50}\n"
        f"  [DEV] OTP pentru {username} ({to_email})\n"
        f"  COD: {otp_code}\n"
        f"  Expiră în {settings.OTP_EXPIRE_MINUTES} minute\n"
        f"{'='*50}"
    )
    print(
        f"\n{'='*50}\n"
        f"  [DEV] OTP pentru {username} ({to_email})\n"
        f"  COD: {otp_code}\n"
        f"  Expiră în {settings.OTP_EXPIRE_MINUTES} minute\n"
        f"{'='*50}"
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
