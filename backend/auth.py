"""
Securitatea autentificării: politică de parolă, hash bcrypt, token-uri JWT și
rate-limiting la login. Funcții pure (fără rute) — rutele sunt în auth_router.py.
"""

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
    """
    Verifică dacă parola respectă politica (lungime minimă + literă mare/mică,
    cifră, caracter special). Întoarce lista mesajelor de eroare; listă goală
    înseamnă parolă validă. Folosită la înregistrare.
    """
    errors = []
    if len(password) < PASSWORD_MIN_LENGTH:
        errors.append(f"minim {PASSWORD_MIN_LENGTH} caractere")
    for pattern, msg in PASSWORD_RULES:
        if not re.search(pattern, password):
            errors.append(msg)
    return errors


# ── Hashing ──────────────────────────────────────────────────────────────────

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Compară parola introdusă (text simplu) cu hash-ul bcrypt din baza de date."""
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def get_password_hash(password: str) -> str:
    """Generează hash-ul bcrypt al unei parole (salvat în DB, niciodată textul brut)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


# ── JWT ──────────────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creează un token JWT semnat cu SECRET_KEY, cu data de expirare inclusă.
    `data` conține de obicei username, rol și id-ul utilizatorului. Tokenul e
    trimis clientului la login și prezentat apoi la fiecare cerere protejată.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decodează și validează un token JWT (semnătură + expirare). Întoarce
    payload-ul (dict) dacă e valid, sau dict gol dacă tokenul e invalid/expirat.
    """
    try:
        decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return decoded
    except jwt.PyJWTError:
        return {}


# ── Rate limiting ────────────────────────────────────────────────────────────

def check_rate_limit(username: str) -> tuple[bool, int]:
    """
    Verifică dacă un cont e blocat temporar din cauza prea multor încercări
    eșuate de login. Întoarce (este_blocat, secunde_rămase); dacă blocarea a
    expirat, o resetează automat. Protejează împotriva atacurilor de tip
    brute-force pe parolă.
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
    """
    Înregistrează o încercare eșuată de login. După MAX_LOGIN_ATTEMPTS eșuări
    consecutive, contul e blocat pentru LOGIN_LOCKOUT_MINUTES minute.
    """
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
    """Șterge contorul de încercări eșuate (apelat după un login reușit)."""
    _login_attempts.pop(username, None)
