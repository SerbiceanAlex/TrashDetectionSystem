from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend import auth, database as db, schemas
from backend.config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _mask_email(email: str) -> str:
    """Mask email: 'user@example.com' → 'u***@example.com'"""
    parts = email.split("@")
    if len(parts) != 2:
        return "***@***"
    local = parts[0]
    if len(local) <= 1:
        masked = local + "***"
    else:
        masked = local[0] + "***" + local[-1]
    return f"{masked}@{parts[1]}"


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: AsyncSession = Depends(db.get_db)
) -> db.User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = auth.decode_access_token(token)
    username: str = payload.get("username")
    if username is None:
        raise credentials_exception
    
    # Query user
    result = await session.execute(select(db.User).where(db.User.username == username))
    user = result.scalar_one_or_none()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[db.User, Depends(get_current_user)]
) -> db.User:
    return current_user


@router.post("/register", response_model=schemas.UserOut)
async def register_user(
    user_in: schemas.UserCreate,
    session: AsyncSession = Depends(db.get_db)
):
    # Validate password policy
    pw_errors = auth.validate_password(user_in.password)
    if pw_errors:
        raise HTTPException(
            status_code=422,
            detail="Parola nu îndeplinește cerințele: " + "; ".join(pw_errors)
        )

    # Check if username or email exists
    result_u = await session.execute(select(db.User).where(db.User.username == user_in.username))
    if result_u.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Username already registered")
        
    result_e = await session.execute(select(db.User).where(db.User.email == user_in.email))
    if result_e.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = auth.get_password_hash(user_in.password)
    
    # First user is admin, others are 'user'
    count_res = await session.execute(select(db.User))
    first_user = count_res.first() is None
    role = "admin" if first_user else "user"

    new_user = db.User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed_pw,
        role=role
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    
    print(f"[MAIL MOCK] Trimis email de confirmare la: {new_user.email}")
    
    return new_user


@router.post("/login")
async def login_step1(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: AsyncSession = Depends(db.get_db)
):
    """
    Step 1: Verify username + password → send OTP to email.
    Returns either OTPRequired (need OTP) or Token (if OTP disabled).
    """
    username = form_data.username

    # Rate limit check
    is_locked, remaining_sec = auth.check_rate_limit(username)
    if is_locked:
        raise HTTPException(
            status_code=429,
            detail=f"Prea multe încercări. Încearcă din nou în {remaining_sec} secunde."
        )

    # Find user
    result = await session.execute(select(db.User).where(db.User.username == username))
    user = result.scalar_one_or_none()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        auth.record_failed_login(username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilizator sau parolă incorectă",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Password correct → reset rate limiter
    auth.reset_login_attempts(username)

    # Invalidate any existing unused OTP codes for this user
    old_codes = await session.execute(
        select(db.OTPCode).where(
            db.OTPCode.user_id == user.id,
            db.OTPCode.is_used == 0
        )
    )
    for old_otp in old_codes.scalars().all():
        old_otp.is_used = 1

    # Generate new OTP
    otp_code = auth.generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)

    new_otp = db.OTPCode(
        user_id=user.id,
        code=otp_code,
        expires_at=expires_at,
    )
    session.add(new_otp)
    await session.commit()

    # Send OTP via email
    await auth.send_otp_email(user.email, otp_code, user.username)

    return schemas.OTPRequired(
        otp_required=True,
        email_hint=_mask_email(user.email),
        message="Cod de verificare trimis pe email"
    )


@router.post("/verify-otp", response_model=schemas.Token)
async def login_step2(
    otp_data: schemas.OTPVerify,
    session: AsyncSession = Depends(db.get_db)
):
    """
    Step 2: Verify OTP code → return JWT token.
    """
    now = datetime.now(timezone.utc)

    # Rate limit check
    otp_key = f"otp:{otp_data.username}"
    is_locked, remaining_sec = auth.check_rate_limit(otp_key)
    if is_locked:
        raise HTTPException(
            status_code=429,
            detail=f"Prea multe încercări OTP. Încearcă din nou în {remaining_sec} secunde."
        )

    # Find user
    result = await session.execute(select(db.User).where(db.User.username == otp_data.username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="Utilizator invalid")

    # Find valid OTP code
    otp_result = await session.execute(
        select(db.OTPCode).where(
            db.OTPCode.user_id == user.id,
            db.OTPCode.code == otp_data.code,
            db.OTPCode.is_used == 0,
            db.OTPCode.expires_at > now
        ).order_by(db.OTPCode.created_at.desc())
    )
    otp_record = otp_result.scalar_one_or_none()

    if not otp_record:
        auth.record_failed_login(otp_key)
        raise HTTPException(
            status_code=401,
            detail="Cod invalid sau expirat"
        )

    # Mark OTP as used
    otp_record.is_used = 1
    await session.commit()

    # Reset OTP rate limiter
    auth.reset_login_attempts(otp_key)

    # Issue JWT
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"username": user.username, "role": user.role, "id": user.id},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/resend-otp", response_model=schemas.OTPRequired)
async def resend_otp(
    data: schemas.OTPVerify,
    session: AsyncSession = Depends(db.get_db)
):
    """Resend a new OTP code (invalidates previous ones)."""
    # Just needs username, code field is ignored
    result = await session.execute(select(db.User).where(db.User.username == data.username))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="Utilizator invalid")

    # Invalidate old codes
    old_codes = await session.execute(
        select(db.OTPCode).where(db.OTPCode.user_id == user.id, db.OTPCode.is_used == 0)
    )
    for old_otp in old_codes.scalars().all():
        old_otp.is_used = 1

    # Generate new
    otp_code = auth.generate_otp()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
    new_otp = db.OTPCode(user_id=user.id, code=otp_code, expires_at=expires_at)
    session.add(new_otp)
    await session.commit()

    await auth.send_otp_email(user.email, otp_code, user.username)

    return schemas.OTPRequired(
        otp_required=True,
        email_hint=_mask_email(user.email),
        message="Cod nou trimis pe email"
    )


@router.get("/password-rules")
async def password_rules():
    """Return password policy rules for the frontend to display."""
    return {
        "min_length": auth.PASSWORD_MIN_LENGTH,
        "rules": [msg for _, msg in auth.PASSWORD_RULES],
    }


@router.get("/me", response_model=schemas.UserOut)
async def read_users_me(
    current_user: Annotated[db.User, Depends(get_current_active_user)]
):
    return current_user
