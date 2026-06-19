from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend import auth, database as db, schemas
from backend.config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


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


async def get_current_user_optional(
    token: str = Depends(OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)),
    session: AsyncSession = Depends(db.get_db),
) -> "db.User | None":
    """Optional auth — returns None instead of raising 401 when no token is provided."""
    if not token:
        return None
    try:
        payload = auth.decode_access_token(token)
        username: str = payload.get("username")
        if not username:
            return None
        result = await session.execute(select(db.User).where(db.User.username == username))
        return result.scalar_one_or_none()
    except Exception:
        return None


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
        raise HTTPException(status_code=400, detail="Numele de utilizator este deja înregistrat.")
        
    result_e = await session.execute(select(db.User).where(db.User.email == user_in.email))
    if result_e.scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail="Emailul este deja înregistrat.")

    hashed_pw = auth.get_password_hash(user_in.password)

    # First user is admin and gets their own org; others join org 1 (default)
    count_res = await session.execute(select(db.User))
    first_user = count_res.first() is None
    role = "admin" if first_user else "user"

    # Resolve organization
    if first_user:
        org = await db.create_organization(session, f"{user_in.username}'s Organization")
    else:
        org = await db.get_or_create_default_org(session)

    new_user = db.User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed_pw,
        role=role,
        organization_id=org.id,
    )
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)

    return new_user


@router.post("/login")
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: AsyncSession = Depends(db.get_db)
):
    """
    Verify username + password and return a JWT token.
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

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"username": user.username, "role": user.role, "id": user.id},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


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
