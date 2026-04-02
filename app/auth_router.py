from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app import auth, database as db, schemas

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


@router.post("/register", response_model=schemas.UserOut)
async def register_user(
    user_in: schemas.UserCreate,
    session: AsyncSession = Depends(db.get_db)
):
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
    
    # Mock sending confirmation email
    print(f"[MAIL MOCK] Trimis email de confirmare la: {new_user.email}")
    
    return new_user


@router.post("/login", response_model=schemas.Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: AsyncSession = Depends(db.get_db)
):
    result = await session.execute(select(db.User).where(db.User.username == form_data.username))
    user = result.scalar_one_or_none()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"username": user.username, "role": user.role, "id": user.id},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=schemas.UserOut)
async def read_users_me(
    current_user: Annotated[db.User, Depends(get_current_active_user)]
):
    return current_user
