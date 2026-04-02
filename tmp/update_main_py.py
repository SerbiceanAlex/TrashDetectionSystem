import sys
import re

file_path = "d:/TrashDetectionSystem/app/main.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update detect endpoint arguments and add user auth logic
content = re.sub(
    r'async def detect\(\s+background_tasks: BackgroundTasks,\s+file: UploadFile = File\(\.\.\.\),\s+det_conf: float = Query\(default=0\.50, ge=0\.05, le=0\.95, description="Detector confidence threshold"\),\s+latitude: float = Query\(default=None, description="GPS latitude"\),\s+longitude: float = Query\(default=None, description="GPS longitude"\),\s+session: AsyncSession = Depends\(db\.get_db\),\s+\):',
    r'async def detect(\n    background_tasks: BackgroundTasks,\n    file: UploadFile = File(...),\n    det_conf: float = Query(default=0.50, ge=0.05, le=0.95, description="Detector confidence threshold"),\n    latitude: float = Query(default=None, description="GPS latitude"),\n    longitude: float = Query(default=None, description="GPS longitude"),\n    session: AsyncSession = Depends(db.get_db),\n    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,\n):',
    content
)

# 2. Assign reporter_id and reward points in detect endpoint
if 'current_user = None' not in content:
    content = content.replace(
        'async def detect(\n    background_tasks: BackgroundTasks,\n    file: UploadFile = File(...),\n    det_conf: float = Query(default=0.50, ge=0.05, le=0.95, description="Detector confidence threshold"),\n    latitude: float = Query(default=None, description="GPS latitude"),\n    longitude: float = Query(default=None, description="GPS longitude"),\n    session: AsyncSession = Depends(db.get_db),\n    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,\n):',
        'async def detect(\n    background_tasks: BackgroundTasks,\n    file: UploadFile = File(...),\n    det_conf: float = Query(default=0.50, ge=0.05, le=0.95, description="Detector confidence threshold"),\n    latitude: float = Query(default=None, description="GPS latitude"),\n    longitude: float = Query(default=None, description="GPS longitude"),\n    session: AsyncSession = Depends(db.get_db),\n    token: Annotated[Optional[str], Depends(oauth2_scheme)] = None,\n):\n    # Optional User Auth for points\n    current_user = None\n    if token:\n        from app.auth import decode_access_token\n        try:\n            payload = decode_access_token(token)\n            if payload and "username" in payload:\n                from sqlalchemy import select\n                res = await session.execute(select(db.User).where(db.User.username == payload["username"]))\n                current_user = res.scalar_one_or_none()\n        except Exception:\n            pass'
    )

# 3. Assign reporter_id to DetectionSession and add points
content = content.replace(
    '        gps_source=gps_src,\n    )',
    '        gps_source=gps_src,\n        reporter_id=current_user.id if current_user else None\n    )\n    if current_user:\n        current_user.points += 10'
)

# 4. Update resolve_session endpoint
content = content.replace(
    'async def resolve_session(\n    session_id: int,\n    session: AsyncSession = Depends(db.get_db),\n):',
    'async def resolve_session(\n    session_id: int,\n    session: AsyncSession = Depends(db.get_db),\n    current_user: Annotated[db.User, Depends(get_current_active_user)] = Depends(),\n):'
)

content = content.replace(
    '    ds.is_resolved = 1\n    ds.resolved_at = datetime.utcnow()\n    await session.commit()',
    '    if ds.is_resolved:\n        return {"detail": "Acest focar este deja marcat ca fiind curățat."}\n\n    ds.is_resolved = 1\n    ds.resolved_at = datetime.utcnow()\n    ds.resolver_id = current_user.id\n    \n    # Reward points\n    current_user.points += 50\n    \n    await session.commit()'
)

# 5. Restrict deletion to admins
content = content.replace(
    'async def delete_session(\n    session_id: int,\n    session: AsyncSession = Depends(db.get_db),\n):',
    'async def delete_session(\n    session_id: int,\n    session: AsyncSession = Depends(db.get_db),\n    current_user: Annotated[db.User, Depends(get_current_active_user)] = Depends(),\n):\n    if current_user.role != "admin":\n        raise HTTPException(status_code=403, detail="Numai administratorii pot șterge raportări.")'
)

content = content.replace(
    'async def delete_video_session(\n    session_id: int,\n    session: AsyncSession = Depends(db.get_db),\n):',
    'async def delete_video_session(\n    session_id: int,\n    session: AsyncSession = Depends(db.get_db),\n    current_user: Annotated[db.User, Depends(get_current_active_user)] = Depends(),\n):\n    if current_user.role != "admin":\n        raise HTTPException(status_code=403, detail="Numai administratorii pot șterge sesiuni video.")'
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Main app successfully updated with Auth & Gamification logic.")
