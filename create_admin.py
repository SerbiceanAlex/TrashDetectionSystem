"""
Utility script — creare utilizatori în baza de date.

Utilizare:
    .venv\Scripts\python.exe create_admin.py

Creează:
  - admin  /  admin@trash.local  /  Admin1234!   (rol: admin)
  - demo   /  demo@trash.local   /  Demo1234!    (rol: user)

Dacă utilizatorii există deja, îi sare.
"""

import asyncio
import sys
from pathlib import Path

# Asigurăm că importă modulele din proiect
sys.path.insert(0, str(Path(__file__).parent))

from backend import database as db
from backend.auth import get_password_hash
from sqlalchemy import select


USERS_TO_CREATE = [
    {
        "username": "admin",
        "email":    "admin@trash.local",
        "password": "Admin1234!",
        "role":     "admin",
        "points":   0,
    },
    {
        "username": "demo",
        "email":    "demo@trash.local",
        "password": "Demo1234!",
        "role":     "user",
        "points":   0,
    },
]


async def main():
    await db.create_tables()

    async with db.AsyncSessionLocal() as session:
        for u in USERS_TO_CREATE:
            # Verifică dacă există deja
            existing = await session.scalar(
                select(db.User).where(db.User.username == u["username"])
            )
            if existing:
                print(f"  [SKIP]    {u['username']!r} există deja (rol: {existing.role})")
                continue

            new_user = db.User(
                username=u["username"],
                email=u["email"],
                hashed_password=get_password_hash(u["password"]),
                role=u["role"],
                points=u["points"],
            )
            session.add(new_user)
            await session.commit()
            await session.refresh(new_user)
            print(f"  [CREAT]   {u['username']!r}  rol={u['role']}  id={new_user.id}")

    print("\nGata. Utilizatori disponibili:")
    print("  username: admin   parola: Admin1234!   rol: admin")
    print("  username: demo    parola: Demo1234!    rol: user")


if __name__ == "__main__":
    asyncio.run(main())
