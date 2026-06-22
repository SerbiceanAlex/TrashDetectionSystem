"""
Pregătește baza de date SQLite locală pentru o rulare curată (teză/local).

Scriptul e conservator implicit:
  - face un backup cu timestamp înainte de orice scriere;
  - asigură un cont de admin și unul de utilizator standard;
  - curăță notificările;
  - (opțional) resetează datele generate sau atribuie rândurile vechi userului local;
  - mută/șterge fișiere de DB vechi din backend/ când e sigur.

Operațiile distructive (ștergere utilizatori în plus, resetare runtime) cer flag-uri explicite.

Utilizare:
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py --apply
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py --apply --prune-users
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py --apply --assign-legacy-events
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py --apply --reset-runtime
"""

from __future__ import annotations

import sys
# Diacriticele românești se afișează corect indiferent de codarea consolei.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.config import settings  # noqa: E402
from backend.auth import get_password_hash  # noqa: E402

DB_PATH = settings.REPO_ROOT / "data" / "trash_detection.db"
LEGACY_DB_PATH = ROOT / "backend" / "trash_detection.db"
STALE_DB_PATHS = [ROOT / "backend" / "trashdet.db", LEGACY_DB_PATH]
BACKUP_DIR = settings.REPO_ROOT / "data" / "backups"


LOCAL_USERS = [
    {
        "username": "admin",
        "email": "admin@trash.local",
        "password": "Admin1234!",
        "role": "admin",
        "points": 0,
    },
    {
        "username": "operator",
        "email": "operator@trash.local",
        "password": "Operator1234!",
        "role": "user",
        "points": 0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pregătește baza de date locală TrashDet.")
    parser.add_argument("--apply", action="store_true", help="Scrie efectiv modificările. Fără el, doar simulare (dry-run).")
    parser.add_argument(
        "--prune-users",
        action="store_true",
        help="Șterge utilizatorii care nu sunt conturile locale documentate (admin/operator). Cere --apply.",
    )
    parser.add_argument(
        "--reset-runtime",
        action="store_true",
        help="Șterge incidentele/sesiunile/notificările generate. Cere --apply.",
    )
    parser.add_argument(
        "--reset-local-passwords",
        action="store_true",
        help="Resetează parolele conturilor locale la cele documentate.",
    )
    parser.add_argument(
        "--assign-legacy-events",
        action="store_true",
        help="Atribuie incidentele/sesiunile video fără proprietar userului local standard. Cere --apply.",
    )
    return parser.parse_args()


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    """True dacă tabelul există în baza de date."""
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def backup_db() -> Path:
    """Copiază baza de date într-un fișier de backup cu timestamp."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"trash_detection_{stamp}.db"
    shutil.copy2(DB_PATH, dst)
    return dst


def count(cur: sqlite3.Cursor, table: str) -> int:
    """Numărul de rânduri dintr-un tabel (0 dacă nu există)."""
    if not table_exists(cur, table):
        return 0
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def ensure_organization(cur: sqlite3.Cursor) -> None:
    """Asigură organizația implicită (id=1)."""
    cur.execute("SELECT id FROM organizations WHERE id = 1")
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO organizations (id, name, created_at) VALUES (1, ?, ?)",
            ("TrashDet Organization", datetime.now().isoformat(sep=" ")),
        )
        print("  [CREARE] organizația #1 TrashDet Organization")
    else:
        cur.execute("UPDATE organizations SET name = ? WHERE id = 1", ("TrashDet Organization",))
        print("  [UPDATE] organizația #1 -> TrashDet Organization")


def ensure_user(cur: sqlite3.Cursor, user: dict[str, object], reset_passwords: bool) -> None:
    """Creează sau actualizează un cont local (rol, email, opțional parolă)."""
    cur.execute("SELECT id, role FROM users WHERE username = ?", (user["username"],))
    row = cur.fetchone()
    if row:
        user_id, old_role = int(row[0]), row[1]
        fields = ["email = ?", "role = ?", "points = ?", "organization_id = 1"]
        params: list[object] = [user["email"], user["role"], user["points"]]
        if reset_passwords:
            fields.append("hashed_password = ?")
            params.append(get_password_hash(str(user["password"])))
        params.append(user_id)
        cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", params)
        msg = f"  [UPDATE] user {user['username']} rol {old_role!r} -> {user['role']!r}"
        if reset_passwords:
            msg += " + parolă resetată"
        print(msg)
        return

    cur.execute(
        """
        INSERT INTO users
            (username, email, hashed_password, role, points, organization_id, created_at)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        """,
        (
            user["username"],
            user["email"],
            get_password_hash(str(user["password"])),
            user["role"],
            user["points"],
            datetime.now().isoformat(sep=" "),
        ),
    )
    print(f"  [CREARE] user {user['username']} rol={user['role']}")


def get_user_id(cur: sqlite3.Cursor, username: str) -> int | None:
    """Întoarce id-ul unui utilizator după username, sau None."""
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def clean_notifications(cur: sqlite3.Cursor) -> None:
    """Șterge toate notificările."""
    if table_exists(cur, "notifications"):
        before = count(cur, "notifications")
        cur.execute("DELETE FROM notifications")
        print(f"  [DELETE] notificări: {before}")


def assign_legacy_runtime_to_local_user(cur: sqlite3.Cursor) -> None:
    """Atribuie rândurile vechi fără proprietar userului local, pentru o rulare coerentă."""
    local_user_id = get_user_id(cur, "operator")
    if local_user_id is None:
        print("  [SKIP] atribuire runtime vechi: userul local lipsește")
        return

    if table_exists(cur, "littering_events"):
        cur.execute("UPDATE littering_events SET organization_id = 1 WHERE organization_id IS NULL")
        cur.execute("SELECT COUNT(*) FROM littering_events WHERE reporter_id IS NULL")
        before = int(cur.fetchone()[0])
        cur.execute("UPDATE littering_events SET reporter_id = ? WHERE reporter_id IS NULL", (local_user_id,))
        print(f"  [UPDATE] incidente fără proprietar -> user local: {before}")

    if table_exists(cur, "video_sessions"):
        cur.execute("UPDATE video_sessions SET organization_id = 1 WHERE organization_id IS NULL")
        cur.execute("SELECT COUNT(*) FROM video_sessions WHERE user_id IS NULL")
        before = int(cur.fetchone()[0])
        cur.execute("UPDATE video_sessions SET user_id = ? WHERE user_id IS NULL", (local_user_id,))
        print(f"  [UPDATE] sesiuni video fără proprietar -> user local: {before}")


def prune_users(cur: sqlite3.Cursor) -> None:
    """Șterge utilizatorii care nu sunt conturile locale documentate."""
    keep = tuple(user["username"] for user in LOCAL_USERS)
    placeholders = ",".join("?" for _ in keep)
    cur.execute(f"SELECT COUNT(*) FROM users WHERE username NOT IN ({placeholders})", keep)
    before = int(cur.fetchone()[0])
    cur.execute(f"DELETE FROM users WHERE username NOT IN ({placeholders})", keep)
    print(f"  [DELETE] utilizatori în plus: {before}")


def reset_runtime(cur: sqlite3.Cursor) -> None:
    """Șterge toate datele generate la rulare (incidente, sesiuni, notificări)."""
    for table in ("video_sessions", "notifications", "littering_events"):
        if not table_exists(cur, table):
            continue
        n = count(cur, table)
        cur.execute(f"DELETE FROM {table}")
        print(f"  [DELETE] {table}: {n}")


def ensure_db_location() -> None:
    """Mută baza de date veche din backend/ în data/ dacă lipsește cea curentă."""
    if DB_PATH.exists() or not LEGACY_DB_PATH.exists():
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_DB_PATH, DB_PATH)
    print(f"  [MIGRARE] DB veche copiată în {DB_PATH}")


def remove_stale_db() -> None:
    """Șterge (sau arhivează) fișierele de DB vechi rămase în backend/."""
    for stale_db_path in STALE_DB_PATHS:
        if not stale_db_path.exists() or stale_db_path == DB_PATH:
            continue
        if stale_db_path.stat().st_size == 0:
            stale_db_path.unlink()
            print(f"  [DELETE] DB goală veche: {stale_db_path}")
        else:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = BACKUP_DIR / f"{stale_db_path.stem}_legacy_{stamp}{stale_db_path.suffix}"
            shutil.move(str(stale_db_path), str(archived))
            print(f"  [ARHIVĂ] DB veche mutată în: {archived}")


def print_summary(cur: sqlite3.Cursor) -> None:
    """Afișează numărul de rânduri pe tabele și conturile locale."""
    print("\nSumar:")
    for table in ["organizations", "users", "littering_events", "video_sessions", "notifications"]:
        if table_exists(cur, table):
            print(f"  {table:<22} {count(cur, table)}")

    print("\nConturi locale:")
    for user in LOCAL_USERS:
        print(f"  {user['username']:<8} / {user['password']:<12} / rol={user['role']}")


def main() -> int:
    args = parse_args()

    ensure_db_location()
    if not DB_PATH.exists():
        print(f"EROARE: baza de date nu există: {DB_PATH}")
        return 1
    if (args.prune_users or args.reset_runtime or args.reset_local_passwords or args.assign_legacy_events) and not args.apply:
        print("EROARE: flag-urile distructive/de resetare cer --apply")
        return 2

    print("Pregătire DB locală TrashDet")
    print(f"DB: {DB_PATH}")
    print(f"Mod: {'APPLY' if args.apply else 'DRY-RUN'}")

    if not args.apply:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        print_summary(cur)
        con.close()
        print("\nDoar simulare. Rulează din nou cu --apply pentru a scrie modificările.")
        return 0

    backup = backup_db()
    print(f"\n[BACKUP] {backup}")

    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        ensure_organization(cur)
        for user in LOCAL_USERS:
            ensure_user(cur, user, args.reset_local_passwords)
        clean_notifications(cur)
        if args.reset_runtime:
            reset_runtime(cur)
        if args.assign_legacy_events:
            assign_legacy_runtime_to_local_user(cur)
        if args.prune_users:
            prune_users(cur)
        con.commit()
        remove_stale_db()
        print_summary(cur)
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    print("\nGata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
