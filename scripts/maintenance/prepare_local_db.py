"""
Prepare the local SQLite database for a clean thesis/local run.

The script is intentionally conservative by default:
  - creates a timestamped DB backup before any write;
  - ensures one admin account and one standard local user account;
  - cleans OTP/test-login junk;
  - creates readable local organization and monitored locations;
  - removes legacy authority/webhook rows because the final thesis build keeps
    evidence review local instead of forwarding by email/webhook;
  - removes stale legacy DB files from backend/ when safe.

Destructive cleanup such as deleting extra users or runtime incidents requires
explicit flags.

Usage:
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py --apply
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py --apply --prune-users
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py --apply --assign-legacy-events
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_local_db.py --apply --reset-runtime
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
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

LOCAL_LOCATIONS = [
    {
        "name": "Parcul Cetate - Camera principala",
        "address": "Alba Iulia, zona Cetate",
        "latitude": 46.0679,
        "longitude": 23.5708,
    },
    {
        "name": "Campus universitar - Zona verde",
        "address": "Alba Iulia, campus universitar",
        "latitude": 46.0710,
        "longitude": 23.5728,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TrashDet local database.")
    parser.add_argument("--apply", action="store_true", help="Actually write changes. Without this, dry-run only.")
    parser.add_argument(
        "--prune-users",
        action="store_true",
        help="Delete users that are not the documented local admin/user accounts. Requires --apply.",
    )
    parser.add_argument(
        "--reset-runtime",
        action="store_true",
        help="Delete generated events/sessions/records/notifications and legacy integrations. Requires --apply.",
    )
    parser.add_argument(
        "--prune-locations",
        action="store_true",
        help="Delete monitored locations that are not the local configured locations. Requires --apply.",
    )
    parser.add_argument(
        "--reset-local-passwords",
        action="store_true",
        help="Reset local user passwords to the documented local passwords.",
    )
    parser.add_argument(
        "--assign-legacy-events",
        action="store_true",
        help="Assign existing unowned littering events/video sessions to the local standard user. Requires --apply.",
    )
    return parser.parse_args()


def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None


def column_exists(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def backup_db() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"trash_detection_{stamp}.db"
    shutil.copy2(DB_PATH, dst)
    return dst


def count(cur: sqlite3.Cursor, table: str) -> int:
    if not table_exists(cur, table):
        return 0
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cur.fetchone()[0])


def ensure_organization(cur: sqlite3.Cursor) -> None:
    cur.execute("SELECT id FROM organizations WHERE id = 1")
    if cur.fetchone() is None:
        cur.execute(
            """
            INSERT INTO organizations
                (id, name, plan, subscription_active, max_cameras, max_incidents_month, created_at)
            VALUES
                (1, 'TrashDet Organization', 'pro', 1, 10, 999999, ?)
            """,
            (datetime.now().isoformat(sep=" "),),
        )
        print("  [CREATE] organization #1 TrashDet Organization")
    else:
        cur.execute(
            """
            UPDATE organizations
            SET name = ?, plan = ?, subscription_active = 1,
                max_cameras = 10, max_incidents_month = 999999
            WHERE id = 1
            """,
            ("TrashDet Organization", "pro"),
        )
        print("  [UPDATE] organization #1 -> TrashDet Organization / pro")


def ensure_user(cur: sqlite3.Cursor, user: dict[str, object], reset_passwords: bool) -> None:
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
        msg = f"  [UPDATE] user {user['username']} role {old_role!r} -> {user['role']!r}"
        if reset_passwords:
            msg += " + password reset"
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
    print(f"  [CREATE] user {user['username']} role={user['role']}")


def ensure_locations(cur: sqlite3.Cursor) -> None:
    if not table_exists(cur, "monitored_locations"):
        return

    cur.execute("DELETE FROM monitored_locations WHERE name IN (?, ?)", tuple(loc["name"] for loc in LOCAL_LOCATIONS))
    admin_id = get_user_id(cur, "admin")
    for loc in LOCAL_LOCATIONS:
        cur.execute(
            """
            INSERT INTO monitored_locations
                (name, address, latitude, longitude, rtsp_url, alert_email,
                 is_active, created_at, created_by, organization_id)
            VALUES (?, ?, ?, ?, NULL, NULL, 1, ?, ?, 1)
            """,
            (
                loc["name"],
                loc["address"],
                loc["latitude"],
                loc["longitude"],
                datetime.now().isoformat(sep=" "),
                admin_id,
            ),
        )
        print(f"  [UPSERT] location {loc['name']}")


def keep_locations(cur: sqlite3.Cursor) -> None:
    if not table_exists(cur, "monitored_locations"):
        return
    keep = tuple(loc["name"] for loc in LOCAL_LOCATIONS)
    placeholders = ",".join("?" for _ in keep)
    cur.execute(f"SELECT COUNT(*) FROM monitored_locations WHERE name NOT IN ({placeholders})", keep)
    before = int(cur.fetchone()[0])
    cur.execute(f"DELETE FROM monitored_locations WHERE name NOT IN ({placeholders})", keep)
    print(f"  [DELETE] extra monitored locations: {before}")


def get_user_id(cur: sqlite3.Cursor, username: str) -> int | None:
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def clean_otp(cur: sqlite3.Cursor) -> None:
    if table_exists(cur, "otp_codes"):
        before = count(cur, "otp_codes")
        cur.execute("DELETE FROM otp_codes")
        print(f"  [DELETE] otp_codes rows: {before}")


def clean_notifications(cur: sqlite3.Cursor) -> None:
    if table_exists(cur, "notifications"):
        before = count(cur, "notifications")
        cur.execute("DELETE FROM notifications")
        print(f"  [DELETE] notifications rows: {before}")


def clean_legacy_integrations(cur: sqlite3.Cursor) -> None:
    """Remove rows from discontinued integration tables, if the old DB has them."""
    for table in ("authority_contacts", "webhook_configs"):
        if not table_exists(cur, table):
            continue
        before = count(cur, table)
        cur.execute(f"DELETE FROM {table}")
        print(f"  [DELETE] legacy {table}: {before}")


def assign_legacy_runtime_to_local_user(cur: sqlite3.Cursor) -> None:
    """Attach old unowned runtime rows to the local standard user for a coherent run."""
    local_user_id = get_user_id(cur, "operator")
    if local_user_id is None:
        print("  [SKIP] legacy runtime assignment: local user missing")
        return

    if table_exists(cur, "littering_events"):
        cur.execute("UPDATE littering_events SET organization_id = 1 WHERE organization_id IS NULL")
        cur.execute("SELECT COUNT(*) FROM littering_events WHERE reporter_id IS NULL")
        before = int(cur.fetchone()[0])
        cur.execute("UPDATE littering_events SET reporter_id = ? WHERE reporter_id IS NULL", (local_user_id,))
        print(f"  [UPDATE] unowned littering events -> local user: {before}")

    if table_exists(cur, "video_sessions"):
        cur.execute("UPDATE video_sessions SET organization_id = 1 WHERE organization_id IS NULL")
        cur.execute("SELECT COUNT(*) FROM video_sessions WHERE user_id IS NULL")
        before = int(cur.fetchone()[0])
        cur.execute("UPDATE video_sessions SET user_id = ? WHERE user_id IS NULL", (local_user_id,))
        print(f"  [UPDATE] unowned video sessions -> local user: {before}")

    if table_exists(cur, "detection_sessions"):
        cur.execute("UPDATE detection_sessions SET organization_id = 1 WHERE organization_id IS NULL")
        cur.execute("SELECT COUNT(*) FROM detection_sessions WHERE reporter_id IS NULL")
        before = int(cur.fetchone()[0])
        cur.execute("UPDATE detection_sessions SET reporter_id = ? WHERE reporter_id IS NULL", (local_user_id,))
        print(f"  [UPDATE] unowned image sessions -> local user: {before}")


def prune_users(cur: sqlite3.Cursor) -> None:
    keep = tuple(user["username"] for user in LOCAL_USERS)
    placeholders = ",".join("?" for _ in keep)
    cur.execute(f"SELECT COUNT(*) FROM users WHERE username NOT IN ({placeholders})", keep)
    before = int(cur.fetchone()[0])
    cur.execute(f"DELETE FROM users WHERE username NOT IN ({placeholders})", keep)
    print(f"  [DELETE] extra users: {before}")


def reset_runtime(cur: sqlite3.Cursor) -> None:
    tables = [
        "detection_records",
        "video_sessions",
        "detection_sessions",
        "notifications",
        "webhook_configs",
        "authority_contacts",
        "littering_events",
    ]
    for table in tables:
        if not table_exists(cur, table):
            continue
        n = count(cur, table)
        cur.execute(f"DELETE FROM {table}")
        print(f"  [DELETE] {table}: {n}")


def ensure_db_location() -> None:
    if DB_PATH.exists() or not LEGACY_DB_PATH.exists():
        return
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LEGACY_DB_PATH, DB_PATH)
    print(f"  [MIGRATE] copied legacy DB to {DB_PATH}")


def remove_stale_db() -> None:
    for stale_db_path in STALE_DB_PATHS:
        if not stale_db_path.exists() or stale_db_path == DB_PATH:
            continue
        if stale_db_path.stat().st_size == 0:
            stale_db_path.unlink()
            print(f"  [DELETE] stale empty DB: {stale_db_path}")
        else:
            BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived = BACKUP_DIR / f"{stale_db_path.stem}_legacy_{stamp}{stale_db_path.suffix}"
            shutil.move(str(stale_db_path), str(archived))
            print(f"  [ARCHIVE] legacy DB moved to: {archived}")


def print_summary(cur: sqlite3.Cursor) -> None:
    print("\nSummary:")
    for table in [
        "organizations",
        "users",
        "monitored_locations",
        "littering_events",
        "video_sessions",
        "detection_records",
        "notifications",
        "otp_codes",
    ]:
        if table_exists(cur, table):
            print(f"  {table:<22} {count(cur, table)}")

    print("\nLocal accounts:")
    for user in LOCAL_USERS:
        print(f"  {user['username']:<8} / {user['password']:<12} / role={user['role']}")


def main() -> int:
    args = parse_args()

    ensure_db_location()
    if not DB_PATH.exists():
        print(f"ERROR: database not found: {DB_PATH}")
        return 1
    if (args.prune_users or args.reset_runtime or args.reset_local_passwords or args.prune_locations or args.assign_legacy_events) and not args.apply:
        print("ERROR: destructive/reset flags require --apply")
        return 2

    print("TrashDet local DB preparation")
    print(f"DB: {DB_PATH}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    if not args.apply:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        print_summary(cur)
        con.close()
        print("\nDry-run only. Re-run with --apply to write changes.")
        return 0

    backup = backup_db()
    print(f"\n[BACKUP] {backup}")

    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        ensure_organization(cur)
        for user in LOCAL_USERS:
            ensure_user(cur, user, args.reset_local_passwords)
        clean_otp(cur)
        clean_notifications(cur)
        clean_legacy_integrations(cur)
        if args.reset_runtime:
            reset_runtime(cur)
        if args.assign_legacy_events:
            assign_legacy_runtime_to_local_user(cur)
        ensure_locations(cur)
        if args.prune_locations:
            keep_locations(cur)
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

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
