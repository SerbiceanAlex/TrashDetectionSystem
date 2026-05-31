"""
Prepare the local SQLite database for a clean thesis/demo run.

The script is intentionally conservative by default:
  - creates a timestamped DB backup before any write;
  - ensures one admin account and one operator account;
  - cleans OTP/test-login junk;
  - creates readable demo organization, monitored locations and authority contact;
  - removes the stale empty backend/trashdet.db file if present.

Destructive cleanup such as deleting extra users or runtime incidents requires
explicit flags.

Usage:
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_demo_db.py
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_demo_db.py --apply
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_demo_db.py --apply --prune-users
    .venv\\Scripts\\python.exe scripts\\maintenance\\prepare_demo_db.py --apply --reset-runtime
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "backend" / "trash_detection.db"
STALE_DB_PATH = ROOT / "backend" / "trashdet.db"
BACKUP_DIR = ROOT / "backend" / "backups"

sys.path.insert(0, str(ROOT))

from backend.auth import get_password_hash  # noqa: E402


DEMO_USERS = [
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

DEMO_LOCATIONS = [
    {
        "name": "Parcul Cetate - Camera principala",
        "address": "Alba Iulia, zona Cetate",
        "latitude": 46.0679,
        "longitude": 23.5708,
        "alert_email": "admin@trash.local",
    },
    {
        "name": "Campus universitar - Zona verde",
        "address": "Alba Iulia, campus universitar",
        "latitude": 46.0710,
        "longitude": 23.5728,
        "alert_email": "admin@trash.local",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TrashDet demo database.")
    parser.add_argument("--apply", action="store_true", help="Actually write changes. Without this, dry-run only.")
    parser.add_argument(
        "--prune-users",
        action="store_true",
        help="Delete users that are not demo accounts. Requires --apply.",
    )
    parser.add_argument(
        "--reset-runtime",
        action="store_true",
        help="Delete generated events/sessions/records/notifications/webhooks/authorities. Requires --apply.",
    )
    parser.add_argument(
        "--prune-locations",
        action="store_true",
        help="Delete monitored locations that are not demo locations. Requires --apply.",
    )
    parser.add_argument(
        "--reset-demo-passwords",
        action="store_true",
        help="Reset demo user passwords to the documented local demo passwords.",
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
                (1, 'TrashDet Demo Organization', 'pro', 1, 10, 999999, ?)
            """,
            (datetime.now().isoformat(sep=" "),),
        )
        print("  [CREATE] organization #1 TrashDet Demo Organization")
    else:
        cur.execute(
            """
            UPDATE organizations
            SET name = ?, plan = ?, subscription_active = 1,
                max_cameras = 10, max_incidents_month = 999999
            WHERE id = 1
            """,
            ("TrashDet Demo Organization", "pro"),
        )
        print("  [UPDATE] organization #1 -> TrashDet Demo Organization / pro")


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

    cur.execute("DELETE FROM monitored_locations WHERE name IN (?, ?)", tuple(loc["name"] for loc in DEMO_LOCATIONS))
    admin_id = get_user_id(cur, "admin")
    for loc in DEMO_LOCATIONS:
        cur.execute(
            """
            INSERT INTO monitored_locations
                (name, address, latitude, longitude, rtsp_url, alert_email,
                 is_active, created_at, created_by, organization_id)
            VALUES (?, ?, ?, ?, NULL, ?, 1, ?, ?, 1)
            """,
            (
                loc["name"],
                loc["address"],
                loc["latitude"],
                loc["longitude"],
                loc["alert_email"],
                datetime.now().isoformat(sep=" "),
                admin_id,
            ),
        )
        print(f"  [UPSERT] location {loc['name']}")


def keep_locations(cur: sqlite3.Cursor) -> None:
    if not table_exists(cur, "monitored_locations"):
        return
    keep = tuple(loc["name"] for loc in DEMO_LOCATIONS)
    placeholders = ",".join("?" for _ in keep)
    cur.execute(f"SELECT COUNT(*) FROM monitored_locations WHERE name NOT IN ({placeholders})", keep)
    before = int(cur.fetchone()[0])
    cur.execute(f"DELETE FROM monitored_locations WHERE name NOT IN ({placeholders})", keep)
    print(f"  [DELETE] extra monitored locations: {before}")


def ensure_authority(cur: sqlite3.Cursor) -> None:
    if not table_exists(cur, "authority_contacts"):
        return
    admin_id = get_user_id(cur, "admin")
    if admin_id is None:
        return
    cur.execute(
        "DELETE FROM authority_contacts WHERE name = ?",
        ("Primaria Alba Iulia - Serviciul salubritate",),
    )
    cur.execute(
        """
        INSERT INTO authority_contacts
            (name, email, area_description, created_by, organization_id, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (
            "Primaria Alba Iulia - Serviciul salubritate",
            "salubritate@example.local",
            "Contact demonstrativ pentru transmiterea incidentelor validate.",
            admin_id,
            datetime.now().isoformat(sep=" "),
        ),
    )
    print("  [UPSERT] authority contact")


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


def prune_users(cur: sqlite3.Cursor) -> None:
    keep = tuple(user["username"] for user in DEMO_USERS)
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


def remove_stale_db() -> None:
    if not STALE_DB_PATH.exists():
        return
    if STALE_DB_PATH.stat().st_size == 0:
        STALE_DB_PATH.unlink()
        print(f"  [DELETE] stale empty DB: {STALE_DB_PATH}")
    else:
        print(f"  [KEEP] stale DB is not empty, manual review needed: {STALE_DB_PATH}")


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
        "authority_contacts",
        "webhook_configs",
        "otp_codes",
    ]:
        if table_exists(cur, table):
            print(f"  {table:<22} {count(cur, table)}")

    print("\nDemo accounts:")
    for user in DEMO_USERS:
        print(f"  {user['username']:<8} / {user['password']:<12} / role={user['role']}")


def main() -> int:
    args = parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: database not found: {DB_PATH}")
        return 1
    if (args.prune_users or args.reset_runtime or args.reset_demo_passwords or args.prune_locations) and not args.apply:
        print("ERROR: destructive/reset flags require --apply")
        return 2

    print("TrashDet demo DB preparation")
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
        for user in DEMO_USERS:
            ensure_user(cur, user, args.reset_demo_passwords)
        clean_otp(cur)
        clean_notifications(cur)
        if args.reset_runtime:
            reset_runtime(cur)
        ensure_locations(cur)
        if args.prune_locations:
            keep_locations(cur)
        ensure_authority(cur)
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
