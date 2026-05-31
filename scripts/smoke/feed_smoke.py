"""
Manual smoke test for the incident feed endpoint.

Requires a running local server and an admin account. Safe to import during
pytest collection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests

__test__ = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.config import settings

BASE = settings.APP_BASE_URL.rstrip("/")


def main() -> int:
    for creds in [
        {"username": "admin", "password": "Admin1234!"},
        {"username": "admin_test", "password": "TestPass1!"},
    ]:
        resp = requests.post(f"{BASE}/api/auth/login", data=creds, timeout=5)
        if resp.status_code == 200 and "access_token" in resp.json():
            token = resp.json()["access_token"]
            break
    else:
        print("Nu am putut obtine token admin.")
        return 1

    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{BASE}/api/littering/events?limit=2", headers=headers, timeout=5)
    print("STATUS:", resp.status_code)
    data = resp.json()
    print("TYPE:", type(data))
    print("KEYS:", list(data.keys()) if isinstance(data, dict) else None)
    print("BODY:", json.dumps(data, indent=2, default=str)[:2000])
    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
