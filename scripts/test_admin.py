"""
Manual smoke test for the current admin/B2B API.

This script expects the FastAPI server to be running locally:

    .venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8000
    .venv\\Scripts\\python.exe scripts\\test_admin.py

It is safe to import; pytest collection will not execute requests.
"""

from __future__ import annotations

import sys

import requests

__test__ = False


BASE = "http://localhost:8000"


def check(label: str, condition: bool, detail: str = "") -> bool:
    if condition:
        print(f"  OK   {label}")
        return True
    print(f"  FAIL {label} {detail}")
    return False


def admin_token() -> str | None:
    credentials = [
        {"username": "admin", "password": "Admin123!"},
        {"username": "admin_test", "password": "TestPass1!"},
    ]
    for creds in credentials:
        try:
            resp = requests.post(f"{BASE}/api/auth/login", data=creds, timeout=5)
        except requests.RequestException:
            return None
        if resp.status_code == 200 and "access_token" in resp.json():
            print(f"  Login user: {creds['username']}")
            return resp.json()["access_token"]
    return None


def main() -> int:
    errors = 0

    print("=" * 60)
    print("ADMIN/B2B FLOW")
    print("=" * 60)

    token = admin_token()
    if not check("Admin login", token is not None, "server pornit si user admin existent?"):
        return 1

    headers = {"Authorization": f"Bearer {token}"}

    resp = requests.get(f"{BASE}/api/auth/me", headers=headers, timeout=5)
    errors += 0 if check("/api/auth/me", resp.status_code == 200, resp.text[:120]) else 1

    resp = requests.get(f"{BASE}/api/dashboard/b2b", headers=headers, timeout=5)
    ok_dashboard = resp.status_code == 200 and "pending_review" in resp.json()
    errors += 0 if check("B2B dashboard", ok_dashboard, resp.text[:120]) else 1

    resp = requests.get(f"{BASE}/api/littering/events?limit=5", headers=headers, timeout=5)
    ok_events = resp.status_code == 200 and "items" in resp.json()
    errors += 0 if check("Incident list", ok_events, resp.text[:120]) else 1
    if ok_events:
        print(f"       -> {resp.json().get('total', 0)} incidente")

    resp = requests.get(f"{BASE}/api/video/sessions?limit=5", headers=headers, timeout=5)
    ok_sessions = resp.status_code == 200 and "items" in resp.json()
    errors += 0 if check("Video sessions", ok_sessions, resp.text[:120]) else 1

    resp = requests.get(f"{BASE}/api/reports/stats", headers=headers, timeout=5)
    ok_reports = resp.status_code == 200 and "total_incidents" in resp.json()
    errors += 0 if check("Report stats", ok_reports, resp.text[:120]) else 1

    print("=" * 60)
    print("ALL CHECKS PASSED" if errors == 0 else f"FAILED: {errors} check(s)")
    print("=" * 60)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
