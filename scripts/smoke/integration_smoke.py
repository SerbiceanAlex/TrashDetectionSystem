"""
Integration test — rulează cu serverul pornit pe port 8000.

Testează:
  1. Batch detection (POST /api/detect cu imagine reală)
  2. Creare manuală LitteringEvent în DB
  3. REST API incidente (list / get / patch status)
  4. Simulare WebSocket monitor cu frame real

Rulare:
    .venv\\Scripts\\python.exe scripts\\smoke\\integration_smoke.py
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import websockets

__test__ = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.config import settings

BASE = settings.APP_BASE_URL.rstrip("/")
WS_BASE = BASE.replace("https://", "wss://").replace("http://", "ws://")
TEST_IMAGE = ROOT / "datasets" / "raw" / "images" / "park01_download_img_001.jpg"

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
RESET = "\033[0m"
BOLD  = "\033[1m"

_pass = lambda msg: (print(f"{GREEN}[  OK  ]{RESET} {msg}"), _counters.__setitem__('ok', _counters['ok'] + 1))
_fail = lambda msg: (print(f"{RED}[ FAIL ]{RESET} {msg}"), _counters.__setitem__('fail', _counters['fail'] + 1))
_info = lambda msg: print(f"{YELLOW}[ INFO ]{RESET} {msg}")
_counters = {'ok': 0, 'fail': 0}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_admin_token(client: httpx.Client) -> str:
    """Login as admin and return JWT token."""
    for creds in [
        {"username": "admin", "password": "Admin1234!"},
        {"username": "operator", "password": "Operator1234!"},
    ]:
        # OAuth2PasswordRequestForm requires form data
        r = client.post(f"{BASE}/api/auth/login", data=creds)
        if r.status_code == 200:
            data = r.json()
            if "access_token" in data:
                _info(f"Login reușit ca: {creds['username']}")
                return data["access_token"]
    return ""


def sep(title: str):
    print(f"\n{'='*60}")
    print(f"  {BOLD}{title}{RESET}")
    print('='*60)


# ── TEST 1: Batch detection ───────────────────────────────────────────────────

def test_batch_detection(client: httpx.Client, token: str = "") -> int | None:
    sep("TEST 1 — Batch Detection (POST /api/detect)")
    if not TEST_IMAGE.exists():
        _fail(f"Imagine test lipsă: {TEST_IMAGE}")
        return None

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with open(TEST_IMAGE, "rb") as f:
        r = client.post(
            f"{BASE}/api/detect?det_conf=0.30",
            files={"file": ("test.jpg", f, "image/jpeg")},
            headers=headers,
            timeout=60,
        )

    if r.status_code != 200:
        _fail(f"HTTP {r.status_code}: {r.text[:200]}")
        return None

    data = r.json()
    session_id = data.get("session_id")
    objects = data.get("total_objects", 0)
    ms = data.get("inference_ms", 0)
    _pass(f"Detecție reușită: {objects} obiecte în {ms:.0f}ms | session_id={session_id}")

    if objects > 0:
        mats = [d["material"] for d in data.get("detections", [])]
        _info(f"Materiale detectate: {mats}")
        _pass("Clasificator material a rulat pe crop-uri reale")
    else:
        _info("0 obiecte — imaginea poate fi prea curată sau conf prea mare")

    # Check annotated image URL
    ann = data.get("annotated_url", "")
    if ann:
        r2 = client.get(f"{BASE}{ann}", timeout=10)
        if r2.status_code == 200:
            _pass(f"Imagine adnotată accesibilă: {ann}")
        else:
            _fail(f"Imaginea adnotată nu e accesibilă: {ann} → {r2.status_code}")

    return session_id


# ── TEST 2: Create littering event in DB ──────────────────────────────────────

def test_create_littering_event(token: str) -> int | None:
    sep("TEST 2 — Creare LitteringEvent în DB (direct Python)")
    try:
        import asyncio
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from backend import database as db

        async def _create():
            await db.create_tables()
            async with db.AsyncSessionLocal() as session:
                evt = await db.create_littering_event(
                    session,
                    material="plastic",
                    det_score=0.82,
                    person_present=True,
                    person_count=1,
                    latitude=44.4268,
                    longitude=26.1025,
                    address="Parcul Herăstrău, București",
                )
                return evt.id

        event_id = asyncio.run(_create())
        _pass(f"LitteringEvent creat cu id={event_id}")
        return event_id
    except Exception as e:
        _fail(f"Eroare: {e}")
        return None


# ── TEST 3: REST API incidente ────────────────────────────────────────────────

def test_littering_api(client: httpx.Client, token: str, event_id: int):
    sep("TEST 3 — REST API /api/littering/events")

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # List
    r = client.get(f"{BASE}/api/littering/events", headers=headers, timeout=10)
    if r.status_code == 200:
        data = r.json()
        _pass(f"GET /api/littering/events → total={data['total']}")
    elif r.status_code == 401:
        _info("GET /api/littering/events → 401 (nu ești admin — e corect)")
        return
    else:
        _fail(f"GET /api/littering/events → HTTP {r.status_code}")
        return

    # Get by ID
    if event_id:
        r2 = client.get(f"{BASE}/api/littering/events/{event_id}", headers=headers, timeout=10)
        if r2.status_code == 200:
            ev = r2.json()
            _pass(f"GET /api/littering/events/{event_id} → material={ev['material']} status={ev['status']}")
        else:
            _fail(f"GET /api/littering/events/{event_id} → HTTP {r2.status_code}")

    # PATCH status → reviewed
    if event_id and token:
        r3 = client.patch(
            f"{BASE}/api/littering/events/{event_id}/status",
            headers={**headers, "Content-Type": "application/json"},
            content=json.dumps({"status": "reviewed", "notes": "Test automat"}),
            timeout=10,
        )
        if r3.status_code == 200:
            _pass(f"PATCH status → reviewed ✓ (notes salvate)")
        else:
            _fail(f"PATCH status → HTTP {r3.status_code}: {r3.text[:200]}")

    # Filter by status
    r4 = client.get(f"{BASE}/api/littering/events?status=pending", headers=headers, timeout=10)
    if r4.status_code == 200:
        _pass(f"Filter ?status=pending → total={r4.json()['total']}")

    # Filter by material
    r5 = client.get(f"{BASE}/api/littering/events?material=plastic", headers=headers, timeout=10)
    if r5.status_code == 200:
        _pass(f"Filter ?material=plastic → total={r5.json()['total']}")


# ── TEST 4: Monitor WebSocket cu frame real ───────────────────────────────────

async def test_monitor_websocket(token: str):
    sep("TEST 4 — Monitor WebSocket (/ws/video/monitor)")

    if not TEST_IMAGE.exists():
        _fail(f"Imagine test lipsă: {TEST_IMAGE}")
        return

    frame = cv2.imread(str(TEST_IMAGE))
    if frame is None:
        _fail("Nu s-a putut citi imaginea de test")
        return

    # Resize la 640px pentru viteză
    h, w = frame.shape[:2]
    scale = 640 / max(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg_bytes = buf.tobytes()

    uri = f"{WS_BASE}/ws/video/monitor?det_conf=0.30&person_conf=0.40"

    try:
        async with websockets.connect(uri, open_timeout=10) as ws:
            _info("WebSocket conectat — trimit 5 frame-uri...")

            alerts = []
            frame_responses = []

            for i in range(5):
                await ws.send(jpeg_bytes)
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    msg = json.loads(raw)
                    if msg.get("type") == "alert":
                        alerts.append(msg)
                        _info(f"  Frame {i+1}: ALERT! event_id={msg.get('event_id')} material={msg.get('material')}")
                    else:
                        state = msg.get("state", "?")
                        persons = msg.get("persons", 0)
                        trash = msg.get("trash", 0)
                        fps = msg.get("fps", 0)
                        frame_responses.append(msg)
                        _info(f"  Frame {i+1}: state={state} persons={persons} trash={trash} fps={fps:.1f}")
                except asyncio.TimeoutError:
                    _fail(f"  Frame {i+1}: timeout la răspuns (>15s)")
                    break

            if frame_responses:
                _pass(f"Monitor WS funcționează — {len(frame_responses)} răspunsuri primite")
            if alerts:
                _pass(f"ALERT generat! event_id={alerts[0]['event_id']}")
            else:
                _info("Niciun alert — normal (nu e persoană+gunoi în imagine)")

    except Exception as e:
        _fail(f"WebSocket error: {e}")


# ── TEST 5: Batch API multi-imagine ──────────────────────────────────────────

def test_batch_api(client: httpx.Client, token: str = ""):
    sep("TEST 5 — Batch API (POST /api/detect/batch)")
    if not TEST_IMAGE.exists():
        _fail("Imagine test lipsă")
        return

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with open(TEST_IMAGE, "rb") as f1:
        img_bytes = f1.read()

    # Trimitem aceeași imagine de 2 ori simulând batch
    r = client.post(
        f"{BASE}/api/detect/batch?det_conf=0.30",
        files=[
            ("files", ("img1.jpg", img_bytes, "image/jpeg")),
            ("files", ("img2.jpg", img_bytes, "image/jpeg")),
        ],
        headers=headers,
        timeout=120,
    )

    if r.status_code == 200:
        data = r.json()
        total_files = data.get("total_files", 0)
        total_obj = data.get("total_objects", 0)
        _pass(f"Batch OK: {total_files} fișiere, {total_obj} obiecte totale")
        for res in data.get("results", []):
            _info(f"  {res.get('filename','?')} → {res.get('total_objects',0)} obiecte")
    elif r.status_code == 404:
        _info("Endpoint /api/detect/batch nu există (OK — nu e obligatoriu)")
    else:
        _fail(f"Batch → HTTP {r.status_code}: {r.text[:200]}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*60}")
    print("  INTEGRATION TEST — Trash Detection System")
    print(f"  Server: {BASE}")
    print(f"{'='*60}{RESET}")

    # Check server
    try:
        r = httpx.get(f"{BASE}/api/sessions?limit=1", timeout=5)
        _pass(f"Server accesibil — HTTP {r.status_code}")
    except Exception as e:
        _fail(f"Server INACCESIBIL pe {BASE}: {e}")
        print("\nPornește serverul mai întâi:")
        print("  .venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000")
        sys.exit(1)

    with httpx.Client(timeout=30) as client:
        # Get admin token
        token = get_admin_token(client)
        if token:
            _pass(f"Login admin reușit — token obținut")
        else:
            _info("Nu s-a putut face login ca admin — testele protejate vor fi sărite")

        # Run tests
        session_id = test_batch_detection(client, token)
        event_id = test_create_littering_event(token)
        test_littering_api(client, token, event_id)
        test_batch_api(client, token)

    # Async test for WebSocket
    asyncio.run(test_monitor_websocket(token))

    sep("SUMAR FINAL")
    ok, fail = _counters['ok'], _counters['fail']
    print(f"\n  {GREEN}{ok} PASS{RESET} / {RED if fail else GREEN}{fail} FAIL{RESET}\n")
    print(f"{GREEN}Rulează serverul și deschide {BASE}{RESET}")
    print(f"Login admin → tab {BOLD}Scanare → Monitor{RESET} pentru test live.")
    print(f"Sau tab {BOLD}Admin → Incidente{RESET} pentru a vedea evenimentele create.")
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
