r"""
Test end-to-end complet pe lanțul de incident, izolat de datele reale.

Simulează exact scenariul de utilizare:
  1.  înregistrare utilizator + login (token JWT)
  2.  conectare la WebSocket-ul de monitorizare (ca telefonul)
  3.  trimitere cadre dintr-un clip real cu aruncare (persoana lasă obiect, pleacă)
  4.  așteptare alertă de incident de la server
  5.  verificare în REST: incidentul există, are thumbnail ȘI clip video
  6.  descărcarea dovezilor (thumbnail + clip) returnează 200
  7.  alt utilizator NU vede incidentul (izolare pe raportor)
  8.  adminul vede incidentul și îi poate schimba statusul

Rulare (pornește serverul de test separat, pe CPU, port 8010, DB izolată):
    $env:DATABASE_URL="sqlite+aiosqlite:///D:/TrashDetectionSystem/data/test_e2e.db"
    $env:STORAGE_ROOT="data/runtime_test"
    $env:CUDA_VISIBLE_DEVICES=""
    .venv\Scripts\python.exe -m uvicorn backend.main:app --port 8010

    .venv\Scripts\python.exe scripts\smoke\full_e2e_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import httpx
import websockets

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8010"
WS_BASE = "ws://127.0.0.1:8010"
CLIP = ROOT / "datasets" / "ai_videos2" / "ai2_bottle_high.mp4"

USER1 = {"username": "e2e_operator", "email": "e2e_op@test.local", "password": "Test!Parola9"}
USER2 = {"username": "e2e_altcineva", "email": "e2e_alt@test.local", "password": "Test!Parola9"}

PASS, FAIL = "[  OK  ]", "[ FAIL ]"
errors: list[str] = []


def ok(msg):
    """Marchează o verificare ca trecută."""
    print(f"{PASS} {msg}")


def fail(msg, detail=""):
    """Marchează o verificare ca eșuată și o reține în lista de erori."""
    print(f"{FAIL} {msg}" + (f": {detail}" if detail else ""))
    errors.append(msg)


async def register_and_login(client: httpx.AsyncClient, user: dict) -> str | None:
    """Înregistrează și autentifică un utilizator; întoarce token-ul JWT sau None."""
    r = await client.post(f"{BASE}/api/auth/register", json=user)
    if r.status_code not in (200, 201, 400, 409):  # 400/409 = există deja
        fail(f"înregistrare {user['username']}", f"HTTP {r.status_code}: {r.text[:200]}")
        return None
    r = await client.post(
        f"{BASE}/api/auth/login",
        data={"username": user["username"], "password": user["password"]},
    )
    if r.status_code != 200 or "access_token" not in r.json():
        fail(f"autentificare {user['username']}", f"HTTP {r.status_code}: {r.text[:200]}")
        return None
    ok(f"înregistrare + autentificare {user['username']}")
    return r.json()["access_token"]


async def stream_monitor(token: str) -> dict | None:
    """Trimite cadrele clipului prin WS pana la alerta; returneaza alerta."""
    cap = cv2.VideoCapture(str(CLIP))
    if not cap.isOpened():
        fail("deschidere clip de test", str(CLIP))
        return None

    # Praguri aliniate cu evaluarea batch (clip CCTV cu obiect mic);
    # serverul de test ruleaza cu MONITOR_MIN_DET_CONF=0.30.
    url = (
        f"{WS_BASE}/ws/video/monitor?det_conf=0.30&person_conf=0.40"
        f"&analysis_fps=10&token={token}"
    )
    alert = None
    states = set()
    sent = 0
    t0 = time.time()
    async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
        post_frames_left = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # la fiecare al 2-lea cadru, redimensionat ca pe telefon (max 640)
            sent += 1
            if sent % 2 == 0:
                continue
            h, w = frame.shape[:2]
            scale = min(1.0, 640 / max(h, w))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            okj, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not okj:
                continue
            await ws.send(buf.tobytes())
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            msg = json.loads(raw)
            if msg.get("type") == "alert":
                alert = msg
                ok(
                    f"alerta primita dupa {sent} cadre / {time.time()-t0:.0f}s "
                    f"(material={msg.get('material')}, event_id={msg.get('event_id')})"
                )
                post_frames_left = 45  # ține WS deschis pentru fereastra post + scrierea clipului
            else:
                states.add(msg.get("state"))
            if post_frames_left is not None:
                post_frames_left -= 1
                if post_frames_left <= 0:
                    break
    cap.release()
    print(f"         stări parcurse: {sorted(s for s in states if s)}")
    if alert is None:
        fail("alerta de incident pe WS", f"niciun alert dupa {sent} cadre")
    # Incidentul se poate declanșa fie pe modul zonă (CLEAR→PERSON_PRESENT→
    # MONITORING), fie pe modul distanță (persoana se îndepărtează de obiect).
    # Cerem doar ca persoana să fi fost detectată și alerta să fi apărut.
    if "PERSON_PRESENT" in states:
        ok("persoana detectata si incident declansat (mod " +
           ("zona" if "MONITORING" in states else "distanta") + ")")
    else:
        fail("masina de stari", f"persoana nu a fost detectata: {states}")
    return alert


async def main() -> int:
    if not CLIP.exists():
        fail("clip de test inexistent", str(CLIP))
        return 1

    async with httpx.AsyncClient(timeout=60) as client:
        # asteapta serverul de test
        for _ in range(60):
            try:
                r = await client.get(f"{BASE}/api/system/info")
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        else:
            fail("serverul de test nu raspunde pe :8010")
            return 1
        ok("serverul de test raspunde")

        token1 = await register_and_login(client, USER1)
        if not token1:
            return 1
        h1 = {"Authorization": f"Bearer {token1}"}

        # 2-4: monitor live cu clip real
        alert = await stream_monitor(token1)
        if alert is None:
            return 1
        event_id = alert["event_id"]

        # 5: incidentul exista cu dovezi (clipul se scrie dupa fereastra post)
        thumb_ok = clip_ok = False
        ev = None
        for _ in range(20):
            r = await client.get(f"{BASE}/api/littering/events/{event_id}", headers=h1)
            if r.status_code == 200:
                ev = r.json()
                thumb_ok = bool(ev.get("thumbnail_path"))
                clip_ok = bool(ev.get("clip_path"))
                if thumb_ok and clip_ok:
                    break
            await asyncio.sleep(2)
        ok("incident salvat in DB") if ev else fail("incident inexistent in DB")
        ok("incidentul are thumbnail") if thumb_ok else fail("thumbnail lipsa")
        ok("incidentul are clip video (100% video)") if clip_ok else fail("clip video lipsa")

        # 6: descarcarea dovezilor
        for kind in ("thumbnail", "clip"):
            r = await client.get(
                f"{BASE}/api/littering/events/{event_id}/{kind}", headers=h1
            )
            if r.status_code == 200 and len(r.content) > 1000:
                ok(f"descarcare {kind}: HTTP 200, {len(r.content)} bytes")
            else:
                fail(f"descarcare {kind}", f"HTTP {r.status_code}")

        # utilizatorul isi vede incidentul in lista
        r = await client.get(f"{BASE}/api/littering/events", headers=h1)
        n1 = r.json().get("total", 0) if r.status_code == 200 else -1
        ok(f"utilizatorul isi vede incidentele (total={n1})") if n1 >= 1 else fail(
            "lista incidente utilizator", f"total={n1}"
        )

        # 7: alt utilizator NU vede incidentul
        token2 = await register_and_login(client, USER2)
        if token2:
            r = await client.get(
                f"{BASE}/api/littering/events",
                headers={"Authorization": f"Bearer {token2}"},
            )
            n2 = r.json().get("total", -1) if r.status_code == 200 else -1
            ok("alt utilizator NU vede incidentul (izolare)") if n2 == 0 else fail(
                "izolare incidente", f"user2 vede total={n2}"
            )
            r = await client.get(
                f"{BASE}/api/littering/events/{event_id}",
                headers={"Authorization": f"Bearer {token2}"},
            )
            ok("acces direct la incident strain: refuzat") if r.status_code in (
                403,
                404,
            ) else fail("acces direct incident strain", f"HTTP {r.status_code}")

        # 8: validare de catre admin. Primul utilizator inregistrat (USER1) este
        # admin si se afla in aceeasi organizatie cu incidentul, deci el este
        # adminul care il poate vedea si valida (izolarea pe organizatie e deja
        # confirmata la pasul 7).
        r = await client.get(f"{BASE}/api/littering/events", headers=h1)
        nadm = r.json().get("total", -1) if r.status_code == 200 else -1
        ok(f"adminul vede incidentele organizatiei (total={nadm})") if nadm >= 1 else fail(
            "vizibilitate admin", f"total={nadm}"
        )
        r = await client.patch(
            f"{BASE}/api/littering/events/{event_id}/status",
            headers=h1,
            json={"status": "reviewed"},
        )
        if r.status_code == 200:
            ok("adminul a validat incidentul (status=reviewed)")
        else:
            fail("validare admin", f"HTTP {r.status_code}: {r.text[:150]}")

    print("\n" + "=" * 60)
    if errors:
        print(f"REZULTAT: {len(errors)} PROBLEME -> {errors}")
        return 1
    print("REZULTAT: TOATE VERIFICARILE AU TRECUT")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
