"""
test_batch_and_monitor.py
=========================
Testează automat:
  1. BATCH  — trimite 5 imagini din datasets/parks_detect_final/images/test la /api/detect
  2. MONITOR — simulează starea mașinii LitteringDetector cu frame-uri reale
               (fără cameră: folosește imaginile de park din dataset)

Rulare:
    .venv/Scripts/python.exe scripts/smoke/batch_monitor_smoke.py
"""

import asyncio
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

__test__ = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.config import settings

API = settings.APP_BASE_URL.rstrip("/")
IMAGES_DIR = ROOT / "datasets" / "parks_detect_final" / "images" / "test"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def header(title: str):
    print(f"\n{BOLD}{CYAN}{'═'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*60}{RESET}")

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def info(msg): print(f"  {YELLOW}→{RESET} {msg}")


def get_token() -> str | None:
    """Login ca admin și returnează JWT token."""
    for creds in [
        {"username": "admin", "password": "Admin1234!"},
        {"username": "operator", "password": "Operator1234!"},
    ]:
        try:
            r = requests.post(f"{API}/api/auth/login", data=creds, timeout=5)
            if r.ok:
                return r.json().get("access_token")
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — BATCH
# ─────────────────────────────────────────────────────────────────────────────

def test_batch(token: str | None):
    header("TEST 1 — BATCH (upload 5 imagini simultan)")

    imgs = sorted(IMAGES_DIR.glob("*.jpg"))[:5]
    if not imgs:
        fail(f"Nicio imagine găsită în {IMAGES_DIR}")
        return

    info(f"Imagini găsite: {len(imgs)}")
    for p in imgs:
        info(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Trimite una câte una (endpoint batch nu există, dar /api/detect acceptă câte una)
    # Sau folosim /api/detect/batch dacă există
    ok_count = 0
    total_objects = 0

    for img_path in imgs:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            try:
                r = requests.post(
                    f"{API}/api/detect?det_conf=0.35",
                    files=files,
                    headers=headers,
                    timeout=30,
                )
                if r.ok:
                    data = r.json()
                    n = data.get("total_objects", 0)
                    ms = data.get("inference_ms", 0)
                    total_objects += n
                    ok_count += 1
                    ok(f"{img_path.name:40s}  {n:2d} obiecte  {ms:.0f}ms")
                else:
                    fail(f"{img_path.name}: HTTP {r.status_code} — {r.text[:80]}")
            except requests.exceptions.ConnectionError:
                fail("Serverul nu răspunde! Porneste uvicorn mai intai.")
                return
            except Exception as e:
                fail(f"{img_path.name}: {e}")

    print()
    if ok_count == len(imgs):
        ok(f"BATCH OK — {ok_count}/{len(imgs)} imagini procesate, {total_objects} obiecte total")
    else:
        fail(f"BATCH PARTIAL — {ok_count}/{len(imgs)} imagini OK")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — MONITOR state machine (fără server, direct Python)
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_detections(n_trash: int, base_x: int = 100, base_y: int = 100):
    """Generează n_trash detecții false de gunoi cu track_id-uri unice."""
    dets = []
    for i in range(n_trash):
        dets.append({
            "track_id": 1000 + i,
            "box": (base_x + i*40, base_y, base_x + i*40 + 30, base_y + 30),
            "det_score": 0.75,
            "material_name": "plastic",
        })
    return dets


def test_monitor_state_machine():
    header("TEST 2 — MONITOR (state machine LitteringDetector)")

    try:
        from backend.littering_detector import LitteringDetector, DetectorState
    except ImportError as e:
        fail(f"Import eșuat: {e}")
        return

    info("Creez LitteringDetector(fps=10, monitor_seconds=3, pre_event_seconds=2)")
    det = LitteringDetector(fps=10, monitor_seconds=3.0, pre_event_seconds=2.0)

    # Creez un frame fals (640x480 verde)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 50, 30)

    # ── Faza 1: CLEAR (nicio persoană, niciun gunoi)
    event = det.update(frame, [], [])
    assert det.current_state == "CLEAR", f"Așteptat CLEAR, got {det.current_state}"
    ok("Faza 1 CLEAR — nicio persoană, niciun gunoi → state=CLEAR")

    # ── Faza 2: Persoană apare (3 frame-uri)
    person_box = (200, 100, 400, 450)  # (x1,y1,x2,y2)
    trash_existing = _make_fake_detections(1, base_x=50)  # gunoi PRE-existent
    for i in range(3):
        event = det.update(frame, trash_existing, [person_box])
    assert det.current_state == "PERSON_PRESENT", f"Așteptat PERSON_PRESENT, got {det.current_state}"
    ok("Faza 2 PERSON_PRESENT — persoana e în cadru → state=PERSON_PRESENT")

    # ── Faza 3: Persoana pleacă → MONITORING
    for i in range(5):
        event = det.update(frame, trash_existing, [])
    assert det.current_state == "MONITORING", f"Așteptat MONITORING, got {det.current_state}"
    ok("Faza 3 MONITORING — persoana a plecat → state=MONITORING, countdown pornit")

    # ── Faza 4: Gunoi NOU apare în zona fostei persoane → EVENT!
    trash_new = trash_existing + [
        {
            "track_id": 9999,  # track_id nou = obiect nou!
            "box": (220, 200, 350, 380),  # în centrul zonei persoanei
            "det_score": 0.80,
            "material_name": "plastic",
        }
    ]
    event = det.update(frame, trash_new, [])

    if event is not None:
        ok(f"Faza 4 EVENT DETECTAT! material={event.material}, det_score={event.det_score:.2f}")
        ok(f"  trash_box={event.trash_box}, person_box={event.person_box}")
    else:
        fail("Faza 4 — eveniment NEFOCALIZAT (ar trebui să se detecteze)")
        info("  Posibil: track_id 9999 nu e în zona expanded a persoanei")
        info("  Zona persoanei: " + str(det._person_zones))

    print()
    ok("State machine test complet") if event else fail("State machine test EȘUAT")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — MONITOR cu imagini reale din dataset (simulare video)
# ─────────────────────────────────────────────────────────────────────────────

def test_monitor_with_real_images():
    header("TEST 3 — MONITOR cu imagini reale (simulare inferență)")

    try:
        from backend.inference import load_models, run_pipeline_frame, detect_persons
        from backend.littering_detector import LitteringDetector
        from backend.config import settings
    except ImportError as e:
        fail(f"Import eșuat: {e}")
        return

    info("Încarc modelele (poate dura 10-30s prima dată)...")
    try:
        load_models()
        ok("Modele încărcate")
    except Exception as e:
        fail(f"Eroare la încărcare modele: {e}")
        return

    imgs = sorted(IMAGES_DIR.glob("*.jpg"))[:5]
    if not imgs:
        fail("Nicio imagine găsită")
        return

    det = LitteringDetector(fps=5, monitor_seconds=5.0, pre_event_seconds=2.0)

    info(f"Procesez {len(imgs)} imagini ca dacă ar fi frame-uri dintr-un video...")
    print()

    total_trash = 0
    for i, img_path in enumerate(imgs):
        frame = cv2.imread(str(img_path))
        if frame is None:
            fail(f"Nu pot citi {img_path.name}")
            continue

        t0 = time.perf_counter()
        trash_dets, _, ms = run_pipeline_frame(frame, det_conf=0.35)
        person_boxes = detect_persons(frame, conf=0.40)
        elapsed = (time.perf_counter() - t0) * 1000

        total_trash += len(trash_dets)

        print(f"  Frame {i+1:2d} | {img_path.name:40s} | "
              f"trash={len(trash_dets):2d}  persons={len(person_boxes)}  "
              f"{elapsed:.0f}ms  state={det.current_state}")

        event = det.update(frame, trash_dets, person_boxes)
        if event:
            ok(f"  → EVENT la frame {i+1}! material={event.material}")

    print()
    ok(f"Total gunoi detectat: {total_trash} obiecte în {len(imgs)} frame-uri")
    info("Notă: evenimentul de littering necesită secvența PERSON_PRESENT → MONITORING → trash nou")
    info("      Imaginile statice nu vor declanșa event (nu există persoană care aruncă)")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Verifică că endpoint-urile /api/littering exist
# ─────────────────────────────────────────────────────────────────────────────

def test_littering_endpoints(token: str | None):
    header("TEST 4 — REST endpoints /api/littering")

    if not token:
        fail("Nu am token de admin — skip endpoint test")
        info("  Porneste serverul si inregistreaza user admin")
        return

    headers = {"Authorization": f"Bearer {token}"}

    # GET /api/littering/events
    try:
        r = requests.get(f"{API}/api/littering/events", headers=headers, timeout=5)
        if r.ok:
            data = r.json()
            ok(f"GET /api/littering/events → total={data['total']}, items={len(data['items'])}")
        else:
            fail(f"GET /api/littering/events → HTTP {r.status_code}: {r.text[:100]}")
    except requests.exceptions.ConnectionError:
        fail("Serverul nu răspunde la /api/littering/events")
        return

    # GET /api/littering/events cu filtru
    r = requests.get(f"{API}/api/littering/events?status=pending&limit=5",
                     headers=headers, timeout=5)
    if r.ok:
        data = r.json()
        ok(f"GET /api/littering/events?status=pending → total={data['total']}")
    else:
        fail(f"Filter endpoint → HTTP {r.status_code}")

    # Test 404
    r = requests.get(f"{API}/api/littering/events/99999", headers=headers, timeout=5)
    if r.status_code == 404:
        ok("GET /api/littering/events/99999 → 404 Not Found (corect)")
    else:
        fail(f"Expected 404, got {r.status_code}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{BOLD}TrashDet — Test Batch + Monitor{RESET}")
    print(f"API: {API}")
    print(f"Images: {IMAGES_DIR}")

    # Încearcă conectarea la server
    server_up = False
    try:
        r = requests.get(f"{API}/api/stats", timeout=3)
        server_up = True
        ok(f"Server pornit la {API}")
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError):
        info("Server OFFLINE — testele de API vor fi skipped")
        info("  Porneste cu: .venv/Scripts/uvicorn backend.main:app --reload --port 8000")

    token = None
    if server_up:
        token = get_token()
        if token:
            ok(f"Login admin reușit (JWT obținut)")
        else:
            info("Login eșuat — unele teste vor fi skipped")
            info("  Rulează prepare_local_db.py pentru conturile admin/operator locale")

    # ── Rulează testele ──
    test_monitor_state_machine()  # Nu necesită server

    test_monitor_with_real_images()  # Nu necesită server, dar încarcă modelele

    if server_up:
        test_batch(token)
        test_littering_endpoints(token)
    else:
        header("TEST 1 + 4 — SKIP (server offline)")
        info("Porneste serverul pentru testele de API/Batch")

    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}Teste complete.{RESET}\n")
