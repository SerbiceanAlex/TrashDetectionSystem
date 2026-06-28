"""
Teste pentru mașina de stări de abandonare (backend/littering_detector.py).

Validează logica TEMPORALĂ — partea cea mai importantă a proiectului — cu
traiectorii sintetice (persoană + obiect pe cadre), fără inferență ML:
  • un obiect ȚINUT în mână NU declanșează incident;
  • aruncarea + îndepărtarea persoanei DECLANȘEAZĂ incident;
  • un obiect DUS cu persoana (cărat afară) NU declanșează.

Distanțele se scalează din înălțimea casetei persoanei (~1,70 m), deci controlăm
distanța prin poziția în pixeli. Cadre mici (240x320) pentru memorie/viteză.
"""

import numpy as np
import pytest

from backend.littering_detector import LitteringDetector

FPS = 25.0
H, W = 240, 320


def _frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


def _person(cx, h=160, w=54):
    """Casetă persoană centrată pe cx; înălțime fixă → scară constantă."""
    return (int(cx - w / 2), 40, int(cx + w / 2), 40 + h)


def _trash(tid, cx, cy=180, s=30, material="plastic"):
    return {
        "track_id": tid,
        "box": (int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)),
        "material_name": material,
        "det_score": 0.9,
    }


def test_obiect_tinut_in_mana_nu_declanseaza():
    """Un obiect NOU, ținut lângă persoană (suprapus) și static, NU e incident."""
    d = LitteringDetector(fps=FPS)
    # persoana intră prima (fără obiect) → obiectul nu devine „preexistent"
    for _ in range(3):
        d.update(_frame(), [], [_person(240)], [10])
    incident = None
    # obiect NOU, suprapus peste persoană (ținut în mână), static, ~3 secunde
    for _ in range(75):
        ev = d.update(_frame(), [_trash(1, 240, 120)], [_person(240)], [10])
        incident = incident or ev
    assert incident is None, "Ținerea unui obiect în mână NU trebuie să declanșeze incident"


def test_aruncare_si_plecare_declanseaza():
    """Persoana lasă un obiect NOU și se îndepărtează clar → incident (mod distanță)."""
    d = LitteringDetector(fps=FPS)
    for _ in range(3):
        d.update(_frame(), [], [_person(240)], [10])
    # obiectul apare lângă persoană (nou), persoana încă aproape câteva cadre
    for _ in range(3):
        d.update(_frame(), [_trash(1, 285, 180)], [_person(240)], [10])
    # persoana se îndepărtează spre stânga; obiectul rămâne pe loc (static)
    incident = None
    for step in range(1, 40):
        px = max(25, 240 - step * 28)
        ev = d.update(_frame(), [_trash(1, 285, 180)], [_person(px)], [10])
        incident = incident or ev
        if incident:
            break
    assert incident is not None, "Aruncarea + plecarea TREBUIE să declanșeze incident"
    assert incident.detection_method == "distance"
    assert incident.material == "plastic"


def test_obiect_carat_cu_persoana_nu_declanseaza():
    """Obiectul se mișcă ÎMPREUNĂ cu persoana (cărat afară din cadru) → fără incident."""
    d = LitteringDetector(fps=FPS)
    for _ in range(3):
        d.update(_frame(), [], [_person(240)], [10])
    incident = None
    for step in range(1, 40):
        px = max(25, 240 - step * 28)
        # obiectul rămâne suprapus peste persoană și se mișcă odată cu ea
        ev = d.update(_frame(), [_trash(1, px, 120)], [_person(px)], [10])
        incident = incident or ev
    assert incident is None, "Un obiect cărat cu persoana NU trebuie să declanșeze incident"


def test_scena_fara_persoana_nu_declanseaza():
    """Obiecte vizibile fără nicio persoană (preexistente) → niciodată incident."""
    d = LitteringDetector(fps=FPS)
    incident = None
    for _ in range(50):
        ev = d.update(_frame(), [_trash(1, 160, 120)], [], [])
        incident = incident or ev
    assert incident is None, "Obiectele preexistente, fără persoană, NU declanșează"
