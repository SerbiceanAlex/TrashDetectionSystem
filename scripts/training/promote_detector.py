"""
Promovează un checkpoint validat de detector în slotul de producție.

Backend-ul citește mereu:
    models/detector/production/best.pt

Folosește scriptul doar după ce un model candidat trece validarea.

Exemplu:
    .venv\\Scripts\\python.exe scripts\\training\\promote_detector.py ^
        --candidate runs\\detect\\parks-trash-final\\weights\\best.pt ^
        --name parks-trash-final
"""

from __future__ import annotations

import sys
# Diacriticele românești se afișează corect indiferent de codarea consolei.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PRODUCTION = REPO / "models" / "detector" / "production" / "best.pt"
MANIFEST = REPO / "models" / "detector" / "production" / "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promovează un checkpoint de detector în producție")
    parser.add_argument("--candidate", required=True, help="Calea către best.pt candidat")
    parser.add_argument("--name", required=True, help="Nume lizibil al modelului, ex. parks-trash-final")
    parser.add_argument("--note", default="", help="Notă opțională de validare")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    """Absolutizează o cale relativă față de rădăcina proiectului."""
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    return path


def sha256(path: Path) -> str:
    """Calculează hash-ul SHA-256 al unui fișier (citit în bucăți de 1 MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    candidate = resolve_path(args.candidate)
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint-ul candidat nu există: {candidate}")

    PRODUCTION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, PRODUCTION)

    manifest = {
        "active_model": args.name,
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(candidate.relative_to(REPO) if candidate.is_relative_to(REPO) else candidate),
        "production": str(PRODUCTION.relative_to(REPO)),
        "sha256": sha256(PRODUCTION),
        "note": args.note,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Detector promovat:")
    print(f"  nume       : {args.name}")
    print(f"  sursă      : {candidate}")
    print(f"  producție  : {PRODUCTION}")
    print(f"  sha256     : {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
