"""
Promote a validated detector checkpoint to the production model slot.

The backend always reads:
    models/detector/production/best.pt

Use this script only after a candidate model passes validation.

Example:
    .venv\\Scripts\\python.exe scripts\\training\\promote_detector.py ^
        --candidate runs\\detect\\parks-trash-A6-curated\\weights\\best.pt ^
        --name A6-curated
"""

from __future__ import annotations

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
    parser = argparse.ArgumentParser(description="Promote detector checkpoint to production")
    parser.add_argument("--candidate", required=True, help="Path to candidate best.pt")
    parser.add_argument("--name", required=True, help="Human-readable model name, e.g. A6-curated")
    parser.add_argument("--note", default="", help="Optional validation note")
    return parser.parse_args()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    candidate = resolve_path(args.candidate)
    if not candidate.exists():
        raise FileNotFoundError(f"Candidate checkpoint not found: {candidate}")

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

    print("Promoted detector:")
    print(f"  name       : {args.name}")
    print(f"  source     : {candidate}")
    print(f"  production : {PRODUCTION}")
    print(f"  sha256     : {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
