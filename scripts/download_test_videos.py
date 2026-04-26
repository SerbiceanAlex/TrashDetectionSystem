"""
download_test_videos.py
========================
Cauta si descarca clipuri potrivite pentru testarea sistemului de
detectie a actului de aruncare ilegala.

Foloseste cautare YouTube prin yt-dlp (primele rezultate disponibile).

Rulare:
    .venv/Scripts/python.exe scripts/download_test_videos.py

Output: datasets/test_videos/  (gitignored)
"""

import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("yt-dlp nu e instalat. Ruleaza: pip install yt-dlp")
    sys.exit(1)

OUT_DIR = Path(__file__).resolve().parents[1] / "datasets" / "test_videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Termeni de cautare — primul rezultat disponibil va fi descarcat
SEARCH_QUERIES = [
    ("ytsearch3:littering caught on camera CCTV 2023 short",  "littering_cctv"),
    ("ytsearch3:person throws trash illegal dumping surveillance", "illegal_dumping"),
    ("ytsearch2:aruncarea gunoiului camera surprins", "gunoi_ro"),
]

def download_search(query: str, out_name: str, max_duration: int = 120) -> bool:
    """Descarca primul video disponibil din rezultatele cautarii."""
    ydl_opts = {
        "format": "best[height<=480][ext=mp4]/best[height<=480]/best",
        "outtmpl": str(OUT_DIR / f"{out_name}.%(ext)s"),
        "quiet": False,
        "no_warnings": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration < {max_duration}"),
        "playlistend": 3,        # incearca primele 3 rezultate
        "noplaylist": False,
        "merge_output_format": "mp4",
        "js_runtimes": "nodejs",  # foloseste Node.js daca e disponibil
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([query])
        # Check if file was actually saved
        saved = list(OUT_DIR.glob(f"{out_name}.*"))
        return len(saved) > 0
    except Exception as e:
        print(f"  Eroare: {e}")
        return False


print(f"Output: {OUT_DIR}")
print(f"Cautam {len(SEARCH_QUERIES)} seturi de clipuri...\n")

success = 0
for query, name in SEARCH_QUERIES:
    short_q = query.replace("ytsearch3:", "").replace("ytsearch2:", "")
    print(f"Cautare: '{short_q}'")
    ok = download_search(query, name)
    if ok:
        success += 1
        files = list(OUT_DIR.glob(f"{name}.*"))
        print(f"  Salvat: {[f.name for f in files]}\n")
    else:
        print(f"  Niciun rezultat disponibil\n")

# Afiseaza toate fisierele descarcate
all_videos = list(OUT_DIR.glob("*.mp4")) + list(OUT_DIR.glob("*.webm"))
print(f"\n{'='*50}")
print(f"Total videoclipuri disponibile: {len(all_videos)}")
for v in sorted(all_videos):
    size_mb = v.stat().st_size / 1024 / 1024
    print(f"  {v.name}  ({size_mb:.1f} MB)")

print(f"\nCum testezi:")
print("  1. Deschide http://127.0.0.1:8000")
print("  2. Login -> tab 'Scanare' -> sub-tab 'Video'")
print("  3. Upload unul dintre clipurile de mai sus")
print("  4. Asteapta bara de progres (poate dura 1-3 min)")
print("  5. Admin -> Incidente -> vezi evenimentele detectate automat")
