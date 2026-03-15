# Trash Detection System (Step 1)

## Setup
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Run (video preview)
```bash
python -m src.video_preview --source 0
# or:
python -m src.video_preview --source path\to\video.mp4
```

Notes:
- `--source 0` = webcam default.
- Press `q` to quit.