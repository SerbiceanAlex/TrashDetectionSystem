# Scripts

Utility scripts are grouped by purpose:

- `data/` - dataset preparation, frame extraction, YOLO filtering and validation.
- `training/` - model training and promotion helpers.
- `evaluation/` - metric and video-event evaluation scripts used for the thesis.
- `presentation/` - visual run scripts for presentation or professor review.
- `maintenance/` - local database/admin/reset helpers.
- `smoke/` - quick manual checks against a running local server.

For a normal local app run, you usually need only:

```powershell
.\.venv\Scripts\python.exe start_https.py
.\.venv\Scripts\python.exe -m scripts.maintenance.reset_data
```
