# Local Runtime Data

This folder is for local files generated while the application runs.

- `trash_detection.db` - local SQLite database, ignored by git.
- `backups/` - local database backups, ignored by git.
- `runtime/uploads/` - original uploaded images.
- `runtime/annotated/` - annotated detection images.
- `runtime/videos/` - uploaded and annotated videos.
- `runtime/littering/` - incident clips and thumbnails.

Keep source code in `backend/`, UI files in `frontend/`, training assets in
`datasets/` and `models/`, and generated app data here.
