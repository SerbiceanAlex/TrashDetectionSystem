# Trash Detection System — Sistem Two-Stage de Detecție a Deșeurilor

Sistem two-stage de detecție și clasificare a deșeurilor în spații verzi urbane (parcuri), implementat cu YOLOv8.

**Lucrare de licență — Universitatea Politehnica București, 2026.**

---

## Arhitectură

```
Imagine/Video → [Stage 1: Detector YOLO] → bounding boxes (clasa: trash)
                                                    ↓
                                    [Stage 2: Clasificator YOLO]
                                                    ↓
                            material: glass / metal / paper / plastic / other
```

- **Stage 1** (`src/detect_two_stage.py`): YOLOv8s, imgsz=640, clasă unică `trash`, antrenat pe dataset Parks adnotat manual din videoclipuri de parc
- **Stage 2** (`src/detect_two_stage.py`): YOLOv8n-cls, imgsz=224, antrenat pe TrashNet + crops extrase din dataset-ul de parcuri

---

## Rezultate Finale

### Detector — Experiment A

| Experiment | Model | imgsz | Precision | Recall | **mAP50** | mAP50-95 |
|------------|-------|-------|-----------|--------|-----------|----------|
| A22 (baseline) | YOLOv8n | 416 | 0.707 | 0.286 | 0.393 | 0.281 |
| **A3-final** ✅ productie | **YOLOv8s** | **640** | **0.623** | **0.406** | **0.443** | **0.321** |

### Clasificator — Experiment B (metrici pe test set)

| Experiment | Model | Dataset | **Acc Top-1** | **F1 macro** | Imagini test |
|------------|-------|---------|---------------|--------------|--------------|
| **B2** ✅ productie | YOLOv8n-cls | TrashNet + parks crops | **91.1%** | 0.881 | 257 |
| B3 | YOLOv8n-cls | TrashNet + parks crops (extins) | 91.3% | 0.879 | 299 |

### Pipeline End-to-End — Experiment C

| Experiment | Detector | Clasificator | Imagini cu detecții | **Rată detecție** | Total detecții | Viteză |
|------------|----------|--------------|---------------------|-------------------|----------------|--------|
| C1 (baseline) | A22 | B2 | 9 / 225 | 4.0% | 10 | 21.4 ms/img |
| **C2** ✅ best | **A3-final** | **B2** | **219 / 225** | **97.3%** | **739** | 60.7 ms/img |

**Distribuție materiale detectate (C2, 739 detecții):**

| Material | Nr. detecții | Procent |
|----------|-------------|---------|
| paper    | 276 | 37.3% |
| metal    | 267 | 36.1% |
| glass    | 94  | 12.7% |
| plastic  | 77  | 10.4% |
| other    | 25  | 3.4%  |

---

## Modele antrenate

| Model | Cale | Rol |
|-------|------|-----|
| **Detector A3-final** | `runs/detect/parks-trash-A3-final/weights/best.pt` | Stage 1 — detectează obiecte trash |
| **Clasificator B2** | `runs/classify/parks-cls-B2/weights/best.pt` | Stage 2 — clasifică materialul |

---

## Notebook-uri

Întreaga pipeline este documentată și reproductibilă prin notebook-uri Jupyter:

| Notebook | Scop |
|----------|------|
| `notebooks/data/01_data_preparation.ipynb` | Pregătire dataset detecție (split train/val/test) |
| `notebooks/data/02_classification_data.ipynb` | Pregătire dataset clasificare (crops + split) |
| `notebooks/data/03_annotate_parks_crops.ipynb` | Export crops, antrenare B, evaluare clasificator |
| `notebooks/training/01_train_detector.ipynb` | Antrenare detectori — Experiment A |
| `notebooks/training/02_train_classifier.ipynb` | Antrenare clasificatori — Experiment B |
| `notebooks/evaluation/01_evaluate_detector.ipynb` | Evaluare detector A22 și A3-final pe test set |
| `notebooks/evaluation/02_evaluate_classifier.ipynb` | Evaluare clasificator B2 și B3 pe test set |
| `notebooks/evaluation/03_inference_demo.ipynb` | Demo vizual two-stage pe test set |
| `notebooks/evaluation/04_pipeline_C1_C2.ipynb` | Pipeline end-to-end C1 vs C2 |
| `notebooks/evaluation/05_thesis_figures.ipynb` | Generare figuri și tabele pentru teză |

---

## Structura proiectului

```
TrashDetectionSystem/
├── src/
│   └── detect_two_stage.py      # Pipeline two-stage (CLI + modul importabil de backend/)
├── backend/
│   ├── main.py                  # FastAPI router, endpoint-uri REST + WebSocket
│   ├── auth.py                  # JWT (PyJWT), bcrypt, OTP, rate limiting, password policy
│   ├── auth_router.py           # Endpoint-uri /auth/register, /auth/login, /auth/verify-otp, /auth/me
│   ├── inference.py             # Thread-safe two-stage pipeline (singleton)
│   ├── video.py                 # WebSocket handler pentru inferență live pe video
│   ├── database.py              # SQLAlchemy async engine + modele ORM
│   ├── schemas.py               # Pydantic schemas
│   └── geo.py                   # Geocodare coordonate GPS (Nominatim)
├── frontend/
│   ├── static/                  # CSS, JS, manifest PWA, service worker
│   └── templates/               # HTML (Jinja2) — base + partials + tabs
├── scripts/
│   ├── train_classifier.py      # Antrenare clasificator (apelat din notebook)
│   ├── evaluate_classifier.py   # Evaluare clasificator (apelat din notebook)
│   ├── export_yolo_crops.py     # Export crops din detecții
│   ├── split_classification_dataset.py   # Split all→train/val/test (clasificare)
│   ├── merge_classification_datasets.py  # Merge TrashNet + parks crops
│   ├── validate_yolo_dataset.py          # Validare format dataset YOLO
│   └── report_classification_dataset_stats.py  # Statistici dataset clasificare
├── notebooks/
│   ├── data/                    # Pregătire date
│   ├── training/                # Antrenare modele
│   └── evaluation/              # Evaluare, demo, figuri teză
├── datasets/
│   ├── parks_detect_full/       # Dataset detecție (train/val/test, adnotat manual)
│   ├── parks_cls/               # Dataset clasificare (train/val/test)
│   ├── mixed_cls/               # Dataset clasificare extins (TrashNet + parks)
│   └── trashnet_cls/            # TrashNet original
├── results/
│   ├── detector/                # JSON metrici A22, A3-final
│   ├── classifier/              # JSON metrici B2, B3
│   └── pipeline/                # JSON/CSV sumare C1, C2
├── requirements.txt
└── README.md
```

---

## Aplicație Web

Interfață web fullstack cu autentificare, GPS, video live și statistici interactive.

**Stack:** FastAPI · SQLAlchemy 2.0 async · SQLite (aiosqlite) · Alpine.js 3 · Chart.js 4 · Leaflet 1.9.4

```bash
# Pornire server (din directorul rădăcină al proiectului)
.venv\Scripts\uvicorn backend.main:app --reload --port 8000
```

Deschide `http://127.0.0.1:8000` în browser.

**Funcționalități:**
- **Autentificare** — înregistrare/login cu JWT, parole hashed bcrypt, roluri user/admin
- **Detectare imagine** — upload drag & drop (limită 20 MB), slider confidence, imagine adnotată instant, coordonate GPS opționale
- **Detectare batch** — upload multiple imagini simultan, raport agregat per sesiune
- **Video live** — inferență two-stage în timp real prin WebSocket (stream MJPEG)
- **Hartă GPS** — toate detecțiile pe hartă Leaflet cu filtre pe material și status (rezolvat/nerezolvat)
- **Statistici** — pie chart materiale, bar chart detecții pe zi, carduri sumar global, statistici personale, grafic săptămânal
- **Leaderboard** — clasament utilizatori după număr de detecții raportate
- **Notificări** — sistem in-app de notificări per utilizator
- **Export CSV** — descarcă toate detecțiile filtrabile (compatibil Excel)
- **PWA** — Progressive Web App, instalabil pe mobil/desktop (manifest + service worker)
- **API docs** — Swagger UI la `/docs`, ReDoc la `/redoc`

> Variabila de mediu `TRASHDET_SECRET_KEY` poate înlocui cheia JWT implicită în producție.

---

## Setup

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Utilizare

### Inferență pe video/webcam (CLI)

```bash
# Pipeline two-stage complet (detector + clasificator material)
python -m src.detect_two_stage --source path/to/video.mp4 \
    --detector runs/detect/parks-trash-A3-final/weights/best.pt \
    --classifier runs/classify/parks-cls-B2/weights/best.pt \
    --show --save
```

### Antrenare modele

Antrenarea se face din notebook-uri Jupyter (reproductibil, cu logging vizual):

```
notebooks/training/01_train_detector.ipynb   → Experiment A (detector)
notebooks/training/02_train_classifier.ipynb  → Experiment B (clasificator)
```

### Evaluare

Evaluarea completă se rulează din notebook-uri:

```
notebooks/evaluation/01_evaluate_detector.ipynb    → metrici A22 + A3-final
notebooks/evaluation/02_evaluate_classifier.ipynb  → metrici B2 + B3
notebooks/evaluation/04_pipeline_C1_C2.ipynb       → pipeline C1 vs C2
notebooks/evaluation/05_thesis_figures.ipynb       → figuri și tabele finale pentru teză
```

### Pregătire dataset

```bash
# Split dataset clasificare
python scripts/split_classification_dataset.py \
    --source-root datasets/parks_cls_unsorted \
    --out-root datasets/parks_cls --clear

# Validare format YOLO
python scripts/validate_yolo_dataset.py

# Statistici dataset
python scripts/report_classification_dataset_stats.py
```