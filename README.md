# TrashDetectionSystem

**Sistem inteligent pentru detectarea aruncării ilegale a deșeurilor în spații publice**, prin analiză video și viziune artificială.

Lucrare de licență — accent pe componenta de Machine Learning: detectorul de deșeuri, clasificatorul de material și **algoritmul temporal** care decide dacă un obiect apărut în scenă reprezintă un incident real. O aplicație web susține validarea practică prin monitorizare live, analiză video, administrare și păstrarea locală a dovezilor.

---

## ✨ Ideea cheie

Un deșeu **detectat nu este automat un incident**. Sistemul nu reacționează la un singur cadru, ci la un **comportament în timp**:

1. **Detecție** — modelul localizează obiectele de tip deșeu în cadre;
2. **Asociere** — analizează relația persoană ↔ obiect;
3. **Context temporal** — verifică dacă obiectul rămâne în zonă după ce persoana pleacă;
4. **Dovadă** — salvează momentul relevant (thumbnail, clip, metadate);
5. **Validare** — un operator confirmă, respinge sau arhivează.

Astfel se evită alarmele false (cineva care doar ține un obiect în mână) și se păstrează decizia umană în buclă.

---

## 📊 Performanță

| Model | Arhitectură | Rezultate |
|---|---|---|
| **Detector deșeuri** | YOLOv8s | mAP50 **91%** · precizie **92%** · recall **82%** |
| **Clasificator material** | YOLOv8n-cls (5 clase) | **92%** TrashNet · **97%** obiecte reale · **≈96%** la distanță simulată |
| **Detector persoane** | YOLOv8n (COCO) | preantrenat |

Clasificatorul a fost antrenat **mixt** (TrashNet + crop-uri reale de cameră + augmentare de distanță) pentru a fi bun atât pe obiecte generale, cât și pe domeniul real al aplicației (*domain adaptation*), fără supra-specializare.

---

## 🧠 Pipeline ML

```text
Cadru video
  → Detector YOLOv8s          localizează deșeurile
  → ByteTrack                 urmărește obiectul între cadre
  → Clasificator YOLOv8n-cls  estimează materialul (la incident)
  → Algoritm temporal         decide dacă există o aruncare → incident + dovadă
```

Greutățile active (configurabile prin `.env`):

```text
models/
├── detector/production/best.pt   # detector deșeuri (final)
├── classify/B2/best.pt           # clasificator material (mixt, final)
└── pretrained/yolov8n.pt         # detector persoane (COCO)
```

`runs/` păstrează doar antrenările finale (proveniență); modelul activ este cel promovat în `models/`.

---

## 🗂️ Structura proiectului

```text
TrashDetectionSystem/
├── backend/
│   ├── main.py                 # API FastAPI și rute HTTP
│   ├── video.py                # WebSocket monitor live + procesare upload (cu dedup casete)
│   ├── inference.py            # încărcare modele, inferență, tracking, blur GDPR
│   ├── littering_detector.py   # algoritmul temporal de decizie a incidentului
│   ├── database.py             # modele SQLAlchemy + helpers DB
│   ├── schemas.py              # scheme Pydantic
│   ├── config.py               # configurare centralizată din .env
│   └── ml/two_stage.py         # pipeline detector + clasificator
├── frontend/
│   ├── templates/              # pagini Alpine/Jinja
│   └── static/{js,css}/        # logică UI, monitorizare, admin
├── models/                     # greutăți finale (ignorat de git)
│   ├── detector/production/    # detector deșeuri
│   ├── classify/B2/            # clasificator material
│   └── pretrained/             # baze YOLO (persoane, clasificator)
├── datasets/                   # date locale (ignorat de git)
├── results/                    # metrici + figuri folosite în lucrare
├── runs/                       # antrenări finale (proveniență, ignorat de git)
├── data/                       # SQLite local + dovezi generate (ignorat de git)
├── scripts/
│   ├── data/                   # pregătire/validare/export dataseturi
│   ├── training/               # antrenare detector + clasificator
│   ├── evaluation/             # evaluări standalone
│   ├── maintenance/            # reset DB, conturi locale
│   └── smoke/                  # test E2E pe server pornit
├── tests/                      # 31 teste pytest (API, securitate, DB)
├── certs/                      # certificat HTTPS local (ignorat de git)
├── start_https.py              # pornire locală HTTPS (camera telefonului)
└── requirements.txt
```

Dataseturile păstrate pentru reproducere/evaluare:

- `datasets/parks_detect_final/` — set final pentru **detector** (6213/775/780 train/val/test);
- `datasets/trashnet_cls/` — set de bază pentru **clasificator**;
- `datasets/real_material/` — crop-uri reale de cameră (domain adaptation);
- `datasets/material_mixed/` — set mixt final pentru clasificator.

---

## 🚀 Pornire rapidă

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe start_https.py
```

Pentru camera telefonului: rulează serverul HTTPS și deschide adresa afișată în consolă, **de pe același Wi-Fi**. Browserul poate avertiza pentru certificatul self-signed local → `Advanced` → `Proceed`.

Comenzi server:

```powershell
.\.venv\Scripts\python.exe start_https.py            # pornire pe HTTPS 8443
.\.venv\Scripts\python.exe start_https.py --restart  # dacă portul e ocupat de o rulare veche
.\.venv\Scripts\python.exe start_https.py --no-reload # stabil pentru prezentare
.\.venv\Scripts\python.exe start_https.py --stop     # oprire
.\.venv\Scripts\python.exe start_https.py --port 9444 # port alternativ
```

Conturi locale:

```text
admin    / Admin1234!     rol: admin
operator / Operator1234!  rol: utilizator
```

---

## ✅ Testare

```powershell
# Suita automată (API, securitate, izolare pe organizație, DB)
.\.venv\Scripts\python.exe -m pytest tests/ -q          # 31 teste

# Test E2E pe server pornit (necesită un clip prin SMOKE_CLIP)
$env:SMOKE_CLIP="D:/cale/catre/clip_aruncare.mp4"
.\.venv\Scripts\python.exe scripts\smoke\full_e2e_test.py

# Evaluare detector pe setul de test
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_video_events.py --help
```

---

## ⚙️ Configurare (`.env`)

Setările care diferă între local și server se află în `.env` (citite prin `backend/config.py`). `.env.example` conține valorile necesare: `SECRET_KEY`, `APP_BASE_URL`, `DATABASE_URL`, `STORAGE_ROOT`, căile modelelor, limite de upload, praguri de inferență și retenția dovezilor.

- Local rulează cu **SQLite** implicit (`data/trash_detection.db`).
- Pentru deploy pe server: `DATABASE_URL` poate indica PostgreSQL, iar `APP_BASE_URL` domeniul serverului.
- Dovezile video mai vechi de `LITTERING_FILE_RETENTION_DAYS` se curăță automat; metadata incidentelor rămâne în DB.

---

## 🛠️ Tehnologii

**Backend:** FastAPI · SQLAlchemy (async) · SQLite · JWT · WebSocket
**ML:** Ultralytics YOLOv8 · ByteTrack · OpenCV · PyTorch (CUDA)
**Frontend:** Alpine.js · Jinja2 · TailwindCSS

---

*Lucrare de licență — Universitatea „1 Decembrie 1918" din Alba Iulia, Facultatea de Informatică și Inginerie. Autor: Serbicean Alexandru. Coordonator: Lect. univ. drd. Incze Arpad.*
