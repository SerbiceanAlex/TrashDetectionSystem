# TrashDetectionSystem

Sistem inteligent pentru detectarea actului de aruncare ilegala a deseurilor in spatii publice prin analiza video in timp real.

Lucrarea este orientata pe partea de ML: detectorul de deseuri, clasificatorul de material si algoritmul temporal care decide daca un obiect aparut in scena reprezinta un incident de littering. Aplicatia web ramane demonstratia practica a pipeline-ului.

## Directia proiectului

Proiectul a pornit de la ideea simpla de detectie a deseurilor in imagini si a evoluat catre o problema mai puternica pentru licenta: identificarea momentului in care o persoana arunca ilegal un obiect. Sistemul coreleaza temporal trei elemente:

1. prezenta unei persoane in cadru;
2. disparitia sau indepartarea persoanei din zona;
3. aparitia/stabilizarea unui obiect de tip deseu in zona monitorizata.

Rezultatul este salvat ca dovada: metadata, thumbnail, clip scurt si scoruri de incredere.

## Modele ML

Pipeline-ul folosit de aplicatie este two-stage:

```text
Frame video
  -> Detector YOLOv8 final: localizeaza obiectele trash
  -> Clasificator YOLOv8 B2: estimeaza materialul
  -> Behavioral Engine: decide daca exista act de aruncare
```

Greutatile finale folosite de aplicatie stau in `models/`:

```text
models/
├── detector/production/best.pt
└── classify/B2/best.pt
```

`runs/` ramane zona de experimente si antrenari YOLO. Modelul activ este checkpoint-ul promovat in `models/detector/production/best.pt`.

## Structura proiectului

```text
TrashDetectionSystem/
├── backend/
│   ├── main.py                 # API FastAPI si rute HTTP
│   ├── video.py                # WebSocket live monitor si procesare video upload
│   ├── inference.py            # incarcare modele, inferenta, tracking, blur GDPR
│   ├── littering_detector.py   # algoritmul temporal de detectie a actului de aruncare
│   ├── database.py             # modele SQLAlchemy si helpers DB
│   ├── schemas.py              # scheme Pydantic pentru API
│   └── ml/
│       └── two_stage.py        # pipeline YOLO detector + clasificator
├── frontend/
│   ├── templates/              # pagini Alpine/Jinja
│   └── static/
│       ├── js/                 # logica UI, auth, monitor, admin
│       └── css/                # stiluri
├── models/
│   ├── detector/production/    # model detector final folosit in aplicatie
│   └── classify/B2/            # model clasificator folosit in aplicatie
├── notebooks/
│   ├── training/               # experimente si antrenari
│   └── evaluation/             # evaluari, comparatii, figuri pentru licenta
├── scripts/
│   ├── data/                   # pregatire dataseturi, split, validare, export cropuri
│   ├── training/               # scripturi de antrenare
│   ├── evaluation/             # evaluari standalone
│   ├── demos/                  # demo vizual pentru profesor/prezentare
│   ├── maintenance/            # reset DB, creare admin/demo user
│   └── smoke/                  # teste manuale rapide pe server pornit
├── tests/                      # teste automate pytest
├── results/                    # rezultate si metrici folosite in lucrare
├── start_https.py              # pornire locala HTTPS pentru camera telefonului
├── requirements.txt
└── PROJECT_SUCCESS_PATH.md     # status proiect si detectorul final in productie
```

Directoarele `datasets/`, `runs/`, `outputs/`, `backend/uploads/`, `backend/videos/` si fisierele `.pt` sunt artefacte locale si nu sunt versionate.

## Pornire rapida

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe start_https.py
```

Pentru camera telefonului, ruleaza serverul HTTPS si deschide adresa afisata in consola de pe acelasi Wi-Fi.
Browserul va afisa un avertisment de certificat local/self-signed; pentru demo local se continua cu
`Advanced` / `Help me understand` -> `Proceed`.

Comenzi utile pentru serverul local:

```powershell
# Pornire normala pe portul local HTTPS 8443
.\.venv\Scripts\python.exe start_https.py

# Serverul ruleaza in terminalul curent; Ctrl+C il opreste si logurile raman vizibile.

# Daca portul este deja ocupat de o rulare veche
.\.venv\Scripts\python.exe start_https.py --restart

# Daca vrei doar sa deschizi serverul deja pornit, fara sa-l atasezi la terminal
.\.venv\Scripts\python.exe start_https.py --reuse

# Oprire server local
.\.venv\Scripts\python.exe start_https.py --stop

# Port alternativ, doar daca 8443 este ocupat de alt program
.\.venv\Scripts\python.exe start_https.py --port 9444

# Fara deschidere automata in browser
.\.venv\Scripts\python.exe start_https.py --no-open

# Fara reload automat, util pentru un demo mai tacut
.\.venv\Scripts\python.exe start_https.py --no-reload
```

Pentru a reduce cererile repetate de Windows Firewall, permite Python/Uvicorn pe retele private
sau ruleaza o singura data in PowerShell pornit ca Administrator:

```powershell
New-NetFirewallRule -DisplayName "TrashDet HTTPS Local" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443,9444 -Profile Private
```

## Configurare locală și server-ready

Setările care diferă între rularea locală și un server se află în `.env` și sunt citite prin `backend/config.py`.
Fișierul `.env.example` conține valorile necesare pentru:

```text
SECRET_KEY
APP_BASE_URL
DATABASE_URL
DETECTOR_WEIGHTS
CLASSIFIER_WEIGHTS
PERSON_DETECTOR_WEIGHTS
MAX_UPLOAD_MB
LIVE_IMGSZ
DEFAULT_DET_CONF
MONITOR_MIN_DET_CONF
LITTERING_FILE_RETENTION_DAYS
```

Pentru licență, aplicația rulează local cu SQLite implicit (`backend/trash_detection.db`). Pentru un deploy ulterior pe server, `DATABASE_URL` poate indica o bază PostgreSQL, iar `APP_BASE_URL` trebuie setat la domeniul/URL-ul serverului. Modelele rămân configurabile prin căi relative la rădăcina proiectului.

Retenția probelor video este activă implicit: fișierele din `backend/littering/` mai vechi de `LITTERING_FILE_RETENTION_DAYS` sunt curățate automat, iar metadata incidentelor rămâne în baza de date.

## Comenzi utile

```powershell
# Demo vizual pe clip
.\.venv\Scripts\python.exe scripts\demos\demo_littering.py --video datasets\test_videos\clip.mp4

# Demo live camera
.\.venv\Scripts\python.exe scripts\demos\demo_littering.py --camera 0

# Evaluare video pe clipuri selectate manual
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_video_events.py --manifest scripts\evaluation\video_manifest_template.csv --frame-skip 1

# Pregatire baza de date demo: admin/operator, locatie, autoritate, OTP cleanup
.\.venv\Scripts\python.exe scripts\maintenance\prepare_demo_db.py --apply --prune-locations --reset-demo-passwords

# Reset date generate local
.\.venv\Scripts\python.exe -m scripts.maintenance.reset_data
```

Conturi demo locale:

```text
admin    / Admin1234!    rol: admin
operator / Operator1234! rol: user/operator
```

Adminul vede panoul de administrare, utilizatorii, locațiile, incidentele, autoritățile și storage-ul. Operatorul folosește monitorizarea și fluxul de incidente fără acces la administrarea organizației.

## Focus pentru lucrare

In redactare, aplicatia web trebuie tratata ca suport practic. Partea principala merita sa fie:

1. pregatirea dataseturilor si antrenarea modelelor;
2. evaluarea detectorului final si a clasificatorului B2;
3. logica Behavioral Engine si criteriile temporale;
4. experimentele pe clipuri reale si analiza erorilor;
5. limite, GDPR si directii viitoare.
