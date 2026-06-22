# TrashDetectionSystem

Sistem inteligent pentru detectarea actului de aruncare ilegală a deșeurilor în spații publice, folosind analiză video și modele de viziune artificială.

Lucrarea este orientată pe componenta de Machine Learning: detectorul de deșeuri, clasificatorul de material și algoritmul temporal care decide dacă un obiect apărut în scenă reprezintă un posibil incident. Aplicația web susține validarea practică a pipeline-ului prin monitorizare, administrare și păstrarea locală a dovezilor.

## Direcția proiectului

Proiectul a pornit de la detecția deșeurilor în imagini și a evoluat către o problemă mai potrivită pentru lucrarea de licență: identificarea momentului în care o persoană aruncă sau abandonează ilegal un obiect. Sistemul corelează temporal trei elemente:

1. prezența unei persoane în cadru;
2. dispariția sau îndepărtarea persoanei din zonă;
3. apariția și stabilizarea unui obiect de tip deșeu în zona monitorizată.

Rezultatul este salvat ca dovadă locală: metadata, thumbnail, clip scurt, status și scoruri de încredere.

## Modele ML

Pipeline-ul folosit de aplicație este în două etape:

```text
Frame video
  -> Detector YOLOv8 final: localizează obiectele de tip deșeu
  -> Clasificator YOLOv8 B2: estimează materialul
  -> Behavioral Engine: decide dacă există un posibil act de aruncare
```

Greutățile finale folosite de aplicație sunt în `models/`:

```text
models/
├── detector/production/best.pt
└── classify/B2/best.pt
```

`runs/` rămâne zona de experimente și antrenări YOLO. Modelul activ este checkpoint-ul promovat în `models/detector/production/best.pt`.

## Structura proiectului

```text
TrashDetectionSystem/
├── backend/
│   ├── main.py                 # API FastAPI și rute HTTP
│   ├── video.py                # WebSocket live monitor și procesare video upload
│   ├── inference.py            # încărcare modele, inferență, tracking, blur GDPR
│   ├── littering_detector.py   # algoritmul temporal de detecție a actului de aruncare
│   ├── database.py             # modele SQLAlchemy și helpers DB
│   ├── schemas.py              # scheme Pydantic pentru API
│   └── ml/
│       └── two_stage.py        # pipeline YOLO detector + clasificator
├── frontend/
│   ├── templates/              # pagini Alpine/Jinja
│   └── static/
│       ├── js/                 # logică UI, autentificare, monitorizare, admin
│       └── css/                # stiluri
├── models/
│   ├── detector/production/    # model detector final folosit în aplicație
│   └── classify/B2/            # model clasificator folosit în aplicație
├── data/
│   ├── README.md               # explică datele locale generate
│   ├── trash_detection.db      # SQLite local, ignorat de git
│   └── runtime/                # uploaduri, video, thumbnail-uri și dovezi generate
├── certs/                      # certificat HTTPS local generat automat, ignorat de git
├── datasets/                   # dataseturi locale pentru antrenare/evaluare
├── notebooks/
│   ├── training/               # experimente și antrenări
│   └── evaluation/             # evaluări, comparații și figuri tehnice
├── scripts/
│   ├── data/                   # pregătire dataseturi, split, validare, export cropuri
│   ├── training/               # scripturi de antrenare
│   ├── evaluation/             # evaluări standalone
│   ├── maintenance/            # reset DB, creare conturi locale
│   └── smoke/                  # teste manuale rapide pe server pornit
├── tests/                      # teste automate pytest
├── results/                    # rezultate și metrici folosite în lucrare
├── start_https.py              # pornire locală HTTPS pentru camera telefonului
└── requirements.txt
```

Directoarele `datasets/`, `runs/`, `outputs/`, `data/runtime/`, baza locală `data/trash_detection.db` și fișierele `.pt` sunt artefacte locale și nu sunt versionate.

Dataseturile locale păstrate pentru reproducere și evaluare sunt:

- `datasets/parks_detect_final/` - setul final pentru detectorul de deșeuri;
- `datasets/trashnet_cls/` - setul final pentru clasificatorul de material;
- `datasets/test_videos/` - clipuri pentru smoke test, evaluare video și prezentare.

## Pornire rapidă

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe start_https.py
```

Pentru camera telefonului, rulează serverul HTTPS și deschide adresa afișată în consolă de pe același Wi-Fi. Browserul poate afișa un avertisment pentru certificatul local/self-signed; pentru rulare locală se continuă cu `Advanced` / `Help me understand` -> `Proceed`.

Comenzi utile pentru serverul local:

```powershell
# Pornire normală pe portul local HTTPS 8443
.\.venv\Scripts\python.exe start_https.py

# Serverul rulează în terminalul curent; Ctrl+C îl oprește și logurile rămân vizibile.

# Dacă portul este deja ocupat de o rulare veche
.\.venv\Scripts\python.exe start_https.py --restart

# Dacă vrei doar să deschizi serverul deja pornit, fără să-l atașezi la terminal
.\.venv\Scripts\python.exe start_https.py --reuse

# Oprire server local
.\.venv\Scripts\python.exe start_https.py --stop

# Port alternativ, doar dacă 8443 este ocupat de alt program
.\.venv\Scripts\python.exe start_https.py --port 9444

# Fără deschidere automată în browser
.\.venv\Scripts\python.exe start_https.py --no-open

# Fără reload automat, util pentru o prezentare mai stabilă
.\.venv\Scripts\python.exe start_https.py --no-reload
```

Pentru a reduce cererile repetate de Windows Firewall, permite Python/Uvicorn pe rețele private sau rulează o singură dată în PowerShell pornit ca Administrator:

```powershell
New-NetFirewallRule -DisplayName "TrashDet HTTPS Local" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8443,9444 -Profile Private
```

## Configurare locală și server-ready

Setările care diferă între rularea locală și un server se află în `.env` și sunt citite prin `backend/config.py`. Fișierul `.env.example` conține valorile necesare pentru:

```text
SECRET_KEY
APP_BASE_URL
DATABASE_URL
STORAGE_ROOT
DETECTOR_WEIGHTS
CLASSIFIER_WEIGHTS
PERSON_DETECTOR_WEIGHTS
MAX_UPLOAD_MB
LIVE_IMGSZ
DEFAULT_DET_CONF
MONITOR_MIN_DET_CONF
LITTERING_FILE_RETENTION_DAYS
```

Pentru licență, aplicația rulează local cu SQLite implicit (`data/trash_detection.db`). Pentru un deploy ulterior pe server, `DATABASE_URL` poate indica o bază PostgreSQL, iar `APP_BASE_URL` trebuie setat la domeniul/URL-ul serverului. Modelele rămân configurabile prin căi relative la rădăcina proiectului.

Retenția probelor video este activă implicit: fișierele din `data/runtime/littering/` mai vechi de `LITTERING_FILE_RETENTION_DAYS` sunt curățate automat, iar metadata incidentelor rămâne în baza de date.

## Comenzi utile

```powershell
# Evaluare video pe clipuri selectate manual
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_video_events.py --manifest scripts\evaluation\video_manifest_template.csv --frame-skip 1

# Pregătire baza de date locală: admin/utilizator, locație, autoritate, OTP cleanup
.\.venv\Scripts\python.exe scripts\maintenance\prepare_local_db.py --apply --prune-locations --reset-local-passwords

# Reset date generate local
.\.venv\Scripts\python.exe -m scripts.maintenance.reset_data
```

Conturi locale:

```text
admin    / Admin1234!    rol: admin
operator / Operator1234! rol: utilizator
```

Adminul vede panoul de administrare, utilizatorii, locațiile, incidentele, autoritățile și stocarea locală. Utilizatorul folosește monitorizarea și fluxul de incidente fără acces la administrarea organizației.

## Focus până la prezentare

Lucrarea de licență a fost mutată în afara workspace-ului aplicației. Din acest punct, proiectul rămâne concentrat pe stabilizarea demonstrației reale:

1. monitorizare live cu telefonul și cameră locală;
2. detecție pe scenarii filmate real, nu doar pe clipuri de pe internet;
3. upload video cu incidente generate corect;
4. panou admin/utilizator clar și verificabil;
5. scenariu de prezentare reproductibil pentru comisie.
