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
  -> Detector YOLOv8 A4-8010: localizeaza obiectele trash
  -> Clasificator YOLOv8 B2: estimeaza materialul
  -> Behavioral Engine: decide daca exista act de aruncare
```

Greutatile finale folosite de aplicatie stau in `models/`:

```text
models/
├── detector/A4-8010/best.pt
└── classify/B2/best.pt
```

`runs/` ramane zona de experimente si antrenari YOLO. Cand alegi un model final, copiezi checkpoint-ul castigator in `models/`.

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
│   ├── detector/A4-8010/       # model detector folosit in aplicatie
│   └── classify/B2/          # model clasificator folosit in aplicatie
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
└── ACTION_PLAN.md
```

Directoarele `datasets/`, `runs/`, `outputs/`, `backend/uploads/`, `backend/videos/` si fisierele `.pt` sunt artefacte locale si nu sunt versionate.

## Pornire rapida

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py start_https.py
```

Pentru camera telefonului, ruleaza serverul HTTPS si deschide adresa afisata in consola de pe acelasi Wi-Fi.

## Comenzi utile

```powershell
# Demo vizual pe clip
.\.venv\Scripts\python.exe scripts\demos\demo_littering.py --video datasets\test_videos\clip.mp4

# Demo live camera
.\.venv\Scripts\python.exe scripts\demos\demo_littering.py --camera 0

# Creare user admin/demo local
.\.venv\Scripts\python.exe scripts\maintenance\create_admin.py

# Reset date generate local
.\.venv\Scripts\python.exe -m scripts.maintenance.reset_data
```

## Focus pentru lucrare

In redactare, aplicatia web trebuie tratata ca suport practic. Partea principala merita sa fie:

1. pregatirea dataseturilor si antrenarea modelelor;
2. evaluarea detectorului A4-8010 si a clasificatorului B2;
3. logica Behavioral Engine si criteriile temporale;
4. experimentele pe clipuri reale si analiza erorilor;
5. limite, GDPR si directii viitoare.
