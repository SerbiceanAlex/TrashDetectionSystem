# TrashDetectionSystem — Sistem Inteligent de Monitorizare a Deșeurilor

Sistem two-stage bazat pe **YOLOv8** pentru detecția și clasificarea deșeurilor în spații verzi urbane, cu o componentă activă de **Behavioral Engine** capabilă să identifice abandonul ilegal de deșeuri (littering) în timp real din fluxuri video.

**Lucrare de licență — Universitatea 1 Decembrie 1918 Alba Iulia, 2026.**
**Autor:** Serbicean Alexandru

---

## Scopul Proiectului

Platforma a evoluat de la o aplicație de tip "gamification/reporting" la un **Sistem Administrativ de Monitorizare** cu următoarele capabilități principale:
1. **Monitorizare Live (Behavioral Engine):** Analizează fluxul camerei (laptop sau telefon) pentru a detecta momentul exact când o persoană abandonează gunoi.
2. **Offline Video Processing:** Permite încărcarea clipurilor CCTV (MP4) și procesarea lor asincronă pentru identificarea actelor de littering.
3. **GDPR by Design:** Toate dovezile video/foto salvate au fețele anonimizate (blurate) automat prin algoritmi de detecție facială, respectând Art. 25 GDPR.
4. **Mobile-First & HTTPS:** Optimizat pentru a rula pe telefoane mobile (iPhone) pentru calitatea superioară a camerei, utilizând un flux securizat HTTPS.

---

## Arhitectură Modele AI (Two-Stage)

Sistemul folosește o arhitectură în doi pași pentru precizie maximă și reducerea alarmelor false (FP):

```
Imagine/Video → [Stage 1: Detector YOLO] → bounding boxes (clasa: trash)
                                                    ↓
                                    [Stage 2: Clasificator YOLO]
                                                    ↓
                            material: glass / metal / paper / plastic / other
```

### Modele în producție:
- **Detector A4-8010:** YOLOv8s, imgsz=640. Antrenat pe un dataset combinat (Parks + TACO). Are un **False Positive Rate de 0%** pe videoclipurile reale din parcuri și o performanță de **mAP50=0.666**. 
  - Locație: `runs/detect/parks-trash-A4-8010/weights/best.pt`
- **Clasificator B2:** YOLOv8n-cls, imgsz=224. Antrenat pe TrashNet și secțiuni extrase din datasetul Parks. Acuratețe Top-1 de **91.1%** pe test set.
  - Locație: `runs/classify/parks-cls-B2/weights/best.pt`

---

## Aplicația Web (FastAPI + Alpine.js)

Interfața web este minimalistă, centrată strict pe funcționalitățile de detecție, renunțând la modulele inutile (SaaS, gamificare) pentru stabilitate maximă:

1. **Dashboard:** O privire de ansamblu (Placeholder vizual minimalist).
2. **Monitor:** 
   - Mod **Live Camera**: Monitorizare fullscreen, cu indicatori de stare (LIBER, PERSOANĂ, VERIFICARE, INCIDENT).
   - Mod **Upload Video**: Permite analiza clipurilor înregistrate, rulează într-un thread de fundal, generând incidente la finalizarea analizei.
3. **Incidente:** Galerie cu toate dovezile de aruncare ilegală a gunoiului (clipuri scurte extrase automat, plus metadata despre material și scorul de încredere).

---

## Setup & Pornire

### Cerințe:
- Python 3.11+
- Mediu virtual (`.venv`)
- Pachetul `cryptography` pentru certificate HTTPS

### Instalare:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install cryptography
```

### Pornire Server (Mod HTTPS pentru iPhone/Mobil):
Pentru a accesa camera de înaltă calitate a telefonului, serverul trebuie să ruleze pe HTTPS (restricție impusă de browsere pe mobile):

```bash
python start_https.py
```
*Scriptul va genera automat un certificat SSL self-signed și va porni serverul pe `0.0.0.0:8443`.*

**Conectare de pe telefon:**
1. Conectează telefonul la același Wi-Fi ca PC-ul.
2. Deschide Safari și accesează adresa afișată în consolă (ex: `https://192.168.1.X:8443`).
3. La avertismentul de securitate, selectează `Avansat -> Vizitează site-ul`.
4. Mergi în tab-ul **Monitor** și pornește feed-ul. Interfața va intra automat în modul *Fullscreen Monitor*.

---

## Structura Proiectului (Actualizată)

```text
TrashDetectionSystem/
├── src/                         # Scripturi utilitare și pipeline-uri standalone
├── backend/
│   ├── main.py                  # API Endpoints (FastAPI) & rute statice
│   ├── inference.py             # Logica core YOLO, tracking (DeepSORT) și Face Blurring
│   ├── video.py                 # Handlere WebSocket pentru Live Monitor și Uploads
│   ├── database.py              # ORM SQLAlchemy
│   ├── config.py                # Configurație (variabile de mediu)
│   └── uploads/, videos/, littering/ # Directoare generate automat pentru dovezi
├── frontend/
│   ├── static/js/video.js       # Logica de client Alpine.js (WebSocket, canvas, poll)
│   └── templates/tabs/          # Structura HTML (dashboard, scan, incidents)
├── scripts/
│   └── demo_littering.py        # Script standalone pentru testarea detecției pe fișiere video
├── start_https.py               # Launcher pentru mediu HTTPS local
├── ACTION_PLAN.md               # Documentul viu cu pașii de dezvoltare și testare
└── README.md                    # Prezentarea generală a proiectului
```

---

## Următorii Pași (Vezi `ACTION_PLAN.md`)
Sistemul este stabil și pregătit pentru testarea finală. Următoarea etapă constă în generarea datelor reale de test prin:
1. Plasarea telefonului ca sursă live (CCTV).
2. Simulare de aruncare ilegală în cadru.
3. Confirmarea înregistrării clipului în tab-ul "Incidente" cu fețele anonimizate corect.