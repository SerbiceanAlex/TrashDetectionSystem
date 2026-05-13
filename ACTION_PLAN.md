# Plan de Acțiuni — TrashDetectionSystem
**Lucrare de licență UAB Alba Iulia 2026 — Serbicean Alexandru**
**Ultima actualizare: 13 mai 2026**

---

## Starea curentă ✅

| Component | Status | Detalii |
|-----------|--------|---------|
| Model detector A4-8010 | ✅ Antrenat | YOLOv8s, mAP50=0.690, split 80/10/10, 232 test |
| Clasificator B2 | ✅ Antrenat | YOLOv8n-cls, 92.22% Top-1, 257 test |
| Site minimalist | ✅ Gata | 3 tabs: Dashboard + Monitor + Incidente |
| Monitor live | ✅ Gata | Webcam direct, fără configurare locații |
| Demo standalone | ✅ Gata | `scripts/demo_littering.py` |
| Notebooks evaluare | ✅ Gata | 5 notebooks adaptate la A4-8010 |
| Figuri teză | ✅ Generate | `outputs/thesis_figures/F1-F8` |
| Upload video fix | ✅ Fix 13 mai | Auth optional → funcționează fără login |
| HTTPS / iPhone | ✅ Adăugat | `start_https.py` → certificat auto-generat |

---

## ETAPA 0 — Testare pe iPhone cu cameră bună (ACUM)

### 0.1 Pornire server HTTPS (accesibil pe telefon)

```bash
# Instalează cryptography (o singură dată)
pip install cryptography

# Pornește cu HTTPS pe toate interfețele
python start_https.py
```

Terminalul va afișa ceva de genul:
```
  Local:   https://localhost:8443
  iPhone:  https://192.168.1.X:8443
```

### 0.2 Conectare iPhone

1. **Același Wi-Fi** ca laptopul
2. Deschide **Safari** → `https://192.168.1.X:8443`
3. La avertisment SSL → apasă **"Avansat" → "Vizitează site-ul"**
4. *(Opțional pentru WebSocket stabil)* Settings → General → About →  
   Certificate Trust Settings → activează certificatul TrashDetectionSystem

### 0.3 Testare Monitor pe iPhone

1. Mergi la tab **Monitor**
2. Apasă **"Pornește feed"**
3. Permite accesul la cameră
4. Testezi scenariul cu camera iPhone (calitate mult mai bună!)

> **De ce iPhone e mai bun:** Camera laptop = VGA/720p, fixă, slab în lumină slabă.
> Camera iPhone = 4K/12MP, stabilizare optică, HDR → detecție mai precisă.

---

## ETAPA 1 — Testare și validare sistem (Această săptămână: 13-16 mai)

### 1.1 Testare Monitor Live (PRIORITATE MAXIMĂ — mâine 14 mai)

**Fișier:** `frontend/templates/tabs/scan.html` + `frontend/static/js/video.js`
**Acțiune:**
```
1. Pornești serverul: python start_https.py  (sau uvicorn pe port 8000)
2. Mergi la http://localhost:8000 → Monitor
3. Apasă "Pornește feed"
4. Testezi scenariul:
   - Intri în cadru → badge PORTOCALIU "PERSOANĂ DETECTATĂ"
   - Ieși complet → badge MOV "MONITORIZARE ACTIVĂ" + progress bar
   - Pui un obiect pe masă → badge ROȘU "ARUNCARE ILEGALĂ!"
   - Mergi la Incidente → vezi clipul salvat
```

**Ce urmărești:**
- Starea se schimbă corect (CLEAR → PERSON → MONITORING → ALERT)
- Bounding boxes apar pe persoane și gunoi
- Clipul se salvează în `backend/littering/event_XXXXXX/`
- Thumbnail apare în tab-ul Incidente

**Dacă nu merge:** verifică `backend/video.py` funcția `handle_monitor_ws`

---

### 1.2 Testare Upload Video (acum fix!)

Upload video funcționează **fără login** după fix-ul din 13 mai.

```
1. Monitor tab → "Analizează video"
2. Selectează un fișier MP4
3. Urmărești progress bar-ul
4. La final → vezi rezultatele în Incidente
```

**Fix aplicat:** `get_current_user_optional` în loc de `get_current_active_user`
(endpoint-ul nu mai returnează 401 pentru utilizatori neautentificați)

---

### 1.3 Testare Demo Script pe clipuri reale

**Fișier:** `scripts/demo_littering.py`
**Acțiune:**
```bash
# Testează pe clipurile disponibile (alege cel mai clar)
python scripts/demo_littering.py --video datasets/test_videos/littering_cctv_2024.mp4
python scripts/demo_littering.py --video datasets/test_videos/dumping_neighbor_00001.mp4
python scripts/demo_littering.py --video datasets/test_videos/illegal_dumping_cctv.mp4

# Salvează demo cu output video
python scripts/demo_littering.py --video datasets/test_videos/littering_cctv_2024.mp4 --save
```

**Ce urmărești:**
- Câte alerte detectează pe clip (numărul din terminal la final)
- Vizual: bounding boxes corecte pe persoane și gunoi
- Stările se schimbă logic
- Latența: în câte secunde după eveniment apare ALERT

**Documentează:** notează numărul de TP/FN per clip pentru teză

---

### 1.3 Testare personală cu webcam (demo pentru profesor)

**Acțiune:**
```bash
python scripts/demo_littering.py --camera 0 --save
```

**Scenariul recomandat:**
1. Stai în fața camerei 3-4 secunde (cu un obiect în mână)
2. Ieși complet din cadru
3. Din exterior, pune obiectul pe masă/jos
4. Aștepți 3-5 secunde
5. Sistemul generează ALERT + salvează clip

**Filmează ecranul** pentru demonstrația la profesor!

---

## ETAPA 2 — Rulare notebook-uri evaluare (14-15 mai)

**Ordine obligatorie:**

```
1. notebooks/evaluation/01_evaluate_detector.ipynb
   → Metrici A4-8010: mAP50, F1, confidence sweep, speed
   → Timp: ~15 min (GPU)

2. notebooks/evaluation/02_evaluate_classifier.ipynb
   → Metrici B2: accuracy, ROC curves, per-class F1
   → Timp: ~5 min

3. notebooks/evaluation/03_detector_comparison.ipynb
   → Grafice A22→A3→A4→A4-8010 evoluție
   → Timp: ~5 min (cross-eval pe GPU)

4. notebooks/evaluation/04_inference_demo.ipynb
   → Pipeline A4-8010 + B2 pe 232 imagini test
   → Timp: ~10 min

5. notebooks/evaluation/05_thesis_figures.ipynb
   → Toate figurile pentru teză (F1-F8)
   → Timp: ~3 min
```

**Atenție:** după fiecare notebook, verifică că nu sunt erori înainte de a trece la următorul.

---

## ETAPA 3 — Commit și curățare finală (15 mai)

**Fișiere de commit:**
```bash
git add backend/config.py
git add backend/main.py
git add frontend/templates/
git add frontend/static/js/app.js
git add scripts/demo_littering.py
git add notebooks/
git add results/
git add README.md ACTION_PLAN.md
git commit -m "feat(demo+ui): monitor live direct, site minimalist, demo littering script"
```

---

## ETAPA 4 — Redactare teză (16-24 mai)

### Structura capitolelor (40-70 pagini, format UAB)

```
Pagina de titlu
Cuprins
Introducere (2-3 pag)
  - Problema aruncării ilegale
  - Obiectivele lucrării
  - Structura lucrării

Capitol 1 — Fundamente teoretice (8-10 pag)
  1.1 Rețele neuronale convoluționale (CNN)
  1.2 Arhitecturi de object detection (YOLO, R-CNN)
  1.3 YOLOv8 — arhitectură și inovații
  1.4 Tracking video (ByteTrack)
  1.5 Transfer learning și fine-tuning
  1.6 Metrici de evaluare (mAP, F1, IoU)

Capitol 2 — Date și pregătire dataset (6-8 pag)
  2.1 Sursele de date (Parks + TACO)
  2.2 Procesarea și augmentarea datelor
  2.3 Split-ul dataset 80/10/10
  2.4 Analiza distribuției claselor

Capitol 3 — Antrenarea modelelor (8-10 pag)
  3.1 Evoluția experimentelor A22→A3→A4→A4-8010
  3.2 Configurația de antrenare
  3.3 Analiza curbelor de convergență
  3.4 Clasificatorul B2

Capitol 4 — Algoritmul de detecție a aruncării (8-10 pag)
  4.1 Definirea problemei temporale
  4.2 Mașina de stări (CLEAR→PERSON→MONITORING→ALERT)
  4.3 Detecție persoane și tracking ByteTrack
  4.4 Fereastra de monitorizare (10 secunde)
  4.5 GDPR: anonimizare facială
  4.6 Evidența evenimentului (clip + thumbnail + hash)

Capitol 5 — Evaluare și rezultate (10-12 pag)
  5.1 Metrici detector A4-8010 (mAP50=0.690, F1=0.703)
  5.2 Threshold optim (confidence sweep)
  5.3 Speed benchmark (105 FPS)
  5.4 Metrici clasificator B2 (92.22%, AUC=0.987)
  5.5 Evaluare sistem littering (FPR=0% pe footage negativ)
  5.6 Comparație cu literatura de specialitate

Capitol 6 — Integrare practică (3-4 pag)
  6.1 Arhitectura aplicației (FastAPI + Alpine.js)
  6.2 WebSocket real-time
  6.3 Demo și rezultate vizuale

Concluzii (2-3 pag)
  - Contribuții originale
  - Limitări ale sistemului
  - Direcții viitoare

Bibliografie (min. 10 surse IEEE)
```

### Surse bibliografice necesare (format IEEE)
```
[1] J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," CVPR, 2016.
[2] G. Jocher et al., "Ultralytics YOLOv8," GitHub, 2023. https://github.com/ultralytics/ultralytics
[3] M. Fulton et al., "Robotic Detection of Marine Litter Using Deep Visual Detection Models," ICRA, 2019.
[4] S. Proença et al., "TACO: Trash Annotations in Context for Litter Detection," arXiv, 2020.
[5] D. Yang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," ECCV, 2022.
[6] K. He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
[7] A. Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection," arXiv, 2020.
[8] G. Huang et al., "TrashNet Dataset," Stanford University, 2017.
[9] T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," ECCV, 2014.
[10] GDPR, "Regulation (EU) 2016/679 — General Data Protection Regulation," EUR-Lex, 2016.
```

---

## ETAPA 5 — Prezentare PowerPoint (pentru susținere)

### Structura prezentare (10 minute, 8-10 slide-uri)

```
Slide 1: Titlu + autor + coordonator
Slide 2: Problema — aruncarea ilegală, de ce contează
Slide 3: Arhitectura sistemului (diagramă flux)
Slide 4: Dataset și experimentele (A22→A4-8010, grafic mAP50)
Slide 5: Rezultatele modelului (tabel metrici + grafice)
Slide 6: Algoritmul de surprindere a actului (state machine diagram)
Slide 7: Demo vizual (screenshot/video cu detecția)
Slide 8: Concluzii + contribuții originale
```

**Graficele sunt gata** în `outputs/thesis_figures/`

---

## Fișiere cheie de urmărit

| Fișier | Rol | Când îl modifici |
|--------|-----|-----------------|
| `backend/config.py:22` | Model de producție | Dacă antrenezi un model mai bun |
| `backend/littering_detector.py` | State machine | Dacă ajustezi parametri detecție |
| `backend/video.py` | WebSocket procesare | Dacă ajustezi confidența |
| `scripts/demo_littering.py` | Demo standalone | Testare și demo profesor |
| `notebooks/evaluation/01_evaluate_detector.ipynb` | Metrici oficiale | După orice antrenare nouă |
| `outputs/thesis_figures/` | Figuri teză | Regenerezi după fiecare evaluare |

---

## Calendar deadline

```
13 mai (azi)   → Testare Monitor Live + demo script
14 mai (mâine) → Prezentare progres profesor Arpad (ora 10, H1.5)
14-15 mai      → Rulare notebook-uri evaluare complete
15 mai         → Commit final stare stabilă
16-22 mai      → Redactare teză (cap. 1-5)
22-23 mai      → Revizii + finalizare
24 mai         → PREDARE
```

---

## Parametri importanți sistem (nu modifica fără motiv)

```python
# backend/config.py
DETECTOR_WEIGHTS  = "runs/detect/parks-trash-A4-8010/weights/best.pt"
CLASSIFIER_WEIGHTS = "runs/classify/parks-cls-B2/weights/best.pt"

# backend/video.py (constantele littering detector)
DET_CONF = 0.30          # confidence trash
PERSON_CONF = 0.40       # confidence persoane
MONITOR_SECONDS = 10.0   # fereastra de monitorizare
PRE_EVENT_SECONDS = 5.0  # clip inainte de eveniment
```
