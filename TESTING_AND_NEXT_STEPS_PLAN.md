# Testing and Next Steps Plan

Scop: proiectul ramane axat pe modelul final si pe demonstratia ca sistemul detecteaza evenimente de aruncare ilegala in video, nu doar obiecte statice.

## 0. Starea Inghetata A Proiectului

Artefacte active:

- Detector final: `models/detector/production/best.pt`
- Dataset final: `datasets/parks_detect_final`
- Run final: `runs/detect/parks-trash-final`
- Rezultate finale: `results/detector/parks-trash-final-test.json`
- Smoke video: `scripts/smoke/pipeline_e2e_smoke.py`
- Evaluare video batch: `scripts/evaluation/evaluate_video_events.py`
- Smoke sistem: `scripts/smoke/littering_system_smoke.py`

Metrici finale detector:

| Metric | Value |
|---|---:|
| Precision | 0.9234 |
| Recall | 0.8229 |
| F1 | 0.8703 |
| mAP50 | 0.9102 |
| mAP50-95 | 0.6834 |

Regula: nu mai schimbam modelul final sau datasetul final fara motiv puternic. De acum testam, documentam si integram.

Storage live:

- clipurile si thumbnail-urile incidentelor se pastreaza local in `backend/littering`;
- retention implicit: `30` zile (`LITTERING_FILE_RETENTION_DAYS`);
- backend-ul ruleaza cleanup automat la pornire si apoi la fiecare `24` ore;
- DB-ul pastreaza metadatele incidentului, dar sterge fisierele vechi ca sa nu creasca disk-ul la infinit.

## 1. Verificare Tehnica De Baza

Ruleaza dupa orice cleanup sau modificare importanta:

```powershell
.\.venv\Scripts\python.exe scripts\data\validate_yolo_dataset.py --data datasets\parks_detect_final\dataset.yaml
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('models/detector/production/best.pt'); print('production model OK')"
.\.venv\Scripts\python.exe scripts\smoke\littering_system_smoke.py
.\.venv\Scripts\python.exe scripts\smoke\pipeline_e2e_smoke.py
```

Criteriu de succes:

- dataset valid;
- modelul se incarca;
- smoke sistem trece;
- smoke video detecteaza cel putin un eveniment pe clipul de referinta;
- FPS peste 25 pe laptop.

## 2. Testare Detector Static

Obiectiv: confirmam ca detectorul final ramane solid pe setul test.

Ruleaza notebookul:

- `notebooks/evaluation/01_evaluate_detector.ipynb`

Ce trebuie extras pentru licenta:

- precision;
- recall;
- F1;
- mAP50;
- mAP50-95;
- confusion matrix;
- curbe PR/F1;
- exemple vizuale cu ground truth vs predictii.

Rezultat asteptat:

- mAP50 in jur de `0.91`;
- F1 in jur de `0.87`.

## 3. Testare Clasificator Material

Obiectiv: verificam componenta secundara care clasifica materialul obiectului detectat.

Ruleaza:

- `notebooks/evaluation/02_evaluate_classifier.ipynb`

Ce verificam:

- acuratete pe clase;
- clase confundate frecvent;
- daca rezultatul clasificatorului este suficient pentru demo.

Important: detectorul este piesa principala a lucrarii. Clasificatorul este suport informativ, nu metricul central.

## 4. Testare Video Si Eveniment Temporal

Obiectiv: demonstram ca sistemul detecteaza actul de aruncare, nu doar gunoi static.

Smoke rapid pe clipul de referinta:

```powershell
.\.venv\Scripts\python.exe scripts\smoke\pipeline_e2e_smoke.py
```

Evaluare batch pe mai multe clipuri:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_video_events.py --clips all --frame-skip 1
```

Pentru o verificare mai scurta:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_video_events.py --clips littering_cctv_2024.mp4,dumping_neighbor_00001.mp4,cctv_parking_away_00001.mp4 --frame-skip 1 --out results\video_events\quick_video_event_eval
```

Rezultatele se salveaza in:

- `results/video_events/*.csv`
- `results/video_events/*.json`

Apoi eticheteaza manual clipurile din:

- `datasets/test_videos`

Pentru fiecare clip notam:

| Clip | Are eveniment real? | Sistemul detecteaza? | Timp detectie | False positive? | Observatii |
|---|---|---|---|---|---|

Categorii de clipuri:

- pozitive: persoana arunca/lasa obiect;
- negative: persoana trece fara sa arunce;
- dificile: obiect mic, lumina slaba, fundal aglomerat.

Criteriu de succes:

- detecteaza evenimentele clare;
- nu alerteaza constant pe clipuri negative;
- functioneaza peste 25 FPS.

## 5. Testare In Aplicatie

Obiectiv: verificam demo-ul complet pentru comisie.

Pasi:

1. Porneste serverul:

```powershell
py start_https.py
```

2. Intra in interfata web.
3. Testeaza:

- upload video;
- monitor live;
- detectie persoana;
- detectie trash;
- generare incident;
- thumbnail/metadata;
- afisare in dashboard/admin.

Criteriu de succes:

- nu apar erori in consola backend;
- nu apar blocaje in UI;
- incidentul se salveaza;
- demo-ul poate fi repetat de 2-3 ori.

## 6. Test Set Real Propriu

Obiectiv: cresterea credibilitatii lucrarii prin test independent.

Colectare recomandata:

- 3-5 locatii exterioare: parc, alee, zona verde, parcare langa zona verde;
- 10-20 clipuri scurte;
- 50-100 cadre relevante extrase;
- 20-30 cadre negative fara gunoi sau fara act de aruncare.

Nu trebuie sa antrenam neaparat pe aceste date. Ele pot fi folosite ca test independent.

Ce raportam:

- cate cazuri reale a prins;
- unde greseste;
- exemple vizuale;
- limite: distanta, obiecte mici, ocluzii, lumina.

## 7. Ce Scriem In Licenta

Fir narativ recomandat:

1. problema: detectia aruncarii ilegale in spatii publice;
2. pipeline: detector + clasificator + motor temporal;
3. date: dataset final filtrat, 80/10/10;
4. antrenare: detector final YOLOv8s, 150 epoci;
5. rezultate detector: mAP50 `0.9102`;
6. testare video: evenimente detectate in timp real;
7. limitari: false positives, obiecte mici, dependenta de perspectiva;
8. directii viitoare: mai multe clipuri reale, pose/gesture detection, dataset local extins.

Nu incarcam lucrarea cu toate experimentele esuate. Le mentionam doar scurt ca motivatie pentru filtrarea datelor.

## 8. Ordinea De Lucru De Acum

1. Commit/snapshot la starea curata.
2. Ruleaza verificarile tehnice de baza.
3. Ruleaza notebookul de evaluare detector.
4. Ruleaza smoke video pe 3-5 clipuri.
5. Testeaza aplicatia web end-to-end.
6. Colecteaza un mic test set real.
7. Scrie capitolul de rezultate cu tabele si imagini.
8. Pregateste demo-ul final.

## 9. Criteriu Final De Gata

Proiectul este pregatit pentru licenta cand:

- `models/detector/production/best.pt` este singurul detector folosit;
- `datasets/parks_detect_final` este singurul dataset detector activ;
- smoke tests trec;
- aplicatia genereaza incident pe video;
- exista tabel final cu metrici;
- exista 3-5 imagini/predictii pentru lucrare;
- exista un clip demo sigur pentru prezentare.
