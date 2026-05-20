# Labeling Guidelines — Dataset A5 Open Challenge
**Standard de adnotare pentru detector single-class `trash`**
**Versiune:** 1.0 — 15 mai 2026

---

## SCOPUL DOCUMENTULUI

Acest document definește **EXACT** ce se adnotează și cum, pentru ca dataset-ul să fie consistent. Dacă 2 persoane adnotează același cadru folosind aceste reguli, rezultatul trebuie să fie identic.

**De ce contează:** un model antrenat pe date inconsistente învață **zgomotul** adnotării, nu obiectul. Un dataset de 1000 cadre cu reguli clare bate un dataset de 5000 cadre cu reguli mixte.

---

## CLASA UNICĂ — `trash`

Modelul detectează **o singură clasă**: `trash`. Toate obiectele de deșeu primesc aceeași etichetă.

**Clasificarea materialului** (plastic / metal / sticlă / hârtie / altul) se face în etapa următoare de clasificatorul B2 — **NU în adnotare**.

---

## REGULA #1 — CE ADNOTĂM CA `trash`

### ✅ Se ADNOTEAZĂ

| Scenariu | Bbox |
|----------|------|
| Sticlă plastic / sticlă de sticlă (în mână, pe jos, în zbor) | Strâns pe obiect |
| Doză aluminiu | Strâns pe obiect |
| Pungă plastic (mototolită sau întinsă) | Pe conturul vizibil |
| Ambalaj / hârtie / șervețel | Pe conturul vizibil |
| Pahar carton/plastic (gol) | Strâns pe obiect |
| Cutie de mâncare (pizza box, kebab box) | Strâns pe obiect |
| Chiștoc țigară | Cât mai strâns |
| Recipient băutură (cup with straw) | Pe întreaga formă |
| **Obiect în mână gata să fie aruncat** | Pe obiect, NU pe mână |
| **Obiect în zbor (cădere/aruncare)** | Pe obiect |
| **Obiect pe jos (post-aruncare)** | Strâns |

### ❌ NU se ADNOTEAZĂ

| Scenariu | Motiv |
|----------|-------|
| Obiecte funcționale (telefon, geantă, ghiozdan, jacheta) | Nu sunt deșeu |
| Obiecte de îmbrăcăminte | Excepție: doar dacă sunt CLAR aruncate ca deșeu |
| Frunze, crengi, materie organică naturală | Nu sunt "trash" în context urban |
| Obiecte pe sol care arată ca trash dar sunt parte din scenă (gulere de șuruburi, capace) | Ambiguu — skip |
| Obiecte complet ocluzate (>80% acoperite) | Nu se mai vede obiectul |
| Obiecte sub 0.5% din suprafața frame-ului (< 30×30 pixeli la 1080p) | Prea mici, distrug recall metric |

---

## REGULA #2 — STRÂNGEREA BOUNDING BOX (TIGHT BBOX)

Bbox-ul trebuie să fie **STRÂNS pe obiect** — fără margini libere, dar fără să taie din obiect.

### Corect:
```
┌──────────┐
│  ████    │  bbox trash — strâns la sticlă, fără spațiu mare
│  ████    │
│  ████    │
└──────────┘
```

### Greșit (prea larg):
```
┌──────────────┐
│              │
│    ████      │  margine prea mare jur-împrejur
│    ████      │
│              │
└──────────────┘
```

### Greșit (taie):
```
████          │  bbox prea mic, taie din obiect
████          │
█             │
```

**Toleranță acceptată:** ±2-3 pixeli pe fiecare latură.

---

## REGULA #3 — OCLUZIE PARȚIALĂ

**Întrebare cheie:** Cât din obiect trebuie să fie vizibil pentru a-l adnota?

### Pragul: ≥ 30% vizibil din suprafața obiectului

| Vizibilitate | Acțiune |
|--------------|---------|
| 100% — obiect complet vizibil | ✅ Adnotează bbox strâns pe partea vizibilă |
| 70-99% — ocluzie ușoară (mână, marginea altui obiect) | ✅ Adnotează bbox pe partea vizibilă |
| 30-70% — ocluzie semnificativă (jumătate ascuns după corp) | ✅ Adnotează doar partea vizibilă |
| 10-30% — ocluzie majoră (doar un colț se vede) | ⚠️ Adnotează DOAR dacă e clar că e trash (nu confuz) |
| < 10% — practic invizibil | ❌ NU adnotează |

**Bbox-ul include DOAR partea vizibilă**, nu obiectul "imaginar" întreg.

```
┌────┬─────┐
│████│xxxxx│   ████ = partea vizibilă din sticlă
│████│xxxxx│   xxxx = partea ascunsă după corp
└────┴─────┘
     ↑
   bbox aici, doar pe ████
```

---

## REGULA #4 — OBIECT ÎN MÂNĂ

Acesta e **scenariul cheie** al tezei. Adnotăm OBIECTUL, nu mâna.

### Exemplu: persoană ține o sticlă

```
     ▒▒▒▒▒
     ▒▒▒▒▒    ← cap persoană (NU adnotăm)
     ▒▒▒▒▒
  ▒▒▒▒▒▒▒▒    ← corp persoană (NU adnotăm)
  ▒▒▒▒▒▒▒▒
  ▒▒█████▒    ← mână + sticlă: bbox doar pe █████ (sticla)
       ▒▒
```

**Regula:** dacă mâna acoperă o parte din sticlă, **bbox e pe partea vizibilă a sticlei**. Nu include mâna.

---

## REGULA #5 — OBIECTE ÎN ZBOR / CĂDERE

Obiectele în mișcare (cădere, aruncare) **se adnotează**, chiar și cu motion blur.

**Atenție:** dacă obiectul e atât de "blurred" încât nu se mai poate identifica ca trash (e o pată), **nu se adnotează**.

```
   ●        ← frame 1: obiect clar
   ╱        
  ●         ← frame 2: obiect cu motion blur ușor — ADNOTĂM
 ╱          
●           ← frame 3: pată complet neclară — NU adnotăm
```

---

## REGULA #6 — OBIECTE MULTIPLE ADIACENTE

| Scenariu | Cum adnotăm |
|----------|-------------|
| 2 sticle separate, distanță > 20px | 2 bbox-uri separate |
| 2 sticle lipite (≤ 20px distanță) | 2 bbox-uri (1 fiecare) |
| Pungă cu 3 sticle vizibile prin folie | **1 bbox pe pungă** (pungă = trash container) |
| Grămadă de gunoi nesegregat | 1 bbox pe întreaga grămadă |

**Regula generală:** un bbox per **obiect distinct identificabil**. Dacă nu poți distinge clar limita, e 1 bbox.

---

## REGULA #7 — CADRE NEGATIVE (fără trash)

**CRITIC pentru a evita false positives:**

Cadrele unde **persoana NU aruncă** (merge prin cadru cu sau fără obiect, dar nu lasă nimic) trebuie:
- Să existe în dataset cu fișier `.txt` GOL (zero bbox-uri)
- Reprezintă ~30% din totalul de cadre

Lipsa cadrelor negative = modelul învață că orice persoană + obiect = aruncare.

### Cum marchezi un cadru ca negativ:

În YOLO format, fișierul `imagine.txt` rămâne **gol** (sau nu există). NU pune linie cu coordonate "0 0 0 0 0" — aceea ar fi un bbox invalid.

---

## REGULA #8 — DIMENSIUNE MINIMĂ BBOX

**Prag minim:** bbox cu suprafață < 0.05% din frame se ignoră.

La 1920×1080 (Full HD): 0.05% = ~62×34 pixeli minim.
La 1280×720 (HD): 0.05% = ~42×23 pixeli minim.

Motivul: obiecte sub acest prag sunt **prea mici pentru detecție robustă** la imgsz=640 (resize la 1/3 = obiectul devine ~10×10 px, sub limita YOLO de 8×8).

---

## REGULA #9 — REGULI SPECIFICE PER SCENARIU

### Scenariul A — Aruncare din mișcare (de mers)
- Adnotează obiectul în mână (cadru pre-aruncare)
- Adnotează obiectul în zbor (cadru aruncare)
- Adnotează obiectul pe jos (cadre post-aruncare)
- Persoana din cadre = nu adnotăm (vom folosi yolov8n COCO pentru persoane)

### Scenariul B — Drop simplu (lăsat jos)
- Adnotează obiectul în mână
- Adnotează obiectul în momentul când atinge solul
- Adnotează obiectul rămas pe jos (în următoarele 1-3 secunde)

### Scenariul C — Persoană trece fără să arunce (NEGATIVE)
- Dacă obiectul e ținut în mână și nu cade: **nu adnotăm obiectul** (este negativ)
- Cadre `.txt` gol = important pentru training

**De ce nu adnotăm sticla în mână la negative?** Pentru că vrem ca modelul să fie **mai puțin sensibil** când obiectul nu cade. În cadrele pozitive (aruncare), obiectul în mână e adnotat — modelul învață contextul "aruncă" prin frame-urile următoare.

### Scenariul D — Gunoi există deja înainte de cadru
- Adnotează obiectul pe jos
- Dacă persoana ridică gunoiul (curăță) — nu adnotăm persoana

---

## REGULA #10 — CE EVITĂM

❌ **NU adnotăm sticla "imaginară"** — dacă vezi doar un colț, adnotează DOAR colțul vizibil
❌ **NU adnotăm obiecte ambigue** — dacă nu ești sigur că e trash, skip
❌ **NU folosim bbox foarte larg** — cu margine de 10-20% în jurul obiectului
❌ **NU adnotăm dacă obiectul e parte din decor permanent** (coș de gunoi cu pungi vizibile = nu adnotăm, nu e littering)

---

## VALIDARE — CHECKLIST PE FIECARE CADRU

Înainte să salvezi adnotarea, verifică:

- [ ] Toate obiectele de tip trash din cadru sunt adnotate?
- [ ] Bbox-urile sunt strânse (fără margini libere)?
- [ ] Obiecte sub 0.05% din frame — sunt skip?
- [ ] Cadre negative — fișier `.txt` gol?
- [ ] Obiecte cu ocluzie > 70% — sunt skip?

---

## TOOLS RECOMANDATE

### Roboflow (recomandat — UI simplu, free 10k img)
1. Cont gratuit pe roboflow.com
2. Upload cadre, single class "trash"
3. Draw bbox tool
4. Export → YOLO v8 format

### CVAT (auto-hosted, mai serios)
- Pentru dataset mare > 10k cadre
- Mai complex, suportă track-uri video

### LabelImg (offline, simplu)
- Open source, Python
- Bun pentru sesiuni rapide

---

## METADATA — ATAȘĂM LA FIECARE CADRU ADNOTAT

Pentru audit/teză, atașăm informație în fișier separat `meta.csv`:

```csv
filename,source,scenario,occlusion,quality,annotator,date
clip_001_frame_005.jpg,stipra,handheld,clear,good,alex,2026-05-15
clip_001_frame_006.jpg,stipra,throwing,clear,good,alex,2026-05-15
```

Câmpuri:
- **scenario**: `handheld | throwing | falling | ground | negative`
- **occlusion**: `clear | partial | heavy`
- **quality**: `good | blurry | dark`

Cu metadata, putem face **error analysis** în teză: "ce tip de cadre rateaza modelul cel mai des?"

---

## EXEMPLE DE REFERINȚĂ

**Pentru consistență**, atunci când adnotăm primele 50 de cadre, fiecare adnotator trebuie să consulte acest document de **2 ori**:
1. Înainte să înceapă (citire integrală)
2. La cadrul 25 (review rapid pentru calibrare)

După 50 cadre, regulile vor deveni intuitive.

---

## REVIZUIRE — INTER-ANNOTATOR AGREEMENT

Dacă 2+ persoane adnotează:
- 50 cadre comune adnotate de ambii
- Calculezi IoU mediu între bbox-urile lor
- **Ținta: IoU > 0.85** = consistență bună
- IoU < 0.70 = reluați regulile, calibrarea

Pentru o singură persoană (cazul tău): re-vezi primele 50 cadre după 1 zi — vezi dacă ai aplicat reguli identice.

---

## REFERINȚE ÎN TEZĂ

Capitolul 2 (Date și pregătire experimentală) va cita explicit acest document:

> *"Adnotarea s-a făcut conform unui ghid intern (LABELING_GUIDELINES.md) care definește regulile de adnotare pentru clasa unică `trash`. Pragul minim de vizibilitate al obiectului este 30%, dimensiunea minimă a bbox-ului este 0.05% din suprafața cadrului, iar cadrele negative (fără aruncare) reprezintă aproximativ 30% din total."*

---

**Cu acest document, dataset-ul A5_open_challenge va avea calitate de publicație academică.**
