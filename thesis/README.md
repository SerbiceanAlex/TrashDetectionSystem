# Licență LaTeX

Titlu: **Sistem inteligent pentru detectarea deșeurilor în spații verzi utilizând procesarea imaginilor video**

## Regulă principală

Ghidul oficial UAB 2026 este sursa principală și se respectă înaintea oricărui alt șablon sau preferință. Regula este notată separat în `NORME_OBLIGATORII_UAB_2026.md`.

## Norme aplicate din ghidul UAB 2026

- format A4;
- margini: stânga 2,5 cm, dreapta 2 cm, sus 2 cm, jos 2 cm;
- Times New Roman, 12 pt;
- spațiere 1,5;
- text justificat;
- indentare paragraf 1,25 cm;
- număr de pagină centrat în subsol;
- pagina de titlu este numerotată intern, dar numărul nu se afișează;
- cuprinsul și fiecare capitol încep pe pagină impară;
- figurile și tabelele se numerotează automat;
- citările sunt numerice, în stil IEEE.

## Observații din fișierele primite

Ghidul oficial de licență are prioritate. Șablonul de proiect de cercetare este util doar ca inspirație pentru abstract/metodologie, dar folosește APA și structură generică de cercetare. Pentru lucrarea de licență la Informatică folosim citare IEEE, conform ghidului oficial.

Fișierele sau textele de tip proiect/articol scurt, inclusiv cele care cer abstract, cuvinte cheie, APA, text la un rând sau 4-10 pagini, nu se folosesc drept șablon principal pentru lucrarea de licență. Ele pot ajuta doar la formularea concisă a motivației, metodelor, rezultatelor și concluziilor.

Analiza detaliată este în `ANALIZA_FISIERE.md`.

Textele extrase din ghiduri și șabloane sunt păstrate în `sources/extracted/`.
Ele sunt materiale de referință pentru redactare, nu capitole finale ale lucrării.

## Buget recomandat

Ținta noastră este o lucrare compactă, nu una umflată artificial: aproximativ **30-40 pagini de conținut redactat** și aproximativ **40-45 pagini în PDF-ul final**, după includerea copertei, paginilor albe, cuprinsului, bibliografiei, figurilor și tabelelor.

Astfel respectăm ghidul oficial, care recomandă 40-70 pagini, dar ne orientăm spre limita inferioară.

- Introducere: 3-4 pagini;
- Capitolul 1, Contextul problemei și cerințele domeniului: 5-6 pagini;
- Capitolul 2, Fundamente teoretice și tehnologii utilizate: 6-7 pagini;
- Capitolul 3, Date, antrenarea modelelor și metodologia experimentală: 6-7 pagini;
- Capitolul 4, Proiectarea și implementarea sistemului: 7-8 pagini;
- Capitolul 5, Testare, rezultate și interpretare: 6-7 pagini;
- Concluzii: 2-3 pagini;
- Bibliografie: 1-2 pagini.

## Structura curentă a lucrării

Structura LaTeX este stabilită astfel:

1. Copertă conform Anexei 1 din ghid;
2. pagină albă;
3. pagină de titlu conform Anexei 2 din ghid;
4. pagină albă;
5. cuprins;
6. introducere nenumerotată;
7. cinci capitole numerotate;
8. concluzii nenumerotate;
9. bibliografie nenumerotată în stil IEEE.

## Compilare

Se recomandă VS Code + LaTeX Workshop + MiKTeX. Documentul trebuie compilat cu XeLaTeX pentru suport corect Times New Roman și diacritice:

```powershell
cd thesis
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

MiKTeX este instalat local. Pentru generarea PDF-ului rulează:

```powershell
.\build.ps1
```

PDF-ul rezultat apare în `thesis/main.pdf`.
