# Licență LaTeX

Titlu: **Sistem inteligent de detecție a aruncării ilegale de deșeuri în spații publice prin analiza video în timp real**

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

Fișierul `detalii de scriere.doc` pare orientat către un proiect/articol scurt de 4-10 pagini, deci nu îl folosim ca șablon principal pentru licență.

Analiza detaliată este în `ANALIZA_FISIERE.md`.

## Buget recomandat

Ținta noastră este o lucrare compactă, nu una umflată artificial: aproximativ **30-40 pagini de conținut redactat** și aproximativ **40-45 pagini în PDF-ul final**, după includerea copertei, paginilor albe, cuprinsului, bibliografiei, figurilor și tabelelor.

Astfel respectăm ghidul oficial, care recomandă 40-70 pagini, dar ne orientăm spre limita inferioară.

- Introducere: 3-4 pagini;
- Capitolul 1: 6-7 pagini;
- Capitolul 2: 6-7 pagini;
- Capitolul 3: 6-7 pagini;
- Capitolul 4: 4-5 pagini;
- Capitolul 5: 6-7 pagini;
- Concluzii: 2-3 pagini;
- Bibliografie: 1-2 pagini.

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
