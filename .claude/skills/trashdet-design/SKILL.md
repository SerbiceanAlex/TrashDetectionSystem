---
name: trashdet-design
description: Convențiile de design/UI ale aplicației TrashDet — folosește-le ori de câte ori lucrezi la frontend (HTML/CSS/Alpine): culori, carduri, badge-uri de status, etichete material, select-uri, chip-uri sursă, afișare SHA. Invocă la „design", „UI", „stilizează", „card incident", „badge", „temă".
---

# Sistem de design — TrashDet

Aplicație web (FastAPI + Alpine.js + TailItem utility classes + CSS variables), temă **dark/light** cu accent **emerald**. Frontend în `frontend/templates/tabs/*.html`, stiluri în `frontend/static/css/style.css`.

**Aplică ACESTE convenții la orice modificare de UI — nu inventa culori/pattern-uri noi.**

## Variabile CSS (folosește-le, NU culori hardcodate pentru fundal/text)
- Fundaluri: `var(--bg-surface)` (card), `var(--bg-raised)` (interior/input), `var(--bg-border)` (chenar)
- Text: `var(--tx-primary)` (titlu), `var(--tx-secondary)` (normal), `var(--tx-muted)` (secundar/gri)
- Dark/light se comută prin `darkMode` (Alpine) — culorile de accent se aleg adesea `darkMode? '#…' : '#…'`.

## Accente pe zone
- **Incidente:** roșu (`#ef4444`) — hero, alerte
- **Admin:** indigo/violet (`#4f46e5`, `#6366f1`, `#818cf8`)
- **Brand/confirmare:** emerald (`#10b981`, `#34d399`)
- Focus pe input-uri/select-uri: border `#6366f1` + glow `0 0 0 3px rgba(99,102,241,.15)` (NU roșu).

## Carduri
- `rounded-2xl`, `background:var(--bg-surface)`, `border:1px solid var(--bg-border)`.
- **Bară de status pe jos** (3px gradient) — verde=confirmat, galben=așteptare, albastru=arhivat, gri=fals:
  `linear-gradient(90deg, <culoare>, transparent)` cu
  `reviewed→#10b981 · pending→#f59e0b · forwarded→#3b82f6 · else→#6b7280`.

## Badge-uri de STATUS incident
- `pending` → **„În așteptare"** (amber `#f59e0b`)
- `reviewed` → **„Confirmat"** (green `#10b981`)
- `forwarded` → **„Arhivat"** (blue `#3b82f6`)
- `dismissed` → **„Fals pozitiv"** (gray `#6b7280`)

## Badge-uri de MATERIAL
- plastic→**Plastic** (albastru) · paper→**Hârtie** (amber) · glass→**Sticlă** (cyan) · metal→**Metal** (violet) · other→**Altele** · necunoscut→**„Necunoscut"** (NU „Neconfirmat"/„Necesită validare"; „Altele" e categorie reală, „Necunoscut" = clasificator nesigur).

## Chip de SURSĂ incident (câmpul `ev.source`)
- `upload` → iconă `upload` + **„Video încărcat"**
- altfel → iconă `radio` + **„Monitor live"**

## Select-uri (dropdown)
- Adaugă clasa **`.filter-select`** → chevron custom (ascunde săgeata nativă boxată). Definită în `style.css`.

## Integritate / dovezi
- Afișează **SHA-256** (`ev.image_hash`) cu iconă `shield-check` verde + font-mono trunchiat (full pe `title`). E atuul de „chain of custody".
- ID-uri incident **stabile, cu goluri** (NU renumerota — sunt legate de fișiere/hash/watermark). Numerotare secvențială DOAR pentru liste fără dovezi (ex. tabel Utilizatori).

## Limbă & ton
- Tot UI-ul în **română**. Etichete clare, fără jargon (ex. „Necunoscut", nu „unknown").
- Liste de procesări/upload: arată **ora încărcării**, nu „obj/fr" tehnice.

## Reguli generale
- Iconițe: **lucide** (`data-lucide="…"`). În `<template x-if>` merg static; pentru dinamice preferă două template-uri x-if (lucide.createIcons re-rulează).
- După modificări de **template/CSS**: utilizatorul trebuie **hard-refresh** (Ctrl+Shift+R) — cache de browser.
- Validează HTML/JS: `node --check frontend/static/js/<f>.js` pentru JS.
