"""
Genereaza capturi de ecran reale din aplicatie pentru lucrare (Selenium + Edge headless).

Cerinte: serverul pornit local (implicit http://127.0.0.1:8020) cu baza de date reala.
Creeaza un cont admin demo (admin_demo) daca nu exista, face login in UI si salveaza:
  - thesis/capturi_aplicatie/09_dashboard_incidente.png  (Dashboard cu statistici si incidente)
  - thesis/capturi_aplicatie/10_lista_incidente.png      (Tab-ul Incidente cu lista plina)

Rulare:
    .venv\\Scripts\\python.exe scripts\\presentation\\capture_ui_screenshots.py
"""

from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import httpx
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8020"
OUT = ROOT / "thesis" / "capturi_aplicatie"
DB = ROOT / "data" / "trash_detection.db"
USER = {"username": "admin_demo", "email": "admin_demo@local.test", "password": "Demo!Parola9"}


def ensure_admin() -> None:
    with httpx.Client(timeout=30) as c:
        for _ in range(60):
            try:
                if c.get(f"{BASE}/api/system/info").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            sys.exit("Serverul nu raspunde pe " + BASE)
        c.post(f"{BASE}/api/auth/register", json=USER)  # 400 daca exista deja - ok
    con = sqlite3.connect(DB)
    con.execute("UPDATE users SET role='admin', organization_id=1 WHERE username=?", (USER["username"],))
    con.commit()
    con.close()
    print("cont admin_demo pregatit")


def main() -> None:
    ensure_admin()
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1600,1000")
    opts.add_argument("--force-device-scale-factor=1")
    drv = webdriver.Edge(options=opts)
    try:
        drv.get(f"{BASE}/app")
        time.sleep(4)

        # Deschide modalul de login si autentifica-te
        drv.execute_script("""
            const root = document.querySelector('[x-data]');
            const app = Alpine.$data(root);
            app.openAuth('login');
        """)
        time.sleep(1)
        drv.execute_script(f"""
            const app = Alpine.$data(document.querySelector('[x-data]'));
            app.loginData.username = '{USER["username"]}';
            app.loginData.password = '{USER["password"]}';
            app.login();
        """)
        time.sleep(6)  # login + loadDashboard + grafice

        OUT.mkdir(parents=True, exist_ok=True)
        drv.save_screenshot(str(OUT / "09_dashboard_incidente.png"))
        print("salvat 09_dashboard_incidente.png")

        drv.execute_script("Alpine.$data(document.querySelector('[x-data]')).goTo('incidents');")
        time.sleep(5)  # incarcare lista + thumbnails
        drv.save_screenshot(str(OUT / "10_lista_incidente.png"))
        print("salvat 10_lista_incidente.png")
    finally:
        drv.quit()


if __name__ == "__main__":
    main()
