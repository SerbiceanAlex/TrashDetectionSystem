#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
start_https.py -- Porneste TrashDetectionSystem cu HTTPS pe reteaua locala.

Folosire:
    .venv\\Scripts\\python.exe start_https.py

Accesare de pe iPhone (aceeasi retea Wi-Fi):
    https://<IP-ul-tau>:8443

NOTA: Browserul va afisa avertisment 'certificat neincredzut'.
Apasa 'Avansat' -> 'Viziteaza site-ul' pentru a continua.
"""

import datetime
import os
import socket
import subprocess
import sys
from pathlib import Path

# -- Determina IP-ul local ---------------------------------------------------
def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

LOCAL_IP = get_local_ip()
CERT_DIR = Path(__file__).parent / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE  = CERT_DIR / "key.pem"


def generate_self_signed_cert():
    """Genereaza certificat self-signed cu cryptography sau openssl CLI."""
    CERT_DIR.mkdir(exist_ok=True)

    if CERT_FILE.exists() and KEY_FILE.exists():
        print("[HTTPS] Certificat existent: " + str(CERT_FILE))
        return

    print("[HTTPS] Generez certificat SSL self-signed...")

    # Incearca cu cryptography (Python-nativ, nu necesita openssl CLI)
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, LOCAL_IP),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "TrashDetectionSystem"),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                    x509.IPAddress(ipaddress.ip_address(LOCAL_IP)),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        KEY_FILE.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
        print("[HTTPS] Certificat generat cu cryptography -> " + str(CERT_FILE))
        return

    except ImportError:
        pass  # fallback la openssl CLI

    # Fallback: openssl CLI
    try:
        subj = "/CN=" + LOCAL_IP + "/O=TrashDetectionSystem"
        san  = "subjectAltName=IP:" + LOCAL_IP + ",IP:127.0.0.1,DNS:localhost"
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(KEY_FILE),
            "-out", str(CERT_FILE),
            "-days", "365", "-nodes",
            "-subj", subj,
            "-addext", san,
        ], check=True, capture_output=True)
        print("[HTTPS] Certificat generat cu openssl CLI -> " + str(CERT_FILE))
        return

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print("[HTTPS] EROARE: Nu pot genera certificat.")
    print("  Ruleaza: .venv\\Scripts\\python.exe -m pip install cryptography")
    sys.exit(1)


if __name__ == "__main__":
    generate_self_signed_cert()

    HOST = "0.0.0.0"
    PORT = 8443

    print()
    print("=" * 60)
    print("  TrashDetectionSystem -- HTTPS Server")
    print("=" * 60)
    print("  Local:   https://localhost:" + str(PORT))
    print("  iPhone:  https://" + LOCAL_IP + ":" + str(PORT))
    print()
    print("  IMPORTANT -- acceptare certificat pe iPhone:")
    print("  1. Deschide Safari -> https://" + LOCAL_IP + ":" + str(PORT))
    print("  2. Apasa 'Avansat' -> 'Viziteaza site-ul'")
    print("  3. Settings -> General -> About -> Certificate Trust Settings")
    print("     -> Activeaza certificatul TrashDetectionSystem")
    print("=" * 60)
    print()

    # Porneste uvicorn cu SSL
    venv_python = Path(sys.executable)
    cmd = [
        str(venv_python), "-m", "uvicorn",
        "backend.main:app",
        "--host", HOST,
        "--port", str(PORT),
        "--ssl-keyfile", str(KEY_FILE),
        "--ssl-certfile", str(CERT_FILE),
        "--reload",
        "--reload-dir", "backend",
        "--reload-dir", "frontend",
    ]
    print("[CMD] " + " ".join(cmd))
    print()
    os.execv(str(venv_python), cmd)
