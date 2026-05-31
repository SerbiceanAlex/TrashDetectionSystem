#!/usr/bin/env python3
"""Start TrashDet over HTTPS on the local network.

This is the comfortable launcher for the thesis/demo setup:

    .venv\\Scripts\\python.exe start_https.py
    .venv\\Scripts\\python.exe start_https.py --restart

It prints the desktop/mobile URLs, sets APP_BASE_URL for this process, and
prevents the common WinError 10048 by detecting an already-running server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
import socket
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


REPO = Path(__file__).resolve().parent
CERT_DIR = REPO / "certs"
CERT_FILE = CERT_DIR / "cert.pem"
KEY_FILE = CERT_DIR / "key.pem"
DEFAULT_PORT = 8443


class C:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start TrashDet HTTPS server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host. Keep 0.0.0.0 for LAN/mobile access.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"HTTPS port. Default: {DEFAULT_PORT}.")
    parser.add_argument("--ip", default="", help="Advertised LAN IP. Auto-detected when omitted.")
    parser.add_argument("--base-url", default="", help="Override APP_BASE_URL for this run.")
    parser.add_argument("--restart", action="store_true", help="Stop the existing TrashDet server on this port first.")
    parser.add_argument("--stop", action="store_true", help="Stop the existing TrashDet server on this port and exit.")
    parser.add_argument("--reuse", action="store_true", help="Reuse an already-running server instead of restarting it here.")
    parser.add_argument("--auto-port", action="store_true", help="Use the next free port if --port is occupied. Optional fallback only.")
    parser.add_argument("--open", dest="open_browser", action="store_true", default=True, help="Open the app URL in the default browser.")
    parser.add_argument("--no-open", dest="open_browser", action="store_false", help="Do not open the browser automatically.")
    parser.add_argument("--open-when-ready", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reload", dest="reload", action="store_true", default=True, help="Enable uvicorn reload.")
    parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable uvicorn reload.")
    return parser.parse_args()


def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def cert_matches_lan_ip(lan_ip: str) -> bool:
    """Return True when the existing local cert is still valid for this LAN IP."""
    if not CERT_FILE.exists() or not KEY_FILE.exists():
        return False

    try:
        decoded = ssl._ssl._test_decode_cert(str(CERT_FILE))
        san_values = decoded.get("subjectAltName", [])
        names = {(kind, value) for kind, value in san_values}

        not_after = decoded.get("notAfter")
        if not_after:
            expires = dt.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=dt.timezone.utc)
            if expires <= dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1):
                return False

        return (
            ("DNS", "localhost") in names
            and ("IP Address", "127.0.0.1") in names
            and ("IP Address", lan_ip) in names
        )
    except Exception:
        return False


def generate_self_signed_cert(lan_ip: str) -> None:
    CERT_DIR.mkdir(exist_ok=True)
    if cert_matches_lan_ip(lan_ip):
        return
    if CERT_FILE.exists() or KEY_FILE.exists():
        print(f"{C.YELLOW}[HTTPS]{C.RESET} Regenerating certificate for LAN IP {lan_ip}...")
        CERT_FILE.unlink(missing_ok=True)
        KEY_FILE.unlink(missing_ok=True)

    print("[HTTPS] Generating self-signed certificate...")
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print(f"{C.RED}[ERROR]{C.RESET} Missing dependency: cryptography")
        print(f"        Install it with: {C.CYAN}.venv\\Scripts\\python.exe -m pip install cryptography{C.RESET}")
        raise SystemExit(1)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "TrashDet"),
        ]
    )
    now = dt.datetime.now(dt.timezone.utc)
    san_values = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address(lan_ip)),
    ]
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + dt.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_values), critical=False)
        .sign(key, hashes.SHA256())
    )

    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    KEY_FILE.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )


def listening_pids(port: int) -> set[int]:
    """Return Windows PIDs listening on port using netstat."""
    if os.name != "nt":
        return set()
    proc = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: set[int] = set()
    needle = f":{port}"
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[1].endswith(needle):
            if parts[3].upper() == "LISTENING":
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    pass
    return pids


def find_free_port(start_port: int, attempts: int = 10) -> int:
    for port in range(start_port, start_port + attempts):
        if not listening_pids(port):
            return port
    raise RuntimeError(f"No free port found in range {start_port}-{start_port + attempts - 1}")


def process_command_line(pid: int) -> str:
    if os.name != "nt":
        return ""
    command = (
        "(Get-CimInstance Win32_Process -Filter "
        f"\"ProcessId={pid}\").CommandLine"
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    return (proc.stdout or "").strip()


def is_trashdet_process(pid: int) -> bool:
    cmd = process_command_line(pid).lower()
    return "uvicorn" in cmd and "backend.main:app" in cmd


def stop_existing_server(port: int, *, force: bool = False) -> None:
    pids = listening_pids(port)
    if not pids:
        return

    for pid in sorted(pids):
        if not is_trashdet_process(pid):
            if not force:
                print(f"{C.RED}[ERROR]{C.RESET} Port {port} is used by another process (PID {pid}).")
                print("        Close that app or choose another port.")
                raise SystemExit(1)
            print(f"{C.YELLOW}[SERVER]{C.RESET} Trying to stop process on port {port} (PID {pid})...")
        else:
            print(f"{C.YELLOW}[SERVER]{C.RESET} Stopping existing TrashDet server (PID {pid})...")
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
    time.sleep(1.0)


def server_is_healthy(port: int) -> bool:
    url = f"https://127.0.0.1:{port}/api/system/info"
    ctx = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=3) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_server(port: int, timeout_sec: float = 90.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if server_is_healthy(port):
            return True
        time.sleep(0.35)
    return False


def print_urls(lan_ip: str, port: int, base_url: str) -> None:
    app_url = f"https://{lan_ip}:{port}/app"
    local_url = f"https://localhost:{port}/app"
    health_url = f"https://{lan_ip}:{port}/api/system/info"
    print()
    print(f"{C.GREEN}{'=' * 72}{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}  TrashDet HTTPS server is ready{C.RESET}")
    print(f"{C.GREEN}{'=' * 72}{C.RESET}")
    print(f"  Desktop local : {C.CYAN}{local_url}{C.RESET}")
    print(f"  Desktop LAN   : {C.CYAN}{app_url}{C.RESET}")
    print(f"  Mobile        : {C.CYAN}{app_url}{C.RESET}")
    print(f"  API health    : {C.CYAN}{health_url}{C.RESET}")
    print(f"  APP_BASE_URL  : {C.CYAN}{base_url}{C.RESET}")
    print()
    print(f"  {C.YELLOW}Browser warning is expected with the local self-signed certificate.{C.RESET}")
    print("  Chrome/Opera: Advanced/Help me understand -> Proceed.")
    print("  If no Proceed button appears, type thisisunsafe on that warning page.")
    print("  For phone camera access, use HTTPS and the same Wi-Fi network.")
    print(f"{C.GREEN}{'=' * 72}{C.RESET}")
    print()


def open_browser(lan_ip: str, port: int) -> None:
    url = f"https://{lan_ip}:{port}/app"
    print(f"{C.GREEN}[BROWSER]{C.RESET} Opening {C.CYAN}{url}{C.RESET}")
    webbrowser.open(url)


def launch_browser_when_ready(lan_ip: str, port: int) -> None:
    cmd = [
        str(Path(sys.executable)),
        str(Path(__file__).resolve()),
        "--open-when-ready",
        "--ip",
        lan_ip,
        "--port",
        str(port),
    ]
    kwargs = {
        "cwd": str(REPO),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(cmd, **kwargs)
    print(f"{C.GREEN}[BROWSER]{C.RESET} Browser will open after the server is ready.")


def main() -> int:
    args = parse_args()
    lan_ip = args.ip.strip() or get_local_ip()

    generate_self_signed_cert(lan_ip)

    if args.open_when_ready:
        if wait_for_server(args.port, timeout_sec=90.0):
            open_browser(lan_ip, args.port)
            return 0
        return 1

    if args.stop:
        stop_existing_server(args.port, force=True)
        print(f"{C.GREEN}[OK]{C.RESET} No TrashDet server is listening on port {args.port}.")
        return 0

    if args.restart:
        stop_existing_server(args.port, force=True)
        if listening_pids(args.port):
            if not args.auto_port:
                print(f"{C.RED}[ERROR]{C.RESET} Port {args.port} is still occupied after restart attempt.")
                print(f"        Best local-demo fallback:")
                print(f"        {C.CYAN}.venv\\Scripts\\python.exe start_https.py --port 9444{C.RESET}")
                return 1
            old_port = args.port
            args.port = find_free_port(args.port + 1)
            print(f"{C.YELLOW}[PORT]{C.RESET} {old_port} is still busy, using {args.port} instead.")
    elif listening_pids(args.port):
        if server_is_healthy(args.port):
            base_url = args.base_url.strip() or f"https://{lan_ip}:{args.port}"
            if args.reuse:
                print_urls(lan_ip, args.port, base_url)
                print(f"{C.GREEN}[OK]{C.RESET} Server already running on port {args.port}.")
                print("     This terminal is not attached to its logs.")
                print("     To run it here and stop with Ctrl+C, use:")
                print(f"     {C.CYAN}.venv\\Scripts\\python.exe start_https.py --restart --port {args.port}{C.RESET}")
                if args.open_browser:
                    open_browser(lan_ip, args.port)
                return 0

            print(f"{C.YELLOW}[SERVER]{C.RESET} Server already runs on port {args.port}.")
            print(f"         Restarting it in this terminal so Ctrl+C stops it and logs stay visible.")
            stop_existing_server(args.port, force=True)
            if listening_pids(args.port):
                print(f"{C.RED}[ERROR]{C.RESET} Port {args.port} is still occupied after restart attempt.")
                print(f"        Use fallback: {C.CYAN}.venv\\Scripts\\python.exe start_https.py --port 9444{C.RESET}")
                return 1
        if args.auto_port:
            old_port = args.port
            args.port = find_free_port(args.port + 1)
            print(f"{C.YELLOW}[PORT]{C.RESET} {old_port} is occupied, using {args.port} instead.")
        else:
            print(f"{C.RED}[ERROR]{C.RESET} Port {args.port} is already occupied.")
            print(f"        Use a stable fallback: {C.CYAN}.venv\\Scripts\\python.exe start_https.py --port 9444{C.RESET}")
            return 1

    base_url = args.base_url.strip() or f"https://{lan_ip}:{args.port}"

    os.environ["APP_BASE_URL"] = base_url

    print_urls(lan_ip, args.port, base_url)
    if args.open_browser:
        launch_browser_when_ready(lan_ip, args.port)
    cmd = [
        str(Path(sys.executable)),
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ssl-keyfile",
        str(KEY_FILE),
        "--ssl-certfile",
        str(CERT_FILE),
    ]
    if args.reload:
        cmd.extend(["--reload", "--reload-dir", "backend", "--reload-dir", "frontend"])

    print(f"{C.GREEN}[CMD]{C.RESET} " + " ".join(cmd))
    print()
    os.execv(str(Path(sys.executable)), cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
