"""One-command setup and launch for HealthPredict.

    Windows:      double-click start.bat        (or:  py start.py)
    macOS/Linux:  ./start.sh                    (or:  python3 start.py)

What it does (each step is skipped when already done):
  1. checks for Python 3.10 - 3.12
  2. creates a virtual environment in .venv and installs requirements.txt
  3. repairs SQLAlchemy if Windows Application Control blocks its compiled DLL
  4. trains the ML models if they are missing or cannot be loaded (~8 minutes, first run only)
  5. creates the demo accounts
  6. starts the web app and opens it in your browser

Options:  --port 5000   --lan (allow other devices on the network)   --retrain   --no-browser
Uses only the Python standard library, so it runs before anything is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VPY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
# Portable build: a bundled Python runtime with all packages pre-installed in ./python
PORTABLE_PY = ROOT / "python" / "python.exe"
PORTABLE = os.name == "nt" and PORTABLE_PY.exists()
if PORTABLE:
    VPY = PORTABLE_PY
    
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONHOME", None)
    os.environ.pop("PYTHONPATH", None)
REQ = ROOT / "requirements.txt"
MARK = VENV / ".requirements.sha256"


def say(msg: str, kind: str = "*") -> None:
    print(f"\n[{kind}] {msg}", flush=True)


def run(cmd, **kw) -> int:
    return subprocess.call([str(c) for c in cmd], cwd=ROOT, **kw)


def ok(cmd, env=None) -> tuple[bool, str]:
    p = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, env=env)
    return p.returncode == 0, (p.stdout + p.stderr)


def check_python() -> None:
    v = sys.version_info
    if not ((3, 10) <= (v.major, v.minor) <= (3, 12)):
        say(f"This project needs Python 3.10, 3.11 or 3.12 (you are running {v.major}.{v.minor}).", "!")
        print("    Install Python 3.12 from https://www.python.org/downloads/release/python-3128/\n"
              "    (tick 'Add python.exe to PATH'), then run start.bat / start.sh again.")
        sys.exit(1)


def ensure_venv() -> None:
    if VPY.exists():
        return
    say("Creating virtual environment (.venv) ...")
    if run([sys.executable, "-m", "venv", VENV]) != 0 or not VPY.exists():
        say("Could not create the virtual environment.", "!")
        if sys.platform.startswith("linux"):
            print("    On Debian/Ubuntu run:  sudo apt install python3-venv")
        sys.exit(1)


def ensure_requirements() -> None:
    digest = hashlib.sha256(REQ.read_bytes()).hexdigest()
    if MARK.exists() and MARK.read_text().strip() == digest:
        return
    say("Installing Python packages (first run only, a few minutes) ...")
    run([VPY, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    if run([VPY, "-m", "pip", "install", "-r", REQ]) != 0:
        say("Package installation failed - check your internet connection and try again.", "!")
        sys.exit(1)
    MARK.write_text(digest)


def repair_blocked_dlls() -> None:
    """Some locked-down Windows PCs block SQLAlchemy's optional compiled speed-ups."""
    good, out = ok([VPY, "-c", "import sqlalchemy.orm"])
    if good:
        return
    if "DLL load failed" not in out and "Application Control" not in out:
        say("SQLAlchemy failed to import:\n" + out, "!")
        sys.exit(1)
    say("Windows blocked a compiled SQLAlchemy file - installing the pure-Python version ...")
    ver = next((l.split("==")[1].strip() for l in REQ.read_text().splitlines()
                if l.lower().startswith("sqlalchemy==")), None)
    env = dict(os.environ, DISABLE_SQLALCHEMY_CEXT="1")
    pkg = f"sqlalchemy=={ver}" if ver else "sqlalchemy"
    subprocess.call([str(VPY), "-m", "pip", "install", "--force-reinstall", "--no-deps", "--no-binary",
                     "sqlalchemy", pkg], cwd=ROOT, env=env)
    good, out = ok([VPY, "-c", "import sqlalchemy.orm"])
    if not good:
        say("SQLAlchemy still cannot be imported:\n" + out, "!")
        sys.exit(1)


def ensure_models(force: bool) -> None:
    probe = ("from ml.predict import Registry; r = Registry().load(); "
             "import sys; print(r.errors); sys.exit(1 if r.errors else 0)")
    if not force:
        good, out = ok([VPY, "-c", probe])
        if good:
            return
        say("Models are missing or could not be loaded:\n    " + out.strip().splitlines()[-1])
    say("Training the machine-learning models (about 6-10 minutes, only needed once) ...")
    if run([VPY, "-m", "ml.train_all", "--force"]) != 0:
        say("Model training failed - see the messages above.", "!")
        sys.exit(1)
    run([VPY, "-m", "ml.figures"])


def free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "your-ip"


def main() -> None:
    ap = argparse.ArgumentParser(description="Set up and start HealthPredict")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--lan", action="store_true", help="listen on all network interfaces")
    ap.add_argument("--retrain", action="store_true", help="retrain all models before starting")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    print("=" * 64 + "\n  HealthPredict - Mental Health, Brain Stroke & Diabetes\n" + "=" * 64)
    if PORTABLE:
        say("Portable build: using the bundled Python runtime (nothing to install).")
    else:
        check_python()
        ensure_venv()
        ensure_requirements()
        repair_blocked_dlls()
    ensure_models(args.retrain)
    run([VPY, "-m", "flask", "--app", "run", "seed-demo"])

    port = free_port(args.port)
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 64)
    print(f"  HealthPredict is starting at  {url}")
    if args.lan:
        print(f"  Other devices on your network:  http://{lan_ip()}:{port}")
    print("\n  Demo logins")
    print("    Patient : ganesh@example.com          / ganesh12345")
    print("    Doctor  : doctor@healthpredict.local  / doctor12345")
    print("    Admin   : admin@healthpredict.local   / admin12345")
    print("\n  Sample inputs to try: test_inputs.txt")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 64 + "\n", flush=True)

    if not args.no_browser:
        threading.Timer(3.0, lambda: webbrowser.open(url)).start()
    env = dict(os.environ, PORT=str(port), HOST=host)
    try:
        subprocess.call([str(VPY), "run.py"], cwd=ROOT, env=env)
    except KeyboardInterrupt:
        pass
    print("\nServer stopped. Run start again any time.")


if __name__ == "__main__":
    main()
