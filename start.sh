#!/usr/bin/env bash
# HealthPredict - one-command setup and launch (macOS / Linux)
cd "$(dirname "$0")" || exit 1
for py in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null; then
    exec "$py" start.py "$@"
  fi
done
echo "[!] Python 3.10, 3.11 or 3.12 was not found."
echo "    macOS:  brew install python@3.12     Ubuntu/Debian:  sudo apt install python3.12 python3.12-venv"
exit 1
