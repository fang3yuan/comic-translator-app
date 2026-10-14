#!/usr/bin/env bash
# Comic Translator - install (first time) and run on macOS / Linux:  bash start.sh
set -e
cd "$(dirname "$0")"

PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY="$cand"; break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (or your package manager)."
  exit 1
fi

if ! "$PY" -c 'import tkinter' 2>/dev/null; then
  echo "Python has no Tk. Install it first:"
  echo "  Ubuntu/Debian: sudo apt install python3-tk"
  echo "  Fedora:        sudo dnf install python3-tkinter"
  echo "  macOS (brew):  brew install python-tk"
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating private Python environment..."
  "$PY" -m venv .venv
fi

if [ ! -f ".venv/.installed" ]; then
  echo "Installing required packages (first run only, 2-5 minutes)..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
  echo installed > .venv/.installed
fi

exec .venv/bin/python run.py "$@"
