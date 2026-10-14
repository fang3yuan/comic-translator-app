#!/usr/bin/env bash
# One-command build for macOS / Linux:  ./build.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
[ -d .venv-build ] || "$PY" -m venv .venv-build
. .venv-build/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller comic_translator.spec --noconfirm --clean
OS="$(uname -s | tr A-Z a-z)"
( cd dist && tar -czf "ComicTranslator-$OS.tar.gz" ComicTranslator )
echo "Done: dist/ComicTranslator/ComicTranslator  (archive: dist/ComicTranslator-$OS.tar.gz)"
