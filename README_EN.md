# Comic Translator (EN → AR)

Local manga/comic page translator: OCR → erase original text (texture preserved) → **Gemini** translation → Arabic typesetting.

## Highlights
- **Gemini only.** One or many API keys, **one request per page** (all detected texts at once, structured JSON answer), at most 1 request/second.
- **No bubble detection.** Arabic goes into the detected text box; the box grows only into verified empty, uniformly coloured space, so text never leaves the bubble.
- **Length hints.** Every item is sent with a `max_chars` budget derived from its box so the font does not shrink to nothing.
- **Custom notes** (names, tone, glossary) are added to the prompt.
- **Light LaMa.** Limited threads, lowered process priority, GPU used if available; falls back to OpenCV.
- **One-click build** (`build_windows.bat` / `build.sh`) and GitHub Actions that build and publish releases.

## Quick start
```bash
pip install -r requirements.txt
python run.py                 # GUI  (Translation tab → paste key → Test my keys)
python run.py cli pages/ -o out --key AIza...
```
Get a free key: https://aistudio.google.com/apikey — see [docs/GEMINI.md](docs/GEMINI.md).
Build the app: see [docs/BUILD.md](docs/BUILD.md). Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). License: MIT.
