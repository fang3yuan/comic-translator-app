# Changelog

## 2.2.0
- **Eraser models panel:** pick LaMa Manga / LaMa / MI-GAN / OpenCV with one click; install one model only (download or import a `.onnx`), delete it, choose a backup eraser.
- **Smarter erasing:** mask thickness fitted to the paper, plane fill with matching grain for gradients, quality check with automatic rollback to the next method.
- **Gemini:** model picker with backup models and automatic switch when a model's quota ends, series memory (summary, names, last lines), optional page-picture mode that fixes OCR mistakes, self-healing answers (missing blocks keep the English text instead of failing the page).
- **Interface:** status bar, scrollable tabs, Eraser tab, health check before translating with a "fix it" button, every setting reachable from the UI.
- Removed dead code and the old per-model settings (migrated automatically).

## 2.0.0
- Translation is Gemini-only: one or several API keys, one request per page, structured JSON answer, max 1 request/second.
- No bubble detection: the translation goes into the detected text box and grows only into verified empty space.
- LaMa runs light (limited threads, low priority, optional GPU); OCR trimmed.
- One-click builds (`build_windows.bat`, `build.sh`) and GitHub Actions that build and publish releases.
