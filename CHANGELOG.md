# Changelog

## 2.0.0
- Translation is Gemini-only: one or several API keys, one request per page, structured JSON answer, max 1 request/second.
- No bubble detection: the translation goes into the detected text box and grows only into verified empty space.
- LaMa runs light (limited threads, low priority, optional GPU); OCR trimmed.
- One-click builds (`build_windows.bat`, `build.sh`) and GitHub Actions that build and publish releases.
