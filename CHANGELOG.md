# Changelog

Notable project changes are documented here. Entries summarize user-facing changes and do not replace the Git history.

## [2.2.0]

- Added an eraser-model panel for selecting LaMa Manga, LaMa, MI-GAN, or OpenCV.
- Added model management options, including downloading or importing ONNX models, removing installed models, and selecting a fallback eraser.
- Improved text-removal processing with mask sizing, background fill handling, quality checks, and fallback behavior.
- Added Gemini model selection and fallback options when a model reaches its quota.
- Added series context to help maintain consistency of summaries, names, and recent dialogue.
- Added an optional page-image translation mode to help address OCR errors.
- Improved handling of incomplete model responses so missing blocks do not automatically fail the entire page.
- Improved the interface with a status bar, scrollable tabs, an eraser tab, and a pre-translation health check.
- Removed obsolete code and migrated older model settings automatically.

## [2.0.0]

- Standardized translation around Google Gemini.
- Combined detected text blocks into a structured page-level translation request.
- Improved placement of translated text within detected text regions.
- Reduced OCR overhead and tuned LaMa execution to limit resource usage.
- Added one-click build scripts and GitHub Actions release automation.

---

For earlier changes, see the repository's [commit history](https://github.com/fang3yuan/comic-translator-app/commits/main).
