# Changelog

Notable user-facing changes are summarized here. For the full history, see the [commit log](https://github.com/fang3yuan/comic-translator-app/commits/main).

## 2.2.0

- Added an eraser-model panel for selecting available inpainting engines.
- Added model management options for supported ONNX models.
- Improved text-removal processing, including mask handling and fallback behavior.
- Added Gemini model selection and fallback options where supported.
- Added series context to help maintain consistency across pages.
- Added an optional page-image translation mode to help address OCR errors.
- Improved handling of incomplete model responses.
- Improved the interface with status feedback, scrollable tabs, and pre-translation checks.
- Migrated older model settings.

## 2.0.0

- Standardized translation around Google Gemini.
- Combined detected text blocks into a structured page-level translation request.
- Improved placement of translated text within detected regions.
- Tuned OCR and LaMa execution to reduce unnecessary resource usage.
- Added build scripts and GitHub Actions release automation.
