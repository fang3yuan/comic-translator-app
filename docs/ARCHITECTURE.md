# Architecture (for developers)

[← Home](../README.md) · English technical overview. User guides are in Arabic.

## Pipeline

```
image ─► OCREngine (RapidOCR, downscaled detection) ─► Detector.group_lines / build_regions
      ─► Detector.find_room  (text box grown into verified empty space)
      ─► is_credit_text → TextRegion.ignored
      ├─► GeminiTranslator.translate_items (ONE request per page)       ─┐
      └─► Detector.build_mask → Inpainter (flat → patch → LaMa/OpenCV) ─┴─► Typesetter.render_page
```

Translation (network) and erasing (CPU) run concurrently in `ComicPipeline.process_image`.

## Modules

| Module | Responsibility |
|--------|----------------|
| `src/core/ocr.py` | RapidOCR wrapper, tiling for tall pages, text cleaning, source colour/size estimate |
| `src/core/detector.py` | Line grouping, `find_room` (free-space growth), reading order, ink-only erase mask |
| `src/core/inpainter.py` | Flat fill → texture patch copy → LaMa ONNX (limited threads, GPU if present) / OpenCV |
| `src/core/renderer.py` | Arabic shaping, binary-search rectangle font fitting |
| `src/core/pipeline.py` | Orchestration, `length_budget`, batch export, project save/load |
| `src/translators/gemini.py` | Gemini REST client: keys, throttle, retries, JSON parsing, `check_keys` |
| `src/utils/sysutils.py` | Lower process priority, thread budgets |
| `src/utils/color_utils.py` | Lab colour-distance ink/background separation |
| `src/utils/text_filters.py` | Credit / website / watermark detection |
| `src/config.py` | Settings (versioned), frozen-app paths (`RESOURCE_DIR`, user dir) |
| `src/ui/` | CustomTkinter GUI |
| `comic_translator.spec`, `build*.{bat,sh}`, `.github/workflows/build.yml` | PyInstaller packaging and CI releases |

## Design decisions

- **No bubble detection.** `find_room` samples the background colour (Lab) in a ring around the text; if the ring is flat it grows each side 2 px at a time while ≥ 98.5 % of the new strip matches that colour, symmetrically and with a 4 px safety margin. Textured surroundings are not grown.
- **Gemini protocol.** Items `{id, text, role, max_chars}` + page context + optional user notes in one request; JSON-schema response `{"translations":[{"id","text"}]}`. Missing ids are re-requested; >40 items are chunked.
- **Keys and limits.** Round-robin keys; 429 → per-key cooldown (from `retryDelay`, 30 s default, 1 h for daily quota); invalid keys are dropped; ≥ 1 s between any two requests; 404 → next model; 400 → drop optional fields (thinking, schema).
- **Length budget.** `max(0.7·n + 3, min(suggest_max_chars, fits-in-box))` characters per item.
- **Light on the machine.** LaMa/OCR thread caps, below-normal priority, `enable_mem_pattern=False`.
- **Erasing.** Ink pixels are those on the background→ink colour line in Lab space, so borders and art inside the polygon are kept.

## Settings

`config/settings.json`, versioned (`SETTINGS_VERSION`) with migrations in `Settings`. Key options: `translation.*`, `ocr.max_side`, `ocr.tile_height`, `inpainting.*`, `detector.mask_dilate`, `styles.<role>.*`.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest -q
python -m pyflakes src tests run.py
```

Tests avoid network access; translators are tested with fake HTTP responses.

## Packaging

`pyinstaller comic_translator.spec` builds `dist/ComicTranslator/` (entry `packaging/launcher.py`). LaMa is not bundled; Setup downloads it into the user `models/` folder. CI builds Windows/macOS/Linux and attaches archives to releases on `v*` tags.
