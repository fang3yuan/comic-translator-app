# Model weights

| File | Used by | How to get it |
|------|---------|---------------|
| `lama_fp32.onnx` | LaMa inpainting (optional) | Setup window, or `python run.py cli --download-models` |

Without `lama_fp32.onnx` the app falls back to OpenCV inpainting. The OCR models ship inside the `rapidocr` package.
