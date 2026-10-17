<div align="center">

# Comic Translator

**AI-assisted comic and manhwa translation · English to Arabic**

Detect text, clean the original lettering, translate with Google Gemini, and render Arabic text back into the image.

[Download](https://github.com/fang3yuan/comic-translator-app/releases/latest) · [Releases](https://github.com/fang3yuan/comic-translator-app/releases) · [Issues](https://github.com/fang3yuan/comic-translator-app/issues) · [الواجهة العربية](https://github.com/fang3yuan/comic-translator-app/blob/main/README.md)

![License](https://img.shields.io/github/license/fang3yuan/comic-translator-app?style=flat-square)
![Latest Release](https://img.shields.io/github/v/release/fang3yuan/comic-translator-app?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/fang3yuan/comic-translator-app?style=flat-square)

</div>

---

## Overview

**Comic Translator** is a desktop application for translating comic and manhwa pages into Arabic. It helps detect text regions, remove the original lettering while attempting to preserve the artwork, translate the extracted text, and render the Arabic translation back into the image.

The application uses **Google Gemini** for translation and offers multiple text-removal options. Processing time and output quality depend on the source image, selected model, and available hardware.

## Features

- **AI translation:** Translate detected text with Google Gemini and provide additional context such as character names, tone, and terminology.
- **Page-level translation request:** Send detected text blocks together in a structured request instead of issuing a separate request for every block.
- **Text-region rendering:** Place translated text inside detected regions, with expansion when suitable space is available.
- **Style-aware rendering:** Match the original text color and appearance as closely as possible.
- **Multiple erasing options:** Choose among available engines such as LaMa Manga, LaMa, MI-GAN, or OpenCV, depending on installed models and configuration.
- **Gemini key management:** Configure one or multiple API keys and use fallback model options where supported by the current configuration.
- **Series context:** Tools intended to improve consistency of names and dialogue across pages, depending on the current settings.
- **GUI and CLI:** Use the graphical interface or run translation tasks from the command line.
- **Windows packaging:** The repository includes build and installation scripts. A ready-to-run installer is available only when attached to a published release.

## Quick start

### Option 1: Download a release

1. Open the [latest release page](https://github.com/fang3yuan/comic-translator-app/releases/latest).
2. Download the installer if an installer asset is attached to that release.
3. Run it and follow the setup instructions.

> A GitHub Release page does not guarantee that a Windows installer is attached. If no installer is available, use the source setup instructions or check back later.

### Option 2: Use the installation script

Download the project using **Code → Download ZIP** on the [repository page](https://github.com/fang3yuan/comic-translator-app), then run `Install.bat` on Windows if it is included in your downloaded version. Follow the script's prompts and only run files you trust.

### Option 3: Run from source

Requirements and setup steps may vary by release and operating system. With a suitable Python environment, install the dependencies and launch the app:

```bash
pip install -r requirements.txt
python run.py
```

To inspect the available CLI options:

```bash
python run.py cli --help
```

These commands are for running from source. For dependency or model issues, check the [repository](https://github.com/fang3yuan/comic-translator-app) and [open issues](https://github.com/fang3yuan/comic-translator-app/issues).

## Google Gemini setup

A valid API key is required for Gemini translation.

1. Create a key in [Google AI Studio](https://aistudio.google.com/apikey).
2. Enter it in the application's translation section.
3. Use the key-test option if it is available in your version.
4. Review Google's current quotas and terms before translating large batches.

**Keep API keys private.** Never commit personal keys to GitHub or include them in screenshots or shared configuration files.

## Workflow

```text
Comic page
    ↓
Text-region detection and OCR
    ↓
Remove original lettering with the selected erasing engine
    ↓
Translate extracted text with Gemini
    ↓
Render Arabic text into the image
    ↓
Save the translated result
```

Some pages may require manual review. Small text, decorative lettering, and complex backgrounds can affect detection, inpainting, and translation quality.

## Performance and privacy

- Text removal can take longer on low-memory computers, particularly when local models are used.
- Gemini translation requires an internet connection and sends the data needed for translation to Google's service.
- Do not submit private or sensitive pages to third-party services unless you understand their privacy policies and terms.
- Image cleanup is not guaranteed to be perfect. Review the output before sharing or publishing it.

## Troubleshooting

- **Translation fails:** Check the API key, internet connection, and Gemini usage limits.
- **Text removal is slow:** Try a lighter available erasing engine, close other applications, or reduce image dimensions.
- **Text is missed or translated incorrectly:** Try a clearer image and review the detected regions and final output.
- **Installation or startup fails:** Include your operating system, installation method, reproduction steps, and error message when [opening an issue](https://github.com/fang3yuan/comic-translator-app/issues).

## Contributing

Bug reports and contributions are welcome. Do not include API keys, credentials, or personal data in issues.

## License

The project is distributed under the MIT License. See the repository's [LICENSE file](https://github.com/fang3yuan/comic-translator-app/blob/main/LICENSE).

---

<div align="center">

**Comic Translator** · Open-source project

</div>
