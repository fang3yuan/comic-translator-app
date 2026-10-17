<div align="center">

# Comic Translator

**AI-assisted comic and manhwa translation · English to Arabic**

Detect text, remove original lettering while preserving artwork where possible, translate with Google Gemini, and render Arabic text into the image.

[Download](https://github.com/fang3yuan/comic-translator-app/releases/latest) · [Releases](https://github.com/fang3yuan/comic-translator-app/releases) · [Issues](https://github.com/fang3yuan/comic-translator-app/issues) · [الواجهة العربية](https://github.com/fang3yuan/comic-translator-app/blob/main/README.md)

![License](https://img.shields.io/github/license/fang3yuan/comic-translator-app?style=flat-square)
![Latest Release](https://img.shields.io/github/v/release/fang3yuan/comic-translator-app?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/fang3yuan/comic-translator-app?style=flat-square)

</div>

---

## Before and after

<div align="center">

![Comic translation before and after](docs/images/00-before-after-demo.png)

<sub>Before-and-after example included in the project.</sub>

</div>

## Overview

**Comic Translator** is a desktop application for translating comic and manhwa pages into Arabic. It reads detected text, removes the original lettering, sends the text to Google Gemini for translation, and renders the Arabic text back into the image.

## Features

- **Gemini translation:** Translate detected text with additional instructions such as character names, tone, and terminology.
- **Structured page translation:** Process detected text blocks in a structured translation request.
- **Text-region rendering:** Place translated text in detected regions while accounting for available space.
- **Text color matching:** Attempt to preserve the original text color when rendering translations.
- **Text-removal engines:** Choose from available options, including LaMa Manga, LaMa, and MI-GAN, depending on installed models and configuration.
- **Gemini key settings:** Configure API keys using the options available in the app.
- **GUI and CLI:** Use the graphical interface or command-line interface.
- **Windows packaging:** Build and installation files are included in the repository; check Releases for currently available assets.

## Quick start

### Download a release

1. Open the [latest release page](https://github.com/fang3yuan/comic-translator-app/releases/latest).
2. Download the installer if one is attached to the release.
3. Run the installer and follow its instructions.

### Install from repository files

Download the repository using **Code → Download ZIP** on the [project page](https://github.com/fang3yuan/comic-translator-app). If `Install.bat` is included in the downloaded version, run it on Windows and follow its prompts.

### Run from source

See the [installation and build guide](https://github.com/fang3yuan/comic-translator-app/blob/main/docs/INSTALL.md) for instructions matching the current version.

## Application interface

<div align="center">

![Comic Translator main window](docs/images/01-main-window.png)

<sub>The main Comic Translator interface.</sub>

</div>

## Workflow

1. **Add pages:** Drag image files into the application.
2. **Detect and read text:** Identify text regions and extract their contents.
3. **Remove original lettering:** Process text regions with the selected erasing engine.
4. **Translate:** Send the text to Gemini with your instructions.
5. **Render:** Place the Arabic translation into the text regions.
6. **Save:** Review the output images in the application's output location.

## Guides

| Guide | Contents |
|---|---|
| [Installation and build](docs/INSTALL.md) | Source setup and installation files |
| [Gemini keys](docs/GEMINI.md) | Key configuration and usage limits |
| [User guide](docs/USAGE.md) | Translation, editing, fonts, and export |
| [Build EXE](docs/BUILD.md) | Packaging and releases |
| [Inpainting engines](docs/INPAINTING.md) | Available engines and models |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Errors and performance issues |
| [FAQ](docs/FAQ.md) | Frequently asked questions |
| [Architecture](docs/ARCHITECTURE.md) | Technical project structure |

## Command line

```bash
python run.py cli --help
```

Use the CLI help or the [user guide](docs/USAGE.md) for the options supported by your version.

## Performance and privacy

- Text removal can take longer on low-memory computers, particularly with local models.
- Gemini translation requires an internet connection and sends the text needed for translation to the service.
- Never commit API keys or include them in screenshots.
- Review the output manually; decorative lettering and complex backgrounds can affect OCR, cleanup, and translation quality.

## Reporting issues

Open an [issue](https://github.com/fang3yuan/comic-translator-app/issues) and include your operating system, launch method, reproduction steps, and error message. Do not include API keys or sensitive information.

## License

Licensed under MIT. See the repository's [LICENSE](https://github.com/fang3yuan/comic-translator-app/blob/main/LICENSE).

---

<div align="center">

**Comic Translator** · Open-source project

</div>
