"""Application launcher: GUI by default, ``cli`` sub-command for batch use."""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Sequence

from src import __version__
from src.config import Settings, prepare_user_dirs, resolve_path
from src.translators.gemini import parse_keys
from src.utils.sysutils import lower_priority


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns:
        Configured ``ArgumentParser`` with ``gui`` and ``cli`` sub-commands.
    """
    parser = argparse.ArgumentParser(prog="comic-translator", description="English -> Arabic comic translator")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("gui", help="Launch the desktop application (default)")

    cli = sub.add_parser("cli", help="Process images / folders without the GUI")
    cli.add_argument("inputs", nargs="*", type=Path, help="Image files or folders")
    cli.add_argument("-o", "--output", type=Path, default=Path("output"), help="Output folder")
    cli.add_argument("--key", action="append", default=[], metavar="GEMINI_KEY",
                     help="Gemini API key (repeat for several keys; or set GEMINI_API_KEY)")
    cli.add_argument("--model", help="Gemini model name")
    cli.add_argument("--order", choices=["rtl", "ltr"], help="Reading order")
    cli.add_argument("--format", choices=["png", "jpg", "webp"], help="Output format")
    cli.add_argument("--inpaint", choices=["auto", "lama", "migan", "telea", "ns"], help="Inpainting backend")
    cli.add_argument("--download-models", action="store_true", help="Download the LaMa ONNX model and exit")
    return parser


def _apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> None:
    """Copy CLI flags into ``settings`` (without saving them).

    Args:
        settings: Settings to modify.
        args: Parsed arguments.
    """
    keys = list(args.key) + parse_keys(os.environ.get("GEMINI_API_KEY", ""))
    if keys:
        settings.set("translation.gemini_keys", parse_keys(keys))
    if args.model:
        settings.set("translation.model", args.model)
    if args.order:
        settings.set("general.reading_order", args.order)
    if args.format:
        settings.set("general.output_format", args.format)
    if args.inpaint:
        settings.set("inpainting.backend", args.inpaint)


def run_cli(args: argparse.Namespace) -> int:
    """Run batch processing from the command line.

    Args:
        args: Parsed ``cli`` arguments.

    Returns:
        Process exit code.
    """
    from src.core.inpainter import ensure_lama_model
    from src.core.pipeline import ComicPipeline
    from src.translators import TranslationError
    from src.utils.image_utils import IMAGE_EXTENSIONS, list_images

    settings = Settings.load()
    _apply_cli_overrides(settings, args)

    if args.download_models:
        target = resolve_path(settings.get("inpainting.lama_model", "models/lama_fp32.onnx"))
        last = {"pct": -1}

        def show(frac: float) -> None:
            pct = int(frac * 100)
            if pct != last["pct"] and pct % 5 == 0:
                last["pct"] = pct
                print(f"\rDownloading LaMa: {pct}%", end="", flush=True)

        try:
            path = ensure_lama_model(target, show)
        except RuntimeError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 1
        print(f"\nLaMa model ready at {path}")
        return 0

    files: List[Path] = []
    for item in args.inputs:
        if item.is_dir():
            files.extend(list_images(item))
        elif item.suffix.lower() in IMAGE_EXTENSIONS and item.exists():
            files.append(item)
        else:
            print(f"Skipping {item}: not an image or folder", file=sys.stderr)
    if not files:
        print("No input images. Usage: python run.py cli PAGE_OR_FOLDER -o output", file=sys.stderr)
        return 2

    pipeline = ComicPipeline(settings, log=lambda m: print(m, flush=True))
    last_msg = {"text": ""}

    def progress(fraction: float, message: str) -> None:
        if message != last_msg["text"]:
            last_msg["text"] = message
            print(f"[{fraction:6.1%}] {message}", flush=True)

    try:
        outputs = pipeline.process_batch(files, args.output, progress, threading.Event())
    except TranslationError as exc:
        print(f"\nTranslation failed:\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    print(f"\n{len(outputs)}/{len(files)} page(s) written to {args.output.resolve()}")
    return 0 if len(outputs) == len(files) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    args = build_parser().parse_args(argv)
    prepare_user_dirs()
    lower_priority()
    if args.command == "cli":
        return run_cli(args)
    try:
        from src.ui.app import run_gui
    except ImportError as exc:
        print(f"GUI dependencies missing ({exc}). Run: pip install -r requirements.txt", file=sys.stderr)
        return 1
    run_gui()
    return 0


if __name__ == "__main__":
    sys.exit(main())
