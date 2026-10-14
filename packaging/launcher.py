"""Entry point of the packaged (PyInstaller) app."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from src.main import main

    sys.exit(main())
