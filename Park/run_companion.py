#!/usr/bin/env python3
"""Launch the live companion API (and built SPA if present).

Run from anywhere:
    python Park/run_companion.py

Or from the Park directory:
    python run_companion.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_park_importable() -> None:
    """Put the parent of Park/ on sys.path so `import Park` works without PYTHONPATH."""
    park_dir = Path(__file__).resolve().parent
    repo_root = park_dir.parent
    root = str(repo_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def main() -> None:
    _ensure_park_importable()
    from Park.companion.server.__main__ import main as server_main

    server_main()


if __name__ == "__main__":
    main()
