"""Run the companion API (and optional built SPA).

Usage:
    python -m Park.companion.server.app
    # or:
    uvicorn Park.companion.server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("COMPANION_HOST", "0.0.0.0")
    port = int(os.environ.get("COMPANION_PORT", "8000"))
    uvicorn.run(
        "Park.companion.server.app:app",
        host=host,
        port=port,
        reload=os.environ.get("COMPANION_RELOAD", "").lower() in ("1", "true", "yes"),
    )


if __name__ == "__main__":
    main()
