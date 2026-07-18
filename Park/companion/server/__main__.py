"""Run the companion API (and optional built SPA).

Prefer the path-safe launcher:
    python Park/run_companion.py

Or, with Park already importable:
    python -m Park.companion.server
"""

from __future__ import annotations

import uvicorn

from Park.companion import settings


def main() -> None:
    uvicorn.run(
        "Park.companion.server.app:app",
        host=settings.HOST,
        port=int(settings.PORT),
        reload=bool(settings.RELOAD),
    )


if __name__ == "__main__":
    main()
