"""FastAPI app: live waits + PPO companion recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from Park import config
from Park.companion import settings
from Park.companion.server.obs import (
    CompanionState,
    WEIGHT_SLIDER_MAX,
    build_live_observation,
    default_preference_weights,
    now_sec_of_day,
    resolve_location_node_id,
)
from Park.companion.server.recommend import Recommender
from Park.companion.server.ride_map import hub_display_name
from Park.companion.server.waits import WaitTimeProvider
from Park.training.features import NUM_RIDES

WEB_DIST = Path(__file__).resolve().parents[1] / "web" / "dist"


class RecommendRequest(BaseModel):
    preference_weights: list[float] = Field(..., min_length=NUM_RIDES, max_length=NUM_RIDES)
    must_dos: list[int] = Field(..., min_length=NUM_RIDES, max_length=NUM_RIDES)
    history: list[int] = Field(..., min_length=NUM_RIDES, max_length=NUM_RIDES)
    location: str = Field(..., description="entrance | hub:<id> | ride:<id>")
    leave_hour: float | None = Field(
        default=None, description="Local hour to leave, e.g. 21.5 for 9:30 PM"
    )
    arrival_hour: float | None = Field(
        default=None, description="Local hour of arrival; default park open"
    )
    party_size: int = Field(default=2, ge=1, le=16)
    force_refresh_waits: bool = False


def _hour_to_sec_since_open(hour: float | None, *, default_sec: int) -> int:
    if hour is None:
        return default_sec
    abs_sec = int(hour * 3600)
    open_sec = config.DAY_START_HOUR * 3600
    return int(np.clip(abs_sec - open_sec, 0, config.DAY_SECONDS + config.CLOSE_DRAIN_SEC))


def create_app(
    *,
    recommender: Recommender | None = None,
    waits: WaitTimeProvider | None = None,
) -> FastAPI:
    app = FastAPI(title="OmniQueue Companion", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.recommender = recommender or Recommender(device=settings.DEVICE)
    app.state.waits = waits or WaitTimeProvider(cache_ttl_sec=settings.WAIT_CACHE_TTL_SEC)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        board = app.state.waits.get_board()
        rec: Recommender = app.state.recommender
        return {
            "ok": True,
            "model_loaded": True,
            "model_path": str(rec.checkpoint_path),
            "model_stub": rec.is_stub,
            "waits_fetched_at": board.fetched_at,
            "waits_error": board.error,
            "open_rides": sum(1 for r in board.rides if r.open),
        }

    @app.get("/api/catalog")
    def catalog() -> dict[str, Any]:
        hubs = []
        for hub_id, _coords in sorted(config.HUB_COORDS.items()):
            hubs.append(
                {
                    "id": hub_id,
                    "key": f"hub:{hub_id}" if hub_id != config.NODE_ENTRANCE else "entrance",
                    "name": hub_display_name(hub_id),
                    "kind": "entrance" if hub_id == config.NODE_ENTRANCE else "hub",
                }
            )
        # Deduplicate entrance listed as hub:0
        seen = set()
        unique_hubs = []
        for h in hubs:
            if h["key"] in seen:
                continue
            seen.add(h["key"])
            unique_hubs.append(h)

        rides = []
        for i, ride in enumerate(config.RIDES):
            rides.append(
                {
                    "id": i,
                    "name": ride["name"],
                    "hub_id": config.RIDE_HUB[i],
                    "hub_name": hub_display_name(config.RIDE_HUB[i]),
                    "location_key": f"ride:{i}",
                    "popularity": ride["popularity"],
                    "duration_min": ride["duration_sec"] / 60.0,
                }
            )
        return {
            "num_rides": NUM_RIDES,
            "weight_slider_max": WEIGHT_SLIDER_MAX,
            "default_preference_weights": default_preference_weights().tolist(),
            "day_start_hour": config.DAY_START_HOUR,
            "day_end_hour": config.DAY_END_HOUR,
            "hubs": unique_hubs,
            "rides": rides,
        }

    @app.get("/api/waits")
    def waits_endpoint(force: bool = False) -> dict[str, Any]:
        board = app.state.waits.get_board(force=force)
        return {
            "fetched_at": board.fetched_at,
            "source": board.source,
            "park_id": board.park_id,
            "error": board.error,
            "rides": [
                {
                    "ride_id": r.ride_id,
                    "name": r.name,
                    "wait_min": r.wait_min,
                    "status": r.status,
                    "open": r.open,
                    "entity_id": r.entity_id,
                }
                for r in board.rides
            ],
        }

    @app.post("/api/recommend")
    def recommend(body: RecommendRequest) -> dict[str, Any]:
        try:
            location_node_id = resolve_location_node_id(body.location)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        now_sec = now_sec_of_day()
        leave_sec = _hour_to_sec_since_open(body.leave_hour, default_sec=config.DAY_SECONDS)
        spawn_sec = _hour_to_sec_since_open(
            body.arrival_hour, default_sec=0
        )

        state = CompanionState(
            preference_weights=np.asarray(body.preference_weights, dtype=np.float32),
            must_dos=np.asarray(body.must_dos, dtype=np.uint8),
            history=np.asarray(body.history, dtype=np.int32),
            location_node_id=location_node_id,
            leave_sec=leave_sec,
            spawn_sec=spawn_sec,
            party_size=body.party_size,
        )
        board = app.state.waits.get_board(force=body.force_refresh_waits)
        try:
            flat, meta = build_live_observation(state, board, now_sec=now_sec)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"obs build failed: {exc}") from exc

        result = app.state.recommender.recommend(flat)
        waits_by_id = {r.ride_id: r for r in board.rides}
        for row in result["distribution"]:
            if row["is_ride"]:
                live = waits_by_id.get(row["action_id"])
                row["wait_min"] = live.wait_min if live else None
                row["status"] = live.status if live else "UNKNOWN"
                row["open"] = live.open if live else False

        return {
            **result,
            "meta": meta,
            "waits_fetched_at": board.fetched_at,
            "waits_error": board.error,
            "now_sec": now_sec,
        }

    if WEB_DIST.is_dir():
        assets = WEB_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIST / "index.html")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str) -> FileResponse:
            candidate = WEB_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")

    return app


app = create_app()
