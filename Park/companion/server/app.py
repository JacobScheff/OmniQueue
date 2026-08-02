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
from Park.companion.server.recommend import ModelRegistry, Recommender
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
    model_version: str | None = Field(
        default=None,
        description="Model tag from settings.MODELS (e.g. v1, v2). Defaults to DEFAULT_MODEL_VERSION.",
    )
    force_slot: int | None = Field(
        default=None,
        description=(
            "Optional route position (0..route_k-1) to pin to force_action; earlier "
            "and later slots still decode autoregressively/naturally. Must be set "
            "together with force_action."
        ),
        ge=0,
    )
    force_action: int | None = Field(
        default=None,
        description=(
            "Action id to pin at force_slot. For slot 0 this may be any legal "
            "action (including exit/idle); for later slots it must be a ride id "
            "that is legal at that stop."
        ),
        ge=0,
        le=NUM_RIDES + 1,
    )
    force_refresh_waits: bool = False


def _hour_to_sec_since_open(hour: float | None, *, default_sec: int) -> int:
    if hour is None:
        return default_sec
    abs_sec = int(hour * 3600)
    open_sec = config.DAY_START_HOUR * 3600
    return int(np.clip(abs_sec - open_sec, 0, config.DAY_SECONDS + config.CLOSE_DRAIN_SEC))


def create_app(
    *,
    registry: ModelRegistry | None = None,
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

    if registry is not None:
        app.state.registry = registry
    elif recommender is not None:
        # Single-model test harness: wrap as a one-entry registry.
        class _Single:
            default_version = settings.DEFAULT_MODEL_VERSION

            def __init__(self, rec: Recommender) -> None:
                self._rec = rec

            def versions(self) -> list[dict]:
                info = self._rec.info()
                vid = info.get("version") or self.default_version
                return [{"id": vid, "label": str(vid).upper(), **info}]

            def get(self, version: str | None = None) -> Recommender:
                return self._rec

        app.state.registry = _Single(recommender)
    else:
        app.state.registry = ModelRegistry(device=settings.DEVICE)

    app.state.waits = waits or WaitTimeProvider(cache_ttl_sec=settings.WAIT_CACHE_TTL_SEC)

    # Warm walk matrix + wait board at boot so the first /api/recommend does not
    # block on a multi-minute all-pairs pathway rebuild (Render free CPU).
    if registry is None and recommender is None:
        from Park.park_graph import get_park_graph

        get_park_graph()
        app.state.waits.get_board()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        board = app.state.waits.get_board()
        reg = app.state.registry
        default = reg.get(reg.default_version)
        return {
            "ok": True,
            "model_loaded": True,
            "default_model_version": reg.default_version,
            "models": reg.versions(),
            "model_path": str(default.checkpoint_path),
            "model_stub": default.is_stub,
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
        reg = app.state.registry
        return {
            "num_rides": NUM_RIDES,
            "weight_slider_max": WEIGHT_SLIDER_MAX,
            "default_preference_weights": default_preference_weights().tolist(),
            "day_start_hour": config.DAY_START_HOUR,
            "day_end_hour": config.DAY_END_HOUR,
            "hubs": unique_hubs,
            "rides": rides,
            "default_model_version": reg.default_version,
            "models": reg.versions(),
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

        try:
            rec = app.state.registry.get(body.model_version)
        except KeyError as exc:
            known = [v["id"] for v in app.state.registry.versions()]
            raise HTTPException(
                status_code=400,
                detail=f"unknown model_version {body.model_version!r}; choose one of {known}",
            ) from exc

        now_sec = now_sec_of_day()
        leave_sec = _hour_to_sec_since_open(body.leave_hour, default_sec=config.DAY_SECONDS)
        spawn_sec = _hour_to_sec_since_open(body.arrival_hour, default_sec=0)

        state = CompanionState(
            preference_weights=np.asarray(body.preference_weights, dtype=np.float32),
            must_dos=np.asarray(body.must_dos, dtype=np.uint8),
            history=np.asarray(body.history, dtype=np.int32),
            location_node_id=location_node_id,
            leave_sec=leave_sec,
            spawn_sec=spawn_sec,
        )
        board = app.state.waits.get_board(force=body.force_refresh_waits)
        try:
            flat, meta = build_live_observation(state, board, now_sec=now_sec)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"obs build failed: {exc}") from exc

        try:
            result = rec.recommend(
                flat, force_slot=body.force_slot, force_action=body.force_action
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        waits_by_id = {r.ride_id: r for r in board.rides}

        def _annotate_dist(rows: list[dict]) -> None:
            for row in rows:
                if row.get("is_ride"):
                    live = waits_by_id.get(row["action_id"])
                    row["wait_min"] = live.wait_min if live else None
                    row["status"] = live.status if live else "UNKNOWN"
                    row["open"] = live.open if live else False

        _annotate_dist(result["distribution"])
        for slot_rows in result.get("distributions_by_slot", []):
            _annotate_dist(slot_rows)

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
