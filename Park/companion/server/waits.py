"""Fetch and cache live Disneyland wait times from ThemeParks.wiki."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from Park.companion.server.ride_map import (
    DISNEYLAND_PARK_ENTITY_ID,
    build_config_name_index,
    resolve_ride_index,
)
from Park import config

logger = logging.getLogger(__name__)

API_BASE = "https://api.themeparks.wiki/v1"
DEFAULT_CACHE_TTL_SEC = 45.0
USER_AGENT = "OmniQueueCompanion/0.1 (+https://github.com/themeparks)"


@dataclass
class RideLiveStatus:
    ride_id: int
    name: str
    wait_min: float | None
    status: str  # OPERATING | DOWN | CLOSED | REFURBISHMENT | UNKNOWN
    open: bool
    entity_id: str | None = None
    last_updated: str | None = None


@dataclass
class LiveBoard:
    fetched_at: float
    rides: list[RideLiveStatus]
    unmatched: list[str] = field(default_factory=list)
    source: str = "themeparks.wiki"
    park_id: str = DISNEYLAND_PARK_ENTITY_ID
    error: str | None = None

    def by_ride_id(self) -> dict[int, RideLiveStatus]:
        return {r.ride_id: r for r in self.rides}


class WaitTimeProvider:
    """Cached live wait board. Safe to share across requests."""

    def __init__(
        self,
        *,
        cache_ttl_sec: float = DEFAULT_CACHE_TTL_SEC,
        park_id: str = DISNEYLAND_PARK_ENTITY_ID,
        timeout_sec: float = 12.0,
    ) -> None:
        self.cache_ttl_sec = cache_ttl_sec
        self.park_id = park_id
        self.timeout_sec = timeout_sec
        self._name_index = build_config_name_index()
        self._board: LiveBoard | None = None
        self._lock_fetch = False

    def get_board(self, *, force: bool = False) -> LiveBoard:
        now = time.time()
        if (
            not force
            and self._board is not None
            and (now - self._board.fetched_at) < self.cache_ttl_sec
            and self._board.error is None
        ):
            return self._board
        try:
            board = self._fetch()
            self._board = board
            return board
        except Exception as exc:  # noqa: BLE001 — surface as board error for clients
            logger.exception("live wait fetch failed")
            if self._board is not None:
                stale = LiveBoard(
                    fetched_at=self._board.fetched_at,
                    rides=list(self._board.rides),
                    unmatched=list(self._board.unmatched),
                    source=self._board.source,
                    park_id=self._board.park_id,
                    error=f"stale cache; refresh failed: {exc}",
                )
                return stale
            empty = self._empty_board(error=str(exc))
            self._board = empty
            return empty

    def _empty_board(self, error: str | None = None) -> LiveBoard:
        rides = [
            RideLiveStatus(
                ride_id=i,
                name=r["name"],
                wait_min=None,
                status="UNKNOWN",
                open=False,
            )
            for i, r in enumerate(config.RIDES)
        ]
        return LiveBoard(fetched_at=time.time(), rides=rides, error=error)

    def _fetch(self) -> LiveBoard:
        url = f"{API_BASE}/entity/{self.park_id}/live"
        with httpx.Client(
            timeout=self.timeout_sec,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()

        live_rows = payload.get("liveData") or []
        by_id: dict[int, RideLiveStatus] = {}
        unmatched: list[str] = []

        for row in live_rows:
            if row.get("entityType") != "ATTRACTION":
                continue
            name = str(row.get("name") or "")
            entity_id = row.get("id")
            ride_id = resolve_ride_index(entity_id, name, self._name_index)
            if ride_id is None:
                unmatched.append(name)
                continue
            status = str(row.get("status") or "UNKNOWN").upper()
            queue = row.get("queue") or {}
            standby = queue.get("STANDBY") or {}
            wait = standby.get("waitTime")
            wait_min = float(wait) if wait is not None else None
            open_ok = status == "OPERATING"
            # DOWN / CLOSED / REFURBISHMENT → closed for the policy mask
            by_id[ride_id] = RideLiveStatus(
                ride_id=ride_id,
                name=config.RIDES[ride_id]["name"],
                wait_min=wait_min if open_ok else None,
                status=status,
                open=open_ok,
                entity_id=str(entity_id) if entity_id else None,
                last_updated=row.get("lastUpdated"),
            )

        rides: list[RideLiveStatus] = []
        for i, ride in enumerate(config.RIDES):
            if i in by_id:
                rides.append(by_id[i])
            else:
                rides.append(
                    RideLiveStatus(
                        ride_id=i,
                        name=ride["name"],
                        wait_min=None,
                        status="UNKNOWN",
                        open=False,
                    )
                )

        return LiveBoard(
            fetched_at=time.time(),
            rides=rides,
            unmatched=sorted(set(unmatched)),
            park_id=self.park_id,
        )
