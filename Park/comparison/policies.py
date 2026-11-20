"""Adapters for the existing trained RL model and history-aware DP planner."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

from Park import config
from Park.comparison.runner import Choice
from Park.planner import Context, Ride, plan
from Park.planner.data import Forecast


class ForecastClient:
    def __init__(self, python, settings):
        self.process = subprocess.Popen(
            [str(python), "-u", str(Path(__file__).with_name("forecast_worker.py"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.metadata = {}
        self.process.stdin.write(json.dumps(settings)+"\n")
        self.process.stdin.flush()

    def forecast(self, now):
        if self.process.poll() is not None:
            raise ValueError("Wait forecast worker exited; check its error output")
        self.process.stdin.write(json.dumps(dict(now=now))+"\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise ValueError("Wait forecast worker failed; check its error output")
        payload = json.loads(line)
        if "error" in payload:
            raise ValueError("Wait forecast failed: "+payload["error"])
        self.metadata = payload["metadata"]
        return payload

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait()
        self.process.stdout.close()


class DPPolicy:
    name = "DP"

    def __init__(self, walks, profile, closing, forecasts, *, seconds=30, beam=1000,
                 step=60, buffer=300, quantile="p90", walk_cost=2.0):
        self.walks, self.profile, self.closing, self.forecasts = walks, profile, closing, forecasts
        self.settings = dict(seconds=seconds, beam=beam, step=step, buffer=buffer, walk_cost=walk_cost)
        self.quantile = quantile

    def choose(self, state, board):
        payload = self.forecasts.forecast(state.now)
        forecast = Forecast(payload, quantile=self.quantile)
        rides = [Ride(j, r["name"], r["duration_sec"], self.profile["preferences"][j],
                      j in self.profile["must_dos"], j in state.completed,
                      queue_closes=self.closing, finish_by=self.closing)
                 for j, r in enumerate(config.RIDES)]
        ctx = Context(rides, state.now, self.closing, self.walks.matrix(state.location),
                      forecast.wait_at, speed=self.profile["speed"],
                      operating_at=forecast.operating_at, **self.settings)
        result = plan(ctx)
        routes = result.pop("routes")
        detail = dict(planner=result, forecast_as_of=forecast.now,
                      forecast_history_ticks=payload["history_ticks"])
        # Keep the proposed itinerary at every replan for later inspection.
        detail["planned_stops"] = routes[0]["stops"] if routes else []
        if routes and routes[0]["stops"]:
            stop = routes[0]["stops"][0]
            return Choice(stop["ride_id"], stop["depart"], stop["join"], detail)
        detail["reason"] = (result.get("reason") or
                            "No feasible unfinished ride; retry with the next observed board")
        return Choice(detail=detail)


class RLPolicy:
    name = "RL"

    def __init__(self, checkpoint, device, walks, profile, opening, closing):
        import numpy as np
        from Park.router.ppo import PPOPolicy
        from Park.park_graph import get_park_graph

        self.np = np
        self.policy = PPOPolicy(checkpoint, device=device, close_margin=0.0)
        from Park.training.features import GUEST_FEAT_DIM, RIDE_DYNAMIC_FEAT_DIM
        model = self.policy.model
        if (model.guest_proj[0].in_features != GUEST_FEAT_DIM+3
                or model.ride_feat_proj[0].in_features != RIDE_DYNAMIC_FEAT_DIM
                or model.num_rides != config.NUM_RIDES or model.env_dim != 3):
            raise ValueError("RL checkpoint observation dimensions do not match this repository's current features; "
                             "use a compatible checkpoint (for example Park/companion/model/v2.pt)")
        if self.policy.meta.get("load_notes"):
            raise ValueError("RL checkpoint does not match the current model: "+self.policy.meta["load_notes"])
        self.park = get_park_graph()
        self.walks, self.profile = walks, profile
        self.opening, self.closing = opening, closing

    def observation(self, state, board):
        """Match companion/native feature units, using the shared verified walks."""
        np = self.np
        from Park.training.features import GUEST_FEAT_DIM, RIDE_DYNAMIC_FEAT_DIM
        prefs = np.asarray(self.profile["preferences"], dtype=np.float32)
        speed = self.profile["speed"]
        unfinished = np.array([j not in state.completed for j in range(config.NUM_RIDES)])
        required = np.array([j in self.profile["must_dos"] for j in range(config.NUM_RIDES)]) & unfinished
        remaining = self.closing-state.now
        elapsed = state.now-self.opening
        guest = np.zeros(GUEST_FEAT_DIM, dtype=np.float32)
        guest[:config.NUM_RIDES] = prefs
        node = config.NODE_ENTRANCE if state.location == config.NUM_RIDES else config.ride_node_id(state.location)
        guest[34:] = [float((prefs[unfinished]**config.PPO_PREF_REWARD_EXP).sum()), speed/2,
                      remaining/config.DAY_SECONDS, self.park.node_to_idx(node)/self.park.num_nodes,
                      min(len(state.completed), 40)/20, int(required.sum())/5,
                      float(state.location < config.NUM_RIDES), 1/16, elapsed/config.DAY_SECONDS]
        ride = np.zeros((config.NUM_RIDES, RIDE_DYNAMIC_FEAT_DIM), dtype=np.float32)
        known_waits = [wait for wait in board if wait is not None]
        mean = sum(known_waits)/len(known_waits) if known_waits else 0.0
        for j, r in enumerate(config.RIDES):
            meters = self.walks.distances[state.location][j]
            exit_m = self.walks.distances[j][config.NUM_RIDES]
            reachable = math.isfinite(meters) and math.isfinite(exit_m)
            walk = math.ceil(meters/speed) if reachable else 360000
            wait = board[j] or 0
            feasible = (reachable and board[j] is not None and unfinished[j]
                        and walk+wait+r["duration_sec"]+math.ceil(exit_m/speed) <= remaining)
            wf, qf = min(walk, 3600)/3600, min(wait, 3600)/3600
            ride[j] = [qf, 0, float(feasible), r["duration_sec"]/900,
                       r["capacity_per_hour"]/3600, wf, float(not unfinished[j])/10,
                       float(required[j]), float(prefs[j]**config.PPO_PREF_REWARD_EXP) if unfinished[j] else 0,
                       min(wf+qf, 2), float(np.clip(qf-mean/3600, -1, 1))]
        # Normalize the close flag to the *recorded* opening/closing window.
        env = np.array([elapsed/(self.closing-self.opening), mean/3600,
                        sum(w is None for w in board)/config.NUM_RIDES], dtype=np.float32)
        return np.concatenate([guest, ride.reshape(-1), env])

    def choose(self, state, board):
        route, _ = self.policy.act_with_route(self.observation(state, board))
        action = int(route[0])
        detail = dict(model_action=action, proposed_route=[int(j) for j in route])
        if action >= config.NUM_RIDES or action < 0:
            detail["reason"] = "RL chose exit/idle; full-day comparison waits for the next observation"
            return Choice(detail=detail)
        return Choice(action, detail=detail)
