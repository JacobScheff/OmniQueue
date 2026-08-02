import { useEffect, useMemo, useRef, useState } from "react";
import type { Catalog, RideInfo, UserState, WaitRow } from "../lib/api";
import { waitLabel, waitLevel } from "../lib/format";
import { SearchIcon } from "./Icons";

type RidesViewProps = {
  catalog: Catalog;
  state: UserState;
  waitById: Map<number, WaitRow>;
  onCommit: (next: UserState) => void;
};

type SortMode = "pref" | "wait" | "az";

function priorityLabel(weight: number, max: number): string {
  if (weight <= 0) return "Skip";
  const frac = weight / Math.max(1, max);
  if (frac < 0.34) return "Low priority";
  if (frac < 0.67) return "Medium priority";
  return "High priority";
}

export default function RidesView({ catalog, state, waitById, onCommit }: RidesViewProps) {
  const [query, setQuery] = useState("");
  const [sortMode, setSortMode] = useState<SortMode>("pref");
  const [localWeights, setLocalWeights] = useState<number[]>(() => [...state.preference_weights]);
  const draggingRef = useRef(false);

  // Live slider feedback follows local drag state; once the drag ends (or an
  // external change like undo/redo lands) it resyncs from committed state.
  useEffect(() => {
    if (draggingRef.current) return;
    setLocalWeights([...state.preference_weights]);
  }, [state.preference_weights]);

  const rideById = useMemo(() => {
    const map = new Map<number, RideInfo>();
    for (const r of catalog.rides) map.set(r.id, r);
    return map;
  }, [catalog]);

  // Ordering always derives from committed weights (not the in-flight drag
  // value) so rows only reshuffle once an edit is released, not every tick.
  const order = useMemo(() => {
    const rides = [...catalog.rides];
    if (sortMode === "az") {
      rides.sort((a, b) => a.name.localeCompare(b.name));
    } else if (sortMode === "wait") {
      rides.sort((a, b) => {
        const wa = waitById.get(a.id);
        const wb = waitById.get(b.id);
        const va = wa?.open && wa.wait_min != null ? wa.wait_min : Number.POSITIVE_INFINITY;
        const vb = wb?.open && wb.wait_min != null ? wb.wait_min : Number.POSITIVE_INFINITY;
        return va !== vb ? va - vb : a.id - b.id;
      });
    } else {
      const weights = state.preference_weights;
      rides.sort((a, b) => {
        const d = weights[b.id] - weights[a.id];
        return d !== 0 ? d : a.id - b.id;
      });
    }
    return rides.map((r) => r.id);
  }, [catalog, sortMode, state.preference_weights, waitById]);

  const visibleRides = useMemo(() => {
    const q = query.trim().toLowerCase();
    return order
      .map((id) => rideById.get(id))
      .filter((r): r is RideInfo => {
        if (!r) return false;
        if (!q) return true;
        return r.name.toLowerCase().includes(q) || r.hub_name.toLowerCase().includes(q);
      });
  }, [order, query, rideById]);

  const setWeight = (rideId: number, value: number) => {
    setLocalWeights((prev) => {
      const next = [...prev];
      next[rideId] = value;
      return next;
    });
  };

  const commitWeight = () => {
    draggingRef.current = false;
    onCommit({ ...state, preference_weights: [...localWeights] });
  };

  const toggleMust = (rideId: number) => {
    const must_dos = [...state.must_dos];
    must_dos[rideId] = must_dos[rideId] ? 0 : 1;
    onCommit({ ...state, must_dos });
  };

  const bumpDone = (rideId: number, delta: number) => {
    const history = [...state.history];
    history[rideId] = Math.max(0, Math.min(20, history[rideId] + delta));
    onCommit({ ...state, history });
  };

  return (
    <div className="view rides-view">
      <div className="rides-toolbar">
        <label className="search-field">
          <SearchIcon size={15} />
          <input
            type="search"
            placeholder="Search rides or lands…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <div className="sort-tabs" role="group" aria-label="Sort rides">
          {(
            [
              ["pref", "My priority"],
              ["wait", "Shortest wait"],
              ["az", "A–Z"],
            ] as [SortMode, string][]
          ).map(([mode, label]) => (
            <button
              key={mode}
              type="button"
              className={`sort-tab${sortMode === mode ? " active" : ""}`}
              onClick={() => setSortMode(mode)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <ul className="ride-cards">
        {visibleRides.map((ride) => {
          const live = waitById.get(ride.id);
          const level = waitLevel(live?.wait_min, live?.open, live?.status);
          const weight = localWeights[ride.id] ?? 0;
          const done = state.history[ride.id] > 0;
          const must = state.must_dos[ride.id] > 0;

          return (
            <li key={ride.id} className="ride-card">
              <div className="ride-card-top">
                <div>
                  <div className="ride-card-name">
                    {ride.name}
                    {must && <span className="tag must">Must-do</span>}
                    {done && <span className="tag done">Done ×{state.history[ride.id]}</span>}
                  </div>
                  <div className="ride-card-hub">{ride.hub_name}</div>
                </div>
                <span className={`wait-chip level-${level}`}>
                  {waitLabel(live?.wait_min, live?.open, live?.status)}
                </span>
              </div>

              <div className="pref-slider-row">
                <input
                  type="range"
                  min={0}
                  max={catalog.weight_slider_max}
                  value={weight}
                  onChange={(e) => {
                    draggingRef.current = true;
                    setWeight(ride.id, Number(e.target.value));
                  }}
                  onPointerUp={commitWeight}
                  onKeyUp={(e) => {
                    if (
                      ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(
                        e.key,
                      )
                    ) {
                      commitWeight();
                    }
                  }}
                  aria-label={`Preference for ${ride.name}`}
                />
                <span className="pref-label">{priorityLabel(weight, catalog.weight_slider_max)}</span>
              </div>

              <div className="ride-card-actions">
                <button
                  type="button"
                  className={`chip-btn${must ? " on" : ""}`}
                  onClick={() => toggleMust(ride.id)}
                >
                  Must-do
                </button>
                <div className="stepper" aria-label={`Times ridden ${ride.name}`}>
                  <span className="stepper-label">Done</span>
                  <button type="button" onClick={() => bumpDone(ride.id, -1)} aria-label="Decrease">
                    −
                  </button>
                  <span>{state.history[ride.id]}</span>
                  <button type="button" onClick={() => bumpDone(ride.id, 1)} aria-label="Increase">
                    +
                  </button>
                </div>
              </div>
            </li>
          );
        })}
        {visibleRides.length === 0 && <li className="empty-hint">No rides match “{query}”.</li>}
      </ul>
    </div>
  );
}
