import { useEffect, useMemo, useState } from "react";
import {
  type Catalog,
  type RecommendResponse,
  type UserState,
  type WaitRow,
  fetchCatalog,
  fetchWaits,
  postRecommend,
} from "./api";
import {
  type StoredBundle,
  loadBundle,
  pushEdit,
  redo,
  saveBundle,
  undo,
} from "./storage";

function formatProb(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function waitLabel(wait: number | null | undefined, open?: boolean, status?: string): string {
  if (status && status !== "OPERATING" && status !== "UNKNOWN") return status;
  if (!open && status) return status;
  if (wait == null) return "—";
  return `${Math.round(wait)} min`;
}

function cloneState(state: UserState): UserState {
  return {
    preference_weights: [...state.preference_weights],
    must_dos: [...state.must_dos],
    history: [...state.history],
    location: state.location,
    leave_hour: state.leave_hour,
    arrival_hour: state.arrival_hour,
    party_size: state.party_size,
  };
}

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [bundle, setBundle] = useState<StoredBundle | null>(null);
  const [waits, setWaits] = useState<WaitRow[]>([]);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [draft, setDraft] = useState<UserState | null>(null);
  const [showAllDist, setShowAllDist] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cat = await fetchCatalog();
        if (cancelled) return;
        setCatalog(cat);
        const loaded = loadBundle(cat);
        setBundle(loaded);
        saveBundle(loaded);
        const board = await fetchWaits(false);
        if (!cancelled) setWaits(board.rides);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!bundle) return;
    saveBundle(bundle);
  }, [bundle]);

  useEffect(() => {
    if (!bundle) return;
    let cancelled = false;
    const run = async () => {
      setBusy(true);
      setError(null);
      try {
        const rec = await postRecommend(bundle.state, false);
        if (!cancelled) setResult(rec);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setBusy(false);
      }
    };
    const t = window.setTimeout(run, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(t);
    };
  }, [bundle]);

  const waitById = useMemo(() => {
    const map = new Map<number, WaitRow>();
    for (const w of waits) map.set(w.ride_id, w);
    return map;
  }, [waits]);

  const sortedRides = useMemo(() => {
    if (!catalog || !bundle) return [];
    const weights = (draft ?? bundle.state).preference_weights;
    return [...catalog.rides].sort((a, b) => weights[b.id] - weights[a.id]);
  }, [catalog, bundle, draft]);

  const openEditor = () => {
    if (!bundle) return;
    setDraft(cloneState(bundle.state));
    setEditOpen(true);
  };

  const commitEditor = () => {
    if (!bundle || !draft) return;
    setBundle(pushEdit(bundle, draft));
    setEditOpen(false);
    setDraft(null);
  };

  const refresh = async () => {
    if (!bundle) return;
    setBusy(true);
    setError(null);
    try {
      const board = await fetchWaits(true);
      setWaits(board.rides);
      const rec = await postRecommend(bundle.state, true);
      setResult(rec);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (!catalog || !bundle) {
    return (
      <div className="app">
        <p className="loading">{error ? error : "Loading companion…"}</p>
      </div>
    );
  }

  const state = bundle.state;
  const edit = draft ?? state;
  const dist = result?.distribution ?? [];
  const visibleDist = showAllDist ? dist : dist.filter((d) => d.legal).slice(0, 8);
  const stub = result?.model.stub;

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1 className="brand">OmniQueue</h1>
          <p className="sub">Live Disneyland routing companion</p>
        </div>
      </header>

      <div className="pill-row">
        <span className="pill">{busy ? "Updating…" : "Live waits"}</span>
        {result && (
          <span className="pill">
            {result.meta.open_rides} open · avg {result.meta.mean_wait_min.toFixed(0)} min
          </span>
        )}
        {stub && <span className="pill warn">Stub model — replace checkpoint</span>}
        {result?.waits_error && <span className="pill bad">Wait feed issue</span>}
      </div>

      {error && <div className="error">{error}</div>}

      <section className="hero-rec">
        <p className="eyebrow">Next action</p>
        <h2 className="rec-title">
          {result ? result.recommended.label : busy ? "Thinking…" : "—"}
        </h2>
        {result && (
          <div className="rec-prob">{formatProb(result.recommended.prob)} confidence</div>
        )}
        <div className="actions">
          <button className="btn" type="button" onClick={() => void refresh()} disabled={busy}>
            Refresh waits
          </button>
          <button className="btn secondary" type="button" onClick={openEditor}>
            Edit prefs
          </button>
        </div>
      </section>

      <section className="section">
        <h2>Probability distribution</h2>
        <ul className="dist-list">
          {visibleDist.map((row) => (
            <li key={row.action_id} className={`dist-item${row.legal ? "" : " illegal"}`}>
              <div>
                <div className="ride-name">{row.label}</div>
                <div className="ride-meta">
                  {row.is_ride
                    ? waitLabel(row.wait_min, row.open, row.status)
                    : row.legal
                      ? "legal"
                      : "masked"}
                </div>
              </div>
              <strong>{formatProb(row.prob)}</strong>
              <div className="bar">
                <span style={{ width: `${Math.min(100, row.prob * 100)}%` }} />
              </div>
            </li>
          ))}
        </ul>
        {dist.length > 8 && (
          <div className="actions">
            <button
              className="btn ghost"
              type="button"
              onClick={() => setShowAllDist((v) => !v)}
            >
              {showAllDist ? "Show top legal" : "Show all actions"}
            </button>
          </div>
        )}
      </section>

      <section className="section">
        <h2>Rides</h2>
        <ul className="ride-list">
          {sortedRides.map((ride) => {
            const live = waitById.get(ride.id);
            const done = state.history[ride.id] > 0;
            const must = state.must_dos[ride.id] > 0 && !done;
            return (
              <li key={ride.id} className="ride-item">
                <div>
                  <div className="ride-name">
                    {ride.name}
                    {must && <span className="tag must">MUST</span>}
                    {done && <span className="tag done">DONE ×{state.history[ride.id]}</span>}
                  </div>
                  <div className="ride-meta">
                    {waitLabel(live?.wait_min, live?.open, live?.status)} · {ride.hub_name}
                  </div>
                </div>
                <div className="ride-meta">pref {Math.round(state.preference_weights[ride.id])}</div>
              </li>
            );
          })}
        </ul>
      </section>

      <div className="dock">
        <button
          className="btn ghost"
          type="button"
          disabled={bundle.past.length === 0}
          onClick={() => setBundle((b) => (b ? undo(b) : b))}
        >
          Undo
        </button>
        <button
          className="btn ghost"
          type="button"
          disabled={bundle.future.length === 0}
          onClick={() => setBundle((b) => (b ? redo(b) : b))}
        >
          Redo
        </button>
        <button className="btn secondary" type="button" onClick={openEditor}>
          Edit
        </button>
      </div>

      {editOpen && draft && (
        <div className="sheet" role="dialog" aria-modal="true">
          <div className="sheet-panel">
            <div className="sheet-head">
              <h2>Your day</h2>
              <button className="btn" type="button" onClick={commitEditor}>
                Done
              </button>
            </div>

            <div className="field-grid">
              <div className="field">
                <label htmlFor="location">Current location</label>
                <select
                  id="location"
                  value={edit.location}
                  onChange={(e) =>
                    setDraft((d) => (d ? { ...d, location: e.target.value } : d))
                  }
                >
                  <optgroup label="Hubs">
                    {catalog.hubs.map((h) => (
                      <option key={h.key} value={h.key}>
                        {h.name}
                      </option>
                    ))}
                  </optgroup>
                  <optgroup label="Rides">
                    {catalog.rides.map((r) => (
                      <option key={r.location_key} value={r.location_key}>
                        {r.name}
                      </option>
                    ))}
                  </optgroup>
                </select>
              </div>

              <div className="field">
                <label htmlFor="party">Party size</label>
                <input
                  id="party"
                  type="number"
                  min={1}
                  max={16}
                  value={edit.party_size}
                  onChange={(e) =>
                    setDraft((d) =>
                      d
                        ? {
                            ...d,
                            party_size: Math.min(16, Math.max(1, Number(e.target.value) || 1)),
                          }
                        : d,
                    )
                  }
                />
              </div>

              <div className="field">
                <label htmlFor="leave">Leaving around (hour)</label>
                <input
                  id="leave"
                  type="number"
                  min={catalog.day_start_hour}
                  max={23}
                  step={0.5}
                  value={edit.leave_hour ?? catalog.day_end_hour}
                  onChange={(e) =>
                    setDraft((d) =>
                      d ? { ...d, leave_hour: Number(e.target.value) } : d,
                    )
                  }
                />
              </div>
            </div>

            <section className="section">
              <h2>Preferences & completions</h2>
              {sortedRides.map((ride) => (
                <div className="pref-row" key={ride.id}>
                  <div className="pref-top">
                    <strong>{ride.name}</strong>
                    <span className="ride-meta">
                      {waitLabel(
                        waitById.get(ride.id)?.wait_min,
                        waitById.get(ride.id)?.open,
                        waitById.get(ride.id)?.status,
                      )}
                    </span>
                  </div>
                  <div className="pref-controls">
                    <input
                      type="range"
                      min={0}
                      max={catalog.weight_slider_max}
                      value={edit.preference_weights[ride.id]}
                      onChange={(e) => {
                        const value = Number(e.target.value);
                        setDraft((d) => {
                          if (!d) return d;
                          const preference_weights = [...d.preference_weights];
                          preference_weights[ride.id] = value;
                          return { ...d, preference_weights };
                        });
                      }}
                    />
                    <button
                      type="button"
                      className={`must-btn${edit.must_dos[ride.id] ? " on" : ""}`}
                      onClick={() =>
                        setDraft((d) => {
                          if (!d) return d;
                          const must_dos = [...d.must_dos];
                          must_dos[ride.id] = must_dos[ride.id] ? 0 : 1;
                          return { ...d, must_dos };
                        })
                      }
                    >
                      Must
                    </button>
                    <div className="stepper" aria-label={`Times ridden ${ride.name}`}>
                      <button
                        type="button"
                        onClick={() =>
                          setDraft((d) => {
                            if (!d) return d;
                            const history = [...d.history];
                            history[ride.id] = Math.max(0, history[ride.id] - 1);
                            return { ...d, history };
                          })
                        }
                      >
                        −
                      </button>
                      <span>{edit.history[ride.id]}</span>
                      <button
                        type="button"
                        onClick={() =>
                          setDraft((d) => {
                            if (!d) return d;
                            const history = [...d.history];
                            history[ride.id] = Math.min(20, history[ride.id] + 1);
                            return { ...d, history };
                          })
                        }
                      >
                        +
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </section>
          </div>
        </div>
      )}
    </div>
  );
}
