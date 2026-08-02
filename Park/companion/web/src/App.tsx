import { useEffect, useMemo, useState } from "react";
import MeView from "./components/MeView";
import PlanView from "./components/PlanView";
import RidesView from "./components/RidesView";
import TabNav, { type TabId } from "./components/TabNav";
import TopBar, { type StatusPill } from "./components/TopBar";
import {
  type Catalog,
  type ForcedPick,
  type RecommendResponse,
  type UserState,
  type WaitRow,
  fetchCatalog,
  fetchWaits,
  postRecommend,
} from "./lib/api";
import { applyTheme, getInitialTheme, type Theme } from "./lib/theme";
import {
  defaultUserState,
  loadBundle,
  pushEdit,
  redo,
  saveBundle,
  undo,
  type StoredBundle,
} from "./lib/storage";

export default function App() {
  const [theme, setTheme] = useState<Theme>(() => getInitialTheme());
  const [activeTab, setActiveTab] = useState<TabId>("plan");
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [bundle, setBundle] = useState<StoredBundle | null>(null);
  const [waits, setWaits] = useState<WaitRow[]>([]);
  const [waitsError, setWaitsError] = useState<string | null>(null);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [forcedPick, setForcedPick] = useState<ForcedPick | null>(null);

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

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
        if (!cancelled) {
          setWaits(board.rides);
          setWaitsError(board.error);
        }
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
        const rec = await postRecommend(bundle.state, false, forcedPick);
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
  }, [bundle, forcedPick]);

  const waitById = useMemo(() => {
    const map = new Map<number, WaitRow>();
    for (const w of waits) map.set(w.ride_id, w);
    return map;
  }, [waits]);

  const topPrefIds = useMemo(() => {
    if (!catalog || !bundle) return [];
    return [...catalog.rides]
      .filter((r) => bundle.state.preference_weights[r.id] > 0)
      .sort((a, b) => bundle.state.preference_weights[b.id] - bundle.state.preference_weights[a.id])
      .slice(0, 3)
      .map((r) => r.id);
  }, [catalog, bundle]);

  const commit = (next: UserState) => {
    setBundle((b) => (b ? pushEdit(b, next) : b));
  };

  const onModelChange = (version: string) => {
    setForcedPick(null);
    setBundle((prev) => {
      if (!prev || prev.state.model_version === version) return prev;
      return pushEdit(prev, { ...prev.state, model_version: version });
    });
  };

  const onForce = (slot: number, actionId: number) => {
    setForcedPick((prev) => (prev && prev.slot === slot && prev.actionId === actionId ? null : { slot, actionId }));
  };

  const onReset = () => {
    if (!catalog) return;
    setForcedPick(null);
    commit(defaultUserState(catalog));
  };

  const onRefreshWaits = async () => {
    if (!bundle) return;
    setBusy(true);
    setError(null);
    try {
      const board = await fetchWaits(true);
      setWaits(board.rides);
      setWaitsError(board.error);
      const rec = await postRecommend(bundle.state, true, forcedPick);
      setResult(rec);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  if (!catalog || !bundle) {
    return (
      <div className="app app-loading">
        <p className="loading">{error ? error : "Loading companion…"}</p>
      </div>
    );
  }

  const state = bundle.state;
  const activeModel = catalog.models.find((m) => m.id === state.model_version);

  const pills: StatusPill[] = [
    { label: busy ? "Updating…" : "Live waits" },
    ...(result
      ? [{ label: `${result.meta.open_rides} open · avg ${Math.round(result.meta.mean_wait_min)} min` }]
      : []),
    ...(activeModel ? [{ label: `Model ${activeModel.label}` }] : []),
    ...(activeModel?.stub ? [{ label: "Stub weights — replace checkpoint", tone: "warn" as const }] : []),
    ...(waitsError ? [{ label: "Wait feed issue", tone: "bad" as const }] : []),
  ];

  return (
    <div className="app">
      <TopBar theme={theme} onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))} pills={pills} />

      {error && <div className="error">{error}</div>}

      <main className="content">
        {activeTab === "plan" && (
          <PlanView
            result={result}
            busy={busy}
            state={state}
            waitById={waitById}
            topPrefIds={topPrefIds}
            forcedPick={forcedPick}
            onForce={onForce}
            onClearForce={() => setForcedPick(null)}
            onRefreshWaits={() => void onRefreshWaits()}
          />
        )}
        {activeTab === "rides" && (
          <RidesView catalog={catalog} state={state} waitById={waitById} onCommit={commit} />
        )}
        {activeTab === "me" && (
          <MeView
            catalog={catalog}
            state={state}
            onCommit={commit}
            onModelChange={onModelChange}
            canUndo={bundle.past.length > 0}
            canRedo={bundle.future.length > 0}
            onUndo={() => setBundle((b) => (b ? undo(b) : b))}
            onRedo={() => setBundle((b) => (b ? redo(b) : b))}
            onReset={onReset}
          />
        )}
      </main>

      <TabNav active={activeTab} onChange={setActiveTab} />
    </div>
  );
}
