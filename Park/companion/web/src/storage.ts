import type { Catalog, UserState } from "./api";

const STORAGE_KEY = "omniqueue-companion-v1";
const HISTORY_KEY = "omniqueue-companion-undo-v1";
const MAX_UNDO = 50;

type StoredBundle = {
  state: UserState;
  past: UserState[];
  future: UserState[];
};

function cloneState(state: UserState): UserState {
  return {
    preference_weights: [...state.preference_weights],
    must_dos: [...state.must_dos],
    history: [...state.history],
    location: state.location,
    leave_hour: state.leave_hour,
    arrival_hour: state.arrival_hour,
    party_size: state.party_size,
    model_version: state.model_version,
  };
}

export function defaultUserState(catalog: Catalog): UserState {
  const n = catalog.num_rides;
  return {
    preference_weights: [...catalog.default_preference_weights],
    must_dos: Array(n).fill(0),
    history: Array(n).fill(0),
    location: "entrance",
    leave_hour: catalog.day_end_hour,
    arrival_hour: catalog.day_start_hour,
    party_size: 2,
    model_version: catalog.default_model_version,
  };
}

export function loadBundle(catalog: Catalog): StoredBundle {
  const fallback: StoredBundle = {
    state: defaultUserState(catalog),
    past: [],
    future: [],
  };
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<StoredBundle> & { state?: UserState };
    if (!parsed.state) return fallback;
    const n = catalog.num_rides;
    const state = normalizeState(parsed.state, n, catalog);
    const past = Array.isArray(parsed.past)
      ? parsed.past.map((s) => normalizeState(s, n, catalog)).slice(-MAX_UNDO)
      : [];
    const future = Array.isArray(parsed.future)
      ? parsed.future.map((s) => normalizeState(s, n, catalog)).slice(-MAX_UNDO)
      : [];
    return { state, past, future };
  } catch {
    return fallback;
  }
}

function normalizeState(state: UserState, n: number, catalog: Catalog): UserState {
  const pad = (arr: number[] | undefined, fill: number) => {
    const base = Array.isArray(arr) ? arr.slice(0, n) : [];
    while (base.length < n) base.push(fill);
    return base;
  };
  const known = new Set(catalog.models.map((m) => m.id));
  const version =
    state.model_version && known.has(state.model_version)
      ? state.model_version
      : catalog.default_model_version;
  return {
    preference_weights: pad(state.preference_weights, 1),
    must_dos: pad(state.must_dos, 0).map((v) => (v ? 1 : 0)),
    history: pad(state.history, 0).map((v) => Math.max(0, Math.floor(v))),
    location: state.location || "entrance",
    leave_hour: state.leave_hour ?? catalog.day_end_hour,
    arrival_hour: state.arrival_hour ?? catalog.day_start_hour,
    party_size: Math.min(16, Math.max(1, state.party_size || 2)),
    model_version: version,
  };
}

export function saveBundle(bundle: StoredBundle): void {
  const payload: StoredBundle = {
    state: cloneState(bundle.state),
    past: bundle.past.slice(-MAX_UNDO).map(cloneState),
    future: bundle.future.slice(-MAX_UNDO).map(cloneState),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  localStorage.setItem(HISTORY_KEY, String(payload.past.length));
}

export function pushEdit(bundle: StoredBundle, next: UserState): StoredBundle {
  return {
    state: cloneState(next),
    past: [...bundle.past, cloneState(bundle.state)].slice(-MAX_UNDO),
    future: [],
  };
}

export function undo(bundle: StoredBundle): StoredBundle {
  if (bundle.past.length === 0) return bundle;
  const prev = bundle.past[bundle.past.length - 1];
  return {
    state: cloneState(prev),
    past: bundle.past.slice(0, -1),
    future: [cloneState(bundle.state), ...bundle.future].slice(0, MAX_UNDO),
  };
}

export function redo(bundle: StoredBundle): StoredBundle {
  if (bundle.future.length === 0) return bundle;
  const next = bundle.future[0];
  return {
    state: cloneState(next),
    past: [...bundle.past, cloneState(bundle.state)].slice(-MAX_UNDO),
    future: bundle.future.slice(1),
  };
}

export type { StoredBundle };
