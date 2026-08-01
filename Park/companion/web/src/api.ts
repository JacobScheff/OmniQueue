export type RideInfo = {
  id: number;
  name: string;
  hub_id: number;
  hub_name: string;
  location_key: string;
  popularity: number;
  duration_min: number;
};

export type HubInfo = {
  id: number;
  key: string;
  name: string;
  kind: string;
};

export type ModelInfo = {
  id: string;
  label: string;
  version: string | null;
  path: string;
  step: number;
  stub: boolean;
  device: string;
  available: boolean;
};

export type Catalog = {
  num_rides: number;
  weight_slider_max: number;
  default_preference_weights: number[];
  day_start_hour: number;
  day_end_hour: number;
  hubs: HubInfo[];
  rides: RideInfo[];
  default_model_version: string;
  models: ModelInfo[];
};

export type WaitRow = {
  ride_id: number;
  name: string;
  wait_min: number | null;
  status: string;
  open: boolean;
  entity_id: string | null;
};

export type DistRow = {
  action_id: number;
  label: string;
  prob: number;
  legal: boolean;
  is_ride: boolean;
  wait_min?: number | null;
  status?: string;
  open?: boolean;
};

export type RouteStop = {
  action_id: number;
  label: string;
  slot: number;
  is_ride: boolean;
};

export type RecommendResponse = {
  recommended: {
    action_id: number;
    label: string;
    prob: number;
    legal: boolean;
  };
  route?: RouteStop[];
  distribution: DistRow[];
  model: {
    version: string | null;
    path: string;
    step: number;
    stub: boolean;
    device: string;
    available: boolean;
  };
  meta: {
    warnings: string[];
    mean_wait_min: number;
    open_rides: number;
    must_remaining: number[];
  };
  waits_fetched_at: number;
  waits_error: string | null;
  now_sec: number;
};

export type UserState = {
  preference_weights: number[];
  must_dos: number[];
  history: number[];
  location: string;
  leave_hour: number | null;
  arrival_hour: number | null;
  party_size: number;
  model_version: string;
};

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${url} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function fetchCatalog(): Promise<Catalog> {
  return getJson("/api/catalog");
}

export function fetchWaits(force = false): Promise<{ rides: WaitRow[]; fetched_at: number; error: string | null }> {
  return getJson(`/api/waits?force=${force ? "true" : "false"}`);
}

export async function postRecommend(
  state: UserState,
  forceRefreshWaits = false,
): Promise<RecommendResponse> {
  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      preference_weights: state.preference_weights,
      must_dos: state.must_dos,
      history: state.history,
      location: state.location,
      leave_hour: state.leave_hour,
      arrival_hour: state.arrival_hour,
      party_size: state.party_size,
      model_version: state.model_version,
      force_refresh_waits: forceRefreshWaits,
    }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || `recommend failed: ${res.status}`);
  }
  return res.json() as Promise<RecommendResponse>;
}
