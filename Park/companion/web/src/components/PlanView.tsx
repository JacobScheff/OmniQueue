import { useMemo, useState } from "react";
import type { RecommendResponse, UserState, WaitRow, ForcedPick } from "../lib/api";
import { formatProb, waitLabel, waitLevel } from "../lib/format";
import { ChevronIcon, InfoIcon, RefreshIcon } from "./Icons";
import RouteTimeline from "./RouteTimeline";

type PlanViewProps = {
  result: RecommendResponse | null;
  busy: boolean;
  state: UserState;
  waitById: Map<number, WaitRow>;
  topPrefIds: number[];
  forcedPick: ForcedPick | null;
  onForce: (slot: number, actionId: number) => void;
  onClearForce: () => void;
  onRefreshWaits: () => void;
};

function confidencePhrase(prob: number): string {
  if (prob >= 0.66) return "Confident pick";
  if (prob >= 0.4) return "Leaning this way";
  return "Close call";
}

function buildWhy(params: {
  result: RecommendResponse;
  state: UserState;
  waitById: Map<number, WaitRow>;
  topPrefIds: number[];
  isForced: boolean;
}): string[] {
  const { result, state, waitById, topPrefIds, isForced } = params;
  const reasons: string[] = [];
  const rec = result.recommended;
  const recIsRide = waitById.has(rec.action_id);
  if (isForced) {
    reasons.push("You pinned this stop yourself — the rest of the plan re-routes around it.");
    return reasons;
  }
  if (!recIsRide) {
    reasons.push(
      rec.label.toLowerCase().includes("exit")
        ? "The model thinks it's time to head out given how much daylight is left."
        : "The model thinks waiting a moment beats moving right now.",
    );
    return reasons;
  }
  const live = waitById.get(rec.action_id);
  if (live) reasons.push(`Current wait is ${waitLabel(live.wait_min, live.open, live.status)}.`);
  if (state.must_dos[rec.action_id] > 0) reasons.push("It's on your must-do list.");
  if (topPrefIds.includes(rec.action_id)) reasons.push("It's one of your top preferences.");
  if (
    result.natural_recommended &&
    result.natural_recommended.action_id !== rec.action_id
  ) {
    reasons.push(`Runner-up was ${result.natural_recommended.label}, close behind.`);
  }
  if (reasons.length === 0) reasons.push("Best balance of wait time and your preferences right now.");
  return reasons;
}

export default function PlanView({
  result,
  busy,
  state,
  waitById,
  topPrefIds,
  forcedPick,
  onForce,
  onClearForce,
  onRefreshWaits,
}: PlanViewProps) {
  const [whyOpen, setWhyOpen] = useState(false);

  const isForced = forcedPick?.slot === 0;
  const why = useMemo(() => {
    if (!result) return [];
    return buildWhy({ result, state, waitById, topPrefIds, isForced });
  }, [result, state, waitById, topPrefIds, isForced]);

  const canForceSlot = (slot: number): boolean => {
    if (!result) return false;
    if (result.model.supports_force_first === false) return false;
    if (slot > 0 && result.model.supports_force_any_slot !== true) return false;
    return true;
  };

  const rec = result?.recommended;
  const recIsRide = rec != null && waitById.has(rec.action_id);
  const liveRec = recIsRide && rec ? waitById.get(rec.action_id) : undefined;
  const recLevel = recIsRide ? waitLevel(liveRec?.wait_min, liveRec?.open, liveRec?.status) : "unknown";

  return (
    <div className="view plan-view">
      <section className="hero">
        <div className="hero-top">
          <p className="eyebrow">{isForced ? "Pinned next stop" : "Up next"}</p>
          {recIsRide && (
            <span className={`wait-chip level-${recLevel}`}>
              {waitLabel(liveRec?.wait_min, liveRec?.open, liveRec?.status)}
            </span>
          )}
        </div>
        <h1 className="hero-title">
          {result ? result.recommended.label : busy ? "Thinking…" : "—"}
        </h1>
        {result && (
          <p className="hero-sub">
            {confidencePhrase(result.recommended.prob)} · {formatProb(result.recommended.prob)} confidence
          </p>
        )}

        {result && (
          <div className="why-box">
            <button type="button" className="why-toggle" onClick={() => setWhyOpen((v) => !v)}>
              <InfoIcon size={14} />
              Why this pick?
              <ChevronIcon direction={whyOpen ? "down" : "right"} size={13} />
            </button>
            {whyOpen && (
              <ul className="why-list">
                {why.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {isForced && (
          <div className="force-banner">
            <span>Pinned by you — tap “Clear pin” to let the model choose again.</span>
            <button type="button" className="btn ghost sm" onClick={onClearForce} disabled={busy}>
              Clear pin
            </button>
          </div>
        )}

        <div className="hero-actions">
          <button type="button" className="btn ghost" onClick={onRefreshWaits} disabled={busy}>
            <RefreshIcon size={15} /> Refresh waits
          </button>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Your route</h2>
          <p className="section-sub">Tap any stop to see — or pin — an alternative.</p>
        </div>
        {result?.route && result.route.length > 0 ? (
          <RouteTimeline
            route={result.route}
            distributionsBySlot={result.distributions_by_slot ?? []}
            waitById={waitById}
            forcedPick={forcedPick}
            canForceSlot={canForceSlot}
            onForce={onForce}
            busy={busy}
          />
        ) : (
          <p className="empty-hint">{busy ? "Building your plan…" : "No plan yet."}</p>
        )}
      </section>
    </div>
  );
}
