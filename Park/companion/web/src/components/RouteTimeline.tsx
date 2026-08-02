import { useState } from "react";
import type { DistRow, ForcedPick, RouteStop, WaitRow } from "../lib/api";
import { formatProb, waitLabel, waitLevel } from "../lib/format";
import { ChevronIcon, CloseIcon, PinIcon } from "./Icons";

type RouteTimelineProps = {
  route: RouteStop[];
  distributionsBySlot: DistRow[][];
  waitById: Map<number, WaitRow>;
  forcedPick: ForcedPick | null;
  canForceSlot: (slot: number) => boolean;
  onForce: (slot: number, actionId: number) => void;
  busy: boolean;
};

const STOP_WORDS = ["Now", "Next", "Then", "Then", "Then", "Then", "Then"];

function stopLabel(slot: number): string {
  return STOP_WORDS[slot] ?? "Then";
}

export default function RouteTimeline({
  route,
  distributionsBySlot,
  waitById,
  forcedPick,
  canForceSlot,
  onForce,
  busy,
}: RouteTimelineProps) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);

  if (route.length === 0) return null;

  return (
    <ol className="timeline" aria-label="Planned route">
      {route.map((stop) => {
        const isOpen = expanded === stop.slot;
        const isForced = forcedPick?.slot === stop.slot;
        const live = stop.is_ride ? waitById.get(stop.action_id) : undefined;
        const level = stop.is_ride ? waitLevel(live?.wait_min, live?.open, live?.status) : "unknown";
        const forceable = canForceSlot(stop.slot);
        const dist = distributionsBySlot[stop.slot] ?? [];
        const visible = showAll ? dist : dist.filter((d) => d.legal).slice(0, 6);

        return (
          <li key={stop.slot} className={`stop${isOpen ? " open" : ""}`}>
            <button
              type="button"
              className={`stop-row${stop.slot === 0 ? " primary" : ""}${isForced ? " forced" : ""}`}
              onClick={() => {
                setExpanded((s) => (s === stop.slot ? null : stop.slot));
                setShowAll(false);
              }}
              aria-expanded={isOpen}
            >
              <span className="stop-dot" data-level={level} aria-hidden="true" />
              <span className="stop-main">
                <span className="stop-label">{stopLabel(stop.slot)}</span>
                <span className="stop-name">
                  {stop.label}
                  {isForced && (
                    <span className="pin-badge">
                      <PinIcon size={11} /> Pinned
                    </span>
                  )}
                </span>
              </span>
              {stop.is_ride && (
                <span className={`wait-chip level-${level}`}>
                  {waitLabel(live?.wait_min, live?.open, live?.status)}
                </span>
              )}
              <ChevronIcon direction={isOpen ? "down" : "right"} className="stop-chevron" />
            </button>

            {isOpen && (
              <div className="stop-alts">
                <p className="alts-hint">
                  {forceable
                    ? "Tap another option to pin it here — the rest of the plan will re-route around it."
                    : "This model can’t pin this stop. Alternatives shown for reference only."}
                </p>
                <ul className="alts-list">
                  {visible.map((row) => {
                    const active = forcedPick?.slot === stop.slot && forcedPick.actionId === row.action_id;
                    const clickable = forceable && row.legal;
                    return (
                      <li key={row.action_id}>
                        <button
                          type="button"
                          className={`alt-item${row.legal ? "" : " illegal"}${active ? " active" : ""}${
                            clickable ? " clickable" : ""
                          }`}
                          disabled={!clickable || busy}
                          onClick={() => onForce(stop.slot, row.action_id)}
                        >
                          <span className="alt-name">{row.label}</span>
                          {row.is_ride && (
                            <span className={`wait-chip small level-${waitLevel(row.wait_min, row.open, row.status)}`}>
                              {waitLabel(row.wait_min, row.open, row.status)}
                            </span>
                          )}
                          <span className="alt-prob">{formatProb(row.prob)}</span>
                          {active && <CloseIcon size={12} className="alt-clear" />}
                        </button>
                      </li>
                    );
                  })}
                </ul>
                {dist.length > 6 && (
                  <button type="button" className="link-btn" onClick={() => setShowAll((v) => !v)}>
                    {showAll ? "Show top options" : "Show all options"}
                  </button>
                )}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
