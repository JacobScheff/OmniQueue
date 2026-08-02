import type { Catalog, UserState } from "../lib/api";
import { formatHour } from "../lib/format";
import { RedoIcon, UndoIcon } from "./Icons";

type MeViewProps = {
  catalog: Catalog;
  state: UserState;
  onCommit: (next: UserState) => void;
  onModelChange: (version: string) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onReset: () => void;
};

export default function MeView({
  catalog,
  state,
  onCommit,
  onModelChange,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onReset,
}: MeViewProps) {
  return (
    <div className="view me-view">
      <section className="section">
        <div className="section-head">
          <h2>Where you are</h2>
        </div>
        <div className="field-card">
          <label htmlFor="location">Current location</label>
          <select
            id="location"
            value={state.location}
            onChange={(e) => onCommit({ ...state, location: e.target.value })}
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

        <div className="field-card two-up">
          <div>
            <label htmlFor="arrival">Arrived around</label>
            <input
              id="arrival"
              type="number"
              min={catalog.day_start_hour}
              max={23}
              step={0.5}
              value={state.arrival_hour ?? catalog.day_start_hour}
              onChange={(e) => onCommit({ ...state, arrival_hour: Number(e.target.value) })}
            />
            <span className="field-hint">{formatHour(state.arrival_hour ?? catalog.day_start_hour)}</span>
          </div>
          <div>
            <label htmlFor="leave">Leaving around</label>
            <input
              id="leave"
              type="number"
              min={catalog.day_start_hour}
              max={23}
              step={0.5}
              value={state.leave_hour ?? catalog.day_end_hour}
              onChange={(e) => onCommit({ ...state, leave_hour: Number(e.target.value) })}
            />
            <span className="field-hint">{formatHour(state.leave_hour ?? catalog.day_end_hour)}</span>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Model</h2>
          <p className="section-sub">Which trained policy plans your route.</p>
        </div>
        <ul className="model-cards">
          {catalog.models.map((m) => {
            const active = state.model_version === m.id;
            return (
              <li key={m.id}>
                <button
                  type="button"
                  className={`model-card${active ? " active" : ""}`}
                  onClick={() => onModelChange(m.id)}
                >
                  <span className="model-card-top">
                    <strong>{m.label}</strong>
                    {m.stub && <span className="tag warn">Stub weights</span>}
                  </span>
                  <span className="model-card-meta">
                    {m.step ? `Trained to step ${m.step.toLocaleString()}` : "Untrained placeholder"}
                    {m.supports_force_any_slot
                      ? " · can pin any stop"
                      : m.supports_force_first
                        ? " · can pin only the first stop"
                        : ""}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>History</h2>
        </div>
        <div className="history-actions">
          <button type="button" className="btn ghost" onClick={onUndo} disabled={!canUndo}>
            <UndoIcon size={15} /> Undo
          </button>
          <button type="button" className="btn ghost" onClick={onRedo} disabled={!canRedo}>
            <RedoIcon size={15} /> Redo
          </button>
          <button type="button" className="btn ghost danger" onClick={onReset}>
            Reset to defaults
          </button>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>What these mean</h2>
        </div>
        <dl className="help-list">
          <dt>Priority</dt>
          <dd>How much you want to ride something. Higher priority nudges the model to route you there sooner.</dd>
          <dt>Must-do</dt>
          <dd>Rides you don't want to miss today. The model works these into your route before it's done.</dd>
          <dt>Done</dt>
          <dd>How many times you've already ridden something. Rides marked done won't be recommended again.</dd>
          <dt>Pinning a stop</dt>
          <dd>
            On the Plan tab, tap any stop in your route to see alternatives — tap one to force it into that
            exact spot. Everything else in the route still re-plans around your pin.
          </dd>
        </dl>
      </section>
    </div>
  );
}
