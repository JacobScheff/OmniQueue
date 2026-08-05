import type { ModelInfo } from "../lib/api";
import type { Theme } from "../lib/theme";
import { MoonIcon, SunIcon } from "./Icons";

export type StatusPill = {
  label: string;
  tone?: "warn" | "bad";
};

type TopBarProps = {
  theme: Theme;
  onToggleTheme: () => void;
  pills: StatusPill[];
  models: ModelInfo[];
  modelVersion: string;
  onModelChange: (version: string) => void;
};

export default function TopBar({
  theme,
  onToggleTheme,
  pills,
  models,
  modelVersion,
  onModelChange,
}: TopBarProps) {
  return (
    <header className="topbar">
      <div className="topbar-row">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            OQ
          </span>
          <div>
            <h1>OmniQueue</h1>
            <p>Your live Disneyland co-pilot</p>
          </div>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={onToggleTheme}
          aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? <SunIcon /> : <MoonIcon />}
        </button>
      </div>
      {models.length > 0 && (
        <div className="model-switch" role="group" aria-label="Model version">
          {models.map((m) => {
            const active = modelVersion === m.id;
            return (
              <button
                key={m.id}
                type="button"
                className={`model-switch-btn${active ? " active" : ""}`}
                aria-pressed={active}
                title={m.stub ? `${m.label} (stub weights)` : m.label}
                onClick={() => onModelChange(m.id)}
              >
                {m.label}
              </button>
            );
          })}
        </div>
      )}
      {pills.length > 0 && (
        <div className="pill-row">
          {pills.map((p) => (
            <span key={p.label} className={`pill${p.tone ? ` ${p.tone}` : ""}`}>
              {p.label}
            </span>
          ))}
        </div>
      )}
    </header>
  );
}
