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
};

export default function TopBar({ theme, onToggleTheme, pills }: TopBarProps) {
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
