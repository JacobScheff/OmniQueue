export type Theme = "light" | "dark";

const THEME_KEY = "omniqueue-companion-theme";

const THEME_COLOR: Record<Theme, string> = {
  dark: "#0b0e13",
  light: "#f5f6fa",
};

export function getInitialTheme(): Theme {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    // localStorage unavailable (private mode) — fall through to system preference.
  }
  const prefersLight =
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-color-scheme: light)").matches;
  return prefersLight ? "light" : "dark";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", THEME_COLOR[theme]);
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Ignore write failures (private mode / storage full).
  }
}
