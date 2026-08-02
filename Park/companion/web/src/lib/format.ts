export type WaitLevel = "good" | "warn" | "bad" | "closed" | "unknown";

export function formatProb(p: number): string {
  return `${(p * 100).toFixed(0)}%`;
}

export function waitLevel(
  wait: number | null | undefined,
  open?: boolean,
  status?: string,
): WaitLevel {
  const isDown = status && status !== "OPERATING" && status !== "UNKNOWN";
  if (isDown || open === false) return "closed";
  if (wait == null) return "unknown";
  if (wait <= 20) return "good";
  if (wait <= 45) return "warn";
  return "bad";
}

export function waitLabel(
  wait: number | null | undefined,
  open?: boolean,
  status?: string,
): string {
  if (status && status !== "OPERATING" && status !== "UNKNOWN") return statusLabel(status);
  if (open === false) return "Closed";
  if (wait == null) return "—";
  return `${Math.round(wait)} min`;
}

function statusLabel(status: string): string {
  switch (status) {
    case "DOWN":
      return "Down";
    case "CLOSED":
      return "Closed";
    case "REFURBISHMENT":
      return "Refurb.";
    default:
      return status;
  }
}

export function formatHour(hour: number): string {
  const h = Math.floor(hour);
  const m = Math.round((hour - h) * 60);
  const period = h >= 12 ? "PM" : "AM";
  const h12 = ((h + 11) % 12) + 1;
  return m === 0 ? `${h12}:00 ${period}` : `${h12}:${String(m).padStart(2, "0")} ${period}`;
}
