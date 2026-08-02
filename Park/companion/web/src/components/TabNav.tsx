import type { ComponentType } from "react";
import { MeIcon, PlanIcon, RidesIcon } from "./Icons";

export type TabId = "plan" | "rides" | "me";

type TabNavProps = {
  active: TabId;
  onChange: (tab: TabId) => void;
};

const TABS: { id: TabId; label: string; icon: ComponentType<{ size?: number }> }[] = [
  { id: "plan", label: "Plan", icon: PlanIcon },
  { id: "rides", label: "Rides", icon: RidesIcon },
  { id: "me", label: "Me", icon: MeIcon },
];

export default function TabNav({ active, onChange }: TabNavProps) {
  return (
    <nav className="tabbar" aria-label="Sections">
      {TABS.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          type="button"
          className={`tab-btn${active === id ? " active" : ""}`}
          onClick={() => onChange(id)}
          aria-current={active === id ? "page" : undefined}
        >
          <Icon size={20} />
          <span>{label}</span>
        </button>
      ))}
    </nav>
  );
}
