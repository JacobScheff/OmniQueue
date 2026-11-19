"""Causal outage-duration estimates learned only from training observations."""

import torch

from ml.data import DOWN, OVERFLOW, TICK_SECONDS, align_day


def fit_breakdown_hazards(days, rides, bins=48, prior_strength=5):
    """Estimate P(reopen next tick | still DOWN, elapsed outage ticks).

    Known operating -> DOWN onsets establish age. Opening outages are left
    censored; unknown/interpolated ticks and session breaks end coverage.
    DOWN -> CLOSED is censored, never mislabeled as a reopening. The final
    bin pools the tail. Sparse rides shrink toward the park's age-specific rate.
    """
    risk = torch.zeros(len(rides), bins, dtype=torch.float64)
    events = torch.zeros_like(risk)
    for day in days:
        tokens = align_day(day, rides)
        ages = torch.zeros(len(rides), dtype=torch.long)
        interpolated = day.get("interpolated_ticks", {})
        for i in range(1, len(tokens)):
            before, after = tokens[i - 1], tokens[i]
            continuous = (day["timestamps"][i] - day["timestamps"][i - 1] == TICK_SECONDS
                          and str(day["timestamps"][i - 1]) not in interpolated
                          and str(day["timestamps"][i]) not in interpolated)
            if not continuous:
                ages.zero_()
                continue
            eligible = (ages > 0) & (before == DOWN) & (after >= 0) & (after <= DOWN)
            columns = eligible.nonzero().flatten()
            indices = (ages[columns] - 1).clamp_max(bins - 1)
            risk[columns, indices] += 1
            events[columns, indices] += (after[columns] <= OVERFLOW).double()
            onset = (before >= 0) & (before <= OVERFLOW) & (after == DOWN)
            ages = torch.where(onset, 1, torch.where(eligible & (after == DOWN), ages + 1, 0))
    if not risk.sum():
        return None  # No evidence: leave the neural distribution alone.
    overall = events.sum() / risk.sum()
    pooled = (events.sum(0) + prior_strength * overall) / (risk.sum(0) + prior_strength)
    hazards = (events + prior_strength * pooled) / (risk + prior_strength)
    return hazards.float().tolist()


def advance_ages(ages, tokens):
    """Update elapsed DOWN ticks for observed or generated states."""
    return torch.where(tokens == DOWN, ages + 1, 0)


def history_ages(history, timestamps=None):
    ages = torch.zeros(history.shape[-1], dtype=torch.long, device=history.device)
    for i, tokens in enumerate(history):
        if timestamps is not None and i and timestamps[i] - timestamps[i - 1] != TICK_SECONDS:
            ages.zero_()
        ages = advance_ages(ages, tokens)
    return ages
