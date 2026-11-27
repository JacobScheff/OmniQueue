"""Causal outage-duration estimates learned only from training observations."""

import torch

from ml.data import CLOSED, DOWN, OVERFLOW, REFURBISHMENT, UNKNOWN, TICK_SECONDS, align_day


# Bump when inference/decoding changes so saved test reports cannot silently
# describe an older forecast policy with the same checkpoint weights.
FORECAST_POLICY_VERSION = 2
UNAVAILABLE_ENTER = .65
UNAVAILABLE_EXIT = .35


def wait_quantiles(probabilities, quantiles=(.1, .5, .9)):
    """Inverse CDF of waits conditional on operating, including overflow.

    Accept [..., vocab] and return [..., quantiles]. A zero-mass conditional
    distribution has no wait; callers must show the unavailable/locked status.
    """
    cdf = probabilities[..., :DOWN].cumsum(-1).contiguous()
    levels = torch.as_tensor(quantiles, device=cdf.device, dtype=cdf.dtype)
    values = (cdf[..., -1:] * levels).contiguous()
    return torch.searchsorted(cdf, values).clamp_max(OVERFLOW)


def decode_availability(probabilities, previous):
    """Decode one forecast tick with hysteresis, seeded by the last observation.

    Availability competes against the *sum* of operating wait probabilities,
    not each individual wait bucket. Ambiguous predictions retain the previous
    availability; strong evidence can still close/reopen within a single tick.
    This is causal over forecast leads: appending more leads changes no prefix.
    Raw probabilities remain untouched and are reported separately.
    """
    numeric = wait_quantiles(probabilities, (.5,)).squeeze(-1)
    return _decode_availability(probabilities, previous, numeric)


def decode_availability_sequence(probabilities, previous):
    """Decode [batch, leads, rides, vocab], batching conditional wait quantiles.

    Only the cheap availability state machine needs a sequential scan. Its
    state is per origin/ride and can continue across bounded lead chunks.
    """
    numeric = wait_quantiles(probabilities, (.5,)).squeeze(-1)
    predictions = []
    for step in range(probabilities.shape[1]):
        previous = _decode_availability(probabilities[:, step], previous, numeric[:, step])
        predictions.append(previous)
    return torch.stack(predictions, dim=1)


def _decode_availability(probabilities, previous, numeric):
    down, closed = probabilities[..., DOWN], probabilities[..., CLOSED]
    unavailable = down + closed
    was_unavailable = (previous == DOWN) | (previous == CLOSED)
    is_unavailable = torch.where(was_unavailable, unavailable > UNAVAILABLE_EXIT,
                                 unavailable >= UNAVAILABLE_ENTER)
    status = torch.where(closed >= down, CLOSED, DOWN)
    # Apply the same evidence margin before changing DOWN <-> CLOSED labels.
    status = torch.where((previous == DOWN) & (closed < UNAVAILABLE_ENTER * unavailable), DOWN, status)
    status = torch.where((previous == CLOSED) & (down < UNAVAILABLE_ENTER * unavailable), CLOSED, status)
    prediction = torch.where(is_unavailable, status, numeric)
    for locked in (REFURBISHMENT, UNKNOWN):
        prediction = torch.where(probabilities[..., locked] == 1, locked, prediction)
    return prediction


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
