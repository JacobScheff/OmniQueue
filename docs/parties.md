# Parties

**Module:** `parties.py`, `park_types.py`

## Overview

Guests are grouped into **parties**. The simulator uses struct-of-arrays via a `PartyPool` containing `Party` dataclass instances.

## Spawn Model

| Parameter | Default |
|-----------|---------|
| Total guests/day | ~50,000 ± 2,500 |
| Party size | `max(1, round(N(3.2, 1.0)))`, no cap |
| Arrival peak | ~11:00 AM (bell curve) |
| Dwell time | Mean **10 h**, σ = 2 h, min 2 h |

## Party Speed

Each party draws per-member speeds from a log-normal distribution and uses the **minimum** (right-skewed, modeling slowest-member pace):

```python
effective_speed = min(lognormal(member_speeds))
```

## Preferences

- Random uniform weights per ride, normalized to sum ≈ 1.
- **Not land-themed.**
- Must-do rides receive a `MUST_DO_PREF_BOOST` multiplier.

## Must-Do Lists

- Count per party: uniform **0–4** rides.
- Unfinished must-dos sort first in `preference_order`.
- Cleared on successful ride completion.

## Balk Thresholds

Precomputed per party at spawn:

```python
balk_sec[r] = BASE_BALK_SEC + BALK_SCALE × preference[r] ** BALK_PREF_EXP
```

Higher preference → willing to wait longer.
