"""Small parsing helpers shared by the App and UI modules (kept out of ``app`` to avoid cycles)."""

from __future__ import annotations

from .constants import STEP_MAX, STEP_MIN


def clamp_steps(text: str) -> int:
    """Parse a steps box value and clamp it to the supported range.

    Raises ``ValueError`` on non-integer input so the caller can restore the prior value.
    """
    return max(STEP_MIN, min(STEP_MAX, int(text)))


def int_or(text: str, fallback: int) -> int:
    try:
        return int(text)
    except ValueError:
        return fallback
