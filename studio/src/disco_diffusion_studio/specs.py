"""Shared, torch-free typed records: a prompt triple and a guidance snapshot.

Kept apart from the heavier preset/session persistence (presets.py) so both the worker (the
diffusion engine) and the UI/session models share these without pulling in TOML/zip machinery.
Both mirror ``RunConfig`` fields where relevant, with import-time checks against drift from it.
"""

from __future__ import annotations

from typing import NamedTuple

from disco_diffusion import RunConfig
from pydantic import BaseModel, ConfigDict


class PromptSpec(NamedTuple):
    """One prompt as a typed, tuple-compatible triple.

    Lives here (torch-free) so both the session models and the worker share one prompt type; it
    unpacks as ``text, weight, muted`` and compares equal to a plain 3-tuple, while giving the
    UI↔worker wire format named, typed fields.
    """

    text: str
    weight: float
    muted: bool


class GuidanceSnapshot(BaseModel):
    """The live-guidance values captured at a checkpoint, so Revert restores them exactly.

    Fully typed (not a loose ``dict[str, float]``) so int knobs — ``clip_guidance_scale`` and
    ``cutn_batches``, the latter used as ``range(cutn_batches)`` — stay ints through save/load.
    Fields mirror the RunConfig live-guidance knobs + eta; the import-time check below guards
    against drift, and the defaults match RunConfig's.
    """

    model_config = ConfigDict(frozen=True)

    clip_guidance_scale: int = 5000
    tv_scale: float = 0.0
    range_scale: float = 150.0
    sat_scale: float = 0.0
    cutn_batches: int = 4
    clamp_max: float = 0.05
    eta: float = 0.8

    @classmethod
    def capture(cls, config: RunConfig) -> GuidanceSnapshot:
        """Snapshot the guidance fields off a RunConfig (a field absent on ``config`` defaults)."""
        return cls(**{f: getattr(config, f) for f in cls.model_fields if hasattr(config, f)})

    def apply_to(self, config: RunConfig) -> None:
        """Write the snapshot back onto a RunConfig — types preserved, no float/int surprises."""
        for field_name, value in self.model_dump().items():
            setattr(config, field_name, value)


# Every GuidanceSnapshot field must name a real RunConfig field — checked at import so a
# rename/typo fails fast.
_UNKNOWN_SNAPSHOT_FIELDS = sorted(set(GuidanceSnapshot.model_fields) - set(RunConfig.model_fields))
if _UNKNOWN_SNAPSHOT_FIELDS:
    raise RuntimeError(f"GuidanceSnapshot fields not on RunConfig: {_UNKNOWN_SNAPSHOT_FIELDS}")
