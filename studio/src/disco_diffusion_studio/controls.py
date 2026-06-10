"""The advanced-control tables (derived from RunConfig's Tunable metadata) + the prompt-row model.

These are shared by the App and the UI builders (``_ui_*.py``), so they live here to keep those
modules free of an import cycle through ``app``. The control tables are *derived from* the
``Tunable`` metadata on the RunConfig fields, so the attribute names are the config's own — there's
no hand-maintained list of strings that could drift from the schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from disco_diffusion import RunConfig
from disco_diffusion.config import Tunable
from pydantic import BaseModel, ConfigDict


class LiveScale(BaseModel):
    """A live guidance knob surfaced as a slider on the Advanced tab.

    It drives a ``RunConfig`` attribute that ``Sampler._cond_fn`` reads fresh every step, so
    dragging the slider retunes the run on the next step — no restart needed.
    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    attr: str
    """The ``RunConfig`` attribute this slider sets."""
    label: str
    """Slider label shown in the panel."""
    lo: float
    """Minimum slider value."""
    hi: float
    """Maximum slider value."""
    is_int: bool
    """Round the value to an int before applying."""
    fmt: str
    """Format string for the value readout."""


class ScheduleField(BaseModel):
    """A cut-schedule knob, edited as a raw schedule string (e.g. ``[12]*400+[4]*600``).

    Schedules are snapshotted when a Sampler is built, so edits apply on the *next* Play.
    """

    model_config = ConfigDict(frozen=True, use_attribute_docstrings=True)

    attr: str
    """The ``RunConfig`` schedule attribute this box edits."""
    label: str
    """Field label shown in the panel."""


def _tunables(group: str) -> list[tuple[str, Tunable]]:
    """The (field name, Tunable) pairs in ``group``, in RunConfig declaration order."""
    out: list[tuple[str, Tunable]] = []
    for name, info in RunConfig.model_fields.items():
        for meta in info.metadata:
            if isinstance(meta, Tunable) and meta.group == group:
                out.append((name, meta))
    return out


# Live guidance knobs surfaced as sliders. Each drives a RunConfig field that Sampler._cond_fn
# reads fresh every step, so dragging one retunes the run on the next step — no restart.
LIVE_SCALES: list[LiveScale] = [
    LiveScale(attr=n, label=t.label, lo=t.lo or 0.0, hi=t.hi or 0.0, is_int=t.is_int, fmt=t.fmt)
    for n, t in _tunables("live")
]

# Cut-schedule knobs surfaced as raw schedule strings; edits apply on the next Play.
SCHEDULES: list[ScheduleField] = [
    ScheduleField(attr=n, label=t.label) for n, t in _tunables("schedule")
]

# Per-run settings listed (read-only) in the sidebar "Current" tab. The live guidance knobs
# (LIVE_SCALES) are shown above these and track session.config every frame; these reflect the
# active run's snapshot (or the pending values when stopped). Config-backed entries take their
# label from the field's Tunable metadata; "steps"/"size" and the model set are synthesised.
CURRENT_PERRUN: list[tuple[str, str]] = [
    ("steps", "Steps"),
    ("size", "Size"),
    *[(n, t.label) for n, t in _tunables("per_run")],
    *[(n, t.label) for n, t in _tunables("schedule")],
    ("clip_models", "CLIP models"),
    ("use_secondary_model", "Secondary"),
]

# The dropdown entry shown when the live settings don't match any saved preset.
CUSTOM_PRESET = "Custom"


@dataclass
class PromptRow:
    text: str = ""
    weight: float = 1.0
    muted: bool = False  # excluded from the conditioning mix, but kept (text + weight preserved)
