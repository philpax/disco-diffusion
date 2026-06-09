"""The shared session state, in one place so the pieces depend on the data, not the whole App.

:class:`SharedState` is the studio's working model — the loaded session, the in-flight worker, the
prompt list, output geometry, the edit timeline, the img2img init image, and the encode cache. It
holds no UI and does no coordination; controllers + areas read and mutate it. :class:`PaintState`
is the painting tools (the live brush settings + the colour palette, which share ``brush.color``).
The App builds both in ``__post_init__`` and hands them to the pieces that need them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from disco_diffusion import DiscoSession, EncodedPrompt

from .controls import PromptRow
from .init_image import InitImage
from .paint import Brush
from .palette import Palette
from .timeline import Timeline
from .worker import GenerationWorker


@dataclass
class SharedState:
    """The working session model the studio's pieces read + mutate (no UI, no coordination)."""

    session: DiscoSession
    width: int
    height: int
    steps: int
    prompts: list[PromptRow]
    # Seed field contents, persisted across UI rebuilds. Always a concrete seed so the value in
    # use is visible upfront and replaying reuses it; "Rnd" rolls a fresh one (random at startup).
    seed_text: str
    # Staged model selection (what the toggles show, pending a weight reload). Seeded from the
    # loaded session; Models drives the reload, Recipe reads it as part of the live recipe.
    clip_selected: set[str]
    secondary_on: bool
    worker: GenerationWorker | None = None
    paused: bool = False
    # The selected preset name (or "Custom"); set by Recipe once it has detected the match.
    preset_selection: str = ""
    # Snapshot of the per-run settings the active run was started with (the "Current" sidebar tab).
    run_snapshot: dict[str, str] = field(default_factory=dict)
    timeline: Timeline = field(default_factory=Timeline)
    init: InitImage = field(default_factory=InitImage)
    # Prompt-embedding cache shared with the worker thread (guarded by cache_lock).
    encode_cache: dict[str, EncodedPrompt] = field(default_factory=dict)
    cache_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class PaintState:
    """The painting tools: the live brush settings + the colour palette (shared brush.color)."""

    brush: Brush
    palette: Palette
