"""Disco Diffusion Studio — interactive control of the sampling loop.

Take manual control of the diffusion loop and steer it live by mixing encoded prompts at
every step. The image is shown at the top; the panel below holds the controls:

  * play / pause + stop, a step counter, and a total-steps box (editable while paused/stopped)
  * a list of prompts, each with a live weight slider (0-2) and a remove button; text applies
    on Enter or when focus leaves the box, and each row shows the normalised mix it contributes
  * width / height (snapped to multiples of 64) and a landscape/portrait flip
  * painting tools (brush kind/size/opacity + palette) that inject onto the canvas to steer
    the diffusion — strokes are noised to the current step and blended into the live latent
  * save

Steps are deliberately slow at full quality: each one is an opportunity to retune the prompt
mix and watch the image respond. Generation runs on a background thread so the UI stays
responsive; all torch/CUDA work happens on that one thread (see worker.py).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame
import pygame_gui
from disco_diffusion import DiscoSession, EncodedPrompt, RunConfig
from disco_diffusion.config import AVAILABLE_CLIP_MODELS, Tunable, parse_schedule
from pydantic import BaseModel, ConfigDict, field_validator
from pygame_gui.core import ObjectID
from pygame_gui.windows import UIFileDialog

from .layout import (
    CTRL_H,
    DEFAULT_H,
    DEFAULT_W,
    DIVIDER_W,
    LABEL_H,
    MARGIN,
    MIN_IMAGE_H,
    MIN_LEFT_PANEL_W,
    MIN_WINDOW_W,
    PAD,
    PANEL_H,
    PROMPT_LIST_H,
    ROW_PITCH,
    SIDEBAR_W_DEFAULT,
    SIDEBAR_W_MAX,
    SIDEBAR_W_MIN,
    Row,
    Stack,
    snap_side,
)
from .paint import BRUSHES, PALETTE, PaintLayer
from .theme import (
    DIVIDER,
    IMAGE_BG,
    MUTED_COLOR,
    PANEL_BG,
    PENDING_COLOR,
    READOUT_COLOR,
    THEME,
    WINDOW_BG,
)
from .worker import GenerationWorker, HistoryEntry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("disco_diffusion_studio")

MIN_ZOOM, MAX_ZOOM = 0.1, 16.0  # view zoom bounds
# Help HUD text per interaction mode (hold right mouse = navigate, release = draw).
DRAW_HELP = (
    "left-drag: paint · scroll: size · shift+scroll: opacity · hold right: pan/zoom"
    " · F: fit · space: play/pause"
)
NAV_HELP = "right-drag: pan · scroll: zoom · release right: draw · F: fit · space: play/pause"
CANVAS_EMPTY_BG = (18, 20, 26)  # placeholder canvas fill shown before the first frame
CANVAS_BORDER = (70, 78, 92)
# Noise-mode re-rolls the patch from fresh noise, which is potent, so its opacity is mapped
# through a gamma curve into a capped injection range: gentle at the low end, and bounded at
# the top so even a full-opacity stroke re-rolls a controlled fraction rather than wiping the
# region. injection = NOISE_MAX_INJECT * opacity**gamma. Overlay stays at the raw opacity.
NOISE_MAX_INJECT = 0.2
NOISE_OPACITY_GAMMA = 2.0
# Debounce for the model auto-reload: changing the CLIP set / secondary toggle queues a reload
# that fires this long after the *last* change, so rapid toggling doesn't reload repeatedly.
RELOAD_DEBOUNCE_MS = 1200


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


class PresetConfig(BaseModel):
    """The config-only half of a preset: guidance scales, eta/Perlin, and the cut schedules.

    Field names match ``RunConfig`` so a preset applies via ``setattr`` over ``model_dump()``.
    """

    model_config = ConfigDict(frozen=True)

    clip_guidance_scale: int
    tv_scale: float
    range_scale: float
    sat_scale: float
    clamp_max: float
    cutn_batches: int
    eta: float
    perlin_init: bool
    cut_overview: str
    cut_innercut: str
    cut_ic_pow: str
    cut_icgray_p: str

    @field_validator("cut_overview", "cut_innercut", "cut_ic_pow", "cut_icgray_p")
    @classmethod
    def _validate_schedule(cls, value: str) -> str:
        parse_schedule(value)  # raises on a malformed schedule string
        return value


# A preset applies via setattr over model_dump(), so every PresetConfig field must name a real
# RunConfig field — check at import so a typo fails fast instead of at apply time.
_UNKNOWN_PRESET_FIELDS = set(PresetConfig.model_fields) - set(RunConfig.model_fields)
if _UNKNOWN_PRESET_FIELDS:
    raise RuntimeError(f"PresetConfig fields not on RunConfig: {sorted(_UNKNOWN_PRESET_FIELDS)}")


class Preset(BaseModel):
    """A one-click full recipe: config knobs (applied now) + a model set (staged for Reload)."""

    model_config = ConfigDict(frozen=True)

    config: PresetConfig
    clip_models: list[str]
    use_secondary_model: bool


# The control tables below are *derived from* the ``Tunable`` metadata on the RunConfig fields
# (see disco_diffusion.config.Tunable), so the attribute names are the config's own — there's
# no separate hand-maintained list of strings that could drift from the schema.
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

# One-click full-recipe presets. "Default" is the port's faithful recipe; "2022 sauce" is the
# high-detail, heavily-regularised, multi-CLIP look from the archive (ic_pow 15, six models).
PRESETS: dict[str, Preset] = {
    "Default": Preset(
        config=PresetConfig(
            clip_guidance_scale=5000,
            tv_scale=0.0,
            range_scale=150.0,
            sat_scale=0.0,
            clamp_max=0.05,
            cutn_batches=4,
            eta=0.8,
            perlin_init=False,
            cut_overview="[12]*400+[4]*600",
            cut_innercut="[4]*400+[12]*600",
            cut_ic_pow="[1]*1000",
            cut_icgray_p="[0.2]*400+[0]*600",
        ),
        clip_models=["ViT-B/32", "ViT-B/16", "RN50"],
        use_secondary_model=True,
    ),
    "2022 sauce": Preset(
        config=PresetConfig(
            clip_guidance_scale=15000,
            tv_scale=250000.0,
            range_scale=10000.0,
            sat_scale=50000.0,
            clamp_max=0.09,
            cutn_batches=1,
            eta=0.8,
            perlin_init=True,
            cut_overview="[18]*200+[14]*200+[4]*400+[2]*200",
            cut_innercut="[2]*200+[6]*200+[8]*400+[18]*200",
            cut_ic_pow="[15]*1000",
            cut_icgray_p="[0.2]*200+[0.1]*200+[0.1]*200+[0.1]*200+[0.1]*200",
        ),
        clip_models=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN101", "RN50", "RN50x4"],
        use_secondary_model=False,
    ),
}


# --- app ---------------------------------------------------------------------


@dataclass
class PromptRow:
    text: str = ""
    weight: float = 1.0


@dataclass
class App:
    session: DiscoSession
    out_dir: Path

    width: int = DEFAULT_W
    height: int = DEFAULT_H
    steps: int = 100
    prompts: list[PromptRow] = field(default_factory=list)

    worker: GenerationWorker | None = None
    paused: bool = False
    _frame_surface: pygame.Surface | None = None
    _frame_key: tuple[int, int] | None = None  # (id(array), index) to detect new frames
    _pending_size: tuple[int, int] | None = None  # coalesced window resize, applied per frame
    _file_dialog: UIFileDialog | None = None  # open Save dialog, if any
    _save_surface: pygame.Surface | None = None  # frame frozen when Save was clicked

    # painting tools
    brush_type: str = "Soft"
    brush_size: float = 48.0  # radius in generation pixels
    brush_strength: float = 0.7
    noise_mode: bool = False  # paint fresh tinted noise (new structure) vs plain colour

    def __post_init__(self) -> None:
        self.width = snap_side(self.width)
        self.height = snap_side(self.height)
        if not self.prompts:
            self.prompts = [PromptRow("a vast alien landscape, oil painting", 1.0)]
        self._encode_cache: dict[str, EncodedPrompt] = {}
        self._cache_lock = threading.Lock()
        # The window size is independent of the generation size: it's seeded once, then owned
        # by the user (resizable). Seed it so the image region — the window minus the sidebar
        # and the bottom panel — matches the canvas aspect, so the canvas fills it with minimal
        # letterboxing, and so the whole thing fits on the desktop. (On a tiling WM the size is
        # overridden anyway; the image is letterboxed into the region, so a flip never resizes.)
        self.sidebar_w = SIDEBAR_W_DEFAULT
        chrome_w = self.sidebar_w + DIVIDER_W  # width taken by the sidebar + its divider
        try:
            sizes = pygame.display.get_desktop_sizes()
            desk_w, desk_h = sizes[0] if sizes and sizes[0][0] > 0 else (1920, 1080)
        except (pygame.error, IndexError):  # headless / older pygame
            desk_w, desk_h = 1920, 1080
        avail_w = max(320, int(desk_w * 0.9) - chrome_w)
        avail_h = max(240, int(desk_h * 0.9) - PANEL_H)
        # Fit the canvas into the available image area, never upscaling past 1:1.
        scale = min(avail_w / self.width, avail_h / self.height, 1.0)
        img_w, img_h = int(self.width * scale), int(self.height * scale)
        # The left column is at least MIN_LEFT_PANEL_W wide; the sidebar always gets its full
        # width on top of that, so the chrome never squeezes either below its minimum.
        self.win_w = max(MIN_LEFT_PANEL_W, img_w) + chrome_w
        self.win_h = max(MIN_IMAGE_H, img_h) + PANEL_H
        self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Disco Diffusion - interactive")
        # Enforce the minimum size natively (SDL) instead of by re-calling set_mode on
        # resize. Re-calling set_mode issues a SetWindowSize that a tiling WM treats as the
        # app demanding geometry, snapping a floated window back under tiling. With a native
        # minimum we just adopt whatever size the window becomes and never fight the WM.
        try:
            window = pygame.Window.from_display_module()
            window.minimum_size = (MIN_WINDOW_W, MIN_IMAGE_H + PANEL_H)
        except (AttributeError, pygame.error):  # older pygame / headless: best-effort only
            pass
        self.manager = pygame_gui.UIManager((self.win_w, self.win_h))
        self.manager.get_theme().load_theme(io.StringIO(json.dumps(THEME)))
        # element-reference tables, rebuilt by _build_ui
        self._row_elements: list[pygame_gui.core.UIElement] = []
        self._remove_buttons: dict[pygame_gui.elements.UIButton, int] = {}
        self._prompt_entries: dict[pygame_gui.elements.UITextEntryLine, int] = {}
        self._weight_sliders: dict[pygame_gui.elements.UIHorizontalSlider, int] = {}
        # The prompt entry that last held keyboard focus, so we can auto-apply its text
        # when focus moves away (no need to press Enter).
        self._focused_entry: pygame_gui.elements.UITextEntryLine | None = None
        # Whether the steps box held focus last frame, so we can commit it on blur (mirrors
        # the prompt boxes; without this, typing a value and clicking away wouldn't apply it).
        self._steps_focused = False

        # Right-hand sidebar: the active tab and the coalesced "rebuild after a resize" flag
        # (applied once per frame in run()). Its width (self.sidebar_w) is seeded above.
        self._sidebar_tab = "settings"  # "settings" | "current"
        self._dragging_divider = False
        self._sidebar_dirty = False
        # Live guidance-scale sliders -> (config attr, is_int, value label, value format).
        self._scale_sliders: dict[
            pygame_gui.elements.UIHorizontalSlider,
            tuple[str, bool, pygame_gui.elements.UILabel, str],
        ] = {}
        # Cut-schedule text boxes -> config attr (validated, applied on next Play).
        self._schedule_entries: dict[pygame_gui.elements.UITextEntryLine, str] = {}
        self._preset_buttons: dict[pygame_gui.elements.UIButton, str] = {}
        # The schedule box that last held focus, so its text is committed on blur (like steps).
        self._focused_schedule: pygame_gui.elements.UITextEntryLine | None = None

        # Model selection (CLIP set + secondary). Changing these needs a full weight reload,
        # so the toggles stage a pending selection and *queue* an auto-reload (debounced); it
        # fires after the user stops changing, and un-queues if they revert to the loaded set.
        self._clip_buttons: dict[pygame_gui.elements.UIButton, str] = {}
        self._clip_selected: set[str] = set(self.session.config.clip_models)
        self._secondary_on: bool = self.session.config.use_secondary_model
        self._reloading = False
        self._reload_thread: threading.Thread | None = None
        self._reload_result: dict[str, object] | None = None
        self._reload_queued_at: int | None = None  # pygame ticks at which to fire the auto-reload

        # "Current" sidebar tab: snapshot of the per-run settings the active run was started
        # with (live knobs are read straight from session.config), plus the label cache.
        self._run_snapshot: dict[str, str] = {}
        self._current_labels: dict[str, pygame_gui.elements.UILabel] = {}

        # Painting state. The paint layer is generation-resolution and persists across runs
        # (re-created only on a size change); strokes are injected into the latent by the
        # worker and the overlay clears once a step has baked them.
        self.brush_color: tuple[int, int, int] = PALETTE[3]
        self._paint_layer = PaintLayer(self.width, self.height)
        self._mouse_pos: tuple[int, int] = (0, 0)
        self._painting = False
        self._last_gen: tuple[float, float] | None = None
        self._paint_awaiting_bake = False
        self._paint_baseline = 0
        self._brush_buttons: dict[pygame_gui.elements.UIButton, str] = {}
        self._swatch_rects: list[tuple[pygame.Rect, tuple[int, int, int]]] = []
        self._color_preview_rect = pygame.Rect(0, 0, 0, 0)
        self._list_inner_w = 0  # row width inside the prompt list (set in _build_ui)

        # View transform: the canvas is decoupled from the window, which is just a viewport
        # onto it. zoom = screen px per canvas px; pan = screen pos of canvas pixel (0, 0).
        self._zoom = 1.0
        self._pan = pygame.Vector2(0, 0)
        self._panning = False
        self._navigating = False  # right mouse held: canvas-navigation mode (pan + scroll-zoom)
        self._hud_font = pygame.font.SysFont(None, 19)

        # Edit history / revert preview. _preview_index None = showing the live frame; an int
        # = showing that checkpoint's image (non-destructive) until Revert commits it.
        self._history: list[HistoryEntry] = []
        self._hist_len = 0
        self._preview_index: int | None = None
        self._preview_prompts: list[PromptRow] | None = None  # prompts shown while previewing
        self._preview_surface: pygame.Surface | None = None
        self._preview_key: int | None = None
        self._history_slider_rect = pygame.Rect(0, 0, 10, 10)
        # Fit the canvas to the available area once the real window size is settled. The window
        # manager may resize the window right after it opens (firing a VIDEORESIZE that only
        # re-clamps), so we (re)fit on the first run-loop frame, not just here.
        self._did_initial_fit = False
        self._build_ui()
        self._fit_view()

    # -- geometry --
    # The window is a left column (image on top, bottom control panel below) plus a full-height
    # right sidebar. A draggable divider at x == _panel_w() sets the split.
    def _panel_w(self) -> int:
        """Width of the left column (image + bottom panel) — everything left of the sidebar."""
        return max(MIN_LEFT_PANEL_W, self.win_w - self.sidebar_w)

    def _divider_x(self) -> int:
        return self._panel_w()

    def _sidebar_rect(self) -> pygame.Rect:
        x = self._panel_w() + DIVIDER_W
        return pygame.Rect(x, 0, max(0, self.win_w - x), self.win_h)

    def _bottom_panel_rect(self) -> pygame.Rect:
        return pygame.Rect(0, self._image_area_h(), self._panel_w(), PANEL_H)

    def _image_area_h(self) -> int:
        return max(0, self.win_h - PANEL_H)

    def _window_size(self) -> tuple[int, int]:
        return (self.win_w, self.win_h)

    def _image_region(self) -> pygame.Rect:
        """The screen area the canvas viewport occupies (above the panel, left of the sidebar)."""
        return pygame.Rect(0, 0, self._panel_w(), self._image_area_h())

    def _canvas_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def _canvas_screen_rect(self) -> pygame.Rect:
        """The canvas bounds in screen coords under the current view transform."""
        w, h = self._canvas_size()
        return pygame.Rect(
            int(self._pan.x), int(self._pan.y), int(w * self._zoom), int(h * self._zoom)
        )

    def _fit_view(self) -> None:
        """Reset the view so the whole canvas fits the viewport, centred."""
        region = self._image_region()
        w, h = self._canvas_size()
        self._zoom = min(region.width / w, region.height / h) if w and h else 1.0
        self._pan = pygame.Vector2(
            region.x + (region.width - w * self._zoom) / 2,
            region.y + (region.height - h * self._zoom) / 2,
        )

    def _zoom_at(self, pos: tuple[int, int], factor: float) -> None:
        """Multiply the zoom by ``factor``, keeping the canvas point under ``pos`` fixed."""
        new_zoom = max(MIN_ZOOM, min(MAX_ZOOM, self._zoom * factor))
        ratio = new_zoom / self._zoom
        anchor = pygame.Vector2(pos)
        self._pan = anchor - (anchor - self._pan) * ratio
        self._zoom = new_zoom
        self._clamp_pan()

    def _clamp_pan(self) -> None:
        """Keep the canvas centre within the viewport so the canvas can't be lost off-screen."""
        region = self._image_region()
        w, h = self._canvas_size()
        cw, ch = w * self._zoom, h * self._zoom
        cx = max(region.left, min(region.right, self._pan.x + cw / 2))
        cy = max(region.top, min(region.bottom, self._pan.y + ch / 2))
        self._pan = pygame.Vector2(cx - cw / 2, cy - ch / 2)

    def _screen_to_canvas(self, pos: tuple[int, int]) -> tuple[float, float] | None:
        """Map a screen position to canvas-pixel coords, or None if outside the canvas."""
        if not self._image_region().collidepoint(pos):
            return None
        cx = (pos[0] - self._pan.x) / self._zoom
        cy = (pos[1] - self._pan.y) / self._zoom
        w, h = self._canvas_size()
        return (cx, cy) if 0 <= cx < w and 0 <= cy < h else None

    def _blit_canvas(self, surf: pygame.Surface) -> None:
        """Blit only the visible part of a canvas-resolution surface under the view transform."""
        w, h = surf.get_size()
        z = self._zoom
        region = self._image_region()
        cx0 = max(0, int((region.left - self._pan.x) / z))
        cy0 = max(0, int((region.top - self._pan.y) / z))
        cx1 = min(w, math.ceil((region.right - self._pan.x) / z))
        cy1 = min(h, math.ceil((region.bottom - self._pan.y) / z))
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = surf.subsurface(pygame.Rect(cx0, cy0, cx1 - cx0, cy1 - cy0))
        dest = (max(1, int((cx1 - cx0) * z)), max(1, int((cy1 - cy0) * z)))
        scaled = pygame.transform.smoothscale(sub, dest)
        self.screen.blit(scaled, (self._pan.x + cx0 * z, self._pan.y + cy0 * z))

    def _typing(self) -> bool:
        """True while a text box has keyboard focus (so shortcut keys don't steal input)."""
        focus = self.manager.get_focus_set()
        return bool(focus) and any(
            isinstance(e, pygame_gui.elements.UITextEntryLine) for e in focus
        )

    def _build_palette(self, rect: pygame.Rect) -> None:
        """Lay out the current-colour preview + swatch rects within ``rect`` (drawn custom)."""
        self._swatch_rects = []
        self._color_preview_rect = pygame.Rect(rect.x, rect.y, CTRL_H, CTRL_H)
        x = rect.x + CTRL_H + 10
        n = len(PALETTE)
        gap = 4
        sw = max(10, min(CTRL_H, (rect.right - x - (n - 1) * gap) // n))
        y = rect.y + (CTRL_H - sw) // 2
        for i, color in enumerate(PALETTE):
            self._swatch_rects.append((pygame.Rect(x + i * (sw + gap), y, sw, sw), color))

    @property
    def running(self) -> bool:
        return (
            self.worker is not None
            and self.worker.is_alive()
            and not self.paused
            and not self.worker.finished
        )

    # -- UI construction --
    def _build_ui(self) -> None:
        self.manager.clear_and_reset()
        self._remove_buttons.clear()
        self._prompt_entries.clear()
        self._weight_sliders.clear()
        self._row_elements.clear()
        self._build_bottom_panel()
        self._build_sidebar()
        self._sync_enabled()

    def _build_bottom_panel(self) -> None:
        """The left column's control panel: transport, history, tools, colours, prompts."""
        ui = pygame_gui.elements
        panel_w = self._panel_w()
        stack = Stack(MARGIN, self._image_area_h() + PAD, panel_w - 2 * MARGIN)

        # Row 1: transport — Play / Stop | step (left-aligned) … status (right-aligned) | Save.
        r = stack.row(CTRL_H)
        self.play_button = ui.UIButton(r.left(110), "Play", self.manager, object_id="#play_button")
        self.stop_button = ui.UIButton(r.left(90), "Stop", self.manager, object_id="#stop_button")
        self.save_button = ui.UIButton(r.right(90), "Save", self.manager, object_id="#save_button")
        self.status_label = ui.UILabel(r.right(170), "", self.manager, object_id="#status_label")
        self.step_label = ui.UILabel(r.fill(), "step 0 / 0", self.manager, object_id="#step_label")

        # Row 2: history scrubber — directly under transport. Drag to preview a checkpoint.
        r = stack.row(CTRL_H)
        ui.UILabel(r.left(54), "History", self.manager)
        self.cancel_button = ui.UIButton(r.right(70), "Cancel", self.manager)
        self.revert_button = ui.UIButton(r.right(70), "Revert", self.manager)
        self.history_label = ui.UILabel(r.right(120), "live", self.manager)
        self._history_slider_rect = r.fill()
        # The slider spans the 0..N step timeline (not the checkpoint count), so a checkpoint's
        # thumb position matches its actual progress; drags snap to the nearest checkpoint.
        self.history_slider = ui.UIHorizontalSlider(
            self._history_slider_rect,
            start_value=self._history_slider_start(),
            value_range=(0.0, float(max(self._history_total(), 1))),
            manager=self.manager,
        )

        # Row 3: painting tools — brush kind, noise toggle, size, opacity, clear
        r = stack.row(CTRL_H)
        self._brush_buttons = {}
        for name in BRUSHES:
            button = ui.UIButton(r.left(64), name, self.manager, object_id="#brush_button")
            self._brush_buttons[button] = name
            if name == self.brush_type:
                button.select()
        # Toggle: deposit fresh tinted noise (new structure) instead of plain colour.
        self.noise_button = ui.UIButton(
            r.left(74), "Noise", self.manager, object_id="#brush_button"
        )
        if self.noise_mode:
            self.noise_button.select()
        # Right group (packed right-to-left, so it reads "Opacity [slider] Clear" left-to-right).
        self.clear_paint_button = ui.UIButton(r.right(64), "Clear", self.manager)
        self.strength_slider = ui.UIHorizontalSlider(
            r.right(104), self.brush_strength, (0.05, 1.0), self.manager
        )
        ui.UILabel(r.right(56), "Opacity", self.manager)
        # Size label + slider, the slider flexing into whatever's left between the two groups.
        ui.UILabel(r.left(36), "Size", self.manager)
        self.size_slider = ui.UIHorizontalSlider(
            r.fill(), self.brush_size, (4.0, 160.0), self.manager
        )

        # Row 4: colour palette — current-colour preview + swatches (custom-drawn) on its own row.
        self._build_palette(stack.row(CTRL_H).fill())

        # Row 5: prompts header — Add + hint (hint fills the remaining width)
        r = stack.row(LABEL_H)
        self.add_button = ui.UIButton(
            r.left(120), "+ Add prompt", self.manager, object_id="#add_button"
        )
        self.hint_label = ui.UILabel(
            r.fill(),
            "weight 0-2 applies instantly · text applies on Enter or click-away · % = mix used",
            self.manager,
            object_id="#hint_label",
        )

        # Fixed-height scrolling prompt list (extra rows scroll), filling the rest of the panel.
        # Pulled up under the header (less the usual row pad) to tighten the gap to the first row.
        list_rect = pygame.Rect(MARGIN, stack.y - 8, panel_w - 2 * MARGIN, PROMPT_LIST_H)
        self.prompt_panel = ui.UIScrollingContainer(list_rect, self.manager)
        # Lay rows out narrower than the viewport so the vertical scrollbar never forces a
        # horizontal one (a horizontal bar appears only when content is wider than the view).
        self._list_inner_w = list_rect.width - 24
        self._rebuild_prompt_rows()

    def _build_sidebar(self) -> None:
        """The full-height right sidebar: a Settings / Current tab pair over a scroll area."""
        ui = pygame_gui.elements
        sb = self._sidebar_rect()
        x = sb.x + PAD
        inner = max(40, sb.width - 2 * PAD)

        r = Row(x, MARGIN, inner, CTRL_H)
        half = (inner - PAD) // 2
        self.tab_settings = ui.UIButton(
            r.left(half), "Settings", self.manager, object_id="#tab_button"
        )
        self.tab_current = ui.UIButton(r.fill(), "Current", self.manager, object_id="#tab_button")

        cont_y = MARGIN + CTRL_H + PAD
        cont_rect = pygame.Rect(x, cont_y, inner, max(60, self.win_h - cont_y - MARGIN))
        self.settings_panel = ui.UIScrollingContainer(cont_rect, self.manager)
        self.current_panel = ui.UIScrollingContainer(cont_rect, self.manager)
        self._sb_inner_w = inner - 24  # leave room for the vertical scrollbar
        self._build_settings_rows()
        self._build_current_rows()
        self._sync_sidebar_tabs()

    def _displayed_prompts(self) -> list[PromptRow]:
        """The prompts shown in the rows: a previewed checkpoint's, else the live set."""
        return self._preview_prompts if self._preview_prompts is not None else self.prompts

    def _rebuild_prompt_rows(self) -> None:
        for el in self._row_elements:
            el.kill()
        self._row_elements.clear()
        self._remove_buttons.clear()
        self._prompt_entries.clear()
        self._weight_sliders.clear()

        container = self.prompt_panel
        inner_w = self._list_inner_w
        ui = pygame_gui.elements
        v_pad = (ROW_PITCH - CTRL_H) // 2  # vertically centre widgets in their row pitch
        prompts = self._displayed_prompts()
        for i, prompt in enumerate(prompts):
            # Pack: [text fills] [slider] [weight readout] [X]. Right-side widgets are taken
            # first so the text entry flexes into whatever width is left.
            r = Row(0, i * ROW_PITCH + v_pad, inner_w, CTRL_H)
            remove = ui.UIButton(
                r.right(30),
                "×",
                self.manager,
                container=container,
                object_id=ObjectID(object_id="#remove_button", class_id="@remove_button"),
            )
            wlabel = ui.UILabel(r.right(104), "", self.manager, container=container)
            slider = ui.UIHorizontalSlider(
                r.right(150),
                start_value=prompt.weight,
                value_range=(0.0, 2.0),
                manager=self.manager,
                container=container,
            )
            entry = ui.UITextEntryLine(r.fill(), self.manager, container=container)
            entry.set_text(prompt.text)
            self._row_elements += [remove, entry, slider, wlabel]
            self._remove_buttons[remove] = i
            self._prompt_entries[entry] = i
            self._weight_sliders[slider] = i
            prompt._wlabel = wlabel  # type: ignore[attr-defined]  # stash for live updates
            prompt._entry = entry  # type: ignore[attr-defined]
            prompt._label_state = None  # type: ignore[attr-defined]  # last (text, colour) shown
        container.set_scrollable_area_dimensions((inner_w, max(len(prompts), 1) * ROW_PITCH + 6))
        self._refresh_rows()

    def _refresh_rows(self) -> None:
        """Update each row's readout: raw weight + normalised share, or a pending badge.

        Mirrors Sampler.set_conditioning (empty rows ignored; remaining weights normalised
        to sum to 1, so the % is exactly the mix the guidance uses). A row whose text box
        differs from the applied prompt shows an amber "edited · Enter" badge instead — this
        is the live "not yet applied" signal. Labels are only mutated when their state
        changes, so this is cheap to call every frame.
        """
        prompts = self._displayed_prompts()
        active = [(i, r.weight) for i, r in enumerate(prompts) if r.text.strip()]
        total = sum(w for _, w in active)
        shares = {i: w / total for i, w in active} if total > 1e-3 else {}
        for i, row in enumerate(prompts):
            wlabel = getattr(row, "_wlabel", None)
            entry = getattr(row, "_entry", None)
            if wlabel is None or entry is None:
                continue
            if entry.get_text() != row.text:
                text, colour = "edited · Enter", PENDING_COLOR
            elif not row.text.strip():
                text, colour = f"{row.weight:.2f}  empty", MUTED_COLOR
            elif not shares:
                text, colour = f"{row.weight:.2f}  off", MUTED_COLOR
            else:
                text, colour = f"{row.weight:.2f}  {shares[i] * 100:.0f}%", READOUT_COLOR
            state = (text, colour)
            if row._label_state == state:  # type: ignore[attr-defined]
                continue
            row._label_state = state  # type: ignore[attr-defined]
            wlabel.text_colour = pygame.Color(*colour)
            wlabel.set_text(text)

    def _build_settings_rows(self) -> None:
        """Build the sidebar Settings tab: output size, then the advanced controls.

        Layout order: output (steps / size, per-run) · guidance sliders (retune live) · per-run
        cut schedules + eta/perlin · model set. Sliders write straight to ``session.config``
        (read live by the worker each step); schedule boxes are validated and applied next Play;
        toggling a model queues a (debounced) auto-reload.
        """
        self._scale_sliders = {}
        self._schedule_entries = {}
        self._preset_buttons = {}
        self._clip_buttons = {}
        ui = pygame_gui.elements
        container = self.settings_panel
        inner_w = self._sb_inner_w
        pitch = CTRL_H + 8
        cfg = self.session.config

        def section(y: int, text: str) -> int:
            ui.UILabel(
                Row(0, y, inner_w, LABEL_H).fill(),
                text,
                self.manager,
                container=container,
                object_id="#section_label",
            )
            return y + LABEL_H + 6

        # Output (per-run): steps, width/height, apply / flip.
        y = section(2, "Output — apply on next Play")
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(54), "Steps", self.manager, container=container)
        self.steps_entry = ui.UITextEntryLine(r.fill(), self.manager, container=container)
        self.steps_entry.set_text(str(self.steps))
        y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(20), "W", self.manager, container=container)
        self.width_entry = ui.UITextEntryLine(
            r.left((inner_w - 56) // 2), self.manager, container=container
        )
        self.width_entry.set_text(str(self.width))
        ui.UILabel(r.left(20), "H", self.manager, container=container)
        self.height_entry = ui.UITextEntryLine(r.fill(), self.manager, container=container)
        self.height_entry.set_text(str(self.height))
        y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        self.apply_button = ui.UIButton(
            r.left((inner_w - PAD) // 2), "Apply size", self.manager, container=container
        )
        self.swap_button = ui.UIButton(r.fill(), "Flip W/H", self.manager, container=container)
        y += pitch

        # Guidance (live): each slider retunes the running step immediately.
        y = section(y + 6, "Guidance — retunes live")
        for sc in LIVE_SCALES:
            r = Row(0, y, inner_w, CTRL_H)
            ui.UILabel(r.left(108), sc.label, self.manager, container=container)
            vlabel = ui.UILabel(r.right(66), "", self.manager, container=container)
            cur = float(getattr(cfg, sc.attr))
            slider = ui.UIHorizontalSlider(
                r.fill(),
                start_value=min(max(cur, sc.lo), sc.hi),
                value_range=(sc.lo, sc.hi),
                manager=self.manager,
                container=container,
            )
            vlabel.set_text(sc.fmt.format(cur))
            self._scale_sliders[slider] = (sc.attr, sc.is_int, vlabel, sc.fmt)
            y += pitch

        # Per-run: presets, eta + perlin, raw cut schedules.
        y = section(y + 6, "Per-run — apply on next Play")
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(50), "Preset", self.manager, container=container)
        pw = (inner_w - 50 - PAD - (len(PRESETS) - 1) * PAD) // max(len(PRESETS), 1)
        for name in PRESETS:
            button = ui.UIButton(r.left(pw), name, self.manager, container=container)
            self._preset_buttons[button] = name
        y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(40), "eta", self.manager, container=container)
        self.perlin_button = ui.UIButton(
            r.right(96), "Perlin init", self.manager, container=container
        )
        self._eta_label = ui.UILabel(r.right(48), "", self.manager, container=container)
        self._eta_slider = ui.UIHorizontalSlider(
            r.fill(),
            start_value=min(max(float(cfg.eta), 0.0), 1.0),
            value_range=(0.0, 1.0),
            manager=self.manager,
            container=container,
        )
        self._eta_label.set_text(f"{cfg.eta:.2f}")
        if cfg.perlin_init:
            self.perlin_button.select()
        y += pitch
        for sch in SCHEDULES:
            ui.UILabel(
                Row(0, y, inner_w, LABEL_H).left(inner_w),
                sch.label,
                self.manager,
                container=container,
            )
            y += LABEL_H + 2
            entry = ui.UITextEntryLine(
                Row(0, y, inner_w, CTRL_H).fill(), self.manager, container=container
            )
            entry.set_text(str(getattr(cfg, sch.attr)))
            self._schedule_entries[entry] = sch.attr
            y += pitch

        # Models: CLIP toggles + secondary. Changing these queues an auto-reload (no button).
        y = section(y + 6, "Models — auto-reloads on change")
        per_row = 2
        bw = (inner_w - (per_row - 1) * PAD) // per_row
        for i, name in enumerate(AVAILABLE_CLIP_MODELS):
            if i % per_row == 0:
                r = Row(0, y, inner_w, CTRL_H)
            button = ui.UIButton(
                r.left(bw), name, self.manager, container=container, object_id="#brush_button"
            )
            self._clip_buttons[button] = name
            if name in self._clip_selected:
                button.select()
            if i % per_row == per_row - 1:
                y += pitch
        if len(AVAILABLE_CLIP_MODELS) % per_row != 0:
            y += pitch
        r = Row(0, y, inner_w, CTRL_H)
        self.secondary_button = ui.UIButton(
            r.fill(),
            "Secondary model",
            self.manager,
            container=container,
            object_id="#brush_button",
        )
        if self._secondary_on:
            self.secondary_button.select()
        y += pitch

        container.set_scrollable_area_dimensions((inner_w, y + 8))

    def _build_current_rows(self) -> None:
        """Build the read-only "Current" tab: name + value label per setting."""
        ui = pygame_gui.elements
        container = self.current_panel
        inner_w = self._sb_inner_w
        pitch = CTRL_H + 4
        self._current_labels = {}
        name_w = 116

        def row(y: int, key: str, name: str) -> int:
            ui.UILabel(
                Row(0, y, inner_w, CTRL_H).left(name_w), name, self.manager, container=container
            )
            value = ui.UILabel(
                Row(name_w + PAD, y, inner_w - name_w - PAD, CTRL_H).fill(),
                "",
                self.manager,
                container=container,
            )
            self._current_labels[key] = value
            return y + pitch

        # Headings are plain — the Current tab reflects what the image is generating with, so the
        # values speak for themselves without a "this run / next run" qualifier.
        y = 2
        ui.UILabel(
            Row(0, y, inner_w, LABEL_H).fill(),
            "Guidance",
            self.manager,
            container=container,
            object_id="#section_label",
        )
        y += LABEL_H + 6
        for sc in LIVE_SCALES:
            y = row(y, sc.attr, sc.label)
        y += 6
        ui.UILabel(
            Row(0, y, inner_w, LABEL_H).fill(),
            "Per-run",
            self.manager,
            container=container,
            object_id="#section_label",
        )
        y += LABEL_H + 6
        for key, name in CURRENT_PERRUN:
            y = row(y, key, name)
        container.set_scrollable_area_dimensions((inner_w, y + 8))
        self._refresh_current()

    def _perrun_values(self) -> dict[str, str]:
        """Display strings for every CURRENT_PERRUN key from the current (pending) state.

        Driven by the key list itself: the two synthesised keys are special-cased, the rest are
        read off ``session.config`` by name (so they can't drift from the listed keys).
        """
        cfg = self.session.config
        out: dict[str, str] = {}
        for key, _label in CURRENT_PERRUN:
            if key == "steps":
                out[key] = str(self.steps)
            elif key == "size":
                out[key] = f"{self.width} × {self.height}"
            elif key == "clip_models":
                out[key] = ", ".join(cfg.clip_models)
            else:
                value = getattr(cfg, key)
                if isinstance(value, bool):
                    out[key] = "on" if value else "off"
                elif isinstance(value, float):
                    out[key] = f"{value:.2f}"
                else:
                    out[key] = str(value)
        return out

    def _refresh_current(self) -> None:
        """Update the Current tab: live knobs from session.config, per-run from the snapshot."""
        if not self._current_labels:
            return
        cfg = self.session.config
        for sc in LIVE_SCALES:  # live: reflect session.config as sliders move
            label = self._current_labels.get(sc.attr)
            if label is not None:
                text = sc.fmt.format(float(getattr(cfg, sc.attr)))
                if label.text != text:
                    label.set_text(text)
        # While a run exists (playing, paused, or done) these reflect that run's frozen snapshot —
        # what the image on screen was generated with; only once fully stopped do they show the
        # pending values the next run would use.
        perrun = self._run_snapshot if self.worker is not None else self._perrun_values()
        for key, _name in CURRENT_PERRUN:
            label = self._current_labels.get(key)
            text = perrun.get(key, "—")
            if label is not None and label.text != text:
                label.set_text(text)

    def _sync_sidebar_tabs(self) -> None:
        """Show exactly one of the Settings / Current panels for the active sidebar tab."""
        settings = self._sidebar_tab == "settings"
        (self.settings_panel.show if settings else self.settings_panel.hide)()
        (self.current_panel.show if not settings else self.current_panel.hide)()
        (self.tab_settings.select if settings else self.tab_settings.unselect)()
        (self.tab_current.select if not settings else self.tab_current.unselect)()

    def _commit_schedule_entry(self, entry: pygame_gui.elements.UITextEntryLine) -> None:
        """Validate a cut-schedule box and store it on the config (applies on next Play).

        Schedules are parsed with the library's own parser; on a malformed string we flag it
        and restore the previous value rather than letting the worker blow up at run start.
        """
        attr = self._schedule_entries.get(entry)
        if attr is None:
            return
        text = entry.get_text().strip()
        if text == str(getattr(self.session.config, attr)):
            return
        try:
            parsed = parse_schedule(text)
        except ValueError:
            self._status("Bad schedule")
            entry.set_text(str(getattr(self.session.config, attr)))
            return
        # cond_fn indexes these over the full 1000-step internal timeline, so a short schedule
        # would IndexError mid-run. Require it to cover 1000 (extra entries are harmless).
        if len(parsed) < 1000:
            self._status("Schedule short")
            entry.set_text(str(getattr(self.session.config, attr)))
            return
        setattr(self.session.config, attr, text)
        self._status("Schedule set")

    def _refresh_advanced_widgets(self) -> None:
        """Re-sync every Advanced widget from the current config (after a preset load)."""
        cfg = self.session.config
        for slider, (attr, _is_int, vlabel, fmt) in self._scale_sliders.items():
            value = float(getattr(cfg, attr))
            slider.set_current_value(value)
            vlabel.set_text(fmt.format(value))
        self._eta_slider.set_current_value(min(max(float(cfg.eta), 0.0), 1.0))
        self._eta_label.set_text(f"{cfg.eta:.2f}")
        (self.perlin_button.select if cfg.perlin_init else self.perlin_button.unselect)()
        for entry, attr in self._schedule_entries.items():
            entry.set_text(str(getattr(cfg, attr)))

    def _apply_preset(self, name: str) -> None:
        """Load a full-recipe preset: config knobs apply now; a model change auto-reloads."""
        preset = PRESETS.get(name)
        if preset is None:
            return
        cfg = self.session.config
        for attr, value in preset.config.model_dump().items():
            setattr(cfg, attr, value)
        self._refresh_advanced_widgets()
        # Stage the model set (CLIP + secondary); a change queues the debounced auto-reload.
        self._clip_selected = set(preset.clip_models)
        self._secondary_on = preset.use_secondary_model
        for button, mname in self._clip_buttons.items():
            (button.select if mname in self._clip_selected else button.unselect)()
        (self.secondary_button.select if self._secondary_on else self.secondary_button.unselect)()
        self._status("Loaded")
        self._update_reload_queue()

    def _toggle_clip_model(self, button: pygame_gui.elements.UIButton) -> None:
        """Stage a CLIP model in/out of the pending selection (queues the auto-reload)."""
        name = self._clip_buttons[button]
        if name in self._clip_selected:
            self._clip_selected.discard(name)
            button.unselect()
        else:
            self._clip_selected.add(name)
            button.select()
        self._update_reload_queue()

    def _start_reload(self) -> None:
        """Rebuild the session with the staged CLIP set / secondary toggle on a worker thread.

        Reloading weights takes ~a minute, so it runs off the UI thread; the run loop polls
        :meth:`_poll_reload` and swaps the session in when it's done. Play stays disabled and
        the staged selection is locked (via _sync_enabled) until then.
        """
        if self._reloading:
            return
        selected = [m for m in AVAILABLE_CLIP_MODELS if m in self._clip_selected]
        if not selected:
            self._status("Pick a model")
            return
        cfg = self.session.config
        # Order is irrelevant for the CLIP set (guidance sums over all models), so compare as
        # sets — otherwise a reselection in a different order would look like a change.
        if set(selected) == set(cfg.clip_models) and self._secondary_on == cfg.use_secondary_model:
            self._status("No change")
            return
        self._stop_run()  # the worker holds the old session; tear it down first
        new_cfg = cfg.model_copy(
            update={"clip_models": selected, "use_secondary_model": self._secondary_on}
        )
        device = self.session.device
        self._reload_result = None
        self._reloading = True
        self._status("Reloading…")
        self._sync_enabled()

        def work() -> None:
            try:
                # Pass the existing device so there's no interactive CPU-fallback prompt.
                new_session = DiscoSession(new_cfg, device=device)
                self._reload_result = {"session": new_session}
            except Exception as exc:  # surfaced on the UI thread by _poll_reload
                log.exception("model reload failed")
                self._reload_result = {"error": str(exc)}

        self._reload_thread = threading.Thread(target=work, daemon=True)
        self._reload_thread.start()

    def _poll_reload(self) -> None:
        """Swap in a reloaded session once the background reload finishes (called per frame)."""
        if not self._reloading or self._reload_thread is None or self._reload_thread.is_alive():
            return
        self._reloading = False
        self._reload_thread = None
        result = self._reload_result
        self._reload_result = None
        if result and "session" in result:
            self.session = result["session"]  # type: ignore[assignment]
            self._encode_cache.clear()  # embeds came from the old CLIP set — now stale
            self._clip_selected = set(self.session.config.clip_models)
            self._secondary_on = self.session.config.use_secondary_model
            self._status("Reloaded")
        else:
            self._status("Reload failed")  # the traceback was logged by the reload thread
        self._build_ui()  # rebuild so the advanced controls reflect the (new) session config
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        """Total-steps + size boxes are editable only when not actively generating."""
        editable = not self.running
        for el in (
            self.steps_entry,
            self.width_entry,
            self.height_entry,
            self.apply_button,
            self.swap_button,
        ):
            (el.enable if editable else el.disable)()
        # History controls are usable only while paused/stopped and there's history to scrub.
        hist_on = editable and len(self._history) > 0
        for hist_el in (self.history_slider, self.revert_button, self.cancel_button):
            (hist_el.enable if hist_on else hist_el.disable)()
        # Prompt rows are read-only while previewing a checkpoint (they show its prompts).
        prompts_on = self._preview_index is None
        prompt_widgets = [self.add_button, *self._prompt_entries, *self._weight_sliders]
        for pw in (*prompt_widgets, *self._remove_buttons):
            (pw.enable if prompts_on else pw.disable)()
        self.play_button.set_text("Pause" if self.running else "Play")
        # Can't resume mid-preview or mid-reload — Revert/Cancel, or wait for the reload.
        play_off = self._preview_index is not None or self._reloading
        (self.play_button.disable if play_off else self.play_button.enable)()

    def _models_match_session(self) -> bool:
        """True when the staged CLIP set + secondary toggle equal the loaded session's."""
        cfg = self.session.config
        return set(self._clip_selected) == set(cfg.clip_models) and (
            self._secondary_on == cfg.use_secondary_model
        )

    def _update_reload_queue(self) -> None:
        """Queue (or cancel) the debounced auto-reload after a model toggle / preset load.

        Changing the CLIP set or secondary toggle needs a full weight reload. Rather than a
        button, we queue the reload to fire shortly after the user stops changing things, and
        cancel it if they land back on the currently-loaded set.
        """
        if self._models_match_session():
            if self._reload_queued_at is not None:
                self._reload_queued_at = None
                self._status("Reload cancelled")
        else:
            self._reload_queued_at = pygame.time.get_ticks() + RELOAD_DEBOUNCE_MS
            self._status("Reload queued")

    def _status(self, text: str) -> None:
        self.status_label.set_text(text)

    # -- prompt snapshot --
    def _prompt_snapshot(self) -> list[tuple[str, float]]:
        return [(r.text, r.weight) for r in self.prompts]

    def _push_prompts(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.set_prompts(self._prompt_snapshot())

    def _commit_prompt_entry(self, entry: pygame_gui.elements.UITextEntryLine) -> None:
        """Apply a prompt text box's contents (on Enter or when focus moves away)."""
        idx = self._prompt_entries.get(entry)
        if idx is None or not (0 <= idx < len(self.prompts)):
            return
        if entry.get_text() == self.prompts[idx].text:
            return
        self.prompts[idx].text = entry.get_text()
        self._refresh_rows()
        self._push_prompts()
        self._request_checkpoint("prompt")

    # -- run lifecycle --
    def _start_run(self) -> None:
        # Adopt any step count typed into the box but not yet Enter-applied: clicking Play
        # moves focus off the box without firing UI_TEXT_ENTRY_FINISHED, so without this the
        # run would silently use the previous value.
        self._commit_steps()
        self._stop_run()
        self.worker = GenerationWorker(
            self.session,
            width=self.width,
            height=self.height,
            steps=self.steps,
            encode_cache=self._encode_cache,
            cache_lock=self._cache_lock,
            perlin=self.session.config.perlin_init,
        )
        self.worker.set_prompts(self._prompt_snapshot())
        if not self._paint_layer.empty():
            self._paint_layer.dirty = True  # re-send any standing strokes to the new run
        # Freeze the per-run settings this run uses, for the "Current" tab to show while it runs.
        self._run_snapshot = self._perrun_values()
        self._history = []
        self._hist_len = 0
        self._preview_index = None
        self.paused = False
        self.worker.start()
        self._status("Running")
        self._sync_enabled()

    def _stop_run(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=5.0)
        self.worker = None
        self.paused = False
        self._history = []
        self._hist_len = 0
        self._preview_index = None
        self._sync_enabled()

    def _toggle_play(self) -> None:
        if self._preview_index is not None:
            self._status("Previewing")
            return
        if self.worker is None or not self.worker.is_alive() or self.worker.finished:
            self._start_run()  # finished -> Play starts a fresh run (Revert continues a branch)
        elif self.paused:
            self.paused = False
            self._preview_index = None  # resume from live, drop any history preview
            self.worker.resume()
            self._status("Running")
            self._sync_enabled()
        else:
            self.paused = True
            self.worker.pause()
            self._status("Paused")
            self._sync_enabled()

    def _apply_size(self, width: int, height: int) -> None:
        # Changing the output shape requires a fresh run. The window does NOT change — the
        # image is letterboxed into the image region — so orientation flips keep proportions.
        self._stop_run()
        self.width = snap_side(width)
        self.height = snap_side(height)
        self.width_entry.set_text(str(self.width))
        self.height_entry.set_text(str(self.height))
        # The paint layer is generation-resolution, so rebuild it for the new size.
        self._paint_layer = PaintLayer(self.width, self.height)
        self._paint_awaiting_bake = False
        # The canvas changed shape: drop the stale frame and refit the view.
        self._frame_surface = None
        self._frame_key = None
        self._fit_view()
        self._status("Size set")

    def _set_sidebar_width(self, width: int) -> None:
        """Set the sidebar width (clamped); the rebuild is coalesced to one per frame."""
        max_w = max(SIDEBAR_W_MIN, min(SIDEBAR_W_MAX, self.win_w - MIN_LEFT_PANEL_W - DIVIDER_W))
        new_w = max(SIDEBAR_W_MIN, min(max_w, int(width)))
        if new_w != self.sidebar_w:
            self.sidebar_w = new_w
            self._sidebar_dirty = True

    def _resize_window(self, w: int, h: int) -> None:
        # Adopt the window's actual new size — do NOT call set_mode (that fights tiling WMs;
        # see __post_init__). The SDL surface tracks the resize on its own; we just relay
        # out the UI. The native minimum size (set at startup) keeps it from going too small.
        if (w, h) == (self.win_w, self.win_h):
            return
        self.win_w, self.win_h = w, h
        # Keep the sidebar within the new window (the left column must still fit).
        self.sidebar_w = max(
            SIDEBAR_W_MIN, min(self.sidebar_w, self.win_w - MIN_LEFT_PANEL_W - DIVIDER_W)
        )
        surface = pygame.display.get_surface()
        if surface is not None:
            self.screen = surface
        self.manager.set_window_resolution((w, h))
        self._build_ui()
        self._clamp_pan()  # keep the canvas in view after the viewport changed

    def _commit_steps(self) -> None:
        """Adopt the steps box value (clamped). Safe to call on Enter, on blur, or at Play.

        Idempotent: re-committing the same value is a no-op, so calling it just before a run
        starts (to pick up a number typed but not Enter-applied) never disturbs anything.
        """
        try:
            value = clamp_steps(self.steps_entry.get_text())
        except ValueError:
            self.steps_entry.set_text(str(self.steps))
            return
        self.steps_entry.set_text(str(value))
        if value == self.steps:
            return
        self.steps = value
        # If a run is paused, changing steps abandons it (respacing is fixed per run).
        if self.worker is not None and self.worker.is_alive():
            self._stop_run()
            self._status("Steps set")

    def _open_save_dialog(self) -> None:
        """Freeze the current frame and open a file dialog to choose where to write it."""
        if self._frame_surface is None:
            self._status("No frame")
            return
        if self._file_dialog is not None and self._file_dialog.alive():
            return  # a dialog is already open
        self._save_surface = self._frame_surface.copy()  # freeze; generation may advance
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        win_w, win_h = self._window_size()
        rect = pygame.Rect(0, 0, 560, 440)
        rect.center = (win_w // 2, win_h // 2)
        self._file_dialog = UIFileDialog(
            rect,
            self.manager,
            window_title="Save image",
            initial_file_path=str(self.out_dir / f"interactive_{stamp}.png"),
            allow_existing_files_only=False,
        )

    def _write_save(self, path_str: str) -> None:
        """Write the frozen frame to the path the dialog returned (defaulting to .png)."""
        if self._save_surface is None:
            return
        path = Path(path_str)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tga"):
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self._save_surface, str(path))
        self._save_surface = None
        self._status("Saved")
        log.info("saved %s", path)

    # -- history / revert --
    def _request_checkpoint(self, label: str) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.checkpoint(label)

    def _set_history_label(self) -> None:
        if self._preview_index is not None and self._preview_index < len(self._history):
            e = self._history[self._preview_index]
            text = f"{e.label} {e.index}/{e.total}"
        else:
            frame = self.worker.latest_frame() if self.worker is not None else None
            text = f"live {frame.index}/{frame.total}" if frame is not None else "live"
        if self.history_label.text != text:  # avoid re-rendering when unchanged
            self.history_label.set_text(text)

    def _sync_preview_prompts(self) -> None:
        """Show the previewed checkpoint's prompts in the rows (live set when not previewing)."""
        want: list[tuple[str, float]] | None = None
        if self._preview_index is not None and self._preview_index < len(self._history):
            want = self._history[self._preview_index].prompts
        cur = (
            [(p.text, p.weight) for p in self._preview_prompts]
            if self._preview_prompts is not None
            else None
        )
        if (list(want) if want is not None else None) == cur:
            return  # already showing the right prompts
        self._preview_prompts = [PromptRow(t, w) for t, w in want] if want is not None else None
        self._rebuild_prompt_rows()
        self._sync_enabled()

    def _refresh_preview_state(self) -> None:
        """Update the prompt rows, label, and Play gating whenever the preview changes."""
        self._sync_preview_prompts()
        self._set_history_label()
        (self.play_button.disable if self._preview_index is not None else self.play_button.enable)()

    def _history_total(self) -> int:
        """The run's display-step total (the history slider's right edge)."""
        if self._history:
            return max(1, self._history[-1].total)
        frame = self.worker.latest_frame() if self.worker is not None else None
        return max(1, frame.total if frame is not None else 1)

    def _live_index(self) -> int:
        """The current live display step (the slider's rightmost snap point)."""
        frame = self.worker.latest_frame() if self.worker is not None else None
        if frame is not None:
            return frame.index
        return self._history[-1].index if self._history else 0

    def _history_slider_start(self) -> float:
        """Where the thumb should sit: a previewed checkpoint's step, else the live step."""
        if self._preview_index is not None and self._preview_index < len(self._history):
            return float(self._history[self._preview_index].index)
        return float(self._live_index())

    def _history_snap(self, value: float) -> int | None:
        """Nearest checkpoint index to a slider step value; None == live (rightmost)."""
        points: list[tuple[float, int | None]] = [
            (float(cp.index), i) for i, cp in enumerate(self._history)
        ]
        points.append((float(self._live_index()), None))  # the live frame
        return min(points, key=lambda p: abs(p[0] - value))[1]

    def _rebuild_history_slider(self) -> None:
        """Recreate the step-space history slider (range follows the run's total steps)."""
        self.history_slider.kill()
        self.history_slider = pygame_gui.elements.UIHorizontalSlider(
            self._history_slider_rect,
            start_value=self._history_slider_start(),
            value_range=(0.0, float(max(self._history_total(), 1))),
            manager=self.manager,
        )

    def _sync_history(self) -> None:
        """Pull the worker's history each frame; rebuild the slider when it changes length."""
        self._history = self.worker.get_history() if self.worker is not None else []
        if len(self._history) != self._hist_len:
            self._hist_len = len(self._history)
            if self._preview_index is not None and self._preview_index >= self._hist_len:
                self._preview_index = None
            self._rebuild_history_slider()
            self._sync_enabled()
        elif self.running and self._preview_index is None and self._history:
            # While actively generating (not previewing), let the thumb track the live step as it
            # advances. When paused/done we leave it alone so a scrub isn't yanked back each frame.
            self.history_slider.set_current_value(float(self._live_index()))
        self._set_history_label()  # keep the live step current as it ticks

    def _displayed_surface(self) -> pygame.Surface | None:
        """The canvas surface to show: a previewed checkpoint, else the live frame."""
        if self._preview_index is not None and self._preview_index < len(self._history):
            entry = self._history[self._preview_index]
            if self._preview_key != id(entry.preview):
                self._preview_key = id(entry.preview)
                self._preview_surface = pygame.surfarray.make_surface(entry.preview.swapaxes(0, 1))
            return self._preview_surface
        return self._frame_surface

    # -- events --
    def _handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.VIDEORESIZE:
            # Resize events stream while dragging; coalesce to one relayout per frame
            # (see run()) instead of rebuilding the UI on every event.
            self._pending_size = (event.w, event.h)
            return True

        if event.type == pygame.MOUSEMOTION:
            self._mouse_pos = event.pos
            if self._dragging_divider:
                self._set_sidebar_width(self.win_w - event.pos[0] - DIVIDER_W // 2)
            elif self._panning:
                self._pan += pygame.Vector2(event.rel)
                self._clamp_pan()
            elif self._painting:
                self._paint_to(event.pos)
            return True
        if event.type == pygame.MOUSEBUTTONDOWN:
            # The draggable divider sits between the left column and the sidebar (full height).
            if event.button == 1 and abs(event.pos[0] - self._divider_x()) <= DIVIDER_W:
                self._dragging_divider = True
                return True
            on_canvas = self._image_region().collidepoint(event.pos)
            if event.button == 3 and on_canvas:  # right held = navigate mode (pan + scroll-zoom)
                self._navigating = True
                self._panning = True
                return True
            if event.button == 2 and on_canvas:  # middle-drag also pans
                self._panning = True
                return True
            if event.button == 1:  # left-drag on the canvas paints
                if self._on_swatch(event.pos):
                    return True
                # No painting while previewing history — it would be invisible and unapplied.
                if self._preview_index is None and self._screen_to_canvas(event.pos) is not None:
                    self._painting = True
                    self._last_gen = None
                    self._paint_to(event.pos)
                return True
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self._dragging_divider = False
                self._painting = False
                self._last_gen = None
            elif event.button == 3:
                self._navigating = False
                self._panning = False
            elif event.button == 2:
                self._panning = False
        if event.type == pygame.MOUSEWHEEL and self._image_region().collidepoint(self._mouse_pos):
            if self._navigating:  # canvas mode: wheel zooms toward the cursor
                self._zoom_at(self._mouse_pos, 1.15**event.y)
            elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
                self.brush_strength = max(0.05, min(1.0, self.brush_strength + event.y * 0.05))
                self.strength_slider.set_current_value(self.brush_strength)
            else:
                self.brush_size = max(4.0, min(160.0, self.brush_size * (1.1**event.y)))
                self.size_slider.set_current_value(self.brush_size)
            return True
        if event.type == pygame.KEYDOWN and not self._typing():
            if event.key == pygame.K_SPACE:
                self._toggle_play()
            elif event.key == pygame.K_f:
                self._fit_view()
            elif event.key == pygame.K_0:
                self._zoom_at(self._image_region().center, 1.0 / self._zoom)

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == self.play_button:
                self._toggle_play()
            elif event.ui_element == self.stop_button:
                self._stop_run()
                self._status("Stopped")
            elif event.ui_element == self.save_button:
                self._open_save_dialog()
            elif event.ui_element in (self.tab_settings, self.tab_current):
                self._sidebar_tab = (
                    "settings" if event.ui_element == self.tab_settings else "current"
                )
                self._sync_sidebar_tabs()
            elif event.ui_element in self._preset_buttons:
                self._apply_preset(self._preset_buttons[event.ui_element])
            elif event.ui_element == self.perlin_button:
                self.session.config.perlin_init = not self.session.config.perlin_init
                on = self.session.config.perlin_init
                (self.perlin_button.select if on else self.perlin_button.unselect)()
                self._status(f"Perlin {'on' if on else 'off'}")
            elif event.ui_element in self._clip_buttons:
                self._toggle_clip_model(event.ui_element)
            elif event.ui_element == self.secondary_button:
                self._secondary_on = not self._secondary_on
                (
                    self.secondary_button.select
                    if self._secondary_on
                    else self.secondary_button.unselect
                )()
                self._update_reload_queue()
            elif event.ui_element == self.add_button:
                self.prompts.append(PromptRow("", 1.0))
                self._rebuild_prompt_rows()
                self._push_prompts()
                self._request_checkpoint("add prompt")
            elif event.ui_element == self.apply_button:
                self._apply_size(
                    int_or(self.width_entry.get_text(), self.width),
                    int_or(self.height_entry.get_text(), self.height),
                )
            elif event.ui_element == self.swap_button:
                self._apply_size(self.height, self.width)
            elif event.ui_element in self._remove_buttons:
                idx = self._remove_buttons[event.ui_element]
                if 0 <= idx < len(self.prompts):
                    self.prompts.pop(idx)
                    self._rebuild_prompt_rows()
                    self._push_prompts()
                    self._request_checkpoint("remove prompt")
            elif event.ui_element in self._brush_buttons:
                self.brush_type = self._brush_buttons[event.ui_element]
                for button, name in self._brush_buttons.items():
                    (button.select if name == self.brush_type else button.unselect)()
            elif event.ui_element == self.noise_button:
                self.noise_mode = not self.noise_mode
                (self.noise_button.select if self.noise_mode else self.noise_button.unselect)()
            elif event.ui_element == self.clear_paint_button:
                self._paint_layer.clear()
            elif event.ui_element == self.revert_button:
                if self._preview_index is not None and self.worker is not None:
                    # Adopt the checkpoint's prompts as the live set, then branch from it.
                    self.prompts = [
                        PromptRow(t, w) for t, w in self._history[self._preview_index].prompts
                    ]
                    self.worker.seek(self._preview_index)
                    self._preview_index = None
                    self._preview_prompts = None
                    # Branching drops any in-flight strokes (the worker clears its queue on seek),
                    # so reset the overlay tracking to stay aligned with the worker's apply count.
                    self._pending_overlays = []
                    self._paint_submitted = self.worker.paint_applied_count
                    self.history_slider.set_current_value(float(self._live_index()))
                    self._rebuild_prompt_rows()  # now-live (reverted) prompts
                    self._push_prompts()  # apply them to the resumed run
                    self._sync_enabled()  # re-enable prompt editing
                    self._refresh_preview_state()
            elif event.ui_element == self.cancel_button:
                self._preview_index = None
                self.history_slider.set_current_value(float(self._live_index()))
                self._refresh_preview_state()

        elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            if event.ui_element == self.size_slider:
                self.brush_size = float(event.value)
            elif event.ui_element == self.strength_slider:
                self.brush_strength = float(event.value)
            elif event.ui_element == self.history_slider:
                # The slider is in step-space; snap the dragged value to the nearest checkpoint
                # (or live), and park the thumb on that checkpoint's actual step position.
                snap_idx = self._history_snap(float(event.value))
                self._preview_index = snap_idx
                if snap_idx is not None:
                    snapped = float(self._history[snap_idx].index)
                else:
                    snapped = float(self._live_index())
                self.history_slider.set_current_value(snapped)
                self._refresh_preview_state()
            elif event.ui_element == self._eta_slider:
                # eta is read when the loop's generator is built, so this lands on the next run.
                self.session.config.eta = float(event.value)
                self._eta_label.set_text(f"{event.value:.2f}")
                self._mark_custom()
            elif event.ui_element in self._scale_sliders:
                attr, is_int, vlabel, fmt = self._scale_sliders[event.ui_element]
                value: float | int = int(round(event.value)) if is_int else float(event.value)
                # session.config is the live config the running Sampler reads each step, so
                # this retunes guidance on the next step (and seeds the next run when stopped).
                setattr(self.session.config, attr, value)
                vlabel.set_text(fmt.format(value))
            else:
                slider_idx = self._weight_sliders.get(event.ui_element)
                if slider_idx is not None and 0 <= slider_idx < len(self.prompts):
                    self.prompts[slider_idx].weight = float(event.value)
                    self._refresh_rows()
                    self._push_prompts()

        elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element == self.steps_entry:
                self._commit_steps()
            elif event.ui_element in (self.width_entry, self.height_entry):
                pass  # applied via the Apply button
            elif event.ui_element in self._prompt_entries:
                self._commit_prompt_entry(event.ui_element)
            elif event.ui_element in self._schedule_entries:
                self._commit_schedule_entry(event.ui_element)

        elif event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
            if event.ui_element is self._file_dialog:
                self._write_save(event.text)
                self._file_dialog = None

        elif event.type == pygame_gui.UI_WINDOW_CLOSE:
            if event.ui_element is self._file_dialog:  # cancelled
                self._file_dialog = None
                self._save_surface = None

        return True

    # -- drawing --
    def _update_frame_surface(self) -> None:
        if self.worker is None:
            return
        frame = self.worker.latest_frame()
        if frame is None:
            return
        key = (id(frame.image), frame.index)
        if key == self._frame_key:
            return
        self._frame_key = key
        # pygame.surfarray expects (W, H, 3), so swap the first two axes.
        self._frame_surface = pygame.surfarray.make_surface(frame.image.swapaxes(0, 1))
        self.step_label.set_text(f"step {frame.index} / {frame.total}")

    def _auto_apply_on_blur(self) -> None:
        """Apply a text box when keyboard focus leaves it (no Enter needed).

        Covers both the prompt boxes and the steps box, so a value typed and then clicked
        away from is adopted just as if Enter had been pressed.
        """
        focus = self.manager.get_focus_set() or set()
        current = next((e for e in self._prompt_entries if e in focus), None)
        if current is not self._focused_entry:
            previous = self._focused_entry
            self._focused_entry = current
            if previous is not None and previous.alive():
                self._commit_prompt_entry(previous)
        steps_focused = self.steps_entry in focus
        if self._steps_focused and not steps_focused:
            self._commit_steps()
        self._steps_focused = steps_focused
        sched = next((e for e in self._schedule_entries if e in focus), None)
        if sched is not self._focused_schedule:
            previous = self._focused_schedule
            self._focused_schedule = sched
            if previous is not None and previous.alive():
                self._commit_schedule_entry(previous)

    def _draw(self) -> None:
        win_w, win_h = self._window_size()
        img_h = self._image_area_h()
        panel_w = self._panel_w()
        self.screen.fill(WINDOW_BG)
        pygame.draw.rect(self.screen, IMAGE_BG, (0, 0, panel_w, img_h))
        pygame.draw.rect(self.screen, PANEL_BG, (0, img_h, panel_w, win_h - img_h))
        pygame.draw.rect(self.screen, PANEL_BG, self._sidebar_rect())  # full-height sidebar
        # Draggable divider band between the left column and the sidebar.
        div = pygame.Rect(panel_w, 0, DIVIDER_W, win_h)
        hot = self._dragging_divider or abs(self._mouse_pos[0] - panel_w) <= DIVIDER_W
        pygame.draw.rect(self.screen, DIVIDER, div)
        grip_x = panel_w + DIVIDER_W // 2
        pygame.draw.line(
            self.screen,
            (110, 120, 140) if hot else (70, 78, 92),
            (grip_x, win_h // 2 - 14),
            (grip_x, win_h // 2 + 14),
            2,
        )
        # Draw the canvas (and unbaked paint overlay) under the view transform, clipped to
        # the viewport so a zoomed/panned canvas never spills into the panel.
        self.screen.set_clip(self._image_region())
        crect = self._canvas_screen_rect()
        surface = self._displayed_surface()
        if surface is not None:
            self._blit_canvas(surface)
        else:
            # No frame yet: show the canvas bounds so the size/aspect is clear before Play.
            pygame.draw.rect(self.screen, CANVAS_EMPTY_BG, crect)
            label = self._hud_font.render(
                f"{self.width} × {self.height} — press Play", True, (140, 147, 160)
            )
            self.screen.blit(label, label.get_rect(center=crect.center))
        # Paint overlay only on the live view (hidden while previewing history).
        if self._preview_index is None and not self._paint_layer.empty():
            self._blit_canvas(self._paint_layer.to_surface())
        pygame.draw.rect(self.screen, CANVAS_BORDER, crect, 1)  # canvas outline at any zoom
        self.screen.set_clip(None)
        pygame.draw.line(self.screen, DIVIDER, (0, img_h), (panel_w, img_h))

    def _draw_tools(self) -> None:
        """Draw the colour palette, the brush-preview ring, and the canvas help HUD."""
        # Palette: current-colour preview + swatches (selected one outlined).
        pygame.draw.rect(self.screen, self.brush_color, self._color_preview_rect, border_radius=5)
        pygame.draw.rect(self.screen, DIVIDER, self._color_preview_rect, width=1, border_radius=5)
        for sr, color in self._swatch_rects:
            pygame.draw.rect(self.screen, color, sr, border_radius=4)
            if color == self.brush_color:
                pygame.draw.rect(self.screen, (255, 255, 255), sr, width=2, border_radius=4)
        region = self._image_region()
        # Brush ring (scaled by zoom) — only in draw mode (not navigating, not previewing).
        if (
            not self._navigating
            and self._preview_index is None
            and region.collidepoint(self._mouse_pos)
        ):
            ring = max(2, int(self.brush_size * self._zoom))
            pygame.draw.circle(self.screen, self.brush_color, self._mouse_pos, ring, 2)
            pygame.draw.circle(self.screen, (255, 255, 255), self._mouse_pos, ring + 1, 1)
        # Help HUD in the corner of the canvas (doesn't cost panel height), per mode.
        text = self._hud_font.render(
            NAV_HELP if self._navigating else DRAW_HELP, True, (210, 214, 222)
        )
        pad = 6
        chip = pygame.Surface(
            (text.get_width() + 2 * pad, text.get_height() + 2 * pad), pygame.SRCALPHA
        )
        chip.fill((0, 0, 0, 120))
        pos = (10, region.bottom - chip.get_height() - 10)
        self.screen.blit(chip, pos)
        self.screen.blit(text, (pos[0] + pad, pos[1] + pad))

    # -- painting --
    def _on_swatch(self, pos: tuple[int, int]) -> bool:
        for sr, color in self._swatch_rects:
            if sr.collidepoint(pos):
                self.brush_color = color
                return True
        return False

    def _paint_to(self, pos: tuple[int, int]) -> None:
        gen = self._screen_to_canvas(pos)
        if gen is None:
            return
        c = self.brush_color
        color01 = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
        tint = 1.0 if self.noise_mode else 0.0
        if self._last_gen is None:
            self._paint_layer.stamp(
                gen[0], gen[1], self.brush_size, color01, self.brush_strength, self.brush_type, tint
            )
        else:
            self._paint_layer.stroke(
                self._last_gen,
                gen,
                self.brush_size,
                color01,
                self.brush_strength,
                self.brush_type,
                tint,
            )
        self._last_gen = gen

    def _sync_paint(self) -> None:
        """Hand new strokes to the worker, and clear the overlay once a step has baked them."""
        layer = self._paint_layer
        if layer.dirty and self.worker is not None and self.worker.is_alive():
            layer.dirty = False
            rgb, alpha, tint = layer.snapshot()
            # Gamma-shape the *injected* mask for noise-mode pixels (tint/alpha = how much of
            # the pixel is noise-mode), keeping the on-screen overlay at the raw opacity.
            frac_noise = np.divide(tint, alpha, out=np.zeros_like(tint), where=alpha > 1e-6)
            shaped = NOISE_MAX_INJECT * alpha**NOISE_OPACITY_GAMMA
            alpha = alpha * (1.0 - frac_noise) + shaped * frac_noise
            self.worker.set_paint(rgb, alpha, tint)
            self._paint_awaiting_bake = True
            self._paint_baseline = self.worker.paint_applied_count
        # Clear the overlay only when a *published frame* that incorporated the paint arrives —
        # not merely when the worker started applying it. The injecting step takes seconds, so
        # clearing on apply would make the stroke vanish before the baked image catches up.
        if self._paint_awaiting_bake and not self._painting and self.worker is not None:
            frame = self.worker.latest_frame()
            if frame is not None and frame.paint_applied > self._paint_baseline:
                layer.clear()
                self._paint_awaiting_bake = False

    # -- main loop --
    def run(self) -> None:
        clock = pygame.time.Clock()
        alive = True
        while alive:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                self.manager.process_events(event)
                if not self._handle_event(event):
                    alive = False
            # Apply at most one window relayout per frame (resize events are coalesced).
            if self._pending_size is not None:
                w, h = self._pending_size
                self._pending_size = None
                self._resize_window(w, h)
            # Fit the canvas to the available area on the first realized frame (after any
            # startup window-manager resize has been applied above).
            if not self._did_initial_fit:
                self._did_initial_fit = True
                self._fit_view()
            # Rebuild once per frame after a sidebar-divider drag (also coalesced).
            if self._sidebar_dirty:
                self._sidebar_dirty = False
                self._build_ui()
                self._clamp_pan()
            # Fire the debounced model auto-reload once its delay has elapsed.
            if (
                self._reload_queued_at is not None
                and not self._reloading
                and pygame.time.get_ticks() >= self._reload_queued_at
            ):
                self._reload_queued_at = None
                self._start_reload()
            # When the run finishes (worker idles) or exits unexpectedly, drop to a paused
            # state so controls/history are usable and the image can still be reverted.
            wk = self.worker
            if wk is not None and not self.paused and (wk.finished or not wk.is_alive()):
                self.paused = True
                if wk.is_alive():
                    wk.pause()
                self._status("Done")
                self._sync_enabled()
            if wk is not None and wk.notice is not None:  # e.g. compile OOM -> eager fallback
                self._status(wk.notice)
                wk.notice = None
            self._poll_reload()  # swap in a reloaded session once its background thread finishes
            self._auto_apply_on_blur()
            # Live "edited · Enter" badge while typing (cheap; only mutates on change).
            self._refresh_rows()
            self._refresh_current()  # keep the "Current" sidebar tab in sync
            self._sync_paint()
            self._sync_history()
            self._update_frame_surface()
            self.manager.update(dt)
            self._draw()
            self.manager.draw_ui(self.screen)
            self._draw_tools()
            pygame.display.flip()
        self._stop_run()


STEP_MIN, STEP_MAX = 1, 1000


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100, help="Initial total step count.")
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the UNet/CLIP for faster steps (off by default). Worth it on a "
        "GPU with free VRAM headroom or for smaller/lighter runs (~1.4x); the first run at "
        "each size warms up (~60s, cached on disk). Robust: compile-time errors and OOM fall "
        "back to eager rather than crashing. Off by default because the heavy presets at "
        "large sizes need ~21GB to compile, which can exceed a shared GPU's free memory - and "
        "eager already runs the 2022 preset at 1280x768 in ~1.5min.",
    )
    ap.add_argument("--cpu", action="store_true", help="Force CPU (very slow).")
    # Defaults are repo-root-relative, matching the library, so weights/outputs are shared
    # with it (run from the repo root: `uv run disco-studio`) rather than re-downloaded.
    ap.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Where model weights live (defaults to the library's models dir).",
    )
    ap.add_argument("--out", type=Path, default=Path("images_out"), help="Where Save writes PNGs.")
    args = ap.parse_args()

    # By default SDL swallows the mouse click that brings an unfocused window to the
    # foreground: that click only focuses the window and is never delivered to the app, so
    # the first click on the studio after another window had focus is silently dropped
    # (buttons highlight on hover but don't fire until a second click). This hint passes that
    # focusing click through as a normal click. Must be set before the video subsystem inits.
    os.environ.setdefault("SDL_MOUSE_FOCUS_CLICKTHROUGH", "1")
    pygame.init()
    config = RunConfig(
        compile=args.compile,
        cpu=args.cpu,
        models_dir=args.models_dir,
        width=snap_side(args.width),
        height=snap_side(args.height),
        steps=args.steps,
    )
    log.info("loading models (this can take a minute)…")
    session = DiscoSession(config)
    log.info("models loaded")

    app = App(
        session=session,
        out_dir=args.out,
        width=args.width,
        height=args.height,
        steps=args.steps,
    )
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
