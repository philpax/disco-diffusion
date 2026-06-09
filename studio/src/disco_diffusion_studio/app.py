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
import os
import random
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame
import pygame_gui
from disco_diffusion import DiscoSession, EncodedPrompt, RunConfig
from disco_diffusion.config import AVAILABLE_CLIP_MODELS, parse_schedule
from PIL import Image
from pygame_gui.windows import UIColourPickerDialog, UIConfirmationDialog, UIMessageWindow

from . import _ui_build, _ui_draw, _ui_events, native_dialog
from .bottom_bar import BottomBar
from .canvas import Canvas
from .constants import (
    APP_TITLE,
    RELOAD_DEBOUNCE_MS,
)
from .controls import CURRENT_PERRUN, CUSTOM_PRESET, LIVE_SCALES, PromptRow
from .init_image import InitImage
from .layout import (
    CTRL_H,
    DEFAULT_H,
    DEFAULT_W,
    DIVIDER_W,
    LABEL_H,
    MIN_IMAGE_H,
    MIN_LEFT_PANEL_W,
    MIN_WINDOW_W,
    PANEL_H,
    PANEL_MIN,
    SIDEBAR_W_DEFAULT,
    Layout,
    snap_side,
)
from .paint import Brush
from .palette import Palette
from .presets import (
    HistoryItem,
    Preset,
    PresetConfig,
    Session,
    load_presets,
    load_session,
    match_preset,
    save_preset,
    save_session,
)
from .reload import ModelReloader
from .sidebar import Sidebar
from .theme import (
    THEME,
    WINDOW_BG,
)
from .timeline import Timeline
from .util import clamp_steps
from .worker import GenerationWorker, HistoryEntry, PromptSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("disco_diffusion_studio")

# Tuning constants (APP_TITLE, BRUSH_*, STEP_*, *_HELP, CANVAS_*, *_MS) live in .constants.


# Preset / PresetConfig and their loading/saving live in .presets (TOML on disk); the advanced-
# control tables (LIVE_SCALES / SCHEDULES / CURRENT_PERRUN) + PromptRow live in .controls.


def compute_window_size(width: int, height: int, sidebar_w: int, panel_h: int) -> tuple[int, int]:
    """The initial window size: the canvas fitted into the image area, plus the chrome.

    Seeded once from the desktop size (the window is user-resizable thereafter). Shared by the
    startup loading screen (``main``) and ``App.__post_init__`` so both open at the same size.
    """
    chrome_w = sidebar_w + DIVIDER_W  # width taken by the sidebar + its divider
    try:
        sizes = pygame.display.get_desktop_sizes()
        desk_w, desk_h = sizes[0] if sizes and sizes[0][0] > 0 else (1920, 1080)
    except (pygame.error, IndexError):  # headless / older pygame
        desk_w, desk_h = 1920, 1080
    avail_w = max(320, int(desk_w * 0.9) - chrome_w)
    avail_h = max(240, int(desk_h * 0.9) - panel_h)
    # Fit the canvas into the available image area, never upscaling past 1:1.
    scale = min(avail_w / width, avail_h / height, 1.0)
    img_w, img_h = int(width * scale), int(height * scale)
    # The left column is at least MIN_LEFT_PANEL_W wide; the sidebar always gets its full width on
    # top of that, so the chrome never squeezes either below its minimum.
    return max(MIN_LEFT_PANEL_W, img_w) + chrome_w, max(MIN_IMAGE_H, img_h) + panel_h


# --- app ---------------------------------------------------------------------


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

    # Late-bound runtime objects + widgets, built in __post_init__/_build_ui (so the UI builders,
    # event router, and renderer can live in the _ui_*.py modules and still see typed attributes).
    # field(init=False) keeps them out of the constructor; they're unset until the build runs.
    manager: pygame_gui.UIManager = field(init=False)
    screen: pygame.Surface = field(init=False)
    layout: Layout = field(init=False)  # window geometry (split + divider positions)
    canvas: Canvas = field(init=False)  # image area: view transform + paint layer + frame
    sidebar: Sidebar = field(init=False)  # right sidebar: Settings/Current tabs + their widgets
    bottom_bar: BottomBar = field(init=False)  # transport / history / paint tools / prompts

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
        sidebar_w = SIDEBAR_W_DEFAULT
        # Bottom-panel height — seeded to its natural default, then user-owned (draggable via a
        # horizontal divider, clamped to [PANEL_MIN, win_h - MIN_IMAGE_H]).
        panel_h = PANEL_H
        # If a window already exists — the startup loading screen creates it at this exact size
        # (see main()) — adopt it as-is, so we don't re-issue a SetWindowSize that a tiling WM
        # would fight, and so there's no visible resize after the models finish loading. Only
        # create the window here when constructed standalone (no loading screen, e.g. tests).
        existing = pygame.display.get_surface()
        if existing is not None:
            self.screen = existing
            win_w, win_h = existing.get_size()
        else:
            win_w, win_h = compute_window_size(self.width, self.height, sidebar_w, panel_h)
            self.screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
        self.layout = Layout(win_w, win_h, sidebar_w, panel_h)
        pygame.display.set_caption(APP_TITLE)
        # Enforce the minimum size natively (SDL) instead of by re-calling set_mode on
        # resize. Re-calling set_mode issues a SetWindowSize that a tiling WM treats as the
        # app demanding geometry, snapping a floated window back under tiling. With a native
        # minimum we just adopt whatever size the window becomes and never fight the WM.
        try:
            window = pygame.Window.from_display_module()
            window.minimum_size = (MIN_WINDOW_W, MIN_IMAGE_H + PANEL_MIN)
        except (AttributeError, pygame.error):  # older pygame / headless: best-effort only
            pass
        self.manager = pygame_gui.UIManager((win_w, win_h))
        self.manager.get_theme().load_theme(io.StringIO(json.dumps(THEME)))
        # The bottom panel (transport / history / paint tools / prompts) owns its widgets, built
        # in _build_ui. The Sidebar (right column) does likewise; the App coordinates the two.
        self.bottom_bar = BottomBar()
        # The prompt entry that last held keyboard focus, so we can auto-apply its text
        # when focus moves away (no need to press Enter).
        self._focused_entry: pygame_gui.elements.UITextEntryLine | None = None
        # Whether the steps box held focus last frame, so we can commit it on blur (mirrors
        # the prompt boxes; without this, typing a value and clicking away wouldn't apply it).
        self._steps_focused = False
        # Seed field contents, persisted across UI rebuilds. Always holds a concrete seed so the
        # value in use is visible upfront and replaying (Play again) clearly reuses it; "Rnd"
        # rolls a fresh one. Seeded random at startup.
        self._seed_text = str(random.randrange(2**31))

        # Right-hand sidebar: owns its widgets (Settings/Current tabs + controls), built in
        # _build_ui. The coalesced "rebuild after a resize" flag is applied once per frame in
        # run(); its width is on self.layout.
        self.sidebar = Sidebar()
        self._dragging_divider = False
        self._sidebar_dirty = False
        # Horizontal divider between the image area and the bottom panel (drag to resize the
        # panel). Like the sidebar divider, the rebuild is coalesced to one per frame.
        self._dragging_panel = False
        self._panel_dirty = False
        # Presets are loaded from studio/presets/*.toml and surfaced as a dropdown that flips to
        # "Custom" once any preset-controlled knob is edited. _applying_preset suppresses that
        # flip while a preset is being applied (so its own widget updates don't read as edits).
        self._presets: dict[str, Preset] = load_presets()
        self._applying_preset = False
        self._save_preset_window: pygame_gui.elements.UIWindow | None = None
        self._save_preset_entry: pygame_gui.elements.UITextEntryLine | None = None
        self._save_preset_ok: pygame_gui.elements.UIButton | None = None
        self._save_preset_cancel: pygame_gui.elements.UIButton | None = None
        # The schedule box that last held focus, so its text is committed on blur (like steps).
        self._focused_schedule: pygame_gui.elements.UITextEntryLine | None = None

        # Model selection (CLIP set + secondary). Changing these needs a full weight reload,
        # so the toggles stage a pending selection and *queue* an auto-reload (debounced); it
        # fires after the user stops changing, and un-queues if they revert to the loaded set.
        self._clip_selected: set[str] = set(self.session.config.clip_models)
        self._secondary_on: bool = self.session.config.use_secondary_model
        self._reloader = ModelReloader()  # background weight reload + debounced trigger
        # pygame ticks at which to drop a guidance-change checkpoint (set when a guidance slider
        # moves, pushed back on each further move, fired once the value settles). None = idle.
        self._guidance_checkpoint_at: int | None = None
        # The preset matching the loaded session (or "Custom"); the dropdown's selected entry.
        self._preset_selection = self._detect_preset()

        # "Current" sidebar tab: snapshot of the per-run settings the active run was started
        # with (live knobs are read straight from session.config), plus the label cache.
        self._run_snapshot: dict[str, str] = {}

        # img2img init image (per-run, applies on next Play): seed the run from an image instead
        # of noise, set by Open…/drag-drop (a file) or "Use current" (the on-screen frame). The
        # InitImage owns the source image, denoise %, and the canvas preview shown before Play.
        self._init = InitImage()
        # "Reset canvas" confirmation (discards the rendered frame so the init preview shows).
        self._confirm_dialog: UIConfirmationDialog | None = None
        # Transient error/notice modal (e.g. loading a session via the init button, or vice versa).
        self._message_window: UIMessageWindow | None = None

        # Colours: the palette + recently-picked colours come from studio/config.toml. The RGB
        # picker prepends to the recents (capped, persisted), and swatches = palette + recents.
        self._palette = Palette.load()
        self._colour_picker: UIColourPickerDialog | None = None

        # Painting. The Brush holds the live brush state (kind/size/opacity/colour/noise); the
        # PaintController owns the active paint layer + stroke lifecycle: on mouse-up a stroke is
        # flushed to the worker as one batch (its own checkpoint) and kept on-screen as a pending
        # overlay until a baked frame incorporates it. brush.color is shared with the palette.
        self.brush = Brush(color=self._palette.default_brush())
        # The Canvas owns the image area: the view transform (zoom/pan), the paint controller
        # (active stroke + overlays), and the latest rendered frame surface.
        self.canvas = Canvas(self)
        self._mouse_pos: tuple[int, int] = (0, 0)
        self._panning = False
        self._navigating = False  # right mouse held: canvas-navigation mode (pan + scroll-zoom)
        self._hud_font = pygame.font.SysFont(None, 19)

        # Edit history / revert preview. The Timeline owns the checkpoint list, the preview cursor
        # (None = live frame; an int = previewing that checkpoint non-destructively until Revert),
        # the Ctrl+Z undo cursor, and the scrub maths. _preview_prompts (the prompt rows shown
        # while previewing) stays on App, as it's prompt-row UI state.
        self._timeline = Timeline()
        self._preview_prompts: list[PromptRow] | None = None  # prompts shown while previewing
        self.bottom_bar._history_slider_rect = pygame.Rect(0, 0, 10, 10)
        # Fit the canvas to the available area once the real window size is settled. The window
        # manager may resize the window right after it opens (firing a VIDEORESIZE that only
        # re-clamps), so the fit happens on the first run-loop frame, not here — by then any
        # startup resize has landed. (_zoom/_pan keep their defaults above until then.)
        self._did_initial_fit = False
        self._build_ui()

    # -- geometry --
    # The window is a left column (image on top, bottom control panel below) plus a full-height
    # right sidebar; the Layout owns the split maths. These thin wrappers keep the call sites
    # short (and let the bottom panel's rebuild be coalesced to one per frame via the dirty flag).
    def _panel_w(self) -> int:
        return self.layout.panel_w()

    def _divider_x(self) -> int:
        return self.layout.divider_x()

    def _sidebar_rect(self) -> pygame.Rect:
        return self.layout.sidebar_rect()

    def _bottom_panel_rect(self) -> pygame.Rect:
        return self.layout.bottom_panel_rect()

    def _image_area_h(self) -> int:
        return self.layout.image_area_h()

    def _set_panel_height(self, height: int) -> None:
        """Set the bottom-panel height (clamped); the rebuild is coalesced to one per frame."""
        if self.layout.set_panel_height(height):
            self._panel_dirty = True

    def _window_size(self) -> tuple[int, int]:
        return self.layout.window_size()

    def _centered_rect(self, w: int, h: int) -> pygame.Rect:
        return self.layout.centered_rect(w, h)

    def _image_region(self) -> pygame.Rect:
        return self.layout.image_region()

    # The view transform / paint layer / frame live on the Canvas; these thin wrappers keep the
    # call sites short (the Canvas supplies the ViewTransform with the live region + canvas size).
    def _canvas_screen_rect(self) -> pygame.Rect:
        return self.canvas.canvas_screen_rect()

    def _fit_view(self) -> None:
        self.canvas.fit()

    def _zoom_at(self, pos: tuple[int, int], factor: float) -> None:
        self.canvas.zoom_at(pos, factor)

    def _clamp_pan(self) -> None:
        self.canvas.clamp_pan()

    def _screen_to_canvas(self, pos: tuple[int, int]) -> tuple[float, float] | None:
        return self.canvas.screen_to_canvas(pos)

    def _blit_canvas(self, surf: pygame.Surface) -> None:
        self.canvas.blit(surf)

    def _typing(self) -> bool:
        """True while a text box has keyboard focus (so shortcut keys don't steal input)."""
        focus = self.manager.get_focus_set()
        return bool(focus) and any(
            isinstance(e, pygame_gui.elements.UITextEntryLine) for e in focus
        )

    def _modal_open(self) -> bool:
        """True while an in-window dialog (RGB picker / save preset / reset confirm) is open.

        Those are ``UIWindow``s pygame_gui draws over the canvas, but our raw mouse/keyboard
        canvas handling runs regardless — so while one is up we suppress painting, panning,
        zooming, swatch clicks, shortcuts, and the brush cursor to keep it from leaking through.
        (The native Save/Open dialogs are separate OS windows that block the loop, so they need
        no tracking here.)
        """
        return any(
            w is not None and w.alive()
            for w in (
                self._colour_picker,
                self._save_preset_window,
                self._confirm_dialog,
                self._message_window,
            )
        )

    def _show_message(self, title: str, message: str) -> None:
        """Pop a small OK-dismissable modal to surface a user error (e.g. wrong load button)."""
        if self._message_window is not None and self._message_window.alive():
            self._message_window.kill()
        rect = self._centered_rect(420, 200)
        self._message_window = UIMessageWindow(
            rect, html_message=message, manager=self.manager, window_title=title
        )

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
        return _ui_build._build_ui(self)

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
        if not self.sidebar._current_labels:
            return
        cfg = self.session.config
        for sc in LIVE_SCALES:  # live: reflect session.config as sliders move
            label = self.sidebar._current_labels.get(sc.attr)
            if label is not None:
                text = sc.fmt.format(float(getattr(cfg, sc.attr)))
                if label.text != text:
                    label.set_text(text)
        # While a run exists (playing, paused, or done) these reflect that run's frozen snapshot —
        # what the image on screen was generated with; only once fully stopped do they show the
        # pending values the next run would use.
        perrun = self._run_snapshot if self.worker is not None else self._perrun_values()
        for key, _name in CURRENT_PERRUN:
            label = self.sidebar._current_labels.get(key)
            text = perrun.get(key, "—")
            if label is not None and label.text != text:
                label.set_text(text)

    def _commit_schedule_entry(self, entry: pygame_gui.elements.UITextEntryLine) -> None:
        """Validate a cut-schedule box and store it on the config (applies on next Play).

        Schedules are parsed with the library's own parser; on a malformed string we flag it
        and restore the previous value rather than letting the worker blow up at run start.
        """
        attr = self.sidebar._schedule_entries.get(entry)
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
        self._mark_custom()

    def _current_preset(self) -> Preset:
        """Capture the live settings (guidance + per-run + schedules + models) as a Preset."""
        config = PresetConfig.from_run_config(self.session.config)
        models = [m for m in AVAILABLE_CLIP_MODELS if m in self._clip_selected]
        return Preset(config=config, clip_models=models, use_secondary_model=self._secondary_on)

    def _detect_preset(self) -> str:
        """The saved preset whose recipe matches the live settings, else "Custom"."""
        return match_preset(self._presets, self._current_preset()) or CUSTOM_PRESET

    def _set_preset_selection(self, name: str) -> None:
        """Set the dropdown's selected entry (rebuilding it, since it has no set-selected API)."""
        if self._preset_selection == name and self.sidebar.preset_dropdown is not None:
            return
        self._preset_selection = name
        self.sidebar.spawn_preset_dropdown(self)

    def _mark_custom(self) -> None:
        """Flip the preset dropdown to "Custom" after the user edits a preset-controlled knob."""
        if self._applying_preset:
            return
        self._set_preset_selection(CUSTOM_PRESET)

    def _apply_recipe(
        self, config: PresetConfig, clip_models: list[str], use_secondary: bool
    ) -> None:
        """Apply a recipe's config knobs (now) + stage its model set (a change auto-reloads).

        Shared by preset and session loads; wrapped in _applying_preset so the widget updates
        don't read as user edits (which would flip the preset dropdown to "Custom").
        """
        self._applying_preset = True
        try:
            cfg = self.session.config
            for attr, value in config.model_dump().items():
                setattr(cfg, attr, value)
            self.sidebar.refresh_advanced_widgets(self)
            self._clip_selected = set(clip_models)
            self._secondary_on = use_secondary
            for button, mname in self.sidebar._clip_buttons.items():
                (button.select if mname in self._clip_selected else button.unselect)()
            sec = self.sidebar.secondary_button
            (sec.select if self._secondary_on else sec.unselect)()
            self._update_reload_queue()
        finally:
            self._applying_preset = False

    def _apply_preset(self, name: str) -> None:
        """Load a full-recipe preset: config knobs apply now; a model change auto-reloads."""
        preset = self._presets.get(name)
        if preset is None:
            return
        self._apply_recipe(preset.config, preset.clip_models, preset.use_secondary_model)
        self._preset_selection = name
        # A preset retunes the live guidance, so record a revert point (discrete change — no
        # debounce); supersede any pending guidance-drag checkpoint.
        self._guidance_checkpoint_at = None
        self._request_checkpoint(f"preset {name}")
        self._status(f"Loaded {name}")

    # -- sessions (full working state) --
    def _current_session(self) -> Session:
        """Capture the whole working state (prompts + output + denoise + recipe) as a Session."""
        recipe = self._current_preset()
        return Session(
            width=self.width,
            height=self.height,
            steps=self.steps,
            seed=self._seed_for_run(),  # also fills the field with the seed in use
            denoise=self._init.denoise,
            prompts=[PromptSpec(r.text, r.weight, r.muted) for r in self.prompts],
            config=recipe.config,
            clip_models=recipe.clip_models,
            use_secondary_model=recipe.use_secondary_model,
        )

    def _current_image(self) -> Image.Image | None:
        """The rendered result currently on the canvas as a PIL image (for the session zip)."""
        surface = self._displayed_surface()
        if surface is None:
            return None
        return surface_to_pil(surface)

    def _history_for_save(self) -> list[tuple[HistoryItem, Image.Image]]:
        """The current edit history as (metadata, preview image) pairs for the session zip."""
        return [
            (
                HistoryItem(
                    label=e.label,
                    step=e.step,
                    index=e.index,
                    total=e.total,
                    prompts=[PromptSpec(t, w, m) for t, w, m in e.prompts],
                    config=e.config,
                ),
                Image.fromarray(e.preview),
            )
            for e in self._timeline.entries
        ]

    def _native_path(self, kind: str, title: str) -> str | None:
        """Open a native file dialog (``kind`` ``"save"``/``"open"``) rooted at ``out_dir``.

        Returns the chosen path, or ``None`` if cancelled or no backend is installed (reported
        via the status line). Ensures ``out_dir`` exists so the dialog opens there.
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        pick = native_dialog.save_file if kind == "save" else native_dialog.open_file
        try:
            return pick(title=title, start_dir=str(self.out_dir))
        except native_dialog.Unavailable:
            self._status("No native dialog (install zenity)")
            return None

    def _save_session(self) -> None:
        """Save the whole working state + result + history to a .zip via the native Save dialog."""
        path = self._native_path("save", "Save session")
        if not path:
            return
        try:
            saved = save_session(
                path, self._current_session(), self._current_image(), self._history_for_save()
            )
        except Exception as exc:  # noqa: BLE001 - surface the failure instead of crashing
            log.exception("saving session failed")
            self._status(f"Save failed: {exc}")
            return
        self._status(f"Saved session {saved.name}")

    def _load_session(self) -> None:
        """Load a session .zip via the native Open dialog and apply it (result -> init image)."""
        path = self._native_path("open", "Open session")
        if not path:
            return
        if not zipfile.is_zipfile(path):  # an image (or other file) picked via the session button
            self._show_message(
                "Not a session",
                f"<b>{Path(path).name}</b> isn't a session bundle (.zip)."
                "<br><br>If it's an image, use the <b>Init image → Open…</b> button to load it.",
            )
            return
        try:
            session, image, history = load_session(path)
        except Exception as exc:  # noqa: BLE001 - bad/old file shouldn't crash the app
            log.exception("loading session failed")
            self._show_message("Bad session", f"Couldn't load <b>{Path(path).name}</b>:<br>{exc}")
            return
        self._apply_session(session)
        # The bundled result becomes the init image so Play continues from it; without one, clear.
        # It's also painted onto the canvas as the (static) final frame, so the timeline treats it
        # as the rightmost endpoint — scrubbing visits it just like a freshly finished run's last
        # step, rather than snapping back to the last recorded checkpoint.
        if image is not None:
            self._set_init_image(image, "session result")
            arr = np.asarray(image.convert("RGB"))  # (H, W, 3)
            self.canvas.frame_surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        else:
            self._clear_init()
            self.canvas.frame_surface = None
        # Restore the scrubbable timeline (previews only, latent=None) — _sync_history keeps it
        # while there's no worker, and Revert continues from a checkpoint's preview via img2img.
        self._timeline.entries = [
            HistoryEntry(
                latent=None,
                step=item.step,
                index=item.index,
                total=item.total,
                preview=np.asarray(preview),
                label=item.label,
                prompts=[PromptSpec(t, w, m) for t, w, m in item.prompts],
                config=item.config,
            )
            for item, preview in history
        ]
        self._timeline.preview_index = None
        self._sync_enabled()

    def _apply_session(self, session: Session) -> None:
        """Adopt a loaded session: stop the run, set output + prompts + the recipe for next Play."""
        self._apply_size(session.width, session.height)  # also stops the run + rebuilds the canvas
        self.steps = session.steps
        self.sidebar.steps_entry.set_text(str(self.steps))
        self._seed_text = str(session.seed)
        self.sidebar.seed_entry.set_text(self._seed_text)
        self._init.denoise = session.denoise
        self.sidebar._init_denoise_slider.set_current_value(float(self._init.denoise))
        self.sidebar._init_denoise_label.set_text(f"{self._init.denoise}%")
        self.prompts = [PromptRow(t, w, m) for t, w, m in session.prompts] or [PromptRow("", 1.0)]
        self.bottom_bar.rebuild_prompt_rows(self)
        self._apply_recipe(session.config, session.clip_models, session.use_secondary_model)
        self._preset_selection = self._detect_preset()
        self.sidebar.spawn_preset_dropdown(self)
        self._run_snapshot = self._perrun_values()
        self._status("Session loaded — press Play")

    def _open_save_preset_dialog(self) -> None:
        """Open a small modal asking for a filename to save the current settings as a preset."""
        if self._save_preset_window is not None and self._save_preset_window.alive():
            return
        rect = self._centered_rect(420, 168)
        ui = pygame_gui.elements
        self._save_preset_window = ui.UIWindow(
            rect, self.manager, window_display_title="Save preset"
        )
        cont = self._save_preset_window
        inner_w = rect.width - 32
        ui.UILabel(pygame.Rect(6, 4, inner_w, LABEL_H), "Filename", self.manager, container=cont)
        self._save_preset_entry = ui.UITextEntryLine(
            pygame.Rect(6, 32, inner_w, CTRL_H), self.manager, container=cont
        )
        self._save_preset_entry.set_text("my-preset")
        self._save_preset_entry.focus()
        self._save_preset_ok = ui.UIButton(
            pygame.Rect(inner_w - 150, 78, 72, CTRL_H),
            "Save",
            self.manager,
            container=cont,
            object_id="#add_button",
        )
        self._save_preset_cancel = ui.UIButton(
            pygame.Rect(inner_w - 70, 78, 72, CTRL_H), "Cancel", self.manager, container=cont
        )

    def _close_save_preset_dialog(self) -> None:
        if self._save_preset_window is not None:
            self._save_preset_window.kill()
        self._save_preset_window = None
        self._save_preset_entry = None
        self._save_preset_ok = None
        self._save_preset_cancel = None

    def _save_current_preset(self) -> None:
        """Write the current settings to presets/<filename>.toml and select the new preset."""
        if self._save_preset_entry is None:
            return
        filename = self._save_preset_entry.get_text().strip() or "preset"
        try:
            name, _path = save_preset(filename, self._current_preset())
        except Exception as exc:  # noqa: BLE001 - surface the failure instead of crashing
            log.exception("saving preset failed")
            self._status(f"Save failed: {exc}")
            return
        self._presets = load_presets()
        self._close_save_preset_dialog()
        self._set_preset_selection(name)
        self._status(f"Saved {name}")

    def _toggle_clip_model(self, button: pygame_gui.elements.UIButton) -> None:
        """Stage a CLIP model in/out of the pending selection (queues the auto-reload)."""
        name = self.sidebar._clip_buttons[button]
        if name in self._clip_selected:
            self._clip_selected.discard(name)
            button.unselect()
        else:
            self._clip_selected.add(name)
            button.select()
        self._update_reload_queue()
        self._mark_custom()

    def _start_reload(self) -> None:
        """Rebuild the session with the staged CLIP set / secondary toggle on a worker thread.

        Reloading weights takes ~a minute, so it runs off the UI thread; the run loop polls
        :meth:`_poll_reload` and swaps the session in when it's done. Play stays disabled and
        the staged selection is locked (via _sync_enabled) until then.
        """
        if self._reloader.reloading:
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
        self._reloader.start(new_cfg, self.session.device)
        self._status("Reloading…")
        self._sync_enabled()

    def _poll_reload(self) -> None:
        """Swap in a reloaded session once the background reload finishes (called per frame)."""
        result = self._reloader.poll()
        if result is None:
            return
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
        return _ui_build._sync_enabled(self)

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
            if self._reloader.queued:
                self._reloader.cancel()
                self._status("Reload cancelled")
        else:
            self._reloader.schedule(pygame.time.get_ticks() + RELOAD_DEBOUNCE_MS)
            self._status("Reload queued")

    def _status(self, text: str) -> None:
        self.bottom_bar.status_label.set_text(text)

    # -- prompt snapshot --
    def _prompt_snapshot(self) -> list[PromptSpec]:
        return [PromptSpec(r.text, r.weight, r.muted) for r in self.prompts]

    def _push_prompts(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.set_prompts(self._prompt_snapshot())

    def _commit_prompt_entry(self, entry: pygame_gui.elements.UITextEntryLine) -> None:
        """Apply a prompt text box's contents (on Enter or when focus moves away)."""
        idx = self.bottom_bar._prompt_entries.get(entry)
        if idx is None or not (0 <= idx < len(self.prompts)):
            return
        if entry.get_text() == self.prompts[idx].text:
            return
        self.prompts[idx].text = entry.get_text()
        self.bottom_bar.refresh_rows(self)
        self._push_prompts()
        self._request_checkpoint("prompt")

    # -- run lifecycle --
    def _seed_for_run(self) -> int:
        """Seed for the next run: the typed value, or a fresh random one (then shown in the field).

        Filling the field with the seed actually used makes every run reproducible and visible.
        """
        try:
            seed = int(self.sidebar.seed_entry.get_text().strip())
            if seed < 0:
                raise ValueError
        except ValueError:
            seed = random.randrange(2**31)  # empty / invalid -> random, then surface it
        self._seed_text = str(seed)
        self.sidebar.seed_entry.set_text(self._seed_text)
        return seed

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
            init_image=self._init.image,
            skip_steps=self._init.skip_steps(self.steps),
            seed=self._seed_for_run(),
        )
        self.worker.set_prompts(self._prompt_snapshot())
        # A fresh worker starts with paint_applied_count == 0; reset the overlay tracking to match
        # and drop stale overlays from the previous run. Any stroke painted before Play (still on
        # the active layer) is flushed as the new run's first paint batch rather than discarded.
        self.canvas.paint.reset_overlays()
        if not self.canvas.paint.layer.empty():
            self.canvas.paint.flush(self.worker, self.brush)
        # Freeze the per-run settings this run uses, for the "Current" tab to show while it runs.
        self._run_snapshot = self._perrun_values()
        self._timeline.entries = []
        self._timeline.hist_len = 0
        self._timeline.preview_index = None
        self._timeline.undo_cursor = None
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
        self._timeline.entries = []
        self._timeline.hist_len = 0
        self._timeline.preview_index = None
        self._sync_enabled()

    def _toggle_play(self) -> None:
        if self._timeline.preview_index is not None:
            self._status("Previewing")
            return
        if self.worker is None or not self.worker.is_alive() or self.worker.finished:
            self._start_run()  # finished -> Play starts a fresh run (Revert continues a branch)
        elif self.paused:
            self.paused = False
            self._timeline.preview_index = None  # resume from live, drop any history preview
            self._timeline.undo_cursor = None  # resuming ends the undo chain
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
        self.sidebar.width_entry.set_text(str(self.width))
        self.sidebar.height_entry.set_text(str(self.height))
        # Generation size changed: rebuild the paint layer + drop the stale frame (Canvas.resize),
        # rebuild the init preview (also generation-res), and refit the view.
        self.canvas.resize(self.width, self.height)
        self._init.rebuild_surface(self.width, self.height)
        self._fit_view()
        self._status("Size set")

    def _set_sidebar_width(self, width: int) -> None:
        """Set the sidebar width (clamped); the rebuild is coalesced to one per frame."""
        if self.layout.set_sidebar_width(width):
            self._sidebar_dirty = True

    def _resize_window(self, w: int, h: int) -> None:
        # Adopt the window's actual new size — do NOT call set_mode (that fights tiling WMs;
        # see __post_init__). The SDL surface tracks the resize on its own; we just relay
        # out the UI. The native minimum size (set at startup) keeps it from going too small.
        # Layout.resize clamps the sidebar + bottom panel to fit the new window.
        if not self.layout.resize(w, h):
            return
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
            value = clamp_steps(self.sidebar.steps_entry.get_text())
        except ValueError:
            self.sidebar.steps_entry.set_text(str(self.steps))
            return
        self.sidebar.steps_entry.set_text(str(value))
        if value == self.steps:
            return
        self.steps = value
        # If a run is paused, changing steps abandons it (respacing is fixed per run).
        if self.worker is not None and self.worker.is_alive():
            self._stop_run()
            self._status("Steps set")

    def _save_image(self) -> None:
        """Save the current frame via the native Save dialog (blocks while it's open)."""
        if self.canvas.frame_surface is None:
            self._status("No frame")
            return
        surface = self.canvas.frame_surface.copy()  # freeze; the worker keeps generating
        path_str = self._native_path("save", "Save image")
        if not path_str:
            return  # cancelled
        path = Path(path_str)
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tga"):
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(surface, str(path))
        self._status("Saved")
        log.info("saved %s", path)

    # -- init image (img2img) --
    def _set_init_image(self, image: Image.Image, label: str) -> None:
        self._init.set(image, label, self.width, self.height)
        status = getattr(self, "_init_status_label", None)
        if status is not None:
            status.set_text(f"Init: {label}")
        self._status(f"Init set ({label})")

    def _load_init_file(self, path_str: str) -> None:
        if zipfile.is_zipfile(path_str):  # a session bundle picked via the init-image button
            self._show_message(
                "Not an image",
                f"<b>{Path(path_str).name}</b> looks like a session bundle, not an image."
                "<br><br>Use the <b>Load…</b> button at the top of the sidebar to open it.",
            )
            return
        try:
            image = Image.open(path_str).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - surface the failure instead of crashing
            log.exception("failed to load init image %s", path_str)
            self._show_message("Bad image", f"Couldn't load <b>{Path(path_str).name}</b>:<br>{exc}")
            return
        self._set_init_image(image, Path(path_str).name)

    def _use_current_as_init(self) -> None:
        """Seed the next run from whatever is currently on the canvas (live frame or preview)."""
        surface = self._displayed_surface()
        if surface is None:
            self._status("No frame")
            return
        self._set_init_image(surface_to_pil(surface), "current result")

    def _clear_init(self) -> None:
        self._init.clear()
        status = getattr(self, "_init_status_label", None)
        if status is not None:
            status.set_text("Init: none")
        self._status("Init cleared")

    def _open_init(self) -> None:
        """Pick an init image via the native Open dialog (blocks while it's open)."""
        path_str = self._native_path("open", "Open init image")
        if path_str:
            self._load_init_file(path_str)

    def _open_reset_confirm(self) -> None:
        """Confirm before discarding the rendered frame (so the init / empty canvas shows again)."""
        if self.canvas.frame_surface is None and self.worker is None:
            self._status("Nothing to clear")
            return
        if self._confirm_dialog is not None and self._confirm_dialog.alive():
            return
        rect = self._centered_rect(360, 200)
        desc = "Discard the current image and stop the run? The init image (if set) is shown again."
        self._confirm_dialog = UIConfirmationDialog(
            rect, desc, self.manager, window_title="Reset canvas", action_short_name="Reset"
        )

    def _reset_canvas(self) -> None:
        """Stop the run and drop the rendered frame, revealing the init preview / empty canvas."""
        self._stop_run()  # also clears history / preview index
        self.canvas.frame_surface = None
        self.canvas.frame_key = None
        self.bottom_bar.step_label.set_text("step 0 / 0")
        self._status("Canvas cleared")

    # -- history / revert --
    def _request_checkpoint(self, label: str) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.checkpoint(label)

    def _set_history_label(self) -> None:
        e = self._timeline.preview_entry()
        if e is not None:
            text = f"{e.label} {e.index}/{e.total}"
        else:
            frame = self.worker.latest_frame() if self.worker is not None else None
            text = f"live {frame.index}/{frame.total}" if frame is not None else "live"
        if self.bottom_bar.history_label.text != text:  # avoid re-rendering when unchanged
            self.bottom_bar.history_label.set_text(text)

    def _sync_preview_prompts(self) -> None:
        """Show the previewed checkpoint's prompts in the rows (live set when not previewing)."""
        entry = self._timeline.preview_entry()
        want: list[PromptSpec] | None = entry.prompts if entry is not None else None
        cur = (
            [PromptSpec(p.text, p.weight, p.muted) for p in self._preview_prompts]
            if self._preview_prompts is not None
            else None
        )
        if (list(want) if want is not None else None) == cur:
            return  # already showing the right prompts
        self._preview_prompts = (
            [PromptRow(t, w, m) for t, w, m in want] if want is not None else None
        )
        self.bottom_bar.rebuild_prompt_rows(self)
        self._sync_enabled()

    def _refresh_preview_state(self) -> None:
        """Update the prompt rows, label, and Play gating whenever the preview changes."""
        self._sync_preview_prompts()
        self._set_history_label()
        previewing = self._timeline.preview_index is not None
        (
            self.bottom_bar.play_button.disable
            if previewing
            else self.bottom_bar.play_button.enable
        )()

    def _do_revert(self) -> None:
        """Branch the run from the previewed checkpoint, restoring its prompts + guidance/eta."""
        if self._timeline.preview_index is None:
            return
        entry = self._timeline.entries[self._timeline.preview_index]
        if self.worker is None:  # loaded-session history (no latent) -> img2img from its preview
            self._revert_loaded(entry)
            return
        # Adopt the checkpoint's prompts as the live set, then branch from it.
        self.prompts = [PromptRow(t, w, m) for t, w, m in entry.prompts]
        # Restore the guidance + eta captured at the checkpoint (undo the changes made since),
        # syncing the sliders; cancel any pending guidance checkpoint.
        entry.config.apply_to(self.session.config)
        self._guidance_checkpoint_at = None
        self.sidebar.refresh_advanced_widgets(self)
        self.worker.seek(self._timeline.preview_index)
        self._timeline.preview_index = None
        self._preview_prompts = None
        # Branching drops any in-flight strokes (the worker clears its queue on seek), so reset
        # the overlay tracking to stay aligned with the worker's apply count.
        self.canvas.paint.reset_overlays(self.worker.paint_applied_count)
        # Park the thumb on the checkpoint's actual step now — the worker processes the seek
        # asynchronously, so _live_index() would still read the stale (forward) live frame here.
        self.bottom_bar.history_slider.set_current_value(float(entry.index))
        self.bottom_bar.rebuild_prompt_rows(self)  # now-live (reverted) prompts
        self._push_prompts()  # apply them to the resumed run
        self._sync_enabled()  # re-enable prompt editing
        self._refresh_preview_state()

    def _revert_loaded(self, entry: HistoryEntry) -> None:
        """Revert into a loaded checkpoint (no latent): continue from its preview via img2img."""
        self.prompts = [PromptRow(t, w, m) for t, w, m in entry.prompts]
        entry.config.apply_to(self.session.config)
        self.sidebar.refresh_advanced_widgets(self)
        self._set_init_image(Image.fromarray(entry.preview), f"history: {entry.label}")
        self._timeline.preview_index = None
        self._preview_prompts = None
        self.bottom_bar.rebuild_prompt_rows(self)
        self._start_run()  # seeds the new run from the checkpoint preview (img2img)

    # -- keyboard shortcuts --
    def _keyboard_revert(self) -> None:
        """Ctrl+Z: step back through the checkpoints, reverting to each in turn (undo).

        A scrubbed preview commits that checkpoint; otherwise the first press targets the latest
        checkpoint and each further press one earlier (tracked by _undo_cursor, since a revert to
        the last checkpoint doesn't truncate history so it can't be read back off the list).
        """
        if self.worker is None or self.running or not self._timeline.entries:
            return
        if self._timeline.preview_index is not None:
            target = self._timeline.preview_index
        elif self._timeline.undo_cursor is not None:
            target = max(0, self._timeline.undo_cursor - 1)
        else:
            target = len(self._timeline.entries) - 1
        self._timeline.undo_cursor = target
        self._timeline.preview_index = target
        self._do_revert()

    def _nudge_brush_size(self, factor: float) -> None:
        """Scale the brush size (clamped) and sync the slider — shared by [ / ] and the wheel."""
        self.brush.nudge_size(factor)
        self.bottom_bar.size_slider.set_current_value(self.brush.size)

    def _nudge_brush_strength(self, delta: float) -> None:
        """Shift the brush opacity (clamped) and sync the slider — shared by the wheel."""
        self.brush.nudge_strength(delta)
        self.bottom_bar.strength_slider.set_current_value(self.brush.strength)

    def _select_palette_index(self, index: int) -> None:
        """Digit keys: pick the nth swatch (palette + recents) as the brush colour, if it exists."""
        colours = self._palette.swatches()
        if 0 <= index < len(colours):
            self.brush.color = colours[index]

    def _history_total(self) -> int:
        """The run's display-step total (the history slider's right edge)."""
        frame = self.worker.latest_frame() if self.worker is not None else None
        return self._timeline.total(frame.total if frame is not None else 1)

    def _live_index(self) -> int:
        """The current live display step (the slider's rightmost snap point)."""
        frame = self.worker.latest_frame() if self.worker is not None else None
        if frame is not None:
            return frame.index
        entries = self._timeline.entries
        if self.canvas.frame_surface is not None and entries:
            # A loaded session's result is a static final frame (no worker): its endpoint is the
            # run's last step, past every recorded checkpoint, so scrubbing can reach it.
            return entries[-1].total
        return entries[-1].index if entries else 0

    def _rebuild_history_slider(self) -> None:
        """Recreate the step-space history slider (range follows the run's total steps)."""
        self.bottom_bar.history_slider.kill()
        self.bottom_bar.history_slider = pygame_gui.elements.UIHorizontalSlider(
            self.bottom_bar._history_slider_rect,
            start_value=self._timeline.slider_start(self._live_index()),
            value_range=(0.0, float(max(self._history_total(), 1))),
            manager=self.manager,
        )

    def _sync_history(self) -> None:
        """Pull the worker's history each frame; rebuild the slider when it changes length.

        With no worker we keep the timeline as-is — empty after a stop, or restored from a loaded
        session (which has no worker until you Play/Revert).
        """
        tl = self._timeline
        if self.worker is not None:
            tl.entries = self.worker.get_history()
        if len(tl.entries) != tl.hist_len:
            grew = len(tl.entries) > tl.hist_len  # a new checkpoint = a fresh edit
            tl.hist_len = len(tl.entries)
            if grew:
                tl.undo_cursor = None  # start a new undo chain (our reverts only shrink)
            if tl.preview_index is not None and tl.preview_index >= tl.hist_len:
                tl.preview_index = None
            self._rebuild_history_slider()
            self._sync_enabled()
        elif self.running and tl.preview_index is None and tl.entries:
            # While actively generating (not previewing), let the thumb track the live step as it
            # advances. When paused/done we leave it alone so a scrub isn't yanked back each frame.
            self.bottom_bar.history_slider.set_current_value(float(self._live_index()))
        self._set_history_label()  # keep the live step current as it ticks

    def _displayed_surface(self) -> pygame.Surface | None:
        """The canvas surface to show: a previewed checkpoint, else the live frame."""
        return self._timeline.preview_surface() or self.canvas.frame_surface

    # -- events --
    def _handle_event(self, event: pygame.event.Event) -> bool:
        return _ui_events._handle_event(self, event)

    def _update_frame_surface(self) -> None:
        self.canvas.update_frame_surface()

    def _auto_apply_on_blur(self) -> None:
        """Apply a text box when keyboard focus leaves it (no Enter needed).

        Covers both the prompt boxes and the steps box, so a value typed and then clicked
        away from is adopted just as if Enter had been pressed.
        """
        focus = self.manager.get_focus_set() or set()
        current = next((e for e in self.bottom_bar._prompt_entries if e in focus), None)
        if current is not self._focused_entry:
            previous = self._focused_entry
            self._focused_entry = current
            if previous is not None and previous.alive():
                self._commit_prompt_entry(previous)
        steps_focused = self.sidebar.steps_entry in focus
        if self._steps_focused and not steps_focused:
            self._commit_steps()
        self._steps_focused = steps_focused
        sched = next((e for e in self.sidebar._schedule_entries if e in focus), None)
        if sched is not self._focused_schedule:
            previous = self._focused_schedule
            self._focused_schedule = sched
            if previous is not None and previous.alive():
                self._commit_schedule_entry(previous)

    def _draw(self) -> None:
        return _ui_draw._draw(self)

    def _draw_tools(self) -> None:
        return _ui_draw._draw_tools(self)

    def _draw_history_ticks(self) -> None:
        return _ui_draw._draw_history_ticks(self)

    def _open_colour_picker(self) -> None:
        """Open the arbitrary-RGB picker, seeded with the current brush colour."""
        if self._colour_picker is not None and self._colour_picker.alive():
            return
        rect = self._centered_rect(420, 400)
        self._colour_picker = UIColourPickerDialog(
            rect,
            self.manager,
            initial_colour=pygame.Color(*self.brush.color),
            window_title="Pick a colour",
        )

    def _apply_picked_colour(self, rgb: tuple[int, int, int]) -> None:
        """Adopt a picked colour as the brush colour and remember it (capped, persisted)."""
        self.brush.color = rgb
        self._palette.remember(rgb)  # records off-palette colours as recents (deduped, persisted)
        # relayout the swatches to include any new recent
        self.bottom_bar.build_palette(self, self.bottom_bar._palette_rect)

    # -- painting --
    def _on_swatch(self, pos: tuple[int, int]) -> bool:
        for sr, color in self.bottom_bar._swatch_rects:
            if sr.collidepoint(pos):
                self.brush.color = color
                return True
        return False

    def _paint_at(self, pos: tuple[int, int]) -> None:
        """Paint into the layer at screen ``pos`` (no-op if outside the canvas)."""
        self.canvas.paint_at(pos)

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
            # Rebuild once per frame after a sidebar- or panel-divider drag (both coalesced).
            if self._sidebar_dirty or self._panel_dirty:
                self._sidebar_dirty = False
                self._panel_dirty = False
                self._build_ui()
                self._clamp_pan()
            # Fire the debounced model auto-reload once its delay has elapsed.
            if self._reloader.due(pygame.time.get_ticks()):
                self._reloader.cancel()  # clear the debounce; _start_reload re-validates the change
                self._start_reload()
            # Drop a "guidance" checkpoint once a guidance slider has settled (no-op if no run).
            if (
                self._guidance_checkpoint_at is not None
                and pygame.time.get_ticks() >= self._guidance_checkpoint_at
            ):
                self._guidance_checkpoint_at = None
                self._request_checkpoint("guidance")
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
            self.bottom_bar.refresh_rows(self)
            self._refresh_current()  # keep the "Current" sidebar tab in sync
            self.canvas.paint.sync(self.worker)
            self._sync_history()
            self._update_frame_surface()
            self.manager.update(dt)
            self._draw()
            self.manager.draw_ui(self.screen)
            self._draw_tools()
            self._draw_history_ticks()
            pygame.display.flip()
        self._stop_run()


def surface_to_pil(surface: pygame.Surface) -> Image.Image:
    """Copy a pygame surface to a PIL image (transposing pygame's column-major (W, H) layout)."""
    arr = pygame.surfarray.array3d(surface).swapaxes(0, 1).astype("uint8")  # (H, W, 3)
    return Image.fromarray(arr)


@dataclass
class _LoadingState:
    """Shared between the loading thread and the loading screen."""

    status: str = "starting"  # the component currently loading
    session: DiscoSession | None = None
    error: Exception | None = None
    done: bool = False


def _loading_screen(screen: pygame.Surface, state: _LoadingState) -> bool:
    """Render a loading screen until the model load finishes; ``False`` if the user closes it."""
    title_font = pygame.font.SysFont(None, 40)
    status_font = pygame.font.SysFont(None, 24)
    hint_font = pygame.font.SysFont(None, 18)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        w, h = screen.get_size()
        cx, cy = w // 2, h // 2
        screen.fill(WINDOW_BG)
        title = title_font.render(APP_TITLE, True, (231, 233, 240))
        screen.blit(title, title.get_rect(center=(cx, cy - 38)))
        if state.error is not None:
            msg = status_font.render(f"Failed to load: {state.error}", True, (239, 107, 129))
            screen.blit(msg, msg.get_rect(center=(cx, cy + 8)))
            hint = hint_font.render("close the window to exit", True, (122, 130, 144))
            screen.blit(hint, hint.get_rect(center=(cx, cy + 42)))
        else:
            dots = "." * (1 + (pygame.time.get_ticks() // 400) % 3)
            msg = status_font.render(f"loading {state.status}{dots}", True, (150, 196, 255))
            screen.blit(msg, msg.get_rect(center=(cx, cy + 8)))
            # An indeterminate progress sweep so it's clearly alive during the long load.
            bar = pygame.Rect(0, 0, min(420, w - 80), 4)
            bar.center = (cx, cy + 46)
            pygame.draw.rect(screen, (38, 44, 56), bar, border_radius=2)
            frac = (pygame.time.get_ticks() % 1400) / 1400.0
            seg_w = bar.width // 4
            sx = max(bar.left, bar.left + int((bar.width + seg_w) * frac) - seg_w)
            seg_w = min(seg_w, bar.right - sx)
            if seg_w > 0:
                rect = (sx, bar.top, seg_w, bar.height)
                pygame.draw.rect(screen, (109, 124, 255), rect, border_radius=2)
        pygame.display.flip()
        clock.tick(30)
        if state.done and state.error is None:
            return True


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

    # Choose the device up front on the main thread: it may prompt on stdin for CPU fallback,
    # which must not happen on the background loading thread (the prompt would be invisible behind
    # the window). The loading thread is then handed the chosen device and never re-prompts.
    from disco_diffusion.session import select_device

    device = select_device(config.cpu)

    # Open the window at its final size now, with a loading screen, so it never resizes after the
    # models finish loading — App adopts this same window (see __post_init__).
    win_w, win_h = compute_window_size(
        snap_side(args.width), snap_side(args.height), SIDEBAR_W_DEFAULT, PANEL_H
    )
    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption(APP_TITLE)
    try:
        pygame.Window.from_display_module().minimum_size = (MIN_WINDOW_W, MIN_IMAGE_H + PANEL_MIN)
    except (AttributeError, pygame.error):  # older pygame / headless: best-effort only
        pass

    # Load the models on a background thread (the device/CPU work) while the loading screen
    # pumps events and shows what's loading. (DiscoSession off the main thread is already how
    # the in-app model reload works.)
    state = _LoadingState()

    def load() -> None:
        try:
            state.session = DiscoSession(
                config, device=device, progress=lambda label: setattr(state, "status", label)
            )
        except Exception as exc:  # noqa: BLE001 - surfaced on the loading screen, re-raised below
            log.exception("model load failed")
            state.error = exc
        finally:
            state.done = True

    log.info("loading models (this can take a minute)…")
    threading.Thread(target=load, daemon=True).start()
    if not _loading_screen(screen, state):  # window closed before the load succeeded
        pygame.quit()
        if state.error is not None:  # it failed (the screen showed the error); surface it
            raise state.error
        return  # the user aborted while still loading
    log.info("models loaded")

    session = state.session
    assert session is not None
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
