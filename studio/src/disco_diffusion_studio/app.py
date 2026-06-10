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
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import pygame
import pygame_gui
from disco_diffusion import DiscoSession, RunConfig
from PIL import Image
from pygame_gui.windows import UIColourPickerDialog, UIConfirmationDialog, UIMessageWindow

from .common import native_dialog
from .common.constants import (
    APP_TITLE,
)
from .common.layout import (
    DEFAULT_H,
    DEFAULT_W,
    DIVIDER_W,
    MIN_IMAGE_H,
    MIN_LEFT_PANEL_W,
    MIN_WINDOW_W,
    PANEL_H,
    PANEL_MIN,
    SIDEBAR_W_DEFAULT,
    Layout,
    snap_side,
)
from .common.signals import Signals
from .common.theme import (
    DIVIDER,
    IMAGE_BG,
    PANEL_BG,
    THEME,
    WINDOW_BG,
)
from .common.util import surface_to_pil
from .controllers.generation import Generation
from .controllers.history import History
from .controllers.models import Models
from .controllers.recipe import Recipe
from .controllers.session_io import SessionIO
from .engine.worker import PromptSpec
from .paint.paint import Brush
from .paint.palette import Palette
from .session.controls import PromptRow
from .session.state import PaintState, SharedState
from .ui import events
from .ui.bottom_bar import BottomBar
from .ui.canvas import Canvas
from .ui.loading import LoadingState, loading_screen
from .ui.sidebar import Sidebar

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
    # Construction inputs (InitVar: passed to __post_init__ to seed SharedState, not kept on App).
    session: InitVar[DiscoSession]
    out_dir: Path
    width: InitVar[int] = DEFAULT_W
    height: InitVar[int] = DEFAULT_H
    steps: InitVar[int] = 100
    prompts: InitVar[list[PromptRow] | None] = None

    _pending_size: tuple[int, int] | None = None  # coalesced window resize, applied per frame

    # The App is a coordinator over its pieces. The screen areas (ui/) own their own widgets +
    # build + event handling; the rest are the value objects / services. field(init=False) keeps
    # the late-bound ones (built in __post_init__/_build_ui) out of the constructor.
    state: SharedState = field(init=False)  # the working session model the pieces read + mutate
    paint: PaintState = field(init=False)  # painting tools: live brush settings + colour palette
    manager: pygame_gui.UIManager = field(init=False)
    screen: pygame.Surface = field(init=False)
    layout: Layout = field(init=False)  # window geometry (split + divider positions)
    canvas: Canvas = field(init=False)  # image area: view transform + paint layer + frame
    sidebar: Sidebar = field(init=False)  # right sidebar: Settings/Current tabs + their widgets
    bottom_bar: BottomBar = field(init=False)  # transport / history / paint tools / prompts
    session_io: SessionIO = field(init=False)  # save/load the working state as a .zip
    history: History = field(init=False)  # checkpoints / preview scrubbing / undo / revert
    generation: Generation = field(init=False)  # run lifecycle: start/stop/size/steps/seed/save
    models: Models = field(init=False)  # staged CLIP/secondary set + debounced weight reload
    recipe: Recipe = field(init=False)  # presets: capture/apply/save + the dropdown selection

    def __post_init__(
        self,
        session: DiscoSession,
        width: int,
        height: int,
        steps: int,
        prompts: list[PromptRow] | None,
    ) -> None:
        # Painting tools (the live brush + colour palette; brush.color is seeded from the palette).
        palette = Palette.load()
        self.paint = PaintState(brush=Brush(color=palette.default_brush()), palette=palette)
        # The working session model the pieces share (session / worker / geometry / prompts /
        # timeline / init image / encode cache). Output size is snapped to a multiple of 64.
        self.state = SharedState(
            session=session,
            width=snap_side(width),
            height=snap_side(height),
            steps=steps,
            prompts=prompts or [PromptRow("a vast alien landscape, oil painting", 1.0)],
            seed_text=str(random.randrange(2**31)),
            clip_selected=set(session.config.clip_models),
            secondary_on=session.config.use_secondary_model,
        )
        # The UI event bus: controllers announce status messages / enablement-invalidation through
        # it instead of reaching back into the bottom bar + sidebar. Listeners are wired below,
        # once those areas exist (see _resync_enabled / bottom_bar.set_status).
        self.signals = Signals()
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
            win_w, win_h = compute_window_size(
                self.state.width, self.state.height, sidebar_w, panel_h
            )
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
        # in _build_ui. The Sidebar (right column) does likewise; the App coordinates the two. Both
        # take their stable infra + shared state up front; siblings/glue come via app per-method.
        infra = (self.manager, self.layout, self.signals, self.state, self.paint)
        self.bottom_bar = BottomBar(*infra)
        # The prompt entry that last held keyboard focus, so we can auto-apply its text
        # when focus moves away (no need to press Enter).
        self._focused_entry: pygame_gui.elements.UITextEntryLine | None = None
        # Whether the steps box held focus last frame, so we can commit it on blur (mirrors
        # the prompt boxes; without this, typing a value and clicking away wouldn't apply it).
        self._steps_focused = False

        # Right-hand sidebar: owns its widgets (Settings/Current tabs + controls), built in
        # _build_ui. The coalesced "rebuild after a resize" flag is applied once per frame in
        # run(); its width is on self.layout.
        self.sidebar = Sidebar(*infra)
        self._dragging_divider = False
        self._sidebar_dirty = False
        # Horizontal divider between the image area and the bottom panel (drag to resize the
        # panel). Like the sidebar divider, the rebuild is coalesced to one per frame.
        self._dragging_panel = False
        self._panel_dirty = False
        # The schedule box that last held focus, so its text is committed on blur (like steps).
        self._focused_schedule: pygame_gui.elements.UITextEntryLine | None = None

        # Model selection (staged CLIP set + secondary) and the debounced background weight
        # reload — changing the model set rebuilds the session off-thread (see models.py).
        self.models = Models(self, self.signals, self.state)
        self.session_io = SessionIO(
            self, self.signals, self.state
        )  # save/load the working state as a .zip
        # The run lifecycle: start/stop/pause, output size + steps, seed, save (see generation.py).
        # It also holds the "Current" tab's per-run snapshot (read straight from session.config for
        # live knobs).
        self.generation = Generation(self, self.signals, self.state, self.paint)
        # Presets / recipes (TOML on disk): capture/apply the full guidance recipe, the dropdown
        # selection, the "Custom" flip on edits, and the save-as modal (see recipe.py).
        self.recipe = Recipe(self, self.signals, self.state)

        # "Reset canvas" confirmation (discards the rendered frame so the init preview shows).
        self._confirm_dialog: UIConfirmationDialog | None = None
        # Transient error/notice modal (e.g. loading a session via the init button, or vice versa).
        self._message_window: UIMessageWindow | None = None
        # The arbitrary-RGB colour picker dialog (None when closed).
        self._colour_picker: UIColourPickerDialog | None = None

        # The Canvas owns the image area: the view transform (zoom/pan), the paint controller
        # (active stroke + overlays), and the latest rendered frame surface.
        self.canvas = Canvas(self.screen, self.layout, self.state, self.paint, self.bottom_bar)
        self._mouse_pos: tuple[int, int] = (0, 0)
        self._panning = False
        self._navigating = False  # right mouse held: canvas-navigation mode (pan + scroll-zoom)
        self._hud_font = pygame.font.SysFont(None, 19)

        # The History controller drives the edit timeline (state.timeline): checkpoint requests,
        # preview scrubbing, undo, revert, and the per-frame sync (see history.py).
        self.history = History(self, self.signals, self.state)
        # Fit the canvas to the available area once the real window size is settled. The window
        # manager may resize the window right after it opens (firing a VIDEORESIZE that only
        # re-clamps), so the fit happens on the first run-loop frame, not here — by then any
        # startup resize has landed. (_zoom/_pan keep their defaults above until then.)
        self._did_initial_fit = False
        # Wire the event bus now that the bottom bar + sidebar exist: status messages go to the
        # bottom bar's status line, and an invalidation re-syncs both areas' widget enablement.
        self.signals.on_status(self.bottom_bar.set_status)
        self.signals.on_invalidate(self._resync_enabled)
        self.signals.on_edited(self.recipe.mark_custom)  # editing a knob flips the preset to Custom
        self._build_ui()

    # -- geometry --
    # Window geometry lives on self.layout, the canvas transform/paint/frame on self.canvas; call
    # those directly (e.g. self.layout.image_region(), self.canvas.fit()). Only _set_panel_height
    # keeps a wrapper, for the coalesced-rebuild dirty flag (its sibling is _set_sidebar_width).
    def _set_panel_height(self, height: int) -> None:
        """Set the bottom-panel height (clamped); the rebuild is coalesced to one per frame."""
        if self.layout.set_panel_height(height):
            self._panel_dirty = True

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
        return self.recipe.save_modal_alive() or any(
            w is not None and w.alive()
            for w in (
                self._colour_picker,
                self._confirm_dialog,
                self._message_window,
            )
        )

    def _show_message(self, title: str, message: str) -> None:
        """Pop a small OK-dismissable modal to surface a user error (e.g. wrong load button)."""
        if self._message_window is not None and self._message_window.alive():
            self._message_window.kill()
        rect = self.layout.centered_rect(420, 200)
        self._message_window = UIMessageWindow(
            rect, html_message=message, manager=self.manager, window_title=title
        )

    @property
    def running(self) -> bool:
        return (
            self.state.worker is not None
            and self.state.worker.is_alive()
            and not self.state.paused
            and not self.state.worker.finished
        )

    # -- UI construction --
    def _build_ui(self) -> None:
        """Rebuild every widget (the bottom panel + sidebar own their own build)."""
        # Preserve the seed field's contents across the rebuild (the widget is recreated).
        if hasattr(self.sidebar, "seed_entry"):
            self.state.seed_text = self.sidebar.seed_text()
        self.manager.clear_and_reset()
        self.bottom_bar.forget_prompt_widgets()  # drop refs the manager just destroyed
        self.bottom_bar.build(self)
        self.sidebar.build(self)
        self.signals.invalidate()

    # -- sessions (full working state) --
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
            self.signals.status("No native dialog (install zenity)")
            return None

    def _resync_enabled(self) -> None:
        """Re-sync widget enablement to the run / preview / reload state (the invalidate handler).

        Wired to ``signals.invalidate()`` — both areas re-evaluate which widgets are enabled.
        """
        self.sidebar.sync_enabled(self)
        self.bottom_bar.sync_enabled(self)

    # -- prompt snapshot --
    def _prompt_snapshot(self) -> list[PromptSpec]:
        return [PromptSpec(r.text, r.weight, r.muted) for r in self.state.prompts]

    def _push_prompts(self) -> None:
        if self.state.worker is not None and self.state.worker.is_alive():
            self.state.worker.set_prompts(self._prompt_snapshot())

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
            self.canvas.screen = surface  # the canvas holds the window surface; keep it current
        self.manager.set_window_resolution((w, h))
        self._build_ui()
        self.canvas.clamp_pan()  # keep the canvas in view after the viewport changed

    # -- init image (img2img) --
    def _set_init_image(self, image: Image.Image, label: str) -> None:
        self.state.init.set(image, label, self.state.width, self.state.height)
        self.sidebar.set_init_status(f"Init: {label}")
        self.signals.status(f"Init set ({label})")

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
        surface = self.history.displayed_surface()
        if surface is None:
            self.signals.status("No frame")
            return
        self._set_init_image(surface_to_pil(surface), "current result")

    def _clear_init(self) -> None:
        self.state.init.clear()
        self.sidebar.set_init_status("Init: none")
        self.signals.status("Init cleared")

    def _open_init(self) -> None:
        """Pick an init image via the native Open dialog (blocks while it's open)."""
        path_str = self._native_path("open", "Open init image")
        if path_str:
            self._load_init_file(path_str)

    def _open_reset_confirm(self) -> None:
        """Confirm before discarding the rendered frame (so the init / empty canvas shows again)."""
        if self.canvas.frame_surface is None and self.state.worker is None:
            self.signals.status("Nothing to clear")
            return
        if self._confirm_dialog is not None and self._confirm_dialog.alive():
            return
        rect = self.layout.centered_rect(360, 200)
        desc = "Discard the current image and stop the run? The init image (if set) is shown again."
        self._confirm_dialog = UIConfirmationDialog(
            rect, desc, self.manager, window_title="Reset canvas", action_short_name="Reset"
        )

    def _reset_canvas(self) -> None:
        """Stop the run and drop the rendered frame, revealing the init preview / empty canvas."""
        self.generation.stop()  # also clears history / preview index
        self.canvas.clear_frame()
        self.bottom_bar.set_step_label("step 0 / 0")
        self.signals.status("Canvas cleared")

    # -- events --
    def _auto_apply_on_blur(self) -> None:
        """Apply a text box when keyboard focus leaves it (no Enter needed).

        Covers both the prompt boxes and the steps box, so a value typed and then clicked
        away from is adopted just as if Enter had been pressed.
        """
        focus = self.manager.get_focus_set() or set()
        current = self.bottom_bar.focused_entry(focus)
        if current is not self._focused_entry:
            previous = self._focused_entry
            self._focused_entry = current
            if previous is not None and previous.alive():
                self.bottom_bar.commit_prompt_entry(self, previous)
        steps_focused = self.sidebar.steps_entry in focus
        if self._steps_focused and not steps_focused:
            self.generation.commit_steps()
        self._steps_focused = steps_focused
        sched = self.sidebar.focused_schedule(focus)
        if sched is not self._focused_schedule:
            previous = self._focused_schedule
            self._focused_schedule = sched
            if previous is not None and previous.alive():
                self.sidebar.commit_schedule_entry(self, previous)

    def _open_colour_picker(self) -> None:
        """Open the arbitrary-RGB picker, seeded with the current brush colour."""
        if self._colour_picker is not None and self._colour_picker.alive():
            return
        rect = self.layout.centered_rect(420, 400)
        self._colour_picker = UIColourPickerDialog(
            rect,
            self.manager,
            initial_colour=pygame.Color(*self.paint.brush.color),
            window_title="Pick a colour",
        )

    # -- rendering --
    def _draw_frame(self) -> None:
        """The window chrome under everything: panel/sidebar backgrounds + the two dividers.

        The overarching frame; the canvas and bottom bar then draw their own regions on top
        (self.canvas.draw / self.bottom_bar.draw), and pygame_gui draws the widgets in between.
        """
        win_w, win_h = self.layout.window_size()
        img_h = self.layout.image_area_h()
        panel_w = self.layout.panel_w()
        self.screen.fill(WINDOW_BG)
        pygame.draw.rect(self.screen, IMAGE_BG, (0, 0, panel_w, img_h))
        pygame.draw.rect(self.screen, PANEL_BG, (0, img_h, panel_w, win_h - img_h))
        pygame.draw.rect(self.screen, PANEL_BG, self.layout.sidebar_rect())  # full-height sidebar
        # Draggable divider band between the left column and the sidebar.
        pygame.draw.rect(self.screen, DIVIDER, pygame.Rect(panel_w, 0, DIVIDER_W, win_h))
        hot = self._dragging_divider or abs(self._mouse_pos[0] - panel_w) <= DIVIDER_W
        grip_x = panel_w + DIVIDER_W // 2
        pygame.draw.line(
            self.screen,
            (110, 120, 140) if hot else (70, 78, 92),
            (grip_x, win_h // 2 - 14),
            (grip_x, win_h // 2 + 14),
            2,
        )
        # Draggable horizontal divider between the image area and the bottom panel, grip lights
        # up on hover/drag (mirrors the sidebar divider).
        pygame.draw.line(self.screen, DIVIDER, (0, img_h), (panel_w, img_h))
        hot_h = self._dragging_panel or (
            self._mouse_pos[0] < panel_w and abs(self._mouse_pos[1] - img_h) <= DIVIDER_W
        )
        grip_cx = panel_w // 2
        pygame.draw.line(
            self.screen,
            (110, 120, 140) if hot_h else (70, 78, 92),
            (grip_cx - 14, img_h),
            (grip_cx + 14, img_h),
            2,
        )

    # -- main loop --
    def run(self) -> None:
        clock = pygame.time.Clock()
        alive = True
        while alive:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                self.manager.process_events(event)
                if not events.handle(self, event):
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
                self.canvas.fit()
            # Rebuild once per frame after a sidebar- or panel-divider drag (both coalesced).
            if self._sidebar_dirty or self._panel_dirty:
                self._sidebar_dirty = False
                self._panel_dirty = False
                self._build_ui()
                self.canvas.clamp_pan()
            # Fire the debounced model auto-reload once its delay has elapsed.
            self.models.tick_reload(pygame.time.get_ticks())
            # Drop a "guidance" checkpoint once a guidance slider has settled (no-op if no run).
            self.history.tick_guidance_checkpoint(pygame.time.get_ticks())
            # When the run finishes (worker idles) or exits unexpectedly, drop to a paused
            # state so controls/history are usable and the image can still be reverted.
            wk = self.state.worker
            if wk is not None and not self.state.paused and (wk.finished or not wk.is_alive()):
                self.state.paused = True
                if wk.is_alive():
                    wk.pause()
                self.signals.status("Done")
                self.signals.invalidate()
            if wk is not None and wk.notice is not None:  # e.g. compile OOM -> eager fallback
                self.signals.status(wk.notice)
                wk.notice = None
            self.models.poll()  # swap in a reloaded session once its background thread finishes
            self._auto_apply_on_blur()
            # Live "edited · Enter" badge while typing (cheap; only mutates on change).
            self.bottom_bar.refresh_rows(self)
            self.sidebar.refresh_current(self)  # keep the "Current" sidebar tab in sync
            self.canvas.paint.sync(self.state.worker)
            self.history.sync()
            self.canvas.update_frame_surface()
            self.manager.update(dt)
            self._draw_frame()  # window chrome + dividers
            self.canvas.draw(self)  # the image area (frame / init / overlays / cursor / HUD)
            self.manager.draw_ui(self.screen)  # the pygame_gui widgets
            self.bottom_bar.draw(self)  # palette + history-slider ticks, on top of the widgets
            pygame.display.flip()
        self.generation.stop()


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
    state = LoadingState()

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
    if not loading_screen(screen, state):  # window closed before the load succeeded
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
