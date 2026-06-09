"""The bottom panel of the left column: transport, history scrubber, paint tools, prompt list.

:class:`BottomBar` owns those widgets *and* builds them (``build`` + ``build_palette`` /
``rebuild_prompt_rows`` / ``refresh_rows``), keeping the left column's panel out of the god-class.
It takes the App for shared state / actions, and routes its own widget events via ``handle``.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pygame
import pygame_gui
from pygame_gui.core import ObjectID, UIElement
from pygame_gui.elements import (
    UIButton,
    UIHorizontalSlider,
    UILabel,
    UIScrollingContainer,
    UITextEntryLine,
)

from ..constants import BRUSH_SIZE_MAX, BRUSH_SIZE_MIN, BRUSH_STRENGTH_MAX, BRUSH_STRENGTH_MIN
from ..controls import PromptRow
from ..layout import CTRL_H, LABEL_H, MARGIN, PAD, ROW_PITCH, Row, Stack
from ..paint import BRUSHES
from ..theme import MUTED_COLOR, PENDING_COLOR, READOUT_COLOR

if TYPE_CHECKING:
    from ..app import App
    from ..layout import Layout
    from ..signals import Signals
    from ..state import PaintState, SharedState

RGB = tuple[int, int, int]


def _empty_rect() -> pygame.Rect:
    return pygame.Rect(0, 0, 0, 0)


@dataclass
class BottomBar:
    """Owns the bottom panel's widgets (transport, history, paint tools, prompts)."""

    # Injected dependencies: the stable UI infra + shared state this area reads/writes. Siblings
    # (history / generation / …) and App glue are still reached via the `app` passed to each method.
    manager: pygame_gui.UIManager
    layout: Layout
    signals: Signals
    state: SharedState
    paint: PaintState

    # Transport row.
    play_button: UIButton = field(init=False)
    stop_button: UIButton = field(init=False)
    reset_button: UIButton = field(init=False)
    save_button: UIButton = field(init=False)
    status_label: UILabel = field(init=False)
    step_label: UILabel = field(init=False)
    # History scrubber.
    history_label: UILabel = field(init=False)
    history_slider: UIHorizontalSlider = field(init=False)
    cancel_button: UIButton = field(init=False)
    revert_button: UIButton = field(init=False)
    # Seeded non-empty so the slider has a valid rect before the first build sets the real one.
    _history_slider_rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 10, 10))
    # Paint tools.
    noise_button: UIButton = field(init=False)
    clear_paint_button: UIButton = field(init=False)
    pick_color_button: UIButton = field(init=False)
    size_slider: UIHorizontalSlider = field(init=False)
    strength_slider: UIHorizontalSlider = field(init=False)
    _brush_buttons: dict[UIButton, str] = field(default_factory=dict)
    # Colour palette (custom-drawn swatches).
    _swatch_rects: list[tuple[pygame.Rect, RGB]] = field(default_factory=list)
    _color_preview_rect: pygame.Rect = field(default_factory=_empty_rect)
    _palette_rect: pygame.Rect = field(default_factory=_empty_rect)
    # Prompt list (rows rebuilt as prompts change).
    add_button: UIButton = field(init=False)
    hint_label: UILabel = field(init=False)
    prompt_panel: UIScrollingContainer = field(init=False)
    _list_inner_w: int = 0  # row width inside the prompt list (set in build)
    _row_elements: list[UIElement] = field(default_factory=list)
    _remove_buttons: dict[UIButton, int] = field(default_factory=dict)
    _mute_buttons: dict[UIButton, int] = field(default_factory=dict)
    _prompt_entries: dict[UITextEntryLine, int] = field(default_factory=dict)
    _weight_sliders: dict[UIHorizontalSlider, int] = field(default_factory=dict)

    def build(self, app: App) -> None:
        """The left column's control panel: transport, history, tools, colours, prompts."""
        ui = pygame_gui.elements
        panel_w = self.layout.panel_w()
        stack = Stack(MARGIN, self.layout.image_area_h() + PAD, panel_w - 2 * MARGIN)

        # Row 1: transport — Play / Stop / Reset | step (left) … status (right) | Save.
        r = stack.row(CTRL_H)
        self.play_button = ui.UIButton(r.left(100), "Play", self.manager, object_id="#play_button")
        self.stop_button = ui.UIButton(r.left(80), "Stop", self.manager, object_id="#stop_button")
        # Reset discards the rendered frame (after a confirm) so the init / empty canvas shows.
        self.reset_button = ui.UIButton(r.left(70), "Reset", self.manager)
        self.save_button = ui.UIButton(r.right(80), "Save", self.manager, object_id="#save_button")
        self.status_label = ui.UILabel(r.right(150), "", self.manager, object_id="#status_label")
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
            start_value=self.state.timeline.slider_start(app.history.live_index()),
            value_range=(0.0, float(max(app.history.total(), 1))),
            manager=self.manager,
        )

        # Row 3: painting tools — brush kind, noise toggle, size, opacity, clear
        r = stack.row(CTRL_H)
        self._brush_buttons = {}
        for name in BRUSHES:
            button = ui.UIButton(r.left(64), name, self.manager, object_id="#brush_button")
            self._brush_buttons[button] = name
            if name == self.paint.brush.type:
                button.select()
        # Toggle: deposit fresh tinted noise (new structure) instead of plain colour.
        self.noise_button = ui.UIButton(
            r.left(74), "Noise", self.manager, object_id="#brush_button"
        )
        if self.paint.brush.noise:
            self.noise_button.select()
        # Right group (packed right-to-left, so it reads "Opacity [slider] Clear" left-to-right).
        self.clear_paint_button = ui.UIButton(r.right(64), "Clear", self.manager)
        self.strength_slider = ui.UIHorizontalSlider(
            r.right(104),
            self.paint.brush.strength,
            (BRUSH_STRENGTH_MIN, BRUSH_STRENGTH_MAX),
            self.manager,
        )
        ui.UILabel(r.right(56), "Opacity", self.manager)
        # Size label + slider, the slider flexing into whatever's left between the two groups.
        ui.UILabel(r.left(36), "Size", self.manager)
        self.size_slider = ui.UIHorizontalSlider(
            r.fill(), self.paint.brush.size, (BRUSH_SIZE_MIN, BRUSH_SIZE_MAX), self.manager
        )

        # Row 4: colour palette — current-colour preview + swatches (custom-drawn), and an
        # "RGB…" button that opens the arbitrary-colour picker. The preview/swatches occupy the
        # space left of the button.
        r = stack.row(CTRL_H)
        self.pick_color_button = ui.UIButton(
            r.right(70), "RGB…", self.manager, object_id="#add_button"
        )
        self.build_palette(app, r.fill())

        # Row 5: prompts header — Add + hint (hint fills the remaining width)
        r = stack.row(LABEL_H)
        self.add_button = ui.UIButton(
            r.left(120), "+ Add prompt", self.manager, object_id="#add_button"
        )
        self.hint_label = ui.UILabel(
            r.fill(),
            "weight 0-2 applies instantly · Enter/click-away applies text · M mutes · % = mix",
            self.manager,
            object_id="#hint_label",
        )

        # Scrolling prompt list (extra rows scroll), flexing to fill the rest of the panel down
        # to its bottom edge — so dragging the panel taller shows more rows. Pulled up under the
        # header (less the usual row pad) to tighten the gap to the first row.
        list_top = stack.y - 8
        list_h = max(ROW_PITCH, self.layout.win_h - MARGIN - list_top)
        list_rect = pygame.Rect(MARGIN, list_top, panel_w - 2 * MARGIN, list_h)
        self.prompt_panel = ui.UIScrollingContainer(list_rect, self.manager)
        # Lay rows out narrower than the viewport so the vertical scrollbar never forces a
        # horizontal one (a horizontal bar appears only when content is wider than the view).
        self._list_inner_w = list_rect.width - 24
        self.rebuild_prompt_rows(app)

    def build_palette(self, app: App, rect: pygame.Rect) -> None:
        """Lay out the current-colour preview + swatch rects within ``rect`` (drawn custom)."""
        self._palette_rect = rect
        self._swatch_rects = []
        self._color_preview_rect = pygame.Rect(rect.x, rect.y, CTRL_H, CTRL_H)
        x = rect.x + CTRL_H + 10
        colours = self.paint.palette.swatches()
        n = max(len(colours), 1)
        gap = 4
        sw = max(8, min(CTRL_H, (rect.right - x - (n - 1) * gap) // n))
        y = rect.y + (CTRL_H - sw) // 2
        for i, color in enumerate(colours):
            self._swatch_rects.append((pygame.Rect(x + i * (sw + gap), y, sw, sw), color))

    def displayed_prompts(self, app: App) -> list[PromptRow]:
        """The prompts shown in the rows: a previewed checkpoint's, else the live set."""
        return (
            app.history.preview_prompts
            if app.history.preview_prompts is not None
            else self.state.prompts
        )

    def focused_entry(self, focus: Collection[object]) -> UITextEntryLine | None:
        """The prompt text box currently holding keyboard focus, if any (for apply-on-blur)."""
        return next((e for e in self._prompt_entries if e in focus), None)

    def forget_prompt_widgets(self) -> None:
        """Drop the prompt-row widget registries *without* killing — for a full UI rebuild.

        ``manager.clear_and_reset()`` has already destroyed every widget, so the stale refs must
        be dropped without re-killing dead elements (unlike rebuild_prompt_rows, which kills live
        rows before rebuilding).
        """
        self._row_elements.clear()
        self._remove_buttons.clear()
        self._mute_buttons.clear()
        self._prompt_entries.clear()
        self._weight_sliders.clear()

    def rebuild_prompt_rows(self, app: App) -> None:
        for el in self._row_elements:
            el.kill()
        self._row_elements.clear()
        self._remove_buttons.clear()
        self._mute_buttons.clear()
        self._prompt_entries.clear()
        self._weight_sliders.clear()

        container = self.prompt_panel
        inner_w = self._list_inner_w
        ui = pygame_gui.elements
        v_pad = (ROW_PITCH - CTRL_H) // 2  # vertically centre widgets in their row pitch
        prompts = self.displayed_prompts(app)
        for i, prompt in enumerate(prompts):
            # Pack: [mute] [text fills] [slider] [weight readout] [X]. Right-side widgets are
            # taken first so the text entry flexes into whatever width is left.
            r = Row(0, i * ROW_PITCH + v_pad, inner_w, CTRL_H)
            mute = ui.UIButton(
                r.left(30), "M", self.manager, container=container, object_id="#brush_button"
            )
            if prompt.muted:
                mute.select()
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
            self._row_elements += [mute, remove, entry, slider, wlabel]
            self._remove_buttons[remove] = i
            self._mute_buttons[mute] = i
            self._prompt_entries[entry] = i
            self._weight_sliders[slider] = i
            prompt._wlabel = wlabel  # type: ignore[attr-defined]  # stash for live updates
            prompt._entry = entry  # type: ignore[attr-defined]
            prompt._label_state = None  # type: ignore[attr-defined]  # last (text, colour) shown
        container.set_scrollable_area_dimensions((inner_w, max(len(prompts), 1) * ROW_PITCH + 6))
        self.refresh_rows(app)

    def refresh_rows(self, app: App) -> None:
        """Update each row's readout: raw weight + normalised share, or a pending badge.

        Mirrors Sampler.set_conditioning (empty rows ignored; remaining weights normalised
        to sum to 1, so the % is exactly the mix the guidance uses). A row whose text box
        differs from the applied prompt shows an amber "edited · Enter" badge instead — this
        is the live "not yet applied" signal. Labels are only mutated when their state
        changes, so this is cheap to call every frame.
        """
        prompts = self.displayed_prompts(app)
        active = [(i, r.weight) for i, r in enumerate(prompts) if r.text.strip() and not r.muted]
        total = sum(w for _, w in active)
        shares = {i: w / total for i, w in active} if total > 1e-3 else {}
        for i, row in enumerate(prompts):
            wlabel = getattr(row, "_wlabel", None)
            entry = getattr(row, "_entry", None)
            if wlabel is None or entry is None:
                continue
            if entry.get_text() != row.text:
                text, colour = "edited · Enter", PENDING_COLOR
            elif row.muted:
                text, colour = f"{row.weight:.2f}  muted", MUTED_COLOR
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

    def handle(self, app: App, event: pygame.event.Event) -> bool:
        """Handle an event targeting a bottom-bar widget; return True if it was ours."""
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            e = event.ui_element
            if e == self.play_button:
                app.generation.toggle_play()
            elif e == self.stop_button:
                app.generation.stop()
                self.signals.status("Stopped")
            elif e == self.reset_button:
                app._open_reset_confirm()
            elif e == self.save_button:
                app.generation.save_image()
            elif e == self.pick_color_button:
                app._open_colour_picker()
            elif e == self.add_button:
                self.state.prompts.append(PromptRow("", 1.0))
                self.rebuild_prompt_rows(app)
                app._push_prompts()
                app.history.request_checkpoint("add prompt")
            elif e in self._remove_buttons:
                idx = self._remove_buttons[e]
                if 0 <= idx < len(self.state.prompts):
                    self.state.prompts.pop(idx)
                    self.rebuild_prompt_rows(app)
                    app._push_prompts()
                    app.history.request_checkpoint("remove prompt")
            elif e in self._mute_buttons:
                idx = self._mute_buttons[e]
                if 0 <= idx < len(self.state.prompts):
                    prompt = self.state.prompts[idx]
                    prompt.muted = not prompt.muted
                    (e.select if prompt.muted else e.unselect)()
                    self.refresh_rows(app)
                    app._push_prompts()  # re-mix conditioning (muted excluded)
                    app.history.request_checkpoint(
                        "mute prompt" if prompt.muted else "unmute prompt"
                    )
            elif e in self._brush_buttons:
                self.paint.brush.type = self._brush_buttons[e]
                for button, name in self._brush_buttons.items():
                    (button.select if name == self.paint.brush.type else button.unselect)()
            elif e == self.noise_button:
                self.paint.brush.noise = not self.paint.brush.noise
                (
                    self.noise_button.select
                    if self.paint.brush.noise
                    else self.noise_button.unselect
                )()
            elif e == self.clear_paint_button:
                app.canvas.paint.layer.clear()
            elif e == self.revert_button:
                app.history.revert()
            elif e == self.cancel_button:
                self.state.timeline.clear_preview()
                self.park_history_thumb(float(app.history.live_index()))
                app.history.refresh_preview_state()
            else:
                return False
            return True
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            e = event.ui_element
            if e == self.size_slider:
                self.paint.brush.size = float(event.value)
            elif e == self.strength_slider:
                self.paint.brush.strength = float(event.value)
            elif e == self.history_slider:
                # Step-space slider: snap the dragged value to the nearest checkpoint (or live)
                # and park the thumb on that checkpoint's actual step position.
                snapped = self.state.timeline.scrub(float(event.value), app.history.live_index())
                self.park_history_thumb(snapped)
                app.history.refresh_preview_state()
            elif e in self._weight_sliders:
                idx = self._weight_sliders[e]
                if 0 <= idx < len(self.state.prompts):
                    self.state.prompts[idx].weight = float(event.value)
                    self.refresh_rows(app)
                    app._push_prompts()
            else:
                return False
            return True
        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element in self._prompt_entries:
                self.commit_prompt_entry(app, event.ui_element)
                return True
        return False

    def on_swatch(self, app: App, pos: tuple[int, int]) -> bool:
        """If ``pos`` hits a palette swatch, adopt it as the brush colour. True if it did."""
        for sr, color in self._swatch_rects:
            if sr.collidepoint(pos):
                self.paint.brush.color = color
                return True
        return False

    def select_palette_index(self, app: App, index: int) -> None:
        """Digit keys: pick the nth swatch (palette + recents) as the brush colour, if it exists."""
        colours = self.paint.palette.swatches()
        if 0 <= index < len(colours):
            self.paint.brush.color = colours[index]

    def apply_picked_colour(self, app: App, rgb: tuple[int, int, int]) -> None:
        """Adopt a picked colour as the brush colour and remember it (capped, persisted)."""
        self.paint.brush.color = rgb
        self.paint.palette.remember(
            rgb
        )  # records off-palette colours as recents (deduped, persisted)
        self.build_palette(app, self._palette_rect)  # relayout to include any new recent

    def set_step_label(self, text: str) -> None:
        """Set the transport step counter (e.g. "step 12 / 100")."""
        self.step_label.set_text(text)

    def set_status(self, text: str) -> None:
        """Set the transport status line (e.g. "Running" / "Paused" / "Saved")."""
        self.status_label.set_text(text)

    def nudge_brush_size(self, app: App, factor: float) -> None:
        """Scale the brush size (clamped) and sync the slider — shared by [ / ] and the wheel."""
        self.paint.brush.nudge_size(factor)
        self.size_slider.set_current_value(self.paint.brush.size)

    def nudge_brush_strength(self, app: App, delta: float) -> None:
        """Shift the brush opacity (clamped) and sync the slider — shared by the wheel."""
        self.paint.brush.nudge_strength(delta)
        self.strength_slider.set_current_value(self.paint.brush.strength)

    def commit_prompt_entry(self, app: App, entry: UITextEntryLine) -> None:
        """Apply a prompt text box's contents (on Enter or when focus moves away)."""
        idx = self._prompt_entries.get(entry)
        if idx is None or not (0 <= idx < len(self.state.prompts)):
            return
        if entry.get_text() == self.state.prompts[idx].text:
            return
        self.state.prompts[idx].text = entry.get_text()
        self.refresh_rows(app)
        app._push_prompts()
        app.history.request_checkpoint("prompt")

    def set_history_label(self, text: str) -> None:
        """Set the history readout (live / previewed checkpoint); skips a no-op re-render."""
        if self.history_label.text != text:
            self.history_label.set_text(text)

    def gate_play_for_preview(self, previewing: bool) -> None:
        """Disable Play while previewing a checkpoint (Revert/Cancel first), else enable it."""
        (self.play_button.disable if previewing else self.play_button.enable)()

    def park_history_thumb(self, step: float) -> None:
        """Park the scrubber thumb on a step without firing a slider-moved event."""
        self.history_slider.set_current_value(step)

    def rebuild_history_slider(self, app: App) -> None:
        """Recreate the step-space history slider (its range follows the run's total steps)."""
        self.history_slider.kill()
        self.history_slider = pygame_gui.elements.UIHorizontalSlider(
            self._history_slider_rect,
            start_value=self.state.timeline.slider_start(app.history.live_index()),
            value_range=(0.0, float(max(app.history.total(), 1))),
            manager=self.manager,
        )

    def sync_enabled(self, app: App) -> None:
        """History scrubbing, prompt editing, and the Play button reflect run / preview state."""
        editable = not app.running
        # History controls are usable only while paused/stopped and there's history to scrub.
        hist_on = editable and len(self.state.timeline.entries) > 0
        for hist_el in (self.history_slider, self.revert_button, self.cancel_button):
            (hist_el.enable if hist_on else hist_el.disable)()
        # Prompt rows are read-only while previewing a checkpoint (they show its prompts).
        prompts_on = self.state.timeline.preview_index is None
        prompt_widgets = [self.add_button, *self._prompt_entries, *self._weight_sliders]
        for pw in (*prompt_widgets, *self._remove_buttons, *self._mute_buttons):
            (pw.enable if prompts_on else pw.disable)()
        self.play_button.set_text("Pause" if app.running else "Play")
        # Can't resume mid-preview or mid-reload — Revert/Cancel, or wait for the reload.
        play_off = self.state.timeline.preview_index is not None or app.models.reloader.reloading
        (self.play_button.disable if play_off else self.play_button.enable)()
