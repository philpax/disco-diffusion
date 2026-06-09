"""The bottom panel of the left column: transport, history scrubber, paint tools, prompt list.

:class:`BottomBar` owns those widgets *and* builds them (``build`` + ``build_palette`` /
``rebuild_prompt_rows`` / ``refresh_rows``), keeping the left column's panel out of the god-class.
It takes the App for shared state / actions, and routes its own widget events via ``handle``.
"""

from __future__ import annotations

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

RGB = tuple[int, int, int]


def _empty_rect() -> pygame.Rect:
    return pygame.Rect(0, 0, 0, 0)


@dataclass
class BottomBar:
    """Owns the bottom panel's widgets (transport, history, paint tools, prompts)."""

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
    _history_slider_rect: pygame.Rect = field(default_factory=_empty_rect)
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
        panel_w = app.layout.panel_w()
        stack = Stack(MARGIN, app.layout.image_area_h() + PAD, panel_w - 2 * MARGIN)

        # Row 1: transport — Play / Stop / Reset | step (left) … status (right) | Save.
        r = stack.row(CTRL_H)
        self.play_button = ui.UIButton(r.left(100), "Play", app.manager, object_id="#play_button")
        self.stop_button = ui.UIButton(r.left(80), "Stop", app.manager, object_id="#stop_button")
        # Reset discards the rendered frame (after a confirm) so the init / empty canvas shows.
        self.reset_button = ui.UIButton(r.left(70), "Reset", app.manager)
        self.save_button = ui.UIButton(r.right(80), "Save", app.manager, object_id="#save_button")
        self.status_label = ui.UILabel(r.right(150), "", app.manager, object_id="#status_label")
        self.step_label = ui.UILabel(r.fill(), "step 0 / 0", app.manager, object_id="#step_label")

        # Row 2: history scrubber — directly under transport. Drag to preview a checkpoint.
        r = stack.row(CTRL_H)
        ui.UILabel(r.left(54), "History", app.manager)
        self.cancel_button = ui.UIButton(r.right(70), "Cancel", app.manager)
        self.revert_button = ui.UIButton(r.right(70), "Revert", app.manager)
        self.history_label = ui.UILabel(r.right(120), "live", app.manager)
        self._history_slider_rect = r.fill()
        # The slider spans the 0..N step timeline (not the checkpoint count), so a checkpoint's
        # thumb position matches its actual progress; drags snap to the nearest checkpoint.
        self.history_slider = ui.UIHorizontalSlider(
            self._history_slider_rect,
            start_value=app._timeline.slider_start(app._live_index()),
            value_range=(0.0, float(max(app._history_total(), 1))),
            manager=app.manager,
        )

        # Row 3: painting tools — brush kind, noise toggle, size, opacity, clear
        r = stack.row(CTRL_H)
        self._brush_buttons = {}
        for name in BRUSHES:
            button = ui.UIButton(r.left(64), name, app.manager, object_id="#brush_button")
            self._brush_buttons[button] = name
            if name == app.brush.type:
                button.select()
        # Toggle: deposit fresh tinted noise (new structure) instead of plain colour.
        self.noise_button = ui.UIButton(r.left(74), "Noise", app.manager, object_id="#brush_button")
        if app.brush.noise:
            self.noise_button.select()
        # Right group (packed right-to-left, so it reads "Opacity [slider] Clear" left-to-right).
        self.clear_paint_button = ui.UIButton(r.right(64), "Clear", app.manager)
        self.strength_slider = ui.UIHorizontalSlider(
            r.right(104),
            app.brush.strength,
            (BRUSH_STRENGTH_MIN, BRUSH_STRENGTH_MAX),
            app.manager,
        )
        ui.UILabel(r.right(56), "Opacity", app.manager)
        # Size label + slider, the slider flexing into whatever's left between the two groups.
        ui.UILabel(r.left(36), "Size", app.manager)
        self.size_slider = ui.UIHorizontalSlider(
            r.fill(), app.brush.size, (BRUSH_SIZE_MIN, BRUSH_SIZE_MAX), app.manager
        )

        # Row 4: colour palette — current-colour preview + swatches (custom-drawn), and an
        # "RGB…" button that opens the arbitrary-colour picker. The preview/swatches occupy the
        # space left of the button.
        r = stack.row(CTRL_H)
        self.pick_color_button = ui.UIButton(
            r.right(70), "RGB…", app.manager, object_id="#add_button"
        )
        self.build_palette(app, r.fill())

        # Row 5: prompts header — Add + hint (hint fills the remaining width)
        r = stack.row(LABEL_H)
        self.add_button = ui.UIButton(
            r.left(120), "+ Add prompt", app.manager, object_id="#add_button"
        )
        self.hint_label = ui.UILabel(
            r.fill(),
            "weight 0-2 applies instantly · Enter/click-away applies text · M mutes · % = mix",
            app.manager,
            object_id="#hint_label",
        )

        # Scrolling prompt list (extra rows scroll), flexing to fill the rest of the panel down
        # to its bottom edge — so dragging the panel taller shows more rows. Pulled up under the
        # header (less the usual row pad) to tighten the gap to the first row.
        list_top = stack.y - 8
        list_h = max(ROW_PITCH, app.layout.win_h - MARGIN - list_top)
        list_rect = pygame.Rect(MARGIN, list_top, panel_w - 2 * MARGIN, list_h)
        self.prompt_panel = ui.UIScrollingContainer(list_rect, app.manager)
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
        colours = app._palette.swatches()
        n = max(len(colours), 1)
        gap = 4
        sw = max(8, min(CTRL_H, (rect.right - x - (n - 1) * gap) // n))
        y = rect.y + (CTRL_H - sw) // 2
        for i, color in enumerate(colours):
            self._swatch_rects.append((pygame.Rect(x + i * (sw + gap), y, sw, sw), color))

    def displayed_prompts(self, app: App) -> list[PromptRow]:
        """The prompts shown in the rows: a previewed checkpoint's, else the live set."""
        return app._preview_prompts if app._preview_prompts is not None else app.prompts

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
                r.left(30), "M", app.manager, container=container, object_id="#brush_button"
            )
            if prompt.muted:
                mute.select()
            remove = ui.UIButton(
                r.right(30),
                "×",
                app.manager,
                container=container,
                object_id=ObjectID(object_id="#remove_button", class_id="@remove_button"),
            )
            wlabel = ui.UILabel(r.right(104), "", app.manager, container=container)
            slider = ui.UIHorizontalSlider(
                r.right(150),
                start_value=prompt.weight,
                value_range=(0.0, 2.0),
                manager=app.manager,
                container=container,
            )
            entry = ui.UITextEntryLine(r.fill(), app.manager, container=container)
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
                app._toggle_play()
            elif e == self.stop_button:
                app._stop_run()
                app._status("Stopped")
            elif e == self.reset_button:
                app._open_reset_confirm()
            elif e == self.save_button:
                app._save_image()
            elif e == self.pick_color_button:
                app._open_colour_picker()
            elif e == self.add_button:
                app.prompts.append(PromptRow("", 1.0))
                self.rebuild_prompt_rows(app)
                app._push_prompts()
                app._request_checkpoint("add prompt")
            elif e in self._remove_buttons:
                idx = self._remove_buttons[e]
                if 0 <= idx < len(app.prompts):
                    app.prompts.pop(idx)
                    self.rebuild_prompt_rows(app)
                    app._push_prompts()
                    app._request_checkpoint("remove prompt")
            elif e in self._mute_buttons:
                idx = self._mute_buttons[e]
                if 0 <= idx < len(app.prompts):
                    prompt = app.prompts[idx]
                    prompt.muted = not prompt.muted
                    (e.select if prompt.muted else e.unselect)()
                    self.refresh_rows(app)
                    app._push_prompts()  # re-mix conditioning (muted excluded)
                    app._request_checkpoint("mute prompt" if prompt.muted else "unmute prompt")
            elif e in self._brush_buttons:
                app.brush.type = self._brush_buttons[e]
                for button, name in self._brush_buttons.items():
                    (button.select if name == app.brush.type else button.unselect)()
            elif e == self.noise_button:
                app.brush.noise = not app.brush.noise
                (self.noise_button.select if app.brush.noise else self.noise_button.unselect)()
            elif e == self.clear_paint_button:
                app.canvas.paint.layer.clear()
            elif e == self.revert_button:
                app._do_revert()
            elif e == self.cancel_button:
                app._timeline.preview_index = None
                self.history_slider.set_current_value(float(app._live_index()))
                app._refresh_preview_state()
            else:
                return False
            return True
        if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
            e = event.ui_element
            if e == self.size_slider:
                app.brush.size = float(event.value)
            elif e == self.strength_slider:
                app.brush.strength = float(event.value)
            elif e == self.history_slider:
                # Step-space slider: snap the dragged value to the nearest checkpoint (or live)
                # and park the thumb on that checkpoint's actual step position.
                snap_idx = app._timeline.snap(float(event.value), app._live_index())
                app._timeline.preview_index = snap_idx
                if snap_idx is not None:
                    snapped = float(app._timeline.entries[snap_idx].index)
                else:
                    snapped = float(app._live_index())
                self.history_slider.set_current_value(snapped)
                app._refresh_preview_state()
            elif e in self._weight_sliders:
                idx = self._weight_sliders[e]
                if 0 <= idx < len(app.prompts):
                    app.prompts[idx].weight = float(event.value)
                    self.refresh_rows(app)
                    app._push_prompts()
            else:
                return False
            return True
        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            if event.ui_element in self._prompt_entries:
                app._commit_prompt_entry(event.ui_element)
                return True
        return False

    def on_swatch(self, app: App, pos: tuple[int, int]) -> bool:
        """If ``pos`` hits a palette swatch, adopt it as the brush colour. True if it did."""
        for sr, color in self._swatch_rects:
            if sr.collidepoint(pos):
                app.brush.color = color
                return True
        return False

    def select_palette_index(self, app: App, index: int) -> None:
        """Digit keys: pick the nth swatch (palette + recents) as the brush colour, if it exists."""
        colours = app._palette.swatches()
        if 0 <= index < len(colours):
            app.brush.color = colours[index]

    def apply_picked_colour(self, app: App, rgb: tuple[int, int, int]) -> None:
        """Adopt a picked colour as the brush colour and remember it (capped, persisted)."""
        app.brush.color = rgb
        app._palette.remember(rgb)  # records off-palette colours as recents (deduped, persisted)
        self.build_palette(app, self._palette_rect)  # relayout to include any new recent

    def sync_enabled(self, app: App) -> None:
        """History scrubbing, prompt editing, and the Play button reflect run / preview state."""
        editable = not app.running
        # History controls are usable only while paused/stopped and there's history to scrub.
        hist_on = editable and len(app._timeline.entries) > 0
        for hist_el in (self.history_slider, self.revert_button, self.cancel_button):
            (hist_el.enable if hist_on else hist_el.disable)()
        # Prompt rows are read-only while previewing a checkpoint (they show its prompts).
        prompts_on = app._timeline.preview_index is None
        prompt_widgets = [self.add_button, *self._prompt_entries, *self._weight_sliders]
        for pw in (*prompt_widgets, *self._remove_buttons, *self._mute_buttons):
            (pw.enable if prompts_on else pw.disable)()
        self.play_button.set_text("Pause" if app.running else "Play")
        # Can't resume mid-preview or mid-reload — Revert/Cancel, or wait for the reload.
        play_off = app._timeline.preview_index is not None or app._reloader.reloading
        (self.play_button.disable if play_off else self.play_button.enable)()
