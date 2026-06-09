"""Sidebar / panel UI construction for the studio App.

Free functions taking the ``App`` (rather than methods) so the ~700-line build cluster lives
outside ``app.py``; ``App``'s widget attributes are declared there so these stay fully typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
import pygame_gui
from pygame_gui.core import ObjectID

from .constants import BRUSH_SIZE_MAX, BRUSH_SIZE_MIN, BRUSH_STRENGTH_MAX, BRUSH_STRENGTH_MIN
from .controls import PromptRow
from .layout import (
    CTRL_H,
    LABEL_H,
    MARGIN,
    PAD,
    ROW_PITCH,
    Row,
    Stack,
)
from .paint import BRUSHES
from .theme import MUTED_COLOR, PENDING_COLOR, READOUT_COLOR

if TYPE_CHECKING:
    from .app import App


def _build_palette(app: App, rect: pygame.Rect) -> None:
    """Lay out the current-colour preview + swatch rects within ``rect`` (drawn custom)."""
    app.bottom_bar._palette_rect = rect
    app.bottom_bar._swatch_rects = []
    app.bottom_bar._color_preview_rect = pygame.Rect(rect.x, rect.y, CTRL_H, CTRL_H)
    x = rect.x + CTRL_H + 10
    colours = app._palette.swatches()
    n = max(len(colours), 1)
    gap = 4
    sw = max(8, min(CTRL_H, (rect.right - x - (n - 1) * gap) // n))
    y = rect.y + (CTRL_H - sw) // 2
    for i, color in enumerate(colours):
        app.bottom_bar._swatch_rects.append((pygame.Rect(x + i * (sw + gap), y, sw, sw), color))


def _build_ui(app: App) -> None:
    # Preserve the seed field's current contents across the rebuild (the widget is recreated).
    if hasattr(app, "seed_entry"):
        app._seed_text = app.sidebar.seed_entry.get_text()
    app.manager.clear_and_reset()
    app.bottom_bar._remove_buttons.clear()
    app.bottom_bar._mute_buttons.clear()
    app.bottom_bar._prompt_entries.clear()
    app.bottom_bar._weight_sliders.clear()
    app.bottom_bar._row_elements.clear()
    app._build_bottom_panel()
    app.sidebar.build(app)
    app._sync_enabled()


def _build_bottom_panel(app: App) -> None:
    """The left column's control panel: transport, history, tools, colours, prompts."""
    ui = pygame_gui.elements
    panel_w = app._panel_w()
    stack = Stack(MARGIN, app._image_area_h() + PAD, panel_w - 2 * MARGIN)

    # Row 1: transport — Play / Stop / Reset | step (left) … status (right) | Save.
    r = stack.row(CTRL_H)
    app.bottom_bar.play_button = ui.UIButton(
        r.left(100), "Play", app.manager, object_id="#play_button"
    )
    app.bottom_bar.stop_button = ui.UIButton(
        r.left(80), "Stop", app.manager, object_id="#stop_button"
    )
    # Reset discards the rendered frame (after a confirm) so the init / empty canvas shows.
    app.bottom_bar.reset_button = ui.UIButton(r.left(70), "Reset", app.manager)
    app.bottom_bar.save_button = ui.UIButton(
        r.right(80), "Save", app.manager, object_id="#save_button"
    )
    app.bottom_bar.status_label = ui.UILabel(
        r.right(150), "", app.manager, object_id="#status_label"
    )
    app.bottom_bar.step_label = ui.UILabel(
        r.fill(), "step 0 / 0", app.manager, object_id="#step_label"
    )

    # Row 2: history scrubber — directly under transport. Drag to preview a checkpoint.
    r = stack.row(CTRL_H)
    ui.UILabel(r.left(54), "History", app.manager)
    app.bottom_bar.cancel_button = ui.UIButton(r.right(70), "Cancel", app.manager)
    app.bottom_bar.revert_button = ui.UIButton(r.right(70), "Revert", app.manager)
    app.bottom_bar.history_label = ui.UILabel(r.right(120), "live", app.manager)
    app.bottom_bar._history_slider_rect = r.fill()
    # The slider spans the 0..N step timeline (not the checkpoint count), so a checkpoint's
    # thumb position matches its actual progress; drags snap to the nearest checkpoint.
    app.bottom_bar.history_slider = ui.UIHorizontalSlider(
        app.bottom_bar._history_slider_rect,
        start_value=app._timeline.slider_start(app._live_index()),
        value_range=(0.0, float(max(app._history_total(), 1))),
        manager=app.manager,
    )

    # Row 3: painting tools — brush kind, noise toggle, size, opacity, clear
    r = stack.row(CTRL_H)
    app.bottom_bar._brush_buttons = {}
    for name in BRUSHES:
        button = ui.UIButton(r.left(64), name, app.manager, object_id="#brush_button")
        app.bottom_bar._brush_buttons[button] = name
        if name == app.brush.type:
            button.select()
    # Toggle: deposit fresh tinted noise (new structure) instead of plain colour.
    app.bottom_bar.noise_button = ui.UIButton(
        r.left(74), "Noise", app.manager, object_id="#brush_button"
    )
    if app.brush.noise:
        app.bottom_bar.noise_button.select()
    # Right group (packed right-to-left, so it reads "Opacity [slider] Clear" left-to-right).
    app.bottom_bar.clear_paint_button = ui.UIButton(r.right(64), "Clear", app.manager)
    app.bottom_bar.strength_slider = ui.UIHorizontalSlider(
        r.right(104),
        app.brush.strength,
        (BRUSH_STRENGTH_MIN, BRUSH_STRENGTH_MAX),
        app.manager,
    )
    ui.UILabel(r.right(56), "Opacity", app.manager)
    # Size label + slider, the slider flexing into whatever's left between the two groups.
    ui.UILabel(r.left(36), "Size", app.manager)
    app.bottom_bar.size_slider = ui.UIHorizontalSlider(
        r.fill(), app.brush.size, (BRUSH_SIZE_MIN, BRUSH_SIZE_MAX), app.manager
    )

    # Row 4: colour palette — current-colour preview + swatches (custom-drawn), and an
    # "RGB…" button that opens the arbitrary-colour picker. The preview/swatches occupy the
    # space left of the button.
    r = stack.row(CTRL_H)
    app.bottom_bar.pick_color_button = ui.UIButton(
        r.right(70), "RGB…", app.manager, object_id="#add_button"
    )
    app._build_palette(r.fill())

    # Row 5: prompts header — Add + hint (hint fills the remaining width)
    r = stack.row(LABEL_H)
    app.bottom_bar.add_button = ui.UIButton(
        r.left(120), "+ Add prompt", app.manager, object_id="#add_button"
    )
    app.bottom_bar.hint_label = ui.UILabel(
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
    app.bottom_bar.prompt_panel = ui.UIScrollingContainer(list_rect, app.manager)
    # Lay rows out narrower than the viewport so the vertical scrollbar never forces a
    # horizontal one (a horizontal bar appears only when content is wider than the view).
    app.bottom_bar._list_inner_w = list_rect.width - 24
    app._rebuild_prompt_rows()


def _displayed_prompts(app: App) -> list[PromptRow]:
    """The prompts shown in the rows: a previewed checkpoint's, else the live set."""
    return app._preview_prompts if app._preview_prompts is not None else app.prompts


def _rebuild_prompt_rows(app: App) -> None:
    for el in app.bottom_bar._row_elements:
        el.kill()
    app.bottom_bar._row_elements.clear()
    app.bottom_bar._remove_buttons.clear()
    app.bottom_bar._mute_buttons.clear()
    app.bottom_bar._prompt_entries.clear()
    app.bottom_bar._weight_sliders.clear()

    container = app.bottom_bar.prompt_panel
    inner_w = app.bottom_bar._list_inner_w
    ui = pygame_gui.elements
    v_pad = (ROW_PITCH - CTRL_H) // 2  # vertically centre widgets in their row pitch
    prompts = app._displayed_prompts()
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
        app.bottom_bar._row_elements += [mute, remove, entry, slider, wlabel]
        app.bottom_bar._remove_buttons[remove] = i
        app.bottom_bar._mute_buttons[mute] = i
        app.bottom_bar._prompt_entries[entry] = i
        app.bottom_bar._weight_sliders[slider] = i
        prompt._wlabel = wlabel  # type: ignore[attr-defined]  # stash for live updates
        prompt._entry = entry  # type: ignore[attr-defined]
        prompt._label_state = None  # type: ignore[attr-defined]  # last (text, colour) shown
    container.set_scrollable_area_dimensions((inner_w, max(len(prompts), 1) * ROW_PITCH + 6))
    app._refresh_rows()


def _refresh_rows(app: App) -> None:
    """Update each row's readout: raw weight + normalised share, or a pending badge.

    Mirrors Sampler.set_conditioning (empty rows ignored; remaining weights normalised
    to sum to 1, so the % is exactly the mix the guidance uses). A row whose text box
    differs from the applied prompt shows an amber "edited · Enter" badge instead — this
    is the live "not yet applied" signal. Labels are only mutated when their state
    changes, so this is cheap to call every frame.
    """
    prompts = app._displayed_prompts()
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


def _sync_enabled(app: App) -> None:
    """Total-steps + size boxes are editable only when not actively generating."""
    editable = not app.running
    for el in (
        app.sidebar.steps_entry,
        app.sidebar.seed_entry,
        app.sidebar.random_seed_button,
        app.sidebar.width_entry,
        app.sidebar.height_entry,
        app.sidebar.apply_button,
        app.sidebar.swap_button,
    ):
        (el.enable if editable else el.disable)()
    # History controls are usable only while paused/stopped and there's history to scrub.
    hist_on = editable and len(app._timeline.entries) > 0
    for hist_el in (
        app.bottom_bar.history_slider,
        app.bottom_bar.revert_button,
        app.bottom_bar.cancel_button,
    ):
        (hist_el.enable if hist_on else hist_el.disable)()
    # Prompt rows are read-only while previewing a checkpoint (they show its prompts).
    prompts_on = app._timeline.preview_index is None
    prompt_widgets = [
        app.bottom_bar.add_button,
        *app.bottom_bar._prompt_entries,
        *app.bottom_bar._weight_sliders,
    ]
    for pw in (*prompt_widgets, *app.bottom_bar._remove_buttons, *app.bottom_bar._mute_buttons):
        (pw.enable if prompts_on else pw.disable)()
    app.bottom_bar.play_button.set_text("Pause" if app.running else "Play")
    # Can't resume mid-preview or mid-reload — Revert/Cancel, or wait for the reload.
    play_off = app._timeline.preview_index is not None or app._reloader.reloading
    (app.bottom_bar.play_button.disable if play_off else app.bottom_bar.play_button.enable)()
