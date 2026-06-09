"""Sidebar / panel UI construction for the studio App.

Free functions taking the ``App`` (rather than methods) so the ~700-line build cluster lives
outside ``app.py``; ``App``'s widget attributes are declared there so these stay fully typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
import pygame_gui
from disco_diffusion.config import AVAILABLE_CLIP_MODELS
from pygame_gui.core import ObjectID
from pygame_gui.elements import UIDropDownMenu, UIScrollingContainer

from .constants import BRUSH_SIZE_MAX, BRUSH_SIZE_MIN, BRUSH_STRENGTH_MAX, BRUSH_STRENGTH_MIN
from .controls import CURRENT_PERRUN, CUSTOM_PRESET, LIVE_SCALES, SCHEDULES, PromptRow
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
    app._build_sidebar()
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


def _build_sidebar(app: App) -> None:
    """The full-height right sidebar: a Settings / Current tab pair over a scroll area."""
    ui = pygame_gui.elements
    sb = app._sidebar_rect()
    x = sb.x + PAD
    inner = max(40, sb.width - 2 * PAD)
    half = (inner - PAD) // 2

    # Session save/load sits at the very top — a global action (works on either tab), so it's
    # not buried in Settings; what it saves/loads is clear from context (the whole session).
    r = Row(x, MARGIN, inner, CTRL_H)
    app.sidebar.save_session_button = ui.UIButton(
        r.left(half), "Save…", app.manager, object_id="#add_button"
    )
    app.sidebar.load_session_button = ui.UIButton(
        r.fill(), "Load…", app.manager, object_id="#add_button"
    )

    tabs_y = MARGIN + CTRL_H + PAD
    r = Row(x, tabs_y, inner, CTRL_H)
    app.sidebar.tab_settings = ui.UIButton(
        r.left(half), "Settings", app.manager, object_id="#tab_button"
    )
    app.sidebar.tab_current = ui.UIButton(r.fill(), "Current", app.manager, object_id="#tab_button")

    cont_y = tabs_y + CTRL_H + PAD
    cont_rect = pygame.Rect(x, cont_y, inner, max(60, app.layout.win_h - cont_y - MARGIN))
    app.sidebar.settings_panel = ui.UIScrollingContainer(cont_rect, app.manager)
    app.sidebar.current_panel = ui.UIScrollingContainer(cont_rect, app.manager)
    app.sidebar._sb_inner_w = inner - 24  # leave room for the vertical scrollbar
    app._build_settings_rows()
    app._build_current_rows()
    app._sync_sidebar_tabs()


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


def _section_label(
    app: App, container: UIScrollingContainer, inner_w: int, y: int, text: str
) -> int:
    """Draw a section heading in the settings panel; return the y below it."""
    pygame_gui.elements.UILabel(
        Row(0, y, inner_w, LABEL_H).fill(),
        text,
        app.manager,
        container=container,
        object_id="#section_label",
    )
    return y + LABEL_H + 6


def _build_settings_rows(app: App) -> None:
    """Build the sidebar Settings tab, one section at a time, threading the y cursor.

    Layout order: output (steps / seed / size) · init image · preset (dropdown + Save) ·
    guidance sliders (retune live) · per-run eta/perlin + cut schedules · model set. Sliders
    write straight to ``session.config`` (read live by the worker each step); schedule boxes
    are validated and applied next Play; toggling a model queues a (debounced) auto-reload.
    """
    app.sidebar._scale_sliders = {}
    app.sidebar._schedule_entries = {}
    app.sidebar._clip_buttons = {}
    app.sidebar.preset_dropdown = None  # cleared by clear_and_reset; respawned below
    container = app.sidebar.settings_panel
    inner_w = app.sidebar._sb_inner_w
    pitch = CTRL_H + 8
    y = 2
    y = app._build_output_section(container, inner_w, pitch, y)
    y = app._build_init_section(container, inner_w, pitch, y)
    y = app._build_preset_section(container, inner_w, pitch, y)
    y = app._build_guidance_section(container, inner_w, pitch, y)
    y = app._build_perrun_section(container, inner_w, pitch, y)
    y = app._build_models_section(container, inner_w, pitch, y)
    container.set_scrollable_area_dimensions((inner_w, y + 8))


def _build_output_section(
    app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
) -> int:
    """Steps, seed (+ Rnd), width/height, apply / flip — all per-run (apply on next Play)."""
    ui = pygame_gui.elements
    y = app._section_label(container, inner_w, y, "Output — apply on next Play")
    r = Row(0, y, inner_w, CTRL_H)
    ui.UILabel(r.left(54), "Steps", app.manager, container=container)
    app.sidebar.steps_entry = ui.UITextEntryLine(r.fill(), app.manager, container=container)
    app.sidebar.steps_entry.set_text(str(app.steps))
    y += pitch
    # Seed: always shows the concrete seed in use (so it's reproducible and visible); Play
    # uses whatever's here, and "Rnd" rolls a fresh one.
    r = Row(0, y, inner_w, CTRL_H)
    ui.UILabel(r.left(54), "Seed", app.manager, container=container)
    app.sidebar.random_seed_button = ui.UIButton(
        r.right(46), "Rnd", app.manager, container=container
    )
    app.sidebar.seed_entry = ui.UITextEntryLine(r.fill(), app.manager, container=container)
    app.sidebar.seed_entry.set_text(app._seed_text)
    y += pitch
    r = Row(0, y, inner_w, CTRL_H)
    ui.UILabel(r.left(20), "W", app.manager, container=container)
    app.sidebar.width_entry = ui.UITextEntryLine(
        r.left((inner_w - 56) // 2), app.manager, container=container
    )
    app.sidebar.width_entry.set_text(str(app.width))
    ui.UILabel(r.left(20), "H", app.manager, container=container)
    app.sidebar.height_entry = ui.UITextEntryLine(r.fill(), app.manager, container=container)
    app.sidebar.height_entry.set_text(str(app.height))
    y += pitch
    r = Row(0, y, inner_w, CTRL_H)
    app.sidebar.apply_button = ui.UIButton(
        r.left((inner_w - PAD) // 2), "Apply size", app.manager, container=container
    )
    app.sidebar.swap_button = ui.UIButton(r.fill(), "Flip W/H", app.manager, container=container)
    return y + pitch


def _build_init_section(
    app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
) -> int:
    """Init image (img2img): Open… / Use current / Clear + the denoise slider (per-run)."""
    ui = pygame_gui.elements
    y = app._section_label(container, inner_w, y + 6, "Init image — applies on next Play")
    app.sidebar._init_status_label = ui.UILabel(
        Row(0, y, inner_w, CTRL_H).fill(),
        f"Init: {app._init.label}",
        app.manager,
        container=container,
    )
    y += pitch
    r = Row(0, y, inner_w, CTRL_H)
    third = (inner_w - 2 * PAD) // 3
    app.sidebar.open_init_button = ui.UIButton(
        r.left(third), "Open…", app.manager, container=container
    )
    app.sidebar.use_current_init_button = ui.UIButton(
        r.left(third), "Use current", app.manager, container=container
    )
    app.sidebar.clear_init_button = ui.UIButton(r.fill(), "Clear", app.manager, container=container)
    y += pitch
    r = Row(0, y, inner_w, CTRL_H)
    ui.UILabel(r.left(80), "Denoise", app.manager, container=container)
    app.sidebar._init_denoise_label = ui.UILabel(r.right(48), "", app.manager, container=container)
    app.sidebar._init_denoise_slider = ui.UIHorizontalSlider(
        r.fill(),
        start_value=float(app._init.denoise),
        value_range=(0.0, 100.0),
        manager=app.manager,
        container=container,
    )
    app.sidebar._init_denoise_label.set_text(f"{app._init.denoise}%")
    return y + pitch


def _build_preset_section(
    app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
) -> int:
    """A dropdown of saved recipes + Save; selecting one applies the whole recipe."""
    ui = pygame_gui.elements
    y = app._section_label(container, inner_w, y + 6, "Preset")
    r = Row(0, y, inner_w, CTRL_H)
    app.sidebar.save_preset_button = ui.UIButton(
        r.right(72), "Save…", app.manager, container=container, object_id="#add_button"
    )
    app.sidebar._preset_dd_rect = r.fill()
    app._spawn_preset_dropdown()
    return y + pitch


def _build_guidance_section(
    app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
) -> int:
    """The live-guidance sliders — each retunes the running step immediately."""
    ui = pygame_gui.elements
    cfg = app.session.config
    y = app._section_label(container, inner_w, y + 6, "Guidance — retunes live")
    for sc in LIVE_SCALES:
        r = Row(0, y, inner_w, CTRL_H)
        ui.UILabel(r.left(108), sc.label, app.manager, container=container)
        vlabel = ui.UILabel(r.right(66), "", app.manager, container=container)
        cur = float(getattr(cfg, sc.attr))
        slider = ui.UIHorizontalSlider(
            r.fill(),
            start_value=min(max(cur, sc.lo), sc.hi),
            value_range=(sc.lo, sc.hi),
            manager=app.manager,
            container=container,
        )
        vlabel.set_text(sc.fmt.format(cur))
        app.sidebar._scale_sliders[slider] = (sc.attr, sc.is_int, vlabel, sc.fmt)
        y += pitch
    return y


def _build_perrun_section(
    app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
) -> int:
    """eta + Perlin init, then the raw cut-schedule text boxes (applied on next Play)."""
    ui = pygame_gui.elements
    cfg = app.session.config
    y = app._section_label(container, inner_w, y + 6, "Per-run — apply on next Play")
    r = Row(0, y, inner_w, CTRL_H)
    ui.UILabel(r.left(40), "eta", app.manager, container=container)
    app.sidebar.perlin_button = ui.UIButton(
        r.right(96), "Perlin init", app.manager, container=container
    )
    app.sidebar._eta_label = ui.UILabel(r.right(48), "", app.manager, container=container)
    app.sidebar._eta_slider = ui.UIHorizontalSlider(
        r.fill(),
        start_value=min(max(float(cfg.eta), 0.0), 1.0),
        value_range=(0.0, 1.0),
        manager=app.manager,
        container=container,
    )
    app.sidebar._eta_label.set_text(f"{cfg.eta:.2f}")
    if cfg.perlin_init:
        app.sidebar.perlin_button.select()
    y += pitch
    for sch in SCHEDULES:
        ui.UILabel(
            Row(0, y, inner_w, LABEL_H).left(inner_w),
            sch.label,
            app.manager,
            container=container,
        )
        y += LABEL_H + 2
        entry = ui.UITextEntryLine(
            Row(0, y, inner_w, CTRL_H).fill(), app.manager, container=container
        )
        entry.set_text(str(getattr(cfg, sch.attr)))
        app.sidebar._schedule_entries[entry] = sch.attr
        y += pitch
    return y


def _build_models_section(
    app: App, container: UIScrollingContainer, inner_w: int, pitch: int, y: int
) -> int:
    """CLIP model toggles + the secondary-model toggle (changing these auto-reloads)."""
    ui = pygame_gui.elements
    y = app._section_label(container, inner_w, y + 6, "Models — auto-reloads on change")
    per_row = 2
    bw = (inner_w - (per_row - 1) * PAD) // per_row
    r = Row(0, y, inner_w, CTRL_H)
    for i, name in enumerate(AVAILABLE_CLIP_MODELS):
        if i % per_row == 0:
            r = Row(0, y, inner_w, CTRL_H)
        button = ui.UIButton(
            r.left(bw), name, app.manager, container=container, object_id="#brush_button"
        )
        app.sidebar._clip_buttons[button] = name
        if name in app._clip_selected:
            button.select()
        if i % per_row == per_row - 1:
            y += pitch
    if len(AVAILABLE_CLIP_MODELS) % per_row != 0:
        y += pitch
    r = Row(0, y, inner_w, CTRL_H)
    app.sidebar.secondary_button = ui.UIButton(
        r.fill(),
        "Secondary model",
        app.manager,
        container=container,
        object_id="#brush_button",
    )
    if app._secondary_on:
        app.sidebar.secondary_button.select()
    return y + pitch


def _build_current_rows(app: App) -> None:
    """Build the read-only "Current" tab: name + value label per setting."""
    ui = pygame_gui.elements
    container = app.sidebar.current_panel
    inner_w = app.sidebar._sb_inner_w
    pitch = CTRL_H + 4
    app.sidebar._current_labels = {}
    name_w = 116

    def row(y: int, key: str, name: str) -> int:
        ui.UILabel(Row(0, y, inner_w, CTRL_H).left(name_w), name, app.manager, container=container)
        value = ui.UILabel(
            Row(name_w + PAD, y, inner_w - name_w - PAD, CTRL_H).fill(),
            "",
            app.manager,
            container=container,
        )
        app.sidebar._current_labels[key] = value
        return y + pitch

    # Headings are plain — the Current tab reflects what the image is generating with, so the
    # values speak for themselves without a "this run / next run" qualifier.
    y = 2
    ui.UILabel(
        Row(0, y, inner_w, LABEL_H).fill(),
        "Guidance",
        app.manager,
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
        app.manager,
        container=container,
        object_id="#section_label",
    )
    y += LABEL_H + 6
    for key, name in CURRENT_PERRUN:
        y = row(y, key, name)
    container.set_scrollable_area_dimensions((inner_w, y + 8))
    app._refresh_current()


def _sync_sidebar_tabs(app: App) -> None:
    """Show exactly one of the Settings / Current panels for the active sidebar tab."""
    settings = app.sidebar._sidebar_tab == "settings"
    (app.sidebar.settings_panel.show if settings else app.sidebar.settings_panel.hide)()
    (app.sidebar.current_panel.show if not settings else app.sidebar.current_panel.hide)()
    (app.sidebar.tab_settings.select if settings else app.sidebar.tab_settings.unselect)()
    (app.sidebar.tab_current.select if not settings else app.sidebar.tab_current.unselect)()


def _refresh_advanced_widgets(app: App) -> None:
    """Re-sync every Advanced widget from the current config (after a preset load)."""
    cfg = app.session.config
    for slider, (attr, _is_int, vlabel, fmt) in app.sidebar._scale_sliders.items():
        value = float(getattr(cfg, attr))
        slider.set_current_value(value)
        vlabel.set_text(fmt.format(value))
    app.sidebar._eta_slider.set_current_value(min(max(float(cfg.eta), 0.0), 1.0))
    app.sidebar._eta_label.set_text(f"{cfg.eta:.2f}")
    (app.sidebar.perlin_button.select if cfg.perlin_init else app.sidebar.perlin_button.unselect)()
    for entry, attr in app.sidebar._schedule_entries.items():
        entry.set_text(str(getattr(cfg, attr)))


def _spawn_preset_dropdown(app: App) -> None:
    """(Re)create the preset dropdown in the settings panel at the stored rect/selection."""
    if app.sidebar.preset_dropdown is not None:
        app.sidebar.preset_dropdown.kill()
    options: list[str | tuple[str, str]] = [*app._presets.keys(), CUSTOM_PRESET]
    selected = app._preset_selection if app._preset_selection in options else CUSTOM_PRESET
    app._preset_selection = selected
    app.sidebar.preset_dropdown = UIDropDownMenu(
        options,
        selected,
        app.sidebar._preset_dd_rect,
        app.manager,
        container=app.sidebar.settings_panel,
    )


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
