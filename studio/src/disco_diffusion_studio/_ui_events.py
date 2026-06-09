"""Event routing for the studio App.

A free function taking the ``App``: the single ``_handle_event`` dispatcher mapping pygame /
pygame_gui events to the App's actions (paint, pan/zoom, dividers, widgets, keyboard shortcuts).
Kept out of ``app.py`` so the ~300-line dispatcher is its own module.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pygame
import pygame_gui

from .constants import GUIDANCE_CHECKPOINT_MS
from .controls import CUSTOM_PRESET, PromptRow
from .layout import DIVIDER_W
from .util import int_or

if TYPE_CHECKING:
    from .app import App


def _handle_event(app: App, event: pygame.event.Event) -> bool:
    if event.type == pygame.QUIT:
        return False

    if event.type == pygame.VIDEORESIZE:
        # Resize events stream while dragging; coalesce to one relayout per frame
        # (see run()) instead of rebuilding the UI on every event.
        app._pending_size = (event.w, event.h)
        return True

    # While a dialog window is open, let pygame_gui own the mouse/keyboard (it processed the
    # event already, before us) and skip our canvas interactions so they don't leak through.
    _MOUSE = (
        pygame.MOUSEMOTION,
        pygame.MOUSEBUTTONDOWN,
        pygame.MOUSEBUTTONUP,
        pygame.MOUSEWHEEL,
    )
    if event.type in _MOUSE and app._modal_open():
        if event.type == pygame.MOUSEMOTION:
            app._mouse_pos = event.pos  # keep the cursor position current for hover/HUD
        return True

    if event.type == pygame.MOUSEMOTION:
        app._mouse_pos = event.pos
        if app._dragging_divider:
            app._set_sidebar_width(app.layout.win_w - event.pos[0] - DIVIDER_W // 2)
        elif app._dragging_panel:
            app._set_panel_height(app.layout.win_h - event.pos[1] - DIVIDER_W // 2)
        elif app._panning:
            app.canvas.view.pan += pygame.Vector2(event.rel)
            app._clamp_pan()
        elif app.canvas.paint.painting:
            app._paint_at(event.pos)
        return True
    if event.type == pygame.MOUSEBUTTONDOWN:
        # The draggable divider sits between the left column and the sidebar (full height).
        if event.button == 1 and abs(event.pos[0] - app._divider_x()) <= DIVIDER_W:
            app._dragging_divider = True
            return True
        # Horizontal divider between the image area and the bottom panel (left column only).
        if (
            event.button == 1
            and event.pos[0] < app._panel_w()
            and abs(event.pos[1] - app._image_area_h()) <= DIVIDER_W
        ):
            app._dragging_panel = True
            return True
        on_canvas = app._image_region().collidepoint(event.pos)
        if event.button == 3 and on_canvas:  # right held = navigate mode (pan + scroll-zoom)
            app._navigating = True
            app._panning = True
            return True
        if event.button == 2 and on_canvas:  # middle-drag also pans
            app._panning = True
            return True
        if event.button == 1:  # left-drag on the canvas paints
            if app._on_swatch(event.pos):
                return True
            # No painting while previewing history — it would be invisible and unapplied.
            on_canvas = app._screen_to_canvas(event.pos) is not None
            if app._timeline.preview_index is None and on_canvas:
                app.canvas.paint.begin()
                app._paint_at(event.pos)
            return True
    if event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            app._dragging_divider = False
            app._dragging_panel = False
            if app.canvas.paint.painting:  # a completed stroke becomes one batch / checkpoint
                app.canvas.paint.end()
                app.canvas.paint.flush(app.worker, app.brush)
            app.canvas.paint.last_gen = None
        elif event.button == 3:
            app._navigating = False
            app._panning = False
        elif event.button == 2:
            app._panning = False
    if event.type == pygame.MOUSEWHEEL and app._image_region().collidepoint(app._mouse_pos):
        if app._navigating:  # canvas mode: wheel zooms toward the cursor
            app._zoom_at(app._mouse_pos, 1.15**event.y)
        elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
            app._nudge_brush_strength(event.y * 0.05)
        else:
            app._nudge_brush_size(1.1**event.y)
        return True
    if event.type == pygame.KEYDOWN and not app._typing() and not app._modal_open():
        if event.mod & pygame.KMOD_CTRL:  # ctrl combos: Save / Revert
            if event.key == pygame.K_s:
                app._save_image()
            elif event.key == pygame.K_z:
                app._keyboard_revert()
        elif event.key == pygame.K_SPACE:
            app._toggle_play()
        elif event.key == pygame.K_f:
            app._fit_view()
        elif event.key == pygame.K_0:
            app._zoom_at(app._image_region().center, 1.0 / app.canvas.view.zoom)
        elif event.key == pygame.K_LEFTBRACKET:
            app._nudge_brush_size(1.0 / 1.1)
        elif event.key == pygame.K_RIGHTBRACKET:
            app._nudge_brush_size(1.1)
        elif pygame.K_1 <= event.key <= pygame.K_9:  # digit -> nth palette/recents swatch
            app._select_palette_index(event.key - pygame.K_1)

    if event.type == pygame_gui.UI_BUTTON_PRESSED:
        if event.ui_element == app.bottom_bar.play_button:
            app._toggle_play()
        elif event.ui_element == app.bottom_bar.stop_button:
            app._stop_run()
            app._status("Stopped")
        elif event.ui_element == app.bottom_bar.reset_button:
            app._open_reset_confirm()
        elif event.ui_element == app.bottom_bar.save_button:
            app._save_image()
        elif event.ui_element in (app.sidebar.tab_settings, app.sidebar.tab_current):
            app.sidebar._sidebar_tab = (
                "settings" if event.ui_element == app.sidebar.tab_settings else "current"
            )
            app.sidebar.sync_tabs(app)
        elif event.ui_element == app.sidebar.save_preset_button:
            app._open_save_preset_dialog()
        elif event.ui_element == app.sidebar.save_session_button:
            app._save_session()
        elif event.ui_element == app.sidebar.load_session_button:
            app._load_session()
        elif event.ui_element == app.bottom_bar.pick_color_button:
            app._open_colour_picker()
        elif event.ui_element == app.sidebar.open_init_button:
            app._open_init()
        elif event.ui_element == app.sidebar.use_current_init_button:
            app._use_current_as_init()
        elif event.ui_element == app.sidebar.clear_init_button:
            app._clear_init()
        elif event.ui_element is app._save_preset_ok:
            app._save_current_preset()
        elif event.ui_element is app._save_preset_cancel:
            app._close_save_preset_dialog()
        elif event.ui_element == app.sidebar.perlin_button:
            app.session.config.perlin_init = not app.session.config.perlin_init
            on = app.session.config.perlin_init
            (app.sidebar.perlin_button.select if on else app.sidebar.perlin_button.unselect)()
            app._status(f"Perlin {'on' if on else 'off'}")
            app._mark_custom()
        elif event.ui_element in app.sidebar._clip_buttons:
            app._toggle_clip_model(event.ui_element)
        elif event.ui_element == app.sidebar.secondary_button:
            app._secondary_on = not app._secondary_on
            (
                app.sidebar.secondary_button.select
                if app._secondary_on
                else app.sidebar.secondary_button.unselect
            )()
            app._update_reload_queue()
            app._mark_custom()
        elif event.ui_element == app.bottom_bar.add_button:
            app.prompts.append(PromptRow("", 1.0))
            app.bottom_bar.rebuild_prompt_rows(app)
            app._push_prompts()
            app._request_checkpoint("add prompt")
        elif event.ui_element == app.sidebar.apply_button:
            app._apply_size(
                int_or(app.sidebar.width_entry.get_text(), app.width),
                int_or(app.sidebar.height_entry.get_text(), app.height),
            )
        elif event.ui_element == app.sidebar.swap_button:
            app._apply_size(app.height, app.width)
        elif event.ui_element == app.sidebar.random_seed_button:
            app._seed_text = str(random.randrange(2**31))  # roll a fresh, visible seed
            app.sidebar.seed_entry.set_text(app._seed_text)
            app._status(f"Seed {app._seed_text}")
        elif event.ui_element in app.bottom_bar._remove_buttons:
            idx = app.bottom_bar._remove_buttons[event.ui_element]
            if 0 <= idx < len(app.prompts):
                app.prompts.pop(idx)
                app.bottom_bar.rebuild_prompt_rows(app)
                app._push_prompts()
                app._request_checkpoint("remove prompt")
        elif event.ui_element in app.bottom_bar._mute_buttons:
            idx = app.bottom_bar._mute_buttons[event.ui_element]
            if 0 <= idx < len(app.prompts):
                prompt = app.prompts[idx]
                prompt.muted = not prompt.muted
                (event.ui_element.select if prompt.muted else event.ui_element.unselect)()
                app.bottom_bar.refresh_rows(app)
                app._push_prompts()  # re-mix conditioning (muted excluded)
                app._request_checkpoint("mute prompt" if prompt.muted else "unmute prompt")
        elif event.ui_element in app.bottom_bar._brush_buttons:
            app.brush.type = app.bottom_bar._brush_buttons[event.ui_element]
            for button, name in app.bottom_bar._brush_buttons.items():
                (button.select if name == app.brush.type else button.unselect)()
        elif event.ui_element == app.bottom_bar.noise_button:
            app.brush.noise = not app.brush.noise
            (
                app.bottom_bar.noise_button.select
                if app.brush.noise
                else app.bottom_bar.noise_button.unselect
            )()
        elif event.ui_element == app.bottom_bar.clear_paint_button:
            app.canvas.paint.layer.clear()
        elif event.ui_element == app.bottom_bar.revert_button:
            app._do_revert()
        elif event.ui_element == app.bottom_bar.cancel_button:
            app._timeline.preview_index = None
            app.bottom_bar.history_slider.set_current_value(float(app._live_index()))
            app._refresh_preview_state()

    elif event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
        if event.ui_element == app.bottom_bar.size_slider:
            app.brush.size = float(event.value)
        elif event.ui_element == app.bottom_bar.strength_slider:
            app.brush.strength = float(event.value)
        elif event.ui_element == app.bottom_bar.history_slider:
            # The slider is in step-space; snap the dragged value to the nearest checkpoint
            # (or live), and park the thumb on that checkpoint's actual step position.
            snap_idx = app._timeline.snap(float(event.value), app._live_index())
            app._timeline.preview_index = snap_idx
            if snap_idx is not None:
                snapped = float(app._timeline.entries[snap_idx].index)
            else:
                snapped = float(app._live_index())
            app.bottom_bar.history_slider.set_current_value(snapped)
            app._refresh_preview_state()
        elif event.ui_element == app.sidebar._eta_slider:
            # eta is read when the loop's generator is built, so this lands on the next run.
            app.session.config.eta = float(event.value)
            app.sidebar._eta_label.set_text(f"{event.value:.2f}")
            app._mark_custom()
        elif event.ui_element == app.sidebar._init_denoise_slider:
            # img2img strength — converted to skip_steps at the next Play.
            app._init.denoise = int(round(event.value))
            app.sidebar._init_denoise_label.set_text(f"{app._init.denoise}%")
        elif event.ui_element in app.sidebar._scale_sliders:
            attr, is_int, vlabel, fmt = app.sidebar._scale_sliders[event.ui_element]
            value: float | int = int(round(event.value)) if is_int else float(event.value)
            # session.config is the live config the running Sampler reads each step, so
            # this retunes guidance on the next step (and seeds the next run when stopped).
            setattr(app.session.config, attr, value)
            vlabel.set_text(fmt.format(value))
            app._mark_custom()
            # Drop a revert point once the drag settles (debounced in run()).
            app._guidance_checkpoint_at = pygame.time.get_ticks() + GUIDANCE_CHECKPOINT_MS
        else:
            slider_idx = app.bottom_bar._weight_sliders.get(event.ui_element)
            if slider_idx is not None and 0 <= slider_idx < len(app.prompts):
                app.prompts[slider_idx].weight = float(event.value)
                app.bottom_bar.refresh_rows(app)
                app._push_prompts()

    elif event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
        if event.ui_element == app.sidebar.preset_dropdown:
            app._preset_selection = event.text
            if event.text != CUSTOM_PRESET:
                app._apply_preset(event.text)

    elif event.type == pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED:
        if event.ui_element is app._colour_picker:
            col = event.colour
            app._apply_picked_colour((col.r, col.g, col.b))

    elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
        if event.ui_element == app.sidebar.steps_entry:
            app._commit_steps()
        elif event.ui_element in (app.sidebar.width_entry, app.sidebar.height_entry):
            pass  # applied via the Apply button
        elif event.ui_element is app._save_preset_entry:
            app._save_current_preset()  # Enter in the filename box saves
        elif event.ui_element in app.bottom_bar._prompt_entries:
            app._commit_prompt_entry(event.ui_element)
        elif event.ui_element in app.sidebar._schedule_entries:
            app._commit_schedule_entry(event.ui_element)

    elif event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
        if event.ui_element is app._confirm_dialog:
            app._reset_canvas()
            app._confirm_dialog = None

    elif event.type == pygame.DROPFILE:  # drag-drop an image onto the window -> init image
        app._load_init_file(event.file)

    elif event.type == pygame_gui.UI_WINDOW_CLOSE:
        if event.ui_element is app._save_preset_window:
            app._close_save_preset_dialog()
        elif event.ui_element is app._colour_picker:
            app._colour_picker = None
        elif event.ui_element is app._confirm_dialog:  # cancelled
            app._confirm_dialog = None

    return True


# -- drawing --
