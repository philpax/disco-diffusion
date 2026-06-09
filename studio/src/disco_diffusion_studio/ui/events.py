"""The top-level event router for the studio App.

Handles the cross-cutting input directly — window/quit, the modal guard, canvas mouse (paint /
pan / zoom), the dividers, keyboard shortcuts, and the modal dialog widgets — then delegates each
remaining widget event to the area that owns it (``bottom_bar.handle`` / ``sidebar.handle``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
import pygame_gui

from ..layout import DIVIDER_W

if TYPE_CHECKING:
    from ..app import App


def handle(app: App, event: pygame.event.Event) -> bool:
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
            app.canvas.clamp_pan()
        elif app.canvas.paint.painting:
            app.canvas.paint_at(event.pos)
        return True
    if event.type == pygame.MOUSEBUTTONDOWN:
        # The draggable divider sits between the left column and the sidebar (full height).
        if event.button == 1 and abs(event.pos[0] - app.layout.divider_x()) <= DIVIDER_W:
            app._dragging_divider = True
            return True
        # Horizontal divider between the image area and the bottom panel (left column only).
        if (
            event.button == 1
            and event.pos[0] < app.layout.panel_w()
            and abs(event.pos[1] - app.layout.image_area_h()) <= DIVIDER_W
        ):
            app._dragging_panel = True
            return True
        on_canvas = app.layout.image_region().collidepoint(event.pos)
        if event.button == 3 and on_canvas:  # right held = navigate mode (pan + scroll-zoom)
            app._navigating = True
            app._panning = True
            return True
        if event.button == 2 and on_canvas:  # middle-drag also pans
            app._panning = True
            return True
        if event.button == 1:  # left-drag on the canvas paints
            if app.bottom_bar.on_swatch(app, event.pos):
                return True
            # No painting while previewing history — it would be invisible and unapplied.
            on_canvas = app.canvas.screen_to_canvas(event.pos) is not None
            if app.state.timeline.preview_index is None and on_canvas:
                app.canvas.paint.begin()
                app.canvas.paint_at(event.pos)
            return True
    if event.type == pygame.MOUSEBUTTONUP:
        if event.button == 1:
            app._dragging_divider = False
            app._dragging_panel = False
            if app.canvas.paint.painting:  # a completed stroke becomes one batch / checkpoint
                app.canvas.paint.end()
                app.canvas.paint.flush(app.state.worker, app.paint.brush)
            app.canvas.paint.last_gen = None
        elif event.button == 3:
            app._navigating = False
            app._panning = False
        elif event.button == 2:
            app._panning = False
    if event.type == pygame.MOUSEWHEEL and app.layout.image_region().collidepoint(app._mouse_pos):
        if app._navigating:  # canvas mode: wheel zooms toward the cursor
            app.canvas.zoom_at(app._mouse_pos, 1.15**event.y)
        elif pygame.key.get_mods() & pygame.KMOD_SHIFT:
            app.bottom_bar.nudge_brush_strength(app, event.y * 0.05)
        else:
            app.bottom_bar.nudge_brush_size(app, 1.1**event.y)
        return True
    if event.type == pygame.KEYDOWN and not app._typing() and not app._modal_open():
        if event.mod & pygame.KMOD_CTRL:  # ctrl combos: Save / Revert
            if event.key == pygame.K_s:
                app.generation.save_image()
            elif event.key == pygame.K_z:
                app.history.keyboard_revert()
        elif event.key == pygame.K_SPACE:
            app.generation.toggle_play()
        elif event.key == pygame.K_f:
            app.canvas.fit()
        elif event.key == pygame.K_0:
            app.canvas.zoom_at(app.layout.image_region().center, 1.0 / app.canvas.view.zoom)
        elif event.key == pygame.K_LEFTBRACKET:
            app.bottom_bar.nudge_brush_size(app, 1.0 / 1.1)
        elif event.key == pygame.K_RIGHTBRACKET:
            app.bottom_bar.nudge_brush_size(app, 1.1)
        elif pygame.K_1 <= event.key <= pygame.K_9:  # digit -> nth palette/recents swatch
            app.bottom_bar.select_palette_index(app, event.key - pygame.K_1)

    # Widget events route to the area that owns the widget; the App keeps the modal dialog
    # widgets (save-preset window, colour picker, reset-confirm) + drag-drop below.
    if app.bottom_bar.handle(app, event):
        return True
    if app.sidebar.handle(app, event):
        return True

    if event.type == pygame_gui.UI_BUTTON_PRESSED:
        if event.ui_element is app.recipe.save_ok:
            app.recipe.save_current()
        elif event.ui_element is app.recipe.save_cancel:
            app.recipe.close_save_dialog()
    elif event.type == pygame_gui.UI_COLOUR_PICKER_COLOUR_PICKED:
        if event.ui_element is app._colour_picker:
            col = event.colour
            app.bottom_bar.apply_picked_colour(app, (col.r, col.g, col.b))
    elif event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
        if event.ui_element is app.recipe.save_entry:
            app.recipe.save_current()  # Enter in the filename box saves
    elif event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
        if event.ui_element is app._confirm_dialog:
            app._reset_canvas()
            app._confirm_dialog = None
    elif event.type == pygame.DROPFILE:  # drag-drop an image onto the window -> init image
        app._load_init_file(event.file)
    elif event.type == pygame_gui.UI_WINDOW_CLOSE:
        if event.ui_element is app.recipe.save_window:
            app.recipe.close_save_dialog()
        elif event.ui_element is app._colour_picker:
            app._colour_picker = None
        elif event.ui_element is app._confirm_dialog:  # cancelled
            app._confirm_dialog = None

    return True


# -- drawing --
