"""Frame rendering / HUD for the studio App.

Free functions taking the ``App``: composite the canvas + paint overlays + chrome, the on-canvas
tool cursor and help HUD, and the history-slider checkpoint ticks. Kept out of ``app.py`` so the
renderer is its own module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from .constants import CANVAS_BORDER, CANVAS_EMPTY_BG, DRAW_HELP, NAV_HELP
from .layout import DIVIDER_W
from .theme import DIVIDER, IMAGE_BG, PANEL_BG, WINDOW_BG

if TYPE_CHECKING:
    from .app import App


def _draw(app: App) -> None:
    win_w, win_h = app.layout.window_size()
    img_h = app.layout.image_area_h()
    panel_w = app.layout.panel_w()
    app.screen.fill(WINDOW_BG)
    pygame.draw.rect(app.screen, IMAGE_BG, (0, 0, panel_w, img_h))
    pygame.draw.rect(app.screen, PANEL_BG, (0, img_h, panel_w, win_h - img_h))
    pygame.draw.rect(app.screen, PANEL_BG, app.layout.sidebar_rect())  # full-height sidebar
    # Draggable divider band between the left column and the sidebar.
    div = pygame.Rect(panel_w, 0, DIVIDER_W, win_h)
    hot = app._dragging_divider or abs(app._mouse_pos[0] - panel_w) <= DIVIDER_W
    pygame.draw.rect(app.screen, DIVIDER, div)
    grip_x = panel_w + DIVIDER_W // 2
    pygame.draw.line(
        app.screen,
        (110, 120, 140) if hot else (70, 78, 92),
        (grip_x, win_h // 2 - 14),
        (grip_x, win_h // 2 + 14),
        2,
    )
    # Draw the canvas (and unbaked paint overlay) under the view transform, clipped to
    # the viewport so a zoomed/panned canvas never spills into the panel.
    app.screen.set_clip(app.layout.image_region())
    crect = app.canvas.canvas_screen_rect()
    surface = app._displayed_surface()
    if surface is not None:
        app.canvas.blit(surface)
    elif app._init.surface is not None:
        # No frame yet but an init image is set: preview it (dimmed) so it's clear the run
        # will seed from it.
        app.canvas.blit(app._init.surface)
        scrim = pygame.Surface(crect.size, pygame.SRCALPHA)
        scrim.fill((10, 12, 16, 120))
        app.screen.blit(scrim, crect.topleft)
        label = app._hud_font.render(
            f"init: {app._init.label} — press Play to evolve", True, (224, 228, 236)
        )
        app.screen.blit(label, label.get_rect(center=crect.center))
    else:
        # No frame yet: show the canvas bounds so the size/aspect is clear before Play.
        pygame.draw.rect(app.screen, CANVAS_EMPTY_BG, crect)
        label = app._hud_font.render(
            f"{app.width} × {app.height} — press Play", True, (140, 147, 160)
        )
        app.screen.blit(label, label.get_rect(center=crect.center))
    # Paint overlays only on the live view (hidden while previewing history): the in-progress
    # stroke plus any flushed strokes not yet baked into a published frame.
    if app._timeline.preview_index is None:
        for overlay, _ in app.canvas.paint.pending_overlays:
            app.canvas.blit(overlay)
        if not app.canvas.paint.layer.empty():
            app.canvas.blit(app.canvas.paint.layer.to_surface())
    pygame.draw.rect(app.screen, CANVAS_BORDER, crect, 1)  # canvas outline at any zoom
    app.screen.set_clip(None)
    # Draggable horizontal divider between the image area and the bottom panel, with a grip
    # that lights up on hover/drag (mirrors the sidebar divider).
    pygame.draw.line(app.screen, DIVIDER, (0, img_h), (panel_w, img_h))
    hot_h = app._dragging_panel or (
        app._mouse_pos[0] < panel_w and abs(app._mouse_pos[1] - img_h) <= DIVIDER_W
    )
    grip_cx = panel_w // 2
    pygame.draw.line(
        app.screen,
        (110, 120, 140) if hot_h else (70, 78, 92),
        (grip_cx - 14, img_h),
        (grip_cx + 14, img_h),
        2,
    )


def _draw_tools(app: App) -> None:
    """Draw the colour palette, the brush-preview ring, and the canvas help HUD."""
    # Palette: current-colour preview + swatches (selected one outlined).
    pygame.draw.rect(
        app.screen, app.brush.color, app.bottom_bar._color_preview_rect, border_radius=5
    )
    pygame.draw.rect(
        app.screen, DIVIDER, app.bottom_bar._color_preview_rect, width=1, border_radius=5
    )
    for sr, color in app.bottom_bar._swatch_rects:
        pygame.draw.rect(app.screen, color, sr, border_radius=4)
        if color == app.brush.color:
            pygame.draw.rect(app.screen, (255, 255, 255), sr, width=2, border_radius=4)
    region = app.layout.image_region()
    # Brush ring (scaled by zoom) — only in draw mode (not navigating, not previewing, and
    # not while a dialog window is up).
    if (
        not app._navigating
        and app._timeline.preview_index is None
        and not app._modal_open()
        and region.collidepoint(app._mouse_pos)
    ):
        ring = max(2, int(app.brush.size * app.canvas.view.zoom))
        pygame.draw.circle(app.screen, app.brush.color, app._mouse_pos, ring, 2)
        pygame.draw.circle(app.screen, (255, 255, 255), app._mouse_pos, ring + 1, 1)
    # Help HUD in the corner of the canvas (doesn't cost panel height), per mode.
    text = app._hud_font.render(NAV_HELP if app._navigating else DRAW_HELP, True, (210, 214, 222))
    pad = 6
    chip = pygame.Surface(
        (text.get_width() + 2 * pad, text.get_height() + 2 * pad), pygame.SRCALPHA
    )
    chip.fill((0, 0, 0, 120))
    pos = (10, region.bottom - chip.get_height() - 10)
    app.screen.blit(chip, pos)
    app.screen.blit(text, (pos[0] + pad, pos[1] + pad))


def _draw_history_ticks(app: App) -> None:
    """Mark each checkpoint's position on the (step-space) history slider.

    Drawn after the UI so the ticks sit on top of the track; hovering the slider shows the
    nearest checkpoint's label so the otherwise-invisible snap points are discoverable.
    """
    tl = app._timeline
    if not tl.entries or not app.bottom_bar.history_slider.is_enabled:
        return
    total = app._history_total()
    track = app.bottom_bar.history_slider.rect
    button = getattr(app.bottom_bar.history_slider, "sliding_button", None)
    button_w = button.rect.width if button is not None else 26
    base_y = track.bottom - 3
    for i, cp in enumerate(tl.entries):
        x = tl.tick_x(float(cp.index), track, button_w, total)
        active = tl.preview_index == i
        kind = tl.tick_colour(cp.label)
        colour = tl.brighten(kind) if active else kind
        height = 9 if active else 5
        width = 2 if active else 1
        pygame.draw.line(app.screen, colour, (x, base_y - height), (x, base_y), width)
    # Hover: surface the nearest checkpoint's label above the slider (and accent its tick).
    if track.collidepoint(app._mouse_pos) and not app._modal_open():
        span = max(1, track.width - button_w)
        mval = (app._mouse_pos[0] - track.left - button_w / 2) / span * float(max(total, 1))
        cp = min(tl.entries, key=lambda c: abs(c.index - mval))
        x = tl.tick_x(float(cp.index), track, button_w, total)
        accent = tl.brighten(tl.tick_colour(cp.label))
        pygame.draw.line(app.screen, accent, (x, base_y - 9), (x, base_y), 2)
        label = f"{cp.label}  {cp.index}/{cp.total}"
        text = app._hud_font.render(label, True, (228, 232, 240))
        pad = 5
        chip = pygame.Surface(
            (text.get_width() + 2 * pad, text.get_height() + 2 * pad), pygame.SRCALPHA
        )
        chip.fill((18, 20, 28, 235))
        cx = x - chip.get_width() // 2
        cx = max(track.left, min(cx, track.right - chip.get_width()))
        cy = track.top - chip.get_height() - 3
        app.screen.blit(chip, (cx, cy))
        app.screen.blit(text, (cx + pad, cy + pad))


# -- colours --
