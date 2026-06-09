"""The bottom panel of the left column: transport, history scrubber, paint tools, prompt list.

:class:`BottomBar` owns those widgets as one named group, so the rest of the app reaches them via
``app.bottom_bar.*``. Built by the ``_ui_build`` helpers and driven by the App's event router.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame
from pygame_gui.core import UIElement
from pygame_gui.elements import (
    UIButton,
    UIHorizontalSlider,
    UILabel,
    UIScrollingContainer,
    UITextEntryLine,
)

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
