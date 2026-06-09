"""The right sidebar: the Settings tab + the read-only Current tab + session save/load.

:class:`Sidebar` owns the sidebar's widgets as one named group, so the rest of the app reaches
them via ``app.sidebar.*`` rather than as ~30 fields on the god-class. It's built by the
``_ui_build`` helpers and driven by the App's event router (which will fold into this class next).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame
from pygame_gui.elements import (
    UIButton,
    UIDropDownMenu,
    UIHorizontalSlider,
    UILabel,
    UIScrollingContainer,
    UITextEntryLine,
)


@dataclass
class Sidebar:
    """Owns the sidebar widgets (tabs, panels, and the Settings/Current controls)."""

    # Tabs + the two scrolling panels they switch between.
    tab_settings: UIButton = field(init=False)
    tab_current: UIButton = field(init=False)
    settings_panel: UIScrollingContainer = field(init=False)
    current_panel: UIScrollingContainer = field(init=False)
    # Session save/load (top of the sidebar, above the tabs).
    save_session_button: UIButton = field(init=False)
    load_session_button: UIButton = field(init=False)
    # Output section.
    steps_entry: UITextEntryLine = field(init=False)
    seed_entry: UITextEntryLine = field(init=False)
    random_seed_button: UIButton = field(init=False)
    width_entry: UITextEntryLine = field(init=False)
    height_entry: UITextEntryLine = field(init=False)
    apply_button: UIButton = field(init=False)
    swap_button: UIButton = field(init=False)
    # Init-image section.
    open_init_button: UIButton = field(init=False)
    use_current_init_button: UIButton = field(init=False)
    clear_init_button: UIButton = field(init=False)
    _init_status_label: UILabel = field(init=False)
    _init_denoise_label: UILabel = field(init=False)
    _init_denoise_slider: UIHorizontalSlider = field(init=False)
    # Preset section.
    save_preset_button: UIButton = field(init=False)
    _preset_dd_rect: pygame.Rect = field(init=False)
    # Per-run section.
    perlin_button: UIButton = field(init=False)
    secondary_button: UIButton = field(init=False)
    _eta_label: UILabel = field(init=False)
    _eta_slider: UIHorizontalSlider = field(init=False)
    # Content width of the settings panel (minus the scrollbar).
    _sb_inner_w: int = field(init=False)

    # State / dicts (populated during build; safe defaults until then).
    preset_dropdown: UIDropDownMenu | None = None
    _sidebar_tab: str = "settings"  # "settings" | "current"
    _scale_sliders: dict[UIHorizontalSlider, tuple[str, bool, UILabel, str]] = field(
        default_factory=dict
    )
    _schedule_entries: dict[UITextEntryLine, str] = field(default_factory=dict)
    _clip_buttons: dict[UIButton, str] = field(default_factory=dict)
    _current_labels: dict[str, UILabel] = field(default_factory=dict)
