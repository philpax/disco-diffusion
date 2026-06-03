"""Colour palette and pygame_gui theme for the studio UI."""

from __future__ import annotations

from typing import Any

# Palette for surfaces we draw ourselves (widget colours live in THEME below).
WINDOW_BG = (15, 17, 21)
PANEL_BG = (26, 30, 39)
IMAGE_BG = (8, 9, 12)
DIVIDER = (44, 50, 62)

# Per-row weight-readout text colours.
READOUT_COLOR = (150, 196, 255)  # the normalised % the guidance is using
MUTED_COLOR = (122, 130, 144)  # empty / off rows
PENDING_COLOR = (245, 176, 66)  # amber: text edited but not yet applied

# A cohesive dark UI with an indigo accent, rounded shapes, and arrow-less sliders.
# Loaded onto the pygame_gui manager in App.__post_init__.
ACCENT = "#6d7cff"
ACCENT_HI = "#828fff"

THEME: dict[str, Any] = {
    "defaults": {
        "colours": {
            "normal_bg": "#262c38",
            "hovered_bg": "#313a4b",
            "disabled_bg": "#1b1f27",
            "selected_bg": ACCENT,
            "active_bg": ACCENT,
            "dark_bg": "#12151c",
            "normal_text": "#e7e9f0",
            "hovered_text": "#ffffff",
            "selected_text": "#ffffff",
            "disabled_text": "#5b626f",
            "active_text": "#ffffff",
            "normal_border": "#39414f",
            "hovered_border": ACCENT,
            "disabled_border": "#262c38",
            "link_text": ACCENT,
        },
        "font": {"name": "noto_sans", "size": "14"},
    },
    "button": {
        "misc": {
            "shape": "rounded_rectangle",
            "shape_corner_radius": "8",
            "border_width": "1",
        }
    },
    "text_entry_line": {
        "colours": {"dark_bg": "#11141b", "normal_border": "#39414f", "text_cursor_colour": ACCENT},
        "misc": {"shape": "rounded_rectangle", "shape_corner_radius": "6", "border_width": "1"},
    },
    "horizontal_slider": {
        "colours": {"normal_bg": "#11141b", "hovered_bg": "#11141b", "disabled_bg": "#1b1f27"},
        "misc": {
            "enable_arrow_buttons": "0",
            "shape": "rounded_rectangle",
            "shape_corner_radius": "5",
            "sliding_button_width": "26",
        },
    },
    "#sliding_button": {
        "colours": {"normal_bg": ACCENT, "hovered_bg": ACCENT_HI, "normal_border": ACCENT},
        "misc": {"shape": "rounded_rectangle", "shape_corner_radius": "5"},
    },
    "label": {"colours": {"normal_text": "#c4cad6"}},
    "#step_label": {
        "colours": {"normal_text": "#ffffff"},
        "font": {"name": "noto_sans", "size": "15", "bold": "1"},
    },
    "#section_label": {
        "colours": {"normal_text": "#8b93a3"},
        "font": {"name": "noto_sans", "size": "13", "bold": "1"},
    },
    "#hint_label": {
        "colours": {"normal_text": "#6f7787"},
        "font": {"name": "noto_sans", "size": "12"},
    },
    "#status_label": {"colours": {"normal_text": "#9aa3b2"}},
    "#play_button": {
        "colours": {
            "normal_bg": ACCENT,
            "hovered_bg": ACCENT_HI,
            "normal_border": ACCENT,
            "normal_text": "#ffffff",
        }
    },
    "#stop_button": {
        "colours": {
            "normal_bg": "#2a2f3a",
            "hovered_bg": "#3a2530",
            "normal_text": "#ef6b81",
            "hovered_text": "#ff8197",
            "normal_border": "#523947",
        }
    },
    "#save_button": {
        "colours": {
            "normal_bg": "#243042",
            "hovered_bg": "#2c3a50",
            "normal_text": "#9fb0ff",
            "normal_border": "#3a4a66",
        }
    },
    "#add_button": {
        "colours": {
            "normal_bg": "#243042",
            "hovered_bg": "#2c3a50",
            "normal_text": "#9fb0ff",
            "normal_border": "#3a4a66",
        }
    },
    "@remove_button": {
        "colours": {
            "normal_bg": "#222732",
            "hovered_bg": "#4a2630",
            "normal_text": "#cf8c98",
            "hovered_text": "#ff8197",
            "normal_border": "#39414f",
        },
        "misc": {"shape": "rounded_rectangle", "shape_corner_radius": "6"},
    },
}
