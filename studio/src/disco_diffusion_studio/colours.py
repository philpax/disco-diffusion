"""The studio colour config — the fixed palette + recently-picked colours — stored as TOML.

Lives in ``studio/config.toml`` (alongside, not inside, the package) so it can be hand-edited,
version-controlled, and saved from the UI. Reading uses the stdlib ``tomllib``; writing uses
``tomli_w``. Kept apart from presets.py so the paint subsystem doesn't depend on the recipe/
session persistence — only on these colours.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel, ConfigDict

log = logging.getLogger("disco_diffusion_studio.colours")

# This file is studio/src/disco_diffusion_studio/colours.py, so parents[2] is the studio project
# dir — the same place ``disco-studio`` is run from. The colour config lives there.
_STUDIO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = _STUDIO_ROOT / "config.toml"

MAX_RECENT = 8  # how many recently-picked colours to remember

RGB = tuple[int, int, int]

# Seed palette, used only if config.toml is missing/empty (a friendly default spread).
DEFAULT_PALETTE: list[str] = [
    "#000000",
    "#ffffff",
    "#7f7f7f",
    "#d64541",
    "#e8833a",
    "#f1c40f",
    "#2ecc71",
    "#3498db",
    "#6c7cff",
    "#9b59b6",
    "#ec70b4",
    "#f5deb3",
]


class ColourConfig(BaseModel):
    """The studio colour state: the fixed palette plus the recently-picked colours."""

    model_config = ConfigDict(frozen=True)

    palette: list[RGB]
    recent: list[RGB]


def _hex_to_rgb(value: str) -> RGB:
    s = value.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _rgb_to_hex(rgb: RGB) -> str:
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def load_colours() -> ColourConfig:
    """Load the palette + recents from config.toml, falling back to the seed palette."""
    if CONFIG_PATH.exists():
        try:
            data = tomllib.loads(CONFIG_PATH.read_text()).get("colours", {})
            palette = [_hex_to_rgb(h) for h in data.get("palette", [])]
            recent = [_hex_to_rgb(h) for h in data.get("recent", [])]
            if palette:
                return ColourConfig(palette=palette, recent=recent)
        except Exception:  # noqa: BLE001 - a malformed config shouldn't break startup
            log.exception("failed to load colour config %s", CONFIG_PATH)
    return ColourConfig(palette=[_hex_to_rgb(h) for h in DEFAULT_PALETTE], recent=[])


def save_colours(colours: ColourConfig) -> None:
    """Persist the palette + recents to config.toml as hex strings."""
    data = {
        "colours": {
            "palette": [_rgb_to_hex(c) for c in colours.palette],
            "recent": [_rgb_to_hex(c) for c in colours.recent],
        }
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(tomli_w.dumps(data))
