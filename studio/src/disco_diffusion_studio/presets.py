"""On-disk presets and the colour config, stored as TOML under the studio project dir.

Presets (full one-click recipes: guidance + per-run knobs + cut schedules + the model set) live
as ``studio/presets/*.toml``; the colour palette and recently-picked colours live in
``studio/config.toml``. Keeping them on disk means they can be hand-edited, version-controlled,
and saved from the UI. Reading uses the stdlib ``tomllib``; writing uses a tiny serialiser below
(our schema is just scalars, strings, and flat lists — no need for a third-party writer).
"""

from __future__ import annotations

import logging
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from disco_diffusion import RunConfig
from disco_diffusion.config import parse_schedule
from pydantic import BaseModel, ConfigDict, field_validator

log = logging.getLogger("disco_diffusion_studio.presets")

# This file is studio/src/disco_diffusion_studio/presets.py, so parents[2] is the studio project
# dir — the same place ``disco-studio`` is run from in the workspace. Presets and the colour
# config live there (alongside, not inside, the importable package).
_STUDIO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = _STUDIO_ROOT / "presets"
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


# --- preset models -----------------------------------------------------------


class PresetConfig(BaseModel):
    """The config half of a preset: guidance scales, eta/Perlin, and the cut schedules.

    Field names match ``RunConfig`` so a preset applies via ``setattr`` over ``model_dump()``.
    """

    model_config = ConfigDict(frozen=True)

    clip_guidance_scale: int
    tv_scale: float
    range_scale: float
    sat_scale: float
    clamp_max: float
    cutn_batches: int
    eta: float
    perlin_init: bool
    cut_overview: str
    cut_innercut: str
    cut_ic_pow: str
    cut_icgray_p: str

    @field_validator("cut_overview", "cut_innercut", "cut_ic_pow", "cut_icgray_p")
    @classmethod
    def _validate_schedule(cls, value: str) -> str:
        parse_schedule(value)  # raises on a malformed schedule string
        return value


# A preset applies via setattr over model_dump(), so every PresetConfig field must name a real
# RunConfig field — check at import so a typo fails fast instead of at apply time.
_UNKNOWN_PRESET_FIELDS = set(PresetConfig.model_fields) - set(RunConfig.model_fields)
if _UNKNOWN_PRESET_FIELDS:
    raise RuntimeError(f"PresetConfig fields not on RunConfig: {sorted(_UNKNOWN_PRESET_FIELDS)}")


class Preset(BaseModel):
    """A one-click full recipe: config knobs (applied now) + a model set (staged for Reload)."""

    model_config = ConfigDict(frozen=True)

    config: PresetConfig
    clip_models: list[str]
    use_secondary_model: bool


class ColourConfig(BaseModel):
    """The studio colour state: the fixed palette plus the recently-picked colours."""

    model_config = ConfigDict(frozen=True)

    palette: list[RGB]
    recent: list[RGB]


# --- tiny TOML writer (our schema only: scalars, strings, flat lists, one table level) --------


def _fmt_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fmt_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)  # repr round-trips floats (e.g. 150.0, 0.05)
    if isinstance(v, str):
        return _fmt_str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_value(x) for x in v) + "]"
    raise TypeError(f"unsupported TOML value: {v!r}")


def _dumps_toml(data: Mapping[str, Any]) -> str:
    """Serialise a flat dict (scalars/lists at top level, dicts as ``[table]`` sections)."""
    lines: list[str] = []
    tables: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {_fmt_value(value)}")
    for name, table in tables:
        if lines:
            lines.append("")
        lines.append(f"[{name}]")
        for key, value in table.items():
            lines.append(f"{key} = {_fmt_value(value)}")
    return "\n".join(lines) + "\n"


# --- presets: load / save ----------------------------------------------------


def _preset_from_data(data: dict) -> tuple[str, Preset]:
    name = str(data["name"])
    config = PresetConfig(**data["config"])
    models = data.get("models", {})
    preset = Preset(
        config=config,
        clip_models=list(models["clip_models"]),
        use_secondary_model=bool(models["use_secondary_model"]),
    )
    return name, preset


def load_presets() -> dict[str, Preset]:
    """Load every ``presets/*.toml``, keyed by display name. "Default" sorts first."""
    out: dict[str, Preset] = {}
    if PRESETS_DIR.is_dir():
        for path in sorted(PRESETS_DIR.glob("*.toml")):
            try:
                name, preset = _preset_from_data(tomllib.loads(path.read_text()))
                out[name] = preset
            except Exception:  # noqa: BLE001 - one bad file shouldn't sink the rest
                log.exception("failed to load preset %s", path)
    return dict(sorted(out.items(), key=lambda kv: (kv[0] != "Default", kv[0].lower())))


def _sanitize_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "preset"
    return re.sub(r"[^A-Za-z0-9 ._-]", "_", stem)


def save_preset(filename: str, preset: Preset) -> tuple[str, Path]:
    """Write ``preset`` to ``presets/<filename>.toml``; returns its (display name, path)."""
    stem = _sanitize_stem(filename)
    path = PRESETS_DIR / f"{stem}.toml"
    data = {
        "name": stem,
        "config": preset.config.model_dump(),
        "models": {
            "clip_models": list(preset.clip_models),
            "use_secondary_model": preset.use_secondary_model,
        },
    }
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(_dumps_toml(data))
    log.info("saved preset %s", path)
    return stem, path


# --- colours: load / save ----------------------------------------------------


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
    CONFIG_PATH.write_text(_dumps_toml(data))
