"""On-disk presets and the colour config, stored as TOML under the studio project dir.

Presets (full one-click recipes: guidance + per-run knobs + cut schedules + the model set) live
as ``studio/presets/*.toml``; the colour palette and recently-picked colours live in
``studio/config.toml``. Keeping them on disk means they can be hand-edited, version-controlled,
and saved from the UI. Reading uses the stdlib ``tomllib``; writing uses ``tomli_w`` (the standard
companion to ``tomllib``, which is read-only).
"""

from __future__ import annotations

import io
import logging
import re
import tomllib
import zipfile
from pathlib import Path
from typing import NamedTuple

import tomli_w
from disco_diffusion import RunConfig
from disco_diffusion.config import parse_schedule
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger("disco_diffusion_studio.presets")

# This file is studio/src/disco_diffusion_studio/presets.py, so parents[2] is the studio project
# dir — the same place ``disco-studio`` is run from in the workspace. Presets and the colour
# config live there (alongside, not inside, the importable package).
_STUDIO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = _STUDIO_ROOT / "presets"
CONFIG_PATH = _STUDIO_ROOT / "config.toml"

MAX_RECENT = 8  # how many recently-picked colours to remember

RGB = tuple[int, int, int]


class PromptSpec(NamedTuple):
    """One prompt as a typed, tuple-compatible triple.

    Lives here (torch-free) so both the session models and the worker share one prompt type; it
    unpacks as ``text, weight, muted`` and compares equal to a plain 3-tuple, while giving the
    UI↔worker wire format named, typed fields.
    """

    text: str
    weight: float
    muted: bool

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
    path.write_text(tomli_w.dumps(data))
    log.info("saved preset %s", path)
    return stem, path


# --- sessions: a full working state (superset of a preset) -------------------


class Session(BaseModel):
    """A whole working state, saved/loaded as a ``.zip`` via the native dialog.

    A superset of a :class:`Preset` (config + model set) plus the prompts and the output
    settings (size / steps / seed / denoise) — enough to reproduce or *resume* a piece of work.
    The zip also bundles the rendered result (``result.png``); loading sets it as the init image
    (with ``denoise``) so pressing Play continues from where you left off.
    """

    model_config = ConfigDict(frozen=True)

    width: int
    height: int
    steps: int
    seed: int
    denoise: int  # init-image denoise % to continue the bundled result with
    prompts: list[PromptSpec]
    config: PresetConfig
    clip_models: list[str]
    use_secondary_model: bool


_SESSION_TOML = "session.toml"
_SESSION_IMAGE = "result.png"


# The on-disk TOML schema, as pydantic models mirroring the [output]/[config]/[models] tables and
# the prompts array. Reading goes through ``model_validate``, so a malformed/old file raises a
# clear ValidationError instead of a KeyError/TypeError deep in manual dict access.
class _PromptDoc(BaseModel):
    text: str
    weight: float
    muted: bool = False


class _OutputDoc(BaseModel):
    width: int
    height: int
    steps: int
    seed: int
    denoise: int = 60


class _ModelsDoc(BaseModel):
    clip_models: list[str]
    use_secondary_model: bool


class _SessionDoc(BaseModel):
    prompts: list[_PromptDoc] = Field(default_factory=list)
    output: _OutputDoc
    config: PresetConfig
    models: _ModelsDoc

    @classmethod
    def from_session(cls, s: Session) -> _SessionDoc:
        return cls(
            prompts=[_PromptDoc(text=t, weight=w, muted=m) for t, w, m in s.prompts],
            output=_OutputDoc(
                width=s.width, height=s.height, steps=s.steps, seed=s.seed, denoise=s.denoise
            ),
            config=s.config,
            models=_ModelsDoc(
                clip_models=s.clip_models, use_secondary_model=s.use_secondary_model
            ),
        )

    def to_session(self) -> Session:
        return Session(
            width=self.output.width,
            height=self.output.height,
            steps=self.output.steps,
            seed=self.output.seed,
            denoise=self.output.denoise,
            prompts=[PromptSpec(p.text, p.weight, p.muted) for p in self.prompts],
            config=self.config,
            clip_models=self.models.clip_models,
            use_secondary_model=self.models.use_secondary_model,
        )


class GuidanceSnapshot(BaseModel):
    """The live-guidance values captured at a checkpoint, so Revert restores them exactly.

    Fully typed (not a loose ``dict[str, float]``) so int knobs — ``clip_guidance_scale`` and
    ``cutn_batches``, the latter used as ``range(cutn_batches)`` — stay ints through save/load.
    Fields mirror the RunConfig live-guidance knobs + eta; the import-time check below guards
    against drift, and the defaults match RunConfig's.
    """

    model_config = ConfigDict(frozen=True)

    clip_guidance_scale: int = 5000
    tv_scale: float = 0.0
    range_scale: float = 150.0
    sat_scale: float = 0.0
    cutn_batches: int = 4
    clamp_max: float = 0.05
    eta: float = 0.8

    @classmethod
    def capture(cls, config: RunConfig) -> GuidanceSnapshot:
        """Snapshot the guidance fields off a RunConfig (a field absent on ``config`` defaults)."""
        return cls(**{f: getattr(config, f) for f in cls.model_fields if hasattr(config, f)})

    def apply_to(self, config: RunConfig) -> None:
        """Write the snapshot back onto a RunConfig — types preserved, no float/int surprises."""
        for field_name, value in self.model_dump().items():
            setattr(config, field_name, value)


# Like PresetConfig, every GuidanceSnapshot field must name a real RunConfig field — checked at
# import so a rename/typo fails fast.
_UNKNOWN_SNAPSHOT_FIELDS = sorted(set(GuidanceSnapshot.model_fields) - set(RunConfig.model_fields))
if _UNKNOWN_SNAPSHOT_FIELDS:
    raise RuntimeError(f"GuidanceSnapshot fields not on RunConfig: {_UNKNOWN_SNAPSHOT_FIELDS}")



class HistoryItem(BaseModel):
    """One saved checkpoint's metadata; its preview image is stored alongside in the zip.

    Latents aren't saved (huge and, given DD's nondeterminism, not worth it) — so reloading
    restores the scrubbable timeline and lets you continue from a checkpoint via img2img on its
    preview, rather than a bit-exact resume.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    step: int
    index: int
    total: int
    prompts: list[PromptSpec] = Field(default_factory=list)
    config: GuidanceSnapshot = Field(default_factory=GuidanceSnapshot)


class _HistoryDoc(BaseModel):
    entries: list[HistoryItem] = Field(default_factory=list)


_SESSION_HISTORY_JSON = "history.json"


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _read_image(zf: zipfile.ZipFile, name: str) -> Image.Image:
    image = Image.open(io.BytesIO(zf.read(name))).convert("RGB")
    image.load()  # fully read before the zip closes
    return image


def save_session(
    path: str,
    session: Session,
    image: Image.Image | None = None,
    history: list[tuple[HistoryItem, Image.Image]] | None = None,
) -> Path:
    """Write a session ``.zip``: settings TOML + the rendered result + the checkpoint history.

    ``history`` is the edit-history checkpoints (metadata + preview); their previews are JPEGs
    under ``history/`` and the metadata is validated JSON, so reloading restores the timeline.
    """
    out = Path(path)
    if out.suffix.lower() != ".zip":
        out = out.with_suffix(".zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_SESSION_TOML, tomli_w.dumps(_SessionDoc.from_session(session).model_dump()))
        if image is not None:
            zf.writestr(_SESSION_IMAGE, _png_bytes(image))
        if history:
            doc = _HistoryDoc(entries=[item for item, _img in history])
            zf.writestr(_SESSION_HISTORY_JSON, doc.model_dump_json())
            for i, (_item, preview) in enumerate(history):
                zf.writestr(f"history/{i:03d}.jpg", _jpeg_bytes(preview))
    log.info("saved session %s", out)
    return out


def load_session(
    path: str,
) -> tuple[Session, Image.Image | None, list[tuple[HistoryItem, Image.Image]]]:
    """Read a session ``.zip``: settings, the bundled result image (or None), and the history.

    The TOML/JSON are validated through pydantic, so a malformed/old file raises a
    ``ValidationError`` (caught by the caller) rather than failing obscurely.
    """
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        session = _SessionDoc.model_validate(tomllib.loads(zf.read(_SESSION_TOML).decode()))
        image = _read_image(zf, _SESSION_IMAGE) if _SESSION_IMAGE in names else None
        history: list[tuple[HistoryItem, Image.Image]] = []
        if _SESSION_HISTORY_JSON in names:
            doc = _HistoryDoc.model_validate_json(zf.read(_SESSION_HISTORY_JSON))
            for i, item in enumerate(doc.entries):
                name = f"history/{i:03d}.jpg"
                if name in names:
                    history.append((item, _read_image(zf, name)))
    return session.to_session(), image, history


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
    CONFIG_PATH.write_text(tomli_w.dumps(data))
