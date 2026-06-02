"""Typed configuration for a Disco Diffusion run.

A single ``RunConfig`` replaces the original notebook's sprawling ``args`` namespace.
Defaults faithfully reproduce the canonical "lighthouse" image.
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# CLIP models available from the vendored OpenAI CLIP.
AVAILABLE_CLIP_MODELS = (
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
    "ViT-L/14@336px",
)

DEFAULT_PROMPTS = [
    "A beautiful painting of a singular lighthouse, shining its light across a "
    "tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.",
    "yellow color scheme",
]


class DiffusionModel(StrEnum):
    """Supported primary diffusion checkpoints (see ``download.py`` for URIs)."""

    finetune_512 = "512x512_diffusion_uncond_finetune_008100"
    uncond_256 = "256x256_diffusion_uncond"
    portrait_512 = "portrait_generator_v001"


class SamplingMode(StrEnum):
    ddim = "ddim"
    plms = "plms"


class PerlinMode(StrEnum):
    mixed = "mixed"
    color = "color"
    gray = "gray"


def parse_schedule(expr: str) -> list[float]:
    """Safely parse a cutout schedule like ``"[12]*400+[4]*600"`` into a per-step list.

    Replaces the original notebook's use of ``eval()`` on these strings.
    """
    result: list[float] = []
    for raw in expr.split("+"):
        term = raw.strip()
        repeated = re.fullmatch(r"\[(-?\d+(?:\.\d+)?)\]\s*\*\s*(\d+)", term)
        single = re.fullmatch(r"\[(-?\d+(?:\.\d+)?)\]", term)
        if repeated:
            result.extend([float(repeated.group(1))] * int(repeated.group(2)))
        elif single:
            result.append(float(single.group(1)))
        else:
            raise ValueError(f"Invalid cut-schedule term: {term!r} in {expr!r}")
    return result


class RunConfig(BaseModel):
    """All settings for a single still-image generation run."""

    model_config = {"extra": "forbid"}

    # Prompts
    prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_PROMPTS))
    image_prompts: list[str] = Field(default_factory=list)

    # Output / batching
    batch_name: str = "TimeToDisco"
    n_batches: int = 1
    seed: int | None = None
    output_dir: Path = Path("images_out")
    models_dir: Path = Path("models")

    # Models
    diffusion_model: DiffusionModel = DiffusionModel.finetune_512
    use_secondary_model: bool = True
    diffusion_sampling_mode: SamplingMode = SamplingMode.ddim
    use_checkpoint: bool = True
    clip_models: list[str] = Field(default_factory=lambda: ["ViT-B/32", "ViT-B/16", "RN50"])

    # Basic generation settings
    steps: int = 250
    width: int = 1280
    height: int = 768
    clip_guidance_scale: int = 5000
    tv_scale: float = 0.0
    range_scale: float = 150.0
    sat_scale: float = 0.0
    cutn_batches: int = 4
    skip_augs: bool = False

    # Init image
    init_image: str | None = None
    init_scale: int = 1000
    skip_steps: int = 10

    # Cutout schedules (per 1000 internal steps)
    cut_overview: str = "[12]*400+[4]*600"
    cut_innercut: str = "[4]*400+[12]*600"
    cut_ic_pow: str = "[1]*1000"
    cut_icgray_p: str = "[0.2]*400+[0]*600"

    # Advanced
    eta: float = 0.8
    clamp_grad: bool = True
    clamp_max: float = 0.05
    randomize_class: bool = True
    clip_denoised: bool = False
    fuzzy_prompt: bool = False
    rand_mag: float = 0.05
    perlin_init: bool = False
    perlin_mode: PerlinMode = PerlinMode.mixed

    # Runtime
    cpu: bool = False
    # torch.compile the UNet and CLIP image encoders. ~2x faster per step once
    # warm; the first run pays a one-time compile cost that is then cached on disk
    # (so subsequent runs are fast). Disable for a faster cold start / debugging.
    compile: bool = True

    @field_validator("clip_models")
    @classmethod
    def _validate_clip_models(cls, value: list[str]) -> list[str]:
        unknown = [m for m in value if m not in AVAILABLE_CLIP_MODELS]
        if unknown:
            raise ValueError(
                f"Unknown CLIP model(s): {unknown}. Available: {list(AVAILABLE_CLIP_MODELS)}"
            )
        if not value:
            raise ValueError("At least one CLIP model is required.")
        return value

    @field_validator("cut_overview", "cut_innercut", "cut_ic_pow", "cut_icgray_p")
    @classmethod
    def _validate_schedule(cls, value: str) -> str:
        parse_schedule(value)  # raises on malformed input
        return value

    @property
    def side_x(self) -> int:
        """Width snapped down to a multiple of 64 (required by the UNet)."""
        return (self.width // 64) * 64

    @property
    def side_y(self) -> int:
        """Height snapped down to a multiple of 64 (required by the UNet)."""
        return (self.height // 64) * 64

    @property
    def is_256_model(self) -> bool:
        return self.diffusion_model == DiffusionModel.uncond_256

    def cut_overview_schedule(self) -> list[float]:
        return parse_schedule(self.cut_overview)

    def cut_innercut_schedule(self) -> list[float]:
        return parse_schedule(self.cut_innercut)

    def cut_ic_pow_schedule(self) -> list[float]:
        return parse_schedule(self.cut_ic_pow)

    def cut_icgray_p_schedule(self) -> list[float]:
        return parse_schedule(self.cut_icgray_p)
