"""Model construction and loading.

Builds the guided-diffusion config for a checkpoint, loads the primary diffusion
UNet, the secondary model, the CLIP guidance models, and the LPIPS perceptual loss.
Ported from the original Disco Diffusion notebook.
"""

from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms as T
from torch import nn

from .config import DiffusionModel, RunConfig
from .download import ensure_model, model_filename
from .secondary import SecondaryDiffusionImageNet2
from .vendor import clip
from .vendor import lpips as lpips_pkg
from .vendor.guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

# CLIP normalization (mean/std) used for all guidance image embeddings.
CLIP_NORMALIZE = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711],
)


def build_model_config(config: RunConfig) -> dict[str, Any]:
    """Construct the guided-diffusion model/diffusion config for this run."""
    model_config = dict(model_and_diffusion_defaults())
    use_fp16 = not config.cpu
    # Gradient checkpointing only trades compute for memory during backprop, but the
    # guidance gradient is taken through the secondary model, not this UNet (and we
    # have ample VRAM). It also forces graph breaks under torch.compile, so disable it
    # whenever we compile.
    use_checkpoint = config.use_checkpoint and not config.compile

    if config.diffusion_model == DiffusionModel.finetune_512:
        model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": 250,
                "image_size": 512,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_checkpoint": use_checkpoint,
                "use_fp16": use_fp16,
                "use_scale_shift_norm": True,
            }
        )
    elif config.diffusion_model == DiffusionModel.uncond_256:
        model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "timestep_respacing": 250,
                "image_size": 256,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_checkpoint": use_checkpoint,
                "use_fp16": use_fp16,
                "use_scale_shift_norm": True,
            }
        )
    elif config.diffusion_model == DiffusionModel.portrait_512:
        model_config.update(
            {
                "attention_resolutions": "32, 16, 8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "rescale_timesteps": True,
                "image_size": 512,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 128,
                "num_heads": 4,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_checkpoint": use_checkpoint,
                "use_fp16": use_fp16,
                "use_scale_shift_norm": True,
            }
        )

    # Respacing for the requested step count (handled the same way as the notebook).
    steps = config.steps
    model_config.update(
        {
            "timestep_respacing": f"ddim{steps}",
            "diffusion_steps": (1000 // steps) * steps if steps < 1000 else steps,
        }
    )
    return model_config


def load_diffusion_model(
    config: RunConfig, model_config: dict[str, Any], device: torch.device
) -> tuple[nn.Module, Any]:
    """Load the primary diffusion UNet and its diffusion process."""
    path = ensure_model(config.diffusion_model.value, config.models_dir)
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.requires_grad_(False).eval().to(device)
    # Disco Diffusion leaves attention/projection params grad-enabled for guidance.
    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    return model, diffusion


def load_secondary_model(config: RunConfig, device: torch.device) -> SecondaryDiffusionImageNet2:
    path = ensure_model("secondary", config.models_dir)
    model = SecondaryDiffusionImageNet2()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    model.eval().requires_grad_(False).to(device)
    # The --fast-fp16-secondary lever: the secondary model only steers the CLIP guidance
    # (it doesn't produce the final pixels), so running it in fp16 is ~1.9x faster for a
    # small (~3dB) systematic departure. Off by default.
    if config.fast_fp16_secondary and device.type == "cuda" and not config.cpu:
        model.half()
    return model


def load_clip_models(config: RunConfig, device: torch.device) -> list[Any]:
    # Typed as Any: CLIP models expose encode_text/encode_image/visual which are not
    # part of the nn.Module interface.
    download_root = str(config.models_dir / "clip")
    models: list[Any] = []
    for name in config.clip_models:
        model = clip.load(name, jit=False, download_root=download_root)[0]
        models.append(model.eval().requires_grad_(False).to(device))
    return models


def load_lpips(device: torch.device) -> nn.Module:
    model: nn.Module = lpips_pkg.LPIPS(net="vgg")
    return model.to(device)


__all__ = [
    "CLIP_NORMALIZE",
    "build_model_config",
    "load_clip_models",
    "load_diffusion_model",
    "load_lpips",
    "load_secondary_model",
    "model_filename",
]
