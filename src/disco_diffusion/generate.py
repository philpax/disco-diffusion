"""The Disco Diffusion generation pipeline (single still image).

A refactor of the notebook's ``do_run`` for the still-image case: it builds the
CLIP target embeddings, defines the guidance ``cond_fn``, and drives the
guided-diffusion DDIM/PLMS sampler. All animation/3D/video logic is gone.
"""

from __future__ import annotations

import gc
import io
import json
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .config import RunConfig, SamplingMode
from .cutouts import MakeCutouts, MakeCutoutsDango
from .losses import range_loss, spherical_dist_loss, tv_loss
from .models import (
    CLIP_NORMALIZE,
    build_model_config,
    load_clip_models,
    load_diffusion_model,
    load_lpips,
    load_secondary_model,
)
from .noise import regen_perlin
from .prompts import parse_prompt
from .secondary import alpha_sigma_to_t
from .tls import ensure_certifi_ssl
from .vendor import clip


def select_device(cpu: bool) -> torch.device:
    """Pick CUDA when available (any architecture, e.g. 3090 or 5090), else CPU."""
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _configure_perf(models_dir: Path) -> None:
    """Enable TF32 matmuls and a persistent on-disk Inductor cache.

    The persistent cache means the one-time ``torch.compile`` warmup is paid only on
    the first run; later runs reuse the compiled kernels.
    """
    torch.set_float32_matmul_precision("high")
    cache_dir = models_dir / ".inductor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir.resolve()))


def _fetch(url_or_path: str) -> io.BytesIO | Any:
    if str(url_or_path).startswith(("http://", "https://")):
        r = requests.get(url_or_path, timeout=60)
        r.raise_for_status()
        return io.BytesIO(r.content)
    return open(url_or_path, "rb")


class Generator:
    """Loads the models once and generates images for a :class:`RunConfig`."""

    def __init__(self, config: RunConfig, device: torch.device | None = None) -> None:
        ensure_certifi_ssl()
        self.config = config
        self.device = device or select_device(config.cpu)
        print(f"Using device: {self.device}")

        if config.compile and self.device.type == "cuda":
            _configure_perf(config.models_dir)

        self.model_config = build_model_config(config)
        self.model, self.diffusion = load_diffusion_model(config, self.model_config, self.device)
        self.secondary_model = (
            load_secondary_model(config, self.device) if config.use_secondary_model else None
        )
        self.clip_models = load_clip_models(config, self.device)
        # LPIPS is only used for the init-image loss; load it lazily to avoid the
        # VGG backbone download when it isn't needed.
        self.lpips_model = load_lpips(self.device) if config.init_image is not None else None

        if config.compile and self.device.type == "cuda":
            # Compile the UNet and the CLIP image encoders. The first run pays a
            # one-time warmup (cached on disk by _configure_perf); later runs are fast.
            print("torch.compile enabled (first run compiles kernels; subsequent runs reuse cache)")
            self.model = torch.compile(self.model)  # type: ignore[assignment]
            for clip_model in self.clip_models:
                clip_model.encode_image = torch.compile(clip_model.encode_image)

    # -- target embeddings ------------------------------------------------
    def _build_model_stats(self) -> list[dict[str, Any]]:
        cfg = self.config
        device = self.device
        cutn = 16
        model_stats: list[dict[str, Any]] = []
        for clip_model in self.clip_models:
            embeds: list[torch.Tensor] = []
            weights: list[float] = []

            for prompt in cfg.prompts:
                _, weight = parse_prompt(prompt)
                txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
                if cfg.fuzzy_prompt:
                    for _ in range(25):
                        embeds.append(
                            (txt + torch.randn(txt.shape, device=device) * cfg.rand_mag).clamp(0, 1)
                        )
                        weights.append(weight)
                else:
                    embeds.append(txt)
                    weights.append(weight)

            if cfg.image_prompts:
                make_cutouts = MakeCutouts(
                    clip_model.visual.input_resolution, cutn, skip_augs=cfg.skip_augs
                )
                for prompt in cfg.image_prompts:
                    path, weight = parse_prompt(prompt)
                    img = Image.open(_fetch(path)).convert("RGB")
                    img = TF.resize(
                        img, min(cfg.side_x, cfg.side_y, *img.size), T.InterpolationMode.LANCZOS
                    )
                    batch = make_cutouts(TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
                    embed = clip_model.encode_image(CLIP_NORMALIZE(batch)).float()
                    embeds.append(embed)
                    weights.extend([weight / cutn] * cutn)

            embeds_t = torch.cat(embeds)
            weights_t = torch.tensor(weights, device=device)
            if weights_t.sum().abs() < 1e-3:
                raise ValueError("The prompt weights must not sum to 0.")
            weights_t /= weights_t.sum().abs()
            model_stats.append(
                {"clip_model": clip_model, "target_embeds": embeds_t, "weights": weights_t}
            )
        return model_stats

    # -- guidance ---------------------------------------------------------
    def _make_cond_fn(
        self,
        model_stats: list[dict[str, Any]],
        init: torch.Tensor | None,
        cur_t: list[int],
    ) -> Callable[..., torch.Tensor]:
        cfg = self.config
        device = self.device
        diffusion = self.diffusion
        cut_overview = cfg.cut_overview_schedule()
        cut_innercut = cfg.cut_innercut_schedule()
        cut_ic_pow = cfg.cut_ic_pow_schedule()
        cut_icgray_p = cfg.cut_icgray_p_schedule()

        def cond_fn(x: torch.Tensor, t: torch.Tensor, y: Any = None) -> torch.Tensor:
            with torch.enable_grad():
                x_is_nan = False
                x = x.detach().requires_grad_()
                n = x.shape[0]
                if self.secondary_model is not None:
                    alpha = torch.tensor(
                        diffusion.sqrt_alphas_cumprod[cur_t[0]], device=device, dtype=torch.float32
                    )
                    sigma = torch.tensor(
                        diffusion.sqrt_one_minus_alphas_cumprod[cur_t[0]],
                        device=device,
                        dtype=torch.float32,
                    )
                    cosine_t = alpha_sigma_to_t(alpha, sigma)
                    sec_dtype = next(self.secondary_model.parameters()).dtype
                    out = self.secondary_model(
                        x.to(sec_dtype), cosine_t[None].repeat([n]).to(sec_dtype)
                    ).pred.float()
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t[0]]
                    x_in = out * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t[0]
                    out = diffusion.p_mean_variance(
                        self.model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
                    )
                    fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t[0]]
                    x_in = out["pred_xstart"] * fac + x * (1 - fac)
                    x_in_grad = torch.zeros_like(x_in)

                t_int = int(t.item()) + 1
                overview_n = int(cut_overview[1000 - t_int])
                inner_n = int(cut_innercut[1000 - t_int])
                n_cuts = overview_n + inner_n
                for model_stat in model_stats:
                    try:
                        input_resolution = model_stat["clip_model"].visual.input_resolution
                    except AttributeError:
                        input_resolution = 224
                    # Recompute per model so each model's grad call has its own graph
                    # (sharing this node across autograd.grad calls double-frees it).
                    x_in_unit = x_in.add(1).div(2)
                    # Draw all cutn_batches cutout sets (identical RNG sequence to the
                    # per-batch loop), then run a single batched CLIP encode + one grad
                    # call. The per-batch gradient mean reduces to a single mean over all
                    # cutn_batches*n_cuts cutouts, so this is numerically equivalent.
                    batches = []
                    for _ in range(cfg.cutn_batches):
                        cuts = MakeCutoutsDango(
                            input_resolution,
                            overview=overview_n,
                            inner_crop=inner_n,
                            ic_size_pow=cut_ic_pow[1000 - t_int],
                            ic_grey_p=cut_icgray_p[1000 - t_int],
                            skip_augs=cfg.skip_augs,
                        )
                        batches.append(cuts(x_in_unit))
                    clip_in = CLIP_NORMALIZE(torch.cat(batches))
                    image_embeds = model_stat["clip_model"].encode_image(clip_in).float()
                    dists = spherical_dist_loss(
                        image_embeds.unsqueeze(1), model_stat["target_embeds"].unsqueeze(0)
                    )
                    dists = dists.view([cfg.cutn_batches * n_cuts, n, -1])
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    x_in_grad += torch.autograd.grad(losses.sum() * cfg.clip_guidance_scale, x_in)[
                        0
                    ]

                tv_losses = tv_loss(x_in)
                range_losses = range_loss(
                    out if self.secondary_model is not None else out["pred_xstart"]
                )
                sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
                loss = (
                    tv_losses.sum() * cfg.tv_scale
                    + range_losses.sum() * cfg.range_scale
                    + sat_losses.sum() * cfg.sat_scale
                )
                if init is not None and cfg.init_scale and self.lpips_model is not None:
                    init_losses = self.lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * cfg.init_scale
                x_in_grad += torch.autograd.grad(loss, x_in)[0]
                if not torch.isnan(x_in_grad).any():
                    grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
                else:
                    x_is_nan = True
                    grad = torch.zeros_like(x)

            if cfg.clamp_grad and not x_is_nan:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=cfg.clamp_max) / magnitude
            return grad

        return cond_fn

    # -- run --------------------------------------------------------------
    def run(self) -> list[Path]:
        cfg = self.config
        device = self.device
        batch_dir = cfg.output_dir / cfg.batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_num = self._next_batch_num(batch_dir)

        seed = cfg.seed if cfg.seed is not None else random.randint(0, 2**32 - 1)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        print(f"Run {cfg.batch_name}({batch_num}), seed {seed}")

        model_stats = self._build_model_stats()

        init = None
        if cfg.init_image is not None:
            init_pil = Image.open(_fetch(cfg.init_image)).convert("RGB")
            init_pil = init_pil.resize((cfg.side_x, cfg.side_y), Image.Resampling.LANCZOS)
            init = TF.to_tensor(init_pil).to(device).unsqueeze(0).mul(2).sub(1)

        self._save_settings(batch_dir, batch_num, seed)

        if cfg.diffusion_sampling_mode == SamplingMode.ddim:
            sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            sample_fn = self.diffusion.plms_sample_loop_progressive

        outputs: list[Path] = []
        cur_t = [0]
        cond_fn = self._make_cond_fn(model_stats, init, cur_t)

        for i in range(cfg.n_batches):
            gc.collect()
            torch.cuda.empty_cache()
            cur_t[0] = self.diffusion.num_timesteps - cfg.skip_steps - 1
            total_steps = cur_t[0]

            if cfg.perlin_init:
                init = regen_perlin(
                    cfg.side_x, cfg.side_y, device, cfg.perlin_mode.value, batch_size=1
                )

            shape = (1, 3, cfg.side_y, cfg.side_x)
            if cfg.diffusion_sampling_mode == SamplingMode.ddim:
                samples = sample_fn(
                    self.model,
                    shape,
                    clip_denoised=cfg.clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=False,
                    skip_timesteps=cfg.skip_steps,
                    init_image=init,
                    randomize_class=cfg.randomize_class,
                    eta=cfg.eta,
                )
            else:
                samples = sample_fn(
                    self.model,
                    shape,
                    clip_denoised=cfg.clip_denoised,
                    model_kwargs={},
                    cond_fn=cond_fn,
                    progress=False,
                    skip_timesteps=cfg.skip_steps,
                    init_image=init,
                    randomize_class=cfg.randomize_class,
                    order=2,
                )

            sample = None
            for step in tqdm(samples, total=total_steps, desc=f"Batch {i + 1}/{cfg.n_batches}"):
                cur_t[0] -= 1
                sample = step

            assert sample is not None
            image_t = sample["pred_xstart"][0]
            image = TF.to_pil_image(image_t.add(1).div(2).clamp(0, 1))
            filename = f"{cfg.batch_name}({batch_num})_{i:04}.png"
            out_path = batch_dir / filename
            image.save(out_path)
            outputs.append(out_path)
            print(f"Saved {out_path}")

        return outputs

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _next_batch_num(batch_dir: Path) -> int:
        existing = list(batch_dir.glob("*_settings.json"))
        return len(existing)

    def _save_settings(self, batch_dir: Path, batch_num: int, seed: int) -> None:
        data = self.config.model_dump(mode="json")
        data["resolved_seed"] = seed
        path = batch_dir / f"{self.config.batch_name}({batch_num})_settings.json"
        path.write_text(json.dumps(data, indent=2))


def generate(config: RunConfig) -> list[Path]:
    """Convenience entry point: build a :class:`Generator` and run it."""
    return Generator(config).run()
