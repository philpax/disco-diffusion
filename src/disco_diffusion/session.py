"""External-control API for driving the diffusion loop step by step.

Where :func:`disco_diffusion.generate.generate` runs a whole batch end-to-end with a
fixed prompt set, this module lets you take the sampling loop apart:

    session = DiscoSession(config)
    sky = session.encode("a clear blue sky")
    storm = session.encode("a violent thunderstorm")

    sampler = session.sampler(width=512, height=512, steps=100)
    for step in sampler:
        # mix / crossfade / swap encoded prompts however you like, per step
        w = step.index / step.total
        sampler.set_conditioning([(sky, 1 - w), (storm, w)])
    sampler.current_pil().save("out.png")

The three pieces:

* :class:`EncodedPrompt` — a prompt encoded once (per CLIP model), ready to mix cheaply.
* :class:`DiscoSession` — owns the loaded models; encodes prompts and builds samplers.
* :class:`Sampler` — a manual iterator over the guided-diffusion loop whose conditioning
  can be changed between any two steps.

``generate.Generator`` is built on top of these primitives, so the batch path and the
interactive path share exactly one copy of the model-loading and guidance code.
"""

from __future__ import annotations

import io
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from .config import RunConfig, SamplingMode
from .cutouts import MakeCutouts, MakeCutoutsDango
from .losses import range_loss, spherical_dist_loss, tv_loss
from .models import (
    CLIP_NORMALIZE,
    build_gaussian_diffusion,
    build_model_config,
    load_clip_models,
    load_diffusion_model,
    load_lpips,
    load_secondary_model,
)
from .noise import regen_perlin
from .secondary import alpha_sigma_to_t
from .tls import ensure_certifi_ssl
from .vendor import clip

# Number of cutouts used to encode an image prompt (matches the notebook / Generator).
_IMAGE_PROMPT_CUTN = 16


def select_device(cpu: bool) -> torch.device:
    """Pick CUDA when available (any architecture, e.g. 3090 or 5090).

    If CUDA is unavailable and ``cpu`` wasn't requested explicitly, warn and ask before
    falling back: a full run on CPU takes hours, so a silent fallback is almost always a
    mistake (an old or invisible driver, or — on NixOS — not being inside the nix-shell).
    """
    if not cpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    if not cpu:
        print(
            "\nWARNING: CUDA is not available, so this would run on CPU — a full run takes "
            "hours.\nUsually the GPU driver isn't visible (on NixOS, enter the nix-shell "
            "first) or is too old for the CUDA 12.8 wheels.",
            file=sys.stderr,
        )
        reply = input("Run on CPU anyway? [y/N] ").strip().lower() if sys.stdin.isatty() else "n"
        if reply not in ("y", "yes"):
            raise SystemExit("Aborted. Fix the GPU/driver, or pass --cpu to use CPU deliberately.")
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


def fetch(url_or_path: str) -> io.BytesIO | Any:
    """Open a local path or download a URL, returning a file-like object."""
    if str(url_or_path).startswith(("http://", "https://")):
        r = requests.get(url_or_path, timeout=60)
        r.raise_for_status()
        return io.BytesIO(r.content)
    return open(url_or_path, "rb")


@dataclass
class EncodedPrompt:
    """A single prompt encoded once into per-CLIP-model embeddings, ready to mix.

    ``embeds`` and ``base_weights`` are parallel to ``DiscoSession.clip_models``: one
    ``[K, D]`` embedding tensor and one ``[K]`` weight vector per model. ``K`` is 1 for a
    plain text prompt, 25 with ``fuzzy_prompt``, and ``_IMAGE_PROMPT_CUTN`` for an image
    prompt (whose ``base_weights`` are ``1/cutn`` so it matches the original split).
    """

    text: str
    embeds: list[torch.Tensor]
    base_weights: list[torch.Tensor]


@dataclass
class StepResult:
    """The outcome of advancing a :class:`Sampler` by one step."""

    index: int  # 1-based step number just completed
    total: int  # total number of steps in the run


class DiscoSession:
    """Loads the models once; encodes prompts and builds :class:`Sampler` objects.

    The model-loading here is shared verbatim with the batch path
    (:class:`disco_diffusion.generate.Generator` wraps a session).
    """

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
            #
            # The UNet forward is convolution-bound (~50% of its runtime is fp16
            # tensor-core GEMMs), so we autotune kernel selection with
            # "max-autotune-no-cudagraphs": it benchmarks Triton/CUTLASS conv+matmul
            # templates and picks the fastest, taking the UNet from ~128 ms to ~109 ms
            # (~14%) on a 1280x768 forward. The choice is lossless (still fp16 tensor
            # cores, within the existing noise floor); only the first-run compile is
            # slower (~90 s vs ~6 s), and that is cached on disk. CUDA graphs are skipped
            # because the forward is GPU-bound, not launch-bound (they gave no speedup).
            print("torch.compile enabled (first run autotunes kernels; later runs reuse cache)")
            self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")  # type: ignore[assignment]
            for clip_model in self.clip_models:
                clip_model.encode_image = torch.compile(clip_model.encode_image)

    # -- encoding ---------------------------------------------------------
    def encode(self, prompt: str) -> EncodedPrompt:
        """Encode a text prompt into per-CLIP-model embeddings.

        The raw ``prompt`` string is tokenized as-is (so a trailing ``:weight`` is encoded
        exactly as the batch path does — :meth:`Sampler.set_conditioning` applies the mix
        weight separately). With ``config.fuzzy_prompt`` the embedding is jittered 25×.
        """
        cfg = self.config
        device = self.device
        embeds: list[torch.Tensor] = []
        base_weights: list[torch.Tensor] = []
        for clip_model in self.clip_models:
            txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()
            if cfg.fuzzy_prompt:
                jittered = [
                    (txt + torch.randn(txt.shape, device=device) * cfg.rand_mag).clamp(0, 1)
                    for _ in range(25)
                ]
                emb = torch.cat(jittered)
            else:
                emb = txt
            embeds.append(emb)
            base_weights.append(torch.ones(emb.shape[0], device=device))
        return EncodedPrompt(text=prompt, embeds=embeds, base_weights=base_weights)

    def encode_image(self, path_or_image: str | Image.Image) -> EncodedPrompt:
        """Encode an image prompt into per-CLIP-model embeddings (via cutouts)."""
        cfg = self.config
        device = self.device
        cutn = _IMAGE_PROMPT_CUTN
        if isinstance(path_or_image, Image.Image):
            img = path_or_image.convert("RGB")
            label = "<image>"
        else:
            img = Image.open(fetch(path_or_image)).convert("RGB")
            label = path_or_image
        img = TF.resize(img, min(cfg.side_x, cfg.side_y, *img.size), T.InterpolationMode.LANCZOS)
        tensor = TF.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1)
        embeds: list[torch.Tensor] = []
        base_weights: list[torch.Tensor] = []
        for clip_model in self.clip_models:
            make_cutouts = MakeCutouts(
                clip_model.visual.input_resolution, cutn, skip_augs=cfg.skip_augs
            )
            batch = make_cutouts(tensor)
            emb = clip_model.encode_image(CLIP_NORMALIZE(batch)).float()
            embeds.append(emb)
            base_weights.append(torch.full((emb.shape[0],), 1.0 / cutn, device=device))
        return EncodedPrompt(text=label, embeds=embeds, base_weights=base_weights)

    # -- samplers ---------------------------------------------------------
    def sampler(
        self,
        *,
        width: int,
        height: int,
        steps: int,
        seed: int | None = None,
        init_image: str | Image.Image | torch.Tensor | None = None,
        skip_steps: int | None = None,
        perlin: bool = False,
        resume_latent: torch.Tensor | None = None,
        resume_step: int | None = None,
    ) -> Sampler:
        """Build a :class:`Sampler` for a fresh run at the given size/step count.

        Only the cheap diffusion schedule is rebuilt for ``steps``; the UNet and CLIP
        models are reused. ``init_image`` may be a path, a PIL image, or an already-built
        ``[1, 3, H, W]`` tensor in ``[-1, 1]``. Pass ``perlin=True`` for a Perlin-noise
        init (ignored when an ``init_image`` is given).

        ``resume_latent``/``resume_step`` (from :meth:`Sampler.state`) continue a run from a
        saved latent at that internal step instead of starting fresh — the loop picks up
        there and runs to the end. The latent is fed as pre-scaled noise so the vendor loop
        reconstructs it exactly. DDIM only (PLMS keeps history the snapshot doesn't carry).
        """
        side_x = (width // 64) * 64
        side_y = (height // 64) * 64
        if resume_latent is not None and resume_step is not None:
            diffusion = self.diffusion_for(steps)
            # Pre-scale so the loop's q_sample(zeros, t, noise) at resume_step yields the latent.
            soma = max(float(diffusion.sqrt_one_minus_alphas_cumprod[resume_step]), 1e-3)
            resume_noise = resume_latent.to(self.device, torch.float32) / soma
            return Sampler(
                self,
                width=side_x,
                height=side_y,
                steps=steps,
                seed=seed,
                init=None,
                skip_steps=0,
                resume_noise=resume_noise,
                resume_step=resume_step,
            )
        init = self._build_init(side_x, side_y, init_image, perlin)
        return Sampler(
            self,
            width=side_x,
            height=side_y,
            steps=steps,
            seed=seed,
            init=init,
            skip_steps=self.config.skip_steps if skip_steps is None else skip_steps,
        )

    def _build_init(
        self,
        side_x: int,
        side_y: int,
        init_image: str | Image.Image | torch.Tensor | None,
        perlin: bool,
    ) -> torch.Tensor | None:
        if init_image is not None:
            if isinstance(init_image, torch.Tensor):
                return init_image.to(self.device)
            if isinstance(init_image, Image.Image):
                pil = init_image
            else:
                pil = Image.open(fetch(init_image))
            pil = pil.convert("RGB").resize((side_x, side_y), Image.Resampling.LANCZOS)
            return TF.to_tensor(pil).to(self.device).unsqueeze(0).mul(2).sub(1)
        if perlin:
            return regen_perlin(
                side_x, side_y, self.device, self.config.perlin_mode.value, batch_size=1
            )
        return None

    def diffusion_for(self, steps: int) -> Any:
        """The diffusion schedule for ``steps`` (reuses the loaded one when unchanged)."""
        if steps == self.config.steps:
            return self.diffusion
        return build_gaussian_diffusion(self.config, steps)


class Sampler:
    """A manual iterator over the guided-diffusion loop with mutable conditioning.

    Construct via :meth:`DiscoSession.sampler`. Call :meth:`set_conditioning` to choose
    the active prompt mix (you may change it between any two steps), then iterate — each
    ``next()`` advances exactly one diffusion step. The guidance ``cond_fn`` reads the
    current conditioning every step, so live changes take effect immediately.
    """

    def __init__(
        self,
        session: DiscoSession,
        *,
        width: int,
        height: int,
        steps: int,
        seed: int | None,
        init: torch.Tensor | None,
        skip_steps: int,
        resume_noise: torch.Tensor | None = None,
        resume_step: int | None = None,
    ) -> None:
        self.session = session
        self.config = session.config
        self.device = session.device
        self.width = width
        self.height = height
        self.steps = steps
        self.diffusion = session.diffusion_for(steps)
        num = self.diffusion.num_timesteps
        # Resuming: start the loop at resume_step fed the (pre-scaled) saved latent; otherwise
        # a normal fresh run honouring skip_steps.
        self._resume_noise = resume_noise
        self.skip_steps = (num - 1 - resume_step) if resume_step is not None else skip_steps
        self._init = None if resume_noise is not None else init

        # Reseeding here would make every batch identical, so it is opt-in: the batch path
        # seeds once and creates each Sampler with seed=None to keep advancing the stream.
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        cfg = self.config
        self._cut_overview = cfg.cut_overview_schedule()
        self._cut_innercut = cfg.cut_innercut_schedule()
        self._cut_ic_pow = cfg.cut_ic_pow_schedule()
        self._cut_icgray_p = cfg.cut_icgray_p_schedule()

        # Mutable conditioning, as the existing [{clip_model, target_embeds, weights}]
        # shape consumed by cond_fn. Empty list => no guidance (zero gradient).
        self._model_stats: list[dict[str, Any]] = []
        # Lazy guidance cache (see RunConfig.guidance_every).
        self._guidance_cache: dict[str, Any] = {"grad": None, "calls": 0}

        # cur_t mirrors the batch path: it starts at num_timesteps - skip - 1 and is
        # decremented *after* each yielded step, so cond_fn sees the right index.
        self._cur_t = [num - self.skip_steps - 1]
        if resume_step is not None:
            # Keep the displayed progress continuous with the original run.
            self.total = num
            self.index = num - 1 - resume_step
        else:
            self.total = num - self.skip_steps
            self.index = 0
        self.done = False
        self._last: dict[str, Any] | None = None
        self._gen: Any = None

    # -- conditioning -----------------------------------------------------
    def set_conditioning(self, items: list[tuple[EncodedPrompt, float]]) -> None:
        """Set the active prompt mix to ``[(encoded_prompt, weight), ...]``.

        Reproduces the batch path's per-model concatenate-then-normalize. If the weights
        sum to ~0 (e.g. every slider at zero), guidance is disabled for the next step(s)
        rather than dividing by zero.
        """
        if not items:
            self._model_stats = []
            return
        model_stats: list[dict[str, Any]] = []
        zero = False
        for i, clip_model in enumerate(self.session.clip_models):
            embeds = torch.cat([p.embeds[i] for p, _ in items])
            weights = torch.cat([p.base_weights[i] * w for p, w in items])
            total = weights.sum().abs()
            if total < 1e-3:
                zero = True
                break
            model_stats.append(
                {
                    "clip_model": clip_model,
                    "target_embeds": embeds,
                    "weights": weights / total,
                }
            )
        self._model_stats = [] if zero else model_stats

    # -- iteration --------------------------------------------------------
    def __iter__(self) -> Sampler:
        return self

    def __next__(self) -> StepResult:
        if self.done:
            raise StopIteration
        if self._gen is None:
            self._gen = self._make_generator()
        try:
            step = next(self._gen)
        except StopIteration:
            self.done = True
            raise
        self._cur_t[0] -= 1
        self.index += 1
        self._last = step
        return StepResult(index=self.index, total=self.total)

    def current_pil(self) -> Image.Image | None:
        """The current denoised prediction as a PIL image (``None`` before the first step)."""
        if self._last is None:
            return None
        img_t = self._last["pred_xstart"][0].add(1).div(2).clamp(0, 1).detach().cpu()
        return TF.to_pil_image(img_t)

    @property
    def has_output(self) -> bool:
        """True once a step has produced a sample — required before ``paint``/``state``."""
        return self._last is not None

    def close(self) -> None:
        """Close the underlying loop generator so its grad-mode context unwinds now.

        The vendor loop suspends inside ``with torch.no_grad():`` at each ``yield``. If an
        abandoned sampler (e.g. after a revert) is left for the GC, that context can exit on
        the worker thread mid-``cond_fn`` and corrupt the thread-local autograd state, so we
        close it deterministically instead.
        """
        if self._gen is not None:
            self._gen.close()
            self._gen = None
        self.done = True

    def state(self) -> tuple[torch.Tensor, int] | None:
        """Snapshot ``(cpu latent, internal step)`` to resume from later (``None`` pre-first-step).

        The latent is the input to the *next* step; pass it back via
        ``DiscoSession.sampler(resume_latent=..., resume_step=...)`` to continue from here.
        """
        if self._last is None:
            return None
        return self._last["sample"].detach().cpu().clone(), int(self._cur_t[0])

    def paint(
        self, rgb01: np.ndarray, alpha01: np.ndarray, tint01: np.ndarray | None = None
    ) -> None:
        """Blend painted pixels into the live sample so the next step incorporates them.

        ``rgb01`` is an ``(H, W, 3)`` image in ``[0, 1]`` and ``alpha01`` an ``(H, W)`` mask
        in ``[0, 1]``, both at the sampler's resolution. The painted RGB is noised to the
        current timestep (DD works in pixel space, so paint maps 1:1 onto the latent) and
        alpha-blended into ``self._last["sample"]`` **in place** — the vendor loop reads that
        same tensor as the next step's input (``img = out["sample"]``), so the paint sticks
        and is then evolved by the diffusion + CLIP guidance. No-op before the first step.

        ``tint01`` (optional, ``(H, W)`` in ``[0, 1]``) selects a *fresh-noise* mode: the
        region is replaced with brand-new noise at the current level (``sqrt(1-ᾱ)·randn``, so
        the model invents new structure there rather than reworking the existing content) with
        the painted colour as that structure's target. A plain colour stroke imposes the
        colour only at its physical weight ``sqrt(ᾱ)``, which is ~0 early and gets washed out;
        ``tint`` boosts that weight toward full strength so the new structure actually takes
        the colour. tint=0 is the physical renoise; tint=1 imposes the colour fully.
        """
        if self._last is None:
            return
        x = self._last["sample"]
        paint = torch.from_numpy(rgb01).to(x.device, x.dtype).permute(2, 0, 1).unsqueeze(0)
        paint = paint.mul(2).sub(1)  # [0,1] -> [-1,1]
        mask = torch.from_numpy(alpha01).to(x.device, x.dtype).unsqueeze(0).unsqueeze(0)
        tt = max(0, int(self._cur_t[0]))
        sa = float(self.diffusion.sqrt_alphas_cumprod[tt])
        soma = float(self.diffusion.sqrt_one_minus_alphas_cumprod[tt])
        # Fresh current-level noise (new structure) with the colour as its clean target.
        noised = sa * paint + soma * torch.randn_like(paint)
        if tint01 is not None:
            tint = torch.from_numpy(tint01).to(x.device, x.dtype).unsqueeze(0).unsqueeze(0)
            noised = noised + tint * (1.0 - sa) * paint  # boost colour weight: sqrt(ᾱ) -> 1
        x.mul_(1 - mask).add_(noised * mask)

    def _make_generator(self) -> Any:
        cfg = self.config
        shape = (1, 3, self.height, self.width)
        # When resuming, the saved (pre-scaled) latent is fed as the loop's noise.
        noise = self._resume_noise
        if cfg.diffusion_sampling_mode == SamplingMode.ddim:
            # With no secondary model the guidance has to run the full UNet to get
            # pred_xstart. The no-grad loop would then run the UNet *twice* per step
            # (once for the step, once in cond_fn), so take the grad-enabled DDIM path
            # instead: it computes the forward once, with grad, and hands it to cond_fn
            # to reuse. With a secondary model the cheap no-grad path is already optimal
            # (guidance uses the surrogate, not the UNet), so leave it alone.
            reuse_forward = self.session.secondary_model is None
            return self.diffusion.ddim_sample_loop_progressive(
                self.session.model,
                shape,
                noise=noise,
                clip_denoised=cfg.clip_denoised,
                model_kwargs={},
                cond_fn=self._cond_fn,
                progress=False,
                skip_timesteps=self.skip_steps,
                init_image=self._init,
                randomize_class=cfg.randomize_class,
                eta=cfg.eta,
                cond_fn_with_grad=reuse_forward,
            )
        return self.diffusion.plms_sample_loop_progressive(
            self.session.model,
            shape,
            noise=noise,
            clip_denoised=cfg.clip_denoised,
            model_kwargs={},
            cond_fn=self._cond_fn,
            progress=False,
            skip_timesteps=self.skip_steps,
            init_image=self._init,
            randomize_class=cfg.randomize_class,
            order=2,
        )

    # -- guidance ---------------------------------------------------------
    def _cond_fn(
        self, x: torch.Tensor, t: torch.Tensor, precomputed_out: Any = None
    ) -> torch.Tensor:
        # ``precomputed_out`` is set only on the grad-enabled DDIM path
        # (``condition_score_with_grad`` passes the model's forward output positionally):
        # there ``x`` already requires grad and the UNet forward is done, so we reuse it
        # instead of running the model again. On the no-grad path it stays None.
        # No active conditioning (no prompts, or all-zero weights) => no guidance.
        if not self._model_stats:
            return torch.zeros_like(x)

        cfg = self.config
        device = self.device
        diffusion = self.diffusion
        model_stats = self._model_stats
        cur_t = self._cur_t
        init = self._init
        secondary_model = self.session.secondary_model
        cut_overview = self._cut_overview
        cut_innercut = self._cut_innercut
        cut_ic_pow = self._cut_ic_pow
        cut_icgray_p = self._cut_icgray_p
        guidance_cache = self._guidance_cache

        calls = guidance_cache["calls"]
        guidance_cache["calls"] = calls + 1
        if (
            cfg.guidance_every > 1
            and guidance_cache["grad"] is not None
            and calls % cfg.guidance_every != 0
        ):
            return guidance_cache["grad"]
        with torch.enable_grad():
            x_is_nan = False
            if precomputed_out is None:
                x = x.detach().requires_grad_()  # build our own graph for the guidance
            else:
                # x already requires grad (from ddim_sample_with_grad); re-detaching here
                # would sever precomputed_out from it. The cut schedule indexes by the
                # rescaled timestep, which the no-grad path receives pre-scaled — match it.
                t = diffusion._scale_timesteps(t)
            n = x.shape[0]
            if secondary_model is not None:
                alpha = torch.tensor(
                    diffusion.sqrt_alphas_cumprod[cur_t[0]], device=device, dtype=torch.float32
                )
                sigma = torch.tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod[cur_t[0]],
                    device=device,
                    dtype=torch.float32,
                )
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                sec_dtype = next(secondary_model.parameters()).dtype
                out = secondary_model(
                    x.to(sec_dtype), cosine_t[None].repeat([n]).to(sec_dtype)
                ).pred.float()
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t[0]]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            else:
                if precomputed_out is not None:
                    out = precomputed_out  # reuse the loop's grad-enabled forward
                else:
                    my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t[0]
                    out = diffusion.p_mean_variance(
                        self.session.model, x, my_t, clip_denoised=False, model_kwargs={}
                    )
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t[0]]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)

            # Defensive: if the autograd graph wasn't built (grad mode disturbed), skip
            # guidance this step rather than crashing the worker thread.
            if not x_in.requires_grad:
                return torch.zeros_like(x)

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
                x_in_grad += torch.autograd.grad(losses.sum() * cfg.clip_guidance_scale, x_in)[0]

            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out if secondary_model is not None else out["pred_xstart"])
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (
                tv_losses.sum() * cfg.tv_scale
                + range_losses.sum() * cfg.range_scale
                + sat_losses.sum() * cfg.sat_scale
            )
            if init is not None and cfg.init_scale and self.session.lpips_model is not None:
                init_losses = self.session.lpips_model(x_in, init)
                loss = loss + init_losses.sum() * cfg.init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_nan = True
                grad = torch.zeros_like(x)

        if cfg.clamp_grad and not x_is_nan:
            magnitude = grad.square().mean().sqrt()
            result = grad * magnitude.clamp(max=cfg.clamp_max) / magnitude
        else:
            result = grad
        guidance_cache["grad"] = result
        return result
