"""The Disco Diffusion generation pipeline (single still image).

A refactor of the notebook's ``do_run`` for the still-image case. The model loading and
the guidance ``cond_fn`` now live in :mod:`disco_diffusion.session`; this module is the
batch-oriented entry point built on top of those primitives. It encodes the configured
prompts, sets them as a :class:`~disco_diffusion.session.Sampler`'s conditioning, and
drives the sampler to completion for each batch.
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import RunConfig
from .prompts import parse_prompt
from .session import DiscoSession, EncodedPrompt, Sampler, StepResult, fetch, select_device

__all__ = [
    "DiscoSession",
    "EncodedPrompt",
    "Generator",
    "Sampler",
    "StepResult",
    "fetch",
    "generate",
    "select_device",
]


class Generator:
    """Loads the models once and generates images for a :class:`RunConfig`.

    A thin batch-oriented wrapper around :class:`~disco_diffusion.session.DiscoSession`.
    """

    def __init__(self, config: RunConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.session = DiscoSession(config, device)
        self.device = self.session.device

    def _build_conditioning(self) -> list[tuple[EncodedPrompt, float]]:
        """Encode the configured text and image prompts into a weighted mix.

        Mirrors the notebook ordering (all text prompts, then all image prompts) so the
        concatenate-then-normalize in ``set_conditioning`` matches the original.
        """
        cfg = self.config
        items: list[tuple[EncodedPrompt, float]] = []
        for prompt in cfg.prompts:
            _, weight = parse_prompt(prompt)
            items.append((self.session.encode(prompt), weight))
        for prompt in cfg.image_prompts:
            path, weight = parse_prompt(prompt)
            items.append((self.session.encode_image(path), weight))
        return items

    def run(self) -> list[Path]:
        cfg = self.config
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

        # Encode prompts once after seeding (fuzzy_prompt jitter must consume RNG here,
        # matching the original ordering) and reuse the mix for every batch.
        conditioning = self._build_conditioning()

        self._save_settings(batch_dir, batch_num, seed)

        outputs: list[Path] = []
        for i in range(cfg.n_batches):
            gc.collect()
            torch.cuda.empty_cache()

            # seed=None keeps the RNG stream advancing across batches (a fresh seed would
            # make every batch identical); perlin overrides any init_image per batch.
            sampler = self.session.sampler(
                width=cfg.width,
                height=cfg.height,
                steps=cfg.steps,
                seed=None,
                init_image=None if cfg.perlin_init else cfg.init_image,
                perlin=cfg.perlin_init,
                skip_steps=cfg.skip_steps,
            )
            sampler.set_conditioning(conditioning)

            for _ in tqdm(sampler, total=sampler.total, desc=f"Batch {i + 1}/{cfg.n_batches}"):
                pass

            image = sampler.current_pil()
            assert image is not None
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
