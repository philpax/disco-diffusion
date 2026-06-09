"""GenerationWorker behaviour against a stub sampler (no torch/models needed).

Locks in the two things we got wrong before: each paint stroke must inject + checkpoint on its
own step (the queue must not be wiped every loop iteration), and each checkpoint must snapshot
the live guidance values for revert.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from disco_diffusion_studio.worker import GenerationWorker, PromptSpec


class _StepResult:
    def __init__(self, index: int, total: int) -> None:
        self.index, self.total = index, total


class StubSampler:
    """The slice of the Sampler API the worker drives — steps slowly, records paint calls."""

    def __init__(self, total: int = 20) -> None:
        self.index = 0
        self.total = total
        self._last: object | None = None
        self.paints: list[float] = []

    @property
    def has_output(self) -> bool:
        return self._last is not None

    def set_conditioning(self, items) -> None:
        pass

    def __iter__(self):
        return self

    def __next__(self) -> _StepResult:
        if self.index >= self.total:
            raise StopIteration
        self.index += 1
        self._last = object()
        time.sleep(0.008)  # slow steps, so the test can interleave paint while it runs
        return _StepResult(self.index, self.total)

    def current_pil(self):
        return Image.new("RGB", (4, 4)) if self._last else None

    def state(self):
        return (object(), self.total - self.index) if self._last else None

    def paint(self, rgb, alpha, tint) -> None:
        self.paints.append(float(alpha.mean()))

    def close(self) -> None:
        pass


def _make_worker(cfg=None) -> GenerationWorker:
    cfg = cfg or SimpleNamespace(clip_guidance_scale=5000, tv_scale=0.0)
    session = SimpleNamespace(
        config=cfg,
        diffusion_for=lambda steps: SimpleNamespace(num_timesteps=20),
        sampler=lambda **kw: StubSampler(),
    )
    return GenerationWorker(
        session, width=64, height=64, steps=20, encode_cache={}, cache_lock=threading.Lock()
    )


def _paint_batch(value: float = 0.5):
    rgb = np.zeros((64, 64, 3), np.float32)
    alpha = np.full((64, 64), value, np.float32)
    tint = np.zeros((64, 64), np.float32)
    return rgb, alpha, tint


def test_each_stroke_is_its_own_checkpoint():
    worker = _make_worker()
    worker.start()
    time.sleep(0.05)  # let it produce output (has_output) before painting
    for i in range(3):
        worker.set_paint(*_paint_batch(), f"paint stroke {i}")
        time.sleep(0.01)
    time.sleep(0.35)
    worker.stop()
    worker.join(timeout=2)
    labels = [entry.label for entry in worker.get_history()]
    assert worker.paint_applied_count == 3  # would be 0 if the queue were wiped each iteration
    assert sum(label.startswith("paint") for label in labels) == 3


def test_checkpoints_snapshot_guidance_and_eta():
    cfg = SimpleNamespace(clip_guidance_scale=5000, tv_scale=0.0, eta=0.8)
    worker = _make_worker(cfg)
    worker.start()
    time.sleep(0.05)
    cfg.clip_guidance_scale = 12345  # "change guidance"
    cfg.eta = 0.3
    worker.checkpoint("guidance")
    time.sleep(0.1)
    worker.stop()
    worker.join(timeout=2)
    history = worker.get_history()
    start = next(e for e in history if e.label == "start")
    guidance = next(e for e in history if e.label == "guidance")
    assert start.config.clip_guidance_scale == 5000
    assert start.config.eta == 0.8
    assert guidance.config.clip_guidance_scale == 12345
    assert guidance.config.eta == 0.3  # eta is captured, so Revert can restore it


def test_muted_and_empty_prompts_excluded_from_conditioning():
    recorded = []

    class RecSampler(StubSampler):
        def set_conditioning(self, items):
            recorded.append(items)

    session = SimpleNamespace(
        config=SimpleNamespace(),
        diffusion_for=lambda steps: SimpleNamespace(num_timesteps=20),
        sampler=lambda **kw: RecSampler(),
        encode=lambda text: f"emb:{text}",
    )
    worker = GenerationWorker(
        session, width=64, height=64, steps=20, encode_cache={}, cache_lock=threading.Lock()
    )
    specs = [
        PromptSpec(text="a", weight=1.0, muted=False),
        PromptSpec(text="b", weight=0.5, muted=True),
        PromptSpec(text="", weight=1.0, muted=False),
    ]
    worker._start_sampler()
    worker.set_prompts(specs)
    worker._apply_pending()
    assert recorded[-1] == [("emb:a", 1.0)]  # only the active, non-muted, non-empty prompt
    # but all prompts are kept on _last_prompts, so a checkpoint/revert preserves the muted one
    assert worker._last_prompts == specs
