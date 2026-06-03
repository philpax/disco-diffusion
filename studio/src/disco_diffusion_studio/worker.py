"""Background generation thread that drives a Disco Diffusion Sampler.

All torch/CUDA work — encoding prompts, stepping the sampler, converting frames — happens
on this one thread. The UI thread only reads the latest published :class:`Frame` and toggles
the pause / stop / pending-prompts flags.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from disco_diffusion import DiscoSession, EncodedPrompt

log = logging.getLogger("disco_diffusion_studio.worker")


@dataclass
class Frame:
    """A published preview frame plus its step counters."""

    image: np.ndarray  # (H, W, 3) uint8
    index: int
    total: int
    paint_applied: int = 0  # value of paint_applied_count when this frame was produced


class GenerationWorker(threading.Thread):
    """Drives one Sampler run on a background thread."""

    def __init__(
        self,
        session: DiscoSession,
        *,
        width: int,
        height: int,
        steps: int,
        encode_cache: dict[str, EncodedPrompt],
        cache_lock: threading.Lock,
    ) -> None:
        super().__init__(daemon=True)
        self._session = session
        self._width = width
        self._height = height
        self._steps = steps
        self._encode_cache = encode_cache
        self._cache_lock = cache_lock

        self._resume = threading.Event()
        self._resume.set()  # start running (not paused)
        # NB: not "_stop" — threading.Thread has an internal _stop() method, and shadowing
        # it breaks is_alive() once the thread finishes.
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._pending: list[tuple[str, float]] | None = None  # prompts awaiting (re)encode
        self._pending_paint: tuple[np.ndarray, np.ndarray] | None = None  # (rgb, alpha)
        self._frame: Frame | None = None
        self.finished = False
        self.paint_applied_count = 0  # bumps each time a paint batch is injected
        self.total = session.diffusion_for(steps).num_timesteps  # skip_steps=0 below

    # -- control (called from UI thread) --
    def set_prompts(self, prompts: list[tuple[str, float]]) -> None:
        with self._lock:
            self._pending = list(prompts)

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._resume.set()  # unblock if paused so the thread can exit

    def latest_frame(self) -> Frame | None:
        with self._lock:
            return self._frame

    def set_paint(self, rgb: np.ndarray, alpha: np.ndarray) -> None:
        with self._lock:
            self._pending_paint = (rgb, alpha)

    def has_pending_paint(self) -> bool:
        with self._lock:
            return self._pending_paint is not None

    # -- encoding (worker thread) --
    def _encode(self, text: str) -> EncodedPrompt:
        with self._cache_lock:
            cached = self._encode_cache.get(text)
        if cached is not None:
            return cached
        encoded = self._session.encode(text)
        with self._cache_lock:
            self._encode_cache[text] = encoded
        return encoded

    def _apply_pending(self, sampler: Any) -> None:
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return
        items = [(self._encode(t), w) for t, w in pending if t.strip()]
        sampler.set_conditioning(items)

    def _apply_pending_paint(self, sampler: Any) -> None:
        with self._lock:
            paint = self._pending_paint
            self._pending_paint = None
        if paint is None:
            return
        sampler.paint(paint[0], paint[1])  # injects into the current sample, in place
        self.paint_applied_count += 1

    # -- run loop --
    def run(self) -> None:
        # skip_steps=0 so any total-step count >= 1 is valid (no init image to skip toward).
        sampler = self._session.sampler(
            width=self._width, height=self._height, steps=self._steps, seed=None, skip_steps=0
        )
        self.total = sampler.total
        log.info("worker started: %dx%d, %d steps", self._width, self._height, sampler.total)

        while not self._stop_event.is_set():
            self._resume.wait()  # blocks while paused
            if self._stop_event.is_set():
                break
            self._apply_pending(sampler)  # re-mix before computing the step
            self._apply_pending_paint(sampler)  # inject painted pixels into the live latent
            try:
                step = next(sampler)
            except StopIteration:
                self.finished = True
                log.info("worker finished")
                break
            pil = sampler.current_pil()
            if pil is not None:
                with self._lock:
                    self._frame = Frame(
                        np.asarray(pil), step.index, step.total, self.paint_applied_count
                    )
