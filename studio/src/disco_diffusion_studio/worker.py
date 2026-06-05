"""Background generation thread that drives a Disco Diffusion Sampler.

All torch/CUDA work — encoding prompts, stepping the sampler, converting frames — happens
on this one thread. The UI thread only reads the latest published :class:`Frame`, the edit
history, and toggles the pause / stop / pending flags.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from disco_diffusion import DiscoSession, EncodedPrompt

log = logging.getLogger("disco_diffusion_studio.worker")

MAX_HISTORY = 60  # cap on edit-history checkpoints (each holds a CPU latent)


@dataclass
class Frame:
    """A published preview frame plus its step counters."""

    image: np.ndarray  # (H, W, 3) uint8
    index: int
    total: int
    paint_applied: int = 0  # value of paint_applied_count when this frame was produced


@dataclass
class HistoryEntry:
    """A revertible checkpoint: the latent to resume from, plus a preview + label."""

    latent: Any  # CPU torch tensor (the sampler's latent at `step`)
    step: int  # internal diffusion step the latent belongs to
    index: int  # display step index when captured
    total: int
    preview: np.ndarray  # (H, W, 3) uint8 — the image at this checkpoint
    label: str
    prompts: list[tuple[str, float]] = field(default_factory=list)  # (text, weight) at capture


class GenerationWorker(threading.Thread):
    """Drives a Sampler on a background thread, with a revertible edit history."""

    def __init__(
        self,
        session: DiscoSession,
        *,
        width: int,
        height: int,
        steps: int,
        encode_cache: dict[str, EncodedPrompt],
        cache_lock: threading.Lock,
        perlin: bool = False,
    ) -> None:
        super().__init__(daemon=True)
        self._session = session
        self._width = width
        self._height = height
        self._steps = steps
        self._encode_cache = encode_cache
        self._cache_lock = cache_lock
        self._perlin = perlin  # seed the fresh run from Perlin noise instead of flat gaussian

        self._resume = threading.Event()
        self._resume.set()  # start running (not paused)
        # NB: not "_stop" — threading.Thread has an internal _stop() method, and shadowing
        # it breaks is_alive() once the thread finishes.
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._pending: list[tuple[str, float]] | None = None  # prompts awaiting (re)encode
        self._pending_paint: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._pending_seek: int | None = None  # history index to revert to
        self._checkpoint_label: str | None = None  # request a checkpoint before the next step
        self._frame: Frame | None = None
        self._sampler: Any = None
        self._last_items: list[tuple[EncodedPrompt, float]] = []  # conditioning to re-apply
        self._last_prompts: list[tuple[str, float]] = []  # (text, weight) shown in the UI
        self.history: list[HistoryEntry] = []
        self._started_history = False
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
        self._resume.set()

    def latest_frame(self) -> Frame | None:
        with self._lock:
            return self._frame

    def set_paint(self, rgb: np.ndarray, alpha: np.ndarray, tint: np.ndarray) -> None:
        with self._lock:
            self._pending_paint = (rgb, alpha, tint)

    def has_pending_paint(self) -> bool:
        with self._lock:
            return self._pending_paint is not None

    def checkpoint(self, label: str) -> None:
        """Request an edit checkpoint (captured before the next step)."""
        with self._lock:
            self._checkpoint_label = label

    def seek(self, index: int) -> None:
        """Revert to history[index] — resume the run from that checkpoint's latent."""
        with self._lock:
            self._pending_seek = index

    def get_history(self) -> list[HistoryEntry]:
        with self._lock:
            return list(self.history)

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

    def _apply_pending(self) -> None:
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return
        self._last_prompts = list(pending)
        self._last_items = [(self._encode(t), w) for t, w in pending if t.strip()]
        self._sampler.set_conditioning(self._last_items)

    def _apply_pending_paint(self) -> None:
        # Needs a stepped latent to inject into; just after a revert there isn't one yet, so
        # keep the paint pending (don't drop it) until the resumed sampler has produced a step.
        if not self._sampler.has_output:
            return
        with self._lock:
            paint = self._pending_paint
            self._pending_paint = None
        if paint is None:
            return
        self._sampler.paint(paint[0], paint[1], paint[2])  # injects into the sample, in place
        self.paint_applied_count += 1

    def _add_checkpoint(self, label: str) -> None:
        state = self._sampler.state()
        pil = self._sampler.current_pil()
        if state is None or pil is None:
            return
        latent, step = state
        entry = HistoryEntry(
            latent=latent,
            step=step,
            index=self._sampler.index,
            total=self._sampler.total,
            preview=np.asarray(pil),
            label=label,
            prompts=list(self._last_prompts),
        )
        with self._lock:
            self.history.append(entry)
            if len(self.history) > MAX_HISTORY:
                self.history.pop(0)

    def _process_seek(self) -> None:
        """If a revert is pending, rebuild the sampler resuming from that checkpoint."""
        with self._lock:
            index = self._pending_seek
            self._pending_seek = None
        if index is None or not (0 <= index < len(self.history)):
            return
        entry = self.history[index]
        if self._sampler is not None:
            self._sampler.close()  # unwind the abandoned loop's grad context deterministically
        self._sampler = self._session.sampler(
            width=self._width,
            height=self._height,
            steps=self._steps,
            resume_latent=entry.latent,
            resume_step=entry.step,
        )
        self._sampler.set_conditioning(self._last_items)
        self.total = self._sampler.total
        self.finished = False
        with self._lock:
            del self.history[index + 1 :]  # branched: drop the abandoned future
            self._frame = Frame(entry.preview, entry.index, entry.total, self.paint_applied_count)
        log.info("reverted to '%s' (step %d/%d)", entry.label, entry.index, entry.total)

    # -- run loop --
    def run(self) -> None:
        # skip_steps=0 so any total-step count >= 1 is valid (no init image to skip toward).
        self._sampler = self._session.sampler(
            width=self._width,
            height=self._height,
            steps=self._steps,
            seed=None,
            skip_steps=0,
            perlin=self._perlin,
        )
        self.total = self._sampler.total
        log.info("worker started: %dx%d, %d steps", self._width, self._height, self.total)

        try:
            while not self._stop_event.is_set():
                self._process_seek()  # revert works while paused, playing, or finished
                if self.finished or not self._resume.is_set():
                    time.sleep(0.03)  # idle (done/paused) but stay responsive to seek/stop
                    continue
                # Checkpoint the pre-edit state before applying paint or a flagged prompt edit.
                with self._lock:
                    label = self._checkpoint_label
                    self._checkpoint_label = None
                    has_paint = self._pending_paint is not None
                if label is not None or has_paint:
                    self._add_checkpoint(label or "paint")

                self._apply_pending()  # re-mix conditioning
                self._apply_pending_paint()  # inject painted pixels
                try:
                    step = next(self._sampler)
                except StopIteration:
                    self.finished = True  # idle (don't exit) so reverts can still resume it
                    log.info("worker finished")
                    continue
                if not self._started_history:  # baseline checkpoint after the first step
                    self._started_history = True
                    self._add_checkpoint("start")
                pil = self._sampler.current_pil()
                if pil is not None:
                    with self._lock:
                        self._frame = Frame(
                            np.asarray(pil), step.index, step.total, self.paint_applied_count
                        )
        except Exception:  # don't die silently and freeze the UI — log and stop the run
            log.exception("generation step failed; stopping")
            self.finished = True
        finally:
            if self._sampler is not None:
                self._sampler.close()  # unwind the loop's grad context on this thread
