"""Background generation thread that drives a Disco Diffusion Sampler.

All torch/CUDA work — encoding prompts, stepping the sampler, converting frames — happens
on this one thread. The UI thread only reads the latest published :class:`Frame`, the edit
history, and toggles the pause / stop / pending flags.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch
from disco_diffusion import DiscoSession, EncodedPrompt, Sampler
from PIL import Image

from .specs import GuidanceSnapshot, PromptSpec

log = logging.getLogger("disco_diffusion_studio.worker")

MAX_HISTORY = 60  # cap on edit-history checkpoints (each holds a CPU latent)


class PaintBatch(NamedTuple):
    """One completed stroke queued for injection: the RGBA paint plus its checkpoint label."""

    rgb: np.ndarray
    alpha: np.ndarray
    tint: np.ndarray
    label: str


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

    latent: torch.Tensor | None  # CPU latent at `step` (None for loaded-session checkpoints)
    step: int  # internal diffusion step the latent belongs to
    index: int  # display step index when captured
    total: int
    preview: np.ndarray  # (H, W, 3) uint8 — the image at this checkpoint
    label: str
    prompts: list[PromptSpec] = field(default_factory=list)  # (text, weight, muted) at capture
    config: GuidanceSnapshot = field(default_factory=GuidanceSnapshot)  # live guidance at capture


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
        init_image: Image.Image | None = None,
        skip_steps: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._session = session
        self._width = width
        self._height = height
        self._steps = steps
        self._encode_cache = encode_cache
        self._cache_lock = cache_lock
        self._perlin = perlin  # seed the fresh run from Perlin noise instead of flat gaussian
        self._seed = seed  # RNG seed for the fresh run (None = don't reseed; same on eager restart)
        # img2img: seed the fresh run from this image, noised to skip_steps in (resized to the
        # generation size by the library). A revert still resumes from a saved latent, not this.
        self._init_image = init_image
        self._init_skip_steps = skip_steps

        self._resume = threading.Event()
        self._resume.set()  # start running (not paused)
        # NB: not "_stop" — threading.Thread has an internal _stop() method, and shadowing
        # it breaks is_alive() once the thread finishes.
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._pending: list[PromptSpec] | None = None  # prompts awaiting (re)encode
        # A FIFO of paint batches. Each is injected on its own step with its own checkpoint, so
        # successive strokes stay separate edits in the history rather than coalescing into one.
        self._pending_paints: list[PaintBatch] = []
        self._pending_seek: int | None = None  # history index to revert to
        self._checkpoint_label: str | None = None  # request a checkpoint before the next step
        self._frame: Frame | None = None
        self._sampler: Sampler | None = None
        self._last_items: list[tuple[EncodedPrompt, float]] = []  # conditioning to re-apply
        self._last_prompts: list[PromptSpec] = []  # (text, weight, muted) shown in the UI
        self.history: list[HistoryEntry] = []
        self._started_history = False
        self.finished = False
        self.paint_applied_count = 0  # bumps each time a paint batch is injected
        self._compile_fallback_done = False  # one-shot: only fall back from compile once
        self.notice: str | None = None  # transient message for the UI to surface
        self.total = session.diffusion_for(steps).num_timesteps  # skip_steps=0 below

    # -- control (called from UI thread) --
    def set_prompts(self, prompts: list[PromptSpec]) -> None:
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

    def set_paint(
        self, rgb: np.ndarray, alpha: np.ndarray, tint: np.ndarray, label: str = "paint"
    ) -> None:
        """Queue a paint batch (one completed stroke); injected on its own step."""
        with self._lock:
            self._pending_paints.append(PaintBatch(rgb, alpha, tint, label))

    def has_pending_paint(self) -> bool:
        with self._lock:
            return bool(self._pending_paints)

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
        # These run-loop helpers only fire after _start_sampler(), so the sampler is live.
        assert self._sampler is not None
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return
        self._last_prompts = list(pending)
        # Muted (or empty) prompts are kept in _last_prompts (so a checkpoint/revert preserves
        # them) but excluded from the conditioning the sampler actually guides on.
        self._last_items = [
            (self._encode(t), w) for t, w, muted in pending if t.strip() and not muted
        ]
        self._sampler.set_conditioning(self._last_items)

    def _apply_pending_paint(self) -> None:
        assert self._sampler is not None
        # Needs a stepped latent to inject into; just after a revert there isn't one yet, so
        # keep the paint pending (don't drop it) until the resumed sampler has produced a step.
        if not self._sampler.has_output:
            return
        with self._lock:
            if not self._pending_paints:
                return
            paint = self._pending_paints.pop(0)  # one stroke per step (FIFO)
        self._sampler.paint(paint.rgb, paint.alpha, paint.tint)  # injects into the sample, in place
        self.paint_applied_count += 1

    def _add_checkpoint(self, label: str) -> None:
        assert self._sampler is not None
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
            config=GuidanceSnapshot.capture(self._session.config),
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
            # Only an actual revert abandons queued strokes — this runs every loop iteration, so
            # clearing unconditionally here would wipe paint before it was ever injected.
            if index is not None:
                self._pending_paints.clear()
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

    def _start_sampler(self) -> None:
        # With no init image, skip_steps=0 so any total-step count >= 1 is valid (nothing to skip
        # toward); with an init image we skip toward it (the library resizes it to the gen size).
        self._sampler = self._session.sampler(
            width=self._width,
            height=self._height,
            steps=self._steps,
            seed=self._seed,
            skip_steps=self._init_skip_steps,
            perlin=self._perlin,
            init_image=self._init_image,
        )
        self.total = self._sampler.total
        if self._last_items:  # restart path: re-apply the conditioning we already had
            self._sampler.set_conditioning(self._last_items)

    def _fallback_to_eager(self) -> None:
        """A compiled-warmup CUDA OOM: drop compile and restart this run eagerly.

        The compiled UNet's warmup memory spike can exceed VRAM at large sizes on a
        shared GPU; eager fits. The OOM happens on the first (warmup) step, so
        restarting from scratch loses no real progress.
        """
        log.warning("CUDA OOM under torch.compile; falling back to eager and restarting")
        self.notice = "Eager fallback"
        self._compile_fallback_done = True
        try:
            if self._sampler is not None:
                self._sampler.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup before we rebuild
            pass
        self._sampler = None
        self._session.disable_compile()
        # The failed compile's tensors are only freed once the OOM exception (and its
        # traceback, which references them) has been cleared by the caller — which is
        # why this runs *after* the except block, not inside it. Collect them, drop the
        # compiled-graph state, and return the freed blocks so the eager run has room.
        # NB: this reclaims PyTorch's own allocations, but torch.compile can leave a few
        # GB of CUDA-module/allocator residue that only a fresh process fully frees, so
        # the eager retry can still OOM on a memory-contended GPU — handled gracefully.
        gc.collect()
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        self._started_history = False
        with self._lock:
            self.history.clear()
        self._start_sampler()

    # -- run loop --
    def run(self) -> None:
        self._start_sampler()
        log.info("worker started: %dx%d, %d steps", self._width, self._height, self.total)

        try:
            while not self._stop_event.is_set():
                self._process_seek()  # revert works while paused, playing, or finished
                assert self._sampler is not None  # _start_sampler() ran; seek only re-points it
                if self.finished or not self._resume.is_set():
                    time.sleep(0.03)  # idle (done/paused) but stay responsive to seek/stop
                    continue
                # Checkpoint the pre-edit state before applying paint or a flagged prompt edit.
                # Each queued stroke carries its own label, so successive paints checkpoint (and
                # bake) one at a time rather than merging into a single edit.
                with self._lock:
                    label = self._checkpoint_label
                    self._checkpoint_label = None
                    paint_label = self._pending_paints[0].label if self._pending_paints else None
                if label is not None or paint_label is not None:
                    self._add_checkpoint(label or paint_label or "paint")

                self._apply_pending()  # re-mix conditioning
                self._apply_pending_paint()  # inject painted pixels
                oom_fallback = False
                try:
                    step = next(self._sampler)
                except StopIteration:
                    self.finished = True  # idle (don't exit) so reverts can still resume it
                    log.info("worker finished")
                    continue
                except torch.cuda.OutOfMemoryError:
                    if self._compile_fallback_done or not self._session.compiled:
                        # Already eager (or compile was off): genuinely out of VRAM.
                        # Stop gracefully with a clear message instead of crashing.
                        log.exception("out of GPU memory")
                        self.notice = "Out of VRAM"
                        self.finished = True
                        continue
                    # Compiled-warmup OOM: drop to eager and restart once. Defer the
                    # actual fallback until the except block exits — while it's active the
                    # exception's traceback pins the failed forward's tensors, so the
                    # memory can't be reclaimed until we're out of the handler.
                    oom_fallback = True
                if oom_fallback:
                    self._fallback_to_eager()
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
