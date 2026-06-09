"""The edit-history / revert controller: checkpoints, preview scrubbing, and undo.

:class:`History` drives the scrubbable timeline that the worker fills with checkpoints. It owns the
prompt rows shown while previewing a checkpoint and the pending "guidance settled" checkpoint, and
orchestrates revert (branch the run from a checkpoint, or img2img from a loaded one). The pure scrub
maths live on :class:`~.timeline.Timeline`; this is the App-facing glue that reads the worker / live
frame and drives the sidebar + bottom-bar + canvas. It takes the App and reaches its pieces through
it; the App holds one and forwards the history events / per-frame sync to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
from PIL import Image

from .controls import PromptRow
from .worker import HistoryEntry, PromptSpec

if TYPE_CHECKING:
    from .app import App


class History:
    """Checkpoint requests, preview scrubbing, undo, and revert for the edit timeline."""

    def __init__(self, app: App) -> None:
        self.app = app
        self.preview_prompts: list[PromptRow] | None = None  # prompts shown while previewing
        # pygame ticks at which to drop a guidance-change checkpoint (set when a guidance slider
        # moves, pushed back on each further move, fired once the value settles). None = idle.
        self.guidance_checkpoint_at: int | None = None

    def request_checkpoint(self, label: str) -> None:
        """Ask the worker to drop a labelled checkpoint (no-op if nothing's running)."""
        worker = self.app.worker
        if worker is not None and worker.is_alive():
            worker.checkpoint(label)

    # -- guidance "settled" checkpoint (debounced) --
    def arm_guidance_checkpoint(self, at: int) -> None:
        """Schedule a "guidance" checkpoint for tick ``at`` (pushed back on each further move)."""
        self.guidance_checkpoint_at = at

    def cancel_guidance_checkpoint(self) -> None:
        """Drop any pending guidance checkpoint (a discrete change supersedes the drag)."""
        self.guidance_checkpoint_at = None

    def tick_guidance_checkpoint(self, now: int) -> None:
        """Fire the pending guidance checkpoint once its settle delay has elapsed."""
        if self.guidance_checkpoint_at is not None and now >= self.guidance_checkpoint_at:
            self.guidance_checkpoint_at = None
            self.request_checkpoint("guidance")

    # -- live-frame reads (the slider's right edge / rightmost snap point) --
    def total(self) -> int:
        """The run's display-step total (the history slider's right edge)."""
        worker = self.app.worker
        frame = worker.latest_frame() if worker is not None else None
        return self.app._timeline.total(frame.total if frame is not None else 1)

    def live_index(self) -> int:
        """The current live display step (the slider's rightmost snap point)."""
        worker = self.app.worker
        frame = worker.latest_frame() if worker is not None else None
        if frame is not None:
            return frame.index
        entries = self.app._timeline.entries
        if self.app.canvas.frame_surface is not None and entries:
            # A loaded session's result is a static final frame (no worker): its endpoint is the
            # run's last step, past every recorded checkpoint, so scrubbing can reach it.
            return entries[-1].total
        return entries[-1].index if entries else 0

    def displayed_surface(self) -> pygame.Surface | None:
        """The canvas surface to show: a previewed checkpoint, else the live frame."""
        return self.app._timeline.preview_surface() or self.app.canvas.frame_surface

    # -- preview state (label + the prompt rows shown while scrubbing) --
    def refresh_label(self) -> None:
        """Update the history readout to the previewed checkpoint, or the live step."""
        app = self.app
        e = app._timeline.preview_entry()
        if e is not None:
            text = f"{e.label} {e.index}/{e.total}"
        else:
            frame = app.worker.latest_frame() if app.worker is not None else None
            text = f"live {frame.index}/{frame.total}" if frame is not None else "live"
        app.bottom_bar.set_history_label(text)

    def sync_preview_prompts(self) -> None:
        """Show the previewed checkpoint's prompts in the rows (live set when not previewing)."""
        app = self.app
        entry = app._timeline.preview_entry()
        want: list[PromptSpec] | None = entry.prompts if entry is not None else None
        cur = (
            [PromptSpec(p.text, p.weight, p.muted) for p in self.preview_prompts]
            if self.preview_prompts is not None
            else None
        )
        if (list(want) if want is not None else None) == cur:
            return  # already showing the right prompts
        self.preview_prompts = (
            [PromptRow(t, w, m) for t, w, m in want] if want is not None else None
        )
        app.bottom_bar.rebuild_prompt_rows(app)
        app._sync_enabled()

    def refresh_preview_state(self) -> None:
        """Update the prompt rows, label, and Play gating whenever the preview changes."""
        self.sync_preview_prompts()
        self.refresh_label()
        self.app.bottom_bar.gate_play_for_preview(self.app._timeline.preview_index is not None)

    # -- revert / undo --
    def revert(self) -> None:
        """Branch the run from the previewed checkpoint, restoring its prompts + guidance/eta."""
        app = self.app
        if app._timeline.preview_index is None:
            return
        entry = app._timeline.entries[app._timeline.preview_index]
        if app.worker is None:  # loaded-session history (no latent) -> img2img from its preview
            self._revert_loaded(entry)
            return
        # Adopt the checkpoint's prompts as the live set, then branch from it.
        app.prompts = [PromptRow(t, w, m) for t, w, m in entry.prompts]
        # Restore the guidance + eta captured at the checkpoint (undo the changes made since),
        # syncing the sliders; cancel any pending guidance checkpoint.
        entry.config.apply_to(app.session.config)
        self.cancel_guidance_checkpoint()
        app.sidebar.refresh_advanced_widgets(app)
        app.worker.seek(app._timeline.preview_index)
        app._timeline.clear_preview()
        self.preview_prompts = None
        # Branching drops any in-flight strokes (the worker clears its queue on seek), so reset
        # the overlay tracking to stay aligned with the worker's apply count.
        app.canvas.paint.reset_overlays(app.worker.paint_applied_count)
        # Park the thumb on the checkpoint's actual step now — the worker processes the seek
        # asynchronously, so live_index() would still read the stale (forward) live frame here.
        app.bottom_bar.park_history_thumb(float(entry.index))
        app.bottom_bar.rebuild_prompt_rows(app)  # now-live (reverted) prompts
        app._push_prompts()  # apply them to the resumed run
        app._sync_enabled()  # re-enable prompt editing
        self.refresh_preview_state()

    def _revert_loaded(self, entry: HistoryEntry) -> None:
        """Revert into a loaded checkpoint (no latent): continue from its preview via img2img."""
        app = self.app
        app.prompts = [PromptRow(t, w, m) for t, w, m in entry.prompts]
        entry.config.apply_to(app.session.config)
        app.sidebar.refresh_advanced_widgets(app)
        app._set_init_image(Image.fromarray(entry.preview), f"history: {entry.label}")
        app._timeline.clear_preview()
        self.preview_prompts = None
        app.bottom_bar.rebuild_prompt_rows(app)
        app._start_run()  # seeds the new run from the checkpoint preview (img2img)

    def keyboard_revert(self) -> None:
        """Ctrl+Z: step back through the checkpoints, reverting to each in turn (undo)."""
        app = self.app
        if app.worker is None or app.running or not app._timeline.entries:
            return
        app._timeline.begin_undo()
        self.revert()

    # -- per-frame sync --
    def sync(self) -> None:
        """Pull the worker's history each frame; rebuild the slider when it changes length.

        With no worker we keep the timeline as-is — empty after a stop, or restored from a loaded
        session (which has no worker until you Play/Revert).
        """
        app = self.app
        tl = app._timeline
        if tl.sync(app.worker.get_history() if app.worker is not None else None):
            app.bottom_bar.rebuild_history_slider(app)
            app._sync_enabled()
        elif app.running and tl.preview_index is None and tl.entries:
            # While actively generating (not previewing), let the thumb track the live step as it
            # advances. When paused/done we leave it alone so a scrub isn't yanked back each frame.
            app.bottom_bar.park_history_thumb(float(self.live_index()))
        self.refresh_label()  # keep the live step current as it ticks
