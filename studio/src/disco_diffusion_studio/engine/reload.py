"""Background model (re)load: rebuilding the DiscoSession weights off the UI thread.

Changing the CLIP set or the secondary-model toggle needs a full weight reload (~a minute), so it
runs on a daemon thread. :class:`ModelReloader` owns just the *mechanism* — a debounce timer plus
the worker thread and its result — while the App keeps the policy (what to stage, when it counts
as a change, and swapping the finished session in). The UI thread drives it: ``schedule``/``cancel``
as the selection changes, ``due``/``start`` to fire the debounced reload, ``poll`` to collect it.
"""

from __future__ import annotations

import logging
import threading

import torch
from disco_diffusion import DiscoSession, RunConfig

log = logging.getLogger("disco_diffusion_studio.reload")

# A reload result is exactly one of {"session": DiscoSession} or {"error": str}.
ReloadResult = dict[str, object]


class ModelReloader:
    """Runs ``DiscoSession(cfg, device)`` on a daemon thread, with a debounced trigger."""

    def __init__(self) -> None:
        self._reloading = False
        self._thread: threading.Thread | None = None
        self._result: ReloadResult | None = None
        self._queued_at: int | None = None  # pygame ticks at which to fire the auto-reload

    @property
    def reloading(self) -> bool:
        return self._reloading

    @property
    def queued(self) -> bool:
        return self._queued_at is not None

    def schedule(self, fire_at: int) -> None:
        """Queue the debounced reload to fire at ``fire_at`` (pygame ticks)."""
        self._queued_at = fire_at

    def cancel(self) -> None:
        """Drop any queued reload (e.g. the selection landed back on the loaded set)."""
        self._queued_at = None

    def due(self, now: int) -> bool:
        """True when a queued reload should fire now (and isn't already running)."""
        return self._queued_at is not None and not self._reloading and now >= self._queued_at

    def start(self, cfg: RunConfig, device: torch.device | None) -> None:
        """Kick off a session rebuild on a daemon thread; clears any pending queue."""
        self._queued_at = None
        self._result = None
        self._reloading = True

        def work() -> None:
            try:
                # Pass the existing device so there's no interactive CPU-fallback prompt.
                self._result = {"session": DiscoSession(cfg, device=device)}
            except Exception as exc:  # surfaced on the UI thread by poll()
                log.exception("model reload failed")
                self._result = {"error": str(exc)}

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def poll(self) -> ReloadResult | None:
        """Return the finished result once (``{"session"|"error": …}``), else ``None``."""
        if not self._reloading or self._thread is None or self._thread.is_alive():
            return None
        self._reloading = False
        self._thread = None
        result, self._result = self._result, None
        return result
