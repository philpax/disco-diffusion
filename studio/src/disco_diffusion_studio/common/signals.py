"""A tiny in-process event bus for UI broadcasts, so emitters don't depend on their listeners.

Two things are pure fire-and-forget broadcasts in the studio: a status-line message, and a
"widget enable/disable state is stale, re-sync it" notification. Both were methods on the App that
every controller reached back into; routing them through :class:`Signals` means a controller just
*announces* ("status changed") without holding a reference to the bottom bar / sidebar that render
the result. Dispatch is synchronous (handlers run inline), so behaviour matches the old direct calls
exactly. The App wires the listeners up once; emitters only need the bus.
"""

from __future__ import annotations

from collections.abc import Callable


class Signals:
    """Broadcast status messages + enablement-invalidation to whoever subscribed."""

    def __init__(self) -> None:
        self._status: list[Callable[[str], None]] = []
        self._invalidate: list[Callable[[], None]] = []
        self._edited: list[Callable[[], None]] = []

    def on_status(self, handler: Callable[[str], None]) -> None:
        """Subscribe to status-line messages (e.g. the bottom bar's status label)."""
        self._status.append(handler)

    def on_invalidate(self, handler: Callable[[], None]) -> None:
        """Subscribe to enablement-invalidation (e.g. an area re-syncing its widgets)."""
        self._invalidate.append(handler)

    def on_edited(self, handler: Callable[[], None]) -> None:
        """Subscribe to "a preset-controlled knob was edited" (e.g. the recipe flips to Custom)."""
        self._edited.append(handler)

    def status(self, text: str) -> None:
        """Announce a status-line message to every subscriber."""
        for handler in self._status:
            handler(text)

    def invalidate(self) -> None:
        """Announce that widget enable/disable state needs re-syncing to every subscriber."""
        for handler in self._invalidate:
            handler()

    def edited(self) -> None:
        """Announce a preset-controlled knob was edited (a guidance slider, a model toggle, …)."""
        for handler in self._edited:
            handler()
