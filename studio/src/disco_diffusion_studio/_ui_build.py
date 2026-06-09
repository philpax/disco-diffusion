"""Sidebar / panel UI construction for the studio App.

Free functions taking the ``App`` (rather than methods) so the ~700-line build cluster lives
outside ``app.py``; ``App``'s widget attributes are declared there so these stay fully typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import App


def _build_ui(app: App) -> None:
    # Preserve the seed field's current contents across the rebuild (the widget is recreated).
    if hasattr(app, "seed_entry"):
        app._seed_text = app.sidebar.seed_entry.get_text()
    app.manager.clear_and_reset()
    app.bottom_bar._remove_buttons.clear()
    app.bottom_bar._mute_buttons.clear()
    app.bottom_bar._prompt_entries.clear()
    app.bottom_bar._weight_sliders.clear()
    app.bottom_bar._row_elements.clear()
    app.bottom_bar.build(app)
    app.sidebar.build(app)
    app._sync_enabled()
