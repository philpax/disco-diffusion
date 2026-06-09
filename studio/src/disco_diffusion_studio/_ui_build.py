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


def _sync_enabled(app: App) -> None:
    """Total-steps + size boxes are editable only when not actively generating."""
    editable = not app.running
    for el in (
        app.sidebar.steps_entry,
        app.sidebar.seed_entry,
        app.sidebar.random_seed_button,
        app.sidebar.width_entry,
        app.sidebar.height_entry,
        app.sidebar.apply_button,
        app.sidebar.swap_button,
    ):
        (el.enable if editable else el.disable)()
    # History controls are usable only while paused/stopped and there's history to scrub.
    hist_on = editable and len(app._timeline.entries) > 0
    for hist_el in (
        app.bottom_bar.history_slider,
        app.bottom_bar.revert_button,
        app.bottom_bar.cancel_button,
    ):
        (hist_el.enable if hist_on else hist_el.disable)()
    # Prompt rows are read-only while previewing a checkpoint (they show its prompts).
    prompts_on = app._timeline.preview_index is None
    prompt_widgets = [
        app.bottom_bar.add_button,
        *app.bottom_bar._prompt_entries,
        *app.bottom_bar._weight_sliders,
    ]
    for pw in (*prompt_widgets, *app.bottom_bar._remove_buttons, *app.bottom_bar._mute_buttons):
        (pw.enable if prompts_on else pw.disable)()
    app.bottom_bar.play_button.set_text("Pause" if app.running else "Play")
    # Can't resume mid-preview or mid-reload — Revert/Cancel, or wait for the reload.
    play_off = app._timeline.preview_index is not None or app._reloader.reloading
    (app.bottom_bar.play_button.disable if play_off else app.bottom_bar.play_button.enable)()
