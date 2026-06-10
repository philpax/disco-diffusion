"""The brush colour palette: a fixed set of swatches plus recently-picked colours.

The fixed swatches come from ``config.toml``; picking an off-palette colour prepends it to the
recents (deduped, capped, persisted). :class:`Palette` owns that model and its persistence, while
the App keeps the rendering (swatch hit-rects, the colour preview) and the RGB picker dialog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .colours import MAX_RECENT, RGB, ColourConfig, load_colours, save_colours

log = logging.getLogger("disco_diffusion_studio.palette")


@dataclass
class Palette:
    """A fixed swatch set plus recently-picked colours; ``swatches()`` is what the UI lays out."""

    fixed: list[RGB]
    recent: list[RGB]

    @classmethod
    def load(cls) -> Palette:
        """Load the palette + recents from config (falling back to the seed palette)."""
        cfg = load_colours()
        return cls(fixed=list(cfg.palette), recent=list(cfg.recent))

    def default_brush(self) -> RGB:
        """A sensible initial brush colour: the 4th fixed swatch, or the first."""
        return self.fixed[3] if len(self.fixed) > 3 else self.fixed[0]

    def swatches(self) -> list[RGB]:
        """The fixed palette followed by any recently-picked colours not already in it."""
        out = list(self.fixed)
        for c in self.recent:
            if c not in out:
                out.append(c)
        return out

    def remember(self, rgb: RGB) -> bool:
        """Record a picked colour as a recent (most-recent-first, deduped, capped) and persist it.

        Fixed-palette colours are already shown as swatches, so they don't earn a recents slot —
        returns ``False`` (nothing changed) in that case, ``True`` when a new recent was added.
        """
        if rgb in self.fixed:
            return False
        self.recent = [rgb, *(c for c in self.recent if c != rgb)][:MAX_RECENT]
        try:
            save_colours(ColourConfig(palette=self.fixed, recent=self.recent))
        except Exception:  # noqa: BLE001 - persistence is best-effort; don't crash the UI
            log.exception("saving colour config failed")
        return True
