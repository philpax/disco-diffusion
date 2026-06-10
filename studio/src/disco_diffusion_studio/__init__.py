"""Disco Diffusion Studio: an interactive UI for steering the sampling loop live.

Built on :class:`disco_diffusion.DiscoSession` — encode prompts once, then mix them per
step while the image forms. Run with ``disco-studio`` (or ``python -m disco_diffusion_studio``).
"""

from .app import App, main

__version__ = "0.1.0"

__all__ = ["App", "main"]
