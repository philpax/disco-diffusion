"""Disco Diffusion: a durable, typed port of the CLIP-guided diffusion art generator.

Two entry points:

* :func:`~disco_diffusion.generate.generate` runs a whole batch from a
  :class:`~disco_diffusion.config.RunConfig` (what the CLI uses).
* :class:`~disco_diffusion.session.DiscoSession` is the external-control API: encode
  prompts once, then drive the sampling loop yourself, changing the prompt mix per step.
  See the Disco Diffusion Studio app (``studio/``) for a live demo.
"""

from .config import RunConfig
from .generate import Generator, generate
from .session import DiscoSession, EncodedPrompt, Sampler, StepResult

__version__ = "6.0.0"

__all__ = [
    "DiscoSession",
    "EncodedPrompt",
    "Generator",
    "RunConfig",
    "Sampler",
    "StepResult",
    "__version__",
    "generate",
]
