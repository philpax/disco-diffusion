"""Guidance loss functions used to steer diffusion sampling.

Ported from the original Disco Diffusion notebook (Katherine Crowson et al.).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def spherical_dist_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared great-circle distance between two batches of embeddings."""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input: torch.Tensor) -> torch.Tensor:
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input: torch.Tensor) -> torch.Tensor:
    """Penalise pixel values that stray outside the [-1, 1] range."""
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])
