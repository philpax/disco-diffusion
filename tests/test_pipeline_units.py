"""CPU-only unit tests for pipeline building blocks (no model downloads)."""

from __future__ import annotations

import torch

from disco_diffusion.cutouts import MakeCutoutsDango
from disco_diffusion.losses import range_loss, spherical_dist_loss, tv_loss
from disco_diffusion.prompts import parse_prompt
from disco_diffusion.secondary import SecondaryDiffusionImageNet2


def test_parse_prompt_weight() -> None:
    assert parse_prompt("a cat") == ("a cat", 1.0)
    assert parse_prompt("a cat:2") == ("a cat", 2.0)
    text, weight = parse_prompt("https://example.com/x.png:3")
    assert text == "https://example.com/x.png"
    assert weight == 3.0


def test_cutouts_dango_output_shape() -> None:
    cut_size = 32
    overview, inner = 4, 6
    cutter = MakeCutoutsDango(cut_size, overview=overview, inner_crop=inner, skip_augs=True)
    img = torch.rand(1, 3, 64, 80)
    out = cutter(img)
    assert out.shape == (overview + inner, 3, cut_size, cut_size)


def test_losses_shapes_and_signs() -> None:
    x = torch.randn(2, 3, 16, 16)
    assert tv_loss(x).shape == (2,)
    assert range_loss(x).shape == (2,)
    # range_loss is zero for in-range input, positive for out-of-range
    assert torch.allclose(range_loss(torch.zeros(1, 3, 8, 8)), torch.zeros(1))
    assert range_loss(torch.full((1, 3, 8, 8), 5.0)).item() > 0

    a = torch.randn(4, 512)
    b = torch.randn(3, 512)
    dists = spherical_dist_loss(a.unsqueeze(1), b.unsqueeze(0))
    assert dists.shape == (4, 3)


def test_secondary_model_forward() -> None:
    model = SecondaryDiffusionImageNet2().eval()
    x = torch.randn(1, 3, 64, 64)
    t = torch.tensor([0.5])
    with torch.no_grad():
        out = model(x, t)
    assert out.pred.shape == x.shape
    assert out.eps.shape == x.shape
