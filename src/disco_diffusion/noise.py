"""Perlin-noise init image generation (optional alternative to an init image).

Ported from the original Disco Diffusion notebook; original perlin helpers from
https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps


def _interp(t: torch.Tensor) -> torch.Tensor:
    return 3 * t**2 - 2 * t**3


def perlin(width: int, height: int, scale: int, device: torch.device) -> torch.Tensor:
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - _interp(xs)
    wy = 1 - _interp(ys)
    dots = torch.zeros((), device=device)
    dots = dots + wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots = dots + (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots = dots + wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots = dots + (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def perlin_ms(
    octaves: list[float], width: int, height: int, grayscale: bool, device: torch.device
) -> torch.Tensor:
    out_array = [torch.tensor(0.5)] if grayscale else [torch.tensor(0.5)] * 3
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] = out_array[i].to(device) + p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(
    side_x: int,
    side_y: int,
    device: torch.device,
    octaves: list[float],
    width: int,
    height: int,
    grayscale: bool,
) -> Image.Image:
    out = perlin_ms(octaves, width, height, grayscale, device)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out_pil = TF.to_pil_image(out.clamp(0, 1)).convert("RGB")
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out_pil = TF.to_pil_image(out.clamp(0, 1).squeeze())
    return ImageOps.autocontrast(out_pil)


def regen_perlin(
    side_x: int,
    side_y: int,
    device: torch.device,
    perlin_mode: str,
    batch_size: int,
) -> torch.Tensor:
    color = perlin_mode in ("color", "mixed")
    gray = perlin_mode in ("gray", "mixed")
    init = create_perlin_noise(
        side_x, side_y, device, [1.5**-i * 0.5 for i in range(12)], 1, 1, not color
    )
    init2 = create_perlin_noise(
        side_x, side_y, device, [1.5**-i * 0.5 for i in range(8)], 4, 4, gray
    )
    init_t = (
        TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
    )
    return init_t.expand(batch_size, -1, -1, -1)
