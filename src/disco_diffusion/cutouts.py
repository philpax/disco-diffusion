"""Image cutout / augmentation modules for CLIP guidance.

``MakeCutouts`` is the original sampler; ``MakeCutoutsDango`` is Dango233's
advanced overview/inner-crop method used by default. Ported from the original
Disco Diffusion notebook with the animation-mode and debug branches removed
(this library only generates still images).
"""

from __future__ import annotations

import math

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import nn
from torch.nn import functional as F

from .vendor.resize_right import resize

# resize_right's resampling is a fixed *linear* operator on pixel values, so for a
# given (in_size, out_size) it equals a constant matrix M (out_size x in_size). The
# Python-heavy part of resize_right is recomputing that matrix every call; here we
# extract it once (by resizing an identity basis), cache it, and apply it as a matmul.
# This reproduces resize_right's output to float-rounding (~3e-7) while being ~10x
# faster, which is the dominant cost of the cutout method at high resolution.
_resize_matrix_cache: dict[tuple[int, int, torch.dtype, str], torch.Tensor] = {}


def _resize_matrix(
    in_sz: int, out_sz: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (in_sz, out_sz, dtype, str(device))
    matrix = _resize_matrix_cache.get(key)
    if matrix is None:
        with torch.no_grad():
            if in_sz == out_sz:
                matrix = torch.eye(in_sz, device=device, dtype=dtype)
            else:
                eye = torch.eye(in_sz, device=device, dtype=dtype).reshape(in_sz, 1, in_sz, 1)
                out = resize(eye, out_shape=[in_sz, 1, out_sz, 1])
                matrix = out.reshape(in_sz, out_sz).T.contiguous()
        _resize_matrix_cache[key] = matrix
    return matrix


def _resample_to(x: torch.Tensor, size: int) -> torch.Tensor:
    """Antialiased resample to ``size x size``, matching resize_right to float noise."""
    _, _, h, w = x.shape
    mat_h = _resize_matrix(h, size, x.device, x.dtype)
    mat_w = _resize_matrix(w, size, x.device, x.dtype)
    out = torch.einsum("oh,nchw->ncow", mat_h, x)
    return torch.einsum("pw,ncow->ncop", mat_w, out)


def _resample_interpolate(x: torch.Tensor, size: int) -> torch.Tensor:
    """Faster native antialiased resample (the ``--fast-interpolate-cutout`` lever).

    bicubic-antialias is ~40x faster than the cached matrix but a *systematic* (not
    random) departure from resize_right's lanczos — a visibly different (still good)
    sample, outside the noise floor.
    """
    return F.interpolate(x, size=(size, size), mode="bicubic", antialias=True)


def sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x: torch.Tensor, a: int) -> torch.Tensor:
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio: float, width: int) -> torch.Tensor:
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0.0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(
    input: torch.Tensor, size: tuple[int, int], align_corners: bool = True
) -> torch.Tensor:
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)


class MakeCutouts(nn.Module):
    """Simple random-crop cutout sampler (used for image prompts)."""

    def __init__(self, cut_size: int, cutn: int, skip_augs: bool = False) -> None:
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(
                    max_size
                    * torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(float(self.cut_size / max_size), 1.0)
                )
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        return torch.cat(cutouts, dim=0)


class MakeCutoutsDango(nn.Module):
    """Dango233's overview + inner-crop cutout method (the default)."""

    def __init__(
        self,
        cut_size: int,
        overview: int = 4,
        inner_crop: int = 0,
        ic_size_pow: float = 0.5,
        ic_grey_p: float = 0.2,
        skip_augs: bool = False,
        fast_resize: bool = False,
    ) -> None:
        super().__init__()
        self.cut_size = cut_size
        self.overview = overview
        self.inner_crop = inner_crop
        self.ic_size_pow = ic_size_pow
        self.ic_grey_p = ic_grey_p
        self.skip_augs = skip_augs
        self._resize = _resample_interpolate if fast_resize else _resample_to
        self.augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
        )
        cutout = self._resize(pad_input, self.cut_size)

        if self.overview > 0:
            if self.overview <= 4:
                if self.overview >= 1:
                    cutouts.append(cutout)
                if self.overview >= 2:
                    cutouts.append(gray(cutout))
                if self.overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = self._resize(pad_input, self.cut_size)
                for _ in range(self.overview):
                    cutouts.append(cutout)

        if self.inner_crop > 0:
            for i in range(self.inner_crop):
                size = int(torch.rand([]) ** self.ic_size_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.ic_grey_p * self.inner_crop):
                    cutout = gray(cutout)
                cutout = self._resize(cutout, self.cut_size)
                cutouts.append(cutout)

        cutouts_t = torch.cat(cutouts)
        if not self.skip_augs:
            cutouts_t = self.augs(cutouts_t)
        return cutouts_t
