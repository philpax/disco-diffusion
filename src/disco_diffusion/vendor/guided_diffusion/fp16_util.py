"""
Helpers for 16-bit precision.

Vendored from kostarion/guided-diffusion (MIT). Trimmed to the inference-only
helpers used by ``unet.py``; the training-time ``MixedPrecisionTrainer`` and its
``logger`` dependency have been removed.
"""

import torch.nn as nn


def convert_module_to_f16(l):
    """Convert primitive modules to float16."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """Convert primitive modules to float32, undoing convert_module_to_f16()."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()
