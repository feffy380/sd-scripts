# Adapted from
# https://github.com/mosaicml/composer/blob/dev/composer/algorithms/low_precision_layernorm/low_precision_layernorm.py
# https://github.com/mosaicml/composer/blob/dev/composer/algorithms/low_precision_groupnorm/low_precision_groupnorm.py

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Low Precision LayerNorm and GroupNorm"""

import torch
import torch.nn.functional as F

orig_layer_norm = torch.nn.LayerNorm
orig_group_norm = torch.nn.GroupNorm


def apply_low_precision_norm():
    torch.nn.LayerNorm = LPLayerNorm
    torch.nn.GroupNorm = LPGroupNorm


def undo_low_precision_norm():
    torch.nn.LayerNorm = orig_layer_norm
    torch.nn.GroupNorm = orig_group_norm


class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(
            self.weight) if self.weight is not None else self.weight  # pyright: ignore[reportUnnecessaryComparison]
        downcast_bias = _cast_if_autocast_enabled(
            self.bias) if self.bias is not None else self.bias  # pyright: ignore[reportUnnecessaryComparison]
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)


class LPGroupNorm(torch.nn.GroupNorm):

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None):
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(
            self.weight) if self.weight is not None else self.weight  # pyright: ignore[reportUnnecessaryComparison]
        downcast_bias = _cast_if_autocast_enabled(
            self.bias) if self.bias is not None else self.bias  # pyright: ignore[reportUnnecessaryComparison]
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.group_norm(downcast_x, self.num_groups, downcast_weight, downcast_bias, self.eps)


def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor