"""Ascend kernel layer: catalogue, planner registry, fused ops and Triton references.

Importing this package does not import torch or torch_npu.  ``fused_ops`` is
loaded on first attribute access, so `infrarsi ascend kernels list` works on a
laptop without PyTorch.
"""
from __future__ import annotations

from . import catalog, registry

_LAZY = ('rms_norm', 'add_rms_norm', 'swiglu', 'gelu', 'rotary_embedding', 'flash_attention', 'cross_entropy',
         'gemm', 'backends', 'explain', 'reset')


def __getattr__(name):
    if name in _LAZY:
        from . import fused_ops
        return getattr(fused_ops, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['catalog', 'registry', *_LAZY]
