"""Ascend edition of InfraRSI: device facts, chip data, kernels and launch knobs.

The package is import-safe on a CPU-only laptop: ``torch_npu`` is only imported
behind ``env.torch_npu()`` and ``kernels.fused_ops`` is lazy.  Start from
``env.detect_family()`` and ``chips.current()``.
"""
from __future__ import annotations

from . import chips, env, knobs, patch, smi, topology

__all__ = ['env', 'smi', 'topology', 'knobs', 'chips', 'numerics', 'patch', 'kernels', 'doctor', 'bench',
           'linkbench']


def __getattr__(name):
    if name in ('kernels', 'doctor', 'bench', 'linkbench', 'numerics'):
        import importlib
        return importlib.import_module(f'.{name}', __name__)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
