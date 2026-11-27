"""Numerics gates for a kernel swap.

InfraRSI's rule for kernels is the same on an NPU as on a GPU: a fused op may
change reduction order and rounding, never the loss.  This module supplies the
comparison used before a kernel candidate is allowed into the search:

* a float64 reference on the first real inputs (not random ones), and
* a per-element relative error bound with an absolute floor for near-zero
  values, reported separately for the rows that matter (the ones with a
  non-negligible reference norm).

``gate`` returns a dict a receipt can store; ``assert_gate`` raises.
"""
from __future__ import annotations

import torch


def _f64(x: torch.Tensor) -> torch.Tensor:
    return x.detach().double()


def compare(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 2e-2, atol: float = 1e-3,
            row_dim: int | None = None) -> dict:
    """Elementwise error of ``actual`` against ``expected`` (both float64 inside)."""
    a, e = _f64(actual), _f64(expected)
    if a.shape != e.shape:
        return {'passed': False, 'reason': f'shape {tuple(a.shape)} != {tuple(e.shape)}'}
    diff = (a - e).abs()
    denom = e.abs().clamp_min(atol)
    rel = diff / denom
    scale = e.abs().max().item() or 1.0
    out = {
        'max_abs': diff.max().item(),
        'max_rel': rel.max().item(),
        'rel_to_scale': (diff.max() / scale).item(),
        'rtol': rtol, 'atol': atol,
        'passed': bool((diff <= atol + rtol * e.abs()).all().item()),
    }
    if row_dim is not None:
        row_norm = e.abs().sum(dim=row_dim)
        keep = row_norm > (row_norm.max() * 1e-6)
        if keep.any():
            out['row_max_rel'] = rel[keep].max().item()
            out['rows_checked'] = int(keep.sum().item())
            out['row_passed'] = bool((diff[keep] <= atol + rtol * e[keep].abs()).all().item())
            out['passed'] = bool(out['passed'] and out['row_passed'])
    return out


def gate(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 2e-2, atol: float = 1e-3,
         row_dim: int | None = None, name: str = '') -> dict:
    res = compare(actual, expected, rtol=rtol, atol=atol, row_dim=row_dim)
    res['name'] = name
    res['ok'] = res['passed']
    return res


def assert_gate(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float = 2e-2, atol: float = 1e-3,
                row_dim: int | None = None, name: str = '') -> dict:
    res = gate(actual, expected, rtol=rtol, atol=atol, row_dim=row_dim, name=name)
    if not res['passed']:
        raise AssertionError(f'{name or "kernel"}: numerics gate failed (max_rel={res.get("max_rel")}, '
                             f'max_abs={res.get("max_abs")})')
    return res


def fp64_reference(fn, *args, **kwargs) -> torch.Tensor:
    """Run ``fn`` on float64 copies of its tensor arguments and return a float64 result."""
    conv = [a.double() if isinstance(a, torch.Tensor) and a.is_floating_point() else a for a in args]
    out = fn(*conv, **kwargs)
    return out[0].double() if isinstance(out, (tuple, list)) else out.double()


__all__ = ['compare', 'gate', 'assert_gate', 'fp64_reference']
