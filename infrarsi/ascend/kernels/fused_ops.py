"""Fused operators for Ascend, with a pure-PyTorch fallback for every one.

The wrappers are the narrow waist between a training job and the kernel
libraries: a job calls ``infrarsi.ascend.kernels.rms_norm`` and gets
``torch_npu.npu_rms_norm`` when it exists, the same maths in PyTorch otherwise.
Every wrapper records which backend answered in ``LAST_BACKEND`` so a receipt
can tell an NPU run from a CPU test.

Numerics contract: the fallbacks compute in float32 (float64 when asked) and
cast back, and the fused ops are only used when the input dtype is one of
bfloat16/float16/float32.  ``numerics.check`` compares the two.
"""
from __future__ import annotations

import math
import os
from typing import Callable

import torch
import torch.nn.functional as F

LAST_BACKEND: dict[str, str] = {}
STRICT = os.environ.get('INFRARSI_ASCEND_STRICT', '') not in ('', '0', 'false')


def _record(op: str, backend: str) -> None:
    LAST_BACKEND[op] = backend


def _npu():
    from .. import env
    return env.torch_npu()


def _try(op: str, fn: Callable, fallback: Callable):
    """Run ``fn``; on any failure fall back (unless INFRARSI_ASCEND_STRICT=1)."""
    if STRICT:
        out = fn()
        _record(op, 'npu')
        return out
    try:
        out = fn()
        _record(op, 'npu')
        return out
    except Exception:                                    # noqa: BLE001 - a missing op is the normal case on CPU
        _record(op, 'torch')
        return fallback()


# ------------------------------------------------------------------ RMSNorm
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``x / sqrt(mean(x^2) + eps) * weight`` over the last dimension."""
    def npu():
        tn = _npu()
        if tn is None or x.device.type != 'npu':
            raise RuntimeError('no npu')
        out = tn.npu_rms_norm(x, weight, eps)
        return out[0] if isinstance(out, (tuple, list)) else out

    def ref():
        dt = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (y * weight.float()).to(dt)

    return _try('rms_norm', npu, ref)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                 eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """``(norm(x + residual), x + residual)`` — the fused residual-add + RMSNorm pair."""
    def npu():
        tn = _npu()
        if tn is None or x.device.type != 'npu':
            raise RuntimeError('no npu')
        out = tn.npu_add_rms_norm(x, residual, weight, eps)
        if isinstance(out, (tuple, list)) and len(out) >= 3:
            return out[0], out[2]
        if isinstance(out, (tuple, list)) and len(out) == 2:
            return out[0], x + residual
        raise RuntimeError('unexpected npu_add_rms_norm output')

    def ref():
        s = x + residual
        return rms_norm(s, weight, eps), s

    return _try('add_rms_norm', npu, ref)


# ------------------------------------------------------------------ SwiGLU
def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``silu(x[..., :d]) * x[..., d:]`` where ``d = x.size(dim) // 2``."""
    def npu():
        tn = _npu()
        if tn is None or x.device.type != 'npu':
            raise RuntimeError('no npu')
        return tn.npu_swiglu(x, dim=dim)

    def ref():
        a, b = x.chunk(2, dim=dim)
        return F.silu(a) * b

    return _try('swiglu', npu, ref)


def gelu(x: torch.Tensor, approximate: str = 'none') -> torch.Tensor:
    def npu():
        tn = _npu()
        if tn is None or x.device.type != 'npu':
            raise RuntimeError('no npu')
        return tn.npu_fast_gelu(x) if approximate == 'tanh' else tn.npu_gelu(x, approximate=approximate)

    def ref():
        return F.gelu(x, approximate=approximate)

    return _try('gelu', npu, ref)


# ------------------------------------------------------------------ RoPE
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def rotary_embedding(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     interleaved: bool = False) -> torch.Tensor:
    """Apply RoPE to ``x`` given cos/sin broadcastable to ``x``.

    ``interleaved`` selects the GPT-NeoX (rotate_half) or the GPT-J (pairwise)
    layout.  ``torch_npu.npu_rotary_mul`` implements the half/rotate_half form;
    the interleaved form stays in PyTorch.
    """
    def npu():
        tn = _npu()
        if tn is None or x.device.type != 'npu' or interleaved:
            raise RuntimeError('no npu or interleaved layout')
        return tn.npu_rotary_mul(x, cos, sin)

    def ref():
        if interleaved:
            x1, x2 = x[..., 0::2], x[..., 1::2]
            c1, c2 = cos[..., 0::2], cos[..., 1::2]
            s1, s2 = sin[..., 0::2], sin[..., 1::2]
            out = torch.empty_like(x)
            out[..., 0::2] = x1 * c1 - x2 * s1
            out[..., 1::2] = x2 * c2 + x1 * s2
            return out
        return x * cos + rotate_half(x) * sin

    return _try('rotary_embedding', npu, ref)


# ------------------------------------------------------------------ attention
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int,
                    scale: float | None = None, causal: bool = True,
                    input_layout: str = 'BNSD') -> torch.Tensor:
    """Scaled dot-product attention through ``npu_fusion_attention``.

    q/k/v are ``[B, N, S, D]`` for BNSD (the NPU layout) or ``[B, S, N, D]`` for
    BSND; the fallback accepts either and calls ``F.scaled_dot_product_attention``.
    """
    def npu():
        tn = _npu()
        if tn is None or q.device.type != 'npu':
            raise RuntimeError('no npu')
        mask = None
        if causal:
            s = q.shape[2] if input_layout == 'BNSD' else q.shape[1]
            mask = torch.triu(torch.ones((s, s), dtype=torch.bool, device=q.device), diagonal=1)
        out = tn.npu_fusion_attention(q, k, v, num_heads, input_layout, atten_mask=mask,
                                      scale=1.0 if scale is None else scale, keep_prob=1.0,
                                      pre_tockens=2147483647, next_tockens=2147483647, inner_precise=0)
        return out[0] if isinstance(out, (tuple, list)) else out

    def ref():
        if input_layout == 'BNSD':
            qq, kk, vv = (t.transpose(1, 2) for t in (q, k, v))
            out = F.scaled_dot_product_attention(qq, kk, vv, is_causal=causal, scale=scale)
            return out.transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)

    return _try('flash_attention', npu, ref)


# ------------------------------------------------------------------ loss
def cross_entropy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100,
                  label_smoothing: float = 0.0, reduction: str = 'mean'):
    """Cross entropy over the last dimension through ``npu_cross_entropy_loss``."""
    def npu():
        tn = _npu()
        if tn is None or logits.device.type != 'npu':
            raise RuntimeError('no npu')
        out = tn.npu_cross_entropy_loss(logits, labels, reduction=reduction,
                                        ignore_index=ignore_index, label_smoothing=label_smoothing)
        return out[0] if isinstance(out, (tuple, list)) else out

    def ref():
        return F.cross_entropy(logits, labels, ignore_index=ignore_index, label_smoothing=label_smoothing,
                               reduction=reduction)

    return _try('cross_entropy', npu, ref)


# ------------------------------------------------------------------ GEMM
def gemm(a: torch.Tensor, b: torch.Tensor, backend: str = 'auto', out: torch.Tensor | None = None):
    """``a @ b`` through DeepGEMM-Ascend when it is installed and the shape qualifies.

    DeepGEMM-Ascend is only used for 2-D BF16/FP16 operands whose M, N, K are
    multiples of 64 (its contiguous-layout alignment); everything else goes to
    the aclnn GEMM behind ``torch.matmul``.
    """
    def deep():
        import deep_gemm
        if a.dim() != 2 or b.dim() != 2 or a.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError('deep_gemm needs 2-D bf16/fp16')
        if any(x % 64 for x in (a.shape[0], b.shape[1], a.shape[1])):
            raise RuntimeError('shape not aligned')
        d = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device) if out is None else out
        deep_gemm.bf16_gemm_nn(a, b, d=d)
        return d

    def torch_mm():
        return torch.matmul(a, b, out=out) if out is not None else torch.matmul(a, b)

    if backend == 'torch':
        _record('gemm', 'torch')
        return torch_mm()
    if backend in ('auto', 'deep_gemm'):
        try:
            res = deep()
            _record('gemm', 'deep_gemm')
            return res
        except Exception:                                # noqa: BLE001
            if backend == 'deep_gemm':
                raise
    _record('gemm', 'torch')
    return torch_mm()


# ------------------------------------------------------------------ introspection
def backends() -> dict[str, str]:
    return dict(LAST_BACKEND)


def explain() -> str:
    if not LAST_BACKEND:
        return 'no fused op has run yet'
    return ', '.join(f'{k}={v}' for k, v in sorted(LAST_BACKEND.items()))


def reset() -> None:
    LAST_BACKEND.clear()


__all__ = ['rms_norm', 'add_rms_norm', 'swiglu', 'gelu', 'rotary_embedding', 'flash_attention',
           'cross_entropy', 'gemm', 'backends', 'explain', 'reset', 'LAST_BACKEND']
