"""Reference Triton kernels that run on Ascend through Triton-Ascend.

These are deliberately small and shape-generic: one program per row for the
row-wise ops, a block over the last dimension.  They are the kernels the
``npu_gemm_library=triton`` and custom-fusion candidates start from, and they
double as executable documentation of the Ascend constraints (32-byte
alignment, power-of-two block sizes, no atomics in the fallback path).

On a host without Triton the module imports; calling a kernel raises
``RuntimeError`` and the caller falls back to ``fused_ops``.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:                                        # noqa: BLE001 - Triton is optional
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rms_norm_kernel(X, W, Y, stride, N, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        y = x * tl.rsqrt(var + eps) * w
        tl.store(Y + row * stride + cols, y, mask=mask)

    @triton.jit
    def _swiglu_kernel(X, Y, stride, N, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        a = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(X + row * stride + N + cols, mask=mask, other=0.0).to(tl.float32)
        y = a * tl.sigmoid(a) * b
        tl.store(Y + row * N + cols, y, mask=mask)

    @triton.jit
    def _rope_kernel(X, COS, SIN, Y, stride, N, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        half = N // 2
        x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
        c = tl.load(COS + row * stride + cols, mask=mask, other=0.0)
        s = tl.load(SIN + row * stride + cols, mask=mask, other=0.0)
        # rotate_half: [:-half] pairs with [+half:] and vice versa
        pair = tl.where(cols < half, cols + half, cols - half)
        xp = tl.load(X + row * stride + pair, mask=mask, other=0.0)
        sign = tl.where(cols < half, -1.0, 1.0)
        y = x * c + sign * xp * s
        tl.store(Y + row * stride + cols, y, mask=mask)


def _need() -> None:
    if not HAS_TRITON:
        raise RuntimeError('Triton is not installed')


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    _need()
    n = x.shape[-1]
    rows = x.numel() // n
    y = torch.empty_like(x)
    _rms_norm_kernel[(rows,)](x, weight, y, x.stride(-2) if x.dim() > 1 else n, n, eps,
                              BLOCK=triton.next_power_of_2(n))
    return y


def swiglu(x: torch.Tensor) -> torch.Tensor:
    _need()
    n = x.shape[-1] // 2
    rows = x.numel() // (2 * n)
    y = torch.empty(x.shape[:-1] + (n,), dtype=x.dtype, device=x.device)
    _swiglu_kernel[(rows,)](x, y, x.stride(-2) if x.dim() > 1 else 2 * n, n,
                            BLOCK=triton.next_power_of_2(n))
    return y


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    _need()
    n = x.shape[-1]
    rows = x.numel() // n
    y = torch.empty_like(x)
    _rope_kernel[(rows,)](x, cos, sin, y, x.stride(-2) if x.dim() > 1 else n, n,
                          BLOCK=triton.next_power_of_2(n))
    return y


__all__ = ['HAS_TRITON', 'rms_norm', 'swiglu', 'rope']
