"""Small kernel benchmarks on NPU or CPU, with the same timing shape as the runner.

``timeit`` returns median/p10/p90 milliseconds over ``iters`` calls after
``warmup``.  On an NPU it synchronizes through ``torch.npu.synchronize`` and uses
device events when they are available; on CPU it uses ``perf_counter``.
"""
from __future__ import annotations

import statistics
import time

import torch


def _sync(device: torch.device) -> None:
    if device.type == 'npu':
        try:
            import torch_npu  # noqa: F401
            torch.npu.synchronize()
        except Exception:                                # noqa: BLE001
            pass
    elif device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, *, warmup: int = 5, iters: int = 20, device: torch.device | None = None) -> dict:
    dev = device or (fn().device if callable(fn) else torch.device('cpu'))
    for _ in range(warmup):
        fn()
    _sync(dev)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {'median_ms': statistics.median(times), 'min_ms': times[0], 'max_ms': times[-1],
            'p10_ms': times[max(0, int(0.1 * len(times)) - 1)], 'p90_ms': times[min(len(times) - 1, int(0.9 * len(times)))],
            'iters': iters, 'device': str(dev)}


def compare(fn_a, fn_b, *, warmup: int = 5, iters: int = 20, device=None) -> dict:
    """Two implementations of the same op: speed ratio b/a (> 1: b is faster)."""
    a = timeit(fn_a, warmup=warmup, iters=iters, device=device)
    b = timeit(fn_b, warmup=warmup, iters=iters, device=device)
    return {'a': a, 'b': b, 'speedup': a['median_ms'] / b['median_ms'] if b['median_ms'] else None}


def bench_kernels(shape=(4096, 4096), dtype=None, device: str | None = None) -> dict:
    """RMSNorm / SwiGLU / GEMM at one shape, each against its PyTorch reference."""
    from .kernels import fused_ops
    dev = torch.device(device or ('npu' if _npu_available() else 'cpu'))
    dt = dtype or (torch.bfloat16 if dev.type == 'npu' else torch.float32)
    x = torch.randn(shape[0], shape[1], dtype=dt, device=dev)
    w = torch.randn(shape[1], dtype=dt, device=dev)
    out = {}
    out['rms_norm'] = compare(lambda: fused_ops.rms_norm(x, w), lambda: _rms_ref(x, w), device=dev)
    x2 = torch.randn(shape[0], shape[1] * 2, dtype=dt, device=dev)
    out['swiglu'] = compare(lambda: fused_ops.swiglu(x2), lambda: _swiglu_ref(x2), device=dev)
    a = torch.randn(shape[0], shape[1], dtype=dt, device=dev)
    b = torch.randn(shape[1], shape[1], dtype=dt, device=dev)
    out['gemm'] = compare(lambda: fused_ops.gemm(a, b), lambda: torch.matmul(a, b), device=dev)
    return out


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:                                    # noqa: BLE001
        return False


def _rms_ref(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


def _swiglu_ref(x):
    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


__all__ = ['timeit', 'compare', 'bench_kernels']
