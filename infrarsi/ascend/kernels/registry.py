"""Which Ascend kernel implements which training op, and whether it is here.

The registry is a *planning* table: it names the callable and the constraints a
shape must meet, but never imports the library at module import time.  The
search uses it to turn "this job spends 40 % in RMSNorm" into a concrete
candidate (``torch_npu.npu_rms_norm``) and to say why a candidate is not
available on this host.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Impl:
    op: str
    library: str
    entry: str                                  # 'module:callable' or 'module.Class'
    dtypes: tuple[str, ...] = ('bf16', 'fp16', 'fp32')
    min_dim: int | None = None
    alignment: int | None = None
    backward: bool = True
    notes: str = ''
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def module(self) -> str:
        return self.entry.split(':')[0].split('.')[0]

    def available(self) -> bool:
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False

    def as_dict(self) -> dict:
        return {'op': self.op, 'library': self.library, 'entry': self.entry, 'dtypes': list(self.dtypes),
                'min_dim': self.min_dim, 'alignment': self.alignment, 'backward': self.backward,
                'notes': self.notes, 'tags': list(self.tags), 'available': self.available()}


# op -> implementations, best first.  "Best" = vendor fused op first, then a
# specialised library, then the generic Triton/TileLang path.
OPS: dict[str, tuple[Impl, ...]] = {
    'rms_norm': (
        Impl('rms_norm', 'torch_npu', 'torch_npu:npu_rms_norm', min_dim=2,
             notes='fused RMSNorm; returns (y, rstd); epsilon is the third argument'),
        Impl('rms_norm', 'mojo_opset', 'mojo_opset:MojoRMSNorm', min_dim=2,
             notes='module replacement; select the Triton backend with MOJO_BACKEND=ttx'),
    ),
    'add_rms_norm': (
        Impl('add_rms_norm', 'torch_npu', 'torch_npu:npu_add_rms_norm', min_dim=2,
             notes='residual add + RMSNorm in one pass; returns (y, rstd, residual_out)'),
    ),
    'swiglu': (
        Impl('swiglu', 'torch_npu', 'torch_npu:npu_swiglu', min_dim=2, alignment=2,
             notes='last dimension must be even: input is [gate | up] concatenated'),
        Impl('swiglu', 'pto_kernels', 'pto_kernels:pto_swiglu', min_dim=2, alignment=2,
             notes='tile-level SwiGLU on PTO-ISA; prebuilt wheels for 910B/910_93'),
        Impl('swiglu', 'mojo_opset', 'mojo_opset:MojoSwiGlu', min_dim=2, alignment=2),
    ),
    'gelu': (
        Impl('gelu', 'torch_npu', 'torch_npu:npu_gelu', min_dim=1,
             notes='npu_fast_gelu is the tanh approximation; npu_gelu takes approximate="tanh"|"none"'),
    ),
    'rotary': (
        Impl('rotary', 'torch_npu', 'torch_npu:npu_rotary_mul', min_dim=2, alignment=2,
             notes='rotate_half layout (half mode); cos/sin must broadcast to x'),
        Impl('rotary', 'mojo_opset', 'mojo_opset:MojoRotaryEmb', min_dim=2, alignment=2),
    ),
    'attention': (
        Impl('attention', 'torch_npu', 'torch_npu:npu_fusion_attention', min_dim=4,
             notes='BNSD layout, keep_prob=1.0 for training; returns (out, softmax_max, softmax_sum, ...)'),
        Impl('attention', 'torch_npu', 'torch_npu:npu_prompt_flash_attention', min_dim=4,
             notes='prefill-oriented flash attention; check the causal-mask convention'),
        Impl('attention', 'pto_kernels', 'pto_kernels:pto_flash_attention', min_dim=4,
             notes='tile-level flash attention; shapes must be multiples of the tile'),
        Impl('attention', 'triton_ascend_kernels', 'triton_ascend_kernels:flash_attention', min_dim=4,
             notes='Triton-Ascend flash attention; A2/A3'),
    ),
    'cross_entropy': (
        Impl('cross_entropy', 'torch_npu', 'torch_npu:npu_cross_entropy_loss', min_dim=2,
             notes='fused CE with label smoothing and ignore_index'),
        Impl('cross_entropy', 'mojo_opset', 'mojo_opset:MojoLinearCrossEntropyFunc', min_dim=2,
             notes='fused linear + CE: takes hidden states and the classifier weight'),
    ),
    'gemm': (
        Impl('gemm', 'torch_npu', 'torch_npu:matmul', min_dim=2,
             notes='aclnn GEMM through torch.matmul; the reference path'),
        Impl('gemm', 'deep_gemm', 'deep_gemm:bf16_gemm_nn', min_dim=2, alignment=64,
             notes='DeepGEMM-Ascend; 2-D BF16/FP16, M/N/K multiples of 64, preallocated d'),
        Impl('gemm', 'catlass', 'catlass', min_dim=2, alignment=64, backward=False,
             notes='C++ templates compiled into the extension; no Python callable until built'),
    ),
    'moe_dispatch': (
        Impl('moe_dispatch', 'torch_npu', 'torch_npu:npu_moe_distribute_dispatch', min_dim=2,
             notes='fused dispatch; pair with npu_moe_distribute_combine'),
    ),
    'moe_combine': (
        Impl('moe_combine', 'torch_npu', 'torch_npu:npu_moe_distribute_combine', min_dim=2),
    ),
    'optimizer': (
        Impl('optimizer', 'torch_npu', 'torch_npu:npu_apply_adam_w', min_dim=1,
             notes='fused AdamW update; the NPU counterpart of torch.optim.AdamW(fused=True)'),
    ),
    'linear_attention': (
        Impl('linear_attention', 'pto_kernels', 'pto_kernels:pto_gdn_mega', min_dim=3,
             notes='GDN fused mega-kernel (Gram/gate preparation + inverse/WY/state/output)'),
        Impl('linear_attention', 'flash_kda', 'flash_kda', min_dim=3, notes='Kimi Delta Attention'),
    ),
}


def candidates(op: str) -> list[Impl]:
    return list(OPS.get(op, ()))


def available(op: str | None = None) -> list[Impl]:
    ops = [op] if op else list(OPS)
    return [impl for o in ops for impl in OPS.get(o, ()) if impl.available()]


def resolve(op: str, prefer: str | None = None) -> Impl | None:
    """The first available implementation of ``op``, optionally restricted to a library."""
    for impl in OPS.get(op, ()):
        if prefer and impl.library != prefer:
            continue
        if impl.available():
            return impl
    return None


def coverage() -> dict:
    """op -> {'implementations': n, 'available': [library]} for the doctor output."""
    return {op: {'implementations': len(v), 'available': [i.library for i in v if i.available()]}
            for op, v in OPS.items()}


def plan(hot_ops: list[str]) -> list[dict]:
    """Concrete replacements for the hot ops, with the first available entry marked."""
    out = []
    for op in hot_ops:
        impls = OPS.get(op) or []
        if not impls:
            out.append({'op': op, 'status': 'no-ascend-impl', 'candidates': []})
            continue
        pick = next((i for i in impls if i.available()), None)
        out.append({'op': op, 'status': 'ready' if pick else 'not-installed',
                    'pick': pick.as_dict() if pick else None,
                    'candidates': [i.as_dict() for i in impls]})
    return out
