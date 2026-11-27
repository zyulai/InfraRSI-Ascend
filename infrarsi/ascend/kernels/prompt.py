"""Prompt builder for Ascend kernel synthesis, driven by the extracted chip data.

The chip record in ``infrarsi/ascend/data/chips.yaml`` exists so that a kernel
generator does not have to guess the target: this module turns it into the
hardware contract (on-chip buffer sizes, alignment, dtypes, arch flags) and a
task template that a code model can fill.

The template is deliberately strict about the numerics gate and about which
files may change, mirroring InfraRSI's kernel skill: a generated kernel is a
candidate, not a result.
"""
from __future__ import annotations

from .. import chips


def language_for(card=None, op: str = 'elementwise') -> str:
    """The most practical kernel language for this card and op."""
    c = card or chips.current()
    arch = c.npu_arch if c else ''
    if arch == 'dav-3510':
        return 'tilelang' if op in ('gemm', 'attention', 'moe') else 'triton'
    if op in ('gemm', 'grouped_gemm'):
        return 'ascendc' if arch == 'dav-2201' else 'tilelang'
    return 'triton'


def kernel_prompt(op: str, *, card=None, shape: str = 'M=4096, N=4096, K=4096', dtype: str = 'bf16',
                  language: str | None = None, reference: str | None = None) -> str:
    c = card or chips.current()
    con = chips.constraints(c)
    lang = language or language_for(c, op)
    hw = chips.prompt_block(c)
    return f"""# Ascend kernel task: {op}

## Target
{hw}

## Contract
- Implement `{op}` for {shape}, dtype `{dtype}`, layout contiguous (row-major) unless the reference says otherwise.
- Language: **{lang}**. Entry point: a Python function with the same argument order as the reference below.
- The kernel must be correct for the full shape, not only the aligned core: handle the tail with the same
  arithmetic (masked loads/stores), never by padding with zeros that enter a reduction.
- Forward and backward (if the op trains) must both exist; a forward-only kernel is an inference kernel and must
  say so in its docstring.

## Hard limits (from infrarsi/ascend/data/chips.yaml)
- On-chip buffers (KB): {con.get('buffers_kb')}
- L2: {con.get('l2_mb')} MB; cube MAC: {con.get('cube_mac')}; cube/vector cores: {con.get('cube_cores')}/{con.get('vector_cores')}
- DMA alignment: {con.get('dma_alignment_bytes')} bytes; arch: {con.get('npu_arch')} ({con.get('arch_dir')}),
  compile with SOC_VERSION={con.get('soc_version')}, base mode {con.get('base_mode')}.
- Supported dtypes: {', '.join(con.get('dtypes') or []) or 'inherit from the reference'}.

## Reference semantics
{reference or 'Use the PyTorch reference in the job; do not re-derive the maths.'}

## Deliverable
1. The kernel source, self-contained, with the tile/block sizes as named constants and a one-line rationale for
   each against the buffer limits above.
2. A CPU/PyTorch fallback with identical semantics, selected when the NPU op is unavailable.
3. A numerics test that compares against a float64 reference on the first real inputs, reporting max absolute
   error, max relative error and the row-wise maximum separately. Tolerance: rtol=2e-2, atol=1e-3 unless the
   job's contract is tighter.
4. A benchmark: median/p10/p90 ms over at least 20 iterations after 5 warmups, plus the achieved TFLOPS or GB/s.

## Forbidden
- Changing the dtype of an accumulation, softmax/LSE, or a reduction to make a tile fit.
- Skipping the tail, the mask, or a row whose length is not a multiple of the tile.
- Reporting a speedup without the numerics gate passing.
"""


def review_checklist() -> list[str]:
    """What a reviewer (or the search's verifier) checks before a generated kernel is screened."""
    return [
        'The kernel compiles for the card\'s arch (SOC_VERSION / CATLASS_ARCH / dav flag), not a neighbouring one.',
        'Every on-chip buffer allocation is <= the card\'s L1/L0A/L0B/L0C/UB size, including double buffering.',
        'All global-memory addresses are 32-byte aligned (or the card\'s stated alignment).',
        'The tail path is exercised by a test whose length is not a multiple of the tile.',
        'The float64 gate passes on real inputs; row-wise maxima are reported, not only the aggregate.',
        'The PyTorch fallback is bit-identical to the pre-change job.',
        'The benchmark includes host-device synchronization and reports a median, not a single run.',
    ]
