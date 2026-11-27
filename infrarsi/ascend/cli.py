"""`infrarsi ascend ...`: the Ascend edition's own commands.

    infrarsi ascend chips [--card 910B2] [--json]
    infrarsi ascend kernels [--op rms_norm]
    infrarsi ascend doctor
    infrarsi ascend bench [--shape 4096x4096]
    infrarsi ascend prompt --op rms_norm [--card 910B2]

These commands never import torch_npu at module import time; `doctor` and
`bench` import it only when they actually probe the device.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import chips, env, knobs, smi, topology


def _print_cards(as_json: bool, card: str | None) -> int:
    if card:
        c = chips.get(card)
        if c is None:
            print(f'unknown card {card!r}; known: {", ".join(chips.all_cards())}', file=sys.stderr)
            return 2
        print(json.dumps(c.as_dict(), ensure_ascii=False, indent=1, default=str))
        return 0
    cards = chips.all_cards()
    if as_json:
        print(json.dumps({k: v.as_dict() for k, v in cards.items()}, ensure_ascii=False, indent=1, default=str))
        return 0
    print(f'{"class":<14} {"arch":<9} {"mem GiB":>8} {"BF16 T":>8} {"cube":>5} {"vector":>7}  sources')
    for name, c in cards.items():
        print(f'{name:<14} {c.npu_arch:<9} {str(c.memory_gib or "-"):>8} {str(c.bf16_tflops_peak or "-"):>8} '
              f'{str(c.cube_cores or "-"):>5} {str(c.vector_cores or "-"):>7}  {len(c.sources)} repo / '
              f'{len(c.public_sources)} public')
    return 0


def _kernels(op: str | None) -> int:
    from .kernels import catalog, registry
    if op:
        print(json.dumps(registry.plan([op]), ensure_ascii=False, indent=1))
        return 0
    print('Kernel libraries:')
    for k in catalog.CATALOG:
        mark = 'OK ' if k.available() else '-- '
        print(f'  {mark}{k.name:<32} {k.repo}  ({k.license})')
    print('\nOp coverage:')
    for name, cov in registry.coverage().items():
        print(f'  {name:<18} {cov["implementations"]} impl, available: {", ".join(cov["available"]) or "none"}')
    return 0


def _doctor() -> int:
    from . import doctor
    print(json.dumps(doctor.report(), ensure_ascii=False, indent=1, default=str))
    return 0


def _bench(shape: str) -> int:
    from . import bench
    try:
        w, h = (int(x) for x in shape.lower().split('x', 1))
    except ValueError:
        print('--shape must be MxN, e.g. 4096x4096', file=sys.stderr)
        return 2
    print(json.dumps(bench.bench_kernels((w, h)), ensure_ascii=False, indent=1, default=str))
    return 0


def _prompt(op: str, card: str | None, shape: str, dtype: str, language: str | None) -> int:
    from .kernels import prompt
    c = chips.get(card) if card else chips.current()
    print(prompt.kernel_prompt(op, card=c, shape=shape, dtype=dtype, language=language))
    return 0


def build() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='infrarsi ascend', description='Ascend NPU edition of InfraRSI')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('chips', help='list the extracted Ascend cards, or show one')
    p.add_argument('--card', help='class or SoC name, e.g. 910B2 / Ascend910_93')
    p.add_argument('--json', action='store_true')
    p.set_defaults(fn=lambda a: _print_cards(a.json, a.card))
    p = sub.add_parser('kernels', help='list the kernel libraries and the op coverage')
    p.add_argument('--op', help='show the implementation plan for one op')
    p.set_defaults(fn=lambda a: _kernels(a.op))
    p = sub.add_parser('doctor', help='NPU environment, health and topology')
    p.set_defaults(fn=lambda a: _doctor())
    p = sub.add_parser('bench', help='RMSNorm / SwiGLU / GEMM microbenchmark')
    p.add_argument('--shape', default='4096x4096')
    p.set_defaults(fn=lambda a: _bench(a.shape))
    p = sub.add_parser('prompt', help='print a kernel-synthesis prompt for this card')
    p.add_argument('--op', required=True)
    p.add_argument('--card')
    p.add_argument('--shape', default='M=4096, N=4096, K=4096')
    p.add_argument('--dtype', default='bf16')
    p.add_argument('--language', choices=['triton', 'ascendc', 'tilelang'])
    p.set_defaults(fn=lambda a: _prompt(a.op, a.card, a.shape, a.dtype, a.language))
    return ap


def main(argv=None) -> int:
    a = build().parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':
    raise SystemExit(main())
