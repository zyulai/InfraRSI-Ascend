#!/usr/bin/env python3
"""Validate the model's chip JSON and emit the InfraRSI data files.

- `chips.yaml`        : what the model extracted from the repositories (verbatim, audited)
- `chips_public.yaml` : the public-spec supplement (fills only gaps)
- `gpus_ascend.yaml`  : the transfer-facing view (repo value first, public for gaps)
- `chips_merged.json` : the effective merged view, for human review
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'InfraRSI'

ARCH = {'dav-1001': 1001, 'dav-2002': 2002, 'dav-2201': 2201, 'dav-3002': 3002, 'dav-3510': 3510}
RANGES = {
    'memory_gib': (1, 1024), 'hbm_tbps': (0.05, 20), 'bf16_tflops_peak': (0.5, 5000),
    'fp16_tflops_peak': (0.5, 5000), 'fp32_tflops_peak': (0.1, 1000), 'int8_tops': (0.5, 10000),
    'fp8_tflops_peak': (0.5, 10000), 'fp4_tflops_peak': (0.5, 20000),
    'cube_cores': (1, 256), 'vector_cores': (1, 512), 'cube_freq_ghz': (0.3, 4.0),
    'l1_kb': (32, 4096), 'l0a_kb': (8, 1024), 'l0b_kb': (8, 1024), 'l0c_kb': (8, 2048),
    'ub_kb': (16, 4096), 'bt_kb': (0, 256), 'l2_mb': (1, 1024),
    'intra_link_gbps': (1, 10000), 'inter_link_gbps': (0.1, 10000), 'tdp_w': (10, 2000),
}
DERIVED_FIELDS = ('bf16_tflops_peak', 'fp16_tflops_peak', 'int8_tops', 'vector_cores', 'l2_mb')
# Product-table values win over a repository value for these SKU-level fields:
# the repos document architecture families, the product tables document a SKU.
PUBLIC_WINS = {'memory_gib', 'memory_type', 'hbm_tbps', 'bf16_tflops_peak', 'fp16_tflops_peak',
               'fp32_tflops_peak', 'int8_tops', 'fp8_tflops_peak', 'fp4_tflops_peak', 'cube_cores',
               'vector_cores', 'cube_freq_ghz', 'l2_mb', 'tdp_w', 'intra_link', 'intra_link_gbps',
               'inter_link', 'inter_link_gbps'}
CARD_KEYS = ('class', 'aliases', 'soc_version', 'npu_arch', 'arch_id', 'arch_dir', 'generation', 'product',
             'memory_gib', 'memory_type', 'hbm_tbps', 'bf16_tflops_peak', 'fp16_tflops_peak', 'fp32_tflops_peak',
             'int8_tops', 'fp8_tflops_peak', 'fp4_tflops_peak', 'cube_cores', 'vector_cores', 'cube_freq_ghz',
             'cube_mac', 'l1_kb', 'l0a_kb', 'l0b_kb', 'l0c_kb', 'ub_kb', 'bt_kb', 'l2_mb', 'sparsity',
             'intra_link', 'intra_link_gbps', 'inter_link', 'inter_link_gbps', 'tdp_w', 'kernel_gen',
             'confidence', 'sources', 'missing', 'derived')


def load(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return data.get('cards') or []


def normalize(card: dict) -> dict:
    out = {k: card[k] for k in CARD_KEYS if k in card and card[k] not in (None, '', [])}
    cls = str(out.get('class') or '').strip()
    out['class'] = cls
    if out.get('npu_arch') not in ARCH:
        raise ValueError(f'{cls}: bad npu_arch {out.get("npu_arch")!r}')
    out['arch_id'] = ARCH[out['npu_arch']]
    for key, (lo, hi) in RANGES.items():
        if key in out:
            v = out[key]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not (lo <= v <= hi):
                raise ValueError(f'{cls}: {key}={v!r} outside {lo}..{hi}')
    if not isinstance(out.get('sources'), list) or not out['sources']:
        raise ValueError(f'{cls}: no sources')
    if out.get('confidence') not in ('high', 'medium', 'low'):
        raise ValueError(f'{cls}: bad confidence {out.get("confidence")!r}')
    if not isinstance(out.get('kernel_gen'), dict):
        raise ValueError(f'{cls}: kernel_gen is not an object')
    return out


def merge() -> list[dict]:
    cards: dict[str, dict] = {}
    for name in ('chips_a.json', 'chips_b.json'):
        p = HERE / name
        if not p.is_file():
            continue
        for c in load(p):
            if not c.get('class'):
                continue
            n = normalize(c)
            cards[n['class']] = n
    return [cards[k] for k in sorted(cards, key=lambda x: (ARCH[cards[x]['npu_arch']], x))]


def public() -> dict:
    p = HERE / 'public_specs.yaml'
    return yaml.safe_load(p.read_text(encoding='utf-8')) or {} if p.is_file() else {}


def effective(card: dict, pub: dict) -> dict:
    out = dict(card)
    p = pub.get(card['class']) or {}
    derived = bool((card.get('kernel_gen') or {}).get('derived'))
    prov = {}
    for k, v in p.items():
        if k in ('sources', 'confidence', 'note', 'notes'):
            continue
        if out.get(k) in (None, '', []) or k in PUBLIC_WINS:
            if v is not None:
                out[k] = v
                prov[k] = 'public'
            else:
                prov[k] = 'repo'
        else:
            prov[k] = 'repo'
    out['provenance'] = prov
    out['public_sources'] = p.get('sources') or []
    if out.get('confidence') == 'low' and p.get('confidence'):
        out['confidence'] = p['confidence']
    return out


def gpus_yaml(cards: list[dict]) -> dict:
    out = {}
    for c in cards:
        entry = {'aliases': c.get('aliases') or [c.get('soc_version', ''), c['class']],
                 'arch': c['npu_arch'], 'memory_gib': c.get('memory_gib')}
        for key in ('bf16_tflops_peak', 'fp8_tflops_peak', 'hbm_tbps', 'intra_link_gbps'):
            if c.get(key) is not None:
                entry[key] = c[key]
        entry['bf16_tflops_measured'] = None
        entry['fp8_tflops_measured'] = None
        bits = [f"generation {c.get('generation')}" if c.get('generation') else '',
                f"{c.get('cube_cores')} cube / {c.get('vector_cores')} vector cores" if c.get('cube_cores') else '',
                f"L1 {c['l1_kb']}KB, UB {c['ub_kb']}KB, L0C {c['l0c_kb']}KB" if c.get('l1_kb') else '',
                'sources: ' + '; '.join(c.get('sources', [])[:3])]
        entry['notes'] = '; '.join(b for b in bits if b)
        out[c['class']] = entry
    return out


def main() -> int:
    raw = merge()
    pub = public()
    eff = [effective(c, pub) for c in raw]
    (HERE / 'chips_merged.json').write_text(
        json.dumps({'schema': 'infrarsi-ascend-chips-v1', 'cards': eff}, ensure_ascii=False, indent=1),
        encoding='utf-8')
    gdir = ROOT / 'infrarsi' / 'experience' / 'data'
    adir = ROOT / 'infrarsi' / 'ascend' / 'data'
    gdir.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)
    (gdir / 'gpus_ascend.yaml').write_text(
        '# Ascend NPU classes extracted from the kernel repositories; the full per-card record and its\n'
        '# provenance is in infrarsi/ascend/data/chips.yaml (repo) + chips_public.yaml (public supplement).\n'
        + yaml.safe_dump(gpus_yaml(eff), sort_keys=False, allow_unicode=True), encoding='utf-8')
    (adir / 'chips.yaml').write_text(
        '# Per-card Ascend data for the kernel generator, extracted by hy4-flash (:8317) from the kernel\n'
        '# repositories under ascend_chip_extract/sources. Every value cites a file+section; gaps are listed\n'
        '# under "missing" and filled by chips_public.yaml at load time.\n'
        + yaml.safe_dump({'schema': 'infrarsi-ascend-chips-v1', 'cards': {c['class']: c for c in raw}},
                         sort_keys=False, allow_unicode=True), encoding='utf-8')
    if (HERE / 'public_specs.yaml').is_file():
        shutil.copyfile(HERE / 'public_specs.yaml', adir / 'chips_public.yaml')
    print(f'{len(raw)} cards -> gpus_ascend.yaml, chips.yaml (+ chips_public.yaml)')
    for c in eff:
        print(f"  {c['class']:<14} {c['npu_arch']:<9} mem={c.get('memory_gib')}GB "
              f"bf16={c.get('bf16_tflops_peak')} cores={c.get('cube_cores')} "
              f"conf={c.get('confidence')} prov={c.get('provenance')}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
