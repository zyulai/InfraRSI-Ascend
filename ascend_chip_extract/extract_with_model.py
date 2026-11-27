#!/usr/bin/env python3
"""Ask the local hy4-flash model (CLIProxyAPI :8317) to extract Ascend chip data.

Usage:
    python3 extract_with_model.py --batch a
    python3 extract_with_model.py --batch b

The evidence pack is built from the repository READMEs downloaded under
`sources/`; the model must cite a file and a section for every non-null value.
Raw responses are kept so a human can audit what the model actually said.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = HERE / 'sources'
ENDPOINT = os.environ.get('HY4_ENDPOINT', 'http://127.0.0.1:8317/v1/chat/completions')
MODEL = os.environ.get('HY4_MODEL', 'cn:hy4flash-free')
KEY = os.environ.get('HY4_KEY', '')

if not KEY:
    raise SystemExit('set HY4_KEY to the CLIProxyAPI key before running (the key is never stored in this repo)')

SCHEMA = {
    'schema': 'infrarsi-ascend-chips-v1',
    'cards': [{
        'class': '910B2',
        'aliases': ['Ascend910B2', 'Ascend 910B2', '910B2'],
        'soc_version': 'Ascend910B2',
        'npu_arch': 'dav-2201',
        'arch_id': 2201,
        'arch_dir': 'arch22',
        'generation': 'A2',
        'product': 'Atlas A2 训练/推理',
        'memory_gib': 64,
        'memory_type': 'HBM2e',
        'hbm_tbps': 1.6,
        'bf16_tflops_peak': None,
        'fp16_tflops_peak': None,
        'fp32_tflops_peak': None,
        'int8_tops': None,
        'fp8_tflops_peak': None,
        'fp4_tflops_peak': None,
        'cube_cores': 24,
        'vector_cores': 48,
        'cube_freq_ghz': 1.8,
        'cube_mac': '16x16x16',
        'l1_kb': 512,
        'l0a_kb': 64,
        'l0b_kb': 64,
        'l0c_kb': 128,
        'ub_kb': 192,
        'bt_kb': 1,
        'l2_mb': 192,
        'sparsity': True,
        'intra_link': 'HCCS',
        'intra_link_gbps': 392,
        'inter_link': 'RoCE',
        'inter_link_gbps': 50,
        'tdp_w': 310,
        'kernel_gen': {
            'supported_dtypes': ['fp16', 'bf16', 'int8', 'fp32'],
            'soc_version_flag': 'Ascend910B2',
            'base_mode': 'MEMORY',
            'dma_alignment_bytes': 32,
            'tile_shape_hint': '16x16x16 cube; L0C 128KB, L1 512KB, UB 192KB',
            'notes': 'A2: Cube:Vector = 1:2, 4:2 sparsity, L1->L0A/B path exists',
        },
        'confidence': 'high',
        'sources': ['npu-hardware-params.md §2.3', 'npu-arch-guide.md §3'],
        'missing': ['bf16_tflops_peak: not stated for this SKU in the sources'],
    }],
    'missing_cards': ['...'],
    'notes': '...',
}

BATCHES = {
    'a': ['Ascend910B1', 'Ascend910B2', 'Ascend910B3', 'Ascend910B4', 'Ascend910B2C', 'Ascend910_93'],
    'b': ['Ascend950PR (PCIE, 28-core)', 'Ascend950PR (Server, 32-core)', 'Ascend950DT',
          'Ascend310P1', 'Ascend310P3', 'Ascend310P5', 'Ascend310P7',
          'Ascend310B1', 'Ascend310B2', 'Ascend310B3', 'Ascend310B4', 'Ascend910'],
}


def read(name: str) -> str:
    p = SOURCES / name
    return p.read_text(encoding='utf-8', errors='replace') if p.is_file() else ''


def evidence_pack() -> str:
    parts = []
    for name in ('npu-hardware-params.md', 'npu-arch-guide.md', 'pto-kernels.md', 'catlass.md',
                 'deepgemm-ascend.md', 'tilelang-ascend.md', 'mojo-opset.md', 'triton-ascend.md'):
        text = read(name)
        if not text:
            continue
        parts.append(f'\n\n===== FILE: {name} =====\n{text[:30000]}')
    # every line of the remaining READMEs that names a card or an arch
    rx = re.compile(r'(Ascend|910|950|310|DAV_|dav-|arch[0-9])', re.I)
    extra = []
    for name in ('triton-ascend-kernels.md', 'flashkda.md', 'flashqla.md', 'deepselect.md', 'pto-isa.md'):
        text = read(name)
        hits = [ln for ln in text.splitlines() if rx.search(ln)]
        if hits:
            extra.append(f'\n\n===== FILE: {name} (card/arch lines) =====\n' + '\n'.join(hits[:80]))
    return ''.join(parts) + ''.join(extra)


def prompt(batch: str) -> str:
    cards = BATCHES[batch]
    return f"""You are a hardware-data extraction engine. Read the SOURCE DOCUMENTS and fill the JSON schema.

TASK
Extract the hardware specification of these Ascend NPU cards, one object per card:
{json.dumps(cards, ensure_ascii=False)}

RULES
1. Use ONLY facts stated in the SOURCE DOCUMENTS. If a field is not stated, OMIT it (do not emit null) and put its
   name in a single compact string "missing": "field1, field2". Never guess.
2. A value you compute from a formula in the sources is allowed only if the formula is stated there; then set
   "derived": true inside that card's "kernel_gen" and lower "confidence" to "medium". Never guess a TFLOPS number.
3. "sources" must cite the FILE name and the section/table, e.g. "npu-hardware-params.md §2.3".
4. Keep the schema keys exactly, but emit compact JSON (no indentation, one line per card is fine).
5. "class" is the short InfraRSI class name: 910B1, 910B2, 910B3, 910B4, 910B2C, 910_93, 950PR-PCIE, 950PR-Server,
   950DT, 310P1, 310P3, 310P5, 310P7, 310B1..B4, 910.
6. For 950PR there are two SKUs (PCIE 28-core and Server 32-core): emit two cards with classes "950PR-PCIE" and
   "950PR-Server", both aliases of Ascend950PR.
7. "kernel_gen" is for a later Ascend kernel generator: on-chip memories (l1/l0a/l0b/l0c/ub/bt/l2 in KB/MB),
   cube MAC shape, cube:vector core ratio, supported dtypes, soc_version flag, base mode, DMA alignment,
   arch directory. Omit what the sources do not say.
8. Output ONLY one fenced ```json block, no prose before or after. Be terse: no explanations, no reasoning in the
   output, compact JSON.

SCHEMA EXAMPLE (values are placeholders, not facts):
{json.dumps(SCHEMA, ensure_ascii=False, indent=1)}

SOURCE DOCUMENTS
{evidence_pack()}
"""


def call(text: str, max_tokens: int = 24000) -> dict:
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': text}],
                       'temperature': 0, 'max_tokens': max_tokens, 'reasoning_effort': 'low'}).encode()
    req = urllib.request.Request(ENDPOINT, data=body, headers={
        'Content-Type': 'application/json', 'Authorization': f'Bearer {KEY}'})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def extract_json(content: str) -> dict | None:
    m = re.search(r'```json\s*(.*?)```', content, re.S)
    text = m.group(1) if m else content
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', choices=sorted(BATCHES), required=True)
    ap.add_argument('--max-tokens', type=int, default=24000)
    a = ap.parse_args()
    p = prompt(a.batch)
    (HERE / f'prompt_{a.batch}.txt').write_text(p, encoding='utf-8')
    print(f'prompt chars={len(p)} batch={a.batch}')
    res = call(p, a.max_tokens)
    msg = res['choices'][0]['message']
    content = msg.get('content') or ''
    (HERE / f'raw_{a.batch}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    data = extract_json(content)
    if data is None:
        print('FAILED to parse JSON; content head:')
        print(content[:1500])
        return 2
    (HERE / f'chips_{a.batch}.json').write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
    print('cards:', [c.get('class') for c in data.get('cards', [])])
    print('usage:', res.get('usage'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
