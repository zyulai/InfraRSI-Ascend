"""Typed access to the per-card Ascend data extracted from the kernel repositories.

`data/chips.yaml` is the machine-readable record the kernel generator reads:
arch id, on-chip memories, cube geometry, dtypes and the compile flags each
card needs.  This module turns it into the two things a generator asks for:

* ``constraints(card)`` — the hard limits a tile/kernel must fit (bytes per
  buffer, alignment, dtype support), and
* ``prompt_block(card)`` — a compact, citable description to put in a kernel
  synthesis prompt.

Nothing here imports torch or torch_npu.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from . import env

DATA = Path(__file__).with_name('data') / 'chips.yaml'
PUBLIC = Path(__file__).with_name('data') / 'chips_public.yaml'

# Fields where a formula-derived repo value yields to a vendor/measured public figure.
DERIVED_FIELDS = ('bf16_tflops_peak', 'fp16_tflops_peak', 'int8_tops', 'vector_cores', 'l2_mb')
# Product-table values win over a repository value for these SKU-level fields.
PUBLIC_WINS = {'memory_gib', 'memory_type', 'hbm_tbps', 'bf16_tflops_peak', 'fp16_tflops_peak',
               'fp32_tflops_peak', 'int8_tops', 'fp8_tflops_peak', 'fp4_tflops_peak', 'cube_cores',
               'vector_cores', 'cube_freq_ghz', 'l2_mb', 'tdp_w', 'intra_link', 'intra_link_gbps',
               'inter_link', 'inter_link_gbps'}


@dataclass(frozen=True)
class Card:
    name: str
    soc_version: str
    npu_arch: str
    arch_id: int
    arch_dir: str | None = None
    generation: str | None = None
    product: str | None = None
    memory_gib: float | None = None
    memory_type: str | None = None
    hbm_tbps: float | None = None
    bf16_tflops_peak: float | None = None
    fp16_tflops_peak: float | None = None
    fp32_tflops_peak: float | None = None
    int8_tops: float | None = None
    fp8_tflops_peak: float | None = None
    fp4_tflops_peak: float | None = None
    cube_cores: int | None = None
    vector_cores: int | None = None
    cube_freq_ghz: float | None = None
    cube_mac: str | None = None
    l1_kb: int | None = None
    l0a_kb: int | None = None
    l0b_kb: int | None = None
    l0c_kb: int | None = None
    ub_kb: int | None = None
    bt_kb: int | None = None
    l2_mb: int | None = None
    sparsity: bool | None = None
    intra_link: str | None = None
    intra_link_gbps: float | None = None
    inter_link: str | None = None
    inter_link_gbps: float | None = None
    tdp_w: float | None = None
    aliases: tuple[str, ...] = ()
    kernel_gen: dict = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    public_sources: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)
    confidence: str | None = None
    missing: str | None = None

    @property
    def build_soc_version(self) -> str:
        """The SOC_VERSION a build uses: a wildcard in the extracted flag falls back to the card name."""
        flag = (self.kernel_gen or {}).get('soc_version_flag')
        if flag and '*' not in flag:
            return flag
        return self.name if self.name.lower().startswith('ascend') else f'Ascend{self.name}'

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, (), {}, [])}


@lru_cache(maxsize=None)
def _load() -> dict[str, Card]:
    if not DATA.is_file():
        return {}
    doc = yaml.safe_load(DATA.read_text(encoding='utf-8')) or {}
    pub = {}
    if PUBLIC.is_file():
        pub = yaml.safe_load(PUBLIC.read_text(encoding='utf-8')) or {}
    out = {}
    for name, d in (doc.get('cards') or {}).items():
        d = dict(d)
        d['name'] = name
        d['aliases'] = tuple(d.get('aliases') or ())
        d['sources'] = tuple(d.get('sources') or ())
        p = pub.get(name) or {}
        if p:
            derived = bool((d.get('kernel_gen') or {}).get('derived'))
            prov = {}
            for k, v in p.items():
                if k in ('sources', 'confidence', 'note', 'notes'):
                    continue
                if d.get(k) in (None, '', []) or k in PUBLIC_WINS:
                    if v is not None:
                        d[k] = v
                        prov[k] = 'public'
                    else:
                        prov[k] = 'repo'
                else:
                    prov[k] = 'repo'
            d['public_sources'] = tuple(p.get('sources') or ())
            d['provenance'] = prov
            if not d.get('confidence') or d.get('confidence') == 'low':
                d['confidence'] = p.get('confidence') or d.get('confidence')
        out[name] = Card(**{k: v for k, v in d.items() if k in Card.__dataclass_fields__})
    return out


def all_cards() -> dict[str, Card]:
    return dict(_load())


def get(name: str | None) -> Card | None:
    """Look a card up by class, SoC name, alias or an SMI string."""
    cards = _load()
    if not name:
        return None
    cls = env.soc_class(name)
    for key in (name, cls):
        if key and key in cards:
            return cards[key]
    n = str(name).lower().replace(' ', '').replace('-', '').replace('_', '')
    for c in cards.values():
        for a in (c.name, c.soc_version, *c.aliases):
            if str(a).lower().replace(' ', '').replace('-', '').replace('_', '') in n or n in str(a).lower().replace(' ', ''):
                return c
    return None


def current() -> Card | None:
    return get(env.soc_version())


def constraints(card: Card | None = None) -> dict:
    """The hard limits a kernel for this card must respect."""
    c = card or current()
    if c is None:
        return {}
    kg = c.kernel_gen or {}
    return {
        'soc_version': c.build_soc_version,
        'npu_arch': c.npu_arch,
        'arch_dir': c.arch_dir,
        'base_mode': kg.get('base_mode'),
        'dma_alignment_bytes': kg.get('dma_alignment_bytes') or 32,
        'buffers_kb': {k: getattr(c, k) for k in ('l1_kb', 'l0a_kb', 'l0b_kb', 'l0c_kb', 'ub_kb', 'bt_kb')
                       if getattr(c, k) is not None},
        'l2_mb': c.l2_mb,
        'cube_mac': c.cube_mac or '16x16x16',
        'cube_cores': c.cube_cores,
        'vector_cores': c.vector_cores,
        'cube_vector_ratio': kg.get('cube_vector_ratio'),
        'dtypes': kg.get('supported_dtypes') or [],
        'sparsity': c.sparsity,
        'notes': kg.get('notes'),
        'sources': list(c.sources),
    }


def prompt_block(card: Card | None = None) -> str:
    """A compact, citable hardware block for a kernel-synthesis prompt."""
    c = card or current()
    if c is None:
        return 'Ascend NPU: card unknown (run `infrarsi ascend chips` on the host).'
    con = constraints(c)
    lines = [f'Ascend card: {c.name} ({c.build_soc_version}), generation {c.generation or "?"}, {c.product or ""}'.rstrip(),
             f'arch: {c.npu_arch} (arch id {c.arch_id}, dir {c.arch_dir}), base mode {con.get("base_mode") or "?"}',
             f'HBM: {c.memory_gib} GiB {c.memory_type or ""}, {c.hbm_tbps or "?"} TB/s'.rstrip(),
             f'cube: {c.cube_cores} cores x {c.cube_freq_ghz} GHz, MAC {c.cube_mac or "16x16x16"}, '
             f'vector: {c.vector_cores} cores',
             f'on-chip KB: ' + ', '.join(f'{k}={v}' for k, v in con['buffers_kb'].items()) + f', L2={c.l2_mb} MB',
             f'dtypes: {", ".join(con["dtypes"]) or "unknown"}; sparsity: {c.sparsity}',
             f'DMA alignment: {con["dma_alignment_bytes"]} bytes; notes: {con.get("notes") or ""}',
             'repo sources: ' + '; '.join(c.sources),
             'public sources: ' + '; '.join(c.public_sources)]
    return '\n'.join(lines)
