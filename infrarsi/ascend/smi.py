"""Parsers for ``npu-smi`` output, and a snapshot in the shape the project runner expects.

``npu-smi info`` has kept one table shape across CANN 6.x-9.x, with small
variations (the chip row may carry the name on either line, and some releases
print the process table in the same output).  The parsers below read cells by
their position inside a row instead of relying on column widths, which is what
makes them survive those variations.

Nothing here imports torch_npu; ``query()`` falls back to the SMI text and
returns empty structures on a host without an NPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import env

ROW = re.compile(r'^\s*\|(.*)\|\s*$')
SEP = re.compile(r'^[\s+\-=|]+$')
NUM_PAIR = re.compile(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)')


@dataclass
class Npu:
    id: int
    name: str = ''
    health: str = ''
    power_w: float | None = None
    temp_c: float | None = None
    aicore_pct: float | None = None
    mem_used_mib: float | None = None
    mem_total_mib: float | None = None
    hbm_used_mib: float | None = None
    hbm_total_mib: float | None = None
    bus_id: str = ''
    processes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            'id': self.id, 'index': self.id, 'name': self.name, 'health': self.health, 'power_w': self.power_w,
            'temp_c': self.temp_c, 'aicore_pct': self.aicore_pct,
            'mem_used_mib': self.mem_used_mib, 'mem_total_mib': self.mem_total_mib,
            'hbm_used_mib': self.hbm_used_mib, 'hbm_total_mib': self.hbm_total_mib,
            'bus_id': self.bus_id, 'processes': list(self.processes),
        }


def _cells(line: str) -> list[str]:
    m = ROW.match(line)
    body = m.group(1) if m else line
    return [c.strip() for c in body.split('|')]


def _num(text: str) -> float | None:
    m = re.search(r'-?\d+(?:\.\d+)?', text or '')
    return float(m.group(0)) if m else None


def _pair(text: str) -> tuple[float | None, float | None]:
    m = NUM_PAIR.search(text or '')
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def parse_info(text: str) -> dict:
    """``npu-smi info`` -> {'npus': [...], 'processes': [...], 'version': str|None}.

    The table has three logical columns (NPU/Name, Health/Bus-Id, the numeric
    block); some releases add pipe separators between the numbers.  Both are
    read: the first cell is split into id + name, the numeric block by position
    or by number pairs.
    """
    npus: dict[int, Npu] = {}
    processes: list[dict] = []
    in_process_table = False
    current: int | None = None
    version = None
    for raw in (text or '').splitlines():
        if not version:
            m = re.search(r'npu-smi\s+\S+\s+Version:\s*(\S+)', raw)
            version = m.group(1) if m else None
        if 'Process id' in raw or 'Process name' in raw:
            in_process_table = True
            continue
        if 'NPU' in raw and 'Name' in raw and 'Health' in raw:
            in_process_table = False
            continue
        if SEP.match(raw) or not raw.strip().startswith('|'):
            continue
        cells = _cells(raw)
        if len(cells) < 3:
            continue
        head = cells[0].split()
        if not head or not head[0].isdigit():
            continue
        npu_id = int(head[0])
        name = ' '.join(head[1:]).strip()
        if not name and current is not None:
            npu_id = current                                # the chip row's second line repeats the chip id, not the NPU id
        if in_process_table:
            if len(cells) >= 4 and cells[1].strip().isdigit():
                processes.append({'npu': npu_id, 'chip': head[1] if len(head) > 1 else '0',
                                  'pid': int(cells[1]), 'name': cells[2].strip(), 'mem_mib': _num(cells[3])})
            elif len(cells) >= 5 and cells[2].strip().isdigit():
                processes.append({'npu': npu_id, 'chip': cells[1].strip(), 'pid': int(cells[2]),
                                  'name': cells[3].strip(), 'mem_mib': _num(cells[4])})
            continue
        n = npus.setdefault(npu_id, Npu(id=npu_id))
        if name:                                        # chip row
            current = npu_id
            n.name = n.name or name
            n.health = n.health or cells[1].strip()
            nums = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', cells[2])]
            if len(nums) >= 1:
                n.power_w = n.power_w if n.power_w is not None else nums[0]
            if len(nums) >= 2:
                n.temp_c = n.temp_c if n.temp_c is not None else nums[1]
            continue
        n.bus_id = n.bus_id or cells[1].strip()         # second row
        block = cells[2]
        pairs = NUM_PAIR.findall(block)
        lead = re.match(r'\s*(-?\d+(?:\.\d+)?)', block)
        if lead:
            n.aicore_pct = n.aicore_pct if n.aicore_pct is not None else float(lead.group(1))
        if pairs:
            m_used, m_total = (float(pairs[0][0]), float(pairs[0][1]))
            if n.mem_used_mib is None:
                n.mem_used_mib, n.mem_total_mib = m_used, m_total
            if len(pairs) > 1:
                n.hbm_used_mib, n.hbm_total_mib = float(pairs[1][0]), float(pairs[1][1])
    for p in processes:
        if p['npu'] in npus:
            npus[p['npu']].processes.append(p)
    return {'npus': [n.as_dict() for n in sorted(npus.values(), key=lambda x: x.id)],
            'processes': processes, 'version': version}


def parse_kv(text: str) -> dict:
    """``npu-smi info -t board|health -i N`` key/value output."""
    out = {}
    for line in (text or '').splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            if k.strip():
                out[k.strip().lower().replace(' ', '_')] = v.strip()
    return out


def parse_topo(text: str) -> dict:
    """``npu-smi info -t topo`` -> {'links': {'0-1': 'HCCS'}, 'intra': 'hccs'|'pcie'|'mixed', 'full_mesh': bool}."""
    def ident(cell: str) -> int | None:
        m = re.fullmatch(r'(?i)(?:npu)?\s*(\d+)', str(cell).strip())
        return int(m.group(1)) if m else None

    links: dict[str, str] = {}
    header: list[int] | None = None
    for raw in (text or '').splitlines():
        if not raw.strip():
            continue
        cells = _cells(raw) if raw.strip().startswith('|') else raw.split()
        if not cells:
            continue
        ids = [ident(c) for c in cells]
        is_header = (header is None and ids[0] is not None and sum(x is not None for x in ids) >= 2
                     and not any(str(c).strip().upper() in ('HCCS', 'HCCL', 'PCIE', 'SYS', 'SIO', 'UB', 'X')
                                 for c in cells[1:]))
        if is_header:
            header = [x for x in ids if x is not None]     # the column ids
            continue
        a = ident(cells[0])
        if a is None:
            continue
        for j, cell in enumerate(cells[1:]):
            if header is None or j >= len(header):
                break
            b = header[j]
            val = cell.strip().upper()
            if val in ('HCCS', 'HCCL', 'PCIE', 'SYS', 'SIO', 'UB'):
                links[f'{a}-{b}'] = val
    vals = set(links.values())
    if not vals:
        intra = None
    elif vals <= {'HCCS', 'HCCL', 'UB'}:
        intra = 'hccs'
    elif 'HCCS' in vals or 'UB' in vals:
        intra = 'mixed'
    else:
        intra = 'pcie'
    n = len({int(k.split('-')[0]) for k in links}) if links else 0
    full = bool(links) and all(f'{i}-{j}' in links for i in range(n) for j in range(n) if i != j)
    return {'links': links, 'intra': intra, 'full_mesh': full, 'count': n}


def parse_processes(text: str) -> list[dict]:
    """``npu-smi info -t proc-mem -i N`` rows -> [{npu, chip, pid, name, mem_mib}]."""
    out = []
    for raw in (text or '').splitlines():
        if not raw.strip().startswith('|'):
            continue
        cells = _cells(raw)
        if len(cells) >= 5 and cells[0].isdigit() and cells[2].strip().isdigit():
            out.append({'npu': int(cells[0]), 'chip': cells[1].strip(), 'pid': int(cells[2]),
                        'name': cells[3].strip(), 'mem_mib': _num(cells[4])})
    return out


def query(device_id: int | None = None) -> dict:
    """Live ``npu-smi info`` (one call), parsed. Empty when the tool is absent."""
    if not env.has_smi():
        return {'npus': [], 'processes': [], 'version': None}
    cmd = ['npu-smi', 'info']
    if device_id is not None:
        cmd += ['-i', str(device_id)]
    return parse_info(env.run(cmd))


def board(device_id: int = 0) -> dict:
    if not env.has_smi():
        return {}
    return parse_kv(env.run(['npu-smi', 'info', '-t', 'board', '-i', str(device_id)]))


def health(device_id: int = 0) -> dict:
    if not env.has_smi():
        return {}
    return parse_kv(env.run(['npu-smi', 'info', '-t', 'health', '-i', str(device_id)]))


def topo() -> dict:
    if not env.has_smi():
        return {'links': {}, 'intra': None, 'full_mesh': False, 'count': 0}
    return parse_topo(env.run(['npu-smi', 'info', '-t', 'topo']))


def snapshot(devices=None) -> dict | None:
    """The project runner's snapshot shape, restricted to ``devices`` (indices or 'npu-N' prefixes)."""
    info = query()
    npus = info.get('npus') or []
    if not npus:
        return None
    keep = []
    for n in npus:
        if devices is None or n['id'] in devices or f'npu-{n["id"]}' in devices:
            keep.append(n)
    procs = [p for p in info.get('processes') or [] if any(p['npu'] == n['id'] for n in keep)]
    return {'gpus': [{'index': n['id'], 'uuid': n['bus_id'] or f'npu-{n["id"]}',
                      'mem_used_mib': n['hbm_used_mib'] if n['hbm_used_mib'] is not None else n['mem_used_mib'],
                      'util': n['aicore_pct'], 'persistence': 'n/a', 'ecc': None,
                      'temp_c': n['temp_c'], 'health': n['health'], 'name': n['name']} for n in keep],
            'apps': len(procs), 'apps_mib': sum(p.get('mem_mib') or 0 for p in procs)}
