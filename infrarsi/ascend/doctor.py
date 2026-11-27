"""Read-only NPU doctor: versions, health, topology, contention.

`report()` is what `infrarsi ascend doctor` prints.  It never raises on a host
without an NPU: every probe degrades to an empty value and the report says so.
"""
from __future__ import annotations

import os

from . import chips, env, knobs, smi, topology


def health_warnings() -> list[str]:
    out = []
    info = smi.query()
    for n in info.get('npus') or []:
        tag = f'NPU {n["id"]} ({n.get("name") or "?"})'
        if n.get('health') and n['health'].upper() != 'OK':
            out.append(f'{tag}: health {n["health"]}')
        if n.get('temp_c') is not None and n['temp_c'] >= 85:
            out.append(f'{tag}: {n["temp_c"]} C (throttling risk)')
        if n.get('hbm_total_mib') and n.get('hbm_used_mib'):
            pct = 100 * n['hbm_used_mib'] / n['hbm_total_mib']
            if pct >= 90:
                out.append(f'{tag}: HBM {pct:.0f}% used before the run')
        if n.get('power_w') is not None and n['power_w'] >= 400:
            out.append(f'{tag}: {n["power_w"]} W')
    if os.environ.get('ASCEND_LAUNCH_BLOCKING') in ('1', 'true'):
        out.append('ASCEND_LAUNCH_BLOCKING=1: every kernel is synchronous; a timing measured now is not the job\'s')
    if os.environ.get('HCCL_INTRA_ROCE_ENABLE') in ('1', 'true'):
        t = topology.intra_link()
        if t.get('intra') == 'hccs':
            out.append('HCCL_INTRA_ROCE_ENABLE=1 while npu-smi reports HCCS links: collectives are routed the slow way')
    return out


def contention(snapshot: dict | None, when: str) -> list[str]:
    """Other processes on the job's NPUs before/after a run (the NPU analogue of gpu_contention)."""
    if not snapshot:
        return []
    busy = [g for g in snapshot.get('gpus') or []
            if (g.get('mem_used_mib') or 0) > 1024 or (g.get('util') or 0) > 10]
    if not busy and not snapshot.get('apps'):
        return []
    gib = sum(g.get('mem_used_mib') or 0 for g in busy) / 1024
    return [f'{len(busy)} NPU(s) in use {when} the run ({gib:.1f} GiB'
            + (f', {snapshot["apps"]} process(es)' if snapshot.get('apps') else '') + ')']


def report() -> dict:
    family = env.detect_family()
    info = smi.query()
    t = topology.intra_link()
    card = chips.current()
    return {
        'family': family,
        'toolkit': str(env.toolkit_home()) if env.toolkit_home() else None,
        'driver': str(env.driver_home()) if env.driver_home() else None,
        'versions': env.versions(),
        'soc': env.soc_version(),
        'card': card.as_dict() if card else None,
        'npus': info.get('npus') or [],
        'smi_version': info.get('version'),
        'topology': {'intra': t.get('intra'), 'full_mesh': t.get('full_mesh'), 'count': t.get('count'),
                     'inter': topology.inter_link(2)},
        'knobs': {k: os.environ.get(k) for k in knobs.LAUNCH_ENV if os.environ.get(k) is not None},
        'warnings': health_warnings(),
    }
