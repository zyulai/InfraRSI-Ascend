"""Ascend interconnect facts: HCCS inside a node, RoCE/UB between nodes.

The node-local probe is ``npu-smi info -t topo`` (HCCS/PCIE cells).  A
multi-node job's fabric is read from the environment the job runs with
(``HCCL_INTRA_ROCE_ENABLE``, ``HCCL_IB_HCA`` ...) and from ``/sys/class/infiniband``
the same way the NVIDIA link probe does it, so a receipt can say whether an
all-reduce went over HCCS, RoCE or the UB fabric.
"""
from __future__ import annotations

import re

from . import env, smi

# A2/A3: 8 cards in a full mesh over HCCS at 392 GB/s per link; 910_93 (A3
# super-node) uses the UB switch at 784 GB/s.  These are the numbers the
# transfer judgment compares, not a measurement.
HCCS_GBPS = {'910B': 392.0, '910_93': 784.0, '950': 2000.0}
ROCE_GBPS = 50.0


def intra_link() -> dict:
    """{'intra': 'hccs'|'pcie'|'mixed'|None, 'full_mesh': bool, 'links': {...}, 'count': n}."""
    return smi.topo()


def inter_link(nodes: int = 1) -> str | None:
    """'roce' | 'ub' | 'tcp' | None for a multi-node job, from the host's devices and env."""
    if nodes <= 1:
        return None
    if env.run('ls /sys/class/infiniband 2>/dev/null').strip():
        layers = env.run('cat /sys/class/infiniband/*/ports/*/link_layer 2>/dev/null').strip().lower()
        if 'ethernet' in layers:
            return 'roce'
        if 'infiniband' in layers:
            return 'ib'
    # a UB fabric has no ib device; the HCCL env names it
    text = env.run("env | grep -E '^HCCL_(INTRA_ROCE_ENABLE|IB_HCA|UB_ENABLE)='")
    if re.search(r'HCCL_UB_ENABLE=1', text):
        return 'ub'
    if re.search(r'HCCL_INTRA_ROCE_ENABLE=1', text) or re.search(r'HCCL_IB_HCA=\S+', text):
        return 'roce'
    return 'tcp'


def summary(nodes: int = 1) -> dict:
    t = intra_link()
    cls = env.soc_class(env.soc_version()) or ''
    gbps = next((v for k, v in HCCS_GBPS.items() if cls.startswith(k)), None)
    return {'intra': t.get('intra'), 'full_mesh': t.get('full_mesh'), 'links': t.get('links') or {},
            'link_gbps': gbps, 'inter': inter_link(nodes)}
