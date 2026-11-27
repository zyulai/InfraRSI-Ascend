"""npu-smi parsers: fixtures copied from the output shapes of CANN 7.x/8.x."""
from infrarsi.ascend import smi

INFO = """
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc2                 Version: 24.1.rc2                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 92.5        41                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          1234 / 65536         |
+===========================+===============+====================================================+
| 1     910B3               | Warning       | 88.0        45                0    / 0             |
| 0                         | 0000:83:00.0  | 12          0    / 0          2048 / 65536         |
+===========================+===============+====================================================+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 12345         | python                   | 1024                    |
+===========================+===============+====================================================+
"""

TOPO = """
        NPU0    NPU1    NPU2
NPU0    X       HCCS    HCCS
NPU1    HCCS    X       HCCS
NPU2    HCCS    HCCS    X
"""


def test_parse_info_two_cards():
    out = smi.parse_info(INFO)
    assert out['version'] == '24.1.rc2'
    assert [n['id'] for n in out['npus']] == [0, 1]
    first = out['npus'][0]
    assert first['name'] == '910B3'
    assert first['health'] == 'OK'
    assert first['power_w'] == 92.5
    assert first['temp_c'] == 41
    assert first['hbm_used_mib'] == 1234
    assert first['hbm_total_mib'] == 65536
    assert first['bus_id'] == '0000:82:00.0'
    assert out['npus'][1]['aicore_pct'] == 12
    assert out['npus'][1]['health'] == 'Warning'
    assert out['processes'][0]['pid'] == 12345
    assert out['npus'][0]['processes'][0]['mem_mib'] == 1024


def test_parse_topo_full_mesh():
    out = smi.parse_topo(TOPO)
    assert out['intra'] == 'hccs'
    assert out['full_mesh'] is True
    assert out['count'] == 3
    assert out['links']['0-1'] == 'HCCS'


def test_snapshot_shape_matches_executor():
    snap = smi.snapshot()
    # no npu-smi on a CPU test host: None, not an exception
    assert snap is None or set(snap) == {'gpus', 'apps', 'apps_mib'}
