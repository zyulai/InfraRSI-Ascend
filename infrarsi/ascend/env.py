"""Ascend environment facts: toolkit layout, versions, visible devices and SoC names.

Everything here is read-only and safe on a machine without an NPU: the helpers
return ``None``/``{}``/``0`` instead of raising.  ``torch_npu`` is imported lazily
because importing it on a host without the driver aborts the interpreter in some
CANN releases.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

FAMILY_ENV = 'INFRARSI_DEVICE'
ASCEND_ALIASES = ('ascend', 'npu', 'ascend-npu', 'ascend_npu')

# The two layouts CANN has used: the toolkit install root and the "latest" symlink.
TOOLKIT_CANDIDATES = (
    '/usr/local/Ascend/ascend-toolkit/latest',
    '/usr/local/Ascend/latest',
    '/usr/local/Ascend/ascend-toolkit',
)
DRIVER_CANDIDATES = ('/usr/local/Ascend/driver', '/usr/local/Ascend/firmware')

SOC_RE = re.compile(r'(?:Ascend)?\s*(910[ _]?[A-Za-z0-9]*|950[ _]?[A-Za-z0-9]*|310[ _]?[A-Za-z0-9]*)', re.I)


def _env_path(*names: str) -> Path | None:
    for n in names:
        v = os.environ.get(n)
        if v and Path(v).is_dir():
            return Path(v)
    return None


def toolkit_home() -> Path | None:
    """ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME, else the first existing default."""
    found = _env_path('ASCEND_HOME_PATH', 'ASCEND_TOOLKIT_HOME', 'ASCEND_OPP_PATH')
    if found is not None:
        # ASCEND_OPP_PATH points at .../opp; the toolkit root is two levels up
        return found.parent.parent if found.name == 'opp' else found
    return next((Path(p) for p in TOOLKIT_CANDIDATES if Path(p).is_dir()), None)


def driver_home() -> Path | None:
    return _env_path('ASCEND_DRIVER_HOME') or next((Path(p) for p in DRIVER_CANDIDATES if Path(p).is_dir()), None)


def has_toolkit() -> bool:
    return toolkit_home() is not None


def has_smi() -> bool:
    return shutil.which('npu-smi') is not None


def torch_npu():
    """The imported ``torch_npu`` module, or None (never raises)."""
    try:
        import torch_npu  # noqa: F401 - the import registers the 'npu' device
        return torch_npu
    except Exception:                                    # noqa: BLE001 - a missing/broken driver must not kill the caller
        return None


def detect_family() -> str:
    """'ascend' | 'nvidia' | 'cpu' for this process, unless INFRARSI_DEVICE says otherwise."""
    forced = (os.environ.get(FAMILY_ENV) or '').strip().lower()
    if forced in ASCEND_ALIASES:
        return 'ascend'
    if forced in ('nvidia', 'gpu', 'cuda'):
        return 'nvidia'
    if forced in ('cpu', 'none'):
        return 'cpu'
    if has_smi() or has_toolkit() or torch_npu() is not None:
        return 'ascend'
    if shutil.which('nvidia-smi'):
        return 'nvidia'
    return 'cpu'


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def _parse_kv(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip().strip('"')
    return out


def versions() -> dict:
    """Versions a receipt can compare: CANN, driver, firmware, torch_npu, torch and HCCL."""
    out: dict = {}
    tk = toolkit_home()
    if tk is not None:
        for rel in ('version.cfg', 'ascend_toolkit_install.info', 'compiler/version.info'):
            kv = _parse_kv(_read(tk / rel))
            if kv:
                out.setdefault('cann', kv.get('version') or kv.get('Version'))
                out.setdefault('cann_package', kv.get('package_name'))
                break
        # some releases keep the version only in a directory name: .../8.2.RC1/
        if not out.get('cann') and tk.name not in ('latest', 'ascend-toolkit'):
            out['cann'] = tk.name
    drv = driver_home()
    if drv is not None:
        kv = _parse_kv(_read(drv / 'version.info'))
        out.update({k.lower(): v for k, v in kv.items() if k.lower() in ('version', 'driver', 'firmware')})
    tn = torch_npu()
    if tn is not None:
        out['torch_npu'] = getattr(tn, '__version__', None)
        try:
            import torch
            out['torch'] = torch.__version__
            out['npu_backend'] = getattr(getattr(torch, 'version', None), 'npu', None)
        except Exception:                                # noqa: BLE001
            pass
        try:
            out['hccl'] = '.'.join(str(x) for x in tn.npu.hccl.version())   # type: ignore[attr-defined]
        except Exception:                                # noqa: BLE001 - not every release exposes it
            pass
    return {k: v for k, v in out.items() if v}


def visible_devices() -> list[str] | None:
    """ASCEND_RT_VISIBLE_DEVICES as a list; None when unset (all devices)."""
    v = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if v is None or v.strip() == '':
        return None
    return [x.strip() for x in v.split(',') if x.strip()]


def device_count() -> int:
    """Number of NPUs this process can use: torch_npu first, then the SMI table."""
    tn = torch_npu()
    if tn is not None:
        try:
            return int(tn.npu.device_count())
        except Exception:                                # noqa: BLE001
            pass
    from .smi import query
    return len(query().get('npus') or [])


def soc_version(device_id: int = 0) -> str | None:
    """'Ascend910B3' / 'Ascend910_93' / 'Ascend950PR' as the runtime reports it."""
    tn = torch_npu()
    if tn is not None:
        try:
            name = tn.npu.get_device_name(device_id)
            return str(name).split('(')[0].strip() if name else None
        except Exception:                                # noqa: BLE001
            pass
    from .smi import query
    npus = query().get('npus') or []
    return npus[device_id].get('name') if device_id < len(npus) else None


def soc_class(name: str | None) -> str | None:
    """Map an SMI/runtime name to the gpus_ascend.yaml class ('910B3', '910_93', '950PR')."""
    if not name:
        return None
    n = name.replace(' ', '_')
    m = re.search(r'(910[_-]?(?:B[1-4]|B[1-4]C|93|C)|950(?:PR|DT)?|310[BP]?)', n, re.I)
    if not m:
        return None
    s = m.group(1).upper().replace('-', '_')
    if s.startswith('910C'):
        return '910_93'
    if s.startswith('910_'):
        return s
    if s.startswith('950'):
        return s if s != '950' else '950PR'
    return s


def npu_arch(name: str | None) -> str | None:
    """The compiler arch of a SoC: dav-2201 (A2/A3) or dav-3510 (A5)."""
    cls = soc_class(name)
    if cls is None:
        return None
    if cls.startswith('950'):
        return 'dav-3510'
    if cls.startswith('910'):
        return 'dav-2201'
    return None


def run(cmd, timeout: float = 20) -> str:
    """stdout of a short command (a list, or a shell string), '' when it cannot run."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors='replace', timeout=timeout,
                              shell=isinstance(cmd, str)).stdout
    except (OSError, subprocess.SubprocessError):
        return ''
