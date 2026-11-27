"""Source scan and safe rewrites."""
from infrarsi.ascend import patch

SRC = '''import torch
import torch.nn.functional as F


def block(x, w):
    h = F.rms_norm(x, (x.shape[-1],), w, 1e-6)
    h = F.gelu(h)
    return F.cross_entropy(h, x)


def attn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)
'''


def test_scan_finds_ops():
    kinds = {s.kind for s in patch.scan(SRC)}
    assert {'rms_norm', 'gelu', 'cross_entropy', 'attention'} <= kinds


def test_apply_rms_norm_and_gelu():
    new, applied = patch.apply(SRC)
    assert applied
    assert 'ascend_ops.rms_norm(x, w, 1e-6)' in new
    assert 'ascend_ops.gelu(h' in new
    assert 'from infrarsi.ascend.kernels import fused_ops as ascend_ops' in new
    # cross_entropy is only a suggestion (weight=/reduction= must be checked)
    assert 'F.cross_entropy' in new


def test_apply_is_idempotent():
    once, _ = patch.apply(SRC)
    twice, applied = patch.apply(once)
    assert twice == once
    assert applied == []


def test_diff_is_unified():
    new, _ = patch.apply(SRC)
    d = patch.diff(SRC, new)
    assert d.startswith('--- a/train.py') and '+++ b/train.py' in d
