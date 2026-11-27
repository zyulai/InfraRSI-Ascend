"""Find the ops in a training script that an Ascend kernel can replace.

``scan`` is a source-level detector: it reports what it found and, for the
rewrites that are drop-in (same semantics, same argument order), the exact
replacement line.  ``apply`` writes only those rewrites and inserts the one
import they need.  Everything else is a *suggestion* with the reason it needs
a human: an attention call needs the head count, a chunk+SwiGLU pair needs the
concatenated tensor, and a custom autograd function needs its backward.

The detector never imports the job.  It reads text, which is what makes it
safe to run on a repository before any dependency is installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

IMPORT = 'from infrarsi.ascend.kernels import fused_ops as ascend_ops'

# (kind, regex, confidence, replacement template or None, reason)
PATTERNS = (
    ('rms_norm', re.compile(r'(?<![\w.])F\.rms_norm\(\s*(?P<x>[^,()]+?)\s*,\s*(?P<shape>\([^)]*\)|[^,()]+?)\s*,'
                            r'\s*(?P<w>[^,()]+?)\s*(?:,\s*(?P<eps>[^,()]+?)\s*)?\)'), 'high',
     'ascend_ops.rms_norm({x}, {w}, {eps})', 'F.rms_norm -> npu_rms_norm'),
    ('gelu', re.compile(r'(?<![\w.])F\.gelu\(\s*(?P<x>[^,()]+?)\s*(?:,\s*(?P<approx>[^,()]+?)\s*)?\)'), 'high',
     'ascend_ops.gelu({x}, {approx})', 'F.gelu -> npu_gelu / npu_fast_gelu'),
    ('cross_entropy', re.compile(r'(?<![\w.])F\.cross_entropy\(\s*(?P<args>[^()]*)\)'), 'medium',
     None, 'F.cross_entropy -> npu_cross_entropy_loss (check weight= and reduction=)'),
    ('attention', re.compile(r'(?<![\w.])F\.scaled_dot_product_attention\(|(?<![\w.])torch\.nn\.functional\.'
                             r'scaled_dot_product_attention\('), 'low', None,
     'SDPA -> npu_fusion_attention (needs num_heads and the BNSD layout)'),
    ('swiglu', re.compile(r'(?<![\w.])F\.silu\(\s*(?P<a>[^()]+?)\s*\)\s*\*'), 'low', None,
     'silu(a) * b -> npu_swiglu on the concatenated [gate | up] tensor'),
    ('rms_norm_module', re.compile(r'(?<![\w.])nn\.RMSNorm\('), 'low', None,
     'nn.RMSNorm -> MojoRMSNorm (MOJO_BACKEND=ttx) or a fused_ops.rms_norm forward'),
)


@dataclass
class Suggestion:
    kind: str
    line: int
    old: str
    new: str | None
    confidence: str
    reason: str
    replaceable: bool = field(default=False)

    def as_dict(self) -> dict:
        return {'kind': self.kind, 'line': self.line, 'old': self.old, 'new': self.new,
                'confidence': self.confidence, 'reason': self.reason, 'replaceable': self.replaceable}


def scan(source: str) -> list[Suggestion]:
    out: list[Suggestion] = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        for kind, rx, conf, tmpl, reason in PATTERNS:
            m = rx.search(line)
            if not m:
                continue
            new = None
            if tmpl:
                groups = {k: (v.strip() if v else None) for k, v in m.groupdict().items()}
                if kind == 'gelu':
                    groups['approx'] = groups.get('approx') or "'none'"
                if kind == 'rms_norm':
                    groups['eps'] = groups.get('eps') or '1e-6'
                try:
                    new = line[:m.start()] + tmpl.format(**groups) + line[m.end():]
                except (KeyError, IndexError):
                    new = None
            out.append(Suggestion(kind, i, line, new, conf, reason, replaceable=new is not None))
            break                                            # one suggestion per line
    return out


def apply(source: str, suggestions: list[Suggestion] | None = None) -> tuple[str, list[Suggestion]]:
    """Apply the replaceable suggestions, adding the import once.  Returns (new source, applied)."""
    sug = [s for s in (scan(source) if suggestions is None else suggestions) if s.replaceable and s.new]
    if not sug:
        return source, []
    lines = source.splitlines()
    for s in sorted(sug, key=lambda x: -x.line):
        if not s.new or lines[s.line - 1] != s.old:          # the file changed under the plan
            continue
        lines[s.line - 1] = s.new
    text = '\n'.join(lines) + ('\n' if source.endswith('\n') else '')
    if IMPORT not in text:
        text = _insert_import(text)
    return text, sug


def _insert_import(text: str) -> str:
    """Insert the import after the last top-level import line, else after the module docstring."""
    lines = text.splitlines()
    last = None
    for i, line in enumerate(lines[:200]):
        if re.match(r'^(?:from\s+\S+\s+import|import\s+\S+)', line):
            last = i
    if last is not None:
        return '\n'.join(lines[:last + 1] + ['', IMPORT] + lines[last + 1:]) + ('\n' if text.endswith('\n') else '')
    return IMPORT + '\n\n' + text


def plan(source: str) -> dict:
    """The scan, grouped by op, plus the registry's candidate for each."""
    from .kernels import registry
    sug = scan(source)
    by: dict[str, list[dict]] = {}
    for s in sug:
        by.setdefault(s.kind, []).append(s.as_dict())
    return {'found': by, 'total': len(sug),
            'plan': registry.plan([k for k in by if k in registry.OPS])}


def diff(old: str, new: str, name: str = 'train.py') -> str:
    import difflib
    return ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                        fromfile=f'a/{name}', tofile=f'b/{name}'))
