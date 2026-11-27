"""The Ascend kernel ecosystem this edition knows how to call.

One entry per repository, with what it gives a training job and how to reach
it from Python.  ``check()`` imports the package (never the repository) and
reports the version, so a receipt can say which kernel implementation a run
actually used.  ``install`` is a hint, not something InfraRSI runs by itself.

The list is intentionally curated: a library is here when it exposes kernels a
*trainer* can call (forward and backward, or a drop-in module), not when it is
only an inference serving project.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KernelLibrary:
    name: str
    repo: str
    license: str
    provides: tuple[str, ...]
    import_name: str | None
    install: str
    ops: tuple[str, ...] = ()
    notes: str = ''
    tags: tuple[str, ...] = field(default_factory=tuple)

    def available(self) -> bool:
        if not self.import_name:
            return False
        try:
            return importlib.util.find_spec(self.import_name) is not None
        except (ImportError, ValueError):
            return False

    def version(self) -> str | None:
        if not self.import_name:
            return None
        try:
            mod = __import__(self.import_name)
            return getattr(mod, '__version__', None)
        except Exception:                                # noqa: BLE001 - a broken install is "not available"
            return None

    def as_dict(self) -> dict:
        return {'name': self.name, 'repo': self.repo, 'license': self.license, 'provides': list(self.provides),
                'import': self.import_name, 'install': self.install, 'ops': list(self.ops), 'notes': self.notes,
                'tags': list(self.tags), 'available': self.available(), 'version': self.version()}


CATALOG: tuple[KernelLibrary, ...] = (
    KernelLibrary(
        name='torch_npu (Ascend for PyTorch)',
        repo='https://github.com/Ascend/pytorch',
        license='BSD-3-Clause',
        provides=('the npu device', 'fused operators', 'HCCL collectives'),
        import_name='torch_npu',
        install='pip install torch_npu==<the torch version>',
        ops=('npu_rms_norm', 'npu_add_rms_norm', 'npu_swiglu', 'npu_rotary_mul', 'npu_fusion_attention',
             'npu_prompt_flash_attention', 'npu_cross_entropy_loss', 'npu_apply_adam_w', 'npu_moe_distribute_dispatch',
             'npu_moe_distribute_combine', 'npu_grouped_matmul', 'npu_scaled_mm'),
        notes='The baseline: every other library below is optional. The fused ops are the first thing to try and '
              'the fallback when a custom kernel fails its numerics gate.',
        tags=('vendor', 'fused-ops', 'required'),
    ),
    KernelLibrary(
        name='triton-ascend-kernels',
        repo='https://gitcode.com/Ascend/triton-ascend-kernels',
        license='MIT',
        provides=('Triton kernels written for Ascend', 'benchmarks'),
        import_name='triton_ascend_kernels',
        install='git clone https://gitcode.com/Ascend/triton-ascend-kernels.git && pip install -e .',
        ops=('flash attention', 'KDA', 'convert_index', 'normalization', 'activation'),
        notes='The official Triton-Ascend operator collection; also the best reference for how a Triton kernel '
              'must be written for the A2/A3 memory hierarchy (L1/UB, 32-byte alignment).',
        tags=('triton', 'attention', 'official'),
    ),
    KernelLibrary(
        name='PTO kernels',
        repo='https://github.com/huawei-csl/pto-kernels',
        license='BSD-3-Clause-Clear',
        provides=('tile-level kernels on PTO-ISA', 'torch_npu operators'),
        import_name='pto_kernels',
        install='pip install pto-kernels',
        ops=('pto_abs', 'pto_swiglu', 'pto_flash_attention', 'pto_simple_matmul', 'pto_tri_inv_*',
             'pto_gdn_chunk_cumsum', 'pto_gdn_chunk_h', 'pto_gdn_chunk_o', 'pto_gdn_wy_fast', 'pto_kda_*'),
        notes='Built on PTO-ISA, the tile-level virtual ISA of CANN. Good for linear-attention building blocks '
              '(GDN/KDA) and small fused elementwise kernels; ships prebuilt wheels for x86_64/aarch64.',
        tags=('pto', 'linear-attention', 'elementwise'),
    ),
    KernelLibrary(
        name='DeepGEMM-Ascend',
        repo='https://github.com/deepseek-ai/DeepGEMM-Ascend',
        license='MIT',
        provides=('BF16/FP8/FP4 GEMM', 'grouped GEMM', 'MQA logits', 'MegaMoE'),
        import_name='deep_gemm',
        install='git clone --recursive https://github.com/deepseek-ai/DeepGEMM-Ascend.git && pip install . --no-build-isolation',
        ops=('bf16_gemm_nn', 'fp8_gemm_nt', 'fp8_fp4_gemm_nn', 'm_grouped_gemm', 'mqa_logits', 'mega_moe'),
        notes='API-compatible with DeepGEMM. Validated on Ascend 950; needs CANN 9.3+ and Bisheng. The scaling '
              'factor layout differs from NVIDIA (UE8M0 pairs packed into int16, MN-major).',
        tags=('gemm', 'fp8', 'moe', '950'),
    ),
    KernelLibrary(
        name='TileLang-Ascend',
        repo='https://github.com/tile-ai/tilelang-ascend',
        license='MIT',
        provides=('a tile DSL for Ascend', 'GEMM/attention/softmax examples'),
        import_name='tilelang',
        install='pip install tilelang-*.whl  (or ./build_wheel_ascend.sh)',
        ops=('gemm', 'flash_attention', 'sparse_flash_attention', 'softmax', 'normalization', 'cross_entropy',
             'dispatch_combine', 'aclgraph'),
        notes='Pythonic tile programming with Cube/Vector scope separation; the same kernel source can be tuned '
              'per shape. Examples are the reference implementations for new fused kernels.',
        tags=('dsl', 'attention', 'gemm'),
    ),
    KernelLibrary(
        name='CATLASS',
        repo='https://github.com/ascend-catlass/catlass',
        license='CANN Open Software License Agreement v2',
        provides=('C++ GEMM templates', 'epilogue templates', 'grouped GEMM'),
        import_name=None,
        install='git clone https://github.com/ascend-catlass/catlass.git && bash scripts/build.sh -DCATLASS_ARCH=2201',
        ops=('basic_matmul', 'group_gemm', 'quant_matmul', 'microscale_gemm'),
        notes='The Ascend counterpart of CUTLASS: header-only templates for matrix-multiply kernels. Use it when '
              'a GEMM shape is fixed and aclnn leaves performance on the table.',
        tags=('cpp', 'gemm', 'templates'),
    ),
    KernelLibrary(
        name='Mojo Opset',
        repo='https://github.com/Ascend/mojo_opset',
        license='Apache-2.0',
        provides=('drop-in LLM/multimodal ops', 'Triton-x backend for 910B/C'),
        import_name='mojo_opset',
        install='pip install -e .  (see the repository README)',
        ops=('MojoRMSNorm', 'MojoLayerNorm', 'MojoResidualAddRMSNorm', 'MojoSwiGlu', 'MojoRotaryEmb', 'MojoSdpa',
             'MojoLinearCrossEntropyFunc', 'MojoMoEGate'),
        notes='A module-level replacement layer: swap nn.RMSNorm / SwiGLU for MojoRMSNorm / MojoSwiGlu and select '
              'the backend with MOJO_BACKEND=ttx. Good when a whole model, not a single kernel, is the unit.',
        tags=('modules', 'triton', 'drop-in'),
    ),
    KernelLibrary(
        name='FlashKDA',
        repo='https://github.com/Ascend/FlashKDA',
        license='See repository',
        provides=('Kimi Delta Attention kernels',),
        import_name=None,
        install='see the repository README',
        ops=('kda_fwd', 'kda_bwd'),
        notes='Attention variants outside the transformer: KDA/GDN-style linear attention, where aclnn has no '
              'fused op yet.',
        tags=('attention', 'linear-attention'),
    ),
    KernelLibrary(
        name='FlashQLA',
        repo='https://github.com/Ascend/FlashQLA',
        license='See repository',
        provides=('linear attention kernels on TileLang',),
        import_name=None,
        install='see the repository README',
        ops=('qla_fwd', 'qla_bwd'),
        notes='TileLang-based linear attention; a source of kernels to adapt rather than a pip dependency.',
        tags=('attention', 'tilelang'),
    ),
    KernelLibrary(
        name='DeepSelect',
        repo='https://github.com/Ascend/DeepSelect',
        license='See repository',
        provides=('TopK kernels for DeepSeek Sparse Attention', 'samplers'),
        import_name=None,
        install='see the repository README',
        ops=('topk_selector', 'sampler'),
        notes='Sparse-attention TopK and sampling kernels; useful when a model spends its time in the indexer.',
        tags=('topk', 'sparse-attention'),
    ),
    KernelLibrary(
        name='ascend-kernel (Chitu)',
        repo='https://github.com/QingCheng-AI/ascend-kernel',
        license='Apache-2.0',
        provides=('Ascend C kernels for the Chitu inference engine',),
        import_name='ascend_kernel',
        install='see the repository README (Ascend C build)',
        ops=('attention', 'norm', 'activation'),
        notes='A compact Ascend C kernel project: a good template for adding a custom operator to torch_npu.',
        tags=('ascend-c', 'inference'),
    ),
    KernelLibrary(
        name='mamba-ascendc',
        repo='https://github.com/So-cean/mamba-ascendc',
        license='Apache-2.0',
        provides=('Mamba-2 SSD forward/backward Ascend C operators',),
        import_name='ascend_kernel',
        install='bash scripts/build_mamba_ascendc_wheel.sh && pip install dist/*.whl --no-deps',
        ops=('mamba2_ssd_fwd', 'mamba2_ssd_bwd'),
        notes='A worked example of a training-capable custom op: AscendC Cube/Vector kernels, autograd wrapper, '
              'Triton-Ascend comparison and A100 baseline.',
        tags=('ascend-c', 'ssm', 'training'),
    ),
    KernelLibrary(
        name='HuggingFace kernels (NPU backend)',
        repo='https://github.com/huggingface/kernels',
        license='Apache-2.0',
        provides=('the kernels package with an NPU backend',),
        import_name='kernels',
        install='pip install kernels==0.12.0',
        ops=('kernel hub operators',),
        notes='kernels 0.11-0.12 support NPU; the hub operator must ship an Ascend build. Check '
              '`kernels.get_kernel(...).backend` before trusting it.',
        tags=('hub', 'community'),
    ),
)


def by_name(name: str) -> KernelLibrary | None:
    n = (name or '').lower().replace('_', '-')
    return next((k for k in CATALOG if n in (k.name.lower().replace('_', '-'), k.import_name)), None)


def available() -> list[KernelLibrary]:
    return [k for k in CATALOG if k.available()]


def report() -> list[dict]:
    return [k.as_dict() for k in CATALOG]
