"""Ascend launch knobs: the environment variables that change NPU training speed.

Each entry is what ``infrarsi init`` can draft as an ``env`` knob and what the
mechanism registry needs to recognise.  Values are the ones CANN/torch_npu
document or that are common in Ascend training recipes; the notes say what a
change costs, because on an NPU most of these trade HBM for communication or
launch overhead.

This is deliberately a *catalogue*, not a policy: `default` is the value the
runtime uses when the variable is unset, and `choices` never includes a value
that changes what the job computes.
"""
from __future__ import annotations

# name -> {type, choices, default, mechanism, notes}
LAUNCH_ENV: dict[str, dict] = {
    'TASK_QUEUE_ENABLE': {
        'type': 'str', 'choices': ['0', '1', '2'], 'default': '1', 'mechanism': 'npu_task_queue',
        'notes': 'torch_npu task-queue offload: 0 = synchronous, 1 = the default partial offload, 2 = full '
                 'offload (lower host overhead, more device memory for the queue).'},
    'COMBINED_ENABLE': {
        'type': 'str', 'choices': ['0', '1'], 'default': '0', 'mechanism': 'npu_combined_ops',
        'notes': 'combine small elementwise/optimizer kernels into one launch; helps short kernels, can hurt '
                 'when the fused shape is not the hot one.'},
    'MULTI_STREAM_MEMORY_REUSE': {
        'type': 'str', 'choices': ['0', '1'], 'default': '0', 'mechanism': 'npu_stream_memory_reuse',
        'notes': 'reuse memory across streams instead of holding a separate pool per stream: saves HBM when '
                 'the job uses several streams, adds a stream dependency.'},
    'PYTORCH_NPU_ALLOC_CONF': {
        'type': 'str', 'choices': ['', 'expandable_segments:True', 'garbage_collection_threshold:0.9',
                                   'expandable_segments:True,garbage_collection_threshold:0.9'],
        'default': '', 'mechanism': 'npu_allocator',
        'notes': 'caching allocator: expandable segments reduce fragmentation for variable shapes; the GC '
                 'threshold trades HBM for allocator time.'},
    'HCCL_BUFFSIZE': {
        'type': 'str', 'choices': ['100', '200', '400', '800'], 'default': '200', 'mechanism': 'hccl_buffsize',
        'notes': 'HCCL buffer in MB per rank: larger buffers cut small-message round trips and cost HBM '
                 '(200 MB x 8 ranks is already 1.6 GB).'},
    'HCCL_ALGO': {
        'type': 'str', 'choices': ['', 'Ring', 'Tree', 'HalvingDoubling'], 'default': '', 'mechanism': 'hccl_algo',
        'notes': 'collective algorithm; empty lets HCCL choose by message size and topology.'},
    'HCCL_DETERMINISTIC': {
        'type': 'str', 'choices': ['false', 'true'], 'default': 'false', 'mechanism': 'hccl_deterministic',
        'notes': 'deterministic reduction order: bitwise repeatable, usually slower on large reductions.'},
    'HCCL_INTRA_ROCE_ENABLE': {
        'type': 'str', 'choices': ['0', '1'], 'default': '0', 'mechanism': 'hccl_intra_roce',
        'notes': 'route intra-node collectives over RoCE instead of HCCS; a workaround for a broken HCCS '
                 'link, never faster when HCCS is healthy.'},
    'HCCL_INTRA_PCIE_ENABLE': {
        'type': 'str', 'choices': ['0', '1'], 'default': '0', 'mechanism': 'hccl_intra_pcie',
        'notes': 'force PCIe for intra-node collectives; for cards without HCCS, much slower than HCCS.'},
    'CPU_AFFINITY_CONF': {
        'type': 'str', 'choices': ['', '1', '2'], 'default': '', 'mechanism': 'npu_cpu_affinity',
        'notes': 'bind the process to its NUMA node: 1 = bind to the NPU-local NUMA node, 2 = bind to the '
                 'NUMA node and reserve the management CPUs.'},
    'ASCEND_LAUNCH_BLOCKING': {
        'type': 'str', 'choices': ['0', '1'], 'default': '0', 'mechanism': 'npu_launch_blocking',
        'notes': 'synchronize after every kernel launch: debugging only, it removes all overlap.'},
    'INF_NAN_MODE_ENABLE': {
        'type': 'str', 'choices': ['0', '1'], 'default': '0', 'mechanism': 'npu_inf_nan_mode',
        'notes': '0 keeps the overflow-check mode that makes Inf/NaN visible; 1 trades that check for speed.'},
    'ACLNN_CACHE_LIMIT': {
        'type': 'str', 'choices': ['1000', '10000', '100000'], 'default': '10000', 'mechanism': 'npu_aclnn_cache',
        'notes': 'number of aclnn kernel descriptors kept cached; a larger cache avoids re-compilation of '
                 'many small shapes at the cost of host memory.'},
}

# Knobs that are not env vars but code-level mechanisms this edition knows how to
# detect or apply (see infrarsi.ascend.kernels and infrarsi.ascend.patch).
CODE_KNOBS: dict[str, dict] = {
    'npu_fusion_ops': {
        'mechanism': 'npu_fusion_ops', 'states': ['off', 'on'],
        'notes': 'replace unfused RMSNorm / SwiGLU / RoPE / cross-entropy with torch_npu fused ops '
                 '(npu_rms_norm, npu_swiglu, npu_rotary_mul, npu_cross_entropy_loss).'},
    'npu_attention': {
        'mechanism': 'npu_attention_backend', 'states': ['math', 'fusion', 'prompt_flash', 'flash'],
        'notes': 'torch_npu.npu_fusion_attention (fusion) vs npu_prompt_flash_attention for long context; '
                 'math is the unfused reference path.'},
    'npu_gemm_library': {
        'mechanism': 'npu_gemm_backend', 'states': ['torch', 'deep_gemm', 'catlass', 'triton'],
        'notes': "matmul backend: torch_npu's aclnn GEMM, DeepGEMM-Ascend, CATLASS templates or a Triton-Ascend "
                 "kernel; shapes must meet each library's alignment rules."},
    'npu_moe_dispatch': {
        'mechanism': 'npu_moe_dispatch', 'states': ['alltoall', 'distribute'],
        'notes': 'MoE token dispatch: alltoall through HCCL vs npu_moe_distribute_dispatch/combine fused '
                 'dispatch+combine.'},
}


def env_choices() -> dict[str, list]:
    """{env: choices} for the manifest drafter (only variables with >= 2 values)."""
    return {k: list(v['choices']) for k, v in LAUNCH_ENV.items() if len(v.get('choices') or []) >= 2}


def env_defaults() -> dict[str, str]:
    return {k: v['default'] for k, v in LAUNCH_ENV.items() if v.get('default') not in (None, '')}


def mechanism_of(env_name: str) -> str | None:
    return (LAUNCH_ENV.get(env_name) or {}).get('mechanism')


def notes_of(env_name: str) -> str:
    return (LAUNCH_ENV.get(env_name) or {}).get('notes', '')
