# InfraRSI-Ascend

**InfraRSI 的昇腾（Ascend NPU）版本**：把 CUDA-first 的训练基础设施优化 harness 接到
Ascend 910B / 910_93 / 950PR / 950DT 上，并把昇腾 kernel 生态作为 git submodule 收进来。

> 上游项目：[TokenRhythm/InfraRSI](https://github.com/TokenRhythm/InfraRSI)（Apache-2.0）
> 本仓库是 Ascend edition，`upstream` 分支是未改动的上游快照，`main` 分支是昇腾改造。

## 仓库结构

```text
InfraRSI-Ascend/
├── infrarsi/                     # InfraRSI 本体
│   └── ascend/                   # 昇腾适配层
│       ├── env.py                # CANN/torch_npu 版本、SoC 型号、设备可见性
│       ├── smi.py                # npu-smi info/topo/board/health 解析
│       ├── topology.py           # HCCS / PCIe / RoCE / UB
│       ├── knobs.py              # HCCL / torch_npu 启动参数
│       ├── chips.py              # 芯片数据加载、约束、kernel prompt
│       ├── kernels/              # kernel 目录、op→实现、融合算子、Triton 参考
│       ├── numerics.py           # FP64 数值门禁
│       ├── patch.py              # 源码扫描与安全替换
│       ├── doctor.py             # NPU 健康检查
│       └── cli.py                # infrarsi ascend ...
├── infrarsi/experience/data/
│   ├── gpus_ascend.yaml          # 18 个 Ascend class（传输判断用）
│   └── mechanisms_ascend.yaml    # 昇腾优化机制
├── infrarsi/ascend/data/
│   ├── chips.yaml                # 从 kernel 仓库提取的芯片数据
│   └── chips_public.yaml         # 公开规格补充
├── ascend_chip_extract/          # 芯片数据提取的可复现实验记录
│   ├── sources/                  # 模型读取的仓库 README/文档
│   ├── raw_a.json / raw_b.json   # hy4-flash 原始响应
│   ├── chips_a.json / chips_b.json
│   ├── public_specs.yaml
│   ├── validate_chips.py
│   └── CHIPS.md                  # 18 张卡速查表
├── tests/cpu/ascend/             # 无需 NPU 的单元测试
└── third_party/                  # 昇腾 kernel 仓库（git submodules）
```

## 快速开始

```bash
git clone --recursive git@github.com:zyulai/InfraRSI-Ascend.git
cd InfraRSI-Ascend
pip install -e .

# 查看芯片数据（无需 NPU）
infrarsi ascend chips
infrarsi ascend chips --card 910B2
infrarsi ascend kernels

# 在 NPU 机器上
infrarsi ascend doctor
infrarsi ascend bench --shape 4096x4096

# 生成 kernel 任务描述
infrarsi ascend prompt --op rms_norm --card 950PR-Server
```

已经 clone 但没带 submodule：

```bash
git submodule update --init --recursive --depth 1
```

## third_party submodules

| 目录 | 仓库 | 用途 |
|---|---|---|
| `pto-kernels` | [huawei-csl/pto-kernels](https://github.com/huawei-csl/pto-kernels) | PTO-ISA tile 级算子（SwiGLU、FlashAttention、GDN/KDA） |
| `pto-isa` | [PTO-ISA/pto-isa](https://github.com/PTO-ISA/pto-isa) | PTO 虚拟指令集 |
| `DeepGEMM-Ascend` | [deepseek-ai/DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend) | BF16/FP8/FP4 GEMM、Grouped GEMM、MegaMoE |
| `tilelang-ascend` | [tile-ai/tilelang-ascend](https://github.com/tile-ai/tilelang-ascend) | TileLang 昇腾后端 |
| `catlass` | [ascend-catlass/catlass](https://github.com/ascend-catlass/catlass) | 昇腾 CUTLASS（C++ GEMM 模板） |
| `mojo_opset` | [Ascend/mojo_opset](https://github.com/Ascend/mojo_opset) | 可直接替换 nn.Module 的算子集 |
| `FlashKDA` | [Ascend/FlashKDA](https://github.com/Ascend/FlashKDA) | Kimi Delta Attention kernel |
| `FlashQLA` | [Ascend/FlashQLA](https://github.com/Ascend/FlashQLA) | TileLang 线性注意力 |
| `DeepSelect` | [Ascend/DeepSelect](https://github.com/Ascend/DeepSelect) | DSA TopK / sampler kernel |
| `ascend-kernel` | [QingCheng-AI/ascend-kernel](https://github.com/QingCheng-AI/ascend-kernel) | Chitu 的 Ascend C 算子 |
| `mamba-ascendc` | [So-cean/mamba-ascendc](https://github.com/So-cean/mamba-ascendc) | Mamba-2 SSD Ascend C 算子 |
| `mskl` | [Ascend/mskl](https://github.com/Ascend/mskl) | MindStudio Kernel Launcher |
| `triton-ascend-kernels` | [gitcode.com/Ascend/triton-ascend-kernels](https://gitcode.com/Ascend/triton-ascend-kernels) | 官方 Triton-Ascend 算子集 |

每个 submodule 保持自己的 LICENSE；本仓库只记录 commit 指针，不复制第三方代码。

## 芯片数据

18 个 class：`910`、`910B1..B4`、`910B2C`、`910_93`、`950PR-PCIE`、`950PR-Server`、`950DT`、
`310P1/3/5/7`、`310B1..B4`。每个记录包含：

- `npu_arch` / `arch_dir`（`dav-2201`→arch22、`dav-3510`→arch35）
- HBM 容量/带宽、BF16/FP16/INT8/FP8/FP4
- cube/vector 核数、频率、Cube MAC
- **L1 / L0A / L0B / L0C / UB / BT / L2**
- dtype、稀疏、32B DMA 对齐
- HCCS / RoCE / UB 带宽
- `sources`（仓库文件+章节）与 `confidence`

数据来源分两层，`infrarsi/ascend/chips.py` 加载时合并并记录 `provenance`：

1. `chips.yaml` — 由本地 `hy4-flash` 模型（CLIProxyAPI `:8317`）从 `ascend_chip_extract/sources/`
   的 kernel 仓库文档中提取，逐值带引用；
2. `chips_public.yaml` — 仓库未写到的 SKU 级参数（显存、TFLOPS、TDP、互联带宽），逐值带公开 URL。

## 测试

```bash
/usr/bin/python3 -m pytest tests/cpu/ascend -q
```

无需 NPU：`npu-smi` 解析器、芯片数据、源码替换、knob/registry 合并都有 fixture 测试。

## 状态

- 芯片数据：18 张卡，仓库提取 + 公开补充，**尚未在真实 NPU 上用 `npu-smi` 校验**
- `npu-smi` 解析器、CLI、芯片数据：已通过 15 个 CPU 测试
- `torch_npu` 融合算子：有纯 PyTorch fallback，但未在 NPU 上跑过
- `executor` / `linkbench` 的 NPU 接线：尚未完成

## License

本仓库的 Ascend 改造代码沿用上游 InfraRSI 的许可证；`third_party/` 下每个 submodule
按其自身 LICENSE 授权。
