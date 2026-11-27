# Ascend chip-data extraction

The per-card records in `InfraRSI/infrarsi/ascend/data/chips.yaml` were extracted
by the local **hy4-flash** model (`cn:hy4flash-free` on CLIProxyAPI `:8317`)
from the Ascend kernel repositories, then audited and supplemented with public
product tables where the repositories only document an architecture family.

## Pipeline

1. `sources/` — the READMEs/docs the model read:
   - `npu-hardware-params.md`, `npu-arch-guide.md` (the authoritative arch tables:
     L1/L0A/L0B/L0C/UB/BT/L2, core counts, arch ids)
   - `pto-kernels.md` (SOC_VERSION → dav arch / base mode)
   - `deepgemm-ascend.md` (950DT validation, CANN 9.3)
   - `tilelang-ascend.md` (A2/A3 memory hierarchy)
   - `catlass.md` (CATLASS_ARCH values)
   - `mojo-opset.md`, `triton-ascend.md`, `flashkda.md`, `flashqla.md`, `deepselect.md`
2. `extract_with_model.py --batch a|b` — sends the evidence pack to
   `http://127.0.0.1:8317/v1/chat/completions`, model `cn:hy4flash-free`,
   `temperature=0`, `max_tokens=48000`. Raw responses are kept in
   `raw_a.json` / `raw_b.json`; parsed output in `chips_a.json` / `chips_b.json`.
3. `public_specs.yaml` — fills the gaps the repositories do not state
   (per-SKU HBM size, TFLOPS, TDP, HCCS/RoCE bandwidth), each with its URL.
4. `validate_chips.py` — checks arch ids, numeric ranges, citations and
   confidence; writes:
   - `InfraRSI/infrarsi/ascend/data/chips.yaml` (repo extraction, verbatim)
   - `InfraRSI/infrarsi/ascend/data/chips_public.yaml` (public supplement)
   - `InfraRSI/infrarsi/experience/data/gpus_ascend.yaml` (transfer view)
   - `chips_merged.json` (effective merged view for review)

## Provenance

`infrarsi/ascend/chips.py` merges the two files at load time:

- `provenance[field] = "repo"` — value from the model's repository extraction.
- `provenance[field] = "public"` — value from a public product table because the
  repositories did not state it (or stated only a formula-derived value).
- `sources` / `public_sources` keep the file+section and URL respectively.

## Current coverage

18 classes: `910`, `910B1..B4`, `910B2C`, `910_93`, `950PR-PCIE`,
`950PR-Server`, `950DT`, `310P1/3/5/7`, `310B1..B4`.

Known gaps (marked in `missing` and/or left null):

- `310B1..B4` device HBM (public tables do not state it).
- `310P*` BF16/TFLOPS (inference parts; no training figure).
- `910_93` cube cores are per-die (dual-die package).
- `950DT` core count and frequency (public roadmap gives memory/TFLOPS only).

## Re-run

```bash
cd ~/Desktop/InfraRSI-Ascend/ascend_chip_extract
export HY4_KEY=<the CLIProxyAPI key>       # never stored in the repository
/usr/bin/python3 extract_with_model.py --batch a --max-tokens 48000
/usr/bin/python3 extract_with_model.py --batch b --max-tokens 48000
/usr/bin/python3 validate_chips.py
cd ../InfraRSI && /usr/bin/python3 -m pytest tests/cpu/ascend -q
```
