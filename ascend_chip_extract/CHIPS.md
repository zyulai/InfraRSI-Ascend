# Ascend cards extracted for InfraRSI

| class | arch | dir | HBM GiB | TB/s | BF16 T | FP8 T | cube | vector | L1 KB | L0C KB | UB KB | BT KB | L2 MB | dtypes | conf |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 910 | dav-1001 | arch10 | 32 | 1.5 | 256 | - | - | - | 1024 | 256 | 256 | - | 32 |  | medium |
| 310P1 | dav-2002 | arch20 | 24 | 0.2 | - | - | - | - | 1024 | 256 | 256 | - | 16 |  | medium |
| 310P3 | dav-2002 | arch20 | 24 | 0.2 | - | - | - | - | 1024 | 256 | 256 | - | 16 |  | medium |
| 310P5 | dav-2002 | arch20 | 24 | 0.2 | - | - | - | - | 1024 | 256 | 256 | - | 16 |  | medium |
| 310P7 | dav-2002 | arch20 | 24 | 0.2 | - | - | - | - | 1024 | 256 | 256 | - | 16 |  | medium |
| 910B1 | dav-2201 | arch22 | 64 | 1.6 | 400 | - | 24 | 48 | 512 | 128 | 192 | 1 | 192 |  | high |
| 910B2 | dav-2201 | arch22 | 64 | 1.6 | 376 | - | 24 | 48 | 512 | 128 | 192 | 1 | 192 | fp16,bf16 | high |
| 910B2C | dav-2201 | arch22 | 64 | 1.6 | 376 | - | 24 | 48 | 512 | 128 | 192 | 1 | 192 |  | high |
| 910B3 | dav-2201 | arch22 | 64 | 1.6 | 313 | - | 20 | 40 | 512 | 128 | 192 | 1 | 192 |  | medium |
| 910B4 | dav-2201 | arch22 | 32 | 0.8 | 280 | - | 20 | 40 | 512 | 128 | 192 | 1 | 96 |  | medium |
| 910_93 | dav-2201 | arch22 | 128 | 3.2 | 800 | - | 24 | 48 | 512 | 128 | 192 | 1 | - |  | medium |
| 310B1 | dav-3002 | arch30 | - | - | - | - | 1 | 1 | 1024 | 128 | 248 | 1 | 4 |  | medium |
| 310B2 | dav-3002 | arch30 | - | - | - | - | 1 | 1 | 1024 | 128 | 248 | 1 | 4 |  | medium |
| 310B3 | dav-3002 | arch30 | - | - | - | - | 1 | 1 | 1024 | 128 | 248 | 1 | 4 |  | medium |
| 310B4 | dav-3002 | arch30 | - | - | - | - | 1 | 1 | 1024 | 128 | 248 | 1 | 4 |  | medium |
| 950DT | dav-3510 | arch35 | 144 | 4.0 | 547 | 1000 | - | - | 512 | 256 | 248 | 4 | - | fp16,bf16,hf32,fp32,s8,fp8_e5m2,fp8_e4m3fn,mxfp8_e5m2,mxfp8_e4m3,mxfp4_e2m1,mxfp4_e1m2,hif8 | high |
| 950PR-PCIE | dav-3510 | arch35 | 112 | 1.4 | 378 | 804 | 28 | 56 | 512 | 256 | 248 | 4 | 112 | fp16,bf16,hf32,fp32,s8,fp8_e5m2,fp8_e4m3fn,mxfp8_e5m2,mxfp8_e4m3,mxfp4_e2m1,mxfp4_e1m2,hif8 | high |
| 950PR-Server | dav-3510 | arch35 | 128 | 1.6 | 432 | 865 | 32 | 64 | 512 | 256 | 248 | 4 | 128 | fp16,bf16,hf32,fp32,s8,fp8_e5m2,fp8_e4m3fn,mxfp8_e5m2,mxfp8_e4m3,mxfp4_e2m1,mxfp4_e1m2,hif8 | high |

Source: chips.yaml (repo extraction) + chips_public.yaml (public supplement); see README.md.
