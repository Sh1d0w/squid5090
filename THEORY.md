# squid5090 — Theoretical Performance Ceilings (`Inkling-2-Mini` on RTX 5090)

## 1. Measured Hardware (`CT 108` on `pve.lan`)

Measured directly with `/tmp/hw_probe` on 2026-11-23 while CT 100 (`frigate`, ~8.1 GiB VRAM) is running concurrently:

| Metric | Value | Notes |
|---|---:|---|
| GPU | NVIDIA GeForce RTX 5090 (`sm_120`, Blackwell) | 170 SMs, 96.0 MiB L2, 100 KiB SMEM/SM |
| Memory | 32,607 MiB GDDR7 (512-bit, 14,001 MHz DDR) | **23,997 MiB (23.43 GiB) free** (Frigate uses 8,116 MiB) |
| Spec Peak Bandwidth | 1,792.1 GB/s | Theoretical bus rate |
| **Measured Pure Read BW (1 GiB, `__ldg`)** | **1,512.9 GB/s** | **Decode bandwidth ceiling reference** |
| Measured D2D Copy BW (1 GiB) | 1,235.1 GB/s | Read + write |
| Measured BF16 Tensor Core GEMM (4096³) | 176.2 TFLOPS | 0.78 ms/iter |
| Measured BF16 Tensor Core GEMM (8192³) | 174.3 TFLOPS | 6.31 ms/iter (sustained) |
| Host CPU / RAM | Threadripper PRO 7975WX (16 vCPU in CT 108) / 64 GiB | NVMe thin-LVM rootfs |
| CUDA Toolkit / Driver | CUDA 13.1 (`V13.1.115`) / Driver `590.48.01` | Native `sm_120` NVFP4/FP8/PDL support |

## 2. Inkling-2-Mini Architecture & Exact Weight Footprint

Checkpoint: `thinkingmachines/Inkling-2-Mini` (`model-00001-of-00001.safetensors` = 19.526 GB, `mtp.safetensors` = 0.636 GB).

- **Loop Geometry**: 24 physical layers expanded into **48 logical visits**:
  - Prelude: layers `0..5` (6 visits)
  - Looped Core: layers `6..17` × **3 passes** (36 visits; layers 6..17 weights reused 3×, separate KV/sconv state per visit)
  - Coda: layers `18..23` (6 visits)
- **Attention**: 24 Q heads, 8 KV heads, `head_dim = 128`, `d_rel = 16`, `rel_extent = 1024`, sliding window `1024` on local layers (20 of 24 physical layers = 40 of 48 visits), full context on global layers (`5, 11, 17, 23` = **8 of 48 visits**).
- **MoE**: 128 routed experts, top-8 per token, `intermediate_size = 1152`, softcapped SwiGLU (`gate_cap=4`, `up_cap=25`).
  - Physical layers `0..19`: **NVFP4** (`e2m1` packed U8 + `f8_e4m3` block-16 scales + per-expert F32 `amax`/`denom`)
  - Physical layers `20..23`: **MXFP8** (`f8_e4m3` + `ue8m0` block-32 scales)
- **Vocab**: `201,024` padded (`200,058` unpadded), `hidden_size = 2304`.
- **DSpark MTP Drafter**: 3 dense layers (`intermediate_size = 9216`, `block_size = 8` → 7 drafts per step) + rank-256 Markov RNN head, tapping target visits `[5, 23, 41, 48]`.

### Memory Footprint by Precision Mode (Text-Only + MTP, excluding 0.637 GB Vision)

| Component | Official (`exact`) | `fp8` Dense | `nvfp4` All-Linear |
|---|---:|---:|---:|
| MoE Experts L0–19 (NVFP4) | 11.466 GB | 11.466 GB | 11.466 GB |
| MoE Experts L20–23 | 4.204 GB (MXFP8) | 4.204 GB (MXFP8) | 2.293 GB (NVFP4) |
| Attention Dense + Router (24 layers) | 1.308 GB (BF16) | 0.654 GB (FP8) | 0.368 GB (NVFP4) |
| Controllers + Norms + SConv + AttnRes | 0.058 GB (BF16/F32) | 0.058 GB | 0.058 GB |
| Embedding Table | 0.926 GB (BF16) | 0.926 GB (BF16) | 0.926 GB (BF16) |
| LM Head (`unembed`) | 0.926 GB (BF16) | 0.463 GB (FP8) | 0.261 GB (NVFP4) |
| DSpark MTP (3 layers + Markov) | 0.636 GB (BF16) | 0.320 GB (FP8) | ~0.240 GB (NVFP4 + pruned/2b head) |
| **Total Resident Weights** | **19.524 GB (18.18 GiB)** | **18.091 GB (16.85 GiB)** | **15.612 GB (14.54 GiB)** |
| **Headroom in 23.43 GiB Free VRAM** | **5.25 GiB** | **6.58 GiB** | **8.89 GiB** |

**Key structural win vs Basalt (CT 107):** 100% of weights stay resident in RTX 5090 GDDR7 with **5.2–8.9 GiB** headroom for KV cache + workspace (zero host-RAM expert spilling).

## 3. Single-User (`c=1`) Decode Theoretical Ceilings (at 1,512.9 GB/s Measured Read BW)

### Bytes Read per Single Target Token (`B = 1`, Short Context)

| Component | `exact` (MB/tok) | `fp8` Dense (MB/tok) | `nvfp4` All-Linear (MB/tok) |
|---|---:|---:|---:|
| Experts L0–19 (44 visits × 8/128) | 1,576.5 | 1,576.5 | 1,576.5 |
| Experts L20–23 (4 visits × 8/128) | 262.8 | 262.8 | 143.3 |
| Attention Dense + Router (48 visits) | 2,616.0 | 1,308.0 | 736.0 |
| Controllers (24 looped visits) | 84.0 | 84.0 | 84.0 |
| LM Head (`unembed`) | 926.3 | 463.2 | 260.5 |
| **Total Bytes / Token (`B = 1`)** | **5,465.6 MB** | **3,694.5 MB** | **2,800.3 MB** |
| **Theoretical Ceiling (`1,512.9 GB/s`)** | **276.8 tok/s** | **409.5 tok/s** | **540.3 tok/s** |
| **Theoretical Ceiling (`1,792.1 GB/s` spec)** | **327.9 tok/s** | **485.1 tok/s** | **640.0 tok/s** |

> **Critical Physics Insight:** In the official checkpoint, BF16 attention projections (2,616 MB across 48 visits) + BF16 LM head (926 MB) account for **64.8% of all decode memory traffic**, dwarfing the NVFP4/MXFP8 MoE experts (1,839 MB = 33.6%)! Quantizing attention + LM head to NVFP4 nearly **doubles** the single-token hardware ceiling before speculation even begins.

### With Built-In DSpark MTP Speculative Decoding (`B = K + 1` Verify + Expert Deduplication)

In a verify pass of `B = K + 1` tokens:
- Dense weights (`D`) and LM head (`H`) are read **ONCE** for all `B` tokens (100% amortized!).
- MoE experts are **grouped/deduplicated across the `B` tokens**: let `u(B)` be the expected multiplier of unique experts vs 1 token. Worst-case i.i.d. is `128/8 * (1 - (120/128)^B)` (`u(4)=3.64`, `u(6)=5.17`, `u(8)=6.45`); real correlated consecutive tokens typically exhibit **~20–35% lower `u(B)`** (`u(4)≈2.8`, `u(6)≈3.9`, `u(8)≈4.9`).
- DSpark draft cost (`NVFP4` 3-layer block + pruned/screened Markov head): ~150 MB total for up to 7 drafts (~0.15–0.35 ms).

| Mode | Verify `B` (`K` drafts) | Assumed Accepted `a` | Est. Expert `u(B)` | Verify + Draft MB | MB / Accepted Tok | **Practical Ceiling (1,513 GB/s)** |
|---|---:|---:|---:|---:|---:|---:|
| `exact` (no MTP) | 1 (0) | 1.0 | 1.00 | 5,466 | 5,466 | **277 tok/s** |
| `nvfp4` (no MTP) | 1 (0) | 1.0 | 1.00 | 2,800 | 2,800 | **540 tok/s** |
| `nvfp4` + MTP (`K=3`, prose) | 4 (3) | 2.6 | 2.9 | 6,218 | 2,392 | **633 tok/s** |
| `nvfp4` + MTP (`K=5`, prose) | 6 (5) | 3.4 | 4.1 | 8,282 | 2,436 | **621 tok/s** |
| `nvfp4` + MTP (`K=7`, code/high-acc) | 8 (7) | 5.2 | 4.9 | 9,658 | 1,857 | **815 tok/s** |
| `nvfp4` + MTP (`K=7`, very high acc) | 8 (7) | 6.5 | 4.5 | 8,970 | 1,380 | **1,096 tok/s** |
| **Adaptive-K MTP** (stops at low conf) | 2..8 | ~3.5–6.0 | minimal! | optimal | ~1,400–2,000 | **750–1,080+ tok/s** |

### Honest Assessment of "100x faster"
- **vs Naive PyTorch / HF BF16** (~10–15 tok/s on 26B looped MoE): **750–1,000+ tok/s** is **50–100× faster**!
- **vs Apple Silicon `nanoattention`** (~39 tok/s on M3 Ultra): **20–28× faster**!
- **vs Basalt on this exact RTX 5090** (~265–325 tok/s with spec=8 and host-RAM expert spill): single-user physics ceiling is **~650–1,100 tok/s** (**2.5–3.5× faster** than Basalt), bounded strictly by GDDR7 read bandwidth (1,513 GB/s) divided by active NVFP4 expert bytes!
- **KV Cache Sizing**:
  - 40 local visits (window 1024) = fixed `40 × 1024 × 2 × 1024 × 2 B` = **167.8 MB** (BF16) or **83.9 MB** (FP8)!
  - 8 global visits = `8 × 2 × 1024 × 2 B` = **32,768 B/token** (BF16) or **16,384 B/token** (FP8)!
  - At 64K context: BF16 KV = **2.27 GiB**, FP8 KV = **1.14 GiB** — fits comfortably in our **8.9 GiB** VRAM headroom!
