# Engineering & Abliteration Learnings (`squid5090`)

A technical post-mortem of everything we learned optimizing [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) from **`76.6 tok/s` to `585.9 tok/s`** single-user decode, **`9,181 tok/s`** chunk prefill, and **`1,048,576` (1M) context** on a single NVIDIA RTX 5090 — plus the full mathematical derivation and practical pipeline for generating custom **1.8 MB runtime-gated Fisher-LDA abliteration** `.safetensors` files with [`tools/ablate.py`](tools/ablate.py).

---

## 1. End-to-End Progression Summary (Single RTX 5090, 32 GiB GDDR7)

| Milestone | Key Engineering Change | Code Decode (`c=1`) | Chat Decode | Single-Token (`mtp=0`) | Prefill (20K / 64K) | Max Context / VRAM |
|---|---|---:|---:|---:|---:|---|
| **v0 Eager Baseline** | All 48 visits implemented from scratch (`exact` BF16+NVFP4/MXFP8) | — | — | 76.6 tok/s | ~900 tok/s | ~200K (`18.8 GiB`) |
| **CUDA Graphs + NVFP4 PTX** | Persistent `cudaGraphExec_t`, Blackwell `cvt.rn.f16x2.e2m1x2`, all-NVFP4 linear load-time quant | — | — | 190.1 tok/s | ~900 tok/s | ~256K (`15.0 GiB`) |
| **Half-Warp GEMV + Mega-Fusions + DSpark `mtp=3`** | 99%-BW `lm_head`, 32K pruned draft vocab, zero-atomic deterministic MoE, TC Split-K FlashDecoding, `B=512` TC prefill | 369.6 tok/s | 307.3 tok/s | 243.8 tok/s | 5,077 / 4,637 tok/s | **1M** (`22.7 GiB`) |
| **Blackwell PDL + TC `mma.m16n8k16` Decode** | `cudaLaunchAttributeProgrammaticStreamSerialization` + `pdl_wait_fenced`, register-permuted TC `moe_up`/`moe_down` & dense NVFP4 GEMVs | 470.2 tok/s | 307.3 tok/s | **259.0–262.8 tok/s** | 5,077 / 4,637 tok/s | **1M** (`22.7 GiB`) |
| **Adaptive `{2, 4, 8}` Switch Graph + Overlap** | Markov online softmax confidence, CUDA 13.1 `cudaGraphCondTypeSwitch`, overlapped next-round draft | **585.9 tok/s** | **332.5 tok/s** | 259.0–262.8 tok/s | 5,077 / 4,637 tok/s | **1M** (`22.7 GiB`) |
| **`B=2048` WGEMM Prefill + 2D-Tiled Short Conv** | Multi-stage `cp.async` `128×128×64` grouped MoE GEMM + 576-CTA 4-tap FIR sconv | **585.9 tok/s** | **332.5 tok/s** | 259.0–262.8 tok/s | **8,755 / 7,742 tok/s** *(9,181 chunk peak)* | **1M** (`23.22 GiB`) |
| **Runtime-Gated Fisher-LDA `--ablation`** | 1.8 MB gated projection fused into decode & prefill inside single `VRAMArena` | **541.1 tok/s** | **332.5 tok/s** *(0 harmless diff)* | 259.0 tok/s | 8,755 / 7,742 tok/s | **1M** (`23.22 GiB`) |

---

## 2. Roofline & Architectural Realities of Inkling-2-Mini on Blackwell (`sm_120a`)

1. **Measured hardware ceilings beat spec-sheet numbers:**
   - RTX 5090 sustained GDDR7 bandwidth measured in-device is **`1,512.9 GB/s`** across 170 SMs, with 96 MiB L2 and 100 KiB dynamic shared memory per SM.
   - Always verify the GPU is in **`P0`** (`14,001 MHz` memory clock, `0%` background utilization) before attributing a 5–10% delta to a kernel change. Background GPU consumers silently clamp the card into `P1` (`13,801 MHz` mem) and add SM contention.
2. **Why Inkling-2-Mini has unusual performance characteristics:**
   - **24 physical layers → 48 logical visits:** layers `0..5`, then layers `6..17` looped **3 times** (`36` looped visits), then layers `18..23`.
   - **AttnRes + cross-pass controllers + 4-tap short convolutions:** every visit includes depth attention over block boundaries, post-attention and post-MLP causal 4-tap FIR convolutions, and on looped passes 1–2 four low-rank controllers. These small memory-bound ops quickly dominate latency unless mega-fused and overlapped.
   - **Hybrid attention enables 1M context on 32 GiB:** 36 visits are local sliding-window attention (`W = 1024`, fixed tiny ring buffer `kv_cap = 1280`), leaving only **12 global visits sharing 8 unique global KV caches** (plus 1 global DSpark drafter layer). Quantizing global KV to group-64 INT4 (`--kv q4`) + pinning the `0.86 GiB` BF16 embedding table in zero-copy host RAM fits **`1,048,576` tokens in `23.22 GiB` VRAM** with zero host-RAM spilling of weights or KV caches.
   - **Bandwidth wall at `B=1`:** Quantizing dense weights, `lm_head`, and layers `20..23` experts to NVFP4 on load cuts per-token HBM traffic from `5.47 GB/tok` to `~2.80 GB/tok`. Even at `~75%` aggregate GDDR7 utilization across 48 visits, `B=1` tops out around `260–270 tok/s`. Breaking `500+ tok/s` requires **multi-token speculative verification** (`B = 2..8`) where dense weights are loaded once for `B` tokens and MoE experts are deduplicated across tokens.

---

## 3. GPU Decode Optimization Learnings (`76.6 → 585.9 tok/s`)

### 3.1 Native Blackwell FP4 Unpack & Streaming Loads
- Using Blackwell's native `cvt.rn.f16x2.e2m1x2` PTX instruction + 128-bit `uint4` streaming loads (`ld.global.nc.L1::no_allocate.L2::256B`) eliminates LUT overhead and L1 thrashing.
- **CUDA 13.1 `ptxas` pitfall:** Do **not** enable `L2::cache_hint` on CUDA 13.1 (`ptxas` codegen bug). Plain `ld.global.nc.L1::no_allocate.L2::256B` compiles cleanly and hits 99% of measured GDDR7 bandwidth on `lm_head` (`1,497 GB/s`).

### 3.2 Blackwell Programmatic Dependent Launch (PDL)
- Launching kernels inside a captured CUDA Graph with `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization` allows the GPU scheduler to start a consumer kernel's prologue (index math, address computation, weight-load issue) **before** the producer kernel's epilogue finishes retiring, synchronizing only at `pdl_wait_fenced(zero)` immediately before reading producer outputs and calling `pdl_trigger()` right after writing outputs.
- Across 48 visits (~600 kernel boundaries per decode step), PDL alone lifted `mtp=0` from `243.8 → 262.8 tok/s` with zero algorithmic change.

### 3.3 Register-Permuted Tensor Core `mma.m16n8k16` for `M = 1..8` MoE & Dense GEMVs
- At `B=1`, CUDA-core `half2` FMA is near-optimal because latency is pure HBM read. At `B = 2..8` (speculative verify), CUDA-core FMA becomes compute- and register-bound, especially inside MoE where 8 tokens share overlapping experts.
- Replacing CUDA-core MoE with `moe_up_tc_kernel` and a **144-CTA fused `moe_down_tc_kernel`** (plus Tensor Core `qkvrg_fused_mma`, `linear_nvfp4_mma` for `K ∈ {2304, 3072, 9216}`, and `lm_head_mma` for `M >= 2`) keeps weight traffic constant while pushing `M = 2..8` compute onto Blackwell Tensor Cores (`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`).
- **Effect:** `mtp=3` jumped from **`369.6 → 470.2 tok/s`** on code.
- **Numerical parity note:** `B=1` (FP16-pair CUDA core) and `B >= 2` (FP32-accumulate Tensor Core MMA) can diverge by 1 ULP on tight argmax ties over 400-token greedy rollouts. Always verify multi-token parity across `B=4`, fixed `B=8`, and adaptive `{2, 4, 8}` — all three share the exact same FP32-accumulate Tensor Core path and produce bit-for-bit identical token streams.

### 3.4 Adaptive `{2, 4, 8}` Speculative Verify via CUDA Graph Conditional Switch
- **The fixed-width dilemma:** On predictable code, verifying `B=8` tokens (`mtp=7`) jumps from `470 → 511 tok/s`. On unpredictable prose/chat, verifying `B=8` *regresses* from `307 → 270 tok/s` because the extra drafted tokens miss and waste MoE/attention bandwidth.
- **Device-side adaptive solution:**
  1. Inside the DSpark Markov argmax head (`cuda/kernels_dspark.cu`), compute online softmax `(m, sum)` under softcap `30.0` in a single pass alongside argmax to emit exact per-draft confidence `p_i = max_v softmax(logits_i)` for drafts `0..6`.
  2. Run a tiny 1-thread device kernel `z_adaptive_select_width_kernel` that picks the verify width:
     - `B = 8` if the first 3 drafts all exceed `--spec-threshold` (default `0.80`)
     - `B = 4` if draft 0 exceeds `--spec-threshold`
     - `B = 2` otherwise
  3. Dispatch the chosen branch directly on the GPU inside a single CUDA 13.1 **`cudaGraphCondTypeSwitch`** graph (`case 0 → B=2`, `case 1 → B=4`, `case 2 → B=8`) — **zero host round-trip, zero graph re-capture**.
  4. **Overlapped next-round drafting:** After launching the conditional switch verify graph, record `done_event` and immediately enqueue the *next* round's DSpark draft graph on the stream before calling `cudaEventSynchronize(done_event)`. While the host copies out accepted tokens and flushes HTTP SSE chunks, the GPU is already drafting the next round.
  5. **Zero-rollback state design:** Because every sconv ring in `squid5090` is a 16-slot modular ring indexed by `(p & 15)` (`max_B + 3 = 8 + 3 = 11 < 16`), a verify batch never overwrites the `P-3..P-1` history needed on rejection, and KV / DSpark caches are position-indexed — rejected positions strictly sit ahead of the updated `*d_pos = P + n_acc` and are harmlessly overwritten on the next step.
- **Effect:** **`585.9 tok/s`** on code (`acc=72.5%`, `dl=3.80`, beating `ingine`'s published `561 tok/s` on the same prompt!) AND **`332.5 tok/s`** on chat (beating both `mtp=3` and fixed `mtp=7` simultaneously!).

---

## 4. GPU Prefill Optimization Learnings (`5,077 → 9,181 tok/s`)

### 4.1 `B=2048` Multi-Stage `cp.async` `128×128×64` Grouped MoE GEMM (`p_wgemm_kernel`)
- At `B=512`, each of the 128 experts receives on average only `512 × 8 / 128 = 32` tokens — too small to saturate Tensor Cores relative to weight-dequant overhead.
- Raising `prefill_batch_max` from `512 → 2048` quadruples expert reuse to ~128 tokens/expert.
- We implemented a multi-stage `cp.async` grouped MoE GEMM (`p_wgemm_kernel<FMT, MODE, K, TM>` in [`cuda/kernels_prefill.cu`](cuda/kernels_prefill.cu)) with:
  - `TM = 32` for `B <= 512` and `TM = 128` for `B > 512`, `TN = 128`, `BK = 64`.
  - **Pigeonhole tile bound:** For `B <= 512` with `TR = 32` (`4,096` token-expert pairs) OR `B <= 2048` with `TR = 128` (`16,384` pairs), the total number of expert tiles across 128 experts is strictly bounded by `128 + 128 = 256` (`<= 256`), requiring **zero host sync** to read tile counts and fixed static scratch memory.
  - 3-stage `cp.async` swizzled FP16 activations + 3–7-stage `cp.async` packed NVFP4/MXFP8 weights (~98–99 KiB dynamic SMEM, fitting cleanly under Blackwell's 100 KiB limit) + Blackwell hardware `cvt.rn.f16x2.e2m1x2` + `ldmatrix.x4` + FP16 MMA.
  - Contiguous sorted activation handoff between `Up` and `Down` (`out_act_h[(pb + row) * INTERMEDIATE]`), eliminating scattered gathers on `Down`.
- **Effect:** Cut MoE prefill time for 2,048 tokens across all 48 visits to **`92.2 ms`** (`1.92 ms/visit`, 2.4× faster MoE), taking 20K prefill from `5,077 → 7,449 tok/s`.

### 4.2 Always Profile After Your Big Win: The 9-CTA Short-Conv Bottleneck
- Running `SQUID_PROF_PREFILL=1` on `B=2048` immediately revealed:
  ```text
  Before: total=266.35ms (7689 tok/s) | pre_sub=52.43ms dense_tc=29.50ms attn_sub=57.27ms post_sub=17.88ms moe=92.15ms commit=17.13ms
  ```
- Why were `pre_sub + post_sub` taking **`70.3 ms` (26% of all prefill time!!)**? Because `p_sconv_post_res_kernel` launched only `2304 / 256 = 9` CTAs on a 170-SM GPU (5% SM occupancy) and looped serially `for (int b = 0; b < 2048; ++b)`!
- **Key insight:** A 4-tap causal short convolution is a pure **FIR** filter (no IIR feedback!) — token `b` only depends on `src[b], src[b-1], src[b-2], src[b-3]`.
- **Subtle concurrency trap avoided:** Tiling across `SCONV_T_BLK = 32` (`576 CTAs`) creates a subtle race if the last Y-CTA writes `ring[((pos0 + b) & 15)]` while CTA `y=0` is still reading `ring[((pos0 - 1) & 15)]`, because for `B=2048`, `(pos0 + 2047) & 15 == (pos0 - 1) & 15`!
- **Clean race-free fix:** Keep `ring` read-only inside the 576-CTA tiled `p_sconv_post_res_kernel` when `ty > 1`, and follow it with a tiny 9-CTA `p_sconv_ring_commit_kernel` (only writes the final 16 tokens, `<2 µs`).
- **Effect:**
  ```text
  After:  total=223.07ms (9181 tok/s) | pre_sub=19.73ms dense_tc=29.45ms attn_sub=57.19ms post_sub=7.67ms moe=92.59ms commit=16.43ms
  ```
  - Saved **`42.9 ms`** per 2,048-token chunk in 30 lines of CUDA!
  - **20K prefill:** **`8,755 tok/s`** (`2,282 ms`) | **64K Needle-in-a-Haystack (PASS):** **`7,742 tok/s`** (`8.46 s`) | **131K prefill:** **`6,509 tok/s`** (`19.97 s`).

---

## 5. TigerStyle Systems Learnings

- **2 `cudaMalloc` arenas (`VRAMArena` + `assert_exact_fill()`):** Two-pass size calculation guarantees exact upfront VRAM accounting (zero fragmentation, zero surprise OOMs at 1M context, and immediate assertion failure if any tensor or buffer size drifts).
- **Zero-malloc hot path (`ServerBuffers`):** Static startup-allocated HTTP, tokenizer, prompt, and SSE buffers yield deterministic `0.07 tok/s` run-to-run variance and zero GC/allocator jitter even during 450+ tok/s live SSE streaming.
- **Warning-free CUDA builds:** `build.zig` treats any `nvcc` warning stderr as a build failure, forcing dead kernels and unused variables to be pruned immediately.

---

## 6. Runtime-Gated Fisher-LDA Abliteration: Math, Intuition & How to Generate Your Own

This section documents the complete abliteration method pioneered by [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) (Apache-2.0), which we ported into `squid5090` (runtime CUDA/Zig execution + offline [`tools/ablate.py`](tools/ablate.py) generation pipeline).

### 6.1 Why Standard Weight-Space Abliteration Breaks on Inkling-2-Mini
Standard abliteration (Arditi et al. / Heretic) projects a refusal direction $r$ out of output weight matrices ($W_{\text{out}} \leftarrow W_{\text{out}} - w\, r (r^\top W_{\text{out}})$). On Inkling-2-Mini, doing that directly in weight space fails for **three** reasons:
1. **Looped weight sharing:** Physical layers `6..17` are executed **3 times** as 36 distinct logical visits (`visits 6..41`). Refusal circuits fire strongly on specific passes/visits, while earlier or later passes through the *same physical weights* perform general reasoning and coding. Editing physical weights corrupts all three passes at once.
2. **4-bit quantization floor:** Re-quantizing a small rank-1 projection back into NVFP4 (`e2m1` + block-16 `e4m3` scales) either rounds the edit to zero or introduces large quantization noise across unrelated weights.
3. **Unconditional damage on harmless text:** Always projecting out $r$ even when the model is NOT refusing shifts harmless token logits, increasing KL divergence and degrading coding/reasoning accuracy.

### 6.2 The 1.8 MB Runtime-Gated Residual-Write Formula
Instead of touching model weights, we keep the base checkpoint **100% untouched** and apply a tiny, per-visit runtime intervention right after `post_attention_convolution` and `post_mlp_convolution` — the **exact** residual-stream write points across all `V = 48` logical visits:

$$y \leftarrow y - w_v \cdot \mathbb{I}\!\left(d_v \cdot y > \tau_v\right) \cdot \max\!\left(r_v \cdot y - c_v,\; 0\right) \cdot r_v$$

Every symbol has a sharp geometric purpose:
1. **$r_v \in \mathbb{R}^{2304}$ (Projected Refusal Direction, unit norm):**
   - Measure mean read activations on `N = 336` harmful prompts (`mlabonne/harmful_behaviors`) vs `336` harmless prompts (`mlabonne/harmless_alpaca`) at two positions: direct-answer start (`<|message_model|><|content_text|>`) and reasoning-turn start (`<|message_model|><|content_thinking|>`).
   - Compute `diff = mean(harmful) - mean(harmless)`, then **orthogonalize against the harmless mean** (`diff <- diff - (diff · ĥ) ĥ`, where `ĥ = normalize(mean(harmless))`) so the direction never cancels the typical harmless activation component.
   - Blend the direct-answer and reasoning-turn unit directions with search parameter `dir_mix ∈ [0, 1]` and re-normalize.
2. **$c_v \in \mathbb{R}$ (Harmless Mean Baseline Offset):**
   - Computed analytically from harmless write statistics as $c_v = \mathbb{E}_{\text{harmless}}[r_v \cdot y]$.
   - Combining $c_v$ with the ReLU gate $\max(r_v \cdot y - c_v, 0)$ ensures we **only** remove excess projection along $r_v$ above normal harmless text — we never push harmless or negative projections further negative.
3. **$d_v \in \mathbb{R}^{2304}$ & $\tau_v \in \mathbb{R}$ (Shrinkage Fisher-LDA Detector):**
   - Why not gate on $r_v \cdot y$ alone? Because harmless activations have huge anisotropic variance; a raw mean-difference vector $r_v$ can accidentally overlap high-variance harmless directions (e.g., code formatting or prose style).
   - Instead, `tools/ablate.py stats` accumulates the full **`2304 × 2304` FP64 covariance matrix $\Sigma_v$** of harmless residual writes across an equal mixture of harmless chat turns, harmless reasoning turns, and general technical text.
   - It also collects 4 harmful-minus-harmless displacement vectors $\delta_{m,v}$ ($m \in \{\text{answer start, reasoning start, reasoning turn, direct answer}\}$), then solves the regularized Fisher Linear Discriminant system:
     $$s_m = \left(\Sigma_v + \lambda\, \overline{\operatorname{diag}(\Sigma_v)}\, I\right)^{-1} \delta_{m,v}, \qquad \lambda = 0.1$$
     $$u_v = \sum_{m=1}^{4} \frac{s_m}{\sqrt{\delta_{m,v}^\top s_m}}, \qquad d_v = \frac{u_v}{\sqrt{u_v^\top \Sigma_v u_v}}$$
   - Notice that final normalization: $\operatorname{Std}_{\text{harmless}}(d_v \cdot y) = \sqrt{d_v^\top \Sigma_v d_v} \equiv 1$!
   - Therefore, setting the threshold to:
     $$\tau_v = \mathbb{E}_{\text{harmless}}[d_v \cdot y] + k_{\text{det}}$$
     places the firing boundary at exactly **$k_{\text{det}}$ harmless standard deviations** above the harmless mean (typically $k_{\text{det}} \in [1.5, 4.5]$). On ordinary coding and chat prompts, $d_v \cdot y \le \tau_v$ almost everywhere, so `k = 0` and the model's output is **bit-for-bit identical** to the unmodified base model!
4. **$w_v \ge 0$ (Heretic Linear-Decay Visit Kernel):**
   - Rather than optimizing 96 independent visit weights (which would overfit), `ablate.py` parameterizes attention and MLP weights across the 48 visits with two 4-parameter triangular kernels `(max_w, pos, min_frac, dist)`:
     $$w_v = \begin{cases} w_{\max} + (f_{\min} w_{\max} - w_{\max}) \frac{|v - p|}{D} & \text{if } |v - p| \le D \\ 0 & \text{otherwise} \end{cases}$$
   - Total Optuna search dimensionality is just **10 continuous parameters**: `dir_mix`, `k_det`, and 4 kernel params each for attention and MLP!
5. **Optuna Multi-Objective TPE Optimization:**
   - Objective 1 (**refusals**): evaluated fast on the RTX 5090 by launching `squid5090 serve --ablation <trial.safetensors>` across direct harmful, effort-none harmful, low-effort reasoning harmful, and mild over-refusal prompt sets.
   - Objective 2 (**capability damage**): `max(KL_answers, KL_text)`, teacher-forced KL divergence against the unmodified model in BF16 (`tools/inkling_ref.py`). Trials exceeding `--kl-cap 0.03` are pruned before serving.
   - Choosing a Pareto-frontier trial yields near-zero refusals with `<0.005–0.02` KL divergence and **zero degradation on harmless coding/chat**.

### 6.3 Safetensors Schema (`~1.77 MB` Total) & Zero-Overhead CUDA Fusion
Exported by `tools/ablate.py export` and loaded by `squid5090 --ablation <path>`:

| Tensor Name | Shape | Dtype | Meaning |
|---|---|---|---|
| `attn_dirs`, `mlp_dirs` | `[48, 2304]` | `F32` | Per-visit refusal directions $r_v$ (unit-normalized on load in FP64→FP32) |
| `attn_det`, `mlp_det` | `[48, 2304]` | `F32` | Per-visit Fisher-LDA detectors $d_v$ (kept at native $\sigma=1$ scale!) |
| `attn_weight`, `mlp_weight` | `[48]` | `F32` | Per-visit ablation strength $w_v$ |
| `attn_offset`, `mlp_offset` | `[48]` | `F32` | Per-visit harmless mean offset $c_v$ |
| `attn_threshold`, `mlp_threshold` | `[48]` | `F32` | Per-visit detector threshold $\tau_v$ |
| `gate` | `[1]` | `F32` | `1.0` for ReLU gate (`floor = 0`), `0.0` for affine (`floor = -FLT_MAX`) |

In `squid5090`:
- [`src/weights.zig`](src/weights.zig) packs `{w_v, c_v, floor, tau_v}` into a compact `[48, 4]` `float4` table inside the **single weight `VRAMArena`** (still 1 `cudaMalloc`, verified by `assert_exact_fill()`).
- In decode ([`cuda/kernels_inkling.cu`](cuda/kernels_inkling.cu)), `apply_ablation_576` is fused directly into the 576-thread vectorized `mega_attn_post_mlp_in_kernel`, `mega_mlp_post_res_kernel`, and `mega_mlp_post_next_depth_norm_kernel` — zero extra kernel launches, only two warp-reduced dot products (`r · y` and `d · y`) over `y` already live in registers!
- In prefill ([`cuda/kernels_prefill.cu`](cuda/kernels_prefill.cu)), `p_ablate_post_res_kernel` runs right after `p_sconv_post_res_kernel` whenever `--ablation` is active.

### 6.4 Generating Your Own Abliterated Safetensors with `squid5090`

> **Note:** The core `squid5090` inference binary requires **zero Python**. Python + PyTorch are only needed offline if you want to run `tools/ablate.py` to generate a new custom `.safetensors` file.

1. **Install offline tooling dependencies** (into a dedicated venv so your system stays clean):
   ```bash
   python3 -m venv .venv-ablate
   source .venv-ablate/bin/activate
   pip install -r tools/requirements.txt
   ```

2. **Build `squid5090`** (used by `tools/ablate.py search` and `verify` for fast refusal evaluation):
   ```bash
   zig build -Doptimize=ReleaseFast
   ```

3. **Run the full abliteration pipeline** (replace `/opt/models/inkling-2-mini` with your model directory, or export `SQUID_MODEL_DIR`):
   ```bash
   export SQUID_MODEL_DIR=/opt/models/inkling-2-mini
   WORK=work/ablate-run1
   mkdir -p "$WORK"

   # 1. Capture per-visit harmful/harmless mean read activations (direct answer & reasoning start)
   python3 tools/ablate.py directions --out "$WORK/directions.pt"
   python3 tools/ablate.py directions --out "$WORK/directions_think.pt" --think

   # 2. Compute 48x2x2304x2304 harmless write covariances & shrinkage Fisher-LDA detectors (~10 GB host RAM)
   python3 tools/ablate.py stats --work "$WORK"

   # 3. Record original model's harmless greedy answers & top-64 logprobs (KL targets)
   python3 tools/ablate.py baseline --out "$WORK/baseline_val.pt" --split val
   python3 tools/ablate.py baseline --out "$WORK/baseline.pt" --split report

   # 4. Run Optuna multi-objective TPE search (KL in PyTorch ref, fast refusals via squid5090 serve!)
   python3 tools/ablate.py search --work "$WORK" --batches 12 --n 8

   # 5. Inspect Pareto-optimal trials (refusals vs KL damage)
   python3 tools/ablate.py show --work "$WORK" --top 20

   # 6. Export your chosen trial N into a 1.8 MB safetensors file
   python3 tools/ablate.py export --work "$WORK" --trial <N> --out "$WORK/custom-abliterated.safetensors"

   # 7. Verify held-out refusals (harmful_test + JailbreakBench) and harmless KL damage
   python3 tools/ablate.py verify \
     --ablation "$WORK/custom-abliterated.safetensors" \
     --baseline "$WORK/baseline.pt" \
     --server ./zig-out/bin/squid5090 \
     --out "$WORK/verify.json"
   ```

4. **Serve with your custom `.safetensors` and A/B test over-refusal:**
   ```bash
   # Serve with your new 1.8 MB ablation file
   ./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
     --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 --spec-threshold 0.80 \
     --ablation "$WORK/custom-abliterated.safetensors"

   # Evaluate 32 mild over-refusal prompts against the live OpenAI server
   python3 tools/refusal_eval.py --base-url http://127.0.0.1:8000/v1 --label ablated --out "$WORK/refusals-ablated.json"
   python3 tools/refusal_eval.py --compare "$WORK/refusals-original.json" "$WORK/refusals-ablated.json"
   ```

---

## 7. Attribution

All kernel designs and abliteration tooling adapted from [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) are licensed under **Apache License 2.0**; see [`NOTICE`](NOTICE) for full attributions.
