# Optimization Audit: `squid5090` vs. `inference-optimisation-skill`

This document records what we learned and applied from [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill) (by 0xSero / Sybil Solutions), what we adapted from Blackwell/Apple-Silicon open-source engines, and — just as importantly — **what techniques from `inference-optimisation-skill` we have *not* implemented yet** and where the remaining headroom lives.

---

## 1. Headline Results (Single RTX 5090, 32 GiB GDDR7)

| Workload | Baseline (`v0` eager) | Current `squid5090` | Speedup |
|---|---:|---:|---:|
| **Single-token decode (`mtp=0`)** | 76.6 tok/s | **259.0–262.8 tok/s** | **3.4×** |
| **Code decode (1,024 tok, adaptive `mtp=7` + prompt-lookup)** | 76.6 tok/s | **589.5 tok/s** (`acc=73.1%`, `dl=3.82`) | **7.7×** |
| **Verbatim code / RAG quote (hybrid prompt-lookup + DSpark)** | 76.6 tok/s | **771.6 tok/s** (`acc=91.5%`, `dl=5.40`, `38/105` lookup hits) | **10.1×** |
| **Chat decode (adaptive `mtp=7`)** | 76.6 tok/s | **332.5 tok/s** (`acc=43.5%`, `dl=1.83`) | **4.3×** |
| **Ablated code decode (`--ablation`, 1.8 MB gated LDA)** | — | **541.1 tok/s** (0 harmless divergence) | — |
| **Cold prefill (20K / 64K Needle PASS / 131K)** | ~900 tok/s | **8,755 / 7,742 / 6,509 tok/s** (**9,181 tok/s** chunk peak) | **~9.7×** |
| **Warm multi-turn prefill (16.1K / 82.5K cached)** | ~18,000–90,000 ms | **71.9 ms / 62.4 ms** | **250–1,400×** |
| **Max context (`--kv q4`, 0 host-RAM spill)** | ~200K (`18.8 GiB`) | **1,048,576 (1M)** in **23.22 GiB** (**108.0 tok/s** `B=4`) | **5.2× ctx** |

---

## 2. What We Learned & Applied from `sybil-solutions/inference-optimisation-skill`

| Skill Principle / Lever | Source in Skill | How We Applied It in `squid5090` | Measured Impact |
|---|---|---|---|
| **Measure real hardware ceilings first, never trust spec sheets** | `SKILL.md` §0–1, `ceilings.md` | Measured real RTX 5090 GDDR7 copy BW (`1,512.9 GB/s`), 170 SMs, 96 MiB L2, and P0 clocks (`14,001 MHz` mem) into [`THEORY.md`](THEORY.md); verified `lm_head` reached `1,497 GB/s` (**99% of ceiling**). | Prevented chasing false bottlenecks; proved `B=1` caps ~265 tok/s and speculation is mandatory for 500+ tok/s. |
| **Structured hill-climbing artifacts (`THEORY.md`, `PLAN.md`, `ledger.jsonl`)** | `SKILL.md` §0, §2.9 | Logged every lever, latency delta, and revert in [`THEORY.md`](THEORY.md), [`PLAN.md`](PLAN.md), and [`ledger.jsonl`](ledger.jsonl). | Zero repeated dead-ends; clean attribution per lever. |
| **Decode graphs first** | `SKILL.md` §3.2, `levers.md` | Captured full 48-visit decode into persistent `cudaGraphExec_t` (and later CUDA 13.1 `cudaGraphCondTypeSwitch` for `{2, 4, 8}`). | `76.6 → 95.8 tok/s` immediately; eliminated ~600 kernel launch overheads/step. |
| **Shrink weight bytes with hardware-native quant** | `levers.md` (Decode) | Load-time NVFP4 quantization of dense, `lm_head`, and L20–23 experts (`5.47 → 2.80 GB/tok`) decoded via Blackwell `cvt.rn.f16x2.e2m1x2`. | `114.3 → 190.1 tok/s` (`mtp=0`), VRAM down to `14.57 GiB`. |
| **Prune the draft LM head vocabulary** | `SKILL.md` §3.3, `levers.md` | Pruned DSpark draft vocab from `200K → 32K` (`draft_vocab_32k.bin`), while target verifies against full 200K (100% lossless). | Cut `dspark_draft` latency in half (`868 → 428 µs`). |
| **Keep sampling/argmax on the GPU** | `SKILL.md` §3.3, `levers.md` | GPU-side `lm_head_argmax`, GPU-side Markov argmax + online softmax confidence, and GPU-side `spec_verify` + `z_adaptive_select_width_kernel`. | Eliminated multi-ms host round-trips per decode round. |
| **Verify must use decode-style attention** | `SKILL.md` §3.3, `levers.md` | Unified Blackwell FP16 Tensor Core Split-K FlashDecoding across all 48 visits + DSpark global layer 1 for both `B=1` and `B=2..8` verify. | 16K real context jumped `145.7 → 338.4 tok/s` (`2.32×`), 1M pos hits `108.0 tok/s`. |
| **Sweep draft length per workload & judge `a / step_time`** | `levers.md`, `methodology.md` | Confirmed fixed `B=8` wins on code (`511 tok/s`) but *loses* on chat (`270` vs `307 tok/s`), motivating device-side adaptive `{2, 4, 8}` switch graphs (`--spec-threshold 0.80`) + hybrid prompt-lookup (`min_match_len = 16`). | **589.5 tok/s** code, **771.6 tok/s** quote, **332.5 tok/s** chat — beats every fixed depth! |
| **Explicitly enable prefix caching & verify `cached_tokens > 0`** | `SKILL.md` §3.4, `levers.md` | Built 4-checkpoint host-RAM local-state prefix cache (`0` extra VRAM) + 4K global-KV side-request shield, reporting `usage.prompt_tokens_details.cached_tokens`. | Turn-2 prefill dropped from `2,110 ms → 71.9 ms` (16K) and `~90 s → 62.4 ms` (82.5K). |
| **Profile prefill by stage before coding kernels** | `levers.md`, `methodology.md` | Built `SQUID_PROF_PREFILL=1`; after `B=2048` WGEMM cut MoE to `92 ms`, profiling exposed `p_sconv_post_res_kernel` taking `70.3 ms` on only 9 CTAs! Tiling it to 576 CTAs took 30 lines and saved `42.9 ms`. | Chunk prefill jumped `7,689 → 9,181 tok/s` (`20K`: `8,755 tok/s`, `64K`: `7,742 tok/s`). |
| **Grouped MoE GEMMs: avoid padding cliffs & reuse decoded weights** | `levers.md` (Prefill) | Replaced `B=512, TM=32` prefill MoE with `B=2048, TM=128 × TN=128 × BK=64` multi-stage `cp.async` grouped GEMM (4× expert weight reuse, pigeonhole `<= 256` tiles). | MoE prefill 2.4× faster (`92.2 ms` for 2,048 tokens across 48 visits). |
| **Avoid silent memory-outside-allocator OOMs** | `levers.md` (Memory) | Replaced 1,359 individual allocations with 2 contiguous `VRAMArena` bump allocators (`assert_exact_fill()`), zero post-init host `malloc`, and `8.8 GiB` VRAM headroom at 1M ctx. | Zero fragmentation, zero OOMs, `0.07 tok/s` run-to-run jitter. |
| **Always check greedy equivalence, numerics, and retrieval needles** | `SKILL.md` §2.6, §2.8, `methodology.md` | Verified bit-exact token parity on Code/Chat/Quote, explained 1-ULP `B=1` (FP16-pair) vs `B>=2` (FP32-accum MMA) tie breaks instead of chasing phantoms, and passed 64K verbatim Needle-in-a-Haystack. | Caught the `min_match_len = 6` prompt-lookup regression (`-21 tok/s`) immediately and fixed it with `min_match_len = 16`. |
| **Dedicated idle GPU (`P0` clocks) & proper token counting** | `SKILL.md` §2.3, `methodology.md` | Stopped background GPU workloads (lifting clock state `P1 → P0`, `+10%` decode) and counted tokens via real tokenizer / `usage` stats (never SSE chunk count). | Clean, reproducible measurements (`±0.1 tok/s`). |

---

## 3. What We Did *Not* Apply Yet from `inference-optimisation-skill`

Auditing every single reference file in `inference-optimisation-skill` reveals **5 genuine unexploited levers** we haven't implemented yet, plus several stack-specific levers that do not apply to `squid5090`:

### A. Genuine Remaining Headroom (Candidates for Future Work)

1. **FP8 / INT8 Tensor Core Prefill Dense GEMMs (`levers.md` — Prefill):**
   - *What the skill teaches:* Prefill is compute-bound, so running dense projections in FP8/INT8 Tensor Cores (as a clearly labelled lossy trade, e.g., per-row scales) gives `~1.5–1.9×` faster linear GEMMs with `<0.2%` NLL change.
   - *Where `squid5090` is today:* `launch_prefill_gemm_h` dequantizes NVFP4/FP8/BF16 weights to an FP16 scratch buffer and runs FP16 `cublasGemmEx`. At `B=2048`, `dense_tc` is `29.5 ms` (`13%` of chunk time). Moving to native Blackwell FP8 `cublasLtMatmul` (or NVFP4 `mxfp4` TC MMA once exposed cleanly) could shave `~12–15 ms` per 2,048-token chunk (pushing chunk prefill toward **`10,000+ tok/s`**).
2. **SASS / `cuobjdump` ISA Disassembly Audit (`levers.md` — Kernel Work & `methodology.md`):**
   - *What the skill teaches:* When dominant kernels appear "at the limit", inspect the compiled assembly (`cuobjdump --dump-sass`) for hidden register spills, bank conflicts, stride moves, or serialized epilogue tails.
   - *Where `squid5090` is today:* We tuned at the PTX/CUDA level (`cvt.rn.f16x2.e2m1x2`, `mma.m16n8k16`, `ldmatrix.x4`, `cp.async`, PDL), but have not yet done a full `sm_120a` SASS disassembly pass on `moe_down_tc_kernel`, `moe_up_tc_kernel`, or `p_wgemm_kernel`.
3. **Multi-User Continuous Batching (`cN` Concurrency, `SKILL.md` §3.4 & `levers.md`):**
   - *What the skill teaches:* Size KV pools and hybrid state slots for `N` concurrent streams (`c4`, `c8`, `c16`) and verify true concurrent overlap with full-length soaks.
   - *Where `squid5090` is today:* `squid5090` is deliberately single-slot (`c=1`) to maximize single-user responsiveness (`589.5 tok/s`) and fit 1M context in `23.22 GiB`. Supporting `c2`–`c4` concurrent users would require paged KV tables and per-slot sconv/local-KV ring banks.
4. **Automated Interleaved `control → candidate → control` Harness + Bootstrap 95% CI & Full 1M-Token 3-Needle Soak (`methodology.md` & `publishing.md`):**
   - *What the skill teaches:* Automate interleaved A/B pairs with bootstrap 95% confidence intervals (`median(B)/median(A) > 1.0`) and run a 3-needle retrieval check at `~50%` and `~100%` of the advertised context length.
   - *Where `squid5090` is today:* We verified real 64K verbatim Needle-in-a-Haystack PASS (`8.46 s`), 131K real prefill (`19.97 s`), and 1M graphed scaling (`longctx` at `1,048,000` pos), but have not yet run a real 1,048,000-token end-to-end 3-needle prefill+retrieval soak (which would take ~3.5 minutes of prefill) or wrapped our CLI bench in an automated bootstrap-CI harness.
5. **Wider Prompt-Lookup Verify Graph (`B=16`) + Early-Exit Markov RNN:**
   - *Why it matters:* Hybrid Prompt-Lookup already jumped verbatim code quotes from `706.9 → 771.6 tok/s` (`91.5%` acceptance) using `B=8` verify (`7` drafts). Adding a `B=16` verify graph solely for long prompt-lookup hits (`>= 16` token suffix matches) could push code-editing/RAG bursts past `1,000 tok/s`, and early-exiting Markov RNN steps `3..6` when draft 0 confidence is below threshold could save `~100–150 µs` on low-confidence chat steps.

### B. Techniques in the Skill That Do Not Apply to `squid5090` (and Why)

- **JIT / Triton / Inductor / DeepGEMM / SYCL cache persistence & `TORCHINDUCTOR_COMPILE_THREADS=1`:** `squid5090` is an AOT-compiled native Zig + `nvcc` binary with zero Python/PyTorch; startup is already `~3.0 s`.
- **vLLM / SGLang / `llama.cpp` / ExLlama engine flags** (`--safetensors-load-strategy eager`, `--max-mamba-cache-size`, `expandable_segments:True`, `FULL_DECODE_ONLY`): All replaced by custom zero-malloc Zig `mmap` loading and static 2-arena `VRAMArena`.
- **Unified-memory / non-NVIDIA workarounds** (`drm-resident-*`, `iogpu.wired_limit_mb`, Intel Xe L1-uncached split-K fences, ROCm HIP graph leaks, Ampere INT4-over-FP8): `squid5090` specifically targets discrete NVIDIA Blackwell (`sm_120a`).
- **Swapping in an off-the-shelf attention library (`oneDNN` / FA2):** Inkling-2-Mini uses L2-clipped Q/K/R, learned relative-position bias projections, attention sinks, and Q4 global KV — requiring our custom `sm_120a` Tensor Core attention kernels.

---

## 4. Techniques We Added *Beyond* `inference-optimisation-skill`

In addition to the skill's methodology, four architectural breakthroughs pushed `squid5090` past `589 tok/s` (`771 tok/s` on quotes) and `9.2K tok/s` prefill:
1. **Blackwell Programmatic Dependent Launch (PDL)** + **register-permuted Tensor Core `mma.m16n8k16` decode MoE** (adapted from [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine)).
2. **CUDA 13.1 Graph Conditional Switch (`cudaGraphCondTypeSwitch`)** for zero-host-round-trip adaptive `{2, 4, 8}` verification + overlapped next-round drafting (adapted from `ingine`).
3. **Hybrid O(1) Prompt-Lookup + DSpark Fallback Drafter** (inspired by [`AtomicBot-ai/atomic-engine`](https://github.com/AtomicBot-ai/atomic-engine/tree/codex/expert-weight-source), upgraded with a zero-malloc 2-entry hash index and `min_match_len = 16` precision filter).
4. **2D Time-Tiled 4-Tap FIR Short-Conv (`p_sconv_post_res_kernel`)** + **Runtime-Gated Fisher-LDA Ablation (`--ablation`)** fused directly into decode and prefill kernels.
