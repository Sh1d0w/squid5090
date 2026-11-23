# squid5090

**A custom, zero-dependency TigerStyle Zig + Blackwell CUDA (`sm_120a`) inference engine and OpenAI-compatible API server for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) — engineered for maximum single-user (`c=1`) decode tokens/sec and up to `1,048,576` (1M) context on a single NVIDIA RTX 5090.**

## What This Repo Does

`squid5090` is a **single-binary, zero-Python** TigerStyle Zig + CUDA engine (`zig-out/bin/squid5090`, built purely with `zig build`) that handles safetensors loading, tokenization, chat templating, GPU inference, and OpenAI-compatible HTTP/SSE serving in one process — zero PyTorch, zero vLLM, zero CMake/Make, zero Python wrappers, zero IPC:

- **100% Native Blackwell GPU Hot Path (`cuda/`):** Implements every specialized Inkling-2-Mini architecture op directly in `sm_120a` CUDA (`cvt.rn.f16x2.e2m1x2`, `mma.sync.aligned.m16n8k16`, `ldmatrix.x4` / `ldmatrix.x2.trans`) — 48 logical visits (3× looped core), AttnRes depth-attention, cross-pass controllers, static + dynamic short convolutions, L2 Q/K/R clipping, relative-bias GQA with sinks, top-8 MoE, and built-in **DSpark MTP** speculative decoding — all captured in persistent **CUDA Graphs**.
- **TigerStyle Systems Architecture (`src/`):** Written in [TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md) Zig with **243 always-on release assertions** (active even in `-Doptimize=fast`), `comptime` geometry invariants, strictly bounded loops, zero recursion, and **zero dynamic memory allocation post-`init()`**.
- **Two-Pass Single-Arena VRAM Allocator:** Computes exact 256-byte-aligned VRAM requirements upfront and allocates the entire model & runtime in **2 contiguous `cudaMalloc` bump arenas** (1 for all 1,000 weight tensors + 1 for all 359 KV/runtime slices), verified by paired `arena.assert_exact_fill()` postconditions.
- **Ultra-Fast Native Zig BPE Tokenizer & Chat Template:** Memory-maps `tokenizer.json` directly into flat open-addressed hash tables + a 1.20 MiB contiguous byte arena in **~135 ms**, encodes at **32.8M tokens/sec** with zero per-request `malloc`s and 100% exact token-for-token parity against Hugging Face `tokenizers`, and renders Inkling-2-Mini's official chat/tool template natively.
- **1M-Token Context on One RTX 5090:** Combines Inkling-2-Mini's hybrid SWA/global layout (only 8 global visits) with `--kv fp8` (up to **432K** context) and `--kv q4` + zero-copy mapped host embeddings (full **1,048,576** context in **22.47 GiB** VRAM) plus unified Tensor Core Split-K FlashDecoding.
- **Built-In Zero-Malloc OpenAI API Server (`squid5090 serve`):** Pure POSIX HTTP/1.1 + SSE server backed by static startup-allocated `ServerBuffers` — streams tokens directly from the GPU callback to the socket with zero per-token heap allocation (and aborts GPU generation immediately on client disconnect or stop-sequence match), exposing `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`.

---

## Speedup Comparison: `squid5090` vs. RTX 5090 Inference Engines (`Inkling-2-Mini`)

Single-user (`c=1`) decode throughput, VRAM footprint, and context limits for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an **NVIDIA RTX 5090** (32 GiB GDDR7). `squid5090` numbers are **6-run medians** (512 generated tokens, coding prompt P1) measured on an RTX 5090 with **23.4–24.0 GiB free VRAM** alongside Frigate (`~8.1 GiB`):

| Inference Engine | Precision / Format | Native DSpark MTP | Decode Speed (`c=1`) | `squid5090` Speedup (`mtp=3`: **369.6 tok/s**) | `squid5090` Speedup (`mtp=0`: **243.8 tok/s**) | Max Context (32 GB / 23.4 GB free) | VRAM Footprint | Source |
|---|---|---|---:|---:|---:|---:|---|---|
| **PyTorch / HF Transformers** | BF16 eager | No | ~10–15 tok/s | **~25–37×** | **~16–24×** | ~64K | >24 GiB | Roofline est. |
| **Apple Silicon `nanoattention`** *(M3 Ultra ref)* | Q4 | No | ~39 tok/s | **~9.5×** | **~6.3×** | — | Unified | Inspiration ref |
| **`squid5090` v0 Baseline** *(CUDA eager)* | `exact` (BF16+NVFP4/MXFP8) | No | **76.63 tok/s** | **4.82×** | **3.18×** | ~200K (`bf16`) | 18.76 GiB | **Measured (same 5090)** |
| **Ollama** *(GGUF backend)* | GGUF 4-bit (`Q4`/`IQ4`) | No | ~90–97 tok/s | **~3.8–4.1×** | **~2.5–2.7×** | ~128K–256K | ~18–22 GiB | Published 5090 ref |
| **`llama.cpp` / Unsloth** (`llama-server`) | GGUF (`IQ4_NL` / `UD-IQ4_XS`) | No | ~93–97 tok/s | **~3.8–4.0×** | **~2.5–2.6×** | ~256K | ~18–22 GiB | Published 5090 ref |
| **vLLM** (`vllm serve`) | Native NVFP4/MXFP8 + FP8 KV | Yes | ~190–250 tok/s | **~1.5–1.9×** | **~1.0–1.3×** | ~256K–512K | ~22–28 GiB | Published ref *(Python overhead at `c=1`)* |
| **SGLang** | Native NVFP4/MXFP8 + FP8 KV | Yes | ~250–350 tok/s | **~1.06–1.48×** | 0.70–0.98× | ~256K–512K | ~22–28 GiB | Published 5090 ref |
| **Basalt** *(same RTX 5090, chat/coding)* | Native NVFP4 + Q8, `spec=8` | Yes | **261–325 tok/s** *(prose/chat: 287–350\*)* | **1.14–1.42×** | 0.75–0.93× | 262,144 (256K) | 22.9 GiB + **8.9 GiB RAM spill** | **Measured (same 5090, CT 107)** |
| **`squid5090` (Zig + CUDA, `mtp=0`)** | All-`nvfp4`, `fp8` KV, Graphed | Off (`mtp=0`) | **243.79 tok/s** *(max 244.0)* | — | 1.00× | **1,048,576 (1M)** | **14.92 GiB** (0 spill) | **Measured (6-run median)** |
| **`squid5090` (Zig + CUDA, `mtp=3`)** | All-`nvfp4`, `fp8`/`q4` KV, Graphed | **On (`mtp=3`)** | **369.6 tok/s** *(peak 379.2 / 408.1 HTTP)* | **1.00×** | **1.52×** | **1,048,576 (1M)** | **14.85–14.92 GiB** (0 spill) | **Measured (6-run median)** |

*\*Methodology note: `llama.cpp` / Ollama / vLLM / SGLang rows reflect published RTX 5090 / community single-user numbers because GGUF runtimes do not support Inkling-2-Mini's native NVFP4 Blackwell PTX or DSpark MTP speculator, and Python-orchestrated servers (`vLLM` / `SGLang`) optimize primarily for batched multi-user throughput. Basalt was measured directly on the same RTX 5090 (CT 107) on general chat/coding prompts (**261–325 tok/s**); on highly repetitive structured/JSON outputs where draft acceptance approaches 100%, published Basalt runs reach ~613–645 tok/s. `squid5090` fits in **14.9 GiB VRAM with zero host-RAM spilling** (8 GiB less than Basalt) and scales to **4× Basalt's max context (1M tokens)**.*

### `squid5090` Long-Context Scaling (Up to 1,048,576 Tokens on One RTX 5090)

| Context Depth | Mode / KV Precision | Decode Speed (`c=1`) | Total VRAM (Weights + KV) | Notes |
|---|---|---:|---:|---|
| **Short context (P1, 512 tok)** | `nvfp4`, `mtp=3`, `fp8`/`q4` KV | **369.3–369.6 tok/s** *(peak 379.2)* | **14.85–14.92 GiB** | 6-run median, 2 `cudaMalloc` arenas |
| **16K real context** *(Needle PASS)* | `nvfp4`, `mtp=3`, `fp8` KV | **338.43 tok/s** *(prefill 17.9 s)* | **15.06 GiB** | Verbatim retrieval PASS |
| **32K real context** *(Needle PASS)* | `nvfp4`, `mtp=3`, `fp8` KV | **302.66 tok/s** *(prefill 36.3 s)* | **15.35 GiB** | Verbatim retrieval PASS |
| **64K real context** *(Needle PASS)* | `nvfp4`, `mtp=3`, `fp8` KV | **285.58 tok/s** *(prefill 75–84 s)* | **15.92 GiB** | Verbatim retrieval PASS |
| **256K** *(graphed `longctx`)* | `nvfp4`, `B=4 @dl=2.5`, `q4` KV | **199.0–202.3 tok/s** *(`B=1`: 105–115)* | **17.2 GiB** | 0 host spill |
| **512K** *(graphed `longctx`)* | `nvfp4`, `B=4 @dl=2.5`, `q4` KV | **151.4 tok/s** *(`B=1`: 72.7)* | **19.8 GiB** | 0 host spill |
| **1M (`1,048,000`)** *(`longctx`)* | `nvfp4`, `B=4 @dl=2.5`, `q4` KV | **100.6 tok/s** *(`B=1`: 46.3)* | **22.47 GiB** | Full 1M context alongside 8.1 GiB Frigate |

---

## Dependencies Needed to Build & Run

### Hardware & Drivers
- **GPU:** NVIDIA Blackwell GPU (`sm_120a`, e.g. RTX 5090) with ≥ 16 GiB free VRAM (≥ 22.5 GiB free for full 1M context).
- **NVIDIA Driver:** 580+ (tested on `590.48.01`).
- **CUDA Toolkit:** CUDA 13.0+ with `nvcc` and `cuBLAS` (tested on CUDA `13.1`, default path `/usr/local/cuda`).

### Toolchain
- **Zig `0.17.1+`** (provides the compiler, linker, and `zig build` system — replaces Make and CMake).
- **Host C++ compiler (`g++`)** required by `nvcc` for CUDA device compilation (`sudo apt-get install -y build-essential git curl`).

*`zig-out/bin/squid5090` links **only** against `libc`, `libstdc++` (for CUDA runtime), `libcudart`, `libcublas`, and `libpthread` — **zero** Python, Boost, OpenSSL, or third-party Zig/C++ packages required!*

---

## Quickstart (Build with `zig build`, Download Model, and Run)

### 1. Clone & Build (`zig build`)
```bash
git clone git@github.com:tensorfish/squid5090.git
cd squid5090

# Build zig-out/bin/squid5090 (compiles all 4 sm_120a CUDA units in parallel + ReleaseFast Zig)
zig build
```

`build.zig` caches CUDA object files by content hash (including header dependencies), so clean builds take ~30 s and incremental rebuilds take ~24 ms.

**Useful `zig build` options:**
```bash
# View all steps and options
zig build --help

# Override CUDA architecture, CUDA toolkit path, or optimization mode
zig build -Dcuda-arch=120a -Dcuda-path=/usr/local/cuda -Doptimize=fast
```

### 2. Download the Model & Where to Put It

Download [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (~20.2 GB total) into any local directory (e.g. `./models/inkling-2-mini` or `/opt/models/inkling-2-mini`):

```bash
# Using huggingface-cli (install with: pip install huggingface_hub)
huggingface-cli download thinkingmachines/Inkling-2-Mini --local-dir ./models/inkling-2-mini
```

Expected files inside `<MODEL_DIR>`:
```text
<MODEL_DIR>/
├── model-00001-of-00001.safetensors   # Main Inkling-2-Mini weights (~19.5 GB)
├── mtp.safetensors                    # Built-in DSpark MTP drafter weights (~0.64 GB)
├── tokenizer.json                     # BPE tokenizer (mmapped natively in Zig)
├── config.json                        # Model config
└── draft_vocab_32k.bin                # (Optional) Pruned 32K draft vocab IDs for faster MTP
```

### 3. Point the Binary to Your Model & Run

Every subcommand takes `<MODEL_DIR>` as its **first positional argument**:

```bash
# Start the native Zig OpenAI API server directly:
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini --port 8000 --mode nvfp4 --kv fp8 --ctx 65536 --mtp 3

# Or build & launch directly via `zig build`:
zig build serve -- /opt/models/inkling-2-mini --port 8000 --mode nvfp4 --kv fp8 --ctx 65536 --mtp 3

# One-shot native CLI generation:
./zig-out/bin/squid5090 generate /opt/models/inkling-2-mini \
  --prompt "Write a Python binary search function with type hints." \
  --mode nvfp4 --mtp 3 --max-tokens 512
```

### 4. Run the OpenAI-Compatible HTTP/SSE API Server

`squid5090 serve` loads weights into 1 contiguous `VRAMArena`, mmaps `tokenizer.json`, allocates static HTTP/SSE `ServerBuffers`, and captures CUDA Graphs — serving both streaming SSE (`stream: true`) and non-streaming JSON with **zero per-request heap allocations**:

```bash
# 64K context server (~15.9 GiB VRAM, FP8 KV)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini --host 0.0.0.0 --port 8000 --mode nvfp4 --kv fp8 --ctx 65536 --mtp 3

# 256K context server (~19.4 GiB VRAM, FP8 KV)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini --host 0.0.0.0 --port 8000 --mode nvfp4 --kv fp8 --ctx 262144 --mtp 3

# Full 1M (1,048,576) context server (~22.5 GiB VRAM, Q4 KV + auto host-pinned embed)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 1048576 --mtp 3
```

**Supported OpenAI Endpoints:**
- `GET /health` — server health, active `max_context`, and `mtp` setting
- `GET /v1/models` — OpenAI model list (`id: "inkling-2-mini"`, override via `--model-name`)
- `POST /v1/chat/completions` — streaming (`stream: true`, SSE + `stream_options.include_usage`) & non-streaming, multi-turn `messages`, `reasoning_content` separation (`<|content_thinking|>` vs `<|content_text|>`), OpenAI `tools` / `tool_calls`, `stop` sequences, `max_tokens` / `max_completion_tokens`, `reasoning_effort`, and per-request `timings`
- `POST /v1/completions` — raw text completions (streaming & non-streaming, `stop`, `echo`) for eval harnesses (`lm-evaluation-harness`, etc.)
- Optional `--api-key <secret>` (or `SQUID_API_KEY` env var) for `Authorization: Bearer <secret>` auth, plus full CORS headers.

*(Note: `squid5090` runs deterministic greedy GPU argmax decoding — ideal for coding, agents, and reproducible evals; `temperature`/`top_p` fields are accepted gracefully for harness compatibility.)*

**Test with `curl`:**
```bash
# Non-streaming chat completion
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write a 1-line Python function to reverse a string."}],
    "max_tokens": 160
  }' | jq .

# Streaming SSE
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Explain CUDA Graphs in 3 bullet points."}],
    "max_tokens": 256,
    "stream": true
  }'
```

**Integrate with OpenAI Python SDK or Any Harness (Open WebUI, LiteLLM, Continue, Aider, `lm-evaluation-harness`):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "Write a Python binary search with type hints."}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

### 5. Run Persistently as a `systemd` Service

A ready-to-use `systemd` unit is provided at [`deploy/squid5090.service`](deploy/squid5090.service). `squid5090 serve` traps `SIGTERM` and `SIGINT` for graceful shutdown and clean VRAM release:

```bash
# Install and start on boot (edit paths/flags in the unit file first if needed)
sudo cp deploy/squid5090.service /etc/systemd/system/squid5090.service
sudo systemctl daemon-reload
sudo systemctl enable --now squid5090.service

# Start / stop / restart / inspect
sudo systemctl start squid5090
sudo systemctl stop squid5090
sudo systemctl restart squid5090
sudo systemctl status squid5090
journalctl -u squid5090 -f -n 100
```

### 6. Run Benchmarks, Long-Context Sweeps & Tokenization

```bash
# 6-run median benchmark (512 tokens)
zig build bench -- /opt/models/inkling-2-mini \
  --prompt "Write a Python binary search function with type hints, docstring, and unit tests." \
  --mode nvfp4 --kv fp8 --mtp 3 --max-tokens 512 --runs 6

# Long-context scaling sweep (432K FP8 or 1M Q4)
zig build longctx -- /opt/models/inkling-2-mini --mode nvfp4 --kv fp8 --ctx 442368
zig build longctx -- /opt/models/inkling-2-mini --mode nvfp4 --kv q4  --ctx 1048576

# Tokenizer diagnostic & throughput check (no GPU required)
zig build run -- tokenize /opt/models/inkling-2-mini --prompt "Hello, world!"

# Pass very long prompts (32K–1M tokens) via file
./zig-out/bin/squid5090 generate /opt/models/inkling-2-mini --tokens-file prompt_ids.csv --mode nvfp4 --ctx 131072 --mtp 3
```

### CLI Flags & Environment Variables

| Option | Values | Description |
|---|---|---|
| `--mode` | `exact` \| `fp8` \| `nvfp4` | Weight precision (`nvfp4` quantizes dense/LM-head/L20–23 experts on load in ~2.15 s) |
| `--kv` | `auto` \| `bf16` \| `fp8` \| `q4` | Global KV cache format (`auto`: `bf16` for `exact`, `fp8` up to 448K, `q4` above 448K) |
| `--mtp` | `0..7` | Built-in DSpark MTP draft count (`0` = autoregressive, `3` = best all-around throughput) |
| `--ctx` | integer (up to `1048576`) | Maximum context window to allocate |
| `--embed-host` | flag | Pin BF16 embedding table (0.86 GiB) in zero-copy mapped host RAM (auto-enabled `>512K`) |
| `--no-graph` | flag | Disable CUDA Graph capture (for debugging) |
| `SQUID_DRAFT_VOCAB` | `32k` (default) \| `49k` \| `full` | Draft head vocabulary size (optional `draft_vocab_32k.bin`; falls back cleanly to full 200K if absent) |
| `SQUID_MTP_BF16` | `1` | Force DSpark drafter weights to BF16 (for A/B testing) |
| `SQUID_API_KEY` | string | Require `Authorization: Bearer <key>` on `squid5090 serve` |

---

## Why TigerStyle Zig? (Measured Head-to-Head vs. Earlier C++20 Engine)

Before consolidating onto a single Zig-only codebase, we benchmarked our original C++20 host engine directly against the TigerStyle Zig implementation on the exact same RTX 5090 and `sm_120a` CUDA kernels (6-run medians):

| Metric / Workload | Earlier C++20 Engine | TigerStyle Zig (`squid5090`) | Delta |
|---|---:|---:|---|
| **Parity (Tokenizer 12/12 + 66.5 KB + GPU Greedy + HTTP)** | Reference | **100% Bitwise Identical** | Exact parity |
| **Total `cudaMalloc` Calls (Weights + Runtime)** | 1,359 | **2** (`1` weights + `1` runtime arena) | **680× fewer** driver allocs |
| **Per-Request / Per-SSE-Token Host `malloc`s** | ~Hundreds (DOM + strings) | **0** (static `ServerBuffers`) | Zero-malloc hot path |
| **Release-Active Safety Assertions** | `CUDA_CHECK` only | **243** always-on `tiger.assert` + `comptime` | Active in `-Doptimize=fast` |
| **Tokenizer Cold Load (`199,998` vocab + `424,413` merges)** | 248.4 ms | **135.1 ms** | **1.84× faster** |
| **66.5 KB (`25,750` tok) Tokenize Wall Time** | 288.4 ms | **141.0 ms** (**`0.79 ms` encode = 32.8M tok/s**) | **2.05× faster** |
| **Short-Ctx P1 (`nvfp4/fp8`, `mtp=0`, 512 tok)** | 241.95 tok/s | **243.79 tok/s** *(max 243.97)* | +0.8% (GPU HBM bound) |
| **Short-Ctx P1 (`nvfp4/fp8`, `mtp=3`, 512 tok)** | 369.61 tok/s | **369.57 tok/s** *(max 373.08)* | Parity (GPU HBM bound) |
| **Short-Ctx P1 (`nvfp4/q4`, `mtp=3`, 512 tok)** | 349.54 tok/s | **369.31 tok/s** *(max 379.20)* | **+5.7%** (tighter host pacing) |
| **1M Context (`1,048,000` pos, `kv=q4`, `22.47 GiB`, `B=4`)** | 100.5 tok/s (`24.89 ms`) | **100.6 tok/s** (`24.86 ms`) | Parity (TC / KV BW bound) |
| **OpenAI HTTP Live SSE (`253` chunks, `ctx=32K`, `mtp=3`)** | 317.89 e2e tok/s (`579.3 ms` TTFT) | **320.43 e2e tok/s** (`574.8 ms` TTFT) | **+2.54 tok/s** (`0.07` vs `1.82` SSE drop) |

TigerStyle's **243 always-on release assertions** added zero measurable GPU overhead while delivering **1.84–2.05× faster tokenization**, **680× fewer CUDA driver allocations**, and **zero-malloc SSE streaming**.

---

## What We Did to Yield These GPU Results

We followed a hardware-first, roofline-driven hill-climbing methodology (full details in [THEORY.md](THEORY.md), [PLAN.md](PLAN.md), and [ledger.jsonl](ledger.jsonl)): measure real hardware ceilings first (`1,512.9 GB/s` GDDR7, 170 SMs, 96 MiB L2), optimize one lever at a time with 6-run medians, verify bitwise parity & long-context needle retrieval, and revert regressions.

| Lever | What Changed | Effect (P1, 512 tok, 6-run median unless noted) |
|---|---|---|
| **v0 — Native Baseline** | Implemented every Inkling-2-Mini op from scratch (48 visits, 3× looped core, depth-attn, cross-pass controllers, static + dynamic sconv, L2 Q/K/R clip, global log-pos scaling, rel-bias GQA + sinks, top-8 MoE, softcapped SwiGLU, final softcap30). | `exact mtp=0` eager: **76.63 tok/s** |
| **v0 + CUDA Graphs** | Captured the full decode step into a persistent `cudaGraphExec_t`, removing ~600+ host launch overheads/token. | `exact mtp=0`: **76.63 → 95.83 tok/s** (+25%) |
| **Levers 1–3 — Blackwell NVFP4 PTX & All-Linear Quant** | Native `sm_120a` PTX `cvt.rn.f16x2.e2m1x2` + `half2` FMA + 128-bit `uint4` loads; load-time NVFP4 quantization of dense/LM-head/L20–23 experts (cutting traffic `5.47 → 2.80 GB/tok`, VRAM **15.0 GiB**, zero RAM spill); fused Q+K+V+R+Gate+Sink projection. | `nvfp4 mtp=0`: **114.34 → 190.10 tok/s** |
| **Lever 4 — DSpark MTP + Expert-Grouped Verify** | Full GPU DSpark 3-layer block drafter (NVFP4) + folded Markov RNN head captured inside per-`K` CUDA Graphs (`K = 1..7`), paired with an **expert-grouped deduplicated MoE verify kernel**. | `nvfp4 mtp=3`: **190.10 → 299.78 tok/s** |
| **Lever 5 / 5.1 — Half-Warp GEMV, Mega-Fusions, 32K Draft Vocab & Overlap** | Half-warp (`16 threads/row`) NVFP4 GEMV (`lm_head` hit **1,497 GB/s = 99% of measured RTX 5090 BW**); 576-thread vectorized mega-kernels fusing `mlp_sconv + residual + depth_attn + norm + tap`; 1-launch 1024-thread NVFP4 controller; pruned 32K draft vocab (`dspark_draft 868 → 428 µs`); `aux_stream` overlap of `dspark_commit` ∥ `lm_head_argmax`. | `nvfp4 mtp=0`: **221.39 tok/s**; `nvfp4 mtp=3`: **368.49 tok/s** |
| **Lever 6 — Zero-`atomicAdd` Deterministic MoE + Global Mega-Fusion** | Replaced non-deterministic `atomicAdd` in `moe_down` with quarter-warp (`8 threads/row`) per-slot writes `[B, 8, 2304]` reduced inline (**91% peak BW** + **100% bitwise parity** across runs & `mtp=0..5`); fused all 5 global sconv/clip/store ops into `mega_global_sconv_prep_store`. | `nvfp4 mtp=3`: **372.80 tok/s** (100% deterministic) |
| **Lever 7A–7C — 1M Context + Unified Blackwell Tensor Core Split-K** | Added `--kv fp8` (up to 432K) and `--kv q4` (group-64 INT4 + 2-op exact FP16 magic-bias nibble unpack) + zero-copy host-pinned `embed` (**full 1,048,576 context in 22.47 GiB**). Replaced scalar attention across **all 48 visits + DSpark global layer 1** with Blackwell FP16 Tensor Core (`m16n8k16` + `ldmatrix.x4` + `ldmatrix.x2.trans`) Split-K FlashDecoding + captured `prefill_b8_graph_exec`. | `nvfp4 mtp=0`: **243.79 tok/s**; **16K real: `145.7 → 338.4 tok/s` (2.32×!)**; **64K real: 285.6 tok/s**; **256K: 202.3 tok/s**; **1M: 100.6 tok/s** |
| **Lever 8 — TigerStyle Zig Consolidation** | Rewrote the entire host engine, tokenizer, VRAM arena allocator, and OpenAI HTTP/SSE server in TigerStyle Zig with `build.zig`, eliminating CMake/Make/C++ host code and all per-request heap allocations. | Tokenizer load **1.84× faster**, **32.8M tok/s** BPE encode, **2** `cudaMalloc` arenas, **zero-malloc SSE** |

---

## Repository Structure

```text
squid5090/
├── build.zig                 # Native Zig build system (parallel cached nvcc sm_120a + Zig ReleaseFast)
├── deploy/
│   └── squid5090.service     # Persistent systemd service unit for the OpenAI API server
├── README.md                 # Quickstart, OpenAI server guide & optimization walkthrough
├── THEORY.md                 # Measured RTX 5090 hardware & roofline ceilings
├── PLAN.md                   # Optimization roadmap & full experiment table
├── ledger.jsonl              # Structured JSONL log of every experiment & gate
├── cuda/                     # Pure Blackwell sm_120a CUDA device kernels & C-ABI bridge
│   ├── config.hpp            # Shared CUDA architectural geometry constants
│   ├── weights.hpp           # Plain POD device-pointer descriptors
│   ├── kernels.hpp           # Kernel launcher declarations
│   ├── kernels_common.cuh    # Blackwell PTX (e2m1, e4m3, m16n8k16, ldmatrix), KV pack/unpack, TC Split-K
│   ├── kernels_gemv.cu       # Half-warp NVFP4/FP8/BF16 GEMV, 99%-BW lm_head, deduplicated zero-atomic MoE
│   ├── kernels_inkling.cu    # Mega-fused depth-attn/sconv, 1-launch controllers, router, TC local/global attention
│   ├── kernels_dspark.cu     # DSpark 3-layer block drafter, TC global layer 1, folded Markov RNN head
│   └── cuda_abi.cu           # Thin extern "C" bridge exposing sm_120a kernels & CUDA runtime to Zig
└── src/                      # 100% TigerStyle Zig engine & OpenAI server
    ├── tiger.zig             # Always-on release assertions, monotonic clock, ByteWriter, POSIX C-ABI
    ├── constants.zig         # Comptime-verified Inkling-2-Mini & DSpark geometry + exact VRAM accounting
    ├── cuda.zig              # Explicit extern "C" kernel & CUDA runtime bindings
    ├── vram_arena.zig        # Single-cudaMalloc bump VRAMArena with paired exact-fill assertions
    ├── safetensors.zig       # POSIX mmap safetensors loader with static 2048-tensor hash table
    ├── tokenizer.zig         # Flat open-addressed BPE tokenizer (32.8M tok/s) & Inkling chat template
    ├── weights.zig           # Two-pass weight loader into 1 contiguous VRAMArena
    ├── engine.zig            # Single runtime VRAMArena, 48-visit CUDA Graph orchestrator & streaming
    ├── server.zig            # Zero-malloc static-buffer POSIX HTTP/1.1 + SSE OpenAI API server
    └── main.zig              # Single-binary CLI entry point (serve, generate, bench, longctx, tokenize)
```
