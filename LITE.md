# `squid5090` — The Plain-English Guide (`LITE.md`)

Everything we learned making a 1M-context LLM run at **589–771 tokens/sec** on a single RTX 5090 — and how to abliterate it without hurting its intelligence — explained simply, with zero CUDA jargon.

---

## Part 1: How to Make Any Local LLM Fast (The 5 Core Ideas)

If you want to optimize an inference engine on your own hardware, almost every big speedup comes from just **five ideas**, in this exact order:

### 1. Know which speed limit you're hitting first
LLM generation has two totally different phases:
- **Generating tokens (Decode)** is like **moving boxes through a narrow doorway** (memory bandwidth bound). To generate *one* word, the GPU has to read billions of weights from VRAM while its math units sit 95% idle.
- **Reading your prompt (Prefill)** is like **factory assembly** (compute bound). Processing thousands of prompt tokens at once keeps the math units busy.

**Rule #1:** Measure your GPU's real memory speed first (an RTX 5090 moves ~`1,513 GB/s`), divide by the size of your model's active weights per token, and you immediately know the theoretical speed limit. Don't guess — do the division!

### 2. Stop making the CPU babysit the GPU (CUDA Graphs)
Normally, the CPU tells the GPU what to do ~600 times for every single token ("run norm, now run matmul, now run attention..."). Those tiny phone calls add up fast.
- **Fix:** Record the entire token step once as a **CUDA Graph**, then tell the GPU *"replay the whole recording"* with **one** call. Instant ~25–100% speedup.

### 3. Shrink the bytes moving through the doorway (4-bit Weights & 4-bit KV Cache)
Because decode is limited by memory bandwidth, halving the bytes you read nearly **doubles** your speed:
- Packing weights into **4-bit NVFP4** (which Blackwell GPUs unpack in hardware) cut our traffic from `5.47 GB/token` to `2.80 GB/token` and shrank the model to `14.6 GiB` VRAM.
- Packing long-context memory (KV cache) into **4-bit Q4** let us fit **1,048,576 tokens (1M context!)** inside `23.2 GiB` on a single 32 GiB RTX 5090 — with zero slow spillover to system RAM.

### 4. Guess ahead for free (Speculative Decoding + Prompt-Lookup)
Once you hit the memory speed limit (~260 tok/s for 1 token at a time), how do you hit **589–771 tok/s**?
- **The trick:** Reading weights from VRAM for **8 tokens at once** takes almost the same time as reading them for **1 token**!
- So we use two tiny, super-fast **drafters** to guess the next 7 tokens ahead of time, then let the big model check all 8 in one shot:
  1. **Prompt-Lookup (microseconds, CPU):** If the last 16 tokens match something already in the prompt or earlier output (editing code, quoting a file, repeating boilerplate), copy the next 7 tokens verbatim and skip the neural drafter completely (**771.6 tok/s**!).
  2. **DSpark Drafter (0.4 ms, GPU):** Otherwise, Inkling-2-Mini's built-in tiny 3-layer drafter predicts the next 7 tokens + how confident it is about each one.
  3. **Adaptive Switch (on-GPU):** If the drafter is very confident (like writing Python), the GPU verifies **8 tokens** at once. If it's unsure (like creative chat), the GPU automatically switches down to **2 or 4 tokens** so it doesn't waste work on bad guesses.
- **100% lossless:** The big model checks every single guessed token and rejects anything that doesn't match. You get the **exact same output**, just 2–3× faster.

### 5. Always profile again after every big win
After we sped up prompt reading (prefill) with a huge `128×128` Tensor Core MoE kernel, we profiled each stage and found a surprise: a tiny 4-tap convolution was taking **26% of all prefill time** because it was only using **9 out of 170 GPU cores**! Spreading it across 576 blocks took 30 lines of code and jumped prefill from `7,689` to **`9,181 tok/s`**!

---

## Part 2: What We Actually Did in `squid5090`

Following the roofline methodology from [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill) plus Blackwell kernel designs from `ingine` and hybrid lookup drafting from `atomic-engine`:

- **Single zero-Python binary (`zig build`):** Replaced Python/PyTorch/vLLM with TigerStyle Zig + native Blackwell `sm_120a` CUDA. Only **2 `cudaMalloc` calls** total, zero per-token heap allocations, `~3 s` cold start.
- **Decode (`76.6 → 589.5 tok/s` code, `771.6 tok/s` code-copy, `332.5 tok/s` chat):** NVFP4 hardware unpack + Blackwell PDL kernel overlap + Tensor Core decode MoE + on-GPU adaptive `{2, 4, 8}` DSpark switch + O(1) Prompt-Lookup drafter.
- **Prefill (`900 → 9,181 tok/s` chunk peak, `8,755 tok/s` at 20K, `7,742 tok/s` at 64K):** `B=2048` multi-stage async Tensor Core MoE + 2D-tiled short-conv + custom FlashAttention.
- **1M Context + Instant Multi-Turn Cache (`23.2 GiB` VRAM):** Q4 global KV cache + host-RAM prefix checkpoints reuse a 16K–82K conversation prefix in **62–72 ms** (up to **1,400× faster** than re-reading it).

---

## Part 3: Abliteration — Why Normal Methods Break & How Ours Works

### What is abliteration?
Safety-tuned models often over-refuse benign requests (creative fiction, villain dialogue, security education, edgy humor) with canned *"I'm sorry, I cannot..."* replies. Inside the model's hidden activations, "refusal" turns out to point mostly along a single direction in vector space. **Abliteration** removes that refusal direction so the model stops refusing.

### Why standard abliteration ruins Inkling-2-Mini
Normally people permanently edit the model's weight matrices to erase the refusal direction. On Inkling-2-Mini that fails badly for two reasons:
1. **Shared looped layers:** Inkling-2-Mini loops its middle 12 layers **3 times** (48 logical steps from 24 physical layers). Editing a physical weight hits all 3 passes at once — breaking general reasoning and coding.
2. **4-bit quantization:** Tiny surgical weight edits get crushed into noise when re-quantized to 4-bit NVFP4.

### The Smart Solution: A 1.8 MB "Bouncer" at Runtime (`--ablation`)
Instead of touching the 19 GB model weights at all, we keep the base model **100% untouched** and load a tiny **1.8 MB `.safetensors` sidecar** that runs at each of the 48 logical steps as a smart, gated filter:

1. **The Detector (the bouncer):** At each step, a trained statistical detector (**Fisher-LDA**, calibrated against normal harmless coding/chat/text) asks: *"Does this activation look abnormally like a refusal — more than $k$ standard deviations away from normal text?"*
   - **On normal coding & chat:** The bouncer says **No** → **nothing happens**! Output is **bit-for-bit identical** to the unmodified base model (`0%` intelligence loss).
   - **When a refusal starts forming:** The bouncer says **Yes** → it gently subtracts *only* the excess refusal signal above normal harmless text.
2. **Per-step control:** Because it runs at runtime across all 48 logical steps, it can act strongly on the exact loop passes that trigger refusal and stay completely off (`weight = 0`) on the passes that do coding and math.
3. **Zero speed penalty on normal text & 541 tok/s when active:** Fused directly into our existing GPU residual kernels — zero extra kernel launches!

### How to Generate Your Own Abliterated `.safetensors` (Using `uv`)

Core `squid5090` needs **zero Python**. Only the offline generator in `tools/` uses Python via [`uv`](https://docs.astral.sh/uv/):

```bash
# 1. Install offline generator deps into an isolated uv environment
uv sync --project tools

# 2. Build squid5090 (used during search to test refusals at 500+ tok/s!)
zig build -Doptimize=ReleaseFast

# 3. Point to your Inkling-2-Mini model folder and run the pipeline
export SQUID_MODEL_DIR=/opt/models/inkling-2-mini
WORK=work/my-ablation && mkdir -p "$WORK"

# Step A: Measure how harmful vs harmless prompts look inside all 48 steps
uv run --project tools tools/ablate.py directions --out "$WORK/directions.pt"
uv run --project tools tools/ablate.py directions --out "$WORK/directions_think.pt" --think

# Step B: Measure normal harmless variance & build the Fisher-LDA "bouncer" detectors
uv run --project tools tools/ablate.py stats --work "$WORK"

# Step C: Record original harmless answers so we can measure any quality drift (KL)
uv run --project tools tools/ablate.py baseline --out "$WORK/baseline_val.pt" --split val
uv run --project tools tools/ablate.py baseline --out "$WORK/baseline.pt" --split report

# Step D: Let Optuna search for the sweet spot (lowest refusals + lowest quality drift)
uv run --project tools tools/ablate.py search --work "$WORK" --batches 12 --n 8

# Step E: Pick the best trial from `show` and export your 1.8 MB .safetensors!
uv run --project tools tools/ablate.py show --work "$WORK" --top 20
uv run --project tools tools/ablate.py export --work "$WORK" --trial <N> --out "$WORK/my-abliterated.safetensors"
```

Then serve it directly with `squid5090`:
```bash
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 \
  --ablation "$WORK/my-abliterated.safetensors"

# Quick 32-prompt over-refusal check (zero dependencies via uv!)
uv run tools/refusal_eval.py --base-url http://127.0.0.1:8000/v1 --label mine
```

*(Want every formula, PTX instruction, and benchmark table? Read [`LEARNINGS.md`](LEARNINGS.md) and [`OPTIMIZATIONS.md`](OPTIMIZATIONS.md).)*
