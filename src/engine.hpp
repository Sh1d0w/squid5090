#pragma once

#include "config.hpp"
#include "kernels.hpp"
#include "weights.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <functional>
#include <vector>

namespace squid {

struct GenStats {
    int prompt_tokens = 0;
    int gen_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int spec_steps = 0;
    int total_drafts = 0;
    int total_accepted = 0; // accepted drafts (excluding free bonus/target token)
    std::vector<int32_t> tokens;
};

class Engine {
public:
    Engine() = default;
    ~Engine();

    bool init(const ModelWeights* weights, int max_context = 16384, bool use_graphs = true, KVFormat kv_fmt = KVFormat::BF16);
    void reset();

    // Run greedy generation from prompt token IDs.
    // num_drafts: 0 = standard autoregressive decode, 1..7 = DSpark MTP speculative decode!
    // Optional on_token(tok) callback is invoked live per emitted token; return false to stop early.
    GenStats generate(
        const std::vector<int32_t>& prompt,
        int max_new_tokens,
        int num_drafts = 0,
        const std::function<bool(int32_t)>& on_token = nullptr);

    // Run a single forward pass on prompt and return last-token logits [VOCAB_UNPADDED] on host (for verification)
    std::vector<float> forward_logits(const std::vector<int32_t>& prompt);

    // Per-stage GPU timing breakdown (in microseconds) for B tokens + optional DSpark draft!
    // If sim_pos > 0, simulates KV occupancy at sim_pos after warming up prompt (for long-ctx profiling).
    void profile_step(const std::vector<int32_t>& prompt, int B, int num_drafts, int sim_pos = 0);

    // Measure raw graphed step latency (ms/step) at synthetic KV occupancy `sim_pos` (for 128K..1M scaling sweeps)
    double bench_step_at_pos(int sim_pos, int num_drafts, int iters = 32);

    size_t runtime_vram_bytes() const { return runtime_vram_bytes_; }

private:
    void record_target_forward(int B, bool commit_dspark, bool capture_logits, bool compute_lm_head = true);
    void ensure_b1_graph();
    void ensure_spec_graph(int num_drafts);
    void ensure_prefill_b8_graph(bool commit_dspark);
    void* alloc_rt(size_t bytes);

    const ModelWeights* w_ = nullptr;
    VisitInfo schedule_[NUM_VISITS];
    int max_ctx_ = 16384;
    bool use_graphs_ = true;
    KVFormat kv_fmt_ = KVFormat::BF16;
    int cur_pos_ = 0;

    cudaStream_t stream_ = nullptr;
    cudaStream_t aux_stream_ = nullptr;
    cudaEvent_t fork_ev_ = nullptr;
    cudaEvent_t join_ev_ = nullptr;
    std::vector<void*> rt_allocs_;
    size_t runtime_vram_bytes_ = 0;

    // Per-visit persistent caches (local visits: BF16; global visits: kv_fmt_)
    void* k_cache_[NUM_VISITS] = {};
    void* v_cache_[NUM_VISITS] = {};
    __half* k_scale_[NUM_VISITS] = {};
    __half* v_scale_[NUM_VISITS] = {};
    int kv_cap_[NUM_VISITS] = {};
    float* k_ring_[NUM_VISITS] = {};
    float* v_ring_[NUM_VISITS] = {};
    float* attn_ring_[NUM_VISITS] = {};
    float* mlp_ring_[NUM_VISITS] = {};

    // DSpark persistent caches (layers 0,2: local BF16; layer 1: global kv_fmt_)
    void* ds_k_cache_[MTP_LAYERS] = {};
    void* ds_v_cache_[MTP_LAYERS] = {};
    __half* ds_k_scale_[MTP_LAYERS] = {};
    __half* ds_v_scale_[MTP_LAYERS] = {};
    int ds_kv_cap_[MTP_LAYERS] = {};
    float* ds_f_ring_ = nullptr; // [16, HIDDEN]
    float* split_buf_ = nullptr; // [600000] Split-K global attention partials (2.4 MB)

    // Static forward scratch buffers (sized for MAX_VERIFY_B = 8)
    int32_t* d_pos_ = nullptr;          // [1]
    int32_t* d_tokens_ = nullptr;       // [MAX_VERIFY_B]
    int32_t* d_next_tokens_ = nullptr;  // [MAX_VERIFY_B]
    int32_t* d_anchor_ = nullptr;       // [1]
    int32_t* d_drafts_ = nullptr;       // [MTP_MAX_DRAFTS]
    int32_t* d_accepted_ = nullptr;     // [MAX_VERIFY_B]
    int32_t* d_num_acc_ = nullptr;      // [1]

    // Host pinned staging for zero-overhead step readback
    int32_t* h_accepted_ = nullptr;     // [MAX_VERIFY_B]
    int32_t* h_num_acc_ = nullptr;      // [1]

    float* hidden_ = nullptr;           // [B, HIDDEN]
    float* prelude_ = nullptr;          // [B, HIDDEN]
    float* partial_ = nullptr;          // [B, HIDDEN]
    float* blocks_ = nullptr;           // [B, 4, HIDDEN]
    float* saved_ = nullptr;            // [B, 1, HIDDEN]
    float* prior_last_ = nullptr;       // [B, HIDDEN]
    float* attn_in_[3] = {};            // 3 x [B, HIDDEN]
    __half* attn_in_h_[3] = {};         // 3 x [B, HIDDEN] half for zero-SMEM L1 GEMV
    float* qrgs_ = nullptr;             // [B, 6552]
    float* k_proj_ = nullptr;           // [B, KV_DIM]
    float* v_proj_ = nullptr;           // [B, KV_DIM]
    float* ck_ = nullptr;               // [B, KV_DIM]
    float* cv_ = nullptr;               // [B, KV_DIM]
    float* sconv_taps_ = nullptr;       // [B, 32]
    float* rel_bias_ = nullptr;         // [B, 24, 1024]
    float* attn_gated_ = nullptr;       // [B, 3072]
    __half* attn_gated_h_ = nullptr;    // [B, 3072] half
    float* o_proj_ = nullptr;           // [B, HIDDEN]
    float* attn_out_ = nullptr;         // [B, HIDDEN]
    float* residual_ = nullptr;         // [B, HIDDEN]
    float* mlp_in_ = nullptr;           // [B, HIDDEN]
    float* mlp_normed_ = nullptr;       // [B, HIDDEN]
    __half* mlp_normed_h_ = nullptr;    // [B, HIDDEN] half
    float* ctrl_msg_ = nullptr;         // [3, B, HIDDEN]
    float* ctrl_cat_ = nullptr;         // [3, B, 2*HIDDEN]
    float* ctrl_d_ = nullptr;           // [3, B, GATE_RANK]
    float* router_logits_ = nullptr;    // [B, 128]
    int32_t* route_ids_ = nullptr;      // [B, TOP_K]
    float* route_w_ = nullptr;          // [B, TOP_K]
    ExpertGroups* exp_groups_ = nullptr;
    __half* moe_act_h_ = nullptr;       // [B, TOP_K, INTERMEDIATE] half
    float* moe_down_ = nullptr;         // [B, HIDDEN] (used when B == 1)
    float* moe_slot_out_ = nullptr;     // [B, TOP_K, HIDDEN] (deterministic zero-atomic B>1!)
    float* ff_out_ = nullptr;           // [B, HIDDEN]
    float* taps_ = nullptr;             // [4, B, HIDDEN]
    float* ds_mix_ = nullptr;           // [3, B, HIDDEN]
    float* ds_kv_ = nullptr;            // [B, 2*KV_DIM]
    float* ds_ws_ = nullptr;            // DSpark block + Markov workspace
    float* lm_blk_val_ = nullptr;       // [B, 1024]
    int32_t* lm_blk_idx_ = nullptr;     // [B, 1024]
    float* d_logits_ = nullptr;         // optional [VOCAB_UNPADDED] for verify

    // Captured CUDA Graphs
    cudaGraphExec_t b1_graph_exec_ = nullptr;
    cudaGraphExec_t spec_graph_exec_[MTP_MAX_DRAFTS + 1] = {};
    cudaGraphExec_t prefill_b8_graph_exec_[2] = {}; // [0]: no dspark commit, [1]: commit dspark
};

} // namespace squid
