#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

namespace squid {

namespace {

__global__ void inc_pos_and_copy_tok_kernel(int32_t* d_pos, int32_t* d_tok_in, const int32_t* d_tok_out) {
    if (threadIdx.x == 0) {
        *d_pos += 1;
        *d_tok_in = *d_tok_out;
    }
}

__global__ void inc_pos_by_kernel(int32_t* d_pos, int delta) {
    if (threadIdx.x == 0) *d_pos += delta;
}

__global__ void assemble_verify_tokens_kernel(
    const int32_t* d_anchor, const int32_t* d_drafts, int32_t* d_tokens, int num_drafts) {
    int i = threadIdx.x;
    if (i == 0) d_tokens[0] = *d_anchor;
    else if (i <= num_drafts) d_tokens[i] = d_drafts[i - 1];
}

} // namespace

Engine::~Engine() {
    if (b1_graph_exec_) cudaGraphExecDestroy(b1_graph_exec_);
    for (int k = 0; k <= MTP_MAX_DRAFTS; ++k) {
        if (spec_graph_exec_[k]) cudaGraphExecDestroy(spec_graph_exec_[k]);
    }
    for (int c = 0; c < 2; ++c) {
        if (prefill_b8_graph_exec_[c]) cudaGraphExecDestroy(prefill_b8_graph_exec_[c]);
    }
    if (h_accepted_) cudaFreeHost(h_accepted_);
    if (h_num_acc_) cudaFreeHost(h_num_acc_);
    for (void* p : rt_allocs_) {
        if (p) cudaFree(p);
    }
    if (fork_ev_) cudaEventDestroy(fork_ev_);
    if (join_ev_) cudaEventDestroy(join_ev_);
    if (aux_stream_) cudaStreamDestroy(aux_stream_);
    if (stream_) cudaStreamDestroy(stream_);
}

void* Engine::alloc_rt(size_t bytes) {
    void* ptr = nullptr;
    size_t aligned = (bytes + 255) & ~size_t(255);
    CUDA_CHECK(cudaMalloc(&ptr, aligned));
    rt_allocs_.push_back(ptr);
    runtime_vram_bytes_ += aligned;
    return ptr;
}

bool Engine::init(const ModelWeights* weights, int max_context, bool use_graphs, KVFormat kv_fmt) {
    w_ = weights;
    max_ctx_ = max_context;
    use_graphs_ = use_graphs;
    kv_fmt_ = kv_fmt;
    build_visit_schedule(schedule_);
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&fork_ev_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&join_ev_, cudaEventDisableTiming));

    auto alloc_kv = [&](int cap, KVFormat fmt, void*& k_out, void*& v_out, __half*& ks_out, __half*& vs_out) {
        if (fmt == KVFormat::FP8) {
            size_t data_bytes = (size_t)cap * KV_DIM * sizeof(uint8_t);
            size_t scale_bytes = (size_t)cap * NUM_KV_HEADS * sizeof(__half);
            k_out = alloc_rt(data_bytes);
            v_out = alloc_rt(data_bytes);
            ks_out = static_cast<__half*>(alloc_rt(scale_bytes));
            vs_out = static_cast<__half*>(alloc_rt(scale_bytes));
        } else if (fmt == KVFormat::Q4) {
            size_t data_bytes = (size_t)cap * (KV_DIM / 2) * sizeof(uint8_t);
            size_t scale_bytes = (size_t)cap * KV_Q4_SCALES_PER_TOK * sizeof(__half);
            k_out = alloc_rt(data_bytes);
            v_out = alloc_rt(data_bytes);
            ks_out = static_cast<__half*>(alloc_rt(scale_bytes));
            vs_out = static_cast<__half*>(alloc_rt(scale_bytes));
        } else {
            size_t data_bytes = (size_t)cap * KV_DIM * sizeof(__nv_bfloat16);
            k_out = alloc_rt(data_bytes);
            v_out = alloc_rt(data_bytes);
            ks_out = nullptr;
            vs_out = nullptr;
        }
    };

    // Allocate persistent per-visit KV caches and 16-slot sconv rings
    for (int v = 0; v < NUM_VISITS; ++v) {
        // Local visits only need SWA_WINDOW + MAX_VERIFY_B slots (always BF16: 169 MB total)
        kv_cap_[v] = schedule_[v].is_global ? max_ctx_ : (SWA_WINDOW + MAX_VERIFY_B);
        KVFormat vfmt = schedule_[v].is_global ? kv_fmt_ : KVFormat::BF16;
        alloc_kv(kv_cap_[v], vfmt, k_cache_[v], v_cache_[v], k_scale_[v], v_scale_[v]);
        k_ring_[v]    = static_cast<float*>(alloc_rt(16 * KV_DIM * sizeof(float)));
        v_ring_[v]    = static_cast<float*>(alloc_rt(16 * KV_DIM * sizeof(float)));
        attn_ring_[v] = static_cast<float*>(alloc_rt(16 * HIDDEN * sizeof(float)));
        mlp_ring_[v]  = static_cast<float*>(alloc_rt(16 * HIDDEN * sizeof(float)));
    }

    // DSpark KV caches & f_ring (cap draft-only global layer at 64K when max_ctx > 512K so 1M target fits!)
    if (w_->dspark.loaded) {
        for (int l = 0; l < MTP_LAYERS; ++l) {
            bool is_loc = is_mtp_local_layer(l);
            ds_kv_cap_[l] = is_loc ? (SWA_WINDOW + MAX_VERIFY_B)
                                   : ((max_ctx_ > 524288) ? 65536 : max_ctx_);
            KVFormat lfmt = is_loc ? KVFormat::BF16 : kv_fmt_;
            alloc_kv(ds_kv_cap_[l], lfmt, ds_k_cache_[l], ds_v_cache_[l], ds_k_scale_[l], ds_v_scale_[l]);
        }
        ds_f_ring_ = static_cast<float*>(alloc_rt(16 * HIDDEN * sizeof(float)));
    }
    split_buf_ = static_cast<float*>(alloc_rt(2050000 * sizeof(float)));

    // Static activation scratch arena for MAX_VERIFY_B = 8
    constexpr int B = MAX_VERIFY_B;
    d_pos_         = static_cast<int32_t*>(alloc_rt(sizeof(int32_t)));
    d_tokens_      = static_cast<int32_t*>(alloc_rt(B * sizeof(int32_t)));
    d_next_tokens_ = static_cast<int32_t*>(alloc_rt(B * sizeof(int32_t)));
    d_anchor_      = static_cast<int32_t*>(alloc_rt(sizeof(int32_t)));
    d_drafts_      = static_cast<int32_t*>(alloc_rt(MTP_MAX_DRAFTS * sizeof(int32_t)));
    d_accepted_    = static_cast<int32_t*>(alloc_rt(B * sizeof(int32_t)));
    d_num_acc_     = static_cast<int32_t*>(alloc_rt(sizeof(int32_t)));

    CUDA_CHECK(cudaMallocHost(&h_accepted_, B * sizeof(int32_t)));
    CUDA_CHECK(cudaMallocHost(&h_num_acc_, sizeof(int32_t)));

    hidden_     = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    prelude_    = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    partial_    = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    blocks_     = static_cast<float*>(alloc_rt(B * 4 * HIDDEN * sizeof(float)));
    saved_      = static_cast<float*>(alloc_rt(B * 1 * HIDDEN * sizeof(float)));
    prior_last_ = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    for (int i = 0; i < 3; ++i) {
        attn_in_[i] = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
        attn_in_h_[i] = static_cast<__half*>(alloc_rt(B * HIDDEN * sizeof(__half)));
    }

    qrgs_       = static_cast<float*>(alloc_rt(B * (Q_DIM + R_DIM + Q_DIM + NUM_HEADS) * sizeof(float)));
    k_proj_     = static_cast<float*>(alloc_rt(B * KV_DIM * sizeof(float)));
    v_proj_     = static_cast<float*>(alloc_rt(B * KV_DIM * sizeof(float)));
    ck_         = static_cast<float*>(alloc_rt(B * KV_DIM * sizeof(float)));
    cv_         = static_cast<float*>(alloc_rt(B * KV_DIM * sizeof(float)));
    sconv_taps_ = static_cast<float*>(alloc_rt(B * 32 * sizeof(float)));
    rel_bias_   = static_cast<float*>(alloc_rt(B * NUM_HEADS * REL_EXTENT * sizeof(float)));
    attn_gated_ = static_cast<float*>(alloc_rt(B * Q_DIM * sizeof(float)));
    attn_gated_h_ = static_cast<__half*>(alloc_rt(B * Q_DIM * sizeof(__half)));
    o_proj_     = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    attn_out_   = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    residual_   = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    mlp_in_     = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    mlp_normed_ = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    mlp_normed_h_ = static_cast<__half*>(alloc_rt(B * HIDDEN * sizeof(__half)));
    ctrl_msg_   = static_cast<float*>(alloc_rt(3 * B * HIDDEN * sizeof(float)));
    ctrl_cat_   = static_cast<float*>(alloc_rt(3 * B * 2 * HIDDEN * sizeof(float)));
    ctrl_d_     = static_cast<float*>(alloc_rt(3 * B * GATE_RANK * sizeof(float)));
    router_logits_ = static_cast<float*>(alloc_rt(B * NUM_EXPERTS * sizeof(float)));
    route_ids_  = static_cast<int32_t*>(alloc_rt(B * TOP_K * sizeof(int32_t)));
    route_w_    = static_cast<float*>(alloc_rt(B * TOP_K * sizeof(float)));
    exp_groups_ = static_cast<ExpertGroups*>(alloc_rt(sizeof(ExpertGroups)));
    moe_act_h_  = static_cast<__half*>(alloc_rt(B * TOP_K * INTERMEDIATE * sizeof(__half)));
    moe_down_   = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    moe_slot_out_ = static_cast<float*>(alloc_rt(B * TOP_K * HIDDEN * sizeof(float)));
    ff_out_     = static_cast<float*>(alloc_rt(B * HIDDEN * sizeof(float)));
    taps_       = static_cast<float*>(alloc_rt(4 * B * HIDDEN * sizeof(float)));
    ds_mix_     = static_cast<float*>(alloc_rt(3 * B * HIDDEN * sizeof(float)));
    ds_kv_      = static_cast<float*>(alloc_rt(B * 2 * KV_DIM * sizeof(float)));
    // DSpark workspace: ~400K floats + 7 * 200058 floats = ~1.8M floats (~7.2 MB)
    ds_ws_      = static_cast<float*>(alloc_rt(2000000 * sizeof(float)));
    lm_blk_val_ = static_cast<float*>(alloc_rt(B * 1024 * sizeof(float)));
    lm_blk_idx_ = static_cast<int32_t*>(alloc_rt(B * 1024 * sizeof(int32_t)));

    const char* kv_str = (kv_fmt_ == KVFormat::FP8) ? "fp8" : (kv_fmt_ == KVFormat::Q4 ? "q4" : "bf16");
    printf("[engine] Initialized max_ctx=%d | kv=%s | runtime VRAM=%.2f GiB | graphs=%s\n",
           max_ctx_, kv_str, runtime_vram_bytes_ / (1024.0 * 1024.0 * 1024.0), use_graphs_ ? "on" : "off");
    reset();
    return true;
}

void Engine::reset() {
    cur_pos_ = 0;
    CUDA_CHECK(cudaMemsetAsync(d_pos_, 0, sizeof(int32_t), stream_));
    for (int v = 0; v < NUM_VISITS; ++v) {
        CUDA_CHECK(cudaMemsetAsync(k_ring_[v], 0, 16 * KV_DIM * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(v_ring_[v], 0, 16 * KV_DIM * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(attn_ring_[v], 0, 16 * HIDDEN * sizeof(float), stream_));
        CUDA_CHECK(cudaMemsetAsync(mlp_ring_[v], 0, 16 * HIDDEN * sizeof(float), stream_));
    }
    if (ds_f_ring_) CUDA_CHECK(cudaMemsetAsync(ds_f_ring_, 0, 16 * HIDDEN * sizeof(float), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::record_target_forward(int B, bool commit_dspark, bool capture_logits, bool compute_lm_head) {
    // 1. Embed + embed_norm
    launch_embed_norm(w_->embed, w_->embed_norm, d_tokens_, hidden_, B, stream_);

    int block_count = 0;
    bool has_prior = false;
    bool prev_mlp_pending = false;

    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        const VisitInfo& v = schedule_[vi];
        const LayerWeights& lw = w_->layers[v.physical_layer];
        int l = v.physical_layer;

        // Optional pre_tap_out for fused mlp_post_res(vi-1) -> hidden -> pre_tap
        float* pre_tap_out = nullptr;

        // Loop pass boundary management
        if (v.is_loop && l == LOOP_START) {
            if (v.pass == 0) {
                // vi == 6: fuse hidden_ -> prelude_ directly into mega_mlp_post_next_depth_norm!
                pre_tap_out = prelude_;
            } else {
                // vi == 18, 30: prev_mlp_pending was flushed at vi == 17, 29
                launch_extract_block(blocks_, 4, 2, saved_, B, stream_);
                launch_rmsnorm(hidden_, nullptr, prior_last_, B, HIDDEN, HIDDEN, RMS_EPS, stream_);
                CUDA_CHECK(cudaMemcpyAsync(hidden_, prelude_, (size_t)B * HIDDEN * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream_));
                block_count = 1;
                has_prior = true;
            }
        }
        bool use_prior = v.is_loop && has_prior;

        // DSpark target taps at visit inputs 5, 23, 41 (all have prev_mlp_pending == true -> 0 memcpys!)
        if (commit_dspark) {
            int tap_slot = (vi == MTP_TARGET_VISITS[0]) ? 0
                         : (vi == MTP_TARGET_VISITS[1]) ? 1
                         : (vi == MTP_TARGET_VISITS[2]) ? 2 : -1;
            if (tap_slot >= 0) {
                pre_tap_out = taps_ + (size_t)tap_slot * B * HIDDEN;
            }
        }

        // FUSED: [optional mlp_post_res(vi-1)] + partial = rmsnorm(hidden) + depth_attn + pre-attn norm (+ half!)
        bool distinct_inputs = v.is_global || use_prior;
        float* partial_dst = (l % BLOCK_LAYERS == 0) ? partial_ : nullptr;
        bool use_half_dense = (lw.w_qrgs.kind != WeightKind::BF16);

        const __nv_bfloat16 *n0 = nullptr, *n1 = nullptr, *n2 = nullptr;
        float *o0 = nullptr, *o1 = nullptr, *o2 = nullptr;
        __half *oh0 = nullptr, *oh1 = nullptr, *oh2 = nullptr;
        int n_out = 1;

        if (!distinct_inputs) {
            n0 = lw.attn_norm; o0 = attn_in_[0];
            oh0 = use_half_dense ? attn_in_h_[0] : nullptr;
            n_out = 1;
        } else if (v.is_global && !use_prior) {
            n0 = lw.q_in_norm; n1 = lw.k_in_norm; n2 = lw.v_in_norm;
            o0 = attn_in_[0];  o1 = attn_in_[1];  o2 = attn_in_[2];
            oh0 = use_half_dense ? attn_in_h_[0] : nullptr;
            oh1 = use_half_dense ? attn_in_h_[1] : nullptr;
            oh2 = use_half_dense ? attn_in_h_[2] : nullptr;
            n_out = 3;
        } else if (v.is_global) {
            o0 = attn_in_[0]; o1 = attn_in_[1]; o2 = attn_in_[2];
            n_out = 3;
        } else {
            o0 = mlp_in_;
            n_out = 1;
        }

        if (prev_mlp_pending) {
            const LayerWeights& prev_lw = w_->layers[schedule_[vi - 1].physical_layer];
            launch_mega_mlp_post_next_depth_norm(
                (B > 1) ? moe_slot_out_ : moe_down_, (B > 1),
                prev_lw.mlp_sconv_w, mlp_ring_[vi - 1], residual_,
                hidden_, pre_tap_out, nullptr, partial_dst, blocks_, block_count,
                w_->attn_queries[l - 1], n0, n1, n2, o0, o1, o2, oh0, oh1, oh2,
                n_out, B, d_pos_, stream_);
            prev_mlp_pending = false;
        } else {
            launch_fused_depth_norm(
                hidden_, partial_dst, blocks_, block_count,
                (l > 0) ? w_->attn_queries[l - 1] : nullptr,
                n0, n1, n2, o0, o1, o2, oh0, oh1, oh2, n_out, B, stream_);
        }

        if (use_prior) {
            int loc = l - LOOP_START;
            float* xs[3] = {v.is_global ? attn_in_[0] : mlp_in_, v.is_global ? attn_in_[1] : mlp_in_, v.is_global ? attn_in_[2] : mlp_in_};
            float* outs[3] = {attn_in_[0], attn_in_[1], attn_in_[2]};
            __half* outs_h[3] = {attn_in_h_[0], attn_in_h_[1], attn_in_h_[2]};
            const float* cp_q[3] = {
                w_->cross_pass_queries + (size_t)(2 * loc) * HIDDEN,
                w_->cross_pass_kv_queries + (size_t)(2 * loc) * HIDDEN,
                w_->cross_pass_kv_queries + (size_t)(2 * loc + 1) * HIDDEN
            };
            const ControllerWeights* cws[3] = {&lw.ctrl[0], &lw.ctrl[1], &lw.ctrl[2]};
            const __nv_bfloat16* pnorms[3] = {
                v.is_global ? lw.q_in_norm : lw.attn_norm,
                v.is_global ? lw.k_in_norm : lw.attn_norm,
                v.is_global ? lw.v_in_norm : lw.attn_norm
            };
            if (w_->mode == QuantMode::NVFP4) {
                launch_controller_mega_nvfp4(
                    xs, saved_, prior_last_, cp_q, cws, pnorms, outs, use_half_dense ? outs_h : nullptr, 3, B, stream_);
            } else {
                launch_controller_merge_batched(
                    xs, saved_, prior_last_, cp_q, cws, pnorms, outs, use_half_dense ? outs_h : nullptr, 3,
                    ctrl_msg_, ctrl_cat_, ctrl_d_, B, stream_);
            }
        }

        const float* in_q = attn_in_[0];
        const float* in_k = distinct_inputs ? attn_in_[1] : attn_in_[0];
        const float* in_v = distinct_inputs ? attn_in_[2] : attn_in_[0];
        const __half* in_qh = attn_in_h_[0];
        const __half* in_kh = distinct_inputs ? attn_in_h_[1] : attn_in_h_[0];
        const __half* in_vh = distinct_inputs ? attn_in_h_[2] : attn_in_h_[0];

        // Attention linear projections:
        if (lw.w_qrgs.kind == WeightKind::NVFP4) {
            launch_qkvrg_fused(lw, in_qh, in_kh, in_vh, qrgs_, k_proj_, v_proj_, B, stream_);
        } else if (lw.w_qrgs.kind == WeightKind::FP8) {
            launch_linear_h(lw.w_qrgs, in_qh, qrgs_, B, false, stream_);
            launch_linear_h(lw.w_k, in_kh, k_proj_, B, false, stream_);
            launch_linear_h(lw.w_v, in_vh, v_proj_, B, false, stream_);
        } else {
            launch_linear(lw.w_qrgs, in_q, qrgs_, B, false, stream_);
            launch_linear(lw.w_k, in_k, k_proj_, B, false, stream_);
            launch_linear(lw.w_v, in_v, v_proj_, B, false, stream_);
        }

        // K & V short convolutions + Q/R/K clip + V norm + KV cache store (1 kernel for BOTH global & local!)
        if (v.is_global) {
            launch_mega_global_sconv_prep_store(
                qrgs_, k_proj_, v_proj_, in_k, in_v, lw.k_sconv_w, lw.v_sconv_w,
                lw.k_sconv_proj, lw.v_sconv_proj, k_ring_[vi], v_ring_[vi],
                lw.q_clip_scale, lw.rel_clip_scale,
                k_cache_[vi], v_cache_[vi], k_scale_[vi], v_scale_[vi], kv_fmt_,
                kv_cap_[vi], B, d_pos_, stream_);
        } else {
            launch_mega_local_sconv_prep_store(
                qrgs_, k_proj_, v_proj_, lw.k_sconv_w, lw.v_sconv_w,
                k_ring_[vi], v_ring_[vi], lw.q_clip_scale, lw.rel_clip_scale,
                static_cast<__nv_bfloat16*>(k_cache_[vi]), static_cast<__nv_bfloat16*>(v_cache_[vi]),
                kv_cap_[vi], B, d_pos_, stream_);
        }

        // Relative bias (inline SMEM!) + Tensor Core Split-K Global/Local Attention with sink + Gated output (+ half!)
        launch_relative_attention_gated(
            qrgs_, lw.rel_logits_proj, k_cache_[vi], v_cache_[vi],
            k_scale_[vi], v_scale_[vi], v.is_global ? kv_fmt_ : KVFormat::BF16,
            split_buf_,
            use_half_dense ? nullptr : attn_gated_, use_half_dense ? attn_gated_h_ : nullptr,
            kv_cap_[vi], v.is_global ? 0 : SWA_WINDOW, B, d_pos_, stream_);

        // O projection
        if (use_half_dense) launch_linear_h(lw.w_o, attn_gated_h_, o_proj_, B, false, stream_);
        else                launch_linear(lw.w_o, attn_gated_, o_proj_, B, false, stream_);

        // MEGA-FUSION 3: attn_sconv + block boundary/residual + mlp_depth_attn + optional mlp_norm (+ half)!
        bool is_boundary = (l % BLOCK_LAYERS == 0);
        launch_mega_attn_post_mlp_in(
            o_proj_, lw.attn_sconv_w, attn_ring_[vi],
            hidden_, partial_, blocks_, block_count, is_boundary,
            residual_, w_->mlp_queries[l],
            use_prior ? nullptr : lw.mlp_norm,
            use_prior ? mlp_in_ : mlp_normed_,
            use_prior ? nullptr : mlp_normed_h_,
            B, d_pos_, stream_);
        if (is_boundary) block_count++;

        if (use_prior) {
            int loc = l - LOOP_START;
            float* xs[1] = {mlp_in_};
            float* outs[1] = {mlp_normed_};
            __half* outs_h[1] = {mlp_normed_h_};
            const float* cp_q[3] = {w_->cross_pass_queries + (size_t)(2 * loc + 1) * HIDDEN, nullptr, nullptr};
            const ControllerWeights* cws[3] = {&lw.ctrl[3], nullptr, nullptr};
            const __nv_bfloat16* pnorms[3] = {lw.mlp_norm, nullptr, nullptr};
            if (w_->mode == QuantMode::NVFP4) {
                launch_controller_mega_nvfp4(
                    xs, saved_, prior_last_, cp_q, cws, pnorms, outs, outs_h, 1, B, stream_);
            } else {
                launch_controller_merge_batched(
                    xs, saved_, prior_last_, cp_q, cws, pnorms, outs, outs_h, 1,
                    ctrl_msg_, ctrl_cat_, ctrl_d_, B, stream_);
            }
        }

        // MoE Router (32x64 parallel + SMEM-staged ExpertGroups) + Deterministic ZERO-atomic MoE!
        if (w_->mode == QuantMode::NVFP4) {
            launch_route_topk_nvfp4(
                mlp_normed_h_, lw.router_nvfp4, lw.router_bias, router_logits_, route_ids_, route_w_,
                (B > 1) ? exp_groups_ : nullptr, nullptr, B, stream_);
        } else {
            launch_route_topk(
                mlp_normed_, lw.router_w, lw.router_bias, router_logits_, route_ids_, route_w_,
                (B > 1) ? exp_groups_ : nullptr, nullptr, B, stream_);
        }
        launch_moe_gate_up_h(lw.w13, mlp_normed_h_, route_ids_, exp_groups_, moe_act_h_, B, stream_);
        launch_moe_down_h(
            lw.w2, moe_act_h_, route_ids_, route_w_, exp_groups_,
            (B > 1) ? moe_slot_out_ : moe_down_, B, stream_, true);

        // Flush mlp_post_res ONLY before loop restarts (vi == 17, 29); otherwise fuse into next visit!
        if (vi == 17 || vi == 29) {
            launch_mega_mlp_post_res(
                (B > 1) ? moe_slot_out_ : moe_down_, (B > 1),
                lw.mlp_sconv_w, mlp_ring_[vi], residual_, hidden_, B, d_pos_, stream_);
            prev_mlp_pending = false;
        } else {
            prev_mlp_pending = true;
        }
    }

    // FUSED: visit 47 mlp_post_res + final output depth attention + final norm (+ tap 3!)
    launch_mega_mlp_post_next_depth_norm(
        (B > 1) ? moe_slot_out_ : moe_down_, (B > 1),
        w_->layers[schedule_[NUM_VISITS - 1].physical_layer].mlp_sconv_w, mlp_ring_[NUM_VISITS - 1],
        residual_, hidden_, nullptr,
        commit_dspark ? (taps_ + (size_t)3 * B * HIDDEN) : nullptr,
        nullptr, blocks_, block_count, w_->output_query,
        w_->final_norm, nullptr, nullptr,
        mlp_normed_, nullptr, nullptr, nullptr, nullptr, nullptr,
        1, B, d_pos_, stream_);

    // Overlap dspark_commit (aux_stream_) CONCURRENTLY with lm_head_argmax (stream_)!
    if (compute_lm_head) {
        if (commit_dspark) {
            CUDA_CHECK(cudaEventRecord(fork_ev_, stream_));
            CUDA_CHECK(cudaStreamWaitEvent(aux_stream_, fork_ev_, 0));
            launch_dspark_commit(
                w_->dspark, taps_, ds_k_cache_, ds_v_cache_, ds_k_scale_, ds_v_scale_,
                ds_kv_cap_, kv_fmt_, ds_f_ring_, ds_mix_, ds_kv_, B, d_pos_, aux_stream_);
            CUDA_CHECK(cudaEventRecord(join_ev_, aux_stream_));
        }
        launch_lm_head_argmax(
            w_->unembed, mlp_normed_, d_next_tokens_, capture_logits ? d_logits_ : nullptr,
            lm_blk_val_, lm_blk_idx_, B, stream_);
        if (commit_dspark) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, join_ev_, 0));
        }
    } else if (commit_dspark) {
        launch_dspark_commit(
            w_->dspark, taps_, ds_k_cache_, ds_v_cache_, ds_k_scale_, ds_v_scale_,
            ds_kv_cap_, kv_fmt_, ds_f_ring_, ds_mix_, ds_kv_, B, d_pos_, stream_);
    }
}

void Engine::ensure_b1_graph() {
    if (b1_graph_exec_ || !use_graphs_) return;
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    record_target_forward(1, false, false, true);
    inc_pos_and_copy_tok_kernel<<<1, 1, 0, stream_>>>(d_pos_, d_tokens_, d_next_tokens_);
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&b1_graph_exec_, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

void Engine::ensure_prefill_b8_graph(bool commit_dspark) {
    int slot = commit_dspark ? 1 : 0;
    if (prefill_b8_graph_exec_[slot] || !use_graphs_) return;
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    record_target_forward(MAX_VERIFY_B, commit_dspark, false, false);
    inc_pos_by_kernel<<<1, 1, 0, stream_>>>(d_pos_, MAX_VERIFY_B);
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&prefill_b8_graph_exec_[slot], graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

void Engine::ensure_spec_graph(int num_drafts) {
    if (spec_graph_exec_[num_drafts] || !use_graphs_) return;
    int B = num_drafts + 1;
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    // 1. Draft `num_drafts` tokens from `d_anchor_` at `d_pos_`
    const __half* ds_ks_c[MTP_LAYERS] = {ds_k_scale_[0], ds_k_scale_[1], ds_k_scale_[2]};
    const __half* ds_vs_c[MTP_LAYERS] = {ds_v_scale_[0], ds_v_scale_[1], ds_v_scale_[2]};
    launch_dspark_draft(
        *w_, d_anchor_, d_pos_, num_drafts,
        ds_k_cache_, ds_v_cache_, ds_ks_c, ds_vs_c, ds_kv_cap_, kv_fmt_, split_buf_,
        ds_f_ring_, ds_ws_, d_drafts_, stream_);
    // 2. Assemble verify batch [anchor, draft_0 .. draft_{K-1}]
    assemble_verify_tokens_kernel<<<1, B, 0, stream_>>>(d_anchor_, d_drafts_, d_tokens_, num_drafts);
    // 3. Verify all B tokens in ONE target forward pass + commit DSpark taps
    record_target_forward(B, true, false, true);
    // 4. Verify on GPU: accept 1..B tokens, advance d_pos_, update d_anchor_
    launch_spec_verify(d_next_tokens_, d_drafts_, B, d_accepted_, d_num_acc_, d_pos_, d_anchor_, stream_);
    CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&spec_graph_exec_[num_drafts], graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
}

std::vector<float> Engine::forward_logits(const std::vector<int32_t>& prompt) {
    reset();
    if (!d_logits_) {
        d_logits_ = static_cast<float*>(alloc_rt((size_t)VOCAB_UNPADDED * sizeof(float)));
    }
    int n = (int)prompt.size();
    for (int i = 0; i < n; ) {
        int b = std::min(MAX_VERIFY_B, n - i);
        bool is_last = (i + b == n);
        if (is_last && b > 1) b = b - 1; // run last token alone so d_logits_ has 1 row!
        CUDA_CHECK(cudaMemcpyAsync(d_tokens_, prompt.data() + i, b * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_, &cur_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        record_target_forward(b, false, (i + b == n));
        cur_pos_ += b;
        i += b;
    }
    std::vector<float> out(VOCAB_UNPADDED);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), d_logits_, VOCAB_UNPADDED * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return out;
}

GenStats Engine::generate(
    const std::vector<int32_t>& prompt,
    int max_new_tokens,
    int num_drafts,
    const std::function<bool(int32_t)>& on_token) {
    GenStats stats;
    stats.prompt_tokens = (int)prompt.size();
    if (prompt.empty() || max_new_tokens <= 0) return stats;
    if ((int)prompt.size() >= max_ctx_) return stats;
    if ((int)prompt.size() + max_new_tokens > max_ctx_) {
        max_new_tokens = max_ctx_ - (int)prompt.size();
    }

    bool use_mtp = (num_drafts > 0 && w_->dspark.loaded);
    if (num_drafts > MTP_MAX_DRAFTS) num_drafts = MTP_MAX_DRAFTS;

    reset();

    // --- PREFILL ---
    int n = (int)prompt.size();
    if (use_graphs_ && n > MAX_VERIFY_B) {
        ensure_prefill_b8_graph(use_mtp);
    }
    auto tp0 = std::chrono::steady_clock::now();
    int i = 0;
    if (use_graphs_ && n > MAX_VERIFY_B) {
        cudaGraphExec_t pgraph = prefill_b8_graph_exec_[use_mtp ? 1 : 0];
        // d_pos_ is already 0 from reset(); graph increments d_pos_ by 8 each chunk!
        while (i + MAX_VERIFY_B < n) {
            CUDA_CHECK(cudaMemcpyAsync(d_tokens_, prompt.data() + i, MAX_VERIFY_B * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaGraphLaunch(pgraph, stream_));
            cur_pos_ += MAX_VERIFY_B;
            i += MAX_VERIFY_B;
        }
    }
    for (; i < n; ) {
        int b = std::min(MAX_VERIFY_B, n - i);
        bool is_last = (i + b == n);
        CUDA_CHECK(cudaMemcpyAsync(d_tokens_, prompt.data() + i, b * sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_pos_, &cur_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        record_target_forward(b, use_mtp, false, is_last);
        cur_pos_ += b;
        i += b;
    }
    // First generated token is d_next_tokens_[last_b - 1]
    int last_b = ((n - 1) % MAX_VERIFY_B) + 1;
    int32_t first_tok = 0;
    CUDA_CHECK(cudaMemcpyAsync(&first_tok, d_next_tokens_ + (last_b - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_, &cur_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    auto tp1 = std::chrono::steady_clock::now();
    stats.prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    stats.tokens.push_back(first_tok);
    stats.gen_tokens = 1;
    if (on_token && !on_token(first_tok)) return stats;
    if (first_tok == EOS_TOKEN || stats.gen_tokens >= max_new_tokens) return stats;

    // Warmup/instantiate graph before timing decode? Graphs are cached on Engine across calls!
    if (!use_mtp) {
        CUDA_CHECK(cudaMemcpyAsync(d_tokens_, &first_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        ensure_b1_graph();
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_anchor_, &first_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        ensure_spec_graph(num_drafts);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // --- DECODE ---
    auto td0 = std::chrono::steady_clock::now();
    if (!use_mtp) {
        while (stats.gen_tokens < max_new_tokens) {
            if (use_graphs_) {
                CUDA_CHECK(cudaGraphLaunch(b1_graph_exec_, stream_));
            } else {
                record_target_forward(1, false, false);
                inc_pos_and_copy_tok_kernel<<<1, 1, 0, stream_>>>(d_pos_, d_tokens_, d_next_tokens_);
            }
            CUDA_CHECK(cudaMemcpyAsync(h_accepted_, d_next_tokens_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            int32_t tok = h_accepted_[0];
            stats.tokens.push_back(tok);
            stats.gen_tokens++;
            cur_pos_++;
            if (on_token && !on_token(tok)) break;
            if (tok == EOS_TOKEN) break;
        }
    } else {
        int B = num_drafts + 1;
        while (stats.gen_tokens < max_new_tokens) {
            if (use_graphs_) {
                CUDA_CHECK(cudaGraphLaunch(spec_graph_exec_[num_drafts], stream_));
            } else {
                const __half* ds_ks_c[MTP_LAYERS] = {ds_k_scale_[0], ds_k_scale_[1], ds_k_scale_[2]};
                const __half* ds_vs_c[MTP_LAYERS] = {ds_v_scale_[0], ds_v_scale_[1], ds_v_scale_[2]};
                launch_dspark_draft(
                    *w_, d_anchor_, d_pos_, num_drafts,
                    ds_k_cache_, ds_v_cache_, ds_ks_c, ds_vs_c, ds_kv_cap_, kv_fmt_, split_buf_,
                    ds_f_ring_, ds_ws_, d_drafts_, stream_);
                assemble_verify_tokens_kernel<<<1, B, 0, stream_>>>(d_anchor_, d_drafts_, d_tokens_, num_drafts);
                record_target_forward(B, true, false, true);
                launch_spec_verify(d_next_tokens_, d_drafts_, B, d_accepted_, d_num_acc_, d_pos_, d_anchor_, stream_);
            }
            CUDA_CHECK(cudaMemcpyAsync(h_num_acc_, d_num_acc_, sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaMemcpyAsync(h_accepted_, d_accepted_, B * sizeof(int32_t), cudaMemcpyDeviceToHost, stream_));
            CUDA_CHECK(cudaStreamSynchronize(stream_));

            int n_acc = *h_num_acc_;
            stats.spec_steps++;
            stats.total_drafts += num_drafts;
            stats.total_accepted += (n_acc - 1);
            cur_pos_ += n_acc;

            bool stop_now = false;
            for (int i = 0; i < n_acc && stats.gen_tokens < max_new_tokens; ++i) {
                int32_t tok = h_accepted_[i];
                stats.tokens.push_back(tok);
                stats.gen_tokens++;
                if (on_token && !on_token(tok)) { stop_now = true; break; }
                if (tok == EOS_TOKEN) { stop_now = true; break; }
            }
            if (stop_now) break;
        }
    }
    auto td1 = std::chrono::steady_clock::now();
    stats.decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
    return stats;
}

void Engine::profile_step(const std::vector<int32_t>& prompt, int B, int num_drafts, int sim_pos) {
    // Warm up caches with prompt + 64 decode tokens so pos ~ 80
    (void)generate(prompt, 64, num_drafts > 0 ? num_drafts : 0);
    if (sim_pos > cur_pos_ && sim_pos + B < max_ctx_) {
        cur_pos_ = sim_pos;
        CUDA_CHECK(cudaMemcpyAsync(d_pos_, &cur_pos_, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    auto timed = [&](auto&& fn) -> double {
        CUDA_CHECK(cudaEventRecord(ev0, stream_));
        fn();
        CUDA_CHECK(cudaEventRecord(ev1, stream_));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        return ms * 1000.0; // microseconds
    };

    double t_embed = 0, t_attn_in = 0, t_ctrl_attn = 0, t_attn_proj = 0, t_sconv_kv = 0;
    double t_prep = 0, t_rel_attn = 0, t_wo = 0, t_attn_post = 0;
    double t_mlp_in = 0, t_ctrl_mlp = 0, t_router = 0, t_group = 0;
    double t_moe_gu = 0, t_moe_dn = 0, t_mlp_post = 0, t_head = 0, t_ds_commit = 0;
    double t_ds_draft = 0;

    if (num_drafts > 0 && w_->dspark.loaded) {
        const __half* ds_ks_c[MTP_LAYERS] = {ds_k_scale_[0], ds_k_scale_[1], ds_k_scale_[2]};
        const __half* ds_vs_c[MTP_LAYERS] = {ds_v_scale_[0], ds_v_scale_[1], ds_v_scale_[2]};
        t_ds_draft = timed([&] {
            launch_dspark_draft(
                *w_, d_anchor_, d_pos_, num_drafts,
                ds_k_cache_, ds_v_cache_, ds_ks_c, ds_vs_c, ds_kv_cap_, kv_fmt_, split_buf_,
                ds_f_ring_, ds_ws_, d_drafts_, stream_);
        });
    }

    t_embed = timed([&] {
        launch_embed_norm(w_->embed, w_->embed_norm, d_tokens_, hidden_, B, stream_);
    });

    int block_count = 0;
    bool has_prior = false;
    bool prev_mlp_pending = false;
    for (int vi = 0; vi < NUM_VISITS; ++vi) {
        const VisitInfo& v = schedule_[vi];
        const LayerWeights& lw = w_->layers[v.physical_layer];
        int l = v.physical_layer;

        float* pre_tap_out = nullptr;
        if (v.is_loop && l == LOOP_START) {
            if (v.pass == 0) {
                pre_tap_out = prelude_;
            } else {
                launch_extract_block(blocks_, 4, 2, saved_, B, stream_);
                launch_rmsnorm(hidden_, nullptr, prior_last_, B, HIDDEN, HIDDEN, RMS_EPS, stream_);
                CUDA_CHECK(cudaMemcpyAsync(hidden_, prelude_, (size_t)B * HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
                block_count = 1;
                has_prior = true;
            }
        }
        bool use_prior = v.is_loop && has_prior;
        bool distinct_inputs = v.is_global || use_prior;

        float* partial_dst = (l % BLOCK_LAYERS == 0) ? partial_ : nullptr;
        bool use_half_dense = (lw.w_qrgs.kind != WeightKind::BF16);

        const __nv_bfloat16 *n0 = nullptr, *n1 = nullptr, *n2 = nullptr;
        float *o0 = nullptr, *o1 = nullptr, *o2 = nullptr;
        __half *oh0 = nullptr, *oh1 = nullptr, *oh2 = nullptr;
        int n_out = 1;
        if (!distinct_inputs) {
            n0 = lw.attn_norm; o0 = attn_in_[0]; oh0 = use_half_dense ? attn_in_h_[0] : nullptr; n_out = 1;
        } else if (v.is_global && !use_prior) {
            n0 = lw.q_in_norm; n1 = lw.k_in_norm; n2 = lw.v_in_norm;
            o0 = attn_in_[0]; o1 = attn_in_[1]; o2 = attn_in_[2];
            oh0 = use_half_dense ? attn_in_h_[0] : nullptr;
            oh1 = use_half_dense ? attn_in_h_[1] : nullptr;
            oh2 = use_half_dense ? attn_in_h_[2] : nullptr;
            n_out = 3;
        } else if (v.is_global) {
            o0 = attn_in_[0]; o1 = attn_in_[1]; o2 = attn_in_[2]; n_out = 3;
        } else {
            o0 = mlp_in_; n_out = 1;
        }

        t_attn_in += timed([&] {
            if (prev_mlp_pending) {
                const LayerWeights& prev_lw = w_->layers[schedule_[vi - 1].physical_layer];
                launch_mega_mlp_post_next_depth_norm(
                    (B > 1) ? moe_slot_out_ : moe_down_, (B > 1),
                    prev_lw.mlp_sconv_w, mlp_ring_[vi - 1], residual_,
                    hidden_, pre_tap_out, nullptr, partial_dst, blocks_, block_count,
                    w_->attn_queries[l - 1], n0, n1, n2, o0, o1, o2, oh0, oh1, oh2,
                    n_out, B, d_pos_, stream_);
                prev_mlp_pending = false;
            } else {
                launch_fused_depth_norm(
                    hidden_, partial_dst, blocks_, block_count,
                    (l > 0) ? w_->attn_queries[l - 1] : nullptr,
                    n0, n1, n2, o0, o1, o2, oh0, oh1, oh2, n_out, B, stream_);
            }
        });

        if (use_prior) {
            int loc = l - LOOP_START;
            float* xs[3] = {v.is_global ? attn_in_[0] : mlp_in_, v.is_global ? attn_in_[1] : mlp_in_, v.is_global ? attn_in_[2] : mlp_in_};
            float* outs[3] = {attn_in_[0], attn_in_[1], attn_in_[2]};
            __half* outs_h[3] = {attn_in_h_[0], attn_in_h_[1], attn_in_h_[2]};
            const float* cp_q[3] = {
                w_->cross_pass_queries + (size_t)(2 * loc) * HIDDEN,
                w_->cross_pass_kv_queries + (size_t)(2 * loc) * HIDDEN,
                w_->cross_pass_kv_queries + (size_t)(2 * loc + 1) * HIDDEN
            };
            const ControllerWeights* cws[3] = {&lw.ctrl[0], &lw.ctrl[1], &lw.ctrl[2]};
            const __nv_bfloat16* pnorms[3] = {
                v.is_global ? lw.q_in_norm : lw.attn_norm,
                v.is_global ? lw.k_in_norm : lw.attn_norm,
                v.is_global ? lw.v_in_norm : lw.attn_norm
            };
            t_ctrl_attn += timed([&] {
                if (w_->mode == QuantMode::NVFP4) {
                    launch_controller_mega_nvfp4(
                        xs, saved_, prior_last_, cp_q, cws, pnorms, outs, use_half_dense ? outs_h : nullptr, 3, B, stream_);
                } else {
                    launch_controller_merge_batched(
                        xs, saved_, prior_last_, cp_q, cws, pnorms, outs, use_half_dense ? outs_h : nullptr, 3,
                        ctrl_msg_, ctrl_cat_, ctrl_d_, B, stream_);
                }
            });
        }

        const float* in_q = attn_in_[0];
        const float* in_k = distinct_inputs ? attn_in_[1] : attn_in_[0];
        const float* in_v = distinct_inputs ? attn_in_[2] : attn_in_[0];
        const __half* in_qh = attn_in_h_[0];
        const __half* in_kh = distinct_inputs ? attn_in_h_[1] : attn_in_h_[0];
        const __half* in_vh = distinct_inputs ? attn_in_h_[2] : attn_in_h_[0];

        t_attn_proj += timed([&] {
            if (lw.w_qrgs.kind == WeightKind::NVFP4) {
                launch_qkvrg_fused(lw, in_qh, in_kh, in_vh, qrgs_, k_proj_, v_proj_, B, stream_);
            } else if (lw.w_qrgs.kind == WeightKind::FP8) {
                launch_linear_h(lw.w_qrgs, in_qh, qrgs_, B, false, stream_);
                launch_linear_h(lw.w_k, in_kh, k_proj_, B, false, stream_);
                launch_linear_h(lw.w_v, in_vh, v_proj_, B, false, stream_);
            } else {
                launch_linear(lw.w_qrgs, in_q, qrgs_, B, false, stream_);
                launch_linear(lw.w_k, in_k, k_proj_, B, false, stream_);
                launch_linear(lw.w_v, in_v, v_proj_, B, false, stream_);
            }
        });

        t_prep += timed([&] {
            if (v.is_global) {
                launch_mega_global_sconv_prep_store(
                    qrgs_, k_proj_, v_proj_, in_k, in_v, lw.k_sconv_w, lw.v_sconv_w,
                    lw.k_sconv_proj, lw.v_sconv_proj, k_ring_[vi], v_ring_[vi],
                    lw.q_clip_scale, lw.rel_clip_scale,
                    k_cache_[vi], v_cache_[vi], k_scale_[vi], v_scale_[vi], kv_fmt_,
                    kv_cap_[vi], B, d_pos_, stream_);
            } else {
                launch_mega_local_sconv_prep_store(
                    qrgs_, k_proj_, v_proj_, lw.k_sconv_w, lw.v_sconv_w,
                    k_ring_[vi], v_ring_[vi], lw.q_clip_scale, lw.rel_clip_scale,
                    static_cast<__nv_bfloat16*>(k_cache_[vi]), static_cast<__nv_bfloat16*>(v_cache_[vi]),
                    kv_cap_[vi], B, d_pos_, stream_);
            }
        });

        t_rel_attn += timed([&] {
            launch_relative_attention_gated(
                qrgs_, lw.rel_logits_proj, k_cache_[vi], v_cache_[vi],
                k_scale_[vi], v_scale_[vi], v.is_global ? kv_fmt_ : KVFormat::BF16,
                split_buf_,
                use_half_dense ? nullptr : attn_gated_, use_half_dense ? attn_gated_h_ : nullptr,
                kv_cap_[vi], v.is_global ? 0 : SWA_WINDOW, B, d_pos_, stream_);
        });

        t_wo += timed([&] {
            if (use_half_dense) launch_linear_h(lw.w_o, attn_gated_h_, o_proj_, B, false, stream_);
            else                launch_linear(lw.w_o, attn_gated_, o_proj_, B, false, stream_);
        });

        bool is_boundary = (l % BLOCK_LAYERS == 0);
        t_attn_post += timed([&] {
            launch_mega_attn_post_mlp_in(
                o_proj_, lw.attn_sconv_w, attn_ring_[vi],
                hidden_, partial_, blocks_, block_count, is_boundary,
                residual_, w_->mlp_queries[l],
                use_prior ? nullptr : lw.mlp_norm,
                use_prior ? mlp_in_ : mlp_normed_,
                use_prior ? nullptr : mlp_normed_h_,
                B, d_pos_, stream_);
        });
        if (is_boundary) block_count++;

        if (use_prior) {
            int loc = l - LOOP_START;
            float* xs[1] = {mlp_in_};
            float* outs[1] = {mlp_normed_};
            __half* outs_h[1] = {mlp_normed_h_};
            const float* cp_q[3] = {w_->cross_pass_queries + (size_t)(2 * loc + 1) * HIDDEN, nullptr, nullptr};
            const ControllerWeights* cws[3] = {&lw.ctrl[3], nullptr, nullptr};
            const __nv_bfloat16* pnorms[3] = {lw.mlp_norm, nullptr, nullptr};
            t_ctrl_mlp += timed([&] {
                if (w_->mode == QuantMode::NVFP4) {
                    launch_controller_mega_nvfp4(
                        xs, saved_, prior_last_, cp_q, cws, pnorms, outs, outs_h, 1, B, stream_);
                } else {
                    launch_controller_merge_batched(
                        xs, saved_, prior_last_, cp_q, cws, pnorms, outs, outs_h, 1,
                        ctrl_msg_, ctrl_cat_, ctrl_d_, B, stream_);
                }
            });
        }

        t_router += timed([&] {
            if (w_->mode == QuantMode::NVFP4) {
                launch_route_topk_nvfp4(
                    mlp_normed_h_, lw.router_nvfp4, lw.router_bias, router_logits_, route_ids_, route_w_,
                    (B > 1) ? exp_groups_ : nullptr, nullptr, B, stream_);
            } else {
                launch_route_topk(
                    mlp_normed_, lw.router_w, lw.router_bias, router_logits_, route_ids_, route_w_,
                    (B > 1) ? exp_groups_ : nullptr, nullptr, B, stream_);
            }
        });

        t_moe_gu += timed([&] {
            launch_moe_gate_up_h(lw.w13, mlp_normed_h_, route_ids_, exp_groups_, moe_act_h_, B, stream_);
        });

        t_moe_dn += timed([&] {
            launch_moe_down_h(
                lw.w2, moe_act_h_, route_ids_, route_w_, exp_groups_,
                (B > 1) ? moe_slot_out_ : moe_down_, B, stream_, true);
        });

        if (vi == 17 || vi == 29) {
            t_mlp_post += timed([&] {
                launch_mega_mlp_post_res(
                    (B > 1) ? moe_slot_out_ : moe_down_, (B > 1),
                    lw.mlp_sconv_w, mlp_ring_[vi], residual_, hidden_, B, d_pos_, stream_);
            });
            prev_mlp_pending = false;
        } else {
            prev_mlp_pending = true;
        }
    }

    t_head = timed([&] {
        launch_mega_mlp_post_next_depth_norm(
            (B > 1) ? moe_slot_out_ : moe_down_, (B > 1),
            w_->layers[schedule_[NUM_VISITS - 1].physical_layer].mlp_sconv_w, mlp_ring_[NUM_VISITS - 1],
            residual_, hidden_, nullptr, nullptr, nullptr, blocks_, block_count, w_->output_query,
            w_->final_norm, nullptr, nullptr, mlp_normed_, nullptr, nullptr, nullptr, nullptr, nullptr,
            1, B, d_pos_, stream_);
        launch_lm_head_argmax(w_->unembed, mlp_normed_, d_next_tokens_, nullptr, lm_blk_val_, lm_blk_idx_, B, stream_);
    });

    if (num_drafts > 0 && w_->dspark.loaded) {
        t_ds_commit = timed([&] {
            launch_dspark_commit(w_->dspark, taps_, ds_k_cache_, ds_v_cache_, ds_k_scale_, ds_v_scale_,
                                 ds_kv_cap_, kv_fmt_, ds_f_ring_, ds_mix_, ds_kv_, B, d_pos_, stream_);
        });
    }

    double tot = t_embed + t_attn_in + t_ctrl_attn + t_attn_proj + t_sconv_kv + t_prep + t_rel_attn
               + t_wo + t_attn_post + t_mlp_in + t_ctrl_mlp + t_router + t_group
               + t_moe_gu + t_moe_dn + t_mlp_post + t_head + t_ds_commit + t_ds_draft;

    printf("\n=== PROFILE BREAKDOWN (B=%d, mtp=%d, pos=%d) ===\n", B, num_drafts, cur_pos_);
    auto p = [&](const char* name, double us) {
        printf("  %-20s: %7.1f us (%5.1f%%)\n", name, us, 100.0 * us / tot);
    };
    p("embed", t_embed);
    p("attn_in_depth_norm", t_attn_in);
    p("ctrl_attn (72x)", t_ctrl_attn);
    p("attn_proj (qkvrg)", t_attn_proj);
    p("sconv_kv", t_sconv_kv);
    p("attn_prep_store", t_prep);
    p("rel_attn_gated", t_rel_attn);
    p("attn_wo", t_wo);
    p("attn_post_sconv", t_attn_post);
    p("mlp_in_depth_norm", t_mlp_in);
    p("ctrl_mlp (24x)", t_ctrl_mlp);
    p("router_topk", t_router);
    p("expert_grouping", t_group);
    p("moe_gate_up", t_moe_gu);
    p("moe_down", t_moe_dn);
    p("mlp_post_sconv", t_mlp_post);
    p("final_head_argmax", t_head);
    p("dspark_commit", t_ds_commit);
    p("dspark_draft", t_ds_draft);
    printf("  %-20s: %7.1f us (%.2f ms)\n", "SUM_EAGER_STAGES", tot, tot / 1000.0);

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
}

double Engine::bench_step_at_pos(int sim_pos, int num_drafts, int iters) {
    bool use_mtp = (num_drafts > 0 && w_->dspark.loaded);
    if (sim_pos > max_ctx_ - iters * MAX_VERIFY_B - 32) {
        sim_pos = std::max(0, max_ctx_ - iters * MAX_VERIFY_B - 32);
    }
    reset();
    // Zero-init global KV caches & scales so synthetic reads are strictly finite (zero NaN)
    for (int v = 0; v < NUM_VISITS; ++v) {
        if (!schedule_[v].is_global) continue;
        size_t dbytes = (kv_fmt_ == KVFormat::FP8) ? ((size_t)kv_cap_[v] * KV_DIM)
                      : (kv_fmt_ == KVFormat::Q4)  ? ((size_t)kv_cap_[v] * (KV_DIM / 2))
                                                   : ((size_t)kv_cap_[v] * KV_DIM * sizeof(__nv_bfloat16));
        CUDA_CHECK(cudaMemsetAsync(k_cache_[v], 0, dbytes, stream_));
        CUDA_CHECK(cudaMemsetAsync(v_cache_[v], 0, dbytes, stream_));
        if (k_scale_[v]) {
            size_t sbytes = (kv_fmt_ == KVFormat::FP8) ? ((size_t)kv_cap_[v] * NUM_KV_HEADS * sizeof(__half))
                                                       : ((size_t)kv_cap_[v] * KV_Q4_SCALES_PER_TOK * sizeof(__half));
            CUDA_CHECK(cudaMemsetAsync(k_scale_[v], 0, sbytes, stream_));
            CUDA_CHECK(cudaMemsetAsync(v_scale_[v], 0, sbytes, stream_));
        }
    }
    if (use_mtp) {
        int l = 1;
        size_t dbytes = (kv_fmt_ == KVFormat::FP8) ? ((size_t)ds_kv_cap_[l] * KV_DIM)
                      : (kv_fmt_ == KVFormat::Q4)  ? ((size_t)ds_kv_cap_[l] * (KV_DIM / 2))
                                                   : ((size_t)ds_kv_cap_[l] * KV_DIM * sizeof(__nv_bfloat16));
        CUDA_CHECK(cudaMemsetAsync(ds_k_cache_[l], 0, dbytes, stream_));
        CUDA_CHECK(cudaMemsetAsync(ds_v_cache_[l], 0, dbytes, stream_));
        if (ds_k_scale_[l]) {
            size_t sbytes = (kv_fmt_ == KVFormat::FP8) ? ((size_t)ds_kv_cap_[l] * NUM_KV_HEADS * sizeof(__half))
                                                       : ((size_t)ds_kv_cap_[l] * KV_Q4_SCALES_PER_TOK * sizeof(__half));
            CUDA_CHECK(cudaMemsetAsync(ds_k_scale_[l], 0, sbytes, stream_));
            CUDA_CHECK(cudaMemsetAsync(ds_v_scale_[l], 0, sbytes, stream_));
        }
    }
    int32_t seed_tok = 150000;
    CUDA_CHECK(cudaMemcpyAsync(d_tokens_, &seed_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_anchor_, &seed_tok, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_pos_, &sim_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));

    if (!use_mtp) ensure_b1_graph();
    else          ensure_spec_graph(num_drafts);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Warmup 4 steps
    for (int i = 0; i < 4; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(d_pos_, &sim_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaGraphLaunch(use_mtp ? spec_graph_exec_[num_drafts] : b1_graph_exec_, stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Timed `iters` steps back-to-back
    CUDA_CHECK(cudaMemcpyAsync(d_pos_, &sim_pos, sizeof(int32_t), cudaMemcpyHostToDevice, stream_));
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0, stream_));
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaGraphLaunch(use_mtp ? spec_graph_exec_[num_drafts] : b1_graph_exec_, stream_));
    }
    CUDA_CHECK(cudaEventRecord(ev1, stream_));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return double(ms) / double(iters);
}

} // namespace squid
