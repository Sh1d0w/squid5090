#include "kernels.hpp"
#include "kernels_common.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace squid {

namespace {

struct Ptr3 {
    const float* p[3];
};
struct MutPtr3 {
    float* p[3];
};
struct HalfPtr3 {
    __half* p[3];
};
struct BF16Ptr3 {
    const __nv_bfloat16* p[3];
};
struct CtrlPtr3 {
    const __nv_bfloat16* down[3];
    const __nv_bfloat16* forget[3];
    const __nv_bfloat16* gain[3];
};

// Embed lookup + RMSNorm(embed_norm)
__global__ void embed_norm_kernel(
    const __nv_bfloat16* __restrict__ embed,
    const __nv_bfloat16* __restrict__ norm_w,
    const int32_t* __restrict__ tokens,
    float* __restrict__ out) {
    int b = blockIdx.x;
    int tid = threadIdx.x; // 256 threads
    int tok = tokens[b];
    const __nv_bfloat16* row = embed + (size_t)tok * HIDDEN;
    float sum_sq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 256) {
        float v = __bfloat162float(row[d]);
        sum_sq = fmaf(v, v, sum_sq);
    }
    sum_sq = warp_sum(sum_sq);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = sum_sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float s = inv_rms;
    float* o = out + (size_t)b * HIDDEN;
    for (int d = tid; d < HIDDEN; d += 256) {
        float v = __bfloat162float(row[d]) * s;
        v = round_bf16(v) * __bfloat162float(norm_w[d]);
        o[d] = round_bf16(v);
    }
}

// Generic RMSNorm per group
__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ out,
    int width, int group_size, float eps) {
    int groups_per_row = width / group_size;
    int b = blockIdx.x / groups_per_row;
    int g = blockIdx.x % groups_per_row;
    int tid = threadIdx.x;
    const float* xi = x + (size_t)b * width + g * group_size;
    float* xo = out + (size_t)b * width + g * group_size;

    float sum_sq = 0.0f;
    for (int d = tid; d < group_size; d += blockDim.x) {
        float v = xi[d];
        sum_sq = fmaf(v, v, sum_sq);
    }
    sum_sq = warp_sum(sum_sq);
    __shared__ float sm[8];
    int nwarps = blockDim.x >> 5;
    if ((tid & 31) == 0) sm[tid >> 5] = sum_sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < nwarps; ++i) tot += sm[i];
        inv_rms = rsqrtf(tot / float(group_size) + eps);
    }
    __syncthreads();
    float s = inv_rms;
    for (int d = tid; d < group_size; d += blockDim.x) {
        float v = xi[d] * s;
        if (weight) v *= __bfloat162float(weight[d]);
        xo[d] = v;
    }
}

// Depth attention over [blocks[b, 0..n_blocks-1], partial[b]] with F32 query
__global__ void depth_attn_kernel(
    const float* __restrict__ blocks,
    int max_blocks,
    int n_blocks,
    const float* __restrict__ partial,
    const float* __restrict__ query,
    float* __restrict__ out) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int n_cand = n_blocks + 1;

    __shared__ float partials[8][8];
    __shared__ float weights[8];

    const float* blk_b = blocks + (size_t)b * max_blocks * HIDDEN;
    const float* par_b = partial + (size_t)b * HIDDEN;

    for (int c = 0; c < n_cand; ++c) {
        const float* src = (c < n_blocks) ? (blk_b + (size_t)c * HIDDEN) : par_b;
        float acc = 0.0f;
        for (int d = tid; d < HIDDEN; d += 256) acc = fmaf(query[d], src[d], acc);
        acc = warp_sum(acc);
        if (lane == 0) partials[warp][c] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float logits[8], mx = -1e30f;
        for (int c = 0; c < n_cand; ++c) {
            float dot = 0.0f;
            for (int w = 0; w < 8; ++w) dot += partials[w][c];
            logits[c] = dot * INV_SQRT_HIDDEN;
            mx = fmaxf(mx, logits[c]);
        }
        float den = 0.0f;
        for (int c = 0; c < n_cand; ++c) { weights[c] = expf(logits[c] - mx); den += weights[c]; }
        float inv_den = 1.0f / den;
        for (int c = 0; c < n_cand; ++c) weights[c] *= inv_den;
    }
    __syncthreads();

    float* o = out + (size_t)b * HIDDEN;
    for (int d = tid; d < HIDDEN; d += 256) {
        float val = 0.0f;
        for (int c = 0; c < n_cand; ++c) {
            const float* src = (c < n_blocks) ? (blk_b + (size_t)c * HIDDEN) : par_b;
            val = fmaf(weights[c], src[d], val);
        }
        o[d] = val;
    }
}

// FUSED: partial = rmsnorm(hidden, null) + depth_attn(blocks, partial, queries[s]) + optional rmsnorm(norms[s])!
// Grid: dim3(B, n_out), Block: 256 threads.
__global__ void __launch_bounds__(256)
fused_depth_norm_kernel(
    const float* __restrict__ hidden,
    float* __restrict__ partial_out,
    const float* __restrict__ blocks,
    int n_blocks,
    const float* __restrict__ queries,
    BF16Ptr3 norms,
    MutPtr3 outs,
    HalfPtr3 outs_h) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    __shared__ float s_partial[HIDDEN];
    __shared__ float sm[8];

    const float* hb = hidden + (size_t)b * HIDDEN;
    float sum_sq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 256) {
        float v = hb[d];
        s_partial[d] = v;
        sum_sq = fmaf(v, v, sum_sq);
    }
    sum_sq = warp_sum(sum_sq);
    if (lane == 0) sm[warp] = sum_sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    float ir = inv_rms;
    float* po = (s == 0 && partial_out) ? (partial_out + (size_t)b * HIDDEN) : nullptr;
    for (int d = tid; d < HIDDEN; d += 256) {
        float pv = s_partial[d] * ir;
        s_partial[d] = pv;
        if (po) po[d] = pv;
    }
    __syncthreads();

    __shared__ float s_val[HIDDEN];
    __half* oh = outs_h.p[s] ? (outs_h.p[s] + (size_t)b * HIDDEN) : nullptr;
    if (queries) {
        const float* q = queries + (size_t)s * HIDDEN;
        const float* blk_b = blocks + (size_t)b * 4 * HIDDEN;
        int n_cand = n_blocks + 1;
        __shared__ float partials[8][8];
        __shared__ float weights[8];

        for (int c = 0; c < n_cand; ++c) {
            const float* src = (c < n_blocks) ? (blk_b + (size_t)c * HIDDEN) : s_partial;
            float acc = 0.0f;
            for (int d = tid; d < HIDDEN; d += 256) acc = fmaf(q[d], src[d], acc);
            acc = warp_sum(acc);
            if (lane == 0) partials[warp][c] = acc;
        }
        __syncthreads();
        if (tid == 0) {
            float logits[8], mx = -1e30f;
            for (int c = 0; c < n_cand; ++c) {
                float dot = 0.0f;
                for (int w = 0; w < 8; ++w) dot += partials[w][c];
                logits[c] = dot * INV_SQRT_HIDDEN;
                mx = fmaxf(mx, logits[c]);
            }
            float den = 0.0f;
            for (int c = 0; c < n_cand; ++c) { weights[c] = expf(logits[c] - mx); den += weights[c]; }
            float inv_d = 1.0f / den;
            for (int c = 0; c < n_cand; ++c) weights[c] *= inv_d;
        }
        __syncthreads();

        float n_sq = 0.0f;
        for (int d = tid; d < HIDDEN; d += 256) {
            float v = 0.0f;
            for (int c = 0; c < n_cand; ++c) {
                const float* src = (c < n_blocks) ? (blk_b + (size_t)c * HIDDEN) : s_partial;
                v = fmaf(weights[c], src[d], v);
            }
            s_val[d] = v;
            n_sq = fmaf(v, v, n_sq);
        }
        const __nv_bfloat16* nw = norms.p[s];
        float* ob = outs.p[s] + (size_t)b * HIDDEN;
        if (nw) {
            n_sq = warp_sum(n_sq);
            if (lane == 0) sm[warp] = n_sq;
            __syncthreads();
            if (tid == 0) {
                float tot = 0.0f;
                for (int i = 0; i < 8; ++i) tot += sm[i];
                inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
            }
            __syncthreads();
            float s2 = inv_rms;
            for (int d = tid; d < HIDDEN; d += 256) {
                float v = s_val[d] * s2 * __bfloat162float(nw[d]);
                ob[d] = v;
                if (oh) oh[d] = __float2half(v);
            }
        } else {
            for (int d = tid; d < HIDDEN; d += 256) {
                float v = s_val[d];
                ob[d] = v;
                if (oh) oh[d] = __float2half(v);
            }
        }
    } else {
        const __nv_bfloat16* nw = norms.p[s];
        float* ob = outs.p[s] + (size_t)b * HIDDEN;
        for (int d = tid; d < HIDDEN; d += 256) {
            float v = s_partial[d];
            if (nw) v = round_bf16(v) * __bfloat162float(nw[d]);
            ob[d] = v;
            if (oh) oh[d] = __float2half(v);
        }
    }
}

// Legacy batched BF16 controller kernels superseded by 1-launch 16-CTA cluster controller_cluster_bf16_kernel.

// Batched RMSNorm for n_streams (1 or 3) in ONE launch (+ optional half copy!)
__global__ void __launch_bounds__(256)
batched_rmsnorm_kernel(MutPtr3 xs, HalfPtr3 xs_h, BF16Ptr3 norms) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    float* xb = xs.p[s] + (size_t)b * HIDDEN;
    __half* xhb = xs_h.p[s] ? (xs_h.p[s] + (size_t)b * HIDDEN) : nullptr;
    const __nv_bfloat16* nw = norms.p[s];
    float sq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 256) { float v = xb[d]; sq = fmaf(v, v, sq); }
    sq = warp_sum(sq);
    __shared__ float sm[8];
    if ((tid & 31) == 0) sm[tid >> 5] = sq;
    __syncthreads();
    __shared__ float inv_rms;
    if (tid == 0) {
        float tot = 0.0f;
        for (int i = 0; i < 8; ++i) tot += sm[i];
        inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float ir = inv_rms;
    for (int d = tid; d < HIDDEN; d += 256) {
        float v = xb[d] * ir * __bfloat162float(nw[d]);
        xb[d] = v;
        if (xhb) xhb[d] = __float2half(v);
    }
}

// Static Short Conv (16-slot ring buffer, safe under speculative verify B<=8!)
__global__ void static_sconv_kernel(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ ring,
    float* __restrict__ out,
    int B, int width, const int32_t* __restrict__ d_pos) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;

    int pos0 = *d_pos;
    float wc0 = __bfloat162float(w[c * 4 + 0]);
    float wc1 = __bfloat162float(w[c * 4 + 1]);
    float wc2 = __bfloat162float(w[c * 4 + 2]);
    float wc3 = __bfloat162float(w[c * 4 + 3]);

    float h0 = (pos0 >= 3) ? ring[((pos0 - 3) & 15) * width + c] : 0.0f;
    float h1 = (pos0 >= 2) ? ring[((pos0 - 2) & 15) * width + c] : 0.0f;
    float h2 = (pos0 >= 1) ? ring[((pos0 - 1) & 15) * width + c] : 0.0f;

    for (int b = 0; b < B; ++b) {
        float cur = x[(size_t)b * width + c];
        int p = pos0 + b;
        ring[(p & 15) * width + c] = cur;
        float acc = wc0 * h0 + wc1 * h1 + wc2 * h2 + wc3 * cur;
        out[(size_t)b * width + c] = (acc + cur) * INV_SQRT_5;
        h0 = h1; h1 = h2; h2 = cur;
    }
}

// Dynamic Short Conv (global layer K & V, 16-slot ring)
__global__ void dynamic_sconv_kernel(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ static_w,
    const float* __restrict__ dyn_taps,
    float* __restrict__ ring,
    float* __restrict__ out,
    int B, const int32_t* __restrict__ d_pos) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= KV_DIM) return;
    int head = c / HEAD_DIM;
    int pos0 = *d_pos;

    float sw0 = __bfloat162float(static_w[0 * KV_DIM + c]);
    float sw1 = __bfloat162float(static_w[1 * KV_DIM + c]);
    float sw2 = __bfloat162float(static_w[2 * KV_DIM + c]);
    float sw3 = __bfloat162float(static_w[3 * KV_DIM + c]);

    float h1 = (pos0 >= 1) ? ring[((pos0 - 1) & 15) * KV_DIM + c] : 0.0f;
    float h2 = (pos0 >= 2) ? ring[((pos0 - 2) & 15) * KV_DIM + c] : 0.0f;
    float h3 = (pos0 >= 3) ? ring[((pos0 - 3) & 15) * KV_DIM + c] : 0.0f;

    for (int b = 0; b < B; ++b) {
        float cur = x[(size_t)b * KV_DIM + c];
        int p = pos0 + b;
        ring[(p & 15) * KV_DIM + c] = cur;
        const float* dt = dyn_taps + b * 32 + head * 4;
        float acc = cur * (sw0 + dt[0])
                  + h1  * (sw1 + dt[1])
                  + h2  * (sw2 + dt[2])
                  + h3  * (sw3 + dt[3]);
        out[(size_t)b * KV_DIM + c] = (acc + cur) * INV_SQRT_9;
        h3 = h2; h2 = h1; h1 = cur;
    }
}

// Fused Q/R/K/V prep + KV cache store
__global__ void attn_prep_store_kernel(
    float* __restrict__ qrgs,
    const float* __restrict__ ck,
    const float* __restrict__ cv,
    const __nv_bfloat16* __restrict__ q_clip_scale,
    const __nv_bfloat16* __restrict__ rel_clip_scale,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int kv_cap,
    bool is_global,
    const int32_t* __restrict__ d_pos) {
    int b = blockIdx.x;
    int kind_id = blockIdx.y;
    int tid = threadIdx.x;
    int pos = (*d_pos) + b;
    float tau = 1.0f;
    if (is_global) {
        float ratio = fmaxf(float(pos + 1) / float(LOG_SCALE_FLOOR), 1.0f);
        tau = 1.0f + LOG_SCALE_ALPHA * logf(ratio);
    }

    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;

    if (kind_id < 24) {
        int h = kind_id;
        float* qp = qrgs + (size_t)b * QRGS_STRIDE + h * HEAD_DIM;
        float val = qp[tid] * __bfloat162float(q_clip_scale[h * HEAD_DIM + tid]);
        float sq = warp_sum(val * val);
        __shared__ float sm[4];
        if ((tid & 31) == 0) sm[tid >> 5] = sq;
        __syncthreads();
        __shared__ float factor;
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f) * tau;
        __syncthreads();
        qp[tid] = val * factor;
    } else if (kind_id < 48) {
        if (tid >= D_REL) return;
        int h = kind_id - 24;
        float* rp = qrgs + (size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL;
        float val = rp[tid] * __bfloat162float(rel_clip_scale[h * D_REL + tid]);
        float sq = val * val;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) sq += __shfl_xor_sync(0xFFFFu, sq, off);
        float factor = fminf(REL_CLIP / (sqrtf(sq) + 1e-6f), 1.0f) * tau;
        rp[tid] = val * factor;
    } else if (kind_id < 56) {
        int h = kind_id - 48;
        const float* kp = ck + (size_t)b * KV_DIM + h * HEAD_DIM;
        float val = kp[tid];
        float sq = warp_sum(val * val);
        __shared__ float sm[4];
        if ((tid & 31) == 0) sm[tid >> 5] = sq;
        __syncthreads();
        __shared__ float factor;
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f);
        __syncthreads();
        int slot = pos % kv_cap;
        k_cache[(size_t)slot * KV_DIM + h * HEAD_DIM + tid] = __float2bfloat16(val * factor);
    } else {
        int h = kind_id - 56;
        const float* vp = cv + (size_t)b * KV_DIM + h * HEAD_DIM;
        float val = vp[tid];
        float sq = warp_sum(val * val);
        __shared__ float sm[4];
        if ((tid & 31) == 0) sm[tid >> 5] = sq;
        __syncthreads();
        __shared__ float inv_rms;
        if (tid == 0) inv_rms = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
        __syncthreads();
        int slot = pos % kv_cap;
        v_cache[(size_t)slot * KV_DIM + h * HEAD_DIM + tid] = __float2bfloat16(val * inv_rms);
    }
}

// ============================================================================
// LEVER 7A: 256-CTA Blackwell Tensor Core (m16n8k16) Split-K Bulk FlashDecoding!
// Loads EVERY K and V token from global KV cache ONCE across all 3*B queries,
// computing QK^T and PV on FP16 Tensor Cores (~350 TFLOPS) -> 100% HBM-BW bound up to 1M!
// ============================================================================
template <int KV_FMT, int B_CONST>
__global__ void __launch_bounds__((GQA_RATIO * B_CONST > 16) ? 128 : 64)
global_attn_bulk_splitk_kernel(
    const float* __restrict__ qrgs,     // [B, 6552]
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale,
    int kv_cap,
    const int32_t* __restrict__ d_pos,
    float* __restrict__ split_buf) {
    pdl_wait();
    pdl_trigger();
    int pos0 = *d_pos;
    int tail_start = pos0 - REL_EXTENT + 1;
    if (tail_start <= 0) return;

    int split = blockIdx.y;
    int chunk = (tail_start + KV_SPLIT_K - 1) / KV_SPLIT_K;
    int t_start = split * chunk;
    int t_end = min(tail_start, t_start + chunk);

    constexpr int Q_CNT = GQA_RATIO * B_CONST;
    constexpr int THREADS = (Q_CNT > 16) ? 128 : 64;
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    run_tc_bulk_splitk<KV_FMT, Q_CNT, QRGS_STRIDE, THREADS>(
        qrgs, k_cache, v_cache, k_scale, v_scale, kv_cap,
        blockIdx.x, split, t_start, t_end, split_buf);
}

// ============================================================================
// LEVER 7B: 136-CTA Blackwell Tensor Core (m16n8k16) Local SWA / Global Tail Attention!
// Handles <= 1024+B tokens with EXACT causal + window mask + inline relative bias!
// Grid = dim3(NUM_KV_HEADS=8, KV_LOCAL_SPLITS=17) -> 1 single 64-token tile per CTA!
// Replaces slow scalar 48-visit attention (3,223 us -> ~450 us at pos >= 1024!).
// ============================================================================
template <int KV_FMT, int B_CONST>
__global__ void __launch_bounds__((GQA_RATIO * B_CONST > 16) ? 256 : 128)
local_tail_attn_tc_kernel(
    const float* __restrict__ qrgs,
    const __nv_bfloat16* __restrict__ rel_proj,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale,
    int kv_cap,
    int window,
    const int32_t* __restrict__ d_pos,
    float* __restrict__ split_buf) {
    constexpr int Q_CNT = GQA_RATIO * B_CONST;
    constexpr int NM = (Q_CNT > 16) ? 2 : 1;
    constexpr int THREADS = (NM == 2) ? 256 : 128;
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;

    pdl_wait();
    pdl_trigger();
    int kvh = blockIdx.x;
    int lsplit = blockIdx.y; // 0..16
    int split_slot = (window == 0) ? (KV_SPLIT_K + lsplit) : lsplit;
    int tid = threadIdx.x;

    int pos0 = *d_pos;
    int win_start = (window > 0) ? max(0, pos0 + 1 - window) : max(0, pos0 - REL_EXTENT + 1);
    int win_end = pos0 + B_CONST;
    int t0 = win_start + lsplit * 64;
    int tile_len = min(64, max(0, win_end - t0));

    float* s_m_out = split_buf + SPLIT_M_OFF;
    float* s_d_out = split_buf + SPLIT_D_OFF;
    float* s_o_out = split_buf + SPLIT_O_OFF;

    if (tile_len <= 0) {
        if (tid < Q_CNT) {
            int gqa = tid % GQA_RATIO;
            int b = tid / GQA_RATIO;
            int h = kvh * GQA_RATIO + gqa;
            int idx = (h * 8 + b) * KV_TOTAL_SPLITS + split_slot;
            s_m_out[idx] = -1e30f;
            s_d_out[idx] = 0.0f;
        }
        return;
    }

    int warp = tid >> 5;
    int lane = tid & 31;
    int grp = lane >> 2;
    int tid4 = lane & 3;
    int slice = warp & 3;
    int m_tile = warp >> 2;

    // 20.7 KB SMEM total -> 4 blocks/SM on RTX 5090!
    __shared__ __half s_r[32][D_REL];      // 1,024 B
    __shared__ __half s_rp[72][D_REL];     // 2,304 B
    __shared__ union {
        __half s_kv[64][HEAD_DIM + 8];     // 17,408 B
        struct {
            float m[8][4];
            float d[8][4];
            float o[4][8][HEAD_DIM];       // 16,384 B
        } red;
    } smem;

    // 1. Load R[0..Q_CNT-1][0..15] into SMEM
    for (int i = tid; i < Q_CNT * D_REL; i += THREADS) {
        int qi = i >> 4;
        int k = i & 15;
        int h = kvh * GQA_RATIO + (qi % GQA_RATIO);
        int b = qi / GQA_RATIO;
        s_r[qi][k] = __float2half(qrgs[(size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL + k]);
    }

    // 2. Load relevant rel_proj columns [d_min .. d_max] transposed into s_rp[0..d_span-1][0..15]
    int d_min = max(0, pos0 - (t0 + tile_len - 1));
    int d_max = min(REL_EXTENT - 1, pos0 + B_CONST - 1 - t0);
    int d_span = (d_max >= d_min) ? (d_max - d_min + 1) : 0;
    for (int i = tid; i < d_span * D_REL; i += THREADS) {
        int di = i >> 4;
        int k = i & 15;
        int dist = d_min + di;
        s_rp[di][k] = __float2half(__bfloat162float(__ldg(&rel_proj[k * REL_EXTENT + dist])));
    }

    // 3. Load K tile into SMEM
    #pragma unroll
    for (int idx = tid; idx < 1024; idx += THREADS) {
        int t_loc = idx >> 4;
        int sub = idx & 15;
        if (t_loc < tile_len) {
            int slot = (t0 + t_loc) % kv_cap;
            *reinterpret_cast<uint4*>(&smem.s_kv[t_loc][sub * 8]) =
                load_kv_8elems_h2x4<KV_FMT>(k_cache, k_scale, slot, kvh, sub);
        } else {
            *reinterpret_cast<uint4*>(&smem.s_kv[t_loc][sub * 8]) = make_uint4(0, 0, 0, 0);
        }
    }

    // 4. Load Q[16, 128] * INV_SQRT_HEAD_DIM into registers
    int q0 = m_tile * 16 + grp;
    int q1 = m_tile * 16 + grp + 8;
    const float* qp0 = nullptr;
    const float* qp1 = nullptr;
    int pos_q0 = pos0, pos_q1 = pos0;
    if (q0 < Q_CNT) {
        int h0 = kvh * GQA_RATIO + (q0 % GQA_RATIO);
        int b0 = q0 / GQA_RATIO;
        pos_q0 = pos0 + b0;
        qp0 = qrgs + (size_t)b0 * QRGS_STRIDE + h0 * HEAD_DIM;
    }
    if (q1 < Q_CNT) {
        int h1 = kvh * GQA_RATIO + (q1 % GQA_RATIO);
        int b1 = q1 / GQA_RATIO;
        pos_q1 = pos0 + b1;
        qp1 = qrgs + (size_t)b1 * QRGS_STRIDE + h1 * HEAD_DIM;
    }

    uint32_t q_reg[8][4];
    #pragma unroll
    for (int kt = 0; kt < 8; ++kt) {
        int k_off = kt * 16;
        if (qp0) {
            __half2 h0 = __floats2half2_rn(qp0[k_off + tid4 * 2] * INV_SQRT_HEAD_DIM,
                                           qp0[k_off + tid4 * 2 + 1] * INV_SQRT_HEAD_DIM);
            __half2 h2 = __floats2half2_rn(qp0[k_off + tid4 * 2 + 8] * INV_SQRT_HEAD_DIM,
                                           qp0[k_off + tid4 * 2 + 9] * INV_SQRT_HEAD_DIM);
            q_reg[kt][0] = *reinterpret_cast<uint32_t*>(&h0);
            q_reg[kt][2] = *reinterpret_cast<uint32_t*>(&h2);
        } else {
            q_reg[kt][0] = 0u;
            q_reg[kt][2] = 0u;
        }
        if (qp1) {
            __half2 h1 = __floats2half2_rn(qp1[k_off + tid4 * 2] * INV_SQRT_HEAD_DIM,
                                           qp1[k_off + tid4 * 2 + 1] * INV_SQRT_HEAD_DIM);
            __half2 h3 = __floats2half2_rn(qp1[k_off + tid4 * 2 + 8] * INV_SQRT_HEAD_DIM,
                                           qp1[k_off + tid4 * 2 + 9] * INV_SQRT_HEAD_DIM);
            q_reg[kt][1] = *reinterpret_cast<uint32_t*>(&h1);
            q_reg[kt][3] = *reinterpret_cast<uint32_t*>(&h3);
        } else {
            q_reg[kt][1] = 0u;
            q_reg[kt][3] = 0u;
        }
    }
    __syncthreads();

    int c0 = tid4 * 2;
    int c1 = tid4 * 2 + 1;
    int c4 = tid4 * 2 + 8;
    int c5 = tid4 * 2 + 9;
    int t_base = slice * 16;
    int w_len = min(16, max(0, tile_len - t_base));

    float m0 = -1e30f, m1 = -1e30f;
    float d0 = 0.0f, d1 = 0.0f;
    float o[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
    }

    uint32_t pa0 = 0u, pa1 = 0u, pa2 = 0u, pa3 = 0u;
    if (w_len > 0) {
        // 5. QK^T via 8x ldmatrix.x4 + 16x Tensor Core m16n8k16 MMAs
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
        int k_ld_row = t_base + ((lane & 16) >> 1) + (lane & 7);
        int k_ld_col = (lane & 8);
        #pragma unroll
        for (int kt = 0; kt < 8; ++kt) {
            uint32_t kb0, kb1, kb2, kb3;
            ldmatrix_x4(kb0, kb1, kb2, kb3, &smem.s_kv[k_ld_row][kt * 16 + k_ld_col]);
            mma_m16n8k16_f32_f16(s0, s1, s2, s3,
                                 q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                 kb0, kb1, s0, s1, s2, s3);
            mma_m16n8k16_f32_f16(s4, s5, s6, s7,
                                 q_reg[kt][0], q_reg[kt][1], q_reg[kt][2], q_reg[kt][3],
                                 kb2, kb3, s4, s5, s6, s7);
        }

        // 6. Add exact relative bias + causal/window mask!
        auto apply_rel = [&](float& s, int col, int qi, int pos_q) {
            if (qi >= Q_CNT || col >= w_len) { s = -1e30f; return; }
            int t = t0 + t_base + col;
            if (t > pos_q || (window > 0 && t < pos_q + 1 - window)) { s = -1e30f; return; }
            int dist = pos_q - t;
            if (dist < REL_EXTENT) {
                const __half2* rh = reinterpret_cast<const __half2*>(&s_r[qi][0]);
                const __half2* ph = reinterpret_cast<const __half2*>(&s_rp[dist - d_min][0]);
                __half2 a0 = __hmul2(rh[0], ph[0]), a1 = __hmul2(rh[1], ph[1]);
                a0 = __hfma2(rh[2], ph[2], a0); a1 = __hfma2(rh[3], ph[3], a1);
                a0 = __hfma2(rh[4], ph[4], a0); a1 = __hfma2(rh[5], ph[5], a1);
                a0 = __hfma2(rh[6], ph[6], a0); a1 = __hfma2(rh[7], ph[7], a1);
                float2 f = __half22float2(__hadd2(a0, a1));
                s += (f.x + f.y);
            }
        };
        apply_rel(s0, c0, q0, pos_q0);
        apply_rel(s1, c1, q0, pos_q0);
        apply_rel(s4, c4, q0, pos_q0);
        apply_rel(s5, c5, q0, pos_q0);
        apply_rel(s2, c0, q1, pos_q1);
        apply_rel(s3, c1, q1, pos_q1);
        apply_rel(s6, c4, q1, pos_q1);
        apply_rel(s7, c5, q1, pos_q1);

        // 7. Single-tile softmax (NaN-safe for all-masked rows!)
        float rmax0 = fmaxf(fmaxf(s0, s1), fmaxf(s4, s5));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 1));
        rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xffffffffu, rmax0, 2));
        float rmax1 = fmaxf(fmaxf(s2, s3), fmaxf(s6, s7));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 1));
        rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xffffffffu, rmax1, 2));

        float ps0 = (s0 > -1e29f) ? __expf(s0 - rmax0) : 0.0f;
        float ps1 = (s1 > -1e29f) ? __expf(s1 - rmax0) : 0.0f;
        float ps4 = (s4 > -1e29f) ? __expf(s4 - rmax0) : 0.0f;
        float ps5 = (s5 > -1e29f) ? __expf(s5 - rmax0) : 0.0f;
        float ps2 = (s2 > -1e29f) ? __expf(s2 - rmax1) : 0.0f;
        float ps3 = (s3 > -1e29f) ? __expf(s3 - rmax1) : 0.0f;
        float ps6 = (s6 > -1e29f) ? __expf(s6 - rmax1) : 0.0f;
        float ps7 = (s7 > -1e29f) ? __expf(s7 - rmax1) : 0.0f;

        float rsum0 = ps0 + ps1 + ps4 + ps5;
        rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 1);
        rsum0 += __shfl_xor_sync(0xffffffffu, rsum0, 2);
        float rsum1 = ps2 + ps3 + ps6 + ps7;
        rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 1);
        rsum1 += __shfl_xor_sync(0xffffffffu, rsum1, 2);
        m0 = rmax0; d0 = rsum0;
        m1 = rmax1; d1 = rsum1;

        __half2 hp0 = __floats2half2_rn(ps0, ps1);
        __half2 hp1 = __floats2half2_rn(ps2, ps3);
        __half2 hp2 = __floats2half2_rn(ps4, ps5);
        __half2 hp3 = __floats2half2_rn(ps6, ps7);
        pa0 = *reinterpret_cast<uint32_t*>(&hp0);
        pa1 = *reinterpret_cast<uint32_t*>(&hp1);
        pa2 = *reinterpret_cast<uint32_t*>(&hp2);
        pa3 = *reinterpret_cast<uint32_t*>(&hp3);
    }
    __syncthreads();

    // 8. Load V tile into the SAME SMEM buffer!
    #pragma unroll
    for (int idx = tid; idx < 1024; idx += THREADS) {
        int t_loc = idx >> 4;
        int sub = idx & 15;
        if (t_loc < tile_len) {
            int slot = (t0 + t_loc) % kv_cap;
            *reinterpret_cast<uint4*>(&smem.s_kv[t_loc][sub * 8]) =
                load_kv_8elems_h2x4<KV_FMT>(v_cache, v_scale, slot, kvh, sub);
        } else {
            *reinterpret_cast<uint4*>(&smem.s_kv[t_loc][sub * 8]) = make_uint4(0, 0, 0, 0);
        }
    }
    __syncthreads();

    // 9. P @ V via 16x hardware ldmatrix.x2.trans + 16x Tensor Core m16n8k16 MMAs
    if (w_len > 0) {
        int v_ld_row = t_base + (lane & 15);
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            uint32_t vb0, vb1;
            ldmatrix_x2_trans(vb0, vb1, &smem.s_kv[v_ld_row][nt * 8]);
            mma_m16n8k16_f32_f16(o[nt][0], o[nt][1], o[nt][2], o[nt][3],
                                 pa0, pa1, pa2, pa3,
                                 vb0, vb1,
                                 o[nt][0], o[nt][1], o[nt][2], o[nt][3]);
        }
    }
    __syncthreads();

    // 10. Reduce 4 token-slices (8 queries at a time) into split_buf[split_slot]
    for (int mt = 0; mt < NM; ++mt) {
        for (int half_m = 0; half_m < 2; ++half_m) {
            if (mt * 16 + half_m * 8 >= Q_CNT) break;
            if (m_tile == mt) {
                if (tid4 == 0) {
                    smem.red.m[grp][slice] = (half_m == 0) ? m0 : m1;
                    smem.red.d[grp][slice] = (half_m == 0) ? d0 : d1;
                }
                #pragma unroll
                for (int nt = 0; nt < 16; ++nt) {
                    smem.red.o[slice][grp][nt * 8 + c0] = (half_m == 0) ? o[nt][0] : o[nt][2];
                    smem.red.o[slice][grp][nt * 8 + c1] = (half_m == 0) ? o[nt][1] : o[nt][3];
                }
            }
            __syncthreads();

            for (int i = tid; i < 8 * HEAD_DIM; i += THREADS) {
                int qi = i >> 7;
                int dim = i & 127;
                int q_glob = mt * 16 + half_m * 8 + qi;
                if (q_glob < Q_CNT) {
                    int gqa = q_glob % GQA_RATIO;
                    int b = q_glob / GQA_RATIO;
                    int h = kvh * GQA_RATIO + gqa;
                    int idx = (h * 8 + b) * KV_TOTAL_SPLITS + split_slot;
                    float gm = fmaxf(fmaxf(smem.red.m[qi][0], smem.red.m[qi][1]),
                                     fmaxf(smem.red.m[qi][2], smem.red.m[qi][3]));
                    float gd = 0.0f, go = 0.0f;
                    #pragma unroll
                    for (int s = 0; s < 4; ++s) {
                        if (smem.red.d[qi][s] > 0.0f) {
                            float sc = __expf(smem.red.m[qi][s] - gm);
                            gd += smem.red.d[qi][s] * sc;
                            go += smem.red.o[s][qi][dim] * sc;
                        }
                    }
                    if (dim == 0) { s_m_out[idx] = gm; s_d_out[idx] = gd; }
                    s_o_out[(size_t)idx * HEAD_DIM + dim] = go;
                }
            }
            __syncthreads();
        }
    }
}

// Fast 128-thread Merge + Attention Sink + Gated Output kernel for BOTH local & global visits!
__global__ void __launch_bounds__(128)
attn_merge_sink_gate_kernel(
    const float* __restrict__ qrgs,
    const float* __restrict__ split_buf,
    float* __restrict__ out_gated,
    __half* __restrict__ out_gated_h,
    int window,
    const int32_t* __restrict__ d_pos) {
    pdl_wait();
    pdl_trigger();
    int h = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x; // 0..127

    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    const float* qrgs_b = qrgs + (size_t)b * QRGS_STRIDE;
    const float* gate_p = qrgs_b + Q_DIM + R_DIM + h * HEAD_DIM;
    float sink_logit = qrgs_b[Q_DIM + R_DIM + Q_DIM + h];

    int base_hb = (h * 8 + b) * KV_TOTAL_SPLITS;
    const float* s_m_in = split_buf + SPLIT_M_OFF + base_hb;
    const float* s_d_in = split_buf + SPLIT_D_OFF + base_hb;
    const float* s_o_in = split_buf + SPLIT_O_OFF;

    int s_begin, s_end;
    if (window > 0) {
        s_begin = 0;
        s_end = KV_LOCAL_SPLITS;
    } else {
        int tail_start = (*d_pos) - REL_EXTENT + 1;
        s_begin = (tail_start > 0) ? 0 : KV_SPLIT_K;
        s_end = KV_TOTAL_SPLITS;
    }

    float g_m = sink_logit;
    for (int s = s_begin; s < s_end; ++s) {
        g_m = fmaxf(g_m, s_m_in[s]);
    }

    float g_d = __expf(sink_logit - g_m);
    float g_o = 0.0f;
    for (int s = s_begin; s < s_end; ++s) {
        float sd = s_d_in[s];
        if (sd > 0.0f) {
            float sc = __expf(s_m_in[s] - g_m);
            g_d += sd * sc;
            g_o += s_o_in[(size_t)(base_hb + s) * HEAD_DIM + tid] * sc;
        }
    }

    float val = (g_o / g_d) * fast_sigmoid(gate_p[tid]);
    size_t oidx = (size_t)b * Q_DIM + h * HEAD_DIM + tid;
    if (out_gated) out_gated[oidx] = val;
    if (out_gated_h) out_gated_h[oidx] = __float2half(val);
}

// Legacy inline kernel kept for fallback when split_buf == nullptr
template <int KV_FMT>
__global__ void __launch_bounds__(256)
relative_attention_gated_inline_kernel(
    const float* __restrict__ qrgs,
    const __nv_bfloat16* __restrict__ rel_proj,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const __half* __restrict__ k_scale,
    const __half* __restrict__ v_scale,
    const float* __restrict__ split_buf,
    float* __restrict__ out_gated,
    __half* __restrict__ out_gated_h,
    int kv_cap,
    int window,
    const int32_t* __restrict__ d_pos) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;

    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    const float* qrgs_b = qrgs + (size_t)b * QRGS_STRIDE;
    const float* qp = qrgs_b + h * HEAD_DIM;
    const float* rp = qrgs_b + Q_DIM + h * D_REL;
    const float* gate_p = qrgs_b + Q_DIM + R_DIM + h * HEAD_DIM;
    float sink_logit = qrgs_b[Q_DIM + R_DIM + Q_DIM + h];

    int pos0 = *d_pos;
    int pos = pos0 + b;
    int max_d = min(pos + 1, REL_EXTENT);

    // Cooperatively compute s_bias[0..max_d-1] directly into 4 KB SMEM!
    __shared__ float s_r[D_REL];
    __shared__ float s_bias[REL_EXTENT];
    if (tid < D_REL) s_r[tid] = rp[tid];
    __syncthreads();
    for (int d = tid; d < max_d; d += 256) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < D_REL; ++k) {
            sum = fmaf(s_r[k], __bfloat162float(__ldg(&rel_proj[k * REL_EXTENT + d])), sum);
        }
        s_bias[d] = sum;
    }
    __syncthreads();

    int tail_start = pos0 - REL_EXTENT + 1;
    bool has_bulk = (window == 0 && split_buf != nullptr && tail_start > 0);
    int first = (window > 0) ? ((pos + 1 > window) ? (pos + 1 - window) : 0)
                             : (has_bulk ? tail_start : 0);
    int kvh = h / GQA_RATIO;

    int d_off = lane * 4;
    float4 qv = *reinterpret_cast<const float4*>(qp + d_off);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float m_val = (warp == 0) ? sink_logit : -1e30f;
    float d_val = (warp == 0) ? 1.0f : 0.0f;

    for (int t = first + warp; t <= pos; t += 8) {
        int slot = t % kv_cap;
        float2 kf0, kf1, vf0, vf1;
        load_kv_4elems<KV_FMT>(k_cache, k_scale, slot, kvh, d_off, kf0, kf1);
        load_kv_4elems<KV_FMT>(v_cache, v_scale, slot, kvh, d_off, vf0, vf1);

        float dot = fmaf(qv.x, kf0.x, fmaf(qv.y, kf0.y, fmaf(qv.z, kf1.x, qv.w * kf1.y)));
        dot = warp_sum(dot);

        int dist = pos - t;
        float rel = (dist < REL_EXTENT) ? s_bias[dist] : 0.0f;
        float score = dot * INV_SQRT_HEAD_DIM + rel;

        float m_new = fmaxf(m_val, score);
        float alpha = __expf(m_val - m_new);
        float p_w = __expf(score - m_new);
        d_val = d_val * alpha + p_w;
        acc0 = fmaf(p_w, vf0.x, acc0 * alpha);
        acc1 = fmaf(p_w, vf0.y, acc1 * alpha);
        acc2 = fmaf(p_w, vf1.x, acc2 * alpha);
        acc3 = fmaf(p_w, vf1.y, acc3 * alpha);
        m_val = m_new;
    }

    __shared__ float sm_m[8], sm_d[8], sm_acc[8][HEAD_DIM];
    if (lane == 0) { sm_m[warp] = m_val; sm_d[warp] = d_val; }
    sm_acc[warp][d_off + 0] = acc0;
    sm_acc[warp][d_off + 1] = acc1;
    sm_acc[warp][d_off + 2] = acc2;
    sm_acc[warp][d_off + 3] = acc3;
    __syncthreads();

    if (tid < HEAD_DIM) {
        float g_m = sm_m[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) g_m = fmaxf(g_m, sm_m[w]);

        int base_hb = (h * MAX_VERIFY_B + b) * KV_TOTAL_SPLITS;
        const float* s_m_in = split_buf + SPLIT_M_OFF;
        const float* s_d_in = split_buf + SPLIT_D_OFF;
        const float* s_o_in = split_buf + SPLIT_O_OFF;
        if (has_bulk) {
            #pragma unroll
            for (int s = 0; s < KV_SPLIT_K; ++s) {
                g_m = fmaxf(g_m, s_m_in[base_hb + s]);
            }
        }

        float g_d = 0.0f, g_o = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (sm_d[w] > 0.0f) {
                float sc = __expf(sm_m[w] - g_m);
                g_d += sm_d[w] * sc;
                g_o += sm_acc[w][tid] * sc;
            }
        }
        if (has_bulk) {
            #pragma unroll
            for (int s = 0; s < KV_SPLIT_K; ++s) {
                float sd = s_d_in[base_hb + s];
                if (sd > 0.0f) {
                    float sc = __expf(s_m_in[base_hb + s] - g_m);
                    g_d += sd * sc;
                    g_o += s_o_in[(size_t)(base_hb + s) * HEAD_DIM + tid] * sc;
                }
            }
        }
        float val = (g_o / g_d) * fast_sigmoid(gate_p[tid]);
        size_t oidx = (size_t)b * Q_DIM + h * HEAD_DIM + tid;
        if (out_gated) out_gated[oidx] = val;
        if (out_gated_h) out_gated_h[oidx] = __float2half(val);
    }
}

// ============================================================================
// ZERO-SMEM, ZERO-SPILL High-Occupancy Parallel MoE Router: 64 blocks (2 warps/block -> 128 experts!)
// ============================================================================

template <int B_CONST>
__global__ void __launch_bounds__(64)
route_logits_parallel_kernel(
    const float* __restrict__ m,               // [B, HIDDEN]
    const __nv_bfloat16* __restrict__ router_w,// [128, HIDDEN]
    float* __restrict__ out_logits,            // [B, 128]
    float* __restrict__ moe_down_zero) {       // optional [B, HIDDEN] zeroed in parallel!
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int e = blockIdx.x * 2 + warp;
    if (e < NUM_EXPERTS) {
        const uint4* wr = reinterpret_cast<const uint4*>(router_w + (size_t)e * HIDDEN);
        float acc[B_CONST];
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

        for (int v = lane; v < (HIDDEN >> 3); v += 32) {
            uint4 w8 = __ldg(wr + v);
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
            float2 wf0 = __bfloat1622float2(wb[0]);
            float2 wf1 = __bfloat1622float2(wb[1]);
            float2 wf2 = __bfloat1622float2(wb[2]);
            float2 wf3 = __bfloat1622float2(wb[3]);
            int koff = v << 3;
            #pragma unroll
            for (int b = 0; b < B_CONST; ++b) {
                const float4* xf = reinterpret_cast<const float4*>(m + (size_t)b * HIDDEN + koff);
                float4 x0 = __ldg(xf), x1 = __ldg(xf + 1);
                acc[b] = fmaf(wf0.x, x0.x, acc[b]); acc[b] = fmaf(wf0.y, x0.y, acc[b]);
                acc[b] = fmaf(wf1.x, x0.z, acc[b]); acc[b] = fmaf(wf1.y, x0.w, acc[b]);
                acc[b] = fmaf(wf2.x, x1.x, acc[b]); acc[b] = fmaf(wf2.y, x1.y, acc[b]);
                acc[b] = fmaf(wf3.x, x1.z, acc[b]); acc[b] = fmaf(wf3.y, x1.w, acc[b]);
            }
        }
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            float val = warp_sum(acc[b]);
            if (lane == 0) out_logits[b * NUM_EXPERTS + e] = val;
        }
    }
    if constexpr (B_CONST > 1) {
        if (moe_down_zero) {
            float4* md4 = reinterpret_cast<float4*>(moe_down_zero);
            constexpr int N4 = (B_CONST * HIDDEN) >> 2;
            for (int i = blockIdx.x * 64 + threadIdx.x; i < N4; i += 64 * 64) {
                md4[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    }
}

// 64-block HALF-WARP NVFP4 MoE Router (64 SMs, 2 experts/block, 16 threads/expert = 9 EXACT iters!)
// Pre-issues all 9 NVFP4 weight & scale loads BEFORE pdl_wait_fenced(zero) to overlap predecessor tail!
template <int B_CONST>
__global__ void __launch_bounds__(32)
route_logits_nvfp4_parallel_kernel(
    const __half* __restrict__ m_h,            // [B, HIDDEN]
    const uint8_t* __restrict__ w_data,        // [128, HIDDEN/2]
    const uint8_t* __restrict__ w_scales,      // [128, HIDDEN/16]
    float gscale,
    float* __restrict__ out_logits,            // [B, 128]
    float* __restrict__ /*moe_down_zero*/,
    uint32_t zero) {
    int sub_e = threadIdx.x >> 4;  // 0 or 1
    int lane = threadIdx.x & 15;   // 0..15
    int e = blockIdx.x * 2 + sub_e;

    constexpr int W_STRIDE = HIDDEN >> 1;
    constexpr int S_STRIDE = HIDDEN >> 4; // 144 = 9 * 16 EXACT!
    const uint2* wr = reinterpret_cast<const uint2*>(w_data + (size_t)e * W_STRIDE);
    const uint8_t* sr = w_scales + (size_t)e * S_STRIDE;

    uint2 w8[9];
    uint8_t s8[9];
    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        int blk = lane + iter * 16;
        w8[iter] = ld_stream_u2(wr + blk);
        s8[iter] = ld_scale_u8(sr + blk);
    }
    pdl_wait_fenced(zero);
    pdl_trigger();

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        int blk = lane + iter * 16;
        float sc = cvt_e4m3_to_float(s8[iter]);
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8[iter].x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8[iter].y, wh[4], wh[5], wh[6], wh[7]);
        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            uint4 xv0 = ld_dep_u4(m_h + (size_t)b * HIDDEN + koff);
            uint4 xv1 = ld_dep_u4(m_h + (size_t)b * HIDDEN + koff + 8);
            const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
            __half2 p0 = __hmul2(wh[0], xb0[0]), p1 = __hmul2(wh[1], xb0[1]);
            p0 = __hfma2(wh[2], xb0[2], p0); p1 = __hfma2(wh[3], xb0[3], p1);
            p0 = __hfma2(wh[4], xb1[0], p0); p1 = __hfma2(wh[5], xb1[1], p1);
            p0 = __hfma2(wh[6], xb1[2], p0); p1 = __hfma2(wh[7], xb1[3], p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
        }
    }
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float val = half_warp_sum(acc[b]) * gscale;
        if (lane == 0) out_logits[b * NUM_EXPERTS + e] = val;
    }
}

// FUSED Top-8 Selection + SMEM-staged ExpertGroups Deduplication in ONE single launch!
// Grid: 1 block, Block: B_CONST * 32 threads (1 warp per token b!).
template <int B_CONST>
__global__ void __launch_bounds__(B_CONST * 32)
route_select_and_group_kernel(
    const float* __restrict__ logits,      // [B, 128]
    const float* __restrict__ router_bias, // [128]
    int32_t* __restrict__ out_ids,         // [B, TOP_K]
    float* __restrict__ out_weights,       // [B, TOP_K]
    ExpertGroups* __restrict__ g,
    uint32_t zero) {
    int b = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    float rb[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) rb[i] = __ldg(router_bias + lane + 32 * i);
    pdl_wait_fenced(zero);
    pdl_trigger();

    float sc[4], bi[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int e = lane + 32 * i;
        float raw = fast_sigmoid(ld_dep_f32(logits + b * NUM_EXPERTS + e));
        sc[i] = raw;
        bi[i] = raw + rb[i];
    }
    int winners[TOP_K];
    float chosen[TOP_K];
    for (int k = 0; k < TOP_K; ++k) {
        float best = -1e30f;
        int best_e = NUM_EXPERTS;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int e = lane + 32 * i;
            if (bi[i] > best || (bi[i] == best && e < best_e)) { best = bi[i]; best_e = e; }
        }
        float top = warp_max(best);
        int cand = (best == top) ? best_e : NUM_EXPERTS;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, off));
        }
        cand = min(cand, NUM_EXPERTS - 1);
        float sel = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (lane + 32 * i == cand) { sel = sc[i]; bi[i] = -1e30f; }
        }
        chosen[k] = warp_sum(sel);
        winners[k] = cand;
    }

    __shared__ int32_t s_ids[B_CONST][TOP_K];
    __shared__ float s_w[B_CONST][TOP_K];
    if (lane == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) total += chosen[k];
        float inv = ROUTE_SCALE / (total + 1e-20f);
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            float w = chosen[k] * inv;
            out_ids[b * TOP_K + k] = winners[k];
            out_weights[b * TOP_K + k] = w;
            if constexpr (B_CONST > 1) {
                s_ids[b][k] = winners[k];
                s_w[b][k] = w;
            }
        }
    }

    if constexpr (B_CONST > 1) {
        if (!g) return;
        __shared__ int8_t exp_to_u[NUM_EXPERTS];
        __shared__ ExpertGroups sg;
        if (threadIdx.x < 32) {
            reinterpret_cast<uint32_t*>(exp_to_u)[threadIdx.x] = 0xFFFFFFFFu;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int n_uniq = 0;
            #pragma unroll
            for (int tb = 0; tb < B_CONST; ++tb) {
                #pragma unroll
                for (int s = 0; s < TOP_K; ++s) {
                    int e = s_ids[tb][s];
                    float rw = s_w[tb][s];
                    int u = (int)exp_to_u[e];
                    if (u < 0) {
                        u = n_uniq++;
                        exp_to_u[e] = (int8_t)u;
                        sg.expert_id[u] = e;
                        sg.token_mask[u] = 0;
                    }
                    sg.token_mask[u] |= uint8_t(1u << tb);
                    sg.slot_idx[u][tb] = uint8_t(s);
                    sg.route_w[u][tb] = rw;
                }
            }
            sg.num_unique = n_uniq;
        }
        __syncthreads();
        // Coalesced parallel copy of ExpertGroups from SMEM -> global!
        constexpr int NWORDS = sizeof(ExpertGroups) / sizeof(uint32_t);
        const uint32_t* src_u = reinterpret_cast<const uint32_t*>(&sg);
        uint32_t* dst_u = reinterpret_cast<uint32_t*>(g);
        for (int i = threadIdx.x; i < NWORDS; i += B_CONST * 32) {
            dst_u[i] = src_u[i];
        }
    }
}

struct CtrlNVFP4Ptr3 {
    const uint8_t* down_d[3];   const uint8_t* down_s[3];   float down_gs[3];
    const uint8_t* forget_d[3]; const uint8_t* forget_s[3]; float forget_gs[3];
    const uint8_t* gain_d[3];   const uint8_t* gain_s[3];   float gain_gs[3];
};

// MEGA-FUSION 1 (v2): 1024-thread (32-warp!) COALESCED single-kernel NVFP4 cross-pass controller!
// Step 2: 32 warps -> ONLY 2 rows/warp for down_w[64, 4608]!
// Step 3: 4 threads/row (256 rows simultaneously!) -> 100% coalesced uint2 loads for forget/gain[2304, 64]!
// Grid: dim3(B, n_streams), Block: 1024 threads (32 warps).
__global__ void __launch_bounds__(1024)
mega_controller_nvfp4_kernel(
    MutPtr3 xs,
    const float* __restrict__ saved,
    const float* __restrict__ prior_last,
    Ptr3 cp_queries,
    CtrlNVFP4Ptr3 cws,
    BF16Ptr3 post_norms,
    MutPtr3 outs,
    HalfPtr3 outs_h) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x; // 0..1023
    int lane = tid & 31;
    int warp = tid >> 5;   // 0..31

    __shared__ __half s_cat[2 * HIDDEN]; // 9,216 B
    __shared__ float s_msg[HIDDEN];      // 9,216 B
    __shared__ float s_m[HIDDEN];        // 9,216 B
    __shared__ __half s_d[GATE_RANK];    // 128 B
    __shared__ float sm0[32], sm1[32], smx[32];

    const float* q = cp_queries.p[s];
    const float* sv = saved + (size_t)b * HIDDEN;
    const float* pl = prior_last + (size_t)b * HIDDEN;
    const float* xb = xs.p[s] + (size_t)b * HIDDEN;

    // 1. Depth attention + norm(x)
    float d0 = 0.0f, d1 = 0.0f, xsq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 1024) {
        float qv = q[d];
        d0 = fmaf(qv, sv[d], d0);
        d1 = fmaf(qv, pl[d], d1);
        float xv = xb[d];
        s_m[d] = xv;
        xsq = fmaf(xv, xv, xsq);
    }
    d0 = warp_sum(d0); d1 = warp_sum(d1); xsq = warp_sum(xsq);
    if (lane == 0) { sm0[warp] = d0; sm1[warp] = d1; smx[warp] = xsq; }
    __syncthreads();

    __shared__ float w0, w1, inv_x;
    if (warp == 0) {
        float t0 = warp_sum(sm0[lane]);
        float t1 = warp_sum(sm1[lane]);
        float tx = warp_sum(smx[lane]);
        if (lane == 0) {
            float l0 = t0 * INV_SQRT_HIDDEN, l1 = t1 * INV_SQRT_HIDDEN;
            float mx = fmaxf(l0, l1);
            float e0 = expf(l0 - mx), e1 = expf(l1 - mx);
            float inv = 1.0f / (e0 + e1);
            w0 = e0 * inv; w1 = e1 * inv;
            inv_x = rsqrtf(tx / float(HIDDEN) + RMS_EPS);
        }
    }
    __syncthreads();

    float ww0 = w0, ww1 = w1, ix = inv_x, msq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 1024) {
        float mv = ww0 * sv[d] + ww1 * pl[d];
        s_msg[d] = mv;
        s_cat[d] = __float2half(s_m[d] * ix);
        msq = fmaf(mv, mv, msq);
    }
    msq = warp_sum(msq);
    if (lane == 0) sm0[warp] = msq;
    __syncthreads();
    __shared__ float inv_m;
    if (warp == 0) {
        float tm = warp_sum(sm0[lane]);
        if (lane == 0) inv_m = rsqrtf(tm / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float im = inv_m;
    for (int d = tid; d < HIDDEN; d += 1024) s_cat[HIDDEN + d] = __float2half(s_msg[d] * im);
    __syncthreads();

    // 2. Down projection: 64 rows x 4608 -> 32 warps = ONLY 2 rows per warp (9 EXACT iters!)
    const uint8_t* dd = cws.down_d[s];
    const uint8_t* ds = cws.down_s[s];
    float dgs = cws.down_gs[s];
    constexpr int D_NBLK = (2 * HIDDEN) >> 4; // 288 = 9 * 32 EXACT!
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int r = warp + pass * 32;
        const uint2* wr = reinterpret_cast<const uint2*>(dd + (size_t)r * HIDDEN);
        const uint8_t* sr = ds + (size_t)r * D_NBLK;
        float acc = 0.0f;
        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int blk = lane + iter * 32;
            uint2 w8 = __ldg(wr + blk);
            float sc = cvt_e4m3_to_float(__ldg(sr + blk));
            const __half2* xh = reinterpret_cast<const __half2*>(s_cat + (blk << 4));
            acc = fmaf(nvfp4_block16_dot(w8, xh), sc, acc);
        }
        float val = warp_sum(acc) * dgs;
        if (lane == 0) s_d[r] = __float2half(val);
    }
    __syncthreads();

    // 3. COALESCED Up projections (4 threads/row = 256 rows in parallel, 9 iterations!)
    const uint8_t* fd = cws.forget_d[s]; const uint8_t* fs = cws.forget_s[s]; float fgs = cws.forget_gs[s];
    const uint8_t* gd = cws.gain_d[s];   const uint8_t* gs = cws.gain_s[s];   float ggs = cws.gain_gs[s];
    int sub = tid & 3;         // 0..3
    int row_grp = tid >> 2;    // 0..255
    const __half2* dh_sub = reinterpret_cast<const __half2*>(s_d + (sub << 4));

    float out_sq = 0.0f;
    #pragma unroll
    for (int iter = 0; iter < 9; ++iter) {
        int c = row_grp + iter * 256;
        uint2 fw8 = __ldg(reinterpret_cast<const uint2*>(fd + (size_t)c * 32) + sub);
        uint2 gw8 = __ldg(reinterpret_cast<const uint2*>(gd + (size_t)c * 32) + sub);
        float fsc = cvt_e4m3_to_float(__ldg(fs + (size_t)c * 4 + sub));
        float gsc = cvt_e4m3_to_float(__ldg(gs + (size_t)c * 4 + sub));
        float af = nvfp4_block16_dot(fw8, dh_sub) * fsc;
        float ag = nvfp4_block16_dot(gw8, dh_sub) * gsc;
        af += __shfl_xor_sync(0xffffffffu, af, 1);
        af += __shfl_xor_sync(0xffffffffu, af, 2);
        ag += __shfl_xor_sync(0xffffffffu, ag, 1);
        ag += __shfl_xor_sync(0xffffffffu, ag, 2);
        if (sub == 0) {
            float merged = (1.0f - fast_tanh(af * fgs)) * s_m[c] + fast_tanh(ag * ggs) * s_msg[c];
            s_m[c] = merged;
            out_sq = fmaf(merged, merged, out_sq);
        }
    }
    out_sq = warp_sum(out_sq);
    if (lane == 0) sm0[warp] = out_sq;
    __syncthreads();
    if (warp == 0) {
        float tot = warp_sum(sm0[lane]);
        if (lane == 0) inv_x = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float ir = inv_x;
    const __nv_bfloat16* nw = post_norms.p[s];
    float* ob = outs.p[s] + (size_t)b * HIDDEN;
    __half* ohb = outs_h.p[s] ? (outs_h.p[s] + (size_t)b * HIDDEN) : nullptr;
    for (int c = tid; c < HIDDEN; c += 1024) {
        float v = s_m[c] * ir * __bfloat162float(nw[c]);
        ob[c] = v;
        if (ohb) ohb[c] = __float2half(v);
    }
}

// Vectorized 4-channel short-conv helper (2x uint4 weight loads + float4 data loads!)
__device__ __forceinline__ float4 sconv_apply_f4(
    const float* __restrict__ src,
    const __nv_bfloat16* __restrict__ sconv_w,
    float* __restrict__ ring,
    int b, int p, int c4, bool write_ring) {
    float4 cur = *reinterpret_cast<const float4*>(src + (size_t)b * HIDDEN + c4);
    float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 t2 = (b >= 1) ? *reinterpret_cast<const float4*>(src + (size_t)(b - 1) * HIDDEN + c4)
                         : ((p >= 1) ? *reinterpret_cast<const float4*>(ring + (size_t)((p - 1) & 15) * HIDDEN + c4) : z4);
    float4 t1 = (b >= 2) ? *reinterpret_cast<const float4*>(src + (size_t)(b - 2) * HIDDEN + c4)
                         : ((p >= 2) ? *reinterpret_cast<const float4*>(ring + (size_t)((p - 2) & 15) * HIDDEN + c4) : z4);
    float4 t0 = (b >= 3) ? *reinterpret_cast<const float4*>(src + (size_t)(b - 3) * HIDDEN + c4)
                         : ((p >= 3) ? *reinterpret_cast<const float4*>(ring + (size_t)((p - 3) & 15) * HIDDEN + c4) : z4);
    if (write_ring) {
        *reinterpret_cast<float4*>(ring + (size_t)(p & 15) * HIDDEN + c4) = cur;
    }
    const uint4* sw = reinterpret_cast<const uint4*>(sconv_w + c4 * 4);
    uint4 sw0 = __ldg(sw), sw1 = __ldg(sw + 1);
    const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&sw0);
    const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&sw1);
    float2 w0a = __bfloat1622float2(b0[0]), w0b = __bfloat1622float2(b0[1]);
    float2 w1a = __bfloat1622float2(b0[2]), w1b = __bfloat1622float2(b0[3]);
    float2 w2a = __bfloat1622float2(b1[0]), w2b = __bfloat1622float2(b1[1]);
    float2 w3a = __bfloat1622float2(b1[2]), w3b = __bfloat1622float2(b1[3]);
    float4 out;
    out.x = (fmaf(w0a.x, t0.x, fmaf(w0a.y, t1.x, fmaf(w0b.x, t2.x, w0b.y * cur.x))) + cur.x) * INV_SQRT_5;
    out.y = (fmaf(w1a.x, t0.y, fmaf(w1a.y, t1.y, fmaf(w1b.x, t2.y, w1b.y * cur.y))) + cur.y) * INV_SQRT_5;
    out.z = (fmaf(w2a.x, t0.z, fmaf(w2a.y, t1.z, fmaf(w2b.x, t2.z, w2b.y * cur.z))) + cur.z) * INV_SQRT_5;
    out.w = (fmaf(w3a.x, t0.w, fmaf(w3a.y, t1.w, fmaf(w3b.x, t2.w, w3b.y * cur.w))) + cur.w) * INV_SQRT_5;
    return out;
}

__device__ __forceinline__ float f4_dot(float4 a, float4 b) {
    return fmaf(a.x, b.x, fmaf(a.y, b.y, fmaf(a.z, b.z, a.w * b.w)));
}

__device__ __forceinline__ void store_f4_and_h4(float4 v, float* dst_f, __half* dst_h) {
    *reinterpret_cast<float4*>(dst_f) = v;
    if (dst_h) {
        __half2 h0 = __floats2half2_rn(v.x, v.y);
        __half2 h1 = __floats2half2_rn(v.z, v.w);
        uint2 packed;
        packed.x = *reinterpret_cast<uint32_t*>(&h0);
        packed.y = *reinterpret_cast<uint32_t*>(&h1);
        *reinterpret_cast<uint2*>(dst_h) = packed;
    }
}

__device__ __forceinline__ float4 load_moe_f4(const float* __restrict__ src, int b, int c4, bool is_slotted) {
    if (!is_slotted) {
        return *reinterpret_cast<const float4*>(src + (size_t)b * HIDDEN + c4);
    }
    const float* base = src + (size_t)b * TOP_K * HIDDEN + c4;
    float4 acc = *reinterpret_cast<const float4*>(base);
    #pragma unroll
    for (int s = 1; s < TOP_K; ++s) {
        float4 v = *reinterpret_cast<const float4*>(base + (size_t)s * HIDDEN);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    return acc;
}

__device__ __forceinline__ float4 sconv_apply_f4_slotted(
    const float* __restrict__ src,
    bool is_slotted,
    const __nv_bfloat16* __restrict__ sconv_w,
    float* __restrict__ ring,
    int b, int p, int c4, bool write_ring) {
    float4 cur = load_moe_f4(src, b, c4, is_slotted);
    float4 z4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 t2 = (b >= 1) ? load_moe_f4(src, b - 1, c4, is_slotted)
                         : ((p >= 1) ? *reinterpret_cast<const float4*>(ring + (size_t)((p - 1) & 15) * HIDDEN + c4) : z4);
    float4 t1 = (b >= 2) ? load_moe_f4(src, b - 2, c4, is_slotted)
                         : ((p >= 2) ? *reinterpret_cast<const float4*>(ring + (size_t)((p - 2) & 15) * HIDDEN + c4) : z4);
    float4 t0 = (b >= 3) ? load_moe_f4(src, b - 3, c4, is_slotted)
                         : ((p >= 3) ? *reinterpret_cast<const float4*>(ring + (size_t)((p - 3) & 15) * HIDDEN + c4) : z4);
    if (write_ring) {
        *reinterpret_cast<float4*>(ring + (size_t)(p & 15) * HIDDEN + c4) = cur;
    }
    const uint4* sw = reinterpret_cast<const uint4*>(sconv_w + c4 * 4);
    uint4 sw0 = __ldg(sw), sw1 = __ldg(sw + 1);
    const __nv_bfloat162* b0 = reinterpret_cast<const __nv_bfloat162*>(&sw0);
    const __nv_bfloat162* b1 = reinterpret_cast<const __nv_bfloat162*>(&sw1);
    float2 w0a = __bfloat1622float2(b0[0]), w0b = __bfloat1622float2(b0[1]);
    float2 w1a = __bfloat1622float2(b0[2]), w1b = __bfloat1622float2(b0[3]);
    float2 w2a = __bfloat1622float2(b1[0]), w2b = __bfloat1622float2(b1[1]);
    float2 w3a = __bfloat1622float2(b1[2]), w3b = __bfloat1622float2(b1[3]);
    float4 out;
    out.x = (fmaf(w0a.x, t0.x, fmaf(w0a.y, t1.x, fmaf(w0b.x, t2.x, w0b.y * cur.x))) + cur.x) * INV_SQRT_5;
    out.y = (fmaf(w1a.x, t0.y, fmaf(w1a.y, t1.y, fmaf(w1b.x, t2.y, w1b.y * cur.y))) + cur.y) * INV_SQRT_5;
    out.z = (fmaf(w2a.x, t0.z, fmaf(w2a.y, t1.z, fmaf(w2b.x, t2.z, w2b.y * cur.z))) + cur.z) * INV_SQRT_5;
    out.w = (fmaf(w3a.x, t0.w, fmaf(w3a.y, t1.w, fmaf(w3b.x, t2.w, w3b.y * cur.w))) + cur.w) * INV_SQRT_5;
    return out;
}

// MEGA-FUSION 6: Global-layer dynamic k_sconv + v_sconv + Q/R/K clip + log-pos tau + V norm + KV store (BF16/FP8/Q4)!
// Replaces 5 sequential kernels per global visit (40 launches -> 8 launches!) with ONE parallel kernel!
// Grid = dim3(64, B), Block = 128 threads (4 warps).
template <int KV_FMT>
__global__ void __launch_bounds__(128)
mega_global_sconv_prep_store_kernel(
    float* __restrict__ qrgs,
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    const float* __restrict__ in_k,
    const float* __restrict__ in_v,
    const __nv_bfloat16* __restrict__ k_sconv_w,    // [4, KV_DIM]
    const __nv_bfloat16* __restrict__ v_sconv_w,    // [4, KV_DIM]
    const __nv_bfloat16* __restrict__ k_sconv_proj, // [32, HIDDEN]
    const __nv_bfloat16* __restrict__ v_sconv_proj, // [32, HIDDEN]
    float* __restrict__ k_ring,
    float* __restrict__ v_ring,
    const __nv_bfloat16* __restrict__ q_clip_scale,
    const __nv_bfloat16* __restrict__ rel_clip_scale,
    void* __restrict__ k_cache,
    void* __restrict__ v_cache,
    __half* __restrict__ k_scale,
    __half* __restrict__ v_scale,
    int kv_cap, const int32_t* __restrict__ d_pos) {
    int kind_id = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    pdl_wait();
    pdl_trigger();
    int p = (*d_pos) + b;
    float ratio = fmaxf(float(p + 1) / float(LOG_SCALE_FLOOR), 1.0f);
    float tau = 1.0f + LOG_SCALE_ALPHA * logf(ratio);
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    __shared__ float sm[4];
    __shared__ float s_dt[4];
    __shared__ float factor;

    if (kind_id < 24) {
        int h = kind_id;
        float* qp = qrgs + (size_t)b * QRGS_STRIDE + h * HEAD_DIM;
        float val = qp[tid] * __bfloat162float(q_clip_scale[h * HEAD_DIM + tid]);
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f) * tau;
        __syncthreads();
        qp[tid] = val * factor;
    } else if (kind_id < 48) {
        if (tid >= D_REL) return;
        int h = kind_id - 24;
        float* rp = qrgs + (size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL;
        float val = rp[tid] * __bfloat162float(rel_clip_scale[h * D_REL + tid]);
        float sq = val * val;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) sq += __shfl_xor_sync(0xFFFFu, sq, off);
        rp[tid] = val * fminf(REL_CLIP / (sqrtf(sq) + 1e-6f), 1.0f) * tau;
    } else if (kind_id < 56) {
        int h = kind_id - 48;
        // 4 warps compute the 4 dynamic taps dt[0..3] for KV head `h` in parallel!
        const uint4* wr = reinterpret_cast<const uint4*>(k_sconv_proj + (size_t)(h * 4 + warp) * HIDDEN);
        const float* x_in = in_k + (size_t)b * HIDDEN;
        float dacc = 0.0f;
        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int v = lane + iter * 32; // 288 = 9 * 32 EXACT (288 * 8 = 2304)!
            uint4 w8 = __ldg(wr + v);
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
            float2 wf0 = __bfloat1622float2(wb[0]), wf1 = __bfloat1622float2(wb[1]);
            float2 wf2 = __bfloat1622float2(wb[2]), wf3 = __bfloat1622float2(wb[3]);
            const float4* xf = reinterpret_cast<const float4*>(x_in + (v << 3));
            float4 x0 = __ldg(xf), x1 = __ldg(xf + 1);
            dacc = fmaf(wf0.x, x0.x, dacc); dacc = fmaf(wf0.y, x0.y, dacc);
            dacc = fmaf(wf1.x, x0.z, dacc); dacc = fmaf(wf1.y, x0.w, dacc);
            dacc = fmaf(wf2.x, x1.x, dacc); dacc = fmaf(wf2.y, x1.y, dacc);
            dacc = fmaf(wf3.x, x1.z, dacc); dacc = fmaf(wf3.y, x1.w, dacc);
        }
        dacc = warp_sum(dacc);
        if (lane == 0) s_dt[warp] = dacc;
        __syncthreads();

        int c = h * HEAD_DIM + tid;
        float cur = k_proj[(size_t)b * KV_DIM + c];
        float h1 = (b >= 1) ? k_proj[(size_t)(b - 1) * KV_DIM + c] : ((p >= 1) ? k_ring[((p - 1) & 15) * KV_DIM + c] : 0.0f);
        float h2 = (b >= 2) ? k_proj[(size_t)(b - 2) * KV_DIM + c] : ((p >= 2) ? k_ring[((p - 2) & 15) * KV_DIM + c] : 0.0f);
        float h3 = (b >= 3) ? k_proj[(size_t)(b - 3) * KV_DIM + c] : ((p >= 3) ? k_ring[((p - 3) & 15) * KV_DIM + c] : 0.0f);
        k_ring[(p & 15) * KV_DIM + c] = cur;
        float acc = cur * (__bfloat162float(k_sconv_w[0 * KV_DIM + c]) + s_dt[0])
                  + h1  * (__bfloat162float(k_sconv_w[1 * KV_DIM + c]) + s_dt[1])
                  + h2  * (__bfloat162float(k_sconv_w[2 * KV_DIM + c]) + s_dt[2])
                  + h3  * (__bfloat162float(k_sconv_w[3 * KV_DIM + c]) + s_dt[3]);
        float val = (acc + cur) * INV_SQRT_9;
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f);
        __syncthreads();
        store_kv_head_128<KV_FMT>(val * factor, p % kv_cap, h, tid, k_cache, k_scale, sm);
    } else {
        int h = kind_id - 56;
        const uint4* wr = reinterpret_cast<const uint4*>(v_sconv_proj + (size_t)(h * 4 + warp) * HIDDEN);
        const float* x_in = in_v + (size_t)b * HIDDEN;
        float dacc = 0.0f;
        #pragma unroll
        for (int iter = 0; iter < 9; ++iter) {
            int v = lane + iter * 32;
            uint4 w8 = __ldg(wr + v);
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&w8);
            float2 wf0 = __bfloat1622float2(wb[0]), wf1 = __bfloat1622float2(wb[1]);
            float2 wf2 = __bfloat1622float2(wb[2]), wf3 = __bfloat1622float2(wb[3]);
            const float4* xf = reinterpret_cast<const float4*>(x_in + (v << 3));
            float4 x0 = __ldg(xf), x1 = __ldg(xf + 1);
            dacc = fmaf(wf0.x, x0.x, dacc); dacc = fmaf(wf0.y, x0.y, dacc);
            dacc = fmaf(wf1.x, x0.z, dacc); dacc = fmaf(wf1.y, x0.w, dacc);
            dacc = fmaf(wf2.x, x1.x, dacc); dacc = fmaf(wf2.y, x1.y, dacc);
            dacc = fmaf(wf3.x, x1.z, dacc); dacc = fmaf(wf3.y, x1.w, dacc);
        }
        dacc = warp_sum(dacc);
        if (lane == 0) s_dt[warp] = dacc;
        __syncthreads();

        int c = h * HEAD_DIM + tid;
        float cur = v_proj[(size_t)b * KV_DIM + c];
        float h1 = (b >= 1) ? v_proj[(size_t)(b - 1) * KV_DIM + c] : ((p >= 1) ? v_ring[((p - 1) & 15) * KV_DIM + c] : 0.0f);
        float h2 = (b >= 2) ? v_proj[(size_t)(b - 2) * KV_DIM + c] : ((p >= 2) ? v_ring[((p - 2) & 15) * KV_DIM + c] : 0.0f);
        float h3 = (b >= 3) ? v_proj[(size_t)(b - 3) * KV_DIM + c] : ((p >= 3) ? v_ring[((p - 3) & 15) * KV_DIM + c] : 0.0f);
        v_ring[(p & 15) * KV_DIM + c] = cur;
        float acc = cur * (__bfloat162float(v_sconv_w[0 * KV_DIM + c]) + s_dt[0])
                  + h1  * (__bfloat162float(v_sconv_w[1 * KV_DIM + c]) + s_dt[1])
                  + h2  * (__bfloat162float(v_sconv_w[2 * KV_DIM + c]) + s_dt[2])
                  + h3  * (__bfloat162float(v_sconv_w[3 * KV_DIM + c]) + s_dt[3]);
        float val = (acc + cur) * INV_SQRT_9;
        float sq = warp_sum(val * val);
        if (lane == 0) sm[warp] = sq;
        __syncthreads();
        if (tid == 0) factor = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
        __syncthreads();
        store_kv_head_128<KV_FMT>(val * factor, p % kv_cap, h, tid, v_cache, v_scale, sm);
    }
}

// MEGA-FUSION 2 (v2): PARALLEL across B tokens! Grid = dim3(64, B), Block = 128!
// Safe on 16-slot ring because b=0..B-1 ONLY reads prior slots pos0-3..pos0-1 from ring and reads
// intra-batch history from read-only k_proj/v_proj, while writing distinct slots (pos0+b)&15!
__global__ void __launch_bounds__(128)
mega_local_sconv_prep_store_kernel(
    float* __restrict__ qrgs,
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    const __nv_bfloat16* __restrict__ k_sconv_w,
    const __nv_bfloat16* __restrict__ v_sconv_w,
    float* __restrict__ k_ring,
    float* __restrict__ v_ring,
    const __nv_bfloat16* __restrict__ q_clip_scale,
    const __nv_bfloat16* __restrict__ rel_clip_scale,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int kv_cap, const int32_t* __restrict__ d_pos) {
    int kind_id = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x;
    pdl_wait();
    pdl_trigger();
    int pos0 = *d_pos;
    int p = pos0 + b;
    constexpr int QRGS_STRIDE = Q_DIM + R_DIM + Q_DIM + NUM_HEADS;
    __shared__ float sm[4];
    __shared__ float factor;

    if (kind_id < 24) {
        int h = kind_id;
        float* qp = qrgs + (size_t)b * QRGS_STRIDE + h * HEAD_DIM;
        float val = qp[tid] * __bfloat162float(q_clip_scale[h * HEAD_DIM + tid]);
        float sq = warp_sum(val * val);
        if ((tid & 31) == 0) sm[tid >> 5] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f);
        __syncthreads();
        qp[tid] = val * factor;
    } else if (kind_id < 48) {
        if (tid >= D_REL) return;
        int h = kind_id - 24;
        float* rp = qrgs + (size_t)b * QRGS_STRIDE + Q_DIM + h * D_REL;
        float val = rp[tid] * __bfloat162float(rel_clip_scale[h * D_REL + tid]);
        float sq = val * val;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1) sq += __shfl_xor_sync(0xFFFFu, sq, off);
        rp[tid] = val * fminf(REL_CLIP / (sqrtf(sq) + 1e-6f), 1.0f);
    } else if (kind_id < 56) {
        int h = kind_id - 48;
        int c = h * HEAD_DIM + tid;
        float cur = k_proj[(size_t)b * KV_DIM + c];
        float t2 = (b >= 1) ? k_proj[(size_t)(b - 1) * KV_DIM + c] : ((p >= 1) ? k_ring[((p - 1) & 15) * KV_DIM + c] : 0.0f);
        float t1 = (b >= 2) ? k_proj[(size_t)(b - 2) * KV_DIM + c] : ((p >= 2) ? k_ring[((p - 2) & 15) * KV_DIM + c] : 0.0f);
        float t0 = (b >= 3) ? k_proj[(size_t)(b - 3) * KV_DIM + c] : ((p >= 3) ? k_ring[((p - 3) & 15) * KV_DIM + c] : 0.0f);
        k_ring[(p & 15) * KV_DIM + c] = cur;
        float val = (__bfloat162float(k_sconv_w[c * 4 + 0]) * t0
                   + __bfloat162float(k_sconv_w[c * 4 + 1]) * t1
                   + __bfloat162float(k_sconv_w[c * 4 + 2]) * t2
                   + __bfloat162float(k_sconv_w[c * 4 + 3]) * cur + cur) * INV_SQRT_5;
        float sq = warp_sum(val * val);
        if ((tid & 31) == 0) sm[tid >> 5] = sq;
        __syncthreads();
        if (tid == 0) factor = fminf(QK_CLIP / (sqrtf(sm[0] + sm[1] + sm[2] + sm[3]) + 1e-6f), 1.0f);
        __syncthreads();
        k_cache[(size_t)(p % kv_cap) * KV_DIM + c] = __float2bfloat16(val * factor);
    } else {
        int h = kind_id - 56;
        int c = h * HEAD_DIM + tid;
        float cur = v_proj[(size_t)b * KV_DIM + c];
        float t2 = (b >= 1) ? v_proj[(size_t)(b - 1) * KV_DIM + c] : ((p >= 1) ? v_ring[((p - 1) & 15) * KV_DIM + c] : 0.0f);
        float t1 = (b >= 2) ? v_proj[(size_t)(b - 2) * KV_DIM + c] : ((p >= 2) ? v_ring[((p - 2) & 15) * KV_DIM + c] : 0.0f);
        float t0 = (b >= 3) ? v_proj[(size_t)(b - 3) * KV_DIM + c] : ((p >= 3) ? v_ring[((p - 3) & 15) * KV_DIM + c] : 0.0f);
        v_ring[(p & 15) * KV_DIM + c] = cur;
        float val = (__bfloat162float(v_sconv_w[c * 4 + 0]) * t0
                   + __bfloat162float(v_sconv_w[c * 4 + 1]) * t1
                   + __bfloat162float(v_sconv_w[c * 4 + 2]) * t2
                   + __bfloat162float(v_sconv_w[c * 4 + 3]) * cur + cur) * INV_SQRT_5;
        float sq = warp_sum(val * val);
        if ((tid & 31) == 0) sm[tid >> 5] = sq;
        __syncthreads();
        if (tid == 0) factor = rsqrtf((sm[0] + sm[1] + sm[2] + sm[3]) / float(HEAD_DIM) + RMS_EPS);
        __syncthreads();
        v_cache[(size_t)(p % kv_cap) * KV_DIM + c] = __float2bfloat16(val * factor);
    }
}

// Runtime-gated directional ablation (adapted from satellitedown/inkling-2-mini-abliterated-ingine, Apache-2.0):
// y <- y - w * max(r . y - offset, floor) * r   if (!abl_d || abl_d . y > threshold)
__device__ __forceinline__ float4 apply_ablation_576(
    float4 y,
    const float* __restrict__ abl_r,
    const float* __restrict__ abl_d,
    const float* __restrict__ abl_p,
    int c4, int lane, int warp,
    float* sm_r, float* sm_d, float* s_coeff) {
    if (!abl_r) return y;
    float4 rv = *reinterpret_cast<const float4*>(abl_r + c4);
    float dr = warp_sum(f4_dot(y, rv));
    if (lane == 0) sm_r[warp] = dr;
    if (abl_d) {
        float4 dv = *reinterpret_cast<const float4*>(abl_d + c4);
        float dd = warp_sum(f4_dot(y, dv));
        if (lane == 0) sm_d[warp] = dd;
    }
    __syncthreads();
    if (warp == 0) {
        float tot_r = warp_sum((lane < 18) ? sm_r[lane] : 0.0f);
        float tot_d = abl_d ? warp_sum((lane < 18) ? sm_d[lane] : 0.0f) : 0.0f;
        if (lane == 0) {
            float4 ap = __ldg(reinterpret_cast<const float4*>(abl_p));
            *s_coeff = (!abl_d || tot_d > ap.w) ? (ap.x * fmaxf(tot_r - ap.y, ap.z)) : 0.0f;
        }
    }
    __syncthreads();
    float k = *s_coeff;
    return make_float4(y.x - k * rv.x, y.y - k * rv.y, y.z - k * rv.z, y.w - k * rv.w);
}

// MEGA-FUSION 3 (v3): 576-thread (18-warp) VECTORIZED float4 REGISTER-CACHED ZERO-SMEM-ARRAY!
// Every thread owns 1 contiguous float4 (4 channels -> 576 * 4 = 2304 EXACT!).
// Loads blocks[] ONCE into registers (2x fewer VRAM reads, 0 KB SMEM arrays, 4x fewer LDG/STG!).
__global__ void __launch_bounds__(576)
mega_attn_post_mlp_in_kernel(
    const float* __restrict__ o_proj,
    const __nv_bfloat16* __restrict__ attn_sconv_w,
    float* __restrict__ attn_ring,
    const float* __restrict__ hidden,
    const float* __restrict__ partial_prev,
    float* __restrict__ blocks,
    int block_count_before,
    bool is_boundary,
    float* __restrict__ residual_out,
    const float* __restrict__ mlp_query,
    const __nv_bfloat16* __restrict__ mlp_norm_w,
    float* __restrict__ mlp_out,
    __half* __restrict__ mlp_out_h,
    const int32_t* __restrict__ d_pos,
    const float* __restrict__ abl_r,
    const float* __restrict__ abl_d,
    const float* __restrict__ abl_p) {
    int b = blockIdx.x;
    int tid = threadIdx.x; // 0..575
    int lane = tid & 31;
    int warp = tid >> 5;   // 0..17
    int c4 = tid << 2;     // 0, 4, ..., 2300
    pdl_wait();
    pdl_trigger();
    int p = (*d_pos) + b;
    int n_blocks = block_count_before + (is_boundary ? 1 : 0);
    int n_cand = n_blocks + 1;

    __shared__ float sm[18];
    __shared__ float sm_abl[18];
    __shared__ float partials[18][8];
    __shared__ float weights[8];
    __shared__ float inv_rms;

    float* blk_b = blocks + (size_t)b * 4 * HIDDEN;

    // 1. Vectorized attn_sconv + optional runtime-gated ablation + block copy/residual -> r_mp in registers!
    float4 ao = sconv_apply_f4(o_proj, attn_sconv_w, attn_ring, b, p, c4, true);
    ao = apply_ablation_576(ao, abl_r, abl_d, abl_p, c4, lane, warp, sm, sm_abl, &inv_rms);
    float4 rv;
    if (is_boundary) {
        float4 pp = *reinterpret_cast<const float4*>(partial_prev + (size_t)b * HIDDEN + c4);
        *reinterpret_cast<float4*>(blk_b + (size_t)block_count_before * HIDDEN + c4) = pp;
        rv = ao;
    } else {
        float4 hv = *reinterpret_cast<const float4*>(hidden + (size_t)b * HIDDEN + c4);
        rv = make_float4(hv.x + ao.x, hv.y + ao.y, hv.z + ao.z, hv.w + ao.w);
    }
    *reinterpret_cast<float4*>(residual_out + (size_t)b * HIDDEN + c4) = rv;

    float sum_sq = warp_sum(f4_dot(rv, rv));
    if (lane == 0) sm[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
        if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float ir = inv_rms;
    float4 r_mp = make_float4(rv.x * ir, rv.y * ir, rv.z * ir, rv.w * ir);

    // 2. Depth attention: load blocks[] ONCE into registers r_blk[0..3]!
    float4 qv = *reinterpret_cast<const float4*>(mlp_query + c4);
    float4 r_blk[4];
    for (int c = 0; c < n_blocks; ++c) {
        r_blk[c] = *reinterpret_cast<const float4*>(blk_b + (size_t)c * HIDDEN + c4);
        float d = warp_sum(f4_dot(qv, r_blk[c]));
        if (lane == 0) partials[warp][c] = d;
    }
    {
        float d = warp_sum(f4_dot(qv, r_mp));
        if (lane == 0) partials[warp][n_blocks] = d;
    }
    __syncthreads();
    if (warp == 0) {
        float logits[8];
        float mx = -1e30f;
        for (int c = 0; c < n_cand; ++c) {
            float dot = warp_sum((lane < 18) ? partials[lane][c] : 0.0f);
            if (lane == 0) {
                logits[c] = dot * INV_SQRT_HIDDEN;
                mx = fmaxf(mx, logits[c]);
            }
        }
        if (lane == 0) {
            float den = 0.0f;
            for (int c = 0; c < n_cand; ++c) { weights[c] = expf(logits[c] - mx); den += weights[c]; }
            float inv_d = 1.0f / den;
            for (int c = 0; c < n_cand; ++c) weights[c] *= inv_d;
        }
    }
    __syncthreads();

    // 3. Weighted sum ENTIRELY IN REGISTERS + optional mlp_norm (+ half!)
    float wm = weights[n_blocks];
    float4 v = make_float4(wm * r_mp.x, wm * r_mp.y, wm * r_mp.z, wm * r_mp.w);
    for (int c = 0; c < n_blocks; ++c) {
        float wc = weights[c];
        v.x = fmaf(wc, r_blk[c].x, v.x);
        v.y = fmaf(wc, r_blk[c].y, v.y);
        v.z = fmaf(wc, r_blk[c].z, v.z);
        v.w = fmaf(wc, r_blk[c].w, v.w);
    }

    float* mo = mlp_out + (size_t)b * HIDDEN + c4;
    __half* moh = mlp_out_h ? (mlp_out_h + (size_t)b * HIDDEN + c4) : nullptr;
    if (mlp_norm_w) {
        float n_sq = warp_sum(f4_dot(v, v));
        if (lane == 0) sm[warp] = n_sq;
        __syncthreads();
        if (warp == 0) {
            float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
            if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
        }
        __syncthreads();
        float s2 = inv_rms;
        uint2 nw4 = __ldg(reinterpret_cast<const uint2*>(mlp_norm_w + c4));
        const __nv_bfloat162* nb = reinterpret_cast<const __nv_bfloat162*>(&nw4);
        float2 nf0 = __bfloat1622float2(nb[0]), nf1 = __bfloat1622float2(nb[1]);
        v.x = v.x * s2 * nf0.x;
        v.y = v.y * s2 * nf0.y;
        v.z = v.z * s2 * nf1.x;
        v.w = v.w * s2 * nf1.y;
        store_f4_and_h4(v, mo, moh);
    } else {
        *reinterpret_cast<float4*>(mo) = v;
    }
}

// MEGA-FUSION 4 (v3): 576-thread vectorized float4 with optional deterministic 8-slot MoE reduction!
__global__ void __launch_bounds__(576)
mega_mlp_post_res_kernel(
    const float* __restrict__ moe_down_or_slots,
    bool moe_is_slotted,
    const __nv_bfloat16* __restrict__ mlp_sconv_w,
    float* __restrict__ mlp_ring,
    const float* __restrict__ residual,
    float* __restrict__ hidden_out,
    float* __restrict__ loopcd_save_out,
    const int32_t* __restrict__ d_pos,
    const float* __restrict__ abl_r,
    const float* __restrict__ abl_d,
    const float* __restrict__ abl_p) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int c4 = tid << 2;
    pdl_wait();
    pdl_trigger();
    int p = (*d_pos) + b;
    __shared__ float sm_r[18];
    __shared__ float sm_d[18];
    __shared__ float s_coeff;
    float4 ff = sconv_apply_f4_slotted(moe_down_or_slots, moe_is_slotted, mlp_sconv_w, mlp_ring, b, p, c4, true);
    ff = apply_ablation_576(ff, abl_r, abl_d, abl_p, c4, lane, warp, sm_r, sm_d, &s_coeff);
    float4 res = *reinterpret_cast<const float4*>(residual + (size_t)b * HIDDEN + c4);
    float4 hv = make_float4(res.x + ff.x, res.y + ff.y, res.z + ff.z, res.w + ff.w);
    *reinterpret_cast<float4*>(hidden_out + (size_t)b * HIDDEN + c4) = hv;
    if (loopcd_save_out) {
        *reinterpret_cast<float4*>(loopcd_save_out + (size_t)b * HIDDEN + c4) = hv;
    }
}

// LoopCD-Hidden (arXiv:2610.02185 Eq. 2) in-place update for batched prefill:
//   h' = h_R + omega * (h_R - h_ref), bit-exact FP32 fmaf matching decode kernel.
__global__ void __launch_bounds__(576)
loopcd_hidden_apply_kernel(
    float* __restrict__ hidden,
    const float* __restrict__ ref_hidden,
    float omega) {
    int b = blockIdx.x;
    int c4 = threadIdx.x << 2;
    size_t off = (size_t)b * HIDDEN + c4;
    float4 hv = *reinterpret_cast<const float4*>(hidden + off);
    float4 rv = *reinterpret_cast<const float4*>(ref_hidden + off);
    hv.x = fmaf(omega, hv.x - rv.x, hv.x);
    hv.y = fmaf(omega, hv.y - rv.y, hv.y);
    hv.z = fmaf(omega, hv.z - rv.z, hv.z);
    hv.w = fmaf(omega, hv.w - rv.w, hv.w);
    *reinterpret_cast<float4*>(hidden + off) = hv;
}

// ============================================================================
// Multi-SM Batched NVFP4 Cross-Pass Controller (32-108 SMs, weight-shared across B!)
// ============================================================================

__global__ void __launch_bounds__(256)
ctrl_prep_cat_h_batched_kernel(
    MutPtr3 xs,
    const float* __restrict__ saved,
    const float* __restrict__ prior_last,
    Ptr3 cp_queries,
    float* __restrict__ out_msg,     // [n_streams, B, HIDDEN]
    __half* __restrict__ out_cat_h,  // [n_streams, B, 2*HIDDEN]
    int B) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    const float* q = cp_queries.p[s];
    const float* sv = saved + (size_t)b * HIDDEN;
    const float* pl = prior_last + (size_t)b * HIDDEN;
    const float* xb = xs.p[s] + (size_t)b * HIDDEN;

    float d0 = 0.0f, d1 = 0.0f, xsq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int d = tid + i * 256;
        float qv = q[d];
        d0 = fmaf(qv, sv[d], d0);
        d1 = fmaf(qv, pl[d], d1);
        float xv = xb[d];
        xsq = fmaf(xv, xv, xsq);
    }
    d0 = warp_sum(d0); d1 = warp_sum(d1); xsq = warp_sum(xsq);
    __shared__ float sm0[8], sm1[8], smx[8];
    if (lane == 0) { sm0[warp] = d0; sm1[warp] = d1; smx[warp] = xsq; }
    __syncthreads();

    __shared__ float w0, w1, inv_x;
    if (tid == 0) {
        float t0 = 0.0f, t1 = 0.0f, tx = 0.0f;
        for (int i = 0; i < 8; ++i) { t0 += sm0[i]; t1 += sm1[i]; tx += smx[i]; }
        float l0 = t0 * INV_SQRT_HIDDEN, l1 = t1 * INV_SQRT_HIDDEN;
        float mx = fmaxf(l0, l1);
        float e0 = expf(l0 - mx), e1 = expf(l1 - mx);
        float inv = 1.0f / (e0 + e1);
        w0 = e0 * inv; w1 = e1 * inv;
        inv_x = rsqrtf(tx / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    float ww0 = w0, ww1 = w1, ix = inv_x;
    float* msg_b = out_msg + ((size_t)s * B + b) * HIDDEN;
    __half* cat_b = out_cat_h + ((size_t)s * B + b) * (2 * HIDDEN);

    float msq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int d = tid + i * 256;
        float mv = ww0 * sv[d] + ww1 * pl[d];
        msg_b[d] = mv;
        cat_b[d] = __float2half(xb[d] * ix);
        msq = fmaf(mv, mv, msq);
    }
    msq = warp_sum(msq);
    if (lane == 0) sm0[warp] = msq;
    __syncthreads();
    __shared__ float inv_m;
    if (tid == 0) {
        float tm = 0.0f;
        for (int i = 0; i < 8; ++i) tm += sm0[i];
        inv_m = rsqrtf(tm / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();
    float im = inv_m;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int d = tid + i * 256;
        cat_b[HIDDEN + d] = __float2half(msg_b[d] * im);
    }
}

// Grid: dim3(32, n_streams), Block: 32 (2 half-warps = 2 rows/CTA, 18 exact iters, weight-shared across B!)
template <int B_CONST>
__global__ void __launch_bounds__(32)
ctrl_down_nvfp4_batched_kernel(
    CtrlNVFP4Ptr3 cws,
    const __half* __restrict__ cat_h, // [n_streams, B, 4608]
    __half* __restrict__ out_d_h) {   // [n_streams, B, 64]
    int s = blockIdx.y;
    int sub = threadIdx.x >> 4;  // 0 or 1
    int lane = threadIdx.x & 15; // 0..15
    int row = blockIdx.x * 2 + sub;

    constexpr int K = 2 * HIDDEN;
    constexpr int D_NBLK = K >> 4; // 288 = 18 * 16 EXACT!
    const uint2* wr = reinterpret_cast<const uint2*>(cws.down_d[s] + (size_t)row * (K >> 1));
    const uint8_t* sr = cws.down_s[s] + (size_t)row * D_NBLK;

    float acc[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) acc[b] = 0.0f;

    #pragma unroll
    for (int iter = 0; iter < 18; ++iter) {
        int blk = lane + iter * 16;
        uint2 w8 = __ldg(wr + blk);
        float sc = cvt_e4m3_to_float(__ldg(sr + blk));
        __half2 wh[8];
        cvt_e2m1x8_to_half2x4(w8.x, wh[0], wh[1], wh[2], wh[3]);
        cvt_e2m1x8_to_half2x4(w8.y, wh[4], wh[5], wh[6], wh[7]);
        int koff = blk << 4;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            const uint4* x4 = reinterpret_cast<const uint4*>(cat_h + ((size_t)s * B_CONST + b) * K + koff);
            uint4 xv0 = __ldg(x4), xv1 = __ldg(x4 + 1);
            const __half2* xb0 = reinterpret_cast<const __half2*>(&xv0);
            const __half2* xb1 = reinterpret_cast<const __half2*>(&xv1);
            __half2 p0 = __hmul2(wh[0], xb0[0]), p1 = __hmul2(wh[1], xb0[1]);
            p0 = __hfma2(wh[2], xb0[2], p0); p1 = __hfma2(wh[3], xb0[3], p1);
            p0 = __hfma2(wh[4], xb1[0], p0); p1 = __hfma2(wh[5], xb1[1], p1);
            p0 = __hfma2(wh[6], xb1[2], p0); p1 = __hfma2(wh[7], xb1[3], p1);
            float2 pf = __half22float2(__hadd2(p0, p1));
            acc[b] = fmaf(pf.x + pf.y, sc, acc[b]);
        }
    }
    float dgs = cws.down_gs[s];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float val = half_warp_sum(acc[b]) * dgs;
        if (lane == 0) out_d_h[((size_t)s * B_CONST + b) * GATE_RANK + row] = __float2half(val);
    }
}

// Grid: dim3(36, n_streams), Block: 256 (64 rows/CTA, 4 threads/row = 1 single uint2 load shared across B!)
template <int B_CONST>
__global__ void __launch_bounds__(256)
ctrl_up_merge_nvfp4_batched_kernel(
    CtrlNVFP4Ptr3 cws,
    MutPtr3 xs,
    const float* __restrict__ msg,
    const __half* __restrict__ d_h,
    MutPtr3 outs) {
    int s = blockIdx.y;
    int tid = threadIdx.x;
    __shared__ __half s_d[B_CONST][GATE_RANK];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        if (tid < GATE_RANK) s_d[b][tid] = d_h[((size_t)s * B_CONST + b) * GATE_RANK + tid];
    }
    __syncthreads();

    int row_in_blk = tid >> 2; // 0..63
    int sub = tid & 3;         // 0..3
    int row = blockIdx.x * 64 + row_in_blk;
    if (row >= HIDDEN) return;

    uint2 fw8 = __ldg(reinterpret_cast<const uint2*>(cws.forget_d[s] + (size_t)row * 32) + sub);
    uint2 gw8 = __ldg(reinterpret_cast<const uint2*>(cws.gain_d[s]   + (size_t)row * 32) + sub);
    float fsc = cvt_e4m3_to_float(__ldg(cws.forget_s[s] + (size_t)row * 4 + sub)) * cws.forget_gs[s];
    float gsc = cvt_e4m3_to_float(__ldg(cws.gain_s[s]   + (size_t)row * 4 + sub)) * cws.gain_gs[s];

    __half2 fh[8], gh[8];
    cvt_e2m1x8_to_half2x4(fw8.x, fh[0], fh[1], fh[2], fh[3]);
    cvt_e2m1x8_to_half2x4(fw8.y, fh[4], fh[5], fh[6], fh[7]);
    cvt_e2m1x8_to_half2x4(gw8.x, gh[0], gh[1], gh[2], gh[3]);
    cvt_e2m1x8_to_half2x4(gw8.y, gh[4], gh[5], gh[6], gh[7]);

    int koff = sub << 4;
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        const __half2* dh = reinterpret_cast<const __half2*>(&s_d[b][koff]);
        __half2 pf0 = __hmul2(fh[0], dh[0]), pf1 = __hmul2(fh[1], dh[1]);
        __half2 pg0 = __hmul2(gh[0], dh[0]), pg1 = __hmul2(gh[1], dh[1]);
        pf0 = __hfma2(fh[2], dh[2], pf0); pf1 = __hfma2(fh[3], dh[3], pf1);
        pg0 = __hfma2(gh[2], dh[2], pg0); pg1 = __hfma2(gh[3], dh[3], pg1);
        pf0 = __hfma2(fh[4], dh[4], pf0); pf1 = __hfma2(fh[5], dh[5], pf1);
        pg0 = __hfma2(gh[4], dh[4], pg0); pg1 = __hfma2(gh[5], dh[5], pg1);
        pf0 = __hfma2(fh[6], dh[6], pf0); pf1 = __hfma2(fh[7], dh[7], pf1);
        pg0 = __hfma2(gh[6], dh[6], pg0); pg1 = __hfma2(gh[7], dh[7], pg1);
        float2 ff = __half22float2(__hadd2(pf0, pf1));
        float2 gg = __half22float2(__hadd2(pg0, pg1));
        float af = (ff.x + ff.y) * fsc;
        float ag = (gg.x + gg.y) * gsc;
        af += __shfl_xor_sync(0xffffffffu, af, 1);
        af += __shfl_xor_sync(0xffffffffu, af, 2);
        ag += __shfl_xor_sync(0xffffffffu, ag, 1);
        ag += __shfl_xor_sync(0xffffffffu, ag, 2);
        if (sub == 0) {
            float xv = xs.p[s][(size_t)b * HIDDEN + row];
            float mv = msg[((size_t)s * B_CONST + b) * HIDDEN + row];
            outs.p[s][(size_t)b * HIDDEN + row] = (1.0f - fast_tanh(af)) * xv + fast_tanh(ag) * mv;
        }
    }
}

// ============================================================================
// MEGA-FUSION 5 (v2): 576-thread (18-warp) VECTORIZED float4 REGISTER-CACHED!
// Fuses mlp_post_res(vi-1) + pre_tap + next attn_in_depth_norm(vi) (+ post_tap)!
// ZERO SMEM arrays, loads blocks[] ONCE into registers, 4x fewer LDG/STG!
// Grid: dim3(B, n_out), Block: 576 threads.
// ============================================================================
__global__ void __launch_bounds__(576)
mega_mlp_post_next_depth_norm_kernel(
    const float* __restrict__ moe_down_or_slots,
    bool moe_is_slotted,
    const __nv_bfloat16* __restrict__ prev_mlp_sconv_w,
    float* __restrict__ prev_mlp_ring,
    const float* __restrict__ residual,
    float* __restrict__ hidden_out,
    float* __restrict__ pre_tap_out,
    float* __restrict__ post_tap_out,
    float* __restrict__ partial_out,
    const float* __restrict__ blocks,
    int n_blocks,
    const float* __restrict__ queries,
    BF16Ptr3 norms,
    MutPtr3 outs,
    HalfPtr3 outs_h,
    const int32_t* __restrict__ d_pos,
    const float* __restrict__ abl_r,
    const float* __restrict__ abl_d,
    const float* __restrict__ abl_p,
    const float* __restrict__ loopcd_h1,
    const float* __restrict__ loopcd_h2,
    const LoopCDConfig* __restrict__ loopcd_cfg) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x; // 0..575
    int lane = tid & 31;
    int warp = tid >> 5;   // 0..17
    int c4 = tid << 2;     // 0, 4, ..., 2300
    pdl_wait();
    pdl_trigger();
    int p = (*d_pos) + b;

    __shared__ float sm[18];
    __shared__ float sm_abl[18];
    __shared__ float partials[18][8];
    __shared__ float weights[8];
    __shared__ float inv_rms;

    // 1. Vectorized mlp_sconv (with deterministic 8-slot MoE sum when B>1!) + optional ablation + residual add -> hv in regs!
    float4 ff = sconv_apply_f4_slotted(moe_down_or_slots, moe_is_slotted, prev_mlp_sconv_w, prev_mlp_ring, b, p, c4, s == 0);
    ff = apply_ablation_576(ff, abl_r, abl_d, abl_p, c4, lane, warp, sm, sm_abl, &inv_rms);
    float4 res = *reinterpret_cast<const float4*>(residual + (size_t)b * HIDDEN + c4);
    float4 hv = make_float4(res.x + ff.x, res.y + ff.y, res.z + ff.z, res.w + ff.w);
    // Optional LoopCD-Hidden (arXiv:2610.02185 Eq. 2) right at the Loop -> Coda boundary (vi == 42):
    //   h' = h_R + omega * (h_R - h_ref) with 0 extra launches and bit-exact skip when omega == 0.0f.
    if (loopcd_cfg) {
        float omega = loopcd_cfg->omega;
        if (omega != 0.0f) {
            const float* ref_buf = (loopcd_cfg->ref_pass == 2) ? loopcd_h2 : loopcd_h1;
            float4 rv = *reinterpret_cast<const float4*>(ref_buf + (size_t)b * HIDDEN + c4);
            hv.x = fmaf(omega, hv.x - rv.x, hv.x);
            hv.y = fmaf(omega, hv.y - rv.y, hv.y);
            hv.z = fmaf(omega, hv.z - rv.z, hv.z);
            hv.w = fmaf(omega, hv.w - rv.w, hv.w);
        }
    }
    if (s == 0) {
        if (hidden_out)  *reinterpret_cast<float4*>(hidden_out  + (size_t)b * HIDDEN + c4) = hv;
        if (pre_tap_out) *reinterpret_cast<float4*>(pre_tap_out + (size_t)b * HIDDEN + c4) = hv;
    }

    float sum_sq = warp_sum(f4_dot(hv, hv));
    if (lane == 0) sm[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
        if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    float ir = inv_rms;
    float4 r_part = make_float4(hv.x * ir, hv.y * ir, hv.z * ir, hv.w * ir);
    if (s == 0 && partial_out) {
        *reinterpret_cast<float4*>(partial_out + (size_t)b * HIDDEN + c4) = r_part;
    }

    // 2. Depth attention: load blocks[] ONCE into registers r_blk[0..3]!
    float4 qv = *reinterpret_cast<const float4*>(queries + (size_t)s * HIDDEN + c4);
    const float* blk_b = blocks + (size_t)b * 4 * HIDDEN;
    int n_cand = n_blocks + 1;
    float4 r_blk[4];
    for (int c = 0; c < n_blocks; ++c) {
        r_blk[c] = *reinterpret_cast<const float4*>(blk_b + (size_t)c * HIDDEN + c4);
        float d = warp_sum(f4_dot(qv, r_blk[c]));
        if (lane == 0) partials[warp][c] = d;
    }
    {
        float d = warp_sum(f4_dot(qv, r_part));
        if (lane == 0) partials[warp][n_blocks] = d;
    }
    __syncthreads();
    if (warp == 0) {
        float logits[8];
        float mx = -1e30f;
        for (int c = 0; c < n_cand; ++c) {
            float dot = warp_sum((lane < 18) ? partials[lane][c] : 0.0f);
            if (lane == 0) {
                logits[c] = dot * INV_SQRT_HIDDEN;
                mx = fmaxf(mx, logits[c]);
            }
        }
        if (lane == 0) {
            float den = 0.0f;
            for (int c = 0; c < n_cand; ++c) { weights[c] = expf(logits[c] - mx); den += weights[c]; }
            float inv_d = 1.0f / den;
            for (int c = 0; c < n_cand; ++c) weights[c] *= inv_d;
        }
    }
    __syncthreads();

    // 3. Weighted sum ENTIRELY IN REGISTERS + optional RMSNorm (+ half + post_tap!)
    float wp = weights[n_blocks];
    float4 v = make_float4(wp * r_part.x, wp * r_part.y, wp * r_part.z, wp * r_part.w);
    for (int c = 0; c < n_blocks; ++c) {
        float wc = weights[c];
        v.x = fmaf(wc, r_blk[c].x, v.x);
        v.y = fmaf(wc, r_blk[c].y, v.y);
        v.z = fmaf(wc, r_blk[c].z, v.z);
        v.w = fmaf(wc, r_blk[c].w, v.w);
    }

    const __nv_bfloat16* nw = norms.p[s];
    float* ob = outs.p[s] + (size_t)b * HIDDEN + c4;
    __half* oh = outs_h.p[s] ? (outs_h.p[s] + (size_t)b * HIDDEN + c4) : nullptr;
    float* post_b = (s == 0 && post_tap_out) ? (post_tap_out + (size_t)b * HIDDEN + c4) : nullptr;
    if (nw) {
        float n_sq = warp_sum(f4_dot(v, v));
        if (lane == 0) sm[warp] = n_sq;
        __syncthreads();
        if (warp == 0) {
            float tot = warp_sum((lane < 18) ? sm[lane] : 0.0f);
            if (lane == 0) inv_rms = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
        }
        __syncthreads();
        float s2 = inv_rms;
        uint2 nw4 = __ldg(reinterpret_cast<const uint2*>(nw + c4));
        const __nv_bfloat162* nb = reinterpret_cast<const __nv_bfloat162*>(&nw4);
        float2 nf0 = __bfloat1622float2(nb[0]), nf1 = __bfloat1622float2(nb[1]);
        v.x = v.x * s2 * nf0.x;
        v.y = v.y * s2 * nf0.y;
        v.z = v.z * s2 * nf1.x;
        v.w = v.w * s2 * nf1.y;
    }
    store_f4_and_h4(v, ob, oh);
    if (post_b) *reinterpret_cast<float4*>(post_b) = v;
}

__global__ void vec_add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void copy_block_kernel(const float* __restrict__ src, float* __restrict__ blocks, int max_blocks, int slot_idx, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * HIDDEN;
    if (i < total) {
        int b = i / HIDDEN;
        int d = i % HIDDEN;
        blocks[((size_t)b * max_blocks + slot_idx) * HIDDEN + d] = src[i];
    }
}

__global__ void extract_block_kernel(const float* __restrict__ blocks, int max_blocks, int slot_idx, float* __restrict__ dst, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * HIDDEN;
    if (i < total) {
        int b = i / HIDDEN;
        int d = i % HIDDEN;
        dst[i] = blocks[((size_t)b * max_blocks + slot_idx) * HIDDEN + d];
    }
}

} // namespace

void launch_embed_norm(
    const __nv_bfloat16* embed, const __nv_bfloat16* embed_norm,
    const int32_t* tokens, float* out, int B, cudaStream_t stream) {
    embed_norm_kernel<<<B, 256, 0, stream>>>(embed, embed_norm, tokens, out);
}

void launch_rmsnorm(
    const float* x, const __nv_bfloat16* weight, float* out,
    int B, int width, int group_size, float eps, cudaStream_t stream) {
    int groups = B * (width / group_size);
    int threads = (group_size >= 256) ? 256 : 128;
    rmsnorm_kernel<<<groups, threads, 0, stream>>>(x, weight, out, width, group_size, eps);
}

void launch_depth_attn(
    const float* blocks, int max_blocks, int n_blocks, const float* partial,
    const float* query, int q_idx, float* out, int B, cudaStream_t stream) {
    depth_attn_kernel<<<B, 256, 0, stream>>>(
        blocks, max_blocks, n_blocks, partial, query + (size_t)q_idx * HIDDEN, out);
}

void launch_fused_depth_norm(
    const float* hidden, float* partial_out, const float* blocks, int n_blocks,
    const float* queries, const __nv_bfloat16* norm0, const __nv_bfloat16* norm1, const __nv_bfloat16* norm2,
    float* out0, float* out1, float* out2, __half* out_h0, __half* out_h1, __half* out_h2,
    int n_out, int B, cudaStream_t stream) {
    BF16Ptr3 np{{norm0, norm1, norm2}};
    MutPtr3 op{{out0, out1, out2}};
    HalfPtr3 ohp{{out_h0, out_h1, out_h2}};
    fused_depth_norm_kernel<<<dim3(B, n_out), 256, 0, stream>>>(
        hidden, partial_out, blocks, n_blocks, queries, np, op, ohp);
}

// ============================================================================
// 1-Launch 16-CTA Cluster + DSMEM BF16 Router & Cross-Pass Controller (fp8 & exact!)
// Adapted from satellitedown/inkling-2-mini-abliterated-ingine gates.cu & router.cu (Apache-2.0).
// Cuts 96 router launches -> 48 and 192 controller launches -> 48 with 100% PDL weight prefetch
// and ZERO KLD / precision loss (exact BF16 weights + FP32 activations)!
// ============================================================================

template <int B_CONST>
__global__ void __launch_bounds__(256)
router_cluster_bf16_kernel(
    const float* __restrict__ m_in,            // [B_CONST, HIDDEN]
    const __nv_bfloat16* __restrict__ router_w,// [128, HIDDEN]
    const float* __restrict__ router_bias,     // [128]
    int32_t* __restrict__ out_ids,             // [B_CONST, TOP_K]
    float* __restrict__ out_weights,           // [B_CONST, TOP_K]
    uint32_t zero) {
    constexpr int CL = 16;
    constexpr int RPC = NUM_EXPERTS / CL; // 8 experts/CTA (1 warp per expert)
    constexpr int NT = 256;               // 8 warps
    constexpr int CH = HIDDEN / (8 * 32); // 9 uint4 chunks per lane
    constexpr int XG = (B_CONST < 4) ? B_CONST : 4;

    __shared__ __align__(16) float s_x[XG][HIDDEN]; // max 36,864 B (< 48 KB!)
    __shared__ __align__(16) float s_logits[NUM_EXPERTS];
    __shared__ __align__(16) float s_part[B_CONST][RPC];
    __shared__ __align__(8) uint64_t mbar;
    __shared__ unsigned long long s_keys[NUM_EXPERTS];
    __shared__ int s_cnt[NT];
    __shared__ float s_sel_s[TOP_K];
    __shared__ int32_t s_sel_e[TOP_K];

    cg::cluster_group cl = cg::this_cluster();
    const int rank = cl.block_rank();
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row = rank * RPC + warp;

    const uint32_t logits_u = cluster_smem_u32(s_logits);
    const uint32_t mb = cluster_smem_u32(&mbar);
    if (tid == 0) {
        cluster_mb_init(mb);
        cluster_fence_mb_init();
        if (rank < B_CONST) cluster_mb_expect_tx(mb, 4u * NUM_EXPERTS);
    }
    cluster_arrive_relaxed();

    // Prefetch ALL router weights + bias BEFORE pdl_wait_fenced!
    const __nv_bfloat16* wr = router_w + (size_t)row * HIDDEN;
    uint4 wv[CH];
    #pragma unroll
    for (int i = 0; i < CH; ++i) {
        wv[i] = ld_stream_u4(wr + (i * 32 + lane) * 8);
    }
    const float bias_val = (tid < NUM_EXPERTS) ? __ldg(router_bias + tid) : 0.0f;

    pdl_wait_fenced(zero);
    pdl_trigger();

    #pragma unroll
    for (int m0 = 0; m0 < B_CONST; m0 += XG) {
        if (m0 > 0) __syncthreads();
        const int cnt = (m0 + XG <= B_CONST) ? XG : (B_CONST - m0);
        const int n_f4 = (cnt * HIDDEN) >> 2;
        const float* x_base = m_in + (size_t)m0 * HIDDEN;
        for (int k = tid; k < n_f4; k += NT) {
            cp_async16_cg(&s_x[0][0] + (k << 2), x_base + (k << 2));
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();

        float acc[XG];
        #pragma unroll
        for (int m = 0; m < XG; ++m) acc[m] = 0.0f;

        #pragma unroll
        for (int i = 0; i < CH; ++i) {
            const int col = (i * 32 + lane) * 8;
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv[i]);
            float2 wf0 = __bfloat1622float2(wb[0]);
            float2 wf1 = __bfloat1622float2(wb[1]);
            float2 wf2 = __bfloat1622float2(wb[2]);
            float2 wf3 = __bfloat1622float2(wb[3]);
            #pragma unroll
            for (int m = 0; m < XG; ++m) {
                if (m < cnt) {
                    const float4* xf = reinterpret_cast<const float4*>(&s_x[m][col]);
                    float4 x0 = xf[0], x1 = xf[1];
                    float a = acc[m];
                    a = fmaf(wf0.x, x0.x, a); a = fmaf(wf0.y, x0.y, a);
                    a = fmaf(wf1.x, x0.z, a); a = fmaf(wf1.y, x0.w, a);
                    a = fmaf(wf2.x, x1.x, a); a = fmaf(wf2.y, x1.y, a);
                    a = fmaf(wf3.x, x1.z, a); a = fmaf(wf3.y, x1.w, a);
                    acc[m] = a;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < XG; ++m) {
            if (m < cnt) {
                float s = warp_sum(acc[m]);
                if (lane == 0) s_part[m0 + m][warp] = s;
            }
        }
    }
    __syncthreads();
    cluster_wait_acquire();

    // Send 8 logits (= 2 float4s) per token to CTA `tok` via DSMEM st.async
    if (tid < B_CONST * 2) {
        int tok = tid % B_CONST;
        int grp = tid / B_CONST; // 0 or 1
        float4 v = *reinterpret_cast<const float4*>(&s_part[tok][grp * 4]);
        cluster_st_async_f4(
            cluster_remote_u32(logits_u + 4u * (rank * RPC + grp * 4), tok),
            v,
            cluster_remote_u32(mb, tok));
    }
    if (rank >= B_CONST) return;
    cluster_mb_wait(mb);

    // Parallel O(1) rank-based top-8 selection (bit-exact lower-index tie break!)
    float sig = 0.0f;
    if (tid < NUM_EXPERTS) {
        sig = fast_sigmoid(s_logits[tid]);
        uint32_t u = __float_as_uint(sig + bias_val + 0.0f);
        uint32_t o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        s_keys[tid] = (static_cast<unsigned long long>(o) << 7) | static_cast<unsigned long long>(NUM_EXPERTS - 1 - tid);
    }
    __syncthreads();
    {
        constexpr int PARTS = NT / NUM_EXPERTS; // 2
        constexpr int W = NUM_EXPERTS / PARTS;  // 64
        int e = tid & (NUM_EXPERTS - 1);
        int part = tid >> 7;
        unsigned long long mine = s_keys[e];
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        #pragma unroll
        for (int j = 0; j < W; j += 4) {
            c0 += (s_keys[part * W + j + 0] > mine);
            c1 += (s_keys[part * W + j + 1] > mine);
            c2 += (s_keys[part * W + j + 2] > mine);
            c3 += (s_keys[part * W + j + 3] > mine);
        }
        s_cnt[tid] = c0 + c1 + c2 + c3;
    }
    __syncthreads();
    if (tid < NUM_EXPERTS) {
        int r = s_cnt[tid] + s_cnt[NUM_EXPERTS + tid];
        if (r < TOP_K) {
            s_sel_e[r] = tid;
            s_sel_s[r] = sig;
        }
    }
    __syncthreads();
    if (tid < TOP_K) {
        float my_s = s_sel_s[tid];
        float sum = my_s;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffu, sum, o);
        out_ids[rank * TOP_K + tid] = s_sel_e[tid];
        out_weights[rank * TOP_K + tid] = my_s / (sum + 1e-20f) * ROUTE_SCALE;
    }
}

constexpr int CTRL_CL = 16;
constexpr int CTRL_CPC = HIDDEN / CTRL_CL; // 144 channels per CTA
constexpr int CTRL_ROW_BYTES = GATE_RANK * 2; // 128 B (64 BF16)
constexpr int CTRL_RS = CTRL_ROW_BYTES + 16;  // 144 B padded stride (conflict-free)
constexpr int CTRL_SLICE = CTRL_CPC * CTRL_RS; // 20,736 B
constexpr int CTRL_DYN_SMEM = 2 * CTRL_SLICE;  // 41,472 B

template <int B_CONST>
__global__ void __launch_bounds__(256)
controller_cluster_bf16_kernel(
    MutPtr3 xs,
    const float* __restrict__ saved,
    const float* __restrict__ prior_last,
    Ptr3 cp_queries,
    CtrlPtr3 cws,
    BF16Ptr3 post_norms,
    MutPtr3 outs,
    HalfPtr3 outs_h,
    uint32_t zero) {
    constexpr int RPC = GATE_RANK / CTRL_CL; // 4 down rows/CTA
    constexpr int NT = 256;                  // 8 warps = 4 rows x 2 halves
    constexpr int DCH = HIDDEN / (8 * 32);   // 9 uint4 chunks per lane
    constexpr int MQ = (B_CONST + 3) / 4;    // 1 (B<=4) or 2 (B=5..8)
    constexpr int XG = (B_CONST < 4) ? B_CONST : 4;

    extern __shared__ __align__(16) uint8_t fg[]; // forget[SLICE], gain[SLICE]
    __shared__ __align__(16) __nv_bfloat16 s_cat[XG][2 * HIDDEN];
    __shared__ __align__(16) float s_d[B_CONST][GATE_RANK];
    __shared__ __align__(16) float s_part[B_CONST][2][RPC];
    __shared__ __align__(16) float s_ssall[CTRL_CL][4 * MQ];
    __shared__ __align__(16) float s_local_ss[4 * MQ];
    __shared__ float s_w0[B_CONST], s_w1[B_CONST], s_inv_rms[B_CONST];
    __shared__ float s_r0[8], s_r1[8], s_r2[8];
    __shared__ __align__(8) uint64_t mbar[2];

    cg::cluster_group cl = cg::this_cluster();
    const int rank = cl.block_rank();
    const int g = blockIdx.x / CTRL_CL;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rl = warp >> 1;  // 0..3
    const int half = warp & 1; // 0 (x) or 1 (msg)

    const uint32_t mb_g = cluster_smem_u32(&mbar[0]);
    const uint32_t mb_s = cluster_smem_u32(&mbar[1]);
    if (tid == 0) {
        cluster_mb_init(mb_g);
        cluster_mb_init(mb_s);
        cluster_fence_mb_init();
        cluster_mb_expect_tx(mb_g, 16u * CTRL_CL * B_CONST);
        cluster_mb_expect_tx(mb_s, 16u * CTRL_CL * MQ);
    }
    cluster_arrive_relaxed();

    // 1. Prefetch ALL weights (forget/gain via cp.async -> SMEM, down -> regs, cp_q & post_norm -> regs) BEFORE PDL wait!
    {
        const uint8_t* fsrc = reinterpret_cast<const uint8_t*>(cws.forget[g] + (size_t)(rank * CTRL_CPC) * GATE_RANK);
        const uint8_t* gsrc = reinterpret_cast<const uint8_t*>(cws.gain[g]   + (size_t)(rank * CTRL_CPC) * GATE_RANK);
        constexpr int RCH = CTRL_ROW_BYTES / 16; // 8 chunks/row
        for (int k = tid; k < CTRL_CPC * RCH; k += NT) {
            int dst = (k / RCH) * CTRL_RS + (k % RCH) * 16;
            cp_async16_cg(fg + dst, fsrc + (size_t)k * 16);
            cp_async16_cg(fg + CTRL_SLICE + dst, gsrc + (size_t)k * 16);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    uint4 wv[DCH];
    {
        const __nv_bfloat16* wr = cws.down[g] + (size_t)(rank * RPC + rl) * (2 * HIDDEN) + half * HIDDEN;
        #pragma unroll
        for (int i = 0; i < DCH; ++i) {
            wv[i] = ld_stream_u4(wr + (i * 32 + lane) * 8);
        }
    }

    float qv[9];
    {
        const float* qp = cp_queries.p[g];
        #pragma unroll
        for (int i = 0; i < 9; ++i) qv[i] = __ldg(qp + tid + i * NT);
    }
    const float post_nw = (tid < CTRL_CPC) ? __bfloat162float(__ldg(post_norms.p[g] + rank * CTRL_CPC + tid)) : 0.0f;

    pdl_wait_fenced(zero);
    pdl_trigger();

    // 2. Stage 1: depth attention + RMSNorm(x) + RMSNorm(msg) + down projection in groups of XG tokens
    const float* x_ptr = xs.p[g];
    #pragma unroll
    for (int m0 = 0; m0 < B_CONST; m0 += XG) {
        if (m0 > 0) __syncthreads();
        const int cnt = (m0 + XG <= B_CONST) ? XG : (B_CONST - m0);
        for (int m = 0; m < cnt; ++m) {
            const int b = m0 + m;
            const float* xb = x_ptr + (size_t)b * HIDDEN;
            const float* sv = saved + (size_t)b * HIDDEN;
            const float* pl = prior_last + (size_t)b * HIDDEN;

            float x_reg[9], s_reg[9], p_reg[9];
            float d0 = 0.0f, d1 = 0.0f, xsq = 0.0f;
            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                int d = tid + i * NT;
                float xv = ld_dep_f32(xb + d);
                float svv = ld_dep_f32(sv + d);
                float plv = ld_dep_f32(pl + d);
                x_reg[i] = xv; s_reg[i] = svv; p_reg[i] = plv;
                d0 = fmaf(qv[i], svv, d0);
                d1 = fmaf(qv[i], plv, d1);
                xsq = fmaf(xv, xv, xsq);
            }
            d0 = warp_sum(d0); d1 = warp_sum(d1); xsq = warp_sum(xsq);
            if (lane == 0) { s_r0[warp] = d0; s_r1[warp] = d1; s_r2[warp] = xsq; }
            __syncthreads();
            if (tid == 0) {
                float t0 = 0.f, t1 = 0.f, tx = 0.f;
                for (int w = 0; w < 8; ++w) { t0 += s_r0[w]; t1 += s_r1[w]; tx += s_r2[w]; }
                float l0 = t0 * INV_SQRT_HIDDEN, l1 = t1 * INV_SQRT_HIDDEN;
                float mx = fmaxf(l0, l1);
                float e0 = expf(l0 - mx), e1 = expf(l1 - mx);
                float inv = 1.0f / (e0 + e1);
                s_w0[b] = e0 * inv;
                s_w1[b] = e1 * inv;
                s_r0[0] = rsqrtf(tx / float(HIDDEN) + RMS_EPS);
            }
            __syncthreads();
            float ww0 = s_w0[b], ww1 = s_w1[b], ix = s_r0[0];
            float m_reg[9];
            float msq = 0.0f;
            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                int d = tid + i * NT;
                float mv = fmaf(ww0, s_reg[i], ww1 * p_reg[i]);
                m_reg[i] = mv;
                s_cat[m][d] = __float2bfloat16_rn(x_reg[i] * ix);
                msq = fmaf(mv, mv, msq);
            }
            msq = warp_sum(msq);
            if (lane == 0) s_r0[warp] = msq;
            __syncthreads();
            if (tid == 0) {
                float tm = 0.f;
                for (int w = 0; w < 8; ++w) tm += s_r0[w];
                s_r0[0] = rsqrtf(tm / float(HIDDEN) + RMS_EPS);
            }
            __syncthreads();
            float im = s_r0[0];
            #pragma unroll
            for (int i = 0; i < 9; ++i) {
                int d = tid + i * NT;
                s_cat[m][HIDDEN + d] = __float2bfloat16_rn(m_reg[i] * im);
            }
        }
        __syncthreads();

        // Down dot products reusing wv[0..8] across all `cnt` tokens in registers!
        float acc[XG];
        #pragma unroll
        for (int m = 0; m < XG; ++m) acc[m] = 0.0f;
        #pragma unroll
        for (int i = 0; i < DCH; ++i) {
            int col = half * HIDDEN + (i * 32 + lane) * 8;
            const __nv_bfloat162* wb = reinterpret_cast<const __nv_bfloat162*>(&wv[i]);
            float2 wf0 = __bfloat1622float2(wb[0]);
            float2 wf1 = __bfloat1622float2(wb[1]);
            float2 wf2 = __bfloat1622float2(wb[2]);
            float2 wf3 = __bfloat1622float2(wb[3]);
            #pragma unroll
            for (int m = 0; m < XG; ++m) {
                if (m < cnt) {
                    uint4 x8 = *reinterpret_cast<const uint4*>(&s_cat[m][col]);
                    const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&x8);
                    float2 xf0 = __bfloat1622float2(xb[0]);
                    float2 xf1 = __bfloat1622float2(xb[1]);
                    float2 xf2 = __bfloat1622float2(xb[2]);
                    float2 xf3 = __bfloat1622float2(xb[3]);
                    float a = acc[m];
                    a = fmaf(wf0.x, xf0.x, a); a = fmaf(wf0.y, xf0.y, a);
                    a = fmaf(wf1.x, xf1.x, a); a = fmaf(wf1.y, xf1.y, a);
                    a = fmaf(wf2.x, xf2.x, a); a = fmaf(wf2.y, xf2.y, a);
                    a = fmaf(wf3.x, xf3.x, a); a = fmaf(wf3.y, xf3.y, a);
                    acc[m] = a;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < XG; ++m) {
            if (m < cnt) {
                float s = warp_sum(acc[m]);
                if (lane == 0) s_part[m0 + m][half][rl] = s;
            }
        }
    }
    __syncthreads();
    cluster_wait_acquire();

    // 3. Broadcast this CTA's 4 down outputs per token to all 16 CTAs in the cluster via DSMEM!
    const uint32_t sd_u = cluster_smem_u32(s_d);
    if (tid < CTRL_CL * B_CONST) {
        int dst_rank = tid & (CTRL_CL - 1);
        int b = tid >> 4;
        float4 v = make_float4(
            s_part[b][0][0] + s_part[b][1][0],
            s_part[b][0][1] + s_part[b][1][1],
            s_part[b][0][2] + s_part[b][1][2],
            s_part[b][0][3] + s_part[b][1][3]);
        cluster_st_async_f4(
            cluster_remote_u32(sd_u + 4u * (b * GATE_RANK + rank * RPC), dst_rank),
            v,
            cluster_remote_u32(mb_g, dst_rank));
    }

    asm volatile("cp.async.wait_all;" ::: "memory");
    cluster_mb_wait(mb_g);
    __syncthreads();

    // 4. Forget/Gain up-projections from padded SMEM + gated mix + per-CTA sum-of-squares!
    float merged[B_CONST];
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) merged[b] = 0.0f;

    if (tid < CTRL_CPC) {
        const uint8_t* frow = fg + tid * CTRL_RS;
        const uint8_t* grow = fg + CTRL_SLICE + tid * CTRL_RS;
        float af[B_CONST], ag[B_CONST];
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) { af[b] = 0.0f; ag[b] = 0.0f; }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint4 fw8 = *reinterpret_cast<const uint4*>(frow + j * 16);
            uint4 gw8 = *reinterpret_cast<const uint4*>(grow + j * 16);
            const __nv_bfloat162* fb = reinterpret_cast<const __nv_bfloat162*>(&fw8);
            const __nv_bfloat162* gb = reinterpret_cast<const __nv_bfloat162*>(&gw8);
            float2 ff0 = __bfloat1622float2(fb[0]), ff1 = __bfloat1622float2(fb[1]);
            float2 ff2 = __bfloat1622float2(fb[2]), ff3 = __bfloat1622float2(fb[3]);
            float2 gf0 = __bfloat1622float2(gb[0]), gf1 = __bfloat1622float2(gb[1]);
            float2 gf2 = __bfloat1622float2(gb[2]), gf3 = __bfloat1622float2(gb[3]);
            #pragma unroll
            for (int b = 0; b < B_CONST; ++b) {
                const float4* dp = reinterpret_cast<const float4*>(&s_d[b][j * 8]);
                float4 d0 = dp[0], d1 = dp[1];
                float a = af[b], c = ag[b];
                a = fmaf(ff0.x, d0.x, a); a = fmaf(ff0.y, d0.y, a);
                a = fmaf(ff1.x, d0.z, a); a = fmaf(ff1.y, d0.w, a);
                a = fmaf(ff2.x, d1.x, a); a = fmaf(ff2.y, d1.y, a);
                a = fmaf(ff3.x, d1.z, a); a = fmaf(ff3.y, d1.w, a);
                c = fmaf(gf0.x, d0.x, c); c = fmaf(gf0.y, d0.y, c);
                c = fmaf(gf1.x, d0.z, c); c = fmaf(gf1.y, d0.w, c);
                c = fmaf(gf2.x, d1.x, c); c = fmaf(gf2.y, d1.y, c);
                c = fmaf(gf3.x, d1.z, c); c = fmaf(gf3.y, d1.w, c);
                af[b] = a; ag[b] = c;
            }
        }
        const int ch = rank * CTRL_CPC + tid;
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            float x_ch = ld_dep_f32(x_ptr + (size_t)b * HIDDEN + ch);
            float m_ch = fmaf(s_w0[b], ld_dep_f32(saved + (size_t)b * HIDDEN + ch),
                              s_w1[b] * ld_dep_f32(prior_last + (size_t)b * HIDDEN + ch));
            merged[b] = (1.0f - fast_tanh(af[b])) * x_ch + fast_tanh(ag[b]) * m_ch;
        }
    }

    if (tid < 4 * MQ) s_local_ss[tid] = 0.0f;
    #pragma unroll
    for (int b = 0; b < B_CONST; ++b) {
        float sq = warp_sum(merged[b] * merged[b]);
        if (lane == 0) s_r0[warp] = sq;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            for (int w = 0; w < 8; ++w) tot += s_r0[w];
            s_local_ss[b] = tot;
        }
        __syncthreads();
    }

    // 5. Exchange per-CTA sum-of-squares across all 16 CTAs via DSMEM, normalize & write outputs!
    const uint32_t ss_u = cluster_smem_u32(s_ssall);
    if (tid < CTRL_CL * MQ) {
        int dst_rank = tid & (CTRL_CL - 1);
        int q = tid >> 4;
        float4 v = *reinterpret_cast<const float4*>(&s_local_ss[q * 4]);
        cluster_st_async_f4(
            cluster_remote_u32(ss_u + 16u * (rank * MQ + q), dst_rank),
            v,
            cluster_remote_u32(mb_s, dst_rank));
    }
    cluster_mb_wait(mb_s);
    __syncthreads();

    if (tid < B_CONST) {
        float tot = 0.0f;
        #pragma unroll
        for (int r = 0; r < CTRL_CL; ++r) tot += s_ssall[r][tid];
        s_inv_rms[tid] = rsqrtf(tot / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    if (tid < CTRL_CPC) {
        const int ch = rank * CTRL_CPC + tid;
        float* ob = outs.p[g];
        __half* ohb = outs_h.p[g];
        #pragma unroll
        for (int b = 0; b < B_CONST; ++b) {
            float v = merged[b] * s_inv_rms[b] * post_nw;
            ob[(size_t)b * HIDDEN + ch] = v;
            if (ohb) ohb[(size_t)b * HIDDEN + ch] = __float2half(v);
        }
    }
}

template <int B_CONST>
static void dispatch_ctrl_cluster_bf16(
    const MutPtr3& xp, const float* saved, const float* prior_last, const Ptr3& qp,
    const CtrlPtr3& cp, const BF16Ptr3& np, const MutPtr3& op, const HalfPtr3& ohp,
    int n_streams, cudaStream_t stream) {
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(
            controller_cluster_bf16_kernel<B_CONST>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, CTRL_DYN_SMEM));
        CUDA_CHECK(cudaFuncSetAttribute(
            controller_cluster_bf16_kernel<B_CONST>,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        attr_set = true;
    }
    launch_pdl_cluster(
        controller_cluster_bf16_kernel<B_CONST>,
        dim3(CTRL_CL * n_streams), 256, CTRL_DYN_SMEM, CTRL_CL, stream,
        xp, saved, prior_last, qp, cp, np, op, ohp, 0u);
}

template <int B_CONST>
static void dispatch_router_cluster_bf16(
    const float* m, const __nv_bfloat16* router_w, const float* router_bias,
    int32_t* out_ids, float* out_weights, cudaStream_t stream) {
    static bool attr_set = false;
    if (!attr_set) {
        CUDA_CHECK(cudaFuncSetAttribute(
            router_cluster_bf16_kernel<B_CONST>,
            cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
        attr_set = true;
    }
    launch_pdl_cluster(
        router_cluster_bf16_kernel<B_CONST>,
        dim3(16), 256, 0, 16, stream,
        m, router_w, router_bias, out_ids, out_weights, 0u);
}

void launch_controller_merge_batched(
    float* const* xs, const float* saved, const float* prior_last,
    const float* cp_queries[3], const ControllerWeights* cws[3], const __nv_bfloat16* post_norms[3],
    float* const* outs, __half* const* outs_h, int n_streams,
    float* scratch_msg, float* scratch_cat, float* scratch_d, int B, cudaStream_t stream) {
    MutPtr3 xp{{xs[0], n_streams > 1 ? xs[1] : xs[0], n_streams > 2 ? xs[2] : xs[0]}};
    MutPtr3 op{{outs[0], n_streams > 1 ? outs[1] : outs[0], n_streams > 2 ? outs[2] : outs[0]}};
    HalfPtr3 ohp{{
        outs_h ? outs_h[0] : nullptr,
        (outs_h && n_streams > 1) ? outs_h[1] : nullptr,
        (outs_h && n_streams > 2) ? outs_h[2] : nullptr
    }};
    Ptr3 qp{{cp_queries[0], n_streams > 1 ? cp_queries[1] : cp_queries[0], n_streams > 2 ? cp_queries[2] : cp_queries[0]}};
    CtrlPtr3 cp;
    BF16Ptr3 np;
    for (int s = 0; s < 3; ++s) {
        int idx = (s < n_streams) ? s : 0;
        cp.down[s] = cws[idx]->down_w;
        cp.forget[s] = cws[idx]->forget_w;
        cp.gain[s] = cws[idx]->gain_w;
        np.p[s] = post_norms[idx];
    }
    (void)scratch_msg; (void)scratch_cat; (void)scratch_d;
    switch (B) {
        case 1: dispatch_ctrl_cluster_bf16<1>(xp, saved, prior_last, qp, cp, np, op, ohp, n_streams, stream); break;
        case 2: dispatch_ctrl_cluster_bf16<2>(xp, saved, prior_last, qp, cp, np, op, ohp, n_streams, stream); break;
        case 3: dispatch_ctrl_cluster_bf16<3>(xp, saved, prior_last, qp, cp, np, op, ohp, n_streams, stream); break;
        case 4: dispatch_ctrl_cluster_bf16<4>(xp, saved, prior_last, qp, cp, np, op, ohp, n_streams, stream); break;
        case 5: dispatch_ctrl_cluster_bf16<5>(xp, saved, prior_last, qp, cp, np, op, ohp, n_streams, stream); break;
        case 6: dispatch_ctrl_cluster_bf16<6>(xp, saved, prior_last, qp, cp, np, op, ohp, n_streams, stream); break;
        case 7: dispatch_ctrl_cluster_bf16<7>(xp, saved, prior_last, qp, cp, np, op, ohp, n_streams, stream); break;
        case 8: dispatch_ctrl_cluster_bf16<8>(xp, saved, prior_last, qp, cp, np, op, ohp, n_streams, stream); break;
    }
}

void launch_static_sconv(
    const float* x, const __nv_bfloat16* w, float* ring, float* out,
    int B, int width, const int32_t* d_pos, cudaStream_t stream) {
    static_sconv_kernel<<<(width + 255) / 256, 256, 0, stream>>>(x, w, ring, out, B, width, d_pos);
}

void launch_dynamic_sconv(
    const float* x, const float* dyn_in, const __nv_bfloat16* static_w, const __nv_bfloat16* weight_proj,
    float* ring, float* out, float* scratch_taps, int B, const int32_t* d_pos, cudaStream_t stream) {
    launch_bf16_linear(weight_proj, dyn_in, scratch_taps, B, NUM_KV_HEADS * CONV_TAPS, HIDDEN, stream);
    dynamic_sconv_kernel<<<(KV_DIM + 255) / 256, 256, 0, stream>>>(x, static_w, scratch_taps, ring, out, B, d_pos);
}

void launch_attn_prep_store(
    float* qrgs, const float* ck, const float* cv,
    const __nv_bfloat16* q_clip_scale, const __nv_bfloat16* rel_clip_scale,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int kv_cap, bool is_global,
    int B, const int32_t* d_pos, cudaStream_t stream) {
    attn_prep_store_kernel<<<dim3(B, 64), 128, 0, stream>>>(
        qrgs, ck, cv, q_clip_scale, rel_clip_scale, k_cache, v_cache, kv_cap, is_global, d_pos);
}

template <int KV_FMT, int B_CONST>
static void dispatch_tc_attn_b(
    const float* qrgs, const __nv_bfloat16* rel_proj,
    const void* k_cache, const void* v_cache,
    const __half* k_scale, const __half* v_scale,
    float* split_buf, float* out_gated, __half* out_gated_h,
    int kv_cap, int window, const int32_t* d_pos, cudaStream_t stream) {
    constexpr int BULK_THREADS  = (GQA_RATIO * B_CONST > 16) ? 128 : 64;
    constexpr int LOCAL_THREADS = (GQA_RATIO * B_CONST > 16) ? 256 : 128;
    if (window == 0) {
        launch_pdl(global_attn_bulk_splitk_kernel<KV_FMT, B_CONST>, dim3(NUM_KV_HEADS, KV_SPLIT_K), BULK_THREADS, stream,
            qrgs, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos, split_buf);
    }
    launch_pdl(local_tail_attn_tc_kernel<KV_FMT, B_CONST>, dim3(NUM_KV_HEADS, KV_LOCAL_SPLITS), LOCAL_THREADS, stream,
        qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, kv_cap, window, d_pos, split_buf);
    launch_pdl(attn_merge_sink_gate_kernel, dim3(NUM_HEADS, B_CONST), HEAD_DIM, stream,
        qrgs, split_buf, out_gated, out_gated_h, window, d_pos);
}

template <int KV_FMT>
static void dispatch_relative_attention_fmt(
    const float* qrgs, const __nv_bfloat16* rel_proj,
    const void* k_cache, const void* v_cache,
    const __half* k_scale, const __half* v_scale,
    float* split_buf, float* out_gated, __half* out_gated_h,
    int kv_cap, int window, int B, const int32_t* d_pos, cudaStream_t stream) {
    if (split_buf != nullptr) {
        switch (B) {
            case 1: dispatch_tc_attn_b<KV_FMT, 1>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos, stream); break;
            case 2: dispatch_tc_attn_b<KV_FMT, 2>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos, stream); break;
            case 3: dispatch_tc_attn_b<KV_FMT, 3>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos, stream); break;
            case 4: dispatch_tc_attn_b<KV_FMT, 4>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos, stream); break;
            case 5: dispatch_tc_attn_b<KV_FMT, 5>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos, stream); break;
            case 6: dispatch_tc_attn_b<KV_FMT, 6>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos, stream); break;
            case 7: dispatch_tc_attn_b<KV_FMT, 7>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos, stream); break;
            case 8: dispatch_tc_attn_b<KV_FMT, 8>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos, stream); break;
        }
        return;
    }
    relative_attention_gated_inline_kernel<KV_FMT><<<dim3(NUM_HEADS, B), 256, 0, stream>>>(
        qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, d_pos);
}

void launch_relative_attention_gated(
    const float* qrgs, const __nv_bfloat16* rel_proj,
    const void* k_cache, const void* v_cache,
    const __half* k_scale, const __half* v_scale,
    KVFormat kv_fmt, float* split_buf, float* out_gated, __half* out_gated_h,
    int kv_cap, int window, int B, const int32_t* d_pos, cudaStream_t stream) {
    if (kv_fmt == KVFormat::FP8) {
        dispatch_relative_attention_fmt<1>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, B, d_pos, stream);
    } else if (kv_fmt == KVFormat::Q4) {
        dispatch_relative_attention_fmt<2>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, B, d_pos, stream);
    } else {
        dispatch_relative_attention_fmt<0>(qrgs, rel_proj, k_cache, v_cache, k_scale, v_scale, split_buf, out_gated, out_gated_h, kv_cap, window, B, d_pos, stream);
    }
}

template <int B_CONST>
static void dispatch_route_bf16(
    const float* m, const __nv_bfloat16* router_w, const float* router_bias,
    float* scratch_logits, int32_t* out_ids, float* out_weights,
    ExpertGroups* groups, float* moe_down_zero, cudaStream_t stream) {
    if (groups == nullptr && moe_down_zero == nullptr) {
        dispatch_router_cluster_bf16<B_CONST>(m, router_w, router_bias, out_ids, out_weights, stream);
        return;
    }
    route_logits_parallel_kernel<B_CONST><<<64, 64, 0, stream>>>(m, router_w, scratch_logits, moe_down_zero);
    launch_pdl(route_select_and_group_kernel<B_CONST>, dim3(1), B_CONST * 32, stream,
        scratch_logits, router_bias, out_ids, out_weights, groups, 0u);
}

void launch_route_topk(
    const float* m, const __nv_bfloat16* router_w, const float* router_bias,
    float* scratch_logits, int32_t* out_ids, float* out_weights,
    ExpertGroups* groups, float* moe_down_zero, int B, cudaStream_t stream) {
    switch (B) {
        case 1: dispatch_route_bf16<1>(m, router_w, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 2: dispatch_route_bf16<2>(m, router_w, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 3: dispatch_route_bf16<3>(m, router_w, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 4: dispatch_route_bf16<4>(m, router_w, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 5: dispatch_route_bf16<5>(m, router_w, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 6: dispatch_route_bf16<6>(m, router_w, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 7: dispatch_route_bf16<7>(m, router_w, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 8: dispatch_route_bf16<8>(m, router_w, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
    }
}

template <int B_CONST>
static void dispatch_route_nvfp4(
    const __half* m_h, const LinearWeight& r, const float* router_bias,
    float* scratch_logits, int32_t* out_ids, float* out_weights,
    ExpertGroups* groups, float* moe_down_zero, cudaStream_t stream) {
    launch_pdl(route_logits_nvfp4_parallel_kernel<B_CONST>, dim3(64), 32, stream,
        m_h, static_cast<const uint8_t*>(r.data), r.scales, r.global_scale, scratch_logits, moe_down_zero, 0u);
    launch_pdl(route_select_and_group_kernel<B_CONST>, dim3(1), B_CONST * 32, stream,
        scratch_logits, router_bias, out_ids, out_weights, groups, 0u);
}

void launch_route_topk_nvfp4(
    const __half* m_h, const LinearWeight& router_nvfp4, const float* router_bias,
    float* scratch_logits, int32_t* out_ids, float* out_weights,
    ExpertGroups* groups, float* moe_down_zero, int B, cudaStream_t stream) {
    switch (B) {
        case 1: dispatch_route_nvfp4<1>(m_h, router_nvfp4, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 2: dispatch_route_nvfp4<2>(m_h, router_nvfp4, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 3: dispatch_route_nvfp4<3>(m_h, router_nvfp4, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 4: dispatch_route_nvfp4<4>(m_h, router_nvfp4, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 5: dispatch_route_nvfp4<5>(m_h, router_nvfp4, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 6: dispatch_route_nvfp4<6>(m_h, router_nvfp4, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 7: dispatch_route_nvfp4<7>(m_h, router_nvfp4, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
        case 8: dispatch_route_nvfp4<8>(m_h, router_nvfp4, router_bias, scratch_logits, out_ids, out_weights, groups, moe_down_zero, stream); break;
    }
}

template <int B_CONST>
static void dispatch_ctrl_nvfp4_b(
    const CtrlNVFP4Ptr3& cp, const MutPtr3& xp, const MutPtr3& op,
    const float* scratch_msg, const __half* scratch_cat_h, __half* scratch_d_h,
    int n_streams, cudaStream_t stream) {
    ctrl_down_nvfp4_batched_kernel<B_CONST><<<dim3(32, n_streams), 32, 0, stream>>>(cp, scratch_cat_h, scratch_d_h);
    ctrl_up_merge_nvfp4_batched_kernel<B_CONST><<<dim3(36, n_streams), 256, 0, stream>>>(
        cp, xp, scratch_msg, scratch_d_h, op);
}

void launch_controller_nvfp4_batched(
    float* const* xs, const float* saved, const float* prior_last,
    const float* cp_queries[3], const ControllerWeights* cws[3], const __nv_bfloat16* post_norms[3],
    float* const* outs, __half* const* outs_h, int n_streams,
    float* scratch_msg, __half* scratch_cat_h, __half* scratch_d_h, int B, cudaStream_t stream) {
    MutPtr3 xp{{xs[0], n_streams > 1 ? xs[1] : xs[0], n_streams > 2 ? xs[2] : xs[0]}};
    MutPtr3 op{{outs[0], n_streams > 1 ? outs[1] : outs[0], n_streams > 2 ? outs[2] : outs[0]}};
    HalfPtr3 ohp{{
        outs_h ? outs_h[0] : nullptr,
        (outs_h && n_streams > 1) ? outs_h[1] : nullptr,
        (outs_h && n_streams > 2) ? outs_h[2] : nullptr
    }};
    Ptr3 qp{{cp_queries[0], n_streams > 1 ? cp_queries[1] : cp_queries[0], n_streams > 2 ? cp_queries[2] : cp_queries[0]}};
    CtrlNVFP4Ptr3 cp;
    BF16Ptr3 np;
    for (int s = 0; s < 3; ++s) {
        int idx = (s < n_streams) ? s : 0;
        cp.down_d[s] = static_cast<const uint8_t*>(cws[idx]->down_nvfp4.data);
        cp.down_s[s] = cws[idx]->down_nvfp4.scales;
        cp.down_gs[s] = cws[idx]->down_nvfp4.global_scale;
        cp.forget_d[s] = static_cast<const uint8_t*>(cws[idx]->forget_nvfp4.data);
        cp.forget_s[s] = cws[idx]->forget_nvfp4.scales;
        cp.forget_gs[s] = cws[idx]->forget_nvfp4.global_scale;
        cp.gain_d[s] = static_cast<const uint8_t*>(cws[idx]->gain_nvfp4.data);
        cp.gain_s[s] = cws[idx]->gain_nvfp4.scales;
        cp.gain_gs[s] = cws[idx]->gain_nvfp4.global_scale;
        np.p[s] = post_norms[idx];
    }
    ctrl_prep_cat_h_batched_kernel<<<dim3(B, n_streams), 256, 0, stream>>>(
        xp, saved, prior_last, qp, scratch_msg, scratch_cat_h, B);
    switch (B) {
        case 1: dispatch_ctrl_nvfp4_b<1>(cp, xp, op, scratch_msg, scratch_cat_h, scratch_d_h, n_streams, stream); break;
        case 2: dispatch_ctrl_nvfp4_b<2>(cp, xp, op, scratch_msg, scratch_cat_h, scratch_d_h, n_streams, stream); break;
        case 3: dispatch_ctrl_nvfp4_b<3>(cp, xp, op, scratch_msg, scratch_cat_h, scratch_d_h, n_streams, stream); break;
        case 4: dispatch_ctrl_nvfp4_b<4>(cp, xp, op, scratch_msg, scratch_cat_h, scratch_d_h, n_streams, stream); break;
        case 5: dispatch_ctrl_nvfp4_b<5>(cp, xp, op, scratch_msg, scratch_cat_h, scratch_d_h, n_streams, stream); break;
        case 6: dispatch_ctrl_nvfp4_b<6>(cp, xp, op, scratch_msg, scratch_cat_h, scratch_d_h, n_streams, stream); break;
        case 7: dispatch_ctrl_nvfp4_b<7>(cp, xp, op, scratch_msg, scratch_cat_h, scratch_d_h, n_streams, stream); break;
        case 8: dispatch_ctrl_nvfp4_b<8>(cp, xp, op, scratch_msg, scratch_cat_h, scratch_d_h, n_streams, stream); break;
    }
    batched_rmsnorm_kernel<<<dim3(B, n_streams), 256, 0, stream>>>(op, ohp, np);
}

void launch_controller_mega_nvfp4(
    float* const* xs, const float* saved, const float* prior_last,
    const float* cp_queries[3], const ControllerWeights* cws[3], const __nv_bfloat16* post_norms[3],
    float* const* outs, __half* const* outs_h, int n_streams, int B, cudaStream_t stream) {
    MutPtr3 xp{{xs[0], n_streams > 1 ? xs[1] : xs[0], n_streams > 2 ? xs[2] : xs[0]}};
    MutPtr3 op{{outs[0], n_streams > 1 ? outs[1] : outs[0], n_streams > 2 ? outs[2] : outs[0]}};
    HalfPtr3 ohp{{
        outs_h ? outs_h[0] : nullptr,
        (outs_h && n_streams > 1) ? outs_h[1] : nullptr,
        (outs_h && n_streams > 2) ? outs_h[2] : nullptr
    }};
    Ptr3 qp{{cp_queries[0], n_streams > 1 ? cp_queries[1] : cp_queries[0], n_streams > 2 ? cp_queries[2] : cp_queries[0]}};
    CtrlNVFP4Ptr3 cp;
    BF16Ptr3 np;
    for (int s = 0; s < 3; ++s) {
        int idx = (s < n_streams) ? s : 0;
        cp.down_d[s] = static_cast<const uint8_t*>(cws[idx]->down_nvfp4.data);
        cp.down_s[s] = cws[idx]->down_nvfp4.scales;
        cp.down_gs[s] = cws[idx]->down_nvfp4.global_scale;
        cp.forget_d[s] = static_cast<const uint8_t*>(cws[idx]->forget_nvfp4.data);
        cp.forget_s[s] = cws[idx]->forget_nvfp4.scales;
        cp.forget_gs[s] = cws[idx]->forget_nvfp4.global_scale;
        cp.gain_d[s] = static_cast<const uint8_t*>(cws[idx]->gain_nvfp4.data);
        cp.gain_s[s] = cws[idx]->gain_nvfp4.scales;
        cp.gain_gs[s] = cws[idx]->gain_nvfp4.global_scale;
        np.p[s] = post_norms[idx];
    }
    mega_controller_nvfp4_kernel<<<dim3(B, n_streams), 1024, 0, stream>>>(
        xp, saved, prior_last, qp, cp, np, op, ohp);
}

void launch_mega_global_sconv_prep_store(
    float* qrgs, const float* k_proj, const float* v_proj, const float* in_k, const float* in_v,
    const __nv_bfloat16* k_sconv_w, const __nv_bfloat16* v_sconv_w,
    const __nv_bfloat16* k_sconv_proj, const __nv_bfloat16* v_sconv_proj,
    float* k_ring, float* v_ring,
    const __nv_bfloat16* q_clip_scale, const __nv_bfloat16* rel_clip_scale,
    void* k_cache, void* v_cache, __half* k_scale, __half* v_scale, KVFormat kv_fmt,
    int kv_cap, int B, const int32_t* d_pos, cudaStream_t stream) {
    if (kv_fmt == KVFormat::FP8) {
        launch_pdl(mega_global_sconv_prep_store_kernel<1>, dim3(64, B), 128, stream,
            qrgs, k_proj, v_proj, in_k, in_v, k_sconv_w, v_sconv_w, k_sconv_proj, v_sconv_proj,
            k_ring, v_ring, q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    } else if (kv_fmt == KVFormat::Q4) {
        launch_pdl(mega_global_sconv_prep_store_kernel<2>, dim3(64, B), 128, stream,
            qrgs, k_proj, v_proj, in_k, in_v, k_sconv_w, v_sconv_w, k_sconv_proj, v_sconv_proj,
            k_ring, v_ring, q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    } else {
        launch_pdl(mega_global_sconv_prep_store_kernel<0>, dim3(64, B), 128, stream,
            qrgs, k_proj, v_proj, in_k, in_v, k_sconv_w, v_sconv_w, k_sconv_proj, v_sconv_proj,
            k_ring, v_ring, q_clip_scale, rel_clip_scale, k_cache, v_cache, k_scale, v_scale, kv_cap, d_pos);
    }
}

void launch_mega_mlp_post_next_depth_norm(
    const float* moe_down_or_slots, bool moe_is_slotted,
    const __nv_bfloat16* prev_mlp_sconv_w, float* prev_mlp_ring,
    const float* residual, float* hidden_out, float* pre_tap_out, float* post_tap_out,
    float* partial_out, const float* blocks, int n_blocks, const float* queries,
    const __nv_bfloat16* norm0, const __nv_bfloat16* norm1, const __nv_bfloat16* norm2,
    float* out0, float* out1, float* out2, __half* out_h0, __half* out_h1, __half* out_h2,
    int n_out, int B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p,
    const float* loopcd_h1, const float* loopcd_h2, const LoopCDConfig* loopcd_cfg,
    cudaStream_t stream) {
    BF16Ptr3 np{{norm0, norm1, norm2}};
    MutPtr3 op{{out0, out1, out2}};
    HalfPtr3 ohp{{out_h0, out_h1, out_h2}};
    launch_pdl(mega_mlp_post_next_depth_norm_kernel, dim3(B, n_out), 576, stream,
        moe_down_or_slots, moe_is_slotted, prev_mlp_sconv_w, prev_mlp_ring, residual, hidden_out,
        pre_tap_out, post_tap_out, partial_out, blocks, n_blocks, queries, np, op, ohp, d_pos,
        abl_r, abl_d, abl_p, loopcd_h1, loopcd_h2, loopcd_cfg);
}

void launch_mega_local_sconv_prep_store(
    float* qrgs, const float* k_proj, const float* v_proj,
    const __nv_bfloat16* k_sconv_w, const __nv_bfloat16* v_sconv_w,
    float* k_ring, float* v_ring,
    const __nv_bfloat16* q_clip_scale, const __nv_bfloat16* rel_clip_scale,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int kv_cap, int B, const int32_t* d_pos, cudaStream_t stream) {
    launch_pdl(mega_local_sconv_prep_store_kernel, dim3(64, B), 128, stream,
        qrgs, k_proj, v_proj, k_sconv_w, v_sconv_w, k_ring, v_ring,
        q_clip_scale, rel_clip_scale, k_cache, v_cache, kv_cap, d_pos);
}

void launch_mega_attn_post_mlp_in(
    const float* o_proj, const __nv_bfloat16* attn_sconv_w, float* attn_ring,
    const float* hidden, const float* partial_prev, float* blocks, int block_count_before, bool is_boundary,
    float* residual_out, const float* mlp_query, const __nv_bfloat16* mlp_norm_w,
    float* mlp_out, __half* mlp_out_h, int B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p, cudaStream_t stream) {
    launch_pdl(mega_attn_post_mlp_in_kernel, dim3(B), 576, stream,
        o_proj, attn_sconv_w, attn_ring, hidden, partial_prev, blocks, block_count_before, is_boundary,
        residual_out, mlp_query, mlp_norm_w, mlp_out, mlp_out_h, d_pos, abl_r, abl_d, abl_p);
}

void launch_mega_mlp_post_res(
    const float* moe_down_or_slots, bool moe_is_slotted, const __nv_bfloat16* mlp_sconv_w, float* mlp_ring,
    const float* residual, float* hidden_out, float* loopcd_save_out, int B, const int32_t* d_pos,
    const float* abl_r, const float* abl_d, const float* abl_p, cudaStream_t stream) {
    launch_pdl(mega_mlp_post_res_kernel, dim3(B), 576, stream,
        moe_down_or_slots, moe_is_slotted, mlp_sconv_w, mlp_ring, residual, hidden_out, loopcd_save_out, d_pos,
        abl_r, abl_d, abl_p);
}

void launch_loopcd_hidden_apply(
    float* hidden, const float* ref_hidden, float omega, int B, cudaStream_t stream) {
    if (B <= 0 || omega == 0.0f) return;
    loopcd_hidden_apply_kernel<<<B, 576, 0, stream>>>(hidden, ref_hidden, omega);
}

void launch_vec_add(const float* a, const float* b, float* out, int n, cudaStream_t stream) {
    vec_add_kernel<<<(n + 255) / 256, 256, 0, stream>>>(a, b, out, n);
}

void launch_copy_block(const float* src, float* blocks, int max_blocks, int slot_idx, int B, cudaStream_t stream) {
    int total = B * HIDDEN;
    copy_block_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, blocks, max_blocks, slot_idx, B);
}

void launch_extract_block(const float* blocks, int max_blocks, int slot_idx, float* dst, int B, cudaStream_t stream) {
    int total = B * HIDDEN;
    extract_block_kernel<<<(total + 255) / 256, 256, 0, stream>>>(blocks, max_blocks, slot_idx, dst, B);
}

} // namespace squid
