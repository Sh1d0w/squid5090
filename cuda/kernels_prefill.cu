// Fast Batched Prefill Kernels (B <= 256):
// 1. Coalesced NVFP4/FP8/BF16 -> FP16 dequant + cuBLAS FP16 Tensor Core GEMM for dense, router & controllers.
// 2. Blackwell mma.sync.aligned.m16n8k16 FP16 Tensor Core Multi-Token MoE (w13 softcapped SwiGLU + w2 zero-atomic).
// 3. 1-launch batched B<=256 causal sconv + MoE slot reduction + residual + depth-attn controller.

#include "config.hpp"
#include "kernels.hpp"
#include "kernels_common.cuh"
#include "weights.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace squid {

namespace {

static cublasHandle_t g_cublas = nullptr;

static cublasHandle_t get_cublas(cudaStream_t stream) {
    if (!g_cublas) {
        cublasStatus_t st = cublasCreate(&g_cublas);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed: %d\n", (int)st);
            std::exit(1);
        }
        cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);
    }
    cublasSetStream(g_cublas, stream);
    return g_cublas;
}

__global__ void p_f32_to_f16_kernel(const float* __restrict__ src, __half* __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void p_dequant_nvfp4_to_f16_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    float global_scale,
    __half* __restrict__ out_f16,
    size_t total_blocks) {
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks) return;
    uint2 w8 = __ldg(reinterpret_cast<const uint2*>(w_data) + bidx);
    float sc = cvt_e4m3_to_float(__ldg(w_scales + bidx)) * global_scale;
    __half2 hsc = __float2half2_rn(sc);
    __half2 h[8];
    cvt_e2m1x8_to_half2x4(w8.x, h[0], h[1], h[2], h[3]);
    cvt_e2m1x8_to_half2x4(w8.y, h[4], h[5], h[6], h[7]);
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = __hmul2(h[i], hsc);
    uint4* dst4 = reinterpret_cast<uint4*>(out_f16 + bidx * 16);
    dst4[0] = *reinterpret_cast<const uint4*>(&h[0]);
    dst4[1] = *reinterpret_cast<const uint4*>(&h[4]);
}

__global__ void p_dequant_fp8_to_f16_kernel(
    const uint8_t* __restrict__ w_data,
    const float* __restrict__ row_scales,
    __half* __restrict__ out_f16,
    int N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * (K >> 1);
    if (idx >= total) return;
    int row = (int)(idx / (K >> 1));
    float sc = __ldg(row_scales + row);
    uint16_t raw2 = __ldg(reinterpret_cast<const uint16_t*>(w_data) + idx);
    __half2 h2 = cvt_e4m3x2_to_half2(raw2);
    reinterpret_cast<__half2*>(out_f16)[idx] = __hmul2(h2, __float2half2_rn(sc));
}

__global__ void p_dequant_bf16_to_f16_kernel(
    const __nv_bfloat16* __restrict__ w_data,
    __half* __restrict__ out_f16,
    size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out_f16[i] = __float2half(__bfloat162float(w_data[i]));
}

// 1-launch batched B<=256 causal hidden-dim sconv + optional 8-slot MoE reduction + residual add + optional block/tap commit!
__global__ void p_sconv_post_res_kernel(
    const float* __restrict__ src_or_slots,
    int is_slotted,
    const __nv_bfloat16* __restrict__ sconv_w,
    float* __restrict__ ring,
    const float* __restrict__ residual,
    float* __restrict__ out_res,
    float* __restrict__ pre_tap_out, // optional (e.g. prelude)
    float* __restrict__ blocks,      // optional: commit partial into blocks[b, block_idx]
    const float* __restrict__ partial,
    int block_idx,
    int is_boundary,
    int B,
    const int32_t* __restrict__ d_pos) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= HIDDEN) return;

    int pos0 = *d_pos;
    float wc0 = __bfloat162float(sconv_w[c * 4 + 0]);
    float wc1 = __bfloat162float(sconv_w[c * 4 + 1]);
    float wc2 = __bfloat162float(sconv_w[c * 4 + 2]);
    float wc3 = __bfloat162float(sconv_w[c * 4 + 3]);

    float h0 = (pos0 >= 3) ? ring[((pos0 - 3) & 15) * HIDDEN + c] : 0.0f;
    float h1 = (pos0 >= 2) ? ring[((pos0 - 2) & 15) * HIDDEN + c] : 0.0f;
    float h2 = (pos0 >= 1) ? ring[((pos0 - 1) & 15) * HIDDEN + c] : 0.0f;

    for (int b = 0; b < B; ++b) {
        float cur;
        if (is_slotted) {
            const float* sb = src_or_slots + (size_t)b * TOP_K * HIDDEN + c;
            cur = sb[0 * HIDDEN] + sb[1 * HIDDEN] + sb[2 * HIDDEN] + sb[3 * HIDDEN]
                + sb[4 * HIDDEN] + sb[5 * HIDDEN] + sb[6 * HIDDEN] + sb[7 * HIDDEN];
        } else {
            cur = src_or_slots[(size_t)b * HIDDEN + c];
        }
        int p = pos0 + b;
        // Only write final 16 slots back to HBM ring (earlier slots are strictly overwritten!)
        if (b >= B - 16) {
            ring[(p & 15) * HIDDEN + c] = cur;
        }
        float conv = (wc0 * h0 + wc1 * h1 + wc2 * h2 + wc3 * cur + cur) * INV_SQRT_5;
        h0 = h1; h1 = h2; h2 = cur;

        float res_v;
        if (is_boundary) {
            if (blocks && partial) {
                blocks[((size_t)b * 4 + block_idx) * HIDDEN + c] = partial[(size_t)b * HIDDEN + c];
            }
            res_v = conv;
        } else {
            res_v = residual[(size_t)b * HIDDEN + c] + conv;
        }
        out_res[(size_t)b * HIDDEN + c] = res_v;
        if (pre_tap_out) pre_tap_out[(size_t)b * HIDDEN + c] = res_v;
    }
}

// Scatter 8-token tap slots [s*4 + tap_slot, 8, HIDDEN] from contiguous [B, HIDDEN] in 1 launch!
__global__ void p_copy_taps_chunked_kernel(
    const float* __restrict__ src, // [B, HIDDEN]
    float* __restrict__ dst_taps,  // [n_sub, 4, 8, HIDDEN]
    int tap_slot,
    int B) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)B * HIDDEN;
    if (idx >= total) return;
    int b = (int)(idx / HIDDEN);
    int c = (int)(idx % HIDDEN);
    int s = b >> 3;
    int sb = b & 7;
    dst_taps[((size_t)(s * 4 + tap_slot) * 8 + sb) * HIDDEN + c] = src[idx];
}

// Batched B<=256 Controller Prep (1 stream, grid = B, block = 256):
// Computes msg = depth_attn(saved, prior_last, cp_q), cat_h = [rmsnorm(x), rmsnorm(msg)] in FP16!
__global__ void __launch_bounds__(256)
p_ctrl_prep_h_kernel(
    const float* __restrict__ x,
    const float* __restrict__ saved,
    const float* __restrict__ prior_last,
    const float* __restrict__ cp_q,
    float* __restrict__ out_msg,   // [B, HIDDEN]
    __half* __restrict__ out_cat_h // [B, 2*HIDDEN]
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    const float* sv = saved + (size_t)b * HIDDEN;
    const float* pl = prior_last + (size_t)b * HIDDEN;
    const float* xb = x + (size_t)b * HIDDEN;

    float d0 = 0.0f, d1 = 0.0f, xsq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 256) {
        float qv = cp_q[d];
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
        float l0 = t0 * INV_SQRT_HIDDEN;
        float l1 = t1 * INV_SQRT_HIDDEN;
        float mx = fmaxf(l0, l1);
        float e0 = expf(l0 - mx), e1 = expf(l1 - mx);
        float inv = 1.0f / (e0 + e1);
        w0 = e0 * inv;
        w1 = e1 * inv;
        inv_x = rsqrtf(tx / float(HIDDEN) + RMS_EPS);
    }
    __syncthreads();

    float ww0 = w0, ww1 = w1, ix = inv_x;
    float* msg_b = out_msg + (size_t)b * HIDDEN;
    __half* cat_b = out_cat_h + (size_t)b * (2 * HIDDEN);

    float msq = 0.0f;
    for (int d = tid; d < HIDDEN; d += 256) {
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
    for (int d = tid; d < HIDDEN; d += 256) {
        cat_b[HIDDEN + d] = __float2half(msg_b[d] * im);
    }
}

// Batched B<=256 Controller Finish (grid = B, block = 256):
// merged = (1 - tanh(forget)) * x + tanh(gain) * msg -> RMSNorm * post_norm -> out (f32) & out_h (f16)
__global__ void __launch_bounds__(256)
p_ctrl_finish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ msg,
    const float* __restrict__ forget,
    const float* __restrict__ gain,
    const __nv_bfloat16* __restrict__ post_norm,
    float* __restrict__ out,
    __half* __restrict__ out_h) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    const float* xb = x + (size_t)b * HIDDEN;
    const float* mb = msg + (size_t)b * HIDDEN;
    const float* fb = forget + (size_t)b * HIDDEN;
    const float* gb = gain + (size_t)b * HIDDEN;
    float* ob = out + (size_t)b * HIDDEN;
    __half* ohb = out_h ? (out_h + (size_t)b * HIDDEN) : nullptr;

    float vals[9]; // 2304 / 256 = 9 exact!
    float sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int d = tid + i * 256;
        float v = (1.0f - fast_tanh(fb[d])) * xb[d] + fast_tanh(gb[d]) * mb[d];
        vals[i] = v;
        sq = fmaf(v, v, sq);
    }
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
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        int d = tid + i * 256;
        float v = vals[i] * ir * __bfloat162float(post_norm[d]);
        ob[d] = v;
        if (ohb) ohb[d] = __float2half(v);
    }
}

// Per-token top-8 selection (1 warp per token, grid = B) — identical tie-breaking to kernels_inkling.cu!
__global__ void p_route_topk_only_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ router_bias,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights) {
    int b = blockIdx.x;
    int lane = threadIdx.x;

    float sc[4], bi[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int e = lane + 32 * i;
        float raw = fast_sigmoid(logits[(size_t)b * NUM_EXPERTS + e]);
        sc[i] = raw;
        bi[i] = raw + router_bias[e];
    }

    int winners[TOP_K];
    float chosen[TOP_K];
    for (int k = 0; k < TOP_K; ++k) {
        float best = -1e30f;
        int best_e = NUM_EXPERTS;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int e = lane + 32 * i;
            if (bi[i] > best || (bi[i] == best && e < best_e)) {
                best = bi[i];
                best_e = e;
            }
        }
        float top = warp_max(best);
        int cand = (best == top) ? best_e : NUM_EXPERTS;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            cand = min(cand, __shfl_xor_sync(0xffffffffu, cand, off));
        }
        float sel = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (lane + 32 * i == cand) {
                sel = sc[i];
                bi[i] = -1e30f;
            }
        }
        chosen[k] = warp_sum(sel);
        winners[k] = cand;
    }

    if (lane == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) total += chosen[k];
        float inv = ROUTE_SCALE / (total + 1e-20f);
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            out_ids[b * TOP_K + k] = winners[k];
            out_weights[b * TOP_K + k] = chosen[k] * inv;
        }
    }
}

__global__ void p_sort_experts_kernel(
    const int32_t* __restrict__ route_ids,
    const float* __restrict__ route_w,
    int B,
    int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ sorted_tok,
    int32_t* __restrict__ sorted_slot,
    float* __restrict__ sorted_w) {
    __shared__ int32_t counts[NUM_EXPERTS];
    __shared__ int32_t cursors[NUM_EXPERTS];
    int tid = threadIdx.x;
    counts[tid] = 0;
    __syncthreads();

    int total = B * TOP_K;
    for (int idx = tid; idx < total; idx += NUM_EXPERTS) {
        atomicAdd(&counts[route_ids[idx]], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int sum = 0;
        int nt = 0;
        int32_t* tile_exp = expert_offsets + 130;
        int32_t* tile_st  = expert_offsets + 386;
        int32_t* tile_en  = expert_offsets + 642;
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            expert_offsets[e] = sum;
            cursors[e] = sum;
            int cnt = counts[e];
            for (int off = 0; off < cnt; off += 32) {
                tile_exp[nt] = e;
                tile_st[nt]  = sum + off;
                tile_en[nt]  = sum + min(off + 32, cnt);
                nt++;
            }
            sum += cnt;
        }
        expert_offsets[NUM_EXPERTS] = sum;
        expert_offsets[129] = nt;
    }
    __syncthreads();

    for (int idx = tid; idx < total; idx += NUM_EXPERTS) {
        int e = route_ids[idx];
        int dst = atomicAdd(&cursors[e], 1);
        sorted_tok[dst] = idx >> 3;
        sorted_slot[dst] = idx & 7;
        sorted_w[dst] = route_w[idx];
    }
}

// Helper: pick 2 consecutive bytes (16 bits) from uint2 (8 bytes) by tid4 (0..3)
__device__ __forceinline__ uint16_t pick_u16_from_u2(uint2 v, int tid4) {
    uint32_t w = (tid4 & 2) ? v.y : v.x;
    return static_cast<uint16_t>((tid4 & 1) ? (w >> 16) : (w & 0xFFFFu));
}

__device__ __forceinline__ void mma_m16n8k16(
    float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Unpack 1 of 4 coalesced NVFP4 blocks (m = 0..3) from 4-thread-coalesced uint2 + pre-decoded half2 scales
__device__ __forceinline__ void nvfp4_shfl_a_frag(
    uint2 w0_raw, uint2 w1_raw, __half2 sc0, __half2 sc1,
    int grp, int tid4, int m,
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
    int src_lane = (grp << 2) + m;
    uint32_t w0_lo = __shfl_sync(0xffffffffu, w0_raw.x, src_lane);
    uint32_t w0_hi = __shfl_sync(0xffffffffu, w0_raw.y, src_lane);
    uint32_t w1_lo = __shfl_sync(0xffffffffu, w1_raw.x, src_lane);
    uint32_t w1_hi = __shfl_sync(0xffffffffu, w1_raw.y, src_lane);
    int sh = tid4 << 3;
    __half2 h0 = __hmul2(cvt_e2m1x2_to_half2((w0_lo >> sh) & 0xFFu), sc0);
    __half2 h1 = __hmul2(cvt_e2m1x2_to_half2((w1_lo >> sh) & 0xFFu), sc1);
    __half2 h2 = __hmul2(cvt_e2m1x2_to_half2((w0_hi >> sh) & 0xFFu), sc0);
    __half2 h3 = __hmul2(cvt_e2m1x2_to_half2((w1_hi >> sh) & 0xFFu), sc1);
    a0 = *reinterpret_cast<const uint32_t*>(&h0);
    a1 = *reinterpret_cast<const uint32_t*>(&h1);
    a2 = *reinterpret_cast<const uint32_t*>(&h2);
    a3 = *reinterpret_cast<const uint32_t*>(&h3);
}

__device__ __forceinline__ __half2 pick_e4m3_scale_h2(uint32_t s_raw, int m) {
    __half2 pair = cvt_e4m3x2_to_half2(static_cast<uint16_t>((m & 2) ? (s_raw >> 16) : (s_raw & 0xFFFFu)));
    return (m & 1) ? __high2half2(pair) : __low2half2(pair);
}

// Unpack 1 of 2 coalesced MXFP8 half-blocks (m = 0..1, each K=16) from 4-thread-coalesced uint2 (32B total)
__device__ __forceinline__ void mxfp8_shfl_a_frag(
    uint2 w0_raw, uint2 w1_raw, __half2 sc0, __half2 sc1,
    int grp, int tid4, int m,
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
    int l_lo = (grp << 2) + (m << 1);
    int l_hi = l_lo + 1;
    uint2 v0_lo = make_uint2(__shfl_sync(0xffffffffu, w0_raw.x, l_lo), __shfl_sync(0xffffffffu, w0_raw.y, l_lo));
    uint2 v0_hi = make_uint2(__shfl_sync(0xffffffffu, w0_raw.x, l_hi), __shfl_sync(0xffffffffu, w0_raw.y, l_hi));
    uint2 v1_lo = make_uint2(__shfl_sync(0xffffffffu, w1_raw.x, l_lo), __shfl_sync(0xffffffffu, w1_raw.y, l_lo));
    uint2 v1_hi = make_uint2(__shfl_sync(0xffffffffu, w1_raw.x, l_hi), __shfl_sync(0xffffffffu, w1_raw.y, l_hi));
    __half2 h0 = __hmul2(cvt_e4m3x2_to_half2(pick_u16_from_u2(v0_lo, tid4)), sc0);
    __half2 h1 = __hmul2(cvt_e4m3x2_to_half2(pick_u16_from_u2(v1_lo, tid4)), sc1);
    __half2 h2 = __hmul2(cvt_e4m3x2_to_half2(pick_u16_from_u2(v0_hi, tid4)), sc0);
    __half2 h3 = __hmul2(cvt_e4m3x2_to_half2(pick_u16_from_u2(v1_hi, tid4)), sc1);
    a0 = *reinterpret_cast<const uint32_t*>(&h0);
    a1 = *reinterpret_cast<const uint32_t*>(&h1);
    a2 = *reinterpret_cast<const uint32_t*>(&h2);
    a3 = *reinterpret_cast<const uint32_t*>(&h3);
}

// BLACKWELL FP16 TENSOR CORE Multi-Token NVFP4 MoE Gate+Up
// 32-byte coalesced HBM loads (K=64/iter, 100% sector efficiency) + N=32 tokens/pass!
// Grid: (INTERMEDIATE / 64 = 18, NUM_EXPERTS = 128), Block: 128 threads (4 warps x 16 rows)
__global__ void __launch_bounds__(128)
p_moe_gate_up_nvfp4_tc_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ x_h,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_slot,
    __half* __restrict__ out_act_h) {
    int tile_id = blockIdx.y;
    if (tile_id >= expert_offsets[129]) return;
    int expert = expert_offsets[130 + tile_id];
    int t0 = expert_offsets[386 + tile_id];
    int end = expert_offsets[642 + tile_id];
    int cnt = end - t0;

    int tid = threadIdx.x;
    int warp = tid >> 5;   // 0..3
    int lane = tid & 31;
    int grp = lane >> 2;   // 0..7
    int tid4 = lane & 3;   // 0..3

    int j_base = blockIdx.x * 64 + warp * 16;
    int j0 = j_base + grp;
    int j1 = j_base + grp + 8;

    constexpr int W_STRIDE = HIDDEN >> 1;  // 1152 bytes = 36 * 32B
    constexpr int S_STRIDE = HIDDEN >> 4;  // 144 bytes  = 36 * 4B
    constexpr int K_ITERS = HIDDEN >> 6;   // 36
    size_t rg0 = ((size_t)expert * INTERMEDIATE + j0) * 2;
    size_t ru0 = rg0 + 1;
    size_t rg1 = ((size_t)expert * INTERMEDIATE + j1) * 2;
    size_t ru1 = rg1 + 1;
    const uint2* wg0 = reinterpret_cast<const uint2*>(w_data + rg0 * W_STRIDE);
    const uint2* wu0 = reinterpret_cast<const uint2*>(w_data + ru0 * W_STRIDE);
    const uint2* wg1 = reinterpret_cast<const uint2*>(w_data + rg1 * W_STRIDE);
    const uint2* wu1 = reinterpret_cast<const uint2*>(w_data + ru1 * W_STRIDE);
    const uint32_t* sg0 = reinterpret_cast<const uint32_t*>(w_scales + rg0 * S_STRIDE);
    const uint32_t* su0 = reinterpret_cast<const uint32_t*>(w_scales + ru0 * S_STRIDE);
    const uint32_t* sg1 = reinterpret_cast<const uint32_t*>(w_scales + rg1 * S_STRIDE);
    const uint32_t* su1 = reinterpret_cast<const uint32_t*>(w_scales + ru1 * S_STRIDE);
    float es = exp_scales[expert];

    __shared__ __half s_x[32][72]; // 4,608 B SMEM (stride 144B = 16B aligned)
    __shared__ int s_tok[32];
    __shared__ int s_slot[32];

    int load_t0 = tid >> 3;        // 0..15
    int load_t1 = 16 + load_t0;    // 16..31
    int load_c = (tid & 7) << 3;   // 0, 8, ..., 56 (8 halfs = uint4)

    if (tid < 32) {
        s_tok[tid] = (tid < cnt) ? sorted_tok[t0 + tid] : 0;
        s_slot[tid] = (tid < cnt) ? sorted_slot[t0 + tid] : 0;
    }
    __syncthreads();

    float dg0[4] = {0, 0, 0, 0}, dg1[4] = {0, 0, 0, 0}, dg2[4] = {0, 0, 0, 0}, dg3[4] = {0, 0, 0, 0};
    float du0[4] = {0, 0, 0, 0}, du1[4] = {0, 0, 0, 0}, du2[4] = {0, 0, 0, 0}, du3[4] = {0, 0, 0, 0};
    int my_tok0 = s_tok[load_t0];
    int my_tok1 = s_tok[load_t1];

    for (int blk4 = 0; blk4 < K_ITERS; ++blk4) {
        int koff = blk4 << 6;
        // Issue 32-byte coalesced weight loads in parallel with s_x loads before __syncthreads()!
        int woff = (blk4 << 2) + tid4;
        uint2 wg0_r = __ldg(wg0 + woff), wg1_r = __ldg(wg1 + woff);
        uint2 wu0_r = __ldg(wu0 + woff), wu1_r = __ldg(wu1 + woff);
        uint32_t sg0_r = __ldg(sg0 + blk4), sg1_r = __ldg(sg1 + blk4);
        uint32_t su0_r = __ldg(su0 + blk4), su1_r = __ldg(su1 + blk4);

        if (load_t0 < cnt) {
            *reinterpret_cast<uint4*>(&s_x[load_t0][load_c]) =
                __ldg(reinterpret_cast<const uint4*>(x_h + (size_t)my_tok0 * HIDDEN + koff + load_c));
        } else {
            *reinterpret_cast<uint4*>(&s_x[load_t0][load_c]) = make_uint4(0, 0, 0, 0);
        }
        if (load_t1 < cnt) {
            *reinterpret_cast<uint4*>(&s_x[load_t1][load_c]) =
                __ldg(reinterpret_cast<const uint4*>(x_h + (size_t)my_tok1 * HIDDEN + koff + load_c));
        } else if (cnt > 16) {
            *reinterpret_cast<uint4*>(&s_x[load_t1][load_c]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int km = (m << 4) + (tid4 << 1);
            uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(&s_x[grp][km]);
            uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(&s_x[grp][km + 8]);
            uint32_t b0_1 = 0, b1_1 = 0, b0_2 = 0, b1_2 = 0, b0_3 = 0, b1_3 = 0;
            if (cnt > 8) {
                b0_1 = *reinterpret_cast<const uint32_t*>(&s_x[8 + grp][km]);
                b1_1 = *reinterpret_cast<const uint32_t*>(&s_x[8 + grp][km + 8]);
            }
            if (cnt > 16) {
                b0_2 = *reinterpret_cast<const uint32_t*>(&s_x[16 + grp][km]);
                b1_2 = *reinterpret_cast<const uint32_t*>(&s_x[16 + grp][km + 8]);
            }
            if (cnt > 24) {
                b0_3 = *reinterpret_cast<const uint32_t*>(&s_x[24 + grp][km]);
                b1_3 = *reinterpret_cast<const uint32_t*>(&s_x[24 + grp][km + 8]);
            }

            uint32_t ag0, ag1, ag2, ag3;
            nvfp4_shfl_a_frag(wg0_r, wg1_r, pick_e4m3_scale_h2(sg0_r, m), pick_e4m3_scale_h2(sg1_r, m), grp, tid4, m, ag0, ag1, ag2, ag3);
            mma_m16n8k16(dg0, ag0, ag1, ag2, ag3, b0_0, b1_0);
            if (cnt > 8)  mma_m16n8k16(dg1, ag0, ag1, ag2, ag3, b0_1, b1_1);
            if (cnt > 16) mma_m16n8k16(dg2, ag0, ag1, ag2, ag3, b0_2, b1_2);
            if (cnt > 24) mma_m16n8k16(dg3, ag0, ag1, ag2, ag3, b0_3, b1_3);

            uint32_t au0, au1, au2, au3;
            nvfp4_shfl_a_frag(wu0_r, wu1_r, pick_e4m3_scale_h2(su0_r, m), pick_e4m3_scale_h2(su1_r, m), grp, tid4, m, au0, au1, au2, au3);
            mma_m16n8k16(du0, au0, au1, au2, au3, b0_0, b1_0);
            if (cnt > 8)  mma_m16n8k16(du1, au0, au1, au2, au3, b0_1, b1_1);
            if (cnt > 16) mma_m16n8k16(du2, au0, au1, au2, au3, b0_2, b1_2);
            if (cnt > 24) mma_m16n8k16(du3, au0, au1, au2, au3, b0_3, b1_3);
        }
        __syncthreads();
    }

    auto write_act = [&](int local_t, int j, float gv, float uv) {
        if (local_t >= cnt) return;
        gv *= es;
        uv *= es;
        float g_cap = fast_tanh(gv * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
        float u_cap = fast_tanh(uv * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
        int tok = s_tok[local_t];
        int slot = s_slot[local_t];
        out_act_h[((size_t)tok * TOP_K + slot) * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(gv)) * u_cap);
    };

    int n0 = tid4 * 2;
    write_act(n0,     j0, dg0[0], du0[0]);
    write_act(n0 + 1, j0, dg0[1], du0[1]);
    write_act(n0,     j1, dg0[2], du0[2]);
    write_act(n0 + 1, j1, dg0[3], du0[3]);
    if (cnt > 8) {
        write_act(8 + n0,     j0, dg1[0], du1[0]);
        write_act(8 + n0 + 1, j0, dg1[1], du1[1]);
        write_act(8 + n0,     j1, dg1[2], du1[2]);
        write_act(8 + n0 + 1, j1, dg1[3], du1[3]);
    }
    if (cnt > 16) {
        write_act(16 + n0,     j0, dg2[0], du2[0]);
        write_act(16 + n0 + 1, j0, dg2[1], du2[1]);
        write_act(16 + n0,     j1, dg2[2], du2[2]);
        write_act(16 + n0 + 1, j1, dg2[3], du2[3]);
    }
    if (cnt > 24) {
        write_act(24 + n0,     j0, dg3[0], du3[0]);
        write_act(24 + n0 + 1, j0, dg3[1], du3[1]);
        write_act(24 + n0,     j1, dg3[2], du3[2]);
        write_act(24 + n0 + 1, j1, dg3[3], du3[3]);
    }
}

// BLACKWELL FP16 TENSOR CORE Multi-Token NVFP4 MoE Down
// Flat tile scheduler (zero tail stragglers!) + 32-byte coalesced HBM loads (K=64/iter) + N=32!
// Grid: (HIDDEN / 64 = 36, 256), Block: 128 threads (4 warps x 16 rows)
__global__ void __launch_bounds__(128)
p_moe_down_nvfp4_tc_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const float* __restrict__ exp_scales,
    const __half* __restrict__ act_h,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_slot,
    const float* __restrict__ sorted_w,
    float* __restrict__ out_slots) {
    int tile_id = blockIdx.y;
    if (tile_id >= expert_offsets[129]) return;
    int expert = expert_offsets[130 + tile_id];
    int t0 = expert_offsets[386 + tile_id];
    int end = expert_offsets[642 + tile_id];
    int cnt = end - t0;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp = lane >> 2;
    int tid4 = lane & 3;

    int r_base = blockIdx.x * 64 + warp * 16;
    int r0 = r_base + grp;
    int r1 = r_base + grp + 8;

    constexpr int W_STRIDE = INTERMEDIATE >> 1; // 576 bytes = 18 * 32B
    constexpr int S_STRIDE = INTERMEDIATE >> 4; // 72 bytes  = 18 * 4B
    constexpr int K_ITERS = INTERMEDIATE >> 6;  // 18
    size_t row0 = (size_t)expert * HIDDEN + r0;
    size_t row1 = (size_t)expert * HIDDEN + r1;
    const uint2* wr0 = reinterpret_cast<const uint2*>(w_data + row0 * W_STRIDE);
    const uint2* wr1 = reinterpret_cast<const uint2*>(w_data + row1 * W_STRIDE);
    const uint32_t* sr0 = reinterpret_cast<const uint32_t*>(w_scales + row0 * S_STRIDE);
    const uint32_t* sr1 = reinterpret_cast<const uint32_t*>(w_scales + row1 * S_STRIDE);
    float es = exp_scales[expert];

    __shared__ __half s_x[32][72];
    __shared__ int s_base[32];
    __shared__ float s_rw[32];

    int load_t0 = tid >> 3;
    int load_t1 = 16 + load_t0;
    int load_c = (tid & 7) << 3;

    if (tid < 32) {
        if (tid < cnt) {
            s_base[tid] = sorted_tok[t0 + tid] * TOP_K + sorted_slot[t0 + tid];
            s_rw[tid] = sorted_w[t0 + tid] * es;
        } else {
            s_base[tid] = 0;
            s_rw[tid] = 0.0f;
        }
    }
    __syncthreads();

    float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
    int my_base0 = s_base[load_t0];
    int my_base1 = s_base[load_t1];

    for (int blk4 = 0; blk4 < K_ITERS; ++blk4) {
        int koff = blk4 << 6;
        int woff = (blk4 << 2) + tid4;
        uint2 w0_r = __ldg(wr0 + woff), w1_r = __ldg(wr1 + woff);
        uint32_t s0_r = __ldg(sr0 + blk4), s1_r = __ldg(sr1 + blk4);

        if (load_t0 < cnt) {
            *reinterpret_cast<uint4*>(&s_x[load_t0][load_c]) =
                __ldg(reinterpret_cast<const uint4*>(act_h + (size_t)my_base0 * INTERMEDIATE + koff + load_c));
        } else {
            *reinterpret_cast<uint4*>(&s_x[load_t0][load_c]) = make_uint4(0, 0, 0, 0);
        }
        if (load_t1 < cnt) {
            *reinterpret_cast<uint4*>(&s_x[load_t1][load_c]) =
                __ldg(reinterpret_cast<const uint4*>(act_h + (size_t)my_base1 * INTERMEDIATE + koff + load_c));
        } else if (cnt > 16) {
            *reinterpret_cast<uint4*>(&s_x[load_t1][load_c]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            int km = (m << 4) + (tid4 << 1);
            uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(&s_x[grp][km]);
            uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(&s_x[grp][km + 8]);
            uint32_t a0, a1, a2, a3;
            nvfp4_shfl_a_frag(w0_r, w1_r, pick_e4m3_scale_h2(s0_r, m), pick_e4m3_scale_h2(s1_r, m), grp, tid4, m, a0, a1, a2, a3);
            mma_m16n8k16(d0, a0, a1, a2, a3, b0_0, b1_0);
            if (cnt > 8) {
                uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(&s_x[8 + grp][km]);
                uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(&s_x[8 + grp][km + 8]);
                mma_m16n8k16(d1, a0, a1, a2, a3, b0_1, b1_1);
            }
            if (cnt > 16) {
                uint32_t b0_2 = *reinterpret_cast<const uint32_t*>(&s_x[16 + grp][km]);
                uint32_t b1_2 = *reinterpret_cast<const uint32_t*>(&s_x[16 + grp][km + 8]);
                mma_m16n8k16(d2, a0, a1, a2, a3, b0_2, b1_2);
            }
            if (cnt > 24) {
                uint32_t b0_3 = *reinterpret_cast<const uint32_t*>(&s_x[24 + grp][km]);
                uint32_t b1_3 = *reinterpret_cast<const uint32_t*>(&s_x[24 + grp][km + 8]);
                mma_m16n8k16(d3, a0, a1, a2, a3, b0_3, b1_3);
            }
        }
        __syncthreads();
    }

    auto write_out = [&](int local_t, int r, float val) {
        if (local_t >= cnt) return;
        out_slots[(size_t)s_base[local_t] * HIDDEN + r] = val * s_rw[local_t];
    };
    int n0 = tid4 * 2;
    write_out(n0,     r0, d0[0]);
    write_out(n0 + 1, r0, d0[1]);
    write_out(n0,     r1, d0[2]);
    write_out(n0 + 1, r1, d0[3]);
    if (cnt > 8) {
        write_out(8 + n0,     r0, d1[0]);
        write_out(8 + n0 + 1, r0, d1[1]);
        write_out(8 + n0,     r1, d1[2]);
        write_out(8 + n0 + 1, r1, d1[3]);
    }
    if (cnt > 16) {
        write_out(16 + n0,     r0, d2[0]);
        write_out(16 + n0 + 1, r0, d2[1]);
        write_out(16 + n0,     r1, d2[2]);
        write_out(16 + n0 + 1, r1, d2[3]);
    }
    if (cnt > 24) {
        write_out(24 + n0,     r0, d3[0]);
        write_out(24 + n0 + 1, r0, d3[1]);
        write_out(24 + n0,     r1, d3[2]);
        write_out(24 + n0 + 1, r1, d3[3]);
    }
}

// BLACKWELL FP16 TENSOR CORE Multi-Token MXFP8 MoE Gate+Up (L20-23 in NVFP4 mode + all Exact/FP8!)
// Flat tile scheduler + 32-byte coalesced HBM loads (K=32/iter) + N=32!
// Grid: (INTERMEDIATE / 64 = 18, 256), Block: 128 threads
__global__ void __launch_bounds__(128)
p_moe_gate_up_mxfp8_tc_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const __half* __restrict__ x_h,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_slot,
    __half* __restrict__ out_act_h) {
    int tile_id = blockIdx.y;
    if (tile_id >= expert_offsets[129]) return;
    int expert = expert_offsets[130 + tile_id];
    int t0 = expert_offsets[386 + tile_id];
    int end = expert_offsets[642 + tile_id];
    int cnt = end - t0;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp = lane >> 2;
    int tid4 = lane & 3;

    int j_base = blockIdx.x * 64 + warp * 16;
    int j0 = j_base + grp;
    int j1 = j_base + grp + 8;

    constexpr int S_STRIDE = HIDDEN >> 5; // 72
    size_t rg0 = ((size_t)expert * INTERMEDIATE + j0) * 2;
    size_t ru0 = rg0 + 1;
    size_t rg1 = ((size_t)expert * INTERMEDIATE + j1) * 2;
    size_t ru1 = rg1 + 1;
    const uint2* wg0 = reinterpret_cast<const uint2*>(w_data + rg0 * HIDDEN);
    const uint2* wu0 = reinterpret_cast<const uint2*>(w_data + ru0 * HIDDEN);
    const uint2* wg1 = reinterpret_cast<const uint2*>(w_data + rg1 * HIDDEN);
    const uint2* wu1 = reinterpret_cast<const uint2*>(w_data + ru1 * HIDDEN);
    const uint8_t* sg0 = w_scales + rg0 * S_STRIDE;
    const uint8_t* su0 = w_scales + ru0 * S_STRIDE;
    const uint8_t* sg1 = w_scales + rg1 * S_STRIDE;
    const uint8_t* su1 = w_scales + ru1 * S_STRIDE;

    __shared__ __half s_x[32][40]; // 2,560 B SMEM (stride 80B = 16B aligned)
    __shared__ int s_tok[32];
    __shared__ int s_slot[32];

    int load_t = tid >> 2;         // 0..31 (ALL 32 tokens in 1 single uint4 load!)
    int load_c = (tid & 3) << 3;   // 0, 8, 16, 24

    if (tid < 32) {
        s_tok[tid] = (tid < cnt) ? sorted_tok[t0 + tid] : 0;
        s_slot[tid] = (tid < cnt) ? sorted_slot[t0 + tid] : 0;
    }
    __syncthreads();

    float dg0[4] = {0, 0, 0, 0}, dg1[4] = {0, 0, 0, 0}, dg2[4] = {0, 0, 0, 0}, dg3[4] = {0, 0, 0, 0};
    float du0[4] = {0, 0, 0, 0}, du1[4] = {0, 0, 0, 0}, du2[4] = {0, 0, 0, 0}, du3[4] = {0, 0, 0, 0};
    int my_tok = s_tok[load_t];

    for (int blk = 0; blk < S_STRIDE; ++blk) {
        int koff = blk << 5;
        int woff = (blk << 2) + tid4;
        uint2 wg0_r = __ldg(wg0 + woff), wg1_r = __ldg(wg1 + woff);
        uint2 wu0_r = __ldg(wu0 + woff), wu1_r = __ldg(wu1 + woff);
        __half2 scg0 = __float2half2_rn(cvt_ue8m0_to_float(__ldg(sg0 + blk)));
        __half2 scg1 = __float2half2_rn(cvt_ue8m0_to_float(__ldg(sg1 + blk)));
        __half2 scu0 = __float2half2_rn(cvt_ue8m0_to_float(__ldg(su0 + blk)));
        __half2 scu1 = __float2half2_rn(cvt_ue8m0_to_float(__ldg(su1 + blk)));

        if (load_t < cnt) {
            *reinterpret_cast<uint4*>(&s_x[load_t][load_c]) =
                __ldg(reinterpret_cast<const uint4*>(x_h + (size_t)my_tok * HIDDEN + koff + load_c));
        } else {
            *reinterpret_cast<uint4*>(&s_x[load_t][load_c]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            int km = (m << 4) + (tid4 << 1);
            uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(&s_x[grp][km]);
            uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(&s_x[grp][km + 8]);
            uint32_t b0_1 = 0, b1_1 = 0, b0_2 = 0, b1_2 = 0, b0_3 = 0, b1_3 = 0;
            if (cnt > 8) {
                b0_1 = *reinterpret_cast<const uint32_t*>(&s_x[8 + grp][km]);
                b1_1 = *reinterpret_cast<const uint32_t*>(&s_x[8 + grp][km + 8]);
            }
            if (cnt > 16) {
                b0_2 = *reinterpret_cast<const uint32_t*>(&s_x[16 + grp][km]);
                b1_2 = *reinterpret_cast<const uint32_t*>(&s_x[16 + grp][km + 8]);
            }
            if (cnt > 24) {
                b0_3 = *reinterpret_cast<const uint32_t*>(&s_x[24 + grp][km]);
                b1_3 = *reinterpret_cast<const uint32_t*>(&s_x[24 + grp][km + 8]);
            }

            uint32_t ag0, ag1, ag2, ag3;
            mxfp8_shfl_a_frag(wg0_r, wg1_r, scg0, scg1, grp, tid4, m, ag0, ag1, ag2, ag3);
            mma_m16n8k16(dg0, ag0, ag1, ag2, ag3, b0_0, b1_0);
            if (cnt > 8)  mma_m16n8k16(dg1, ag0, ag1, ag2, ag3, b0_1, b1_1);
            if (cnt > 16) mma_m16n8k16(dg2, ag0, ag1, ag2, ag3, b0_2, b1_2);
            if (cnt > 24) mma_m16n8k16(dg3, ag0, ag1, ag2, ag3, b0_3, b1_3);

            uint32_t au0, au1, au2, au3;
            mxfp8_shfl_a_frag(wu0_r, wu1_r, scu0, scu1, grp, tid4, m, au0, au1, au2, au3);
            mma_m16n8k16(du0, au0, au1, au2, au3, b0_0, b1_0);
            if (cnt > 8)  mma_m16n8k16(du1, au0, au1, au2, au3, b0_1, b1_1);
            if (cnt > 16) mma_m16n8k16(du2, au0, au1, au2, au3, b0_2, b1_2);
            if (cnt > 24) mma_m16n8k16(du3, au0, au1, au2, au3, b0_3, b1_3);
        }
        __syncthreads();
    }

    auto write_act = [&](int local_t, int j, float gv, float uv) {
        if (local_t >= cnt) return;
        float g_cap = fast_tanh(gv * (1.0f / GATE_SOFTCAP)) * GATE_SOFTCAP;
        float u_cap = fast_tanh(uv * (1.0f / UP_SOFTCAP)) * UP_SOFTCAP;
        int tok = s_tok[local_t];
        int slot = s_slot[local_t];
        out_act_h[((size_t)tok * TOP_K + slot) * INTERMEDIATE + j] = __float2half((g_cap * fast_sigmoid(gv)) * u_cap);
    };

    int n0 = tid4 * 2;
    write_act(n0,     j0, dg0[0], du0[0]);
    write_act(n0 + 1, j0, dg0[1], du0[1]);
    write_act(n0,     j1, dg0[2], du0[2]);
    write_act(n0 + 1, j1, dg0[3], du0[3]);
    if (cnt > 8) {
        write_act(8 + n0,     j0, dg1[0], du1[0]);
        write_act(8 + n0 + 1, j0, dg1[1], du1[1]);
        write_act(8 + n0,     j1, dg1[2], du1[2]);
        write_act(8 + n0 + 1, j1, dg1[3], du1[3]);
    }
    if (cnt > 16) {
        write_act(16 + n0,     j0, dg2[0], du2[0]);
        write_act(16 + n0 + 1, j0, dg2[1], du2[1]);
        write_act(16 + n0,     j1, dg2[2], du2[2]);
        write_act(16 + n0 + 1, j1, dg2[3], du2[3]);
    }
    if (cnt > 24) {
        write_act(24 + n0,     j0, dg3[0], du3[0]);
        write_act(24 + n0 + 1, j0, dg3[1], du3[1]);
        write_act(24 + n0,     j1, dg3[2], du3[2]);
        write_act(24 + n0 + 1, j1, dg3[3], du3[3]);
    }
}

// BLACKWELL FP16 TENSOR CORE Multi-Token MXFP8 MoE Down
// Grid: (HIDDEN / 64 = 36, 256), Block: 128 threads
__global__ void __launch_bounds__(128)
p_moe_down_mxfp8_tc_kernel(
    const uint8_t* __restrict__ w_data,
    const uint8_t* __restrict__ w_scales,
    const __half* __restrict__ act_h,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_slot,
    const float* __restrict__ sorted_w,
    float* __restrict__ out_slots) {
    int tile_id = blockIdx.y;
    if (tile_id >= expert_offsets[129]) return;
    int expert = expert_offsets[130 + tile_id];
    int t0 = expert_offsets[386 + tile_id];
    int end = expert_offsets[642 + tile_id];
    int cnt = end - t0;

    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int grp = lane >> 2;
    int tid4 = lane & 3;

    int r_base = blockIdx.x * 64 + warp * 16;
    int r0 = r_base + grp;
    int r1 = r_base + grp + 8;

    constexpr int S_STRIDE = INTERMEDIATE >> 5; // 36
    size_t row0 = (size_t)expert * HIDDEN + r0;
    size_t row1 = (size_t)expert * HIDDEN + r1;
    const uint2* wr0 = reinterpret_cast<const uint2*>(w_data + row0 * INTERMEDIATE);
    const uint2* wr1 = reinterpret_cast<const uint2*>(w_data + row1 * INTERMEDIATE);
    const uint8_t* sr0 = w_scales + row0 * S_STRIDE;
    const uint8_t* sr1 = w_scales + row1 * S_STRIDE;

    __shared__ __half s_x[32][40];
    __shared__ int s_base[32];
    __shared__ float s_rw[32];

    int load_t = tid >> 2;
    int load_c = (tid & 3) << 3;

    if (tid < 32) {
        if (tid < cnt) {
            s_base[tid] = sorted_tok[t0 + tid] * TOP_K + sorted_slot[t0 + tid];
            s_rw[tid] = sorted_w[t0 + tid];
        } else {
            s_base[tid] = 0;
            s_rw[tid] = 0.0f;
        }
    }
    __syncthreads();

    float d0[4] = {0, 0, 0, 0}, d1[4] = {0, 0, 0, 0}, d2[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
    int my_base = s_base[load_t];

    for (int blk = 0; blk < S_STRIDE; ++blk) {
        int koff = blk << 5;
        int woff = (blk << 2) + tid4;
        uint2 w0_r = __ldg(wr0 + woff), w1_r = __ldg(wr1 + woff);
        __half2 sc0 = __float2half2_rn(cvt_ue8m0_to_float(__ldg(sr0 + blk)));
        __half2 sc1 = __float2half2_rn(cvt_ue8m0_to_float(__ldg(sr1 + blk)));

        if (load_t < cnt) {
            *reinterpret_cast<uint4*>(&s_x[load_t][load_c]) =
                __ldg(reinterpret_cast<const uint4*>(act_h + (size_t)my_base * INTERMEDIATE + koff + load_c));
        } else {
            *reinterpret_cast<uint4*>(&s_x[load_t][load_c]) = make_uint4(0, 0, 0, 0);
        }
        __syncthreads();

        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            int km = (m << 4) + (tid4 << 1);
            uint32_t b0_0 = *reinterpret_cast<const uint32_t*>(&s_x[grp][km]);
            uint32_t b1_0 = *reinterpret_cast<const uint32_t*>(&s_x[grp][km + 8]);
            uint32_t a0, a1, a2, a3;
            mxfp8_shfl_a_frag(w0_r, w1_r, sc0, sc1, grp, tid4, m, a0, a1, a2, a3);
            mma_m16n8k16(d0, a0, a1, a2, a3, b0_0, b1_0);
            if (cnt > 8) {
                uint32_t b0_1 = *reinterpret_cast<const uint32_t*>(&s_x[8 + grp][km]);
                uint32_t b1_1 = *reinterpret_cast<const uint32_t*>(&s_x[8 + grp][km + 8]);
                mma_m16n8k16(d1, a0, a1, a2, a3, b0_1, b1_1);
            }
            if (cnt > 16) {
                uint32_t b0_2 = *reinterpret_cast<const uint32_t*>(&s_x[16 + grp][km]);
                uint32_t b1_2 = *reinterpret_cast<const uint32_t*>(&s_x[16 + grp][km + 8]);
                mma_m16n8k16(d2, a0, a1, a2, a3, b0_2, b1_2);
            }
            if (cnt > 24) {
                uint32_t b0_3 = *reinterpret_cast<const uint32_t*>(&s_x[24 + grp][km]);
                uint32_t b1_3 = *reinterpret_cast<const uint32_t*>(&s_x[24 + grp][km + 8]);
                mma_m16n8k16(d3, a0, a1, a2, a3, b0_3, b1_3);
            }
        }
        __syncthreads();
    }

    auto write_out = [&](int local_t, int r, float val) {
        if (local_t >= cnt) return;
        out_slots[(size_t)s_base[local_t] * HIDDEN + r] = val * s_rw[local_t];
    };
    int n0 = tid4 * 2;
    write_out(n0,     r0, d0[0]);
    write_out(n0 + 1, r0, d0[1]);
    write_out(n0,     r1, d0[2]);
    write_out(n0 + 1, r1, d0[3]);
    if (cnt > 8) {
        write_out(8 + n0,     r0, d1[0]);
        write_out(8 + n0 + 1, r0, d1[1]);
        write_out(8 + n0,     r1, d1[2]);
        write_out(8 + n0 + 1, r1, d1[3]);
    }
    if (cnt > 16) {
        write_out(16 + n0,     r0, d2[0]);
        write_out(16 + n0 + 1, r0, d2[1]);
        write_out(16 + n0,     r1, d2[2]);
        write_out(16 + n0 + 1, r1, d2[3]);
    }
    if (cnt > 24) {
        write_out(24 + n0,     r0, d3[0]);
        write_out(24 + n0 + 1, r0, d3[1]);
        write_out(24 + n0,     r1, d3[2]);
        write_out(24 + n0 + 1, r1, d3[3]);
    }
}

} // namespace

void launch_prefill_warmup_cublas(cudaStream_t stream) {
    cublasHandle_t h = get_cublas(stream);
    void* d_tmp = nullptr;
    if (cudaMalloc(&d_tmp, 2048) == cudaSuccess && d_tmp) {
        cudaMemsetAsync(d_tmp, 0, 2048, stream);
        float alpha = 1.0f, beta = 0.0f;
        const __half* a = reinterpret_cast<const __half*>(d_tmp);
        const __half* b = a + 256;
        float* c = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(d_tmp) + 1024);
        cublasGemmEx(
            h, CUBLAS_OP_T, CUBLAS_OP_N,
            16, 16, 16,
            &alpha,
            a, CUDA_R_16F, 16,
            b, CUDA_R_16F, 16,
            &beta,
            c, CUDA_R_32F, 16,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaStreamSynchronize(stream);
        cudaFree(d_tmp);
    }
}

void launch_prefill_f32_to_f16(const float* src, __half* dst, int count, cudaStream_t stream) {
    p_f32_to_f16_kernel<<<(count + 255) / 256, 256, 0, stream>>>(src, dst, (size_t)count);
}

void launch_prefill_gemm_h(
    const LinearWeight& w,
    const __half* x_h,
    float* y,
    __half* w_f16_scratch,
    int B,
    cudaStream_t stream) {
    if (w.kind == WeightKind::NVFP4) {
        size_t total_blocks = (size_t)w.rows * (w.cols >> 4);
        p_dequant_nvfp4_to_f16_kernel<<<(int)((total_blocks + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.scales, w.global_scale, w_f16_scratch, total_blocks);
    } else if (w.kind == WeightKind::FP8) {
        size_t total_pairs = (size_t)w.rows * (w.cols >> 1);
        p_dequant_fp8_to_f16_kernel<<<(int)((total_pairs + 255) / 256), 256, 0, stream>>>(
            static_cast<const uint8_t*>(w.data), w.row_scales, w_f16_scratch, w.rows, w.cols);
    } else {
        size_t n = (size_t)w.rows * w.cols;
        p_dequant_bf16_to_f16_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(w.data), w_f16_scratch, n);
    }

    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t h = get_cublas(stream);
    cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_T, CUBLAS_OP_N,
        w.rows, B, w.cols,
        &alpha,
        w_f16_scratch, CUDA_R_16F, w.cols,
        x_h, CUDA_R_16F, w.cols,
        &beta,
        y, CUDA_R_32F, w.rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasGemmEx failed: %d (N=%d B=%d K=%d)\n", (int)st, w.rows, B, w.cols);
        std::exit(1);
    }
}

void launch_prefill_sconv_post_res(
    const float* src_or_slots, int is_slotted, const __nv_bfloat16* sconv_w, float* ring,
    const float* residual, float* out_res, float* pre_tap_out,
    float* blocks, const float* partial, int block_idx, int is_boundary,
    int B, const int32_t* d_pos, cudaStream_t stream) {
    p_sconv_post_res_kernel<<<(HIDDEN + 255) / 256, 256, 0, stream>>>(
        src_or_slots, is_slotted, sconv_w, ring, residual, out_res, pre_tap_out,
        blocks, partial, block_idx, is_boundary, B, d_pos);
}

void launch_prefill_copy_taps(const float* src, float* dst_taps, int tap_slot, int B, cudaStream_t stream) {
    int total = B * HIDDEN;
    p_copy_taps_chunked_kernel<<<(total + 255) / 256, 256, 0, stream>>>(src, dst_taps, tap_slot, B);
}

void launch_prefill_local_sconv_prep_store(
    float* qrgs, float* k_proj, float* v_proj,
    const __nv_bfloat16* k_sconv_w, const __nv_bfloat16* v_sconv_w,
    float* k_ring, float* v_ring,
    const __nv_bfloat16* q_clip_scale, const __nv_bfloat16* rel_clip_scale,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int kv_cap, int B, const int32_t* d_pos, cudaStream_t stream) {
    launch_static_sconv(k_proj, k_sconv_w, k_ring, k_proj, B, KV_DIM, d_pos, stream);
    launch_static_sconv(v_proj, v_sconv_w, v_ring, v_proj, B, KV_DIM, d_pos, stream);
    launch_attn_prep_store(
        qrgs, k_proj, v_proj, q_clip_scale, rel_clip_scale,
        k_cache, v_cache, kv_cap, false, B, d_pos, stream);
}

void launch_prefill_controller_one(
    const float* x, const float* saved, const float* prior_last, const float* cp_q,
    const ControllerWeights& cw, const __nv_bfloat16* post_norm,
    float* out, __half* out_h,
    float* scratch_msg, __half* scratch_cat_h, float* scratch_d, __half* scratch_d_h,
    float* scratch_f, float* scratch_g, __half* w_f16_scratch,
    int B, cudaStream_t stream) {
    p_ctrl_prep_h_kernel<<<B, 256, 0, stream>>>(x, saved, prior_last, cp_q, scratch_msg, scratch_cat_h);

    LinearWeight down_w = cw.down_nvfp4;
    LinearWeight forget_w = cw.forget_nvfp4;
    LinearWeight gain_w = cw.gain_nvfp4;
    if (down_w.data == nullptr) {
        down_w = {WeightKind::BF16, GATE_RANK, 2 * HIDDEN, cw.down_w, nullptr, nullptr, 1.0f};
        forget_w = {WeightKind::BF16, HIDDEN, GATE_RANK, cw.forget_w, nullptr, nullptr, 1.0f};
        gain_w = {WeightKind::BF16, HIDDEN, GATE_RANK, cw.gain_w, nullptr, nullptr, 1.0f};
    }

    launch_prefill_gemm_h(down_w, scratch_cat_h, scratch_d, w_f16_scratch, B, stream);
    launch_prefill_f32_to_f16(scratch_d, scratch_d_h, B * GATE_RANK, stream);
    launch_prefill_gemm_h(forget_w, scratch_d_h, scratch_f, w_f16_scratch, B, stream);
    launch_prefill_gemm_h(gain_w, scratch_d_h, scratch_g, w_f16_scratch, B, stream);

    p_ctrl_finish_kernel<<<B, 256, 0, stream>>>(x, scratch_msg, scratch_f, scratch_g, post_norm, out, out_h);
}

void launch_prefill_route_and_sort(
    const float* router_logits, const float* router_bias,
    int32_t* route_ids, float* route_w,
    int32_t* expert_offsets, int32_t* sorted_tok, int32_t* sorted_slot, float* sorted_w,
    int B, cudaStream_t stream) {
    p_route_topk_only_kernel<<<B, 32, 0, stream>>>(router_logits, router_bias, route_ids, route_w);
    p_sort_experts_kernel<<<1, NUM_EXPERTS, 0, stream>>>(
        route_ids, route_w, B, expert_offsets, sorted_tok, sorted_slot, sorted_w);
}

void launch_prefill_moe_gate_up_h(
    const ExpertBank& w13, const __half* x_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot,
    __half* out_act_h, cudaStream_t stream) {
    // Pigeonhole bound for B <= 512, TOP_K=8, NUM_EXPERTS=128, TILE_N=32: max_tiles <= 256!
    dim3 grid(INTERMEDIATE / 64, 256);
    if (w13.is_nvfp4) {
        p_moe_gate_up_nvfp4_tc_kernel<<<grid, 128, 0, stream>>>(
            w13.data, w13.scales, w13.expert_scale, x_h, expert_offsets, sorted_tok, sorted_slot, out_act_h);
    } else {
        p_moe_gate_up_mxfp8_tc_kernel<<<grid, 128, 0, stream>>>(
            w13.data, w13.scales, x_h, expert_offsets, sorted_tok, sorted_slot, out_act_h);
    }
}

void launch_prefill_moe_down_h(
    const ExpertBank& w2, const __half* act_h,
    const int32_t* expert_offsets, const int32_t* sorted_tok, const int32_t* sorted_slot,
    const float* sorted_w, float* out_slots, cudaStream_t stream) {
    dim3 grid(HIDDEN / 64, 256);
    if (w2.is_nvfp4) {
        p_moe_down_nvfp4_tc_kernel<<<grid, 128, 0, stream>>>(
            w2.data, w2.scales, w2.expert_scale, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, out_slots);
    } else {
        p_moe_down_mxfp8_tc_kernel<<<grid, 128, 0, stream>>>(
            w2.data, w2.scales, act_h, expert_offsets, sorted_tok, sorted_slot, sorted_w, out_slots);
    }
}

} // namespace squid
