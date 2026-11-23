#include "weights.hpp"
#include "kernels_common.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace squid {

namespace {

// Device kernel: compute max(|x|) across BF16 tensor for NVFP4 global scale
__global__ void bf16_amax_kernel(const __nv_bfloat16* __restrict__ x, size_t n, float* __restrict__ out_amax) {
    float local_max = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float v = fabsf(__bfloat162float(x[i]));
        local_max = fmaxf(local_max, v);
    }
    local_max = warp_max(local_max);
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        v = warp_max(v);
        if (lane == 0) {
            // atomicMax for positive floats via int reinterpret
            atomicMax(reinterpret_cast<int*>(out_amax), __float_as_int(v));
        }
    }
}

__device__ __forceinline__ uint8_t quantize_e2m1_scalar(float v) {
    uint8_t sign = (v < 0.0f) ? 8u : 0u;
    float a = fabsf(v);
    uint8_t code;
    if      (a < 0.25f) code = 0; // 0.0
    else if (a < 0.75f) code = 1; // 0.5
    else if (a < 1.25f) code = 2; // 1.0
    else if (a < 1.75f) code = 3; // 1.5
    else if (a < 2.50f) code = 4; // 2.0
    else if (a < 3.50f) code = 5; // 3.0
    else if (a < 5.00f) code = 6; // 4.0
    else                code = 7; // 6.0
    return sign | code;
}

// Quantize BF16 [N, K] (K % 16 == 0) -> NVFP4 packed U8 [N, K/2] + F8_E4M3 scales [N, K/16]
__global__ void quant_bf16_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ src,
    uint8_t* __restrict__ dst_w,
    uint8_t* __restrict__ dst_s,
    int N, int K, float global_scale) {
    int blocks_per_row = K / 16;
    size_t total_blocks = (size_t)N * blocks_per_row;
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks) return;

    const __nv_bfloat16* row_in = src + bidx * 16;
    float vals[16];
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = __bfloat162float(row_in[i]);
        bmax = fmaxf(bmax, fabsf(vals[i]));
    }
    float raw_scale = (global_scale > 0.0f) ? (bmax / (6.0f * global_scale)) : 0.0f;
    raw_scale = fminf(raw_scale, 448.0f);
    __nv_fp8_e4m3 s_fp8(raw_scale);
    dst_s[bidx] = s_fp8.__x;
    float s_eff = static_cast<float>(s_fp8) * global_scale;
    float inv_s = (s_eff > 1e-30f) ? (1.0f / s_eff) : 0.0f;

    uint8_t* out_w = dst_w + bidx * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t lo = quantize_e2m1_scalar(vals[2 * i] * inv_s);
        uint8_t hi = quantize_e2m1_scalar(vals[2 * i + 1] * inv_s);
        out_w[i] = lo | (hi << 4);
    }
}

// Quantize BF16 [N, K] -> FP8 E4M3 [N, K] + per-row F32 scale [N]
__global__ void quant_bf16_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ src,
    uint8_t* __restrict__ dst_w,
    float* __restrict__ dst_row_scales,
    int N, int K) {
    int r = blockIdx.x;
    if (r >= N) return;
    const __nv_bfloat16* row_in = src + (size_t)r * K;
    float lmax = 0.0f;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        lmax = fmaxf(lmax, fabsf(__bfloat162float(row_in[c])));
    }
    lmax = warp_max(lmax);
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = lmax;
    __syncthreads();
    __shared__ float inv_scale;
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        v = warp_max(v);
        if (lane == 0) {
            float sc = fmaxf(v / 448.0f, 1e-12f);
            dst_row_scales[r] = sc;
            inv_scale = 1.0f / sc;
        }
    }
    __syncthreads();
    float isc = inv_scale;
    uint8_t* row_out = dst_w + (size_t)r * K;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        __nv_fp8_e4m3 q(__bfloat162float(row_in[c]) * isc);
        row_out[c] = q.__x;
    }
}

// Requantize MXFP8 expert bank [E, rows, cols] -> NVFP4 [E, rows, cols/2] + scales [E, rows, cols/16] + expert_scale [E]
__global__ void mxfp8_expert_amax_kernel(
    const uint8_t* __restrict__ src_w,
    const uint8_t* __restrict__ src_s,
    float* __restrict__ exp_amax,
    int rows, int cols) {
    int e = blockIdx.x;
    size_t expert_elems = (size_t)rows * cols;
    const uint8_t* ew = src_w + (size_t)e * expert_elems;
    const uint8_t* es = src_s + (size_t)e * (expert_elems / 32);
    float lmax = 0.0f;
    for (size_t idx = threadIdx.x; idx < expert_elems; idx += blockDim.x) {
        float sc = cvt_ue8m0_to_float(es[idx >> 5]);
        float v = fabsf(cvt_e4m3_to_float(ew[idx]) * sc);
        lmax = fmaxf(lmax, v);
    }
    lmax = warp_max(lmax);
    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) smem[warp] = lmax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        v = warp_max(v);
        if (lane == 0) exp_amax[e] = fmaxf(v / (448.0f * 6.0f), 1e-12f);
    }
}

__global__ void mxfp8_to_nvfp4_expert_kernel(
    const uint8_t* __restrict__ src_w,
    const uint8_t* __restrict__ src_s,
    const float* __restrict__ exp_global_scale,
    uint8_t* __restrict__ dst_w,
    uint8_t* __restrict__ dst_s,
    int n_experts, int rows, int cols) {
    size_t blocks_per_expert = (size_t)rows * (cols / 16);
    size_t total_blocks = (size_t)n_experts * blocks_per_expert;
    size_t bidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= total_blocks) return;

    int e = (int)(bidx / blocks_per_expert);
    float gscale = exp_global_scale[e];
    size_t elem_off = bidx * 16;
    float mx_scale = cvt_ue8m0_to_float(src_s[elem_off >> 5]);
    float vals[16];
    float bmax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] = cvt_e4m3_to_float(src_w[elem_off + i]) * mx_scale;
        bmax = fmaxf(bmax, fabsf(vals[i]));
    }
    float raw_scale = fminf(bmax / (6.0f * gscale), 448.0f);
    __nv_fp8_e4m3 s_fp8(raw_scale);
    dst_s[bidx] = s_fp8.__x;
    float s_eff = static_cast<float>(s_fp8) * gscale;
    float inv_s = (s_eff > 1e-30f) ? (1.0f / s_eff) : 0.0f;

    uint8_t* out_w = dst_w + bidx * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t lo = quantize_e2m1_scalar(vals[2 * i] * inv_s);
        uint8_t hi = quantize_e2m1_scalar(vals[2 * i + 1] * inv_s);
        out_w[i] = lo | (hi << 4);
    }
}

// Compute DSpark folded matrices:
// fold_bf16[v, r] = sum_k unembed[v, k] * hidden_out_w[k, r]  (V=200058, K=2304, R=256)
// fold_b[v]       = sum_k unembed[v, k] * hidden_out_b[k]
__global__ void compute_dspark_fold_bias_kernel(
    const __nv_bfloat16* __restrict__ unembed,
    const __nv_bfloat16* __restrict__ hidden_out_b,
    float* __restrict__ fold_b,
    int V, int K) {
    int v = blockIdx.x * blockDim.y + threadIdx.y;
    if (v >= V) return;
    int lane = threadIdx.x;
    const __nv_bfloat16* row = unembed + (size_t)v * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        acc = fmaf(__bfloat162float(row[k]), __bfloat162float(hidden_out_b[k]), acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) fold_b[v] = acc;
}

__global__ void compute_dspark_fold_w_kernel(
    const __nv_bfloat16* __restrict__ unembed,
    const __nv_bfloat16* __restrict__ hidden_out_w, // [2304, 256] row-major
    __nv_bfloat16* __restrict__ fold_out,           // [V, 256]
    int V, int K, int R) {
    // Each block computes 1 row v across all 256 output columns!
    int v = blockIdx.x;
    if (v >= V) return;
    int r = threadIdx.x; // 0..255
    __shared__ float u_row[HIDDEN];
    const __nv_bfloat16* u_ptr = unembed + (size_t)v * K;
    for (int k = r; k < K; k += 256) {
        u_row[k] = __bfloat162float(u_ptr[k]);
    }
    __syncthreads();
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc = fmaf(u_row[k], __bfloat162float(hidden_out_w[(size_t)k * R + r]), acc);
    }
    fold_out[(size_t)v * R + r] = __float2bfloat16(acc);
}

// Split DSpark rnn_proj.weight [768, 5120] -> w_sp [768, 512], w_h [768, 2304], w_f [768, 2304]
__global__ void split_dspark_rnn_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ w_sp,
    __nv_bfloat16* __restrict__ w_h,
    __nv_bfloat16* __restrict__ w_f) {
    int o = blockIdx.x;
    constexpr int K_IN = 2 * MTP_MARKOV_RANK + 2 * HIDDEN; // 5120
    const __nv_bfloat16* row = src + (size_t)o * K_IN;
    for (int c = threadIdx.x; c < K_IN; c += 256) {
        __nv_bfloat16 v = row[c];
        if (c < 512) w_sp[(size_t)o * 512 + c] = v;
        else if (c < 512 + HIDDEN) w_h[(size_t)o * HIDDEN + (c - 512)] = v;
        else w_f[(size_t)o * HIDDEN + (c - 512 - HIDDEN)] = v;
    }
}

} // namespace

ModelWeights::~ModelWeights() {
    for (void* p : allocations) {
        if (p) cudaFree(p);
    }
    allocations.clear();
    for (void* p : host_allocations) {
        if (p) cudaFreeHost(p);
    }
    host_allocations.clear();
}

void* ModelWeights::alloc_dev(size_t bytes) {
    void* ptr = nullptr;
    size_t aligned = (bytes + 255) & ~size_t(255);
    CUDA_CHECK(cudaMalloc(&ptr, aligned));
    allocations.push_back(ptr);
    total_vram_bytes += aligned;
    return ptr;
}

static void upload_raw(const SafetensorsFile& sf, const std::string& name, void* dst_dev, size_t expected_bytes) {
    const TensorInfo& info = sf.get(name);
    if (expected_bytes && info.nbytes != expected_bytes) {
        fprintf(stderr, "Tensor %s size mismatch: got %llu, expected %zu\n",
                name.c_str(), (unsigned long long)info.nbytes, expected_bytes);
        std::exit(1);
    }
    CUDA_CHECK(cudaMemcpy(dst_dev, sf.data_ptr(info), info.nbytes, cudaMemcpyHostToDevice));
}

// Helper: quantize a device BF16 buffer [rows, cols] into LinearWeight according to target_kind
static LinearWeight pack_linear_from_bf16_dev(
    ModelWeights& mw,
    const __nv_bfloat16* src_bf16_dev,
    int rows, int cols,
    WeightKind target_kind,
    float* d_amax_scratch) {
    LinearWeight lw;
    lw.rows = rows;
    lw.cols = cols;
    lw.kind = target_kind;

    if (target_kind == WeightKind::BF16) {
        size_t bytes = (size_t)rows * cols * sizeof(__nv_bfloat16);
        lw.data = mw.alloc_dev(bytes);
        CUDA_CHECK(cudaMemcpy(lw.data, src_bf16_dev, bytes, cudaMemcpyDeviceToDevice));
    } else if (target_kind == WeightKind::FP8) {
        lw.data = mw.alloc_dev((size_t)rows * cols);
        lw.row_scales = static_cast<float*>(mw.alloc_dev((size_t)rows * sizeof(float)));
        quant_bf16_to_fp8_kernel<<<rows, 256>>>(src_bf16_dev, static_cast<uint8_t*>(lw.data), lw.row_scales, rows, cols);
        CUDA_CHECK(cudaGetLastError());
    } else { // NVFP4
        CUDA_CHECK(cudaMemset(d_amax_scratch, 0, sizeof(float)));
        size_t n = (size_t)rows * cols;
        int grid = (int)std::min<size_t>((n + 255) / 256, 2048);
        bf16_amax_kernel<<<grid, 256>>>(src_bf16_dev, n, d_amax_scratch);
        float h_amax = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_amax, d_amax_scratch, sizeof(float), cudaMemcpyDeviceToHost));
        lw.global_scale = std::max(h_amax / (448.0f * 6.0f), 1e-12f);

        lw.data = mw.alloc_dev((size_t)rows * (cols / 2));
        lw.scales = static_cast<uint8_t*>(mw.alloc_dev((size_t)rows * (cols / 16)));
        size_t total_blocks = (size_t)rows * (cols / 16);
        int qgrid = (int)((total_blocks + 255) / 256);
        quant_bf16_to_nvfp4_kernel<<<qgrid, 256>>>(
            src_bf16_dev, static_cast<uint8_t*>(lw.data), lw.scales, rows, cols, lw.global_scale);
        CUDA_CHECK(cudaGetLastError());
    }
    return lw;
}

bool ModelWeights::load(const std::string& model_dir, QuantMode quant_mode, bool load_mtp, bool embed_on_host) {
    auto t0 = std::chrono::steady_clock::now();
    mode = quant_mode;

    SafetensorsFile sf;
    std::string main_path = model_dir + "/model-00001-of-00001.safetensors";
    if (!sf.open(main_path)) return false;

    WeightKind dense_kind = (quant_mode == QuantMode::Exact) ? WeightKind::BF16
                          : (quant_mode == QuantMode::FP8)   ? WeightKind::FP8
                                                             : WeightKind::NVFP4;
    // Scratch device buffer for BF16 staging & quantization (max: unembed = 200058 * 2304 * 2 = 922 MB)
    __nv_bfloat16* d_stage = nullptr;
    size_t stage_bytes = (size_t)VOCAB_UNPADDED * HIDDEN * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMalloc(&d_stage, stage_bytes));
    float* d_amax = nullptr;
    CUDA_CHECK(cudaMalloc(&d_amax, sizeof(float)));

    auto load_bf16 = [&](const std::string& name, size_t elems) -> __nv_bfloat16* {
        auto* p = static_cast<__nv_bfloat16*>(alloc_dev(elems * sizeof(__nv_bfloat16)));
        upload_raw(sf, name, p, elems * sizeof(__nv_bfloat16));
        return p;
    };
    auto load_f32 = [&](const std::string& name, size_t elems) -> float* {
        auto* p = static_cast<float*>(alloc_dev(elems * sizeof(float)));
        upload_raw(sf, name, p, elems * sizeof(float));
        return p;
    };

    // Embeddings & norms
    if (embed_on_host) {
        const TensorInfo& ei = sf.get("model.llm.embed.weight");
        size_t ebytes = (size_t)VOCAB_PADDED * HIDDEN * sizeof(__nv_bfloat16);
        void* h_ptr = nullptr;
        CUDA_CHECK(cudaHostAlloc(&h_ptr, ebytes, cudaHostAllocMapped));
        host_allocations.push_back(h_ptr);
        std::memcpy(h_ptr, sf.data_ptr(ei), ebytes);
        void* d_ptr = nullptr;
        CUDA_CHECK(cudaHostGetDevicePointer(&d_ptr, h_ptr, 0));
        embed = static_cast<__nv_bfloat16*>(d_ptr);
        printf("[weights] Embed table pinned in zero-copy host RAM (saved %.2f GiB VRAM)\n",
               ebytes / (1024.0 * 1024.0 * 1024.0));
    } else {
        embed = load_bf16("model.llm.embed.weight", (size_t)VOCAB_PADDED * HIDDEN);
    }
    embed_norm = load_bf16("model.llm.embed_norm.weight", HIDDEN);
    final_norm = load_bf16("model.llm.norm.weight", HIDDEN);

    // Unembed (only first VOCAB_UNPADDED = 200058 rows are usable!)
    {
        const TensorInfo& ui = sf.get("model.llm.unembed.weight");
        size_t usable_bytes = (size_t)VOCAB_UNPADDED * HIDDEN * sizeof(__nv_bfloat16);
        CUDA_CHECK(cudaMemcpy(d_stage, sf.data_ptr(ui), usable_bytes, cudaMemcpyHostToDevice));
        unembed = pack_linear_from_bf16_dev(*this, d_stage, VOCAB_UNPADDED, HIDDEN, dense_kind, d_amax);
    }

    // AttnRes & cross-pass queries
    for (int l = 1; l < NUM_LAYERS; ++l) {
        size_t elems = is_global_layer(l) ? (3 * HIDDEN) : HIDDEN;
        attn_queries[l - 1] = load_f32("model.llm.attn_res.attn_queries." + std::to_string(l - 1), elems);
    }
    for (int l = 0; l < NUM_LAYERS; ++l) {
        mlp_queries[l] = load_f32("model.llm.attn_res.mlp_queries." + std::to_string(l), HIDDEN);
    }
    output_query = load_f32("model.llm.attn_res.output_query", HIDDEN);
    cross_pass_queries = load_f32("model.llm.middle_cycle_cross_pass_queries", 24 * HIDDEN);
    cross_pass_kv_queries = load_f32("model.llm.middle_cycle_cross_pass_kv_queries", 24 * HIDDEN);

    // 24 physical layers
    for (int l = 0; l < NUM_LAYERS; ++l) {
        LayerWeights& lw = layers[l];
        std::string pfx = "model.llm.layers." + std::to_string(l) + ".";
        lw.is_global = is_global_layer(l);

        if (lw.is_global) {
            lw.q_in_norm = load_bf16(pfx + "q_input_norm.weight", HIDDEN);
            lw.k_in_norm = load_bf16(pfx + "k_input_norm.weight", HIDDEN);
            lw.v_in_norm = load_bf16(pfx + "v_input_norm.weight", HIDDEN);
        } else {
            lw.attn_norm = load_bf16(pfx + "attn_norm.weight", HIDDEN);
        }
        lw.mlp_norm = load_bf16(pfx + "mlp_norm.weight", HIDDEN);

        // Concatenate [wq(3072), wr(384), gate(3072), sink(24)] -> 6552 x 2304
        {
            size_t off = 0;
            auto stage_cat = [&](const std::string& name, int r) {
                size_t bytes = (size_t)r * HIDDEN * sizeof(__nv_bfloat16);
                upload_raw(sf, name, reinterpret_cast<uint8_t*>(d_stage) + off, bytes);
                off += bytes;
            };
            stage_cat(pfx + "attn.wq_du.weight", Q_DIM);
            stage_cat(pfx + "attn.wr_du.weight", R_DIM);
            stage_cat(pfx + "attn.gate_attn.weight", Q_DIM);
            stage_cat(pfx + "attn.attn_sink_proj.weight", NUM_HEADS);
            lw.w_qrgs = pack_linear_from_bf16_dev(*this, d_stage, Q_DIM + R_DIM + Q_DIM + NUM_HEADS, HIDDEN, dense_kind, d_amax);
        }

        // Concatenate [wk(1024), wv(1024)] -> 2048 x 2304, and create slices w_k, w_v!
        {
            size_t kv_bytes = (size_t)KV_DIM * HIDDEN * sizeof(__nv_bfloat16);
            upload_raw(sf, pfx + "attn.wk_dv.weight", d_stage, kv_bytes);
            upload_raw(sf, pfx + "attn.wv_dv.weight", reinterpret_cast<uint8_t*>(d_stage) + kv_bytes, kv_bytes);
            lw.w_kv = pack_linear_from_bf16_dev(*this, d_stage, 2 * KV_DIM, HIDDEN, dense_kind, d_amax);

            lw.w_k = lw.w_kv;
            lw.w_k.rows = KV_DIM;
            lw.w_v = lw.w_kv;
            lw.w_v.rows = KV_DIM;
            if (dense_kind == WeightKind::BF16) {
                lw.w_v.data = static_cast<uint8_t*>(lw.w_kv.data) + (size_t)KV_DIM * HIDDEN * 2;
            } else if (dense_kind == WeightKind::FP8) {
                lw.w_v.data = static_cast<uint8_t*>(lw.w_kv.data) + (size_t)KV_DIM * HIDDEN;
                lw.w_v.row_scales = lw.w_kv.row_scales + KV_DIM;
            } else {
                lw.w_v.data = static_cast<uint8_t*>(lw.w_kv.data) + (size_t)KV_DIM * (HIDDEN / 2);
                lw.w_v.scales = lw.w_kv.scales + (size_t)KV_DIM * (HIDDEN / 16);
            }
        }

        // w_o: [2304, 3072]
        {
            size_t wo_bytes = (size_t)HIDDEN * Q_DIM * sizeof(__nv_bfloat16);
            upload_raw(sf, pfx + "attn.wo_ud.weight", d_stage, wo_bytes);
            lw.w_o = pack_linear_from_bf16_dev(*this, d_stage, HIDDEN, Q_DIM, dense_kind, d_amax);
        }

        // Short convolutions
        if (lw.is_global) {
            lw.k_sconv_w = load_bf16(pfx + "attn.k_sconv.static_w", CONV_TAPS * KV_DIM);
            lw.v_sconv_w = load_bf16(pfx + "attn.v_sconv.static_w", CONV_TAPS * KV_DIM);
            lw.k_sconv_proj = load_bf16(pfx + "attn.k_sconv.weight_proj", NUM_KV_HEADS * CONV_TAPS * HIDDEN);
            lw.v_sconv_proj = load_bf16(pfx + "attn.v_sconv.weight_proj", NUM_KV_HEADS * CONV_TAPS * HIDDEN);
        } else {
            lw.k_sconv_w = load_bf16(pfx + "attn.k_sconv.weight", KV_DIM * CONV_TAPS);
            lw.v_sconv_w = load_bf16(pfx + "attn.v_sconv.weight", KV_DIM * CONV_TAPS);
        }
        lw.attn_sconv_w = load_bf16(pfx + "attn_sconv.weight", HIDDEN * CONV_TAPS);
        lw.mlp_sconv_w = load_bf16(pfx + "mlp_sconv.weight", HIDDEN * CONV_TAPS);

        lw.q_clip_scale = load_bf16(pfx + "attn.q_l2_clip.scale", NUM_HEADS * HEAD_DIM);
        lw.rel_clip_scale = load_bf16(pfx + "attn.rel_l2_clip.scale", NUM_HEADS * D_REL);
        lw.rel_logits_proj = load_bf16(pfx + "attn.rel_logits_proj.proj", D_REL * REL_EXTENT);

        // Cross-pass controllers (layers 6..17)
        if (l >= LOOP_START && l < LOOP_END) {
            lw.has_controllers = true;
            const char* stems[4] = {"attn", "k", "v", "mlp"};
            for (int s = 0; s < 4; ++s) {
                std::string cp = pfx + "controllers." + stems[s] + ".";
                if (quant_mode == QuantMode::NVFP4) {
                    upload_raw(sf, cp + "down_weight", d_stage, (size_t)GATE_RANK * 2 * HIDDEN * 2);
                    lw.ctrl[s].down_nvfp4 = pack_linear_from_bf16_dev(*this, d_stage, GATE_RANK, 2 * HIDDEN, WeightKind::NVFP4, d_amax);
                    upload_raw(sf, cp + "forget_weight", d_stage, (size_t)HIDDEN * GATE_RANK * 2);
                    lw.ctrl[s].forget_nvfp4 = pack_linear_from_bf16_dev(*this, d_stage, HIDDEN, GATE_RANK, WeightKind::NVFP4, d_amax);
                    upload_raw(sf, cp + "gain_weight", d_stage, (size_t)HIDDEN * GATE_RANK * 2);
                    lw.ctrl[s].gain_nvfp4 = pack_linear_from_bf16_dev(*this, d_stage, HIDDEN, GATE_RANK, WeightKind::NVFP4, d_amax);
                } else {
                    lw.ctrl[s].down_w = load_bf16(cp + "down_weight", (size_t)GATE_RANK * 2 * HIDDEN);
                    lw.ctrl[s].forget_w = load_bf16(cp + "forget_weight", (size_t)HIDDEN * GATE_RANK);
                    lw.ctrl[s].gain_w = load_bf16(cp + "gain_weight", (size_t)HIDDEN * GATE_RANK);
                }
            }
        }

        // Router
        if (quant_mode == QuantMode::NVFP4) {
            upload_raw(sf, pfx + "mlp.gate.weight", d_stage, (size_t)NUM_EXPERTS * HIDDEN * 2);
            lw.router_nvfp4 = pack_linear_from_bf16_dev(*this, d_stage, NUM_EXPERTS, HIDDEN, WeightKind::NVFP4, d_amax);
        } else {
            lw.router_w = load_bf16(pfx + "mlp.gate.weight", (size_t)NUM_EXPERTS * HIDDEN);
        }
        lw.router_bias = load_f32(pfx + "mlp.gate.bias", NUM_EXPERTS);

        // MoE Experts
        auto load_expert_bank = [&](ExpertBank& eb, const std::string& wname, int rows, int cols) {
            eb.rows = rows;
            eb.cols = cols;
            bool ckpt_nvfp4 = is_nvfp4_expert_layer(l);
            if (ckpt_nvfp4) {
                eb.is_nvfp4 = true;
                size_t w_bytes = (size_t)NUM_EXPERTS * rows * (cols / 2);
                size_t s_bytes = (size_t)NUM_EXPERTS * rows * (cols / 16);
                eb.data = static_cast<uint8_t*>(alloc_dev(w_bytes));
                eb.scales = static_cast<uint8_t*>(alloc_dev(s_bytes));
                upload_raw(sf, pfx + "mlp.experts." + wname + "_weight", eb.data, w_bytes);
                upload_raw(sf, pfx + "mlp.experts." + wname + "_weight_scale", eb.scales, s_bytes);

                // Precompute expert_scale[e] = amax[e] / denom[0]
                const float* h_amax = reinterpret_cast<const float*>(sf.data_ptr(pfx + "mlp.experts." + wname + "_weight_amax"));
                const float* h_denom = reinterpret_cast<const float*>(sf.data_ptr(pfx + "mlp.experts." + wname + "_weight_outer_scale_denominator"));
                float h_sc[NUM_EXPERTS];
                float inv_d = 1.0f / h_denom[0];
                for (int e = 0; e < NUM_EXPERTS; ++e) h_sc[e] = h_amax[e] * inv_d;
                eb.expert_scale = static_cast<float*>(alloc_dev(NUM_EXPERTS * sizeof(float)));
                CUDA_CHECK(cudaMemcpy(eb.expert_scale, h_sc, sizeof(h_sc), cudaMemcpyHostToDevice));
            } else if (quant_mode != QuantMode::NVFP4) {
                // Keep native MXFP8 for layers 20..23
                eb.is_nvfp4 = false;
                size_t w_bytes = (size_t)NUM_EXPERTS * rows * cols;
                size_t s_bytes = (size_t)NUM_EXPERTS * rows * (cols / 32);
                eb.data = static_cast<uint8_t*>(alloc_dev(w_bytes));
                eb.scales = static_cast<uint8_t*>(alloc_dev(s_bytes));
                upload_raw(sf, pfx + "mlp.experts." + wname + "_weight", eb.data, w_bytes);
                upload_raw(sf, pfx + "mlp.experts." + wname + "_weight_scale", eb.scales, s_bytes);
            } else {
                // Aggressive NVFP4 mode: requantize MXFP8 -> NVFP4 on GPU!
                eb.is_nvfp4 = true;
                size_t mx_w_bytes = (size_t)NUM_EXPERTS * rows * cols;
                size_t mx_s_bytes = (size_t)NUM_EXPERTS * rows * (cols / 32);
                uint8_t* tmp_w = reinterpret_cast<uint8_t*>(d_stage);
                uint8_t* tmp_s = tmp_w + mx_w_bytes;
                // d_stage is 922 MB; max mx_w + mx_s is 679.5 + 21.2 = 701 MB < 922 MB!
                upload_raw(sf, pfx + "mlp.experts." + wname + "_weight", tmp_w, mx_w_bytes);
                upload_raw(sf, pfx + "mlp.experts." + wname + "_weight_scale", tmp_s, mx_s_bytes);

                eb.expert_scale = static_cast<float*>(alloc_dev(NUM_EXPERTS * sizeof(float)));
                mxfp8_expert_amax_kernel<<<NUM_EXPERTS, 256>>>(tmp_w, tmp_s, eb.expert_scale, rows, cols);

                eb.data = static_cast<uint8_t*>(alloc_dev((size_t)NUM_EXPERTS * rows * (cols / 2)));
                eb.scales = static_cast<uint8_t*>(alloc_dev((size_t)NUM_EXPERTS * rows * (cols / 16)));
                size_t total_blocks = (size_t)NUM_EXPERTS * rows * (cols / 16);
                mxfp8_to_nvfp4_expert_kernel<<<(int)((total_blocks + 255) / 256), 256>>>(
                    tmp_w, tmp_s, eb.expert_scale, eb.data, eb.scales, NUM_EXPERTS, rows, cols);
                CUDA_CHECK(cudaGetLastError());
            }
        };
        load_expert_bank(lw.w13, "w13", 2 * INTERMEDIATE, HIDDEN);
        load_expert_bank(lw.w2,  "w2",  HIDDEN, INTERMEDIATE);
    }

    // Load DSpark MTP weights if requested
    std::string mtp_path = model_dir + "/mtp.safetensors";
    SafetensorsFile sf_mtp;
    if (load_mtp && sf_mtp.open(mtp_path)) {
        // Always quantize DSpark block projections to NVFP4 in FP8/NVFP4 modes (drafts are verified!)
        const char* env_mtp_bf16 = std::getenv("SQUID_MTP_BF16");
        WeightKind mtp_kind = (quant_mode == QuantMode::Exact || (env_mtp_bf16 && env_mtp_bf16[0] == '1'))
                            ? WeightKind::BF16 : WeightKind::NVFP4;

        auto load_mtp_bf16 = [&](const std::string& name, size_t elems) -> __nv_bfloat16* {
            auto* p = static_cast<__nv_bfloat16*>(alloc_dev(elems * sizeof(__nv_bfloat16)));
            upload_raw(sf_mtp, name, p, elems * sizeof(__nv_bfloat16));
            return p;
        };

        // fuse_weight [4, 4] BF16 -> softmax per row into F32 [4, 4]
        const uint16_t* raw_fuse = reinterpret_cast<const uint16_t*>(sf_mtp.data_ptr("fuse_weight"));
        float alpha[16];
        for (int r = 0; r < 4; ++r) {
            double mx = -1e30;
            double v[4];
            for (int c = 0; c < 4; ++c) {
                uint32_t bits = uint32_t(raw_fuse[r * 4 + c]) << 16;
                float f;
                std::memcpy(&f, &bits, 4);
                v[c] = f;
                mx = std::max(mx, v[c]);
            }
            double sum = 0.0;
            for (int c = 0; c < 4; ++c) { v[c] = std::exp(v[c] - mx); sum += v[c]; }
            for (int c = 0; c < 4; ++c) alpha[r * 4 + c] = static_cast<float>(v[c] / sum);
        }
        dspark.fuse_alpha = static_cast<float*>(alloc_dev(16 * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dspark.fuse_alpha, alpha, sizeof(alpha), cudaMemcpyHostToDevice));

        dspark.mask_embed = load_mtp_bf16("mask_embed", HIDDEN);
        dspark.norm_w = load_mtp_bf16("norm.weight", HIDDEN);
        dspark.markov_dflare_norm = load_mtp_bf16("markov_head.dflare_norm.weight", HIDDEN);
        dspark.markov_w1 = load_mtp_bf16("markov_head.markov_w1.weight", (size_t)VOCAB_PADDED * MTP_MARKOV_RANK);
        dspark.rnn_bias = load_mtp_bf16("markov_head.rnn_proj.bias", 3 * MTP_MARKOV_RANK);
        dspark.hidden_out_w = load_mtp_bf16("markov_head.hidden_out.weight", (size_t)HIDDEN * MTP_MARKOV_RANK);
        dspark.hidden_out_b = load_mtp_bf16("markov_head.hidden_out.bias", HIDDEN);

        // Split rnn_proj.weight [768, 5120]
        {
            size_t rnn_bytes = (size_t)(3 * MTP_MARKOV_RANK) * (2 * MTP_MARKOV_RANK + 2 * HIDDEN) * sizeof(__nv_bfloat16);
            upload_raw(sf_mtp, "markov_head.rnn_proj.weight", d_stage, rnn_bytes);
            dspark.rnn_w_sp = static_cast<__nv_bfloat16*>(alloc_dev((size_t)768 * 512 * sizeof(__nv_bfloat16)));
            dspark.rnn_w_h  = static_cast<__nv_bfloat16*>(alloc_dev((size_t)768 * HIDDEN * sizeof(__nv_bfloat16)));
            dspark.rnn_w_f  = static_cast<__nv_bfloat16*>(alloc_dev((size_t)768 * HIDDEN * sizeof(__nv_bfloat16)));
            split_dspark_rnn_kernel<<<768, 256>>>(d_stage, dspark.rnn_w_sp, dspark.rnn_w_h, dspark.rnn_w_f);
        }

        // Precompute folded Markov head (with optional pruned draft vocab for 4-6x faster drafter!)
        {
            const TensorInfo& ui = sf.get("model.llm.unembed.weight");
            const uint8_t* h_unembed = sf.data_ptr(ui);

            // Check for pruned draft vocabulary file (SQUID_DRAFT_VOCAB=full disables pruning)
            std::vector<int32_t> h_dvocab;
            const char* env_dv = std::getenv("SQUID_DRAFT_VOCAB");
            bool allow_prune = (quant_mode != QuantMode::Exact) && (!env_dv || std::string(env_dv) != "full");
            if (allow_prune) {
                const char* cands[2] = {"/draft_vocab_32k.bin", "/draft_vocab_49k.bin"};
                if (env_dv && std::string(env_dv) == "49k") {
                    cands[0] = "/draft_vocab_49k.bin";
                    cands[1] = "/draft_vocab_32k.bin";
                }
                for (const char* fname : cands) {
                    FILE* fp = fopen((model_dir + fname).c_str(), "rb");
                    if (fp) {
                        fseek(fp, 0, SEEK_END);
                        long sz = ftell(fp);
                        fseek(fp, 0, SEEK_SET);
                        h_dvocab.resize(sz / sizeof(int32_t));
                        if (fread(h_dvocab.data(), sizeof(int32_t), h_dvocab.size(), fp) != h_dvocab.size()) {
                            h_dvocab.clear();
                        }
                        fclose(fp);
                        if (!h_dvocab.empty()) break;
                    }
                }
            }

            if (!h_dvocab.empty()) {
                dspark.draft_vocab_size = (int)h_dvocab.size();
                dspark.draft_vocab_ids = static_cast<int32_t*>(alloc_dev(h_dvocab.size() * sizeof(int32_t)));
                CUDA_CHECK(cudaMemcpy(dspark.draft_vocab_ids, h_dvocab.data(),
                                      h_dvocab.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
                // Gather pruned unembed rows on host into contiguous buffer, upload to d_stage
                std::vector<uint16_t> h_sub((size_t)dspark.draft_vocab_size * HIDDEN);
                const uint16_t* u16 = reinterpret_cast<const uint16_t*>(h_unembed);
                for (int i = 0; i < dspark.draft_vocab_size; ++i) {
                    std::memcpy(h_sub.data() + (size_t)i * HIDDEN,
                                u16 + (size_t)h_dvocab[i] * HIDDEN,
                                HIDDEN * sizeof(uint16_t));
                }
                CUDA_CHECK(cudaMemcpy(d_stage, h_sub.data(), h_sub.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
                dspark.unembed_draft = pack_linear_from_bf16_dev(*this, d_stage, dspark.draft_vocab_size, HIDDEN, mtp_kind, d_amax);
            } else {
                dspark.draft_vocab_size = VOCAB_UNPADDED;
                dspark.draft_vocab_ids = nullptr;
                size_t usable_bytes = (size_t)VOCAB_UNPADDED * HIDDEN * sizeof(__nv_bfloat16);
                CUDA_CHECK(cudaMemcpy(d_stage, h_unembed, usable_bytes, cudaMemcpyHostToDevice));
                dspark.unembed_draft = unembed; // alias full unembed
            }

            int Vd = dspark.draft_vocab_size;
            dspark.fold_b = static_cast<float*>(alloc_dev((size_t)Vd * sizeof(float)));
            dim3 fb_blk(32, 8);
            int fb_grid = (Vd + 7) / 8;
            compute_dspark_fold_bias_kernel<<<fb_grid, fb_blk>>>(d_stage, dspark.hidden_out_b, dspark.fold_b, Vd, HIDDEN);

            __nv_bfloat16* d_fold_bf16 = nullptr;
            CUDA_CHECK(cudaMalloc(&d_fold_bf16, (size_t)Vd * MTP_MARKOV_RANK * sizeof(__nv_bfloat16)));
            compute_dspark_fold_w_kernel<<<Vd, 256>>>(
                d_stage, dspark.hidden_out_w, d_fold_bf16, Vd, HIDDEN, MTP_MARKOV_RANK);
            dspark.fold_w = pack_linear_from_bf16_dev(*this, d_fold_bf16, Vd, MTP_MARKOV_RANK, mtp_kind, d_amax);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(d_fold_bf16));
            printf("[dspark] Draft vocab size=%d (%s)\n", Vd, dspark.draft_vocab_ids ? "pruned" : "full");
        }

        for (int l = 0; l < MTP_LAYERS; ++l) {
            DsparkLayerWeights& dl = dspark.layers[l];
            dl.is_local = is_mtp_local_layer(l);
            std::string p = "layers." + std::to_string(l) + ".";
            dl.input_norm = load_mtp_bf16(p + "input_layernorm.weight", HIDDEN);
            dl.post_norm = load_mtp_bf16(p + "post_attention_layernorm.weight", HIDDEN);
            dl.dflare_norm = load_mtp_bf16(p + "dflare_norm.weight", HIDDEN);
            dl.q_norm = load_mtp_bf16(p + "self_attn.q_norm.weight", HEAD_DIM);
            dl.k_norm = load_mtp_bf16(p + "self_attn.k_norm.weight", HEAD_DIM);
            dl.k_norm_ctx = load_mtp_bf16(p + "self_attn.k_norm_ctx.weight", HEAD_DIM);

            // w_qkv [5120, 2304]
            {
                size_t qb = (size_t)Q_DIM * HIDDEN * 2;
                size_t kb = (size_t)KV_DIM * HIDDEN * 2;
                upload_raw(sf_mtp, p + "self_attn.q_proj.weight", d_stage, qb);
                upload_raw(sf_mtp, p + "self_attn.k_proj.weight", reinterpret_cast<uint8_t*>(d_stage) + qb, kb);
                upload_raw(sf_mtp, p + "self_attn.v_proj.weight", reinterpret_cast<uint8_t*>(d_stage) + qb + kb, kb);
                dl.w_qkv = pack_linear_from_bf16_dev(*this, d_stage, Q_DIM + 2 * KV_DIM, HIDDEN, mtp_kind, d_amax);
            }
            // w_o [2304, 3072]
            {
                upload_raw(sf_mtp, p + "self_attn.o_proj.weight", d_stage, (size_t)HIDDEN * Q_DIM * 2);
                dl.w_o = pack_linear_from_bf16_dev(*this, d_stage, HIDDEN, Q_DIM, mtp_kind, d_amax);
            }
            // w_gu [18432, 2304]
            {
                size_t gb = (size_t)MTP_INTERMEDIATE * HIDDEN * 2;
                upload_raw(sf_mtp, p + "mlp.gate_proj.weight", d_stage, gb);
                upload_raw(sf_mtp, p + "mlp.up_proj.weight", reinterpret_cast<uint8_t*>(d_stage) + gb, gb);
                dl.w_gu = pack_linear_from_bf16_dev(*this, d_stage, 2 * MTP_INTERMEDIATE, HIDDEN, mtp_kind, d_amax);
            }
            // w_down [2304, 9216]
            {
                upload_raw(sf_mtp, p + "mlp.down_proj.weight", d_stage, (size_t)HIDDEN * MTP_INTERMEDIATE * 2);
                dl.w_down = pack_linear_from_bf16_dev(*this, d_stage, HIDDEN, MTP_INTERMEDIATE, mtp_kind, d_amax);
            }
            // w_kv_ctx [2048, 2304]
            {
                size_t kb = (size_t)KV_DIM * HIDDEN * 2;
                upload_raw(sf_mtp, p + "self_attn.k_proj_ctx.weight", d_stage, kb);
                upload_raw(sf_mtp, p + "self_attn.v_proj_ctx.weight", reinterpret_cast<uint8_t*>(d_stage) + kb, kb);
                dl.w_kv_ctx = pack_linear_from_bf16_dev(*this, d_stage, 2 * KV_DIM, HIDDEN, mtp_kind, d_amax);
            }
        }
        dspark.loaded = true;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_stage));
    CUDA_CHECK(cudaFree(d_amax));

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("[weights] Loaded mode=%s in %.2fs | VRAM=%.2f GiB | MTP=%s\n",
           quant_mode == QuantMode::Exact ? "exact" : (quant_mode == QuantMode::FP8 ? "fp8" : "nvfp4"),
           sec, total_vram_bytes / (1024.0 * 1024.0 * 1024.0), dspark.loaded ? "on" : "off");
    return true;
}

} // namespace squid
