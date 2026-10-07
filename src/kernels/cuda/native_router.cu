// Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d; finite F32, 512-expert/10-output path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_router.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
template<bool Restricted = false>
__launch_bounds__(256, 1)
__global__ void route(const float* __restrict__ logits, int32_t* __restrict__ ids,
                      float* __restrict__ weights, const int32_t* resident = nullptr, const int32_t* mode = nullptr) {
    if constexpr (Restricted) { if (*mode != 1) return; }
    // Preserve the pinned 32x8 block geometry; only row zero is active here.
    // blockIdx.x = the token (a multi-token launch; 0 for the single one)
    logits += (size_t) blockIdx.x * 512; ids += (size_t) blockIdx.x * 10; weights += (size_t) blockIdx.x * 10;
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = logits[lane + i * 32];
        if constexpr (Restricted) if (resident[lane + i * 32] < 0) values[i] = -INFINITY;
    }
    __syncthreads();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
        if constexpr (Restricted) if (resident[lane + i * 32] < 0) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            // Deliberately accumulate by WINNING EXPERT lane, not output rank.
            // Multiple selected experts in one lane add in selection order.
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
__global__ void closed_drop(int32_t* ids, float* weights, const int32_t* res,
                             const int32_t* mode, int n) {
    if (*mode != 2) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || res[ids[i]] >= 0) return;
    // Zero-weight dummy resident IDs preserve the grouped planner contract.
    int fallback = 0; while (fallback < 512 && res[fallback] < 0) ++fallback;
    if (fallback < 512) { ids[i] = fallback; weights[i] = 0.0f; }
}
__global__ void rank_keep(float* weights, int n, int keep, int renorm, float min_w, int min_keep) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float* row = weights + (size_t) t * 10;
    float in[10], kept = 0.0f;
    for (int k = 0; k < 10; ++k) in[k] = row[k];
    for (int k = 0; k < 10; ++k) {
        int rank = 0;
        for (int j = 0; j < 10; ++j) rank += in[j] > in[k] || (in[j] == in[k] && j < k);
        row[k] = rank < keep && (rank < min_keep || in[k] >= min_w) ? in[k] : 0.0f;
        kept += row[k];
    }
    if (renorm && kept > 0.0f)
        for (int k = 0; k < 10; ++k) row[k] /= kept;
}
__global__ void la_top20_kernel(const float* logits, int32_t* pred) {
    const int t = blockIdx.x, lane = threadIdx.x;   // one warp per token
    __shared__ float v[512];
    for (int i = lane; i < 512; i += 32) v[i] = logits[(size_t) t * 512 + i];
    __syncwarp();
    for (int r = 0; r < 20; ++r) {
        float best = -INFINITY;
        int id = 512;
        for (int i = lane; i < 512; i += 32)
            if (v[i] > best || (v[i] == best && i < id)) { best = v[i]; id = i; }
        for (int m = 16; m; m >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, m);
            const int oi = __shfl_xor_sync(0xffffffffu, id, m);
            if (ob > best || (ob == best && oi < id)) { best = ob; id = oi; }
        }
        if (id >= 512) id = 0;
        if (lane == 0) { pred[t * 20 + r] = id; v[id] = -INFINITY; }
        __syncwarp();
    }
}
__global__ void la_compare_kernel(const int32_t* ids, const int32_t* pred, const int32_t* res,
                                  unsigned long long* cnt, int n) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    unsigned long long h10 = 0, h20 = 0, nr = 0, nrh = 0;
    for (int k = 0; k < 10; ++k) {
        const int e = ids[t * 10 + k];
        if (e < 0 || e >= 512) continue;
        bool i10 = false, i20 = false;
        for (int j = 0; j < 20; ++j)
            if (pred[t * 20 + j] == e) { i20 = true; if (j < 10) i10 = true; }
        h10 += i10; h20 += i20;
        const bool nres = res != nullptr && res[e] < 0;
        nr += nres; nrh += nres && i20;
    }
    atomicAdd(cnt + 0, 1ull); atomicAdd(cnt + 1, h10); atomicAdd(cnt + 2, h20);
    atomicAdd(cnt + 3, nr); atomicAdd(cnt + 4, nrh);
}
__global__ void count_closed(const int32_t* ids, float* counts, int stride,
                             const int32_t* keep_d, int keep_h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, l = blockIdx.y;
    const int keep = keep_d ? *keep_d : keep_h;
    if (i >= keep * 10) return;
    const int id = ids[(size_t) l * stride * 10 + i];
    if (id >= 0 && id < 512) atomicAdd(counts + l * 512 + id, 1.0f);
}
// docs/PREFILL_SEED.md: one layer's prompt routing added to the per-expert mass and count (router_mass_count)
__global__ void mass_count(const int32_t* ids, const float* weights, float* mass, float* count, int n, int n_expert) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int id = ids[i];
    const float w = weights[i];
    if (id < 0 || id >= n_expert || !(w > 0.0f) || !isfinite(w)) return;
    atomicAdd(mass + id, w);
    atomicAdd(count + id, 1.0f);
}
__launch_bounds__(256, 1)
__global__ void route_multi(const float* __restrict__ logits, int32_t* __restrict__ ids,
                            float* __restrict__ weights, int n_tok) {
    const int tk = (int) blockIdx.x * 8 + (int) threadIdx.y;
    if (tk >= n_tok) return;
    logits += (size_t) tk * 512; ids += (size_t) tk * 10; weights += (size_t) tk * 10;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    selected_sum = max(warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}
void closed_router_apply(float* logits, int32_t* ids, float* weights, int32_t* original_ids,
                         const int32_t* resident, const int32_t* mode, int n_tok, void* stream) {
    if (!stream || !logits || !ids || !weights || !original_ids || !resident || !mode || n_tok < 1 || n_tok > 8)
        throw std::invalid_argument("closed router: invalid spans/window");
    const auto cs = static_cast<cudaStream_t>(stream);
    auto e = cudaMemcpyAsync(original_ids, ids, (size_t) n_tok * 10 * 4, cudaMemcpyDeviceToDevice, cs);
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    route<true><<<(unsigned) n_tok, dim3(32,8), 0, cs>>>(logits, ids, weights, resident, mode);
    closed_drop<<<1,128,0,cs>>>(ids,weights,resident,mode,n_tok*10);
    e = cudaGetLastError(); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void routed_rank_keep(float* weights, int n_tok, int keep, bool renorm, float min_w, int min_keep, void* stream) {
    if (!weights || !stream || n_tok < 1 || keep < 1 || keep > 10 || min_keep < 1 || min_keep > 10 ||
        !(min_w >= 0.0f && min_w < 1.0f))
        throw std::invalid_argument("routed rank keep: invalid window, keep or threshold");
    rank_keep<<<(unsigned) ((n_tok + 63) / 64), 64, 0, static_cast<cudaStream_t>(stream)>>>(
        weights, n_tok, keep, renorm ? 1 : 0, min_w, min_keep);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void routed_lookahead_top20(const float* logits, int32_t* pred, int n_tok, void* stream) {
    if (!logits || !pred || !stream || n_tok < 1) throw std::invalid_argument("lookahead top20: invalid window");
    la_top20_kernel<<<(unsigned) n_tok, 32, 0, static_cast<cudaStream_t>(stream)>>>(logits, pred);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void routed_lookahead_compare(const int32_t* ids, const int32_t* pred, const int32_t* res, unsigned long long* cnt,
                              int n_tok, void* stream) {
    if (!ids || !pred || !cnt || !stream || n_tok < 1) throw std::invalid_argument("lookahead compare: invalid window");
    la_compare_kernel<<<1, 64, 0, static_cast<cudaStream_t>(stream)>>>(ids, pred, res, cnt, n_tok);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void closed_router_count(const int32_t* ids, float* counts, int layers, int stride,
                         const int32_t* keep_device, int keep_host, void* stream) {
    if (!ids || !counts || !stream || layers < 1 || layers > 48 || stride < 1 ||
        (!keep_device && (keep_host < 1 || keep_host > stride)))
        throw std::invalid_argument("closed router: invalid count geometry");
    count_closed<<<dim3((unsigned)((stride*10+255)/256),(unsigned)layers),256,0,
                   static_cast<cudaStream_t>(stream)>>>(ids,counts,stride,keep_device,keep_host);
    auto e=cudaGetLastError(); if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void router_mass_count(const int32_t* ids, const float* weights, float* mass, float* count, int n_tok, int n_expert,
                       void* stream) {
    if (!ids || !weights || !mass || !count || !stream || n_tok < 1 || n_expert < 1)
        throw std::invalid_argument("router mass: invalid window");
    const int n = n_tok * 10;
    mass_count<<<(unsigned) ((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(ids, weights, mass, count,
                                                                                            n, n_expert);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4)
        || overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4)
        || overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route<false><<<1, dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    route_multi<<<(unsigned) ((n_tok + 7) / 8), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, n_tok);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
