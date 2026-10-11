// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu}.
//
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

#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/q8_1_finite.hpp"   // chainfuse: the q8_1 quantizer's helpers (native_mmvq.cu's)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
template<int BlockSize>
__global__ void norm(const float* input, const float* __restrict__ gamma, float* output,
                     int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row_offset = std::size_t(blockIdx.x) * n_cols;
    input += row_offset; output += row_offset;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    // All reads of input for the reduction precede this barrier. Afterwards,
    // each thread reads/writes only its own elements, allowing exact in-place use.
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
// As norm, with gamma[groups * n_cols]: row r is group r % groups and reads its own slice of gamma.
template<int BlockSize>
__global__ void norm_grouped(const float* input, const float* __restrict__ gamma, float* output,
                             int n_cols, int groups, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row_offset = std::size_t(blockIdx.x) * n_cols;
    input += row_offset; output += row_offset;
    gamma += std::size_t(blockIdx.x % unsigned(groups)) * n_cols;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize)
        output[col] = scale * input[col] * gamma[col];
}
__global__ void gate(const float* attn, const float* __restrict__ q_full, float* output,
                     int n_head, int head_dim) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_head) * head_dim) return;
    const std::size_t head = i / head_dim, channel = i % head_dim;
    const float raw = q_full[head * 2 * head_dim + head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + expf(-raw));
    output[i] = attn[i] * sigmoid;
}
std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes,
             float* output, void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument("native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
void check_launch() {
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("native QSA launch: ") + cudaGetErrorString(result));
}
} // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    if (n_cols < 1024)
        norm<256><<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols, epsilon);
    else
        norm<1024><<<unsigned(n_rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols, epsilon);
    check_launch();
}
void native_qsa_rms_norm_grouped(const float* input, const float* gamma, float* output,
                                 int n_cols, int groups, int n_rows, float epsilon, void* stream) {
    if (groups < 1) throw std::invalid_argument("native QSA requires positive bounded dimensions");
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * groups * 4, output, stream);
    if (n_cols < 1024)
        norm_grouped<256><<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols,
                                                                                         groups, epsilon);
    else
        norm_grouped<1024><<<unsigned(n_rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(input, gamma, output, n_cols,
                                                                                           groups, epsilon);
    check_launch();
}
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    gate<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(attn, q_full, output, n_head, head_dim);
    check_launch();
}

// ======================= chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1) ==========================================
// The MTP drafter's chain quantizes the norm's and the gate's outputs to q8_1 right away (native_quantize_q8_1_kernel,
// native_mmvq.cu: thread i is element i, so a warp is one 32-element block; amax = warp_max, sum = warp_sum of the raw
// values, d = q8_1_finite(amax / 127), q = roundf(x / d) clamped, ds = (d, sum) as fp16).  Here the quantizer's
// expressions run in the producing kernel's epilogue on the same lanes: in `norm` lane l of warp w holds column
// 32 w + l + BlockSize k, so a warp's 32 columns are exactly one block in the quantizer's lane order (n_cols is a
// multiple of 32, so a warp is whole or absent in every iteration); in `gate` thread i is element i as there.
// This file compiles with --use_fast_math like native_mmvq.cu, and q8_1_finite.hpp's helpers inline under the same
// flags: the same bytes as the two launches.  Parity: src/kernels/chain_step_bench.cpp.
namespace {
struct Q81Out { __half2 ds; int8_t qs[32]; };   // native_mmvq.cu's Q81Block layout (36 bytes)
static_assert(sizeof(Q81Out) == 36 && alignof(Q81Out) == 4, "q8_1 block layout");
__device__ __forceinline__ float warp_max(float value) {   // native_mmvq.cu's warp_max: the same pairing order
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, 32));
    return value;
}
// the quantizer's block on the warp's 32 lanes: `v` is this lane's value, `i` its element index
__device__ __forceinline__ void q8_1_block(float v, std::size_t i, Q81Out* y) {
    const float amax = warp_max(fabsf(v));
    const float sum = warp_sum(v);
    const float d = q8_1_finite(amax / 127.0f);
    const int8_t q = q8_1_quant(v, d, amax);
    y[i / 32].qs[i % 32] = q;
    if (i % 32 == 0) y[i / 32].ds = q8_1_ds(d, sum);
}
template<int BlockSize>
__global__ void norm_q8_1(const float* input, const float* __restrict__ gamma, float* output, Q81Out* y,
                          int n_cols, float epsilon) {
    const int tid = threadIdx.x;
    const std::size_t row_offset = std::size_t(blockIdx.x) * n_cols;
    input += row_offset;
    if (output != nullptr) output += row_offset;
    y += row_offset / 32;
    float partial = 0.0f;
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float x = input[col];
        partial += x * x;
    }
    __shared__ float sums[32];
    partial = warp_sum(partial);
    const int lane = tid % 32;
    if (lane == 0) sums[tid / 32] = partial;
    __syncthreads();
    partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
    partial = warp_sum(partial);
    const float mean = partial / n_cols;
    const float scale = rsqrtf(mean + epsilon);
    for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
        const float v = scale * input[col] * gamma[col];
        if (output != nullptr) output[col] = v;
        q8_1_block(v, col, y);
    }
}
__global__ void gate_q8_1(const float* attn, const float* __restrict__ q_full, float* output, Q81Out* y,
                          int n_head, int head_dim) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(n_head) * head_dim) return;   // a multiple of 32: whole warps return
    const std::size_t head = i / head_dim, channel = i % head_dim;
    const float raw = q_full[head * 2 * head_dim + head_dim + channel];
    const float sigmoid = 1.0f / (1.0f + expf(-raw));
    const float v = attn[i] * sigmoid;
    if (output != nullptr) output[i] = v;
    q8_1_block(v, i, y);
}
} // namespace

void native_qsa_rms_norm_q8_1(const float* input, const float* gamma, float* output, void* x_q8_1,
                              int n_cols, int n_rows, float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    if (n_cols % 32 != 0 || !valid(x_q8_1, count / 32 * sizeof(Q81Out)))
        throw std::invalid_argument("native QSA norm+q8_1 requires n_cols a multiple of 32 and an aligned q8_1 span");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output != nullptr ? output : const_cast<float*>(input), stream);
    if (n_cols < 1024)
        norm_q8_1<256><<<unsigned(n_rows), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gamma, output, static_cast<Q81Out*>(x_q8_1), n_cols, epsilon);
    else
        norm_q8_1<1024><<<unsigned(n_rows), 1024, 0, static_cast<cudaStream_t>(stream)>>>(
            input, gamma, output, static_cast<Q81Out*>(x_q8_1), n_cols, epsilon);
    check_launch();
}
void native_qsa_gate_q8_1(const float* attn, const float* q_full, float* output, void* x_q8_1,
                          int n_head, int head_dim, void* stream) {
    const auto count = elements(head_dim, n_head);
    if (count % 32 != 0 || !valid(x_q8_1, count / 32 * sizeof(Q81Out)))
        throw std::invalid_argument("native QSA gate+q8_1 requires n_head * head_dim a multiple of 32 and an aligned q8_1 span");
    buffers(attn, count * 4, q_full, count * 8, output != nullptr ? output : const_cast<float*>(attn), stream);
    gate_q8_1<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        attn, q_full, output, static_cast<Q81Out*>(x_q8_1), n_head, head_dim);
    check_launch();
}
} // namespace strata::kernels
