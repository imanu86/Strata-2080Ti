// src/kernels/cuda/topk_lse.cu - see include/strata/kernels/topk_lse.hpp.
#include "strata/kernels/topk_lse.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

constexpr int kThreads = 128;
constexpr int kMaxK = 32;

__device__ __forceinline__ bool better(float xv, int xi, float yv, int yi) {
    return xv > yv || (xv == yv && xi < yi);
}

// One block a row.  Pass 1: every thread keeps the K best of its strided slice (sorted, registers / local memory);
// K rounds of a block-wide argmax over the threads' current heads merge them.  Pass 2: the row's sum of exp(x - max)
// for the log-sum-exp (float, a block reduction).  Row max = the merged top-1.
__global__ void topk_lse_kernel(const float* __restrict__ logits, int n, int K, const int32_t* __restrict__ next,
                                int32_t* __restrict__ ids, float* __restrict__ lps, float* __restrict__ nll) {
    const int r = blockIdx.x;
    const float* row = logits + (size_t) r * (size_t) n;
    float lv[kMaxK];
    int li[kMaxK];
    for (int i = 0; i < kMaxK; ++i) { lv[i] = -CUDART_INF_F; li[i] = 0x7fffffff; }
    for (int v = threadIdx.x; v < n; v += kThreads) {
        const float x = row[v];
        if (x != x) continue;   // NaN never picked
        if (!better(x, v, lv[K - 1], li[K - 1])) continue;
        int i = K - 1;
        while (i > 0 && better(x, v, lv[i - 1], li[i - 1])) { lv[i] = lv[i - 1]; li[i] = li[i - 1]; --i; }
        lv[i] = x; li[i] = v;
    }
    __shared__ float sv[kThreads];
    __shared__ int si[kThreads];
    __shared__ float out_v[kMaxK];
    __shared__ int out_i[kMaxK];
    int cur = 0;
    for (int k = 0; k < K; ++k) {
        sv[threadIdx.x] = cur < K ? lv[cur] : -CUDART_INF_F;
        si[threadIdx.x] = cur < K ? li[cur] : 0x7fffffff;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                const float ov = sv[threadIdx.x + s];
                const int oi = si[threadIdx.x + s];
                if (better(ov, oi, sv[threadIdx.x], si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
            }
            __syncthreads();
        }
        const float wv = sv[0];
        const int wi = si[0];
        if (threadIdx.x == 0) { out_v[k] = wv; out_i[k] = wi; }
        if (cur < K && li[cur] == wi && lv[cur] == wv) ++cur;   // the winner's thread advances
        __syncthreads();
    }
    const float mx = out_v[0] == -CUDART_INF_F ? 0.0f : out_v[0];
    float sum = 0.0f;
    for (int v = threadIdx.x; v < n; v += kThreads) {
        const float x = row[v];
        if (x == x) sum += __expf(x - mx);
    }
    sv[threadIdx.x] = sum;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sv[threadIdx.x] += sv[threadIdx.x + s];
        __syncthreads();
    }
    const float lse = mx + logf(sv[0]);
    if (threadIdx.x < K) {
        const int i = threadIdx.x;
        const bool have = out_i[i] != 0x7fffffff;
        ids[(size_t) r * K + i] = have ? out_i[i] : -1;
        lps[(size_t) r * K + i] = have ? out_v[i] - lse : -CUDART_INF_F;
    }
    if (threadIdx.x == 0 && nll != nullptr) {
        const int32_t nx = next != nullptr ? next[r] : -1;
        nll[r] = nx >= 0 && nx < n ? row[nx] - lse : nanf("");
    }
}

}  // namespace

void row_topk_lse(const float* logits, int n_rows, int n, int K, const int32_t* next, int32_t* ids, float* lps,
                  float* nll, void* stream) {
    if (n_rows <= 0) return;
    if (K < 1 || K > kMaxK || n < 1) throw std::invalid_argument("row_topk_lse: K must be 1..32 and n positive");
    topk_lse_kernel<<<n_rows, kThreads, 0, (cudaStream_t) stream>>>(logits, n, K, next, ids, lps, nll);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("row_topk_lse: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels
