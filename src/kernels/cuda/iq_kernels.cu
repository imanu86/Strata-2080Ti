// src/kernels/cuda/iq_kernels.cu - see include/strata/kernels/iq_kernels.hpp.
//
// The dot products (vec_dot_*_q8_1), the dequantizers and the q8_1 quantizer are transcribed from llama.cpp
// (ggml/src/ggml-cuda/vecdotq.cuh, dequantize.cuh, quantize.cu at the commit in third_party/ggml/VERSION.txt;
// MIT license, third_party/ggml/LICENSE).  The block structs and codebook grids come from its ggml-common.h,
// included unchanged.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/q8_1_finite.hpp"
#include "s26_tsum.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "ggml-common.h"

#include <cstdio>
#include <algorithm>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
__device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
__device__ __forceinline__ int get_int_b4(const void* x, const int& i32) { return ((const int*) x)[i32]; }
__device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
__device__ __forceinline__ int2 get_int_from_table_16(const int& q4, const int8_t* table) {
#if defined(STRATA_HIP_GFX906)
    // AMD: llama.cpp's HIP lookup (vecdotq.cuh) - v_perm_b32 takes 3-bit byte indices, so the low and high halves
    // of the table are looked up and the index MSB picks between them: 4 perms per 8 values.
    const uint32_t* v32 = reinterpret_cast<const uint32_t*>(table);
    const uint32_t q_even = (uint32_t) q4, q_odd = (uint32_t) q4 >> 4;
    const uint32_t el = __builtin_amdgcn_perm(v32[1], v32[0], q_even & 0x07070707u);
    const uint32_t ol = __builtin_amdgcn_perm(v32[1], v32[0], q_odd & 0x07070707u);
    const uint32_t eh = __builtin_amdgcn_perm(v32[3], v32[2], q_even & 0x07070707u);
    const uint32_t oh = __builtin_amdgcn_perm(v32[3], v32[2], q_odd & 0x07070707u);
    return make_int2((int) __builtin_amdgcn_perm(eh, el, 0x03020100u | ((q_even & 0x08080808u) >> 1)),
                     (int) __builtin_amdgcn_perm(oh, ol, 0x03020100u | ((q_odd & 0x08080808u) >> 1)));
#else
    const uint32_t* table32 = (const uint32_t*) table;
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
#endif
}
#define ggml_cuda_dp4a(a, b, c) STRATA_DP4A((a), (b), (c))

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
__device__ __forceinline__ float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = __byte_perm(qe, qo, 0x5140);
        const int qy = __byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = __low2float(bq8_1_chunk->ds);
    return d2 * d8 * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xxs_grid)[aux8[k0 / 2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t* q2 = (const uint16_t*) &q2_packed;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*) iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (iq2s_grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                      const int& kbx, const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
}

__device__ __forceinline__ float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                    const int& kbx, const int& iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const int qs_packed = get_int_b4(bq1->qs, iqs);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[qs[l0 / 2] | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] += delta * sumy;
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    const float d = __half2float(scale.f16) * __low2float(bq8_1[iqs].ds);
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return d * ((sumi[0] + sumf[0]) * sc0 + (sumi[1] + sumf[1]) * sc1);
}

__device__ __forceinline__ float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = __half2float(bq4->d) * __low2float(bq8_1->ds);
    return d * sumi;
}

// IQ4_XS: 256 values as 8 sub-blocks of 32 (6-bit scale each); one call covers one sub-block (iqs = 4 * sub-block),
// and `bq8_1` is the super-block's first q8_1 block, so the call's activation is bq8_1[iqs / 4].  The GSQ-RCO IQ3_S
// file keeps one layer's routed gate/up experts in this format.
__device__ __forceinline__ float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                     const int& kbx, const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = __half2float(bq4->d) * __low2float(bq8_1[iqs / 4].ds);
    return d * sumi;
}

// ---------------------------------------------------------------- Unsloth's UD-Q4_K_XL experts
// Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down: llama.cpp's vec_dot_*_q8_1 (vecdotq.cuh, VDR 2 each), transcribed; the
// Q5_1 min term is the one departure (below).  Via eddoursul/Strata 8029fa9 (iq_dot.cuh) and #255 (Q8_0,
// gopinath87607), which agree with llama.cpp and with each other.
constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q5_0 = 2, VDR_Q4_1 = 2, VDR_Q4_0 = 2, VDR_Q8_0 = 2, VDR_Q6_K = 1;

__device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(const int* __restrict__ v, const int* __restrict__ u,
                                                             const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
                                                             const half2& dm4, const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);   // the min times the sum of the QUANTIZED activations
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}
__device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(const int* __restrict__ vl, const int* __restrict__ vh,
                                                             const int* __restrict__ u, const uint8_t* __restrict__ sc,
                                                             const uint8_t* __restrict__ m, const half2& dm5,
                                                             const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm5f = __half22float2(dm5);
    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}
// the 6-bit scales and mins of the 32-value group pair bq8_offset / 2, branchless (llama.cpp; shared by Q4_K, Q5_K)
__device__ __forceinline__ void k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}
__device__ __forceinline__ float vec_dot_q4_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scale_min(bq4_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
__device__ __forceinline__ float vec_dot_q5_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min(bq5_K->scales, bq8_offset, aux);
    const uint8_t* sc = (const uint8_t*) aux;
    const uint8_t* m = sc + 2;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}
// Q5_1: llama.cpp's integer chain, but the min term multiplies the sum of the QUANTIZED activations (dp4a with
// 0x01010101, times d8) instead of the q8_1 block's `ds.y`, which our quantizer (like llama.cpp's) fills with the sum
// of the ORIGINAL activations.  That is ggml-cpu's convention (its q8_1 `s` is d * sum(q)) and the one the K-quant
// mins above use; the scaled and the min term then see the same activation (eddoursul/Strata measured 1.1-1.2%
// against 1.9% relative error per expert).  Result: sumi * (d5 * d8) + sumu * (m5 * d8).
// Q5_0: llama.cpp's integer chain verbatim (vecdotq.cuh vec_dot_q5_0_q8_1_impl) -- symmetric (one delta, no
// learned min), so unlike Q5_1 there is no min-term rounding choice to make: the constant -16 per-weight offset
// is accounted for via the q8_1 block's `ds.y` (sum of the ORIGINAL activations), exactly as upstream.
__device__ __forceinline__ float vec_dot_q5_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
    }
    const float d5 = __half2float(bq5_0->d);
    const float d8 = __low2float(bq8_1->ds);
    const float s8 = __high2float(bq8_1->ds);
    return d5 * (sumi * d8 - (16.0f * VDR_Q5_0 / QI5_0) * s8);
}
// Q4_0 (plain llama-quantize Q4_0 files, e.g. bartowski's: gate/up and most down projections): the codes are centred
// (q - 8 as signed bytes) before the dot, the exact integer form ggml-cpu's q4_0 x q8_0 dot uses, rather than
// llama.cpp's CUDA chain (uncentred codes minus 8 x the q8_1 block's float sum `ds.y`).  On a real Q4_0 expert
// (2560 x 640) the CUDA chain measured 1.4e-2 relative error against the float product, the centred form 5.3e-3;
// with the CUDA chain native_expert_parity failed its 3e-2 limit on two layers of a Q4_0 Flash-Next file.  The
// centring is the same idea as the Q5_1 min term above: the offset is applied to the activations' own int8 codes.
__device__ __forceinline__ int q4_0_centred(const int v) {
    // Unsigned: (b & 0x08080808) * 0x1E reaches 0xF0F0F0F0, which overflows a signed int - undefined behaviour the
    // sm_60 build (no __dp4a, strata_dp4a fallback) optimised into a wrong value (native_expert_parity gpu rel 1.7e4).
    const uint32_t b = (uint32_t) v ^ 0x08080808u;   // per byte: 0..15 -> (q - 8) as a 4-bit two's complement value
    return (int) (b | ((b & 0x08080808u) * 0x1Eu));  // sign-extend each nibble into its byte (0x08 * 0x1E = 0xF0)
}
__device__ __forceinline__ float vec_dot_q4_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_0* bq4 = (const block_q4_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_0; ++i) {
        const int v = get_int_b2(bq4->qs, iqs + i);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 0) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i), sumi);
        sumi = ggml_cuda_dp4a(q4_0_centred((v >> 4) & 0x0F0F0F0F), get_int_b4(bq8_1->qs, iqs + i + QI4_0), sumi);
    }
    return __half2float(bq4->d) * __low2float(bq8_1->ds) * sumi;
}
// Q4_1 (llama-quantize puts it on a few down projections of its Q4_0 files): the Q5_1 chain above without the fifth
// bit, the same min-term choice (the activations' own int8 codes): sumi * (d4 * d8) + sumu * (m4 * d8).
__device__ __forceinline__ float vec_dot_q4_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q4_1* bq4_1 = (const block_q4_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q4_1; ++i) {
        const int v = get_int_b4(bq4_1->qs, iqs + i);
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI4_1);
        sumi = ggml_cuda_dp4a((v >> 0) & 0x0F0F0F0F, u0, sumi);
        sumi = ggml_cuda_dp4a((v >> 4) & 0x0F0F0F0F, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm4 = __half22float2(bq4_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm4.x * d8) + sumu * (dm4.y * d8);
}
__device__ __forceinline__ float vec_dot_q5_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        int vi0 = (vl >> 0) & 0x0F0F0F0F;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi0, u0, sumi);
        int vi1 = (vl >> 4) & 0x0F0F0F0F;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = ggml_cuda_dp4a(vi1, u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float2 dm5 = __half22float2(bq5_1->dm);
    const float d8 = __low2float(bq8_1->ds);
    return sumi * (dm5.x * d8) + sumu * (dm5.y * d8);
}
// Q6_K: llama.cpp's integer chain verbatim (vecdotq.cuh vec_dot_q6_K_q8_1_impl_mmvq / vec_dot_q6_K_q8_1) -- signed
// per-16-element scales, the -32 offset folded into the per-byte value, the q8_1 block's ds.low for the scales,
// upstream exactly as pulled (no min-term choice exists: Q6_K has no learned min).
__device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(const int vl, const int vh, const int* __restrict__ u,
                                                             const int8_t* __restrict__ scales, const float d,
                                                             const float* __restrict__ d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);   // vi = (vil | vih) - 32
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}
__device__ __forceinline__ float vec_dot_q6_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q6_K* bq6_K = (const block_q6_K*) vbq + kbx;
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
    const int8_t* scales = bq6_K->scales + scale_offset;
    int u[QR6_K];
    float d8[QR6_K];
#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i] = get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2 * i].ds);
    }
    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, __half2float(bq6_K->d), d8);
}
__device__ __forceinline__ float vec_dot_q8_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1,
                                                   const int& kbx, const int& iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i)
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    const float d8_0 = __half2float(bq8_0->d), d8_1 = __low2float(bq8_1->ds);
    return d8_0 * d8_1 * ((float) sumi);
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / VDR_Q4_K, step = VDR_Q4_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / VDR_Q5_K, step = VDR_Q5_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<14> { static constexpr int qk = 256, ipb = QI6_K / VDR_Q6_K, step = VDR_Q6_K;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q6_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<2> { static constexpr int qk = 32, ipb = QI4_0 / VDR_Q4_0, step = VDR_Q4_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<3> { static constexpr int qk = 32, ipb = QI4_1 / VDR_Q4_1, step = VDR_Q4_1;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = QI5_0 / VDR_Q5_0, step = VDR_Q5_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    __device__ static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };

// The formats of each role, one list each so a type cannot be in one switch and missing from another.  Every
// entry is a kernel template for each CUDA architecture of the build, hence two lists rather than one.
#ifdef STRATA_Q6K_EXPERTS   // opt-in build (-DSTRATA_Q6K_EXPERTS=ON): one more instance per kernel, loaded at start
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(14) X(6) X(2) X(3) X(8)
#else
#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(6) X(2) X(3) X(8)
#endif
#define STRATA_D_FMTS(X) X(20) X(23) X(42) X(7) X(6) X(2) X(3) X(8)
#define STRATA_MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(6) X(2) X(3) X(8)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One row against one q8_1 activation, the whole warp: call k = (block, part) is lane-strided.
template<int TY>
__device__ __forceinline__ float row_dot(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
    return warp_sum(s);
}

template<int TY>
__global__ void __launch_bounds__(128) mmvq_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                   const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in,
                                                   int n_out, int ncols) {
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = row_dot<TY>(wr, x + (size_t) c * (n_in / 32), nb, lane);
        if (lane == 0) y[(size_t) c * n_out + row] = s;
    }
}

// ---------------------------------------------------------------- decode once, apply to every column
// The kernels above call Fmt<TY>::dot once per (call, column): each column re-reads the weight words and redoes
// the grid lookups and sign unpacking.  Here each dot is split, as native_mmvq.cu's multi-column traits are, into
// `load` (everything that depends only on the weight: the signed grid words, the integer scales, the fp16 block
// scale as a float) and `apply` (the activation loads, the dp4a chain in the same order, the same integer scale
// step and the same float expression).  `apply(load(...))` does the dot's integer and float operations in the same
// order on the same values, so a column of the kernels below is BITWISE equal to the same column of mmvq_kernel /
// native_gu_kernel / native_down_kernel (iq_multi_parity checks it; STRATA_OLD_IQ_MMVQ=1 keeps the old kernels).
template<int TY> struct Split;
// Formats with a Split below take the decode-once kernels; the others (Q4_K, Q5_K, Q5_1, Q8_0: UD-Q4_K_XL) the
// per-entry ones, which call Fmt<TY>::dot per column exactly as before #242 (the launchers test kSplit at compile
// time, so the multi kernels are never instantiated for a type without a Split).
template<int TY> inline constexpr bool kSplit = false;
template<int TY> inline constexpr bool kStageIqGrid = (TY == 16 || TY == 17 || TY == 22 || TY == 29);

template<> inline constexpr bool kSplit<16> = true;
template<> struct Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const int q2 = get_int_b2(bq2->qs, iqs);
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        const uint2* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint2*) s_grid;
        else grid_lut = (const uint2*) iq2xxs_grid;
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = grid_lut[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[k0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[k0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = __half2float(bq2->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = sumi * r.ls / 8;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
// IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
struct SplitLs2 {
    struct W { int g[8]; int ls0, ls1; float dw; };
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<17> = true;
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
        const uint16_t* q2 = (const uint16_t*) &q2_packed;
        const uint2* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint2*) s_grid;
        else grid_lut = (const uint2*) iq2xs_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid_lut[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.dw = __half2float(bq2->d);
        return r;
    }
};
template<> inline constexpr bool kSplit<22> = true;
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq2->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const uint64_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = (const uint64_t*) s_grid;
        else grid_lut = iq2s_grid;
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (grid_lut + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos[0] ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos[1] ^ signs1, signs1);
        }
        r.dw = __half2float(bq2->d);
        return r;
    }
};
template<> inline constexpr bool kSplit<18> = true;
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 28;
        r.dw = __half2float(bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<21> = true;
template<> struct Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = __half2float(bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<29> = true;
template<> struct Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
        const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
        const int qs_packed = get_int_b4(bq1->qs, iqs);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const uint32_t* grid_lut;
        if constexpr (STAGE_GRID) grid_lut = s_grid;
        else grid_lut = iq1s_grid_gpu;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = grid_lut[qs[l0 / 2] | ((qhl & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
        r.dw = __half2float(scale.f16);
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
            sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] += r.delta[l0 / 2] * sumy;
        }
        const float d = r.dw * __low2float(bq8_1[iqs].ds);
        return d * ((sumi[0] + sumf[0]) * r.sc0 + (sumi[1] + sumf[1]) * r.sc1);
    }
};
template<> inline constexpr bool kSplit<20> = true;
template<> struct Split<20> {   // IQ4_NL
    struct W { int2 v[2]; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
        r.dw = __half2float(bq4->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        const float d = r.dw * __low2float(bq8_1->ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<23> = true;
template<> struct Split<23> {   // IQ4_XS
    struct W { int2 v[4]; int ls; float dw; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
        r.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(bq4->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
            const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
            sumi = ggml_cuda_dp4a(r.v[j].x, u0, sumi);
            sumi = ggml_cuda_dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(bq8_1[iqs / 4].ds);
        return d * sumi;
    }
};
template<> inline constexpr bool kSplit<42> = true;
template<> struct Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d2; };
    template<bool STAGE_GRID = false>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        W r;
        r.d2 = bq2_0->d;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            r.qx[j] = __byte_perm(qe, qo, 0x5140);
            r.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
            const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
            sumi = ggml_cuda_dp4a(u, r.qx[j], sumi);
            sumi = ggml_cuda_dp4a(v, r.qy[j], sumi);
        }
        const float d8 = __low2float(bq8_1_chunk->ds);
        return r.d2 * d8 * sumi;
    }
};

// One row against the n <= NC activations x + off[0..n) (n >= 1, warp-uniform; offsets in q8_1 blocks, 32-bit to
// spare registers), the whole warp.  Per activation this is row_dot: the same calls k, lane-strided the same way,
// summed in the same order, then the same warp_sum.  Only the weight side moves out of the per-activation loop.
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_multi(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                              int nb, int lane, float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c)
        if (EXACT_N || c < n) s[c] = warp_sum(s[c]);
}

// 8-lane sub-warp row dot for K = nb * ipb == 40 (TD == 20, IQ4_NL with n_ff == 640):
// 4 rows per warp (32 rows per 256-thread block), 100% active lanes, bitwise identical addition tree to row_dot_multi.
template<int TY, int NC, bool EXACT_N = false>
__device__ __forceinline__ void row_dot_40_sub8(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                int t, float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = t; k < 40; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (EXACT_N || c < n) {
                float s16 = 0.0f;
                s16 += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
                s[c] = __fadd_rn(s[c], s16);
            }
        }
    }
    float s8[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s8[c] = 0.0f;
    {
        const int k = t + 8, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s8[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    {
        const int k = t + 24, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (EXACT_N || c < n) {
                float s24 = 0.0f;
                s24 += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
                s[c] = __fadd_rn(s[c], __fadd_rn(s8[c], s24));
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

// 16-lane sub-warp row dot for K = nb * ipb == 20 (TD == 42, Q2_0 with n_ff == 640):
// 2 rows per warp (16 rows per 256-thread block), bitwise identical addition tree to row_dot_multi.
template<int TY, int NC, bool EXACT_N = false>
__device__ __forceinline__ void row_dot_20_sub16(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                 int t, float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    {
        const int k = t, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    if (t < 4) {
        const int k = t + 16, kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

// 16-lane sub-warp row dot for K = nb * ipb == 80 (all kSplit gate/up formats with n_embd == 2560):
// 2 rows per warp (16 rows per 256-thread block), 100% active lanes, bitwise identical addition tree to row_dot_multi.
template<int TY, int NC, bool EXACT_N = false, bool STAGE_GRID = false>
__device__ __forceinline__ void row_dot_80_sub16(const uint8_t* row, const block_q8_1* x, const int (&off)[NC], int n,
                                                 int t, float (&s)[NC], const uint32_t* __restrict__ s_grid = nullptr) {
    using F = Fmt<TY>;
    using S = Split<TY>;
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = 0.0f;
    for (int k = t; k < 80; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
    float s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) s16[c] = 0.0f;
    for (int k = t + 16; k < 64; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const typename S::W w = S::template load<STAGE_GRID>(row, kbx, iqs, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (EXACT_N || c < n) s16[c] += S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (EXACT_N || c < n) {
            s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        }
    }
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
__device__ __forceinline__ const uint32_t* stage_iq_grid(uint32_t* s_buf, int tid, int nthreads) {
    if constexpr (!STAGE_GRID) {
        return nullptr;
    } else if constexpr (TY == 16) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2xxs_grid;
        for (int i = tid; i < 256; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 17) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2xs_grid;
        for (int i = tid; i < 512; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 22) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq2s_grid;
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else if constexpr (TY == 29) {
        uint2* dst = (uint2*) s_buf;
        const uint2* src = (const uint2*) iq1s_grid_gpu;
        for (int i = tid; i < 1024; i += nthreads) dst[i] = src[i];
        __syncthreads();
        return s_buf;
    } else {
        return nullptr;
    }
}

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
struct IqGridWords {
    static constexpr int value = !STAGE_GRID ? 4 : (TY == 22 || TY == 29) ? 2048 : (TY == 17) ? 1024 : (TY == 16) ? 512 : 4;
};

// mmvq_kernel with the columns taken NC at a time.  After warp_sum every lane holds the same sum, so lane c stores
// column c.
// #778: `__shared__ alignas(16)` does not compile for gfx1030 (HIP); the attribute after the declarator does, but nvcc
// with MSVC as host does not take it.  Same layout either way.
#if defined(__HIPCC__)
#define STRATA_SHARED_ALIGN16_U32(name, ...) __shared__ uint32_t name[__VA_ARGS__] __attribute__((aligned(16)))
#else
#define STRATA_SHARED_ALIGN16_U32(name, ...) __shared__ alignas(16) uint32_t name[__VA_ARGS__]
#endif
template<int TY, int NC, bool STAGE_GRID = kStageIqGrid<TY>, bool EXACT_N = false>
__global__ void __launch_bounds__(128) mmvq_multi_kernel(const uint8_t* __restrict__ w, size_t row_bytes,
                                                         const block_q8_1* __restrict__ x, float* __restrict__ y,
                                                         int n_in, int n_out, int ncols) {
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, IqGridWords<TY, STAGE_GRID>::value);
    const uint32_t* s_grid = stage_iq_grid<TY, STAGE_GRID>(s_grid_buf, threadIdx.y * 32 + threadIdx.x, 128);
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int nb = n_in / Fmt<TY>::qk, xb = n_in / 32;
    const uint8_t* wr = w + (size_t) row * row_bytes;
    if constexpr (EXACT_N) {
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = c * xb;
        float s[NC];
        row_dot_multi<TY, NC, true, STAGE_GRID>(wr, x, off, NC, nb, lane, s, s_grid);
#pragma unroll
        for (int c = 0; c < NC; ++c)
            if (lane == c) y[(size_t) c * n_out + row] = s[c];
    } else {
        for (int c0 = 0; c0 < ncols; c0 += NC) {
            const int n = min(NC, ncols - c0);
            int off[NC];
#pragma unroll
            for (int c = 0; c < NC; ++c) off[c] = (c0 + min(c, n - 1)) * xb;
            float s[NC];
            row_dot_multi<TY, NC, false, STAGE_GRID>(wr, x, off, n, nb, lane, s, s_grid);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n && lane == c) y[(size_t) (c0 + c) * n_out + row] = s[c];
        }
    }
}

// ---------------------------------------------------------------- grouped native experts
constexpr int GU_ROWS = 8;     // rows per block (one warp each)

// Group g is computed by block row g % gridDim.y: rows y, y + gridDim.y, ... < *n_groups.  The callers size the
// launch for the most groups a call can have (cap: every entry its own expert), but the count is only known on the
// device, and a verify window's PCIe call usually has no group at all: with one block row per possible group every
// unused one was cap x (2 n_ff / 8 + n_embd / 8) blocks that started only to return.  A smaller gridDim.y strides
// instead.  Each group's rows are computed by the same warp code in the same order either way, so the results do
// not depend on gridDim.y (gridDim.y = cap_groups is the one-row-per-group launch).
template<int TG>
__global__ void __launch_bounds__(256) native_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups,
                                                        const int32_t* __restrict__ ent_tok,
                                                        const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                        float* __restrict__ gate, float* __restrict__ up) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; ++e) {
            const float s = row_dot<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
            if (lane == 0) (is_up ? up : gate)[(size_t) e * L.n_ff + r] = s;
        }
    }
}

// native_gu_kernel with the group's entries taken GRP_NC at a time, each weight part decoded once per pass.
// A group has at most one entry per token of the window (kVerifyMaxT = 8; setup writes --spec 4, and a split window
// has halves of <= 4), so 4 takes such windows in one pass; 8 would take ~64-80 registers against ~48 (ptxas -v).
constexpr int GRP_NC = 4;

template<int TG, bool STAGE_GRID = kStageIqGrid<TG>, bool SUB16 = true>
__global__ void __launch_bounds__(256) native_gu_multi_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_tok,
                                                              const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                              float* __restrict__ gate, float* __restrict__ up) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, IqGridWords<TG, STAGE_GRID>::value);
    const uint32_t* s_grid = stage_iq_grid<TG, STAGE_GRID>(s_grid_buf, threadIdx.x, 256);
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    if constexpr (SUB16) {
        if (xb == 80) {
            const int subwarp = threadIdx.x >> 4, t = threadIdx.x & 15;
            const int row = blockIdx.x * 16 + subwarp;            // 0 .. 2*n_ff
            if (row >= 2 * L.n_ff) return;
            const bool is_up = row >= L.n_ff;
            const int r = is_up ? row - (int) L.n_ff : row;
            const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
            float* dst = is_up ? up : gate;
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    if (n == 1) {
                        const int off1[1] = { ent_tok[e] * 80 };
                        float s1[1];
                        row_dot_80_sub16<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, t, s1, s_grid);
                        if (t == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
                    } else if (n == 2) {
                        const int off2[2] = { ent_tok[e] * 80, ent_tok[e + 1] * 80 };
                        float s2[2];
                        row_dot_80_sub16<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, t, s2, s_grid);
                        if (t < 2) dst[(size_t) (e + t) * L.n_ff + r] = s2[t];
                    } else if (n == 3) {
                        const int off3[3] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80 };
                        float s3[3];
                        row_dot_80_sub16<TG, 3, true, STAGE_GRID>(wr, xq, off3, 3, t, s3, s_grid);
                        if (t < 3) dst[(size_t) (e + t) * L.n_ff + r] = s3[t];
                    } else {
                        const int off4[4] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80, ent_tok[e + 3] * 80 };
                        float s4[4];
                        row_dot_80_sub16<TG, 4, true, STAGE_GRID>(wr, xq, off4, 4, t, s4, s_grid);
                        if (t < 4) dst[(size_t) (e + t) * L.n_ff + r] = s4[t];
                    }
                }
            }
            return;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * GU_ROWS + warp;             // 0 .. 2*n_ff
    if (row >= 2 * L.n_ff) return;
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    float* dst = is_up ? up : gate;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            if (n == 1) {
                const int off1[1] = { ent_tok[e] * xb };
                float s1[1];
                row_dot_multi<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, nb, lane, s1, s_grid);
                if (lane == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
            } else if (n == 2) {
                const int off2[2] = { ent_tok[e] * xb, ent_tok[e + 1] * xb };
                float s2[2];
                row_dot_multi<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, nb, lane, s2, s_grid);
                if (lane < 2) dst[(size_t) (e + lane) * L.n_ff + r] = s2[lane];
            } else if (n == 3) {
                const int off3[3] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb };
                float s3[3];
                row_dot_multi<TG, 3, true, STAGE_GRID>(wr, xq, off3, 3, nb, lane, s3, s_grid);
                if (lane < 3) dst[(size_t) (e + lane) * L.n_ff + r] = s3[lane];
            } else {
                const int off4[4] = { ent_tok[e] * xb, ent_tok[e + 1] * xb, ent_tok[e + 2] * xb, ent_tok[e + 3] * xb };
                float s4[4];
                row_dot_multi<TG, 4, true, STAGE_GRID>(wr, xq, off4, 4, nb, lane, s4, s_grid);
                if (lane < 4) dst[(size_t) (e + lane) * L.n_ff + r] = s4[lane];
            }
        }
    }
}

__global__ void swiglu_entries_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                      long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    h[i] = (g / (1.0f + __expf(-g))) * up[i];
}

template<int TD>
__global__ void __launch_bounds__(256) native_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                          const int32_t* __restrict__ grp_start,
                                                          const int32_t* __restrict__ n_groups,
                                                          const int32_t* __restrict__ ent_dst,
                                                          const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                          float* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= L.n_embd) return;
    const size_t off = L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int ng = *n_groups;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; ++e) {
            const float s = row_dot<TD>(wr, hq + (size_t) e * hb, nb, lane);
            if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s;
        }
    }
}

// native_down_kernel with the entries taken GRP_NC at a time
template<int TD, bool SUB_DOWN = true>
__global__ void __launch_bounds__(256) native_down_multi_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                const int32_t* __restrict__ grp_start,
                                                                const int32_t* __restrict__ n_groups,
                                                                const int32_t* __restrict__ ent_dst,
                                                                const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                                float* __restrict__ out) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    __shared__ uint32_t s_hq_buf[GRP_NC * 20 * 9];
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    if constexpr (SUB_DOWN && TD == 20) {
        if (hb == 20) {
            const int r = blockIdx.x * 32 + (threadIdx.x >> 3);
            const int t = threadIdx.x & 7;
            const bool valid_r = (r < L.n_embd);
            const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    __syncthreads();
                    for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                    __syncthreads();
                    if (valid_r) {
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_40_sub8<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            row_dot_40_sub8<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s2[t];
                        } else if (n == 3) {
                            const int off3[3] = { 0, 20, 40 };
                            float s3[3];
                            row_dot_40_sub8<TD, 3, true>(wr, s_hq, off3, 3, t, s3);
                            if (t < 3) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s3[t];
                        } else {
                            const int off4[4] = { 0, 20, 40, 60 };
                            float s4[4];
                            row_dot_40_sub8<TD, 4, true>(wr, s_hq, off4, 4, t, s4);
                            if (t < 4) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s4[t];
                        }
                    }
                }
            }
            return;
        }
    } else if constexpr (SUB_DOWN && TD == 42) {
        if (hb == 20) {
            const int r = blockIdx.x * 16 + (threadIdx.x >> 4);
            const int t = threadIdx.x & 15;
            const bool valid_r = (r < L.n_embd);
            const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int g = blockIdx.y; g < ng; g += gridDim.y) {
                const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
                const int e0 = grp_start[g], e1 = grp_start[g + 1];
                for (int e = e0; e < e1; e += GRP_NC) {
                    const int n = min(GRP_NC, e1 - e);
                    const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                    const int words = n * (20 * 9);
                    __syncthreads();
                    for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                    __syncthreads();
                    if (valid_r) {
                        if (n == 1) {
                            const int off1[1] = { 0 };
                            float s1[1];
                            row_dot_20_sub16<TD, 1, true>(wr, s_hq, off1, 1, t, s1);
                            if (t == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                        } else if (n == 2) {
                            const int off2[2] = { 0, 20 };
                            float s2[2];
                            row_dot_20_sub16<TD, 2, true>(wr, s_hq, off2, 2, t, s2);
                            if (t < 2) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s2[t];
                        } else if (n == 3) {
                            const int off3[3] = { 0, 20, 40 };
                            float s3[3];
                            row_dot_20_sub16<TD, 3, true>(wr, s_hq, off3, 3, t, s3);
                            if (t < 3) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s3[t];
                        } else {
                            const int off4[4] = { 0, 20, 40, 60 };
                            float s4[4];
                            row_dot_20_sub16<TD, 4, true>(wr, s_hq, off4, 4, t, s4);
                            if (t < 4) out[(size_t) ent_dst[e + t] * L.n_embd + r] = s4[t];
                        }
                    }
                }
            }
            return;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    const bool valid_r = (r < L.n_embd);
    const size_t w_off = L.down_off + (size_t) (valid_r ? r : 0) * L.d_row;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {   // the group stride, as native_gu_kernel
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if (hb == 20) {
            const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                const int words = n * (20 * 9);
                __syncthreads();
                for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                __syncthreads();
                if (valid_r) {
                    if (n == 1) {
                        const int off1[1] = { 0 };
                        float s1[1];
                        row_dot_multi<TD, 1, true>(wr, s_hq, off1, 1, nb, lane, s1);
                        if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s1[0];
                    } else if (n == 2) {
                        const int off2[2] = { 0, 20 };
                        float s2[2];
                        row_dot_multi<TD, 2, true>(wr, s_hq, off2, 2, nb, lane, s2);
                        if (lane < 2) out[(size_t) ent_dst[e + lane] * L.n_embd + r] = s2[lane];
                    } else {
                        int off[GRP_NC];
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * 20;
                        float s[GRP_NC];
                        row_dot_multi<TD, GRP_NC>(wr, s_hq, off, n, nb, lane, s);
#pragma unroll
                        for (int c = 0; c < GRP_NC; ++c)
                            if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
                    }
                }
            }
        } else if (valid_r) {
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
                float s[GRP_NC];
                row_dot_multi<TD, GRP_NC>(wr, hq, off, n, nb, lane, s);
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
            }
        }
    }
}

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// value i of a q8_1 row set (the warp holds block i / 32, lane = i % 32)
__device__ __forceinline__ void q8_1_store(const float xi, block_q8_1* __restrict__ y, const long long i) {
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = q8_1_finite(amax / 127.0f);   // #606: q8_1_finite.hpp - the same bits for every finite block
    const int8_t q = q8_1_quant(xi, d, amax);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = q8_1_ds(d, sum);
}

__global__ void quantize_q8_1_kernel(const float* __restrict__ x, block_q8_1* __restrict__ y, long long n) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    q8_1_store(x[i], y, i);
}

// Lab-only gate-first probe (STRATA_GATE_KEEP): per entry keep the `keep` neurons with the largest |silu(gate)|
// (ties kept) and zero `up` elsewhere, so SwiGLU gives h = 0 there.  One block per entry; quality probe only.
__global__ void __launch_bounds__(256) gate_keep_kernel(const float* __restrict__ gate, float* __restrict__ up,
                                                        const int32_t* __restrict__ grp_start,
                                                        const int32_t* __restrict__ n_groups, int n_ff, int keep) {
    const int e = grp_start[0] + (int) blockIdx.x;
    if (e >= grp_start[*n_groups]) return;
    __shared__ float s[1024];
    const float* g = gate + (size_t) e * n_ff;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
        float v = -1.0f;
        if (i < n_ff) { const float x = g[i]; v = fabsf(x / (1.0f + __expf(-x))); }
        s[i] = v;
    }
    __syncthreads();
    for (int k = 2; k <= 1024; k <<= 1)          // bitonic sort, descending
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
                const int l = i ^ j;
                if (l > i) {
                    const bool desc = (i & k) == 0;
                    const float a = s[i], b = s[l];
                    if (desc ? a < b : a > b) { s[i] = b; s[l] = a; }
                }
            }
            __syncthreads();
        }
    const float thr = s[keep - 1];
    float* u = up + (size_t) e * n_ff;
    for (int i = threadIdx.x; i < n_ff; i += blockDim.x) {
        const float x = g[i];
        if (fabsf(x / (1.0f + __expf(-x))) < thr) u[i] = 0.0f;
    }
}

// swiglu_entries_kernel and quantize_q8_1_kernel in one pass, over the call's own entries only:
// [grp_start[0], grp_start[*n_groups]) (a call's entries are contiguous; the verify window's PCIe call starts after
// its VRAM call) instead of all cap_entries twice - nothing reads the others: the down kernel reads its groups'.
// n_ff is a multiple of 32, so a warp is one q8_1 block and the bounds are warp-uniform.  The values are the two
// kernels': the SwiGLU product is rounded on its own - it must not contract into the first add of q8_1_store's block
// sum, as it could not when it went through memory (CUDA: __fmul_rn; HIP's __fmul_rn is a plain product, so there
// the product is written here under contract(off)) - then q8_1_store unchanged: the same blocks, bit for bit.
__global__ void __launch_bounds__(256) swiglu_q8_1_entries_kernel(const float* __restrict__ gate,
                                                                  const float* __restrict__ up,
                                                                  const int32_t* __restrict__ grp_start,
                                                                  const int32_t* __restrict__ n_groups, int n_ff,
                                                                  block_q8_1* __restrict__ hq) {
#if defined(__HIPCC__)
#pragma clang fp contract(off)
#endif
    const long long lo = (long long) grp_start[0] * n_ff, hi = (long long) grp_start[*n_groups] * n_ff;
    for (long long i = lo + (long long) blockIdx.x * blockDim.x + threadIdx.x; i < hi;
         i += (long long) gridDim.x * blockDim.x) {
        const float g = gate[i];
#if defined(__HIPCC__)
        const float h = (g / (1.0f + __expf(-g))) * up[i];
#else
        const float h = __fmul_rn(g / (1.0f + __expf(-g)), up[i]);
#endif
        q8_1_store(h, hq, i);
    }
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
template<typename dst_t> __device__ __forceinline__ dst_t cvt(float v);
template<> __device__ __forceinline__ float cvt<float>(float v) { return v; }
template<> __device__ __forceinline__ __half cvt<__half>(float v) { return __float2half(v); }

template<typename dst_t>
__device__ void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* aux8 = (const uint8_t*) q2;
    const uint8_t* grid = (const uint8_t*) (iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t* grid = (const uint8_t*) (iq2xs_grid + (q2[il] & 511));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* grid = (const uint8_t*) (iq2s_grid + (x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
__device__ void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 0]);
    const uint8_t* grid2 = (const uint8_t*) (iq3xxs_grid + q3[2 * il + 1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint8_t* grid1 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t* grid2 = (const uint8_t*) (iq3s_grid + (qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
__device__ void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = (float) scale.f16 * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2];
    const int8_t* q = (const int8_t*) grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * (q[j] + delta));
}
template<typename dst_t>
__device__ void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
// Q3_K (the Q2_0 file's token_embd): llama.cpp's dequantize_block_q3_K, its 64 threads folded onto 32
template<typename dst_t>
__device__ void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
__device__ void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t>
__device__ void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // one "superblock" = 256 values = 4 blocks of 64; thread tid writes 8 values
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;          // block 0..3, 8 values each
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
#if defined(__HIPCC__) && defined(__gfx1012__) && HIP_VERSION_MAJOR < 7
        // HIP 5.7 on RDNA1 folds the half path's negative scale times +0
        // to +0. Preserve the scale's sign, as the FP32/CPU paths do.
        if constexpr (std::is_same_v<dst_t, __half>) {
            if (code == 1) {
                yy[b * 64 + i] = __ushort_as_half(__half_as_ushort(x[b].d) & 0x8000u);
                continue;
            }
        }
#endif
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}

// llama.cpp's dequantize_q4_K / dequantize_q5_K (dequantize.cuh; q5_K's 64 threads folded onto 32) and the 32-value
// blocks of Q5_1 / Q8_0, 8 of them per 256-value "superblock" (thread tid writes 8 values of block tid % 8).
__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template<typename dst_t>
__device__ void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = __low2half(x[ibs].dm);
    const float dmin = __high2half(x[ibs].dm);
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
template<typename dst_t>
__device__ void dq_q5_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = __low2half(x[ibs].dm);
        const float dmin = __high2half(x[ibs].dm);
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
template<typename dst_t>
__device__ void dq_q5_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // symmetric: one delta (no min), the high bit folded in then centered by the format's fixed -16 offset
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>(((float) ((x[ib].qs[iqs] & 0xf) | xh_0) - 16.0f) * d);
        y[iqs + 16] = cvt<dst_t>(((float) ((x[ib].qs[iqs] >> 4) | xh_1) - 16.0f) * d);
    }
}
template<typename dst_t>
__device__ void dq_q4_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_0* x = (const block_q4_0*) vx + ibs * (QK_K / QK4_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_0 for value pairs iqs, iqs + 16
        y[iqs] = cvt<dst_t>(((float) (x[ib].qs[iqs] & 0xf) - 8.0f) * d);
        y[iqs + 16] = cvt<dst_t>(((float) (x[ib].qs[iqs] >> 4) - 8.0f) * d);
    }
}
template<typename dst_t>
__device__ void dq_q4_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_1* x = (const block_q4_1*) vx + ibs * (QK_K / QK4_1);
    const int ib = tid % 8, il = tid / 8;
    const float2 dm = __half22float2(x[ib].dm);
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q4_1 for value pairs iqs, iqs + 16
        y[iqs] = cvt<dst_t>((float) (x[ib].qs[iqs] & 0xf) * dm.x + dm.y);
        y[iqs + 16] = cvt<dst_t>((float) (x[ib].qs[iqs] >> 4) * dm.x + dm.y);
    }
}
template<typename dst_t>
__device__ void dq_q5_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float2 dm = __half22float2(x[ib].dm);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;                   // llama.cpp's dequantize_q5_1 for value pairs iqs, iqs + 16
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>((float) ((x[ib].qs[iqs] & 0xf) | xh_0) * dm.x + dm.y);
        y[iqs + 16] = cvt<dst_t>((float) ((x[ib].qs[iqs] >> 4) | xh_1) * dm.x + dm.y);
    }
}
template<typename dst_t>
__device__ void dq_q6_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    // llama.cpp's dequantize_q6_K (dequantize.cuh, written for 64 threads) folded onto the kernel's 32: each of the
    // 64 thread roles runs at tid and tid + 32.
    const block_q6_K* x = (const block_q6_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int64_t ip = tt / 32;                 // 0 or 1
        const int64_t il = tt - 32 * ip;            // 0...31
        const int64_t is = 8 * ip + il / 16;
        dst_t* y = yy + 128 * ip + il;
        const float d = __half2float(x[ibs].d);
        const uint8_t* ql = x[ibs].ql + 64 * ip + il;
        const uint8_t qh = x[ibs].qh[32 * ip + il];
        const int8_t* sc = x[ibs].scales + is;
        y[0]  = cvt<dst_t>(d * sc[0] * ((int8_t) ((ql[0]  & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
        y[32] = cvt<dst_t>(d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
        y[64] = cvt<dst_t>(d * sc[4] * ((int8_t) ((ql[0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
        y[96] = cvt<dst_t>(d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32));
    }
}
template<typename dst_t>
__device__ void dq_q8_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = __half2float(x[ib].d);
    dst_t* y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>((float) x[ib].qs[8 * il + j] * d);
}

// BF16 (the token embedding as the checkpoint ships it, tools/embd_bf16_pack.py): 8 values per thread.
template<typename dst_t>
__device__ void dq_bf16(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const uint16_t* x = (const uint16_t*) vx + ibs * 256 + tid * 8;
    for (int j = 0; j < 8; ++j) yy[tid * 8 + j] = cvt<dst_t>(__uint_as_float((uint32_t) x[j] << 16));
}

// Every type below must also be in is_iq() (BF16: embed_type_supported): the host entry points refuse the others,
// so the default is unreachable.
template<typename dst_t>
__device__ __forceinline__ void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
#ifdef STRATA_Q6K_EXPERTS
        case 14: dq_q6_k(vx, ibs, y, tid); break;
#endif
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 6: dq_q5_0(vx, ibs, y, tid); break;
        case 2: dq_q4_0(vx, ibs, y, tid); break;
        case 3: dq_q4_1(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 30: dq_bf16(vx, ibs, y, tid); break;
        default: break;
    }
}

// flat: superblock i -> y + 256 i
template<typename dst_t>
__global__ void dequant_flat_kernel(int ty, const void* __restrict__ vx, dst_t* __restrict__ y) {
    const int64_t i = blockIdx.x;
    dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, threadIdx.x);
}
// gate/up: superblock i of a role matrix (n_embd/256 per row) -> interleaved row 2r + parity
__global__ void dequant_gu_kernel(int ty, const void* __restrict__ gate, const void* __restrict__ up, int64_t per_row,
                                  __half* __restrict__ y) {
    const int64_t i = blockIdx.x;
    const int parity = blockIdx.y;
    const int64_t r = i / per_row, c = i % per_row;
    dq_dispatch<__half>(ty, parity ? up : gate, i, y + ((2 * r + parity) * per_row + c) * QK_K, threadIdx.x);
}

// the types dq_dispatch dequantizes
bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
#ifdef STRATA_Q6K_EXPERTS
           t == 12 || t == 13 || t == 14 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8;
#else
           t == 12 || t == 13 || t == 7 || t == 6 || t == 2 || t == 3 || t == 8;
#endif
}
// values per block of the types the grouped expert kernels take (0 = none)
int gu_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_GU_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_D_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
bool gu_split(int t) {
    switch (t) {
#define STRATA_SP(T) case T: return kSplit<T>;
        STRATA_GU_FMTS(STRATA_SP)
#undef STRATA_SP
        default: return false;
    }
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}
// STRATA_OLD_IQ_MMVQ=1 keeps the per-column kernels (bitwise equal to the new ones; kept for A/B timing)
bool g_old_kernels = env_on("STRATA_OLD_IQ_MMVQ");
bool g_no_sub16_gu = env_on("STRATA_NO_SUB16_GU");
// STRATA_IQ_STAGE_GRID=0 disables staging 64-bit i-quant codebook tables into shared memory
bool g_stage_grid = [] {
    const char* v = std::getenv("STRATA_IQ_STAGE_GRID");
    return v == nullptr || v[0] == '\0' || v[0] != '0';
}();
// The single-matrix mmvq (128-thread blocks, one per 4 rows) pays the 2-8 KB staging per block: on the RTX 5070 its
// IQ2_XS / IQ2_S / IQ1_M calls were 10-20% slower staged at 3-8 columns (iq_multi_parity --bench), while the grouped
// expert kernels gain.  So mmvq stages only with STRATA_IQ_STAGE_GRID_MMVQ=1 (bitwise the same either way).
bool g_stage_grid_mmvq = g_stage_grid && [] {
    const char* v = std::getenv("STRATA_IQ_STAGE_GRID_MMVQ");
    return v != nullptr && v[0] == '1';
}();

template<int TY, bool STAGE_GRID = kStageIqGrid<TY>>
void launch_mmvq_multi_nc(dim3 grid, dim3 block, cudaStream_t s, const uint8_t* W, size_t rb, const block_q8_1* X,
                          float* y, int n_in, int n_out, int ncols) {
    switch (ncols) {
        case 1: mmvq_multi_kernel<TY, 1, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 1); break;
        case 2: mmvq_multi_kernel<TY, 2, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 2); break;
        case 3: mmvq_multi_kernel<TY, 3, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 3); break;
        case 4: mmvq_multi_kernel<TY, 4, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 4); break;
        case 5: mmvq_multi_kernel<TY, 5, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 5); break;
        case 6: mmvq_multi_kernel<TY, 6, STAGE_GRID, true><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, 6); break;
        default:
            if (ncols <= 1) mmvq_multi_kernel<TY, 1, STAGE_GRID, false><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
            else mmvq_multi_kernel<TY, 8, STAGE_GRID, false><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
            break;
    }
}

template<int TY>
void launch_mmvq(const uint8_t* W, size_t rb, const block_q8_1* X, float* y, int n_in, int n_out, int ncols,
                 cudaStream_t s) {
    const dim3 grid((unsigned) ((n_out + 3) / 4)), block(32, 4);
    if constexpr (!kSplit<TY>) mmvq_kernel<TY><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
    else if (g_old_kernels) mmvq_kernel<TY><<<grid, block, 0, s>>>(W, rb, X, y, n_in, n_out, ncols);
    else if constexpr (kStageIqGrid<TY>) {
        if (!g_stage_grid_mmvq) {
            launch_mmvq_multi_nc<TY, false>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
            return;
        }
        launch_mmvq_multi_nc<TY, true>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
    } else {
        launch_mmvq_multi_nc<TY, false>(grid, block, s, W, rb, X, y, n_in, n_out, ncols);
    }
}


// ---------------------------------------------------------------- S26 STRATA_EXPERT_V2=1 (opt-in): IQ3_S gate/up + IQ4_NL down
// Bitwise equal to native_gu_multi_kernel / native_down_multi_kernel (S26 harness, real UD-IQ4_XS expert blobs, T 1-3,
// memcmp of every output): each row's lane sums and warp_sum are the same; a warp interleaves RPW rows (more loads in
// flight) and reads the chunk's activation rows from LDS and, for IQ3_S, the grid table from LDS (the same values).
#if defined(__HIPCC__)
#define STRATA_NT_LOAD(p) __builtin_nontemporal_load(p)
#else   // (opt-in S26 kernels, AMD-tuned: a CUDA build compiles them with plain loads)
#define STRATA_NT_LOAD(p) (*(p))
#endif
template<bool NT> __device__ __forceinline__ int ld16i(const uint16_t* p) {
    if constexpr (NT) return (int) STRATA_NT_LOAD(p); else return (int) *p;
}
template<bool NT> __device__ __forceinline__ int ldb2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = ld16i<NT>(x16 + 2 * i32 + 0) << 0;
    x32 |= ld16i<NT>(x16 + 2 * i32 + 1) << 16;
    return x32;
}
template<bool NT> __device__ __forceinline__ int ld8i(const uint8_t* p) {
    if constexpr (NT) return (int) STRATA_NT_LOAD(p); else return (int) *p;
}
template<bool NT> __device__ __forceinline__ float ldhalf(const half* p) {
    return __half2float(__ushort_as_half((unsigned short) ld16i<NT>(reinterpret_cast<const uint16_t*>(p))));
}
template<int TY, bool NT> struct S26Split;
template<bool NT> struct S26Split<21, NT> {   // IQ3_S: Split<21>::load with the loads made explicit
    using W = Split<21>::W;
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(ldb2<NT>(bq3->qs, iqs + 0), ldb2<NT>(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = ld8i<NT>(bq3->qh + iqs / 2);
        const int signs_packed_32 = ldb2<NT>(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((ld8i<NT>(bq3->scales + iqs / 4) >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = ldhalf<NT>(&bq3->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ b, int iqs) { return Split<21>::apply(r, b, iqs); }
};
template<bool NT> struct S26Split<20, NT> {   // IQ4_NL
    using W = Split<20>::W;
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(ldb2<NT>(bq4->qs, iqs + l), kvalues_iq4nl);
        r.dw = ldhalf<NT>(&bq4->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ b, int iqs) { return Split<20>::apply(r, b, iqs); }
};

// S26 v13+: v6 (4 interleaved rows per warp) with the IQ3_S grid table (LT) and / or the chunk's activation rows (LX)
// read from LDS: the same table values and the same q8_1 bytes, so the same arithmetic and bits.
struct S26IQ3S {   // Split<21>::load with the grid from `grid` (LT) or iq3s_grid itself
    using W = Split<21>::W;
    template<bool LT = true>
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* grid) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = bq3->qh[iqs / 2];
        const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int i0 = qs[l0 + 0] | ((qh << (8 - l0)) & 0x100), i1 = qs[l0 + 1] | ((qh << (7 - l0)) & 0x100);
            const int2 grid_pos = LT ? make_int2(grid[i0], grid[i1]) : make_int2(iq3s_grid[i0], iq3s_grid[i1]);
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = __half2float(bq3->d);
        return r;
    }
};

constexpr int S26_XMAX = 2560 / 32;   // q8_1 blocks of one activation row (n_embd 2560)

// S26 BAL: step I of a warp's RPW x NI items (item p = 32 I + lane): rows QA / QB (lanes >= THR take QB)
template<int I, int NI, int RPW> struct S26Bal {
    static constexpr int P0 = 32 * I, QA = P0 / NI, QB0 = (P0 + 31) / NI, QB = QB0 < RPW ? QB0 : RPW - 1;
    static constexpr int THR = (QA + 1) * NI - P0;
    static_assert(NI >= 32, "a step spans at most two rows");
};

template<bool LT, bool LX, int RPW, bool SL = false, bool TS = false>
__global__ void __launch_bounds__(256) s26_gu_l_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                       const int32_t* __restrict__ grp_start,
                                                       const int32_t* __restrict__ n_groups,
                                                       const int32_t* __restrict__ ent_tok,
                                                       const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                       float* __restrict__ gate, float* __restrict__ up) {
    __shared__ uint32_t s_grid[LT ? 512 : 1];
    __shared__ block_q8_1 s_x[LX ? GRP_NC * S26_XMAX : 1];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    if constexpr (LT)
        for (int i = threadIdx.x; i < 512; i += 256) s_grid[i] = iq3s_grid[i];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * GU_ROWS * RPW + warp;
    const bool live = row0 + GU_ROWS * (RPW - 1) < 2 * L.n_ff;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
    int rr[RPW];
    bool upq[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int row = row0 + q * GU_ROWS;
        upq[q] = row >= L.n_ff;
        rr[q] = upq[q] ? row - (int) L.n_ff : row;
        wr[q] = blob + (upq[q] ? L.up_off : 0) + (size_t) rr[q] * L.gu_row;
    }
    const uint32_t* grid = LT ? s_grid : reinterpret_cast<const uint32_t*>(iq3s_grid);
    const int nb = (int) (L.n_embd / 256), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* xbase = xq;
        int off[GRP_NC];
        if constexpr (LX) {
            __syncthreads();   // the previous chunk is done with s_x
            for (int i = threadIdx.x; i < n * xb; i += 256) {
                const int c = i / xb, j = i - c * xb;
                memcpy(&s_x[c * S26_XMAX + j], &xq[(size_t) ent_tok[e + c] * xb + j], sizeof(s_x[0]));
            }
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * S26_XMAX;
            xbase = s_x;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
        }
        if constexpr (LT || LX) __syncthreads();   // s_grid / s_x ready
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * 8; k += 32) {
            const int kbx = k / 8, iqs = 2 * (k % 8);
            Split<21>::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) { if constexpr (SL && !LT) w[q] = S26Split<21, false>::load(wr[q], kbx, iqs); else w[q] = S26IQ3S::load<LT>(wr[q], kbx, iqs, grid); }
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += Split<21>::apply(w[q], xbase + off[c] + kbx * 8, iqs);
        }
        if constexpr (TS) {   // S26: all RPW x GRP_NC sums in one transposed butterfly (bitwise the same sums)
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC) {
                const int c = j % GRP_NC;
#pragma unroll
                for (int q = 0; q < RPW; ++q)
                    if (j / GRP_NC == q && c < n) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = sum;
            }
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = s[q][c];
    }
}

constexpr int S26_HMAX = 640 / 32;    // q8_1 blocks of one SwiGLU row (n_ff 640)

template<bool LX, int RPW, bool SL = false, bool TS = false, int BAL = 0>
__global__ void __launch_bounds__(256) s26_down_l_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ grp_start,
                                                         const int32_t* __restrict__ n_groups,
                                                         const int32_t* __restrict__ ent_dst,
                                                         const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                         float* __restrict__ out) {
    __shared__ block_q8_1 s_h[LX ? GRP_NC * S26_HMAX : 1];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * 8 * RPW + warp;
    const bool live = r0 + 8 * (RPW - 1) < L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) wr[q] = blob + L.down_off + (size_t) (r0 + 8 * q) * L.d_row;
    const int nb = (int) (L.n_ff / 32), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* hbase = hq;
        int off[GRP_NC];
        if constexpr (LX) {
            __syncthreads();
            for (int i = threadIdx.x; i < n * hb; i += 256) {
                const int c = i / hb, j = i - c * hb;
                memcpy(&s_h[c * S26_HMAX + j], &hq[(size_t) (e + c) * hb + j], sizeof(s_h[0]));
            }
            __syncthreads();
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * S26_HMAX;
            hbase = s_h;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        if constexpr (BAL > 0) {   // S26: 5 items per lane instead of 8 / 4, in BAL groups (loads first)
            constexpr int NI = 2 * S26_HMAX, IT = RPW * NI / 32;
            static_assert(RPW * NI % 32 == 0, "whole steps");
            auto group = [&]<int G0, int... J>(std::integer_sequence<int, J...>) {
                Split<20>::W w[sizeof...(J)];
                int kk[sizeof...(J)];
                bool hh[sizeof...(J)];
                ([&] {
                    using B = S26Bal<G0 + J, NI, RPW>;
                    hh[J] = B::QB != B::QA && lane >= B::THR;
                    kk[J] = B::P0 + lane - (hh[J] ? B::QB : B::QA) * NI;
                    const uint8_t* rp = hh[J] ? wr[B::QB] : wr[B::QA];
                    if constexpr (SL) w[J] = S26Split<20, false>::load(rp, kk[J] / 2, 2 * (kk[J] % 2));
                    else w[J] = Split<20>::load(rp, kk[J] / 2, 2 * (kk[J] % 2));
                }(), ...);
                ([&] {
                    using B = S26Bal<G0 + J, NI, RPW>;
#pragma unroll
                    for (int c = 0; c < GRP_NC; ++c)
                        if (c < n) {
                            const float v = Split<20>::apply(w[J], hbase + off[c] + kk[J] / 2, 2 * (kk[J] % 2));
                            if (hh[J]) s[B::QB][c] += v; else s[B::QA][c] += v;
                        }
                }(), ...);
            };
            constexpr int G = (IT + BAL - 1) / BAL;
            [&]<int... Q>(std::integer_sequence<int, Q...>) {
                (group.template operator()<Q * G>(std::make_integer_sequence<int, (IT - Q * G < G ? IT - Q * G : G)>{}), ...);
            }(std::make_integer_sequence<int, BAL>{});
        } else
        for (int k = lane; k < nb * 2; k += 32) {
            const int kbx = k / 2, iqs = 2 * (k % 2);
            Split<20>::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) { if constexpr (SL) w[q] = S26Split<20, false>::load(wr[q], kbx, iqs); else w[q] = Split<20>::load(wr[q], kbx, iqs); }
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += Split<20>::apply(w[q], hbase + off[c] + kbx, iqs);
        }
        if constexpr (TS) {   // S26: all RPW x GRP_NC sums in one transposed butterfly (bitwise the same sums)
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            const int q = j / GRP_NC, c = j % GRP_NC;
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC && c < n)
                out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = sum;
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = s[q][c];
    }
}


// S26: swiglu_entries_kernel + quantize_q8_1_kernel in one launch (the same h expression, then the same block quantizer,
// contraction off so the product cannot fuse into the first sum); h is still written (it is the scratch the old pair used)
__global__ void s26_swiglu_q8_1_kernel(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ h,
                                       block_q8_1* __restrict__ y, long long n) {
#pragma clang fp contract(off)
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;   // n is a multiple of 32: whole warps only
    const float g = gate[i];
    const float xi = (g / (1.0f + __expf(-g))) * up[i];
    h[i] = xi;
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = make_half2(d, sum);
}
template<bool LT, bool LX, int RG, int RD, bool SL = false, bool FQ = false, bool TS = false, int BD = 0>
void s26_launch_l(const NativeExpertLayout& L, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr,
                  const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                  const block_q8_1* X, float* gate, float* up, float* h, block_q8_1* hq, float* out, long long nh) {
    const dim3 ggu((unsigned) (2 * L.n_ff / (GU_ROWS * RG)), (unsigned) cap_groups);
    s26_gu_l_kernel<LT, LX, RG, SL, TS><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    if (FQ) {
        s26_swiglu_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, hq, nh);
    } else {
        swiglu_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, nh);
        quantize_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(h, hq, nh);
    }
    const dim3 gd((unsigned) (L.n_embd / (8 * RD)), (unsigned) cap_groups);
    s26_down_l_kernel<LX, RD, SL, TS, BD><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}
// ---------------------------------------------------------------- stream B: UD-Q4_K_XL grouped decode experts
// STRATA_EXPERT_V2K=1: s26_gu_l_kernel / s26_down_l_kernel's structure (4 interleaved rows per warp, the chunk's q8_1
// activation rows in LDS, SwiGLU + q8_1 fused, optionally STRATA_TSUM's transposed butterfly) for Q4_K / Q5_K gate/up
// and Q5_1 / Q8_0 down, which otherwise run native_gu_kernel / native_down_kernel (one entry at a time, the weight
// words re-read per column).  Each Fmt<TY>::dot is split into S27<TY>::load (everything that depends on the weight) and
// apply (the activation words and the same integer / float expression), the same impl functions, so a row's lane-strided
// k order, its terms and warp_sum are the old kernels': every output is bitwise the old one's (iq harness, memcmp).
template<int TY> struct S27;
template<> struct S27<12> {   // Q4_K
    struct W { int v[2]; uint16_t aux[2]; half2 dm; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q4_K* b = (const block_q4_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        const int* q4 = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = q4[0];
        r.v[1] = q4[4];
        k_scale_min(b->scales, bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        int u[2 * QR4_K];
        float d8[QR4_K];
#pragma unroll
        for (int i = 0; i < QR4_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        const uint8_t* m = sc + 2;
        return vec_dot_q4_K_q8_1_impl_vmmq(r.v, u, sc, m, r.dm, d8);
    }
};
template<> struct S27<13> {   // Q5_K
    struct W { int vl[2]; int vh[2]; uint16_t aux[2]; half2 dm; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q5_K* b = (const block_q5_K*) vbq + kbx;
        W r;
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        const int* ql = (const int*) (b->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = (const int*) (b->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> bq8_offset;
        r.vh[1] = qh[4] >> bq8_offset;
        k_scale_min(b->scales, bq8_offset, r.aux);
        r.dm = b->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        int u[2 * QR5_K];
        float d8[QR5_K];
#pragma unroll
        for (int i = 0; i < QR5_K; ++i) {
            const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
            u[2 * i + 0] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = (const uint8_t*) r.aux;
        const uint8_t* m = sc + 2;
        return vec_dot_q5_K_q8_1_impl_vmmq(r.vl, r.vh, u, sc, m, r.dm, d8);
    }
};
template<> struct S27<7> {   // Q5_1
    struct W { int vl[VDR_Q5_1]; int qh; half2 dm; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q5_1* b = (const block_q5_1*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) r.vl[i] = get_int_b4(b->qs, iqs + i);
        r.qh = get_int_b4(b->qh, 0);
        r.dm = b->dm;
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0, sumu = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q5_1; ++i) {
            const int vl = r.vl[i];
            const int vh = r.qh >> (4 * (iqs + i));
            const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
            int vi0 = (vl >> 0) & 0x0F0F0F0F;
            vi0 |= (vh << 4) & 0x00000010;
            vi0 |= (vh << 11) & 0x00001000;
            vi0 |= (vh << 18) & 0x00100000;
            vi0 |= (vh << 25) & 0x10000000;
            sumi = ggml_cuda_dp4a(vi0, u0, sumi);
            int vi1 = (vl >> 4) & 0x0F0F0F0F;
            vi1 |= (vh >> 12) & 0x00000010;
            vi1 |= (vh >> 5) & 0x00001000;
            vi1 |= (vh << 2) & 0x00100000;
            vi1 |= (vh << 9) & 0x10000000;
            sumi = ggml_cuda_dp4a(vi1, u1, sumi);
            sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
        }
        const float2 dm5 = __half22float2(r.dm);
        const float d8 = __low2float(bq8_1->ds);
        return sumi * (dm5.x * d8) + sumu * (dm5.y * d8);
    }
};
template<> struct S27<8> {   // Q8_0
    struct W { int q[VDR_Q8_0]; float d; };
    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q8_0* b = (const block_q8_0*) vbq + kbx;
        W r;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) r.q[i] = get_int_b2(b->qs, iqs + i);
        r.d = __half2float(b->d);
        return r;
    }
    __device__ static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int i = 0; i < VDR_Q8_0; ++i) sumi = ggml_cuda_dp4a(r.q[i], get_int_b4(bq8_1->qs, iqs + i), sumi);
        const float d8_1 = __low2float(bq8_1->ds);
        return r.d * d8_1 * ((float) sumi);
    }
};

template<int TG, bool LX, int RPW, bool TS>
__global__ void __launch_bounds__(256) s27_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                     const int32_t* __restrict__ grp_start,
                                                     const int32_t* __restrict__ n_groups,
                                                     const int32_t* __restrict__ ent_tok,
                                                     const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                     float* __restrict__ gate, float* __restrict__ up) {
    using F = Fmt<TG>;
    using S = S27<TG>;
    __shared__ block_q8_1 s_x[LX ? GRP_NC * S26_XMAX : 1];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * GU_ROWS * RPW + warp;
    const bool live = row0 + GU_ROWS * (RPW - 1) < 2 * L.n_ff;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
    int rr[RPW];
    bool upq[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int row = row0 + q * GU_ROWS;
        upq[q] = row >= L.n_ff;
        rr[q] = upq[q] ? row - (int) L.n_ff : row;
        wr[q] = blob + (upq[q] ? L.up_off : 0) + (size_t) rr[q] * L.gu_row;
    }
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* xbase = xq;
        int off[GRP_NC];
        if constexpr (LX) {
            __syncthreads();   // the previous chunk is done with s_x
            for (int i = threadIdx.x; i < n * xb; i += 256) {
                const int c = i / xb, j = i - c * xb;
                memcpy(&s_x[c * S26_XMAX + j], &xq[(size_t) ent_tok[e + c] * xb + j], sizeof(s_x[0]));
            }
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * S26_XMAX;
            xbase = s_x;
            __syncthreads();
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + min(c, n - 1)] * xb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            typename S::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) w[q] = S::load(wr[q], kbx, iqs);
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += S::apply(w[q], xbase + off[c] + kbx * (F::qk / 32), iqs);
        }
        if constexpr (TS) {
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC) {
                const int c = j % GRP_NC;
#pragma unroll
                for (int q = 0; q < RPW; ++q)
                    if (j / GRP_NC == q && c < n) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = sum;
            }
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) (upq[q] ? up : gate)[(size_t) (e + c) * L.n_ff + rr[q]] = s[q][c];
    }
}

template<int TD, bool LX, int RPW, bool TS>
__global__ void __launch_bounds__(256) s27_down_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                       const int32_t* __restrict__ grp_start,
                                                       const int32_t* __restrict__ n_groups,
                                                       const int32_t* __restrict__ ent_dst,
                                                       const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                       float* __restrict__ out) {
    using F = Fmt<TD>;
    using S = S27<TD>;
    __shared__ block_q8_1 s_h[LX ? GRP_NC * S26_HMAX : 1];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = blockIdx.x * 8 * RPW + warp;
    const bool live = r0 + 8 * (RPW - 1) < L.n_embd;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const uint8_t* wr[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) wr[q] = blob + L.down_off + (size_t) (r0 + 8 * q) * L.d_row;
    const int nb = (int) (L.n_ff / F::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    for (int e = e0; e < e1; e += GRP_NC) {
        const int n = min(GRP_NC, e1 - e);
        const block_q8_1* hbase = hq;
        int off[GRP_NC];
        if constexpr (LX) {
            __syncthreads();
            for (int i = threadIdx.x; i < n * hb; i += 256) {
                const int c = i / hb, j = i - c * hb;
                memcpy(&s_h[c * S26_HMAX + j], &hq[(size_t) (e + c) * hb + j], sizeof(s_h[0]));
            }
            __syncthreads();
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = min(c, n - 1) * S26_HMAX;
            hbase = s_h;
        } else {
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) off[c] = (e + min(c, n - 1)) * hb;
        }
        if (!live) continue;
        float s[RPW][GRP_NC];
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c) s[q][c] = 0.0f;
        for (int k = lane; k < nb * F::ipb; k += 32) {
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            typename S::W w[RPW];
#pragma unroll
            for (int q = 0; q < RPW; ++q) w[q] = S::load(wr[q], kbx, iqs);
#pragma unroll
            for (int q = 0; q < RPW; ++q)
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c)
                    if (c < n) s[q][c] += S::apply(w[q], hbase + off[c] + kbx * (F::qk / 32), iqs);
        }
        if constexpr (TS) {
            constexpr int P = s26ts::pow2_ceil(RPW * GRP_NC);
            float v[P];
#pragma unroll
            for (int j = 0; j < P; ++j) v[j] = j < RPW * GRP_NC ? s[j / GRP_NC][j % GRP_NC] : 0.0f;
            const float sum = s26ts::tsum<P>(v, lane);
            const int j = s26ts::tsum_token<P>(lane);
            const int q = j / GRP_NC, c = j % GRP_NC;
            if (lane == s26ts::tsum_lane<P>(j) && j < RPW * GRP_NC && c < n)
                out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = sum;
            continue;
        }
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n) s[q][c] = warp_sum(s[q][c]);
#pragma unroll
        for (int q = 0; q < RPW; ++q)
#pragma unroll
            for (int c = 0; c < GRP_NC; ++c)
                if (c < n && lane == c) out[(size_t) ent_dst[e + c] * L.n_embd + r0 + 8 * q] = s[q][c];
    }
}

template<int TG, int TD, bool LX, int RG, int RD, bool TS>
void s27_launch(const NativeExpertLayout& L, int64_t cap_groups, cudaStream_t s, const unsigned long long* grp_ptr,
                const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok,
                const block_q8_1* X, float* gate, float* up, float* h, block_q8_1* hq, float* out, long long nh) {
    const dim3 ggu((unsigned) (2 * L.n_ff / (GU_ROWS * RG)), (unsigned) cap_groups);
    s27_gu_kernel<TG, LX, RG, TS><<<ggu, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    s26_swiglu_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, hq, nh);
    const dim3 gd((unsigned) (L.n_embd / (8 * RD)), (unsigned) cap_groups);
    s27_down_kernel<TD, LX, RD, TS><<<gd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}
template<int TG, int TD>
void s27_launch_ts(bool ts, const NativeExpertLayout& L, int64_t cap_groups, cudaStream_t s,
                   const unsigned long long* grp_ptr, const int32_t* grp_start, const int32_t* n_groups,
                   const int32_t* ent_dst, const int32_t* ent_tok, const block_q8_1* X, float* gate, float* up, float* h,
                   block_q8_1* hq, float* out, long long nh) {
    if (ts) s27_launch<TG, TD, true, 4, 4, true>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
    else s27_launch<TG, TD, true, 4, 4, false>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
}

// ---------------------------------------------------------------- STRATA_FP_EXPERT_V (opt-in, default 0 = the kernels above)
// Variants 1-3: IQ3_S gate/up + IQ4_NL down at n_embd 2560 / n_ff 640 only (every other case keeps its path).  Variant 4
// (persistent, further below) takes each role on its own: gate/up IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S at n_embd
// 2560, down IQ4_NL / Q2_0 at n_ff 640 (the model's 48 routed layers), the other role keeping its default kernel when its
// format is not covered.  Every variant gives the bitwise outputs of native_gu_multi_kernel<TG, *, true> /
// native_down_multi_kernel<TD, true>: per output row the same lane partial sums, the same __fadd_rn / xor-shuffle tree,
// the same `acc += apply(w, x)` expressions on the same words; only where the codebook is read from (1) and how many
// rows a sub-warp keeps in flight (2) change.
//   1: gate/up with iq3s_grid (512 x u32, 2 KB) staged in shared memory once per block (LDS lookups instead of
//      L1-cached global ones: shorter dependent chain between the qs loads and the dp4a).
//   2: gate/up and down, two rows per sub-warp (gate row r + up row r; down rows r and r + n_embd / 2), both rows'
//      weight loads of a k step before the dp4a work: ~2x the bytes in flight per lane and one activation read for two
//      rows, at ~1.5-2x the registers (down to 2 blocks / SM at 256 threads on Turing; 40 instead of 80 blocks in x).
//   3: gate/up with 1 and 2 together, down as 2.
//   4: persistent gate/up and down (fp_gu_persist_kernel / fp_down_persist_kernel, further below).
int g_fp_variant = [] {
    const char* v = std::getenv("STRATA_FP_EXPERT_V");
    const int x = v != nullptr ? std::atoi(v) : 0;
    return x >= 0 && x <= 4 ? x : 0;
}();

bool fp_pair_ok(const NativeExpertLayout& L) {
    return !g_old_kernels && !g_no_sub16_gu && L.gu_type == 21 && L.d_type == 20 && L.n_embd == 2560 && L.n_ff == 640;
}

// acc[q][c] += row q's dot of the k-th (block, part) call with column c, k = 8 * kbx + iqs / 2 (IQ3_S: 8 calls per block).
// The NR rows' loads come first (row_dot_80_sub16's body for NR rows; STG: the grid from shared memory).
template<int NR, int NC, bool STG>
__device__ __forceinline__ void fp_dot80_acc(const uint8_t* const* row, const block_q8_1* x, const int (&off)[NC], int k,
                                             float (&acc)[NR][NC], const uint32_t* grid) {
    const int kbx = k / 8, iqs = 2 * (k % 8);
    Split<21>::W w[NR];
#pragma unroll
    for (int q = 0; q < NR; ++q) w[q] = S26IQ3S::load<STG>(row[q], kbx, iqs, grid);
#pragma unroll
    for (int q = 0; q < NR; ++q) {
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[q][c] += Split<21>::apply(w[q], x + off[c] + kbx * 8, iqs);
    }
}

// row_dot_80_sub16's tree for NR rows (t = lane in the 16-lane sub-warp): s = calls t, t+32, t+64; s16 = t+16, t+48;
// s = s + s16; xor tree 8, 4, 2, 1.  All 16 lanes end with the sums.
template<int NR, int NC, bool STG>
__device__ __forceinline__ void fp_row_dot_80(const uint8_t* const* row, const block_q8_1* x, const int (&off)[NC], int t,
                                              float (&s)[NR][NC], const uint32_t* grid) {
    float s16[NR][NC];
#pragma unroll
    for (int q = 0; q < NR; ++q) {
#pragma unroll
        for (int c = 0; c < NC; ++c) { s[q][c] = 0.0f; s16[q][c] = 0.0f; }
    }
    fp_dot80_acc<NR, NC, STG>(row, x, off, t, s, grid);
    fp_dot80_acc<NR, NC, STG>(row, x, off, t + 32, s, grid);
    fp_dot80_acc<NR, NC, STG>(row, x, off, t + 64, s, grid);
    fp_dot80_acc<NR, NC, STG>(row, x, off, t + 16, s16, grid);
    fp_dot80_acc<NR, NC, STG>(row, x, off, t + 48, s16, grid);
#pragma unroll
    for (int q = 0; q < NR; ++q) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            s[q][c] = __fadd_rn(s[q][c], s16[q][c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[q][c] += __shfl_xor_sync(0xffffffffu, s[q][c], o);
        }
    }
}

// One chunk of NC entries e .. e + NC - 1 of a group for NR rows: lane t == c stores column c of each row.
template<int NR, int NC, bool STG>
__device__ __forceinline__ void fp_gu_chunk(const uint8_t* const* row, const block_q8_1* xq, const int32_t* ent_tok, int e,
                                            int t, int r, size_t n_ff, float* const* dst, const uint32_t* grid) {
    int off[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) off[c] = ent_tok[e + c] * 80;
    float s[NR][NC];
    fp_row_dot_80<NR, NC, STG>(row, xq, off, t, s, grid);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (t == c) {
#pragma unroll
            for (int q = 0; q < NR; ++q) dst[q][(size_t) (e + c) * n_ff + (size_t) r] = s[q][c];
        }
    }
}

// native_gu_multi_kernel<21, false, true>'s launch shape (256 threads, 16-lane sub-warps, grid (rows / 16, groups)).
// X2: a sub-warp owns gate row r and up row r (grid x = n_ff / 16); otherwise one row of 2 n_ff as in the original.
template<bool STG, bool X2>
__global__ void __launch_bounds__(256) fp_gu_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                     const int32_t* __restrict__ grp_start,
                                                     const int32_t* __restrict__ n_groups,
                                                     const int32_t* __restrict__ ent_tok,
                                                     const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                     float* __restrict__ gate, float* __restrict__ up) {
    constexpr int NR = X2 ? 2 : 1;
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    __shared__ uint32_t s_grid[STG ? 512 : 1];
    if constexpr (STG) {
        for (int i = threadIdx.x; i < 512; i += 256) s_grid[i] = iq3s_grid[i];
        __syncthreads();
    }
    const int subwarp = threadIdx.x >> 4, t = threadIdx.x & 15;
    const int row = blockIdx.x * 16 + subwarp;
    if (row >= (X2 ? (int) L.n_ff : 2 * (int) L.n_ff)) return;   // whole sub-warp pairs (n_ff % 32 == 0)
    int r;
    size_t w_off[NR];
    float* dst[NR];
    if constexpr (X2) {
        r = row;
        w_off[0] = (size_t) r * L.gu_row;
        w_off[1] = L.up_off + (size_t) r * L.gu_row;
        dst[0] = gate;
        dst[1] = up;
    } else {
        const bool is_up = row >= (int) L.n_ff;
        r = is_up ? row - (int) L.n_ff : row;
        w_off[0] = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
        dst[0] = is_up ? up : gate;
    }
    const uint32_t* grid = s_grid;   // read only when STG
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const uint8_t* rows[NR];
        rows[0] = blob + w_off[0];
        if constexpr (X2) rows[1] = blob + w_off[1];
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            if (n == 1) fp_gu_chunk<NR, 1, STG>(rows, xq, ent_tok, e, t, r, (size_t) L.n_ff, dst, grid);
            else if (n == 2) fp_gu_chunk<NR, 2, STG>(rows, xq, ent_tok, e, t, r, (size_t) L.n_ff, dst, grid);
            else if (n == 3) fp_gu_chunk<NR, 3, STG>(rows, xq, ent_tok, e, t, r, (size_t) L.n_ff, dst, grid);
            else fp_gu_chunk<NR, 4, STG>(rows, xq, ent_tok, e, t, r, (size_t) L.n_ff, dst, grid);
        }
    }
}

// Down, IQ4_NL, hb == 20: acc[q][c] += row q's dot of the call k (block k / 2, part k % 2) with the staged h row c
// (20 q8_1 blocks each).  Both rows' loads before the dp4a work, as fp_dot80_acc.
template<int NC>
__device__ __forceinline__ void fp_dot40_acc(const uint8_t* const* row, const block_q8_1* x, int k, float (&acc)[2][NC]) {
    const int kbx = k / 2, iqs = 2 * (k % 2);
    Split<20>::W w[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) w[q] = Split<20>::load(row[q], kbx, iqs);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[q][c] += Split<20>::apply(w[q], x + c * 20 + kbx, iqs);
    }
}

// row_dot_40_sub8's tree for two rows (t = lane in the 8-lane sub-warp): s = calls t, t+32; s16 = t+16; s8 = t+8;
// s24 = t+24; s = s + s16; s = s + (s8 + s24); xor tree 4, 2, 1.
template<int NC>
__device__ __forceinline__ void fp_row_dot_40_x2(const uint8_t* const* row, const block_q8_1* x, int t, float (&s)[2][NC]) {
    float s16[2][NC], s8[2][NC], s24[2][NC];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int c = 0; c < NC; ++c) { s[q][c] = 0.0f; s16[q][c] = 0.0f; s8[q][c] = 0.0f; s24[q][c] = 0.0f; }
    }
    fp_dot40_acc<NC>(row, x, t, s);
    fp_dot40_acc<NC>(row, x, t + 32, s);
    fp_dot40_acc<NC>(row, x, t + 16, s16);
    fp_dot40_acc<NC>(row, x, t + 8, s8);
    fp_dot40_acc<NC>(row, x, t + 24, s24);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            s[q][c] = __fadd_rn(s[q][c], s16[q][c]);
            s[q][c] = __fadd_rn(s[q][c], __fadd_rn(s8[q][c], s24[q][c]));
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[q][c] += __shfl_xor_sync(0xffffffffu, s[q][c], o);
        }
    }
}

template<int NC>
__device__ __forceinline__ void fp_down_chunk(const uint8_t* const* row, const block_q8_1* s_hq, const int32_t* ent_dst,
                                              int e, int t, int r, int hrows, size_t n_embd, float* out) {
    float s[2][NC];
    fp_row_dot_40_x2<NC>(row, s_hq, t, s);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (t == c) {
            float* o = out + (size_t) ent_dst[e + c] * n_embd + (size_t) r;
            o[0] = s[0][c];
            o[hrows] = s[1][c];
        }
    }
}

// native_down_multi_kernel<20, true>'s launch shape with a sub-warp (8 lanes, 32 per block) owning rows r and
// r + n_embd / 2; grid x = n_embd / 64.  The h chunk staging and its barriers are the original's.
__global__ void __launch_bounds__(256) fp_down_x2_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                         const int32_t* __restrict__ grp_start,
                                                         const int32_t* __restrict__ n_groups,
                                                         const int32_t* __restrict__ ent_dst,
                                                         const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                         float* __restrict__ out) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    __shared__ uint32_t s_hq_buf[GRP_NC * 20 * 9];
    const int hrows = (int) (L.n_embd >> 1);
    const int r = blockIdx.x * 32 + (threadIdx.x >> 3);
    const int t = threadIdx.x & 7;
    const bool valid_r = (r < hrows);
    const int rc = valid_r ? r : 0;
    const size_t w_off0 = L.down_off + (size_t) rc * L.d_row;
    const size_t w_off1 = L.down_off + (size_t) (rc + hrows) * L.d_row;
    const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const uint8_t* const rows[2] = { blob + w_off0, blob + w_off1 };
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
            const int words = n * (20 * 9);
            __syncthreads();
            for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
            __syncthreads();
            if (valid_r) {
                if (n == 1) fp_down_chunk<1>(rows, s_hq, ent_dst, e, t, r, hrows, (size_t) L.n_embd, out);
                else if (n == 2) fp_down_chunk<2>(rows, s_hq, ent_dst, e, t, r, hrows, (size_t) L.n_embd, out);
                else if (n == 3) fp_down_chunk<3>(rows, s_hq, ent_dst, e, t, r, hrows, (size_t) L.n_embd, out);
                else fp_down_chunk<4>(rows, s_hq, ent_dst, e, t, r, hrows, (size_t) L.n_embd, out);
            }
        }
    }
}

// ---------------------------------------------------------------- STRATA_FP_EXPERT_V=4: persistent gate/up and down
// Grid = k x SM count blocks of 256 threads (k = STRATA_FP_EXPERT_PERSIST_K, default 2) instead of 80 x cap_groups blocks
// that mostly exit at once.  A block takes the work items w = blockIdx.x, + gridDim.x, ... in that fixed order, item w =
// (group w / ntiles, tile w % ntiles) with a tile the rows one block of the default kernels owns (16 gate/up rows; 32
// down rows for IQ4_NL, 16 for Q2_0), each row on the same sub-warp layout (16 / 8 / 16 lanes, t = lane in it) with the
// same per-lane calls, the same accumulators and the same __fadd_rn / xor tree as row_dot_80_sub16 / row_dot_40_sub8 /
// row_dot_20_sub16: an output does not depend on which block computed it, and equals the default kernels' bit for bit.
// Formats (the model's routed experts at n_embd 2560 / n_ff 640): gate/up IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18),
// IQ3_S (21), IQ2_S (22) - the kSplit formats with 8 calls per 256-block, 80 per row - through FpGu<TG>; down IQ4_NL (20)
// and Q2_0 (42) through FpDown<TD>.  The gate/up format's codebook grid is staged in shared memory once per block (the
// same values the default kernels read from the global table or their per-block copy).
// Register prefetch: the item's weight words (raw: the lane's k steps x 3-4 registers) are loaded, through the read-only
// path (ld.global.nc; 16-bit and 8-bit loads, as Split<TY>::load's get_int_b2 and byte reads: the ggml blocks are 2-byte
// aligned, so no wider load is legal; plain loads valid on sm_75 and sm_86, nothing from sm_80+), one item of work
// before they are decoded: gate/up refills each raw slot with the next item's step as soon as the step is consumed
// (fp_pchunk80<.., REFILL>), down keeps a second buffer for the next item (fp_down_persist_kernel); the decode from the
// raw words is Split<TY>::load's arithmetic on the same words.  Per row the addition order is untouched; only the order
// in which the loads and the independent rows are issued changes.
// STRATA_FP_PERSIST_MINB (build-time, default 2): the min blocks / SM of both kernels' __launch_bounds__(256, .), i.e.
// the register cap (2 = 128 registers; 1 lets ptxas use up to 255 at 8 warps / SM) - ptxas -v tells whether a format
// still spills, the bench which setting is faster.
#ifndef STRATA_FP_PERSIST_MINB
#define STRATA_FP_PERSIST_MINB 2
#endif
int g_fp_persist_k = [] {
    const char* v = std::getenv("STRATA_FP_EXPERT_PERSIST_K");
    const int x = v != nullptr ? std::atoi(v) : 2;
    return x >= 1 && x <= 16 ? x : 2;
}();

int fp_persist_blocks() {   // k x SMs of the current device, the SM count read once per device
    int dev = 0;
    cudaGetDevice(&dev);
    static int sms[64] = {};
    const bool cacheable = dev >= 0 && dev < 64;
    int sm = cacheable ? sms[dev] : 0;
    if (sm <= 0) {
        sm = 0;
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        if (sm <= 0) sm = 1;
        if (cacheable) sms[dev] = sm;
    }
    return g_fp_persist_k * sm;
}

// the gate/up and down formats the persistent kernels take (every other format keeps the default kernels under V=4)
template<int TG> inline constexpr bool kFpPersistGu = (TG == 16 || TG == 17 || TG == 18 || TG == 21 || TG == 22);
template<int TD> inline constexpr bool kFpPersistDown = (TD == 20 || TD == 42);

// the shapes: gate/up rows of n_embd 2560 (80 calls, row_dot_80_sub16's layout), down rows of n_ff 640 (hb == 20)
bool fp_persist_gu_ok(const NativeExpertLayout& L) {
    return !g_old_kernels && !g_no_sub16_gu && L.n_embd == 2560 && L.n_ff % 32 == 0;
}
bool fp_persist_down_ok(const NativeExpertLayout& L) { return !g_old_kernels && !g_no_sub16_gu && L.n_ff == 640; }

// the read-only path (ld.global.nc): the 16-bit and 8-bit loads behind Split<TY>::load's get_int_b2 and byte reads
__device__ __forceinline__ unsigned int fp_ld16(const void* p) { return (unsigned int) __ldg((const unsigned short*) p); }
__device__ __forceinline__ unsigned int fp_ld8(const uint8_t* p) { return (unsigned int) __ldg(p); }
__device__ __forceinline__ int fp_ld32b2(const void* p, int i32) {   // get_int_b2
    const unsigned short* p16 = (const unsigned short*) p;
    return (int) ((unsigned int) __ldg(p16 + 2 * i32) | ((unsigned int) __ldg(p16 + 2 * i32 + 1) << 16));
}
__device__ __forceinline__ float fp_half_bits(unsigned int h) {   // __half2float of the fp16 bits (the block's d)
    return __half2float(__ushort_as_half((unsigned short) (h & 0xFFFFu)));
}

// FpGu<TG>: a gate/up format of the persistent kernel.  Raw = the words one lane's k step (block kbx, part iqs) reads,
// exactly the loads of Split<TG>::load, unmodified; load() takes them through the read-only path; decode() is
// Split<TG>::load's arithmetic on those words with the format's grid read from `grid` (the block's shared copy);
// kGridWords = u32 words of that table, table() its global copy.
template<int TG> struct FpGu;

template<> struct FpGu<21> {   // IQ3_S: qs words iqs, iqs + 1; the signs word iqs / 2; m = qh byte | scales byte << 8 | d << 16
    struct Raw { int q0, q1, sg; unsigned int m; };
    static constexpr int kGridWords = 512;   // iq3s_grid: 512 x u32
    __device__ static const uint32_t* table() { return iq3s_grid; }
    __device__ static Raw load(const uint8_t* row, int kbx, int iqs) {
        const block_iq3_s* b = (const block_iq3_s*) row + kbx;
        Raw a;
        a.q0 = fp_ld32b2(b->qs, iqs + 0);
        a.q1 = fp_ld32b2(b->qs, iqs + 1);
        a.sg = fp_ld32b2(b->signs, iqs / 2);
        a.m = fp_ld8(b->qh + iqs / 2) | (fp_ld8(b->scales + iqs / 4) << 8) | (fp_ld16(&b->d) << 16);
        return a;
    }
    __device__ static Split<21>::W decode(const Raw& a, int iqs, const uint32_t* grid) {
        const int2 qs_packed = make_int2(a.q0, a.q1);
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = (int) (a.m & 0xFFu);
        const int signs_packed_32 = a.sg;
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        Split<21>::W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                            grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        const int sc = (int) ((a.m >> 8) & 0xFFu);
        r.ls = 1 + 2 * ((sc >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = fp_half_bits(a.m >> 16);
        return r;
    }
};

template<> struct FpGu<16> {   // IQ2_XXS: qs words iqs (grid indices) and iqs + 1 (signs + scale); d
    struct Raw { int q2; unsigned int aux, d; };
    static constexpr int kGridWords = 512;   // iq2xxs_grid: 256 x u64
    __device__ static const uint32_t* table() { return (const uint32_t*) iq2xxs_grid; }
    __device__ static Raw load(const uint8_t* row, int kbx, int iqs) {
        const block_iq2_xxs* b = (const block_iq2_xxs*) row + kbx;
        Raw a;
        a.q2 = fp_ld32b2(b->qs, iqs);
        a.aux = (unsigned int) fp_ld32b2(b->qs, iqs + 1);
        a.d = fp_ld16(&b->d);
        return a;
    }
    __device__ static Split<16>::W decode(const Raw& a, int, const uint32_t* grid) {
        const int q2 = a.q2;
        const uint8_t* aux8 = (const uint8_t*) &q2;
        const uint32_t aux32 = a.aux;
        const uint2* grid_lut = (const uint2*) grid;
        Split<16>::W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = grid_lut[aux8[k0 / 2]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[k0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[k0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = fp_half_bits(a.d);
        return r;
    }
};

template<> struct FpGu<17> {   // IQ2_XS: qs words iqs, iqs + 1; m = scales byte | d << 16
    struct Raw { int q0, q1; unsigned int m; };
    static constexpr int kGridWords = 1024;   // iq2xs_grid: 512 x u64
    __device__ static const uint32_t* table() { return (const uint32_t*) iq2xs_grid; }
    __device__ static Raw load(const uint8_t* row, int kbx, int iqs) {
        const block_iq2_xs* b = (const block_iq2_xs*) row + kbx;
        Raw a;
        a.q0 = fp_ld32b2(b->qs, iqs + 0);
        a.q1 = fp_ld32b2(b->qs, iqs + 1);
        a.m = fp_ld8(b->scales + iqs / 2) | (fp_ld16(&b->d) << 16);
        return a;
    }
    __device__ static Split<17>::W decode(const Raw& a, int, const uint32_t* grid) {
        const int2 q2_packed = make_int2(a.q0, a.q1);
        const uint16_t* q2 = (const uint16_t*) &q2_packed;
        const uint2* grid_lut = (const uint2*) grid;
        Split<17>::W r;
        r.ls0 = (int) (a.m & 0x0Fu);
        r.ls1 = (int) ((a.m >> 4) & 0x0Fu);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid_lut[q2[l0 / 2] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.dw = fp_half_bits(a.m >> 16);
        return r;
    }
};

template<> struct FpGu<22> {   // IQ2_S: the qs word iqs / 2; the signs word QK_K / 32 + iqs / 2; m = qh | scales << 8 | d << 16
    struct Raw { int qs, sg; unsigned int m; };
    static constexpr int kGridWords = 2048;   // iq2s_grid: 1024 x u64
    __device__ static const uint32_t* table() { return (const uint32_t*) iq2s_grid; }
    __device__ static Raw load(const uint8_t* row, int kbx, int iqs) {
        const block_iq2_s* b = (const block_iq2_s*) row + kbx;
        Raw a;
        a.qs = fp_ld32b2(b->qs, iqs / 2);
        a.sg = fp_ld32b2(b->qs, QK_K / 32 + iqs / 2);
        a.m = fp_ld8(b->qh + iqs / 2) | (fp_ld8(b->scales + iqs / 2) << 8) | (fp_ld16(&b->d) << 16);
        return a;
    }
    __device__ static Split<22>::W decode(const Raw& a, int, const uint32_t* grid) {
        const int qs_packed = a.qs;
        const uint8_t* qs = (const uint8_t*) &qs_packed;
        const int qh = (int) (a.m & 0xFFu);
        const int signs_packed_32 = a.sg;
        const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
        const uint64_t* grid_lut = (const uint64_t*) grid;
        Split<22>::W r;
        r.ls0 = (int) ((a.m >> 8) & 0x0Fu);
        r.ls1 = (int) ((a.m >> 12) & 0x0Fu);
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int* grid_pos = (const int*) (grid_lut + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            r.g[l0 + 0] = __vsub4(grid_pos[0] ^ signs0, signs0);
            r.g[l0 + 1] = __vsub4(grid_pos[1] ^ signs1, signs1);
        }
        r.dw = fp_half_bits(a.m >> 16);
        return r;
    }
};

template<> struct FpGu<18> {   // IQ3_XXS: qs words iqs, iqs + 1 (grid indices); the aux word QK_K / 16 + iqs / 2 (signs + scale); d
    struct Raw { int q0, q1; unsigned int aux, d; };
    static constexpr int kGridWords = 256;   // iq3xxs_grid: 256 x u32
    __device__ static const uint32_t* table() { return iq3xxs_grid; }
    __device__ static Raw load(const uint8_t* row, int kbx, int iqs) {
        const block_iq3_xxs* b = (const block_iq3_xxs*) row + kbx;
        Raw a;
        a.q0 = fp_ld32b2(b->qs, iqs);
        a.q1 = fp_ld32b2(b->qs, iqs + 1);
        a.aux = (unsigned int) fp_ld32b2(b->qs, QK_K / 16 + iqs / 2);
        a.d = fp_ld16(&b->d);
        return a;
    }
    __device__ static Split<18>::W decode(const Raw& a, int, const uint32_t* grid) {
        const int2 q3_packed = make_int2(a.q0, a.q1);
        const uint8_t* q3 = (const uint8_t*) &q3_packed;
        const uint32_t aux32 = a.aux;
        Split<18>::W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = __vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 28;
        r.dw = fp_half_bits(a.d);
        return r;
    }
};

// the lane's 5 k steps of a gate/up row, in the order row_dot_80_sub16 sums them: t, t+32, t+64 (s), t+16, t+48 (s16);
// with kb = t / 8 the blocks are kb, kb + 4, kb + 8, kb + 2, kb + 6 (all 5 steps have the same iqs = 2 (t % 8): every
// FpGu format has 8 calls per 256-block at step 2)
template<int TG>
__device__ __forceinline__ void fp_gu_load5(typename FpGu<TG>::Raw (&raw)[5], const uint8_t* row, int kb, int iqs) {
    raw[0] = FpGu<TG>::load(row, kb, iqs);
    raw[1] = FpGu<TG>::load(row, kb + 4, iqs);
    raw[2] = FpGu<TG>::load(row, kb + 8, iqs);
    raw[3] = FpGu<TG>::load(row, kb + 2, iqs);
    raw[4] = FpGu<TG>::load(row, kb + 6, iqs);
}

// acc[c] += the row's call (kbx, iqs) against activation c: Split<TG>::apply on the decoded words, as row_dot_80_sub16
template<int TG, int NC>
__device__ __forceinline__ void fp_pstep80(const typename FpGu<TG>::Raw& a, int kbx, int iqs, const uint32_t* grid,
                                           const block_q8_1* x, const int (&off)[NC], float (&acc)[NC]) {
    const typename Split<TG>::W w = FpGu<TG>::decode(a, iqs, grid);
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] += Split<TG>::apply(w, x + off[c] + kbx * 8, iqs);
}

// One chunk pass of NC entries for one row from the lane's raw words: row_dot_80_sub16's tree, lane t == c stores.
// REFILL: the item's last pass - as soon as step j is consumed its raw slot takes the same step of the next item's row
// `next` (block-uniform; nullptr = no next item), so the next item's loads are in flight while this item finishes and
// the live raw words never exceed one item's (a second full buffer next to the first took the kernel past the 128
// registers of 2 blocks / SM: ptxas spilled 80-360 bytes per thread, in the chunk paths).  The lead is the same as a
// full double buffer's: a word is loaded one item of work before it is used.
template<int TG, int NC, bool REFILL>
__device__ __forceinline__ void fp_pchunk80(typename FpGu<TG>::Raw (&raw)[5], int kb, int iqs, const block_q8_1* xq,
                                            const int32_t* ent_tok, int e, int t, int r, size_t n_ff, float* dst,
                                            const uint32_t* grid, const uint8_t* next) {
    int off[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) off[c] = ent_tok[e + c] * 80;
    float s[NC], s16[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) { s[c] = 0.0f; s16[c] = 0.0f; }
    fp_pstep80<TG, NC>(raw[0], kb, iqs, grid, xq, off, s);
    if constexpr (REFILL) { if (next != nullptr) raw[0] = FpGu<TG>::load(next, kb, iqs); }
    fp_pstep80<TG, NC>(raw[1], kb + 4, iqs, grid, xq, off, s);
    if constexpr (REFILL) { if (next != nullptr) raw[1] = FpGu<TG>::load(next, kb + 4, iqs); }
    fp_pstep80<TG, NC>(raw[2], kb + 8, iqs, grid, xq, off, s);
    if constexpr (REFILL) { if (next != nullptr) raw[2] = FpGu<TG>::load(next, kb + 8, iqs); }
    fp_pstep80<TG, NC>(raw[3], kb + 2, iqs, grid, xq, off, s16);
    if constexpr (REFILL) { if (next != nullptr) raw[3] = FpGu<TG>::load(next, kb + 2, iqs); }
    fp_pstep80<TG, NC>(raw[4], kb + 6, iqs, grid, xq, off, s16);
    if constexpr (REFILL) { if (next != nullptr) raw[4] = FpGu<TG>::load(next, kb + 6, iqs); }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
        for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
        if (t == c) dst[(size_t) (e + c) * n_ff + (size_t) r] = s[c];
    }
}

// work item w -> its row's weights: group w / ntiles, row (w % ntiles) * 16 + subwarp of the 2 n_ff gate/up rows (as the
// default kernel; clamped to row 0 if past the end, such a sub-warp does not compute)
__device__ __forceinline__ const uint8_t* fp_gu_item(const unsigned long long* grp_ptr, const NativeExpertLayout& L, int w,
                                                     int ntiles, int subwarp, int& g, int& row) {
    g = w / ntiles;
    row = (w - g * ntiles) * 16 + subwarp;
    const int rc = row < 2 * (int) L.n_ff ? row : 0;
    const bool is_up = rc >= (int) L.n_ff;
    const int r = is_up ? rc - (int) L.n_ff : rc;
    return (const uint8_t*) grp_ptr[g] + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
}

template<int TG>
__global__ void __launch_bounds__(256, STRATA_FP_PERSIST_MINB) fp_gu_persist_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                               const int32_t* __restrict__ grp_start,
                                                               const int32_t* __restrict__ n_groups,
                                                               const int32_t* __restrict__ ent_tok,
                                                               const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                               float* __restrict__ gate, float* __restrict__ up) {
    using G = FpGu<TG>;
    STRATA_SHARED_ALIGN16_U32(s_grid, G::kGridWords);
    {
        const uint32_t* src = G::table();
        for (int i = threadIdx.x; i < G::kGridWords; i += 256) s_grid[i] = src[i];
    }
    __syncthreads();   // the only barrier: the loop below is barrier-free
    const uint32_t* grid = s_grid;
    const int nff = (int) L.n_ff;
    const int ng = *n_groups;
    const int ntiles = (2 * nff + 15) / 16;
    const int total = ng * ntiles;
    const int subwarp = threadIdx.x >> 4, t = threadIdx.x & 15;
    const int kb = t >> 3, iqs = 2 * (t & 7);
    typename G::Raw cur[5] = {};
    int w = blockIdx.x, g = 0, row = 0;
    if (w < total) {
        const uint8_t* wr = fp_gu_item(grp_ptr, L, w, ntiles, subwarp, g, row);
        fp_gu_load5<TG>(cur, wr, kb, iqs);
    }
    for (; w < total; w += gridDim.x) {
        const int wn = w + (int) gridDim.x;
        int gn = 0, rown = 0;
        const uint8_t* next = nullptr;   // the next item's row: its words replace this item's as they are consumed
        if (wn < total) next = fp_gu_item(grp_ptr, L, wn, ntiles, subwarp, gn, rown);
        bool refilled = false;
        if (row < 2 * nff) {   // whole sub-warp pairs (n_ff % 32 == 0)
            const bool is_up = row >= nff;
            const int r = is_up ? row - nff : row;
            float* dst = is_up ? up : gate;
            const int e0 = grp_start[g], e1 = grp_start[g + 1];
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = min(GRP_NC, e1 - e);
                const bool last = e + GRP_NC >= e1;   // the item's last chunk: its last pass refills the raw words
                // A chunk of 3 or 4 entries goes as two passes of <= 2 columns: a column's sum chain is the same (the
                // same 5 steps, the same apply values, the same order), only the raw words are decoded once more
                // (registers already in hand, grid in shared memory); fewer live accumulators and activation words.
                // The second pass sees lane t - 2 so that its `t == c` store is lane 2 + c's, the default kernel's
                // lane for column 2 + c.
                if (n == 1) {
                    if (last) fp_pchunk80<TG, 1, true>(cur, kb, iqs, xq, ent_tok, e, t, r, (size_t) nff, dst, grid, next);
                    else fp_pchunk80<TG, 1, false>(cur, kb, iqs, xq, ent_tok, e, t, r, (size_t) nff, dst, grid, nullptr);
                } else if (n == 2) {
                    if (last) fp_pchunk80<TG, 2, true>(cur, kb, iqs, xq, ent_tok, e, t, r, (size_t) nff, dst, grid, next);
                    else fp_pchunk80<TG, 2, false>(cur, kb, iqs, xq, ent_tok, e, t, r, (size_t) nff, dst, grid, nullptr);
                } else if (n == 3) {
                    fp_pchunk80<TG, 2, false>(cur, kb, iqs, xq, ent_tok, e, t, r, (size_t) nff, dst, grid, nullptr);
                    if (last) fp_pchunk80<TG, 1, true>(cur, kb, iqs, xq, ent_tok, e + 2, t - 2, r, (size_t) nff, dst, grid, next);
                    else fp_pchunk80<TG, 1, false>(cur, kb, iqs, xq, ent_tok, e + 2, t - 2, r, (size_t) nff, dst, grid, nullptr);
                } else {
                    fp_pchunk80<TG, 2, false>(cur, kb, iqs, xq, ent_tok, e, t, r, (size_t) nff, dst, grid, nullptr);
                    if (last) fp_pchunk80<TG, 2, true>(cur, kb, iqs, xq, ent_tok, e + 2, t - 2, r, (size_t) nff, dst, grid, next);
                    else fp_pchunk80<TG, 2, false>(cur, kb, iqs, xq, ent_tok, e + 2, t - 2, r, (size_t) nff, dst, grid, nullptr);
                }
                refilled |= last;
            }
        }
        if (next != nullptr) {
            if (!refilled) fp_gu_load5<TG>(cur, next, kb, iqs);   // an idle sub-warp, or a group without entries
            g = gn;
            row = rown;
        }
    }
}

// FpDown<TD>: a down format of the persistent kernel at n_ff 640 (hb == 20 q8_1 blocks per staged h row).  kLanes lanes
// per row and kRows rows per tile (the rows one block of the default kernel owns), kSteps raw k steps per lane;
// load_steps() the lane's weight words in the order the default tree sums them; chunk() that tree over the NC staged h
// rows, lane t == c storing column c.
template<int TD> struct FpDown;

template<> struct FpDown<20> {   // IQ4_NL: row_dot_40_sub8 (40 calls, 8 lanes, 32 rows per block)
    struct Raw { int a, b; unsigned int d; };   // the two qs words (iqs, iqs + 1) and d
    static constexpr int kLanes = 8, kRows = 32, kSteps = 5;
    __device__ static Raw load(const uint8_t* row, int kbx, int iqs) {
        const block_iq4_nl* b = (const block_iq4_nl*) row + kbx;
        Raw r;
        r.a = fp_ld32b2(b->qs, iqs + 0);
        r.b = fp_ld32b2(b->qs, iqs + 1);
        r.d = fp_ld16(&b->d);
        return r;
    }
    // the lane's 5 k steps in row_dot_40_sub8's order: t, t+32 (s), t+16 (s16), t+8 (s8), t+24 (s24); with kb = t / 2
    // the blocks are kb, kb + 16, kb + 8, kb + 4, kb + 12 (all with iqs = 2 (t % 2))
    __device__ static void load_steps(Raw (&raw)[5], const uint8_t* row, int t) {
        const int kb = t >> 1, iqs = 2 * (t & 1);
        raw[0] = load(row, kb, iqs);
        raw[1] = load(row, kb + 16, iqs);
        raw[2] = load(row, kb + 8, iqs);
        raw[3] = load(row, kb + 4, iqs);
        raw[4] = load(row, kb + 12, iqs);
    }
    template<int NC>
    __device__ static void step(const Raw& a, int kbx, int iqs, const block_q8_1* s_hq, float (&acc)[NC]) {
        Split<20>::W w;   // Split<20>::load from the raw words
        w.v[0] = get_int_from_table_16(a.a, kvalues_iq4nl);
        w.v[1] = get_int_from_table_16(a.b, kvalues_iq4nl);
        w.dw = fp_half_bits(a.d);
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[c] += Split<20>::apply(w, s_hq + c * 20 + kbx, iqs);
    }
    template<int NC>
    __device__ static void chunk(const Raw (&raw)[5], int t, const block_q8_1* s_hq, const int32_t* ent_dst, int e, int r,
                                 size_t n_embd, float* out) {
        const int kb = t >> 1, iqs = 2 * (t & 1);
        float s[NC], s16[NC], s8[NC], s24[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) { s[c] = 0.0f; s16[c] = 0.0f; s8[c] = 0.0f; s24[c] = 0.0f; }
        step<NC>(raw[0], kb, iqs, s_hq, s);
        step<NC>(raw[1], kb + 16, iqs, s_hq, s);
        step<NC>(raw[2], kb + 8, iqs, s_hq, s16);
        step<NC>(raw[3], kb + 4, iqs, s_hq, s8);
        step<NC>(raw[4], kb + 12, iqs, s_hq, s24);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            s[c] = __fadd_rn(s[c], s16[c]);
            s[c] = __fadd_rn(s[c], __fadd_rn(s8[c], s24[c]));
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
            if (t == c) out[(size_t) ent_dst[e + c] * n_embd + (size_t) r] = s[c];
        }
    }
};

template<> struct FpDown<42> {   // Q2_0: row_dot_20_sub16 (20 calls, 16 lanes, 16 rows per block)
    struct Raw { unsigned int q01, q23, d; };   // the call's 4 int16 qs words (pairs) and d
    static constexpr int kLanes = 16, kRows = 16, kSteps = 2;
    __device__ static Raw load(const uint8_t* row, int kbx, int iqs) {
        const block_q2_0* b = (const block_q2_0*) row + kbx;
        const uint8_t* q = b->qs + 8 * iqs;   // (const int16_t*) qs + iqs * 4
        Raw r;
        r.q01 = fp_ld16(q) | (fp_ld16(q + 2) << 16);
        r.q23 = fp_ld16(q + 4) | (fp_ld16(q + 6) << 16);
        r.d = fp_ld16(&b->d);
        return r;
    }
    // row_dot_20_sub16: call k = t (block t / 2, part t % 2) into s; k = t + 16 (block 8 + t / 2) into s16 for t < 4
    // only - the row has 10 blocks, lanes 4..15 have no second call and load nothing
    __device__ static void load_steps(Raw (&raw)[2], const uint8_t* row, int t) {
        raw[0] = load(row, t >> 1, t & 1);
        if (t < 4) raw[1] = load(row, 8 + (t >> 1), t & 1);
    }
    template<int NC>
    __device__ static void step(const Raw& a, int kbx, int iqs, const block_q8_1* s_hq, float (&acc)[NC]) {
        Split<42>::W w;   // Split<42>::load from the raw words: q = (int) (int16_t) qs[j], sign-extended as there
        w.d2 = fp_half_bits(a.d);
        const int q[4] = { (int) (int16_t) (a.q01 & 0xFFFFu), (int) (int16_t) (a.q01 >> 16),
                           (int) (int16_t) (a.q23 & 0xFFFFu), (int) (int16_t) (a.q23 >> 16) };
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q[j] >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q[j] >> 2);
            w.qx[j] = __byte_perm(qe, qo, 0x5140);
            w.qy[j] = __byte_perm(qe, qo, 0x7362);
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) acc[c] += Split<42>::apply(w, s_hq + c * 20 + kbx * 2, iqs);
    }
    template<int NC>
    __device__ static void chunk(const Raw (&raw)[2], int t, const block_q8_1* s_hq, const int32_t* ent_dst, int e, int r,
                                 size_t n_embd, float* out) {
        float s[NC], s16[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) { s[c] = 0.0f; s16[c] = 0.0f; }
        step<NC>(raw[0], t >> 1, t & 1, s_hq, s);
        if (t < 4) step<NC>(raw[1], 8 + (t >> 1), t & 1, s_hq, s16);
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            s[c] = __fadd_rn(s[c], s16[c]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s[c] += __shfl_xor_sync(0xffffffffu, s[c], o);
            if (t == c) out[(size_t) ent_dst[e + c] * n_embd + (size_t) r] = s[c];
        }
    }
};

// work item w -> its row: group w / ntiles, row (w % ntiles) * rows_per_tile + sub of the n_embd down rows (clamped to
// row 0 if past the end, as the default kernel; such a row does not compute)
__device__ __forceinline__ const uint8_t* fp_down_item(const unsigned long long* grp_ptr, const NativeExpertLayout& L,
                                                       int w, int ntiles, int rows_per_tile, int sub, int& g, int& r) {
    g = w / ntiles;
    r = (w - g * ntiles) * rows_per_tile + sub;
    return (const uint8_t*) grp_ptr[g] + L.down_off + (size_t) (r < (int) L.n_embd ? r : 0) * L.d_row;
}

template<int TD>
__global__ void __launch_bounds__(256, STRATA_FP_PERSIST_MINB) fp_down_persist_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                                 const int32_t* __restrict__ grp_start,
                                                                 const int32_t* __restrict__ n_groups,
                                                                 const int32_t* __restrict__ ent_dst,
                                                                 const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                                 float* __restrict__ out) {
    using D = FpDown<TD>;
    __shared__ uint32_t s_hq_buf[GRP_NC * 20 * 9];
    const int nemb = (int) L.n_embd;
    const int ng = *n_groups;
    const int ntiles = (nemb + D::kRows - 1) / D::kRows;
    const int total = ng * ntiles;   // block-uniform: every barrier below is reached by all 256 threads
    const int sub = (int) threadIdx.x / D::kLanes, t = (int) threadIdx.x % D::kLanes;
    const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
    typename D::Raw cur[D::kSteps] = {}, nxt[D::kSteps] = {};
    int w = blockIdx.x, g = 0, r = 0;
    if (w < total) {
        const uint8_t* wr = fp_down_item(grp_ptr, L, w, ntiles, D::kRows, sub, g, r);
        D::load_steps(cur, wr, t);
    }
    int staged = -1;   // the chunk (its first entry) s_hq_buf holds; chunk starts are unique over all groups
    for (; w < total; w += gridDim.x) {
        const int wn = w + (int) gridDim.x;
        int gn = 0, rn = 0;
        if (wn < total) {   // the next item's weights in flight during this item's staging and work
            const uint8_t* wrn = fp_down_item(grp_ptr, L, wn, ntiles, D::kRows, sub, gn, rn);
            D::load_steps(nxt, wrn, t);
        }
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            if (e != staged) {
                const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
                const int words = n * (20 * 9);
                __syncthreads();   // the previous chunk's readers are done
                for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
                __syncthreads();
                staged = e;
            }
            if (r < nemb) {
                if (n == 1) D::template chunk<1>(cur, t, s_hq, ent_dst, e, r, (size_t) nemb, out);
                else if (n == 2) D::template chunk<2>(cur, t, s_hq, ent_dst, e, r, (size_t) nemb, out);
                else if (n == 3) D::template chunk<3>(cur, t, s_hq, ent_dst, e, r, (size_t) nemb, out);
                else D::template chunk<4>(cur, t, s_hq, ent_dst, e, r, (size_t) nemb, out);
            }
        }
        if (wn < total) {
#pragma unroll
            for (int j = 0; j < D::kSteps; ++j) cur[j] = nxt[j];
            g = gn;
            r = rn;
        }
    }
}

// ---------------------------------------------------------------- when the persistent kernels run (fp_expert_bench, lab/expert-persist)
// fp_expert_bench 300 -1 20 on the owner's cards (build a173807): V=4 bitwise on every pair; at T = 1 it beats the default
// kernels by up to 2x on some pairs (the default's 800-1600 short blocks per layer at ~10-15% of the DRAM peak), from
// T = 2 on the default kernels win by 1.0-1.3x.  So V=4 is taken per call:
//   STRATA_FP_EXPERT_V4_TMAX=N (hot lever EXPERT_V4_TMAX, graphs recaptured): with N > 0 only for calls of T <= N tokens
//     (T = the window tokens behind the call's groups, passed by the caller; 0 = unknown, then only when N == 0; N == 0 =
//     every call, the behaviour before this lever);
//   STRATA_FP_EXPERT_V4_PAIRS: the (device, gate/up / down format) pairs that take it - "gu/d,gu/d" for every device or
//     "dev:gu/d,gu/d;dev:..." per CUDA ordinal (a device with no entry of its own takes the device-less entries; unset or
//     empty = every pair), e.g. "0:18/20,21/42,22/42;1:18/20,17/42,18/42".
int g_fp_v4_tmax = [] {
    const char* v = std::getenv("STRATA_FP_EXPERT_V4_TMAX");
    const int x = v != nullptr ? std::atoi(v) : 0;
    return x > 0 ? x : 0;
}();

struct FpV4Pair { int dev, gu, d; };   // dev < 0: every device
std::vector<FpV4Pair> g_fp_v4_pairs;

bool fp_v4_pairs_parse(const char* spec, std::vector<FpV4Pair>& out) {
    out.clear();
    if (spec == nullptr) return true;
    std::string s(spec);
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(';', i);
        if (j == std::string::npos) j = s.size();
        std::string sec = s.substr(i, j - i);
        i = j + 1;
        if (sec.empty()) continue;
        int dev = -1;
        const size_t c = sec.find(':');
        if (c != std::string::npos) {
            char* end = nullptr;
            dev = (int) std::strtol(sec.c_str(), &end, 10);
            if (end == nullptr || *end != ':' || dev < 0) return false;
            sec = sec.substr(c + 1);
        }
        size_t k = 0;
        while (k < sec.size()) {
            size_t m = sec.find(',', k);
            if (m == std::string::npos) m = sec.size();
            const std::string item = sec.substr(k, m - k);
            k = m + 1;
            if (item.empty()) continue;
            const size_t sl = item.find('/');
            if (sl == std::string::npos || sl == 0 || sl + 1 >= item.size()) return false;
            char* e1 = nullptr;
            char* e2 = nullptr;
            const long gu = std::strtol(item.c_str(), &e1, 10), d = std::strtol(item.c_str() + sl + 1, &e2, 10);
            if (e1 == nullptr || *e1 != '/' || e2 == nullptr || *e2 != '\0' || gu <= 0 || d <= 0) return false;
            out.push_back(FpV4Pair{dev, (int) gu, (int) d});
        }
    }
    return true;
}

bool g_fp_v4_pairs_init = [] {
    const char* v = std::getenv("STRATA_FP_EXPERT_V4_PAIRS");
    if (v != nullptr && v[0] != '\0' && !fp_v4_pairs_parse(v, g_fp_v4_pairs)) {
        std::fprintf(stderr, "STRATA_FP_EXPERT_V4_PAIRS '%s' is not \"gu/d,...\" or \"dev:gu/d,...;...\": ignored (every pair)\n", v);
        g_fp_v4_pairs.clear();
    }
    return true;
}();

bool fp_v4_pair_ok(int gu, int d) {
    if (g_fp_v4_pairs.empty()) return true;
    int dev = -1;
    cudaGetDevice(&dev);
    bool own = false;   // the device has entries of its own: only those count
    for (const FpV4Pair& p : g_fp_v4_pairs) own = own || p.dev == dev;
    for (const FpV4Pair& p : g_fp_v4_pairs)
        if ((own ? p.dev == dev : p.dev < 0) && p.gu == gu && p.d == d) return true;
    return false;
}

// the persistent kernels for this call?  (the pair / shape checks of each role come on top)
bool fp_v4_now(const NativeExpertLayout& L, int tokens) {
    if (g_fp_variant != 4) return false;
    if (g_fp_v4_tmax > 0 && (tokens <= 0 || tokens > g_fp_v4_tmax)) return false;
    return fp_v4_pair_ok(L.gu_type, L.d_type);
}

// ---------------------------------------------------------------- STRATA_FP_DEF_TILES_T1: the default kernels' rows in fewer blocks at T = 1
// The bench's T = 1 reading (above) points at the default launch's shape, not its arithmetic: 80 x groups gate/up blocks
// and 80-160 x groups down blocks that each live for one row pass (one entry per row) - the persistent kernels, few
// long-lived blocks over the same rows, run the same work in half the time.  STRATA_FP_DEF_TILES_T1=k (default 1 = the
// default launch; the bench's "0t<k>" rows) keeps the default kernels' row code and launches, for calls of T == 1,
// grid.x = ceil(tiles / k) blocks that each take k row tiles in turn (tile = the rows one default block owns: 16
// gate/up rows; 32 IQ4_NL / 16 Q2_0 down rows), the codebook staged once per block and the h chunk once per (group,
// chunk) for all its tiles.  Every row is still one sub-warp running row_dot_80_sub16 / row_dot_40_sub8 /
// row_dot_20_sub16 on the same words: bitwise the default's.  (A "bigger grid" is not available without changing the
// per-row tree: a row is already one 16- or 8-lane sub-warp; the only free shape is rows per block, and the evidence
// says fewer, longer blocks.)
int g_fp_def_tiles_t1 = [] {
    const char* v = std::getenv("STRATA_FP_DEF_TILES_T1");
    const int x = v != nullptr ? std::atoi(v) : 1;
    return x >= 1 && x <= 64 ? x : 1;
}();

template<int TG, bool STAGE_GRID>
__global__ void __launch_bounds__(256) fp_gu_tiles_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                           const int32_t* __restrict__ grp_start,
                                                           const int32_t* __restrict__ n_groups,
                                                           const int32_t* __restrict__ ent_tok,
                                                           const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                           float* __restrict__ gate, float* __restrict__ up) {
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    STRATA_SHARED_ALIGN16_U32(s_grid_buf, IqGridWords<TG, STAGE_GRID>::value);
    const uint32_t* s_grid = stage_iq_grid<TG, STAGE_GRID>(s_grid_buf, threadIdx.x, 256);
    const int subwarp = threadIdx.x >> 4, t = threadIdx.x & 15;
    const int nff = (int) L.n_ff, ntiles = (2 * nff + 15) / 16;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
            const int row = tile * 16 + subwarp;
            if (row >= 2 * nff) continue;   // whole warps (n_ff % 32 == 0: a tile is all rows or none)
            const bool is_up = row >= nff;
            const int r = is_up ? row - nff : row;
            const uint8_t* wr = blob + (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
            float* dst = is_up ? up : gate;
            for (int e = e0; e < e1; e += GRP_NC) {   // native_gu_multi_kernel's SUB16 chunk code
                const int n = min(GRP_NC, e1 - e);
                if (n == 1) {
                    const int off1[1] = { ent_tok[e] * 80 };
                    float s1[1];
                    row_dot_80_sub16<TG, 1, true, STAGE_GRID>(wr, xq, off1, 1, t, s1, s_grid);
                    if (t == 0) dst[(size_t) e * L.n_ff + r] = s1[0];
                } else if (n == 2) {
                    const int off2[2] = { ent_tok[e] * 80, ent_tok[e + 1] * 80 };
                    float s2[2];
                    row_dot_80_sub16<TG, 2, true, STAGE_GRID>(wr, xq, off2, 2, t, s2, s_grid);
                    if (t < 2) dst[(size_t) (e + t) * L.n_ff + r] = s2[t];
                } else if (n == 3) {
                    const int off3[3] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80 };
                    float s3[3];
                    row_dot_80_sub16<TG, 3, true, STAGE_GRID>(wr, xq, off3, 3, t, s3, s_grid);
                    if (t < 3) dst[(size_t) (e + t) * L.n_ff + r] = s3[t];
                } else {
                    const int off4[4] = { ent_tok[e] * 80, ent_tok[e + 1] * 80, ent_tok[e + 2] * 80, ent_tok[e + 3] * 80 };
                    float s4[4];
                    row_dot_80_sub16<TG, 4, true, STAGE_GRID>(wr, xq, off4, 4, t, s4, s_grid);
                    if (t < 4) dst[(size_t) (e + t) * L.n_ff + r] = s4[t];
                }
            }
        }
    }
}

template<int TD>
__global__ void __launch_bounds__(256) fp_down_tiles_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                             const int32_t* __restrict__ grp_start,
                                                             const int32_t* __restrict__ n_groups,
                                                             const int32_t* __restrict__ ent_dst,
                                                             const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                             float* __restrict__ out) {
    using D = FpDown<TD>;   // kLanes lanes per row, kRows rows per tile, as the default kernel's block
    const int ng = *n_groups;
    if (blockIdx.y >= ng) return;
    __shared__ uint32_t s_hq_buf[GRP_NC * 20 * 9];
    const block_q8_1* s_hq = reinterpret_cast<const block_q8_1*>(s_hq_buf);
    const int nemb = (int) L.n_embd, ntiles = (nemb + D::kRows - 1) / D::kRows;
    const int sub = (int) threadIdx.x / D::kLanes, t = (int) threadIdx.x % D::kLanes;
    for (int g = blockIdx.y; g < ng; g += gridDim.y) {
        const uint8_t* blob = (const uint8_t*) grp_ptr[g];
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        for (int e = e0; e < e1; e += GRP_NC) {
            const int n = min(GRP_NC, e1 - e);
            const uint32_t* src = reinterpret_cast<const uint32_t*>(hq + (size_t) e * 20);
            const int words = n * (20 * 9);
            __syncthreads();   // block-uniform: the previous chunk's readers are done
            for (int i = threadIdx.x; i < words; i += 256) s_hq_buf[i] = src[i];
            __syncthreads();
            for (int tile = blockIdx.x; tile < ntiles; tile += gridDim.x) {
                const int r = tile * D::kRows + sub;
                if (r >= nemb) continue;   // whole warps (n_embd % 32 == 0)
                const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
                float* o = out + (size_t) r;
                // native_down_multi_kernel<TD, true>'s chunk code (row_dot_40_sub8 for IQ4_NL, row_dot_20_sub16 for Q2_0)
#define STRATA_FP_TILE_DOT(NC, OFF, S) \
                if constexpr (TD == 20) row_dot_40_sub8<TD, NC, true>(wr, s_hq, OFF, NC, t, S); \
                else row_dot_20_sub16<TD, NC, true>(wr, s_hq, OFF, NC, t, S);
                if (n == 1) {
                    const int off1[1] = { 0 };
                    float s1[1];
                    STRATA_FP_TILE_DOT(1, off1, s1)
                    if (t == 0) o[(size_t) ent_dst[e] * L.n_embd] = s1[0];
                } else if (n == 2) {
                    const int off2[2] = { 0, 20 };
                    float s2[2];
                    STRATA_FP_TILE_DOT(2, off2, s2)
                    if (t < 2) o[(size_t) ent_dst[e + t] * L.n_embd] = s2[t];
                } else if (n == 3) {
                    const int off3[3] = { 0, 20, 40 };
                    float s3[3];
                    STRATA_FP_TILE_DOT(3, off3, s3)
                    if (t < 3) o[(size_t) ent_dst[e + t] * L.n_embd] = s3[t];
                } else {
                    const int off4[4] = { 0, 20, 40, 60 };
                    float s4[4];
                    STRATA_FP_TILE_DOT(4, off4, s4)
                    if (t < 4) o[(size_t) ent_dst[e + t] * L.n_embd] = s4[t];
                }
#undef STRATA_FP_TILE_DOT
            }
        }
    }
}

// the default kernels' rows in ceil(tiles / k) blocks for this call?  (T == 1 and the default sub-warp shapes)
bool fp_def_tiles_now(const NativeExpertLayout& L, int tokens) {
    return g_fp_def_tiles_t1 > 1 && tokens == 1 && !g_old_kernels && !g_no_sub16_gu && L.n_embd == 2560 && L.n_ff == 640;
}

// The launch counter of STRATA_FP_DEF_TILES_T1 (native_expert_tiles_report): on the device, because the decode replays
// captured graphs - a host counter would count captures, one per (stage, T), not launches.  With the lever on every
// native_expert_grouped call launches fp_tiles_count_kernel (one block, ~1-2 us inside a graph) with the slot of what
// the call got: 0 = T 1 tiled, 1 = T 1 but the persistent kernels (V=4), 2 = T 1 but the S26 / S27 path (STRATA_EXPERT_V2 /
// V2K), 3 = T 1 but a shape or switch the tiles kernels do not take (old kernels, STRATA_NO_SUB16_GU, n_embd / n_ff),
// 4 = tokens unknown (0: a caller that does not pass T), 5 = T > 1.
__device__ unsigned long long g_fp_tiles_ctr[8];
__global__ void fp_tiles_count_kernel(int slot) {
    if (threadIdx.x == 0) atomicAdd(&g_fp_tiles_ctr[slot], 1ull);
}


template<int TG>
void launch_gu(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
               const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L,
               float* gate, float* up, int tokens) {
    if constexpr (kFpPersistGu<TG>) {   // STRATA_FP_EXPERT_V=4 (for this call: fp_v4_now): fp_gu_persist_kernel
        if (fp_v4_now(L, tokens) && fp_persist_gu_ok(L)) {
            fp_gu_persist_kernel<TG><<<dim3((unsigned) fp_persist_blocks()), 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            return;
        }
    }
    if constexpr (kSplit<TG>) {   // STRATA_FP_DEF_TILES_T1: the default rows in fewer blocks of k tiles (T == 1)
        if (fp_def_tiles_now(L, tokens)) {
            const unsigned nt = (unsigned) ((2 * L.n_ff + 15) / 16), k = (unsigned) g_fp_def_tiles_t1;
            const dim3 gt((nt + k - 1) / k, grid.y);
            if constexpr (kStageIqGrid<TG>) {
                if (g_stage_grid) {
                    fp_gu_tiles_kernel<TG, true><<<gt, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
                    return;
                }
            }
            fp_gu_tiles_kernel<TG, false><<<gt, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            return;
        }
    }
    if constexpr (TG == 21) {   // STRATA_FP_EXPERT_V=1..3: see fp_gu_kernel (the IQ3_S / IQ4_NL pair only)
        const int v = g_fp_variant;
        if (v >= 1 && v <= 3 && fp_pair_ok(L)) {
            const dim3 g2((unsigned) (((v >= 2 ? L.n_ff : 2 * L.n_ff) + 15) / 16), grid.y);
            if (v == 1) fp_gu_kernel<true, false><<<g2, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            else if (v == 2) fp_gu_kernel<false, true><<<g2, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            else fp_gu_kernel<true, true><<<g2, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            return;
        }
    }
    if constexpr (!kSplit<TG>) native_gu_kernel<TG><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    else if (g_old_kernels) native_gu_kernel<TG><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    else if constexpr (kStageIqGrid<TG>) {
        if (!g_stage_grid) {
            if (g_no_sub16_gu) native_gu_multi_kernel<TG, false, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            else native_gu_multi_kernel<TG, false, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
            return;
        }
        if (g_no_sub16_gu) native_gu_multi_kernel<TG, true, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        else native_gu_multi_kernel<TG, true, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    } else {
        if (g_no_sub16_gu) native_gu_multi_kernel<TG, false, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
        else native_gu_multi_kernel<TG, false, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up);
    }
}

template<int TD>
void launch_down(dim3 grid, cudaStream_t s, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout& L,
                 float* out, int tokens) {
    if constexpr (kFpPersistDown<TD>) {   // STRATA_FP_EXPERT_V=4 (for this call: fp_v4_now): fp_down_persist_kernel
        if (fp_v4_now(L, tokens) && fp_persist_down_ok(L)) {
            fp_down_persist_kernel<TD><<<dim3((unsigned) fp_persist_blocks()), 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            return;
        }
        if (fp_def_tiles_now(L, tokens)) {   // STRATA_FP_DEF_TILES_T1: the default rows in fewer blocks of k tiles (T == 1)
            const unsigned nt = (unsigned) ((L.n_embd + FpDown<TD>::kRows - 1) / FpDown<TD>::kRows), k = (unsigned) g_fp_def_tiles_t1;
            const dim3 gt((nt + k - 1) / k, grid.y);
            fp_down_tiles_kernel<TD><<<gt, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            return;
        }
    }
    if constexpr (TD == 20) {   // STRATA_FP_EXPERT_V=2..3: see fp_down_x2_kernel (the IQ3_S / IQ4_NL pair only)
        if (g_fp_variant >= 2 && g_fp_variant <= 3 && fp_pair_ok(L)) {
            const dim3 g2((unsigned) ((L.n_embd / 2 + 31) / 32), grid.y);
            fp_down_x2_kernel<<<g2, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
            return;
        }
    }
    if constexpr (!kSplit<TD>) native_down_kernel<TD><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else if (g_old_kernels) native_down_kernel<TD><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else if (g_no_sub16_gu) native_down_multi_kernel<TD, false><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
    else native_down_multi_kernel<TD, true><<<grid, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
}

}  // namespace

void iq_set_old_kernels(bool old) { g_old_kernels = old; }
bool iq_old_kernels() { return g_old_kernels; }
void native_expert_set_fp_variant(int v) { g_fp_variant = v >= 0 && v <= 4 ? v : 0; }
int native_expert_fp_variant() { return g_fp_variant; }
void native_expert_set_fp_v4_tmax(int tmax) { g_fp_v4_tmax = tmax > 0 ? tmax : 0; }
int native_expert_fp_v4_tmax() { return g_fp_v4_tmax; }
bool native_expert_set_fp_v4_pairs(const char* spec) {
    std::vector<FpV4Pair> parsed;
    if (!fp_v4_pairs_parse(spec, parsed)) return false;
    g_fp_v4_pairs = std::move(parsed);
    return true;
}
void native_expert_set_fp_def_tiles_t1(int k) { g_fp_def_tiles_t1 = k >= 1 && k <= 64 ? k : 1; }
int native_expert_fp_def_tiles_t1() { return g_fp_def_tiles_t1; }

std::string native_expert_tiles_report(bool reset) {
    char line[512];
    std::snprintf(line, sizeof line, "strata expert tiles T1 (STRATA_FP_DEF_TILES_T1=%d):", g_fp_def_tiles_t1);
    std::string out = line;
#if defined(__HIPCC__)
    out += " n/a on HIP";
    (void) reset;
#else
    int n_dev = 0, cur = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0) return out + " no device";
    cudaGetDevice(&cur);
    unsigned long long tot[8] = {};
    std::string per;
    for (int d = 0; d < n_dev; ++d) {
        if (cudaSetDevice(d) != cudaSuccess) continue;
        unsigned long long c[8] = {};
        if (cudaMemcpyFromSymbol(c, g_fp_tiles_ctr, sizeof c) != cudaSuccess) continue;   // a device the lib never ran on
        if (reset) {
            const unsigned long long z[8] = {};
            cudaMemcpyToSymbol(g_fp_tiles_ctr, z, sizeof z);
        }
        for (int i = 0; i < 6; ++i) tot[i] += c[i];
        std::snprintf(line, sizeof line, " dev %d: %llu tiled, %llu V4, %llu S26, %llu shape, %llu unknown, %llu T>1;", d, c[0], c[1],
                      c[2], c[3], c[4], c[5]);
        per += line;
    }
    cudaSetDevice(cur);
    std::snprintf(line, sizeof line,
                  " %llu launches at T=1 tiled, %llu launches at T=1 not tiled (%llu persistent V4, %llu S26/S27 path, %llu shape or "
                  "switch), %llu launches with T unknown, %llu launches at T>1 [per device:%s]",
                  tot[0], tot[1] + tot[2] + tot[3], tot[1], tot[2], tot[3], tot[4], tot[5], per.empty() ? " none" : per.c_str());
    out += line;
#endif
    return out;
}

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 14: return (size_t) (n / 256) * sizeof(block_q6_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 2: return (size_t) (n / 32) * sizeof(block_q4_0);
        case 3: return (size_t) (n / 32) * sizeof(block_q4_1);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 30: return (size_t) n * 2;   // BF16: the token embedding only (iq_embed_rows, iq_dequant_f32)
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    quantize_q8_1_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(x, (block_q8_1*) y, n);
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    cudaStream_t s = (cudaStream_t) stream;
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
#define STRATA_MMVQ(T) case T: launch_mmvq<T>(W, rb, X, y, n_in, n_out, ncols, s); break;
        STRATA_MMVQ_FMTS(STRATA_MMVQ)
#undef STRATA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<__half><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, (__half*) dst);
    check("iq_dequant_f16");
}

namespace {
__global__ void embed_rows_kernel(int ty, const uint8_t* __restrict__ table, size_t row_bytes,
                                  const int32_t* __restrict__ tokens, int64_t n_embd, float* __restrict__ y) {
    const int t = blockIdx.y;
    const int64_t b = blockIdx.x;
    const uint8_t* row = table + (size_t) tokens[t] * row_bytes;
    dq_dispatch<float>(ty, row, b, y + (size_t) t * n_embd + b * QK_K, threadIdx.x);
}
}  // namespace

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    embed_rows_kernel<<<dim3((unsigned) (n_embd / 256), (unsigned) n_tok), 32, 0, (cudaStream_t) stream>>>(
        t, (const uint8_t*) table, row_bytes, tokens, n_embd, out);
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    dequant_flat_kernel<float><<<(unsigned) (n / 256), 32, 0, (cudaStream_t) stream>>>(t, src, dst);
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    // checked like the other entry points: an unknown type used to leave `dst` unwritten, a wrong prompt and no error
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_gu_f16: type %d / %lld\n", t, (long long) n_embd); std::exit(1); }
    const int64_t per_row = n_embd / 256;
    dequant_gu_kernel<<<dim3((unsigned) (n_ff * per_row), 2), 32, 0, (cudaStream_t) stream>>>(t, gate, up, per_row,
                                                                                           (__half*) dst);
    check("iq_dequant_gu_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}


#if defined(STRATA_HIP_GFX906)
// ---- AMD layouts for the grouped native experts (STRATA_EXP_MODE; 0 = the CUDA one above).
// 1 (W64): a row per 64-lane wavefront - 4x the wavefronts, ~1-2 calls per lane, a 64-lane butterfly.
// 2 (R2):  a 32-lane logical warp computes TWO rows in one loop - two independent load chains in flight.
// Either way a (row, entry) sum does not depend on the window size.
template<int TY>
__device__ __forceinline__ float row_dot64(const uint8_t* row, const block_q8_1* x, int nb, int lane) {
    using F = Fmt<TY>;
    float s = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 64) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        s += F::dot(row, x + kbx * (F::qk / 32), kbx, iqs);
    }
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) s += __shfl_xor(s, o, 64);
    return s;
}
template<int TY>
__device__ __forceinline__ void row_dot2(const uint8_t* r0, const uint8_t* r1, const block_q8_1* x, int nb, int lane,
                                         float& o0, float& o1) {
    using F = Fmt<TY>;
    float s0 = 0.0f, s1 = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* xk = x + kbx * (F::qk / 32);
        const float a = F::dot(r0, xk, kbx, iqs);
        const float b = F::dot(r1, xk, kbx, iqs);
        s0 += a;
        s1 += b;
    }
    o0 = warp_sum(s0);
    o1 = warp_sum(s1);
}
template<int TY, int NR>
__device__ __forceinline__ void row_dotn(const uint8_t* const* r, const block_q8_1* x, int nb, int lane, float* o) {
    using F = Fmt<TY>;
    float acc[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = 0.0f;
    for (int k = lane; k < nb * F::ipb; k += 32) {
        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
        const block_q8_1* xk = x + kbx * (F::qk / 32);
        float d[NR];
#pragma unroll
        for (int i = 0; i < NR; ++i) d[i] = F::dot(r[i], xk, kbx, iqs);
#pragma unroll
        for (int i = 0; i < NR; ++i) acc[i] += d[i];
    }
#pragma unroll
    for (int i = 0; i < NR; ++i) o[i] = warp_sum(acc[i]);
}
template<int TG, int MODE>
__global__ void __launch_bounds__(256) native_gu_amd_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_tok,
                                                            const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                            float* __restrict__ gate, float* __restrict__ up) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = 2 * (int) L.n_ff;
    auto wrow = [&](int row) -> const uint8_t* {
        const bool is_up = row >= L.n_ff;
        return blob + (is_up ? L.up_off : 0) + (size_t) (is_up ? row - (int) L.n_ff : row) * L.gu_row;
    };
    auto put = [&](int row, int e, float v) {
        const bool is_up = row >= L.n_ff;
        (is_up ? up : gate)[(size_t) e * L.n_ff + (is_up ? row - (int) L.n_ff : row)] = v;
    };
    if constexpr (MODE == 1) {
        const int lane = threadIdx.x & 63, row = blockIdx.x * 4 + (threadIdx.x >> 6);
        if (row >= nrow) return;
        const uint8_t* wr = wrow(row);
        for (int e = e0; e < e1; ++e) {
            const float v = row_dot64<TG>(wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
            if (lane == 0) put(row, e, v);
        }
    } else if constexpr (MODE == 4) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int row0 = blockIdx.x * 32 + warp;
        if (row0 >= nrow) return;
        const uint8_t* w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = wrow(row0 + 8 * i < nrow ? row0 + 8 * i : row0);
        for (int e = e0; e < e1; ++e) {
            float o[4];
            row_dotn<TG, 4>(w, xq + (size_t) ent_tok[e] * xb, nb, lane, o);
            if (lane == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) if (row0 + 8 * i < nrow) put(row0 + 8 * i, e, o[i]);
        }
    } else {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int row0 = blockIdx.x * 16 + warp, row1 = row0 + 8;
        if (row0 >= nrow) return;
        const bool two = row1 < nrow;
        const uint8_t* w0 = wrow(row0);
        const uint8_t* w1 = wrow(two ? row1 : row0);
        for (int e = e0; e < e1; ++e) {
            float a, b;
            row_dot2<TG>(w0, w1, xq + (size_t) ent_tok[e] * xb, nb, lane, a, b);
            if (lane == 0) { put(row0, e, a); if (two) put(row1, e, b); }
        }
    }
}
template<int TD, int MODE>
__global__ void __launch_bounds__(256) native_down_amd_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_dst,
                                                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                              float* __restrict__ out) {
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = (int) L.n_embd;
    if constexpr (MODE == 1) {
        const int lane = threadIdx.x & 63, r = blockIdx.x * 4 + (threadIdx.x >> 6);
        if (r >= nrow) return;
        const uint8_t* wr = blob + L.down_off + (size_t) r * L.d_row;
        for (int e = e0; e < e1; ++e) {
            const float v = row_dot64<TD>(wr, hq + (size_t) e * hb, nb, lane);
            if (lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = v;
        }
    } else if constexpr (MODE == 4) {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int r0 = blockIdx.x * 32 + warp;
        if (r0 >= nrow) return;
        const uint8_t* w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = blob + L.down_off + (size_t) (r0 + 8 * i < nrow ? r0 + 8 * i : r0) * L.d_row;
        for (int e = e0; e < e1; ++e) {
            float o[4];
            row_dotn<TD, 4>(w, hq + (size_t) e * hb, nb, lane, o);
            if (lane == 0)
#pragma unroll
                for (int i = 0; i < 4; ++i) if (r0 + 8 * i < nrow) out[(size_t) ent_dst[e] * L.n_embd + r0 + 8 * i] = o[i];
        }
    } else {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        const int r0 = blockIdx.x * 16 + warp, r1 = r0 + 8;
        if (r0 >= nrow) return;
        const bool two = r1 < nrow;
        const uint8_t* w0 = blob + L.down_off + (size_t) r0 * L.d_row;
        const uint8_t* w1 = blob + L.down_off + (size_t) (two ? r1 : r0) * L.d_row;
        for (int e = e0; e < e1; ++e) {
            float a, b;
            row_dot2<TD>(w0, w1, hq + (size_t) e * hb, nb, lane, a, b);
            if (lane == 0) {
                out[(size_t) ent_dst[e] * L.n_embd + r0] = a;
                if (two) out[(size_t) ent_dst[e] * L.n_embd + r1] = b;
            }
        }
    }
}
// ---- 5 (LDS): gate/up of the grid formats (IQ2_S, IQ3_XXS, IQ3_S) with the codebook grid and the group's q8_1
// activations in LDS.  The bench showed gate/up compute-bound, not bandwidth-bound (T = 2 costs ~1.8x T = 1 on the
// same weights, 110-130 GB/s): each call issues its grid lookups and eight activation loads through the vector
// memory path after the weight load.  Same calls, same lane-strided k, same R2 row pairs, same warp_sum: the
// output is bitwise the mode-2 one.
template<int TY> struct GridOf;
template<> struct GridOf<22> { using T = uint64_t; static constexpr int N = 1024; __device__ static const T* src() { return iq2s_grid; } };
template<> struct GridOf<18> { using T = uint32_t; static constexpr int N = 256; __device__ static const T* src() { return iq3xxs_grid; } };
template<> struct GridOf<21> { using T = uint32_t; static constexpr int N = 512; __device__ static const T* src() { return iq3s_grid; } };
template<> struct GridOf<23> { using T = uint32_t; static constexpr int N = 1; __device__ static const T* src() { return iq3s_grid; } };   // IQ4_XS: no grid

// SG = 1 (mode 6): the signs without byte-SIMD ops, which gfx906 lacks (strata_vcmpne4 / strata_vsub4 unpack to
// ~12 word ops each).  m = the sign nibble as 0x00/0xff bytes; (g ^ m) is g or -g-1 per byte, so
// dp4a(g ^ m, u) - dp4a(m, u) = sum(s g u) exactly - the same integers, the same floats.
__device__ __forceinline__ int nib_mask(uint32_t n) {
    const uint32_t b = __umul24(n & 0xFu, 0x204081u) & 0x01010101u;
    return (int) ((b << 8) - b);
}
template<int TY, int SG> struct DotG;
template<int SG> struct DotG<22, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint64_t* grid) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const int qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq2->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int* grid_pos = (const int*) (grid + (qs[l0 / 2] | ((qh << (8 - l0)) & 0x300)));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        int& acc = l0 < 4 ? sumi0 : sumi1;
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs_packed_8[l0 / 2]), m1 = nib_mask(signs_packed_8[l0 / 2] >> 4);
            acc = ggml_cuda_dp4a(grid_pos[0] ^ m0, u0, acc);
            acc = ggml_cuda_dp4a(grid_pos[1] ^ m1, u1, acc);
            acc -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);
            acc = ggml_cuda_dp4a(grid_l, u0, acc);
            acc = ggml_cuda_dp4a(grid_h, u1, acc);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };
template<int SG> struct DotG<18, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t* grid) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* q3 = (const uint8_t*) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs), m1 = nib_mask(signs >> 4);
            sumi = ggml_cuda_dp4a(grid_pos.x ^ m0, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_pos.y ^ m1, u1, sumi);
            sumi -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
        }
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };
template<int SG> struct DotG<21, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t* grid) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t* qs = (const uint8_t*) &qs_packed;
    const int qh = bq3->qh[iqs / 2];
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
                                        grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if constexpr (SG == 1) {
            const int m0 = nib_mask(signs_packed_8[l0 / 2]), m1 = nib_mask(signs_packed_8[l0 / 2] >> 4);
            sumi = ggml_cuda_dp4a(grid_pos.x ^ m0, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_pos.y ^ m1, u1, sumi);
            sumi -= ggml_cuda_dp4a(m1, u1, ggml_cuda_dp4a(m0, u0, 0));
        } else {
            const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
            sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
            sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
        }
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs / 2].ds);
    return d * sumi;
} };

template<int SG> struct DotG<23, SG> { __device__ static __forceinline__ float f(const void* vbq, const block_q8_1* bq8_1, int kbx, int iqs,
                                                      const uint32_t*) { return vec_dot_iq4_xs_q8_1(vbq, bq8_1, kbx, iqs); } };

constexpr int LDS_RB = 64;    // gate/up rows per block (4 R2 passes of 16)
constexpr int LDS_NT = 4;     // entries whose activations sit in LDS at once

template<int TG, int SG, bool TI = false>
__global__ void __launch_bounds__(256) native_gu_lds_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                            const int32_t* __restrict__ grp_start,
                                                            const int32_t* __restrict__ n_groups,
                                                            const int32_t* __restrict__ ent_tok,
                                                            const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                            float* __restrict__ gate, float* __restrict__ up) {
    using GT = typename GridOf<TG>::T;
    using F = Fmt<TG>;
    __shared__ GT sgrid[GridOf<TG>::N];
    extern __shared__ int sx_raw[];   // LDS_NT tokens x xb blocks of q8_1 (36 bytes = 9 ints each)
    const int g = blockIdx.y;
    if (g >= *n_groups) return;       // uniform over the block
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const GT* gsrc = GridOf<TG>::src();
    for (int i = tid; i < GridOf<TG>::N; i += 256) sgrid[i] = gsrc[i];
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = 2 * (int) L.n_ff;
    const block_q8_1* sx = (const block_q8_1*) sx_raw;
    auto wrow = [&](int row) -> const uint8_t* {
        const bool is_up = row >= L.n_ff;
        return blob + (is_up ? L.up_off : 0) + (size_t) (is_up ? row - (int) L.n_ff : row) * L.gu_row;
    };
    auto put = [&](int row, int e, float v) {
        const bool is_up = row >= L.n_ff;
        (is_up ? up : gate)[(size_t) e * L.n_ff + (is_up ? row - (int) L.n_ff : row)] = v;
    };
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        for (int j = 0; j < cn; ++j) {
            const int* src = (const int*) (xq + (size_t) ent_tok[c0 + j] * xb);
            for (int i = tid; i < xb * 9; i += 256) sx_raw[j * xb * 9 + i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < LDS_RB / 16; ++p) {
            const int row0 = blockIdx.x * LDS_RB + p * 16 + warp, row1 = row0 + 8;
            if (row0 >= nrow) break;
            const bool two = row1 < nrow;
            const uint8_t* w0 = wrow(row0);
            const uint8_t* w1 = wrow(two ? row1 : row0);
            if constexpr (TI) {   // mode 7: the tokens inside the k loop - one weight load per k for all of them
                auto pass = [&](auto ntc) {
                    constexpr int NTC = decltype(ntc)::value;
                    float s0[NTC], s1[NTC];
#pragma unroll
                    for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                    for (int k = lane; k < nb * F::ipb; k += 32) {
                        const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                        for (int j = 0; j < NTC; ++j) {
                            const block_q8_1* xk = sx + (size_t) j * xb + kbx * (F::qk / 32);
                            const float a = DotG<TG, SG>::f(w0, xk, kbx, iqs, sgrid);
                            const float b = DotG<TG, SG>::f(w1, xk, kbx, iqs, sgrid);
                            s0[j] += a;
                            s1[j] += b;
                        }
                    }
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                        if (lane == 0) { put(row0, c0 + j, a); if (two) put(row1, c0 + j, b); }
                    }
                };
                switch (cn) {
                    case 1: pass(std::integral_constant<int, 1>{}); break;
                    case 2: pass(std::integral_constant<int, 2>{}); break;
                    case 3: pass(std::integral_constant<int, 3>{}); break;
                    default: pass(std::integral_constant<int, 4>{}); break;
                }
                continue;
            }
            for (int j = 0; j < cn; ++j) {
                const block_q8_1* x = sx + (size_t) j * xb;
                float s0 = 0.0f, s1 = 0.0f;
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
                    const block_q8_1* xk = x + kbx * (F::qk / 32);
                    const float a = DotG<TG, SG>::f(w0, xk, kbx, iqs, sgrid);
                    const float b = DotG<TG, SG>::f(w1, xk, kbx, iqs, sgrid);
                    s0 += a;
                    s1 += b;
                }
                s0 = warp_sum(s0);
                s1 = warp_sum(s1);
                if (lane == 0) { put(row0, c0 + j, s0); if (two) put(row1, c0 + j, s1); }
            }
        }
    }
}

// ---- 7 (down): the group's q8_1 h rows in LDS, the tokens inside the k loop, the R2 row pairs of mode 2 -
// the same per-(row, entry) sums in the same order: bitwise the mode-2 output.
template<int TD>
__global__ void __launch_bounds__(256) native_down_lds_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_dst,
                                                              const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                                                              float* __restrict__ out) {
    using F = Fmt<TD>;
    extern __shared__ int sh_raw[];   // LDS_NT entries x hb blocks of q8_1
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_ff / F::qk), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int nrow = (int) L.n_embd;
    const block_q8_1* sh = (const block_q8_1*) sh_raw;
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        {
            const int* src = (const int*) (hq + (size_t) c0 * hb);   // the chunk's entries are contiguous
            for (int i = tid; i < cn * hb * 9; i += 256) sh_raw[i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < LDS_RB / 16; ++p) {
            const int r0 = blockIdx.x * LDS_RB + p * 16 + warp, r1 = r0 + 8;
            if (r0 >= nrow) break;
            const bool two = r1 < nrow;
            const uint8_t* w0 = blob + L.down_off + (size_t) r0 * L.d_row;
            const uint8_t* w1 = blob + L.down_off + (size_t) (two ? r1 : r0) * L.d_row;
            auto pass = [&](auto ntc) {
                constexpr int NTC = decltype(ntc)::value;
                float s0[NTC], s1[NTC];
#pragma unroll
                for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const block_q8_1* xk = sh + (size_t) j * hb + kbx * (F::qk / 32);
                        const float a = F::dot(w0, xk, kbx, iqs);
                        const float b = F::dot(w1, xk, kbx, iqs);
                        s0[j] += a;
                        s1[j] += b;
                    }
                }
#pragma unroll
                for (int j = 0; j < NTC; ++j) {
                    const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                    if (lane == 0) {
                        out[(size_t) ent_dst[c0 + j] * L.n_embd + r0] = a;
                        if (two) out[(size_t) ent_dst[c0 + j] * L.n_embd + r1] = b;
                    }
                }
            };
            switch (cn) {
                case 1: pass(std::integral_constant<int, 1>{}); break;
                case 2: pass(std::integral_constant<int, 2>{}); break;
                case 3: pass(std::integral_constant<int, 3>{}); break;
                default: pass(std::integral_constant<int, 4>{}); break;
            }
        }
    }
}

// ---- 8: mode 7's gate/up with SwiGLU and the q8_1 quantization in the epilogue.  A block takes h rows
// [r0, r0 + 32): gate rows r0.. and up rows r0.. (64 weight rows, the same per-(row, entry) sums as mode 7), so
// one 32-lane warp per entry holds exactly one q8_1 block of h and runs quantize_q8_1_kernel's own code on it -
// bitwise the separate kernels' hq, two launches fewer per call.
template<int TG>
__global__ void __launch_bounds__(256) native_gu_fused_kernel(const unsigned long long* __restrict__ grp_ptr,
                                                              const int32_t* __restrict__ grp_start,
                                                              const int32_t* __restrict__ n_groups,
                                                              const int32_t* __restrict__ ent_tok,
                                                              const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                                                              block_q8_1* __restrict__ hq) {
    using GT = typename GridOf<TG>::T;
    using F = Fmt<TG>;
    __shared__ GT sgrid[GridOf<TG>::N];
    __shared__ float res[LDS_NT][64];
    extern __shared__ int sx_raw[];
    const int g = blockIdx.y;
    if (g >= *n_groups) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const GT* gsrc = GridOf<TG>::src();
    for (int i = tid; i < GridOf<TG>::N; i += 256) sgrid[i] = gsrc[i];
    const uint8_t* blob = (const uint8_t*) grp_ptr[g];
    const int nb = (int) (L.n_embd / F::qk), xb = (int) (L.n_embd / 32), hb = (int) (L.n_ff / 32);
    const int e0 = grp_start[g], e1 = grp_start[g + 1];
    const int r0 = blockIdx.x * 32;
    const block_q8_1* sx = (const block_q8_1*) sx_raw;
    auto wrow = [&](int i) -> const uint8_t* {   // i < 32: gate row r0 + i, else up row r0 + i - 32
        return blob + (i < 32 ? (size_t) 0 : L.up_off) + (size_t) (r0 + (i & 31)) * L.gu_row;
    };
    for (int c0 = e0; c0 < e1; c0 += LDS_NT) {
        const int cn = min(LDS_NT, e1 - c0);
        __syncthreads();
        for (int j = 0; j < cn; ++j) {
            const int* src = (const int*) (xq + (size_t) ent_tok[c0 + j] * xb);
            for (int i = tid; i < xb * 9; i += 256) sx_raw[j * xb * 9 + i] = src[i];
        }
        __syncthreads();
        for (int p = 0; p < 4; ++p) {
            const int i0 = p * 16 + warp, i1 = i0 + 8;
            const uint8_t* w0 = wrow(i0);
            const uint8_t* w1 = wrow(i1);
            auto pass = [&](auto ntc) {
                constexpr int NTC = decltype(ntc)::value;
                float s0[NTC], s1[NTC];
#pragma unroll
                for (int j = 0; j < NTC; ++j) { s0[j] = 0.0f; s1[j] = 0.0f; }
                for (int k = lane; k < nb * F::ipb; k += 32) {
                    const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
#pragma unroll
                    for (int j = 0; j < NTC; ++j) {
                        const block_q8_1* xk = sx + (size_t) j * xb + kbx * (F::qk / 32);
                        const float a = DotG<TG, 1>::f(w0, xk, kbx, iqs, sgrid);
                        const float b = DotG<TG, 1>::f(w1, xk, kbx, iqs, sgrid);
                        s0[j] += a;
                        s1[j] += b;
                    }
                }
#pragma unroll
                for (int j = 0; j < NTC; ++j) {
                    const float a = warp_sum(s0[j]), b = warp_sum(s1[j]);
                    if (lane == 0) { res[j][i0] = a; res[j][i1] = b; }
                }
            };
            switch (cn) {
                case 1: pass(std::integral_constant<int, 1>{}); break;
                case 2: pass(std::integral_constant<int, 2>{}); break;
                case 3: pass(std::integral_constant<int, 3>{}); break;
                default: pass(std::integral_constant<int, 4>{}); break;
            }
        }
        __syncthreads();
        if (warp < cn) {   // swiglu_entries_kernel + quantize_q8_1_kernel, one block of 32 h values per entry
            const float gg = res[warp][lane], uu = res[warp][32 + lane];
            const float xi = (gg / (1.0f + __expf(-gg))) * uu;
            float amax = fabsf(xi), sum = xi;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
                sum += __shfl_xor_sync(0xffffffffu, sum, o);
            }
            const float d = q8_1_finite(amax / 127.0f);   // #606, as q8_1_store: finite blocks bit for bit
            const int8_t q = q8_1_quant(xi, d, amax);
            block_q8_1* y = hq + (size_t) (c0 + warp) * hb + blockIdx.x;
            y->qs[lane] = q;
            if (lane == 0) y->ds = q8_1_ds(d, sum);
        }
    }
}

int g_exp_mode = -1;   // native_expert_set_mode (the bench); -1 = STRATA_EXP_MODE, default 7
int exp_mode() {
    static const int m = [] { const char* v = std::getenv("STRATA_EXP_MODE"); return v ? std::atoi(v) : 7; }();
    return g_exp_mode >= 0 ? g_exp_mode : m;
}
#endif
int g_exp_phase = 0;   // the bench: 0 all, 1 gate/up + swiglu + quantize only, 2 down only

void native_expert_set_mode(int mode, int phase) {
#if defined(STRATA_HIP_GFX906)
    g_exp_mode = mode;
#else
    (void) mode;
#endif
    g_exp_phase = phase;
}

namespace {
bool g_grouped_v1 = env_on("STRATA_GROUPED_V1");
}  // namespace

void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups, int tokens) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0) { std::fprintf(stderr, "native_expert_grouped: n_ff %lld\n", (long long) L.n_ff); std::exit(1); }
    cudaStream_t s = (cudaStream_t) stream;
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa);
    const auto* X = (const block_q8_1*) x_q8_1;
    static const bool v2 = [] { const char* v = std::getenv("STRATA_EXPERT_V2"); return v && v[0] == '1'; }();
    static const bool v2k = [] { const char* v = std::getenv("STRATA_EXPERT_V2K"); return v && v[0] == '1'; }();
    const bool s26 = v2 && L.gu_type == 21 && L.d_type == 20 && L.n_embd == 2560 && L.n_ff == 640;
    const bool s27 = v2k && (L.gu_type == 12 || L.gu_type == 13) && (L.d_type == 7 || L.d_type == 8) && L.n_embd == 2560 &&
                     L.n_ff == 640;
    if (g_fp_def_tiles_t1 > 1) {   // STRATA_FP_DEF_TILES_T1 on: count what this call gets (fp_tiles_count_kernel)
        int slot;
        if (tokens <= 0) slot = 4;
        else if (tokens > 1) slot = 5;
        else if (s26 || s27) slot = 2;
        else if (fp_v4_now(L, tokens) && (fp_persist_gu_ok(L) || fp_persist_down_ok(L))) slot = 1;
        else if (fp_def_tiles_now(L, tokens)) slot = 0;
        else slot = 3;
        fp_tiles_count_kernel<<<1, 32, 0, s>>>(slot);
    }
    if (s26) {   // S26: see s26_gu_l_kernel
        // S26 STRATA_TSUM=1: the sums as one transposed butterfly per warp (s26_tsum.cuh; bitwise the same values)
        static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
        // (+ down rows' items spread evenly over the lanes, one load group: bitwise, harness 1.09-1.12x vs 1.03-1.06x;
        // the same for gate / up was 0.6-0.8x)
        if (ts) s26_launch_l<true, true, 4, 4, true, true, true, 1>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h,
                                                             hq, out, (long long) cap_entries * L.n_ff);
        else s26_launch_l<true, true, 4, 4, true, true>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h,
                                                        hq, out, (long long) cap_entries * L.n_ff);
        check("native_expert_grouped (v2)");
        return;
    }
    // stream B: UD-Q4_K_XL's Q4_K / Q5_K gate/up and Q5_1 / Q8_0 down (see S27 above)
    if (s27) {
        static const bool ts = [] { const char* v = std::getenv("STRATA_TSUM"); return v && v[0] == '1'; }();
        const long long nh = (long long) cap_entries * L.n_ff;
#define STRATA_V2K(G, D) s27_launch_ts<G, D>(ts, L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh)
        if (L.gu_type == 12 && L.d_type == 7) STRATA_V2K(12, 7);
        else if (L.gu_type == 12) STRATA_V2K(12, 8);
        else if (L.d_type == 7) STRATA_V2K(13, 7);
        else STRATA_V2K(13, 8);
#undef STRATA_V2K
        check("native_expert_grouped (v2k)");
        return;
    }
    const bool v1 = g_grouped_v1;
    const int64_t gy = (v1 || grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    const int gu_rows = (!g_old_kernels && !g_no_sub16_gu && L.n_embd == 2560 && gu_split(L.gu_type)) ? 16 : GU_ROWS;
    const dim3 ggu((unsigned) ((2 * L.n_ff + gu_rows - 1) / gu_rows), (unsigned) gy);
#if defined(STRATA_HIP_GFX906)
    const int em0 = exp_mode();
#endif
#if defined(STRATA_HIP_GFX906)
    const bool fused_gu = em0 == 8 && (L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 || L.gu_type == 23) &&
                          L.n_ff % 32 == 0;
    if (fused_gu && g_exp_phase != 2) {
        const dim3 gl((unsigned) (L.n_ff / 32), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
        switch (L.gu_type) {
            case 18: native_gu_fused_kernel<18><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 21: native_gu_fused_kernel<21><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            case 23: native_gu_fused_kernel<23><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
            default: native_gu_fused_kernel<22><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, hq); break;
        }
        check("native_expert_grouped/gu fused");
    }
    if (g_exp_phase != 2 && !fused_gu) {
#else
    if (g_exp_phase != 2) {
#endif
#if defined(STRATA_HIP_GFX906)
    const bool lds_gu = (em0 == 5 || em0 == 6 || em0 == 7 || em0 == 8) && (L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 || L.gu_type == 23);
    if (lds_gu) {
        const dim3 gl((unsigned) ((2 * L.n_ff + LDS_RB - 1) / LDS_RB), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_embd / 32) * sizeof(block_q8_1);
        switch (L.gu_type) {
            case 18: if (em0 >= 7) native_gu_lds_kernel<18, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<18, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<18, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 21: if (em0 >= 7) native_gu_lds_kernel<21, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<21, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<21, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            case 23: if (em0 >= 7) native_gu_lds_kernel<23, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<23, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
            default: if (em0 >= 7) native_gu_lds_kernel<22, 1, true><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else if (em0 == 6) native_gu_lds_kernel<22, 1><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); else native_gu_lds_kernel<22, 0><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        }
    } else {
    const int em = exp_mode() >= 5 ? 2 : exp_mode();
    const dim3 ggu_amd((unsigned) ((2 * L.n_ff + (em == 1 ? 3 : em == 4 ? 31 : 15)) / (em == 1 ? 4 : em == 4 ? 32 : 16)), (unsigned) cap_groups);
#define STRATA_GU_AMD(T) \
    case T: if (em == 1) native_gu_amd_kernel<T, 1><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); \
            else if (em == 4) native_gu_amd_kernel<T, 4><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); \
            else native_gu_amd_kernel<T, 2><<<ggu_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
    // the AMD layouts cover the IQ packs' gate/up types; Unsloth's Q4_K / Q5_K take the CUDA layout below
    const bool amd_gu = L.gu_type == 16 || L.gu_type == 17 || L.gu_type == 18 || L.gu_type == 21 || L.gu_type == 22 ||
                        L.gu_type == 23 || L.gu_type == 29 || L.gu_type == 42;
    if ((em == 1 || em == 2 || em == 4) && amd_gu) {
        switch (L.gu_type) {
            STRATA_GU_AMD(16) STRATA_GU_AMD(17) STRATA_GU_AMD(18) STRATA_GU_AMD(21) STRATA_GU_AMD(22) STRATA_GU_AMD(23)
            STRATA_GU_AMD(29) STRATA_GU_AMD(42)
        }
    } else
#undef STRATA_GU_AMD
#endif
    switch (L.gu_type) {
#define STRATA_GU(T) case T: launch_gu<T>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up, tokens); break;
        STRATA_GU_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
#if defined(STRATA_HIP_GFX906)
    }
#endif
    check("native_expert_grouped/gu");
    static const double gate_keep_frac = [] {
        const char* v = std::getenv("STRATA_GATE_KEEP");
        const double f = v != nullptr ? std::atof(v) : 0.0;
        return f > 0.0 && f < 1.0 ? f : 0.0;
    }();
    if (gate_keep_frac > 0.0 && L.n_ff <= 1024) {
        const int keep = std::max(1, (int) std::lround(gate_keep_frac * (double) L.n_ff));
        if (keep < L.n_ff)
            gate_keep_kernel<<<(unsigned) cap_entries, 256, 0, s>>>(gate, up, grp_start, n_groups, (int) L.n_ff, keep);
        check("native_expert_grouped/gate_keep");
    }
    const long long nh = (long long) cap_entries * L.n_ff;
#if defined(STRATA_HIP_GFX906)
    // gfx906: the separate SwiGLU and q8_1 passes stay the default (the A/B baseline); the RDNA one-pass
    // kernel (bitwise the same) with STRATA_HIP_SWIGLU_FUSED=1
    static const bool sw_fused = env_on("STRATA_HIP_SWIGLU_FUSED");
    const bool sw_v1 = v1 || !sw_fused;
#else
    const bool sw_v1 = v1;
#endif
    if (sw_v1) {
        swiglu_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, h, nh);
        quantize_q8_1_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(h, hq, nh);
    } else {
        swiglu_q8_1_entries_kernel<<<(unsigned) ((nh + 255) / 256), 256, 0, s>>>(gate, up, grp_start, n_groups,
                                                                                (int) L.n_ff, hq);
    }
    check("native_expert_grouped/swiglu");
    }
    if (g_exp_phase == 1) return;
    const int d_rows = (!g_old_kernels && !g_no_sub16_gu && L.n_ff == 640) ? (L.d_type == 20 ? 32 : (L.d_type == 42 ? 16 : 8)) : 8;
    const dim3 gd((unsigned) ((L.n_embd + d_rows - 1) / d_rows), (unsigned) gy);
#if defined(STRATA_HIP_GFX906)
    if ((em0 == 7 || em0 == 8) && (L.d_type == 20 || L.d_type == 42)) {
        const dim3 gl((unsigned) ((L.n_embd + LDS_RB - 1) / LDS_RB), (unsigned) cap_groups);
        const size_t sh = (size_t) LDS_NT * (size_t) (L.n_ff / 32) * sizeof(block_q8_1);
        if (L.d_type == 20) native_down_lds_kernel<20><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        else native_down_lds_kernel<42><<<gl, 256, sh, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out);
        check("native_expert_grouped/down");
        return;
    }
    const int em = exp_mode() >= 5 ? 2 : exp_mode();
    const dim3 gd_amd((unsigned) ((L.n_embd + (em == 1 ? 3 : em == 4 ? 31 : 15)) / (em == 1 ? 4 : em == 4 ? 32 : 16)), (unsigned) cap_groups);
#define STRATA_D_AMD(T) \
    case T: if (em == 1) native_down_amd_kernel<T, 1><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); \
            else if (em == 4) native_down_amd_kernel<T, 4><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); \
            else native_down_amd_kernel<T, 2><<<gd_amd, 256, 0, s>>>(grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
    // the AMD layouts cover the IQ packs' down types; the others (Unsloth's Q4_K / Q5_K / Q5_1 / Q8_0) take the
    // CUDA layout below
    if ((em == 1 || em == 2 || em == 4) && (L.d_type == 20 || L.d_type == 23 || L.d_type == 42)) {
        switch (L.d_type) {
            STRATA_D_AMD(20) STRATA_D_AMD(23) STRATA_D_AMD(42)
        }
    } else
#undef STRATA_D_AMD
#endif
    switch (L.d_type) {
#define STRATA_DOWN(T) case T: launch_down<T>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out, tokens); break;
        STRATA_D_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}

}  // namespace strata::kernels
