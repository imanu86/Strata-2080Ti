// src/kernels/cpu/iq_1tok_avx2.cpp - lab: single-token IQ2_XXS rows, bit-identical to ggml-cpu's AVX2 vec_dot.
//
// A verify window's CPU experts hold one token in ~3/4 of the cases, and a one-token group runs ggml-cpu's
// ggml_vec_dot_iq2_xxs_q8_K (the #152 rule keeps the multi-token kernels for two or more).  Disassembled (MSVC,
// /arch:AVX2), that loop spends ~40 instructions per 32 weights, about half of them scalar index work: the four
// 8-bit grid indices and the four 7-bit sign indices are each shifted, masked and loaded on their own, then put
// into a vector with vpinsrq/vinsertf128.  These variants build the same two vectors differently:
//   - kGather: one vpgatherqq for the four grid rows and one for the four sign rows of each 32 weights, from
//     indices unpacked with vector shifts;
//   - kScalar: ggml's own sequence (the reference arm, to time against with identical code generation).
// The integer arithmetic (sign, maddubs, madd with 2*ls+1, the two partial sums) and the float order (one fmadd
// per 256-weight block, ggml's hsum_float_8, the final 0.125) are ggml's, so the result is bit-identical.
#include "strata/kernels/cpu/iq_1tok.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>

#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// ggml-cpu/arch/x86/quants.c hsum_float_8, the exact order
inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// keven_signs_q2xs as ggml lays it out (ggml-cpu keeps it static): byte k = 0xFF when bit k of
// ksigns_iq2xs[i] (i's 7 bits plus an even-parity bit 7) is set, 0x01 otherwise.  constexpr: no static constructor
// in an AVX2 translation unit (see iq_avx2.cpp, #391).
struct EvenSigns {
    uint64_t v[128];
    constexpr EvenSigns() : v{} {
        for (int i = 0; i < 128; ++i) {
            int par = 0;
            for (int k = 0; k < 7; ++k) par ^= (i >> k) & 1;
            const int s = i | (par << 7);
            uint64_t r = 0;
            for (int k = 0; k < 8; ++k) r |= (uint64_t) (((s >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            v[i] = r;
        }
    }
};
static constexpr EvenSigns even_signs{};

inline float dot_scalar(int n, const block_iq2_xxs* x, const block_q8_K* y) {
    const uint64_t* signs64 = even_signs.v;
    const int nb = n / QK_K;
    uint32_t aux32[4];
    const uint8_t* aux8 = (const uint8_t*) aux32;
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = h2f(x[i].d) * y[i].d;
        const uint16_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            std::memcpy(aux32, q2, 4 * sizeof(uint32_t)); q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >> 7) & 127], signs64[(aux32[1] >> 0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >> 7) & 127], signs64[(aux32[3] >> 0) & 127]);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
            const uint16_t ls1 = aux32[1] >> 28, ls2 = aux32[3] >> 28;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125f * hsum_float_8(accumf);
}

inline float dot_gather(int n, const block_iq2_xxs* x, const block_q8_K* y) {
    const long long* grid = (const long long*) iq2xxs_grid;
    const long long* signs = (const long long*) even_signs.v;
    const int nb = n / QK_K;
    // 4 sign indices of a word: shifts 0, 7, 14, 21 (as 64-bit lanes, masked to 7 bits)
    const __m256i sh = _mm256_setr_epi64x(0, 7, 14, 21);
    const __m256i m7 = _mm256_set1_epi64x(127);
    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = h2f(x[i].d) * y[i].d;
        const uint32_t* q2 = (const uint32_t*) x[i].qs;
        const int8_t* q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256(), sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const uint32_t a0 = q2[0], a1 = q2[1], a2 = q2[2], a3 = q2[3];
            q2 += 4;
            // grid rows: the 4 bytes of a0 (resp. a2) zero-extended to 64-bit indices
            const __m256i gi1 = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128((int) a0));
            const __m256i gi2 = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128((int) a2));
            const __m256i q2_1 = _mm256_i64gather_epi64(grid, gi1, 8);
            const __m256i q2_2 = _mm256_i64gather_epi64(grid, gi2, 8);
            const __m256i si1 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x((long long) a1), sh), m7);
            const __m256i si2 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x((long long) a3), sh), m7);
            const __m256i s2_1 = _mm256_i64gather_epi64(signs, si1, 8);
            const __m256i s2_2 = _mm256_i64gather_epi64(signs, si2, 8);
            const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(q8_1, s2_1));
            const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(q8_2, s2_2));
            const uint16_t ls1 = a1 >> 28, ls2 = a3 >> 28;
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, _mm256_set1_epi16(2 * ls1 + 1)));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, _mm256_set1_epi16(2 * ls2 + 1)));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125f * hsum_float_8(accumf);
}

}  // namespace

float iq2xxs_dot_1tok(Iq1TokVariant v, int n, const void* row, const void* act) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) row;
    const block_q8_K* y = (const block_q8_K*) act;
    return v == Iq1TokVariant::kGather ? dot_gather(n, x, y) : dot_scalar(n, x, y);
}

}  // namespace strata::kernels::cpu
