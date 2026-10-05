// src/kernels/cpu/q2_avx2.cpp - plan v0.3 P6: the Q2_0 expert rows and the activation quantizer for CPUs
// without AVX-512 (Intel Core 12th-14th gen and Core Ultra, AMD Zen 2/3).
//
// Compiled with AVX2 only, so nothing here can fault on those CPUs.  The arithmetic is the AVX-512 kernels':
// codes 0..3 against the int8 activation per 32-value chunk, times the weight scale and the chunk scale, minus
// the weight scale times the chunk's `hx` (the -1 code offset); the quantizer is the scalar rule, bit for bit.
#include "strata/kernels/cpu/expert.hpp"

#include <immintrin.h>

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h)));
}

// 16 bytes of 2-bit codes (value i in byte i/4, bits 2*(i%4)) -> 64 codes in value order, two 32-byte vectors
inline void unpack64(const uint8_t* codes, __m256i& lo, __m256i& hi) {
    const __m128i b = _mm_loadu_si128((const __m128i*) codes);
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);   // bytes 0..7
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);   // bytes 8..15
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));  // values 0..31
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));  // values 32..63
}

template <int NT>
inline void row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    const __m256i ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nblocks; ++b) {
        const uint8_t* blk = row + (size_t) b * 18;
        const float d = h2f(blk);
        __m256i lo, hi;
        unpack64(blk + 2, lo, hi);
        for (int t = 0; t < NT; ++t) {
            const int8_t* q = a[t]->q + b * 64;
            const __m256i s0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i*) q)), ones);
            const __m256i s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i*) (q + 32))), ones);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b]), _mm256_cvtepi32_ps(s0), acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b + 1]), _mm256_cvtepi32_ps(s1), acc[t]);
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[t]), _mm256_extractf128_ps(acc[t], 1));
        const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
        res[t] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr[t];
    }
}

template <int NT>
void rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

// ============================ the same arithmetic, cheaper code unpacking (default) ============================
//
// Bitwise the kernel above: `lo`/`hi` come out byte for byte as unpack64's, and every float operation (d times the
// chunk scale, the two FMAs per block in block order, the -1 offset correction, the reduction tree) is the same
// operation on the same operands in the same order.  What changes:
//   * THE UNPACK: 8 vector ops per block instead of ~17.  The 16 code bytes are broadcast to both 128-bit lanes and
//     zero-extended to 16-bit words in the order (bytes 0-3, 8-11 | 4-7, 12-15); with x = w | w << 6, word i of
//     x & 0x0303 holds the codes of values 4i, 4i+1 and word i of (x >> 4) & 0x0303 those of 4i+2, 4i+3; the two
//     16-bit interleaves then give values 0..31 and 32..63 in order.
//   * d is broadcast once per block for all tokens, and d * scale is a vector multiply by the broadcast scale
//     (the same rounding as the scalar product), so the per-chunk scalar-to-vector moves are gone.
//   * R rows of a short matrix are worked on together (4 for one token, 2 for two), so their FMA chains overlap;
//     a row's own sequence of operations does not depend on R.
// STRATA_Q2_LEGACY=1 keeps the kernel above (A/B).
inline void unpack64_fast(const uint8_t* codes, __m256i& lo, __m256i& hi) {
    const __m256i idx = _mm256_setr_epi8(0, -128, 1, -128, 2, -128, 3, -128, 8, -128, 9, -128, 10, -128, 11, -128,
                                         4, -128, 5, -128, 6, -128, 7, -128, 12, -128, 13, -128, 14, -128, 15, -128);
    const __m256i m = _mm256_set1_epi16(0x0303);
    const __m256i x2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*) codes));
    const __m256i w = _mm256_shuffle_epi8(x2, idx);
    const __m256i x = _mm256_or_si256(w, _mm256_slli_epi16(w, 6));
    const __m256i c01 = _mm256_and_si256(x, m);                           // word i: codes of 4i, 4i+1
    const __m256i c23 = _mm256_and_si256(_mm256_srli_epi16(x, 4), m);     // word i: codes of 4i+2, 4i+3
    lo = _mm256_unpacklo_epi16(c01, c23);   // values 0..15 | 16..31
    hi = _mm256_unpackhi_epi16(c01, c23);   // values 32..47 | 48..63
}

template <int NT, int R>
inline void rows_fast_group(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out,
                            int r) {
    __m256 acc[R][NT];
    float corr[R][NT];
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) { acc[k][t] = _mm256_setzero_ps(); corr[k][t] = 0.f; }
    const __m256i ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nblocks; ++b) {
        for (int k = 0; k < R; ++k) {
            const uint8_t* blk = w + (size_t) (r + k) * row_bytes + (size_t) b * 18;
            const float d = h2f(blk);
            const __m256 dv = _mm256_set1_ps(d);
            __m256i lo, hi;
            unpack64_fast(blk + 2, lo, hi);
            for (int t = 0; t < NT; ++t) {
                const int8_t* q = a[t]->q + b * 64;
                const __m256i s0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i*) q)), ones);
                const __m256i s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i*) (q + 32))), ones);
                acc[k][t] = _mm256_fmadd_ps(_mm256_mul_ps(dv, _mm256_broadcast_ss(&a[t]->scale[2 * b])),
                                            _mm256_cvtepi32_ps(s0), acc[k][t]);
                acc[k][t] = _mm256_fmadd_ps(_mm256_mul_ps(dv, _mm256_broadcast_ss(&a[t]->scale[2 * b + 1])),
                                            _mm256_cvtepi32_ps(s1), acc[k][t]);
                corr[k][t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
            }
        }
    }
    for (int k = 0; k < R; ++k)
        for (int t = 0; t < NT; ++t) {
            const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[k][t]), _mm256_extractf128_ps(acc[k][t], 1));
            const __m128 s = _mm_add_ps(h, _mm_movehl_ps(h, h));
            out[t][r + k] = _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s))) - corr[k][t];
        }
}

template <int NT, int R>
void rows_fast(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    int r = r0;
    for (; r + R <= r1; r += R) rows_fast_group<NT, R>(w, row_bytes, nblocks, a, out, r);
    for (; r < r1; ++r) rows_fast_group<NT, 1>(w, row_bytes, nblocks, a, out, r);
}

const bool kQ2Legacy = std::getenv("STRATA_Q2_LEGACY") != nullptr && std::getenv("STRATA_Q2_LEGACY")[0] == '1';

}  // namespace

void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                                      float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_0_gguf_rows_multi_avx2_legacy(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

void q2_0_gguf_rows_multi_avx2(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                               float* const* out, int r0, int r1) {
    if (kQ2Legacy) {
        q2_0_gguf_rows_multi_avx2_legacy(w, row_bytes, nblocks, a, nt, out, r0, r1);
        return;
    }
    switch (nt) {
        // short rows (a down projection: 10 blocks) are worked on in groups; long ones (gate/up: 40 blocks) one at a
        // time, where interleaved rows break the hardware prefetcher's single stream (measured 0.89x from RAM)
        case 1:
            if (nblocks <= 16) rows_fast<1, 4>(w, row_bytes, nblocks, a, out, r0, r1);
            else rows_fast<1, 1>(w, row_bytes, nblocks, a, out, r0, r1);
            break;
        case 2:
            if (nblocks <= 16) rows_fast<2, 2>(w, row_bytes, nblocks, a, out, r0, r1);
            else rows_fast<2, 1>(w, row_bytes, nblocks, a, out, r0, r1);
            break;
        case 3: rows_fast<3, 1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: rows_fast<4, 1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default:
            for (int t0 = 0; t0 < nt; t0 += 4) {
                const int k = nt - t0 < 4 ? nt - t0 : 4;
                q2_0_gguf_rows_multi_avx2(w, row_bytes, nblocks, a + t0, k, out + t0, r0, r1);
            }
    }
}

void act_quant_q8_1_avx2(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 half = _mm256_set1_ps(0.5f), mhalf = _mm256_set1_ps(-0.5f), zero = _mm256_setzero_ps();
    const __m256i lo = _mm256_set1_epi32(-127), hi = _mm256_set1_epi32(127);
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        __m256 v[4];
        __m256 m = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_loadu_ps(xb + 8 * i);
            m = _mm256_max_ps(m, _mm256_and_ps(v[i], absmask));
        }
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        const float amax = _mm_cvtss_f32(_mm_max_ss(h, _mm_movehdup_ps(h)));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        const __m256 vinv = _mm256_set1_ps(inv);
        __m256i sum = _mm256_setzero_si256();
        alignas(32) int32_t qi[QKA];
        for (int i = 0; i < 4; ++i) {
            const __m256 t = _mm256_mul_ps(v[i], vinv);
            const __m256 r = _mm256_add_ps(t, _mm256_blendv_ps(mhalf, half, _mm256_cmp_ps(t, zero, _CMP_GE_OQ)));
            __m256i q = _mm256_cvttps_epi32(r);
            q = _mm256_min_epi32(_mm256_max_epi32(q, lo), hi);
            sum = _mm256_add_epi32(sum, q);
            _mm256_store_si256((__m256i*) (qi + 8 * i), q);
        }
        for (int j = 0; j < QKA; ++j) a.q[k * QKA + j] = (int8_t) qi[j];
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        const int32_t total = _mm_cvtsi128_si32(s4);
        a.scale[k] = s;
        a.sum[k] = total;
        a.hx[k] = s * (float) total;
    }
}

}  // namespace strata::kernels::cpu
