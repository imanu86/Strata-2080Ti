#pragma once
// Lab: single-token i-quant expert rows, bit-identical to ggml-cpu's AVX2 vec_dot (iq_1tok_avx2.cpp).
// AVX2 CPUs only: the caller checks the CPU before calling.

namespace strata::kernels::cpu {

enum class Iq1TokVariant { kScalar, kGather, kVsign };

/// One IQ2_XXS row (n weights) against one q8_K activation, the same float ggml_vec_dot_iq2_xxs_q8_K returns.
float iq2xxs_dot_1tok(Iq1TokVariant v, int n, const void* row, const void* act);

}  // namespace strata::kernels::cpu
