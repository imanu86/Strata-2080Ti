#pragma once

namespace strata::kernels {

// Set before session construction/capture; existing CUDA graphs retain their
// selected kernels. Disabled by default. Integration chooses the call sites.
void native_qsa_set_enabled(bool enabled);
bool native_qsa_enabled();

// Pinned single-token CUDA F32 arithmetic from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. No allocation or synchronization;
// both functions require a nonnull caller-owned stream and four-byte alignment.
// Positive dimensions with n_cols*n_rows (or head_dim*n_head) <= INT_MAX.
// Finite inputs and finite intermediate sums/products are caller preconditions.

// Contiguous input/output [n_cols,n_rows], gamma[n_cols] broadcast over rows.
// Each row has independent RMS statistics. Exact output==input is supported;
// otherwise all three spans must be disjoint. Epsilon must be finite and >= 0;
// epsilon==0 additionally requires a nonzero sum of squares in every row.
// Pinned 256-thread (width<1024) or 1024-thread XOR reduction, (scale*x)*gamma.
void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
                                  int n_cols, int n_rows, float epsilon, void* stream);

// The same per row with gamma[n_cols * groups]: row r is normalized on its own and scaled by gamma's slice
// r % groups (the MTP's hidden input, one row per hyper-connection stream: --mtp-hnorm stream).
void native_qsa_rms_norm_grouped(const float* input, const float* gamma, float* output,
                                 int n_cols, int groups, int n_rows, float epsilon, void* stream);

// attn/output [head_dim,n_head]; q_full [2*head_dim,n_head], each row is
// [query channels, gate channels]. Output is attn*sigmoid(second-half gate).
// Exact output==attn is supported; all other spans must be disjoint.
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream);

// chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1): native_qsa_rms_norm_weighted followed by native_quantize_q8_1
// of its output (n_cols * n_rows contiguous values, n_cols a multiple of 32) in ONE launch: the norm kernel, then the
// quantizer's expressions in its epilogue on the warp's 32 consecutive columns (the quantizer's own block and lane
// order).  `output` may be null (the normalized row is not kept); `x_q8_1` is native_q8_1_bytes(n_cols, n_rows).
// Same file and flags as the norm, the quantizer's helpers (q8_1_finite.hpp) under those flags: the same bytes.
void native_qsa_rms_norm_q8_1(const float* input, const float* gamma, float* output, void* x_q8_1,
                              int n_cols, int n_rows, float epsilon, void* stream);
// native_qsa_gate_apply followed by native_quantize_q8_1 of its output (n_head * head_dim a multiple of 32) in one
// launch; `output` may be null.
void native_qsa_gate_q8_1(const float* attn, const float* q_full, float* output, void* x_q8_1,
                          int n_head, int head_dim, void* stream);

} // namespace strata::kernels
