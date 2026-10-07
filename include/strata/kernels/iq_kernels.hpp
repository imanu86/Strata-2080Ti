// include/strata/kernels/iq_kernels.hpp - the i-quant formats (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S,
// IQ4_NL) and Q2_0 on the GPU for the IQ2_XS / IQ3_XXS model files, and Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's
// UD-Q4_K_XL (gate/up Q4_K or Q5_K, down Q5_1 or Q8_0, a Q8_0 embedding).
//
// The block layouts, codebook grids and dot products are llama.cpp's (ggml-common.h, ggml-cuda/vecdotq.cuh,
// ggml-cuda/dequantize.cuh; MIT, see third_party/ggml/LICENSE and VERSION.txt), so a weight means exactly what it
// means in llama.cpp.  Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {

/// ggml type ids handled here.
bool iq_supported(int ggml_type) noexcept;
/// The token-embedding types iq_embed_rows and iq_dequant_f32 read: the i-quants above and BF16 (30).
bool embed_type_supported(int ggml_type) noexcept;
/// Bytes of one row of `n` values of `ggml_type` (n a multiple of the type's block).
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept;

/// q8_1 blocks for `n_rows` rows of `n_cols` floats (n_cols a multiple of 32): y is n_rows * n_cols/32 blocks.
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream);

/// y[c][r] = W[r] . x[c] for `ncols` columns of q8_1 activations (x stride n_in/32 blocks per column).
void iq_mmvq(int ggml_type, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// Dequantize `n` contiguous values (n a multiple of 256) to fp16 / fp32.
void iq_dequant_f16(int ggml_type, const void* src, int64_t n, uint16_t* dst, void* stream);
void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream);
/// Rows `tokens[0..n_tok)` (device ids) of a GGUF embedding table (`row_bytes` per row; the table may be mapped
/// host memory) dequantized to fp32, `n_embd` per row (a multiple of 256).
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream);
/// One expert's gate and up matrices (n_ff rows of n_embd each) into the interleaved fp16 layout the prompt path
/// uses: row 2r = gate row r, row 2r+1 = up row r.
void iq_dequant_gu_f16(int ggml_type, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst,
                       void* stream);

/// The layout of one native expert blob: [gate rows | up rows | down rows], raw GGUF blocks.
struct NativeExpertLayout {
    int gu_type = -1, d_type = -1;
    int64_t n_embd = 0, n_ff = 0;
    size_t gu_row = 0, d_row = 0;       // bytes per row
    size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
    size_t bytes = 0;                   // the whole blob
};
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff);
/// Whether `native_expert_grouped` has kernels for this gate/up and down type pair at these dimensions, and the
/// prompt path's dequantizer takes both (checked for every layer at startup, before anything is allocated).
bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept;

/// Bytes of scratch `native_expert_grouped` needs for `cap_entries` entries.
size_t native_expert_scratch_bytes(int64_t cap_entries, int64_t n_ff);

/// Grouped experts in the native format: group g's blob at device address grp_ptr[g]; its entries
/// [grp_start[g], grp_start[g+1]) read token ent_tok[e]'s q8_1 activation (n_embd/32 blocks per token in x_q8_1)
/// and write row ent_dst[e] of `out` (n_embd floats).  Counts are read on the device.
/// `grid_groups` (1 .. cap_groups; 0 = cap_groups) groups run side by side, a block row each striding over the rest:
/// a call that usually has few groups or none (the verify window's PCIe share) launches less for the ones it does
/// not have.  `tokens` = the window tokens behind the call's groups (T; 0 = unknown): the opt-in kernel choices below
/// (STRATA_FP_EXPERT_V4_TMAX, STRATA_FP_DEF_TILES_T1) read it.  The results depend on neither.
void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups = 0, int tokens = 0);
/// true: `native_expert_grouped`'s launches before the group stride (STRATA_GROUPED_V1=1 at startup) - a block row
/// per possible group, SwiGLU and the q8_1 quantization as two kernels over all cap_entries.  Bitwise the same results
/// (native_grouped_parity checks it); kept for A/B timing.  Set before graph capture; captured graphs keep theirs.
void native_grouped_set_v1(bool v1);

/// The bench only: the AMD kernel layout (STRATA_EXP_MODE values; -1 = the environment's) and the phase
/// (0 all, 1 gate/up + SwiGLU + quantize, 2 down).
void native_expert_set_mode(int mode, int phase);
/// `iq_mmvq` and `native_expert_grouped` decode each weight part once and apply it to every column / entry;
/// true selects the older kernels that decode it again per column (STRATA_OLD_IQ_MMVQ=1 at startup).  Both give
/// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
void iq_set_old_kernels(bool old);
bool iq_old_kernels();
/// Opt-in kernel variants of `native_expert_grouped`: 0 = the default kernels; 1 = gate/up with the IQ3_S codebook in
/// shared memory, 2 = two rows per sub-warp (gate/up and down), 3 = gate/up 1 + 2, down 2 - these three for IQ3_S
/// gate/up + IQ4_NL down at n_embd 2560 / n_ff 640 only (every other case keeps its kernels); 4 = persistent gate/up and
/// down (k x SM-count blocks, STRATA_FP_EXPERT_PERSIST_K = k, default 2, taking (group, row tile) items in a fixed order,
/// the next item's weights loaded ahead in registers, the codebook staged once per block) for gate/up IQ2_XXS (16),
/// IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22) at n_embd 2560 and down IQ4_NL (20), Q2_0 (42) at n_ff 640, each
/// role on its own (a role whose format is not covered keeps its default kernel).  All bitwise the same results as 0
/// (fp_expert_bench checks it).  The default is the environment's STRATA_FP_EXPERT_V (0 when unset); values outside
/// 0..4 count as 0.  Set before graph capture; captured graphs keep the kernels they captured.
void native_expert_set_fp_variant(int v);
int native_expert_fp_variant();
/// When variant 4 runs (fp_expert_bench: at T = 1 it beats the default kernels on some pairs, from T = 2 on it loses):
/// `tmax` > 0 = only for calls of `tokens` <= tmax (0 = every call, the default; a call with tokens 0 = unknown runs it
/// only when tmax is 0).  The default is the environment's STRATA_FP_EXPERT_V4_TMAX; hot lever EXPERT_V4_TMAX (graphs
/// recaptured as for EXPERT_V).
void native_expert_set_fp_v4_tmax(int tmax);
int native_expert_fp_v4_tmax();
/// The (device, gate/up / down format) pairs variant 4 runs on: "gu/d,gu/d" for every device, "dev:gu/d,...;dev:..."
/// per CUDA ordinal (a device without entries of its own takes the device-less ones); "" or nullptr = every pair.
/// False (table unchanged) on a malformed spec.  The default is the environment's STRATA_FP_EXPERT_V4_PAIRS.
bool native_expert_set_fp_v4_pairs(const char* spec);
/// STRATA_FP_DEF_TILES_T1: for calls of tokens == 1, the default kernels' rows in ceil(tiles / k) blocks that each take
/// k row tiles in turn (1 = the default launch, the default).  Bitwise the default's results (fp_expert_bench "0t<k>").
void native_expert_set_fp_def_tiles_t1(int k);
int native_expert_fp_def_tiles_t1();
/// With STRATA_FP_DEF_TILES_T1 > 1 every `native_expert_grouped` call also launches a one-block counting kernel (so a
/// captured graph counts at every replay): the line "strata expert tiles T1: N launches at T=1 tiled, M launches at
/// T=1 not tiled (why), ..., K launches at T>1" over every device (`reset` zeroes the counters).  Synchronous
/// (cudaMemcpyFromSymbol): call it between requests.  Hot lever DEF_TILES_T1 (graphs recaptured).
std::string native_expert_tiles_report(bool reset);

}  // namespace strata::kernels
