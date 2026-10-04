#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
// Lab-only: keep original router IDs, then restrict to resident experts (mode 1)
// or drop nonresident contributions without renormalizing (mode 2). Mode 0 is exact.
// Caller must guarantee >=10 resident experts; mode/residency remain device data.
void closed_router_apply(float* logits, int32_t* ids, float* weights, int32_t* original_ids,
                         const int32_t* resident, const int32_t* mode, int n_tok, void* stream);
// Accumulate only committed rows of [layers,stride,10], into [layers,512].
void closed_router_count(const int32_t* ids, float* counts, int layers, int stride,
                         const int32_t* keep_device, int keep_host, void* stream);
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
}
