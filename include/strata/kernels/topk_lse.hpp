// include/strata/kernels/topk_lse.hpp - lab (mtp_capture): a row's top-K and log-sum-exp on the device.
//
// The MTP distillation capture (core/mtp_capture.hpp) wants, for every prompt position, the head's K most probable
// tokens with their log-probabilities and the log-probability of the prompt's own next token.  Done on the host that
// is a heap over 248k logits plus 248k exp() per token: ~2 ms a token on one core, which caps a capture at ~500
// tokens/s however fast the GPU reads the prompt.  Here one block a row does it in place of the host.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// For `n_rows` rows of `n` floats: `ids[r * K + i]` = the K largest values' indices by descending value (ties: the lower
/// index first), `lps[r * K + i]` = that value minus the row's log-sum-exp (its log-probability under softmax), and, when
/// `next` is given, `nll[r]` = logits[r][next[r]] - lse (NaN when next[r] is outside [0, n)).  1 <= K <= 32; the sums are
/// float.  CUDA only.
void row_topk_lse(const float* logits, int n_rows, int n, int K, const int32_t* next, int32_t* ids, float* lps,
                  float* nll, void* stream);

}  // namespace strata::kernels
