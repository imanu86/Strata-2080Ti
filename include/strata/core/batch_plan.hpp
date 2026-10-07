#pragma once
// The rows of a batch window with MTP drafts (--batch-mtp), as pure logic so a CPU test can pin it down:
//   * batch_slot_keep: how many of a slot's rows a window keeps (its current token + the drafts the target agreed
//     with), cut where the slot ends, so the slot's sessions never hold a row its emitted tokens do not;
//   * batch_plan_drafts (STRATA_BATCH_MTP_DRAFTS, opt-in): how many drafts each slot verifies in the next window, from
//     its drafter's own probabilities, so a slot with a confident drafter verifies more rows and one with a shaky
//     drafter none - within a bound on the window's rows.
//
// Default off: without STRATA_BATCH_MTP_DRAFTS every slot verifies its one draft (0.1.40 --batch-mtp), and
// batch_slot_keep with two rows is exactly that path's rule (see the test).

#include <cstdint>

namespace strata::core {

/// The rows of a slot's group a window keeps: 1 + the drafts accepted (row j's pick equals row j+1's token), cut at
/// the first row where the slot ends - its pick is an end token, a stop is pending, it reaches max_new, or the
/// context's end - because the emission ends the slot at that row and the kept rows must be the emitted ones.
/// `out[j]`: the target's pick at the slot's row j; `tok[j]`: the token row j fed (tok[0] the current one, then the
/// drafts); `rows`: the slot's rows in the window (>= 1); `produced`/`p`: the slot's count and position before the window.
/// A row past the first also needs its next position inside the context (p + keep + 2 <= max_context, as 0.1.40).
template <class IsEos>
inline int batch_slot_keep(const int32_t* out, const int32_t* tok, int rows, IsEos is_eos, bool stop,
                           int64_t produced, int64_t max_new, int64_t p, int64_t max_context) {
    int a = 0;   // drafts accepted
    while (a + 1 < rows && out[a] == tok[a + 1]) ++a;
    int keep = 1;
    while (keep <= a) {
        const int j = keep - 1;   // the last kept row: extending past it needs it not to end the slot
        const bool ends = is_eos(out[j]) || stop || produced + j + 1 >= max_new || p + j + 2 > max_context;
        if (ends || p + keep + 2 > max_context) break;
        ++keep;
    }
    return keep;
}

/// The drafts slot i may verify at most: what its drafter proposed (`avail`), no more than its remaining tokens need
/// (a window emits at most 1 + drafts) and inside the context (each row's position + 1 below it, as 0.1.40's p + 3).
inline int batch_slot_draft_cap(int avail, int64_t produced, int64_t max_new, int64_t p, int64_t max_context) {
    int64_t c = avail;
    if (max_new - produced - 1 < c) c = max_new - produced - 1;
    if (max_context - p - 2 < c) c = max_context - p - 2;
    return c < 0 ? 0 : (int) c;
}

/// Each slot's drafts for the next window (d[i] in [0, cap[i]]).  Every slot has its current row; the rows left
/// (up to `max_rows` in all) go one by one to the draft with the highest chance of being accepted - draft j of slot i
/// is accepted with prob[i][0] * .. * prob[i][j] (the drafter's probabilities, conditional per step) - while it
/// raises the window's expected tokens per millisecond: a window of R rows costs base_ms + row_ms * R, and a row
/// with gain g pays when g * (base_ms + row_ms * R) > row_ms * E (E: the expected tokens so far).  `min_gain`: a
/// floor on g (0 = only the cost model).  `scale`: multiplies every probability (the drafter's calibration;
/// 1 = taken as they are).  Returns the window's rows (n + the drafts).
inline int batch_plan_drafts(int n, const int* cap, const float* const* prob, int max_rows, double base_ms,
                             double row_ms, double min_gain, double scale, int* d) {
    for (int i = 0; i < n; ++i) d[i] = 0;
    int rows = n;
    double E = n;                 // each slot's current row yields one token
    double gain[64];              // the chance of slot i's next draft (all before it accepted too)
    if (n > 64) return rows;
    for (int i = 0; i < n; ++i) {
        gain[i] = cap[i] > 0 ? scale * (double) prob[i][0] : 0.0;
        if (gain[i] > 1.0) gain[i] = 1.0;
    }
    while (rows < max_rows) {
        int best = -1;
        for (int i = 0; i < n; ++i)
            if (d[i] < cap[i] && (best < 0 || gain[i] > gain[best])) best = i;
        if (best < 0) break;
        const double g = gain[best];
        if (g <= 0.0 || g < min_gain || g * (base_ms + row_ms * rows) <= row_ms * E) break;
        ++d[best];
        ++rows;
        E += g;
        gain[best] = d[best] < cap[best] ? g * scale * (double) prob[best][d[best]] : 0.0;
        if (gain[best] > 1.0) gain[best] = 1.0;
    }
    return rows;
}

}  // namespace strata::core
