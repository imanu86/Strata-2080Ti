#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::program::pack_prefill {

// A hint can execute before its GEN is validated. Apply the prompt bounds before caching or reading it.
inline bool valid_hint(const std::vector<int64_t>& ids, int64_t max_context, int64_t vocab) {
    return ids.size() >= 2 && max_context > 9 && ids.size() <= (size_t) (max_context - 9) &&
           std::all_of(ids.begin(), ids.end(), [vocab](int64_t t) { return t >= 0 && t < vocab; });
}

// Deferred draft reconstruction keeps FP32 residuals in RAM, never in the expert cache's VRAM.
// Longer suffixes fall back to an ordinary read; a cancelled hint cannot retain unbounded residual storage.
inline constexpr size_t max_draft_bytes = 256u * 1024u * 1024u;
inline bool draft_rows_fit(int64_t rows, int64_t width) {
    return rows > 0 && width > 0 && (uint64_t) width <= max_draft_bytes / sizeof(float) &&
           (uint64_t) rows <= max_draft_bytes / sizeof(float) / (uint64_t) width;
}

// Chunks may end inside a segment, but callbacks must cover its suffix exactly once and in order.
inline bool next_chunk(int64_t begin, int64_t end, int64_t width, size_t stored_floats,
                       int64_t pos, int64_t rows, int64_t row_width) {
    return width > 0 && width == row_width && stored_floats % (size_t) width == 0 &&
           pos >= begin && pos <= end && rows > 0 && rows <= end - pos &&
           (uint64_t) (pos - begin) == stored_floats / (size_t) width;
}

}  // namespace strata::program::pack_prefill
