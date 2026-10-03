#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace strata::prefill::detail {

// Rounding the buffers up must not select a streamed ring for a segment that
// cannot use it. E.g. 1007 actual tokens run routed staging, even if their
// rounded 1024-token layout would reserve hundreds of streamed slots.
inline int64_t request_chunk(int64_t tokens, int64_t max_chunk, int64_t stream_min,
                             bool preserve_routed) {
    if (tokens <= 0 || max_chunk <= 0) return 0;
    const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255
                                ? tokens : ((tokens + 255) / 256) * 256;
    const int64_t planned = std::min(max_chunk, rounded);
    return preserve_routed && tokens < stream_min && planned >= stream_min ? tokens : planned;
}

}  // namespace strata::prefill::detail
