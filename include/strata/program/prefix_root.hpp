#pragma once

#include <cstdint>

namespace strata::program {

// The root turn before the current turn. The returned index
// excludes the boundary token from the prefix; ids[index] is its saved lookahead.
// No token after that boundary is needed: the boundary itself is the lookahead.
// Callers retain their original exclusive current-turn bound.
template <class Ids>
inline int64_t prefix_root_boundary(const Ids& ids, int64_t last_turn,
                                    int64_t turn_token, int64_t min_tokens, bool skip_short_turns) {
    if (turn_token < 0 || min_tokens <= 0 || last_turn <= 1 ||
        last_turn > static_cast<int64_t>(ids.size())) return -1;
    for (int64_t i = 1; i < last_turn; ++i)
        if (ids[static_cast<decltype(ids.size())>(i)] == turn_token) {
            if (i >= min_tokens) return i;
            if (!skip_short_turns) return -1; // preserve the ordinary RAM root policy
        }
    return -1;
}

} // namespace strata::program
