#include "strata/program/pack_prefill.hpp"
#include <cassert>
#include <limits>

int main() {
    using namespace strata::program::pack_prefill;
    assert(valid_hint({1, 99}, 11, 100));
    assert(!valid_hint({1, 100}, 11, 100));
    assert(!valid_hint({-1, 1}, 11, 100));
    assert(!valid_hint({1}, 11, 100));
    assert(!valid_hint({1, 2, 3}, 11, 100));
    assert(!valid_hint({1, 2}, 8, 100));
    assert(draft_rows_fit(1600, 10240));
    assert(!draft_rows_fit(8192, 10240));
    assert(!draft_rows_fit(std::numeric_limits<int64_t>::max(), 10240));
    assert(!draft_rows_fit(1, std::numeric_limits<int64_t>::max()));
    assert(next_chunk(20, 30, 4, 0, 20, 3, 4));
    assert(next_chunk(20, 30, 4, 12, 23, 7, 4));
    assert(!next_chunk(20, 30, 4, 12, 22, 8, 4)); // duplicate or out of order
    assert(!next_chunk(20, 30, 4, 12, 24, 6, 4)); // hole
    assert(!next_chunk(20, 30, 4, 12, 23, 8, 4)); // beyond the segment
    assert(!next_chunk(20, 30, 4, 12, 23, 7, 8)); // wrong residual geometry
    assert(!next_chunk(20, 30, 4, 13, 23, 7, 4)); // partial row
}
