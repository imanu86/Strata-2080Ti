#include "strata/prefill/request_chunk.hpp"

#include <cstdio>
#include <cstdlib>
#include <limits>

using strata::prefill::detail::request_chunk;

int main() {
    int checked = 0;
    auto check = [&](bool ok) { ++checked; if (!ok) { std::fprintf(stderr, "case %d failed\n", checked); std::exit(1); } };
    check(request_chunk(0, 8192, 1024, true) == 0);
    check(request_chunk(1007, 0, 1024, true) == 0);
    check(request_chunk(1007, 8192, 1024, false) == 1024);
    check(request_chunk(1007, 8192, 1024, true) == 1007);
    check(request_chunk(1023, 8192, 1024, true) == 1023);
    check(request_chunk(1024, 8192, 1024, true) == 1024);
    check(request_chunk(1025, 8192, 1024, true) == 1280);
    check(request_chunk(768, 8192, 1024, true) == 768);
    check(request_chunk(769, 8192, 1024, true) == 769);
    check(request_chunk(1007, 512, 1024, true) == 512);
    check(request_chunk(999, 8192, 1000, true) == 999);
    check(request_chunk(1000, 8192, 1000, true) == 1024);
    check(request_chunk(1007, 8192, 1000, true) == 1024);
    const int64_t hi = std::numeric_limits<int64_t>::max();
    check(request_chunk(hi, hi, 1024, true) == hi);
    check(request_chunk(hi - 255, hi, 1024, true) == hi - 255);
    check(request_chunk(hi - 256, hi, 1024, true) == hi - 255);
    check(request_chunk(hi - 254, hi, 1024, true) == hi - 254);
    // Across caps and non-grid thresholds: enough room for the actual chunk,
    // no false crossing, monotonic planning, and no change to streamed requests.
    for (const int64_t threshold : {1, 1000, 1024, 1500, 2048}) {
        for (const int64_t cap : {128, 512, 1000, 1024, 8192}) {
            int64_t previous = 0;
            for (int64_t t = 1; t <= 10000; ++t) {
                const int64_t n = request_chunk(t, cap, threshold, true);
                check(n >= std::min(t, cap) && n <= cap && n >= previous);
                if (t < threshold) check(n < threshold);
                if (t >= threshold || cap < threshold)
                    check(n == request_chunk(t, cap, threshold, false));
                previous = n;
            }
        }
    }
    std::printf("prefill request chunk: %d checks passed\n", checked);
}
