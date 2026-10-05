// src/kernels/cpu/q2_exact_parity.cpp - the default AVX2 Q2_0 row kernel against the previous one
// (q2_0_gguf_rows_multi_avx2_legacy): every output must be bitwise the same.
//
// Random Q2_0 rows (codes, fp16 scales across a wide range, zeros and subnormals included), activations quantized
// by act_quant_q8_1_avx2 (zero chunks and mixed magnitudes included), 1-9 tokens, row ranges with remainders,
// padded and unpadded rows.  --bench times both kernels on one thread for the model's shapes (down: 2560 rows of
// 10 blocks; gate/up: 640 rows of 40), weights hot in cache and streamed from a buffer larger than the L3.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uni() { return (float) ((next() >> 40) & 0xFFFFFF) / 16777216.f; }   // [0, 1)
    float gauss() {
        float a = 0.f;
        for (int i = 0; i < 6; ++i) a += uni();
        return a - 3.f;
    }
};

uint16_t rand_half(Rng& g) {
    const uint64_t r = g.next();
    const uint16_t sign = (uint16_t) ((r & 1) << 15);
    switch ((r >> 1) % 32) {
        case 0: return sign;                                         // zero
        case 1: return (uint16_t) (sign | ((r >> 8) & 0x3FF));       // subnormal
        default: {
            const uint16_t e = (uint16_t) (5 + (r >> 8) % 21);        // 2^-10 .. 2^10
            return (uint16_t) (sign | (e << 10) | ((r >> 16) & 0x3FF));
        }
    }
}

void fill_rows(Rng& g, std::vector<uint8_t>& w, int rows, int nblocks, size_t row_bytes) {
    w.assign((size_t) rows * row_bytes, 0);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nblocks; ++b) {
            uint8_t* blk = w.data() + (size_t) r * row_bytes + (size_t) b * 18;
            const uint16_t d = rand_half(g);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) g.next();
        }
}

void fill_act(Rng& g, c::ActQ& a, int n) {
    std::vector<float> x((size_t) n);
    for (int k = 0; k < n / 32; ++k) {
        const uint64_t mode = g.next() % 8;
        const float mag = mode == 0 ? 0.f : std::ldexp(1.f, (int) (g.next() % 24) - 12);
        for (int j = 0; j < 32; ++j) x[(size_t) k * 32 + j] = mag * g.gauss();
    }
    c::act_quant_q8_1_avx2(x.data(), n, a);
}

int parity() {
    Rng g;
    long long cases = 0, values = 0, differ = 0;
    const int shapes[][2] = {{2560, 10}, {640, 40}, {37, 1}, {64, 2}, {129, 7}, {200, 40}, {33, 3}};
    for (const auto& sh : shapes) {
        for (int pad = 0; pad <= 6; pad += 6) {
            const int rows = sh[0], nb = sh[1];
            const size_t row_bytes = (size_t) nb * 18 + (size_t) pad;
            std::vector<uint8_t> w;
            fill_rows(g, w, rows, nb, row_bytes);
            static c::ActQ acts[9];
            for (int rep = 0; rep < 4; ++rep) {
                for (auto& a : acts) fill_act(g, a, nb * 64);
                for (int nt = 1; nt <= 9; ++nt) {
                    const int r0 = rep == 0 ? 0 : (int) (g.next() % (uint64_t) rows);
                    const int r1 = rep == 0 ? rows : r0 + (int) (g.next() % (uint64_t) (rows - r0 + 1));
                    std::vector<float> o1((size_t) nt * rows, -7.f), o2((size_t) nt * rows, -7.f);
                    const c::ActQ* ap[9];
                    float* p1[9];
                    float* p2[9];
                    for (int t = 0; t < nt; ++t) {
                        ap[t] = &acts[t];
                        p1[t] = o1.data() + (size_t) t * rows;
                        p2[t] = o2.data() + (size_t) t * rows;
                    }
                    c::q2_0_gguf_rows_multi_avx2_legacy(w.data(), row_bytes, nb, ap, nt, p1, r0, r1);
                    c::q2_0_gguf_rows_multi_avx2(w.data(), row_bytes, nb, ap, nt, p2, r0, r1);
                    for (size_t i = 0; i < o1.size(); ++i) {
                        uint32_t u1, u2;
                        std::memcpy(&u1, &o1[i], 4);
                        std::memcpy(&u2, &o2[i], 4);
                        if (u1 != u2) {
                            if (differ < 5)
                                std::printf("  differ: rows %d nb %d pad %d nt %d [%d,%d) idx %zu: %.9g vs %.9g\n", rows,
                                            nb, pad, nt, r0, r1, i, o1[i], o2[i]);
                            ++differ;
                        }
                    }
                    values += (long long) nt * (r1 - r0);
                    ++cases;
                }
            }
        }
    }
    std::printf("q2_exact_parity: %lld cases, %lld row outputs, %lld differ -> %s\n", cases, values, differ,
                differ ? "FAIL" : "PASS");
    return differ ? 1 : 0;
}

template <class F>
double time_ms(F&& f, int reps) {
    f();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
}

void bench() {
    Rng g;
    std::printf("\n--bench: one thread, ms per matrix (all rows), legacy -> default\n");
    std::printf("%-8s %5s %4s %3s | %9s %9s | %6s\n", "weights", "rows", "nb", "nt", "legacy", "default", "speedup");
    const int shapes[][2] = {{2560, 10}, {640, 40}};
    for (const auto& sh : shapes) {
        const int rows = sh[0], nb = sh[1];
        const size_t row_bytes = (size_t) nb * 18;
        const size_t mat = (size_t) rows * row_bytes;
        const int nmat_cold = (int) ((96u << 20) / mat) + 1;   // > L3: every matrix comes from RAM
        std::vector<uint8_t> w;
        fill_rows(g, w, rows * nmat_cold, nb, row_bytes);
        static c::ActQ acts[4];
        for (auto& a : acts) fill_act(g, a, nb * 64);
        std::vector<float> o((size_t) 4 * rows);
        const c::ActQ* ap[4] = {&acts[0], &acts[1], &acts[2], &acts[3]};
        float* op[4] = {o.data(), o.data() + rows, o.data() + 2 * rows, o.data() + 3 * rows};
        for (int cold = 0; cold <= 1; ++cold)
            for (int nt = 1; nt <= 4; ++nt) {
                int m = 0;
                auto run = [&](bool legacy) {
                    const uint8_t* base = w.data() + (cold ? (size_t) (m++ % nmat_cold) * mat : 0);
                    if (legacy) c::q2_0_gguf_rows_multi_avx2_legacy(base, row_bytes, nb, ap, nt, op, 0, rows);
                    else c::q2_0_gguf_rows_multi_avx2(base, row_bytes, nb, ap, nt, op, 0, rows);
                };
                const int reps = cold ? nmat_cold * 3 : 300;
                const double tl = time_ms([&] { run(true); }, reps);
                const double tf = time_ms([&] { run(false); }, reps);
                std::printf("%-8s %5d %4d %3d | %9.4f %9.4f | %5.2fx\n", cold ? "RAM" : "cache", rows, nb, nt, tl, tf,
                            tl / tf);
            }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!c::cpu_avx2_ok()) {
        std::printf("q2_exact_parity: no AVX2 on this CPU - skipped\n");
        return 77;
    }
    const int rc = parity();
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) bench();
    return rc;
}
