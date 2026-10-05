// src/kernels/cpu/iq256_dump.cpp - lab: the AVX2 multi-token i-quant kernels (iq256_rows, iq256_gu_rows) on fixed
// pseudo-random inputs, every output written to a file.  Two builds (before/after a kernel change) must write the
// same bytes: a change that replaces a kernel in place leaves no legacy copy to compare against in one binary.
//
//     iq256_dump <out.bin>              every iq256 format, 1-9 tokens, row ranges with remainders
//     iq256_dump <out.bin> --bench      + one thread, us per gate/up matrix (640 rows x 10 blocks) for IQ3_S and IQ2_S
//
// STRATA_IQ256_GATHER=1 selects the gather path where a format has one (read once at startup): run both ways.
// Weights: random block bytes after an fp16 d spanning zeros, subnormals and 2^-10..2^10 (every index, sign and
// scale bit pattern is a valid encoding).  Activations: Q8_K blocks with codes in [-127, 127] (ggml's range),
// consistent bsums, zero blocks and mixed scales.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"

#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {

struct BlockQ8K {   // ggml's block_q8_K
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
static_assert(sizeof(BlockQ8K) == 292, "block_q8_K layout");

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

uint16_t rand_half(Rng& g) {
    const uint64_t r = g.next();
    const uint16_t sign = (uint16_t) ((r & 1) << 15);
    switch ((r >> 1) % 32) {
        case 0: return sign;
        case 1: return (uint16_t) (sign | ((r >> 8) & 0x3FF));
        default: {
            const uint16_t e = (uint16_t) (5 + (r >> 8) % 21);
            return (uint16_t) (sign | (e << 10) | ((r >> 16) & 0x3FF));
        }
    }
}

void fill_rows(Rng& g, std::vector<uint8_t>& w, int rows, int nb, size_t bsize, size_t row_bytes) {
    w.assign((size_t) rows * row_bytes, 0);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + (size_t) r * row_bytes + (size_t) b * bsize;
            const uint16_t d = rand_half(g);
            std::memcpy(blk, &d, 2);
            for (size_t i = 2; i < bsize; ++i) blk[i] = (uint8_t) g.next();
        }
}

void fill_act(Rng& g, std::vector<BlockQ8K>& a, int nb) {
    a.assign((size_t) nb, BlockQ8K{});
    for (auto& b : a) {
        const uint64_t mode = g.next() % 8;
        b.d = mode == 0 ? 0.f : std::ldexp(1.f + (float) (g.next() % 1024) / 1024.f, (int) (g.next() % 24) - 20);
        if (g.next() % 2) b.d = -b.d;
        for (int i = 0; i < 256; ++i) b.qs[i] = mode == 1 ? 0 : (int8_t) ((int) (g.next() % 255) - 127);
        for (int j = 0; j < 16; ++j) {
            int s = 0;
            for (int i = 0; i < 16; ++i) s += b.qs[16 * j + i];
            b.bsums[j] = (int16_t) s;
        }
    }
}

uint64_t fnv(uint64_t h, const void* p, size_t n) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001B3ull;
    return h;
}

int dump(const char* path) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "iq256_dump: cannot write %s\n", path);
        return 2;
    }
    Rng g;
    const int types[] = {16, 17, 18, 21, 22, 23};
    const int shapes[][2] = {{1, 1}, {7, 2}, {33, 3}, {64, 1}, {129, 4}, {640, 10}};   // rows, QK_K blocks per row
    long long cases = 0, values = 0;
    uint64_t h = 0xCBF29CE484222325ull;
    for (int ty : types) {
        if (!c::iq256_supported(ty)) continue;
        const size_t bsize = ggml_type_size((ggml_type) ty);
        for (const auto& sh : shapes) {
            const int rows = sh[0], nb = sh[1], n = nb * 256;
            const size_t row_bytes = (size_t) nb * bsize;
            std::vector<uint8_t> w, gu;
            fill_rows(g, w, rows, nb, bsize, row_bytes);
            fill_rows(g, gu, 2 * rows, nb, bsize, row_bytes);
            std::vector<BlockQ8K> acts[9];
            for (int rep = 0; rep < 3; ++rep) {
                for (auto& a : acts) fill_act(g, a, nb);
                for (int nt = 1; nt <= 9; ++nt) {
                    const int r0 = rep == 0 ? 0 : (int) (g.next() % (uint64_t) rows);
                    const int r1 = rep == 0 ? rows : r0 + (int) (g.next() % (uint64_t) (rows - r0 + 1));
                    std::vector<float> o((size_t) nt * rows, -7.f), ff((size_t) nt * rows, -7.f);
                    const void* ap[9];
                    float* op[9];
                    float* fp[9];
                    for (int t = 0; t < nt; ++t) {
                        ap[t] = acts[t].data();
                        op[t] = o.data() + (size_t) t * rows;
                        fp[t] = ff.data() + (size_t) t * rows;
                    }
                    c::iq256_rows(ty, w.data(), row_bytes, n, ap, nt, op, r0, r1);
                    c::iq256_gu_rows(ty, gu.data(), row_bytes, (size_t) rows * row_bytes, n, ap, nt, fp, r0, r1);
                    std::fwrite(o.data(), sizeof(float), o.size(), f);
                    std::fwrite(ff.data(), sizeof(float), ff.size(), f);
                    h = fnv(fnv(h, o.data(), o.size() * 4), ff.data(), ff.size() * 4);
                    values += 2LL * nt * (r1 - r0);
                    ++cases;
                }
            }
        }
    }
    std::fclose(f);
    std::printf("iq256_dump: %lld cases, %lld row outputs, fnv %016llx -> %s\n", cases, values, (unsigned long long) h,
                path);
    return 0;
}

void bench() {
    Rng g;
    std::printf("\n--bench: one thread, us per gate/up matrix pair (640 rows x 10 QK_K blocks)\n");
    std::printf("%-6s %-6s %3s | %9s\n", "type", "weights", "nt", "us");
    const int rows = 640, nb = 10, n = nb * 256;
    for (int ty : {21, 22}) {
        const size_t bsize = ggml_type_size((ggml_type) ty), row_bytes = (size_t) nb * bsize;
        const size_t mat = 2 * (size_t) rows * row_bytes;
        const int nmat_cold = (int) ((96u << 20) / mat) + 1;   // > L3: every matrix comes from RAM
        std::vector<uint8_t> w;
        fill_rows(g, w, 2 * rows * nmat_cold, nb, bsize, row_bytes);
        std::vector<BlockQ8K> acts[4];
        for (auto& a : acts) fill_act(g, a, nb);
        std::vector<float> o((size_t) 4 * rows);
        const void* ap[4] = {acts[0].data(), acts[1].data(), acts[2].data(), acts[3].data()};
        float* op[4] = {o.data(), o.data() + rows, o.data() + 2 * rows, o.data() + 3 * rows};
        for (int cold = 0; cold <= 1; ++cold)
            for (int nt = 1; nt <= 4; ++nt) {
                int m = 0;
                auto run = [&] {
                    const uint8_t* base = w.data() + (cold ? (size_t) (m++ % nmat_cold) * mat : 0);
                    c::iq256_gu_rows(ty, base, row_bytes, (size_t) rows * row_bytes, n, ap, nt, op, 0, rows);
                };
                const int reps = cold ? nmat_cold * 3 : 300;
                double best = 1e30;
                for (int round = 0; round < 3; ++round) {
                    run();
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < reps; ++i) run();
                    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                    best = us < best ? us : best;
                }
                std::printf("%-6s %-6s %3d | %9.2f\n", ty == 21 ? "IQ3_S" : "IQ2_S", cold ? "RAM" : "cache", nt, best);
            }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: iq256_dump <out.bin> [--bench]\n");
        return 2;
    }
    if (!c::cpu_avx2_ok()) {
        std::printf("iq256_dump: no AVX2 on this CPU - skipped\n");
        return 77;
    }
    const int rc = dump(argv[1]);
    if (rc == 0 && argc > 2 && std::strcmp(argv[2], "--bench") == 0) bench();
    return rc;
}
