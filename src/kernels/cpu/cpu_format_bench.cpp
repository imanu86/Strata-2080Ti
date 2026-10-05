// src/kernels/cpu/cpu_format_bench.cpp - lab: CPU expert throughput per weight format, and the RAM read ceiling.
//
//     cpu_format_bench <shard1.gguf> [experts=48] [iters=3] [threads=1,8]
//
// Real expert blobs of the model's own formats (layers 0/1/35: IQ2_XS/IQ2_XXS/IQ3_XXS gate+up, IQ4_NL/Q2_0 down)
// and the same layer-35 gate/up weights re-quantized by ggml to IQ3_S, IQ4_XS, Q4_K and Q8_0 (down kept; speed only:
// the re-quantized rows carry no imatrix, so they say nothing about quality). Each case runs
// the pool's own row kernels (native_gu_rows / native_down_rows) over `experts` distinct blobs (more than L3),
// single-threaded and on 8 threads, for 1 and 4 tokens. Prints us per expert, GB/s of weights read and G weights/s.
// A matrix is compute-bound when its GB/s stays well below the measured RAM read ceiling and grows with threads.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace c = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;

namespace {

struct Case {
    std::string name;
    c::NativeFmt f;
    std::vector<uint8_t> blobs;   // experts * f.bytes
};

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

double ram_read_gbs(int threads, size_t bytes) {
    std::vector<uint64_t> buf(bytes / 8);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = i * 2654435761ull;
    std::atomic<uint64_t> sink{0};
    double best = 0;
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t)
            th.emplace_back([&, t] {
                const size_t a = buf.size() * t / threads, b = buf.size() * (t + 1) / threads;
                uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                for (size_t i = a; i + 3 < b; i += 4) { s0 += buf[i]; s1 += buf[i + 1]; s2 += buf[i + 2]; s3 += buf[i + 3]; }
                sink += s0 + s1 + s2 + s3;
            });
        for (auto& x : th) x.join();
        best = std::max(best, (double) bytes / (ms_since(t0) * 1e6));
    }
    return best;
}

// Runs one matrix kind ("gu" or "down") of every expert, split over `threads`, `iters` times; returns ms per pass.
double run(const Case& cs, int experts, bool gu, int nt, int threads, int iters) {
    const c::NativeFmt& f = cs.f;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) nt * f.n_embd), h((size_t) nt * f.n_ff);
    for (auto& v : x) v = nd(rng);
    for (auto& v : h) v = 0.05f * nd(rng);
    std::vector<std::vector<uint8_t>> act(nt, std::vector<uint8_t>(c::kNativeActBytes + 64));
    std::vector<std::vector<uint8_t>> hq(nt, std::vector<uint8_t>(c::kNativeHBytes + 64));
    std::vector<c::ActQ> a2(nt);   // a Q2_0 down takes the pool's own kernel (q2_rows_any), not native_down_rows
    const void* a[c::MAXT];
    const void* q[c::MAXT];
    const c::ActQ* q2[c::MAXT];
    const bool q2_down = !gu && f.d_type == 42;
    for (int t = 0; t < nt; ++t) {
        c::native_quant_act(f, x.data() + (size_t) t * f.n_embd, act[t].data());
        if (q2_down) c::act_quant_any(h.data() + (size_t) t * f.n_ff, (int) f.n_ff, a2[t]);
        else c::native_quant_h(f, h.data() + (size_t) t * f.n_ff, hq[t].data());
        a[t] = act[t].data();
        q[t] = hq[t].data();
        q2[t] = &a2[t];
    }
    double best = 1e30;
    for (int it = 0; it < iters + 1; ++it) {
        const auto t0 = Clock::now();
        std::vector<std::thread> th;
        for (int w = 0; w < threads; ++w)
            th.emplace_back([&, w] {
                std::vector<float> ffb((size_t) c::MAXT * f.n_ff), ob((size_t) c::MAXT * f.n_embd);
                float* ff[c::MAXT];
                float* out[c::MAXT];
                for (int t = 0; t < c::MAXT; ++t) { ff[t] = ffb.data() + (size_t) t * f.n_ff; out[t] = ob.data() + (size_t) t * f.n_embd; }
                for (int e = w; e < experts; e += threads) {
                    const uint8_t* blob = cs.blobs.data() + (size_t) e * f.bytes;
                    if (gu) c::native_gu_rows(f, blob, a, nt, ff, 0, (int) f.n_ff);
                    else if (q2_down) c::q2_rows_any(blob + f.down_off, f.d_row, (int) (f.n_ff / 64), q2, nt, out, 0, (int) f.n_embd);
                    else c::native_down_rows(f, blob, q, nt, out, 0, (int) f.n_embd);
                }
            });
        for (auto& t : th) t.join();
        const double ms = ms_since(t0);
        if (it > 0) best = std::min(best, ms);   // the first pass warms code and TLBs
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) { std::fprintf(stderr, "usage: cpu_format_bench shard1.gguf [experts=48] [iters=3] [threads=1,8]\n"); return 2; }
    const int experts = argc > 2 ? std::atoi(argv[2]) : 48;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 3;
    std::vector<int> thread_list;
    for (const char* p = argc > 4 ? argv[4] : "1,8"; *p;) {
        thread_list.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p) ++p;
    }
    std::fprintf(stderr, "apro %s\n", argv[1]);
    strata::GgufFile gguf(argv[1]);
    std::fprintf(stderr, "gguf aperto, %zu tensori\n", gguf.tensors().size());
    auto tensor = [&](int layer, const char* role) -> const strata::TensorInfo* {
        const std::string want = "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
        for (const auto& t : gguf.tensors()) if (t.name == want) return &t;
        return nullptr;
    };
    std::vector<Case> cases;
    cases.reserve(16);   // `base` below references cases[2] while more cases are appended
    std::string err;
    // the model's own formats, real rows
    for (int layer : {0, 1, 35}) {
        const auto *g = tensor(layer, "gate"), *u = tensor(layer, "up"), *d = tensor(layer, "down");
        if (!g || !u || !d) { std::fprintf(stderr, "layer %d missing\n", layer); return 2; }
        Case cs;
        if (!c::native_fmt(g->type, d->type, c::H, c::FF, cs.f, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
        cs.name = "L" + std::to_string(layer) + " " + ggml_type_name((ggml_type) g->type) + "/" +
                  (d->type == 42 ? std::string("q2_0") : std::string(ggml_type_name((ggml_type) d->type)));
        cs.blobs.resize((size_t) experts * cs.f.bytes);
        const size_t gub = cs.f.up_off, db = cs.f.bytes - cs.f.down_off;
        for (int e = 0; e < experts; ++e) {
            uint8_t* p = cs.blobs.data() + (size_t) e * cs.f.bytes;
            std::memcpy(p, gguf.tensor_data(*g) + (size_t) e * gub, gub);
            std::memcpy(p + cs.f.up_off, gguf.tensor_data(*u) + (size_t) e * gub, gub);
            std::memcpy(p + cs.f.down_off, gguf.tensor_data(*d) + (size_t) e * db, db);
        }
        std::fprintf(stderr, "  caso %s pronto (%zu byte/esperto)\n", cs.name.c_str(), cs.f.bytes);
        cases.push_back(std::move(cs));
    }
    // layer 35's gate/up re-quantized to other formats (down kept as IQ4_NL)
    {
        const Case& base = cases[2];
        const auto* tr = ggml_get_type_traits((ggml_type) base.f.gu_type);
        std::vector<float> imat(c::H, 1.0f);
        for (int ty : {21 /*IQ3_S*/, 23 /*IQ4_XS*/, 12 /*Q4_K*/, 8 /*Q8_0*/}) {
            Case cs;
            if (!c::native_fmt(ty, base.f.d_type, c::H, c::FF, cs.f, err)) { std::fprintf(stderr, "type %d: %s\n", ty, err.c_str()); continue; }
            cs.name = std::string("L35 ") + ggml_type_name((ggml_type) ty) + "/iq4_nl (riquantizzato)";
            ggml_quantize_init((ggml_type) ty);
            cs.blobs.resize((size_t) experts * cs.f.bytes);
            std::vector<float> rows((size_t) c::FF * c::H);
            for (int e = 0; e < experts; ++e) {
                const uint8_t* src = base.blobs.data() + (size_t) e * base.f.bytes;
                uint8_t* dst = cs.blobs.data() + (size_t) e * cs.f.bytes;
                for (int m = 0; m < 2; ++m) {   // gate, then up
                    tr->to_float(src + (m ? base.f.up_off : 0), rows.data(), (int64_t) c::FF * c::H);
                    ggml_quantize_chunk((ggml_type) ty, rows.data(), dst + (m ? cs.f.up_off : 0), 0, c::FF, c::H, imat.data());
                }
                std::memcpy(dst + cs.f.down_off, src + base.f.down_off, base.f.bytes - base.f.down_off);
            }
            cases.push_back(std::move(cs));
        }
    }
    const unsigned hw = std::thread::hardware_concurrency();
    std::printf("RAM lettura: 1 thread %.1f GB/s, 8 thread %.1f GB/s, %u thread %.1f GB/s\n", ram_read_gbs(1, 1ull << 30),
                ram_read_gbs(8, 1ull << 30), hw, ram_read_gbs((int) hw, 1ull << 30));
    std::printf("%-34s %-5s %3s %3s %9s %8s %8s\n", "formato", "matr", "thr", "tok", "us/esp", "GB/s", "Gpesi/s");
    for (const Case& cs : cases)
        for (bool gu : {true, false}) {
            if (!gu && cs.name.find("riquantizzato") != std::string::npos) continue;   // same down as L35
            const double bytes = gu ? (double) cs.f.down_off : (double) (cs.f.bytes - cs.f.down_off);
            const double weights = gu ? 2.0 * c::FF * c::H : (double) c::FF * c::H;
            for (int threads : thread_list)
                for (int nt : {1, 4}) {
                    const double ms = run(cs, experts, gu, nt, threads, iters);
                    const double us = ms * 1e3 / experts;
                    std::printf("%-34s %-5s %3d %3d %9.1f %8.1f %8.2f\n", cs.name.c_str(), gu ? "gu" : "down", threads, nt, us,
                                bytes * experts / (ms * 1e6), weights * experts / (ms * 1e6));
                }
        }
    return 0;
}
