// src/kernels/dense_bench.cpp - the dense projections of a decode verify window at the model's shapes and formats:
// the default kernels against STRATA_DENSE_WR (lab/speed2, docs/SPEED2.md), bit for bit and in microseconds, per
// card, per T.
//
//     build/dense_bench [reps = 300] [device = -1 (all)] [shapes = model] [choices = def,wr,wr:r1,wr:r4,wr:b2,wr:s1,wr:a1:k0,...] [T = 1,2,3,4]
//
// `shapes`: "model" = the stage-1 layers' dense matrices of Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (qkv 2560 -> 10240 in
// Q6_K / IQ4_XS / Q4_K / Q5_K, gate 2560 -> 6144 in IQ4_XS / Q4_K / Q6_K / Q5_K, out 6144 -> 2560 in Q6_K / Q4_K,
// q 2560 -> 12288 in Q6_K / Q4_K / IQ4_XS, kv 2560 -> 512 Q6_K, shexp 2560 -> 640 in IQ4_XS / Q4_K), or a comma list
// of substrings of those names ("qkv,out").  `choices`:
//   def                 the engine's default path: the single-column kernels at T = 1, STRATA_B6_MMVQ_ROWS=1 (two rows
//                       a block) above; the reference for the bitwise check
//   b6[:<mode>]         STRATA_B6_MMVQ_ROWS at that mode (1 = 2 rows, 2 / 4 / 8)
//   wr[:r<n>][:s<0|1>][:b<n>][:m<n>][:t<0|1>][:a<0|1>][:k<n>]
//                       STRATA_DENSE_WR with r = rows per warp (1, 2, 4), s = activations staged in shared memory,
//                       b = min blocks per SM (2 | 4; 0 = auto), m = the smallest launch (blocks) it takes, t = 1 also at
//                       T = 1, a = 1 every format (the default mask leaves Q6_K out), k = the longest row it takes (0 =
//                       any; the default 2560 leaves the 6144-input output projections out); a missing field takes the
//                       engine's value (the environment's STRATA_DENSE_WR_* or its default).  A choice that would not take a call (native_mmvq_dense_wr_takes) is reported as
//                       "default" for that cell: the time is the default kernel's.
// Per (device, shape, T, choice): the output against `def`'s, bit for bit (memcmp), and against T single-column calls;
// the median of 5 rounds of `reps` back-to-back native_mmvq calls between two CUDA events.  The weights rotate over
// >= 96 MB of copies (random bytes, sane fp16 block scales), so every call reads them cold from DRAM as the decode
// reads each layer's.
//
// One line per cell:  dev T shape choice us GB/s pct identical|DIFFERENT 1col=ok|DIFF [default]   (GB/s = weight bytes
// / us; pct = of the card's peak 2 x memory clock x bus width / 8).  Lines starting with '#' are context.  Exit code 1
// if any choice differs from `def` or from the single-column calls, or a CUDA error occurred.  GPU time only.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace K = strata::kernels;

namespace {

struct Shape {
    const char* name;
    int type;          // GGML type id
    int n_in, n_out;
    int block_bytes;   // per 256 elements
    int scale_at[2];   // byte offsets of the fp16 scales in a block; -1 = none
};
const Shape kModel[] = {
    {"qkv Q6_K", 14, 2560, 10240, 210, {208, -1}},
    {"qkv IQ4_XS", 23, 2560, 10240, 136, {0, -1}},
    {"qkv Q4_K", 12, 2560, 10240, 144, {0, 2}},
    {"qkv Q5_K", 13, 2560, 10240, 176, {0, 2}},
    {"gate IQ4_XS", 23, 2560, 6144, 136, {0, -1}},
    {"gate Q4_K", 12, 2560, 6144, 144, {0, 2}},
    {"gate Q6_K", 14, 2560, 6144, 210, {208, -1}},
    {"gate Q5_K", 13, 2560, 6144, 176, {0, 2}},
    {"out Q6_K", 14, 6144, 2560, 210, {208, -1}},
    {"out Q4_K", 12, 6144, 2560, 144, {0, 2}},
    {"q Q6_K", 14, 2560, 12288, 210, {208, -1}},
    {"q Q4_K", 12, 2560, 12288, 144, {0, 2}},
    {"q IQ4_XS", 23, 2560, 12288, 136, {0, -1}},
    {"kv Q6_K", 14, 2560, 512, 210, {208, -1}},
    {"shexp IQ4_XS", 23, 2560, 640, 136, {0, -1}},
    {"shexp Q4_K", 12, 2560, 640, 144, {0, 2}},
};

struct Choice {
    std::string name;
    int kind = 0;           // 0 def, 1 b6, 2 wr
    int b6_mode = 1;
    K::NativeDenseWrParams p;
};
K::NativeDenseWrParams g_engine_wr;   // the engine's values at startup (environment or defaults)

bool parse_choice(const std::string& s, Choice& c) {
    c = Choice{};
    c.name = s;
    c.p = g_engine_wr;
    c.p.on = 0;
    if (s == "def") return true;
    if (s.rfind("b6", 0) == 0) {
        c.kind = 1;
        if (s.size() > 2) { if (s[2] != ':') return false; c.b6_mode = std::atoi(s.c_str() + 3); }
        return c.b6_mode == 1 || c.b6_mode == 2 || c.b6_mode == 4 || c.b6_mode == 8;
    }
    if (s.rfind("wr", 0) != 0) return false;
    c.kind = 2;
    c.p.on = 1;
    size_t i = 2;
    while (i < s.size()) {
        if (s[i] != ':' || i + 1 >= s.size()) return false;
        const char k = s[i + 1];
        char* end = nullptr;
        const long v = std::strtol(s.c_str() + i + 2, &end, 10);
        if (end == s.c_str() + i + 2) return false;
        switch (k) {
            case 'r': c.p.rows = (int) v; break;
            case 's': c.p.stage = (int) v; break;
            case 'b': c.p.min_blocks_per_sm = (int) v; break;
            case 'm': c.p.min_launch_blocks = (int) v; break;
            case 't': c.p.t1 = (int) v; break;
            case 'a': c.p.types_mask = v ? ~0u : K::NativeDenseWrParams{}.types_mask; break;   // a1 = every format (Q6_K too)
            case 'k': c.p.max_n_in = (int) v; break;                                          // k0 = any row length
            default: return false;
        }
        i = (size_t) (end - s.c_str());
    }
    return true;
}

// the engine's path for `c` at T: sets the lab knobs, returns whether DENSE_WR takes the call
bool apply_choice(const Choice& c, const Shape& sh, int T) {
    K::native_mmvq_set_dense_wr(c.p);
    K::native_mmvq_set_b6_rows(c.kind == 1 ? c.b6_mode : 1);   // the engine's config: STRATA_B6_MMVQ_ROWS=1
    return c.kind == 2 && K::native_mmvq_dense_wr_takes(sh.type, sh.n_in, sh.n_out, T);
}

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("# CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}
uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        const size_t b = s.find(',', a);
        out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::max(20, std::atoi(argv[1])) : 300;
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    const std::string shapes_arg = argc > 3 ? argv[3] : "model";
    const std::string choices_arg = argc > 4 ? argv[4] : "def,wr,wr:r1,wr:r4,wr:b2,wr:s1,wr:a1:k0,wr:a1:k0:r4,wr:a1:k0:b2";
    const std::string t_arg = argc > 5 ? argv[5] : "1,2,3,4";
    g_engine_wr = K::native_mmvq_dense_wr();
    std::vector<Choice> choices;
    for (const std::string& s : split(choices_arg)) {
        Choice c;
        if (!parse_choice(s, c)) { std::printf("# bad choice '%s'\n", s.c_str()); return 1; }
        choices.push_back(c);
    }
    if (choices.empty() || choices[0].name != "def") { std::printf("# the first choice must be def\n"); return 1; }
    std::vector<int> ts;
    for (const std::string& s : split(t_arg)) { const int t = std::atoi(s.c_str()); if (t >= 1 && t <= 8) ts.push_back(t); }
    std::vector<const Shape*> shapes;
    for (const Shape& sh : kModel) {
        bool take = shapes_arg == "model";
        if (!take) for (const std::string& s : split(shapes_arg)) if (!s.empty() && std::string(sh.name).find(s) != std::string::npos) take = true;
        if (take) shapes.push_back(&sh);
    }
    int n_dev = 0;
    if (!ck(cudaGetDeviceCount(&n_dev), "device count")) return 1;
    long long bad = 0;
    for (int dev = 0; dev < n_dev; ++dev) {
        if (only >= 0 && dev != only) continue;
        cudaSetDevice(dev);
        cudaDeviceProp pr{};
        cudaGetDeviceProperties(&pr, dev);
        const double peak = 2.0 * pr.memoryClockRate * 1e3 * (pr.memoryBusWidth / 8) / 1e9;   // GB/s
        std::printf("# device %d: %s (sm_%d%d, %d SMs, L2 %d KB, peak %.0f GB/s), %d reps x 5 rounds, median\n", dev, pr.name, pr.major,
                    pr.minor, pr.multiProcessorCount, pr.l2CacheSize / 1024, peak, reps);
        std::printf("# dev T shape           choice        us     GB/s   pct  vs-def  vs-1col\n");
        cudaStream_t s;
        if (!ck(cudaStreamCreate(&s), "stream")) return 1;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (const Shape* shp : shapes) {
            const Shape& sh = *shp;
            if (!K::native_mmvq_supported(sh.type)) { std::printf("# %s: unsupported type %d\n", sh.name, sh.type); continue; }
            std::mt19937 rng(77u + (unsigned) sh.type * 131u + (unsigned) sh.n_out);
            const size_t wbytes = K::native_mmvq_weight_bytes(sh.type, sh.n_in, sh.n_out);
            std::vector<uint8_t> w(wbytes);
            for (auto& x : w) x = (uint8_t) (rng() & 0xff);
            const size_t nb = (size_t) sh.n_out * (size_t) (sh.n_in / 256);
            for (size_t k = 0; k < nb; ++k)
                for (int at : sh.scale_at)
                    if (at >= 0) { const uint16_t h = sane_half(rng); std::memcpy(&w[k * (size_t) sh.block_bytes + (size_t) at], &h, 2); }
            const size_t copies = std::max<size_t>(1, std::min<size_t>(12, (96u << 20) / wbytes + 1));
            std::vector<void*> dw;
            bool ok = true;
            for (size_t k = 0; k < copies && ok; ++k) {
                void* d = nullptr;
                ok = ck(cudaMalloc(&d, wbytes), "malloc w") && ck(cudaMemcpy(d, w.data(), wbytes, cudaMemcpyHostToDevice), "copy w");
                dw.push_back(d);
            }
            std::vector<float> x((size_t) 8 * sh.n_in);
            std::normal_distribution<float> nx(0.f, 1.f);
            for (auto& v : x) v = nx(rng);
            float* xf = nullptr; void* xq = nullptr; float *y = nullptr, *yd = nullptr, *y1 = nullptr;
            const size_t ybytes = (size_t) 8 * sh.n_out * 4;
            ok = ok && ck(cudaMalloc(&xf, x.size() * 4), "malloc x") && ck(cudaMemcpy(xf, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x") &&
                 ck(cudaMalloc(&xq, K::native_q8_1_bytes(sh.n_in, 8)), "malloc xq") && ck(cudaMalloc(&y, ybytes), "y") &&
                 ck(cudaMalloc(&yd, ybytes), "yd") && ck(cudaMalloc(&y1, ybytes), "y1");
            if (!ok) return 1;
            K::quantize_q8_1_rows(xf, 8, sh.n_in, xq, s);
            for (int T : ts) {
                std::vector<uint32_t> hd((size_t) T * sh.n_out), h1(hd.size()), h((size_t) T * sh.n_out);
                // the single-column reference (def's path at ncols = 1)
                apply_choice(choices[0], sh, 1);
                for (int j = 0; j < T; ++j)
                    K::native_mmvq(sh.type, dw[0], (const uint8_t*) xq + K::native_q8_1_bytes(sh.n_in, 1) * (size_t) j, y1 + (size_t) j * sh.n_out, sh.n_in, sh.n_out, 1, s);
                if (!ck(cudaStreamSynchronize(s), "1col")) return 1;
                cudaMemcpy(h1.data(), y1, h1.size() * 4, cudaMemcpyDeviceToHost);
                for (size_t ci = 0; ci < choices.size(); ++ci) {
                    const Choice& c = choices[ci];
                    const bool takes = apply_choice(c, sh, T);
                    const bool is_default = c.kind == 2 && !takes;
                    for (int r = 0; r < 20; ++r) K::native_mmvq(sh.type, dw[(size_t) r % dw.size()], xq, y, sh.n_in, sh.n_out, T, s);   // warm-up
                    double best = 0.0;
                    std::vector<double> rounds;
                    for (int round = 0; round < 5; ++round) {
                        cudaEventRecord(e0, s);
                        for (int r = 0; r < reps; ++r) K::native_mmvq(sh.type, dw[(size_t) r % dw.size()], xq, y, sh.n_in, sh.n_out, T, s);
                        cudaEventRecord(e1, s);
                        if (!ck(cudaEventSynchronize(e1), "round")) return 1;
                        float ms = 0.f;
                        cudaEventElapsedTime(&ms, e0, e1);
                        rounds.push_back(ms * 1e3 / reps);
                    }
                    std::sort(rounds.begin(), rounds.end());
                    best = rounds[2];
                    K::native_mmvq(sh.type, dw[0], xq, ci == 0 ? yd : y, sh.n_in, sh.n_out, T, s);
                    if (!ck(cudaStreamSynchronize(s), "check")) return 1;
                    if (ci == 0) {
                        cudaMemcpy(hd.data(), yd, hd.size() * 4, cudaMemcpyDeviceToHost);
                        h = hd;
                    } else {
                        cudaMemcpy(h.data(), y, h.size() * 4, cudaMemcpyDeviceToHost);
                    }
                    const bool same = std::memcmp(h.data(), hd.data(), h.size() * 4) == 0;
                    const bool same1 = std::memcmp(h.data(), h1.data(), h.size() * 4) == 0;
                    long long nonfinite = 0;
                    for (uint32_t u : h) { float f; std::memcpy(&f, &u, 4); if (!std::isfinite(f)) ++nonfinite; }
                    if (!same || !same1 || nonfinite) ++bad;
                    const double gbs = wbytes / best / 1e3;
                    std::printf("%d %d %-15s %-10s %8.1f %8.1f %5.1f%%  %s  %s%s%s\n", dev, T, sh.name, c.name.c_str(), best, gbs, 100.0 * gbs / peak,
                                same ? "identical" : "DIFFERENT", same1 ? "1col=ok" : "1col=DIFF", is_default ? " default" : "",
                                nonfinite ? " NONFINITE" : "");
                }
            }
            for (void* d : dw) cudaFree(d);
            cudaFree(xf); cudaFree(xq); cudaFree(y); cudaFree(yd); cudaFree(y1);
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaStreamDestroy(s);
    }
    K::native_mmvq_set_dense_wr(g_engine_wr);
    std::printf("# dense_bench: %s (%lld cells differ)\n", bad ? "FAILED" : "ok", bad);
    return bad ? 1 : 0;
}
