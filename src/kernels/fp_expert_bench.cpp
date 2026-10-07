// src/kernels/fp_expert_bench.cpp - what the VRAM experts of a decode verify window cost, per kernel variant, per card,
// per expert format pair.
//
//     build/fp_expert_bench [reps = 300] [device, -1 = all] [groups = 20] [pairs = all]
//
// The model's routed expert formats at its geometry (n_embd 2560, n_ff 640): gate/up IQ2_XXS (16), IQ2_XS (17), IQ3_XXS
// (18), IQ3_S (21), IQ2_S (22); down IQ4_NL (20), Q2_0 (42).  `pairs` is a comma list of gu/down type pairs (e.g.
// "21/20,16/42"); the default runs every one of the 5 x 2 pairs.  Each pair goes through `native_expert_grouped` (gate/up
// + SwiGLU + down) with the opt-in kernel choices of STRATA_FP_EXPERT_V / native_expert_set_fp_variant and friends:
//   0        the default kernels
//   4        persistent gate/up and down (STRATA_FP_EXPERT_PERSIST_K blocks per SM, default 2)
//   1 2 3    the IQ3_S / IQ4_NL pair only: IQ3_S codebook in shared memory, two rows per sub-warp, 1 + 2
//   0t<k>    T = 1 only: the default kernels' rows in ceil(tiles / k) blocks of k row tiles each (STRATA_FP_DEF_TILES_T1=k)
//   auto     the selection the engine makes with STRATA_FP_EXPERT_V=4: variant 4 for T <= STRATA_FP_EXPERT_V4_TMAX (the
//            environment's, else 1 here) on the pairs of STRATA_FP_EXPERT_V4_PAIRS (unset = every pair), the default
//            kernels otherwise, with STRATA_FP_DEF_TILES_T1 as set in the environment (unset = 1, the default launch).
//
// For T = 1..4 tokens a verify window is built: every token picks 10 experts out of `groups` distinct ones (at most
// 10 T; every expert picked by at least one token, a group holds the tokens that picked it, 1..T entries).  Per variant:
// the output buffer against variant 0's, bit for bit (memcmp), and the time of `reps` back-to-back calls with CUDA events
// (median of 5 rounds).  The experts rotate over a pool of >= 96 MB of distinct synthetic blobs (random bytes, sane fp16
// block scales, as b6_mmvq_bench), so every call reads its weights cold from DRAM as the decode does each layer.
//
// One line per (device, T, pair, variant):  dev T gu/d v us GB/s pct identical|DIFFERENT   (GB/s = groups x blob bytes
// / call time; pct = GB/s over the peak DRAM bandwidth 2 x memory clock x bus width / 8 of the device attributes, the
// DDR factor 2 of GDDR6).
// Lines starting with '#' are context.  Exit code 1 if any variant differs or a CUDA error occurred.  GPU time only.
// STRATA_NO_SUB16_GU and STRATA_EXPERT_V2 route around the variants (every row would then be the default kernels).
#include "strata/kernels/iq_kernels.hpp"

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

constexpr int kEmbd = 2560, kFf = 640, kTop = 10, kMaxT = 4;
constexpr int kMaxEntries = kMaxT * kTop;           // every token picks exactly kTop experts

struct Pair { int gu, d; };
const Pair kModelPairs[] = { {16, 20}, {17, 20}, {18, 20}, {21, 20}, {22, 20}, {16, 42}, {17, 42}, {18, 42}, {21, 42}, {22, 42} };

const char* type_name(int t) {
    switch (t) {
        case 16: return "IQ2_XXS"; case 17: return "IQ2_XS"; case 18: return "IQ3_XXS"; case 21: return "IQ3_S";
        case 22: return "IQ2_S"; case 20: return "IQ4_NL"; case 42: return "Q2_0"; default: return "?";
    }
}
// values per block (the i-quants' super-block, IQ4_NL's 32, Q2_0's 64); every block starts with its fp16 scale d
int qk_of(int t) { return t == 20 ? 32 : t == 42 ? 64 : 256; }

// A kernel choice the bench times: variant 0-4, or 0 with STRATA_FP_DEF_TILES_T1 = tiles, or the engine's selection.
struct Choice {
    std::string name;
    int variant = 0;     // native_expert_set_fp_variant
    int tiles_t1 = 1;    // native_expert_set_fp_def_tiles_t1
    int tmax = 0;        // native_expert_set_fp_v4_tmax (auto only; 0 = every call)
    bool pairs_env = false;   // auto: the environment's STRATA_FP_EXPERT_V4_PAIRS table (else every pair)
};

int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}

// the choices for a pair at T: 0 and 4 everywhere, 1-3 for the IQ3_S / IQ4_NL pair they were written for, the tiles
// launches at T = 1, and the engine's selection last
std::vector<Choice> choices_of(const Pair& p, int T) {
    std::vector<Choice> c;
    c.push_back({"0", 0, 1, 0, false});
    if (p.gu == 21 && p.d == 20)
        for (int v = 1; v <= 3; ++v) c.push_back({std::to_string(v), v, 1, 0, false});
    c.push_back({"4", 4, 1, 0, false});
    if (T == 1)
        for (int k : {2, 4, 8}) c.push_back({"0t" + std::to_string(k), 0, k, 0, false});
    Choice a;
    a.name = "auto";
    a.variant = 4;
    a.tmax = std::max(0, env_int("STRATA_FP_EXPERT_V4_TMAX", 1));
    a.tiles_t1 = std::max(1, env_int("STRATA_FP_DEF_TILES_T1", 1));
    a.pairs_env = true;
    c.push_back(a);
    return c;
}

bool g_cuda_error = false;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    g_cuda_error = true;
    return false;
}

uint16_t sane_half(std::mt19937& rng) {   // a finite fp16 of magnitude ~2^-10 .. 2^-6
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// One expert blob [gate rows | up rows | down rows]: random bytes (any qs / qh / signs / scales nibble / grid index /
// IQ4_NL nibble / Q2_0 pair is a valid code) with a sane fp16 `d` in every block (offset 0 of every format's block).
void fill_blob(uint8_t* b, const K::NativeExpertLayout& L, std::mt19937& rng) {
    for (size_t i = 0; i + 4 <= L.bytes; i += 4) {
        const uint32_t r = rng();
        std::memcpy(b + i, &r, 4);
    }
    const size_t gu_block = K::iq_row_bytes(L.gu_type, qk_of(L.gu_type)), d_block = K::iq_row_bytes(L.d_type, qk_of(L.d_type));
    const size_t gu_blocks = (size_t) 2 * (size_t) L.n_ff * (size_t) (L.n_embd / qk_of(L.gu_type));   // gate, then up: contiguous
    for (size_t k = 0; k < gu_blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(b + k * gu_block, &h, 2);
    }
    const size_t d_blocks = (size_t) L.n_embd * (size_t) (L.n_ff / qk_of(L.d_type));
    for (size_t k = 0; k < d_blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(b + L.down_off + k * d_block, &h, 2);
    }
}

struct Window {
    int groups = 0, entries = 0;
    std::vector<int32_t> start, tok, dst;   // group g: entries [start[g], start[g + 1]); tok = token, dst = out row
};

// T tokens, kTop experts each, `groups_req` distinct experts in all (capped at kTop * T, floored at kTop).  Expert j is
// forced onto token j % T so that every group has at least one entry; the rest of each token's picks are random.
Window build_window(int T, int groups_req, std::mt19937& rng) {
    Window w;
    const int G = std::min(std::max(groups_req, kTop), kTop * T);
    std::vector<std::vector<int>> picks((size_t) G);
    for (int k = 0; k < T; ++k) {
        std::vector<char> in((size_t) G, 0);
        int cnt = 0;
        for (int j = k; j < G; j += T) { in[(size_t) j] = 1; ++cnt; }   // <= ceil(G / T) <= kTop
        std::vector<int> rest;
        for (int j = 0; j < G; ++j)
            if (!in[(size_t) j]) rest.push_back(j);
        std::shuffle(rest.begin(), rest.end(), rng);
        for (size_t i = 0; cnt < kTop; ++i, ++cnt) in[(size_t) rest[i]] = 1;
        for (int j = 0; j < G; ++j)
            if (in[(size_t) j]) picks[(size_t) j].push_back(k);
    }
    w.groups = G;
    w.start.push_back(0);
    for (int g = 0; g < G; ++g) {
        for (int k : picks[(size_t) g]) {
            w.tok.push_back(k);
            w.dst.push_back((int32_t) w.dst.size());
        }
        w.start.push_back((int32_t) w.dst.size());
    }
    w.entries = (int) w.dst.size();
    return w;
}

struct DevMem {
    std::vector<void*> ptrs;
    ~DevMem() { for (void* p : ptrs) cudaFree(p); }
    void* alloc(size_t n) {
        void* p = nullptr;
        if (!ck(cudaMalloc(&p, n), "malloc")) return nullptr;
        ptrs.push_back(p);
        return p;
    }
};

struct StreamEvents {
    cudaStream_t s = nullptr;
    cudaEvent_t a = nullptr, b = nullptr;
    ~StreamEvents() {
        if (a) cudaEventDestroy(a);
        if (b) cudaEventDestroy(b);
        if (s) cudaStreamDestroy(s);
    }
};

// the kernel choice in force for the next calls (the engine's setters; the pairs table from the environment or none)
void apply_choice(const Choice& c, const char* pairs_env) {
    K::native_expert_set_fp_variant(c.variant);
    K::native_expert_set_fp_def_tiles_t1(c.tiles_t1);
    K::native_expert_set_fp_v4_tmax(c.tmax);
    K::native_expert_set_fp_v4_pairs(c.pairs_env ? pairs_env : "");
}

bool run_device(int dev, int reps, int groups_req, const std::vector<uint8_t>& pool, size_t P, size_t stride,
                const std::vector<float>& xh, const K::NativeExpertLayout& L, const Pair& pair, const char* pairs_env,
                long long& bad) {
    if (!ck(cudaSetDevice(dev), "set device")) return false;
    cudaDeviceProp pr{};
    cudaGetDeviceProperties(&pr, dev);
    std::printf("# device %d: %s (sm_%d%d, L2 %d KB), %d reps x 5 rounds, <= %d groups, pool %d blobs x %.2f MB\n", dev,
                pr.name, pr.major, pr.minor, pr.l2CacheSize / 1024, reps, groups_req, (int) P, (double) L.bytes / 1e6);
    int mclk_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&mclk_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    const double peak = 2.0 * (double) mclk_khz * 1e3 * (double) bus_bits / 8.0;   // bytes / s
    std::printf("# device %d peak DRAM bandwidth %.1f GB/s (memory clock %d kHz, bus %d bit)\n", dev, peak / 1e9, mclk_khz, bus_bits);
    const int gmax = std::min(groups_req, kMaxEntries);
    DevMem mem;
    auto* dpool = (uint8_t*) mem.alloc(pool.size());
    auto* dx = (float*) mem.alloc(xh.size() * sizeof(float));
    void* dxq = mem.alloc((size_t) kMaxT * kEmbd / 32 * 36);
    void* dscr = mem.alloc(K::native_expert_scratch_bytes(kMaxEntries, kFf));
    auto* dout = (float*) mem.alloc((size_t) kMaxEntries * kEmbd * sizeof(float));
    auto* dptr = (unsigned long long*) mem.alloc(P * (size_t) gmax * sizeof(unsigned long long));
    auto* dstart = (int32_t*) mem.alloc((size_t) (gmax + 1) * sizeof(int32_t));
    auto* dn = (int32_t*) mem.alloc(sizeof(int32_t));
    auto* ddst = (int32_t*) mem.alloc((size_t) kMaxEntries * sizeof(int32_t));
    auto* dtok = (int32_t*) mem.alloc((size_t) kMaxEntries * sizeof(int32_t));
    if (!dpool || !dx || !dxq || !dscr || !dout || !dptr || !dstart || !dn || !ddst || !dtok) return false;
    StreamEvents se;
    if (!ck(cudaStreamCreate(&se.s), "stream") || !ck(cudaEventCreate(&se.a), "event") ||
        !ck(cudaEventCreate(&se.b), "event"))
        return false;
    void* stream = (void*) se.s;
    if (!ck(cudaMemcpy(dpool, pool.data(), pool.size(), cudaMemcpyHostToDevice), "copy pool") ||
        !ck(cudaMemcpy(dx, xh.data(), xh.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x"))
        return false;
    K::quantize_q8_1_rows(dx, kMaxT, kEmbd, dxq, stream);
    if (!ck(cudaStreamSynchronize(se.s), "quantize")) return false;

    std::mt19937 rng(20261007u);
    bool ok = true;
    for (int T = 1; T <= kMaxT && ok; ++T) {
        const Window w = build_window(T, groups_req, rng);
        const int G = w.groups, NE = w.entries;
        // call i reads G experts starting at blob (i * G) % P: a fresh set until the pool wraps (>= 96 MB later)
        std::vector<unsigned long long> hp(P * (size_t) G);
        for (size_t i = 0; i < P; ++i)
            for (int g = 0; g < G; ++g)
                hp[i * (size_t) G + (size_t) g] = (unsigned long long) (dpool + ((i * (size_t) G + (size_t) g) % P) * stride);
        ok = ck(cudaMemcpy(dptr, hp.data(), hp.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice), "copy ptrs") &&
             ck(cudaMemcpy(dstart, w.start.data(), w.start.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy start") &&
             ck(cudaMemcpy(dn, &G, sizeof(int32_t), cudaMemcpyHostToDevice), "copy n") &&
             ck(cudaMemcpy(ddst, w.dst.data(), w.dst.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy dst") &&
             ck(cudaMemcpy(dtok, w.tok.data(), w.tok.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy tok");
        if (!ok) break;
        std::printf("# T=%d: %d groups, %d entries, %.1f MB of weights per call\n", T, G, NE, (double) G * (double) L.bytes / 1e6);
        auto call = [&](size_t i) {   // the engine passes the window's tokens (T) for the per-call kernel choices
            K::native_expert_grouped(L, dptr + (i % P) * (size_t) G, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, stream, 0, T);
        };
        std::vector<float> ref, res((size_t) NE * kEmbd);
        const std::vector<Choice> choices = choices_of(pair, T);
        for (size_t vi = 0; vi < choices.size() && ok; ++vi) {
            const Choice& c = choices[vi];
            apply_choice(c, pairs_env);
            ok = ck(cudaMemset(dout, 0, (size_t) NE * kEmbd * sizeof(float)), "memset out");
            call(0);
            ok = ok && ck(cudaStreamSynchronize(se.s), "run") && ck(cudaGetLastError(), "kernel") &&
                 ck(cudaMemcpy(res.data(), dout, res.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
            if (!ok) break;
            bool same = true;
            if (vi == 0) {   // variant 0: the reference
                ref = res;
                long long nonfinite = 0;
                for (float f : ref) nonfinite += std::isfinite(f) ? 0 : 1;
                if (nonfinite) std::printf("# T=%d: variant 0 output has %lld non-finite values\n", T, nonfinite);
            } else {
                same = std::memcmp(ref.data(), res.data(), ref.size() * sizeof(float)) == 0;
            }
            for (size_t i = 0; i < 100; ++i) call(i);   // warm-up: the card's clocks ramp up from idle
            double rounds[5];
            for (double& r : rounds) {
                cudaEventRecord(se.a, se.s);
                for (int i = 0; i < reps; ++i) call((size_t) i);
                cudaEventRecord(se.b, se.s);
                cudaEventSynchronize(se.b);
                float ms = 0.0f;
                ok = ok && ck(cudaEventElapsedTime(&ms, se.a, se.b), "elapsed") && ck(cudaGetLastError(), "kernel");
                r = 1000.0 * (double) ms / reps;
            }
            std::sort(rounds, rounds + 5);
            const double us = rounds[2];
            const double gbps = (double) G * (double) L.bytes / (us * 1e-6) / 1e9;
            std::printf("%d %d %d/%d %s %.1f %.1f %.1f%% %s\n", dev, T, L.gu_type, L.d_type, c.name.c_str(), us, gbps,
                        peak > 0.0 ? 100.0 * gbps * 1e9 / peak : 0.0, same ? "identical" : "DIFFERENT");
            if (!same) ++bad;
        }
    }
    apply_choice(Choice{"0", 0, 1, 0, false}, pairs_env);
    return ok;
}

// "gu/d,gu/d,..." -> pairs; false on a malformed item
bool parse_pairs(const std::string& spec, std::vector<Pair>& out) {
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string item = spec.substr(i, j - i);
        i = j + 1;
        if (item.empty()) continue;
        const size_t s = item.find('/');
        if (s == std::string::npos || s == 0 || s + 1 >= item.size()) return false;
        Pair p;
        p.gu = std::atoi(item.substr(0, s).c_str());
        p.d = std::atoi(item.substr(s + 1).c_str());
        if (p.gu <= 0 || p.d <= 0) return false;
        out.push_back(p);
    }
    return !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::max(20, std::atoi(argv[1])) : 300;
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    const int groups_req = argc > 3 ? std::min(64, std::max(kTop, std::atoi(argv[3]))) : 20;
    std::vector<Pair> pairs;
    if (argc > 4 && std::string(argv[4]) != "all") {
        if (!parse_pairs(argv[4], pairs)) {
            std::printf("fp_expert_bench: pairs '%s' is not a comma list of gu/down ggml type ids\n", argv[4]);
            return 1;
        }
    } else {
        pairs.assign(std::begin(kModelPairs), std::end(kModelPairs));
    }
    int n_dev = 0;
    if (!ck(cudaGetDeviceCount(&n_dev), "device count") || n_dev == 0) {
        std::printf("fp_expert_bench: no CUDA device\n");
        return 1;
    }
    if (only >= n_dev) {
        std::printf("fp_expert_bench: device %d, only %d present\n", only, n_dev);
        return 1;
    }
    for (const Pair& p : pairs) {
        if (!K::native_expert_supported(p.gu, p.d, kEmbd, kFf)) {
            std::printf("fp_expert_bench: native_expert_supported(%d, %d, %d, %d) is false\n", p.gu, p.d, kEmbd, kFf);
            return 1;
        }
    }
    for (const char* name : {"STRATA_NO_SUB16_GU", "STRATA_EXPERT_V2"}) {
        const char* v = std::getenv(name);
        if (v != nullptr && v[0] != '\0' && v[0] != '0') std::printf("# warning: %s is set, the variants are bypassed\n", name);
    }
    const char* pairs_env_raw = std::getenv("STRATA_FP_EXPERT_V4_PAIRS");
    const std::string pairs_env = pairs_env_raw != nullptr ? pairs_env_raw : "";
    if (!pairs_env.empty() && !K::native_expert_set_fp_v4_pairs(pairs_env.c_str())) {
        std::printf("fp_expert_bench: STRATA_FP_EXPERT_V4_PAIRS '%s' is malformed\n", pairs_env.c_str());
        return 1;
    }
    K::iq_set_old_kernels(false);
    std::printf("# n_embd %d, n_ff %d, %d pair(s); choices 0 and 4 per pair, 1-3 for 21/20, 0t2/0t4/0t8 at T=1, auto\n", kEmbd, kFf,
                (int) pairs.size());
    std::printf("# auto: V=4 for T <= %d (STRATA_FP_EXPERT_V4_TMAX, 1 when unset) on pairs %s, STRATA_FP_DEF_TILES_T1=%d\n",
                std::max(0, env_int("STRATA_FP_EXPERT_V4_TMAX", 1)), pairs_env.empty() ? "(all)" : pairs_env.c_str(),
                std::max(1, env_int("STRATA_FP_DEF_TILES_T1", 1)));
    std::printf("# dev T gu/d v us GB/s pct identical|DIFFERENT\n");
    long long bad = 0;
    bool ok = true;
    for (const Pair& p : pairs) {
        if (!ok) break;
        const K::NativeExpertLayout L = K::native_expert_layout(p.gu, p.d, kEmbd, kFf);
        const size_t stride = (L.bytes + 255) / 256 * 256;
        const size_t P = std::max<size_t>((size_t) (96u << 20) / stride + 1, (size_t) 2 * (size_t) groups_req);
        std::printf("# pair %d/%d: gate/up %s, down %s: blob %.2f MB (gu_row %d, d_row %d), pool %.0f MB\n", p.gu, p.d,
                    type_name(p.gu), type_name(p.d), (double) L.bytes / 1e6, (int) L.gu_row, (int) L.d_row,
                    (double) (P * stride) / 1e6);
        std::mt19937 rng(77u + (unsigned) (p.gu * 131 + p.d));
        std::vector<uint8_t> pool(P * stride, 0);
        for (size_t i = 0; i < P; ++i) fill_blob(pool.data() + i * stride, L, rng);
        std::vector<float> xh((size_t) kMaxT * kEmbd);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (float& v : xh) v = nd(rng);
        for (int dev = 0; dev < n_dev; ++dev) {
            if (only >= 0 && dev != only) continue;
            ok = run_device(dev, reps, groups_req, pool, P, stride, xh, L, p, pairs_env.c_str(), bad) && ok;
        }
    }
    std::printf("\nfp_expert_bench: %s (%lld variant comparisons differ%s)\n", (bad || !ok || g_cuda_error) ? "FAILED" : "ok",
                bad, (!ok || g_cuda_error) ? ", CUDA error" : "");
    return (bad || !ok || g_cuda_error) ? 1 : 0;
}
