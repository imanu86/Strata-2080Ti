// src/kernels/fp_expert_bench.cpp - what the VRAM experts of a decode verify window cost, per kernel variant, per card.
//
//     build/fp_expert_bench [reps = 300] [device, -1 = all] [groups = 20]
//
// The model's own expert format (IQ3_S gate/up, IQ4_NL down, n_embd 2560, n_ff 640: a 2.33 MB blob per expert) through
// `native_expert_grouped` (gate/up + SwiGLU + down) with the opt-in kernel variants STRATA_FP_EXPERT_V / native_expert_
// set_fp_variant: 0 = the default kernels, 1 = IQ3_S codebook in shared memory, 2 = two rows per sub-warp, 3 = 1 + 2,
// 4 = persistent gate/up and down (STRATA_FP_EXPERT_PERSIST_K blocks per SM, default 2).
//
// For T = 1..4 tokens a verify window is built: every token picks 10 experts out of `groups` distinct ones (at most
// 10 T; every expert picked by at least one token, a group holds the tokens that picked it, 1..T entries).  Per variant:
// the output buffer against variant 0's, bit for bit (memcmp), and the time of `reps` back-to-back calls with CUDA events
// (median of 5 rounds).  The experts rotate over a pool of >= 96 MB of distinct synthetic blobs (random bytes, sane fp16
// block scales, as b6_mmvq_bench), so every call reads its weights cold from DRAM as the decode does each layer.
//
// One line per (device, T, variant):  dev T v us GB/s pct identical|DIFFERENT   (GB/s = groups x blob bytes / call time;
// pct = GB/s over the peak DRAM bandwidth 2 x memory clock x bus width / 8 of the device attributes, the DDR factor 2
// of GDDR6).
// Lines starting with '#' are context.  Exit code 1 if any variant differs or a CUDA error occurred.  GPU time only.
// STRATA_NO_SUB16_GU and STRATA_EXPERT_V2 route around the variants (all four rows would then be the default kernels).
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

namespace {

constexpr int kEmbd = 2560, kFf = 640, kTop = 10, kMaxT = 4, kGuType = 21, kDType = 20;
constexpr int kIq3sBlock = 110, kIq4nlBlock = 18;   // block_iq3_s {d, qs[64], qh[8], signs[32], scales[4]}, block_iq4_nl {d, qs[16]}
constexpr int kMaxEntries = kMaxT * kTop;           // every token picks exactly kTop experts

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

// One expert blob [gate rows | up rows | down rows]: random bytes (any qs / qh / signs / scales nibble / IQ4_NL nibble
// is a valid code) with a sane fp16 `d` in every block.
void fill_blob(uint8_t* b, const K::NativeExpertLayout& L, std::mt19937& rng) {
    for (size_t i = 0; i + 4 <= L.bytes; i += 4) {
        const uint32_t r = rng();
        std::memcpy(b + i, &r, 4);
    }
    const size_t gu_blocks = (size_t) 2 * (size_t) L.n_ff * (size_t) (L.n_embd / 256);   // gate, then up: contiguous
    for (size_t k = 0; k < gu_blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(b + k * kIq3sBlock, &h, 2);
    }
    const size_t d_blocks = (size_t) L.n_embd * (size_t) (L.n_ff / 32);
    for (size_t k = 0; k < d_blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(b + L.down_off + k * kIq4nlBlock, &h, 2);
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

bool run_device(int dev, int reps, int groups_req, const std::vector<uint8_t>& pool, size_t P, size_t stride,
                const std::vector<float>& xh, const K::NativeExpertLayout& L, long long& bad) {
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
    const int variant_before = K::native_expert_fp_variant();
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
        auto call = [&](size_t i) {
            K::native_expert_grouped(L, dptr + (i % P) * (size_t) G, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, stream);
        };
        std::vector<float> ref, res((size_t) NE * kEmbd);
        for (int v = 0; v <= 4 && ok; ++v) {
            K::native_expert_set_fp_variant(v);
            ok = ck(cudaMemset(dout, 0, (size_t) NE * kEmbd * sizeof(float)), "memset out");
            call(0);
            ok = ok && ck(cudaStreamSynchronize(se.s), "run") && ck(cudaGetLastError(), "kernel") &&
                 ck(cudaMemcpy(res.data(), dout, res.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out");
            if (!ok) break;
            bool same = true;
            if (v == 0) {
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
            std::printf("%d %d %d %.1f %.1f %.1f%% %s\n", dev, T, v, us, gbps, peak > 0.0 ? 100.0 * gbps * 1e9 / peak : 0.0,
                        same ? "identical" : "DIFFERENT");
            if (!same) ++bad;
        }
    }
    K::native_expert_set_fp_variant(variant_before);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int reps = argc > 1 ? std::max(20, std::atoi(argv[1])) : 300;
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    const int groups_req = argc > 3 ? std::min(64, std::max(kTop, std::atoi(argv[3]))) : 20;
    int n_dev = 0;
    if (!ck(cudaGetDeviceCount(&n_dev), "device count") || n_dev == 0) {
        std::printf("fp_expert_bench: no CUDA device\n");
        return 1;
    }
    if (only >= n_dev) {
        std::printf("fp_expert_bench: device %d, only %d present\n", only, n_dev);
        return 1;
    }
    if (!K::native_expert_supported(kGuType, kDType, kEmbd, kFf)) {
        std::printf("fp_expert_bench: native_expert_supported(%d, %d, %d, %d) is false\n", kGuType, kDType, kEmbd, kFf);
        return 1;
    }
    for (const char* name : {"STRATA_NO_SUB16_GU", "STRATA_EXPERT_V2"}) {
        const char* v = std::getenv(name);
        if (v != nullptr && v[0] != '\0' && v[0] != '0') std::printf("# warning: %s is set, the variants are bypassed\n", name);
    }
    K::iq_set_old_kernels(false);
    const K::NativeExpertLayout L = K::native_expert_layout(kGuType, kDType, kEmbd, kFf);
    const size_t stride = (L.bytes + 255) / 256 * 256;
    const size_t P = std::max<size_t>((size_t) (96u << 20) / stride + 1, (size_t) 2 * (size_t) groups_req);
    std::printf("# gate/up type %d, down type %d, n_embd %d, n_ff %d: blob %.2f MB (gu_row %d, d_row %d), pool %.0f MB\n",
                kGuType, kDType, kEmbd, kFf, (double) L.bytes / 1e6, (int) L.gu_row, (int) L.d_row,
                (double) (P * stride) / 1e6);
    std::printf("# dev T v us GB/s pct identical|DIFFERENT\n");
    std::mt19937 rng(77u);
    std::vector<uint8_t> pool(P * stride, 0);
    for (size_t i = 0; i < P; ++i) fill_blob(pool.data() + i * stride, L, rng);
    std::vector<float> xh((size_t) kMaxT * kEmbd);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float& v : xh) v = nd(rng);
    long long bad = 0;
    bool ok = true;
    for (int dev = 0; dev < n_dev; ++dev) {
        if (only >= 0 && dev != only) continue;
        ok = run_device(dev, reps, groups_req, pool, P, stride, xh, L, bad) && ok;
    }
    std::printf("\nfp_expert_bench: %s (%lld variant comparisons differ%s)\n", (bad || !ok || g_cuda_error) ? "FAILED" : "ok",
                bad, (!ok || g_cuda_error) ? ", CUDA error" : "");
    return (bad || !ok || g_cuda_error) ? 1 : 0;
}
