// src/kernels/fp_shexp_bench.cpp - STRATA_FP_SHEXP_FUSE: the shared expert of a verify window (shared_expert_multi) with
// its launches fused (level 1: the sigmoid gate in the down kernel's epilogue; level 2: also the SwiGLU + q8_1 in its
// prologue) against the default sequence, bit for bit on every output, and each one's time.
//
//     build/fp_shexp_bench [iters = 500] [device, -1 = all]
//
// The owner's model geometry: gate Q4_K [640][2560], up IQ3_S [640][2560], down IQ4_NL [2560][640], the scalar gate a
// BF16 [2560] row, native BF16 gate on (as the engine runs it), lfuse 0, no ready q8_1.  Synthetic weight bytes with
// small fp16 scales (the time does not depend on the values, the sums stay finite), random normal activations.  Per
// device and T = 1..4: the default call and each fusion level on the same inputs, `out` (T x 2560 floats) and the raw
// gate values `g` compared with memcmp; then the time of each: the median of 3 rounds of `iters` back-to-back calls.
// Exit code 1 on any difference or CUDA error.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/shared_expert.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <vector>

namespace K = strata::kernels;

namespace {

constexpr int N_EMBD = 2560, N_FF = 640, T_MAX = 4, T_BUF = 8, LEVELS = 3;
constexpr int GATE_TYPE = 12, UP_TYPE = 21, DOWN_TYPE = 20;

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

uint16_t small_half(std::mt19937& rng) {   // 2^-13 .. 2^-10, either sign
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((2u + (r >> 10) % 4u) << 10) | (r & 0x3ffu));
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// random bytes with a fp16 scale at each byte offset of `scale_at` in every block
std::vector<uint8_t> make_weights(std::mt19937& rng, int type, int n_in, int n_out, int block_elems, int block_bytes,
                                  const int* scale_at, int n_scale) {
    std::vector<uint8_t> w(K::native_mmvq_weight_bytes(type, n_in, n_out));
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const size_t nb = (size_t) n_out * (size_t) (n_in / block_elems);
    for (size_t k = 0; k < nb; ++k)
        for (int s = 0; s < n_scale; ++s) {
            const uint16_t h = small_half(rng);
            std::memcpy(&w[k * (size_t) block_bytes + (size_t) scale_at[s]], &h, 2);
        }
    return w;
}

struct Dev {
    void *wg = nullptr, *wu = nullptr, *wd = nullptr, *q8 = nullptr;
    uint16_t* gi = nullptr;
    float *x = nullptr, *gate = nullptr, *up = nullptr, *g = nullptr, *out[LEVELS] = {nullptr, nullptr, nullptr};
    cudaStream_t s = nullptr;
    K::NativeSharedWeights nw;
    void release() {
        cudaFree(wg); cudaFree(wu); cudaFree(wd); cudaFree(q8); cudaFree(gi); cudaFree(x); cudaFree(gate); cudaFree(up);
        cudaFree(g);
        for (float* o : out) cudaFree(o);
        if (s) cudaStreamDestroy(s);
    }
};

void run(const Dev& d, int T, int level) {
    K::shared_expert_set_fp_fuse(level);
    K::shared_expert_multi(T, d.x, nullptr, d.nw, d.gi, d.gate, d.up, d.g, d.out[level], N_EMBD, N_FF, d.s, nullptr, 0);
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int iters = argc > 1 ? std::max(10, std::atoi(argv[1])) : 500;
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    int n_dev = 0;
    if (!ck(cudaGetDeviceCount(&n_dev), "device count")) return 1;
    K::shared_expert_set_native_bf16(true);   // the engine's choice (--native-bf16-extra, generate.cpp): the BF16 MMVF scalar gate

    // host data, identical on every device
    std::mt19937 rng(20261007u);
    const int q4k_scales[2] = {0, 2}, one_scale[1] = {0};
    const std::vector<uint8_t> hg = make_weights(rng, GATE_TYPE, N_EMBD, N_FF, 256, 144, q4k_scales, 2);   // Q4_K
    const std::vector<uint8_t> hu = make_weights(rng, UP_TYPE, N_EMBD, N_FF, 256, 110, one_scale, 1);       // IQ3_S
    const std::vector<uint8_t> hd = make_weights(rng, DOWN_TYPE, N_FF, N_EMBD, 32, 18, one_scale, 1);       // IQ4_NL
    std::vector<uint16_t> hgi((size_t) N_EMBD);
    std::normal_distribution<float> nw_gate(0.f, 0.05f), nx(0.f, 1.f);
    for (auto& v : hgi) v = bf16(nw_gate(rng));
    std::vector<float> hx((size_t) T_BUF * N_EMBD);
    for (auto& v : hx) v = nx(rng);

    long long bad = 0;
    for (int dev = 0; dev < n_dev; ++dev) {
        if (only >= 0 && dev != only) continue;
        if (!ck(cudaSetDevice(dev), "set device")) return 1;
        cudaDeviceProp pr{};
        cudaGetDeviceProperties(&pr, dev);
        std::printf("\n=== device %d: %s (sm_%d%d), %d calls per round, median of 3 rounds, us per call ===\n", dev, pr.name,
                    pr.major, pr.minor, iters);
        std::printf("%3s %2s | %10s | %10s | %10s | %s\n", "dev", "T", "default", "fuse1", "fuse2", "out + g");

        Dev d;
        bool ok = ck(cudaStreamCreate(&d.s), "stream");
        const size_t q8_bytes = K::native_q8_1_bytes(std::max(N_EMBD, N_FF), T_BUF);
        ok = ok && ck(cudaMalloc(&d.wg, hg.size()), "malloc gate") && ck(cudaMalloc(&d.wu, hu.size()), "malloc up") &&
             ck(cudaMalloc(&d.wd, hd.size()), "malloc down") && ck(cudaMalloc(&d.q8, q8_bytes), "malloc q8_1") &&
             ck(cudaMalloc(&d.gi, hgi.size() * 2), "malloc gate_inp") && ck(cudaMalloc(&d.x, hx.size() * 4), "malloc x") &&
             ck(cudaMalloc(&d.gate, (size_t) T_BUF * N_FF * 4), "malloc gate act") &&
             ck(cudaMalloc(&d.up, (size_t) T_BUF * N_FF * 4), "malloc up act") &&
             ck(cudaMalloc(&d.g, (size_t) T_BUF * 4), "malloc g");
        for (int l = 0; l < LEVELS && ok; ++l) ok = ck(cudaMalloc(&d.out[l], (size_t) T_BUF * N_EMBD * 4), "malloc out");
        ok = ok && ck(cudaMemcpy(d.wg, hg.data(), hg.size(), cudaMemcpyHostToDevice), "copy gate") &&
             ck(cudaMemcpy(d.wu, hu.data(), hu.size(), cudaMemcpyHostToDevice), "copy up") &&
             ck(cudaMemcpy(d.wd, hd.data(), hd.size(), cudaMemcpyHostToDevice), "copy down") &&
             ck(cudaMemcpy(d.gi, hgi.data(), hgi.size() * 2, cudaMemcpyHostToDevice), "copy gate_inp") &&
             ck(cudaMemcpy(d.x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "copy x");
        if (!ok) {
            d.release();
            return 1;
        }
        d.nw.gate_type = GATE_TYPE; d.nw.gate_data = d.wg;
        d.nw.up_type = UP_TYPE; d.nw.up_data = d.wu;
        d.nw.down_type = DOWN_TYPE; d.nw.down_data = d.wd;
        d.nw.q8_1 = d.q8;

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        for (int T = 1; T <= T_MAX; ++T) {
            std::vector<uint32_t> res[LEVELS];
            double us[LEVELS] = {0, 0, 0};
            bool err = false;
            try {
                for (int l = 0; l < LEVELS; ++l) {
                    cudaMemset(d.out[l], 0xff, (size_t) T_BUF * N_EMBD * 4);   // NaN pattern: every output must be written
                    cudaMemset(d.g, 0xff, (size_t) T_BUF * 4);
                    run(d, T, l);
                    if (!ck(cudaStreamSynchronize(d.s), "run")) { err = true; break; }
                    res[l].resize((size_t) T * N_EMBD + (size_t) T);
                    cudaMemcpy(res[l].data(), d.out[l], (size_t) T * N_EMBD * 4, cudaMemcpyDeviceToHost);
                    cudaMemcpy(res[l].data() + (size_t) T * N_EMBD, d.g, (size_t) T * 4, cudaMemcpyDeviceToHost);
                }
                for (int l = 0; l < LEVELS && !err; ++l) {
                    for (int i = 0; i < 20; ++i) run(d, T, l);   // warm-up
                    double round[3];
                    for (int r = 0; r < 3; ++r) {
                        cudaEventRecord(e0, d.s);
                        for (int i = 0; i < iters; ++i) run(d, T, l);
                        cudaEventRecord(e1, d.s);
                        cudaEventSynchronize(e1);
                        float ms = 0;
                        cudaEventElapsedTime(&ms, e0, e1);
                        round[r] = 1e3 * (double) ms / iters;
                    }
                    std::sort(round, round + 3);
                    us[l] = round[1];
                }
            } catch (const std::exception& e) {
                std::printf("shared_expert_multi: %s\n", e.what());
                err = true;
            }
            if (!err && !ck(cudaGetLastError(), "last error")) err = true;
            if (err) {
                ++bad;
                std::printf("%3d %2d | CUDA / launch error\n", dev, T);
                break;
            }
            long long nonfinite = 0;
            for (size_t i = 0; i < (size_t) T * N_EMBD; ++i) {
                float f;
                std::memcpy(&f, &res[0][i], 4);
                if (!(f - f == 0.0f)) ++nonfinite;
            }
            bool same = true;
            for (int l = 1; l < LEVELS; ++l) same = same && std::memcmp(res[0].data(), res[l].data(), res[0].size() * 4) == 0;
            if (!same || nonfinite) ++bad;
            std::printf("%3d %2d | %10.1f | %10.1f | %10.1f | %s%s\n", dev, T, us[0], us[1], us[2],
                        same ? "bitwise equal" : "DIFFERS", nonfinite ? " NONFINITE" : "");
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        d.release();
    }
    K::shared_expert_set_fp_fuse(-1);
    std::printf("\nfp_shexp_bench: %s (%lld device/width comparisons failed)\n", bad ? "FAILED" : "ok", bad);
    return bad ? 1 : 0;
}
