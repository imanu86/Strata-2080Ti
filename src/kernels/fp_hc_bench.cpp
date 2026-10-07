// src/kernels/fp_hc_bench.cpp - fused_gr_read_multi with the norm folded into the down projection
// (STRATA_FP_HC_FUSE_NORM, CUDA) against the default read: bit for bit on every output (lo, rs, inject, mixed, the
// in-place R, the xn scratch) and each one's time, per card, token count, pending write and inject rows.
//
//     build/fp_hc_bench [iters=500] [T min=1] [T max=6] [device=-1: every card]
//
// Exit code 1 on any difference, any CUDA error, or a card where the fused-norm read does not run (its check failed).
#include "strata/kernels/fused_gr.hpp"

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

static int g_cuda_errors = 0;

static bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "fp_hc_bench: %s: %s\n", what, cudaGetErrorString(e));
    ++g_cuda_errors;
    return false;
}
#define CK(x) ck((x), #x)

static uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// Returns the number of failures on this card (differences, a fused-norm read that does not run).
static int run_device(int dev, int iters, int t_lo, int t_hi) {
    if (!CK(cudaSetDevice(dev))) return 1;
    cudaDeviceProp prop;
    if (!CK(cudaGetDeviceProperties(&prop, dev))) return 1;
    std::printf("device %d: %s (sm_%d%d)\n", dev, prop.name, prop.major, prop.minor);

    // The check runs the fused-norm read against the plain one on this card (and picks the default read's variant);
    // the flag has to be on when it runs.  Then the flag is flipped per run.
    K::fused_gr_set_fp_fuse_norm(1);
    K::fused_gr_check();
    if (!K::fused_gr_fp_fuse_norm_active()) {
        std::printf("dev %d: the fused-norm read does not run here (its check failed, the card does not run the staged "
                    "read - see STRATA_HC_SPLIT - or this is not a CUDA build)\n", dev);
        return 1;
    }

    const int N = 2560, HC = 4, D = N * HC, LR = 320, TM = K::kFusedGrMaxT;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto fill = [&](std::vector<float>& v, float sc) { for (auto& x : v) x = sc * nd(rng); };
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    fill(R, 1.0f); fill(bo, 0.5f); fill(inj, 1.0f); fill(wn, 0.3f);
    for (auto& x : wn) x += 1.0f;
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    for (auto& x : wd) x = bf16(0.02f * nd(rng));
    for (auto& x : wu) x = bf16(0.05f * nd(rng));
    for (auto& x : wi) x = bf16(0.02f * nd(rng));

    float *dR = nullptr, *dbo = nullptr, *dinj = nullptr, *dwn = nullptr, *dxn = nullptr, *dlo = nullptr, *drs = nullptr,
          *dio = nullptr, *dmix = nullptr;
    uint16_t *dwd = nullptr, *dwu = nullptr, *dwi = nullptr;
    bool ok = CK(cudaMalloc((void**) &dR, R.size() * 4)) && CK(cudaMalloc((void**) &dbo, bo.size() * 4)) &&
              CK(cudaMalloc((void**) &dinj, inj.size() * 4)) && CK(cudaMalloc((void**) &dwn, wn.size() * 4)) &&
              CK(cudaMalloc((void**) &dxn, (size_t) TM * D * 4)) && CK(cudaMalloc((void**) &dlo, (size_t) TM * LR * 4)) &&
              CK(cudaMalloc((void**) &drs, (size_t) TM * HC * 4)) && CK(cudaMalloc((void**) &dio, (size_t) TM * HC * 4)) &&
              CK(cudaMalloc((void**) &dmix, (size_t) TM * N * 4)) && CK(cudaMalloc((void**) &dwd, wd.size() * 2)) &&
              CK(cudaMalloc((void**) &dwu, wu.size() * 2)) && CK(cudaMalloc((void**) &dwi, wi.size() * 2));
    if (ok)
        ok = CK(cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice)) &&
             CK(cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice)) &&
             CK(cudaMemcpy(dwn, wn.data(), wn.size() * 4, cudaMemcpyHostToDevice)) &&
             CK(cudaMemcpy(dwd, wd.data(), wd.size() * 2, cudaMemcpyHostToDevice)) &&
             CK(cudaMemcpy(dwu, wu.data(), wu.size() * 2, cudaMemcpyHostToDevice)) &&
             CK(cudaMemcpy(dwi, wi.data(), wi.size() * 2, cudaMemcpyHostToDevice));
    cudaStream_t s = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    if (ok) ok = CK(cudaStreamCreate(&s)) && CK(cudaEventCreate(&e0)) && CK(cudaEventCreate(&e1));
    int failures = 0;
    if (!ok) failures = 1;

    for (int T = t_lo; ok && T <= t_hi; ++T)
        for (int apply = 0; ok && apply < 2; ++apply)
            for (int inject = 0; ok && inject < 2; ++inject) {
                K::FusedGrArgs a[K::kFusedGrMaxT];
                for (int t = 0; t < T; ++t) {
                    a[t].R = dR + (size_t) t * D; a[t].R_out = dR + (size_t) t * D; a[t].apply = apply != 0;
                    a[t].bo_prev = dbo + (size_t) t * N; a[t].inj_prev = dinj + (size_t) t * HC;
                    a[t].w_norm = dwn; a[t].w_down = dwd; a[t].w_up = dwu; a[t].w_inject = inject ? dwi : nullptr;
                    a[t].lo = dlo + (size_t) t * LR; a[t].rs = drs + (size_t) t * HC;
                    a[t].inject_out = dio + (size_t) t * HC; a[t].mixed = dmix + (size_t) t * N;   // != inj_prev
                }
                std::vector<float> out[2];
                for (int f = 0; f < 2; ++f) {       // f = 0: the default read, 1: the fused-norm read
                    ok = CK(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice)) &&
                         CK(cudaMemset(dlo, 0xff, (size_t) TM * LR * 4)) && CK(cudaMemset(drs, 0xff, (size_t) TM * HC * 4)) &&
                         CK(cudaMemset(dio, 0xff, (size_t) TM * HC * 4)) && CK(cudaMemset(dmix, 0xff, (size_t) TM * N * 4)) &&
                         CK(cudaMemset(dxn, 0xff, (size_t) TM * D * 4));
                    if (!ok) break;
                    K::fused_gr_set_fp_fuse_norm(f);
                    K::fused_gr_read_multi(a, T, dxn, s);
                    ok = CK(cudaStreamSynchronize(s));
                    if (!ok) break;
                    auto& o = out[f];
                    o.resize((size_t) TM * (D + LR + HC + HC + N + D));
                    float* p = o.data();
                    ok = CK(cudaMemcpy(p, dR, (size_t) TM * D * 4, cudaMemcpyDeviceToHost));
                    p += (size_t) TM * D;
                    ok = ok && CK(cudaMemcpy(p, dlo, (size_t) TM * LR * 4, cudaMemcpyDeviceToHost));
                    p += (size_t) TM * LR;
                    ok = ok && CK(cudaMemcpy(p, drs, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost));
                    p += (size_t) TM * HC;
                    ok = ok && CK(cudaMemcpy(p, dio, (size_t) TM * HC * 4, cudaMemcpyDeviceToHost));
                    p += (size_t) TM * HC;
                    ok = ok && CK(cudaMemcpy(p, dmix, (size_t) TM * N * 4, cudaMemcpyDeviceToHost));
                    p += (size_t) TM * N;
                    ok = ok && CK(cudaMemcpy(p, dxn, (size_t) TM * D * 4, cudaMemcpyDeviceToHost));
                    if (!ok) break;
                }
                if (!ok) { ++failures; break; }
                const bool same = out[0].size() == out[1].size() &&
                                  std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
                double us[2][3];
                for (int round = 0; ok && round < 3; ++round)
                    for (int f = 0; ok && f < 2; ++f) {
                        K::fused_gr_set_fp_fuse_norm(f);
                        for (int i = 0; i < 20; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        ok = CK(cudaEventRecord(e0, s));
                        for (int i = 0; i < iters; ++i) K::fused_gr_read_multi(a, T, dxn, s);
                        ok = ok && CK(cudaEventRecord(e1, s)) && CK(cudaEventSynchronize(e1));
                        float ms = 0;
                        ok = ok && CK(cudaEventElapsedTime(&ms, e0, e1));
                        us[f][round] = 1e3 * (double) ms / iters;
                    }
                if (!ok) { ++failures; break; }
                double med[2];
                for (int f = 0; f < 2; ++f) {
                    std::sort(us[f], us[f] + 3);
                    med[f] = us[f][1];
                }
                std::printf("dev %d T %d apply %d inject %d | default %7.1f us | fused %7.1f us | %s\n", dev, T, apply,
                            inject, med[0], med[1], same ? "bitwise equal" : "DIFFERS");
                if (!same) ++failures;
            }
    K::fused_gr_set_fp_fuse_norm(-1);
    if (s != nullptr) cudaStreamSynchronize(s);
    if (e0 != nullptr) cudaEventDestroy(e0);
    if (e1 != nullptr) cudaEventDestroy(e1);
    if (s != nullptr) cudaStreamDestroy(s);
    cudaFree(dR); cudaFree(dbo); cudaFree(dinj); cudaFree(dwn); cudaFree(dxn); cudaFree(dlo); cudaFree(drs);
    cudaFree(dio); cudaFree(dmix); cudaFree(dwd); cudaFree(dwu); cudaFree(dwi);
    return failures;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int iters = argc > 1 ? std::atoi(argv[1]) : 500;
    int t_lo = argc > 2 ? std::atoi(argv[2]) : 1, t_hi = argc > 3 ? std::atoi(argv[3]) : 6;
    const int device = argc > 4 ? std::atoi(argv[4]) : -1;
    if (iters < 1) iters = 500;
    t_lo = std::max(t_lo, 1);
    t_hi = std::min(t_hi, (int) K::kFusedGrMaxT);
    int n_dev = 0;
    if (!CK(cudaGetDeviceCount(&n_dev)) || n_dev < 1) {
        std::fprintf(stderr, "fp_hc_bench: no CUDA device\n");
        return 1;
    }
    if (device >= n_dev) {
        std::fprintf(stderr, "fp_hc_bench: device %d does not exist (%d found)\n", device, n_dev);
        return 1;
    }
    int failures = 0;
    for (int dev = 0; dev < n_dev; ++dev) {
        if (device >= 0 && dev != device) continue;
        failures += run_device(dev, iters, t_lo, t_hi);
    }
    std::printf("fp_hc_bench: %d differing or failing, %d CUDA errors\n", failures, g_cuda_errors);
    return failures || g_cuda_errors ? 1 : 0;
}
