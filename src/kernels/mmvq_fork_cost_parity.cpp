// src/kernels/mmvq_fork_cost_parity.cpp - reuse upstream mmvq_il_parity's synthetic byte-parity fixture,
// comparing the current fork native/B6 path to IL for the actual split16 stage0 dense shapes.
// --bench uses BAB (auto-table IL/native/IL) for kernel-only and interleave+kernel separately.
// Warm-up is outside each arm metric; the full interleave is not amortized across projections.
// No table/default is written or changed.
// The same Q8_1 bytes feed both paths; finite random scales, uncached rotating weight copies.
// No model, no engine/session, no persistent allocations; CUDA/HIP runtime errors fail the process.
//
// mmvq_fork_cost_parity [--device N] [--case N] [--cols 2|3|4] [--bench [--reps 200]]
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

namespace {
using namespace strata::kernels;

int g_cols = 0;
int g_reps = 200;
struct Case {
    const char* name;
    int type, n_in, n_out, block_elems, block_bytes, scale_at[2];
};
// Unique actual-model shapes/formats in the first 16 layers; odd rows are extra boundary controls.
const Case CASES[] = {
    {"IQ4_XS 2560x10240", 23, 2560, 10240, 256, 136, {0, -1}},
    {"IQ4_XS 2560x6144", 23, 2560, 6144, 256, 136, {0, -1}},
    {"IQ4_XS 2560x12288", 23, 2560, 12288, 256, 136, {0, -1}},
    {"IQ4_XS 6144x2560", 23, 6144, 2560, 256, 136, {0, -1}},
    {"IQ4_XS 2560x512", 23, 2560, 512, 256, 136, {0, -1}},
    {"Q4_K 2560x10240", 12, 2560, 10240, 256, 144, {0, 2}},
    {"Q4_K 2560x6144", 12, 2560, 6144, 256, 144, {0, 2}},
    {"Q4_K 2560x12288", 12, 2560, 12288, 256, 144, {0, 2}},
    {"Q4_K 6144x2560", 12, 6144, 2560, 256, 144, {0, 2}},
    {"Q4_K 2560x512", 12, 2560, 512, 256, 144, {0, 2}},
    {"Q5_K 6144x2560", 13, 6144, 2560, 256, 176, {0, 2}},
    {"Q6_K 2560x10240", 14, 2560, 10240, 256, 210, {208, -1}},
    {"Q6_K 2560x6144", 14, 2560, 6144, 256, 210, {208, -1}},
    {"Q6_K 6144x2560", 14, 6144, 2560, 256, 210, {208, -1}},
    {"Q6_K 2560x512", 14, 2560, 512, 256, 210, {208, -1}},
    {"Q4_K odd rows 2560x2051", 12, 2560, 2051, 256, 144, {0, 2}},
    {"Q6_K odd rows 2560x2051", 14, 2560, 2051, 256, 210, {208, -1}},
};

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}
bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
}

int run_case(const Case& c, bool bench, cudaStream_t s) {
    const std::size_t wbytes = native_mmvq_weight_bytes(c.type, c.n_in, c.n_out);
    std::mt19937 rng(77u + (unsigned) c.type * 31u + (unsigned) c.n_out);
    std::vector<uint8_t> w(wbytes);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t n_blocks = (std::size_t) c.n_out * (std::size_t) (c.n_in / c.block_elems);
    if (n_blocks * (std::size_t) c.block_bytes != wbytes) {
        std::printf("%s: weight bytes mismatch\n", c.name);
        return 1;
    }
    for (std::size_t k = 0; k < n_blocks; ++k)
        for (int at : c.scale_at)
            if (at >= 0) {
                const uint16_t h = sane_half(rng);
                std::memcpy(&w[k * (std::size_t) c.block_bytes + (std::size_t) at], &h, 2);
            }
    const int copies = bench ? (int) std::max<std::size_t>(1, std::min<std::size_t>(24, (192u << 20) / wbytes)) : 1;
    std::vector<void*> dw(copies, nullptr);
    for (int i = 0; i < copies; ++i) {
        if (!ck(cudaMalloc(&dw[i], wbytes), "malloc w") || !ck(cudaMemcpy(dw[i], w.data(), wbytes, cudaMemcpyHostToDevice), "copy w"))
            return 1;
    }
    int bad = 0;
    for (int T = 2; T <= 4; ++T) {
        if (g_cols != 0 && T != g_cols) continue;
        std::vector<float> x((std::size_t) T * c.n_in);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr;
        void *xq = nullptr, *xil = nullptr;
        float *yref = nullptr, *yil = nullptr;
        const std::size_t ybytes = (std::size_t) T * c.n_out * 4;
        if (!ck(cudaMalloc(&dx, x.size() * 4), "malloc x") ||
            !ck(cudaMalloc(&xq, native_q8_1_bytes(c.n_in, T)), "malloc xq") ||
            !ck(cudaMalloc(&xil, native_q8_1_il_bytes(c.n_in, T)), "malloc xil") ||
            !ck(cudaMalloc((void**) &yref, ybytes), "malloc yref") || !ck(cudaMalloc((void**) &yil, ybytes), "malloc yil"))
            return 1;
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x");
        quantize_q8_1_rows(dx, T, c.n_in, xq, s);
        native_q8_1_interleave(xq, xil, c.n_in, T, s);
        native_mmvq(c.type, dw[0], xq, yref, c.n_in, c.n_out, T, s);
        ck(cudaStreamSynchronize(s), "ref");
        std::vector<uint32_t> a((std::size_t) T * c.n_out), b(a.size());
        ck(cudaMemcpy(a.data(), yref, ybytes, cudaMemcpyDeviceToHost), "read ref");
        for (int r : {0, 1, 2, 4}) {
            native_mmvq_il_tune(r);
            ck(cudaMemset(yil, 0xff, ybytes), "poison output");
            ck(cudaMemset(xil, 0xff, native_q8_1_il_bytes(c.n_in, T)), "poison interleave");
            native_q8_1_interleave(xq, xil, c.n_in, T, s);
            native_mmvq_il(c.type, dw[0], xq, xil, yil, c.n_in, c.n_out, T, s);
            ck(cudaStreamSynchronize(s), "il");
            ck(cudaMemcpy(b.data(), yil, ybytes, cudaMemcpyDeviceToHost), "read il");
            long long diff = 0, nonfinite = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                float fa;
                std::memcpy(&fa, &a[i], 4);
                if (!std::isfinite(fa)) ++nonfinite;
                if (a[i] != b[i]) ++diff;
            }
            if (diff != 0 || nonfinite != 0) {
                ++bad;
                std::printf("FAIL %-26s T=%d rows=%d: %lld bits differ, %lld non-finite of %zu\n", c.name, T, r, diff, nonfinite, a.size());
            }
        }
        native_mmvq_il_tune(0);   // parity covers forced rows; timing covers only the production auto table
        if (bench && native_mmvq_il_supported(c.type, T, c.n_out)) {
            cudaEvent_t e0, e1;
            ck(cudaEventCreate(&e0), "event");
            ck(cudaEventCreate(&e1), "event");
            for (int total = 0; total < 2; ++total) {
                double us[3] = {};
                for (int arm = 0; arm < 3; ++arm) {   // B1(auto IL) -> A(native/B6) -> B2(auto IL)
                    auto call = [&](int i) {
                        if (arm == 1) native_mmvq(c.type, dw[i % copies], xq, yil, c.n_in, c.n_out, T, s);
                        else {
                            if (total) native_q8_1_interleave(xq, xil, c.n_in, T, s);
                            native_mmvq_il(c.type, dw[i % copies], xq, xil, yil, c.n_in, c.n_out, T, s);
                        }
                    };
                    // Every arm starts the same weight rotation at zero and writes the same output buffer.
                    for (int i = 0; i < g_reps; ++i) call(i);
                    ck(cudaStreamSynchronize(s), "arm warm");
                    ck(cudaEventRecord(e0, s), "record");
                    for (int i = 0; i < g_reps; ++i) call(i);
                    ck(cudaEventRecord(e1, s), "record");
                    ck(cudaEventSynchronize(e1), "sync");
                    float ms = 0.0f;
                    ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                    us[arm] = 1000.0 * ms / g_reps;
                }
                const double ratio = us[1] > 0.0 ? (us[0] + us[2]) / (2.0 * us[1]) : std::nan("");
                std::printf("BENCH %-26s T=%d %-22s B1 %.2f A %.2f B2 %.2f us meanB/A %.4f\n", c.name, T,
                            total ? "interleave+kernel" : "kernel-only", us[0], us[1], us[2], ratio);
            }
            ck(cudaEventDestroy(e0), "destroy event");
            ck(cudaEventDestroy(e1), "destroy event");
        } else if (bench) {
            std::printf("BENCH %-26s T=%d auto IL unsupported: native fallback parity checked, no IL timing\n", c.name, T);
        }
        ck(cudaFree(dx), "free x"); ck(cudaFree(xq), "free xq"); ck(cudaFree(xil), "free xil");
        ck(cudaFree(yref), "free yref"); ck(cudaFree(yil), "free yil");
    }
    for (void* p : dw) ck(cudaFree(p), "free weights");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    int device = 0, only_case = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc) only_case = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) g_cols = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) g_reps = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: mmvq_fork_cost_parity [--device N] [--case N] [--cols 2|3|4] [--bench [--reps N]]\n");
            return 2;
        }
    }
    const int n_cases = (int) (sizeof(CASES) / sizeof(CASES[0]));
    if (device < 0 || only_case < -1 || only_case >= n_cases || (g_cols && (g_cols < 2 || g_cols > 4)) ||
        g_reps < 1 || g_reps > 1000) return 2;
    const char* tsum = std::getenv("STRATA_TSUM");
    if (tsum != nullptr && tsum[0] == '1') {
        std::fprintf(stderr, "STRATA_TSUM=1 bypasses B6; unset it for this native/B6 comparison\n");
        return 2;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    if (device >= devices) return 2;
    ck(cudaSetDevice(device), "device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "properties");
    std::printf("mmvq_fork_cost_parity device%d %s cc%d.%d; multi_exact=1 B6rows=1 (native baseline), reps%d\n",
                device, prop.name, prop.major, prop.minor, g_reps);
    native_mmvq_set_multi_exact(true);
    native_mmvq_set_b6_rows(1);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    int bad = 0;
    for (int i = 0; i < n_cases; ++i) {
        if (only_case != -1 && only_case != i) continue;
        std::printf("CASE%d %s\n", i, CASES[i].name);
        bad += run_case(CASES[i], bench, s);
    }
    native_mmvq_il_tune(0);
    ck(cudaStreamDestroy(s), "destroy stream");
    std::printf("%s\n", bad ? "mmvq_fork_cost_parity: FAIL" : "mmvq_fork_cost_parity: PASS (finite, byte-identical; unsupported IL uses native fallback)");
    return bad ? 1 : 0;
}
