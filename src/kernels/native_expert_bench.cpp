// src/kernels/native_expert_bench.cpp - fork «Strata adattivo»: how fast the GPU's grouped native expert kernel reads
// a layer's experts, on real GGUF rows, with the shape of a verify window (a few tokens, a group per expert).
//
//     build/native_expert_bench <shard1.gguf> [layer] [experts] [tokens] [iters]
//
// Uploads `experts` real experts of `layer`, routes each token to about two thirds of them (the entries of a verify
// window: every group holds the tokens that picked its expert), quantizes random activations to q8_1 and times
// `native_expert_grouped` (gate/up, SwiGLU, down) with CUDA events.  Prints microseconds per call and the effective
// bandwidth: every expert's whole blob is read once per call.  No profiler and no counters needed.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: native_expert_bench <shard1.gguf> [layer] [experts] [tokens] [iters]\n");
        return 2;
    }
    strata::GgufFile gguf(argv[1]);
    const int l = argc > 2 ? std::atoi(argv[2]) : 20;
    const int G = argc > 3 ? std::atoi(argv[3]) : 18;
    const int T = argc > 4 ? std::atoi(argv[4]) : 3;
    const int iters = argc > 5 ? std::atoi(argv[5]) : 200;
    const int64_t H = 2560, FF = 640;
    const strata::TensorInfo* t[3] = {};
    const char* roles[3] = {"gate", "up", "down"};
    for (const auto& ti : gguf.tensors())
        for (int r = 0; r < 3; ++r)
            if (ti.name == "blk." + std::to_string(l) + ".ffn_" + roles[r] + "_exps.weight") t[r] = &ti;
    if (!t[0] || !t[1] || !t[2]) { std::printf("layer %d: no expert tensors\n", l); return 1; }
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt((int) t[0]->type, (int) t[2]->type, H, FF, f, err)) { std::printf("%s\n", err.c_str()); return 1; }
    const size_t stride = (f.bytes + 255) / 256 * 256;
    std::vector<uint8_t> blobs((size_t) G * stride);
    for (int e = 0; e < G; ++e) {
        uint8_t* b = blobs.data() + (size_t) e * stride;
        std::memcpy(b, gguf.tensor_data(*t[0]) + (size_t) e * f.up_off, f.up_off);
        std::memcpy(b + f.up_off, gguf.tensor_data(*t[1]) + (size_t) e * f.up_off, f.up_off);
        std::memcpy(b + f.down_off, gguf.tensor_data(*t[2]) + (size_t) e * (f.bytes - f.down_off), f.bytes - f.down_off);
    }
    // entries: group g holds the tokens that picked expert g (about two thirds of them, at least one)
    std::vector<int32_t> start(1, 0), dst, tok;
    for (int g = 0; g < G; ++g) {
        for (int k = 0; k < T; ++k)
            if ((g * 7 + k * 3) % 3 != 2 || k == 0) { tok.push_back(k); dst.push_back((int32_t) dst.size()); }
        start.push_back((int32_t) dst.size());
    }
    const int NE = (int) dst.size();
    std::vector<float> x((size_t) T * H);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = nd(rng);
    const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, H, FF);
    void *dblob, *dx, *dxq, *dscr;
    float* dout;
    unsigned long long* dptr;
    int32_t *dstart, *dn, *ddst, *dtok;
    cudaMalloc(&dblob, blobs.size());
    cudaMalloc(&dx, x.size() * 4);
    cudaMalloc(&dxq, (size_t) T * H / 32 * 36);
    cudaMalloc(&dscr, strata::kernels::native_expert_scratch_bytes(NE, FF));
    cudaMalloc((void**) &dout, (size_t) NE * H * 4);
    cudaMalloc((void**) &dptr, (size_t) G * 8);
    cudaMalloc((void**) &dstart, (size_t) (G + 1) * 4);
    cudaMalloc((void**) &dn, 4);
    cudaMalloc((void**) &ddst, (size_t) NE * 4);
    cudaMalloc((void**) &dtok, (size_t) NE * 4);
    cudaMemcpy(dblob, blobs.data(), blobs.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    std::vector<unsigned long long> p((size_t) G);
    for (int g = 0; g < G; ++g) p[(size_t) g] = (unsigned long long) ((uint8_t*) dblob + (size_t) g * stride);
    cudaMemcpy(dptr, p.data(), (size_t) G * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dstart, start.data(), (size_t) (G + 1) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dn, &G, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(ddst, dst.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dtok, tok.data(), (size_t) NE * 4, cudaMemcpyHostToDevice);
    cudaStream_t s;
    cudaStreamCreate(&s);
    strata::kernels::quantize_q8_1_rows((const float*) dx, T, H, dxq, s);
    for (int i = 0; i < 20; ++i)
        strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, s);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, s);
    for (int i = 0; i < iters; ++i)
        strata::kernels::native_expert_grouped(L, dptr, dstart, dn, ddst, dtok, G, NE, dxq, dscr, dout, s);
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    const cudaError_t e = cudaGetLastError();
    const double us = 1000.0 * ms / iters, gb = (double) G * f.bytes / 1e9;
    std::printf("layer %d gate/up type %d down type %d, %d experts x %.2f MB, %d tokens, %d entries: %.1f us per call, "
                "%.0f GB/s%s\n", l, f.gu_type, f.d_type, G, f.bytes / 1e6, T, NE, us, gb / (us * 1e-6),
                e == cudaSuccess ? "" : (std::string(" - CUDA error: ") + cudaGetErrorString(e)).c_str());
    return e == cudaSuccess ? 0 : 1;
}
