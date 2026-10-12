// src/kernels/chain_step_bench.cpp - chainfuse (docs/CHAINFUSE.md): the MTP drafter's chain step, segment by segment,
// default kernel sequence against the STRATA_CHAIN_FUSE2 fused kernels - bit for bit (memcmp of every output the step
// goes on to read) and timed (CUDA events, B-A-B: fused, default, fused).  GPU, synthetic data at the drafter's real
// shapes (n_embd 2560, hc 4, 24 query heads x 256, 2 KV heads, 32k-cell window, 48,542-token draft head), no model.
//
// Segments (the step's 54 kernels and their microseconds on the 2080 Ti come from nsys, DET-cp-nsys5):
//   attn    attn_chunk_kernel_pre75 (240 us) + attn_merge_v2_kernel (50 us)  vs  attn_chunk_kernel_stage (4 variants) + attn_merge_v3
//   norm    norm + native_quantize_q8_1 (2560 and 4 x 2560 values)           vs  norm_q8_1
//   gate    gate + native_quantize_q8_1 (6144 values)                       vs  gate_q8_1
//   tail    sampler_greedy / argmax_rows + row_top_prob_split + map_ids + mtp_select + force_token (~34 us)  vs  mtp_head_tail
//   post    combine_k10_vec4 + post                                          vs  combine_k10_post
//   graph   (--graph) a captured graph of 50 one-block kernels with and without a fork / join to a side stream: the cost
//           of a cross-stream edge inside a graph on this driver (the step shows a 33 us bubble after its fork / join)
//
//   chain_step_bench [--device N] [--iters 200] [--attn 1|2|3|4] [--graph]   exit 0: every segment bit-identical (77: no GPU)
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_gr_postops.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
cudaStream_t st = nullptr;   // every segment runs and is timed on this stream (the native kernels refuse a null one)
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "chain_step_bench: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return p;
}
template <typename T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
template <typename T> std::vector<T> download(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
template <typename T> bool same(const T* a, const T* b, size_t n, const char* what) {
    const std::vector<T> ha = download(a, n), hb = download(b, n);
    if (std::memcmp(ha.data(), hb.data(), n * sizeof(T)) == 0) return true;
    size_t first = 0;
    while (first < n && std::memcmp(&ha[first], &hb[first], sizeof(T)) == 0) ++first;
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) count += std::memcmp(&ha[i], &hb[i], sizeof(T)) != 0;
    std::printf("  FAIL %s: %zu of %zu elements differ, first at %zu\n", what, count, n, first);
    return false;
}
uint16_t f16_value(uint32_t r) { return (uint16_t) ((r & 0x8000u) | ((11u + (r >> 16) % 5u) << 10) | (r & 0x3FFu)); }
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }

struct Timer {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    Timer() { ck(cudaEventCreate(&e0), "event"); ck(cudaEventCreate(&e1), "event"); }
    ~Timer() { cudaEventDestroy(e0); cudaEventDestroy(e1); }
    // microseconds per call over `iters` back-to-back calls, warmed once
    template <typename F> double us(F&& f, int iters) {
        for (int i = 0; i < 8; ++i) f();
        ck(cudaDeviceSynchronize(), "warm");
        ck(cudaEventRecord(e0, st), "record");
        for (int i = 0; i < iters; ++i) f();
        ck(cudaEventRecord(e1, st), "record");
        ck(cudaEventSynchronize(e1), "sync");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return 1000.0 * ms / iters;
    }
};
// B-A-B: fused, default, fused; prints the three and the mean ratio
template <typename FB, typename FA>
void bab(Timer& t, const char* what, FB&& fused, FA&& dflt, int iters, double& sum_a, double& sum_b) {
    const double b1 = t.us(fused, iters), a = t.us(dflt, iters), b2 = t.us(fused, iters);
    const double b = 0.5 * (b1 + b2);
    sum_a += a;
    sum_b += b;
    std::printf("  %-34s default %8.1f us | fused %8.1f us (%8.1f, %8.1f) | fused/default %.3f\n", what, a, b, b1, b2,
                a > 0 ? b / a : 0.0);
}

constexpr int N = 2560, HC = 4, NH = 24, HD = 256, NKV = 2, NVOCAB = 48542, K10 = 10;
constexpr int64_t WINDOW = 32768;

// ---------------------------------------------------------------- attention: pre75 + v2 against mlp + v3
bool seg_attn(std::mt19937& rng, Timer& t, int iters, int attn_variant, double& sa, double& sb) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t PS = s.page_size, H = s.n_head_kv, D = s.head_dim;
    const int64_t cells = WINDOW + 1024, pages = cells / PS, rows = pages * H * PS;
    const int64_t cap = WINDOW;
    bool ok = true;
    // INT8 (the Daily's --kv int8) and FP16 pools over the same rows
    std::vector<int8_t> c8((size_t) (rows * D));
    for (auto& x : c8) x = (int8_t) ((int) (rng() % 255u) - 127);
    int8_t* k8 = upload(c8);
    for (auto& x : c8) x = (int8_t) ((int) (rng() % 255u) - 127);
    int8_t* v8 = upload(c8);
    std::vector<uint16_t> sc((size_t) (rows * (D / k::KV_Q8_GROUP)));
    for (auto& x : sc) x = f16_scale_q8(rng());
    uint16_t* k8s = upload(sc);
    for (auto& x : sc) x = f16_scale_q8(rng());
    uint16_t* v8s = upload(sc);
    std::vector<uint16_t> h16((size_t) (rows * D));
    for (auto& x : h16) x = f16_value(rng());
    uint16_t* k16 = upload(h16);
    for (auto& x : h16) x = f16_value(rng());
    uint16_t* v16 = upload(h16);
    // page tables: every page resident (the drafter's ring), and one with a run of 64 unresident pages and scattered ones
    std::vector<int32_t> tab((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
    std::shuffle(tab.begin(), tab.end(), rng);
    int32_t* table_full = upload(tab);
    for (int64_t i = 2048 / PS; i < (2048 + 256) / PS; ++i) tab[(size_t) i] = -1;
    for (int64_t i = 0; i < pages; ++i) if (rng() % 64u == 0) tab[(size_t) i] = -1;
    int32_t* table_holes = upload(tab);
    float* q = dalloc<float>((size_t) (NH * D));
    int32_t* ids = dalloc<int32_t>((size_t) cap);
    int32_t* steps = dalloc<int32_t>((size_t) k::kStepCount);
    const size_t scratch_floats = (size_t) k::qsa_decode_attn_scratch_floats(cap, s);
    float* scratch_a = dalloc<float>(scratch_floats);
    float* scratch_b = dalloc<float>(scratch_floats);
    float* attn_a = dalloc<float>((size_t) (NH * D));
    float* attn_b = dalloc<float>((size_t) (NH * D));
    const int n_chunks = (int) ((cap + 63) / 64);
    const size_t used_floats = (size_t) n_chunks * NH * (HD + 2);   // the partials the chunk pass writes
    std::normal_distribution<float> nd(0.0f, 1.0f);
    {
        std::vector<int32_t> hid((size_t) cap);
        for (int64_t j = 0; j < cap; ++j) hid[(size_t) j] = (int32_t) j;   // window_ids: consecutive cells
        ck(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids");
    }
    auto pools = [&](int fmt, const int32_t* tabp) {
        k::QsaAttnPools p;
        p.page_table = tabp;
        if (fmt == 1) { p.k_q = k8; p.v_q = v8; p.k_scale = k8s; p.v_scale = v8s; }
        else { p.k_pool = k16; p.v_pool = v16; }
        return p;
    };
    auto set_case = [&](int width) {
        std::vector<float> hq((size_t) (NH * D));
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        std::vector<int32_t> hst((size_t) k::kStepCount, 0);
        hst[k::kStepWidth] = width;
        hst[k::kStepPos] = width;
        hst[1] = width;
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
    };
    const int widths[] = {1, 63, 64, 65, 1000, 2051, 16384, 31000, 32767, 32768};
    long long cases = 0, failed = 0;
    for (int fmt : {1, 0}) {
        for (int holes = 0; holes < 2; ++holes) {
            const int32_t* tabp = holes ? table_holes : table_full;
            for (int width : widths) {
                set_case(width);
                ck(cudaMemset(scratch_a, 0xFF, scratch_floats * 4), "scratch NaN");
                ck(cudaMemset(scratch_b, 0xFF, scratch_floats * 4), "scratch NaN");
                k::qsa_decode_attn_chunks_only(q, pools(fmt, tabp), ids, steps, cap, s, scratch_a, 1, 1, st);
                k::qsa_decode_attn_merge_only(steps, cap, s, scratch_a, attn_a, 1, 1, st);
                for (int kern : {2, 3, 4, 5}) {
                    ck(cudaMemset(scratch_b, 0xFF, scratch_floats * 4), "scratch NaN");
                    ck(cudaMemset(attn_b, 0x7F, (size_t) NH * D * 4), "attn fill");
                    k::qsa_decode_attn_chunks_only(q, pools(fmt, tabp), ids, steps, cap, s, scratch_b, 1, kern, st);
                    k::qsa_decode_attn_merge_only(steps, cap, s, scratch_b, attn_b, 1, 2, st);
                    ck(cudaDeviceSynchronize(), "attn sync");
                    char what[128];
                    std::snprintf(what, sizeof what, "attn partials fmt %s holes %d width %d variant %d", fmt ? "int8" : "fp16",
                                  holes, width, kern - 1);
                    bool c_ok = same(scratch_a, scratch_b, used_floats, what);
                    std::snprintf(what, sizeof what, "attn output fmt %s holes %d width %d variant %d", fmt ? "int8" : "fp16",
                                  holes, width, kern - 1);
                    c_ok = same(attn_a, attn_b, (size_t) NH * D, what) && c_ok;
                    // the merge alone on the reference partials, and the production fuse2 entry
                    ck(cudaMemset(attn_b, 0x7F, (size_t) NH * D * 4), "attn fill");
                    k::qsa_decode_attn_merge_only(steps, cap, s, scratch_a, attn_b, 1, 2, st);
                    ck(cudaDeviceSynchronize(), "merge sync");
                    std::snprintf(what, sizeof what, "attn merge v3 on pre75 partials fmt %s holes %d width %d",
                                  fmt ? "int8" : "fp16", holes, width);
                    c_ok = same(attn_a, attn_b, (size_t) NH * D, what) && c_ok;
                    ck(cudaMemset(attn_b, 0x7F, (size_t) NH * D * 4), "attn fill");
                    k::qsa_decode_attn_batch_fuse2(q, pools(fmt, tabp), ids, steps, cap, s, scratch_b, attn_b, 1,
                                                   kern - 1, 1, st);
                    ck(cudaDeviceSynchronize(), "fuse2 sync");
                    std::snprintf(what, sizeof what, "attn batch_fuse2 fmt %s holes %d width %d variant %d",
                                  fmt ? "int8" : "fp16", holes, width, kern - 1);
                    c_ok = same(attn_a, attn_b, (size_t) NH * D, what) && c_ok;
                    ++cases;
                    if (!c_ok) ++failed;
                }
            }
        }
    }
    std::printf("attn: %lld cases, %lld differ -> %s\n", cases, failed, failed ? "FAIL" : "bit-identical");
    ok = failed == 0;
    // timing at the drafter's shape: INT8, every page resident, 32768 cells
    set_case((int) WINDOW);
    const k::QsaAttnPools p8 = pools(1, table_full);
    std::printf("attn timing (int8, 32768 cells, one query):\n");
    for (int kk : {2, 3, 4, 5}) {
        char what[64];
        std::snprintf(what, sizeof what, "chunk pass stage variant %d vs pre75", kk - 1);
        double da = 0, db = 0;
        bab(t, what,
            [&] { k::qsa_decode_attn_chunks_only(q, p8, ids, steps, cap, s, scratch_b, 1, kk, st); },
            [&] { k::qsa_decode_attn_chunks_only(q, p8, ids, steps, cap, s, scratch_a, 1, 1, st); }, iters, da, db);
    }
    {
        double da = 0, db = 0;
        bab(t, "merge v3 vs v2",
            [&] { k::qsa_decode_attn_merge_only(steps, cap, s, scratch_a, attn_b, 1, 2, st); },
            [&] { k::qsa_decode_attn_merge_only(steps, cap, s, scratch_a, attn_a, 1, 1, st); }, iters, da, db);
    }
    char what[64];
    std::snprintf(what, sizeof what, "attention segment (variant %d + merge v3)", attn_variant);
    bab(t, what,
        [&] { k::qsa_decode_attn_batch_fuse2(q, p8, ids, steps, cap, s, scratch_b, attn_b, 1, attn_variant, 1, st); },
        [&] {
            k::qsa_decode_attn_chunks_only(q, p8, ids, steps, cap, s, scratch_a, 1, 1, st);
            k::qsa_decode_attn_merge_only(steps, cap, s, scratch_a, attn_a, 1, 1, st);
        },
        iters, sa, sb);
    for (void* p : {(void*) k8, (void*) v8, (void*) k8s, (void*) v8s, (void*) k16, (void*) v16, (void*) table_full,
                    (void*) table_holes, (void*) q, (void*) ids, (void*) steps, (void*) scratch_a, (void*) scratch_b,
                    (void*) attn_a, (void*) attn_b})
        ck(cudaFree(p), "free");
    return ok;
}

// ---------------------------------------------------------------- norm + q8_1
bool seg_norm(std::mt19937& rng, Timer& t, int iters, double& sa, double& sb) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    bool ok = true;
    for (int cols : {N, HC * N}) {
        std::vector<float> hx((size_t) cols), hg((size_t) cols);
        for (auto& x : hx) x = nd(rng) * (rng() % 97 == 0 ? 40.0f : 1.0f);   // a few outliers: full int8 range
        for (auto& g : hg) g = 1.0f + 0.1f * nd(rng);
        float* x = upload(hx);
        float* g = upload(hg);
        float* y = dalloc<float>((size_t) cols);
        const size_t qbytes = k::native_q8_1_bytes(cols, 1);
        uint8_t* qa = dalloc<uint8_t>(qbytes);
        uint8_t* qb = dalloc<uint8_t>(qbytes);
        ck(cudaMemset(qb, 0xAB, qbytes), "fill");
        k::native_qsa_rms_norm_weighted(x, g, y, cols, 1, 1e-6f, st);
        k::native_quantize_q8_1(y, qa, cols, 1, st);
        k::native_qsa_rms_norm_q8_1(x, g, nullptr, qb, cols, 1, 1e-6f, st);
        ck(cudaDeviceSynchronize(), "norm sync");
        char what[64];
        std::snprintf(what, sizeof what, "norm+q8_1 %d values", cols);
        ok = same(qa, qb, qbytes, what) && ok;
        bab(t, what, [&] { k::native_qsa_rms_norm_q8_1(x, g, nullptr, qb, cols, 1, 1e-6f, st); },
            [&] {
                k::native_qsa_rms_norm_weighted(x, g, y, cols, 1, 1e-6f, st);
                k::native_quantize_q8_1(y, qa, cols, 1, st);
            },
            iters, sa, sb);
        for (void* p : {(void*) x, (void*) g, (void*) y, (void*) qa, (void*) qb}) ck(cudaFree(p), "free");
    }
    return ok;
}

// ---------------------------------------------------------------- gate + q8_1
bool seg_gate(std::mt19937& rng, Timer& t, int iters, double& sa, double& sb) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> ha((size_t) NH * HD), hq((size_t) NH * 2 * HD);
    for (auto& x : ha) x = nd(rng);
    for (auto& x : hq) x = 3.0f * nd(rng);
    float* attn = upload(ha);
    float* qfull = upload(hq);
    float* out = dalloc<float>((size_t) NH * HD);
    const size_t qbytes = k::native_q8_1_bytes(NH * HD, 1);
    uint8_t* qa = dalloc<uint8_t>(qbytes);
    uint8_t* qb = dalloc<uint8_t>(qbytes);
    ck(cudaMemset(qb, 0xAB, qbytes), "fill");
    k::native_qsa_gate_apply(attn, qfull, out, NH, HD, st);
    k::native_quantize_q8_1(out, qa, NH * HD, 1, st);
    k::native_qsa_gate_q8_1(attn, qfull, nullptr, qb, NH, HD, st);
    ck(cudaDeviceSynchronize(), "gate sync");
    const bool ok = same(qa, qb, qbytes, "gate+q8_1");
    bab(t, "gate+q8_1 (6144 values)", [&] { k::native_qsa_gate_q8_1(attn, qfull, nullptr, qb, NH, HD, st); },
        [&] {
            k::native_qsa_gate_apply(attn, qfull, out, NH, HD, st);
            k::native_quantize_q8_1(out, qa, NH * HD, 1, st);
        },
        iters, sa, sb);
    for (void* p : {(void*) attn, (void*) qfull, (void*) out, (void*) qa, (void*) qb}) ck(cudaFree(p), "free");
    return ok;
}

// ---------------------------------------------------------------- the head tail
bool seg_tail(std::mt19937& rng, Timer& t, int iters, double& sa, double& sb) {
    std::normal_distribution<float> nd(0.0f, 2.0f);
    const int64_t stride = (int64_t) HC * N;
    float* logits = dalloc<float>((size_t) NVOCAB);
    std::vector<int32_t> htab((size_t) NVOCAB);
    for (int i = 0; i < NVOCAB; ++i) htab[(size_t) i] = (int32_t) (rng() % 248320u);
    int32_t* table = upload(htab);
    std::vector<float> hr((size_t) stride);
    for (auto& x : hr) x = nd(rng);
    float* R_src = upload(hr);
    int32_t* row_dev = dalloc<int32_t>(2);   // {a, 0}: the chain reads row 0
    float* R_a = dalloc<float>((size_t) stride);
    float* R_b = dalloc<float>((size_t) stride);
    int32_t* ids_a = dalloc<int32_t>(4);
    int32_t* ids_b = dalloc<int32_t>(4);
    float* pr_a = dalloc<float>(4);
    float* pr_b = dalloc<float>(4);
    int32_t* tok_a = dalloc<int32_t>(4);
    int32_t* tok_b = dalloc<int32_t>(4);
    int32_t* out_a = dalloc<int32_t>(16);
    int32_t* out_b = dalloc<int32_t>(16);
    float* op_a = dalloc<float>(16);
    float* op_b = dalloc<float>(16);
    std::vector<int32_t> hforce(16, -1);
    hforce[2] = 777;   // step 2 forced
    int32_t* force = upload(hforce);
    uint8_t* top_scratch = dalloc<uint8_t>((size_t) k::row_top_prob_scratch_bytes(1));
    uint8_t* arg_scratch = dalloc<uint8_t>((size_t) k::argmax_rows_scratch_bytes(1));
    uint8_t* tail_scratch = dalloc<uint8_t>((size_t) std::max<uint64_t>(64, k::mtp_head_tail_scratch_bytes()));
    k::SamplerParams sp;
    sp.greedy = true;
    sp.temperature = 0.0f;
    auto dflt = [&](int j, bool argmax_rows_pick) {
        if (argmax_rows_pick) k::argmax_rows(logits, 1, NVOCAB, arg_scratch, ids_a, st);
        else k::sample_tokens(logits, 1, NVOCAB, nullptr, 0, sp, ids_a, st);
        k::row_top_prob_split(logits, 1, NVOCAB, ids_a, pr_a, top_scratch, st);
        k::map_ids(ids_a, table, 1, st);
        k::mtp_select(R_src, stride, ids_a, row_dev + 1, R_a, tok_a, out_a, j, nullptr, pr_a, op_a, force);
    };
    auto fused = [&](int j) {
        k::mtp_head_tail(logits, NVOCAB, table, R_src, stride, row_dev + 1, R_b, ids_b, pr_b, tok_b, out_b, j, op_b, force,
                         tail_scratch, st);
    };
    bool ok = true;
    for (int c = 0; c < 12; ++c) {
        std::vector<float> hl((size_t) NVOCAB);
        for (auto& x : hl) x = nd(rng);
        if (c == 1) hl[12345] = hl[777] = 50.0f;                                   // a tie: the lower index wins
        if (c == 2) for (int i = 0; i < 100; ++i) hl[(size_t) (rng() % NVOCAB)] = std::numeric_limits<float>::quiet_NaN();
        if (c == 3) std::fill(hl.begin(), hl.end(), -std::numeric_limits<float>::infinity());   // none: 0
        if (c == 4) hl[NVOCAB - 1] = 90.0f;                                          // the last one
        if (c == 5) hl[0] = 90.0f;
        if (c == 6) std::fill(hl.begin(), hl.end(), 1.0f);                           // all equal: index 0
        ck(cudaMemcpy(logits, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice), "logits");
        for (int j : {0, 2}) {
            for (int argmax_rows_pick = 0; argmax_rows_pick < 2; ++argmax_rows_pick) {
                ck(cudaMemset(R_b, 0xAB, (size_t) stride * 4), "fill");
                ck(cudaMemset(out_b, 0xAB, 64), "fill");
                ck(cudaMemset(op_b, 0xAB, 64), "fill");
                dflt(j, argmax_rows_pick != 0);
                fused(j);
                ck(cudaDeviceSynchronize(), "tail sync");
                char what[96];
                std::snprintf(what, sizeof what, "tail case %d j %d (%s) R", c, j, argmax_rows_pick ? "argmax_rows" : "sampler");
                bool c_ok = same(R_a, R_b, (size_t) stride, what);
                std::snprintf(what, sizeof what, "tail case %d j %d ids/probs", c, j);
                c_ok = same(ids_a, ids_b, 1, what) && c_ok;
                c_ok = same(pr_a, pr_b, 1, what) && c_ok;
                std::snprintf(what, sizeof what, "tail case %d j %d tok", c, j);
                c_ok = same(tok_a, tok_b, 1, what) && c_ok;
                std::snprintf(what, sizeof what, "tail case %d j %d out[j]", c, j);
                c_ok = same(out_a + j, out_b + j, 1, what) && c_ok;
                std::snprintf(what, sizeof what, "tail case %d j %d out_p[j]", c, j);
                c_ok = same(op_a + j, op_b + j, 1, what) && c_ok;
                ok = ok && c_ok;
            }
        }
    }
    std::printf("tail: %s\n", ok ? "bit-identical" : "FAIL");
    {
        std::vector<float> hl((size_t) NVOCAB);
        for (auto& x : hl) x = nd(rng);
        ck(cudaMemcpy(logits, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice), "logits");
    }
    bab(t, "head tail (5 launches -> 1)", [&] { fused(1); }, [&] { dflt(1, false); }, iters, sa, sb);
    {
        double da = 0, db = 0;
        bab(t, "  (default with argmax_rows)", [&] { fused(1); }, [&] { dflt(1, true); }, iters, da, db);
    }
    for (void* p : {(void*) logits, (void*) table, (void*) R_src, (void*) row_dev, (void*) R_a, (void*) R_b, (void*) ids_a,
                    (void*) ids_b, (void*) pr_a, (void*) pr_b, (void*) tok_a, (void*) tok_b, (void*) out_a, (void*) out_b,
                    (void*) op_a, (void*) op_b, (void*) force, (void*) top_scratch, (void*) arg_scratch,
                    (void*) tail_scratch})
        ck(cudaFree(p), "free");
    return ok;
}

// ---------------------------------------------------------------- combine + post
bool seg_post(std::mt19937& rng, Timer& t, int iters, double& sa, double& sb) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hp((size_t) K10 * N), hw((size_t) K10), hs((size_t) N), hr((size_t) HC * N), hi((size_t) HC);
    for (auto& x : hp) x = nd(rng);
    float wsum = 0.0f;
    for (auto& x : hw) { x = std::fabs(nd(rng)) + 0.01f; wsum += x; }
    for (auto& x : hw) x /= wsum;
    for (auto& x : hs) x = nd(rng);
    for (auto& x : hr) x = nd(rng);
    for (auto& x : hi) x = nd(rng);
    float* parts = upload(hp);
    float* w = upload(hw);
    float* shared = upload(hs);
    float* R0 = upload(hr);
    float* inj = upload(hi);
    float* R_a = dalloc<float>((size_t) HC * N);
    float* R_b = dalloc<float>((size_t) HC * N);
    float* y_a = dalloc<float>((size_t) N);
    float* y_b = dalloc<float>((size_t) N);
    auto reset = [&] {
        ck(cudaMemcpy(R_a, R0, (size_t) HC * N * 4, cudaMemcpyDeviceToDevice), "reset");
        ck(cudaMemcpy(R_b, R0, (size_t) HC * N * 4, cudaMemcpyDeviceToDevice), "reset");
    };
    reset();
    k::native_moe_combine(parts, w, shared, y_a, N, K10, st);
    k::native_gr_post(R_a, y_a, inj, R_a, N, HC, st);
    k::native_moe_combine_k10_post(parts, w, shared, y_b, R_b, inj, R_b, N, HC, st);
    ck(cudaDeviceSynchronize(), "post sync");
    bool ok = same(y_a, y_b, (size_t) N, "combine y");
    ok = same(R_a, R_b, (size_t) HC * N, "combine+post R") && ok;
    // timing: in place, as the step does (R drifts; only the time matters here)
    bab(t, "combine+post (2 launches -> 1)",
        [&] { k::native_moe_combine_k10_post(parts, w, shared, y_b, R_b, inj, R_b, N, HC, st); },
        [&] {
            k::native_moe_combine(parts, w, shared, y_a, N, K10, st);
            k::native_gr_post(R_a, y_a, inj, R_a, N, HC, st);
        },
        iters, sa, sb);
    for (void* p : {(void*) parts, (void*) w, (void*) shared, (void*) R0, (void*) inj, (void*) R_a, (void*) R_b, (void*) y_a,
                    (void*) y_b})
        ck(cudaFree(p), "free");
    return ok;
}

// ---------------------------------------------------------------- a graph's cross-stream edge
// 50 one-block map_ids launches captured into a graph, straight; then the same with launches 30..35 on a side stream
// forked after launch 29 and joined before launch 42 (the drafter's step forks its shared expert the same way).  The
// difference is what a fork / join costs inside a graph on this driver (the step shows a 33 us bubble after its join).
void seg_graph(Timer& t, int iters) {
    int32_t* ids = dalloc<int32_t>(64);
    int32_t* table = dalloc<int32_t>(64);
    cudaStream_t cs = nullptr, side = nullptr;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    ck(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking), "stream");
    cudaEvent_t fork = nullptr, join = nullptr;
    ck(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming), "event");
    ck(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "event");
    auto capture = [&](bool branch) {
        cudaGraph_t g = nullptr;
        cudaGraphExec_t e = nullptr;
        ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
        for (int i = 0; i < 50; ++i) {
            if (branch && i == 30) {
                ck(cudaEventRecord(fork, cs), "fork");
                ck(cudaStreamWaitEvent(side, fork, 0), "fork wait");
                for (int j = 0; j < 6; ++j) k::map_ids(ids + 32, table, 8, side);
                ck(cudaEventRecord(join, side), "join");
            }
            if (branch && i == 42) ck(cudaStreamWaitEvent(cs, join, 0), "join wait");
            k::map_ids(ids, table, 8, cs);
        }
        ck(cudaStreamEndCapture(cs, &g), "end capture");
        ck(cudaGraphInstantiate(&e, g, 0), "instantiate");
        ck(cudaGraphDestroy(g), "graph destroy");
        ck(cudaGraphUpload(e, cs), "upload");
        return e;
    };
    cudaGraphExec_t straight = capture(false), forked = capture(true);
    // one launch at a time, synchronized: the graph's own span, not the queue's throughput
    auto one = [&](cudaGraphExec_t e) {
        ck(cudaGraphLaunch(e, cs), "graph launch");
        ck(cudaStreamSynchronize(cs), "graph sync");
    };
    for (int i = 0; i < 8; ++i) { one(straight); one(forked); }
    double us_s[3] = {}, us_f[3] = {};
    for (int arm = 0; arm < 3; ++arm) {
        cudaGraphExec_t e = arm == 1 ? straight : forked;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) one(e);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        (arm == 1 ? us_s : us_f)[arm] = us;
    }
    std::printf("  graph of 50 one-block kernels, launch+sync: straight %8.1f us | with a fork/join %8.1f / %8.1f us "
                "(difference = the cross-stream edge's cost inside a graph)\n", us_s[1], us_f[0], us_f[2]);
    (void) t;
    ck(cudaGraphExecDestroy(straight), "destroy");
    ck(cudaGraphExecDestroy(forked), "destroy");
    ck(cudaEventDestroy(fork), "destroy");
    ck(cudaEventDestroy(join), "destroy");
    ck(cudaStreamDestroy(cs), "destroy");
    ck(cudaStreamDestroy(side), "destroy");
    ck(cudaFree(ids), "free");
    ck(cudaFree(table), "free");
}
}  // namespace

int main(int argc, char** argv) {
    int device = 0, iters = 200, attn_variant = 1;   // 1: the only variant without spills on sm_75 (cuobjdump)
    bool graph = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--attn") == 0 && i + 1 < argc) attn_variant = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--graph") == 0) graph = true;
        else {
            std::fprintf(stderr, "usage: chain_step_bench [--device N] [--iters 200] [--attn 1|2|3|4] [--graph]\n");
            return 2;
        }
    }
    if (attn_variant < 1 || attn_variant > 4) { std::fprintf(stderr, "--attn 1..4\n"); return 2; }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("SKIP: no GPU");
        return 77;
    }
    ck(cudaSetDevice(device), "cudaSetDevice");
    ck(cudaStreamCreate(&st), "stream");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::printf("chain_step_bench on device %d (%s, %d.%d), %d iterations, attention variant %d\n", device, prop.name,
                prop.major, prop.minor, iters, attn_variant);
    if (!k::qsa_decode_attn_fuse2_built()) std::puts("note: the chainfuse attention kernels are not built here");
    std::mt19937 rng(20261011u);
    Timer t;
    double sum_a = 0.0, sum_b = 0.0;
    bool ok = true;
    std::puts("-- attention (chunk pass + merge)");
    ok = seg_attn(rng, t, iters, attn_variant, sum_a, sum_b) && ok;
    std::puts("-- norm + q8_1");
    ok = seg_norm(rng, t, iters, sum_a, sum_b) && ok;
    std::puts("-- gate + q8_1");
    ok = seg_gate(rng, t, iters, sum_a, sum_b) && ok;
    std::puts("-- head tail");
    ok = seg_tail(rng, t, iters, sum_a, sum_b) && ok;
    std::puts("-- combine + post");
    ok = seg_post(rng, t, iters, sum_a, sum_b) && ok;
    std::printf("fused segments: default %.1f us, fused %.1f us, saved %.1f us a step (the step was ~930 us on the 2080 Ti;"
                " the kernels the fusions leave alone are ~%.0f us of it)\n", sum_a, sum_b, sum_a - sum_b, 930.0 - sum_a);
    if (graph) {
        std::puts("-- graph fork/join cost");
        seg_graph(t, iters);
    }
    std::printf("chain_step_bench: %s\n", ok ? "PASS (every segment bit-identical)" : "FAIL");
    return ok ? 0 : 1;
}
