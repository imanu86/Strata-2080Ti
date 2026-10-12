// src/kernels/topk_grid_parity.cpp - SPEED4: the decode QSA top-k on 8 ordinary CTAs per query (qsa_block_topk_grid,
// STRATA_TOPK_GRID=1) against qsa_block_topk's one-CTA dispatch and qsa_block_topk_ref, bit for bit (GPU, synthetic,
// no model). The fixtures are decode_cluster_parity's: real-like relu sums and adversarial rows (all equal, a few
// values, NaN, +-inf, signed zeros, denormals, monotone, all NaN, quantized), contexts from below the selection width to
// 262,144 cells, every tail weight, 1/5/8 queries, two capacities. Plus: one CUDA graph replayed with new inputs, many
// back-to-back calls (the arrival counter's reuse), and two streams with their own buffers running at once (the slots).
//
//   topk_grid_parity --selftest    parity (exit 0 = identical everywhere, 1 = a difference, 2 = a CUDA error)
//   topk_grid_parity --bench       per-call times, one-CTA dispatch vs grid, at 32K / 131K / 262K and 1-5 queries
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

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

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T> struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc(&p, count * sizeof(T) + 64), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    void put(const std::vector<T>& h) {
        ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
        ck(cudaDeviceSynchronize(), "up landed");
    }
    std::vector<T> get() const {
        std::vector<T> h(n);
        ck(cudaDeviceSynchronize(), "before down");
        ck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
        return h;
    }
};

// the grid kernel keys its exchange slot on the `ids` pointer (8 per device; the engine's buffers are stable), so every
// case writes into this one persistent buffer and the result is copied out
int32_t* g_grid_ids = nullptr;
int g_declined = 0;   // cases the grid kernel declined (does not fit with a query to spare): nq > 5 only
int32_t* grid_ids(size_t n) {
    static size_t have = 0;
    if (n > have) {
        if (g_grid_ids) { ck(cudaDeviceSynchronize(), "grid ids sync"); cudaFree(g_grid_ids); }
        have = n < (size_t) 8 * 4096 ? (size_t) 8 * 4096 : n;
        ck(cudaMalloc(&g_grid_ids, have * sizeof(int32_t)), "grid ids");
    }
    return g_grid_ids;
}

const float kNaN = std::numeric_limits<float>::quiet_NaN(), kInf = std::numeric_limits<float>::infinity();

enum Pattern { kRandom, kAllEqual, kFewValues, kNaNs, kInfs, kSignedZero, kDenormal, kAscending, kDescending, kAllNaN,
               kQuantized, kPatterns };
const char* pattern_name(int p) {
    static const char* n[] = {"random", "all-equal", "few-values", "nan", "inf", "signed-zero", "denormal", "ascending",
                              "descending", "all-nan", "quantized"};
    return n[p];
}

void fill_row(float* row, int64_t n_kv, int64_t n_bid, int pattern, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> pick(0, 99);
    for (int64_t b = 0; b <= n_bid; ++b) {
        float v = 0.f;
        const float g = nd(rng);
        switch (pattern) {
            case kRandom: v = std::fmax(0.f, g) * 3.f + std::fmax(0.f, nd(rng)); break;
            case kAllEqual: v = 1.f; break;
            case kFewValues: v = (float) (pick(rng) % 4) * 0.5f; break;
            case kNaNs: v = pick(rng) < 15 ? kNaN : std::fabs(g); break;
            case kInfs: {
                const int r = pick(rng);
                v = r < 5 ? kInf : r < 15 ? -kInf : r < 30 ? -std::fabs(g) : g;
                break;
            }
            case kSignedZero: v = pick(rng) < 50 ? -0.0f : (pick(rng) < 50 ? 0.0f : -std::fabs(g) * 1e-3f); break;
            case kDenormal: v = (float) (pick(rng) % 7) * 1e-41f; break;
            case kAscending: v = (float) b; break;
            case kDescending: v = (float) (n_bid - b); break;
            case kAllNaN: v = kNaN; break;
            case kQuantized: v = std::round(std::fabs(g) * 8.f) / 8.f; break;
        }
        row[b] = v;
    }
    if (n_kv % 4 != 0 && pattern != kAllNaN && pattern != kAllEqual) row[n_bid] += 1e9f;
}

struct Inputs {
    std::vector<float> sc;
    std::vector<int32_t> steps;
};

Inputs make_inputs(int64_t ctx, int64_t max_blocks, int nq, int pattern, std::mt19937& rng) {
    const k::QsaShapes s = k::qsa_real_shapes();
    Inputs in;
    in.sc.assign((size_t) (nq * max_blocks), 0.f);
    in.steps.assign((size_t) nq * k::kStepCount, 0);
    for (int i = 0; i < nq; ++i) {
        const int64_t n_kv = ctx - (nq - 1 - i);
        int32_t* st = in.steps.data() + (size_t) i * k::kStepCount;
        st[k::kStepPos] = (int32_t) (n_kv - 1);
        st[k::kStepNKv] = (int32_t) n_kv;
        st[k::kStepNBid] = (int32_t) (n_kv / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(n_kv, s);
        fill_row(in.sc.data() + (size_t) i * max_blocks, n_kv, n_kv / s.idx_block, pattern, rng);
    }
    return in;
}

int64_t count_diff(const std::vector<int32_t>& a, const std::vector<int32_t>& b, int64_t* first = nullptr) {
    int64_t d = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            if (d == 0 && first) *first = (int64_t) i;
            ++d;
        }
    return d;
}

int run_case(int64_t ctx, int64_t capacity, int nq, int pattern, std::mt19937& rng, bool verbose) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / 4 + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const Inputs in = make_inputs(ctx, max_blocks, nq, pattern, rng);
    Dev<float> d_sc(in.sc.size());
    const size_t n_ids = (size_t) (nq * cap);
    Dev<int32_t> d_st(in.steps.size()), d_new(n_ids), d_old(n_ids), d_ref(n_ids);
    d_sc.put(in.sc);
    d_st.put(in.steps);
    ck(cudaMemset(d_new.p, 0xA5, n_ids * 4), "memset");
    ck(cudaMemset(d_old.p, 0xA5, n_ids * 4), "memset");
    ck(cudaMemset(d_ref.p, 0xA5, n_ids * 4), "memset");
    int32_t* gid = grid_ids(n_ids);
    ck(cudaMemset(gid, 0xA5, n_ids * 4), "memset grid");
    const bool ran = k::qsa_block_topk_grid(d_sc.p, d_st.p, nq, max_blocks, cap, s, gid, nullptr);
    if (ran) ck(cudaMemcpy(d_new.p, gid, n_ids * 4, cudaMemcpyDeviceToDevice), "copy grid ids");
    if (!ran) {   // a grid that does not fit the card with a query to spare is declined by design (decode has 1-5 queries)
        if (nq <= 5) {
            std::printf("FAIL ctx %lld cap %lld nq %d: the grid kernel did not run\n", (long long) ctx, (long long) capacity, nq);
            return 1;
        }
        ++g_declined;
        return 0;
    }
    k::qsa_block_topk(d_sc.p, d_st.p, nq, max_blocks, cap, s, d_old.p, nullptr);
    k::qsa_block_topk_ref(d_sc.p, d_st.p, nq, max_blocks, cap, s, d_ref.p, nullptr);
    ck(cudaDeviceSynchronize(), "sync");
    const auto a = d_new.get(), b = d_old.get(), r = d_ref.get();
    int64_t first = -1;
    const int64_t d_old_n = count_diff(a, b, &first), d_ref_n = count_diff(a, r);
    const bool ok = d_old_n == 0 && d_ref_n == 0;
    if (!ok || verbose)
        std::printf("%s ctx %7lld cap %7lld nq %d %-11s: %lld ids differ from the one-CTA dispatch, %lld from the "
                    "reference%s\n", ok ? "ok  " : "FAIL", (long long) ctx, (long long) capacity, nq,
                    pattern_name(pattern), (long long) d_old_n, (long long) d_ref_n,
                    first >= 0 ? (" (first at " + std::to_string(first) + ")").c_str() : "");
    return ok ? 0 : 1;
}

// the grid launch captured in a graph, replayed over new inputs; then 500 back-to-back replays (the counter's reuse)
int run_graph_case(std::mt19937& rng) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int nq = 5;
    const int64_t capacity = 131072 + 272, max_blocks = capacity / 4 + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    Dev<float> d_sc((size_t) (nq * max_blocks));
    Dev<int32_t> d_st((size_t) nq * k::kStepCount), d_new((size_t) (nq * cap)), d_old((size_t) (nq * cap));
    ck(cudaMemset(d_new.p, 0, d_new.n * 4), "memset");
    ck(cudaMemset(d_old.p, 0, d_old.n * 4), "memset");
    cudaStream_t cs;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    ck(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal), "begin capture");
    const bool l1 = k::qsa_block_topk_grid(d_sc.p, d_st.p, nq, max_blocks, cap, s, d_new.p, cs);
    ck(cudaStreamEndCapture(cs, &g), "end capture");
    if (!l1) { std::printf("FAIL graph: the grid kernel did not run\n"); return 1; }
    ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    int fails = 0;
    for (int rep = 0; rep < 6; ++rep) {
        const int64_t ctx = 100000 + rep * 4001 + rep % 4;
        const Inputs in = make_inputs(ctx, max_blocks, nq, rep % 2 ? kFewValues : kRandom, rng);
        d_sc.put(in.sc);
        d_st.put(in.steps);
        const int launches = rep == 5 ? 500 : 1;
        for (int i = 0; i < launches; ++i) ck(cudaGraphLaunch(ge, cs), "graph launch");
        k::qsa_block_topk(d_sc.p, d_st.p, nq, max_blocks, cap, s, d_old.p, cs);
        ck(cudaStreamSynchronize(cs), "graph sync");
        const int64_t d = count_diff(d_new.get(), d_old.get());
        if (d) ++fails;
        std::printf("%s graph replay %d (ctx %lld, %d launch%s): %lld ids differ\n", d ? "FAIL" : "ok  ", rep,
                    (long long) ctx, launches, launches > 1 ? "es" : "", (long long) d);
    }
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaStreamDestroy(cs);
    return fails;
}

// two streams, two buffers (two slots), 300 calls each at once
int run_two_streams(std::mt19937& rng) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int nq = 4;
    const int64_t capacity = 131072 + 272, max_blocks = capacity / 4 + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    int fails = 0;
    Dev<float> sa((size_t) (nq * max_blocks)), sb((size_t) (nq * max_blocks));
    Dev<int32_t> ta((size_t) nq * k::kStepCount), tb((size_t) nq * k::kStepCount);
    Dev<int32_t> ga((size_t) (nq * cap)), gb((size_t) (nq * cap)), ra((size_t) (nq * cap)), rb((size_t) (nq * cap));
    const Inputs ia = make_inputs(120001, max_blocks, nq, kRandom, rng), ib = make_inputs(90003, max_blocks, nq, kQuantized, rng);
    sa.put(ia.sc); ta.put(ia.steps); sb.put(ib.sc); tb.put(ib.steps);
    for (auto* d : {&ga, &gb, &ra, &rb}) ck(cudaMemset(d->p, 0, d->n * 4), "memset");
    cudaStream_t s1, s2;
    ck(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), "stream");
    ck(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking), "stream");
    for (int i = 0; i < 300; ++i) {
        if (!k::qsa_block_topk_grid(sa.p, ta.p, nq, max_blocks, cap, s, ga.p, s1) ||
            !k::qsa_block_topk_grid(sb.p, tb.p, nq, max_blocks, cap, s, gb.p, s2)) {
            std::printf("FAIL two streams: the grid kernel did not run\n");
            return 1;
        }
    }
    k::qsa_block_topk(sa.p, ta.p, nq, max_blocks, cap, s, ra.p, s1);
    k::qsa_block_topk(sb.p, tb.p, nq, max_blocks, cap, s, rb.p, s2);
    ck(cudaDeviceSynchronize(), "two streams sync");
    const int64_t da = count_diff(ga.get(), ra.get()), db = count_diff(gb.get(), rb.get());
    if (da || db) ++fails;
    std::printf("%s two streams x 300 calls: %lld / %lld ids differ\n", da || db ? "FAIL" : "ok  ", (long long) da,
                (long long) db);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    return fails;
}

template <typename F> float time_us(cudaStream_t cs, F&& f, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 5; ++i) f();
    cudaEventRecord(a, cs);
    for (int i = 0; i < reps; ++i) f();
    cudaEventRecord(b, cs);
    ck(cudaEventSynchronize(b), "time");
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms * 1000.f / (float) reps;
}

float median5(float a[5]) {
    for (int i = 0; i < 5; ++i)
        for (int j = i + 1; j < 5; ++j)
            if (a[j] < a[i]) { const float x = a[i]; a[i] = a[j]; a[j] = x; }
    return a[2];
}

int bench() {
    std::mt19937 rng(11);
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    cudaStream_t cs;
    ck(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    int dev = 0;
    cudaDeviceProp prop{};
    ck(cudaGetDevice(&dev), "device");
    ck(cudaGetDeviceProperties(&prop, dev), "props");
    std::printf("# device %d: %s (sm_%d%d, %d SMs); QSA top-k per call (us), median of 5 rounds x 300 calls; capacity = "
                "context + 272\n", dev, prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    std::printf("# ctx nq one-CTA grid speedup\n");
    int bad = 0;
    for (const int64_t ctx : {32768LL, 131072LL, 262144LL - 272}) {
        for (const int nq : {1, 2, 3, 4, 5}) {
            const int64_t capacity = ctx + 272, max_blocks = capacity / 4 + 2;
            const Inputs in = make_inputs(ctx, max_blocks, nq, kRandom, rng);
            Dev<float> d_sc(in.sc.size());
            Dev<int32_t> d_st(in.steps.size()), d_a((size_t) (nq * cap)), d_b((size_t) (nq * cap));
            d_sc.put(in.sc);
            d_st.put(in.steps);
            const float* sp = d_sc.p;
            const int32_t* st = d_st.p;
            int32_t* gb = grid_ids((size_t) (nq * cap));
            float to[5], tg[5];
            for (int r = 0; r < 5; ++r) {
                to[r] = time_us(cs, [&] { k::qsa_block_topk(sp, st, nq, max_blocks, cap, s, d_a.p, cs); }, 300);
                tg[r] = time_us(cs, [&] { k::qsa_block_topk_grid(sp, st, nq, max_blocks, cap, s, gb, cs); }, 300);
            }
            const float a = median5(to), b = median5(tg);
            ck(cudaMemcpy(d_b.p, gb, (size_t) (nq * cap) * 4, cudaMemcpyDeviceToDevice), "copy grid ids");
            const int64_t d = count_diff(d_a.get(), d_b.get());
            bad += d != 0;
            std::printf("%6lld %d %7.1f %6.1f %5.2fx %s\n", (long long) ctx, nq, a, b, a / b, d ? "DIFFERENT" : "identical");
        }
    }
    cudaStreamDestroy(cs);
    std::printf("# topk_grid bench: %s\n", bad ? "DIFFERENT cells" : "ok (all identical)");
    return bad ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const bool do_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const bool verbose = argc > 2 && std::strcmp(argv[2], "-v") == 0;
    if (const char* d = std::getenv("TOPK_GRID_DEVICE")) ck(cudaSetDevice(std::atoi(d)), "set device");
    std::mt19937 rng(20261012);
    int fails = run_graph_case(rng);   // first: the first call of the process happens inside a capture, as in the engine
    if (do_bench) return fails ? 1 : bench();
    fails += run_two_streams(rng);
    int cases = 0;
    const int64_t ctxs[] = {7, 2050, 2051, 2052, 2053, 2054, 2055, 3001, 9000, 32768, 32769, 65538, 100003,
                            131072, 135170, 135171, 200001, 262142, 262143, 262144};
    for (const int64_t ctx : ctxs) {
        for (const int64_t capacity : {ctx + 272, (int64_t) 262144 + 272}) {
            if (capacity < ctx) continue;
            for (int p = 0; p < kPatterns; ++p) {
                const int nq = (p % 3 == 0) ? 1 : (p % 3 == 1 ? 5 : 8);
                if (nq > ctx) continue;
                fails += run_case(ctx, capacity, nq, p, rng, verbose);
                ++cases;
            }
        }
    }
    std::printf("topk_grid_parity: %d cases + graph + two streams, %d failed, %d declined (nq 8 that does not fit)\n", cases,
                fails, g_declined);
    return fails ? 1 : 0;
}
