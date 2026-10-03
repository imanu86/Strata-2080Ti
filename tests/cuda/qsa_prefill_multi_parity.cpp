// Laboratory driver: compare the complete initialized score/selection buffers between
// separate off/on processes of one executable. Synthetic kernel evidence, not model quality.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace k = strata::kernels;
void ck(cudaError_t e, const char* stage) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(e));
}
template<class T> T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T)), "allocate");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
float value(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return (static_cast<int>(state & 65535) - 32768) * (1.f / 65536.f);
}
template<class F> float timed(F fn, int reps) {
    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "event begin"); ck(cudaEventCreate(&end), "event end");
    ck(cudaEventRecord(begin), "record begin");
    for (int i = 0; i < reps; ++i) fn();
    ck(cudaEventRecord(end), "record end"); ck(cudaEventSynchronize(end), "sync event");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, begin, end), "elapsed");
    ck(cudaEventDestroy(begin), "destroy begin"); ck(cudaEventDestroy(end), "destroy end");
    return ms / reps;
}
int main(int argc, char** argv) try {
    if (argc != 8) throw std::runtime_error("usage: driver context queries reps capacity output.bin pattern active");
    const int64_t context = std::atoll(argv[1]), nq = std::atoll(argv[2]);
    const int reps = std::atoi(argv[3]);
    const int64_t capacity = std::atoll(argv[4]);
    const int pattern = std::atoi(argv[6]), active_enabled = std::atoi(argv[7]);
    if (context <= 0 || nq <= 0 || nq > context || nq > 8192 || context > capacity ||
        capacity > 262144 || reps < 1 || reps > 20 || pattern < 0 || pattern > 2)
        throw std::runtime_error("invalid bounded test dimensions");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "device");
    if (prop.major != 7 || prop.minor != 5) throw std::runtime_error("this local test requires SM75");
    const k::QsaShapes shape = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / 4 + 2;
    const int64_t cap = k::qsa_selection_width(k::kTopkMaxCells, shape);
    uint32_t seed = 1729;
    std::vector<float> pooled(static_cast<size_t>(max_blocks * 128)), dead(128), query(static_cast<size_t>(nq * 512));
    for (auto& x : pooled) x = value(seed);
    for (auto& x : dead) x = value(seed);
    for (auto& x : query) x = value(seed);
    if (pattern == 1) { // Many exact ties and signed zeros.
        for (size_t i = 0; i < pooled.size(); ++i) pooled[i] = i % 256 < 128 ? 0.f : -0.f;
        std::fill(dead.begin(), dead.end(), 0.f);
    } else if (pattern == 2) {
        for (size_t i = 0; i < pooled.size(); i += 257) pooled[i] = std::numeric_limits<float>::quiet_NaN();
        for (size_t i = 31; i < pooled.size(); i += 509) pooled[i] = std::numeric_limits<float>::infinity();
        for (size_t i = 71; i < query.size(); i += 193) query[i] = -0.f;
    }
    std::vector<int32_t> steps(static_cast<size_t>(nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        auto* st = steps.data() + i * k::kStepCount;
        const int64_t nkv = context - nq + i + 1;
        st[k::kStepPos] = static_cast<int32_t>(nkv - 1);
        st[k::kStepNKv] = static_cast<int32_t>(nkv);
        st[k::kStepNBid] = static_cast<int32_t>(nkv / shape.idx_block);
        st[k::kStepWidth] = static_cast<int32_t>(k::qsa_selection_width(nkv, shape));
    }
    const int64_t active = active_enabled ? steps[(nq - 1) * k::kStepCount + k::kStepNBid] + 1 : -1;
    const float sentinel = -1234567.25f;
    std::vector<float> scores(static_cast<size_t>(nq * max_blocks), sentinel);
    std::vector<int32_t> ids(static_cast<size_t>(nq * cap), -1234567);
    auto* dp = upload(pooled); auto* dd = upload(dead); auto* dq = upload(query);
    auto* ds = upload(steps); auto* dc = upload(scores); auto* di = upload(ids);
    auto score = [&] { k::qsa_block_scores(dp, dd, dq, ds, nq, max_blocks, shape, dc, nullptr, active); };
    auto select = [&] { k::qsa_block_topk(dc, ds, nq, max_blocks, cap, shape, di, nullptr, active); };
    score(); select(); ck(cudaDeviceSynchronize(), "warmup");
    const float score_ms = timed(score, reps), topk_ms = timed(select, reps);
    ck(cudaMemcpy(scores.data(), dc, scores.size() * sizeof(float), cudaMemcpyDeviceToHost), "download scores");
    ck(cudaMemcpy(ids.data(), di, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "download IDs");
    for (int64_t i = 0; i < nq; ++i) {
        const auto* st = steps.data() + i * k::kStepCount;
        for (int64_t j = st[k::kStepNBid] + 1; j < max_blocks; ++j)
            if (scores[i * max_blocks + j] != sentinel) throw std::runtime_error("score row padding changed");
        for (int64_t j = 0; j < st[k::kStepWidth]; ++j) {
            const int id = ids[i * cap + j];
            if (id < 0 || id >= st[k::kStepNKv] || (j && id <= ids[i * cap + j - 1]))
                throw std::runtime_error("invalid selection order or bounds");
        }
        for (int64_t j = st[k::kStepWidth]; j < cap; ++j)
            if (ids[i * cap + j] != -1234567) throw std::runtime_error("selection row padding changed");
    }
    FILE* out = std::fopen(argv[5], "wb");
    if (!out) throw std::runtime_error("cannot open output");
    const int64_t header[] = {context, nq, capacity, max_blocks, cap, pattern, active};
    bool ok = std::fwrite(header, sizeof(header), 1, out) == 1;
    ok = std::fwrite(scores.data(), sizeof(float), scores.size(), out) == scores.size() && ok;
    ok = std::fwrite(ids.data(), sizeof(int32_t), ids.size(), out) == ids.size() && ok;
    ok = std::fclose(out) == 0 && ok;
    if (!ok) throw std::runtime_error("output write incomplete");
    std::printf("context=%lld queries=%lld pattern=%d active=%lld scores_ms=%.6f topk_ms=%.6f score_values=%zu id_values=%zu\n",
                static_cast<long long>(context), static_cast<long long>(nq), pattern, static_cast<long long>(active),
                score_ms, topk_ms, scores.size(), ids.size());
    ck(cudaFree(dp), "free"); ck(cudaFree(dd), "free"); ck(cudaFree(dq), "free");
    ck(cudaFree(ds), "free"); ck(cudaFree(dc), "free"); ck(cudaFree(di), "free");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "qsa_prefill_multi_parity: %s\n", e.what());
    return 1;
}
