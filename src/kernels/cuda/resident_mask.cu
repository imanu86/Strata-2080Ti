// src/kernels/cuda/resident_mask.cu - STRATA_RESIDENT_MASK=1: resident-only routing (opt-in, changes the output).
//
// An adaptive pruning mask: the experts no GPU holds are masked out of the router's top-10 selection (a -inf bias on
// the selection, in effect), so the 10 are chosen among the resident ones and the next-best resident expert takes
// the place of a masked one.  The mask is the complement of the resident set and follows it: the table is rebuilt
// from the host residency tables (the primary's, a layer split's stages', the expert servers') before a verify
// window is launched and the layers that changed are uploaded (a few KB after an adaptive round, nothing otherwise).
//
// The WEIGHTS are the router's own on the selected set: p = softmax over the 512 unbiased logits, then
// w_i = p_i / max(sum of the 10 selected p, 2^-14) - the native router's expressions (native_router.cu), applied to
// the experts the mask let through.  A token whose router top-10 holds no masked expert keeps the router's bits.
//
// Unlike STRATA_ROUTE_PRIOR (a finite logit bonus) and STRATA_ROUTE_RESIDENT (tail ranks, within a margin) every rank
// moves and nothing non-resident is ever picked while 10 resident experts exist in the layer; unlike
// STRATA_ROUTE_TAIL_SKIP nothing is zeroed.
//
// The sensor (REAP-loop style): per routing, how many of the router's picks were replaced and the top-10 weight
// they carried; a replaced pick is also counted as demand for the adaptive tier, so a masked expert that the router
// keeps asking for is swapped in.
#include "strata/kernels/resident_mask.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace strata::kernels::resident_mask {
namespace {

__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

// One warp per token; 512 experts = 16 values per lane (as the native router).  `res` is the layer's mask row
// (>= 0: resident on some GPU).  ids / weights hold the router's top-10 on entry.
__global__ void resident_mask_k(const float* __restrict__ logits, const int32_t* __restrict__ res,
                                int32_t* __restrict__ ids, float* __restrict__ weights,
                                unsigned long long* __restrict__ stats, unsigned int* __restrict__ demand) {
    const int lane = threadIdx.x;
    const float* lg = logits + (size_t) blockIdx.x * 512;
    int32_t* id = ids + (size_t) blockIdx.x * 10;
    float* w = weights + (size_t) blockIdx.x * 10;
    // the router's picks (every lane reads the same ten: the branch below is uniform across the warp)
    int32_t old_id[10];
    float old_w[10];
    int n_masked = 0;
    for (int r = 0; r < 10; ++r) {
        old_id[r] = id[r];
        old_w[r] = w[r];
        n_masked += (old_id[r] >= 0 && old_id[r] < 512 && res[old_id[r]] < 0) ? 1 : 0;
    }
    __syncwarp();   // every lane has read the picks before any lane writes them
    // HOOK (no automatic widening yet): a controller that wants to lift the mask for a step would test here - e.g.
    // the weight of the masked picks (the sum of old_w over them) against a threshold - and return, keeping the
    // router's own picks for this token; the sensor below already measures that weight.
    if (n_masked == 0) {
        if (lane == 0 && stats != nullptr) {
            atomicAdd(stats + 0, 10ull);
            atomicAdd(stats + 3, 1ull);
        }
        return;
    }
    // the router's probabilities: softmax over ALL 512 unbiased logits (native_router.cu's expressions)
    float p[16];
    unsigned int resident = 0u, taken = 0u;
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        float v = lg[e];
        if (v != v) v = -FLT_MAX;   // a NaN logit is never picked
        p[i] = v;
        mx = fmaxf(mx, v);
        if (res[e] >= 0) resident |= 1u << i;
    }
    mx = wmax(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        p[i] = expf(p[i] - mx);
        sum += p[i];
    }
    const float reciprocal = 1.0f / wsum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) p[i] *= reciprocal;
    // the top-10 again: a resident expert before any masked one (class 2 over class 1), then the larger probability,
    // then the smaller index - the router's order within a class.  A masked expert is only picked when the layer
    // holds fewer than 10 resident ones (it then runs through the miss path, as without the mask).
    int32_t new_id[10];
    float new_p[10];
    float own_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        int bc = 0, be = 1 << 30;
        float bv = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if ((taken >> i) & 1u) continue;
            const int c = ((resident >> i) & 1u) ? 2 : 1;
            if (c > bc || (c == bc && p[i] > bv)) { bc = c; bv = p[i]; be = lane + 32 * i; }
        }
#pragma unroll
        for (int m = 16; m; m >>= 1) {
            const int oc = __shfl_xor_sync(0xffffffffu, bc, m, 32);
            const float ov = __shfl_xor_sync(0xffffffffu, bv, m, 32);
            const int oe = __shfl_xor_sync(0xffffffffu, be, m, 32);
            if (oc > bc || (oc == bc && (ov > bv || (ov == bv && oe < be)))) { bc = oc; bv = ov; be = oe; }
        }
        if ((be & 31) == lane) {
            taken |= 1u << (be >> 5);
            own_sum += bv;   // by winning expert lane, as the native router accumulates it
        }
        new_id[rank] = be;   // the same on every lane
        new_p[rank] = bv;
    }
    const float inverse_selected_sum = 1.0f / fmaxf(wsum(own_sum), 6.103515625e-5f);
    if (lane == 0) {
        // the sensor: the router's picks that are gone and the top-10 weight the router gave them; every masked
        // pick is demand for the adaptive tier (also one the layer kept for want of 10 resident experts)
        int rerouted = 0;
        float mass = 0.0f;
        for (int r = 0; r < 10; ++r) {
            bool kept = false;
            for (int q = 0; q < 10; ++q) kept = kept || new_id[q] == old_id[r];
            if (!kept) { ++rerouted; mass += old_w[r]; }
            // demand, weighted and bounded: min(1, 10 x the router's top-10 weight) in 1/256 - a pick at or above the
            // uniform share counts as one routed entry, a tail pick as a few hundredths of one
            if (demand != nullptr && old_id[r] >= 0 && old_id[r] < 512 && res[old_id[r]] < 0)
                atomicAdd(demand + old_id[r], (unsigned int) (fminf(fmaxf(10.0f * old_w[r], 0.0f), 1.0f) * 256.0f + 0.5f));
        }
        if (stats != nullptr) {
            atomicAdd(stats + 0, 10ull);
            atomicAdd(stats + 1, (unsigned long long) rerouted);
            atomicAdd(stats + 2, (unsigned long long) (fminf(fmaxf(mass, 0.0f), 1.0f) * 16777216.0f + 0.5f));
            atomicAdd(stats + 3, 1ull);
        }
    }
    if (lane < 10) {
        id[lane] = new_id[lane];
        w[lane] = new_p[lane] * inverse_selected_sum;
    }
}

// ---- the host side: one table, one counter block and one demand block per device ----
constexpr int kMaxDev = 16;
struct Dev {
    std::atomic<int32_t*> table{nullptr};   ///< [n_layers * n_expert], >= 0: resident; published last
    unsigned long long* stats = nullptr;    ///< 8 x uint64: [0..3] the verify windows (decode), [4..7] the prompt path
    unsigned int* demand = nullptr;         ///< [n_layers * n_expert], 1/256 of a pick
    cudaStream_t stream = nullptr;          ///< its own non-blocking stream: no copy here touches the legacy stream
};
struct State {
    std::mutex mu;
    Dev dev[kMaxDev];
    int64_t n_layers = 0, n_expert = 0;
    std::vector<const int32_t*> sources;
    std::vector<int32_t> shadow, next;      ///< what the devices hold / the union being built
    bool have_shadow = false;
    std::vector<float>* sink = nullptr;
    std::vector<unsigned int> demand_host;
    std::atomic<int> hold{0};
    std::atomic<bool> lent{false};
    std::atomic<long long> last_ns{0};
    long long refreshes = 0, updates = 0, resident = 0;
    double demand_folded = 0.0;
    Counters reported;
    long long requests = 0;
};
State& state() {
    static State s;
    return s;
}
long long now_ns() {
    return (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
// the current device set to `device` for a scope (the caller's device is put back)
struct Current {
    int previous = -1;
    explicit Current(int device) {
        int cur = -1;
        if (cudaGetDevice(&cur) == cudaSuccess && cur != device && cudaSetDevice(device) == cudaSuccess) previous = cur;
    }
    ~Current() { if (previous >= 0) cudaSetDevice(previous); }
    Current(const Current&) = delete;
    Current& operator=(const Current&) = delete;
};
// STRATA_RESIDENT_MASK_DEMAND=N: the masked decode picks are folded into the adaptive tier's usage every N refreshes
// (default 16, about three times a second while decoding; 0: never - nothing is counted on the device either)
int demand_every() {
    static const int n = [] {
        const char* v = std::getenv("STRATA_RESIDENT_MASK_DEMAND");
        return v != nullptr ? std::max(0, std::atoi(v)) : 16;
    }();
    return n;
}
// STRATA_RESIDENT_MASK_DEMAND_GAIN=f (default 1): what a full masked pick adds to the usage of its expert, in routed
// entries.  THE DEFAULT, AND WHY: with no demand at all a masked expert is never routed, so its usage stays 0 and the
// adaptive tier can never bring it in - the resident set freezes on the profile.  With +1 per pick (the first
// version) every tail pick of every prompt token counted, 4 x 10^7 a request, and the tier churned.  So: decode
// picks only, each worth min(1, 10 x its router weight) - an expert the router really wanted heats up like a routed
// one, a rank-9 pick at 2% adds 0.2.  0.25 or 0.5 slow it further; STRATA_RESIDENT_MASK_DEMAND=0 is the off arm.
float demand_gain() {
    static const float f = [] {
        const char* v = std::getenv("STRATA_RESIDENT_MASK_DEMAND_GAIN");
        return v != nullptr ? (float) std::max(0.0, std::atof(v)) : 1.0f;
    }();
    return f;
}

}  // namespace

int mode() {
    static const int m = [] {
        const char* v = std::getenv("STRATA_RESIDENT_MASK");
        const int n = v != nullptr ? std::atoi(v) : 0;
        return n == 2 ? 2 : n != 0 ? 1 : 0;
    }();
    return m;
}
bool requested() { return mode() != 0; }
bool prefill_masked() { return mode() == 1; }

bool prepare_device(int device, int64_t n_layers, int64_t n_expert, std::string& err) {
    if (!requested()) return true;
    if (device < 0 || device >= kMaxDev) { err = "STRATA_RESIDENT_MASK: device " + std::to_string(device) + " is out of range"; return false; }
    if (n_expert != 512 || n_layers < 1) {
        err = "STRATA_RESIDENT_MASK: the mask kernel is the 512-expert / top-10 one; this model routes " +
              std::to_string(n_expert) + " experts";
        return false;
    }
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.n_layers == 0) { s.n_layers = n_layers; s.n_expert = n_expert; }
    if (s.n_layers != n_layers || s.n_expert != n_expert) { err = "STRATA_RESIDENT_MASK: two geometries in one run"; return false; }
    Dev& d = s.dev[device];
    if (d.table.load(std::memory_order_acquire) != nullptr) return true;
    const size_t n = (size_t) (n_layers * n_expert);
    const Current on(device);
    int32_t* table = nullptr;
    bool ok = cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking) == cudaSuccess &&
              cudaMalloc((void**) &table, n * sizeof(int32_t)) == cudaSuccess &&
              cudaMalloc((void**) &d.stats, 8 * sizeof(unsigned long long)) == cudaSuccess &&
              cudaMalloc((void**) &d.demand, n * sizeof(unsigned int)) == cudaSuccess;
    // every expert allowed (0 >= 0) until a source says otherwise; a device prepared later starts from the current mask
    ok = ok && (s.have_shadow ? cudaMemcpyAsync(table, s.shadow.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, d.stream)
                              : cudaMemsetAsync(table, 0, n * sizeof(int32_t), d.stream)) == cudaSuccess &&
         cudaMemsetAsync(d.stats, 0, 8 * sizeof(unsigned long long), d.stream) == cudaSuccess &&
         cudaMemsetAsync(d.demand, 0, n * sizeof(unsigned int), d.stream) == cudaSuccess &&
         cudaStreamSynchronize(d.stream) == cudaSuccess;
    if (!ok) {
        (void) cudaGetLastError();
        err = "STRATA_RESIDENT_MASK: the mask table could not be allocated on CUDA" + std::to_string(device);
        return false;
    }
    d.table.store(table, std::memory_order_release);
    return true;
}

void add_source(const int32_t* table, const char* what) {
    if (!requested() || table == nullptr) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    for (const int32_t* t : s.sources)
        if (t == table) return;
    s.sources.push_back(table);
    std::fprintf(stderr, "strata resident-mask: source %zu: %s\n", s.sources.size(), what != nullptr ? what : "a residency table");
}

void set_demand_sink(std::vector<float>* usage) {
    if (!requested()) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.sink = usage;
}

void hold_enter() { if (requested()) state().hold.fetch_add(1, std::memory_order_acq_rel); }
void hold_leave() { if (requested()) state().hold.fetch_sub(1, std::memory_order_acq_rel); }
void set_lent(bool lent) { if (requested()) state().lent.store(lent, std::memory_order_release); }

void refresh(bool force) {
    if (!requested()) return;
    State& s = state();
    const long long now = now_ns();
    if (!force) {
        // a prompt's loan is out (or its tables are being edited): the mask stays the one of before the loan
        if (s.hold.load(std::memory_order_acquire) > 0 || s.lent.load(std::memory_order_acquire)) return;
        // a layer split stages one window on every stage: one pass is enough for all of them
        if (now - s.last_ns.load(std::memory_order_relaxed) < 500000) return;
    }
    std::unique_lock<std::mutex> lk(s.mu, std::defer_lock);
    if (force) lk.lock();
    else if (!lk.try_lock()) return;
    if (s.sources.empty() || s.n_layers == 0) return;
    s.last_ns.store(now, std::memory_order_relaxed);
    const int64_t NL = s.n_layers, NE = s.n_expert;
    const size_t n = (size_t) (NL * NE);
    // the union of the sources: resident on ANY GPU (the entries are read while the adaptive tier may be writing
    // one - a stale entry is one expert let through or held back for a window, never a wrong result: a picked
    // expert that is not there takes the miss path)
    s.next.resize(n);
    {
        const int32_t* a = s.sources[0];
        for (size_t i = 0; i < n; ++i) s.next[i] = a[i] >= 0 ? 0 : -1;
        for (size_t k = 1; k < s.sources.size(); ++k) {
            const int32_t* b = s.sources[k];
            for (size_t i = 0; i < n; ++i)
                if (b[i] >= 0) s.next[i] = 0;
        }
    }
    int64_t lo = NL, hi = -1;   // the dirty layers
    if (!s.have_shadow) { lo = 0; hi = NL - 1; }
    else
        for (int64_t l = 0; l < NL; ++l)
            if (std::memcmp(s.next.data() + l * NE, s.shadow.data() + l * NE, (size_t) NE * sizeof(int32_t)) != 0) {
                lo = std::min(lo, l);
                hi = l;
            }
    const bool fold = s.sink != nullptr && demand_every() > 0 && (++s.refreshes % demand_every()) == 0;
    if (hi < 0 && !fold) return;
    int cur = -1;
    (void) cudaGetDevice(&cur);
    bool moved = false;
    if (hi >= 0) {
        s.shadow.swap(s.next);
        s.have_shadow = true;
        ++s.updates;
        long long res = 0;
        for (size_t i = 0; i < n; ++i) res += s.shadow[i] >= 0;
        s.resident = res;
    }
    if (fold && s.demand_host.size() != n) s.demand_host.assign(n, 0u);
    for (int d = 0; d < kMaxDev; ++d) {
        Dev& dv = s.dev[d];
        int32_t* const table = dv.table.load(std::memory_order_acquire);
        if (table == nullptr) continue;
        if (d != cur) { if (cudaSetDevice(d) != cudaSuccess) { (void) cudaGetLastError(); continue; } moved = true; }
        else if (moved) { (void) cudaSetDevice(d); }
        bool ok = true;
        if (hi >= 0)
            ok = cudaMemcpyAsync(table + lo * NE, s.shadow.data() + lo * NE, (size_t) ((hi - lo + 1) * NE) * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, dv.stream) == cudaSuccess;
        if (ok && fold)
            ok = cudaMemcpyAsync(s.demand_host.data(), dv.demand, n * sizeof(unsigned int), cudaMemcpyDeviceToHost, dv.stream) == cudaSuccess &&
                 cudaMemsetAsync(dv.demand, 0, n * sizeof(unsigned int), dv.stream) == cudaSuccess;
        ok = ok && cudaStreamSynchronize(dv.stream) == cudaSuccess;
        if (!ok) {
            (void) cudaGetLastError();
            std::fprintf(stderr, "strata resident-mask: the table on CUDA%d could not be refreshed; it keeps the previous mask\n", d);
            continue;
        }
        // the masked decode picks as demand, in the unit the pool counts a routed entry with (weighted in the kernel,
        // scaled by the gain; a count lost to the pool thread's own += is a count, not a fault)
        if (fold && s.sink != nullptr && s.sink->size() == n) {
            float* u = s.sink->data();
            const float gain = demand_gain() / 256.0f;
            for (size_t i = 0; i < n; ++i)
                if (s.demand_host[i] != 0u) {
                    const float add = gain * (float) s.demand_host[i];
                    u[i] += add;
                    s.demand_folded += (double) add;
                }
        }
    }
    if (moved && cur >= 0) (void) cudaSetDevice(cur);
}

void residency_changed() {
    if (!requested()) return;
    state().last_ns.store(0, std::memory_order_relaxed);   // a table was just edited: no rate limit for this pass
    refresh(false);
}

bool apply(const float* logits, int32_t* ids, float* weights, int64_t layer, int64_t n_expert, int64_t k, int n_tok,
           void* stream, std::string& err, bool prefill) {
    if (n_tok < 1) return true;
    if (n_expert != 512 || k != 10) {
        err = "STRATA_RESIDENT_MASK: the mask kernel is the 512-expert / top-10 one; this router is " +
              std::to_string(n_expert) + " / " + std::to_string(k);
        return false;
    }
    State& s = state();
    int d = -1;
    if (cudaGetDevice(&d) != cudaSuccess) d = -1;
    int32_t* const table = d >= 0 && d < kMaxDev ? s.dev[d].table.load(std::memory_order_acquire) : nullptr;
    if (table == nullptr || layer < 0 || layer >= s.n_layers || stream == nullptr) {
        // never partial: a routing path nobody prepared a table for stops the run instead of routing unmasked
        err = "STRATA_RESIDENT_MASK: no mask table on CUDA" + std::to_string(d) + " for layer " + std::to_string(layer) +
              " (a routing path the mask was not prepared for); run without STRATA_RESIDENT_MASK";
        return false;
    }
    const Dev& dv = s.dev[d];
    // the prompt path: its own counters, and no demand (a prompt's tokens would drown the decode's)
    const bool count_demand = !prefill && dv.demand != nullptr && demand_every() > 0;
    resident_mask_top10(logits, table + layer * 512, ids, weights, n_tok, dv.stats + (prefill ? 4 : 0),
                        count_demand ? dv.demand + layer * 512 : nullptr, stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { err = std::string("STRATA_RESIDENT_MASK: ") + cudaGetErrorString(e); return false; }
    return true;
}

bool read(Counters& c) {
    c = Counters{};
    if (!requested()) return false;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    int cur = -1;
    (void) cudaGetDevice(&cur);
    bool moved = false, ok = true;
    for (int d = 0; d < kMaxDev; ++d) {
        Dev& dv = s.dev[d];
        if (dv.table.load(std::memory_order_acquire) == nullptr) continue;
        if (d != cur) { if (cudaSetDevice(d) != cudaSuccess) { (void) cudaGetLastError(); ok = false; continue; } moved = true; }
        else if (moved) { (void) cudaSetDevice(d); }
        unsigned long long h[8] = {};
        if (cudaMemcpyAsync(h, dv.stats, sizeof h, cudaMemcpyDeviceToHost, dv.stream) != cudaSuccess ||
            cudaStreamSynchronize(dv.stream) != cudaSuccess) {
            (void) cudaGetLastError();
            ok = false;
            continue;
        }
        for (int part = 0; part < 2; ++part) {
            Part& q = part == 0 ? c.decode : c.prefill;
            q.choices += h[part * 4 + 0];
            q.rerouted += h[part * 4 + 1];
            q.masked_mass += (double) h[part * 4 + 2] / 16777216.0;
            q.tokens += h[part * 4 + 3];
        }
    }
    if (moved && cur >= 0) (void) cudaSetDevice(cur);
    return ok;
}

void report(std::FILE* f) {
    if (!requested() || f == nullptr) return;
    Counters c;
    if (!read(c)) return;
    State& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    auto delta = [](const Part& now, const Part& was) {
        Part d;
        d.choices = now.choices - was.choices;
        d.rerouted = now.rerouted - was.rerouted;
        d.tokens = now.tokens - was.tokens;
        d.masked_mass = now.masked_mass - was.masked_mass;
        return d;
    };
    const Part d = delta(c.decode, s.reported.decode), p = delta(c.prefill, s.reported.prefill);
    s.reported = c;
    ++s.requests;
    char pf[160];
    if (!prefill_masked()) std::snprintf(pf, sizeof pf, "prompt path not masked (STRATA_RESIDENT_MASK=2)");
    else
        std::snprintf(pf, sizeof pf, "prompt path: rerouted %llu of %llu choices (%.2f%%), masked weight mass %.2f%%", p.rerouted,
                      p.choices, p.choices ? 100.0 * (double) p.rerouted / (double) p.choices : 0.0,
                      p.tokens ? 100.0 * p.masked_mass / (double) p.tokens : 0.0);
    // the first figures are the DECODE's (the verify windows); the prompt path's follow, apart
    std::fprintf(f, "strata resident-mask: request %lld, rerouted %llu of %llu choices (%.2f%%), masked weight mass %.2f%% "
                    "(decode); %s [%lld of %lld pairs resident, %lld mask updates, demand added to the adaptive tier %.0f]\n",
                 s.requests, d.rerouted, d.choices, d.choices ? 100.0 * (double) d.rerouted / (double) d.choices : 0.0,
                 d.tokens ? 100.0 * d.masked_mass / (double) d.tokens : 0.0, pf, s.resident,
                 (long long) (s.n_layers * s.n_expert), s.updates, s.demand_folded);
}

void resident_mask_top10(const float* logits, const int32_t* mask_layer, int32_t* ids, float* weights, int n_tok,
                         unsigned long long* stats, unsigned int* demand_layer, void* stream) {
    if (n_tok <= 0) return;
    resident_mask_k<<<(unsigned) n_tok, 32, 0, (cudaStream_t) stream>>>(logits, mask_layer, ids, weights, stats, demand_layer);
}

}  // namespace strata::kernels::resident_mask
