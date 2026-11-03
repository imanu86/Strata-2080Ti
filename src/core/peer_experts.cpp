// src/core/peer_experts.cpp - see include/strata/core/peer_experts.hpp.
// STRATA_EP_SPLIT_PREFILL (docs/EP_SPLIT_PREFILL.md): with the variable on, the prompt path of the peer card's stage
// reads this cache and res_ as its resident tier (cache(), res_table()).  Tappa 4b (s.13.2): the tier's tail is LENT
// to that prompt path for a prompt read (cache_mut(), res_table_mut(), lend_guard()): generate.cpp lays the stage's
// prompt buffers in the slots >= first after a quiesce(), marks their pairs kNotResident, and refills the same slots
// from the source arena before the next decode window.  With the guard up adapt() does not swap and fused_begin()
// refuses.  Off, none of it is called and the tier is never lent.
#include "strata/core/peer_experts.hpp"

#include "strata/core/device.hpp"   // primary_device() (STRATA_PRIMARY_DEVICE)
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/hadamard.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/verify_kernels.hpp"   // gpu_stamp (STRATA_PEER_STAMP)
#include "strata/kernels/window_limits.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::core {

static bool g_peer_portable = false;
void set_peer_portable(bool on) { g_peer_portable = on; }
bool peer_portable() { return g_peer_portable; }

namespace {

constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t CAP = strata::kernels::kWindowMaxT * 10;   // entries per layer: a window's rows x top-10

struct Meta {                           // one block, uploaded with one copy per layer
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count[4];                   // [0] groups, [1] entries
    int32_t out_row[CAP];               // multi-GPU: compact row -> entry row in the host `out`
};
// STRATA_PEER_ONE_COPY: the plan's bytes in the shared block, rounded so the rows after it start 256-byte aligned
constexpr size_t kMetaPad = (sizeof(Meta) + 255) / 256 * 256;

/// Switches the calling thread to `dev` and back.  The engine's own thread lives on device 0.
struct On {
    int prev = 0;
    explicit On(int dev) { cudaGetDevice(&prev); if (prev != dev) cudaSetDevice(dev); }
    ~On() { int cur = 0; cudaGetDevice(&cur); if (cur != prev) cudaSetDevice(prev); }
};

/// #2666: `cudaDeviceCanAccessPeer` said yes and `cudaDeviceEnablePeerAccess` succeeded, and on a pair of GPUs behind a host bridge
/// (2x RTX A4000, "PHB" in nvidia-smi topo) `cudaMemcpyPeerAsync` then returned success having moved nothing: the peer's input rows
/// were zeros and the rows it sent back never arrived, so the first prompt's MoE output was garbage and the logits NaN.  Without
/// peer access enabled the same copies are right (the driver stages them).  So the pair is asked once, with a small copy each way,
/// whether the direct path really carries data.  true = both ways exact (or nothing could be tested: the old trust is kept).
bool p2p_copies_data(int device) {
    constexpr size_t kN = 1 << 14;   // floats
    std::vector<float> h(kN), back(kN);
    for (size_t i = 0; i < kN; ++i) h[i] = 1.0f + (float) ((i * 2654435761u) % 4093u);
    float* d[2] = {nullptr, nullptr};
    const int dev[2] = {0, device};
    bool ok = true, tested = true;
    for (int k = 0; k < 2 && tested; ++k) {
        On on(dev[k]);
        if (cudaMalloc((void**) &d[k], kN * 4) != cudaSuccess) { cudaGetLastError(); tested = false; }
    }
    for (int dir = 0; dir < 2 && tested && ok; ++dir) {   // 0: device 0 -> peer, 1: peer -> device 0
        const int from = dir == 0 ? 0 : 1, to = 1 - from;
        {
            On on(dev[from]);
            if (cudaMemcpy(d[from], h.data(), kN * 4, cudaMemcpyHostToDevice) != cudaSuccess) { tested = false; break; }
        }
        {
            On on(dev[to]);
            if (cudaMemset(d[to], 0, kN * 4) != cudaSuccess) { tested = false; break; }
        }
        if (cudaMemcpyPeer(d[to], dev[to], d[from], dev[from], kN * 4) != cudaSuccess) { tested = false; break; }
        {
            On on(dev[to]);
            if (cudaMemcpy(back.data(), d[to], kN * 4, cudaMemcpyDeviceToHost) != cudaSuccess) { tested = false; break; }
        }
        ok = std::memcmp(back.data(), h.data(), kN * 4) == 0;
    }
    for (int k = 0; k < 2; ++k)
        if (d[k] != nullptr) { On on(dev[k]); cudaFree(d[k]); }
    if (!tested) { cudaGetLastError(); return true; }
    return ok;
}

bool ck(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("peer experts: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

/// STRATA_PEER_DIRECT (default on): launch() scatters its rows straight into the host rows it is given.
bool peer_direct() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_DIRECT"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}

// ---- STRATA_PEER_FUSED helpers (docs/PEER_FUSED.md)

/// An integer knob, `def` when unset or malformed.
int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return def;
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    return end == v ? def : (int) x;
}

/// The kernel's `err` word as words (peer_fused::Err).
const char* fused_err_name(uint32_t e) {
    using namespace strata::core::peer_fused;
    switch (e) {
        case kErrNone: return "no error";
        case kErrTimeoutSeq: return "timeout waiting for the primary's doorbell";
        case kErrTimeoutReady: return "timeout waiting for block 0's plan";
        case kErrTimeoutBarrier: return "timeout at a grid barrier";
        case kErrAbort: return "aborted by the host";
        case kErrPlanOverflow: return "more entries in a step than the plan holds";
        case kErrBadWindow: return "a window geometry the kernel was not built for";
        case kErrBadType: return "a layer's gate/up or down type the kernel has no body for";
        case kErrTimeoutXq: return "timeout waiting for the primary's q8_1 rows of a step (STRATA_PEER2_STAGE)";
        case kErrTimeoutDone: return "timeout waiting for every block's count of the step before last (STRATA_PEER2_SKIP)";
        default: return "unknown";
    }
}

inline void store_u32(uint32_t* p, uint32_t v) { *(volatile uint32_t*) p = v; }
inline uint32_t load_u32(const uint32_t* p) { return *(const volatile uint32_t*) p; }

// STRATA_PEER_NOSWITCH: once refused by the driver the queries go back under the device switch
bool g_noswitch_refused = false;
void noswitch_refuse(cudaError_t e) {
    if (!g_noswitch_refused)
        std::fprintf(stderr, "strata: STRATA_PEER_NOSWITCH: the peer's stream refused a query without the device switch (%s): "
                             "switching the device for the peer's waits from here on\n", cudaGetErrorString(e));
    g_noswitch_refused = true;
    cudaGetLastError();
}

}  // namespace

bool PeerExperts::noswitch() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_NOSWITCH"); return v != nullptr && std::atoi(v) != 0; }();
    return on && !g_noswitch_refused;
}

bool PeerExperts::stamp() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_STAMP"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

bool PeerExperts::refill_flush() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_REFILL_FLUSH"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

bool PeerExperts::one_copy() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PEER_ONE_COPY"); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

bool PeerExperts::fused_env() {
    static const bool on = [] { const char* v = std::getenv(peer_fused::kEnvOn); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

bool PeerExperts::fused_hostwait() {
    static const bool on = [] { const char* v = std::getenv(peer_fused::kEnvHostWait); return v != nullptr && std::atoi(v) != 0; }();
    return on;
}

PeerExperts::~PeerExperts() { close(); }

void PeerExperts::close() {
    if (device_ < 0) return;
    // STRATA_PEER_FUSED: a window's kernel may still be waiting for a doorbell that will never come: make it leave
    // (abort word + bounded drain) before the stream it runs on is destroyed
    if (fused_on_) fused_abort();
    {
        On on(device_);
        if (stream_) cudaStreamSynchronize(stream_);
        fused_close();
        if (refill_) cudaStreamSynchronize(refill_);
        cache_.close();
        if (d_stage_) cudaFree(d_stage_);   // STRATA_PEER_ONE_COPY: d_x_ and d_meta_ point into it
        else { if (d_x_) cudaFree(d_x_); if (d_meta_) cudaFree(d_meta_); }
        if (d_out_) cudaFree(d_out_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (h_stage_) cudaFreeHost(h_stage_);
        else { if (h_x_) cudaFreeHost(h_x_); if (h_meta_) cudaFreeHost(h_meta_); }
        if (h_out_) cudaFreeHost(h_out_);
        if (h_stamp_) cudaFreeHost(h_stamp_);
        if (refill_ev_) cudaEventDestroy(refill_ev_);
        if (stream_) cudaStreamDestroy(stream_);
        if (refill_) cudaStreamDestroy(refill_);
    }
    d_x_ = d_out_ = h_x_ = h_out_ = nullptr;
    d_meta_ = h_meta_ = d_scratch_ = nullptr;
    d_stage_ = h_stage_ = nullptr;
    d_q8_ = nullptr;
    d_sx_ = d_sh_ = nullptr;
    h_stamp_ = m_stamp_ = nullptr;
    stamp_layer_ = -1;
    stream_ = refill_ = nullptr;
    refill_ev_ = nullptr;
    lent_.store(false, std::memory_order_release);   // tappa 4b: a closed tier lends nothing
    lent_slots_ = 0;
    device_ = -1;
}

bool PeerExperts::open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
                       ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
                       std::string& err, const PeerExperts* before, int ordinal, double share) {
    close();
    ordinal_ = ordinal;
    left_.clear();
    left_count_ = took_left_ = 0;
    int count = 0;
    if (!ck(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 1 || device >= count) {
        err = "peer experts: CUDA device " + std::to_string(device) + " is not visible (" + std::to_string(count) +
              " devices; set CUDA_VISIBLE_DEVICES)";
        return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    src_ = &src;
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    {   // direct access both ways (NVLink or another P2P path): the prompt path copies activations and results over it
        int a = 0, b = 0;
        cudaDeviceCanAccessPeer(&a, primary_device(), device);
        cudaDeviceCanAccessPeer(&b, device, primary_device());
        if (const char* np = std::getenv("STRATA_PEER_NO_P2P"); np != nullptr && std::atoi(np) != 0) a = 0;   // test knob: pretend the pair has no P2P
        p2p_ = a && b;
        if (p2p_) {
            On on0(primary_device());
            cudaError_t e0 = cudaDeviceEnablePeerAccess(device, 0);
            if (e0 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e0 = cudaSuccess; }
            On on1(device);
            cudaError_t e1 = cudaDeviceEnablePeerAccess(primary_device(), 0);
            if (e1 == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e1 = cudaSuccess; }
            p2p_ = e0 == cudaSuccess && e1 == cudaSuccess;
            // recommend, never force: the direct path is used only if a test copy each way arrives exact (#2666)
            if (p2p_ && !p2p_copies_data(device)) {
                std::fprintf(stderr, "strata: peer GPU %d: peer access is enabled but a test copy between the two GPUs arrived wrong "
                                     "(a host bridge between them, no real P2P): the prompt path takes the host route instead "
                                     "(STRATA_PEER_NO_P2P=1 does that by hand)\n", device);
                { On q0(0); cudaDeviceDisablePeerAccess(device); cudaGetLastError(); }
                { On q1(device); cudaDeviceDisablePeerAccess(0); cudaGetLastError(); }
                p2p_ = false;
            }
        }
    }
    On on(device);
    device_ = device;
    // scratch first, so the slots take what is really left
    int64_t ff = strata::kernels::cpu::FF;
    if (lay.native)
        for (const auto& f : lay.fmt) ff = std::max<int64_t>(ff, f.n_ff);
    const size_t scratch = std::max<size_t>((size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, ff),
                                            strata::kernels::native_expert_scratch_bytes(CAP, ff));
    // STRATA_PEER_ONE_COPY: the plan (kMetaPad bytes) and the activation rows in one block on each side, one copy a layer
    const bool one = one_copy();
    const size_t stage = kMetaPad + (size_t) CAP * H * sizeof(float);
    const bool alloc_ok =
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        ck(cudaStreamCreateWithFlags(&refill_, cudaStreamNonBlocking), "refill stream", err) &&
        ck(cudaEventCreateWithFlags(&refill_ev_, cudaEventDisableTiming), "event", err) &&
        (one ? ck(cudaHostAlloc(&h_stage_, stage, cudaHostAllocPortable | cudaHostAllocMapped), "plan + input staging", err) &&
               ck(cudaMalloc(&d_stage_, stage), "plan + input", err)
             : ck(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err) &&
               ck(cudaHostAlloc(&h_meta_, sizeof(Meta), cudaHostAllocPortable | cudaHostAllocMapped), "plan staging", err) &&
               ck(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err) &&
               ck(cudaMalloc(&d_meta_, sizeof(Meta)), "plan", err)) &&
        ck(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err) &&
        ck(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err) &&
        ck(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activations", err) &&
        ck(cudaMalloc(&d_scratch_, scratch), "scratch", err);
    if (!alloc_ok) { close(); return false; }
    if (one) {   // the plan at the start of the block, the rows after it
        h_meta_ = h_stage_;
        d_meta_ = d_stage_;
        h_x_ = (float*) ((uint8_t*) h_stage_ + kMetaPad);
        d_x_ = (float*) ((uint8_t*) d_stage_ + kMetaPad);
    }
    if (stamp()) {   // STRATA_PEER_STAMP: the stamp page, mapped so the host reads it without a copy
        const size_t n = (size_t) n_layers * 2;
        if (!ck(cudaHostAlloc((void**) &h_stamp_, n * sizeof(unsigned long long), cudaHostAllocPortable | cudaHostAllocMapped), "stamp page", err) ||
            !ck(cudaHostGetDevicePointer((void**) &m_stamp_, h_stamp_, 0), "stamp page device pointer", err)) {
            close();
            return false;
        }
        std::memset(h_stamp_, 0, n * sizeof(unsigned long long));
        spans_us_.clear();
        span_us_total = 0;
    }
    if (strata::kernels::cpu::hadamard_spec().any()) {   // APR: this device's copies of the sign vectors
        d_sx_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) H), (int) H);
        d_sh_ = strata::kernels::hadamard_device_signs(strata::kernels::cpu::hadamard_signs((int) ff), (int) ff);
    }

    // the pairs the primary does not hold, in rank order, as many as fit
    size_t free_b = 0, total_b = 0;
    if (!ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t reserve = (uint64_t) std::max(reserve_mib, 128) << 20;
    const uint64_t budget = free_b > reserve ? free_b - reserve : 0;
    std::vector<std::pair<int32_t, int32_t>> pick;
    std::vector<int64_t> sizes;
    uint64_t used = 0;
    // s.11.3 (STRATA_PEER2_SHARE): the first server leaves one candidate every 1/share to the second, counted over its
    // candidates in rank order up to where its budget ends (the pairs after that end go to the second anyway)
    const bool leave = ordinal == 0 && before == nullptr && share > 0.0 && share <= 0.5;
    if (leave) left_.assign((size_t) (n_layers * n_expert), 0);
    int64_t cand = 0;
    for (const auto& pr : ranked) {
        if (primary.slot_of(pr.first, pr.second) >= 0) continue;
        // two servers: the second takes the NEXT pairs of the ranking, the ones the first server did not fit either
        // (its tier ends where its budget ended, in rank order), so a pair lives on one card
        if (before != nullptr && before->has(pr.first, pr.second)) continue;
        if (leave) {
            const int64_t c = cand++;
            if ((int64_t) ((double) (c + 1) * share) > (int64_t) ((double) c * share)) {
                left_[(size_t) (pr.first * n_expert + pr.second)] = 1;
                ++left_count_;
                continue;   // the second server's: it skips what this one holds, so it finds this pair free
            }
        }
        const uint64_t b = lay.blob_bytes(pr.first);
        const uint64_t b256 = lay.native ? (b + 255) / 256 * 256 : lay.max_blob;
        if (used + b256 > budget) break;
        if (max_slots > 0 && (int64_t) pick.size() >= max_slots) break;
        used += b256;
        pick.push_back(pr);
        sizes.push_back((int64_t) b);
        if (before != nullptr && before->left_for_next(pr.first, pr.second)) ++took_left_;   // the second: a hot pair the first left it
    }
    if (leave) {
        // the pairs left after the first server's end were never its (the second takes them as the next of the
        // ranking): only those left BEFORE the end count as "left", so the log says what really moved
        int64_t beyond = 0;
        if (!pick.empty()) {
            const auto& last = pick.back();
            bool past = false;
            for (const auto& pr : ranked) {
                if (past && left_[(size_t) (pr.first * n_expert + pr.second)] != 0) { left_[(size_t) (pr.first * n_expert + pr.second)] = 0; ++beyond; }
                if (pr == last) past = true;
            }
        }
        left_count_ -= beyond;
    }
    if (pick.empty()) { err = "peer experts: no room or no expert left for the peer"; close(); return false; }
    // STRATA_EP_SPLIT_PREFILL: the stage's prompt MMQ reads a resident expert in its slot only when the slot has the
    // guard (ExpertCache::set_slot_guard, Prefill::slot_guard_bytes); it is NOT asked here because strata_prefill links
    // strata_engine (this file), so the call would be circular: the stage falls back to the gather for those experts
    // (slower, the same bits), as docs/EP_SPLIT_PREFILL.md s.10 says.  Not an allocation, not a behavior change.
    // Tappa 4b (s.13.2): the slots are sized a layer (open_sized), so `slot_offsets()` exists and generate.cpp's
    // `split_lend_slots` / `relayout` can take the tier's tail (the slots >= first: the pairs picked last here, the
    // coldest in rank order) as the stage's prompt buffers, as it does with the primary's cache.  The fill order
    // below is what makes the tail the cold end: slot 0 is the hottest pair the primary does not hold.
    const bool opened = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                   : cache_.open((int64_t) pick.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!opened) { err = "peer experts: " + err; close(); return false; }
    // open() zeroes the arena with cudaMemset on the legacy stream, which is not ordered against the non-blocking
    // refill stream: wait for it, or the zeroing can land on top of the fills
    if (!ck(cudaDeviceSynchronize(), "arena zeroing", err)) { close(); return false; }
    // mapped reads: advise the next kAhead pairs so their reads overlap
    constexpr size_t kAhead = 256;
    const bool advise = src.advise_pairs(pick.data(), (int64_t) std::min(kAhead, pick.size()));
    for (size_t i = 0; i < pick.size(); ++i) {
        const auto& pr = pick[i];
        if (advise && i + kAhead < pick.size()) (void) src.advise_pairs(&pick[i + kAhead], 1);
        const int32_t slot = cache_.admit(pr.first, pr.second);
        const uint8_t* b = src.blob(pr.first, pr.second);
        if (slot < 0 || b == nullptr || !cache_.fill_slot(slot, b, refill_, err, (int64_t) lay.blob_bytes(pr.first))) {
            err = "peer experts: fill failed: " + err;
            close();
            return false;
        }
        src.note_async_read(b, refill_);   // #1237
        res_[(size_t) (pr.first * n_expert + pr.second)] = slot;
    }
    if (!ck(cudaStreamSynchronize(refill_), "fill", err)) { close(); return false; }
    const auto& f0 = pick.front();
    if (!cache_.verify_slot(res_[(size_t) (f0.first * n_expert + f0.second)], src.blob(f0.first, f0.second), err,
                            (int64_t) lay.blob_bytes(f0.first))) {
        err = "peer experts: " + err;
        close();
        return false;
    }
    resident_ = (int64_t) pick.size();
    // STRATA_PEER_FUSED: the persistent kernel's shape, page and buffers (still under the peer's device).  A pack the
    // fused kernel is not built for keeps the per-layer path with a line on stderr; a CUDA error closes the tier.
    if (fused_env() && !fused_open(err)) { close(); return false; }
    return true;
}

bool PeerExperts::launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                         const int32_t* kind, std::string& err, float* out, int32_t mine) {
    const bool direct = peer_direct() && out != nullptr;
    launched_direct_ = false;
    launched_rows_ = 0;
    row_of_.clear();
    const int64_t n = n_tok * k;
    if (n > CAP) { err = "peer experts: window too large"; return false; }
    const auto t0 = std::chrono::steady_clock::now();
    struct Timed {   // the host time of this call, whichever way it returns
        const std::chrono::steady_clock::time_point t0;
        double& ms;
        ~Timed() { ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    } timed{t0, ms_launch};
    Meta& m = *(Meta*) h_meta_;
    int groups = 0, rows = 0;
    // distinct experts in routing order; each one's entries become compact rows (`mine`: this server's kind code,
    // 2 for the first server as always, 3 for the second)
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != mine) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == mine && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        // STRATA_PEER_FUSED: inside a window the slot comes from the window's table (the one has() answered from)
        const int32_t slot = window_open_ ? own_win_[(size_t) (layer * n_expert_ + ids[i])]
                                          : res_[(size_t) (layer * n_expert_ + ids[i])];
        if (slot < 0) { err = "peer experts: an entry was planned for the peer but is not resident"; return false; }
        m.ptr[groups] = (unsigned long long) cache_.device_slot(slot);
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == mine && ids[j] == ids[i]) {
                m.dst[rows] = rows;
                m.tok[rows] = (int32_t) (j / k);
                m.out_row[rows] = (int32_t) j;
                row_of_.push_back((int32_t) j);
                ++rows;
            }
        ++groups;
        ++experts_;
    }
    if (window_open_) {
        // STRATA_PEER_FUSED: the kernel on the peer reads the same ids from the mapped page, plans with the same
        // table and writes the rows itself: here only the host's count of the step, for fused_end's check against
        // the kernel's `cnt[s]`.  No copy, no launch, nothing on the stream.  The plan above went into h_meta_ as
        // always; nothing reads it.
        if (out == nullptr) { err = "peer experts: fused: the step has no host rows for the kernel to write"; return false; }
        if (step_ >= win_steps_) {
            err = "peer experts: fused: step " + std::to_string(step_) + " is past the window's " + std::to_string(win_steps_);
            return false;
        }
        expect_[(size_t) step_] = ((uint32_t) groups << 16) | (uint32_t) rows;
        if (groups > 0) {
            m.start[groups] = rows;
            m.count[0] = groups;
            m.count[1] = rows;
            entries_ += rows;
            ++launches;
        } else if (skip_ && h_empty_ != nullptr) {
            // s.11.1 (STRATA_PEER2_SKIP): nothing of this server at this step - said to the primary's graph node (and to
            // finish() under HOSTWAIT) with the step's ring in the empty word, so nobody waits the kernel's round trip
            // over the link for a flag that carries no rows.  The kernel still serves the step (cnt[s] = 0, its flag).
            store_u32(h_empty_ + step_, (uint32_t) step_ + 1u);
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        launched_rows_ = rows;
        launched_direct_ = true;   // the kernel writes the mapped rows: the pool must not zero them
        ++step_;
        return true;
    }
    if (groups == 0) return true;
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    entries_ += rows;
    launched_rows_ = rows;
    ++launches;
    // direct: x is the verifier's pinned (portable) doorbell row block - copied from where it is
    // (STRATA_PEER_ONE_COPY: into the shared block first, so the plan and the rows go up in one copy)
    const bool one = h_stage_ != nullptr;
    if (!direct || one) std::memcpy(h_x_, x, (size_t) (n_tok * H) * sizeof(float));
    On on(device_);
    const cudaStream_t s = stream_;
    // STRATA_PEER_STAMP: the layer's GPU span starts here, before the first copy (the stamp is a one-thread kernel)
    if (m_stamp_ != nullptr) { stamp_layer_ = layer; strata::kernels::gpu_stamp(m_stamp_, (int) (layer * 2), s); }
    if (one) {
        if (!ck(cudaMemcpyAsync(d_stage_, h_stage_, kMetaPad + (size_t) (n_tok * H) * sizeof(float), cudaMemcpyHostToDevice, s), "plan + input", err))
            return false;
    } else if (!ck(cudaMemcpyAsync(d_x_, direct ? x : h_x_, (size_t) (n_tok * H) * sizeof(float), cudaMemcpyHostToDevice, s), "input", err) ||
               !ck(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(Meta), cudaMemcpyHostToDevice, s), "plan", err)) {
        return false;
    }
    Meta* dm = (Meta*) d_meta_;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        const auto& f = lay.fmt[(size_t) layer];
        // APR: Hadamard-folded gate/up take H (s * x); d_x_ is this device's private copy, rotated in place
        if (f.had_gu) strata::kernels::hadamard_rows(d_x_, d_x_, n_tok, H, f.had_block, d_sx_, s);
        strata::kernels::quantize_q8_1_rows(d_x_, n_tok, H, d_q8_, s);
        auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        if (f.had_d) { L.had_block = f.had_block; L.d_signs = d_sh_; }
        strata::kernels::native_expert_grouped(L, dm->ptr, dm->start, dm->count, dm->dst, dm->tok, groups, rows, d_q8_,
                                               d_scratch_, d_out_, s);
    } else {
        err = "peer experts: only native packs are supported";
        return false;
    }
    if (direct) {   // the rows straight into the host `out` (zero-copy, coalesced float4 writes)
        try {
            strata::kernels::scatter_rows_f32(d_out_, out, dm->out_row, rows, H, s);
        } catch (const std::exception& e) { err = std::string("peer experts: scatter: ") + e.what(); return false; }
        launched_direct_ = true;
        if (m_stamp_ != nullptr) strata::kernels::gpu_stamp(m_stamp_, (int) (layer * 2 + 1), s);   // after the last kernel
        return true;
    }
    if (!ck(cudaMemcpyAsync(h_out_, d_out_, (size_t) rows * H * sizeof(float), cudaMemcpyDeviceToHost, s), "results", err))
        return false;
    if (m_stamp_ != nullptr) strata::kernels::gpu_stamp(m_stamp_, (int) (layer * 2 + 1), s);
    return true;
}

bool PeerExperts::finish(float* out, std::string& err) {
    if (window_open_) {
        // STRATA_PEER_FUSED: the kernel raises flag_out after the step's rows and the primary's graph waits on it
        // (wait_flag_ge on flag_out_dev0): nothing to do here.  STRATA_PEER_FUSED_HOSTWAIT=1 (bring-up): no graph
        // node, so the pool thread waits for the flag before it raises its own.
        launched_rows_ = 0;
        if (!fused_hostwait()) return true;
        // s.11.1 (STRATA_PEER2_SKIP): an empty step is not waited for here either (the graph's node, when there is one,
        // passes on the empty word): the step's rows are nobody's
        if (skip_ && step_ > 0 && expect_[(size_t) step_ - 1] == 0) return true;
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t want = (uint32_t) step_;   // the step just counted is ring step_ (= s + 1)
        for (unsigned spins = 0;; ++spins) {
            if (load_u32(h_flag_out_) >= want) break;   // UINT32_MAX from release_gpu_waits passes too: the verifier is failing the window
            if ((spins & 1023u) == 0u) {
                const uint32_t ew = load_u32(h_err_);
                if (ew != 0) {
                    err = "peer experts: fused: the kernel left with error " + std::to_string(ew) + " (" + fused_err_name(ew) +
                          ") before step " + std::to_string(step_ - 1);
                    return false;
                }
                if (load_u32(h_abort_) != 0) { err = "peer experts: fused: the window was aborted"; return false; }
                const auto el = std::chrono::steady_clock::now() - t0;
                if ((unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(el).count() > wait_ns_) {
                    err = "peer experts: fused: no flag from the kernel for step " + std::to_string(step_ - 1) + " within " +
                          std::to_string(wait_ns_ / 1000000ull) + " ms (STRATA_PEER_FUSED_HOSTWAIT)";
                    return false;
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }
    if (launched_rows_ == 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    // spin on the stream (a blocking sync would sleep the pool thread and wake it late)
    cudaError_t e = cudaErrorNotReady;
    if (noswitch()) {   // STRATA_PEER_NOSWITCH: the query under the calling thread's device (the primary's)
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (e != cudaSuccess) noswitch_refuse(e);   // asked again below, under the peer's device
    }
    if (e != cudaSuccess) {
        On on(device_);
        while ((e = cudaStreamQuery(stream_)) == cudaErrorNotReady) {
        }
        if (!ck(e, "finish", err)) return false;
    }
    ms_wait += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (h_stamp_ != nullptr && stamp_layer_ >= 0) {   // STRATA_PEER_STAMP: the stream is done, both stamps have landed
        const volatile unsigned long long* st = h_stamp_ + stamp_layer_ * 2;
        const unsigned long long a = st[0], b = st[1];
        if (b >= a) {
            const float us = (float) ((double) (b - a) * 1e-3);
            spans_us_.push_back(us);
            span_us_total += us;
        }
        stamp_layer_ = -1;
    }
    if (launched_direct_) { launched_rows_ = 0; return true; }
    for (size_t r = 0; r < row_of_.size(); ++r)
        std::memcpy(out + (size_t) row_of_[r] * H, h_out_ + r * H, (size_t) H * sizeof(float));
    launched_rows_ = 0;
    return true;
}

bool PeerExperts::adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err, const PeerExperts* other) {
    // STRATA_PEER_FUSED: the window's kernel may be reading any slot right now; the swaps wait for the next gap
    if (window_open_) return true;
    // STRATA_EP_SPLIT_PREFILL tappa 4b: the tier's tail holds the stage's prompt buffers (lend_guard): no slot is
    // free to swap into, and res_ is the prompt's to write; the swaps wait for the refill.  true, nothing queued.
    if (lent_.load(std::memory_order_acquire)) return true;
    if (!pending_.empty() || max_swaps <= 0) return true;
    const auto t0 = std::chrono::steady_clock::now();
    struct Timed {   // the host time of this call (the decode's adapt thread), whichever way it returns
        const std::chrono::steady_clock::time_point t0;
        double& ms;
        ~Timed() { ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    } timed{t0, ms_adapt};
    const auto& lay = strata::kernels::cpu::expert_layout();
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const float* u = usage + l * n_expert_;
        const int32_t* r0 = res0 + l * n_expert_;
        const int32_t* r1 = res_.data() + l * n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            if (r1[e] >= 0) vict.emplace_back(u[e], e);
            // two servers: a pair the other server holds or is bringing in is no candidate (one card a pair)
            else if (r0[e] < 0 && u[e] >= 2.0f && (other == nullptr || !other->claims(l, e))) cand.emplace_back(u[e], e);
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return true;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    if ((int) swaps.size() > max_swaps) swaps.resize((size_t) max_swaps);
    On on(device_);
    for (const Swap& s : swaps) {
        const size_t in = (size_t) (s.layer * n_expert_ + s.in), out = (size_t) (s.layer * n_expert_ + s.out);
        const int32_t slot = res_[out];
        const uint8_t* b = src_->blob(s.layer, s.in);
        if (slot < 0 || b == nullptr) continue;
        if (!ck(cudaMemcpyAsync(cache_.device_slot(slot), b, (size_t) lay.blob_bytes(s.layer), cudaMemcpyHostToDevice,
                                refill_), "refill", err))
            return false;
        res_[out] = kNotResident;              // evicted now: the CPU computes it meanwhile
        pending_.emplace_back((int32_t) in, slot);
        ++swaps_;
    }
    if (!ck(cudaEventRecord(refill_ev_, refill_), "refill event", err)) return false;
    // STRATA_PEER_REFILL_FLUSH: under WDDM the copies sit in the driver's buffer until a call flushes the stream
    // (#2188); the query is that call, and the pool's poll_pending finds them landed during the next window
    if (refill_flush()) (void) cudaStreamQuery(refill_);
    return true;
}

void PeerExperts::apply_pending(bool wait) {
    // STRATA_EP_SPLIT_PREFILL tappa 4b: with the lend guard up pending_ is empty by contract (generate.cpp's lend
    // calls quiesce() -> apply_pending(true) BEFORE raising it, and adapt() queues nothing while it is up), so this
    // returns on the first line; no guard check here, on purpose: if the contract were ever broken, holding the
    // entries back would make them land on slots the refill has meanwhile given back to their own pairs.
    if (pending_.empty()) return;
    if (noswitch()) {   // STRATA_PEER_NOSWITCH: the event query works on any device's event
        const cudaError_t q = wait ? cudaEventSynchronize(refill_ev_) : cudaEventQuery(refill_ev_);
        if (q == cudaSuccess) { for (const auto& [i, slot] : pending_) res_[(size_t) i] = slot; pending_.clear(); return; }
        if (q != cudaErrorNotReady) noswitch_refuse(q);   // refused: asked again below under the peer's device
        else if (!wait) return;
    }
    {
        On on(device_);
        if (wait) cudaEventSynchronize(refill_ev_);
        else if (cudaEventQuery(refill_ev_) != cudaSuccess) return;
    }
    for (const auto& [i, slot] : pending_) res_[(size_t) i] = slot;
    pending_.clear();
}

// STRATA_EP_SPLIT_PREFILL (docs/EP_SPLIT_PREFILL.md s.6, s.13.2): before the stage reads the slots, and inside
// generate.cpp's `lend` before the relayout writes the tail and the guard goes up (tappa 4b).  The window check comes
// FIRST: a fused window's kernel lives on stream_ waiting for a doorbell, and synchronizing that stream would wait with
// it; the design never calls this with a window open (adapt and the fused server run between decode windows, the prompt
// after them), so an open window is reported, not waited for.  Then the swaps of an earlier adapt() land (res_ final,
// pending_ empty: what the lend guard relies on), and both streams drain: no copy into a slot in flight while the stage
// reads them or the relayout overwrites them.  No allocation, nothing swapped.
bool PeerExperts::quiesce(std::string& err) {
    if (device_ < 0) return true;   // no tier: nothing to quiesce
    if (window_open_.load(std::memory_order_acquire)) {
        err = "peer experts: a fused window is still open";
        return false;
    }
    apply_pending(true);
    On on(device_);
    if (!ck(cudaStreamSynchronize(stream_), "quiesce: stream", err)) return false;
    if (!ck(cudaStreamSynchronize(refill_), "quiesce: refill stream", err)) return false;
    return true;
}

// ====================================================================================================================
// STRATA_PEER_FUSED=1 (docs/PEER_FUSED.md s.6): the peer's expert server as one persistent kernel a window.  Everything
// below runs only when fused_on_ (fused_open said yes); off, none of it is reached and nothing is allocated.
// ====================================================================================================================

bool PeerExperts::hook_begin(void* ctx, const peer_fused::WindowHost& w, std::string& err) {
    return static_cast<PeerExperts*>(ctx)->fused_begin(w, err);
}
bool PeerExperts::hook_end(void* ctx, std::string& err) { return static_cast<PeerExperts*>(ctx)->fused_end(err); }
void PeerExperts::hook_flush(void* ctx) { static_cast<PeerExperts*>(ctx)->fused_flush(); }
void PeerExperts::hook_abort(void* ctx) { static_cast<PeerExperts*>(ctx)->fused_abort(); }

/// open()'s fused part, under the peer's device.  Resolves the kernel's shape for the pack (one n_embd / n_ff, the
/// gate/up and down types free to differ a layer: the kernel picks the body from the layer's layout in VRAM),
/// allocates the mapped page, the VRAM tables and the layout table, fills the slot pointers and the hook.  A pack or
/// a setting the fused kernel does not cover is refused with one line on stderr and `true` (the per-layer path
/// stays); only a CUDA error returns false.
bool PeerExperts::fused_open(std::string& err) {
    using namespace peer_fused;
    fused_on_ = false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    auto refuse = [this](const std::string& why) {
        std::fprintf(stderr, "strata: STRATA_PEER_FUSED: the fused expert server is not available for this run on GPU %d (%s): "
                             "the peer keeps the per-layer path\n", device_, why.c_str());
        return true;
    };
    // two servers (docs/PEER_FUSED2.md s.5): the second server's kernel can be left off by itself - its card may drive
    // the desktop, where the per-layer path's short kernels are the safe choice - while the first keeps its own
    const bool second = ordinal_ > 0;
    if (second && env_int(kEnvOn2, 1) == 0) return refuse(std::string(kEnvOn2) + "=0: the second server keeps the per-layer path");
    if (!lay.native) return refuse("not a native pack");
    if (!peer_direct()) return refuse("STRATA_PEER_DIRECT=0: the kernel writes the mapped rows itself");
    if ((int64_t) lay.fmt.size() < n_layers_) return refuse("the pack's layout has fewer layers than the model");
    if (n_layers_ <= 0) return refuse("no layers");
    // the knobs, read once here (the names are the contract's)
    const int minb = env_int(kEnvMinb, kMinBlocksDefault);
    // (the second server's cap of its own, STRATA_PEER2_FUSED_BLOCKS, so a card that drives the desktop can be left
    // SMs for it; unset, the common cap)
    const int blocks_cap = second ? env_int(kEnvBlocks2, env_int(kEnvBlocks, 0)) : env_int(kEnvBlocks, 0);
    const bool coop = env_int(kEnvCoop, 1) != 0;
    // the second server's waits are bounded under Windows' 2 s TDR by default (STRATA_PEER2_WAIT_MS): a kernel on the
    // desktop's card that outlives the timeout takes the display driver down with it; the first server keeps the
    // 20 s of the host's own layer wait
    const int wait_ms = second ? env_int(kEnvWaitMs2, kWaitMs2Default) : env_int(kEnvWaitMs, 20000);
    wait_ns_ = wait_ms > 0 ? (unsigned long long) wait_ms * 1000000ull : kWaitNsDefault;
    // s.11: the skip of the empty steps (on by default on the second server, whose link is the slow one; off on the
    // first) and the staged q8_1 rows (the second server only, off by default)
    skip_ = env_int(second ? kEnvSkip2 : kEnvSkip, second ? 1 : 0) != 0;
    stage_ = second && env_int(kEnvStage2, 0) != 0;
    // one layout a layer (s.3, d_lay_): the geometry must be the pack's one (the buffers are sized once), the types
    // are the layer's own; peer_fused_shape checks every layer against the kernel's bodies and this device's dispatch
    fused_lays_.assign((size_t) n_layers_, strata::kernels::NativeExpertLayout{});
    for (int64_t l = 0; l < n_layers_; ++l) {
        const auto& f = lay.fmt[(size_t) l];
        if (f.had_gu || f.had_d)
            return refuse("layer " + std::to_string(l) + " is Hadamard-folded (APR): hadamard_rows is not in the kernel");
        const auto L = strata::kernels::native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
        if (L.n_embd != H) return refuse("layer " + std::to_string(l) + ": n_embd " + std::to_string((long long) L.n_embd));
        if (L.bytes == 0 || L.gu_row == 0 || L.d_row == 0)
            return refuse("layer " + std::to_string(l) + ": no native layout for gate/up " + std::to_string(f.gu_type) + ", down " + std::to_string(f.d_type));
        fused_lays_[(size_t) l] = L;
    }
    {
        std::string why;
        if (!strata::kernels::peer_fused_shape(fused_lays_.data(), (int) n_layers_, minb, blocks_cap, coop, shape_, why))
            return refuse(why);
    }
    fused_L_ = fused_lays_[0];   // the geometry (n_embd, n_ff: the same on every layer, checked by peer_fused_shape)
    if (shape_.blocks <= 0 || shape_.blocks_per_sm <= 0) return refuse("the kernel has no resident blocks on this device");
    if (fused_L_.n_ff % 32 != 0) return refuse("n_ff " + std::to_string((long long) fused_L_.n_ff) + " is not a multiple of 32");
    if (hq_offset(CAP, fused_L_.n_ff) + q8_bytes(CAP, fused_L_.n_ff) > strata::kernels::native_expert_scratch_bytes(CAP, fused_L_.n_ff))
        return refuse("the hq plane does not fit the scratch native_expert_grouped sizes");

    // ---- the mapped page (s.2b): host view, the peer's view (the kernel), the primary's view of flag_out (the graph)
    if (!ck(cudaHostAlloc((void**) &h_page_, kPageBytes, cudaHostAllocPortable | cudaHostAllocMapped), "fused page", err)) return false;
    std::memset(h_page_, 0, kPageBytes);
    uint8_t* m_page = nullptr;
    if (!ck(cudaHostGetDevicePointer((void**) &m_page, h_page_, 0), "fused page device pointer", err)) return false;
    h_flag_out_ = (uint32_t*) (h_page_ + kOffFlagOut);   m_flag_out_ = (uint32_t*) (m_page + kOffFlagOut);
    h_abort_ = (uint32_t*) (h_page_ + kOffAbort);        m_abort_ = (uint32_t*) (m_page + kOffAbort);
    h_err_ = (uint32_t*) (h_page_ + kOffErr);            m_err_ = (uint32_t*) (m_page + kOffErr);
    h_exit_ = (uint32_t*) (h_page_ + kOffExit);          m_exit_ = (uint32_t*) (m_page + kOffExit);
    h_cnt_ = (uint32_t*) (h_page_ + kOffCnt);            m_cnt_ = (uint32_t*) (m_page + kOffCnt);
    h_stamp_fused_ = (unsigned long long*) (h_page_ + kOffStamp);
    m_stamp_fused_ = (unsigned long long*) (m_page + kOffStamp);
    // s.11: the empty words (host-written, primary-read) and the xq ring (primary-written, peer-read) live in the page
    h_empty_ = skip_ ? (uint32_t*) (h_page_ + kOffEmpty) : nullptr;
    m_empty_ = skip_ ? (const uint32_t*) (m_page + kOffEmpty) : nullptr;
    h_xq_seq_ = stage_ ? (uint32_t*) (h_page_ + kOffXqSeq) : nullptr;
    m_xq_seq_ = stage_ ? (uint32_t*) (m_page + kOffXqSeq) : nullptr;
    if (stage_) {   // the xq page: kMaxTok rows of n_embd/32 q8_1 blocks, written by the primary, read by the kernel
        if (!ck(cudaHostAlloc((void**) &h_xq_, q8_bytes(kMaxTok, fused_L_.n_embd), cudaHostAllocPortable | cudaHostAllocMapped), "fused xq page", err)) return false;
        std::memset(h_xq_, 0, q8_bytes(kMaxTok, fused_L_.n_embd));
        if (!ck(cudaHostGetDevicePointer((void**) &m_xq_, h_xq_, 0), "fused xq page device pointer", err)) return false;
    }
    {
        On on0(primary_device());
        uint8_t* p0 = nullptr;
        if (!ck(cudaHostGetDevicePointer((void**) &p0, h_page_, 0), "fused page device pointer (primary)", err)) return false;
        flag_out_dev0_ = (const uint32_t*) (p0 + kOffFlagOut);
        empty_dev0_ = skip_ ? (const uint32_t*) (p0 + kOffEmpty) : nullptr;
        xq_seq_dev0_ = stage_ ? (uint32_t*) (p0 + kOffXqSeq) : nullptr;
        if (stage_ && !ck(cudaHostGetDevicePointer((void**) &xq_dev0_, h_xq_, 0), "fused xq page device pointer (primary)", err)) return false;
    }

    // ---- VRAM (s.3) and the pinned ownership rows (s.4); d_q8_, d_scratch_ and d_out_ are the per-layer path's
    const size_t own_n = (size_t) (n_layers_ * n_expert_);
    const int64_t slots = cache_.slots();
    if (!ck(cudaMalloc((void**) &d_own_, own_n * sizeof(int32_t)), "fused ownership table", err) ||
        !ck(cudaMemset(d_own_, 0xFF, own_n * sizeof(int32_t)), "fused ownership table zeroing", err) ||   // every word -1
        !ck(cudaHostAlloc((void**) &h_own_, own_n * sizeof(int32_t), cudaHostAllocPortable), "fused ownership staging", err) ||
        !ck(cudaMalloc((void**) &d_slot_ptr_, (size_t) std::max<int64_t>(slots, 1) * sizeof(unsigned long long)), "fused slot table", err) ||
        !ck(cudaMalloc((void**) &d_plan_, 2 * sizeof(Plan)), "fused plan", err) ||
        !ck(cudaMemset(d_plan_, 0, 2 * sizeof(Plan)), "fused plan zeroing", err) ||
        !ck(cudaMalloc((void**) &d_sync_, sizeof(Sync)), "fused sync words", err) ||
        !ck(cudaMemset(d_sync_, 0, sizeof(Sync)), "fused sync words zeroing", err) ||
        !ck(cudaMalloc((void**) &d_lay_, fused_lays_.size() * sizeof(strata::kernels::NativeExpertLayout)), "fused layout table", err) ||
        !ck(cudaMemcpy(d_lay_, fused_lays_.data(), fused_lays_.size() * sizeof(strata::kernels::NativeExpertLayout), cudaMemcpyHostToDevice),
            "fused layout table copy", err))
        return false;
    std::memset(h_own_, 0xFF, own_n * sizeof(int32_t));
    {   // slot -> blob address: the slots do not move (adapt refills a victim's slot in place, the same layer)
        std::vector<unsigned long long> ptrs((size_t) std::max<int64_t>(slots, 1), 0ull);
        for (int64_t s = 0; s < slots; ++s) ptrs[(size_t) s] = (unsigned long long) cache_.device_slot((int32_t) s);
        if (!ck(cudaMemcpy(d_slot_ptr_, ptrs.data(), ptrs.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice), "fused slot table copy", err))
            return false;
    }
    // the memsets above ran on the legacy stream, which the non-blocking stream_ is not ordered against
    if (!ck(cudaDeviceSynchronize(), "fused tables", err)) return false;
    own_win_.assign(own_n, kNotResident);
    own_shadow_.assign(own_n, kNotResident);
    expect_.assign((size_t) kMaxSteps, 0u);
    for (auto& mp : map_cache_) mp = MappedPtr{};
    step_ = win_steps_ = 0;
    epoch_ = 0;
    window_open_ = false;

    // ---- the verifier's handle
    hook_ = Hook{};
    hook_.ctx = this;
    hook_.begin = &PeerExperts::hook_begin;
    hook_.end = &PeerExperts::hook_end;
    hook_.flush = &PeerExperts::hook_flush;
    hook_.abort = &PeerExperts::hook_abort;
    hook_.flag_out_host = h_flag_out_;
    hook_.flag_out_dev0 = fused_hostwait() ? nullptr : flag_out_dev0_;   // HOSTWAIT: no graph node, finish() waits
    // s.11: the skip's empty words go with the graph node (under HOSTWAIT finish() reads expect_ instead); the staged
    // rows need the primary's publish node, which the graph adds when it sees the page
    hook_.empty_dev0 = skip_ && hook_.flag_out_dev0 != nullptr ? empty_dev0_ : nullptr;
    hook_.xq_dev0 = stage_ ? xq_dev0_ : nullptr;
    hook_.xq_seq_dev0 = stage_ ? xq_seq_dev0_ : nullptr;
    fused_on_ = true;
    std::fprintf(stderr, "strata: STRATA_PEER_FUSED: the peer%s on GPU %d serves each window with one persistent kernel: %lld layers, gate/up types {%s}, "
                         "down types {%s} (n_embd %lld, n_ff %lld), minb %d, %d blocks (%d per SM x %d SMs), %zu B dynamic shared "
                         "(grid %zu B), %s launch, waits bounded at %d ms%s%s%s%s\n",
                 second ? " (second server)" : "", device_,
                 (long long) n_layers_, mask_text(shape_.gu_mask).c_str(), mask_text(shape_.d_mask).c_str(),
                 (long long) fused_L_.n_embd, (long long) fused_L_.n_ff, shape_.minb, shape_.blocks,
                 shape_.blocks_per_sm, shape_.sms, shape_.dyn_shared, shape_.grid_bytes, shape_.cooperative ? "cooperative" : "plain", wait_ms,
                 fused_hostwait() ? "; STRATA_PEER_FUSED_HOSTWAIT: the pool thread waits for the kernel's flag, no graph node" : "",
                 stamp() ? "; STRATA_PEER_STAMP: the spans come from the kernel's own stamps" : "",
                 skip_ ? (second ? "; STRATA_PEER2_SKIP: a step with no entry here does not hold the primary" : "; STRATA_PEER_SKIP: a step with no entry here does not hold the primary") : "",
                 stage_ ? "; STRATA_PEER2_STAGE: the primary publishes the step's rows as q8_1 (2 880 B a row instead of 10 240)" : "");
    return true;
}

/// Frees fused_open's allocations (the caller is under the peer's device and has drained the stream).
void PeerExperts::fused_close() {
    if (h_page_) cudaFreeHost(h_page_);
    if (h_xq_) cudaFreeHost(h_xq_);
    h_xq_ = m_xq_ = xq_dev0_ = nullptr;
    h_empty_ = nullptr;
    m_empty_ = nullptr;
    empty_dev0_ = nullptr;
    h_xq_seq_ = m_xq_seq_ = xq_seq_dev0_ = nullptr;
    skip_ = stage_ = false;
    if (h_own_) cudaFreeHost(h_own_);
    if (d_own_) cudaFree(d_own_);
    if (d_slot_ptr_) cudaFree(d_slot_ptr_);
    if (d_plan_) cudaFree(d_plan_);
    if (d_sync_) cudaFree(d_sync_);
    if (d_lay_) cudaFree(d_lay_);
    d_lay_ = nullptr;
    fused_lays_.clear();
    h_page_ = nullptr;
    h_flag_out_ = m_flag_out_ = h_abort_ = m_abort_ = h_err_ = m_err_ = h_exit_ = m_exit_ = h_cnt_ = m_cnt_ = nullptr;
    h_stamp_fused_ = m_stamp_fused_ = nullptr;
    flag_out_dev0_ = nullptr;
    d_own_ = nullptr;
    d_slot_ptr_ = nullptr;
    d_plan_ = nullptr;
    d_sync_ = nullptr;
    h_own_ = nullptr;
    own_win_.clear();
    own_shadow_.clear();
    expect_.clear();
    for (auto& mp : map_cache_) mp = MappedPtr{};
    hook_ = peer_fused::Hook{};
    host_res_fused_ = nullptr;
    other_res_fused_ = nullptr;
    window_open_ = false;
    fused_on_ = false;
    step_ = win_steps_ = 0;
}

/// cudaStreamQuery(stream_) without the device switch (the stream carries its context, as STRATA_PEER_NOSWITCH
/// relies on); if the driver refuses it, the same query under the peer's device.
cudaError_t PeerExperts::fused_query() {
    cudaError_t e = cudaStreamQuery(stream_);
    if (e == cudaSuccess || e == cudaErrorNotReady) return e;
    cudaGetLastError();
    On on(device_);
    return cudaStreamQuery(stream_);
}

/// Polls the stream until it is idle, fails, or `limit_s` passed (then cudaErrorNotReady).  A sleep between polls:
/// this waits for a kernel to leave, not for a layer's rows.
cudaError_t PeerExperts::fused_drain(double limit_s, double spin_s) {
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const cudaError_t e = fused_query();
        if (e != cudaErrorNotReady) return e;
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el > limit_s) return cudaErrorNotReady;
        // s.11.1: with the skip the kernel may still be serving the window's last EMPTY steps (microseconds) when the
        // primary's sync returns; fused_end asks for a busy stretch first, so that tail never costs a 1 ms sleep
        if (el < spin_s) continue;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

/// The peer's device pointer of one of the verifier's mapped pages (the caller is under the peer's device).  Cached
/// by host address: the pages move only when the verifier reallocates them.
bool PeerExperts::fused_map(const void* host, void** dev, std::string& err) {
    if (host == nullptr) { err = "peer experts: fused: the verifier handed a null mapped page"; return false; }
    for (auto& mp : map_cache_)
        if (mp.host == host) { *dev = mp.dev; return true; }
    void* d = nullptr;
    if (!ck(cudaHostGetDevicePointer(&d, const_cast<void*>(host), 0), "fused: a mapped page's device pointer", err)) return false;
    // the oldest entry goes (four pages, four entries: a reallocation replaces one)
    MappedPtr* slot = &map_cache_[0];
    for (auto& mp : map_cache_)
        if (mp.host == nullptr) { slot = &mp; break; }
    if (slot->host != nullptr) {
        for (size_t i = 1; i < sizeof map_cache_ / sizeof map_cache_[0]; ++i) map_cache_[i - 1] = map_cache_[i];
        slot = &map_cache_[sizeof map_cache_ / sizeof map_cache_[0] - 1];
    }
    *slot = MappedPtr{host, d};
    *dev = d;
    return true;
}

/// Before the window's graph launch (hook.begin): the ownership table, the words, the kernel.
bool PeerExperts::fused_begin(const peer_fused::WindowHost& w, std::string& err) {
    using namespace peer_fused;
    if (!fused_on_) { err = "peer experts: fused: the fused server is off"; return false; }
    // STRATA_EP_SPLIT_PREFILL tappa 4b: a window over slots that hold the stage's prompt buffers would read them as
    // experts; the design refills before the first window (generate.cpp), so this is an error, never a different text
    if (lent_.load(std::memory_order_acquire)) { err = "peer experts: tier slots are lent to the prompt path"; return false; }
    On on(device_);
    // (1) the previous window's kernel must have left: two persistent grids on one card would deadlock each other
    {
        const cudaError_t e = cudaStreamQuery(stream_);
        if (e == cudaErrorNotReady) { err = "peer experts: fused: the previous window's kernel is still running"; return false; }
        if (!ck(e, "fused: the stream before the window", err)) return false;
    }
    // the stream is idle, so a window still marked open was left so by a fused_abort / fused_end whose 5 s drain gave
    // up while the kernel was alive (window_open_ stays up until then, to keep adapt() off the slots it reads): the
    // kernel has left now, the window closes here
    window_open_ = false;
    // the geometry the page and the plan were sized for
    if (w.G < 1 || w.G > 2) { err = "peer experts: fused: " + std::to_string(w.G) + " row groups a layer (1 or 2)"; return false; }
    if (w.lb < 0 || w.le <= w.lb || w.le > n_layers_) {
        err = "peer experts: fused: layers [" + std::to_string(w.lb) + ", " + std::to_string(w.le) + ") of " + std::to_string(n_layers_);
        return false;
    }
    const int steps = (w.le - w.lb) * w.G;
    if (steps > kMaxSteps) { err = "peer experts: fused: " + std::to_string(steps) + " steps a window, the page holds " + std::to_string(kMaxSteps); return false; }
    if (w.T <= 0 || w.T > kMaxTok) { err = "peer experts: fused: " + std::to_string(w.T) + " rows a window (at most " + std::to_string(kMaxTok) + ")"; return false; }
    if (w.k <= 0 || (int64_t) w.T * w.k > CAP) { err = "peer experts: fused: " + std::to_string((long long) w.T * w.k) + " entries a step (at most " + std::to_string((long long) CAP) + ")"; return false; }
    if (w.n_embd != fused_L_.n_embd) { err = "peer experts: fused: the verifier's n_embd " + std::to_string((long long) w.n_embd) + " is not the kernel's " + std::to_string((long long) fused_L_.n_embd); return false; }
    for (int g = 0; g < w.G; ++g)
        if (w.gtb[g] < 0 || w.gte[g] < w.gtb[g] || w.gte[g] > w.T) { err = "peer experts: fused: group " + std::to_string(g) + " rows [" + std::to_string(w.gtb[g]) + ", " + std::to_string(w.gte[g]) + ") of " + std::to_string(w.T); return false; }
    {   // the window's layers are the ones the layout table (d_lay_) was built from at open: the pack does not change
        // under a run, this only says so if it ever did
        const auto& lay = strata::kernels::cpu::expert_layout();
        if ((int64_t) fused_lays_.size() < w.le) { err = "peer experts: fused: the layout table is shorter than the window"; return false; }
        for (int l = w.lb; l < w.le; ++l) {
            const auto& f = lay.fmt[(size_t) l];
            const auto& L = fused_lays_[(size_t) l];
            if (f.gu_type != L.gu_type || f.d_type != L.d_type || f.n_embd != L.n_embd || f.n_ff != L.n_ff || f.had_gu || f.had_d) {
                err = "peer experts: fused: layer " + std::to_string(l) + " has another layout than the kernel's table (gate/up " +
                      std::to_string(f.gu_type) + ", down " + std::to_string(f.d_type) + ", n_ff " + std::to_string((long long) f.n_ff) + ")";
                return false;
            }
        }
    }
    // (2) the window's ownership (s.4): the peer computes what it holds and the primary does not; the device gets
    // only the layers whose row changed since the last window (the swaps of adapt), from the pinned rows
    const size_t own_n = (size_t) (n_layers_ * n_expert_);
    if (other_res_fused_ == nullptr) {
        for (size_t i = 0; i < own_n; ++i)
            own_win_[i] = (res_[i] >= 0 && (host_res_fused_ == nullptr || host_res_fused_[i] < 0)) ? res_[i] : kNotResident;
    } else {
        // two servers, the second: a pair the first server holds is the first's (the dispatch asks it before this
        // one), so it is left out here as the primary's are - the two kernels never plan the same entry.  The first's
        // table is read at the same moment the first's own fused_begin reads it (both begins run back to back on the
        // verifier's thread before the graph launch), so the two window tables agree.
        const int32_t* other = other_res_fused_;
        for (size_t i = 0; i < own_n; ++i)
            own_win_[i] = (res_[i] >= 0 && (host_res_fused_ == nullptr || host_res_fused_[i] < 0) && other[i] < 0) ? res_[i] : kNotResident;
    }
    const size_t row_bytes = (size_t) n_expert_ * sizeof(int32_t);
    for (int64_t l = 0; l < n_layers_; ++l) {
        const size_t o = (size_t) (l * n_expert_);
        if (std::memcmp(own_win_.data() + o, own_shadow_.data() + o, row_bytes) == 0) continue;
        std::memcpy(h_own_ + o, own_win_.data() + o, row_bytes);
        if (!ck(cudaMemcpyAsync(d_own_ + o, h_own_ + o, row_bytes, cudaMemcpyHostToDevice, stream_), "fused: ownership row", err)) return false;
        std::memcpy(own_shadow_.data() + o, own_win_.data() + o, row_bytes);
    }
    // (3) the words: the page by the host (volatile stores, then a fence: the kernel reads them through the PCIe
    // aperture), the kernel's own words by a memset on its stream, before the launch
    store_u32(h_flag_out_, 0);
    store_u32(h_abort_, 0);
    store_u32(h_err_, 0);
    store_u32(h_exit_, 0);
    for (int s = 0; s < steps; ++s) {
        store_u32(h_cnt_ + s, 0);
        *(volatile unsigned long long*) (h_stamp_fused_ + 2 * s) = 0;
        *(volatile unsigned long long*) (h_stamp_fused_ + 2 * s + 1) = 0;
        if (h_empty_ != nullptr) store_u32(h_empty_ + s, 0);   // s.11.1: no step is empty until launch() says so
    }
    if (h_xq_seq_ != nullptr) store_u32(h_xq_seq_, 0);        // s.11.2: no q8_1 rows published yet
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!ck(cudaMemsetAsync(d_sync_, 0, sizeof(Sync), stream_), "fused: sync words", err)) return false;
    // (4) the verifier's pages as the peer sees them
    void *seq_d = nullptr, *x_d = nullptr, *ids_d = nullptr, *ymiss_d = nullptr;
    if (!fused_map(w.seq_h, &seq_d, err) || !fused_map(w.x_h, &x_d, err) || !fused_map(w.ids_h, &ids_d, err) ||
        !fused_map(w.ymiss_h, &ymiss_d, err))
        return false;
    // (5) the launch argument
    Window win;
    win.steps = steps;
    win.G = w.G;
    win.lb = w.lb;
    win.T = w.T;
    win.k = w.k;
    for (int g = 0; g < 2; ++g) { win.gtb[g] = w.gtb[g]; win.gte[g] = w.gte[g]; }
    win.n_embd = fused_L_.n_embd;
    win.n_ff = fused_L_.n_ff;
    win.n_expert = n_expert_;
    win.epoch = ++epoch_;
    win.nblocks = shape_.blocks;
    win.wait_ns = wait_ns_;
    win.seq = (const uint32_t*) seq_d;
    win.x = (const float*) x_d;
    win.ids = (const int32_t*) ids_d;
    win.ymiss = (float*) ymiss_d;
    win.flag_out = m_flag_out_;
    win.abort = m_abort_;
    win.err = m_err_;
    win.exit = m_exit_;
    win.cnt = m_cnt_;
    win.stamp = m_stamp_fused_;
    win.own = d_own_;
    win.slot_ptr = d_slot_ptr_;
    win.plan = d_plan_;
    win.q8 = d_q8_;
    win.hq = (uint8_t*) d_scratch_ + hq_offset(CAP, fused_L_.n_ff);   // where native_expert_grouped puts hq
    win.out = d_out_;
    win.sync = d_sync_;
    win.lay = d_lay_;                    // one layout a layer: the kernel picks each step's bodies from it
    win.n_layers = (int) n_layers_;
    // s.11: with the skip the primary rings ahead of the flags on empty steps (the kernel guards its own double buffer);
    // with the stage the kernel copies the primary's q8_1 rows from the xq page.  Off: 0 / null, the kernel of before.
    win.ahead = skip_ ? 1 : 0;
    win.empty = skip_ ? m_empty_ : nullptr;   // the kernel reads a step's ids, then this word: raised = the host found it empty
    win.xq = stage_ ? (const void*) m_xq_ : nullptr;
    win.xq_seq = stage_ ? (const uint32_t*) m_xq_seq_ : nullptr;
    // (6) one launch a window, then a query so WDDM hands it to the card now instead of at the next driver call
    std::string lerr;
    if (!strata::kernels::peer_fused_launch(win, shape_, stream_, lerr)) {
        err = "peer experts: fused: launch: " + lerr;
        return false;
    }
    {
        const cudaError_t q = cudaStreamQuery(stream_);
        if (q != cudaSuccess && q != cudaErrorNotReady && !ck(q, "fused: the stream after the launch", err)) return false;
    }
    window_open_ = true;
    step_ = 0;
    win_steps_ = steps;
    win_G_ = w.G;
    win_lb_ = w.lb;
    return true;
}

/// After the window's cudaStreamSynchronize (hook.end): the kernel has left, no error, every step's count agrees.
bool PeerExperts::fused_end(std::string& err) {
    using namespace peer_fused;
    if (!fused_on_) return true;
    const bool was_open = window_open_;
    // (s.11.1: with the skip the kernel may be a few empty steps behind the primary's sync: a busy 500 us first)
    const cudaError_t e = fused_drain(5.0, skip_ ? 0.0005 : 0.0);
    // from here has / launch / finish are the per-layer path again - unless the kernel is still alive: then the window
    // stays open (adapt() must not refill the slots it reads) and the verifier's guard aborts it; the next begin
    // closes it once the stream is idle
    if (e != cudaErrorNotReady) window_open_ = false;
    if (!was_open) return true;
    ++fused_windows_;
    fused_steps_ += step_;
    if (e == cudaErrorNotReady) { err = "peer experts: fused: the window's kernel is still running 5 s after the window ended"; return false; }
    if (!ck(e, "fused: the window's kernel", err)) return false;
    std::atomic_thread_fence(std::memory_order_acquire);
    const uint32_t ew = load_u32(h_err_);
    if (ew != 0) {
        // the step the error hit: the first whose flag stamp is missing (flag_out itself is UINT32_MAX after an error
        // exit, and two steps' flags may land out of order)
        int s = -1;
        for (int i = 0; i < win_steps_; ++i)
            if (*(const volatile unsigned long long*) (h_stamp_fused_ + 2 * i + 1) == 0) { s = i; break; }
        err = "peer experts: fused: the kernel left with error " + std::to_string(ew) + " (" + fused_err_name(ew) + ")";
        if (s >= 0) err += " at step " + std::to_string(s) + " (layer " + std::to_string(win_lb_ + s / win_G_) + ", group " + std::to_string(s % win_G_) + ")";
        return false;
    }
    const uint32_t ex = load_u32(h_exit_);
    if (ex != epoch_) {
        err = "peer experts: fused: the kernel did not report its exit (exit word " + std::to_string(ex) + ", epoch " + std::to_string(epoch_) + ")";
        return false;
    }
    if (step_ != win_steps_) {
        err = "peer experts: fused: the host served " + std::to_string(step_) + " of the window's " + std::to_string(win_steps_) + " steps";
        return false;
    }
    for (int s = 0; s < step_; ++s) {
        const uint32_t dev = load_u32(h_cnt_ + s), host = expect_[(size_t) s];
        if (dev == host) continue;
        err = "peer experts: fused: plan mismatch at step " + std::to_string(s) + " (layer " + std::to_string(win_lb_ + s / win_G_) +
              "): device " + std::to_string(dev >> 16) + " x " + std::to_string(dev & 0xFFFFu) + ", host " +
              std::to_string(host >> 16) + " x " + std::to_string(host & 0xFFFFu);
        return false;
    }
    // the kernel's stamps: doorbell seen -> flag raised, a span a step (the work, without the submission gaps)
    for (int s = 0; s < step_; ++s) {
        const unsigned long long a = *(const volatile unsigned long long*) (h_stamp_fused_ + 2 * s);
        const unsigned long long b = *(const volatile unsigned long long*) (h_stamp_fused_ + 2 * s + 1);
        if (a == 0 || b < a) continue;
        const float us = (float) ((double) (b - a) * 1e-3);
        spans_us_.push_back(us);
        span_us_total += us;
    }
    return true;
}

/// From the verifier's layer-wait loop (hook.flush): under WDDM the peer's stream sits in the driver's buffer until a
/// call touches it; the query is that call.
void PeerExperts::fused_flush() {
    if (!fused_on_) return;
    (void) fused_query();
}

/// From release_gpu_waits and close() (hook.abort): the kernel leaves at its next poll; the stream is drained within
/// 5 s.  The window closes when the kernel has left (the per-layer path serves until the next begin); if the drain
/// gives up it stays open - a kernel still on the card reads the slots, so adapt() must keep off them - and the next
/// fused_begin closes it once the stream is idle (close() drains the stream before it frees anything).
void PeerExperts::fused_abort() {
    if (!fused_on_ || h_abort_ == nullptr) return;
    store_u32(h_abort_, 1);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const cudaError_t e = fused_drain(5.0);
    if (e == cudaErrorNotReady) {
        std::fprintf(stderr, "strata: STRATA_PEER_FUSED: the peer's kernel did not leave within 5 s of the abort: the window stays "
                             "open (no slot refill) until its stream is idle\n");
        return;
    }
    if (e != cudaSuccess)
        cudaGetLastError();   // the stream's own error: the verifier reports the window; the next begin asks the stream again
    window_open_ = false;
}

}  // namespace strata::core
