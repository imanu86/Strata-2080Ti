// src/core/ep_split.cpp - see include/strata/core/ep_split.hpp (STRATA_EP_SPLIT_PREFILL, docs/EP_SPLIT_PREFILL.md).
//
// The C piece of docs/EP_SPLIT_PREFILL.md s.9: the knobs, the admission (s.7), the state mover (s.5), the sync
// bookkeeping, the mid-prompt checkpoint merge (s.6) and the one-line forms.  Plain host code and cudaMemcpyAsync:
// no kernel of its own (the only device call besides the copies is kv_stream_reset, a kernel the K/V streaming
// already has).  Written, not compiled (docs/EP_SPLIT_PREFILL.md s.11).
#include "strata/core/ep_split.hpp"

#include "strata/core/conversation_snapshot.hpp"   // conversation_kv_part_sizes, conversation_state_sizes, conversation_session_sizes
#include "strata/core/geometry.hpp"                // is_qsa_layer
#include "strata/core/on_device.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/kv_stream.hpp"            // KvHostPools, kv_stream_reset
#include "strata/kernels/qsa.hpp"                  // qsa_real_shapes

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

namespace strata::core::ep_split {

// ---------------------------------------------------------------- knobs (read once)

namespace {

bool env_flag(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::atoi(v) != 0;
}

/// `name` as an integer, `def` when unset or not a number.
int64_t env_i64(const char* name, int64_t def) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return def;
    char* end = nullptr;
    const long long x = std::strtoll(v, &end, 10);
    return end == v ? def : (int64_t) x;
}

constexpr uint64_t kMiB = (uint64_t) 1 << 20;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool cuda_fail(std::string& err, const char* what, cudaError_t e) {
    err = std::string("ep split: ") + what + ": " + cudaGetErrorString(e);
    (void) cudaGetLastError();   // the sticky error of this call is reported once, here
    return false;
}

bool fail(std::string& err, const char* what) {
    err = std::string("ep split: ") + what;
    return false;
}

}  // namespace

bool env_on() {
    static const bool on = env_flag(kEnvOn);
    return on;
}

int64_t env_layer() {
    static const int64_t k = env_i64(kEnvLayer, kLayerDefault);
    return k;
}

size_t env_piece_bytes() {
    static const size_t piece = [] {
        const int64_t mib = env_i64(kEnvPieceMib, -1);
        if (mib < 0) return kPieceBytesDefault;
        return (size_t) std::max<int64_t>(mib, 1) << 20;   // floor 1 MiB
    }();
    return piece;
}

int env_threads() {
    static const int n = [] {
        const int64_t t = env_i64(kEnvThreads, kThreadsDefault);
        return (int) std::min<int64_t>(std::max<int64_t>(t, 1), 16);
    }();
    return n;
}

bool env_trace() {
    static const bool on = env_flag(kEnvTrace);
    return on;
}

int64_t env_reserve_mib() {
    static const int64_t mib = [] {
        const int64_t m = env_i64(kEnvReserveMib, -1);
        return m < 0 ? (int64_t) -1 : m;
    }();
    return mib;
}

// ---- tappa 4c (s.14): read once; meaningful at level 2 only (generate.cpp asks them behind env_level() == 2)

int env_level() {
    static const int level = (int) std::max<int64_t>(0, env_i64(kEnvOn, 0));
    return level;
}

int64_t env_min_tokens() {
    static const int64_t n = std::max<int64_t>(0, env_i64(kEnvMinTokens, kMinTokensDefault));
    return n;
}

bool env_dense_loan() {
    static const bool on = env_i64(kEnvDenseLoan, 1) != 0;
    return on;
}

bool env_overlap() {
    static const bool on = env_i64(kEnvOverlap, 1) != 0;
    return on;
}

bool env_dense_sync() {
    static const bool on = env_flag(kEnvDenseSync);
    return on;
}

// ---------------------------------------------------------------- tappa 4c: pure rules

TailRegion tail_region(int64_t slots, const uint64_t* slot_offsets, uint64_t slot_bytes, uint64_t cache_bytes,
                       uint64_t bytes, int64_t keep) {
    TailRegion r;
    r.bytes = bytes;
    if (slots <= 0 || cache_bytes == 0 || bytes == 0 || bytes > cache_bytes) return r;
    if (slot_offsets == nullptr && slot_bytes == 0) return r;
    // the region ends at the arena's end and starts at the aligned-down offset that still holds `bytes`
    const uint64_t off = (cache_bytes - bytes) / kTailAlign * kTailAlign;
    // the first slot whose END lies past the region's start: slot s is [start(s), end(s)); end(s) > off <=> overlap
    int64_t first = slots;
    for (int64_t s = slots - 1; s >= 0; --s) {
        const uint64_t end = slot_offsets != nullptr ? slot_offsets[s + 1] : (uint64_t) (s + 1) * slot_bytes;
        if (end <= off) break;
        first = s;
    }
    if (first >= slots) return r;             // (cannot happen with bytes > 0: the last slot ends at cache_bytes)
    if (first < std::max<int64_t>(keep, 0)) return r;   // fewer than `keep` slots would stay
    r.first = first;
    r.offset = off;
    return r;
}

// ---------------------------------------------------------------- tappa 4c: the dense mirror

DenseMirror::~DenseMirror() { close(); }

void DenseMirror::close() {
    if (host_ != nullptr) {
        if (pinned_) (void) cudaFreeHost(host_);
        else std::free(host_);
    }
    host_ = nullptr;
    pinned_ = false;
    bytes_ = 0;
    device_ = -1;
    dev_base_ = nullptr;
    last_ms_ = 0;
    restores_ = 0;
    (void) cudaGetLastError();
}

bool DenseMirror::capture(const void* dev_base, uint64_t bytes, int device, std::string& err) {
    close();
    if (dev_base == nullptr || bytes == 0) return fail(err, "dense mirror: nothing to capture");
    if (device < 0) return fail(err, "dense mirror: a device index is negative");
    // Portable: the stage's card reads it; Mapped would be pointless (no kernel reads a page of it).  A pinned
    // allocation of ~2 GiB can fail under WDDM: the pageable copy keeps the engine running, at the pageable rate.
    void* h = nullptr;
    cudaError_t e = cudaHostAlloc(&h, (size_t) bytes, cudaHostAllocPortable);
    if (e == cudaSuccess && h != nullptr) {
        pinned_ = true;
    } else {
        (void) cudaGetLastError();
        h = std::malloc((size_t) bytes);
        if (h == nullptr) return fail(err, "dense mirror: the host copy could not be allocated");
        pinned_ = false;
    }
    host_ = h;
    bytes_ = bytes;
    device_ = device;
    dev_base_ = dev_base;
    const OnDevice on(device);
    e = cudaMemcpy(host_, dev_base, (size_t) bytes, cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        cuda_fail(err, "dense mirror: D2H", e);
        close();
        return false;
    }
    return true;
}

bool DenseMirror::restore(void* dev_base, cudaStream_t stream, bool wait, std::string& err) {
    if (!valid()) return fail(err, "dense mirror: nothing captured");
    if (dev_base != dev_base_) return fail(err, "dense mirror: the restore address is not the captured one (the WeightRefs point at the captured one)");
    const OnDevice on(device_);
    const auto t0 = std::chrono::steady_clock::now();
    cudaError_t e = cudaMemcpyAsync(dev_base, host_, (size_t) bytes_, cudaMemcpyHostToDevice, stream);
    if (e != cudaSuccess) return cuda_fail(err, "dense mirror: H2D", e);
    last_ms_ = 0;
    if (wait) {
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) return cuda_fail(err, "dense mirror: H2D sync", e);
        last_ms_ = ms_since(t0);
    }
    ++restores_;
    progress_beat();
    return true;
}

// ---------------------------------------------------------------- admission (docs/EP_SPLIT_PREFILL.md s.7, in table order)

bool admit(const Config& c, std::string& why) {
    why.clear();
    if (!c.on) return false;
    if (c.peer_device < 0) { why = "needs --peer-device N"; return false; }
    if (c.layer_split) {
        why = "not with --layer-split (the EP prefill splits the prompt itself: STRATA_EP_SPLIT_LAYER=K)";
        return false;
    }
    if (c.expert_cache_remote) { why = "not with a remote expert cache"; return false; }
    if (c.batch > 0 || c.batch_concurrent) { why = "not with --batch"; return false; }
    // (--vision is allowed: the stage carries its own image-position table on its card, as a split's stages do; the
    // measuring configuration runs with --vision, and a refusal here would have compared the engine with itself)
    if (c.prefill_head_alias) { why = "not with --prefill-head-alias"; return false; }
    if (c.prompt_cache_every > 0 && (c.ckpt_async || c.ckpt_vram_slots > 0)) {
        // the mid-prompt checkpoint is saved in two parts by two threads on two cards; conversation_checkpoint_save's
        // STRATA_CKPT_ASYNC path keeps ONE global staging buffer that it frees and reallocates when the card changes,
        // outside its mutex (a D2H of the other part may be in flight from it), and STRATA_CKPT_VRAM_SLOTS would create
        // slots on the peer card that the merge then never uses (dev_tag cleared).  The same race exists today with
        // --layer-split; not brought into --peer-device by this variable.
        why = "not with STRATA_CKPT_ASYNC / STRATA_CKPT_VRAM_SLOTS and --prompt-cache-every (the mid-prompt checkpoint parts are saved from two cards on two threads through one staging buffer)";
        return false;
    }
    if (c.pipeline_windows > 0) { why = "not with --pipeline-windows"; return false; }
    if (c.layer_major) { why = "not with STRATA_PREFILL_LAYER_MAJOR"; return false; }
    if (!c.serve) { why = "needs --serve"; return false; }
    if (!c.native_pack || !c.expert_profile) {
        why = "needs the native expert pack and --expert-profile (the peer tier)";
        return false;
    }
    if (c.primary_device != 0) { why = "needs the primary on CUDA0"; return false; }
    if (c.layer < 2 || c.layer >= c.n_layers - 1) {
        why = "STRATA_EP_SPLIT_LAYER must be in [2, n_layers - 1)";
        return false;
    }
    if (c.kv_grow || c.elastic) { why = "not with the elastic K/V or cache"; return false; }
    return true;
}

// ---------------------------------------------------------------- the state mover

namespace {

/// The four pool pointers of one QSA state in the order of conversation_kv_part_sizes' first four parts
/// (k, v, k_scale, v_scale), chosen by format the way conversation_snapshot.cpp's file-local `pools()` does
/// (conversation_snapshot.cpp:81-93, read on 8cb752a4): K8V4 = k_q + v_q4 + k_scale; Q4 = k_q4 + v_q4; INT8 = the
/// four; fp16 = k_pool + v_pool.  `host`: the pinned copy's pointers (kv_mode 1, the truth of a streamed layer).
struct PoolPtrs {
    const void* p[4] = {nullptr, nullptr, nullptr, nullptr};
};

PoolPtrs pool_ptrs(const QsaState& st, bool host) {
    PoolPtrs r;
    const strata::kernels::KvHostPools& h = st.host;
    if (st.kv_hybrid) {
        r.p[0] = host ? (const void*) h.k_q : st.k_q;
        r.p[1] = host ? (const void*) h.v_q4 : st.v_q4;
        r.p[2] = host ? (const void*) h.k_scale : st.k_scale;
    } else if (st.kv_q4) {
        r.p[0] = host ? (const void*) h.k_q4 : st.k_q4;
        r.p[1] = host ? (const void*) h.v_q4 : st.v_q4;
    } else if (st.kv_int8) {
        r.p[0] = host ? (const void*) h.k_q : st.k_q;
        r.p[1] = host ? (const void*) h.v_q : st.v_q;
        r.p[2] = host ? (const void*) h.k_scale : st.k_scale;
        r.p[3] = host ? (const void*) h.v_scale : st.v_scale;
    } else {
        r.p[0] = host ? (const void*) h.k_pool : st.k_pool;
        r.p[1] = host ? (const void*) h.v_pool : st.v_pool;
    }
    return r;
}

const char* part_name(size_t i) {
    static const char* const names[4] = {"k", "v", "k_scale", "v_scale"};
    return i < 4 ? names[i] : "?";
}

/// Evicts blocks [b0, b1) of a streamed layer's residency map on stream `s` (the map's card is current): page_table[b]
/// = -1 for every block of the range and slot_block[slot] = -1 for the slot each resident one held, so the slot is a
/// free victim for the next resolve (kv_stream.cu's sweep takes a slot whose slot_block is negative before any other)
/// and the block is re-read from the host copy on its next selection.  Every other block of the map stays resident.
/// A range that is half the map or more, or one holding more than kResetAboveResident resident blocks, takes the
/// whole kv_stream_reset instead (one kernel; what a first turn or a restore amounts to anyway).  No resolve may be
/// in flight on the map (the caller holds both sessions still).  Complete on return.
bool evict_blocks(const strata::kernels::KvStreamMap& m, int64_t b0, int64_t b1, cudaStream_t s, std::string& err) {
    if (b1 <= b0) return true;
    if (m.page_table == nullptr || m.slot_block == nullptr || m.n_blocks <= 0 || m.n_slots <= 0)
        return fail(err, "a streamed QSA state has no residency map");
    b0 = std::max<int64_t>(b0, 0);
    b1 = std::min<int64_t>(b1, m.n_blocks);
    if (b1 <= b0) return true;
    const int64_t n = b1 - b0;
    bool whole = n * 2 >= m.n_blocks;
    std::vector<int32_t> table;
    if (!whole) {
        table.resize((size_t) n);
        cudaError_t e = cudaMemcpyAsync(table.data(), m.page_table + b0, (size_t) n * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost, s);
        if (e == cudaSuccess) e = cudaStreamSynchronize(s);
        if (e != cudaSuccess) return cuda_fail(err, "residency map read", e);
        int64_t resident = 0;
        for (const int32_t sl : table) resident += sl >= 0 ? 1 : 0;
        whole = resident > kResetAboveResident;
    }
    if (whole) {
        strata::kernels::kv_stream_reset(m, (void*) s);
    } else {
        // -1 is 0xFFFFFFFF: a byte memset writes it
        cudaError_t e = cudaMemsetAsync(m.page_table + b0, 0xFF, (size_t) n * sizeof(int32_t), s);
        for (int64_t i = 0; e == cudaSuccess && i < n; ++i) {
            const int32_t sl = table[(size_t) i];
            if (sl >= 0 && sl < m.n_slots) e = cudaMemsetAsync(m.slot_block + sl, 0xFF, sizeof(int32_t), s);
        }
        if (e != cudaSuccess) return cuda_fail(err, "residency map eviction", e);
    }
    const cudaError_t e = cudaStreamSynchronize(s);
    if (e != cudaSuccess) return cuda_fail(err, "residency map sync", e);
    return true;
}

}  // namespace

StateMover::~StateMover() { close(); }

bool StateMover::init(const ModelGeometry& g, int64_t lb, int64_t le, int primary_device, int peer_device,
                      int64_t max_cells, std::string& err) {
    close();
    if (lb < 0 || le < lb || le > g.n_layers) return fail(err, "layer range outside the model");
    if (primary_device < 0 || peer_device < 0) return fail(err, "a device index is negative");
    if (max_cells < 0) return fail(err, "negative max_cells");
    g_ = &g;
    lb_ = lb;
    le_ = le;
    max_cells_ = max_cells;
    // (the devices first: a failed step below calls close(), which frees under them; close() puts them back to -1)
    primary_dev_ = primary_device;
    peer_dev_ = peer_device;
    piece_ = env_piece_bytes();
    // Portable (every card reads and writes them), NOT Mapped: they are the staging of plain copies, no kernel reads
    // a page of them.
    for (int i = 0; i < 2; ++i) {
        const cudaError_t e = cudaHostAlloc(&piece_h_[i], piece_, cudaHostAllocPortable);
        if (e != cudaSuccess) {
            piece_h_[i] = nullptr;
            cuda_fail(err, "staging piece", e);
            close();
            return false;
        }
    }
    {
        const OnDevice on(primary_device);
        cudaError_t e = cudaStreamCreateWithFlags(&s_primary_, cudaStreamNonBlocking);
        if (e != cudaSuccess) { s_primary_ = nullptr; cuda_fail(err, "primary stream", e); close(); return false; }
        e = cudaEventCreateWithFlags(&ev_[0], cudaEventDisableTiming);
        if (e != cudaSuccess) { ev_[0] = nullptr; cuda_fail(err, "primary event", e); close(); return false; }
    }
    {
        const OnDevice on(peer_device);
        cudaError_t e = cudaStreamCreateWithFlags(&s_peer_, cudaStreamNonBlocking);
        if (e != cudaSuccess) { s_peer_ = nullptr; cuda_fail(err, "peer stream", e); close(); return false; }
        e = cudaEventCreateWithFlags(&ev_[1], cudaEventDisableTiming);
        if (e != cudaSuccess) { ev_[1] = nullptr; cuda_fail(err, "peer event", e); close(); return false; }
    }
    return true;
}

void StateMover::close() {
    // The streams and events were created under their card and are freed under it; the pieces are host memory
    // (any device).  Safe on a never-opened or half-opened mover: every handle is null until it was created.
    if (s_primary_ != nullptr || ev_[0] != nullptr) {
        const OnDevice on(primary_dev_);
        if (s_primary_ != nullptr) { (void) cudaStreamSynchronize(s_primary_); (void) cudaStreamDestroy(s_primary_); }
        if (ev_[0] != nullptr) (void) cudaEventDestroy(ev_[0]);
        s_primary_ = nullptr;
        ev_[0] = nullptr;
    }
    if (s_peer_ != nullptr || ev_[1] != nullptr) {
        const OnDevice on(peer_dev_);
        if (s_peer_ != nullptr) { (void) cudaStreamSynchronize(s_peer_); (void) cudaStreamDestroy(s_peer_); }
        if (ev_[1] != nullptr) (void) cudaEventDestroy(ev_[1]);
        s_peer_ = nullptr;
        ev_[1] = nullptr;
    }
    for (int i = 0; i < 2; ++i) {
        if (piece_h_[i] != nullptr) (void) cudaFreeHost(piece_h_[i]);
        piece_h_[i] = nullptr;
    }
    (void) cudaGetLastError();
    g_ = nullptr;
    lb_ = le_ = max_cells_ = 0;
    primary_dev_ = peer_dev_ = -1;
    piece_ = 0;
    last_ = Stats{};
    handovers_ = 0;
    bytes_total_ = 0;
    ms_total_ = 0;
}

bool StateMover::to_primary(const SessionState& peer_ss, SessionState& primary_ss, int64_t c0, int64_t c1,
                            bool running_state, Stats& st, std::string& err) {
    return move(peer_ss, primary_ss, Dir::kToPrimary, c0, c1, running_state, st, err);
}

bool StateMover::to_peer(const SessionState& primary_ss, SessionState& peer_ss, int64_t c0, int64_t c1,
                         bool running_state, Stats& st, std::string& err) {
    return move(primary_ss, peer_ss, Dir::kToPeer, c0, c1, running_state, st, err);
}

bool StateMover::rows_to_primary(const float* src_peer, float* dst_primary, int64_t T, int64_t floats_per_row,
                                 std::string& err) {
    if (!valid()) return fail(err, "mover not open");
    if (T < 0 || floats_per_row <= 0) return fail(err, "bad row count");
    if (T == 0) return true;
    if (src_peer == nullptr || dst_primary == nullptr) return fail(err, "null row buffer");
    if ((uint64_t) T > std::numeric_limits<uint64_t>::max() / ((uint64_t) floats_per_row * sizeof(float)))
        return fail(err, "row bytes overflow");
    const size_t bytes = (size_t) ((uint64_t) T * (uint64_t) floats_per_row * sizeof(float));
    return copy_d2d(src_peer, peer_dev_, dst_primary, primary_dev_, bytes, err);
}

bool StateMover::move(const SessionState& from, SessionState& to, Dir dir, int64_t c0, int64_t c1,
                      bool running_state, Stats& st, std::string& err) {
    st = Stats{};
    if (!valid() || g_ == nullptr) return fail(err, "mover not open");
    if (c0 < 0 || c1 < c0 || c1 > max_cells_) return fail(err, "cell range outside the sessions");
    const ModelGeometry& g = *g_;
    const int da_dev = dir == Dir::kToPrimary ? peer_dev_ : primary_dev_;
    const int a_dev = dir == Dir::kToPrimary ? primary_dev_ : peer_dev_;
    cudaStream_t s_dest = a_dev == primary_dev_ ? s_primary_ : s_peer_;
    const auto t_all = std::chrono::steady_clock::now();
    st.cells0 = c0;
    st.cells1 = c1;

    const strata::kernels::QsaShapes shapes = strata::kernels::qsa_real_shapes();
    if (shapes.page_size <= 0 || shapes.idx_block <= 0 || g.idx_key_dim <= 0) return fail(err, "bad QSA shapes");
    const int64_t page = shapes.page_size;
    const int64_t idx_dim = g.idx_key_dim;
    // whole pages: [c0 rounded down, c1 rounded up) capped at the capacity
    const int64_t c0p = c0 - c0 % page;
    const int64_t c1p = std::min<int64_t>(max_cells_, (c1 + page - 1) / page * page);
    const bool cells = c1 > c0;

    ConversationStateSizes z{};
    if (running_state && !conversation_state_sizes(g, z, err)) return false;

    // the global ordinals of the first layer in range: q QSA states and gi GDN rows lie below lb_
    int64_t q = 0, gi = 0;
    for (int64_t l = 0; l < lb_; ++l) (is_qsa_layer(g, l) ? q : gi)++;

    for (int64_t l = lb_; l < le_; ++l) {
        if (!is_qsa_layer(g, l)) {
            // ---- a GDN layer: its state row, with `running_state` only
            const int64_t ordinal = gi++;
            if (!running_state) continue;
            if (ordinal < from.gdn_ord0 || ordinal >= from.gdn_ord0 + from.gdn_alloc ||
                ordinal < to.gdn_ord0 || ordinal >= to.gdn_ord0 + to.gdn_alloc)
                return fail(err, "GDN ordinal outside the carve");
            if (from.gdn_state == nullptr || to.gdn_state == nullptr) return fail(err, "a session has no GDN state");
            const uint64_t row = gdn_row_floats(g);
            const float* src = from.gdn_state + (size_t) (ordinal - from.gdn_ord0) * row;
            float* dst = to.gdn_state + (size_t) (ordinal - to.gdn_ord0) * row;
            const size_t bytes = (size_t) (row * sizeof(float));
            const auto t0 = std::chrono::steady_clock::now();
            if (!copy_d2d(src, da_dev, dst, a_dev, bytes, err)) return false;
            st.ms_state += ms_since(t0);
            st.state_bytes += bytes;
            ++st.gdn_layers;
            continue;
        }

        // ---- a QSA layer: the K/V pages, the pooled rows, and with `running_state` the indexer's small parts
        const int64_t ordinal = q++;
        if (ordinal < from.qsa_ord0 || ordinal >= from.qsa_ord0 + from.qsa_alloc ||
            ordinal < to.qsa_ord0 || ordinal >= to.qsa_ord0 + to.qsa_alloc)
            return fail(err, "QSA ordinal outside the carve");
        if (from.qsa_states == nullptr || to.qsa_states == nullptr) return fail(err, "a session has no QSA states");
        const QsaState& A = from.qsa_states[ordinal];
        QsaState& B = to.qsa_states[ordinal];
        if (A.kv_mode != B.kv_mode || A.kv_int8 != B.kv_int8 || A.kv_q4 != B.kv_q4 || A.kv_hybrid != B.kv_hybrid ||
            A.kv_rot != B.kv_rot || A.max_cells != B.max_cells || A.idx_pooled_rows != B.idx_pooled_rows)
            return fail(err, "the two QSA states differ in format, kv_mode, max_cells or pooled rows");
        if (A.kv_mode == 2) return fail(err, "a drafter ring (kv_mode 2) in the range");
        if (A.kv_mode != 0 && A.kv_mode != 1) return fail(err, "unknown kv_mode");
        if (A.max_cells < max_cells_) return fail(err, "a QSA state is smaller than the mover's max_cells");
        if (A.kv_mode == 1 && (!A.host.present() || !B.host.present()))
            return fail(err, "a streamed QSA state has no host copy");
        ++st.qsa_layers;

        if (cells) {
            // The byte ranges of the four pools.  Read on 8cb752a4 in src/core/conversation_snapshot.cpp:
            //   :48  `const int64_t cells = ((upto + s.page_size - 1) / s.page_size) * s.page_size;`
            //   :53  `const int64_t pooled = index && upto > 0 ? upto / s.idx_block + 1 : 0;`
            //   :61  `product(l.data, {cells, g.n_head_kv, per})`, scales `{cells, n_head_kv, head_dim/64*2}`
            //   :521 `sizes = {l.data, l.value_data, l.scales, l.value_scales, l.pooled};`
            // so the five parts are k, v, k_scale, v_scale, pooled in that order, every part's bytes are linear in
            // whole pages of cells (the cells come first: the pools are [page][kv_head][page_size][head_dim],
            // layer.hpp:210, page-major), and the slice of pages [c0p / page, c1p / page) is the byte range
            // [sizes(c0p)[i], sizes(c1p)[i]) of part i.  Part 4 (pooled) is moved by rows below, not from here.
            std::array<uint64_t, 5> s0{}, s1{};
            if (!conversation_kv_part_sizes(A, g, c0p, true, s0, err) ||
                !conversation_kv_part_sizes(A, g, c1p, true, s1, err)) return false;
            const bool host = A.kv_mode == 1;
            const PoolPtrs pa = pool_ptrs(A, host);
            const PoolPtrs pb = pool_ptrs(B, host);
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < 4; ++i) {
                if (s1[i] < s0[i]) return fail(err, "K/V part bytes not monotone");
                const uint64_t bytes = s1[i] - s0[i];
                const bool have_a = pa.p[i] != nullptr, have_b = pb.p[i] != nullptr;
                if (bytes == 0 || (!have_a && !have_b)) {
                    if (have_a != have_b && bytes != 0) {
                        err = std::string("ep split: K/V part ") + part_name(i) + " present on one side only";
                        return false;
                    }
                    continue;
                }
                if (have_a != have_b) {
                    err = std::string("ep split: K/V part ") + part_name(i) + " present on one side only";
                    return false;
                }
                const uint8_t* src = static_cast<const uint8_t*>(pa.p[i]) + s0[i];
                uint8_t* dst = static_cast<uint8_t*>(const_cast<void*>(pb.p[i])) + s0[i];
                if (host) {
                    if (!copy_h2h(src, dst, (size_t) bytes, err)) return false;
                } else {
                    if (!copy_d2d(src, da_dev, dst, a_dev, (size_t) bytes, err)) return false;
                }
                st.kv_bytes += bytes;
            }
            if (host) {
                // The target's VRAM slots may hold the old bytes of the rewritten blocks: those blocks leave its
                // residency map, the rest of the map keeps the decode's working set.  (conversation_snapshot.cpp:413
                // resets the whole map after rewriting a whole host copy; a turn's handover rewrites a few hundred
                // cells, and a full reset made the first windows of every turn re-read their selections from the host.)
                st.kv_host_memcpy = true;
                const OnDevice on(a_dev);
                if (!evict_blocks(B.map, c0p / page, (c1p + page - 1) / page, s_dest, err)) return false;
            }
            st.ms_kv += ms_since(t0);

            // the indexer's pooled rows [c0 / idx_block, c1 / idx_block], the row being filled included
            const int64_t r0 = c0 / shapes.idx_block, r1 = c1 / shapes.idx_block;
            if (r1 + 1 > A.idx_pooled_rows) return fail(err, "pooled rows outside idx_pooled");
            if (A.idx_pooled == nullptr || B.idx_pooled == nullptr) return fail(err, "a QSA state has no idx_pooled");
            const size_t row_bytes = (size_t) idx_dim * sizeof(float);
            const auto t1 = std::chrono::steady_clock::now();
            const size_t bytes = (size_t) (r1 + 1 - r0) * row_bytes;
            if (!copy_d2d(A.idx_pooled + (size_t) r0 * idx_dim, da_dev, B.idx_pooled + (size_t) r0 * idx_dim, a_dev,
                          bytes, err)) return false;
            st.pooled_bytes += bytes;
            st.ms_pooled += ms_since(t1);
        }

        if (running_state) {
            if (A.idx_tail == nullptr || B.idx_tail == nullptr || A.idx_dead == nullptr || B.idx_dead == nullptr ||
                A.idx_block_pos == nullptr || B.idx_block_pos == nullptr)
                return fail(err, "a QSA state has no indexer tail / dead / block position");
            const auto t0 = std::chrono::steady_clock::now();
            if (!copy_d2d(A.idx_tail, da_dev, B.idx_tail, a_dev, z.tail, err) ||
                !copy_d2d(A.idx_dead, da_dev, B.idx_dead, a_dev, z.dead, err) ||
                !copy_d2d(A.idx_block_pos, da_dev, B.idx_block_pos, a_dev, z.block_pos, err)) return false;
            st.state_bytes += z.tail + z.dead + z.block_pos;
            st.ms_state += ms_since(t0);
        }
    }

    // nothing of the mover's in flight when the caller gets the sessions back
    {
        const OnDevice on(primary_dev_);
        const cudaError_t e = cudaStreamSynchronize(s_primary_);
        if (e != cudaSuccess) return cuda_fail(err, "primary stream sync", e);
    }
    {
        const OnDevice on(peer_dev_);
        const cudaError_t e = cudaStreamSynchronize(s_peer_);
        if (e != cudaSuccess) return cuda_fail(err, "peer stream sync", e);
    }
    st.ms = ms_since(t_all);
    last_ = st;
    ++handovers_;
    bytes_total_ += st.kv_bytes + st.pooled_bytes + st.state_bytes;
    ms_total_ += st.ms;
    return true;
}

bool StateMover::copy_d2d(const void* src, int from_dev, void* dst, int to_dev, size_t bytes, std::string& err) {
    // THE RELAY (two pieces, two events, two streams; no P2P).  Piece i goes D2H on `from`'s stream into
    // piece_h_[i & 1] and H2D on `to`'s stream out of it.  The header fixes one event per card, and
    // cudaEventRecord refuses an event and a stream of different cards (the CUDA guide's "Stream and Event
    // Behavior"), so ev_from is only ever recorded on s_from and ev_to only on s_to; a cudaStreamWaitEvent across
    // cards is allowed.  With one event per side a two-deep pipeline still fits, by ORDERING THE HOST CALLS:
    //   iteration i:  (1) i >= 2: s_from waits ev_to      - ev_to was last recorded after H2D(i-2): piece i & 1 is free
    //                 (2) D2H(i) on s_from into piece i & 1
    //                 (3) i >= 1: s_to waits ev_from       - ev_from was last recorded after D2H(i-1)
    //                             H2D(i-1) on s_to out of piece (i-1) & 1; record ev_to on s_to
    //                 (4) record ev_from on s_from          - after D2H(i)
    //   after the loop: s_to waits ev_from; H2D(n-1); then both streams are synchronized (complete on return).
    // Every wait reads the event as it was recorded before it (a later record does not move a wait already
    // issued), so D2H(i) overlaps H2D(i-1) and never overruns the piece H2D(i-2) is reading.  from == to (the bench
    // on one card, where primary == peer) works the same with two streams of the same card.
    if (bytes == 0) return true;
    if (src == nullptr || dst == nullptr) return fail(err, "null copy pointer");
    if (!valid() || piece_ == 0) return fail(err, "mover not open");
    if ((from_dev != primary_dev_ && from_dev != peer_dev_) || (to_dev != primary_dev_ && to_dev != peer_dev_))
        return fail(err, "copy between cards the mover does not own");
    cudaStream_t s_from = from_dev == primary_dev_ ? s_primary_ : s_peer_;
    cudaEvent_t ev_from = from_dev == primary_dev_ ? ev_[0] : ev_[1];
    cudaStream_t s_to = to_dev == primary_dev_ ? s_primary_ : s_peer_;
    cudaEvent_t ev_to = to_dev == primary_dev_ ? ev_[0] : ev_[1];
    if (s_from == s_to) {   // one card for both roles (primary == peer): the other pair of the same card
        s_to = s_from == s_primary_ ? s_peer_ : s_primary_;
        ev_to = ev_from == ev_[0] ? ev_[1] : ev_[0];
    }
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(dst);
    const size_t n = (bytes + piece_ - 1) / piece_;
    size_t prev_off = 0, prev_len = 0;   // piece i-1
    for (size_t i = 0; i < n; ++i) {
        const int p = (int) (i & 1);
        const size_t off = i * piece_;
        const size_t len = std::min(piece_, bytes - off);
        {
            const OnDevice on(from_dev);
            if (i >= 2) {
                const cudaError_t e = cudaStreamWaitEvent(s_from, ev_to, 0);
                if (e != cudaSuccess) return cuda_fail(err, "wait for the piece", e);
            }
            const cudaError_t e = cudaMemcpyAsync(piece_h_[p], s + off, len, cudaMemcpyDeviceToHost, s_from);
            if (e != cudaSuccess) return cuda_fail(err, "D2H", e);
        }
        if (i >= 1) {
            const OnDevice on(to_dev);
            cudaError_t e = cudaStreamWaitEvent(s_to, ev_from, 0);
            if (e != cudaSuccess) return cuda_fail(err, "wait for the D2H", e);
            e = cudaMemcpyAsync(d + prev_off, piece_h_[1 - p], prev_len, cudaMemcpyHostToDevice, s_to);
            if (e != cudaSuccess) return cuda_fail(err, "H2D", e);
            e = cudaEventRecord(ev_to, s_to);
            if (e != cudaSuccess) return cuda_fail(err, "record after the H2D", e);
        }
        {
            const OnDevice on(from_dev);
            const cudaError_t e = cudaEventRecord(ev_from, s_from);
            if (e != cudaSuccess) return cuda_fail(err, "record after the D2H", e);
        }
        prev_off = off;
        prev_len = len;
        progress_beat();
    }
    {
        const OnDevice on(to_dev);
        cudaError_t e = cudaStreamWaitEvent(s_to, ev_from, 0);
        if (e != cudaSuccess) return cuda_fail(err, "wait for the last D2H", e);
        e = cudaMemcpyAsync(d + prev_off, piece_h_[(n - 1) & 1], prev_len, cudaMemcpyHostToDevice, s_to);
        if (e != cudaSuccess) return cuda_fail(err, "last H2D", e);
        e = cudaEventRecord(ev_to, s_to);
        if (e != cudaSuccess) return cuda_fail(err, "record after the last H2D", e);
        e = cudaStreamSynchronize(s_to);
        if (e != cudaSuccess) return cuda_fail(err, "destination stream sync", e);
    }
    {
        const OnDevice on(from_dev);
        const cudaError_t e = cudaStreamSynchronize(s_from);
        if (e != cudaSuccess) return cuda_fail(err, "source stream sync", e);
    }
    progress_beat();
    return true;
}

bool StateMover::copy_h2h(const void* src, void* dst, size_t bytes, std::string& err) {
    if (bytes == 0) return true;
    if (src == nullptr || dst == nullptr) return fail(err, "null host copy pointer");
    constexpr size_t kSlice = (size_t) 1 << 20;
    const size_t slices = (bytes + kSlice - 1) / kSlice;
    const int threads = bytes < ((size_t) 8 << 20) ? 1 : std::max(1, std::min(env_threads(), (int) slices));
    const uint8_t* s = static_cast<const uint8_t*>(src);
    uint8_t* d = static_cast<uint8_t*>(dst);
    // slice k goes to thread k % threads; the calling thread is thread 0 and beats once per slice of its own
    // (the workers' slices are counted by it too, after the join, so the beat total is one per slice)
    auto run = [&](int t, bool beat) {
        for (size_t k = (size_t) t; k < slices; k += (size_t) threads) {
            const size_t off = k * kSlice;
            std::memcpy(d + off, s + off, std::min(kSlice, bytes - off));
            if (beat) progress_beat();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve((size_t) std::max(threads - 1, 0));
    for (int t = 1; t < threads; ++t) pool.emplace_back(run, t, false);
    run(0, true);
    for (auto& th : pool) th.join();
    for (size_t k = 0; k < slices; ++k)
        if (k % (size_t) threads != 0) progress_beat();
    return true;
}

// ---------------------------------------------------------------- what the peer stage is known to hold

int64_t Sync::valid_for(const int32_t* ids_now, int64_t n) {
    if (cells > 0 && cells <= n && ids_now != nullptr && (int64_t) ids.size() == cells &&
        std::memcmp(ids.data(), ids_now, (size_t) cells * sizeof(int32_t)) == 0)
        return cells;
    invalidate();
    return 0;
}

void Sync::note(const int32_t* ids_now, int64_t cells_now) {
    if (cells_now <= 0 || ids_now == nullptr) { invalidate(); return; }
    cells = cells_now;
    ids.assign(ids_now, ids_now + cells_now);
}

// ---------------------------------------------------------------- mid-prompt checkpoints

bool checkpoint_merge_tail(ConversationCheckpoint& full, const ConversationCheckpoint& tail, const SessionState& full_ss,
                           const SessionState& tail_ss, const ModelGeometry& g, std::string& err) {
    if (full.ids != tail.ids) return fail(err, "checkpoint parts of different token counts");
    // STRATA_CKPT_ASYNC: a part's GDN bytes may still be landing on the background thread (conversation_checkpoint_save
    // returns before them); every reader of a checkpoint's bytes waits first, as the restore and the snapshots do
    conversation_checkpoint_join(nullptr);
    ConversationStateSizes zf{}, zt{};
    if (!conversation_session_sizes(g, full_ss, zf, err) || !conversation_session_sizes(g, tail_ss, zt, err)) return false;
    if (zf.tail != zt.tail || zf.dead != zt.dead || zf.block_pos != zt.block_pos)
        return fail(err, "checkpoint parts of different indexer sizes");

    // GDN rows: the tail's [gdn_ord0, + gdn_alloc) inside the full's
    if (full.gdn.size() != zf.gdn || tail.gdn.size() != zt.gdn) return fail(err, "checkpoint GDN bytes of the wrong size");
    if (tail_ss.gdn_alloc > 0) {
        if (full_ss.gdn_alloc <= 0 || zf.gdn % (size_t) full_ss.gdn_alloc != 0) return fail(err, "full checkpoint without GDN rows");
        const size_t row_bytes = zf.gdn / (size_t) full_ss.gdn_alloc;
        if (row_bytes == 0 || zt.gdn != row_bytes * (size_t) tail_ss.gdn_alloc) return fail(err, "GDN row bytes differ between the parts");
        if (tail_ss.gdn_ord0 < full_ss.gdn_ord0 ||
            tail_ss.gdn_ord0 + tail_ss.gdn_alloc > full_ss.gdn_ord0 + full_ss.gdn_alloc)
            return fail(err, "the tail's GDN rows are outside the full carve");
        std::memcpy(full.gdn.data() + (size_t) (tail_ss.gdn_ord0 - full_ss.gdn_ord0) * row_bytes, tail.gdn.data(), zt.gdn);
        full.dev_tag = 0;   // the device copy of `gdn` (STRATA_CKPT_VRAM_SLOTS) no longer matches these bytes
    }

    // indexer tail / dead / block_pos: owned order j = ordinal - qsa_ord0 in each part (conversation_state.cpp:421-424)
    const size_t nf = (size_t) std::max<int64_t>(full_ss.qsa_alloc, 0), nt = (size_t) std::max<int64_t>(tail_ss.qsa_alloc, 0);
    if (full.tails.size() != nf * zf.tail || full.dead.size() != nf * zf.dead || full.block_pos.size() != nf * zf.block_pos ||
        tail.tails.size() != nt * zt.tail || tail.dead.size() != nt * zt.dead || tail.block_pos.size() != nt * zt.block_pos)
        return fail(err, "checkpoint indexer bytes of the wrong size");
    for (size_t j = 0; j < nt; ++j) {
        const int64_t q = tail_ss.qsa_ord0 + (int64_t) j;
        if (q < full_ss.qsa_ord0 || q >= full_ss.qsa_ord0 + full_ss.qsa_alloc) return fail(err, "a tail QSA ordinal is outside the full carve");
        const size_t at = (size_t) (q - full_ss.qsa_ord0);
        std::memcpy(full.tails.data() + at * zf.tail, tail.tails.data() + j * zt.tail, zt.tail);
        std::memcpy(full.dead.data() + at * zf.dead, tail.dead.data() + j * zt.dead, zt.dead);
        std::memcpy(full.block_pos.data() + at * zf.block_pos, tail.block_pos.data() + j * zt.block_pos, zt.block_pos);
    }
    // `ple` stays the full's: the PLE is layer 1's, the stage never has it
    return true;
}

// ---------------------------------------------------------------- the stage's prompt buffers and the peer's reserve

int64_t reserve_add_mib(uint64_t prompt_bytes) {
    const int64_t forced = env_reserve_mib();
    if (forced >= 0) return forced;
    const uint64_t total = prompt_bytes > std::numeric_limits<uint64_t>::max() - kReserveHeadroomBytes
                               ? std::numeric_limits<uint64_t>::max()
                               : prompt_bytes + kReserveHeadroomBytes;
    return (int64_t) ((total + kMiB - 1) / kMiB);
}

// ---------------------------------------------------------------- the one-line forms

std::string line_on(int64_t lb, int64_t le, int dev, int64_t prompt_mib, int64_t reserve0_mib, int64_t reserve1_mib) {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "strata ep-split: on: layers [%lld, %lld) on CUDA%d; prompt buffers %lld MiB from the peer's reserve (%lld -> %lld MiB)",
                  (long long) lb, (long long) le, dev, (long long) prompt_mib, (long long) reserve0_mib,
                  (long long) reserve1_mib);
    return buf;
}

std::string line_off(const std::string& why) {
    return "strata: STRATA_EP_SPLIT_PREFILL off: " + why;
}

std::string line_request(int64_t fresh, int64_t min_tokens, bool staged) {
    char buf[256];
    if (staged)
        std::snprintf(buf, sizeof buf, "strata ep-split: %lld tokens to read: staged read on two cards (threshold %lld)",
                      (long long) fresh, (long long) min_tokens);
    else
        std::snprintf(buf, sizeof buf, "strata ep-split: %lld tokens to read: the primary reads every layer with the servers' "
                                       "prompt shares (below the threshold %lld)", (long long) fresh, (long long) min_tokens);
    return buf;
}

std::string line_trace(const char* dir, const Stats& st) {
    char buf[384];
    std::snprintf(buf, sizeof buf,
                  "strata ep-split: %s: cells [%lld, %lld), %d QSA + %d GDN layers, K/V %.1f MB (%s) %.1f ms, "
                  "pooled %.1f MB %.1f ms, state %.1f MB %.1f ms, total %.1f ms",
                  dir != nullptr ? dir : "?", (long long) st.cells0, (long long) st.cells1, st.qsa_layers, st.gdn_layers,
                  (double) st.kv_bytes / 1e6, st.kv_host_memcpy ? "host memcpy" : "D2H+H2D", st.ms_kv,
                  (double) st.pooled_bytes / 1e6, st.ms_pooled, (double) st.state_bytes / 1e6, st.ms_state, st.ms);
    return buf;
}

}  // namespace strata::core::ep_split
