// tools/ep_split_test.cpp - EP_SPLIT_PREFILL (docs/EP_SPLIT_PREFILL.md s.5, s.11; include/strata/core/ep_split.hpp): the
// state mover's bit test, bit for bit, plus the sync bookkeeping and the mid-prompt checkpoint merge.
//
// No model, no kernels of its own: a synthetic ModelGeometry with the real structure (the compiled defaults - the
// GDN/QSA shapes of Qwen3.8-Flash-Next - over kNLayers layers, qsa_interval 4: QSA at 3, 7, 11), and FOUR sessions
// carved by session_bytes / session_init exactly as generate.cpp carves the main session and a stage's:
//
//   full   [0, kNLayers)  on the primary card   (the main session: every layer)
//   tail   [kLb, kNLayers) on the peer card     (the EP stage's session: layers [K, n))
//   tail2  [kLb, kNLayers) on the peer card     (a second stage session, for the round trip)
//   ref    [0, kNLayers)  on the primary card   (the reference a merged checkpoint is compared against)
//
// Every part the mover moves (the four K/V pools, idx_pooled, idx_tail, idx_dead, idx_block_pos of every QSA state,
// the GDN rows) is filled with deterministic pseudo-random bytes (splitmix64 seeded by session, ordinal and part), the
// expected bytes of the destination are computed ON THE HOST from the contract (pages [floor(c0 / p) p, ceil(c1 / p) p)
// of every pool with the byte ranges of conversation_kv_part_sizes, pooled rows [c0 / idx_block, c1 / idx_block],
// the small parts and the GDN rows with `running_state`), and the destination is read back and memcmp'd whole: so the
// cells and layers the mover must NOT touch are checked as strictly as the ones it must.  In kv_mode 1 (--kv-resident:
// `kv_resident` > 0 here) the authoritative bytes are the pinned host copies (QsaState::host); a stale block is
// planted in the TARGET's VRAM slots with a page table naming it before the handover, and the page table must be all
// -1 afterwards (kv_stream_reset: docs/EP_SPLIT_PREFILL.md s.10).
//
// The checks, each printing `OK <check>` or `FAIL <check>: <detail>` (exit 1 at the first failure):
//   1  gdn_row_floats(g) against session_bytes (two carves that differ by one GDN layer) and conversation_state_sizes
//   2  to_primary of cells [0, kN) with the running state: the full session's [kLb, n) equals the tail's, the rest
//      intact, the page tables reset, the Stats as expected, the stage's session untouched
//   4  to_peer into tail2 (the round trip): tail2 holds the tail's bytes
//   3  a partial handover [c0, c1) with c0 mid-page and c1 not a page multiple, running_state false: whole pages
//      only, nothing else
//   5  rows_to_primary: T x (hc * n_embd) floats from the peer card to the primary
//   6  Sync: note / valid_for on equal ids, different ids, a shorter request
//   7  checkpoint_merge_tail against a checkpoint of a session built with the same bytes; different ids refused
//   8  errors: kv_mode 2 in the range (a copy of the QsaState array with the mode forced), c1 > max_cells; a range
//      refusal writes nothing
//   Tappa 4c (docs/EP_SPLIT_PREFILL.md s.14):
//   9  tail_region: the dense weights' region at the end of a tier (uniform and sized slots, the keep floor, a
//      region larger than the arena)
//   10 the overlapped handover: the cells cross chunk by chunk without the running state (the page and the pooled
//      row two chunks share are written twice), then the last chunk with it - the full session equals one whole
//      handover byte for byte; in kv_mode 1 a resident block inside the range leaves the map, one outside stays
//   11 DenseMirror: capture, overwrite, restore (waited) gives the bytes back at the same address; another address
//      is refused; restore before capture is refused
//   12 staged_read and the threshold's default
//
//   ep_split_test [primary_dev=0] [peer_dev=0] [max_cells=8192] [kv_resident=0]
//
// The same card twice is the default (the mover must work with from_dev == to_dev); with two cards give both.  With
// `kv_resident` > 0, qsa_set_kv_resident is called BEFORE the carves, so every QSA state is streamed (kv_mode 1);
// that needs max_cells above qsa_kv_resident_min() (layer.cpp's kv_plan), so a smaller max_cells is raised and said.
// STRATA_EP_SPLIT_TEST_KV=int8|q4 carves the K/V in that format (default: fp16).  Exit 77 without a CUDA device.
//
// Written, not compiled (docs/EP_SPLIT_PREFILL.md s.11).  Not in the default build: `cmake --build <dir> --target
// ep_split_test`.
#include "strata/core/ep_split.hpp"

#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace strata::core;
namespace ep = strata::core::ep_split;

namespace {

// ---------------------------------------------------------------- the synthetic model and the handover
constexpr int64_t kNLayers = 12;   ///< QSA at 3, 7, 11 (qsa_interval 4): the tail holds TWO QSA states (ordinals 1, 2)
constexpr int64_t kLb = 4;         ///< the stage's first layer (K)
constexpr int64_t kLe = kNLayers;
constexpr int64_t kExperts = 10;   ///< experts per token (session_bytes' k)
constexpr int64_t kN = 1157;       ///< cells of the first handover and tokens of the checkpoints: not a page multiple

// ---------------------------------------------------------------- reporting
int g_checks = 0;
const char* g_check = "";

[[noreturn]] void fail(const std::string& detail) {
    std::fprintf(stderr, "FAIL %s: %s\n", g_check, detail.c_str());
    std::exit(1);
}
void expect(bool cond, const std::string& detail) {
    ++g_checks;
    if (!cond) fail(detail);
}
/// The same for a detail the condition's own call fills (`err`, `where`): the argument order of a call is unspecified,
/// so the string is bound by reference here and read only after the condition ran.
void expect(bool cond, const char* what, const std::string& filled) {
    ++g_checks;
    if (!cond) fail(std::string(what) + filled);
}
void ok() { std::printf("OK %s\n", g_check); std::fflush(stdout); }
void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}
std::string num(int64_t v) { return std::to_string(v); }

// ---------------------------------------------------------------- deterministic bytes
uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
std::vector<uint8_t> pattern(uint64_t seed, size_t bytes) {
    std::vector<uint8_t> v(bytes);
    uint64_t x = seed;
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) { const uint64_t w = splitmix(x); std::memcpy(v.data() + i, &w, 8); }
    if (i < bytes) { const uint64_t w = splitmix(x); std::memcpy(v.data() + i, &w, bytes - i); }
    return v;
}
/// One seed per (session fill, ordinal, part): two sessions filled with the same `base` hold the same bytes for the
/// ordinals they share (that is how `ref` gets `full`'s bytes below kLb and `tail`'s from kLb on).
uint64_t seed_of(uint64_t base, int64_t ordinal, int part) {
    return (base * 0x100000001B3ull) ^ ((uint64_t) ordinal << 24) ^ ((uint64_t) part << 8) ^ 0x51ull;
}

// ---------------------------------------------------------------- a carved session
struct Carved {
    SessionState ss;
    void* base = nullptr;
    int dev = 0;
    int64_t lo = 0, hi = 0;
    const char* name = "";
};

void carve(Carved& c, const ModelGeometry& g, int64_t max_cells, int dev, int64_t lo, int64_t hi, const char* name) {
    const OnDevice on(dev);
    c.dev = dev; c.lo = lo; c.hi = hi; c.name = name;
    const uint64_t bytes = session_bytes(g, max_cells, kExperts, lo, hi);
    cuda_ok(cudaMalloc(&c.base, bytes), name);
    if (session_init(g, max_cells, kExperts, c.base, c.ss, lo, hi) == 0)
        fail(std::string(name) + ": session_init returned 0 (a half-built QSA state)");
    session_zero(c.ss, g, nullptr, nullptr);
    cuda_ok(cudaDeviceSynchronize(), name);
}
void release(Carved& c) {
    const OnDevice on(c.dev);
    session_release(c.ss);
    delete[] c.ss.qsa_states;
    c.ss.qsa_states = nullptr;
    if (c.base != nullptr) (void) cudaFree(c.base);
    c.base = nullptr;
}

/// The four pool parts in conversation_kv_save's order (k, v, k_scale, v_scale): the authoritative ones (the host copy
/// when streamed) or, with `resident`, the VRAM slots.  Mirrors conversation_snapshot.cpp's pools().
std::array<void*, 4> part_ptrs(const QsaState& st, bool resident) {
    const bool host = st.kv_mode != 0 && !resident;
    if (st.kv_hybrid)
        return {host ? (void*) st.host.k_q : (void*) st.k_q, host ? (void*) st.host.v_q4 : (void*) st.v_q4,
                host ? (void*) st.host.k_scale : (void*) st.k_scale, nullptr};
    if (st.kv_q4)
        return {host ? (void*) st.host.k_q4 : (void*) st.k_q4, host ? (void*) st.host.v_q4 : (void*) st.v_q4, nullptr, nullptr};
    if (st.kv_int8)
        return {host ? (void*) st.host.k_q : (void*) st.k_q, host ? (void*) st.host.v_q : (void*) st.v_q,
                host ? (void*) st.host.k_scale : (void*) st.k_scale, host ? (void*) st.host.v_scale : (void*) st.v_scale};
    return {host ? (void*) st.host.k_pool : (void*) st.k_pool, host ? (void*) st.host.v_pool : (void*) st.v_pool, nullptr, nullptr};
}

/// The bytes of each part at `upto` cells (whole pages, cells first: the identity layout) - the contract's own rule
/// for the slice [bytes(c0'), bytes(c1')) of a pool.
std::array<uint64_t, 5> sizes_at(const QsaState& st, const ModelGeometry& g, int64_t upto) {
    std::array<uint64_t, 5> s{};
    std::string err;
    if (!conversation_kv_part_sizes(st, g, upto, true, s, err)) fail("conversation_kv_part_sizes: " + err);
    return s;
}

void copy_bytes(void* dst, const void* src, size_t n, const char* what) {
    if (n == 0) return;
    if (dst == nullptr || src == nullptr) fail(std::string(what) + ": null pointer");
    cuda_ok(cudaMemcpy(dst, src, n, cudaMemcpyDefault), what);
}

// ---------------------------------------------------------------- a session's movable state as host bytes
struct QsaImg {
    std::array<std::vector<uint8_t>, 4> pool;   ///< every cell of the authoritative pools
    std::vector<uint8_t> pooled, tail, dead, block_pos;
};
struct Image {
    std::map<int64_t, QsaImg> qsa;                 ///< by global QSA ordinal
    std::map<int64_t, std::vector<uint8_t>> gdn;   ///< by global GDN ordinal
    std::vector<uint8_t> ple;
};

ConversationStateSizes state_sizes(const Carved& c, const ModelGeometry& g) {
    ConversationStateSizes z;
    std::string err;
    if (!conversation_session_sizes(g, c.ss, z, err)) fail(std::string(c.name) + ": conversation_session_sizes: " + err);
    return z;
}

/// An image of `c`'s carve filled from `base` (every ordinal the session owns).
Image make_image(const Carved& c, const ModelGeometry& g, uint64_t base) {
    Image im;
    const ConversationStateSizes z = state_sizes(c, g);
    for (int64_t j = 0; j < c.ss.qsa_alloc; ++j) {
        const int64_t q = c.ss.qsa_ord0 + j;
        const QsaState& st = c.ss.qsa_states[q];
        const auto whole = sizes_at(st, g, st.max_cells);
        QsaImg& iq = im.qsa[q];
        for (int i = 0; i < 4; ++i) iq.pool[(size_t) i] = pattern(seed_of(base, q, i), (size_t) whole[(size_t) i]);
        iq.pooled = pattern(seed_of(base, q, 4), (size_t) st.idx_pooled_rows * (size_t) g.idx_key_dim * 4);
        iq.tail = pattern(seed_of(base, q, 5), z.tail);
        iq.dead = pattern(seed_of(base, q, 6), z.dead);
        iq.block_pos = pattern(seed_of(base, q, 7), z.block_pos);
    }
    for (int64_t j = 0; j < c.ss.gdn_alloc; ++j) {
        const int64_t o = c.ss.gdn_ord0 + j;
        im.gdn[o] = pattern(seed_of(base, 1000 + o, 8), (size_t) ep::gdn_row_floats(g) * 4);
    }
    im.ple = pattern(seed_of(base, 2000, 9), z.ple);
    return im;
}

void write_image(const Carved& c, const ModelGeometry& g, const Image& im) {
    const OnDevice on(c.dev);
    for (const auto& [q, iq] : im.qsa) {
        if (q < c.ss.qsa_ord0 || q >= c.ss.qsa_ord0 + c.ss.qsa_alloc)
            fail(std::string(c.name) + ": image ordinal " + num(q) + " outside the carve");
        const QsaState& st = c.ss.qsa_states[q];
        const auto p = part_ptrs(st, false);
        for (int i = 0; i < 4; ++i) copy_bytes(p[(size_t) i], iq.pool[(size_t) i].data(), iq.pool[(size_t) i].size(), "write pool");
        copy_bytes(st.idx_pooled, iq.pooled.data(), iq.pooled.size(), "write idx_pooled");
        copy_bytes(st.idx_tail, iq.tail.data(), iq.tail.size(), "write idx_tail");
        copy_bytes(st.idx_dead, iq.dead.data(), iq.dead.size(), "write idx_dead");
        copy_bytes(st.idx_block_pos, iq.block_pos.data(), iq.block_pos.size(), "write idx_block_pos");
    }
    for (const auto& [o, row] : im.gdn) {
        if (o < c.ss.gdn_ord0 || o >= c.ss.gdn_ord0 + c.ss.gdn_alloc)
            fail(std::string(c.name) + ": GDN ordinal " + num(o) + " outside the carve");
        copy_bytes(c.ss.gdn_state + (size_t) (o - c.ss.gdn_ord0) * (size_t) ep::gdn_row_floats(g), row.data(), row.size(), "write gdn");
    }
    copy_bytes(c.ss.ple_hist, im.ple.data(), im.ple.size(), "write ple");
    cuda_ok(cudaDeviceSynchronize(), "write_image");
}

Image read_image(const Carved& c, const ModelGeometry& g) {
    const OnDevice on(c.dev);
    Image im = make_image(c, g, 0);   // (the shapes; every byte is overwritten below)
    for (auto& [q, iq] : im.qsa) {
        const QsaState& st = c.ss.qsa_states[q];
        const auto p = part_ptrs(st, false);
        for (int i = 0; i < 4; ++i) copy_bytes(iq.pool[(size_t) i].data(), p[(size_t) i], iq.pool[(size_t) i].size(), "read pool");
        copy_bytes(iq.pooled.data(), st.idx_pooled, iq.pooled.size(), "read idx_pooled");
        copy_bytes(iq.tail.data(), st.idx_tail, iq.tail.size(), "read idx_tail");
        copy_bytes(iq.dead.data(), st.idx_dead, iq.dead.size(), "read idx_dead");
        copy_bytes(iq.block_pos.data(), st.idx_block_pos, iq.block_pos.size(), "read idx_block_pos");
    }
    for (auto& [o, row] : im.gdn)
        copy_bytes(row.data(), c.ss.gdn_state + (size_t) (o - c.ss.gdn_ord0) * (size_t) ep::gdn_row_floats(g), row.size(), "read gdn");
    copy_bytes(im.ple.data(), c.ss.ple_hist, im.ple.size(), "read ple");
    cuda_ok(cudaDeviceSynchronize(), "read_image");
    return im;
}

bool same_vec(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, const std::string& what, std::string& where) {
    if (got.size() != want.size()) {
        where = what + ": " + num((int64_t) got.size()) + " bytes, want " + num((int64_t) want.size());
        return false;
    }
    if (got.empty() || std::memcmp(got.data(), want.data(), got.size()) == 0) return true;
    size_t i = 0;
    while (got[i] == want[i]) ++i;
    where = what + ": byte " + num((int64_t) i) + " is " + num(got[i]) + ", want " + num(want[i]);
    return false;
}
bool same_image(const Image& got, const Image& want, std::string& where) {
    if (got.qsa.size() != want.qsa.size() || got.gdn.size() != want.gdn.size()) { where = "image shapes differ"; return false; }
    static const char* part_names[4] = {"k", "v", "k_scale", "v_scale"};
    for (const auto& [q, wq] : want.qsa) {
        const auto it = got.qsa.find(q);
        if (it == got.qsa.end()) { where = "QSA ordinal " + num(q) + " missing"; return false; }
        const QsaImg& gq = it->second;
        const std::string tag = "QSA ordinal " + num(q) + " ";
        for (int i = 0; i < 4; ++i)
            if (!same_vec(gq.pool[(size_t) i], wq.pool[(size_t) i], tag + part_names[i], where)) return false;
        if (!same_vec(gq.pooled, wq.pooled, tag + "idx_pooled", where) || !same_vec(gq.tail, wq.tail, tag + "idx_tail", where) ||
            !same_vec(gq.dead, wq.dead, tag + "idx_dead", where) || !same_vec(gq.block_pos, wq.block_pos, tag + "idx_block_pos", where))
            return false;
    }
    for (const auto& [o, wrow] : want.gdn) {
        const auto it = got.gdn.find(o);
        if (it == got.gdn.end()) { where = "GDN ordinal " + num(o) + " missing"; return false; }
        if (!same_vec(it->second, wrow, "GDN ordinal " + num(o) + " row", where)) return false;
    }
    return same_vec(got.ple, want.ple, "ple_hist", where);
}

/// The contract applied on the host: what `dst` (the destination's image before the handover) must hold after the
/// cells [c0, c1) of every ordinal both images have (the handover's range) crossed from `src`.
void apply_move(Image& dst, const Image& src, const Carved& dst_c, const ModelGeometry& g, int64_t c0, int64_t c1, bool running) {
    const auto s = strata::kernels::qsa_real_shapes();
    const int64_t page = s.page_size;
    const int64_t c0p = c0 / page * page, c1p = (c1 + page - 1) / page * page;
    for (auto& [q, dq] : dst.qsa) {
        const auto it = src.qsa.find(q);
        if (it == src.qsa.end()) continue;
        const QsaImg& sq = it->second;
        const QsaState& st = dst_c.ss.qsa_states[q];
        const auto b0 = sizes_at(st, g, c0p), b1 = sizes_at(st, g, c1p);
        for (int i = 0; i < 4; ++i) {
            const size_t lo = (size_t) b0[(size_t) i], hi = (size_t) b1[(size_t) i];
            if (hi > lo) std::copy(sq.pool[(size_t) i].begin() + (std::ptrdiff_t) lo, sq.pool[(size_t) i].begin() + (std::ptrdiff_t) hi,
                                   dq.pool[(size_t) i].begin() + (std::ptrdiff_t) lo);
        }
        const size_t row_bytes = (size_t) g.idx_key_dim * 4;
        const size_t r0 = (size_t) (c0 / s.idx_block) * row_bytes, r1 = (size_t) (c1 / s.idx_block + 1) * row_bytes;
        std::copy(sq.pooled.begin() + (std::ptrdiff_t) r0, sq.pooled.begin() + (std::ptrdiff_t) r1, dq.pooled.begin() + (std::ptrdiff_t) r0);
        if (running) { dq.tail = sq.tail; dq.dead = sq.dead; dq.block_pos = sq.block_pos; }
    }
    if (running)
        for (auto& [o, row] : dst.gdn) {
            const auto it = src.gdn.find(o);
            if (it != src.gdn.end()) row = it->second;
        }
}

// ---------------------------------------------------------------- the streamed state's VRAM slots
/// kv_mode 1 only: a block resident in slot `slot` of every pool of ordinal `q`, with the page table and the slot map
/// naming it - the stale resident block a handover into this state must invalidate.
void plant_resident(const Carved& c, const ModelGeometry& g, int64_t q, int64_t block, int32_t slot, uint64_t seed) {
    const QsaState& st = c.ss.qsa_states[q];
    if (st.kv_mode != 1) return;
    const OnDevice on(c.dev);
    const auto whole = sizes_at(st, g, st.max_cells);
    const auto res = part_ptrs(st, true);
    for (int i = 0; i < 4; ++i) {
        if (whole[(size_t) i] == 0 || res[(size_t) i] == nullptr) continue;
        const size_t per_page = (size_t) (whole[(size_t) i] / (uint64_t) st.n_pages);
        const auto bytes = pattern(seed + (uint64_t) i, per_page);
        copy_bytes((uint8_t*) res[(size_t) i] + (size_t) slot * per_page, bytes.data(), per_page, "plant slot");
    }
    const int32_t b = (int32_t) block;
    copy_bytes(st.page_table + block, &slot, sizeof(int32_t), "plant page_table");
    copy_bytes(st.map.slot_block + slot, &b, sizeof(int32_t), "plant slot_block");
    cuda_ok(cudaDeviceSynchronize(), "plant_resident");
}
bool table_all_free(const Carved& c, int64_t q) {
    const QsaState& st = c.ss.qsa_states[q];
    const OnDevice on(c.dev);
    std::vector<int32_t> tab((size_t) st.n_pages);
    copy_bytes(tab.data(), st.page_table, tab.size() * sizeof(int32_t), "read page_table");
    return std::all_of(tab.begin(), tab.end(), [](int32_t x) { return x == -1; });
}
/// kv_mode 1: the slot the page table names for `block` (-1: not resident).
int32_t table_entry(const Carved& c, int64_t q, int64_t block) {
    const QsaState& st = c.ss.qsa_states[q];
    const OnDevice on(c.dev);
    int32_t v = -1;
    copy_bytes(&v, st.page_table + block, sizeof(int32_t), "read page_table entry");
    return v;
}

std::string stats_line(const ep::Stats& st) {
    return "cells [" + num(st.cells0) + ", " + num(st.cells1) + "), qsa " + num(st.qsa_layers) + ", gdn " + num(st.gdn_layers) +
           ", kv_bytes " + num((int64_t) st.kv_bytes) + ", pooled " + num((int64_t) st.pooled_bytes) + ", state " +
           num((int64_t) st.state_bytes) + (st.kv_host_memcpy ? ", host memcpy" : ", D2H+H2D");
}

}  // namespace

int main(int argc, char** argv) {
    int primary = argc > 1 ? std::atoi(argv[1]) : 0;
    int peer = argc > 2 ? std::atoi(argv[2]) : 0;
    int64_t max_cells = argc > 3 ? std::atoll(argv[3]) : 8192;
    const int64_t kv_resident = argc > 4 ? std::atoll(argv[4]) : 0;

    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0) {
        std::fprintf(stderr, "ep_split_test: no CUDA device\n");
        return 77;
    }
    g_check = "args";
    if (primary < 0 || primary >= n_dev || peer < 0 || peer >= n_dev)
        fail("devices " + num(primary) + " / " + num(peer) + " with " + num(n_dev) + " visible");
    const auto shapes = strata::kernels::qsa_real_shapes();
    if (max_cells < kN + shapes.page_size) fail("max_cells must be at least " + num(kN + shapes.page_size));
    max_cells = (max_cells + shapes.page_size - 1) / shapes.page_size * shapes.page_size;   // whole pages (as the engine's)

    // --kv-resident: the carve streams a layer only past qsa_kv_resident_min() cells (layer.cpp kv_plan), so a smaller
    // max_cells would carve kv_mode 0 states and test the wrong path
    if (kv_resident > 0) {
        const int64_t need = std::max(kv_resident, qsa_kv_resident_min()) + 4096;
        if (max_cells < need) {
            std::printf("ep_split_test: kv_resident %lld needs max_cells > %lld to stream: max_cells %lld -> %lld\n",
                        (long long) kv_resident, (long long) std::max(kv_resident, qsa_kv_resident_min()),
                        (long long) max_cells, (long long) need);
            max_cells = need;
        }
        qsa_set_kv_resident(kv_resident);
    }
    if (const char* fmt = std::getenv("STRATA_EP_SPLIT_TEST_KV")) {
        if (std::strcmp(fmt, "int8") == 0) qsa_set_kv_int8(true);
        else if (std::strcmp(fmt, "q4") == 0) qsa_set_kv_q4(true);
        else fail(std::string("STRATA_EP_SPLIT_TEST_KV=") + fmt + " (int8 or q4)");
        std::printf("ep_split_test: K/V format %s\n", fmt);
    }

    // The geometry: the compiled defaults (the real GDN / QSA shapes) over kNLayers layers.  conversation_snapshot_test
    // shrinks the GDN state (ssm_state_size 2, ssm_v_heads 2, ssm_conv_channels 8) because it carves nothing; this test
    // carves real sessions, so the rows are the real 3.1 MiB and the GDN handover moves what the engine's moves.
    ModelGeometry g;
    g.n_layers = kNLayers;
    const int64_t n_qsa = g.n_qsa_layers(), n_gdn = g.n_gdn_layers();
    int64_t qsa_in_range = 0, gdn_in_range = 0;
    for (int64_t l = kLb; l < kLe; ++l) (is_qsa_layer(g, l) ? qsa_in_range : gdn_in_range) += 1;
    std::string err;

    Carved full, tail, tail2, ref;
    carve(full, g, max_cells, primary, 0, kNLayers, "full");
    carve(tail, g, max_cells, peer, kLb, kLe, "tail");
    carve(tail2, g, max_cells, peer, kLb, kLe, "tail2");
    carve(ref, g, max_cells, primary, 0, kNLayers, "ref");

    g_check = "carve";
    const int kv_mode = full.ss.qsa_states[full.ss.qsa_ord0].kv_mode;
    expect(kv_mode == (kv_resident > 0 ? 1 : 0), "kv_mode " + num(kv_mode) + " with kv_resident " + num(kv_resident));
    expect(full.ss.qsa_ord0 == 0 && full.ss.qsa_alloc == n_qsa && full.ss.gdn_ord0 == 0 && full.ss.gdn_alloc == n_gdn,
           "the full carve: qsa " + num(full.ss.qsa_ord0) + "+" + num(full.ss.qsa_alloc) + ", gdn " + num(full.ss.gdn_ord0) + "+" + num(full.ss.gdn_alloc));
    expect(tail.ss.qsa_alloc == qsa_in_range && tail.ss.gdn_alloc == gdn_in_range && tail.ss.qsa_ord0 + tail.ss.qsa_alloc == n_qsa &&
           tail.ss.gdn_ord0 + tail.ss.gdn_alloc == n_gdn,
           "the tail carve: qsa " + num(tail.ss.qsa_ord0) + "+" + num(tail.ss.qsa_alloc) + ", gdn " + num(tail.ss.gdn_ord0) + "+" + num(tail.ss.gdn_alloc) +
           " (want " + num(qsa_in_range) + " QSA, " + num(gdn_in_range) + " GDN)");
    for (int64_t j = 0; j < tail.ss.qsa_alloc; ++j) {
        const QsaState& a = tail.ss.qsa_states[tail.ss.qsa_ord0 + j];
        const QsaState& b = full.ss.qsa_states[tail.ss.qsa_ord0 + j];
        expect(a.kv_mode == b.kv_mode && a.max_cells == b.max_cells && a.n_pages == b.n_pages && a.idx_pooled_rows == b.idx_pooled_rows,
               "the two carves of QSA ordinal " + num(tail.ss.qsa_ord0 + j) + " differ");
    }
    std::printf("ep_split_test: %lld layers, QSA %lld, GDN %lld; stage [%lld, %lld) = %lld QSA + %lld GDN; max_cells %lld, kv_mode %d, "
                "primary CUDA%d, peer CUDA%d\n", (long long) kNLayers, (long long) n_qsa, (long long) n_gdn, (long long) kLb,
                (long long) kLe, (long long) qsa_in_range, (long long) gdn_in_range, (long long) max_cells, kv_mode, primary, peer);

    // ---------------------------------------------------------------- 1: the duplicated GDN row formula
    g_check = "1 gdn_row_floats";
    {
        const uint64_t row = ep::gdn_row_floats(g) * 4;
        // layer 0 is a GDN layer and the carves [0, n) and [1, n) hold the same QSA states: they differ by one row
        expect(!is_qsa_layer(g, 0), "layer 0 is a QSA layer");
        const uint64_t with = session_bytes(g, max_cells, kExperts, 0, kNLayers);
        const uint64_t without = session_bytes(g, max_cells, kExperts, 1, kNLayers);
        expect(with - without == row, "session_bytes [0, n) - [1, n) = " + num((int64_t) (with - without)) + ", gdn_row_floats * 4 = " + num((int64_t) row));
        ConversationStateSizes z;
        expect(conversation_state_sizes(g, z, err), "conversation_state_sizes: ", err);
        expect(z.gdn == row * (uint64_t) n_gdn, "conversation_state_sizes.gdn " + num((int64_t) z.gdn) + " != " + num(n_gdn) + " rows of " + num((int64_t) row));
        const ConversationStateSizes zt = state_sizes(tail, g);
        expect(zt.gdn == row * (uint64_t) tail.ss.gdn_alloc, "conversation_session_sizes(tail).gdn " + num((int64_t) zt.gdn) + " != gdn_alloc rows");
        ok();
    }

    ep::StateMover m;
    g_check = "mover init";
    expect(m.init(g, kLb, kLe, primary, peer, max_cells, err), "init: ", err);
    expect(m.valid() && m.lb() == kLb && m.le() == kLe, "init left the mover invalid or with another range");
    ep::Stats stats;
    std::string where;

    // ---------------------------------------------------------------- 2: the full handover to the primary
    g_check = "2 to_primary";
    Image im_full = make_image(full, g, 0xA), im_tail = make_image(tail, g, 0xB);
    write_image(full, g, im_full);
    write_image(tail, g, im_tail);
    for (int64_t j = 0; j < tail.ss.qsa_alloc; ++j)   // a stale resident block in the TARGET's slots (kv_mode 1)
        plant_resident(full, g, tail.ss.qsa_ord0 + j, 1, 0, 0xC0 + (uint64_t) j * 16);
    Image want;
    {
        stats = ep::Stats{};
        expect(m.to_primary(tail.ss, full.ss, 0, kN, true, stats, err), "to_primary: ", err);
        want = im_full;
        apply_move(want, im_tail, full, g, 0, kN, true);
        Image got = read_image(full, g);
        expect(same_image(got, want, where), where);
        for (int64_t j = 0; j < tail.ss.qsa_alloc; ++j)
            expect(kv_mode != 1 || table_all_free(full, tail.ss.qsa_ord0 + j),
                   "page_table of QSA ordinal " + num(tail.ss.qsa_ord0 + j) + " not reset after the handover (kv_stream_reset)");
        const int64_t c1p = (kN + shapes.page_size - 1) / shapes.page_size * shapes.page_size;
        expect(stats.cells0 == 0 && stats.cells1 >= kN && stats.cells1 <= c1p, "stats " + stats_line(stats));
        expect(stats.qsa_layers == qsa_in_range && stats.gdn_layers == gdn_in_range, "stats " + stats_line(stats));
        expect(stats.kv_bytes > 0 && stats.pooled_bytes > 0 && stats.state_bytes > 0, "stats " + stats_line(stats));
        expect(stats.kv_host_memcpy == (kv_mode == 1), "stats " + stats_line(stats));
        expect(m.handovers() == 1 && m.bytes_total() > 0, "handovers " + num(m.handovers()) + ", bytes_total " + num((int64_t) m.bytes_total()));
        Image got_tail = read_image(tail, g);
        expect(same_image(got_tail, im_tail, where), "the stage's session changed: ", where);
        std::printf("   %s, %.2f ms\n", stats_line(stats).c_str(), stats.ms);
        ok();
    }

    // ---------------------------------------------------------------- 4: the round trip back to a stage session
    g_check = "4 to_peer round trip";
    {
        Image im_tail2 = read_image(tail2, g);   // session_zero'd, whatever the host copies hold: the baseline
        for (int64_t j = 0; j < tail2.ss.qsa_alloc; ++j)
            plant_resident(tail2, g, tail2.ss.qsa_ord0 + j, 2, 0, 0xD0 + (uint64_t) j * 16);
        stats = ep::Stats{};
        expect(m.to_peer(full.ss, tail2.ss, 0, kN, true, stats, err), "to_peer: ", err);
        Image want2 = im_tail2;
        apply_move(want2, im_tail, tail2, g, 0, kN, true);   // full holds the tail's bytes there, so tail2 must hold the tail's
        Image got = read_image(tail2, g);
        expect(same_image(got, want2, where), where);
        for (int64_t j = 0; j < tail2.ss.qsa_alloc; ++j)
            expect(kv_mode != 1 || table_all_free(tail2, tail2.ss.qsa_ord0 + j),
                   "page_table of QSA ordinal " + num(tail2.ss.qsa_ord0 + j) + " not reset after the handover back");
        expect(stats.qsa_layers == qsa_in_range && stats.gdn_layers == gdn_in_range && stats.kv_bytes > 0, "stats " + stats_line(stats));
        Image got_full = read_image(full, g);
        expect(same_image(got_full, want, where), "the primary's session changed on the way back: ", where);
        expect(m.handovers() == 2, "handovers " + num(m.handovers()));
        std::printf("   %s, %.2f ms\n", stats_line(stats).c_str(), stats.ms);
        ok();
    }

    // ---------------------------------------------------------------- 3: a partial handover, whole pages only
    g_check = "3 partial pages";
    {
        im_full = make_image(full, g, 0xD);   // a fresh "before" for the full session
        write_image(full, g, im_full);
        const int64_t page = shapes.page_size;
        const int64_t c0 = 25 * page + page / 2, c1 = 60 * page + 1;   // pages [25, 61) change, rows [25, 60] of the indexer
        stats = ep::Stats{};
        expect(m.to_primary(tail.ss, full.ss, c0, c1, false, stats, err), "to_primary: ", err);
        want = im_full;
        apply_move(want, im_tail, full, g, c0, c1, false);
        Image got = read_image(full, g);
        expect(same_image(got, want, where), where);
        expect(stats.cells0 >= c0 / page * page && stats.cells0 <= c0 && stats.cells1 >= c1 && stats.cells1 <= (c1 + page - 1) / page * page,
               "stats " + stats_line(stats));
        expect(stats.qsa_layers == qsa_in_range && stats.gdn_layers == 0, "stats " + stats_line(stats) + " (running_state false: no GDN row)");
        expect(stats.state_bytes == 0, "stats " + stats_line(stats) + " (running_state false: no state bytes)");
        std::printf("   %s, %.2f ms\n", stats_line(stats).c_str(), stats.ms);
        ok();
    }

    // ---------------------------------------------------------------- 9: the dense weights' region at the tier's end (tappa 4c)
    g_check = "9 tail_region";
    {
        // uniform slots: 100 slots of 1000 bytes
        ep::TailRegion r = ep::tail_region(100, nullptr, 1000, 100000, 2500, 128);
        expect(r.first < 0, "a region that leaves fewer than `keep` slots was accepted");
        r = ep::tail_region(100, nullptr, 1000, 100000, 2500, 10);
        expect(r.first >= 0, "a 2500-byte region in a 100000-byte arena refused");
        expect(r.offset % ep::kTailAlign == 0 && r.offset <= 100000 - 2500 && 100000 - r.offset < 2500 + ep::kTailAlign,
               "offset " + num((int64_t) r.offset) + " is not the aligned-down start of the last 2500 bytes");
        // slot s is [1000 s, 1000 (s + 1)): the first slot whose end lies past the offset
        expect(r.first == (int64_t) (r.offset / 1000), "first " + num(r.first) + " for offset " + num((int64_t) r.offset));
        expect(r.bytes == 2500, "bytes " + num((int64_t) r.bytes));
        // sized slots: offsets 0, 300, 1000, 1500, 4000 (4 slots, arena 4000 bytes); a 700-byte region starts at
        // 3300 -> aligned down to 3072 -> overlaps slot 3 only ([1500, 4000))
        const uint64_t off[5] = {0, 300, 1000, 1500, 4000};
        r = ep::tail_region(4, off, 0, 4000, 700, 0);
        expect(r.first == 3 && r.offset == 3072, "sized slots: first " + num(r.first) + ", offset " + num((int64_t) r.offset));
        // a region reaching into slot 2 ([1000, 1500)): 2600 bytes -> start 1400 -> aligned 1280 -> slot 2
        r = ep::tail_region(4, off, 0, 4000, 2600, 0);
        expect(r.first == 2 && r.offset == 1280, "sized slots, two slots: first " + num(r.first) + ", offset " + num((int64_t) r.offset));
        // keep 3 of 4: that region would leave 2 -> refused
        r = ep::tail_region(4, off, 0, 4000, 2600, 3);
        expect(r.first < 0, "the keep floor was not applied on sized slots");
        // larger than the arena, empty arena, zero bytes: refused without touching the result's bytes
        expect(ep::tail_region(4, off, 0, 4000, 5000, 0).first < 0, "a region larger than the arena was accepted");
        expect(ep::tail_region(0, nullptr, 1000, 0, 100, 0).first < 0, "an empty arena gave a region");
        expect(ep::tail_region(4, off, 0, 4000, 0, 0).first < 0, "a zero-byte region was accepted");
        ok();
    }

    // ---------------------------------------------------------------- 10: the overlapped handover (tappa 4c, s.14.3)
    g_check = "10 overlapped handover";
    {
        // full: fresh bytes, then the tail's cells chunk by chunk (no running state), then the last chunk with it;
        // ref: the same fresh bytes, then ONE whole handover.  The two must be identical, and equal the host contract.
        im_full = make_image(full, g, 0x1A);
        write_image(full, g, im_full);
        write_image(ref, g, im_full);
        const int64_t page = shapes.page_size;
        const int64_t n_blocks_range = (kN + page - 1) / page;   // blocks [0, ceil(kN / page))
        int64_t n_pages_q = 0;
        for (int64_t j = 0; j < tail.ss.qsa_alloc; ++j) {   // kv_mode 1: a stale block inside the range, one outside it
            const int64_t q = tail.ss.qsa_ord0 + j;
            n_pages_q = full.ss.qsa_states[q].n_pages;
            plant_resident(full, g, q, 1, 0, 0xE0 + (uint64_t) j * 16);
            if (n_pages_q - 1 >= n_blocks_range) plant_resident(full, g, q, n_pages_q - 1, 1, 0xF0 + (uint64_t) j * 16);
        }
        // chunks that do not fall on page boundaries: the page they share is written twice
        const int64_t cuts[] = {0, 301, 602, 903, kN};
        int64_t last_from = 0;
        const int64_t h0 = m.handovers();
        for (int i = 0; i + 1 < (int) (sizeof cuts / sizeof cuts[0]); ++i) {
            stats = ep::Stats{};
            expect(m.to_primary(tail.ss, full.ss, cuts[i], cuts[i + 1], false, stats, err), "chunk handover: ", err);
            expect(stats.gdn_layers == 0 && stats.state_bytes == 0, "a chunk handover moved the running state: " + stats_line(stats));
            last_from = cuts[i];
        }
        stats = ep::Stats{};
        expect(m.to_primary(tail.ss, full.ss, last_from, kN, true, stats, err), "last chunk with the running state: ", err);
        expect(stats.gdn_layers == gdn_in_range && stats.state_bytes > 0, "the last handover did not move the running state: " + stats_line(stats));
        expect(m.handovers() == h0 + 5, "handovers " + num(m.handovers() - h0) + " (want 5)");
        stats = ep::Stats{};
        expect(m.to_primary(tail.ss, ref.ss, 0, kN, true, stats, err), "whole handover into ref: ", err);
        const Image got = read_image(full, g), got_ref = read_image(ref, g);
        expect(same_image(got, got_ref, where), "chunk by chunk differs from one whole handover: " + where);
        want = im_full;
        apply_move(want, im_tail, full, g, 0, kN, true);
        expect(same_image(got, want, where), "chunk by chunk differs from the host contract: " + where);
        if (kv_mode == 1) {
            // the eviction is per block: every block of the range left the map, a resident block past it stays
            // (unless the range was most of the map or held more than kResetAboveResident resident blocks: then the
            // whole map was reset, and the block past the range is gone too - said, not failed)
            const bool partial = n_blocks_range * 2 < n_pages_q && n_blocks_range <= ep::kResetAboveResident;
            for (int64_t j = 0; j < tail.ss.qsa_alloc; ++j) {
                const int64_t q = tail.ss.qsa_ord0 + j;
                expect(table_entry(full, q, 1) == -1, "block 1 of QSA ordinal " + num(q) + " is still resident after the handover");
                if (n_pages_q - 1 >= n_blocks_range) {
                    const int32_t far = table_entry(full, q, n_pages_q - 1);
                    if (partial) expect(far == 1, "block " + num(n_pages_q - 1) + " of QSA ordinal " + num(q) + " (outside the range) was evicted");
                    else std::printf("   (the range is most of the map: the whole map was reset, block %lld -> %d)\n", (long long) (n_pages_q - 1), far);
                }
            }
        }
        std::printf("   5 handovers, the last with the state: %s, %.2f ms\n", stats_line(stats).c_str(), stats.ms);
        ok();
    }

    // ---------------------------------------------------------------- 11: the dense mirror (tappa 4c, s.14.2)
    g_check = "11 DenseMirror";
    {
        const uint64_t bytes = (uint64_t) 3 << 20 | 0x1234;   // not a multiple of anything
        const std::vector<uint8_t> weights = pattern(0xDE5E, (size_t) bytes), other = pattern(0x0DD, (size_t) bytes);
        uint8_t* dev = nullptr;
        cudaStream_t s = nullptr;
        {
            const OnDevice on(peer);
            cuda_ok(cudaMalloc((void**) &dev, (size_t) bytes), "cudaMalloc the dense region");
            cuda_ok(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
            cuda_ok(cudaMemcpy(dev, weights.data(), (size_t) bytes, cudaMemcpyHostToDevice), "upload the weights");
        }
        ep::DenseMirror mir;
        err.clear();
        expect(!mir.valid() && !mir.restore(dev, s, true, err) && !err.empty(), "restore before capture was accepted");
        expect(mir.capture(dev, bytes, peer, err), "capture: ", err);
        expect(mir.valid() && mir.bytes() == bytes && mir.device() == peer && mir.dev_base() == dev, "the mirror's shape");
        std::printf("   mirror of %llu bytes in %s RAM\n", (unsigned long long) bytes, mir.pinned() ? "pinned" : "pageable");
        {   // the experts write over the region (what a refill does)
            const OnDevice on(peer);
            cuda_ok(cudaMemcpy(dev, other.data(), (size_t) bytes, cudaMemcpyHostToDevice), "overwrite the region");
        }
        err.clear();
        expect(!mir.restore(dev + 256, s, true, err) && !err.empty(), "a restore to another address was accepted");
        expect(mir.restore(dev, s, true, err), "restore: ", err);
        expect(mir.restores() == 1 && mir.last_ms() >= 0.0, "restores " + num(mir.restores()));
        std::vector<uint8_t> back((size_t) bytes);
        {
            const OnDevice on(peer);
            cuda_ok(cudaMemcpy(back.data(), dev, (size_t) bytes, cudaMemcpyDeviceToHost), "read the region back");
        }
        expect(same_vec(back, weights, "dense region", where), where);
        // the queued form: ordered on the stream before a later copy on the same stream
        {
            const OnDevice on(peer);
            cuda_ok(cudaMemcpy(dev, other.data(), (size_t) bytes, cudaMemcpyHostToDevice), "overwrite the region again");
        }
        expect(mir.restore(dev, s, false, err), "queued restore: ", err);
        expect(mir.last_ms() == 0.0 && mir.restores() == 2, "a queued restore reported a time");
        {
            const OnDevice on(peer);
            cuda_ok(cudaMemcpyAsync(back.data(), dev, (size_t) bytes, cudaMemcpyDeviceToHost, s), "read back on the stream");
            cuda_ok(cudaStreamSynchronize(s), "stream sync");
        }
        expect(same_vec(back, weights, "dense region after the queued restore", where), where);
        mir.close();
        expect(!mir.valid() && mir.bytes() == 0, "close left the mirror valid");
        {
            const OnDevice on(peer);
            (void) cudaStreamDestroy(s);
            (void) cudaFree(dev);
        }
        ok();
    }

    // ---------------------------------------------------------------- 12: the threshold (tappa 4c, s.14.1)
    g_check = "12 staged_read";
    {
        expect(ep::staged_read(100, 0) && ep::staged_read(0, 0), "min_tokens 0 is not 'always staged'");
        expect(!ep::staged_read(16383, 16384) && ep::staged_read(16384, 16384) && ep::staged_read(200000, 16384), "the threshold is not inclusive at min_tokens");
        expect(!ep::staged_read(0, 1), "an empty read above a threshold of 1");
        if (std::getenv(ep::kEnvMinTokens) == nullptr)
            expect(ep::env_min_tokens() == ep::kMinTokensDefault, "the default threshold is " + num(ep::env_min_tokens()) + ", want " + num(ep::kMinTokensDefault));
        if (std::getenv(ep::kEnvOn) == nullptr) expect(ep::env_level() == 0 && !ep::env_on(), "the level reads on with the variable unset");
        ok();
    }

    // ---------------------------------------------------------------- 5: the drafter's rows
    g_check = "5 rows_to_primary";
    {
        const int64_t T = 37, fpr = g.hc * g.n_embd;
        const size_t bytes = (size_t) T * (size_t) fpr * sizeof(float);
        std::vector<float> rows((size_t) T * (size_t) fpr), back(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) rows[i] = (float) ((int) (i % 1999) - 1000) * 0.03125f;
        float* src = nullptr;
        float* dst = nullptr;
        {
            const OnDevice on(peer);
            cuda_ok(cudaMalloc(&src, bytes), "cudaMalloc rows on the peer");
            cuda_ok(cudaMemcpy(src, rows.data(), bytes, cudaMemcpyHostToDevice), "upload rows");
            // the mover's streams are non-blocking: without this its D2H can read src before the upload lands (11/11: FAIL on two cards)
            cuda_ok(cudaDeviceSynchronize(), "sync the upload");
        }
        {
            const OnDevice on(primary);
            cuda_ok(cudaMalloc(&dst, bytes), "cudaMalloc rows on the primary");
            cuda_ok(cudaMemset(dst, 0xFF, bytes), "memset rows");
            cuda_ok(cudaDeviceSynchronize(), "sync the memset");   // the same, for the memset racing the mover's H2D
        }
        expect(m.rows_to_primary(src, dst, T, fpr, err), "rows_to_primary: ", err);
        {
            const OnDevice on(primary);
            cuda_ok(cudaMemcpy(back.data(), dst, bytes, cudaMemcpyDeviceToHost), "read rows back");
        }
        if (std::memcmp(back.data(), rows.data(), bytes) != 0) {
            size_t i = 0;
            while (std::memcmp(&back[i], &rows[i], sizeof(float)) == 0) ++i;
            fail("float " + num((int64_t) i) + " (row " + num((int64_t) (i / (size_t) fpr)) + ") differs");
        }
        ++g_checks;
        { const OnDevice on(peer); (void) cudaFree(src); }
        { const OnDevice on(primary); (void) cudaFree(dst); }
        ok();
    }

    // ---------------------------------------------------------------- 6: what the stage is known to hold
    g_check = "6 Sync";
    {
        std::vector<int32_t> ids(150);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = 1000 + (int32_t) i;
        ep::Sync sync;
        expect(sync.cells == 0 && sync.valid_for(ids.data(), (int64_t) ids.size()) == 0, "a fresh Sync holds cells");
        sync.note(ids.data(), 100);
        expect(sync.cells == 100 && sync.ids.size() == 100, "note(100): cells " + num(sync.cells) + ", ids " + num((int64_t) sync.ids.size()));
        expect(sync.valid_for(ids.data(), 150) == 100, "valid_for on the same ids != 100");
        expect(sync.valid_for(ids.data(), 100) == 100, "valid_for on exactly the kept prefix != 100");
        std::vector<int32_t> other = ids;
        other[50] += 1;
        expect(sync.valid_for(other.data(), 150) == 0, "valid_for on different ids != 0");
        expect(sync.cells == 0 && sync.ids.empty(), "the kept prefix was not dropped after a mismatch");
        sync.note(ids.data(), 100);
        expect(sync.valid_for(ids.data(), 50) == 0, "valid_for on a shorter request (cells > n) != 0");
        sync.note(ids.data(), 100);
        sync.invalidate();
        expect(sync.cells == 0 && sync.ids.empty() && sync.valid_for(ids.data(), 150) == 0, "invalidate left cells");
        ok();
    }

    // ---------------------------------------------------------------- 7: the mid-prompt checkpoint from two parts
    g_check = "7 checkpoint_merge_tail";
    {
        im_full = make_image(full, g, 0xE);   // the full session: its [kLb, n) rows and indexer parts are the STALE ones
        write_image(full, g, im_full);
        // ref: full's bytes below the stage, the tail's from the stage on (written directly, not through the mover)
        Image im_ref = im_full;
        for (auto& [q, iq] : im_ref.qsa) { const auto it = im_tail.qsa.find(q); if (it != im_tail.qsa.end()) iq = it->second; }
        for (auto& [o, row] : im_ref.gdn) { const auto it = im_tail.gdn.find(o); if (it != im_tail.gdn.end()) row = it->second; }
        write_image(ref, g, im_ref);
        std::vector<int32_t> ids((size_t) kN);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = 7 + (int32_t) i * 3;
        ConversationCheckpoint cf, ct, cr;
        cf.ids = ct.ids = cr.ids = ids;
        {
            const OnDevice on(primary);
            expect(conversation_checkpoint_save(cf, full.ss, g, err), "checkpoint of the full session: ", err);
            conversation_checkpoint_join(nullptr);
            expect(conversation_checkpoint_save(cr, ref.ss, g, err), "checkpoint of the reference session: ", err);
            conversation_checkpoint_join(nullptr);
        }
        {
            const OnDevice on(peer);
            expect(conversation_checkpoint_save(ct, tail.ss, g, err), "checkpoint of the stage's session: ", err);
            conversation_checkpoint_join(nullptr);
        }
        const ConversationCheckpoint cf0 = cf;
        expect(cf0.gdn != cr.gdn && cf0.tails != cr.tails, "the fixture is not stale: the full checkpoint already equals the reference");
        expect(ep::checkpoint_merge_tail(cf, ct, full.ss, tail.ss, g, err), "merge: ", err);
        expect(cf.ids == ids, "the merge changed the ids");
        expect(same_vec(cf.gdn, cr.gdn, "gdn", where), where);
        expect(same_vec(cf.tails, cr.tails, "tails", where), where);
        expect(same_vec(cf.dead, cr.dead, "dead", where), where);
        expect(same_vec(cf.block_pos, cr.block_pos, "block_pos", where), where);
        expect(cf.ple == cf0.ple && cf.ple == cr.ple, "the merge touched the PLE history (it belongs to the full session)");
        expect(cf.imgs == cf0.imgs && cf.used == cf0.used && cf.pinned == cf0.pinned, "the merge touched the checkpoint's metadata");
        // different ids between the two parts: refused, with a reason
        ConversationCheckpoint ct_bad = ct, cf2 = cf0;
        ct_bad.ids[0] += 1;
        err.clear();
        expect(!ep::checkpoint_merge_tail(cf2, ct_bad, full.ss, tail.ss, g, err) && !err.empty(), "parts with different ids merged");
        expect(cf2.gdn == cf0.gdn && cf2.tails == cf0.tails, "a refused merge changed the full checkpoint");
        ok();
    }

    // ---------------------------------------------------------------- 8: refusals
    g_check = "8 errors";
    {
        // a ring (kv_mode 2) in the range: a copy of the stage's QsaState array with the mode forced on its first state.
        // (The mover checks a QSA state when it reaches it, so the GDN rows before it may have crossed already: a
        // refused handover ends the request with ERR, docs/EP_SPLIT_PREFILL.md s.7 - no atomicity is claimed or checked.)
        std::vector<QsaState> forced(tail.ss.qsa_states, tail.ss.qsa_states + (size_t) n_qsa);
        forced[(size_t) tail.ss.qsa_ord0].kv_mode = 2;
        QsaState* saved = tail.ss.qsa_states;
        tail.ss.qsa_states = forced.data();
        err.clear();
        const bool ring = m.to_primary(tail.ss, full.ss, 0, kN, true, stats, err);
        tail.ss.qsa_states = saved;
        expect(!ring && !err.empty(), "a kv_mode 2 state in the range was accepted");
        std::printf("   kv_mode 2 refused: %s\n", err.c_str());
        // a kv_mode that differs between the two states of one ordinal (the other direction, the target forced)
        std::vector<QsaState> forced_full(full.ss.qsa_states, full.ss.qsa_states + (size_t) n_qsa);
        forced_full[(size_t) tail.ss.qsa_ord0].kv_mode = 2;
        saved = full.ss.qsa_states;
        full.ss.qsa_states = forced_full.data();
        err.clear();
        const bool ring_to = m.to_peer(full.ss, tail2.ss, 0, kN, true, stats, err);
        full.ss.qsa_states = saved;
        expect(!ring_to && !err.empty(), "a kv_mode 2 state in the source was accepted");
        // c1 past the sessions' capacity: refused before any copy (the range is checked up front), so nothing crosses
        const Image before = read_image(full, g);
        err.clear();
        expect(!m.to_primary(tail.ss, full.ss, 0, max_cells + 1, true, stats, err) && !err.empty(), "c1 > max_cells was accepted");
        std::printf("   c1 > max_cells refused: %s\n", err.c_str());
        err.clear();
        expect(!m.to_peer(full.ss, tail2.ss, 0, max_cells + shapes.page_size, true, stats, err) && !err.empty(), "c1 > max_cells was accepted (to_peer)");
        const Image after = read_image(full, g);
        expect(same_image(after, before, where), "a range refusal wrote into the full session: ", where);
        ok();
    }

    m.close();
    release(full); release(tail); release(tail2); release(ref);
    std::printf("ep_split_test: all OK (%d checks, primary CUDA%d, peer CUDA%d, kv_mode %d)\n", g_checks, primary, peer, kv_mode);
    return 0;
}
