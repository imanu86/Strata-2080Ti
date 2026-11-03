// include/strata/core/peer_experts.hpp - multi-GPU: a second expert tier on another GPU.
//
// The primary GPU (device 0) keeps everything it had: the dense weights, the KV cache, the MTP layer, its own
// adaptive expert tier and the captured verify graph.  This class adds an expert tier on a SECOND card.  From
// the verify graph's point of view nothing changes: for each layer the pool thread decides which routed experts
// the primary computes (its VRAM tier, the PCIe share), and every remaining expert is a "CPU row" the graph reads
// from mapped host memory after the pool raises its flag.  The peer takes the rows whose experts it holds: the
// pool thread launches them on the peer right after publishing the primary's plan, the CPU computes the rest at
// the same time, and the peer's rows are written into the same mapped rows before the pool returns.
//
// The peer computes with the SAME kernels as the primary's hit path (q8_1 activations + native_expert_grouped),
// from the same activation values, so an expert gives bit-identical rows on either card.
//
// The tier follows the conversation like the primary's: `adapt` swaps the most-routed experts that neither card
// holds into the peer's least-used slots (per layer, so a slot keeps its layer's blob size).
//
// Two servers (--peer-device A,B, docs/PEER_FUSED2.md): generate.cpp opens TWO of these, one a card, in order.  The
// second (`ordinal` 1) is filled with the ranked pairs that neither the primary nor the first holds (`open`'s
// `before`), computes the entries the pool marks kind 3 (`launch`'s `mine`; the first keeps kind 2), and its adapt
// never takes a pair the other server holds or is bringing in (`adapt`'s `other`, `claims`): a pair lives on ONE
// server.  Each one has its own streams, buffers, mapped page and persistent kernel; the primary's graph waits on
// both flags.  With one server every one of these parameters is at its default and the code path is the one above.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/peer_fused.hpp"      // STRATA_PEER_FUSED: the persistent-kernel contract (Hook, Window, Shape)
#include "strata/kernels/iq_kernels.hpp"   // NativeExpertLayout

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// multi-GPU: set once at startup when a peer device is configured (--peer-device N).  The mapped host buffers
/// the peer card writes into are then allocated with cudaHostAllocPortable; without a peer the flag stays off.
void set_peer_portable(bool on);
bool peer_portable();

class PeerExperts {
public:
    PeerExperts() = default;
    ~PeerExperts();
    PeerExperts(const PeerExperts&) = delete;
    PeerExperts& operator=(const PeerExperts&) = delete;

    /// Opens the tier on `device`: fills it with the first pairs of `ranked` that `primary` does not hold, as many
    /// as fit in the card's free memory minus `reserve_mib` (`max_slots` > 0 caps the count).  Two servers: `before`
    /// is the server opened before this one (its pairs are skipped too, so a pair is on one card) and `ordinal` its
    /// place (0 the first, as today; 1 the second: it reads the STRATA_PEER2_* knobs).  `before` must be open and idle
    /// (it is: the servers open one after the other at startup) and have the same n_layers x n_expert.
    bool open(int device, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
              ExpertSource& src, int64_t n_layers, int64_t n_expert, int reserve_mib, int64_t max_slots,
              std::string& err, const PeerExperts* before = nullptr, int ordinal = 0);
    void close();
    bool valid() const { return device_ >= 0; }
    int device() const { return device_; }
    int ordinal() const { return ordinal_; }   ///< 0: the first (or only) server; 1: the second (--peer-device A,B)

    /// Whether the peer holds (layer, expert) right now.  STRATA_PEER_FUSED: inside a fused window the answer comes
    /// from the window's ownership table (own_win_, what the kernel on the peer plans with), not from res_, so the
    /// host's plan and the device's plan agree even when poll_pending / apply_pending move res_ mid-window.
    bool has(int64_t layer, int64_t expert) const {
        const size_t idx = (size_t) (layer * n_expert_ + expert);
        return window_open_.load(std::memory_order_relaxed) ? own_win_[idx] >= 0 : res_[idx] >= 0;
    }

    /// Two servers: whether this server holds (layer, expert) or is bringing it in (a swap of adapt() whose copy has
    /// not landed: in pending_, not yet in res_).  The other server's adapt() asks this before it takes a candidate,
    /// so a pair never ends up on both cards; has() alone would miss the swap in flight.  Called on the adapt thread
    /// between decode windows, the same thread that writes pending_ (adapt) - apply_pending runs at the decode's join
    /// or on the pool thread inside a window, never beside an adapt.
    bool claims(int64_t layer, int64_t expert) const {
        const size_t idx = (size_t) (layer * n_expert_ + expert);
        if (res_[idx] >= 0) return true;
        for (const auto& p : pending_)
            if ((size_t) p.first == idx) return true;
        return false;
    }

    /// The device address of (layer, expert)'s blob on the peer, or null when it is not resident.
    const uint8_t* slot_ptr(int64_t layer, int64_t expert) {
        const int32_t sl = res_[(size_t) (layer * n_expert_ + expert)];
        return sl >= 0 ? cache_.device_slot(sl) : nullptr;
    }
    /// Whether the primary (device 0) and the peer can read each other's memory directly (NVLink / PCIe P2P).
    bool p2p() const { return p2p_; }

    // ---- STRATA_EP_SPLIT_PREFILL=1 (opt-in, lab/upsync-v0.2.9, docs/EP_SPLIT_PREFILL.md s.2, s.6, s.13.2): the peer
    // card also runs a prompt STAGE (layers [K, n_layers) with its own dense weights and session), whose prompt path
    // takes this tier's cache and residency table as its RESIDENT tier (`Prefill::init(..., &cache(), res_table(), ...)`).
    // Tappa 4b: the TAIL of the tier (the slots >= `first`, the coldest pairs: the tier is filled in rank order) IS
    // LENT to that prompt path for the length of a prompt read, exactly as the primary's cache lends its own tail
    // ("the prompt path borrows N cache slots", generate.cpp `lend` / `refill`): the stage's prompt buffers are laid
    // out in the lent slots (`Prefill::relayout`), the pairs that lived there leave res_ (kNotResident: has() says no,
    // the stage streams them, the primary's PeerPrefill does not plan them here), and before the next decode window
    // the same slots are refilled from the source arena (`ExpertCache::fill_slot_queued` + `sync_queued`, on the
    // cache's own queue) and res_ restored.  Between `lend_guard(true)` and `lend_guard(false)` the tier is LENT:
    // adapt() does nothing, fused_begin() refuses, apply_pending() finds nothing (the caller quiesce()d before the
    // lend, so no copy of an earlier adapt is in flight when relayout writes the slots).  Never lent during a fused
    // window or an adapt: generate.cpp lends after quiesce(), refills before the first window.  The slots never
    // move (d_slot_ptr_ is slot -> address, filled once at open): a lent slot keeps its layer's blob size and gets
    // the SAME pair back at refill.  Off, none of this is called and nothing of the tier is lent.
    /// The tier's cache, read-only: the stage's prompt path reads a resident expert in its slot (a hit) and streams
    /// the rest from host memory itself.  No slot is lent or claimed through this handle (see cache_mut()).
    const ExpertCache& cache() const { return cache_; }
    /// The tier's cache, writable (tappa 4b): the handle generate.cpp's `lend` / `refill` use for the peer's part:
    /// `device_slot(first)` as the borrowed region a `Prefill::relayout` lays the stage's prompt buffers in, and
    /// `fill_slot_queued` / `sync_queued` (under the peer's device) to put the pairs back in the same slots at refill.
    /// Never used to open, close, grow or shrink the cache: the slots' count, sizes and addresses stay the open()'s.
    ExpertCache& cache_mut() { return cache_; }
    /// The tier's residency table, n_layers x n_expert -> the peer's slot or kNotResident: the same format as the
    /// primary's `host_res` (`Prefill::init(..., host_res, ...)`).  Valid while the tier is open (res_ is sized once
    /// at open and never reallocated after it; adapt / apply_pending change entries in place, between decode
    /// windows; lend / refill, through res_table_mut(), between a prompt's lend and its refill).  Outside a fused
    /// window it is the table has() answers from, so the stage and the pool thread agree.
    const int32_t* res_table() const { return res_.data(); }
    /// The same table, writable (tappa 4b): generate.cpp's `lend` puts kNotResident in the entries of the pairs whose
    /// slot is >= `first` (the lent tail) and its `refill` puts the slot back once the pair's bytes are in it again.
    /// Entries only: res_ is never resized or reallocated through this (the pointer stays the one handed out).  Written
    /// from the request thread while no fused window is open and no adapt is in flight (the lend guard says so).
    int32_t* res_table_mut() { return res_.data(); }
    /// Tappa 4b: raises / lowers the lend guard.  `on` with `slots` lent: from here until lend_guard(false) adapt()
    /// returns true without swapping, fused_begin() fails with "tier slots are lent to the prompt path" (never in
    /// the design: an error, not a different text) and apply_pending() has nothing pending (the caller quiesce()d
    /// first).  has() / launch() / finish() do not change: a lent pair is kNotResident in res_, which is enough.
    /// `false` clears the count.  Not called with the variable off.
    void lend_guard(bool on, int64_t slots = 0) {
        lent_slots_ = on ? slots : 0;
        lent_.store(on, std::memory_order_release);
    }
    bool lent() const { return lent_.load(std::memory_order_acquire); }   ///< the guard is up
    int64_t lent_slots() const { return lent_slots_; }                     ///< slots lent now (0 when not lent); for the log line
    /// Before a staged prompt read, and (tappa 4b) inside generate.cpp's `lend` BEFORE the relayout and the guard:
    /// lands the swaps of an earlier adapt() (apply_pending(true): res_ is final, pending_ empty) and waits for the
    /// tier's two streams, so no copy into a slot is in flight while the stage reads, or the relayout writes, the
    /// slots.  false with `err` when a fused window is still open (a kernel that may still write the slots: never in
    /// the design, said with an error and not with a different text) or on a CUDA error.  A closed tier: nothing to
    /// do, true.  Harmless with the guard already up (nothing pending, idle streams: the prompt's refill runs on the
    /// cache's queue, not on these streams).
    bool quiesce(std::string& err);

    /// Starts the peer's share of one layer: the entries i with kind[i] == `mine` (2 for the first server, 3 for the
    /// second: the pool's kind codes) (x: n_tok rows of n_embd floats).
    /// multi-GPU: with `out` (pinned, portable host rows) the peer writes its rows straight into
    /// out[entry] with a scatter kernel (no D2H copy + host memcpy in finish); STRATA_PEER_DIRECT=0: the old way.
    bool launch(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind,
                std::string& err, float* out = nullptr, int32_t mine = 2);
    /// Waits for the launched share and writes its rows into `out` (row i = entry i, n_embd floats).
    bool finish(float* out, std::string& err);

    /// Adaptive tier: swap up to `max_swaps` experts routed at least twice that neither the primary (`res0`, slot
    /// or -1) nor the peer holds into the peer's least-used slots.  Copies run on the peer's refill stream; a
    /// swapped-in expert becomes resident at `apply_pending`.  Call only between verify windows.  Two servers:
    /// `other` is the other server; a pair it holds or is bringing in (claims) is no candidate here.
    bool adapt(const float* usage, const int32_t* res0, int max_swaps, std::string& err, const PeerExperts* other = nullptr);
    void apply_pending(bool wait);
    /// STRATA_PEER_REFILL_FLUSH=1 (opt-in, lab/upsync-v0.2.9, PIPELINE3_PIANO tappa 0b): adapt() queries the refill
    /// stream right after its last copy, so WDDM hands the copies to the card now instead of at the next driver call
    /// (#2188), and the pool thread calls poll_pending() after every layer: a non-blocking apply_pending, so the swaps
    /// land in the tables inside the window and the decode's join of the adapt thread finds nothing left to wait for.
    /// Off, nothing changes.
    static bool refill_flush();
    void poll_pending() { if (refill_flush() && !pending_.empty()) apply_pending(false); }
    /// STRATA_PEER_ONE_COPY=1 (opt-in, lab/upsync-v0.2.9): the layer's plan and its activation rows live in one pinned
    /// block (the plan first, the rows after it) with one device block laid out the same, so launch() queues ONE host
    /// to device copy a layer instead of two (plan + rows).  The rows are memcpy'd into the block first, also on the
    /// direct path (a copy of n_tok x n_embd floats on the pool thread).  The kernels and their inputs do not change.
    static bool one_copy();
    double ms_adapt = 0;                              ///< host time spent in adapt() (the swaps' copies queued)

    int64_t resident() const { return resident_; }
    /// the last launch writes its rows into the host rows itself (the caller must not touch them)
    bool launched_direct() const { return launched_direct_; }
    double gib() const { return cache_.gib(); }
    int64_t entries() const { return entries_; }      ///< routed entries the peer computed
    int64_t experts() const { return experts_; }      ///< distinct (layer, expert) the peer computed
    int64_t swaps() const { return swaps_; }
    double ms_wait = 0;                               ///< host time spent in finish() waiting for the peer
    double ms_launch = 0;                             ///< host time spent in launch() (plan, copies, kernel launches)
    int64_t launches = 0;                             ///< layers with a share on the peer (a launch and a finish each)
    /// STRATA_PEER_NOSWITCH=1 (opt-in, lab/upsync-v0.2.9): finish() and apply_pending() query the peer's stream and
    /// event without switching the calling thread's device (the stream carries its own context, as the verifier's
    /// STRATA_P2_FLUSH_NOSWITCH does).  No GPU work changes.  If the driver refuses a query the engine says so once
    /// and goes back to the switch.
    static bool noswitch();
    /// STRATA_PEER_STAMP=1 (opt-in, lab/upsync-v0.2.9, PIPELINE3_PIANO tappa 0): launch() puts two one-thread
    /// %globaltimer stamps on the peer's stream, before its first copy and after its last kernel, into a mapped page
    /// (two words per layer); finish() reads the layer's GPU span once the stream is done.  The expert work is the
    /// same: the stamps say how long the peer's stream is busy for a layer, against the host's launch + finish.
    static bool stamp();
    /// the GPU spans (us) of the layers finished so far, in order; the decode timing line reads a window range of them
    /// (STRATA_PEER_FUSED: one span a step, from the kernel's own stamps: doorbell seen -> flag raised)
    const std::vector<float>& spans_us() const { return spans_us_; }
    double span_us_total = 0;                         ///< their sum (us)

    // ---- STRATA_PEER_FUSED=1 (opt-in, lab/upsync-v0.2.9, docs/PEER_FUSED.md s.6): the peer's expert server as ONE
    // persistent kernel per verify window.  The verifier drives it through fused_hook() (Verifier::set_peer_fused):
    // begin before the graph launch, end after the window's sync, flush in its wait loop, abort from
    // release_gpu_waits.  Inside a window launch() only counts (groups, rows) for the end-of-window check and
    // finish() returns at once (or spins on the kernel's flag with STRATA_PEER_FUSED_HOSTWAIT=1); has() answers from
    // the window's ownership table.  Off, or outside a fused window (the prompt path, a run without the hook):
    // has / launch / finish are exactly the ones above.
    /// STRATA_PEER_FUSED=1, read once.
    static bool fused_env();
    /// The fused server is on: fused_env(), a native pack whose every layer has a gate/up and a down type the kernel
    /// has a body for (one geometry, the types free to differ a layer), and the buffers allocated at open().
    bool fused_on() const { return fused_on_; }
    /// The verifier's handle (stable: a member).  Null when the fused server is off.
    const peer_fused::Hook* fused_hook() const { return fused_on_ ? &hook_ : nullptr; }
    /// The primary's residency table (drive.d.host_res; may be null before the token graph's table exists): with the
    /// peer's own res_ it gives the window's ownership (own = res_ >= 0 && host_res < 0).  Set before the first window.
    void fused_set_host_res(const int32_t* host_res) { host_res_fused_ = host_res; }
    /// Two servers: the FIRST server's residency table (its res_table()), set on the SECOND before its first window.
    /// Its window table then leaves out every pair the first holds (own = res_ >= 0 && host_res < 0 && other < 0),
    /// the same precedence the pool's dispatch applies (primary, then the first server, then this one), so the two
    /// kernels never plan the same entry.  Null (the first server, or one server): no third test.
    void fused_set_other_res(const int32_t* other_res) { other_res_fused_ = other_res; }
    /// The hook's four entry points (see the Hook comment in peer_fused.hpp).
    bool fused_begin(const peer_fused::WindowHost& w, std::string& err);
    bool fused_end(std::string& err);
    void fused_flush();
    void fused_abort();
    int64_t fused_windows() const { return fused_windows_; }   ///< windows served by the persistent kernel
    int64_t fused_steps() const { return fused_steps_; }       ///< steps (layer x group) those windows carried
    static bool fused_hostwait();                              ///< STRATA_PEER_FUSED_HOSTWAIT=1, read once

private:
    static bool hook_begin(void* ctx, const peer_fused::WindowHost& w, std::string& err);
    static bool hook_end(void* ctx, std::string& err);
    static void hook_flush(void* ctx);
    static void hook_abort(void* ctx);
    bool fused_open(std::string& err);        ///< open()'s fused part: shape, page, buffers, hook (false = a CUDA error)
    void fused_close();                       ///< frees what fused_open allocated (the kernel already left)
    cudaError_t fused_query();                ///< cudaStreamQuery(stream_) without the device switch, under it if refused
    cudaError_t fused_drain(double limit_s);  ///< polls fused_query until not cudaErrorNotReady or the limit
    bool fused_map(const void* host, void** dev, std::string& err);   ///< the peer's device pointer of a mapped host page (cached)
    int device_ = -1;
    int ordinal_ = 0;                                 ///< open()'s ordinal: 1 = the second server (STRATA_PEER2_* knobs, device in the log lines)
    bool p2p_ = false;
    int64_t n_layers_ = 0, n_expert_ = 0;
    ExpertCache cache_;
    ExpertSource* src_ = nullptr;
    std::vector<int32_t> res_;                        ///< [n_layers * n_expert] -> peer slot or -1 (sized at open, never reallocated after: res_table() hands it out)
    std::vector<std::pair<int32_t, int32_t>> pending_;   ///< (residency index, slot) once the copies land
    int64_t resident_ = 0, entries_ = 0, experts_ = 0, swaps_ = 0;
    // ---- STRATA_EP_SPLIT_PREFILL tappa 4b (docs/EP_SPLIT_PREFILL.md s.13.2): the lend guard.  Up between
    // generate.cpp's lend and refill of the tier's tail; atomic because adapt() runs on the decode's adapt thread
    // and the guard is set from the request thread (the design never overlaps them: the atomic makes the read
    // well-defined, not the overlap allowed).  lent_slots_ is plain: written and read on the request thread.
    std::atomic<bool> lent_{false};
    int64_t lent_slots_ = 0;
    cudaStream_t stream_ = nullptr, refill_ = nullptr;
    cudaEvent_t refill_ev_ = nullptr;
    float* h_x_ = nullptr;   float* d_x_ = nullptr;   // n_tok x n_embd
    float* h_out_ = nullptr; float* d_out_ = nullptr; // compact rows
    void* h_meta_ = nullptr; void* d_meta_ = nullptr; // groups: ptr | start | dst | tok | count
    void* h_stage_ = nullptr; void* d_stage_ = nullptr;   // STRATA_PEER_ONE_COPY: plan + rows in one block (h_meta_/h_x_ point into it)
    uint8_t* d_q8_ = nullptr;
    void* d_scratch_ = nullptr;
    const float* d_sx_ = nullptr;                     // APR: device sign vectors (x: H, h: FF); nullptr = identity
    const float* d_sh_ = nullptr;
    std::vector<int32_t> row_of_;                     ///< compact row -> original entry
    int64_t launched_rows_ = 0;
    bool launched_direct_ = false;
    unsigned long long* h_stamp_ = nullptr;           ///< STRATA_PEER_STAMP: 2 x n_layers ns stamps, mapped (host view)
    unsigned long long* m_stamp_ = nullptr;           ///< the same page as the peer sees it
    int64_t stamp_layer_ = -1;                        ///< the layer whose stamps the next finish() reads
    std::vector<float> spans_us_;

    // ---- STRATA_PEER_FUSED (docs/PEER_FUSED.md s.2b, s.3, s.4)
    bool fused_on_ = false;
    /// A fused window is open: has/launch/finish take the fused path and adapt() does not touch the slots.  It stays
    /// true as long as the window's kernel may still be running (fused_abort leaves it up when its 5 s drain gives up:
    /// a kernel that has not left still reads the slots); the next fused_begin / fused_end lowers it once the stream
    /// is idle.  Atomic: written by the pool thread and by fused_abort, which release_gpu_waits may call from the
    /// stall watchdog's thread; read by the pool thread (has) and the adapt thread.
    std::atomic<bool> window_open_{false};
    peer_fused::Shape shape_;                         ///< the kernel instance and grid peer_fused_shape resolved at open
    strata::kernels::NativeExpertLayout fused_L_;     ///< layer 0's layout: the pack's geometry (n_embd, n_ff the same on every layer)
    std::vector<strata::kernels::NativeExpertLayout> fused_lays_;   ///< one layout a layer (the types may differ a layer), what d_lay_ holds
    strata::kernels::NativeExpertLayout* d_lay_ = nullptr;          ///< VRAM copy of fused_lays_: the kernel reads lay[l] a step
    peer_fused::Hook hook_;                           ///< what the verifier holds (fused_hook())
    const int32_t* host_res_fused_ = nullptr;         ///< the primary's residency table (fused_set_host_res)
    const int32_t* other_res_fused_ = nullptr;        ///< two servers: the first server's table, on the second (fused_set_other_res)
    // the peer's mapped page (kPageBytes, Portable | Mapped): host view and the peer's device view of each word
    uint8_t* h_page_ = nullptr;
    uint32_t* h_flag_out_ = nullptr;   uint32_t* m_flag_out_ = nullptr;
    uint32_t* h_abort_ = nullptr;      uint32_t* m_abort_ = nullptr;
    uint32_t* h_err_ = nullptr;        uint32_t* m_err_ = nullptr;
    uint32_t* h_exit_ = nullptr;       uint32_t* m_exit_ = nullptr;
    uint32_t* h_cnt_ = nullptr;        uint32_t* m_cnt_ = nullptr;
    unsigned long long* h_stamp_fused_ = nullptr;   unsigned long long* m_stamp_fused_ = nullptr;
    const uint32_t* flag_out_dev0_ = nullptr;       ///< flag_out as the primary device sees it (the graph's wait node)
    // VRAM
    int32_t* d_own_ = nullptr;                        ///< n_layers x n_expert: the peer's slot or -1 (what the kernel plans with)
    unsigned long long* d_slot_ptr_ = nullptr;        ///< cache_.slots(): slot -> device address of its blob
    peer_fused::Plan* d_plan_ = nullptr;              ///< [2]: the kernel's double-buffered step plan
    peer_fused::Sync* d_sync_ = nullptr;              ///< the kernel's barrier / ready / done words
    // host
    int32_t* h_own_ = nullptr;                        ///< pinned n_layers x n_expert: the rows the per-layer copies go up from
    std::vector<int32_t> own_win_;                    ///< the window's ownership table (the one truth: has() reads it)
    std::vector<int32_t> own_shadow_;                 ///< what d_own_ holds (only the changed layers are copied)
    std::vector<uint32_t> expect_;                    ///< [kMaxSteps] the host's groups << 16 | rows a step
    int step_ = 0;                                    ///< steps launch() counted in the open window
    int win_steps_ = 0;                               ///< steps the open window has
    int win_G_ = 1, win_lb_ = 0;                      ///< for the mismatch message (step -> layer)
    uint32_t epoch_ = 0;                              ///< the window's number (the kernel writes it to `exit`)
    unsigned long long wait_ns_ = peer_fused::kWaitNsDefault;   ///< STRATA_PEER_FUSED_WAIT_MS
    int64_t fused_windows_ = 0, fused_steps_ = 0;
    struct MappedPtr { const void* host = nullptr; void* dev = nullptr; };
    MappedPtr map_cache_[4];                          ///< seq / x / ids / ymiss: the peer's device pointers, by host address
};

}  // namespace strata::core
