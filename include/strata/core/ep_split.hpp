// include/strata/core/ep_split.hpp - STRATA_EP_SPLIT_PREFILL=1 (opt-in, lab/upsync-v0.2.9, docs/EP_SPLIT_PREFILL.md):
// the prompt read in two LAYER stages, the reply written EXPERT-PARALLEL.
//
// Today `--peer-device` reads the prompt on the primary alone (every dense layer on the 3080 Ti, the peer computing
// only the experts it holds for each chunk: 1414-1476 t/s at 128K against the layer split's 2243, -35%), and writes
// the reply with the primary over all 48 layers and the peer as expert server (STRATA_PEER_FUSED: 85-87 t/s, the best
// writing path).  With STRATA_EP_SPLIT_PREFILL=1 the engine keeps that writing path and reads the prompt as the layer
// split does: the primary runs layers [0, K) with its session, the peer card runs layers [K, n_layers) with ITS OWN
// dense weights, its own session (K/V, GDN, indexer) and the peer tier's expert cache as its resident tier, the two
// pipelined chunk by chunk as Prefill's stages already are.  When the prompt is read, the running state of layers
// [K, n_layers) crosses from the peer card to the primary's full session (no P2P on this rig: pinned staging with
// D2H + H2D copies, or a host-to-host memcpy for a streamed K/V whose truth is in pinned RAM already); the decode then
// runs exactly as today.  Before the next turn's prompt the cells the decode wrote (and whatever the peer stage
// does not hold any more) cross back.  Off (the default), nothing changes: no object, no allocation, no line.
//
// This header is the CONTRACT between the pieces (docs/EP_SPLIT_PREFILL.md s.9): generate.cpp (the wiring: the stage,
// the admission, the handovers at the request's seams, the checkpoint merge), peer_experts.* (the peer tier lends its
// cache and residency table to the stage's prompt path), ep_split.cpp (the environment, the admission, the state mover,
// the sync bookkeeping, the checkpoint merge) and tools/ep_split_test.cpp (the mover's bit test).  Plain host code:
// no device code, no kernels.  Written, not compiled (docs/EP_SPLIT_PREFILL.md s.11).
#pragma once

#include "strata/core/conversation_cache.hpp"   // ConversationCheckpoint
#include "strata/core/layer.hpp"                // ModelGeometry, QsaState
#include "strata/core/session.hpp"              // SessionState

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core::ep_split {

// ---------------------------------------------------------------- knobs (read once by ep_split.cpp; the names are the contract)

inline constexpr const char* kEnvOn = "STRATA_EP_SPLIT_PREFILL";          ///< 1: the EP prefill (master switch)
inline constexpr const char* kEnvLayer = "STRATA_EP_SPLIT_LAYER";         ///< K: the first layer of the peer card's stage (default 20)
inline constexpr const char* kEnvPieceMib = "STRATA_EP_SPLIT_PIECE_MIB";  ///< staging piece of the D2H + H2D copies (default 8)
inline constexpr const char* kEnvThreads = "STRATA_EP_SPLIT_THREADS";     ///< host threads for the streamed K/V's memcpy (default 4)
inline constexpr const char* kEnvTrace = "STRATA_EP_SPLIT_TRACE";         ///< 1: one stderr line per handover (bytes, ms per part)
inline constexpr const char* kEnvReserveMib = "STRATA_EP_SPLIT_RESERVE_MIB";   ///< bytes added to --peer-reserve-mib for the stage's prompt buffers (default: computed, s.4b)
// Tappa 4c (docs/EP_SPLIT_PREFILL.md s.14): STRATA_EP_SPLIT_PREFILL=2 is "1 plus the three pieces below"; with 1 none of
// them exists and the engine is tappa 4b's, bit for bit.  Each piece has its own switch, read only at level 2.
inline constexpr const char* kEnvMinTokens = "STRATA_EP_SPLIT_MIN_TOKENS";   ///< the staged read only when a request reads at least this many fresh tokens (default kMinTokensDefault; 0: always, as tappa 4b)
inline constexpr const char* kEnvDenseLoan = "STRATA_EP_SPLIT_DENSE_LOAN";   ///< 1 (default at level 2): the stage's dense weights live in the peer tier's tail only during a staged read
inline constexpr const char* kEnvOverlap = "STRATA_EP_SPLIT_OVERLAP";        ///< 1 (default at level 2): a chunk's cells cross to the primary while the stage reads the next one
inline constexpr const char* kEnvDenseSync = "STRATA_EP_SPLIT_DENSE_SYNC";   ///< 1: the dense weights' H2D is waited for in lend (default 0: queued on the stage's compute stream, which orders it before the stage's kernels)
inline constexpr const char* kEnvSessionLoan = "STRATA_EP_SPLIT_SESSION_LOAN";   ///< 1 (default, with the dense loan): the stage's session too lives in the tier's tail during a staged read only (s.14.2ter)

inline constexpr int64_t kLayerDefault = 20;
/// Break-even of the staged read against today's two-server read (s.14.1, from the 11/11 figures: two servers 969 /
/// 1489 / 1567 t/s at 4K / 32K / 128K, the layer split 897 / 1816 / 2128, a handover plus the dense weights' upload
/// 0.2-0.4 s): the staged read is slower at 4K, even at ~12K, 5 % ahead at 16K and 36 % ahead at 128K.  The default
/// sits at the first chunk count (two of 8192) where the two-stage pipeline overlaps at all.
inline constexpr int64_t kMinTokensDefault = 16384;
/// Alignment of the regions carved out of the tier's tail (the dense weights, the super-chunk): cudaMalloc's.
inline constexpr uint64_t kTailAlign = 256;
inline constexpr size_t kPieceBytesDefault = (size_t) 8 << 20;
inline constexpr int kThreadsDefault = 4;
/// Headroom added to the computed prompt-buffer bytes when the peer's reserve is raised.  THE SAME 512 MiB the chunk
/// scan's `own_fits` demands beside the buffers on every card without a loan (generate.cpp, kHeadroom): the reserve
/// is computed at the largest chunk the scan may pick (o.prefill_chunk before the scan: prefill_auto_max with
/// --prefill auto), so with this headroom the peer card passes own_fits for every smaller chunk by construction,
/// whatever --peer-reserve-mib is; with less (256 MiB, the first draft) a small --peer-reserve-mib made the peer fail
/// own_fits and step the SHARED chunk down, CUDA0's too - a slower read with different bits from today's split.
inline constexpr uint64_t kReserveHeadroomBytes = (uint64_t) 512 << 20;
/// A streamed K/V handover evicts only the rewritten blocks from the target's residency map, one 4-byte write per
/// resident block among them; above this many resident blocks in the range (or a range that is most of the map) the
/// whole map is reset instead (kv_stream_reset, one kernel), which is what a first turn or a restore amounts to anyway.
inline constexpr int64_t kResetAboveResident = 1024;

bool env_on();                 ///< STRATA_EP_SPLIT_PREFILL=1
int64_t env_layer();           ///< STRATA_EP_SPLIT_LAYER or kLayerDefault
size_t env_piece_bytes();      ///< STRATA_EP_SPLIT_PIECE_MIB << 20 or kPieceBytesDefault (floor 1 MiB)
int env_threads();             ///< STRATA_EP_SPLIT_THREADS or kThreadsDefault (1..16)
bool env_trace();              ///< STRATA_EP_SPLIT_TRACE=1
int64_t env_reserve_mib();     ///< STRATA_EP_SPLIT_RESERVE_MIB, -1 when unset (then generate.cpp computes it)
int env_level();               ///< STRATA_EP_SPLIT_PREFILL as a number (0 off, 1 tappa 4/4b, 2 tappa 4c); env_on() is level != 0
int64_t env_min_tokens();      ///< STRATA_EP_SPLIT_MIN_TOKENS or kMinTokensDefault (floor 0)
bool env_dense_loan();         ///< STRATA_EP_SPLIT_DENSE_LOAN, default 1
bool env_overlap();            ///< STRATA_EP_SPLIT_OVERLAP, default 1
bool env_dense_sync();         ///< STRATA_EP_SPLIT_DENSE_SYNC=1
bool env_session_loan();       ///< STRATA_EP_SPLIT_SESSION_LOAN, default 1

// ---------------------------------------------------------------- tappa 4c: pure rules (ep_split_test checks them)

/// The request reads `fresh` tokens in batched parts (n - read_from): staged (two layer stages) when at least
/// `min_tokens`; below it the primary reads every layer with the two servers' prompt shares, as without the variable.
inline bool staged_read(int64_t fresh, int64_t min_tokens) { return min_tokens <= 0 || fresh >= min_tokens; }

/// A region of `bytes` bytes at the END of an expert cache's arena (the slots are lent for the time the region is
/// in use: the dense weights of the stage during a staged read).  `slots` slots, laid out by `slot_offsets` (slot s
/// is [slot_offsets[s], slot_offsets[s + 1]); null: `slot_bytes` each), the arena `cache_bytes` long.
///   offset: where the region starts, kTailAlign-aligned down, so [offset, cache_bytes) holds `bytes`;
///   first:  the first slot the region overlaps - every slot in [first, slots) must be lent while it is in use;
///           -1 when fewer than `keep` slots would stay (the loan's floor), or the region does not fit at all.
struct TailRegion {
    int64_t first = -1;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};
TailRegion tail_region(int64_t slots, const uint64_t* slot_offsets, uint64_t slot_bytes, uint64_t cache_bytes,
                       uint64_t bytes, int64_t keep);

// ---------------------------------------------------------------- tappa 4c: the stage's dense weights between prompts

/// The stage's dense weights (its WeightTable arena and its NativeDense matrices, laid out back to back in the
/// tier's tail region by generate.cpp) kept in RAM between prompts: `capture` copies the region D2H once at start,
/// `restore` puts the SAME bytes back at the SAME device address before a staged read (the WeightRefs keep their
/// pointers: nothing is rebased), queued on the stage's compute stream so the stage's first kernel finds them.
/// Pinned (Portable) when the allocation succeeds, pageable otherwise (`pinned()` false, said in one line by the
/// caller: the copy then runs at the pageable rate).  Written, not compiled.
class DenseMirror {
public:
    DenseMirror() = default;
    ~DenseMirror();
    DenseMirror(const DenseMirror&) = delete;
    DenseMirror& operator=(const DenseMirror&) = delete;

    /// D2H of [dev_base, dev_base + bytes) on `device`, synchronous.  Allocates the host copy.
    bool capture(const void* dev_base, uint64_t bytes, int device, std::string& err);
    /// H2D of the captured bytes to `dev_base` (the same address, checked) on `stream` of `device()`; with `wait`
    /// the call returns when the copy is complete and `last_ms()` has its time; without, the copy is queued on the
    /// stream (ordered before everything the caller issues on it afterwards) and `last_ms()` is 0.
    bool restore(void* dev_base, cudaStream_t stream, bool wait, std::string& err);
    void close();
    bool valid() const { return host_ != nullptr; }
    bool pinned() const { return pinned_; }
    uint64_t bytes() const { return bytes_; }
    int device() const { return device_; }
    const void* dev_base() const { return dev_base_; }
    double last_ms() const { return last_ms_; }
    int64_t restores() const { return restores_; }

private:
    void* host_ = nullptr;
    bool pinned_ = false;
    uint64_t bytes_ = 0;
    int device_ = -1;
    const void* dev_base_ = nullptr;
    double last_ms_ = 0;
    int64_t restores_ = 0;
};

// ---------------------------------------------------------------- admission

/// What generate.cpp knows after the options are parsed and before anything is loaded.  Every field is a plain copy
/// of an option or a probe; `admit` owns the rule, generate.cpp owns the values.
struct Config {
    bool on = false;                  ///< env_on()
    bool serve = false;               ///< --serve
    int peer_device = -1;             ///< --peer-device N (-1: none)
    bool layer_split = false;         ///< --layer-split given (refused with the peer already; the EP prefill is not asked that way)
    bool expert_cache_remote = false; ///< --expert-cache-device1..3
    int batch = 0;                    ///< --batch
    bool vision = false;              ///< --vision (allowed: the stage gets its own image-position table on its card, as a split's stages do; kept for the record)
    bool prefill_head_alias = false;  ///< --prefill-head-alias (CUDA0's hybrid prompt layout: not with a second stage)
    bool ckpt_async = false;          ///< STRATA_CKPT_ASYNC=1 (one global staging buffer for the checkpoint copies)
    int ckpt_vram_slots = 0;          ///< STRATA_CKPT_VRAM_SLOTS (VRAM slots of the saved GDN rows, created on the current card)
    int64_t prompt_cache_every = 0;   ///< --prompt-cache-every (mid-prompt checkpoints: saved in two parts, from two cards, on two threads)
    bool kv_grow = false;             ///< --kv-grow / STRATA_KV_GROW (refused with the peer already)
    bool elastic = false;             ///< --elastic / --vram-elastic (refused with the peer already)
    int pipeline_windows = 0;         ///< --pipeline-windows (the decode is not split: 0 only)
    bool layer_major = false;         ///< STRATA_PREFILL_LAYER_MAJOR reads as on ("0" is off, not a request; declines with a peer already: refused for clarity)
    bool batch_concurrent = false;    ///< STRATA_BATCH_CONCURRENT_READ
    bool native_pack = false;         ///< the native expert pack (the only one the peer tier and the fused server take)
    bool expert_profile = false;      ///< --expert-profile given (the peer needs it)
    bool session_files = false;       ///< a request may ask session files (refused with the peer already)
    int64_t n_layers = 0;             ///< g.n_layers
    int64_t layer = kLayerDefault;    ///< env_layer()
    int n_devices = 0;                ///< cudaGetDeviceCount
    int primary_device = 0;           ///< STRATA_PRIMARY_DEVICE (0 only)
};

/// true: the EP prefill runs.  false with `why` set: the variable is on but a condition fails - generate.cpp prints
/// ONE line `strata: STRATA_EP_SPLIT_PREFILL off: <why>` and runs exactly as without the variable.  false with `why`
/// empty: the variable is unset (nothing is printed).  Pure: no CUDA call.
bool admit(const Config& c, std::string& why);

// ---------------------------------------------------------------- the state that crosses

/// Floats of one GDN layer's state row: the recurrence (ssm_state_size x ssm_v_heads x ssm_state_size) and the conv
/// history (ssm_conv_channels x (ssm_d_conv - 1)).  THE SAME FORMULA AS session.cpp's gdn_state_floats (file-local
/// there): `gdn_state + (ordinal - gdn_ord0) * gdn_row_floats(g)` is layer `ordinal`'s row.  ep_split_test checks it
/// against session_bytes / session_init.
inline uint64_t gdn_row_floats(const ModelGeometry& g) {
    return (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size +
           (uint64_t) g.ssm_conv_channels * (g.ssm_d_conv - 1);
}

/// What one handover moved, for the trace line and the request-end line.
struct Stats {
    int64_t cells0 = 0, cells1 = 0;   ///< the K/V cells moved: [cells0, cells1)
    int qsa_layers = 0;               ///< QSA states in the range
    int gdn_layers = 0;               ///< GDN rows in the range
    uint64_t kv_bytes = 0;            ///< K/V bytes (the four pools; host memcpy when streamed, D2H + H2D when resident)
    uint64_t pooled_bytes = 0;        ///< the indexer's pooled rows of those cells
    uint64_t state_bytes = 0;         ///< GDN rows + indexer tail / dead / block position
    bool kv_host_memcpy = false;      ///< the K/V went host to host (kv_mode 1)
    double ms_kv = 0, ms_pooled = 0, ms_state = 0, ms = 0;
};

/// Moves the running state of layers [lb, le) between the peer stage's session (carved [lb, le) on the peer card)
/// and the primary's full session (carved [0, n_layers) on the primary).  Owns two pinned staging pieces (double
/// buffered, env_piece_bytes() each), one non-blocking stream and one event per card.  Every method takes the
/// current device as it finds it and restores it (OnDevice inside); every copy is complete on return; the caller
/// has synchronized both sessions' streams before (nothing of either session in flight).
///
/// What crosses, per direction, for every layer l in [lb, le):
///   * a QSA layer: the K/V cells [c0, c1) of the four pools (k, v, k_scale, v_scale: those the format has), whole
///     pages ([c0 rounded down to page_size, c1 rounded up)); the indexer's pooled rows [c0 / idx_block, c1 / idx_block]
///     (idx_pooled); and, with `running_state`, idx_tail, idx_dead, idx_block_pos (conversation_state_sizes' tail /
///     dead / block_pos bytes).  kv_mode 0 (resident): pool bytes D2H into a piece, H2D from it, pipelined on the two
///     streams.  kv_mode 1 (streamed, --kv-resident): the truth is the pinned host copy (QsaState::host), so the pool
///     bytes go host to host (memcpy, env_threads() threads, 1 MiB slices with progress_beat), and in the target's
///     residency map ONLY the rewritten blocks [c0 / page, ceil(c1 / page)) are evicted (page_table -1, their slots
///     freed: slot_block -1), because its VRAM slots may hold the old bytes of those blocks; every other resident
///     block stays, so a turn's handover of a few hundred cells does not throw away the decode's working set (a full
///     kv_stream_reset did, and the first windows of every turn re-read their selections from the host).  The full
///     reset is used only when the range is most of the map or has more than kResetAboveResident resident blocks.
///     kv_mode 2 is a drafter's ring and never in the range (refused).
///   * a GDN layer, with `running_state`: its row of gdn_row_floats(g) floats, D2H + H2D.
///   * never: the PLE history (layer 1, never in [lb, le)), the per-token counts (step, pos), the residual.
/// The byte ranges of the pools come from conversation_kv_part_sizes (sizes at `upto` tokens: cells come first in the
/// identity layout, whole pages, so part bytes at c0' and at c1' bound the slice) and the pointers from QsaState's
/// own fields (k_q / v_q / k_scale / v_scale, k_q4 / v_q4, k_pool / v_pool, the hybrid's k_q + k_scale + v_q4; the
/// same names under QsaState::host when streamed).  Both states must have the same format, kv_mode, page_size and
/// max_cells (the two sessions are carved by the same session_init rules; checked anyway, an error otherwise).
class StateMover {
public:
    StateMover() = default;
    ~StateMover();
    StateMover(const StateMover&) = delete;
    StateMover& operator=(const StateMover&) = delete;

    /// `primary_device`, `peer_device`: the two cards; `max_cells`: the sessions' capacity (bounds every range).
    bool init(const ModelGeometry& g, int64_t lb, int64_t le, int primary_device, int peer_device, int64_t max_cells,
              std::string& err);
    void close();
    bool valid() const { return primary_dev_ >= 0; }
    int64_t lb() const { return lb_; }
    int64_t le() const { return le_; }

    /// Peer stage -> primary: the cells [c0, c1) and, with `running_state`, the GDN rows and the indexer's small
    /// parts of every layer in [lb, le).  After it the primary's session holds, for those layers, what the stage
    /// computed; the stage's session is untouched.
    bool to_primary(const SessionState& peer_ss, SessionState& primary_ss, int64_t c0, int64_t c1, bool running_state,
                    Stats& st, std::string& err);
    /// Primary -> peer stage: the same the other way (the cells the decode wrote, or the whole prefix after a restore).
    bool to_peer(const SessionState& primary_ss, SessionState& peer_ss, int64_t c0, int64_t c1, bool running_state,
                 Stats& st, std::string& err);
    /// T rows of `floats_per_row` floats from a device buffer on the peer card to one on the primary (the last stage's
    /// final residual rows for the drafter's prompt pass, which lives on the primary with the EP prefill: hc * n_embd
    /// floats a row).  Through the pinned pieces, pipelined; complete on return.
    bool rows_to_primary(const float* src_peer, float* dst_primary, int64_t T, int64_t floats_per_row, std::string& err);
    const Stats& last() const { return last_; }
    /// Sums since open, for the request-end line: handovers, bytes, ms.
    int64_t handovers() const { return handovers_; }
    uint64_t bytes_total() const { return bytes_total_; }
    double ms_total() const { return ms_total_; }

private:
    enum class Dir { kToPrimary, kToPeer };
    bool move(const SessionState& from, SessionState& to, Dir dir, int64_t c0, int64_t c1, bool running_state,
              Stats& st, std::string& err);
    /// D2H on `from_dev`'s stream into piece p, H2D on `to_dev`'s stream from it, piece after piece (an event per piece).
    bool copy_d2d(const void* src, int from_dev, void* dst, int to_dev, size_t bytes, std::string& err);
    /// Host to host in 1 MiB slices over env_threads() threads, a progress_beat per slice.
    bool copy_h2h(const void* src, void* dst, size_t bytes, std::string& err);
    const ModelGeometry* g_ = nullptr;
    int64_t lb_ = 0, le_ = 0, max_cells_ = 0;
    int primary_dev_ = -1, peer_dev_ = -1;
    size_t piece_ = 0;
    void* piece_h_[2] = {nullptr, nullptr};          ///< pinned (Portable), the two staging pieces
    cudaStream_t s_primary_ = nullptr, s_peer_ = nullptr;
    cudaEvent_t ev_[2] = {nullptr, nullptr};
    Stats last_;
    int64_t handovers_ = 0;
    uint64_t bytes_total_ = 0;
    double ms_total_ = 0;
};

// ---------------------------------------------------------------- what the peer stage is known to hold

/// Which cells of the primary's session the peer stage's session is known to hold, for layers [lb, le): the first
/// `cells` tokens, whose ids are kept so a request on another conversation (or on this one after a restore) is
/// recognized.  Kept by generate.cpp beside the main session; written after every handover; invalidated wherever the
/// main session's cells are rewritten other than by appending: a conversation-cache restore (whole, prefix, borrowed,
/// from disk), a batch slot copy, a checkpoint mount or rewind, a session zero.  The bytes of a cell depend on who
/// computed it (the stage for prompt cells, the primary's decode for generated ones), so "same ids" alone is not
/// "same bytes": that is why every rewrite of the main session invalidates, and only appends keep the prefix.
struct Sync {
    int64_t cells = 0;
    std::vector<int32_t> ids;   ///< the first `cells` tokens

    void invalidate() { cells = 0; ids.clear(); }
    /// The cells the stage holds for a request whose tokens are `ids_now` (n of them): `cells` when the kept prefix
    /// matches ids_now[0, cells) and cells <= n, else 0 (and the kept prefix is dropped).
    int64_t valid_for(const int32_t* ids_now, int64_t n);
    /// After a handover: the stage and the primary agree on [0, cells_now) of `ids_now`.
    void note(const int32_t* ids_now, int64_t cells_now);
};

// ---------------------------------------------------------------- mid-prompt checkpoints

/// A full-session checkpoint from two parts taken at the same token count: `full` was saved from the primary's full
/// session (conversation_checkpoint_save on the [0, n_layers) carve: its GDN rows and indexer parts of layers >= lb
/// are stale mid-prompt, the stage is ahead of the primary's copy of those layers), `tail` from the peer stage's
/// session (carve [lb, le)).  Overwrites in `full` the GDN rows [tail_ss.gdn_ord0, + gdn_alloc) and the indexer tail /
/// dead / block_pos of the QSA ordinals [tail_ss.qsa_ord0, + qsa_alloc) with the tail's bytes (row and part sizes from
/// conversation_session_sizes of each session).  `full.ids == tail.ids` required.  No CUDA call of its own: it waits
/// for a STRATA_CKPT_ASYNC copy in flight (conversation_checkpoint_join) before reading either part's bytes, and
/// clears `full.dev_tag` (the VRAM copy of its GDN rows no longer matches the merged bytes).
bool checkpoint_merge_tail(ConversationCheckpoint& full, const ConversationCheckpoint& tail, const SessionState& full_ss,
                           const SessionState& tail_ss, const ModelGeometry& g, std::string& err);

// ---------------------------------------------------------------- the stage's prompt buffers and the peer's reserve

/// MiB to add to --peer-reserve-mib before PeerExperts::open, so the stage's own prompt buffers (no loan: the peer
/// tier's slots are never lent) fit beside the peer tier: STRATA_EP_SPLIT_RESERVE_MIB when set, else
/// `prompt_bytes` (prefill::Prefill::bytes_needed_owned(g, stage_ss, chunk) computed by generate.cpp) +
/// kReserveHeadroomBytes, rounded up to MiB.
int64_t reserve_add_mib(uint64_t prompt_bytes);

/// The one-line forms of the start and request-end lines (so every piece prints the same words):
///   on:   "strata ep-split: on: layers [K, N) on CUDA<d>; prompt buffers <m> MiB from the peer's reserve (<r0> -> <r1> MiB)"
///   off:  "strata: STRATA_EP_SPLIT_PREFILL off: <why>"
///   trace: "strata ep-split: <to primary|to peer>: cells [c0, c1), <q> QSA + <n> GDN layers, K/V <mb> MB (<host memcpy|D2H+H2D>)
///           <ms_kv> ms, pooled <mb> MB <ms> ms, state <mb> MB <ms> ms, total <ms> ms"
std::string line_on(int64_t lb, int64_t le, int dev, int64_t prompt_mib, int64_t reserve0_mib, int64_t reserve1_mib);
std::string line_off(const std::string& why);
std::string line_trace(const char* dir, const Stats& st);
/// Tappa 4c, per request: "strata ep-split: <fresh> tokens to read: staged read (threshold <min>)" or
/// "... the primary reads every layer with the servers' shares (below the threshold <min>)".
std::string line_request(int64_t fresh, int64_t min_tokens, bool staged);

}  // namespace strata::core::ep_split
