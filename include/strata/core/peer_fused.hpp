// include/strata/core/peer_fused.hpp - STRATA_PEER_FUSED=1 (opt-in, lab/upsync-v0.2.9, docs/PEER_FUSED.md): the peer
// GPU's expert server as ONE persistent kernel per verify window.
//
// Today PeerExperts::launch queues, per layer, one or two host-to-device copies, hadamard (APR), quantize_q8_1_rows,
// native_expert_grouped (mode 8: the fused gate/up kernel + the LDS down kernel) and scatter_rows_f32 on the peer's
// stream, and finish() spins on the stream (~290-330 us of GPU span a layer on the RTX 2080 Ti, STRATA_PEER_STAMP).
// With STRATA_PEER_FUSED=1 the peer gets one launch a window: a kernel whose blocks are all resident waits, step by
// step, for the primary's doorbell (`m_seq_`, mapped host memory), reads the step's activation rows and routed ids
// from the verifier's mapped pages, plans on the device from an ownership table, quantizes, runs the SAME device bodies
// as today's kernels (native_gu_fused_core, native_down_lds_body) for THE LAYER'S OWN TYPES (a mixed pack - Flash-Next
// GSQ-RCO IQ3_XXS has five gate/up types and two down types over its 48 layers - is one kernel, the body chosen a
// step from the layer's layout in VRAM), scatters the rows into the mapped `m_ymiss_` rows with 16-byte streaming
// stores, fences (system scope) and raises an output flag the primary's graph waits on with one wait_flag_ge node.
// No host in the path: it plans the CPU misses as before, counts the peer's entries for the end-of-window check, and
// flushes the stream under WDDM.  The rows are bit for bit those of today's peer path in mode 8.  Off (the default),
// nothing changes: no allocation, no graph node, the per-layer path.
//
// This header is the CONTRACT between the pieces (docs/PEER_FUSED.md s.10): the kernel side (src/kernels/cuda/
// peer_fused.cuh, included inside iq_kernels.cu's translation unit), the host side (PeerExperts, peer_experts.cpp),
// the primary's verifier (verify.cpp) and the bit-identity test (tools/peer_fused_test.cu).  It is included from .cpp
// and .cu files alike: no device code, no CUDA kernels, plain structs with fixed layouts.
#pragma once

#include "strata/kernels/iq_kernels.hpp"      // NativeExpertLayout
#include "strata/kernels/window_limits.hpp"   // kWindowMaxT

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core::peer_fused {

// ---------------------------------------------------------------- constants

/// Threads per block of the persistent kernel (the device bodies it reuses are written for 256).
inline constexpr int kThreads = 256;
/// Most rows (tokens) one step can carry: a verify window's rows (the doorbell publishes a group of them a step).
inline constexpr int kMaxTok = strata::kernels::kWindowMaxT;
/// Entries per step the plan can hold: a window's rows x top-10 (peer_experts.cpp's CAP; the kernel refuses more).
inline constexpr int64_t kCap = (int64_t) strata::kernels::kWindowMaxT * 10;
/// Steps per window the mapped page holds: (le - lb) * G, with G <= 2 groups a layer (48 layers x 2 = 96 fit).
inline constexpr int kMaxSteps = 256;
/// Default bound of every wait inside the kernel (ns): the host's own layer wait is 20 s (verify.cpp); a bench-like
/// 200 ms would be a false positive under paging, PLE gathers or disk prefetch between two doorbells.
inline constexpr unsigned long long kWaitNsDefault = 20000000000ull;
/// Default __launch_bounds__ minimum blocks per SM: <= 128 registers, no spills for the reused bodies (64-78 regs).
inline constexpr int kMinBlocksDefault = 2;
/// Dynamic shared memory the kernel asks for: computed by peer_fused_shape from the pack's layouts (the gate/up
/// activations LDS_NT x n_embd/32 x sizeof(block_q8_1), the 4 x 64 result tile, the largest IQ grid among the pack's
/// gate/up types; or LDS_NT x n_ff/32 x 36 for down; or phase B's plan scratch - the largest of the three).  Here only
/// the sanity cap: sm_75 has 64 KiB per SM.
inline constexpr size_t kDynSharedMax = 48 * 1024;

/// The gate/up and down types the persistent kernel has a body for (docs/PEER_FUSED.md s.5): the IQ gate/up types
/// native_expert_grouped runs as "fused gate/up (mode 8)" - IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21),
/// IQ2_S (22) - and the down types of the LDS down kernel - Q2_0 (42), IQ4_NL (20).  A pack may mix them freely layer
/// by layer; every layer must have one of each.  Masks over these (bit t = type t) describe a pack in Shape.
inline constexpr bool gu_type_fused(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22; }
inline constexpr bool d_type_fused(int t) { return t == 42 || t == 20; }
inline constexpr uint32_t type_bit(int t) { return t >= 0 && t < 32 ? (1u << t) : 0u; }

/// The peer's mapped page (cudaHostAllocPortable | cudaHostAllocMapped), one allocation of kPageBytes, laid out at
/// fixed byte offsets so the host, the peer kernel and the primary's graph agree without a struct crossing the
/// compilers.  Every synchronization word sits on its own 128-byte line.
inline constexpr size_t kPageBytes = 8192;
inline constexpr size_t kOffFlagOut = 0;      ///< uint32: the kernel writes `ring` after the step's rows (UINT32_MAX when it leaves on an error); the primary waits on it
inline constexpr size_t kOffAbort = 128;      ///< uint32: the host raises it (nonzero) to make the kernel leave at its next poll (every wait, and every step before its doorbell)
inline constexpr size_t kOffErr = 256;        ///< uint32 (Err): the kernel's first error; 0 while all is well
inline constexpr size_t kOffExit = 384;       ///< uint32: the last block to leave writes the window's epoch here
inline constexpr size_t kOffCnt = 512;        ///< uint32[kMaxSteps]: per step `groups << 16 | rows`, written before the flag
inline constexpr size_t kOffStamp = 2048;     ///< uint64[2 * kMaxSteps]: %globaltimer when the step's doorbell was seen / its flag raised
static_assert(kOffCnt + (size_t) kMaxSteps * 4 <= kOffStamp, "the count words overlap the stamps");
static_assert(kOffStamp + (size_t) 2 * kMaxSteps * 8 <= kPageBytes, "the stamps overflow the page");

/// Environment knobs (read once by the host side; the names are the contract, the parsing is PeerExperts').
inline constexpr const char* kEnvOn = "STRATA_PEER_FUSED";                 ///< 1: the fused server (master switch)
inline constexpr const char* kEnvHostWait = "STRATA_PEER_FUSED_HOSTWAIT";  ///< 1: no graph node; finish() spins on flag_out (bring-up)
inline constexpr const char* kEnvMinb = "STRATA_PEER_FUSED_MINB";          ///< 2 (default) | 3 | 4: the __launch_bounds__ instance
inline constexpr const char* kEnvBlocks = "STRATA_PEER_FUSED_BLOCKS";      ///< cap on resident blocks per SM (default: the occupancy)
inline constexpr const char* kEnvCoop = "STRATA_PEER_FUSED_COOP";          ///< 0: plain launch even where cooperative launch exists
inline constexpr const char* kEnvWaitMs = "STRATA_PEER_FUSED_WAIT_MS";     ///< bound of the kernel's waits (default 20000)

// ---------------------------------------------------------------- device-visible structs (VRAM / kernel argument)

/// Why the kernel left early (the mapped `err` word).  0 is "no error".
enum Err : uint32_t {
    kErrNone = 0,
    kErrTimeoutSeq = 1,       ///< block 0 waited wait_ns for the doorbell of a step
    kErrTimeoutReady = 2,     ///< a block waited wait_ns for block 0's plan (sync.lready)
    kErrTimeoutBarrier = 3,   ///< a block waited wait_ns at a grid barrier
    kErrAbort = 4,            ///< the host raised `abort`
    kErrPlanOverflow = 5,     ///< more than kCap entries in a step (cannot happen with a well-formed window)
    kErrBadWindow = 6,        ///< steps > kMaxSteps, n_tok > kMaxTok, a null pointer or a geometry the kernel was not built for
    kErrBadType = 7,          ///< a step's layer has a gate/up or down type the kernel has no body for (lay[l]; fused_open checks the pack)
};

/// One step's plan, written by block 0 into VRAM and read by every block: peer_experts.cpp's `Meta`, same fields,
/// same meaning, so the device plan and the host's `launch()` plan can be compared field by field.
///   ptr[g]      device address of group g's blob (slot_ptr[own[layer * n_expert + e]])
///   start[g]    first compact row of group g; start[groups] = rows
///   dst[r]      = r (the down body writes out[dst[r] * n_embd + ...], compact)
///   tok[r]      the entry's token within the step's rows: j / k
///   out_row[r]  the entry index j within the step (the row's place in m_ymiss_ is (tb * k + j) * n_embd)
///   count[0]    groups, count[1] rows (count[2], count[3] unused, 0)
struct Plan {
    unsigned long long ptr[kCap];
    int32_t start[kCap + 1];
    int32_t dst[kCap];
    int32_t tok[kCap];
    int32_t out_row[kCap];
    int32_t count[4];
};

/// The kernel's own synchronization words (VRAM, 128 bytes, zeroed by the host before every launch).
///   bar      the grid barrier in ONE word: blocks arrived in the low 16 bits, the generation in the high 16.  The
///            arriving block learns its generation from the value its atomicAdd returns (no separate load that could
///            be satisfied after the atomic); the last one adds (1 << 16) - nblocks, which clears the count and opens
///            the next generation in one atomic.  nblocks < 65536 (checked by the kernel).
///   done[2]  blocks finished with a step's scatter, step s on done[s & 1], NEVER reset: step s is the (s >> 1)-th use
///            of its word, so its last block is the one whose atomicAdd returns ((s >> 1) + 1) * nblocks - 1.  Two words
///            because a step without barriers (groups == 0) lets a fast block count the next step while slow blocks
///            still write the rows of this one (with G = 2 the primary rings step s + 1 before it waits flag(s)); a
///            block can never be two steps ahead (step s + 2 is rung only after flag(s)), so the two words suffice.
struct alignas(128) Sync {
    unsigned bar;         ///< grid barrier: arrived (low 16 bits) | generation (high 16 bits)
    unsigned lready;      ///< the step whose plan block 0 has published (= ring of that step)
    unsigned done[2];     ///< blocks finished with the scatter of a step of that parity (monotone, see above)
    unsigned quit;        ///< nonzero: leave the step loop (an error or an abort was seen)
    unsigned exited;      ///< blocks that left (the last writes the mapped `exit` word)
    unsigned pad[26];
};
static_assert(sizeof(Sync) == 128, "Sync is one 128-byte line");

/// Everything the kernel needs for one window, passed BY VALUE as the launch argument.  Pointers marked "mapped" are
/// host memory as seen by the PEER device (cudaHostGetDevicePointer under the peer's device); the others are VRAM.
struct Window {
    // ---- geometry (the verifier's: Verifier::run's T, G, lb_, le_, gtb/gte; the model's k, n_embd, n_ff, n_expert)
    int steps = 0;           ///< (le - lb) * G; step s is layer lb + s / G, group s % G, ring s + 1
    int G = 1;               ///< row groups a layer (1 or 2)
    int lb = 0;              ///< first layer of the stage (0 with a peer: no layer split)
    int T = 0;               ///< rows in the window
    int k = 0;               ///< routed experts a row
    int gtb[2] = {0, 0};     ///< first row of group g
    int gte[2] = {0, 0};     ///< one past the last row of group g
    int64_t n_embd = 0;      ///< N (2560): the same on every layer (lay[l].n_embd)
    int64_t n_ff = 0;        ///< the expert's hidden width (640): the same on every layer (lay[l].n_ff)
    int64_t n_expert = 0;    ///< experts a layer (512)
    int n_layers = 0;        ///< entries of `lay`; every step's layer lb + s / G is below it
    uint32_t epoch = 0;      ///< the window's number, written to `exit` when the kernel leaves
    int nblocks = 0;         ///< the grid: every block resident (the barrier counts to it; < 65536, the barrier word's low half)
    unsigned long long wait_ns = kWaitNsDefault;   ///< bound of every wait
    // ---- the primary's mapped pages (peer device pointers)
    const uint32_t* seq = nullptr;     ///< m_seq_: the doorbell ring, >= s + 1 means step s is published
    const float* x = nullptr;          ///< m_x_: T x n_embd fp32 rows; the step reads rows [gtb, gte)
    const int32_t* ids = nullptr;      ///< m_ids_: T x k routed ids; the step reads [gtb * k, gte * k)
    float* ymiss = nullptr;            ///< m_ymiss_: T x k x n_embd fp32 rows; entry j of the step is row gtb * k + j
    // ---- the peer's mapped page (peer device pointers, see kOff*)
    uint32_t* flag_out = nullptr;
    const uint32_t* abort = nullptr;
    uint32_t* err = nullptr;
    uint32_t* exit = nullptr;
    uint32_t* cnt = nullptr;
    unsigned long long* stamp = nullptr;
    // ---- VRAM
    const int32_t* own = nullptr;              ///< n_layers x n_expert: the peer's slot or -1 ("the peer computes it")
    const unsigned long long* slot_ptr = nullptr;   ///< slot -> device address of its blob
    Plan* plan = nullptr;                      ///< [2]: step s uses plan[s & 1]
    void* q8 = nullptr;                        ///< kMaxTok x n_embd/32 block_q8_1
    void* hq = nullptr;                        ///< the gate/up body's output (scratch + peer_fused_hq_offset)
    float* out = nullptr;                      ///< kCap x n_embd compact rows
    Sync* sync = nullptr;
    /// n_layers layouts, one a layer (native_expert_layout(gu_type, d_type, n_embd, n_ff) of the pack's layer):
    /// the step's bodies are chosen from lay[l].gu_type / lay[l].d_type and read its row strides and offsets.
    /// Written by the host before the launch, never during a window.  Every layer: gu_type_fused, d_type_fused,
    /// n_embd / n_ff as above, had_block 0, d_signs null, gu_nib false (fused_open and peer_fused_shape check).
    const strata::kernels::NativeExpertLayout* lay = nullptr;
};

/// What peer_fused_shape resolved for a pack on a device: the kernel instance and its grid.
struct Shape {
    uint32_t gu_mask = 0;            ///< the pack's gate/up types, bit t = type t (over gu_type_fused: 16, 17, 18, 21, 22)
    uint32_t d_mask = 0;             ///< the pack's down types, bit t = type t (42, 20)
    int minb = kMinBlocksDefault;    ///< the __launch_bounds__ instance (2, or 3 / 4 when built)
    size_t grid_bytes = 0;           ///< the largest IQ grid among the pack's gate/up types (part of dyn_shared)
    size_t dyn_shared = 0;           ///< dynamic shared bytes the launch passes
    int blocks_per_sm = 0;           ///< cudaOccupancyMaxActiveBlocksPerMultiprocessor, capped by STRATA_PEER_FUSED_BLOCKS
    int sms = 0;                     ///< multiprocessors of the device
    int blocks = 0;                  ///< blocks_per_sm * sms: the grid
    bool cooperative = false;        ///< launched with cudaLaunchCooperativeKernel (co-residency enforced by the driver)
};

// ---------------------------------------------------------------- host-side structs (verifier <-> PeerExperts)

/// What the verifier hands PeerExperts at the start of a window (host pointers of its mapped pages; the peer resolves
/// its own device pointers).  Filled in Verifier::run after stage_inputs, before the graph launch.
struct WindowHost {
    int T = 0, G = 1, lb = 0, le = 0, k = 0;
    int64_t n_embd = 0;
    int gtb[2] = {0, 0}, gte[2] = {0, 0};
    const uint32_t* seq_h = nullptr;   ///< h_seq_
    const float* x_h = nullptr;        ///< h_x_
    const int32_t* ids_h = nullptr;    ///< h_ids_
    float* ymiss_h = nullptr;          ///< h_ymiss_
};

/// The verifier's handle on the fused peer, installed with Verifier::set_peer_fused before init.  PeerExperts owns the
/// object (fused_hook()); the function pointers are PeerExperts' static trampolines with ctx = the PeerExperts.
///   begin   after stage_inputs, before the graph launch: tables, words, kernel launch, stream query (WDDM)
///   end     after the window's cudaStreamSynchronize: the kernel has left, err == 0, cnt[s] == the host's counts
///   flush   from the host's layer-wait loop beside the 2 ms flush of cs_: cudaStreamQuery on the peer's stream
///   abort   from release_gpu_waits (between two raises of flag_out_host to UINT32_MAX: a step in flight may still
///           write its ring over the first) and on early returns: raises `abort`, drains the peer's stream (bounded)
/// flag_out_dev0 is the primary's device pointer of flag_out: the graph adds wait_flag_ge(flag_out_dev0, ring) before
/// its wait on m_flag_ at every step.  Null with STRATA_PEER_FUSED_HOSTWAIT=1: no node, finish() waits instead.
/// The kernel writes flag_out = UINT32_MAX itself when it leaves on an error (a timeout, the abort): the primary's
/// waits pass, the window fails at `end` (err != 0), and the rows are discarded.
struct Hook {
    void* ctx = nullptr;
    bool (*begin)(void* ctx, const WindowHost& w, std::string& err) = nullptr;
    bool (*end)(void* ctx, std::string& err) = nullptr;
    void (*flush)(void* ctx) = nullptr;
    void (*abort)(void* ctx) = nullptr;
    uint32_t* flag_out_host = nullptr;         ///< the mapped word (host view): release_gpu_waits raises it to UINT32_MAX
    const uint32_t* flag_out_dev0 = nullptr;   ///< the same word as the primary device sees it (null: no graph wait)
};

/// Byte offset of `hq` inside a scratch of native_expert_scratch_bytes(cap_entries, n_ff): the gate, up and h float
/// planes come first, each rounded to 256 bytes, exactly as native_expert_grouped lays them out (iq_kernels.cu).
inline size_t hq_offset(int64_t cap_entries, int64_t n_ff) {
    const size_t f = (size_t) cap_entries * (size_t) n_ff * sizeof(float), fa = (f + 255) & ~(size_t) 255;
    return 3 * fa;
}

/// Bytes of one q8_1 image of `rows` activation rows of width `n_cols` (36 bytes a block of 32 values).
inline size_t q8_bytes(int64_t rows, int64_t n_cols) { return (size_t) rows * (size_t) (n_cols / 32) * 36; }

/// The pack's types as text for a message: "16 17 21" from a mask.
inline std::string mask_text(uint32_t mask) {
    std::string s;
    for (int t = 0; t < 32; ++t)
        if (mask & (1u << t)) { if (!s.empty()) s += ' '; s += std::to_string(t); }
    return s.empty() ? "none" : s;
}

}  // namespace strata::core::peer_fused

namespace strata::kernels {

/// Resolves, for the pack's `n_layers` layouts `lay` (one a layer, in layer order) on the CURRENT device (the caller
/// switched to the peer), whether native_expert_grouped would run exactly "fused gate/up (mode 8) + LDS down" for
/// every one of them, and sizes the kernel: exp_mode() == 8 on this device, every layer's gu_type in {16, 17, 18, 21,
/// 22} (16 only where the fused gate/up takes it, kExpLds16) and d_type in {42, 20}, one n_embd and one n_ff for the
/// whole pack (both multiples of 32), no nibble slot, no APR (had_block == 0, d_signs == nullptr), none of the
/// STRATA_EXPERT_V2 / V2K / EXPDEC / GROUPED_V1 / GU_SWIGLU / DOWN_WS branches taken, g_exp_phase == 0.  Fills `out`
/// (the type masks, the largest grid, dynamic shared bytes, occupancy-derived grid, cooperative launch availability)
/// and returns true; or returns false with `why` naming the layer.  Called at PeerExperts::open, never inside a graph
/// capture.
bool peer_fused_shape(const NativeExpertLayout* lay, int n_layers, int minb, int blocks_per_sm_cap, bool allow_cooperative,
                      strata::core::peer_fused::Shape& out, std::string& why);

/// Launches the persistent kernel for one window on `stream` (the caller switched to the peer's device and zeroed
/// `w.sync` on the same stream).  Uses cudaLaunchCooperativeKernel when `shape.cooperative`, else a plain launch with
/// `shape.blocks` blocks of kThreads threads and `shape.dyn_shared` dynamic shared bytes.  Returns false with `err` on a
/// launch error (cudaGetLastError is consumed).  Does not query or synchronize the stream: the host does.
bool peer_fused_launch(const strata::core::peer_fused::Window& w, const strata::core::peer_fused::Shape& shape,
                       cudaStream_t stream, std::string& err);

}  // namespace strata::kernels
