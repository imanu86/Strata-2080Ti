// src/kernels/cuda/peer_fused.cuh - STRATA_PEER_FUSED=1 (opt-in, lab/upsync-v0.2.9, docs/PEER_FUSED.md s.5, s.8): the
// peer GPU's expert server as ONE persistent kernel a verify window, and the two host functions of the contract
// (include/strata/core/peer_fused.hpp): peer_fused_shape (the instance and the grid) and peer_fused_launch.
//
// This file is NOT a translation unit.  iq_kernels.cu includes it once, inside `namespace strata::kernels`, after
// native_expert_grouped: the kernel reuses native_gu_fused_core / native_down_lds_body and everything they stand on
// (Fmt, DotG, GridOf, the IQ grids, warp_sum, q8_1_store, LDS_NT / LDS_RB, block_q8_1), and the host side re-makes
// native_expert_grouped's dispatch decision from the same statics (exp_mode, g_exp_phase, g_grouped_v1, nib_layout,
// sm75_down_em, down_ws_on).  No -rdc, no second copy of the tables.  Headers are included by iq_kernels.cu at file
// scope (strata/core/peer_fused.hpp among them): nothing is included here, because an #include inside a namespace
// would nest whatever the header declares.
//
// The kernel, step by step (s.5): block 0 waits the primary's doorbell (m_seq_, mapped) and plans on the device from
// the ownership table in the order PeerExperts::launch plans on the host (s.8); every block then quantizes its share
// of the step's rows (quantize_q8_1_kernel's own thread -> value map), reads the LAYER'S layout from VRAM (w.lay[l]:
// a mixed pack has another gate/up or down type on the next layer), runs the fused gate/up items and the LDS down
// items one (bx, g) each with the body of THAT type - the same bodies as the kernels of today with the block index as
// an argument - and scatters the compact rows into the mapped m_ymiss_ rows with streaming 16-byte stores; the last
// block of the step writes cnt[s] and raises flag_out.  Three grid barriers a step (our own: no cooperative_groups),
// none between the scatter and the next step, none at all on a step without entries (groups == 0): the done counters
// are therefore one a step parity and never reset (Sync::done), because with G = 2 the primary rings step s + 1 before
// it waits flag(s) and a fast block may count an empty step s + 1 while slow blocks still write the rows of s.  Every
// wait is bounded (wait_ns) and checks the host's `abort` and the kernel's own `quit`; block 0 also reads `abort` once
// a step before its doorbell, so a window the host gave up on is not served to its end on doorbells the released
// primary keeps ringing.  An error makes every block leave: the last one out writes the window's epoch into `exit`
// and flag_out = UINT32_MAX (the primary's waits pass; `end` fails the window on `err`).  A step never raises its own
// flag once an error was seen.
//
// One kernel for every type (s.5): the five IQ gate/up bodies (IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S 21, IQ2_S 22)
// and the two down bodies (Q2_0 42, IQ4_NL 20) are all in the instance, and a block-uniform switch on the layer's
// type picks one a phase.  The gate/up core takes its shared buffers as arguments so the five bodies share ONE grid
// area (sized by the pack's largest grid: 8 KiB for IQ2_S) instead of five static ones; the dynamic buffer is
// [activations LDS_NT x n_embd/32 q8_1 blocks | the 4 x 64 result tile | the grid] for gate/up and the down body's own
// LDS_NT x n_ff/32 blocks for down (the two phases are separated by a barrier).
//
// Memory and visibility (no P2P under WDDM): the primary's pages are read with __ldcv (never cached) and written with
// __stwt + __threadfence_system; the words in VRAM (Sync, the plan, q8, hq, out) are published with __threadfence
// and acquired with __threadfence on the reading side, so plain loads after a wait or a barrier see what another SM
// wrote during the kernel (the L1 of the reader is invalidated by its own fence).  This holds for weak loads only,
// never for ld.global.nc (the read-only cache is not coherent inside a launch): the bodies this kernel reuses take
// their plan / q8 / hq pointers WITHOUT __restrict__ for that reason (iq_kernels.cu, native_gu_fused_core and
// native_down_lds_body) - those buffers are rewritten by other blocks between the steps of one launch.  The layouts
// (w.lay) are written by the host before the launch and never during it.

namespace pf = ::strata::core::peer_fused;

#if STRATA_EXP_LAYOUTS && !defined(__HIPCC__)
#define STRATA_PEER_FUSED_BUILT 1
#else
#define STRATA_PEER_FUSED_BUILT 0   // HIP, or a build without the AMD layouts: the bodies the kernel reuses do not exist
#endif

static_assert(pf::kCap <= pf::kThreads, "the device plan takes one thread an entry: kCap must not exceed kThreads");
static_assert(pf::kThreads == 256, "the reused bodies (native_gu_fused_core, native_down_lds_body) are written for 256 threads");

#if STRATA_PEER_FUSED_BUILT

namespace {

// ---------------------------------------------------------------- shared-memory layout (host and device agree)

/// Bytes of dynamic shared memory the device plan (phase B, block 0) borrows from the bodies' buffer: four int arrays
/// of kCap (id, first, lead, size).  peer_fused_shape folds it into Shape::dyn_shared.
constexpr size_t kPfPlanScratchBytes = (size_t) 4 * (size_t) pf::kCap * sizeof(int);
/// The gate/up core's result tile (res[LDS_NT][64]), placed after the activations in the dynamic buffer.
constexpr size_t kPfResBytes = (size_t) LDS_NT * 64 * sizeof(float);
/// The gate/up core's activation area: LDS_NT entries of n_embd/32 q8_1 blocks (36 B each, so always a multiple of 16:
/// the tile and the grid behind it stay 16-byte aligned).
__host__ __device__ constexpr size_t pf_sx_bytes(int64_t n_embd) { return (size_t) LDS_NT * (size_t) (n_embd / 32) * sizeof(block_q8_1); }
static_assert(sizeof(block_q8_1) == 36, "the q8_1 block is 36 bytes (the activation area's alignment argument)");
/// Bytes of a gate/up type's grid in shared memory (GridOf), 0 for a type the kernel has no body for.
template<int TG> constexpr size_t pf_grid_bytes_of() { return (size_t) GridOf<TG>::N * sizeof(typename GridOf<TG>::T); }
inline size_t pf_grid_bytes(int t) {
    switch (t) {
        case 16: return pf_grid_bytes_of<16>();
        case 17: return pf_grid_bytes_of<17>();
        case 18: return pf_grid_bytes_of<18>();
        case 21: return pf_grid_bytes_of<21>();
        case 22: return pf_grid_bytes_of<22>();
        default: return 0;
    }
}
/// Slack a block's wait for block 0's plan word gets over wait_ns: block 0 may itself wait wait_ns for the doorbell,
/// and its own timeout (kErrTimeoutSeq) must be the one reported, not the others' (kErrTimeoutReady).
constexpr unsigned long long kPfReadySlackNs = 1000000000ull;

// ---------------------------------------------------------------- device helpers

/// The GPU's clock in ns (%globaltimer, as strata_now_ns in verify_kernels.cu).
__device__ __forceinline__ unsigned long long pf_now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
/// Yield while polling (a backoff hint): pre-Volta CUDA has no __nanosleep and the loop just spins.
__device__ __forceinline__ void pf_pause(unsigned ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    (void) ns;
#else
    __nanosleep(ns);
#endif
}
__device__ __forceinline__ uint32_t pf_ld(const uint32_t* p) { return *(volatile const uint32_t*) p; }
__device__ __forceinline__ void pf_st(uint32_t* p, uint32_t v) { *(volatile uint32_t*) p = v; }

/// The first error wins: the mapped `err` word (0 while all is well), then the kernel's own `quit`, which every wait
/// checks.  Two blocks writing at once write the same kind of thing; no atomic on the mapped word is needed.
__device__ __forceinline__ void pf_fail(const pf::Window& w, uint32_t code) {
    if (pf_ld(w.err) == 0) pf_st(w.err, code);
    __threadfence_system();
    pf_st(&w.sync->quit, 1u);
    __threadfence();
}

/// Grid barrier (s.5): thread 0 of every block arrives on Sync::bar, ONE word - the count in the low 16 bits, the
/// generation in the high 16 - and takes its generation from the value its atomicAdd returns (a separate load of a
/// generation word could be satisfied after the atomic, which is the load-buffering case the PTX model allows and
/// cooperative_groups avoids the same way).  The last one adds (1 << 16) - nblocks: the count is 0 again and the
/// generation is g + 1, in one atomic; the others spin until the high half changes, with __nanosleep(32), checking
/// `quit` every 256 turns, the host's `abort` (mapped) and the clock every 4096 (-> kErrTimeoutBarrier).
/// __syncthreads before and after; every thread fences after it so its plain loads see the other blocks' writes.
/// Returns false, uniformly over the block, when the step loop must be left.
__device__ bool pf_grid_barrier(const pf::Window& w, int nblocks, int* s_ok) {
    __syncthreads();
    if (threadIdx.x == 0) {
        pf::Sync* sy = w.sync;
        int ok = 1;
        __threadfence();   // this block's global writes (q8, hq, out) are at L2 before it counts itself arrived
        const unsigned old = atomicAdd(&sy->bar, 1u);
        const unsigned g = old >> 16;
        if ((old & 0xFFFFu) == (unsigned) nblocks - 1) {
            __threadfence();
            atomicAdd(&sy->bar, (1u << 16) - (unsigned) nblocks);   // count -> 0, generation -> g + 1
        } else {
            const unsigned long long t0 = pf_now_ns();
            unsigned spins = 0;
            while ((pf_ld(&sy->bar) >> 16) == g) {
                pf_pause(32);
                ++spins;
                if ((spins & 255u) == 0 && pf_ld(&sy->quit) != 0) { ok = 0; break; }
                if ((spins & 4095u) == 0) {
                    if (pf_ld(w.abort) != 0) { pf_fail(w, pf::kErrAbort); ok = 0; break; }
                    if (pf_now_ns() - t0 > w.wait_ns) { pf_fail(w, pf::kErrTimeoutBarrier); ok = 0; break; }
                }
            }
        }
        *s_ok = ok;
    }
    __syncthreads();
    __threadfence();   // acquire: what the other blocks wrote before arriving is visible to this thread's loads
    return *s_ok != 0;
}

/// Phase B (block 0, one thread an entry): the step's plan from the mapped ids and the ownership table, in exactly
/// the order PeerExperts::launch plans it on the host (s.8): kind == 2 there is owned(i) here (the same own_win_
/// table); the groups are the owned experts in order of first appearance; a group's rows are its entries in entry
/// order; dst[r] = r, tok[r] = i / k, out_row[r] = i; start[groups] = rows.  `scratch` is kPfPlanScratchBytes of the
/// dynamic shared buffer (the bodies are not running in this block yet).  Thread 0 leaves groups / rows in
/// *s_groups / *s_rows; the caller fences, syncs and publishes lready.
__device__ __forceinline__ void pf_plan(const pf::Window& w, pf::Plan* plan, int l, int tb, int n, int* scratch,
                                        int* s_groups, int* s_rows) {
    int* s_id = scratch;                   // the entry's expert id
    int* s_first = s_id + pf::kCap;        // the first entry (<= i) of the same expert
    int* s_lead = s_first + pf::kCap;      // 1: owned and the first of its expert - a group's lead
    int* s_size = s_lead + pf::kCap;       // a lead's rows: the entries of its expert
    const int tid = threadIdx.x;
    const bool live = tid < n;
    int e = -1;
    if (live) e = __ldcv(w.ids + (size_t) tb * (size_t) w.k + tid);
    if (tid < pf::kCap) s_id[tid] = e;
    __syncthreads();
    int first = tid;
    bool owned = false;
    if (live) {
        owned = e >= 0 && (int64_t) e < w.n_expert && w.own[(size_t) l * (size_t) w.n_expert + e] >= 0;
        for (int j = 0; j < tid; ++j) if (s_id[j] == e) { first = j; break; }
    }
    const int lead = (live && owned && first == tid) ? 1 : 0;
    int size = 0;
    if (lead) for (int j = tid; j < n; ++j) if (s_id[j] == e) ++size;
    if (tid < pf::kCap) { s_first[tid] = first; s_lead[tid] = lead; s_size[tid] = size; }
    __syncthreads();
    if (live && owned) {
        // the group is the rank of the entry's lead among the leads, its start the rows of the leads before it, the
        // row that start plus the entries of the same expert before this one
        int g = 0, start = 0, before = 0;
        for (int j = 0; j < first; ++j) { g += s_lead[j]; start += s_size[j]; }
        for (int j = 0; j < tid; ++j) if (s_id[j] == e) ++before;
        const int r = start + before;
        plan->dst[r] = r;
        plan->tok[r] = tid / w.k;
        plan->out_row[r] = tid;
        if (lead) {
            plan->ptr[g] = w.slot_ptr[w.own[(size_t) l * (size_t) w.n_expert + e]];
            plan->start[g] = start;
        }
    }
    if (tid == 0) {
        int groups = 0, rows = 0;
        for (int j = 0; j < n; ++j) { groups += s_lead[j]; rows += s_size[j]; }
        plan->start[groups] = rows;
        plan->count[0] = groups;
        plan->count[1] = rows;
        plan->count[2] = 0;
        plan->count[3] = 0;
        *s_groups = groups;
        *s_rows = rows;
    }
}

// ---------------------------------------------------------------- the kernel

/// The persistent peer server (s.5): one launch a window, every block resident (Shape::blocks = occupancy x SMs),
/// kThreads threads, dynamic shared = Shape::dyn_shared (the bodies' buffers and phase B's scratch are the same
/// buffer).  MINB is the __launch_bounds__ minimum blocks an SM; the types are the layer's (w.lay[l]), not the
/// instance's.
template<int MINB>
__global__ void __launch_bounds__(pf::kThreads, MINB) peer_fused_kernel(const pf::Window w) {
    extern __shared__ int pf_dyn[];   // the bodies' dynamic buffer (their own extern declarations alias it)
    __shared__ int s_ok;
    __shared__ int s_groups, s_rows;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nblocks = (int) gridDim.x;
    bool ok = true;

    // ---- the window, checked once by every block (uniformly): kErrBadWindow leaves before any step
    {
        bool bad = w.steps <= 0 || w.steps > pf::kMaxSteps || w.G < 1 || w.G > 2 || w.k <= 0 || w.T < 0 ||
                   w.n_embd <= 0 || w.n_embd % 32 != 0 || w.n_ff <= 0 || w.n_ff % 32 != 0 || w.n_expert <= 0 ||
                   w.nblocks != nblocks || nblocks >= 65536 ||   // the barrier counts in 16 bits
                   w.lay == nullptr || w.n_layers <= 0 || w.lb < 0 || w.lb + (w.steps + w.G - 1) / w.G > w.n_layers ||
                   w.seq == nullptr || w.x == nullptr || w.ids == nullptr || w.ymiss == nullptr ||
                   w.flag_out == nullptr || w.abort == nullptr || w.err == nullptr || w.exit == nullptr || w.cnt == nullptr ||
                   w.stamp == nullptr || w.own == nullptr || w.slot_ptr == nullptr || w.plan == nullptr || w.q8 == nullptr ||
                   w.hq == nullptr || w.out == nullptr || w.sync == nullptr;
        for (int g = 0; g < w.G && g < 2; ++g) {
            const int nt = w.gte[g] - w.gtb[g];
            if (w.gtb[g] < 0 || nt < 0 || nt > pf::kMaxTok || (int64_t) nt * w.k > pf::kCap) bad = true;
        }
        if (bad) {
            if (blockIdx.x == 0 && tid == 0 && w.err != nullptr && w.sync != nullptr) pf_fail(w, pf::kErrBadWindow);
            ok = false;
        }
    }

    for (int s = 0; ok && s < w.steps; ++s) {
        const int l = w.lb + s / w.G, grp = s % w.G;
        const int tb = w.gtb[grp], n_tok = w.gte[grp] - tb, n = n_tok * w.k;
        const uint32_t want = (uint32_t) s + 1u;
        pf::Plan* plan = w.plan + (s & 1);

        if (blockIdx.x == 0) {
            // ---- A: the doorbell (mapped, volatile), bounded; abort, quit and the clock every 1024 turns.  `abort`
            // is read once BEFORE the spin too: a released primary runs its graph to the end and rings every doorbell
            // left, so a doorbell found already rung must not start a step of a window the host has given up on.
            if (tid == 0) {
                int o = 1;
                const unsigned long long t0 = pf_now_ns();
                unsigned spins = 0;
                if (pf_ld(w.abort) != 0) { pf_fail(w, pf::kErrAbort); o = 0; }
                while (o && pf_ld(w.seq) < want) {
                    pf_pause(64);
                    if ((++spins & 1023u) == 0) {
                        if (pf_ld(w.abort) != 0) { pf_fail(w, pf::kErrAbort); o = 0; break; }
                        if (pf_ld(&w.sync->quit) != 0) { o = 0; break; }
                        if (pf_now_ns() - t0 > w.wait_ns) { pf_fail(w, pf::kErrTimeoutSeq); o = 0; break; }
                    }
                }
                if (o) {
                    __threadfence_system();   // the primary's rows and ids were published before its doorbell
                    *(volatile unsigned long long*) (w.stamp + 2 * s) = pf_now_ns();
                }
                s_ok = o;
            }
            __syncthreads();
            if (!s_ok) { ok = false; break; }
            // ---- B: the plan, then its word in VRAM for the other blocks
            pf_plan(w, plan, l, tb, n, pf_dyn, &s_groups, &s_rows);
            __threadfence();
            __syncthreads();
            if (tid == 0) pf_st(&w.sync->lready, want);
        } else {
            // the other blocks: block 0's plan word (VRAM, never the link), bounded with a slack over block 0's own
            // wait; `quit` every 1024 turns, the host's `abort` (mapped) and the clock every 4096: block 0 may itself
            // be stuck where it cannot read the abort (a barrier nobody completes), these blocks must not depend on it
            if (tid == 0) {
                int o = 1;
                const unsigned long long t0 = pf_now_ns();
                unsigned spins = 0;
                while (pf_ld(&w.sync->lready) < want) {
                    pf_pause(32);
                    if ((++spins & 1023u) == 0) {
                        if (pf_ld(&w.sync->quit) != 0) { o = 0; break; }
                        if ((spins & 4095u) == 0) {
                            if (pf_ld(w.abort) != 0) { pf_fail(w, pf::kErrAbort); o = 0; break; }
                            if (pf_now_ns() - t0 > w.wait_ns + kPfReadySlackNs) { pf_fail(w, pf::kErrTimeoutReady); o = 0; break; }
                        }
                    }
                }
                s_ok = o;
            }
            __syncthreads();
            if (!s_ok) { ok = false; break; }
            __threadfence();   // acquire: block 0's plan
            if (tid == 0) { s_groups = __ldcg(&plan->count[0]); s_rows = __ldcg(&plan->count[1]); }
            __syncthreads();
        }
        const int groups = s_groups, rows = s_rows;

        if (groups > 0) {
            // the layer's layout (VRAM, written by the host before the launch): every thread reads the same words,
            // so the switches below are uniform over the block and the grid; the bodies take it by value as today
            const NativeExpertLayout L = w.lay[l];
            // ---- C: the step's rows quantized (x mapped -> q8 VRAM): a global warp a q8_1 block, quantize_q8_1_kernel's
            // thread -> value map (lane = i % 32), the same q8_1_store
            {
                const float* x = w.x + (size_t) tb * (size_t) w.n_embd;
                block_q8_1* q8 = (block_q8_1*) w.q8;
                const int nwg = (int) ((long long) n_tok * w.n_embd / 32);
                for (int wg = (int) blockIdx.x * (pf::kThreads / 32) + warp; wg < nwg; wg += nblocks * (pf::kThreads / 32)) {
                    const long long i = (long long) wg * 32 + lane;
                    q8_1_store(__ldcv(x + i), q8, i);
                }
            }
            if (!pf_grid_barrier(w, nblocks, &s_ok)) { ok = false; break; }
            // ---- D: gate/up + SwiGLU + q8_1 (native_gu_fused_core of the layer's type), the items (bx, g) of today's
            // grid one block each; the core's shared buffers are slices of the dynamic buffer: activations, tile, grid
            {
                const int nbx = (int) (w.n_ff / 32);
                const int items = nbx * groups;
                const size_t sxb = pf_sx_bytes(w.n_embd);
                float (*res)[64] = (float (*)[64]) ((unsigned char*) pf_dyn + sxb);
                void* grid = (unsigned char*) pf_dyn + sxb + kPfResBytes;
                auto run = [&](auto tg) {
                    constexpr int TG = decltype(tg)::value;
                    using GT = typename GridOf<TG>::T;
                    for (int it = (int) blockIdx.x; it < items; it += nblocks)
                        native_gu_fused_core<TG>(plan->ptr, plan->start, &plan->count[0], plan->tok, (const block_q8_1*) w.q8, L,
                                                 (block_q8_1*) w.hq, it % nbx, it / nbx, (GT*) grid, res, pf_dyn);
                };
                bool bad_type = false;
                switch (L.gu_type) {
                    case 16: run(std::integral_constant<int, 16>{}); break;
                    case 17: run(std::integral_constant<int, 17>{}); break;
                    case 18: run(std::integral_constant<int, 18>{}); break;
                    case 21: run(std::integral_constant<int, 21>{}); break;
                    case 22: run(std::integral_constant<int, 22>{}); break;
                    default: bad_type = true; break;
                }
                if (bad_type) {   // uniform: every block leaves before the barrier, nobody waits for it
                    if (blockIdx.x == 0 && tid == 0) pf_fail(w, pf::kErrBadType);
                    ok = false;
                    break;
                }
            }
            if (!pf_grid_barrier(w, nblocks, &s_ok)) { ok = false; break; }
            // ---- E: down (native_down_lds_body of the layer's type), the items (bx, g) of today's grid one block each
            {
                const int nbx = (int) ((w.n_embd + LDS_RB - 1) / LDS_RB);
                const int items = nbx * groups;
                bool bad_type = false;
                switch (L.d_type) {
                    case 42:
                        for (int it = (int) blockIdx.x; it < items; it += nblocks)
                            native_down_lds_body<42>(plan->ptr, plan->start, &plan->count[0], plan->dst, (const block_q8_1*) w.hq, L,
                                                     w.out, it % nbx, it / nbx);
                        break;
                    case 20:
                        for (int it = (int) blockIdx.x; it < items; it += nblocks)
                            native_down_lds_body<20>(plan->ptr, plan->start, &plan->count[0], plan->dst, (const block_q8_1*) w.hq, L,
                                                     w.out, it % nbx, it / nbx);
                        break;
                    default: bad_type = true; break;
                }
                if (bad_type) {
                    if (blockIdx.x == 0 && tid == 0) pf_fail(w, pf::kErrBadType);
                    ok = false;
                    break;
                }
            }
            if (!pf_grid_barrier(w, nblocks, &s_ok)) { ok = false; break; }
        }

        // ---- F: the compact rows into the mapped m_ymiss_ rows (entry j of the step is row tb * k + j), 16 bytes a
        // store, streaming; then the step's count and flag by the last block, counted on done[s & 1] with no reset
        // (Sync::done): with G = 2 the primary rings step s + 1 before it waits flag(s), so when s + 1 has no entries
        // (no barriers) a fast block may already be counting s + 1 while slow blocks still write the rows of s.  The
        // parity word keeps the two counts apart; step s + 2 (the same word) starts only after flag(s), that is after
        // every increment of s.  Two steps' last blocks may then write flag_out out of order (s + 2 before s + 1) -
        // harmless: every value written means every step below it is complete, and the waits are >=.
        {
            const int n4 = (int) (w.n_embd / 4);
            const long long total = (long long) rows * n4;
            const float4* out4 = (const float4*) w.out;
            float4* ym4 = (float4*) w.ymiss;
            for (long long idx = (long long) blockIdx.x * pf::kThreads + tid; idx < total; idx += (long long) nblocks * pf::kThreads) {
                const int r = (int) (idx / n4), c = (int) (idx % n4);
                const int j = plan->out_row[r];
                __stwt(ym4 + ((size_t) tb * (size_t) w.k + (size_t) j) * (size_t) n4 + c, out4[(size_t) r * (size_t) n4 + c]);
            }
        }
        __threadfence_system();   // every thread: its rows are in host memory before the block counts itself done
        __syncthreads();
        if (tid == 0) {
            const unsigned old = atomicAdd(&w.sync->done[s & 1], 1u);
            if (old == (unsigned) ((s >> 1) + 1) * (unsigned) nblocks - 1u) {   // the last block of THIS step
                __threadfence();
                if (pf_ld(&w.sync->quit) == 0) {   // never the step's flag after an error: the exit below raises UINT32_MAX
                    pf_st(w.cnt + s, ((uint32_t) groups << 16) | (uint32_t) rows);
                    *(volatile unsigned long long*) (w.stamp + 2 * s + 1) = pf_now_ns();
                    __threadfence_system();
                    pf_st(w.flag_out, want);
                    __threadfence_system();
                }
            }
        }
    }

    // ---- exit (the steps done, or an error / abort seen): the last block out writes the window's epoch, and after an
    // error flag_out = UINT32_MAX, so the primary's wait_flag_ge nodes (unbounded) pass whatever step they wait for:
    // the host sees err != 0 at `end` and fails the window, the rows are discarded.  Without it a barrier timeout on
    // the window's last steps would hold the primary (and the host's cudaStreamSynchronize) until an outside release.
    __syncthreads();
    if (tid == 0 && w.sync != nullptr && w.exit != nullptr) {
        __threadfence();
        const unsigned old = atomicAdd(&w.sync->exited, 1u);
        if (old == (unsigned) nblocks - 1) {
            pf_st(w.exit, w.epoch);
            __threadfence_system();
            if (pf_ld(&w.sync->quit) != 0 && w.flag_out != nullptr) {
                pf_st(w.flag_out, UINT32_MAX);
                __threadfence_system();
            }
        }
    }
}

// ---------------------------------------------------------------- host helpers

using PfKernel = void (*)(const pf::Window);

/// The compiled instance for minb, or null.  MINB 2 always; MINB 3 and 4 only with the compile-time macro
/// STRATA_PEER_FUSED_MINB_ALL (two more large kernels otherwise).  One instance holds every type's body.
PfKernel pf_instance(int minb) {
    if (minb == 2) return peer_fused_kernel<2>;
#ifdef STRATA_PEER_FUSED_MINB_ALL
    if (minb == 3) return peer_fused_kernel<3>;
    if (minb == 4) return peer_fused_kernel<4>;
#endif
    return nullptr;
}

}  // namespace

#endif  // STRATA_PEER_FUSED_BUILT

// ---------------------------------------------------------------- the contract's host functions (peer_fused.hpp)

bool peer_fused_shape(const NativeExpertLayout* lay, int n_layers, int minb, int blocks_per_sm_cap, bool allow_cooperative,
                      pf::Shape& out, std::string& why) {
    out = pf::Shape{};
#if !STRATA_PEER_FUSED_BUILT
    (void) lay; (void) n_layers; (void) minb; (void) blocks_per_sm_cap; (void) allow_cooperative;
    why = "the fused peer server is not built into this engine (a HIP build, or one without the AMD expert layouts)";
    return false;
#else
    char buf[256];
    if (lay == nullptr || n_layers <= 0) { why = "no layers"; return false; }
    // ---- the pack, layer by layer: only what native_expert_grouped runs as "fused gate/up (mode 8) + LDS down" (s.8),
    // one n_embd and one n_ff for the whole pack (the buffers are sized once), the types free to change a layer
    const int64_t n_embd = lay[0].n_embd, n_ff = lay[0].n_ff;
    if (n_ff <= 0 || n_ff % 32 != 0 || n_embd <= 0 || n_embd % 32 != 0) {
        std::snprintf(buf, sizeof buf, "n_embd %lld, n_ff %lld: both must be positive multiples of 32", (long long) n_embd, (long long) n_ff);
        why = buf;
        return false;
    }
    for (int l = 0; l < n_layers; ++l) {
        const NativeExpertLayout& L = lay[l];
        if (nib_layout(L) || L.gu_nib) {
            std::snprintf(buf, sizeof buf, "layer %d: 4-bit gate/up slots (STRATA_EXPERT_NIBBLE): not in the fused server", l);
            why = buf;
            return false;
        }
        if (!pf::gu_type_fused(L.gu_type) || (L.gu_type == 16 && !kExpLds16)) {
            std::snprintf(buf, sizeof buf, "layer %d: gate/up type %d: the fused server is built for IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21) and IQ2_S (22)", l, L.gu_type);
            why = buf;
            return false;
        }
        if (!pf::d_type_fused(L.d_type)) {
            std::snprintf(buf, sizeof buf, "layer %d: down type %d: the fused server is built for Q2_0 (42) and IQ4_NL (20)", l, L.d_type);
            why = buf;
            return false;
        }
        if (L.n_embd != n_embd || L.n_ff != n_ff) {
            std::snprintf(buf, sizeof buf, "layer %d: n_embd %lld, n_ff %lld while layer 0 has %lld, %lld: one geometry a pack", l,
                          (long long) L.n_embd, (long long) L.n_ff, (long long) n_embd, (long long) n_ff);
            why = buf;
            return false;
        }
        if (L.had_block != 0 || L.d_signs != nullptr) {
            std::snprintf(buf, sizeof buf, "layer %d: APR (Hadamard-folded weights): hadamard_rows is not in the fused server", l);
            why = buf;
            return false;
        }
        if (L.tail_off != 0) {
            std::snprintf(buf, sizeof buf, "layer %d: a blob tail (NVFP4 scales) the fused bodies do not read", l);
            why = buf;
            return false;
        }
        out.gu_mask |= pf::type_bit(L.gu_type);
        out.d_mask |= pf::type_bit(L.d_type);
        out.grid_bytes = std::max(out.grid_bytes, pf_grid_bytes(L.gu_type));
    }
    // ---- native_expert_grouped's dispatch on THIS device (the caller switched to the peer): mode 8 and none of the
    // other branches.  V2K takes gate/up 12 / 13 only, GU_SWIGLU only where em0 == 0, EXPDEC's gate/up only where the
    // fused gate/up is not taken: unreachable with the types and the mode above.
    const int em0 = exp_mode();
    if (em0 != 8) {
        std::snprintf(buf, sizeof buf, "exp_mode() is %d on this device, not 8: native_expert_grouped would not take the fused gate/up + LDS down (STRATA_EXP_MODE=8)", em0);
        why = buf;
        return false;
    }
    if (g_exp_phase != 0) {
        std::snprintf(buf, sizeof buf, "the bench's expert phase is %d (native_expert_set_mode)", g_exp_phase);
        why = buf;
        return false;
    }
    if (g_grouped_v1) { why = "STRATA_GROUPED_V1 is on: the v1 launches, not the fused gate/up"; return false; }
    {
        const char* v = std::getenv("STRATA_EXPERT_V2");
        if (v != nullptr && v[0] == '1' && (out.gu_mask & pf::type_bit(21)) && (out.d_mask & pf::type_bit(20)) && n_embd == 2560 && n_ff == 640) {
            why = "STRATA_EXPERT_V2=1 takes the S26 kernels for the IQ3_S / IQ4_NL layers of this pack";
            return false;
        }
    }
    {
        const int emd = sm75_down_em(em0, pf::kCap);
        if (emd != 7 && emd != 8) {
            std::snprintf(buf, sizeof buf, "sm75_down_em(%d) is %d: the down would not take the LDS kernel", em0, emd);
            why = buf;
            return false;
        }
    }
#if STRATA_DOWN_WS_BUILT
    for (int l = 0; l < n_layers; ++l)
        if (down_ws_on(lay[l])) {
            std::snprintf(buf, sizeof buf, "layer %d: STRATA_DOWN_WS is on: the down takes the weight-stationary kernel", l);
            why = buf;
            return false;
        }
#endif
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { why = "cudaGetDevice failed"; (void) cudaGetLastError(); return false; }
    // lds_down_ok: Q2_0 always, IQ4_NL off gfx1151 (a HIP device, never this build)
    if ((out.d_mask & pf::type_bit(20)) && gfx1151_exp8(dev)) { why = "IQ4_NL down on gfx1151 keeps the CUDA layout"; return false; }
    // ---- the instance
    const bool minb_ok = minb == 2
#ifdef STRATA_PEER_FUSED_MINB_ALL
                         || minb == 3 || minb == 4
#endif
        ;
    if (!minb_ok) {
        std::snprintf(buf, sizeof buf, "STRATA_PEER_FUSED_MINB=%d: this engine has the MINB 2 instance only (3 and 4 need STRATA_PEER_FUSED_MINB_ALL at build time)", minb);
        why = buf;
        return false;
    }
    const PfKernel k = pf_instance(minb);
    if (k == nullptr) {
        std::snprintf(buf, sizeof buf, "no kernel instance for MINB %d", minb);
        why = buf;
        return false;
    }
    // ---- dynamic shared: the largest of the gate/up core's buffer (activations + tile + the pack's largest grid),
    // the down body's (LDS_NT entries of q8_1 of n_ff) and phase B's scratch, one buffer
    const size_t dyn_gu = pf_sx_bytes(n_embd) + kPfResBytes + out.grid_bytes;
    const size_t dyn_d = (size_t) LDS_NT * (size_t) (n_ff / 32) * sizeof(block_q8_1);
    const size_t dyn = std::max(std::max(dyn_gu, dyn_d), kPfPlanScratchBytes);
    if (dyn > pf::kDynSharedMax) {
        std::snprintf(buf, sizeof buf, "dynamic shared %zu B exceeds the cap of %zu B", dyn, pf::kDynSharedMax);
        why = buf;
        return false;
    }
    int occ = 0;
    {
        const cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, pf::kThreads, dyn);
        if (e != cudaSuccess) {
            why = std::string("cudaOccupancyMaxActiveBlocksPerMultiprocessor: ") + cudaGetErrorString(e);
            (void) cudaGetLastError();
            return false;
        }
    }
    if (occ < 1) {
        std::snprintf(buf, sizeof buf, "the instance (MINB %d, %zu B dynamic shared) does not fit one block an SM", minb, dyn);
        why = buf;
        return false;
    }
    int sms = 0, coop = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || sms < 1) {
        why = "cudaDevAttrMultiProcessorCount failed";
        (void) cudaGetLastError();
        return false;
    }
    if (cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev) != cudaSuccess) { coop = 0; (void) cudaGetLastError(); }
    out.minb = minb;
    out.dyn_shared = dyn;
    out.blocks_per_sm = blocks_per_sm_cap > 0 ? std::min(occ, blocks_per_sm_cap) : occ;
    out.sms = sms;
    out.blocks = out.blocks_per_sm * sms;
    out.cooperative = allow_cooperative && coop != 0;
    if (out.blocks < 1) { why = "the grid came out empty"; return false; }
    return true;
#endif
}

bool peer_fused_launch(const pf::Window& w, const pf::Shape& shape, cudaStream_t stream, std::string& err) {
#if !STRATA_PEER_FUSED_BUILT
    (void) w; (void) shape; (void) stream;
    err = "peer fused: the fused peer server is not built into this engine";
    return false;
#else
    const PfKernel k = pf_instance(shape.minb);
    if (k == nullptr) {
        err = "peer fused: no kernel instance for MINB " + std::to_string(shape.minb);
        return false;
    }
    if (shape.blocks < 1 || w.nblocks != shape.blocks) {
        err = "peer fused: the window's nblocks (" + std::to_string(w.nblocks) + ") is not the shape's grid (" + std::to_string(shape.blocks) + ")";
        return false;
    }
    if (w.lay == nullptr || w.n_layers <= 0) { err = "peer fused: the window has no layouts"; return false; }
    cudaError_t e = cudaSuccess;
    if (shape.cooperative) {   // co-residency enforced by the driver: a grid that does not fit fails here, clearly
        pf::Window arg = w;
        void* args[] = { (void*) &arg };
        e = cudaLaunchCooperativeKernel((const void*) k, dim3((unsigned) shape.blocks), dim3((unsigned) pf::kThreads), args,
                                        shape.dyn_shared, stream);
    } else {
        k<<<(unsigned) shape.blocks, (unsigned) pf::kThreads, shape.dyn_shared, stream>>>(w);
        e = cudaGetLastError();
    }
    if (e != cudaSuccess) {
        err = std::string("peer fused: launch (") + (shape.cooperative ? "cooperative" : "plain") + ", " + std::to_string(shape.blocks) +
              " blocks): " + cudaGetErrorString(e);
        (void) cudaGetLastError();
        return false;
    }
    return true;
#endif
}
