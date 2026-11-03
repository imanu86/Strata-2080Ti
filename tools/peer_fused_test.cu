// tools/peer_fused_test.cu - PEER_FUSED (docs/PEER_FUSED.md s.8, lab/upsync-v0.2.9): the persistent expert server of
// STRATA_PEER_FUSED=1 (peer_fused_kernel, launched through peer_fused_launch) against today's peer path, bit for bit.
//
// No model, no engine: `n_slots` synthetic expert blobs with a VALID structure (fp16 block scales, finite and small;
// random code bytes - the kernels only index 8-bit lookups and dp4a with them), an ownership table layer x expert
// with ~40% of the pairs on a slot, and the verifier's mapped pages faked on the host: `seq` (the doorbell), `x`
// (kMaxTok x n_embd fp32), `ids` (kMaxTok x k int32), `ymiss` (kMaxTok x k x n_embd fp32).  Per window the host plays
// the primary IN THE ORDER OF ITS GRAPH (verify.cpp record_window: pre(l, 0), pre(l, 1), post(l, 0), pre(l + 1, 0),
// post(l, 1), ...): it writes the rows and ids of the first G steps, fences, rings them, then for every step s spins on
// `flag_out >= s + 1`, snapshots the step's rows of `ymiss` and only then rings step s + G - so with G = 2 the doorbell
// of s + 1 is up while the kernel still works on s, as in the engine.  When the kernel has left (cudaStreamSynchronize), every step is
// recomputed with the path peer_experts.cpp runs today, on the same device and a second stream - the host plan of
// PeerExperts::launch (kind == 2 <=> owned, groups in order of first appearance, Meta ptr/start/dst/tok/out_row),
// cudaMemcpyAsync of the rows, quantize_q8_1_rows, native_expert_grouped (mode 8: fused gate/up + LDS down),
// scatter_rows_f32 into a separate mapped page `ymiss_ref` - and the rows are memcmp'd: entry j of the step at
// (gtb * k + j) * n_embd.  Entries the peer does not own must keep the sentinel the host wrote before the ring (the
// pool's CPU rows live there).  cnt[s] is checked against groups << 16 | rows and the stamps against each other.
//
// The reference runs AFTER the window, not beside it: the persistent kernel keeps every SM's register budget of its
// blocks while it spins, and a second kernel on the same device might never get a slot (the engine never needs one:
// the peer's device runs the server alone).  The snapshots make the comparison the same either way.
//
// Ids are drawn in [-1, n_expert) with repeats on purpose: the same expert on several tokens, experts the peer does
// not own, steps where it owns nothing (groups == 0: cnt[s] == 0 and the flag raised anyway).  With G = 2 half of the
// odd steps that follow a step with entries are FORCED empty: a step without barriers right behind one with rows,
// its doorbell already up, is the case where a fast block counts the empty step while slow blocks still write the
// rows of the full one (Sync::done, one counter a parity).  x is uniform in [-8, 8] with a few 1e4 values and, now and
// then, a whole q8_1 block of them (the block sum passes 65504: q8_1_finite).
//
// Then two endurance windows: (a) the host rings steps 0 and 1, never rings step 2 and raises `abort` after 100 ms:
// the kernel must leave with err == kErrAbort, exit == epoch, flag_out == UINT32_MAX (the error exit raises it so the
// primary's waits pass), the stream drained within 2 s; (b) wait_ns = 200 ms and no doorbell at all: err ==
// kErrTimeoutSeq, flag_out == UINT32_MAX.  Exit code 0 only if every row of every window is equal and both endurance
// windows behave; 2 if peer_fused_shape refuses the layout on this device (its `why` is printed).
//
//   peer_fused_test [device=0] [gu_type=18] [d_type=42] [n_expert=512] [n_slots=64] [windows=20] [seed]
//                   [--primary D]
//
// --primary D: the fake primary is a kernel on device D (the 3080 Ti) that writes x / ids into the mapped pages and
// rings `seq` (the form of k_push in tools/ep_bench.cu: streaming stores, system fence, the flag), then a kernel that
// waits `flag_out` (k_wait_copy's form), so the flag/payload order between two cards without P2P is what is tested,
// not the host's write order.  The host still plans and recomputes the reference.
//
// Knobs read as the engine reads them (peer_fused.hpp): STRATA_PEER_FUSED_MINB, _BLOCKS, _COOP, _WAIT_MS; STRATA_EXP_MODE
// (mode 8 is forced through native_expert_set_mode when it is anything else: the fused server exists only where
// today's path is "fused gate/up + LDS down").
#include "strata/core/peer_fused.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace pf = strata::core::peer_fused;
using strata::kernels::NativeExpertLayout;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace {

constexpr int64_t N = 2560;      // n_embd of Qwen3.8-Flash-Next
constexpr int64_t FF = 640;      // the expert's hidden width
constexpr int K = 10;            // routed experts a row
constexpr int NL = 4;            // layers of the synthetic model: steps a window = NL * G
constexpr uint8_t kSentKernel = 0xA5;   // the bytes of a ymiss row nobody wrote (the kernel's page)
constexpr uint8_t kSentRef = 0x5A;      // the same for the reference page: an unwritten row never compares equal
constexpr double kFlagWaitS = 20.0;     // the host's bound on a step's flag (the engine's layer wait is 20 s too)

using Clock = std::chrono::steady_clock;
double secs_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// the host's half of the mapped protocol: everything written before is visible before what is written after
void host_fence() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
#if defined(_MSC_VER)
    _mm_sfence();
#elif defined(__GNUC__)
    __sync_synchronize();
#endif
}

// fp32 -> fp16 bits, round to nearest even (the scales are normal halves; the two ends are clamped, never reached)
uint16_t f16_bits(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t exp = (int32_t) ((u >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = u & 0x7fffffu;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7bffu);
    uint32_t h = sign | ((uint32_t) exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    uint64_t u64() { return g(); }
    uint32_t below(uint32_t n) { return (uint32_t) (g() % n); }
    float uni(float lo, float hi) { return lo + (hi - lo) * (float) ((g() >> 11) * (1.0 / 9007199254740992.0)); }
};

// values a block of the type holds (the GGUF block the kernels index): 0 = not a type this generator knows
int qk_of(int t) { return t == 18 || t == 21 || t == 22 ? 256 : t == 20 ? 32 : t == 42 ? 64 : 0; }

// one synthetic blob at `b` in the layout: random bytes, then a finite, small fp16 scale at the start of every block
// of every gate, up and down row (the GGUF blocks of IQ3_XXS / IQ3_S / IQ2_S / Q2_0 / IQ4_NL all begin with `d`)
void fill_blob(uint8_t* b, const NativeExpertLayout& L, size_t gu_blk, size_t d_blk, Rng& rng) {
    const size_t n8 = L.bytes / 8;
    for (size_t i = 0; i < n8; ++i) { const uint64_t v = rng.u64(); std::memcpy(b + i * 8, &v, 8); }
    for (size_t i = n8 * 8; i < L.bytes; ++i) b[i] = (uint8_t) rng.u64();
    auto stamp = [&](uint8_t* row, size_t row_bytes, size_t blk) {
        for (size_t o = 0; o + blk <= row_bytes; o += blk) {
            const float d = std::ldexp(1.0f + (float) rng.below(256) / 256.0f, -(6 + (int) rng.below(5)));   // [2^-10, 2^-5)
            const uint16_t h = f16_bits(d);
            std::memcpy(row + o, &h, 2);
        }
    };
    for (int64_t r = 0; r < L.n_ff; ++r) {
        stamp(b + (size_t) r * L.gu_row, L.gu_row, gu_blk);
        stamp(b + L.up_off + (size_t) r * L.gu_row, L.gu_row, gu_blk);
    }
    for (int64_t r = 0; r < L.n_embd; ++r) stamp(b + L.down_off + (size_t) r * L.d_row, L.d_row, d_blk);
}

// the host plan of PeerExperts::launch, field for field (peer_experts.cpp :322-346): distinct owned experts in routing
// order, each one's entries as compact rows.  kind[i] == 2 <=> the peer owns ids[i].
void plan_host(const int32_t* ids, const int32_t* kind, int64_t n, int64_t k, const int32_t* own_l,
               const unsigned long long* slot_ptr, pf::Plan& m, int& groups, int& rows) {
    groups = 0;
    rows = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != 2) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        const int32_t slot = own_l[ids[i]];
        m.ptr[groups] = slot_ptr[slot];
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == 2 && ids[j] == ids[i]) {
                m.dst[rows] = rows;
                m.tok[rows] = (int32_t) (j / k);
                m.out_row[rows] = (int32_t) j;
                ++rows;
            }
        ++groups;
    }
    m.start[groups] = rows;
    m.count[0] = groups;
    m.count[1] = rows;
    m.count[2] = m.count[3] = 0;
}

// one step of a window as the host saw it: what it published and what the kernel wrote back
struct StepRec {
    int s = 0, l = 0, tb = 0, n_tok = 0;
    std::vector<int32_t> ids, kind;       // n_tok * K
    std::vector<float> x;                 // n_tok * N
    std::vector<float> rows_out;          // n_tok * K * N: ymiss rows [tb * K, (tb + n_tok) * K) after the flag
    pf::Plan plan{};
    int groups = 0, rows = 0;
    uint32_t cnt = 0;                     // the kernel's cnt[s]
    unsigned long long st0 = 0, st1 = 0;  // stamp[2s], stamp[2s+1]
    unsigned long long primary_wait_ns = 0;   // --primary: what the primary's wait kernel spent on flag_out
};

// ---------------------------------------------------------------- --primary D: the fake primary on another card
__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// k_push's form (tools/ep_bench.cu): the rows and ids to the mapped pages with streaming stores, the block syncs,
// thread 0 fences at system scope and rings.  The ids are 4-byte stores: tb * K * 4 bytes is not 16-byte aligned
// for an odd tb.
__global__ void k_primary_push(uint4* x_box, const uint4* x_src, int n16, uint32_t* ids_box, const int32_t* ids_src,
                               int n_ids, volatile uint32_t* seq, uint32_t want) {
    for (int i = threadIdx.x; i < n16; i += blockDim.x) __stwt(x_box + i, x_src[i]);
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) __stwt(ids_box + i, (uint32_t) ids_src[i]);
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); *seq = want; }
}

// __nanosleep exists from sm_70; an older arch in the build's list just spins (the test runs on the 2080 Ti / 3080 Ti)
__device__ __forceinline__ void pause_ns(unsigned ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(ns);
#else
    (void) ns;
#endif
}

// k_wait_copy's wait (one poller on the link), bounded: on timeout *perr = want and the host fails the step
__global__ void k_primary_wait(const volatile uint32_t* flag, uint32_t want, unsigned long long wait_ns,
                               volatile uint32_t* perr, unsigned long long* waited) {
    if (threadIdx.x != 0) return;
    const unsigned long long t0 = gtime();
    while (*flag < want) {
        pause_ns(64);
        if (gtime() - t0 > wait_ns) { *perr = want; return; }
    }
    __threadfence_system();
    *waited = gtime() - t0;
}

// ---------------------------------------------------------------- host helpers
int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}

// polls the stream until it is idle or `limit_s` passed (the engine's fused_end / fused_abort do the same)
cudaError_t drain(cudaStream_t s, double limit_s) {
    const auto t0 = Clock::now();
    cudaError_t e = cudaErrorNotReady;
    while ((e = cudaStreamQuery(s)) == cudaErrorNotReady && secs_since(t0) < limit_s)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return e;
}

const char* err_name(uint32_t e) {
    switch (e) {
        case pf::kErrNone: return "none";
        case pf::kErrTimeoutSeq: return "kErrTimeoutSeq";
        case pf::kErrTimeoutReady: return "kErrTimeoutReady";
        case pf::kErrTimeoutBarrier: return "kErrTimeoutBarrier";
        case pf::kErrAbort: return "kErrAbort";
        case pf::kErrPlanOverflow: return "kErrPlanOverflow";
        case pf::kErrBadWindow: return "kErrBadWindow";
        default: return "?";
    }
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// the peer's mapped page (peer_fused.hpp s.2b) and the primary's fake pages, host and device views
struct Pages {
    uint8_t* page_h = nullptr;
    uint8_t* page_d = nullptr;            // the test device's pointer
    volatile uint32_t* flag_out = nullptr;
    volatile uint32_t* abort = nullptr;
    volatile uint32_t* err = nullptr;
    volatile uint32_t* exit = nullptr;
    volatile uint32_t* cnt = nullptr;
    volatile unsigned long long* stamp = nullptr;
    uint32_t* seq_h = nullptr;            // 64 B
    float* x_h = nullptr;                 // kMaxTok x N
    int32_t* ids_h = nullptr;             // kMaxTok x K
    float* ymiss_h = nullptr;             // kMaxTok x K x N (the kernel's)
    float* ymiss_ref_h = nullptr;         // the reference's
    // the test device's pointers
    const uint32_t* seq_d = nullptr;
    const float* x_d = nullptr;
    const int32_t* ids_d = nullptr;
    float* ymiss_d = nullptr;
    float* ymiss_ref_d = nullptr;
    // --primary D: that device's pointers of the same pages, and its own words (err, waited ns)
    uint8_t* pwords_h = nullptr;
    uint4* px_d = nullptr;
    uint32_t* pids_d = nullptr;
    uint32_t* pseq_d = nullptr;
    const uint32_t* pflag_d = nullptr;
    uint32_t* perr_d = nullptr;
    unsigned long long* pwait_d = nullptr;
};

void* dev_ptr(void* host) {
    void* d = nullptr;
    CK(cudaHostGetDevicePointer(&d, host, 0));
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    int pos[6] = {0, 18, 42, 512, 64, 20};
    uint64_t seed = (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
    int primary = -1, npos = 0;
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--primary" && a + 1 < argc) { primary = std::atoi(argv[++a]); continue; }
        if (s == "--help" || s == "-h") {
            std::printf("peer_fused_test [device=0] [gu_type=18] [d_type=42] [n_expert=512] [n_slots=64] [windows=20] [seed] [--primary D]\n");
            return 0;
        }
        if (npos < 6) pos[npos] = std::atoi(argv[a]);
        else if (npos == 6) seed = std::strtoull(argv[a], nullptr, 10);
        ++npos;
    }
    const int device = pos[0], gu_type = pos[1], d_type = pos[2];
    const int64_t n_expert = pos[3];
    const int n_slots = std::max(1, pos[4]), windows = std::max(1, pos[5]);
    if (n_expert <= 0 || n_expert > 65536) { std::printf("n_expert %lld out of range\n", (long long) n_expert); return 1; }

    int nd = 0;
    CK(cudaGetDeviceCount(&nd));
    if (device < 0 || device >= nd || (primary >= 0 && (primary >= nd || primary == device))) {
        std::printf("devices: %d; device %d, primary %d\n", nd, device, primary);
        return 1;
    }
    {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, device));
        std::printf("peer_fused_test: CUDA%d %s sm_%d%d SMs %d | gu %d d %d n_expert %lld n_slots %d windows %d seed %llu",
                    device, p.name, p.major, p.minor, p.multiProcessorCount, gu_type, d_type, (long long) n_expert, n_slots,
                    windows, (unsigned long long) seed);
        if (primary >= 0) {
            cudaDeviceProp q;
            CK(cudaGetDeviceProperties(&q, primary));
            int p2p = 0;
            cudaDeviceCanAccessPeer(&p2p, primary, device);
            std::printf(" | primary CUDA%d %s (P2P %d, not used)", primary, q.name, p2p);
        }
        std::printf("\n");
    }

    // the layout and the kernel instance for it on this device (s.5): mode 8 is the fused server's precondition
    const int em = env_int("STRATA_EXP_MODE", -1);
    if (em != 8) strata::kernels::native_expert_set_mode(8, 0);
    CK(cudaSetDevice(device));
    const NativeExpertLayout L = strata::kernels::native_expert_layout(gu_type, d_type, N, FF);
    const int gu_qk = qk_of(gu_type), d_qk = qk_of(d_type);
    if (gu_qk == 0 || d_qk == 0 || L.bytes == 0) {
        std::printf("the synthetic generator knows gate/up 18 / 21 / 22 and down 42 / 20 only (got %d / %d)\n", gu_type, d_type);
        return 2;
    }
    const size_t gu_blk = strata::kernels::iq_row_bytes(gu_type, gu_qk), d_blk = strata::kernels::iq_row_bytes(d_type, d_qk);
    pf::Shape shape;
    {
        std::string why;
        const int minb = env_int(pf::kEnvMinb, pf::kMinBlocksDefault);
        const int cap = env_int(pf::kEnvBlocks, 0);
        const bool coop = env_int(pf::kEnvCoop, 1) != 0;
        if (!strata::kernels::peer_fused_shape(L, minb, cap, coop, shape, why)) {
            std::printf("peer_fused_shape refused the layout on CUDA%d: %s\n", device, why.c_str());
            return 2;
        }
        std::printf("shape: instance <%d, %d> minb %d, dyn shared %zu B, %d blocks/SM x %d SMs = %d blocks, %s launch\n",
                    shape.gu_type, shape.d_type, shape.minb, shape.dyn_shared, shape.blocks_per_sm, shape.sms, shape.blocks,
                    shape.cooperative ? "cooperative" : "plain");
    }
    const unsigned long long wait_ns = (unsigned long long) std::max(1, env_int(pf::kEnvWaitMs, 20000)) * 1000000ull;

    Rng rng(seed);

    // ---- the slots: n_slots synthetic blobs in one arena, 256-byte strides
    const size_t stride = (L.bytes + 255) / 256 * 256;
    uint8_t* d_arena = nullptr;
    CK(cudaMalloc((void**) &d_arena, stride * (size_t) n_slots));
    std::vector<unsigned long long> slot_ptr((size_t) n_slots);
    {
        std::vector<uint8_t> blob(L.bytes);
        for (int s = 0; s < n_slots; ++s) {
            fill_blob(blob.data(), L, gu_blk, d_blk, rng);
            CK(cudaMemcpy(d_arena + stride * (size_t) s, blob.data(), L.bytes, cudaMemcpyHostToDevice));
            slot_ptr[(size_t) s] = (unsigned long long) (uintptr_t) (d_arena + stride * (size_t) s);
        }
    }
    unsigned long long* d_slot_ptr = nullptr;
    CK(cudaMalloc((void**) &d_slot_ptr, sizeof(unsigned long long) * (size_t) n_slots));
    CK(cudaMemcpy(d_slot_ptr, slot_ptr.data(), sizeof(unsigned long long) * (size_t) n_slots, cudaMemcpyHostToDevice));

    // ---- the ownership table: ~40% of the pairs on a slot, the slots dealt from a shuffled order (distinct while
    // n_slots lasts, then the order again: a slot read by two experts is harmless to the comparison)
    std::vector<int32_t> own((size_t) NL * (size_t) n_expert, -1);
    std::vector<std::vector<int32_t>> unowned((size_t) NL);
    {
        std::vector<int32_t> deal((size_t) n_slots);
        for (int s = 0; s < n_slots; ++s) deal[(size_t) s] = s;
        std::shuffle(deal.begin(), deal.end(), rng.g);
        size_t next = 0;
        int owned_pairs = 0;
        for (int l = 0; l < NL; ++l)
            for (int64_t e = 0; e < n_expert; ++e) {
                if (rng.below(100) < 40) {
                    own[(size_t) l * (size_t) n_expert + (size_t) e] = deal[next++ % (size_t) n_slots];
                    ++owned_pairs;
                } else {
                    unowned[(size_t) l].push_back((int32_t) e);
                }
            }
        std::printf("ownership: %d of %lld pairs on the peer (%d slots of %zu B)\n", owned_pairs, (long long) (NL * n_expert),
                    n_slots, L.bytes);
    }
    int32_t* d_own = nullptr;
    CK(cudaMalloc((void**) &d_own, own.size() * 4));
    CK(cudaMemcpy(d_own, own.data(), own.size() * 4, cudaMemcpyHostToDevice));

    // ---- the kernel's VRAM (s.3) and the reference's
    pf::Plan* d_plan = nullptr;
    void* d_q8 = nullptr;
    uint8_t* d_scratch = nullptr;
    float* d_out = nullptr;
    pf::Sync* d_sync = nullptr;
    const size_t scratch_bytes = strata::kernels::native_expert_scratch_bytes(pf::kCap, FF);
    CK(cudaMalloc((void**) &d_plan, 2 * sizeof(pf::Plan)));
    CK(cudaMalloc(&d_q8, pf::q8_bytes(pf::kMaxTok, N)));
    CK(cudaMalloc((void**) &d_scratch, scratch_bytes));
    CK(cudaMalloc((void**) &d_out, (size_t) pf::kCap * N * sizeof(float)));
    CK(cudaMalloc((void**) &d_sync, sizeof(pf::Sync)));
    CK(cudaMemset(d_plan, 0, 2 * sizeof(pf::Plan)));
    CK(cudaMemset(d_sync, 0, sizeof(pf::Sync)));
    float* d_x_ref = nullptr;
    void* d_q8_ref = nullptr;
    uint8_t* d_scratch_ref = nullptr;
    float* d_out_ref = nullptr;
    pf::Plan* d_plan_ref = nullptr;
    pf::Plan* h_plan_ref = nullptr;
    CK(cudaMalloc((void**) &d_x_ref, (size_t) pf::kMaxTok * N * sizeof(float)));
    CK(cudaMalloc(&d_q8_ref, pf::q8_bytes(pf::kMaxTok, N)));
    CK(cudaMalloc((void**) &d_scratch_ref, scratch_bytes));
    CK(cudaMalloc((void**) &d_out_ref, (size_t) pf::kCap * N * sizeof(float)));
    CK(cudaMalloc((void**) &d_plan_ref, sizeof(pf::Plan)));
    CK(cudaMallocHost((void**) &h_plan_ref, sizeof(pf::Plan)));

    // ---- the mapped pages (s.2): Portable | Mapped, the device pointers taken under the device that uses them
    Pages P;
    const size_t x_bytes = (size_t) pf::kMaxTok * N * sizeof(float), ids_bytes = (size_t) pf::kMaxTok * K * sizeof(int32_t);
    const size_t ymiss_bytes = (size_t) pf::kMaxTok * K * N * sizeof(float);
    const unsigned hflags = cudaHostAllocPortable | cudaHostAllocMapped;
    CK(cudaHostAlloc((void**) &P.page_h, pf::kPageBytes, hflags));
    CK(cudaHostAlloc((void**) &P.seq_h, 64, hflags));
    CK(cudaHostAlloc((void**) &P.x_h, x_bytes, hflags));
    CK(cudaHostAlloc((void**) &P.ids_h, ids_bytes, hflags));
    CK(cudaHostAlloc((void**) &P.ymiss_h, ymiss_bytes, hflags));
    CK(cudaHostAlloc((void**) &P.ymiss_ref_h, ymiss_bytes, hflags));
    CK(cudaHostAlloc((void**) &P.pwords_h, 64, hflags));
    std::memset(P.page_h, 0, pf::kPageBytes);
    std::memset(P.seq_h, 0, 64);
    std::memset(P.x_h, 0, x_bytes);
    std::memset(P.ids_h, 0, ids_bytes);
    std::memset(P.ymiss_h, kSentKernel, ymiss_bytes);
    std::memset(P.ymiss_ref_h, kSentRef, ymiss_bytes);
    std::memset(P.pwords_h, 0, 64);
    P.flag_out = (volatile uint32_t*) (P.page_h + pf::kOffFlagOut);
    P.abort = (volatile uint32_t*) (P.page_h + pf::kOffAbort);
    P.err = (volatile uint32_t*) (P.page_h + pf::kOffErr);
    P.exit = (volatile uint32_t*) (P.page_h + pf::kOffExit);
    P.cnt = (volatile uint32_t*) (P.page_h + pf::kOffCnt);
    P.stamp = (volatile unsigned long long*) (P.page_h + pf::kOffStamp);
    P.page_d = (uint8_t*) dev_ptr(P.page_h);
    P.seq_d = (const uint32_t*) dev_ptr(P.seq_h);
    P.x_d = (const float*) dev_ptr(P.x_h);
    P.ids_d = (const int32_t*) dev_ptr(P.ids_h);
    P.ymiss_d = (float*) dev_ptr(P.ymiss_h);
    P.ymiss_ref_d = (float*) dev_ptr(P.ymiss_ref_h);
    volatile uint32_t* perr_h = (volatile uint32_t*) P.pwords_h;
    volatile unsigned long long* pwait_h = (volatile unsigned long long*) (P.pwords_h + 8);

    cudaStream_t s_kernel = nullptr, s_ref = nullptr, s_primary = nullptr;
    CK(cudaStreamCreateWithFlags(&s_kernel, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s_ref, cudaStreamNonBlocking));

    // --primary D: that device's views of the pages, its staging of a step's rows and ids, its stream
    uint4* d_px = nullptr;
    int32_t* d_pids = nullptr;
    float* h_px = nullptr;
    int32_t* h_pids = nullptr;
    if (primary >= 0) {
        CK(cudaSetDevice(primary));
        P.px_d = (uint4*) dev_ptr(P.x_h);
        P.pids_d = (uint32_t*) dev_ptr(P.ids_h);
        P.pseq_d = (uint32_t*) dev_ptr(P.seq_h);
        P.pflag_d = (const uint32_t*) (((uint8_t*) dev_ptr(P.page_h)) + pf::kOffFlagOut);
        P.perr_d = (uint32_t*) dev_ptr(P.pwords_h);
        P.pwait_d = (unsigned long long*) (((uint8_t*) dev_ptr(P.pwords_h)) + 8);
        CK(cudaMalloc((void**) &d_px, x_bytes));
        CK(cudaMalloc((void**) &d_pids, ids_bytes));
        CK(cudaMallocHost((void**) &h_px, x_bytes));
        CK(cudaMallocHost((void**) &h_pids, ids_bytes));
        CK(cudaStreamCreateWithFlags(&s_primary, cudaStreamNonBlocking));
        CK(cudaSetDevice(device));
    }

    // ---- the window argument that does not change between windows
    pf::Window base{};
    base.lb = 0;
    base.k = K;
    base.n_embd = N;
    base.n_ff = FF;
    base.n_expert = n_expert;
    base.nblocks = shape.blocks;
    base.wait_ns = wait_ns;
    base.seq = P.seq_d;
    base.x = P.x_d;
    base.ids = P.ids_d;
    base.ymiss = P.ymiss_d;
    base.flag_out = (uint32_t*) (P.page_d + pf::kOffFlagOut);
    base.abort = (const uint32_t*) (P.page_d + pf::kOffAbort);
    base.err = (uint32_t*) (P.page_d + pf::kOffErr);
    base.exit = (uint32_t*) (P.page_d + pf::kOffExit);
    base.cnt = (uint32_t*) (P.page_d + pf::kOffCnt);
    base.stamp = (unsigned long long*) (P.page_d + pf::kOffStamp);
    base.own = d_own;
    base.slot_ptr = d_slot_ptr;
    base.plan = d_plan;
    base.q8 = d_q8;
    base.hq = d_scratch + pf::hq_offset(pf::kCap, FF);
    base.out = d_out;
    base.sync = d_sync;
    base.L = L;

    uint32_t epoch = 0;
    bool all_ok = true;
    int64_t peer_entries_total = 0;

    // ---- fused_begin's part: the words, the sentinel, d_sync, the launch, the WDDM push
    auto begin_window = [&](pf::Window& w) -> bool {
        const cudaError_t q = cudaStreamQuery(s_kernel);
        if (q != cudaSuccess) {
            std::printf("  the previous window's kernel is still on the stream (%s)\n", cudaGetErrorString(q));
            return false;
        }
        w.epoch = ++epoch;
        *P.flag_out = 0; *P.abort = 0; *P.err = 0; *P.exit = 0;
        for (int s = 0; s < pf::kMaxSteps; ++s) P.cnt[s] = 0;
        for (int s = 0; s < 2 * pf::kMaxSteps; ++s) P.stamp[s] = 0;
        *(volatile uint32_t*) P.seq_h = 0;
        *perr_h = 0;
        host_fence();
        CK(cudaMemsetAsync(d_sync, 0, sizeof(pf::Sync), s_kernel));
        std::string err;
        if (!strata::kernels::peer_fused_launch(w, shape, s_kernel, err)) {
            std::printf("  peer_fused_launch: %s\n", err.c_str());
            return false;
        }
        (void) cudaStreamQuery(s_kernel);   // WDDM: submit now
        return true;
    };

    // the host's spin on flag_out >= want with the engine's bound, the stream flushed every ~2 ms (hook.flush)
    auto wait_flag = [&](uint32_t want, std::string& why) -> bool {
        const auto t0 = Clock::now();
        auto last_flush = t0;
        while (*P.flag_out < want) {
            if (*P.err != 0) { why = std::string("the kernel reported ") + err_name(*P.err); return false; }
            if (secs_since(t0) > kFlagWaitS) { why = "no flag within " + std::to_string((int) kFlagWaitS) + " s"; return false; }
            if (secs_since(last_flush) > 0.002) { (void) cudaStreamQuery(s_kernel); last_flush = Clock::now(); }
        }
        host_fence();
        return true;
    };

    // after an error or a stuck window: abort, drain within 5 s, so the process never exits over a live kernel
    auto abort_window = [&]() {
        host_fence();
        *P.abort = 1;
        host_fence();
        const cudaError_t e = drain(s_kernel, 5.0);
        if (e == cudaErrorNotReady) std::printf("  the kernel did not leave within 5 s of abort: leaving it to the driver\n");
        else if (e != cudaSuccess) std::printf("  stream error after abort: %s\n", cudaGetErrorString(e));
    };

    // ---- publishes one step: the host writes x / ids (or the primary's kernel does), fences, rings seq = s + 1.  With
    // --primary the push alone goes on s_primary here; the wait on flag_out is queued by primary_wait, so the stream
    // carries the graph's order push(0), push(1), wait(0), push(2), wait(1), ... (G = 2) and not push / wait a step.
    // The step's ymiss rows get the sentinel first: the kernel must not find a stale row of the step before (G steps
    // back, the same group) under what it writes, nor must a non-owned entry keep one.
    auto publish = [&](const StepRec& r, uint32_t want) {
        const int n = r.n_tok * K;
        std::memset(P.ymiss_h + (size_t) r.tb * K * N, kSentKernel, (size_t) n * N * sizeof(float));
        if (primary < 0) {
            std::memcpy(P.x_h + (size_t) r.tb * N, r.x.data(), (size_t) r.n_tok * N * sizeof(float));
            std::memcpy(P.ids_h + (size_t) r.tb * K, r.ids.data(), (size_t) n * sizeof(int32_t));
            host_fence();
            *(volatile uint32_t*) P.seq_h = want;
            host_fence();
            return;
        }
        // --primary: the rows and ids go to D's VRAM, D's kernel moves them into the mapped pages and rings.  The
        // staging buffers are per step (G of them in flight): the slice of the group's rows.
        float* hx = h_px + (size_t) r.tb * N;
        int32_t* hid = h_pids + (size_t) r.tb * K;
        std::memcpy(hx, r.x.data(), (size_t) r.n_tok * N * sizeof(float));
        std::memcpy(hid, r.ids.data(), (size_t) n * sizeof(int32_t));
        host_fence();
        CK(cudaSetDevice(primary));
        CK(cudaMemcpyAsync(d_px + (size_t) r.tb * N / 4, hx, (size_t) r.n_tok * N * sizeof(float), cudaMemcpyHostToDevice, s_primary));
        CK(cudaMemcpyAsync(d_pids + (size_t) r.tb * K, hid, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice, s_primary));
        k_primary_push<<<1, 256, 0, s_primary>>>(P.px_d + (size_t) r.tb * N / 4, d_px + (size_t) r.tb * N / 4, (int) (r.n_tok * N / 4),
                                                 P.pids_d + (size_t) r.tb * K, d_pids + (size_t) r.tb * K, n, P.pseq_d, want);
        CK(cudaGetLastError());
        (void) cudaStreamQuery(s_primary);   // WDDM: submit now
        CK(cudaSetDevice(device));
    };
    // --primary: the primary's wait on flag_out >= want, queued after the pushes already on its stream (post(l, grp))
    auto primary_wait = [&](uint32_t want) {
        CK(cudaSetDevice(primary));
        k_primary_wait<<<1, 32, 0, s_primary>>>(P.pflag_d, want, wait_ns, P.perr_d, P.pwait_d);
        CK(cudaGetLastError());
        (void) cudaStreamQuery(s_primary);
        CK(cudaSetDevice(device));
    };

    // ---- draws a step's ids and rows: repeats on purpose, unowned experts, -1 holes, now and then (or when asked:
    // force_empty) nothing owned
    auto draw_step = [&](StepRec& r, bool force_empty) {
        const int n = r.n_tok * K;
        r.ids.assign((size_t) n, -1);
        r.kind.assign((size_t) n, 0);
        const int32_t* own_l = own.data() + (size_t) r.l * (size_t) n_expert;
        const bool nothing_owned = !unowned[(size_t) r.l].empty() && (force_empty || rng.below(6) == 0);
        for (int i = 0; i < n; ++i) {
            const uint32_t d = rng.below(16);
            int32_t e;
            if (d == 0) e = -1;
            else if (nothing_owned) e = unowned[(size_t) r.l][rng.below((uint32_t) unowned[(size_t) r.l].size())];
            else if (d <= 5 && i > 0) e = r.ids[rng.below((uint32_t) i)];   // the same expert again, this or another token
            else e = (int32_t) rng.below((uint32_t) n_expert);
            r.ids[(size_t) i] = e;
            r.kind[(size_t) i] = e >= 0 && own_l[e] >= 0 ? 2 : (e >= 0 ? 0 : -1);
        }
        r.x.resize((size_t) r.n_tok * N);
        for (int t = 0; t < r.n_tok; ++t) {
            float* row = r.x.data() + (size_t) t * N;
            for (int64_t c = 0; c < N; ++c) {
                const uint32_t d = rng.below(4096);
                row[c] = d == 0 ? (rng.below(2) ? 1e4f : -1e4f) : d == 1 ? (rng.below(2) ? 3e7f : -3e7f) : rng.uni(-8.0f, 8.0f);
            }
            if (rng.below(16) == 0) {   // a whole q8_1 block of 1e4: the block sum passes 65504 (q8_1_finite's clamp)
                const int64_t b = rng.below((uint32_t) (N / 32));
                const float v = rng.below(2) ? 1e4f : -1e4f;
                for (int64_t c = 0; c < 32; ++c) row[b * 32 + c] = v;
            }
        }
        plan_host(r.ids.data(), r.kind.data(), n, K, own_l, slot_ptr.data(), r.plan, r.groups, r.rows);
    };

    // ---- the reference of one step on s_ref (today's path, peer_experts.cpp :361-382), compared to the snapshot
    auto check_step = [&](const StepRec& r, int win) -> bool {
        const int n = r.n_tok * K;
        bool ok = true;
        const uint32_t expect = ((uint32_t) r.groups << 16) | (uint32_t) r.rows;
        if (r.cnt != expect) {
            std::printf("  FAIL finestra %d passo %d (strato %d): cnt %ux%u, host %dx%d\n", win, r.s, r.l, r.cnt >> 16, r.cnt & 0xffffu,
                        r.groups, r.rows);
            ok = false;
        }
        if (r.st1 < r.st0 || r.st0 == 0) {
            std::printf("  FAIL finestra %d passo %d: stamp %llu -> %llu\n", win, r.s, r.st0, r.st1);
            ok = false;
        }
        std::memset(P.ymiss_ref_h + (size_t) r.tb * K * N, kSentRef, (size_t) n * N * sizeof(float));
        if (r.groups > 0) {
            std::memcpy(h_plan_ref, &r.plan, sizeof(pf::Plan));
            CK(cudaMemcpyAsync(d_x_ref, r.x.data(), (size_t) r.n_tok * N * sizeof(float), cudaMemcpyHostToDevice, s_ref));
            CK(cudaMemcpyAsync(d_plan_ref, h_plan_ref, sizeof(pf::Plan), cudaMemcpyHostToDevice, s_ref));
            strata::kernels::quantize_q8_1_rows(d_x_ref, r.n_tok, N, d_q8_ref, s_ref);
            strata::kernels::native_expert_grouped(L, d_plan_ref->ptr, d_plan_ref->start, d_plan_ref->count, d_plan_ref->dst,
                                                   d_plan_ref->tok, r.groups, r.rows, d_q8_ref, d_scratch_ref, d_out_ref, s_ref);
            strata::kernels::scatter_rows_f32(d_out_ref, P.ymiss_ref_d + (size_t) r.tb * K * N, d_plan_ref->out_row, r.rows, N, s_ref);
            CK(cudaStreamSynchronize(s_ref));
            host_fence();
        }
        // every entry of the step: owned -> the reference's row; not owned -> the sentinel untouched
        std::vector<uint8_t> sent((size_t) N * sizeof(float), kSentKernel);
        for (int j = 0; j < n && ok; ++j) {
            const float* got = r.rows_out.data() + (size_t) j * N;
            const uint8_t* want = r.kind[(size_t) j] == 2 ? (const uint8_t*) (P.ymiss_ref_h + ((size_t) r.tb * K + (size_t) j) * N)
                                                           : sent.data();
            if (std::memcmp(got, want, (size_t) N * sizeof(float)) == 0) continue;
            int64_t c = 0;
            while (c < N && std::memcmp((const uint8_t*) got + c * 4, want + c * 4, 4) == 0) ++c;
            uint32_t a = 0, b = 0;
            std::memcpy(&a, (const uint8_t*) got + c * 4, 4);
            std::memcpy(&b, want + c * 4, 4);
            int cr = -1;
            for (int q = 0; q < r.rows; ++q) if (r.plan.out_row[q] == j) { cr = q; break; }
            std::printf("  FAIL finestra %d passo %d (strato %d, tb %d): voce %d (riga compatta %d, token %d, esperto %d, %s) colonna %lld: "
                        "kernel %08x, riferimento %08x\n", win, r.s, r.l, r.tb, j, cr, j / K, r.ids[(size_t) j],
                        r.kind[(size_t) j] == 2 ? "del peer" : "non del peer: sentinella attesa", (long long) c, a, b);
            ok = false;
        }
        return ok;
    };

    // ================================================================ the windows
    for (int win = 0; win < windows && all_ok; ++win) {
        const int T = 1 + (int) rng.below((uint32_t) pf::kMaxTok);
        const int G = T >= 2 && rng.below(2) ? 2 : 1;
        const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
        const int steps = NL * G;
        pf::Window w = base;
        w.steps = steps; w.G = G; w.T = T;
        w.gtb[0] = gtb[0]; w.gtb[1] = gtb[1]; w.gte[0] = gte[0]; w.gte[1] = gte[1];
        std::vector<StepRec> recs((size_t) steps);
        for (int s = 0; s < steps; ++s) {
            StepRec& r = recs[(size_t) s];
            r.s = s; r.l = s / G;
            r.tb = gtb[s % G];
            r.n_tok = gte[s % G] - gtb[s % G];
            // G = 2: half of the group-1 steps behind a group-0 step with entries own nothing - the step without
            // barriers whose doorbell is already up while the kernel still writes the rows of the step before
            const bool force_empty = G == 2 && (s % G) == 1 && recs[(size_t) s - 1].rows > 0 && rng.below(2) == 0;
            draw_step(r, force_empty);
        }
        if (!begin_window(w)) { all_ok = false; break; }
        bool win_ok = true;
        std::string why;
        // the graph's order: the first G steps are rung before any flag is waited for; flag(s) seen -> ring s + G
        for (int s = 0; s < G && s < steps; ++s) publish(recs[(size_t) s], (uint32_t) s + 1);
        for (int s = 0; s < steps && win_ok; ++s) {
            StepRec& r = recs[(size_t) s];
            const int n = r.n_tok * K;
            if (primary >= 0) {   // the primary's wait kernel sees the flag first; the host waits for its stream
                primary_wait((uint32_t) s + 1);
                CK(cudaSetDevice(primary));
                const cudaError_t e = drain(s_primary, kFlagWaitS + 2.0);
                CK(cudaSetDevice(device));
                if (e != cudaSuccess) { why = std::string("the primary's stream: ") + cudaGetErrorString(e); win_ok = false; break; }
                host_fence();
                if (*perr_h != 0) { why = "the primary's wait kernel timed out on flag_out"; win_ok = false; break; }
                r.primary_wait_ns = *pwait_h;
            }
            if (!wait_flag((uint32_t) s + 1, why)) { win_ok = false; break; }
            r.rows_out.resize((size_t) n * N);
            std::memcpy(r.rows_out.data(), P.ymiss_h + (size_t) r.tb * K * N, (size_t) n * N * sizeof(float));
            r.cnt = P.cnt[s];
            r.st0 = P.stamp[2 * s];
            r.st1 = P.stamp[2 * s + 1];
            if (s + G < steps) publish(recs[(size_t) s + G], (uint32_t) (s + G) + 1);   // pre(l + 1, grp): the same group's rows again
        }
        if (!win_ok) {
            std::printf("  FAIL finestra %d: %s (err %s, flag_out %u)\n", win, why.c_str(), err_name(*P.err), *P.flag_out);
            abort_window();
            all_ok = false;
            break;
        }
        // fused_end's part: the kernel has left, no error, the epoch written
        {
            const cudaError_t e = drain(s_kernel, 5.0);
            if (e != cudaSuccess) {
                std::printf("  FAIL finestra %d: the kernel did not leave after the last flag (%s)\n", win, cudaGetErrorString(e));
                abort_window();
                all_ok = false;
                break;
            }
            host_fence();
            if (*P.err != 0) { std::printf("  FAIL finestra %d: err %s after the window\n", win, err_name(*P.err)); all_ok = false; break; }
            if (*P.exit != w.epoch) { std::printf("  FAIL finestra %d: exit %u, epoch %u\n", win, *P.exit, w.epoch); all_ok = false; break; }
        }
        // the reference, step by step, and the comparison
        std::vector<double> spans, pwaits;
        int64_t entries = 0;
        for (int s = 0; s < steps; ++s) {
            const StepRec& r = recs[(size_t) s];
            entries += r.rows;
            spans.push_back((double) (r.st1 - r.st0) * 1e-3);
            if (primary >= 0) pwaits.push_back((double) r.primary_wait_ns * 1e-3);
            if (!check_step(r, win)) { win_ok = false; break; }
        }
        peer_entries_total += entries;
        std::printf("finestra %3d: T %2d G %d passi %2d voci del peer %4lld  flag visto -> alzato p50 %8.1f us", win, T, G, steps,
                    (long long) entries, median(spans));
        if (primary >= 0) std::printf("  attesa padrona p50 %8.1f us", median(pwaits));
        std::printf("  %s\n", win_ok ? "PASS" : "FAIL");
        if (!win_ok) all_ok = false;
    }

    // ================================================================ endurance (s.9)
    // (a) steps 0 and 1 served, step 2 never rung, abort after 100 ms: kErrAbort, exit == epoch, the stream drained in 2 s
    if (all_ok) {
        const int T = 1 + (int) rng.below((uint32_t) pf::kMaxTok);
        pf::Window w = base;
        w.steps = NL; w.G = 1; w.T = T;
        w.gtb[0] = 0; w.gtb[1] = T; w.gte[0] = T; w.gte[1] = T;
        bool ok = begin_window(w);
        std::string why;
        for (int s = 0; s < 2 && ok; ++s) {
            StepRec r;
            r.s = s; r.l = s; r.tb = 0; r.n_tok = T;
            draw_step(r, false);
            std::memset(P.ymiss_h, kSentKernel, (size_t) T * K * N * sizeof(float));
            // the host doorbell even with --primary: the primary's wait kernel must not sit on a flag that never comes
            std::memcpy(P.x_h, r.x.data(), (size_t) T * N * sizeof(float));
            std::memcpy(P.ids_h, r.ids.data(), (size_t) T * K * sizeof(int32_t));
            host_fence();
            *(volatile uint32_t*) P.seq_h = (uint32_t) s + 1;
            host_fence();
            if (!wait_flag((uint32_t) s + 1, why)) { std::printf("  tenuta (a): passo %d: %s\n", s, why.c_str()); ok = false; }
        }
        if (ok) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            host_fence();
            const bool still = *P.err == 0 && *P.flag_out == 2 && cudaStreamQuery(s_kernel) == cudaErrorNotReady;
            const auto t0 = Clock::now();
            host_fence();
            *P.abort = 1;
            host_fence();
            const cudaError_t e = drain(s_kernel, 2.0);
            const double ms = secs_since(t0) * 1e3;
            host_fence();
            // the error exit raises flag_out to UINT32_MAX (the primary's waits must pass): 2 before, MAX after
            const bool good = still && e == cudaSuccess && *P.err == pf::kErrAbort && *P.exit == w.epoch && *P.flag_out == UINT32_MAX;
            std::printf("tenuta (a) abort al passo 2: %s | prima dell'abort: in attesa %s | uscita in %.1f ms (%s), err %s, exit %u (epoch %u), "
                        "flag_out %u (atteso %u)\n", good ? "PASS" : "FAIL", still ? "si" : "no", ms, cudaGetErrorString(e), err_name(*P.err),
                        *P.exit, w.epoch, *P.flag_out, UINT32_MAX);
            if (!good) { if (e == cudaErrorNotReady) abort_window(); ok = false; }
        } else {
            abort_window();
        }
        if (!ok) all_ok = false;
    }
    // (b) wait_ns = 200 ms, no doorbell at all: kErrTimeoutSeq, exit == epoch, the stream drained
    if (all_ok) {
        pf::Window w = base;
        w.steps = NL; w.G = 1; w.T = 1;
        w.gtb[0] = 0; w.gtb[1] = 1; w.gte[0] = 1; w.gte[1] = 1;
        w.wait_ns = 200ull * 1000000ull;
        bool ok = begin_window(w);
        if (ok) {
            const auto t0 = Clock::now();
            const cudaError_t e = drain(s_kernel, 2.0);
            const double ms = secs_since(t0) * 1e3;
            host_fence();
            const bool good = e == cudaSuccess && *P.err == pf::kErrTimeoutSeq && *P.exit == w.epoch && *P.flag_out == UINT32_MAX;
            std::printf("tenuta (b) wait_ns 200 ms senza doorbell: %s | uscita in %.1f ms (%s), err %s, exit %u (epoch %u), flag_out %u (atteso %u)\n",
                        good ? "PASS" : "FAIL", ms, cudaGetErrorString(e), err_name(*P.err), *P.exit, w.epoch, *P.flag_out, UINT32_MAX);
            if (!good) { if (e == cudaErrorNotReady) abort_window(); ok = false; }
        }
        if (!ok) all_ok = false;
    }

    // ---- the end: never destroy a stream under a live kernel (the aborts above drained it, or said they could not)
    if (cudaStreamQuery(s_kernel) == cudaErrorNotReady) {
        std::printf("the kernel is still on the stream at exit: %s\n", all_ok ? "FAIL" : "FAIL (see above)");
        std::fflush(stdout);
        std::_Exit(1);
    }
    CK(cudaStreamDestroy(s_kernel));
    CK(cudaStreamDestroy(s_ref));
    if (primary >= 0) {
        CK(cudaSetDevice(primary));
        (void) drain(s_primary, 2.0);
        CK(cudaStreamDestroy(s_primary));
        cudaFree(d_px); cudaFree(d_pids);
        cudaFreeHost(h_px); cudaFreeHost(h_pids);
        CK(cudaSetDevice(device));
    }
    cudaFree(d_arena); cudaFree(d_slot_ptr); cudaFree(d_own);
    cudaFree(d_plan); cudaFree(d_q8); cudaFree(d_scratch); cudaFree(d_out); cudaFree(d_sync);
    cudaFree(d_x_ref); cudaFree(d_q8_ref); cudaFree(d_scratch_ref); cudaFree(d_out_ref); cudaFree(d_plan_ref);
    cudaFreeHost(h_plan_ref);
    cudaFreeHost(P.page_h); cudaFreeHost(P.seq_h); cudaFreeHost(P.x_h); cudaFreeHost(P.ids_h);
    cudaFreeHost(P.ymiss_h); cudaFreeHost(P.ymiss_ref_h); cudaFreeHost(P.pwords_h);
    std::printf("%s: %d finestre, %lld voci del peer, seme %llu\n", all_ok ? "PASS" : "FAIL", windows, (long long) peer_entries_total,
                (unsigned long long) seed);
    return all_ok ? 0 : 1;
}
