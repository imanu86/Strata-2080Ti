// tools/peer_fused_test.cu - PEER_FUSED (docs/PEER_FUSED.md s.8, lab/upsync-v0.2.9): the persistent expert server of
// STRATA_PEER_FUSED=1 (peer_fused_kernel, launched through peer_fused_launch) against today's peer path, bit for bit.
//
// No model, no engine: a synthetic model of `nl` layers whose expert layers may MIX types (as Qwen3.8-Flash-Next
// GSQ-RCO IQ3_XXS does: five gate/up types and two down types over 48 layers), `n_slots` synthetic expert blobs a
// layer with a VALID structure (fp16 block scales, finite and small; random code bytes - the kernels only index 8- or
// 9-bit lookups and dp4a with them), an ownership table layer x expert with ~40% of the pairs on a slot of that
// layer, and the verifier's mapped pages faked on the host: `seq` (the doorbell), `x` (kMaxTok x n_embd fp32), `ids`
// (kMaxTok x k int32), `ymiss` (kMaxTok x k x n_embd fp32).  Per window the host plays the primary IN THE ORDER OF ITS
// GRAPH (verify.cpp record_window: pre(l, 0), pre(l, 1), post(l, 0), pre(l + 1, 0), post(l, 1), ...): it writes the
// rows and ids of the first G steps, fences, rings them, then for every step s spins on `flag_out >= s + 1`,
// snapshots the step's rows of `ymiss` and only then rings step s + G - so with G = 2 the doorbell of s + 1 is up
// while the kernel still works on s, as in the engine.  When the kernel has left (cudaStreamSynchronize), every step
// is recomputed with the path peer_experts.cpp runs today, on the same device and a second stream - the host plan of
// PeerExperts::launch (kind == 2 <=> owned, groups in order of first appearance, Meta ptr/start/dst/tok/out_row),
// cudaMemcpyAsync of the rows, quantize_q8_1_rows, native_expert_grouped with THE LAYER'S layout (mode 8: fused
// gate/up + LDS down), scatter_rows_f32 into a separate mapped page `ymiss_ref` - and the rows are memcmp'd: entry j
// of the step at (gtb * k + j) * n_embd.  Entries the peer does not own must keep the sentinel the host wrote before
// the ring (the pool's CPU rows live there).  cnt[s] is checked against groups << 16 | rows and the stamps against
// each other.
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
// windows behave; 2 if peer_fused_shape refuses the pack on this device (its `why` is printed).
//
//   peer_fused_test [device=0] [gu_type=0] [d_type=0] [n_expert=512] [n_slots=16] [windows=20] [seed]
//                   [--primary D] [--second D2] [--skip] [--stage]
//
// --skip (docs/PEER_FUSED2.md s.11.1, STRATA_PEER2_SKIP): the host (or the primary's kernel) plays the primary's node
// of that section: a step on which a server has no entry is NOT waited for on that server's flag (the empty word of
// its page, kOffEmpty, says so, as PeerExperts::launch writes it; with --primary the wait kernel polls flag OR word as
// wait_flag_ge_any does), so the next doorbell goes up while that kernel may still be on the empty step - several
// steps ahead over a run of empty ones.  Both kernels get Window::ahead = 1 (block 0's wait on done[s & 1] before it
// rewrites the plan) and Window::empty (their page's words: the ids first, then the word).  cnt[s] and the stamps are
// read after the window (the kernel serves every step, skipped or not) and the rows are compared as before: the point
// is that nothing of a later step lands in an earlier one.  --skip-delay US (default 200, 0 = off) sets
// Window::test_delay_ns: block 0 waits that long after a doorbell before it reads the ids, so the next doorbell of a
// skipped step lands FIRST every time - the engine's race on the slow link (1fe8d044: `plan mismatch, device 2 x 2,
// host 0 x 0`), which this bench's fast kernel never lost by itself, made certain.
//
// --stage (s.11.2, STRATA_PEER2_STAGE; needs --second): every window is served TWICE on the same steps - first as before,
// then with the second server reading the step's rows already quantized: the host quantizes every step's rows on the
// first device BEFORE the window's kernels go up (quantize_q8_1_rows: the bytes the reference uses; once the kernels
// are up that device's SMs are the first server's, and a quantize queued beside it would run only when the kernel
// leaves - 1fe8d044 did that and the first server timed out on its doorbell), and during the window copies them into
// the second server's xq page and raises its ring (kOffXqSeq) before the doorbell (the engine's publish node does it
// right after, on the primary; the kernel waits the ring only on a step with entries).  Both runs are compared bit for
// bit, and for every step the second server's span is printed without and with the stage: what the fewer bytes over
// its link are worth.  With --primary the q8_1 rows still come from the host (the fake primary's push kernel does not
// quantize).
//
// --second D2 (docs/PEER_FUSED2.md, two expert servers): a SECOND persistent kernel on device D2 beside the one on
// `device`, as the engine runs it with --peer-device A,B: its own mapped page, ownership table, slot arena, plan,
// sync words and stream; the same doorbell page, x, ids and ymiss rows.  The pairs are dealt to the two servers
// DISJOINTLY (~40% to the first, ~30% of the rest to the second: kind 2 and kind 3, the pool's codes), the host
// waits both flags a step, and every row is compared bit for bit with the reference computed ON THE FIRST DEVICE
// (the arenas hold the same synthetic blobs slot for slot on both cards, so the reference of a second-server entry
// is today's path on the first card over the same bytes: the claim that an expert gives the same bits on either
// card is what this checks, a 2080 Ti against a 3060 for instance).  cnt and stamps are checked on both pages; the
// endurance windows abort and time out both kernels.  Any two devices will do.
//
// gu_type 0 (the default): the MIXED model - the ten (gate/up, down) pairs of Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS's
// expert layers (tools/iq_pack.py native_experts.txt), one layer each in their order of first appearance: (17, 20),
// (16, 42), (17, 42), (21, 42), (22, 42), (21, 20), (22, 20), (16, 20), (18, 20), (18, 42) - so one window crosses
// every body of the kernel and every type change between two consecutive steps.  A gu_type > 0 is a uniform model of
// 4 layers of (gu_type, d_type), as before.  n_slots is per layer.
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
constexpr uint8_t kSentKernel = 0xA5;   // the bytes of a ymiss row nobody wrote (the kernel's page)
constexpr uint8_t kSentRef = 0x5A;      // the same for the reference page: an unwritten row never compares equal
constexpr double kFlagWaitS = 20.0;     // the host's bound on a step's flag (the engine's layer wait is 20 s too)

// the mixed model: the pack's ten (gate/up, down) pairs in their order of first appearance (layers 0, 1, 5, 8, 12,
// 28, 29, 31, 35, 36 of Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS)
constexpr int kMixed[][2] = { {17, 20}, {16, 42}, {17, 42}, {21, 42}, {22, 42}, {21, 20}, {22, 20}, {16, 20}, {18, 20}, {18, 42} };
constexpr int kMixedLayers = (int) (sizeof kMixed / sizeof kMixed[0]);

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
int qk_of(int t) { return t == 16 || t == 17 || t == 18 || t == 21 || t == 22 ? 256 : t == 20 ? 32 : t == 42 ? 64 : 0; }

// one layer of the synthetic model: its types, its layout, the GGUF block sizes of its gate/up and down rows
struct Layer {
    int gu = 0, d = 0;
    NativeExpertLayout L;
    size_t gu_blk = 0, d_blk = 0;
};

// one synthetic blob at `b` in the layout: random bytes, then a finite, small fp16 scale at the start of every block
// of every gate, up and down row (the GGUF blocks of IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S / Q2_0 / IQ4_NL all
// begin with `d`)
void fill_blob(uint8_t* b, const Layer& ly, Rng& rng) {
    const NativeExpertLayout& L = ly.L;
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
        stamp(b + (size_t) r * L.gu_row, L.gu_row, ly.gu_blk);
        stamp(b + L.up_off + (size_t) r * L.gu_row, L.gu_row, ly.gu_blk);
    }
    for (int64_t r = 0; r < L.n_embd; ++r) stamp(b + L.down_off + (size_t) r * L.d_row, L.d_row, ly.d_blk);
}

// the host plan of PeerExperts::launch, field for field (peer_experts.cpp :322-346): distinct owned experts in routing
// order, each one's entries as compact rows.  kind[i] == mine <=> this server owns ids[i] (2 the first server, 3 the
// second, as the pool marks them).
void plan_host(const int32_t* ids, const int32_t* kind, int64_t n, int64_t k, const int32_t* own_l,
               const unsigned long long* slot_ptr, pf::Plan& m, int& groups, int& rows, int32_t mine = 2) {
    groups = 0;
    rows = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (kind[i] != mine) continue;
        bool seen = false;
        for (int64_t j = 0; j < i; ++j)
            if (kind[j] == mine && ids[j] == ids[i]) { seen = true; break; }
        if (seen) continue;
        const int32_t slot = own_l[ids[i]];
        m.ptr[groups] = slot_ptr[slot];
        m.start[groups] = rows;
        for (int64_t j = i; j < n; ++j)
            if (kind[j] == mine && ids[j] == ids[i]) {
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
    // --second: the second server's plan (kind 3, its slots in the FIRST device's arena for the reference), counts and stamps
    pf::Plan plan2{};
    int groups2 = 0, rows2 = 0;
    uint32_t cnt2 = 0;
    unsigned long long st0_2 = 0, st1_2 = 0;
    // --stage: the step's rows as q8_1 (n_tok x N/32 blocks of 36 B), made on the first device BEFORE the window's
    // kernels go up (quantize_q8_1_rows: the reference's bytes); publish() only copies them into the xq page.  Made
    // during the window they would wait for the first server's persistent kernel to leave (it holds every SM).
    std::vector<uint8_t> xq;
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

// k_wait_copy's wait (one poller on the link), bounded: on timeout *perr = want and the host fails the step.
// `empty` (--skip, null otherwise): the server's empty word of the step, polled with the flag as wait_flag_ge_any does -
// the step passes on either.
__global__ void k_primary_wait(const volatile uint32_t* flag, uint32_t want, unsigned long long wait_ns,
                               volatile uint32_t* perr, unsigned long long* waited, const volatile uint32_t* empty) {
    if (threadIdx.x != 0) return;
    const unsigned long long t0 = gtime();
    while (*flag < want && !(empty != nullptr && *empty == want)) {
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
        case pf::kErrBadType: return "kErrBadType";
        case pf::kErrTimeoutXq: return "kErrTimeoutXq";
        case pf::kErrTimeoutDone: return "kErrTimeoutDone";
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
    // --second D2: the second server's own page (host view, its words) and D2's pointers of the shared pages
    uint8_t* page2_h = nullptr;
    uint8_t* page2_d = nullptr;
    volatile uint32_t* flag_out2 = nullptr;
    volatile uint32_t* abort2 = nullptr;
    volatile uint32_t* err2 = nullptr;
    volatile uint32_t* exit2 = nullptr;
    volatile uint32_t* cnt2 = nullptr;
    volatile unsigned long long* stamp2 = nullptr;
    const uint32_t* seq_d2 = nullptr;
    const float* x_d2 = nullptr;
    const int32_t* ids_d2 = nullptr;
    float* ymiss_d2 = nullptr;
    const uint32_t* pflag2_d = nullptr;     // --primary with --second: the primary's pointer of the second's flag
    unsigned long long* pwait2_d = nullptr; // its wait on it (pwords_h + 16)
    // --skip: the empty words of each page (host view: written by the host as PeerExperts::launch does; the primary's
    // view for its wait kernel)
    volatile uint32_t* empty = nullptr;
    volatile uint32_t* empty2 = nullptr;
    const uint32_t* pempty_d = nullptr;
    const uint32_t* pempty2_d = nullptr;
    // --stage: the second server's xq page (host view, D2's view) and its ring on the second page
    uint8_t* xq_h = nullptr;
    const uint8_t* xq_d2 = nullptr;
    volatile uint32_t* xq_seq2 = nullptr;
};

void* dev_ptr(void* host) {
    void* d = nullptr;
    CK(cudaHostGetDevicePointer(&d, host, 0));
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    int pos[6] = {0, 0, 0, 512, 16, 20};
    uint64_t seed = (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
    int primary = -1, second = -1, npos = 0;
    bool skip = false, stage = false;
    int skip_delay_us = 200;   // --skip: block 0's delay before it reads a step's ids (Window::test_delay_ns), 0 = none
    for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--primary" && a + 1 < argc) { primary = std::atoi(argv[++a]); continue; }
        if (s == "--second" && a + 1 < argc) { second = std::atoi(argv[++a]); continue; }
        if (s == "--skip") { skip = true; continue; }
        if (s == "--skip-delay" && a + 1 < argc) { skip_delay_us = std::max(0, std::atoi(argv[++a])); continue; }
        if (s == "--stage") { stage = true; continue; }
        if (s == "--help" || s == "-h") {
            std::printf("peer_fused_test [device=0] [gu_type=0 (mixed model)] [d_type=0] [n_expert=512] [n_slots=16 a layer] [windows=20] [seed] "
                        "[--primary D] [--second D2] [--skip] [--skip-delay US (200)] [--stage]\n");
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
    if (device < 0 || device >= nd || (primary >= 0 && (primary >= nd || primary == device)) ||
        (second >= 0 && (second >= nd || second == device || second == primary))) {
        std::printf("devices: %d; device %d, primary %d, second %d (three different devices)\n", nd, device, primary, second);
        return 1;
    }
    if (stage && second < 0) {
        std::printf("--stage needs --second D2 (the staged q8_1 rows are the second server's): ignored\n");
        stage = false;
    }

    // ---- the synthetic model: the mixed one (gu_type 0) or 4 uniform layers
    std::vector<Layer> layers;
    if (gu_type == 0) {
        for (int l = 0; l < kMixedLayers; ++l) { Layer ly; ly.gu = kMixed[l][0]; ly.d = kMixed[l][1]; layers.push_back(ly); }
    } else {
        for (int l = 0; l < 4; ++l) { Layer ly; ly.gu = gu_type; ly.d = d_type; layers.push_back(ly); }
    }
    const int nl = (int) layers.size();
    {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, device));
        std::printf("peer_fused_test: CUDA%d %s sm_%d%d SMs %d | %s model, %d layers:", device, p.name, p.major, p.minor,
                    p.multiProcessorCount, gu_type == 0 ? "mixed" : "uniform", nl);
        for (const Layer& ly : layers) std::printf(" %d/%d", ly.gu, ly.d);
        std::printf(" | n_expert %lld n_slots %d a layer windows %d seed %llu", (long long) n_expert, n_slots, windows,
                    (unsigned long long) seed);
        if (primary >= 0) {
            cudaDeviceProp q;
            CK(cudaGetDeviceProperties(&q, primary));
            int p2p = 0;
            cudaDeviceCanAccessPeer(&p2p, primary, device);
            std::printf(" | primary CUDA%d %s (P2P %d, not used)", primary, q.name, p2p);
        }
        if (second >= 0) {
            cudaDeviceProp q;
            CK(cudaGetDeviceProperties(&q, second));
            std::printf(" | second server CUDA%d %s sm_%d%d SMs %d", second, q.name, q.major, q.minor, q.multiProcessorCount);
        }
        if (skip) std::printf(" | --skip: an empty step is not waited for (Window::ahead on both kernels)");
        if (stage) std::printf(" | --stage: every window served twice, the second time with the second server's q8_1 rows staged");
        std::printf("\n");
    }

    // the layouts and the kernel's shape for them on this device (s.5): mode 8 is the fused server's precondition
    const int em = env_int("STRATA_EXP_MODE", -1);
    if (em != 8) strata::kernels::native_expert_set_mode(8, 0);
    CK(cudaSetDevice(device));
    std::vector<NativeExpertLayout> lays((size_t) nl);
    for (int l = 0; l < nl; ++l) {
        Layer& ly = layers[(size_t) l];
        ly.L = strata::kernels::native_expert_layout(ly.gu, ly.d, N, FF);
        const int gu_qk = qk_of(ly.gu), d_qk = qk_of(ly.d);
        if (gu_qk == 0 || d_qk == 0 || ly.L.bytes == 0) {
            std::printf("the synthetic generator knows gate/up 16 / 17 / 18 / 21 / 22 and down 42 / 20 only (layer %d: %d / %d)\n", l, ly.gu, ly.d);
            return 2;
        }
        ly.gu_blk = strata::kernels::iq_row_bytes(ly.gu, gu_qk);
        ly.d_blk = strata::kernels::iq_row_bytes(ly.d, d_qk);
        lays[(size_t) l] = ly.L;
    }
    pf::Shape shape;
    {
        std::string why;
        const int minb = env_int(pf::kEnvMinb, pf::kMinBlocksDefault);
        const int cap = env_int(pf::kEnvBlocks, 0);
        const bool coop = env_int(pf::kEnvCoop, 1) != 0;
        if (!strata::kernels::peer_fused_shape(lays.data(), nl, minb, cap, coop, shape, why)) {
            std::printf("peer_fused_shape refused the model on CUDA%d: %s\n", device, why.c_str());
            return 2;
        }
        std::printf("shape: gate/up types {%s}, down types {%s}, minb %d, dyn shared %zu B (grid %zu B), %d blocks/SM x %d SMs = %d blocks, %s launch\n",
                    pf::mask_text(shape.gu_mask).c_str(), pf::mask_text(shape.d_mask).c_str(), shape.minb, shape.dyn_shared,
                    shape.grid_bytes, shape.blocks_per_sm, shape.sms, shape.blocks, shape.cooperative ? "cooperative" : "plain");
    }
    const unsigned long long wait_ns = (unsigned long long) std::max(1, env_int(pf::kEnvWaitMs, 20000)) * 1000000ull;
    NativeExpertLayout* d_lay = nullptr;
    CK(cudaMalloc((void**) &d_lay, lays.size() * sizeof(NativeExpertLayout)));
    CK(cudaMemcpy(d_lay, lays.data(), lays.size() * sizeof(NativeExpertLayout), cudaMemcpyHostToDevice));
    // --second: the kernel's shape on THAT device (its occupancy, its SM count, its own instance in the fat binary),
    // its layout table and, as the engine does for the second server, the wait bound of STRATA_PEER2_WAIT_MS
    pf::Shape shape2;
    NativeExpertLayout* d_lay2 = nullptr;
    unsigned long long wait_ns2 = wait_ns;
    if (second >= 0) {
        CK(cudaSetDevice(second));
        std::string why;
        const int minb = env_int(pf::kEnvMinb, pf::kMinBlocksDefault);
        const int cap = env_int(pf::kEnvBlocks, 0);
        const bool coop = env_int(pf::kEnvCoop, 1) != 0;
        if (!strata::kernels::peer_fused_shape(lays.data(), nl, minb, cap, coop, shape2, why)) {
            std::printf("peer_fused_shape refused the model on the second server CUDA%d: %s\n", second, why.c_str());
            return 2;
        }
        std::printf("shape (second server CUDA%d): minb %d, dyn shared %zu B, %d blocks/SM x %d SMs = %d blocks, %s launch\n", second,
                    shape2.minb, shape2.dyn_shared, shape2.blocks_per_sm, shape2.sms, shape2.blocks, shape2.cooperative ? "cooperative" : "plain");
        wait_ns2 = (unsigned long long) std::max(1, env_int(pf::kEnvWaitMs2, pf::kWaitMs2Default)) * 1000000ull;
        CK(cudaMalloc((void**) &d_lay2, lays.size() * sizeof(NativeExpertLayout)));
        CK(cudaMemcpy(d_lay2, lays.data(), lays.size() * sizeof(NativeExpertLayout), cudaMemcpyHostToDevice));
        CK(cudaSetDevice(device));
    }

    Rng rng(seed);

    // ---- the slots: n_slots synthetic blobs a layer in the layer's layout (an arena a layer, 256-byte strides);
    // the global slot index is l * n_slots + s, so the ownership table of layer l deals only its own slots
    std::vector<uint8_t*> d_arena((size_t) nl, nullptr);
    std::vector<unsigned long long> slot_ptr((size_t) nl * (size_t) n_slots);
    // --second: the same blobs, slot for slot, in an arena on the second device (the reference of its entries runs on
    // the first device over the first arena: same bytes, so the comparison is card against card)
    std::vector<uint8_t*> d_arena2((size_t) nl, nullptr);
    std::vector<unsigned long long> slot_ptr2(second >= 0 ? slot_ptr.size() : 0);
    {
        size_t total = 0;
        for (int l = 0; l < nl; ++l) {
            const Layer& ly = layers[(size_t) l];
            const size_t stride = (ly.L.bytes + 255) / 256 * 256;
            CK(cudaMalloc((void**) &d_arena[(size_t) l], stride * (size_t) n_slots));
            if (second >= 0) {
                CK(cudaSetDevice(second));
                CK(cudaMalloc((void**) &d_arena2[(size_t) l], stride * (size_t) n_slots));
                CK(cudaSetDevice(device));
            }
            std::vector<uint8_t> blob(ly.L.bytes);
            for (int s = 0; s < n_slots; ++s) {
                fill_blob(blob.data(), ly, rng);
                CK(cudaMemcpy(d_arena[(size_t) l] + stride * (size_t) s, blob.data(), ly.L.bytes, cudaMemcpyHostToDevice));
                slot_ptr[(size_t) l * (size_t) n_slots + (size_t) s] = (unsigned long long) (uintptr_t) (d_arena[(size_t) l] + stride * (size_t) s);
                if (second >= 0) {
                    CK(cudaSetDevice(second));
                    CK(cudaMemcpy(d_arena2[(size_t) l] + stride * (size_t) s, blob.data(), ly.L.bytes, cudaMemcpyHostToDevice));
                    CK(cudaSetDevice(device));
                    slot_ptr2[(size_t) l * (size_t) n_slots + (size_t) s] = (unsigned long long) (uintptr_t) (d_arena2[(size_t) l] + stride * (size_t) s);
                }
            }
            total += stride * (size_t) n_slots;
        }
        std::printf("slots: %d a layer x %d layers, %.1f MiB of blobs%s\n", n_slots, nl, (double) total / (1024.0 * 1024.0),
                    second >= 0 ? " on each of the two cards" : "");
    }
    unsigned long long* d_slot_ptr = nullptr;
    CK(cudaMalloc((void**) &d_slot_ptr, sizeof(unsigned long long) * slot_ptr.size()));
    CK(cudaMemcpy(d_slot_ptr, slot_ptr.data(), sizeof(unsigned long long) * slot_ptr.size(), cudaMemcpyHostToDevice));
    unsigned long long* d_slot_ptr2 = nullptr;
    if (second >= 0) {
        CK(cudaSetDevice(second));
        CK(cudaMalloc((void**) &d_slot_ptr2, sizeof(unsigned long long) * slot_ptr2.size()));
        CK(cudaMemcpy(d_slot_ptr2, slot_ptr2.data(), sizeof(unsigned long long) * slot_ptr2.size(), cudaMemcpyHostToDevice));
        CK(cudaSetDevice(device));
    }

    // ---- the ownership table: ~40% of the pairs on a slot of the layer, the slots dealt from a shuffled order
    // (distinct while n_slots lasts, then the order again: a slot read by two experts is harmless to the comparison)
    std::vector<int32_t> own((size_t) nl * (size_t) n_expert, -1);
    // --second: a second, DISJOINT table (~30% of the pairs the first does not own, on a slot of the second arena);
    // `unowned` lists the pairs of neither server (the steps forced empty draw from it: empty on both kernels)
    std::vector<int32_t> own2(second >= 0 ? own.size() : 0, -1);
    std::vector<std::vector<int32_t>> unowned((size_t) nl);
    {
        int owned_pairs = 0, owned_pairs2 = 0;
        for (int l = 0; l < nl; ++l) {
            std::vector<int32_t> deal((size_t) n_slots), deal2((size_t) n_slots);
            for (int s = 0; s < n_slots; ++s) deal[(size_t) s] = deal2[(size_t) s] = l * n_slots + s;
            std::shuffle(deal.begin(), deal.end(), rng.g);
            if (second >= 0) std::shuffle(deal2.begin(), deal2.end(), rng.g);
            size_t next = 0, next2 = 0;
            for (int64_t e = 0; e < n_expert; ++e) {
                if (rng.below(100) < 40) {
                    own[(size_t) l * (size_t) n_expert + (size_t) e] = deal[next++ % (size_t) n_slots];
                    ++owned_pairs;
                } else if (second >= 0 && rng.below(100) < 30) {
                    own2[(size_t) l * (size_t) n_expert + (size_t) e] = deal2[next2++ % (size_t) n_slots];
                    ++owned_pairs2;
                } else {
                    unowned[(size_t) l].push_back((int32_t) e);
                }
            }
        }
        if (second >= 0)
            std::printf("ownership: %d of %lld pairs on the first server, %d on the second (disjoint)\n", owned_pairs, (long long) (nl * n_expert), owned_pairs2);
        else
            std::printf("ownership: %d of %lld pairs on the peer\n", owned_pairs, (long long) (nl * n_expert));
    }
    int32_t* d_own = nullptr;
    CK(cudaMalloc((void**) &d_own, own.size() * 4));
    CK(cudaMemcpy(d_own, own.data(), own.size() * 4, cudaMemcpyHostToDevice));
    int32_t* d_own2 = nullptr;
    if (second >= 0) {
        CK(cudaSetDevice(second));
        CK(cudaMalloc((void**) &d_own2, own2.size() * 4));
        CK(cudaMemcpy(d_own2, own2.data(), own2.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaSetDevice(device));
    }

    // ---- the kernel's VRAM (s.3) and the reference's (n_ff is the pack's one: the scratch and hq are sized once)
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
    // --second: the second kernel's VRAM, on its device
    pf::Plan* d_plan2 = nullptr;
    void* d_q8_2 = nullptr;
    uint8_t* d_scratch2 = nullptr;
    float* d_out2 = nullptr;
    pf::Sync* d_sync2 = nullptr;
    if (second >= 0) {
        CK(cudaSetDevice(second));
        CK(cudaMalloc((void**) &d_plan2, 2 * sizeof(pf::Plan)));
        CK(cudaMalloc(&d_q8_2, pf::q8_bytes(pf::kMaxTok, N)));
        CK(cudaMalloc((void**) &d_scratch2, scratch_bytes));
        CK(cudaMalloc((void**) &d_out2, (size_t) pf::kCap * N * sizeof(float)));
        CK(cudaMalloc((void**) &d_sync2, sizeof(pf::Sync)));
        CK(cudaMemset(d_plan2, 0, 2 * sizeof(pf::Plan)));
        CK(cudaMemset(d_sync2, 0, sizeof(pf::Sync)));
        CK(cudaSetDevice(device));
    }

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
    // --second: its own page (the engine's PeerExperts allocates one a server) and D2's view of the shared pages
    cudaStream_t s_kernel2 = nullptr;
    if (second >= 0) {
        CK(cudaHostAlloc((void**) &P.page2_h, pf::kPageBytes, hflags));
        std::memset(P.page2_h, 0, pf::kPageBytes);
        P.flag_out2 = (volatile uint32_t*) (P.page2_h + pf::kOffFlagOut);
        P.abort2 = (volatile uint32_t*) (P.page2_h + pf::kOffAbort);
        P.err2 = (volatile uint32_t*) (P.page2_h + pf::kOffErr);
        P.exit2 = (volatile uint32_t*) (P.page2_h + pf::kOffExit);
        P.cnt2 = (volatile uint32_t*) (P.page2_h + pf::kOffCnt);
        P.stamp2 = (volatile unsigned long long*) (P.page2_h + pf::kOffStamp);
        CK(cudaSetDevice(second));
        P.page2_d = (uint8_t*) dev_ptr(P.page2_h);
        P.seq_d2 = (const uint32_t*) dev_ptr(P.seq_h);
        P.x_d2 = (const float*) dev_ptr(P.x_h);
        P.ids_d2 = (const int32_t*) dev_ptr(P.ids_h);
        P.ymiss_d2 = (float*) dev_ptr(P.ymiss_h);
        CK(cudaStreamCreateWithFlags(&s_kernel2, cudaStreamNonBlocking));
        CK(cudaSetDevice(device));
    }
    // --skip: the empty words of each page (the engine's PeerExperts::launch writes them; here the host does, in publish)
    P.empty = (volatile uint32_t*) (P.page_h + pf::kOffEmpty);
    if (second >= 0) P.empty2 = (volatile uint32_t*) (P.page2_h + pf::kOffEmpty);
    // --stage: the second server's xq page (kMaxTok rows of q8_1 blocks) and its ring, D2's views
    if (stage) {
        const size_t xq_bytes = pf::q8_bytes(pf::kMaxTok, N);
        CK(cudaHostAlloc((void**) &P.xq_h, xq_bytes, hflags));
        std::memset(P.xq_h, 0, xq_bytes);
        P.xq_seq2 = (volatile uint32_t*) (P.page2_h + pf::kOffXqSeq);
        CK(cudaSetDevice(second));
        P.xq_d2 = (const uint8_t*) dev_ptr(P.xq_h);
        CK(cudaSetDevice(device));
    }

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
        if (second >= 0) {   // the primary waits the second server's flag too (one more wait node a step, after the first's)
            P.pflag2_d = (const uint32_t*) (((uint8_t*) dev_ptr(P.page2_h)) + pf::kOffFlagOut);
            P.pwait2_d = (unsigned long long*) (((uint8_t*) dev_ptr(P.pwords_h)) + 16);
        }
        if (skip) {   // --skip: the empty words as the primary's wait kernel reads them (wait_flag_ge_any's second word)
            P.pempty_d = (const uint32_t*) (((uint8_t*) dev_ptr(P.page_h)) + pf::kOffEmpty);
            if (second >= 0) P.pempty2_d = (const uint32_t*) (((uint8_t*) dev_ptr(P.page2_h)) + pf::kOffEmpty);
        }
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
    base.n_layers = nl;
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
    base.lay = d_lay;
    base.ahead = skip ? 1 : 0;   // --skip: the doorbell may run ahead of the flags (s.11.1)
    base.empty = skip ? (const uint32_t*) (P.page_d + pf::kOffEmpty) : nullptr;   // the ids, then this word (the host's verdict)
    base.test_delay_ns = skip ? (unsigned long long) skip_delay_us * 1000ull : 0ull;   // the next doorbell lands before the ids are read
    // --second: the second kernel's argument - D2's pointers, its own page, tables and VRAM, the shared doorbell
    pf::Window base2 = base;
    if (second >= 0) {   // (--stage: xq and xq_seq are set per run, off then on, in the window loop: the kernel wants both or neither)
        base2.nblocks = shape2.blocks;
        base2.wait_ns = wait_ns2;
        base2.seq = P.seq_d2;
        base2.x = P.x_d2;
        base2.ids = P.ids_d2;
        base2.ymiss = P.ymiss_d2;
        base2.flag_out = (uint32_t*) (P.page2_d + pf::kOffFlagOut);
        base2.abort = (const uint32_t*) (P.page2_d + pf::kOffAbort);
        base2.err = (uint32_t*) (P.page2_d + pf::kOffErr);
        base2.exit = (uint32_t*) (P.page2_d + pf::kOffExit);
        base2.cnt = (uint32_t*) (P.page2_d + pf::kOffCnt);
        base2.stamp = (unsigned long long*) (P.page2_d + pf::kOffStamp);
        base2.empty = skip ? (const uint32_t*) (P.page2_d + pf::kOffEmpty) : nullptr;   // its own page's words (base's point at the first's)
        base2.own = d_own2;
        base2.slot_ptr = d_slot_ptr2;
        base2.plan = d_plan2;
        base2.q8 = d_q8_2;
        base2.hq = d_scratch2 + pf::hq_offset(pf::kCap, FF);
        base2.out = d_out2;
        base2.sync = d_sync2;
        base2.lay = d_lay2;
    }

    uint32_t epoch = 0;
    bool all_ok = true;
    int64_t peer_entries_total = 0, peer2_entries_total = 0;

    // ---- fused_begin's part: the words, the sentinel, d_sync, the launch, the WDDM push.  --second: the same for
    // the second kernel right after (w2 carries the same geometry), as the verifier calls the two begins back to back
    auto begin_window = [&](pf::Window& w, pf::Window& w2) -> bool {
        const cudaError_t q = cudaStreamQuery(s_kernel);
        if (q != cudaSuccess) {
            std::printf("  the previous window's kernel is still on the stream (%s)\n", cudaGetErrorString(q));
            return false;
        }
        if (second >= 0) {
            const cudaError_t q2 = cudaStreamQuery(s_kernel2);
            if (q2 != cudaSuccess) {
                std::printf("  the previous window's kernel is still on the second server's stream (%s)\n", cudaGetErrorString(q2));
                return false;
            }
        }
        w.epoch = ++epoch;
        *P.flag_out = 0; *P.abort = 0; *P.err = 0; *P.exit = 0;
        for (int s = 0; s < pf::kMaxSteps; ++s) { P.cnt[s] = 0; P.empty[s] = 0; }
        for (int s = 0; s < 2 * pf::kMaxSteps; ++s) P.stamp[s] = 0;
        if (second >= 0) {
            w2.epoch = epoch;
            *P.flag_out2 = 0; *P.abort2 = 0; *P.err2 = 0; *P.exit2 = 0;
            for (int s = 0; s < pf::kMaxSteps; ++s) { P.cnt2[s] = 0; P.empty2[s] = 0; }
            for (int s = 0; s < 2 * pf::kMaxSteps; ++s) P.stamp2[s] = 0;
            if (P.xq_seq2 != nullptr) *P.xq_seq2 = 0;   // --stage: no q8_1 rows published yet (fused_begin's zeroing)
        }
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
        if (second >= 0) {
            CK(cudaSetDevice(second));
            CK(cudaMemsetAsync(d_sync2, 0, sizeof(pf::Sync), s_kernel2));
            const bool ok2 = strata::kernels::peer_fused_launch(w2, shape2, s_kernel2, err);
            (void) cudaStreamQuery(s_kernel2);
            CK(cudaSetDevice(device));
            if (!ok2) {
                std::printf("  peer_fused_launch (second server): %s\n", err.c_str());
                return false;   // the caller aborts: the first kernel is already waiting for a doorbell
            }
        }
        return true;
    };

    // the host's spin on flag_out >= want with the engine's bound, the streams flushed every ~2 ms (hook.flush);
    // --second: both flags (the graph has one wait node a server, the first's then the second's).  --skip: a server
    // with no entry at the step (skip1 / skip2) is not waited for - the primary's node passes on the empty word
    auto wait_flag = [&](uint32_t want, std::string& why, bool skip1 = false, bool skip2 = false) -> bool {
        const auto t0 = Clock::now();
        auto last_flush = t0;
        while ((!skip1 && *P.flag_out < want) || (second >= 0 && !skip2 && *P.flag_out2 < want)) {
            if (*P.err != 0) { why = std::string("the kernel reported ") + err_name(*P.err); return false; }
            if (second >= 0 && *P.err2 != 0) { why = std::string("the second server's kernel reported ") + err_name(*P.err2); return false; }
            if (secs_since(t0) > kFlagWaitS) { why = "no flag within " + std::to_string((int) kFlagWaitS) + " s"; return false; }
            if (secs_since(last_flush) > 0.002) {
                (void) cudaStreamQuery(s_kernel);
                if (second >= 0) (void) cudaStreamQuery(s_kernel2);   // (a stream query works from any current device)
                last_flush = Clock::now();
            }
        }
        host_fence();
        return true;
    };

    // after an error or a stuck window: abort, drain within 5 s, so the process never exits over a live kernel
    // (--second: both kernels, as release_gpu_waits aborts both hooks)
    auto abort_window = [&]() {
        host_fence();
        *P.abort = 1;
        if (second >= 0) *P.abort2 = 1;
        host_fence();
        const cudaError_t e = drain(s_kernel, 5.0);
        if (e == cudaErrorNotReady) std::printf("  the kernel did not leave within 5 s of abort: leaving it to the driver\n");
        else if (e != cudaSuccess) std::printf("  stream error after abort: %s\n", cudaGetErrorString(e));
        if (second >= 0) {
            const cudaError_t e2 = drain(s_kernel2, 5.0);
            if (e2 == cudaErrorNotReady) std::printf("  the second server's kernel did not leave within 5 s of abort: leaving it to the driver\n");
            else if (e2 != cudaSuccess) std::printf("  second server: stream error after abort: %s\n", cudaGetErrorString(e2));
        }
    };

    // ---- publishes one step: the host writes x / ids (or the primary's kernel does), fences, rings seq = s + 1.  With
    // --primary the push alone goes on s_primary here; the wait on flag_out is queued by primary_wait, so the stream
    // carries the graph's order push(0), push(1), wait(0), push(2), wait(1), ... (G = 2) and not push / wait a step.
    // The step's ymiss rows get the sentinel first: the kernel must not find a stale row of the step before (G steps
    // back, the same group) under what it writes, nor must a non-owned entry keep one.
    bool stage_run = false;   // --stage: the current run of the window stages the second server's rows
    auto publish = [&](const StepRec& r, uint32_t want) {
        const int n = r.n_tok * K;
        std::memset(P.ymiss_h + (size_t) r.tb * K * N, kSentKernel, (size_t) n * N * sizeof(float));
        if (skip) {   // --skip: the empty word of a server with nothing at this step, as PeerExperts::launch writes it
            if (r.groups == 0) P.empty[r.s] = want;
            if (second >= 0 && r.groups2 == 0) P.empty2[r.s] = want;
        }
        if (stage_run) {
            // --stage: the step's q8_1 rows (made before the window: r.xq, the reference's bytes) into the second server's
            // xq page at the step's row offset, then its ring - before the doorbell here (the engine's publish node runs
            // right after it; the kernel reads the ring only once it has the doorbell and entries).  No CUDA call here:
            // the first device's SMs are the first server's kernel's for the length of the window.
            const size_t off = (size_t) r.tb * (size_t) (N / 32) * 36;
            std::memcpy(P.xq_h + off, r.xq.data(), r.xq.size());
            host_fence();
            *P.xq_seq2 = want;
        }
        host_fence();
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
        const int s = (int) want - 1;
        // (--skip: the step's empty word beside the flag, as the engine's wait_flag_ge_any node)
        k_primary_wait<<<1, 32, 0, s_primary>>>(P.pflag_d, want, wait_ns, P.perr_d, P.pwait_d, skip ? P.pempty_d + s : nullptr);
        CK(cudaGetLastError());
        if (second >= 0) {   // then the second server's flag: the graph's second node of the step
            k_primary_wait<<<1, 32, 0, s_primary>>>(P.pflag2_d, want, wait_ns, P.perr_d, P.pwait2_d, skip ? P.pempty2_d + s : nullptr);
            CK(cudaGetLastError());
        }
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
        const int32_t* own2_l = second >= 0 ? own2.data() + (size_t) r.l * (size_t) n_expert : nullptr;
        const bool nothing_owned = !unowned[(size_t) r.l].empty() && (force_empty || rng.below(6) == 0);
        for (int i = 0; i < n; ++i) {
            const uint32_t d = rng.below(16);
            int32_t e;
            if (d == 0) e = -1;
            else if (nothing_owned) e = unowned[(size_t) r.l][rng.below((uint32_t) unowned[(size_t) r.l].size())];
            else if (d <= 5 && i > 0) e = r.ids[rng.below((uint32_t) i)];   // the same expert again, this or another token
            else e = (int32_t) rng.below((uint32_t) n_expert);
            r.ids[(size_t) i] = e;
            // the pool's codes: 2 the first server, 3 the second (asked after the first), 0 neither, -1 a hole
            r.kind[(size_t) i] = e >= 0 && own_l[e] >= 0 ? 2 : (e >= 0 && own2_l != nullptr && own2_l[e] >= 0) ? 3 : (e >= 0 ? 0 : -1);
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
        // the second server's plan with the FIRST arena's addresses: it is the reference's plan (run on the first
        // device over the same bytes) and the host's count against cnt2; the kernel on D2 plans from d_own2 / d_slot_ptr2
        if (second >= 0) plan_host(r.ids.data(), r.kind.data(), n, K, own2_l, slot_ptr.data(), r.plan2, r.groups2, r.rows2, 3);
    };

    // ---- the reference of one step on s_ref (today's path, peer_experts.cpp launch, with the layer's own layout),
    // compared to the snapshot
    auto check_step = [&](const StepRec& r, int win) -> bool {
        const int n = r.n_tok * K;
        const Layer& ly = layers[(size_t) r.l];
        bool ok = true;
        const uint32_t expect = ((uint32_t) r.groups << 16) | (uint32_t) r.rows;
        if (r.cnt != expect) {
            std::printf("  FAIL finestra %d passo %d (strato %d, %d/%d): cnt %ux%u, host %dx%d\n", win, r.s, r.l, ly.gu, ly.d,
                        r.cnt >> 16, r.cnt & 0xffffu, r.groups, r.rows);
            ok = false;
        }
        if (r.st1 < r.st0 || r.st0 == 0) {
            std::printf("  FAIL finestra %d passo %d: stamp %llu -> %llu\n", win, r.s, r.st0, r.st1);
            ok = false;
        }
        if (second >= 0) {   // the second server's count and stamps, from its own page
            const uint32_t expect2 = ((uint32_t) r.groups2 << 16) | (uint32_t) r.rows2;
            if (r.cnt2 != expect2) {
                std::printf("  FAIL finestra %d passo %d (strato %d, %d/%d): secondo server: cnt %ux%u, host %dx%d\n", win, r.s, r.l, ly.gu, ly.d,
                            r.cnt2 >> 16, r.cnt2 & 0xffffu, r.groups2, r.rows2);
                ok = false;
            }
            if (r.st1_2 < r.st0_2 || r.st0_2 == 0) {
                std::printf("  FAIL finestra %d passo %d: secondo server: stamp %llu -> %llu\n", win, r.s, r.st0_2, r.st1_2);
                ok = false;
            }
        }
        std::memset(P.ymiss_ref_h + (size_t) r.tb * K * N, kSentRef, (size_t) n * N * sizeof(float));
        // the reference of each server's entries, today's path on the FIRST device: the first's plan, then the
        // second's (its rows are other entries of the same step: disjoint rows of the same page)
        for (int which = 0; which < (second >= 0 ? 2 : 1); ++which) {
            const pf::Plan& plan = which == 0 ? r.plan : r.plan2;
            const int groups = which == 0 ? r.groups : r.groups2, rows = which == 0 ? r.rows : r.rows2;
            if (groups <= 0) continue;
            std::memcpy(h_plan_ref, &plan, sizeof(pf::Plan));
            CK(cudaMemcpyAsync(d_x_ref, r.x.data(), (size_t) r.n_tok * N * sizeof(float), cudaMemcpyHostToDevice, s_ref));
            CK(cudaMemcpyAsync(d_plan_ref, h_plan_ref, sizeof(pf::Plan), cudaMemcpyHostToDevice, s_ref));
            strata::kernels::quantize_q8_1_rows(d_x_ref, r.n_tok, N, d_q8_ref, s_ref);
            strata::kernels::native_expert_grouped(ly.L, d_plan_ref->ptr, d_plan_ref->start, d_plan_ref->count, d_plan_ref->dst,
                                                   d_plan_ref->tok, groups, rows, d_q8_ref, d_scratch_ref, d_out_ref, s_ref);
            strata::kernels::scatter_rows_f32(d_out_ref, P.ymiss_ref_d + (size_t) r.tb * K * N, d_plan_ref->out_row, rows, N, s_ref);
            CK(cudaStreamSynchronize(s_ref));
            host_fence();
        }
        // every entry of the step: owned (by either server) -> the reference's row; not owned -> the sentinel untouched
        std::vector<uint8_t> sent((size_t) N * sizeof(float), kSentKernel);
        for (int j = 0; j < n && ok; ++j) {
            const float* got = r.rows_out.data() + (size_t) j * N;
            const bool owned = r.kind[(size_t) j] == 2 || r.kind[(size_t) j] == 3;
            const uint8_t* want = owned ? (const uint8_t*) (P.ymiss_ref_h + ((size_t) r.tb * K + (size_t) j) * N)
                                        : sent.data();
            if (std::memcmp(got, want, (size_t) N * sizeof(float)) == 0) continue;
            int64_t c = 0;
            while (c < N && std::memcmp((const uint8_t*) got + c * 4, want + c * 4, 4) == 0) ++c;
            uint32_t a = 0, b = 0;
            std::memcpy(&a, (const uint8_t*) got + c * 4, 4);
            std::memcpy(&b, want + c * 4, 4);
            int cr = -1;
            if (r.kind[(size_t) j] == 3) { for (int q = 0; q < r.rows2; ++q) if (r.plan2.out_row[q] == j) { cr = q; break; } }
            else { for (int q = 0; q < r.rows; ++q) if (r.plan.out_row[q] == j) { cr = q; break; } }
            std::printf("  FAIL finestra %d passo %d (strato %d, gate/up %d, down %d, tb %d): voce %d (riga compatta %d, token %d, esperto %d, %s) "
                        "colonna %lld: kernel %08x, riferimento %08x\n", win, r.s, r.l, ly.gu, ly.d, r.tb, j, cr, j / K, r.ids[(size_t) j],
                        r.kind[(size_t) j] == 2 ? "del peer" : r.kind[(size_t) j] == 3 ? "del secondo server" : "di nessun server: sentinella attesa",
                        (long long) c, a, b);
            ok = false;
        }
        return ok;
    };

    // ================================================================ the windows
    for (int win = 0; win < windows && all_ok; ++win) {
        const int T = 1 + (int) rng.below((uint32_t) pf::kMaxTok);
        const int G = T >= 2 && rng.below(2) ? 2 : 1;
        const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
        const int steps = nl * G;
        pf::Window w = base;
        w.steps = steps; w.G = G; w.T = T;
        w.gtb[0] = gtb[0]; w.gtb[1] = gtb[1]; w.gte[0] = gte[0]; w.gte[1] = gte[1];
        pf::Window w2 = base2;   // --second: the same geometry on the second kernel
        w2.steps = steps; w2.G = G; w2.T = T;
        w2.gtb[0] = gtb[0]; w2.gtb[1] = gtb[1]; w2.gte[0] = gte[0]; w2.gte[1] = gte[1];
        std::vector<StepRec> recs((size_t) steps);
        for (int s = 0; s < steps; ++s) {
            StepRec& r = recs[(size_t) s];
            r.s = s; r.l = s / G;
            r.tb = gtb[s % G];
            r.n_tok = gte[s % G] - gtb[s % G];
            // G = 2: half of the group-1 steps behind a group-0 step with entries own nothing - the step without
            // barriers whose doorbell is already up while the kernel still writes the rows of the step before
            const bool force_empty = G == 2 && (s % G) == 1 && (recs[(size_t) s - 1].rows > 0 || recs[(size_t) s - 1].rows2 > 0) && rng.below(2) == 0;
            draw_step(r, force_empty);
        }
        if (stage) {
            // --stage: every step's q8_1 rows NOW, on the first device while it is free (quantize_q8_1_rows on s_ref: the
            // reference's bytes).  Once the window's kernels are up the first server's one holds every SM of this
            // device and a quantize queued beside it would run only when that kernel leaves (1fe8d044: kErrTimeoutSeq
            // of the first server after 20 s, the window served on the second's rows meanwhile).
            for (StepRec& r : recs) {
                const size_t bytes = (size_t) r.n_tok * (size_t) (N / 32) * 36;
                r.xq.resize(bytes);
                CK(cudaMemcpyAsync(d_x_ref, r.x.data(), (size_t) r.n_tok * N * sizeof(float), cudaMemcpyHostToDevice, s_ref));
                strata::kernels::quantize_q8_1_rows(d_x_ref, r.n_tok, N, d_q8_ref, s_ref);
                CK(cudaMemcpyAsync(r.xq.data(), d_q8_ref, bytes, cudaMemcpyDeviceToHost, s_ref));
                CK(cudaStreamSynchronize(s_ref));
            }
        }
        // ---- one run of the window: begin, the graph's doorbell order, the snapshots, end.  --stage: run twice on the
        // same steps (the second time with the second server's rows staged); the records of a run overwrite the last.
        // The counts and stamps are read AFTER the window: with --skip the host does not wait an empty step's flag, so
        // at snapshot time the kernel may not have served it yet (it serves every step before it leaves).
        auto serve = [&](bool stage_on) -> bool {
            stage_run = stage_on;
            w2.xq = stage_on ? (const void*) P.xq_d2 : nullptr;
            w2.xq_seq = stage_on ? (const uint32_t*) (P.page2_d + pf::kOffXqSeq) : nullptr;
            if (!begin_window(w, w2)) { abort_window(); return false; }
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
                // (--skip: a server with nothing at this step is not waited for: the next doorbell goes up at once)
                if (!wait_flag((uint32_t) s + 1, why, skip && r.groups == 0, skip && r.groups2 == 0)) { win_ok = false; break; }
                r.rows_out.resize((size_t) n * N);
                std::memcpy(r.rows_out.data(), P.ymiss_h + (size_t) r.tb * K * N, (size_t) n * N * sizeof(float));
                if (s + G < steps) publish(recs[(size_t) s + G], (uint32_t) (s + G) + 1);   // pre(l + 1, grp): the same group's rows again
            }
            if (!win_ok) {
                std::printf("  FAIL finestra %d%s: %s (err %s, flag_out %u", win, stage_on ? " (con stage)" : "", why.c_str(), err_name(*P.err), *P.flag_out);
                if (second >= 0) std::printf("; secondo server: err %s, flag_out %u", err_name(*P.err2), *P.flag_out2);
                std::printf(")\n");
                abort_window();
                return false;
            }
            // fused_end's part: the kernel has left, no error, the epoch written (--second: both kernels)
            const cudaError_t e = drain(s_kernel, 5.0);
            const cudaError_t e2 = second >= 0 ? drain(s_kernel2, 5.0) : cudaSuccess;
            if (e != cudaSuccess || e2 != cudaSuccess) {
                std::printf("  FAIL finestra %d: a kernel did not leave after the last flag (%s; second server %s)\n", win, cudaGetErrorString(e),
                            second >= 0 ? cudaGetErrorString(e2) : "-");
                abort_window();
                return false;
            }
            host_fence();
            // (which server, and both pages' words: an error of one kernel at the end of a served window is a wait that
            // outlived the window - a doorbell or an xq ring nobody raised, or a quantize queued on a busy device)
            auto words = [&]() {
                std::printf(" | primo server CUDA%d: err %s, exit %u (epoch %u), flag_out %u, seq %u", device, err_name(*P.err), *P.exit, w.epoch,
                            *P.flag_out, *(volatile uint32_t*) P.seq_h);
                if (second >= 0)
                    std::printf("; secondo server CUDA%d: err %s, exit %u (epoch %u), flag_out %u, xq_seq %u", second, err_name(*P.err2), *P.exit2,
                                w2.epoch, *P.flag_out2, P.xq_seq2 != nullptr ? *P.xq_seq2 : 0u);
                std::printf("\n");
            };
            if (*P.err != 0) { std::printf("  FAIL finestra %d%s: primo server: err %s after the window", win, stage_on ? " (con stage)" : "", err_name(*P.err)); words(); return false; }
            if (*P.exit != w.epoch) { std::printf("  FAIL finestra %d: primo server: exit %u, epoch %u", win, *P.exit, w.epoch); words(); return false; }
            if (second >= 0) {
                if (*P.err2 != 0) { std::printf("  FAIL finestra %d%s: secondo server: err %s after the window", win, stage_on ? " (con stage)" : "", err_name(*P.err2)); words(); return false; }
                if (*P.exit2 != w2.epoch) { std::printf("  FAIL finestra %d: secondo server: exit %u, epoch %u", win, *P.exit2, w2.epoch); words(); return false; }
            }
            for (int s = 0; s < steps; ++s) {   // every step's count and stamps, now that both kernels have served them all
                StepRec& r = recs[(size_t) s];
                r.cnt = P.cnt[s];
                r.st0 = P.stamp[2 * s];
                r.st1 = P.stamp[2 * s + 1];
                if (second >= 0) {
                    r.cnt2 = P.cnt2[s];
                    r.st0_2 = P.stamp2[2 * s];
                    r.st1_2 = P.stamp2[2 * s + 1];
                }
            }
            return true;
        };
        // the reference of every step against the run's snapshots; the spans of the run
        auto check_run = [&](std::vector<double>& spans, std::vector<double>& spans2, std::vector<double>& pwaits) -> bool {
            spans.clear(); spans2.clear(); pwaits.clear();
            bool ok = true;
            for (int s = 0; s < steps; ++s) {
                const StepRec& r = recs[(size_t) s];
                spans.push_back((double) (r.st1 - r.st0) * 1e-3);
                if (second >= 0) spans2.push_back((double) (r.st1_2 - r.st0_2) * 1e-3);
                if (primary >= 0) pwaits.push_back((double) r.primary_wait_ns * 1e-3);
                if (ok && !check_step(r, win)) ok = false;
            }
            return ok;
        };
        bool win_ok = serve(false);
        std::vector<double> spans, spans2, pwaits;
        if (win_ok) win_ok = check_run(spans, spans2, pwaits);
        else { all_ok = false; break; }
        int64_t entries = 0, entries2 = 0;
        for (const StepRec& r : recs) { entries += r.rows; entries2 += r.rows2; }
        peer_entries_total += entries;
        peer2_entries_total += entries2;
        std::printf("finestra %3d: T %2d G %d passi %2d voci del peer %4lld  flag visto -> alzato p50 %8.1f us", win, T, G, steps,
                    (long long) entries, median(spans));
        if (second >= 0) std::printf("  2o server: voci %4lld p50 %8.1f us", (long long) entries2, median(spans2));
        if (primary >= 0) std::printf("  attesa padrona p50 %8.1f us", median(pwaits));
        if (stage && win_ok) {
            // --stage: the same window again, the second server's rows staged; every row compared again, then the
            // second server's span of every step without and with the stage (its entries say what the step moved)
            const std::vector<double> spans2_off = spans2;
            std::vector<double> spans_b, spans2_b, pwaits_b;
            win_ok = serve(true) && check_run(spans_b, spans2_b, pwaits_b);
            if (win_ok) {
                std::printf("  con stage: 2o server p50 %8.1f us (senza %8.1f us)", median(spans2_b), median(spans2_off));
                if (primary >= 0) std::printf("  attesa padrona p50 %8.1f us", median(pwaits_b));
            }
            std::printf("  %s\n", win_ok ? "PASS" : "FAIL");
            if (win_ok)
                for (int s = 0; s < steps; ++s) {
                    const StepRec& r = recs[(size_t) s];
                    std::printf("    passo %2d strato %d (%d/%d) righe %d: 2o server voci %3d: senza stage %8.1f us, con stage %8.1f us\n", s, r.l,
                                layers[(size_t) r.l].gu, layers[(size_t) r.l].d, r.n_tok, r.rows2, spans2_off[(size_t) s], spans2_b[(size_t) s]);
                }
        } else {
            std::printf("  %s\n", win_ok ? "PASS" : "FAIL");
        }
        if (!win_ok) all_ok = false;
    }

    // ================================================================ endurance (s.9)
    // (a) steps 0 and 1 served, step 2 never rung, abort after 100 ms: kErrAbort, exit == epoch, the stream drained in 2 s
    if (all_ok) {
        const int T = 1 + (int) rng.below((uint32_t) pf::kMaxTok);
        pf::Window w = base;
        w.steps = nl; w.G = 1; w.T = T;
        w.gtb[0] = 0; w.gtb[1] = T; w.gte[0] = T; w.gte[1] = T;
        pf::Window w2 = base2;
        w2.steps = nl; w2.G = 1; w2.T = T;
        w2.gtb[0] = 0; w2.gtb[1] = T; w2.gte[0] = T; w2.gte[1] = T;
        bool ok = begin_window(w, w2);
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
            const bool still2 = second < 0 || (*P.err2 == 0 && *P.flag_out2 == 2 && cudaStreamQuery(s_kernel2) == cudaErrorNotReady);
            const auto t0 = Clock::now();
            host_fence();
            *P.abort = 1;
            if (second >= 0) *P.abort2 = 1;
            host_fence();
            const cudaError_t e = drain(s_kernel, 2.0);
            const cudaError_t e2 = second >= 0 ? drain(s_kernel2, 2.0) : cudaSuccess;
            const double ms = secs_since(t0) * 1e3;
            host_fence();
            // the error exit raises flag_out to UINT32_MAX (the primary's waits must pass): 2 before, MAX after
            const bool good = still && e == cudaSuccess && *P.err == pf::kErrAbort && *P.exit == w.epoch && *P.flag_out == UINT32_MAX;
            const bool good2 = second < 0 || (still2 && e2 == cudaSuccess && *P.err2 == pf::kErrAbort && *P.exit2 == w2.epoch && *P.flag_out2 == UINT32_MAX);
            std::printf("tenuta (a) abort al passo 2: %s | prima dell'abort: in attesa %s | uscita in %.1f ms (%s), err %s, exit %u (epoch %u), "
                        "flag_out %u (atteso %u)\n", good && good2 ? "PASS" : "FAIL", still ? "si" : "no", ms, cudaGetErrorString(e), err_name(*P.err),
                        *P.exit, w.epoch, *P.flag_out, UINT32_MAX);
            if (second >= 0)
                std::printf("tenuta (a) secondo server: %s | in attesa %s | stream %s, err %s, exit %u (epoch %u), flag_out %u\n", good2 ? "PASS" : "FAIL",
                            still2 ? "si" : "no", cudaGetErrorString(e2), err_name(*P.err2), *P.exit2, w2.epoch, *P.flag_out2);
            if (!good || !good2) { if (e == cudaErrorNotReady || e2 == cudaErrorNotReady) abort_window(); ok = false; }
        } else {
            abort_window();
        }
        if (!ok) all_ok = false;
    }
    // (b) wait_ns = 200 ms, no doorbell at all: kErrTimeoutSeq, exit == epoch, the stream drained
    if (all_ok) {
        pf::Window w = base;
        w.steps = nl; w.G = 1; w.T = 1;
        w.gtb[0] = 0; w.gtb[1] = 1; w.gte[0] = 1; w.gte[1] = 1;
        w.wait_ns = 200ull * 1000000ull;
        pf::Window w2 = base2;
        w2.steps = nl; w2.G = 1; w2.T = 1;
        w2.gtb[0] = 0; w2.gtb[1] = 1; w2.gte[0] = 1; w2.gte[1] = 1;
        w2.wait_ns = 200ull * 1000000ull;
        bool ok = begin_window(w, w2);
        if (ok) {
            const auto t0 = Clock::now();
            const cudaError_t e = drain(s_kernel, 2.0);
            const cudaError_t e2 = second >= 0 ? drain(s_kernel2, 2.0) : cudaSuccess;
            const double ms = secs_since(t0) * 1e3;
            host_fence();
            const bool good = e == cudaSuccess && *P.err == pf::kErrTimeoutSeq && *P.exit == w.epoch && *P.flag_out == UINT32_MAX;
            const bool good2 = second < 0 || (e2 == cudaSuccess && *P.err2 == pf::kErrTimeoutSeq && *P.exit2 == w2.epoch && *P.flag_out2 == UINT32_MAX);
            std::printf("tenuta (b) wait_ns 200 ms senza doorbell: %s | uscita in %.1f ms (%s), err %s, exit %u (epoch %u), flag_out %u (atteso %u)\n",
                        good && good2 ? "PASS" : "FAIL", ms, cudaGetErrorString(e), err_name(*P.err), *P.exit, w.epoch, *P.flag_out, UINT32_MAX);
            if (second >= 0)
                std::printf("tenuta (b) secondo server: %s | stream %s, err %s, exit %u (epoch %u), flag_out %u\n", good2 ? "PASS" : "FAIL",
                            cudaGetErrorString(e2), err_name(*P.err2), *P.exit2, w2.epoch, *P.flag_out2);
            if (!good || !good2) { if (e == cudaErrorNotReady || e2 == cudaErrorNotReady) abort_window(); ok = false; }
        } else {
            abort_window();
        }
        if (!ok) all_ok = false;
    }

    // ---- the end: never destroy a stream under a live kernel (the aborts above drained it, or said they could not)
    if (cudaStreamQuery(s_kernel) == cudaErrorNotReady || (second >= 0 && cudaStreamQuery(s_kernel2) == cudaErrorNotReady)) {
        std::printf("a kernel is still on its stream at exit: %s\n", all_ok ? "FAIL" : "FAIL (see above)");
        std::fflush(stdout);
        std::_Exit(1);
    }
    CK(cudaStreamDestroy(s_kernel));
    CK(cudaStreamDestroy(s_ref));
    if (second >= 0) {
        CK(cudaSetDevice(second));
        CK(cudaStreamDestroy(s_kernel2));
        for (uint8_t* a : d_arena2) cudaFree(a);
        cudaFree(d_slot_ptr2); cudaFree(d_own2); cudaFree(d_lay2);
        cudaFree(d_plan2); cudaFree(d_q8_2); cudaFree(d_scratch2); cudaFree(d_out2); cudaFree(d_sync2);
        CK(cudaSetDevice(device));
        cudaFreeHost(P.page2_h);
    }
    if (stage) cudaFreeHost(P.xq_h);
    if (primary >= 0) {
        CK(cudaSetDevice(primary));
        (void) drain(s_primary, 2.0);
        CK(cudaStreamDestroy(s_primary));
        cudaFree(d_px); cudaFree(d_pids);
        cudaFreeHost(h_px); cudaFreeHost(h_pids);
        CK(cudaSetDevice(device));
    }
    for (uint8_t* a : d_arena) cudaFree(a);
    cudaFree(d_slot_ptr); cudaFree(d_own); cudaFree(d_lay);
    cudaFree(d_plan); cudaFree(d_q8); cudaFree(d_scratch); cudaFree(d_out); cudaFree(d_sync);
    cudaFree(d_x_ref); cudaFree(d_q8_ref); cudaFree(d_scratch_ref); cudaFree(d_out_ref); cudaFree(d_plan_ref);
    cudaFreeHost(h_plan_ref);
    cudaFreeHost(P.page_h); cudaFreeHost(P.seq_h); cudaFreeHost(P.x_h); cudaFreeHost(P.ids_h);
    cudaFreeHost(P.ymiss_h); cudaFreeHost(P.ymiss_ref_h); cudaFreeHost(P.pwords_h);
    std::printf("%s: %d finestre (%s, %d strati), %lld voci del peer", all_ok ? "PASS" : "FAIL", windows,
                gu_type == 0 ? "modello misto" : "modello uniforme", nl, (long long) peer_entries_total);
    if (second >= 0) std::printf(" + %lld del secondo server (CUDA%d)", (long long) peer2_entries_total, second);
    std::printf(", seme %llu\n", (unsigned long long) seed);
    return all_ok ? 0 : 1;
}
