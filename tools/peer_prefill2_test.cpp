// tools/peer_prefill2_test.cpp - PEER_PREFILL2 (docs/PEER_PREFILL2.md s.6, lab/upsync-v0.2.9): the second expert server's
// prompt rows (STRATA_PEER2_PREFILL=1, prefill.cpp's rows-in path) bit for bit against the primary's own MMQ path.
//
// No model, no engine: a synthetic model of `nl` expert layers that MIX types as the pack does (the ten (gate/up, down)
// pairs of Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, one layer each), `n_experts` synthetic blobs a layer with a VALID
// structure (finite, small fp16 block scales; random code bytes: MMQ indexes 8- or 9-bit lookups and dp4a with them),
// the SAME blobs slot for slot on both cards, T tokens of activations x (uniform in [-8, 8], a few 1e4 values), K
// routed experts a token with their weights, and the primary's row layout (rows in expert-id order, `off[e]`, `src`).
//
// Per chunk, what the engine does, step for step (prefill.cpp `peer_layer`, rows_in):
//   primary   quantize x's rows in expert order into Xq (q8_1, the bytes its own gate/up product reads); the
//             REFERENCE: every expert's rows through the MMQ path (gather_native of groups of up to 16 experts,
//             gate/up product, swiglu, quantize, down product) into Dm_A;
//   second    a contiguous range of experts [e0, e1) is "the second server's" (a block of rows [at, at + nr) of the
//             layout): its q8_1 bytes go D2H into a mapped page with ONE 2D copy per group (the MMQ layout is
//             column-block major with the buffer's row count as the stride - Q8Layout below - so a group's rows are
//             re-pitched from the layer's row count to the group's), H2D into the server's xq_blk; the same groups
//             run there with the server's own arena (each group's rows its own buffer, total_rows = its rows,
//             relative bounds), the FP32 rows come back D2H through a mapped page (host_rows), as the engine does;
//   check     the server's rows memcmp'd against Dm_A's rows of the same experts, and the per-token weighted sum
//             (w[t, k] * row[slot[t, k]] in k order, the combine's order) computed on the host from each set:
//             "stessa somma per token".
// Every layer of every chunk prints OK or FAIL (the first differing row, its max |diff| and the max relative
// difference); the exit code is 0 only if every row of every layer is equal.
//
// The launch regime decides the bits (docs/PEER_PREFILL2.md s.5): mmq.cuh's launch_mul_mat_q runs stream-k (K split
// across CUDA blocks, partial sums + a fixup: another float order) when the launch's tiles fill the card's SM waves
// under 90 %, and one block per tile with K in sequence otherwise; the tile count is I-tiles x ntx x n_experts.  The
// primary's full groups of 16 fill the waves on the 3080 Ti (80 SMs) and the 3060 (28 SMs, 95 %); a group of 2-10
// experts does not.  So the server's launch is PADDED to 16 experts (the engine does the same: the repeated bounds give
// the padding experts no rows, their tiles are skipped), and the reference is made of full groups too (n_experts a
// multiple of 16, every expert routed: the default 32 with T 512 x K 10 rows).  Then two cards with the same MMQ
// configuration (the 3080 Ti and the 3060, both sm_86) give the same bits; a card of another architecture (the 2080 Ti,
// sm_75: other I / K tiles, dp4a) differs in the last bits from row 0, a fact about MMQ, not about the route.  The 3/11
// night run without the padding: FAIL from row 128 (= J, the second column tile) on every server group under 16
// experts, OK on the one group with 11 (91.7 %).  mmq::whole_tiles is printed per type pair: it asks the same of every
// visible card, so on this rig it says no because of the 2080 Ti even where the two sm_86 cards agree.
//
// The timing of the server's part (events on its stream: the H2D of the block, the products, the D2H of the rows) is
// printed per layer with the bytes moved, so the GB/s the cost model prices the route at can be read against a measure.
//
//   peer_prefill2_test [primary=0] [second=1] [T=512] [n_experts=32] [chunks=4] [seed=1] [gu_type d_type]
//
// Any two devices; the same device twice works (the route is then a loop through host memory).  A gu_type > 0 is a
// uniform model of 4 layers of (gu_type, d_type).  Exit 77 without MMQ in the build or without two CUDA devices.
//
// Written, not compiled (docs/PEER_PREFILL2.md s.7).  Not in the default build: `cmake --build <dir> --target
// peer_prefill2_test`.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mmq = strata::prefill::mmq;
using strata::kernels::NativeExpertLayout;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

namespace {

constexpr int64_t N = 2560;      // n_embd of Qwen3.8-Flash-Next
constexpr int64_t FF = 640;      // the expert's hidden width
constexpr int K = 10;            // routed experts a row
constexpr int MMQ_GROUP = 16;    // experts per MMQ launch (prefill.cpp)
constexpr size_t MMQ_TAIL = 4096;   // the zeroed tail after a group's last expert (prefill.cpp)

// the mixed model: the pack's ten (gate/up, down) pairs in their order of first appearance (tools/peer_fused_test.cu)
constexpr int kMixed[][2] = { {17, 20}, {16, 42}, {17, 42}, {21, 42}, {22, 42}, {21, 20}, {22, 20}, {16, 20}, {18, 20}, {18, 42} };
constexpr int kMixedLayers = (int) (sizeof kMixed / sizeof kMixed[0]);

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

// one layer of the synthetic model: its types, its layout, the GGUF block sizes, the MMQ matrix bytes
struct Layer {
    int gu = 0, d = 0;
    NativeExpertLayout L;
    size_t gu_blk = 0, d_blk = 0;
    size_t gub = 0, db = 0;   // mmq::matrix_bytes of gate+up ([1280, N]) and down ([N, FF])
    bool ok = true;           // the layout holds what the gather reads
};

// one synthetic blob at `b` in the layout: random bytes, then a finite, small fp16 scale at the start of every block
// of every gate, up and down row (the same generator as tools/peer_fused_test.cu: a .cu this .cpp cannot include)
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

// one card's half of the bench: its arena of blobs, its MMQ buffers, its context
struct Card {
    int dev = -1;
    cudaStream_t s = nullptr;
    std::vector<uint8_t*> arena;    // per layer: n_experts blobs, L.bytes apart (256-byte strides)
    std::vector<size_t> stride;
    uint8_t *grp_gu = nullptr, *grp_d = nullptr;
    void *xq = nullptr, *hq = nullptr;
    float *gu = nullptr, *h = nullptr, *dm = nullptr;
    int32_t *ident = nullptr, *bounds = nullptr;
    mmq::Context* ctx = nullptr;
    cudaEvent_t e0 = nullptr, e1 = nullptr;
};

void card_open(Card& c, int dev, const std::vector<Layer>& layers, int n_experts, int64_t rows_cap, size_t gu_max, size_t d_max,
               size_t xq_bytes, size_t hq_bytes) {
    c.dev = dev;
    CK(cudaSetDevice(dev));
    CK(cudaStreamCreateWithFlags(&c.s, cudaStreamNonBlocking));
    CK(cudaEventCreate(&c.e0));
    CK(cudaEventCreate(&c.e1));
    c.arena.resize(layers.size(), nullptr);
    c.stride.resize(layers.size(), 0);
    for (size_t l = 0; l < layers.size(); ++l) {
        if (!layers[l].ok) continue;
        c.stride[l] = (layers[l].L.bytes + 255) / 256 * 256;
        CK(cudaMalloc((void**) &c.arena[l], c.stride[l] * (size_t) n_experts));
    }
    CK(cudaMalloc((void**) &c.grp_gu, (size_t) MMQ_GROUP * gu_max + MMQ_TAIL));
    CK(cudaMalloc((void**) &c.grp_d, (size_t) MMQ_GROUP * d_max + MMQ_TAIL));
    CK(cudaMalloc(&c.xq, xq_bytes + MMQ_TAIL));
    CK(cudaMalloc(&c.hq, hq_bytes + MMQ_TAIL));
    CK(cudaMalloc((void**) &c.gu, (size_t) rows_cap * 1280 * 4));
    CK(cudaMalloc((void**) &c.h, (size_t) rows_cap * FF * 4));
    CK(cudaMalloc((void**) &c.dm, (size_t) rows_cap * N * 4));
    CK(cudaMalloc((void**) &c.ident, (size_t) (rows_cap + 1024) * 4));   // slack for the padded launch's fixup ids (prefill.cpp set_peer2)
    CK(cudaMalloc((void**) &c.bounds, (size_t) (2 * (1024 + 1024 / MMQ_GROUP + 2)) * 4));
    c.ctx = new mmq::Context();
    mmq::iota(c.ident, rows_cap + 1024, c.s);
    CK(cudaStreamSynchronize(c.s));
}

void card_close(Card& c) {
    if (c.dev < 0) return;
    CK(cudaSetDevice(c.dev));
    CK(cudaStreamSynchronize(c.s));
    delete c.ctx;
    for (uint8_t* a : c.arena) if (a) cudaFree(a);
    cudaFree(c.grp_gu); cudaFree(c.grp_d); cudaFree(c.xq); cudaFree(c.hq); cudaFree(c.gu); cudaFree(c.h); cudaFree(c.dm);
    cudaFree(c.ident); cudaFree(c.bounds);
    cudaEventDestroy(c.e0); cudaEventDestroy(c.e1);
    cudaStreamDestroy(c.s);
    c.dev = -1;
}

// the groups the engine makes of `ord` (prefill.cpp peer_layer, compact: up to MMQ_GROUP experts; the row cap G is
// the chunk and is never hit here)
std::vector<std::pair<size_t, size_t>> groups_of(size_t n) {
    std::vector<std::pair<size_t, size_t>> groups;
    for (size_t g0 = 0; g0 < n; g0 += MMQ_GROUP) groups.push_back({g0, std::min(n, g0 + MMQ_GROUP)});
    return groups;
}

// The MMQ q8 layout (moe_mmq.cu, llama.cpp's quantize_mmq_q8_1; docs/PEER_PREFILL2.md s.1): a buffer quantized with
// `rows` rows holds block (kb, r) - the 128 values [kb * 128, +128) of row r, block_q8_1_mmq, `blk` bytes - at
// index kb * rows + r (COLUMN-BLOCK MAJOR, the row count as the stride), kb < pad512(N) / 128 column blocks, plus a
// fixed tail of 128 blocks; a product reads it with the same row count as total_rows.
struct Q8Layout {
    size_t blk = 0, rb = 0;   // bytes of one block, bytes of one row over all its column blocks
    int64_t kb = 0;           // column blocks a row
    size_t tail = 0;          // the fixed tail (bytes)
};
bool q8_layout(Q8Layout& q) {
    q.tail = mmq::q8_bytes(0, N);
    q.rb = mmq::q8_bytes(2, N) - mmq::q8_bytes(1, N);
    q.blk = q.tail / 128;
    if (q.tail == 0 || q.tail % 128 != 0 || q.rb == 0 || q.blk == 0 || q.rb % q.blk != 0 || mmq::q8_bytes(1, N) != q.tail + q.rb ||
        mmq::q8_bytes(3, N) - mmq::q8_bytes(2, N) != q.rb)
        return false;
    q.kb = (int64_t) (q.rb / q.blk);
    return true;
}

// the engine's per-group sequence (prefill.cpp peer_layer, compact): experts `ord` of layer `l` with their rows
// [bounds_h[j], bounds_h[j + 1]) (n + 1 of them), the FP32 rows into dm (row r at dm + r * N): gather_native of
// each blob, the zeroed tails, gate/up product, swiglu, quantize, down product - the same calls with the same
// arguments on either card.  `whole` (the primary's way, prefill.cpp's own MMQ path): `xq` is the whole layer's
// q8 buffer quantized with `rows_total` rows, the gate/up product reads it with total_rows = rows_total and the
// group's ABSOLUTE bounds; otherwise (the server's way) each group's rows are their own buffer at xq + r0 * rb
// (re-pitched to the group by the caller's 2D copy), read with total_rows = the group's rows and RELATIVE bounds.
void run_groups_l(Card& c, const std::vector<Layer>& layers, size_t l, const std::vector<int32_t>& ord,
                  const std::vector<int32_t>& cnt, const std::vector<int32_t>& bounds_h, const void* xq, const Q8Layout& q,
                  bool whole, int64_t rows_total, float* dm) {
    const Layer& ly = layers[l];
    CK(cudaSetDevice(c.dev));
    const size_t n = ord.size();
    const std::vector<std::pair<size_t, size_t>> groups = groups_of(n);
    const size_t ng = groups.size();
    std::vector<int32_t> bh(n + 1 + ng * (MMQ_GROUP + 1));
    for (size_t j = 0; j <= n; ++j) bh[j] = bounds_h[j];
    for (size_t g2 = 0; g2 < ng; ++g2)
        for (size_t i = 0; i <= (size_t) MMQ_GROUP; ++i)
            bh[n + 1 + g2 * (MMQ_GROUP + 1) + i] = bh[std::min(groups[g2].second, groups[g2].first + i)] - bh[groups[g2].first];
    CK(cudaMemcpyAsync(c.bounds, bh.data(), bh.size() * 4, cudaMemcpyHostToDevice, c.s));
    for (size_t g2 = 0; g2 < ng; ++g2) {
        const size_t j0 = groups[g2].first, j1 = groups[g2].second;
        const int ngx = (int) (j1 - j0);
        int64_t maxr = 0;
        for (size_t j = j0; j < j1; ++j) {
            const int32_t e = ord[j];
            maxr = std::max<int64_t>(maxr, cnt[(size_t) e]);
            const uint8_t* bd = c.arena[l] + c.stride[l] * (size_t) e;
            const size_t q = j - j0;
            mmq::gather_native(bd, bd + ly.L.up_off, ly.gub / 2, bd + ly.L.down_off, ly.db,
                               c.grp_gu + q * ly.gub, c.grp_d + q * ly.db, c.s);
        }
        const int64_t r0 = bh[j0], nr = bh[j1] - r0;
        if (nr <= 0) continue;
        const int32_t* rel = c.bounds + n + 1 + g2 * (MMQ_GROUP + 1);
        CK(cudaMemsetAsync(c.grp_gu + (size_t) ngx * ly.gub, 0, MMQ_TAIL, c.s));
        CK(cudaMemsetAsync(c.grp_d + (size_t) ngx * ly.db, 0, MMQ_TAIL, c.s));
        // the server's launch is padded to MMQ_GROUP experts (prefill.cpp peer_layer, rows_in: the repeated relative
        // bounds give the padding experts col_diff 0, their tiles are skipped), so its tile count is a full group's and
        // the launch runs in the tiling regime (K in sequence) wherever the primary's full groups do; the primary's own
        // launch keeps its real count, as the engine's does
        const int n_launch = whole ? ngx : MMQ_GROUP;
        mmq::Product gu;
        gu.w = c.grp_gu; gu.type = ly.gu; gu.w_rows = 1280; gu.w_cols = N; gu.expert_bytes = ly.gub;
        gu.n = n_launch; gu.ids = c.ident; gu.max_rows = maxr; gu.ld_dst = 1280;
        if (whole) {   // the primary: the layer's buffer, absolute bounds, its rows into GU at their own rows
            gu.xq = xq; gu.bounds = c.bounds + j0; gu.total_rows = rows_total; gu.dst = c.gu;
        } else {       // the server: the group's own buffer and rows
            gu.xq = (const uint8_t*) xq + (size_t) r0 * q.rb; gu.bounds = rel; gu.total_rows = nr; gu.dst = c.gu;
        }
        c.ctx->run(gu, c.s);
        mmq::swiglu(c.gu + (whole ? (size_t) r0 * 1280 : 0), c.h, nr, FF, false, c.s);
        mmq::quantize(c.h, nullptr, c.hq, ly.d, FF, FF, nr, c.s);
        mmq::Product dn;
        dn.w = c.grp_d; dn.type = ly.d; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = ly.db;
        dn.n = n_launch; dn.xq = c.hq; dn.bounds = rel; dn.ids = c.ident; dn.total_rows = nr; dn.max_rows = maxr;
        dn.dst = dm + (size_t) r0 * N; dn.ld_dst = N;
        c.ctx->run(dn, c.s);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int primary = argc > 1 ? std::atoi(argv[1]) : 0;
    int second = argc > 2 ? std::atoi(argv[2]) : 1;
    const int64_t T = argc > 3 ? std::atoll(argv[3]) : 512;
    const int n_experts = argc > 4 ? std::atoi(argv[4]) : 32;   // a multiple of MMQ_GROUP: the primary's groups all full (s.5)
    const int chunks = argc > 5 ? std::atoi(argv[5]) : 4;
    const uint64_t seed = argc > 6 ? (uint64_t) std::strtoull(argv[6], nullptr, 10) : 1;
    const int gu_type = argc > 7 ? std::atoi(argv[7]) : 0;
    const int d_type = argc > 8 ? std::atoi(argv[8]) : 42;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { std::printf("no CUDA device\n"); return 77; }
    if (!mmq::built()) { std::printf("this build has no MMQ prompt path\n"); return 77; }
    if (primary >= ndev || second >= ndev) { std::printf("devices %d and %d: %d visible\n", primary, second, ndev); return 77; }
    if (n_experts < K || n_experts > 1024) { std::printf("n_experts in [%d, 1024]\n", K); return 1; }

    // ---- the synthetic model
    std::vector<Layer> layers;
    if (gu_type == 0) {
        for (int l = 0; l < kMixedLayers; ++l) { Layer ly; ly.gu = kMixed[l][0]; ly.d = kMixed[l][1]; layers.push_back(ly); }
    } else {
        for (int l = 0; l < 4; ++l) { Layer ly; ly.gu = gu_type; ly.d = d_type; layers.push_back(ly); }
    }
    size_t gu_max = 0, d_max = 0;
    {
        cudaDeviceProp p, q;
        CK(cudaGetDeviceProperties(&p, primary));
        CK(cudaGetDeviceProperties(&q, second));
        std::printf("peer_prefill2_test: primary CUDA%d %s sm_%d%d SMs %d | second server CUDA%d %s sm_%d%d SMs %d | T %lld, %d experts, "
                    "%d chunks, seed %llu\n", primary, p.name, p.major, p.minor, p.multiProcessorCount, second, q.name, q.major,
                    q.minor, q.multiProcessorCount, (long long) T, n_experts, chunks, (unsigned long long) seed);
    }
    for (size_t l = 0; l < layers.size(); ++l) {
        Layer& ly = layers[l];
        ly.L = strata::kernels::native_expert_layout(ly.gu, ly.d, N, FF);
        const int gu_qk = qk_of(ly.gu), d_qk = qk_of(ly.d);
        if (gu_qk == 0 || d_qk == 0 || ly.L.bytes == 0) {
            std::printf("layer %zu: the generator knows gate/up 16 / 17 / 18 / 21 / 22 and down 42 / 20 only (%d / %d): skipped\n", l, ly.gu, ly.d);
            ly.ok = false;
            continue;
        }
        ly.gu_blk = strata::kernels::iq_row_bytes(ly.gu, gu_qk);
        ly.d_blk = strata::kernels::iq_row_bytes(ly.d, d_qk);
        ly.gub = mmq::matrix_bytes(ly.gu, 1280, N);
        ly.db = mmq::matrix_bytes(ly.d, N, FF);
        if (!mmq::supported(ly.gu) || !mmq::supported(ly.d) || !mmq::fits(ly.gu, 1280) || !mmq::fits(ly.d, N)) {
            std::printf("layer %zu: MMQ has no tile for %d / %d on a visible card: skipped\n", l, ly.gu, ly.d);
            ly.ok = false;
            continue;
        }
        if (ly.L.up_off < ly.gub / 2 || ly.L.down_off + ly.db > ly.L.bytes) {
            std::printf("layer %zu: the layout (%zu / %zu / %zu bytes) does not hold the MMQ matrices (%zu / %zu): skipped\n", l,
                        ly.L.up_off, ly.L.down_off, ly.L.bytes, ly.gub, ly.db);
            ly.ok = false;
            continue;
        }
        gu_max = std::max(gu_max, ly.gub);
        d_max = std::max(d_max, ly.db);
        std::printf("layer %zu: gate/up %d down %d, blob %zu B; whole_tiles gate/up %s, down %s\n", l, ly.gu, ly.d, ly.L.bytes,
                    mmq::whole_tiles(ly.gu, 1280, MMQ_GROUP) ? "yes" : "NO (stream-k: another SM count may sum in another order)",
                    mmq::whole_tiles(ly.d, N, MMQ_GROUP) ? "yes" : "NO");
    }
    const int64_t rows_cap = T * K;
    Q8Layout q;
    if (!q8_layout(q)) {
        std::printf("FAIL the MMQ q8 layout is not rows x row bytes + a fixed tail (q8_bytes 0/1/2 rows: %zu, %zu, %zu)\n",
                    mmq::q8_bytes(0, N), mmq::q8_bytes(1, N), mmq::q8_bytes(2, N));
        return 1;
    }
    std::printf("q8 layout: %zu B a block, %lld column blocks a row, %zu B a row, tail %zu B (column-block major: block (kb, r) at kb * rows + r)\n",
                q.blk, (long long) q.kb, q.rb, q.tail);
    const size_t xq_bytes = mmq::q8_bytes(rows_cap, N), hq_bytes = mmq::q8_bytes(rows_cap, FF);

    // ---- the two cards: the same blobs slot for slot
    Card A, B;
    card_open(A, primary, layers, n_experts, rows_cap, gu_max, d_max, xq_bytes, hq_bytes);
    card_open(B, second, layers, n_experts, rows_cap, gu_max, d_max, xq_bytes, hq_bytes);
    {
        Rng rng(seed * 7919 + 11);
        std::vector<uint8_t> blob;
        for (size_t l = 0; l < layers.size(); ++l) {
            if (!layers[l].ok) continue;
            blob.assign(layers[l].L.bytes, 0);
            for (int e = 0; e < n_experts; ++e) {
                fill_blob(blob.data(), layers[l], rng);
                CK(cudaSetDevice(A.dev));
                CK(cudaMemcpy(A.arena[l] + A.stride[l] * (size_t) e, blob.data(), blob.size(), cudaMemcpyHostToDevice));
                CK(cudaSetDevice(B.dev));
                CK(cudaMemcpy(B.arena[l] + B.stride[l] * (size_t) e, blob.data(), blob.size(), cudaMemcpyHostToDevice));
            }
        }
    }
    // the route's mapped pages (Portable | Mapped, as the engine's): the q8_1 block in, the FP32 rows out
    uint8_t* host_xq = nullptr;
    float* host_rows = nullptr;
    CK(cudaHostAlloc((void**) &host_xq, xq_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    CK(cudaHostAlloc((void**) &host_rows, (size_t) rows_cap * N * 4, cudaHostAllocPortable | cudaHostAllocMapped));
    // the primary's activations and routing tables
    float* x_dev = nullptr;
    int32_t* src_dev = nullptr;
    CK(cudaSetDevice(A.dev));
    CK(cudaMalloc((void**) &x_dev, (size_t) T * N * 4));
    CK(cudaMalloc((void**) &src_dev, (size_t) rows_cap * 4));

    Rng rng(seed);
    std::vector<float> x((size_t) T * N), w((size_t) T * K);
    std::vector<int32_t> ids((size_t) T * K), cnt((size_t) n_experts), off((size_t) n_experts + 1), src((size_t) rows_cap),
                         slot((size_t) rows_cap);
    std::vector<float> dm_a((size_t) rows_cap * N), sum_a((size_t) T * N), sum_b((size_t) T * N);
    int fails = 0;
    for (int ch = 0; ch < chunks; ++ch) {
        // the chunk's activations (a few huge values, as the fused bench does) and its routing: K distinct experts a token
        for (auto& v : x) v = rng.below(4096) == 0 ? 1e4f : rng.uni(-8.0f, 8.0f);
        for (int64_t t = 0; t < T; ++t) {
            std::vector<int32_t> pick;
            while ((int) pick.size() < K) {
                const int32_t e = (int32_t) rng.below((uint32_t) n_experts);
                if (std::find(pick.begin(), pick.end(), e) == pick.end()) pick.push_back(e);
            }
            float ws = 0.0f;
            for (int k = 0; k < K; ++k) { ids[(size_t) (t * K + k)] = pick[(size_t) k]; w[(size_t) (t * K + k)] = rng.uni(0.05f, 1.0f); ws += w[(size_t) (t * K + k)]; }
            for (int k = 0; k < K; ++k) w[(size_t) (t * K + k)] /= ws;
        }
        // the primary's row layout: expert-id order (prefill.cpp: off[e], slot = inverse of src)
        std::fill(cnt.begin(), cnt.end(), 0);
        for (int32_t e : ids) ++cnt[(size_t) e];
        {   // the reference's groups must be full (s.5): every expert routed and n_experts a multiple of MMQ_GROUP
            int routed = 0;
            for (int e = 0; e < n_experts; ++e) routed += cnt[(size_t) e] > 0;
            if (routed % MMQ_GROUP != 0)
                std::printf("note: chunk %d routes %d experts, not a multiple of %d: the primary's last group is partial and its "
                            "launch regime may be stream-k (s.5) - a FAIL confined to that group's experts is expected then\n",
                            ch, routed, MMQ_GROUP);
        }
        off[0] = 0;
        for (int e = 0; e < n_experts; ++e) off[(size_t) e + 1] = off[(size_t) e] + cnt[(size_t) e];
        {
            std::vector<int32_t> fill(off.begin(), off.end() - 1);
            for (int64_t i = 0; i < rows_cap; ++i) {
                const int32_t e = ids[(size_t) i];
                const int32_t p = fill[(size_t) e]++;
                slot[(size_t) i] = p;
                src[(size_t) p] = (int32_t) (i / K);
            }
        }
        CK(cudaSetDevice(A.dev));
        CK(cudaMemcpyAsync(x_dev, x.data(), x.size() * 4, cudaMemcpyHostToDevice, A.s));
        CK(cudaMemcpyAsync(src_dev, src.data(), (size_t) rows_cap * 4, cudaMemcpyHostToDevice, A.s));
        // the second server's experts this chunk: a contiguous id range (its rows are one block of the layout)
        const int e0 = (int) rng.below((uint32_t) (n_experts / 2)), e1 = std::min(n_experts, e0 + 1 + (int) rng.below((uint32_t) (n_experts / 2)));
        const int64_t at = off[(size_t) e0], nr_b = off[(size_t) e1] - at;
        for (size_t l = 0; l < layers.size(); ++l) {
            if (!layers[l].ok) continue;
            const Layer& ly = layers[l];
            // primary: the chunk's q8_1 rows in expert order (the engine: quantize_scatter, the same bytes) and the reference
            CK(cudaSetDevice(A.dev));
            mmq::quantize(x_dev, src_dev, A.xq, ly.gu, N, N, rows_cap, A.s);
            std::vector<int32_t> ord_all;
            for (int e = 0; e < n_experts; ++e) if (cnt[(size_t) e] > 0) ord_all.push_back(e);
            std::vector<int32_t> bounds_all(ord_all.size() + 1);
            for (size_t j = 0; j < ord_all.size(); ++j) bounds_all[j] = off[(size_t) ord_all[j]];
            bounds_all[ord_all.size()] = (int32_t) rows_cap;
            run_groups_l(A, layers, l, ord_all, cnt, bounds_all, A.xq, q, true, rows_cap, A.dm);
            CK(cudaMemcpyAsync(dm_a.data(), A.dm, (size_t) rows_cap * N * 4, cudaMemcpyDeviceToHost, A.s));
            // second server: its experts, its groups, its block-relative bounds
            std::vector<int32_t> ord_b;
            for (int e = e0; e < e1; ++e) if (cnt[(size_t) e] > 0) ord_b.push_back(e);
            std::vector<int32_t> bounds_b(ord_b.size() + 1);
            for (size_t j = 0; j < ord_b.size(); ++j) bounds_b[j] = off[(size_t) ord_b[j]] - (int32_t) at;
            bounds_b[ord_b.size()] = (int32_t) nr_b;
            // the block's bytes to the mapped page: ONE 2D copy per group, the group's rows re-pitched from the
            // layer's row count to the group's (the engine's peer_layer, rows_in)
            for (const auto& grp : groups_of(ord_b.size())) {
                const int64_t r0 = bounds_b[grp.first], nr = bounds_b[grp.second] - r0;
                if (nr <= 0) continue;
                CK(cudaMemcpy2DAsync(host_xq + (size_t) r0 * q.rb, (size_t) nr * q.blk,
                                     (const uint8_t*) A.xq + (size_t) (at + r0) * q.blk, (size_t) rows_cap * q.blk,
                                     (size_t) nr * q.blk, (size_t) q.kb, cudaMemcpyDeviceToHost, A.s));
            }
            CK(cudaStreamSynchronize(A.s));
            // second server: the block's rows from the page, the same groups, the rows back through the page
            CK(cudaSetDevice(B.dev));
            CK(cudaEventRecord(B.e0, B.s));
            CK(cudaMemcpyAsync(B.xq, host_xq, (size_t) nr_b * q.rb, cudaMemcpyHostToDevice, B.s));
            run_groups_l(B, layers, l, ord_b, cnt, bounds_b, B.xq, q, false, nr_b, B.dm);
            CK(cudaMemcpyAsync(host_rows, B.dm, (size_t) nr_b * N * 4, cudaMemcpyDeviceToHost, B.s));
            CK(cudaEventRecord(B.e1, B.s));
            CK(cudaStreamSynchronize(B.s));
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, B.e0, B.e1));
            // the rows, bit for bit (and, when they differ, the largest difference relative to the value: a last-bit
            // difference on the 1e9-1e11 values the 1e4 activations make is 1e2-1e4 absolute and ~1e-7 relative)
            int64_t bad = -1;
            double maxd = 0.0, maxrel = 0.0;
            for (int64_t r = 0; r < nr_b; ++r) {
                const float* a = dm_a.data() + (size_t) (at + r) * N;
                const float* b = host_rows + (size_t) r * N;
                if (std::memcmp(a, b, (size_t) N * 4) != 0) {
                    if (bad < 0) bad = r;
                    for (int64_t i = 0; i < N; ++i) {
                        const double d = std::fabs((double) a[i] - (double) b[i]);
                        maxd = std::max(maxd, d);
                        if (d > 0.0) maxrel = std::max(maxrel, d / std::max(1e-30, std::fabs((double) a[i])));
                    }
                }
            }
            // the per-token weighted sums in the combine's k order, from each row set (the block's rows from the server)
            std::fill(sum_a.begin(), sum_a.end(), 0.0f);
            std::fill(sum_b.begin(), sum_b.end(), 0.0f);
            for (int64_t t = 0; t < T; ++t)
                for (int k = 0; k < K; ++k) {
                    const int32_t r = slot[(size_t) (t * K + k)];
                    const float wk = w[(size_t) (t * K + k)];
                    const float* ra = dm_a.data() + (size_t) r * N;
                    const float* rbp = r >= at && r < at + nr_b ? host_rows + (size_t) (r - at) * N : ra;
                    float* sa = sum_a.data() + (size_t) t * N;
                    float* sb = sum_b.data() + (size_t) t * N;
                    for (int64_t i = 0; i < N; ++i) { sa[i] += wk * ra[i]; sb[i] += wk * rbp[i]; }
                }
            const bool sums_eq = std::memcmp(sum_a.data(), sum_b.data(), sum_a.size() * 4) == 0;
            const double mb = ((double) nr_b * (double) q.rb + (double) nr_b * (double) N * 4) / 1048576.0;
            if (bad < 0 && sums_eq) {
                std::printf("OK   chunk %d layer %zu (%d/%d): experts [%d, %d) %zu of them, %lld rows: rows and per-token sums identical; "
                            "server %.2f ms for %.1f MiB over its link (%.2f GB/s incl. the products)\n", ch, l, ly.gu, ly.d, e0, e1,
                            ord_b.size(), (long long) nr_b, ms, mb, ms > 0 ? mb / 1024.0 / (ms * 1e-3) : 0.0);
            } else {
                ++fails;
                std::printf("FAIL chunk %d layer %zu (%d/%d): experts [%d, %d), %lld rows: %s (first differing row %lld, max |diff| %.3g, "
                            "max relative %.3g%s)%s\n",
                            ch, l, ly.gu, ly.d, e0, e1, (long long) nr_b, bad >= 0 ? "rows differ" : "rows equal", (long long) bad, maxd,
                            maxrel, maxrel > 0.0 && maxrel < 1e-5 ? ": last bits, another float order" : "",
                            sums_eq ? "" : "; per-token sums differ");
            }
        }
    }
    std::printf("%s: %d chunk-layers failed\n", fails == 0 ? "PASS" : "FAIL", fails);
    CK(cudaSetDevice(A.dev));
    cudaFree(x_dev);
    cudaFree(src_dev);
    cudaFreeHost(host_xq);
    cudaFreeHost(host_rows);
    card_close(A);
    card_close(B);
    return fails == 0 ? 0 : 1;
}
