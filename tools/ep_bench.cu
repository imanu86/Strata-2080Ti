// tools/ep_bench.cu - expert-parallel between the 3080 Ti and the 2080 Ti with nothing but the GPUs on the path.
//
// The question (docs/PIPELINE3.md): with the 3080 Ti as the owner of every layer (dense, attention, router, its own
// experts, combine) and the 2080 Ti as a server of experts, how many microseconds of a layer does the exchange expose
// when the server is a PERSISTENT kernel that waits on a flag in mapped host memory, reads the layer's input, computes
// its experts and writes the rows back with a flag - against today's --peer-device path, where the host thread launches
// the server's share layer by layer and spins on its stream (peer_experts.cpp: launch / finish)?  No P2P under WDDM
// (Ampere + Turing), so every hop crosses pinned, mapped host memory: the 3080 Ti over Oculink PCIe 4.0 x4, the 2080 Ti
// over x16.
//
// Per layer the owner runs the engine's form (a CUDA graph of 48 layers per window):
//   dense (a bandwidth read of `dense` MiB)  ->  push x (q8_1 rows + ids, flag)  ->  own experts (n_loc blobs)
//   ->  wait the rows' flag, copy the rows in
// and the server, in one launch per window:
//   wait x's flag  ->  copy x in  ->  its experts (n_srv blobs, every block)  ->  rows to mapped host (the last block
//   raises the flag).
// The blobs are read in full with 16-byte loads (the expert kernels are bandwidth-bound: IQ3_XXS blobs of ~1.7 MB);
// the rows are one n_embd fp32 row per routed entry (10 KB), as the combine of today wants, or bf16 (5 KB), or one
// weighted row per token ("summed on the server": the combine's order changes, so the bits do too).
//
// Four forms per configuration, median over the windows, us per layer:
//   floor    the owner alone with only its own experts (the server's share vanishes): what no exchange can beat
//   all      the owner alone computing every expert itself (the server's blobs too): one card, no exchange
//   persist  the server as a persistent kernel, device flags both ways (the design to build)
//   host     the server driven by the host per layer as today: spin on x's flag, H2D copy, kernel launch, spin on the
//            stream, then the host raises the rows' flag (the engine's launch + finish, OnDevice switches included)
//   host-ns  the same without the device switch around the stream spin (STRATA_PEER_NOSWITCH)
// exposed = form - floor.
//
//   nvcc -O2 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -o ep_bench tools/ep_bench.cu
//   ep_bench [windows=40] [dense_mib=62] [blob_kib=1728]        (under bench_lock.py: it owns both GPUs for ~30 s)
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)

constexpr int H = 2560;                 // n_embd of Qwen3.8-Flash-Next
constexpr int ROW16 = H * 4 / 16;       // uint4 per fp32 row
constexpr int CHUNK = 64 * 1024;        // bytes of a blob one block reads at a time
constexpr int L = 48;
constexpr uint64_t WAIT_NS = 200000000ull;   // a stuck hop must not outlive the WDDM watchdog

__device__ __forceinline__ uint64_t gtime() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// bounded wait on a flag (mapped host memory or VRAM)
__device__ __forceinline__ bool wait_ge(const volatile uint32_t* f, uint32_t v, volatile uint32_t* err) {
    const uint64_t t0 = gtime();
    while (*f < v) {
        __nanosleep(64);
        if (gtime() - t0 > WAIT_NS) { *err = 1; return false; }
    }
    __threadfence_system();
    return true;
}

__device__ __forceinline__ uint32_t val(const uint32_t* epoch, int l) { return (*epoch << 8) | (uint32_t) (l + 1); }

__global__ void k_bump(uint32_t* epoch) { *epoch += 1; }

// one block reads CHUNK bytes at p (8 loads in flight per thread) and folds them into acc
__device__ __forceinline__ void read_chunk(const uint4* p, uint32_t& acc) {
    constexpr int N16 = CHUNK / 16;
    for (int base = 0; base < N16; base += 256 * 8) {
        uint4 v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = p[base + u * 256 + threadIdx.x];
#pragma unroll
        for (int u = 0; u < 8; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
}

// the dense part of a layer: `n_chunks` chunks of the pool from `off`, grid-stride
__global__ void k_dense(const uint8_t* pool, size_t pool_bytes, size_t off, int n_chunks, uint32_t* sink) {
    uint32_t acc = 0;
    for (int c = blockIdx.x; c < n_chunks; c += gridDim.x)
        read_chunk((const uint4*) (pool + (off + (size_t) c * CHUNK) % pool_bytes), acc);
    if (acc == 0xFFFFFFFFu) *sink = acc;
}

// `chunks` chunks per expert; the expert's rows for its `ent` entries get this chunk's slice.  `sum_rows` > 0: the
// rows go back summed per token instead (block 0 writes them after its items)
__device__ __forceinline__ void expert_items(const uint8_t* pool, size_t pool_bytes, size_t off, int n_exp, int chunks, int ent,
                                             int sum_rows, float4* rows, bool wt, int row16, uint32_t& acc) {
    const int items = n_exp * chunks;
    const int slice = (row16 + chunks - 1) / chunks;   // uint4 of a row per chunk
    for (int it = blockIdx.x; it < items; it += gridDim.x) {
        const int e = it / chunks, c = it % chunks;
        read_chunk((const uint4*) (pool + (off + ((size_t) e * chunks + c) * CHUNK) % pool_bytes), acc);
        const int s0 = c * slice, s1 = min(row16, s0 + slice);
        const float4 fv = make_float4(__uint_as_float(acc | 0x3f800000u), 1.0f, 2.0f, 3.0f);
        for (int t = 0; t < ent; ++t) {
            float4* r = rows + (size_t) (e * ent + t) * row16;
            for (int i = s0 + (int) threadIdx.x; i < s1; i += blockDim.x) {
                if (wt) __stwt(r + i, fv); else r[i] = fv;
            }
        }
    }
    if (sum_rows > 0 && blockIdx.x == 0) {
        const float4 fv = make_float4(__uint_as_float(acc | 0x3f800000u), 1.0f, 2.0f, 3.0f);
        for (int i = threadIdx.x; i < sum_rows * row16; i += blockDim.x) {
            if (wt) __stwt(rows + i, fv); else rows[i] = fv;
        }
    }
}

// the owner's experts, rows into VRAM
__global__ void k_experts(const uint8_t* pool, size_t pool_bytes, size_t off, int n_exp, int chunks, int ent, float4* rows, int row16,
                          uint32_t* sink) {
    uint32_t acc = 0;
    expert_items(pool, pool_bytes, off, n_exp, chunks, ent, 0, rows, false, row16, acc);
    if (acc == 0xFFFFFFFFu) *sink = acc;
}

// the server's experts as a plain kernel (today's form): rows straight into mapped host memory, no flag
__global__ void k_experts_host(const uint8_t* pool, size_t pool_bytes, size_t off, int n_exp, int chunks, int ent, int sum_rows,
                               float4* rows, int row16, uint32_t* sink) {
    uint32_t acc = 0;
    expert_items(pool, pool_bytes, off, n_exp, chunks, ent, sum_rows, rows, true, row16, acc);
    if (acc == 0xFFFFFFFFu) *sink = acc;
}

// push: x to the mapped inbox (one block), then the flag
__global__ void k_push(volatile uint4* box, const uint4* src, int n16, volatile uint32_t* flag, const uint32_t* epoch, int l) {
    for (int i = threadIdx.x; i < n16; i += blockDim.x) __stwt((uint4*) box + i, src[i]);
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); *flag = val(epoch, l); }
}

// wait folded into the copy (the engine's fused wait), but one poller on the link: block 0 polls the mapped flag and
// releases the other blocks through a VRAM word; then every block copies its share of the rows in.
// stamps[epoch-1][l] = ns block 0 spent waiting
__global__ void k_wait_copy(uint4* dst, const volatile uint4* src, int n16, const volatile uint32_t* flag, volatile uint32_t* lready,
                            const uint32_t* epoch, int l, uint64_t* stamps, volatile uint32_t* err) {
    __shared__ int ok;
    const uint32_t v = val(epoch, l);
    if (threadIdx.x == 0) {
        const uint64_t t0 = gtime();
        if (blockIdx.x == 0) {
            ok = wait_ge(flag, v, err);
            stamps[(size_t) (*epoch - 1) * L + l] = gtime() - t0;
            *lready = v;
        } else {
            ok = 1;
            while (*lready < v) {
                __nanosleep(64);
                if (gtime() - t0 > WAIT_NS) { ok = 0; break; }
            }
        }
    }
    __syncthreads();
    if (!ok) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) dst[i] = __ldcv((const uint4*) src + i);
}

// the server, persistent over a window: block 0 waits x and copies it into VRAM, releases the other blocks through a
// VRAM word; every block computes expert items and writes its slices of the rows to mapped host memory; the last block
// to finish raises the rows' flag.  No grid barrier: a block running ahead waits for the next x, which block 0 only
// sees after the owner consumed this layer's rows.
__global__ void k_server(const volatile uint32_t* flag_x, volatile uint32_t* flag_r, const volatile uint4* box_x, int n16_x, uint4* d_x,
                         volatile uint32_t* xready, unsigned* done, const uint8_t* pool, size_t pool_bytes, int n_exp, int chunks,
                         int ent, int sum_rows, float4* rows, int row16, uint32_t base, volatile uint32_t* err, uint32_t* sink) {
    __shared__ int ok;
    uint32_t acc = 0;
    for (int l = 0; l < L; ++l) {
        const uint32_t v = base | (uint32_t) (l + 1);
        if (blockIdx.x == 0) {
            if (threadIdx.x == 0) ok = wait_ge(flag_x, v, err);
            __syncthreads();
            if (!ok) return;
            for (int i = threadIdx.x; i < n16_x; i += blockDim.x) d_x[i] = __ldcv((const uint4*) box_x + i);
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) *xready = v;
        } else {   // the other blocks poll a VRAM word, never the link
            if (threadIdx.x == 0) {
                const uint64_t t0 = gtime();
                ok = 1;
                while (*xready < v) {
                    __nanosleep(64);
                    if (gtime() - t0 > WAIT_NS) { ok = 0; break; }
                }
            }
            __syncthreads();
            if (!ok) return;
        }
        expert_items(pool, pool_bytes, (size_t) l * n_exp * chunks * CHUNK, n_exp, chunks, ent, sum_rows, rows, true, row16, acc);
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned d = atomicAdd(done, 1u);
            if (d == gridDim.x - 1) { *done = 0; __threadfence_system(); *flag_r = v; }
        }
    }
    if (acc == 0xFFFFFFFFu) *sink = acc;
}

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5))];
}

struct Dev {
    int id = -1, sms = 0;
    cudaStream_t s = nullptr;
    uint8_t* pool = nullptr;      // weights stand-in
    size_t pool_bytes = 0;
    uint32_t* epoch = nullptr;
    unsigned* done = nullptr;
    uint32_t* sink = nullptr;
    uint32_t* xready = nullptr;
    uint4* d_x = nullptr;
    float4* rows = nullptr;       // VRAM rows (owner: its own and the copied-in ones)
    uint64_t* stamps = nullptr;
};

struct Cfg {
    const char* what;
    int T, n_loc, ent_loc, n_srv, ent_srv, row_bytes;
};

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 40;
    const int dense_mib = argc > 2 ? std::atoi(argv[2]) : 62;
    const int blob_kib = argc > 3 ? std::atoi(argv[3]) : 1728;
    const int chunks = (blob_kib * 1024 + CHUNK - 1) / CHUNK;
    const int dense_chunks = dense_mib * 1024 * 1024 / CHUNK;
    int nd = 0;
    CK(cudaGetDeviceCount(&nd));
    int io = -1, is = -1;
    for (int d = 0; d < nd; ++d) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, d));
        std::printf("CUDA%d %s sm_%d%d pci %02x:%02x SMs %d\n", d, p.name, p.major, p.minor, p.pciBusID, p.pciDeviceID, p.multiProcessorCount);
        if (std::strstr(p.name, "3080")) io = d;
        if (std::strstr(p.name, "2080")) is = d;
    }
    if (io < 0 || is < 0) { std::printf("need a 3080 Ti (owner) and a 2080 Ti (server)\n"); return 1; }
    int p01 = 0, p10 = 0;
    cudaDeviceCanAccessPeer(&p01, io, is);
    cudaDeviceCanAccessPeer(&p10, is, io);
    std::printf("peer access owner->server %d, server->owner %d; %d windows x %d layers, dense %d MiB, blob %d KiB (%d chunks)\n",
                p01, p10, W, L, dense_mib, blob_kib, chunks);

    // mapped host memory: flags on their own lines, the x inbox, the rows inbox
    const size_t kRowsMax = 1 << 20;
    uint8_t* h = nullptr;
    CK(cudaHostAlloc((void**) &h, 4096 + 65536 + kRowsMax, cudaHostAllocPortable | cudaHostAllocMapped));
    std::memset(h, 0, 4096 + 65536 + kRowsMax);
    volatile uint32_t* hflag = (uint32_t*) h;   // [0] x, [32] rows, [64] err
    uint4* box_x = (uint4*) (h + 4096);
    float4* box_r = (float4*) (h + 4096 + 65536);
    uint32_t* dflag = nullptr;
    uint4* dbox_x = nullptr;
    float4* dbox_r = nullptr;
    CK(cudaHostGetDevicePointer((void**) &dflag, (void*) hflag, 0));
    CK(cudaHostGetDevicePointer((void**) &dbox_x, box_x, 0));
    CK(cudaHostGetDevicePointer((void**) &dbox_r, box_r, 0));
    volatile uint32_t* derr = dflag + 64;

    Dev O, S;
    O.id = io; S.id = is;
    for (Dev* d : {&O, &S}) {
        CK(cudaSetDevice(d->id));
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, d->id));
        d->sms = p.multiProcessorCount;
        CK(cudaStreamCreateWithFlags(&d->s, cudaStreamNonBlocking));
        d->pool_bytes = (size_t) 1536 << 20;   // cycled: far beyond either L2
        CK(cudaMalloc(&d->pool, d->pool_bytes));
        CK(cudaMemset(d->pool, 0x5a, d->pool_bytes));
        CK(cudaMalloc(&d->epoch, 4)); CK(cudaMemset(d->epoch, 0, 4));
        CK(cudaMalloc(&d->done, 4)); CK(cudaMemset(d->done, 0, 4));
        CK(cudaMalloc(&d->sink, 4));
        CK(cudaMalloc(&d->xready, 4)); CK(cudaMemset(d->xready, 0, 4));
        CK(cudaMalloc(&d->d_x, 65536)); CK(cudaMemset(d->d_x, 1, 65536));
        CK(cudaMalloc(&d->rows, 4 * kRowsMax));   // the owner's own rows, then (at 2 MB) the rows copied in
        CK(cudaMalloc(&d->stamps, sizeof(uint64_t) * (size_t) (W + 1) * L));
        CK(cudaMemset(d->stamps, 0, sizeof(uint64_t) * (size_t) (W + 1) * L));
        CK(cudaDeviceSynchronize());
    }
    int srv_blocks = 0;
    {
        CK(cudaSetDevice(S.id));
        int per_sm = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_server, 256, 0));
        srv_blocks = std::min(per_sm, 2) * S.sms;   // every block resident: the flag protocol needs them all alive
        std::printf("server: %d persistent blocks (%d per SM allowed)\n", srv_blocks, per_sm);
    }
    const int own_blocks = 4 * O.sms;

    // entries = routed (token, expert) pairs; n = distinct experts (the blobs read).  Vpeer (2/11, M): per layer-window
    // at T~2.9: 9.2 entries on the 3080 Ti, 18 on the 2080 Ti, 1.4 on the CPU; distinct ~0.75 of the entries.
    // "hot" = the primary keeps the hottest ranks (no STRATA_PEER_HOT): ~43% of the entries local (PIPELINE2 2b).
    const Cfg cfgs[] = {
        {"T=2 as Vpeer, rows fp32",      2, 5, 6, 9, 12, 10240},
        {"T=3 as Vpeer, rows fp32",      3, 7, 9, 13, 18, 10240},
        {"T=4 as Vpeer, rows fp32",      4, 9, 12, 17, 24, 10240},
        {"T=3 hot-first, rows fp32",     3, 10, 13, 11, 15, 10240},
        {"T=3 as Vpeer, rows bf16",      3, 7, 9, 13, 18, 5120},
        {"T=3 as Vpeer, summed/token",   3, 7, 9, 13, 18, 10240},   // ent_srv below is replaced by T rows back
        {"T=3 flags only (no experts)",  3, 0, 0, 0, 0, 10240},
    };
    std::printf("\n%-32s %4s %7s %7s | %8s %8s %8s %8s %8s | %10s %10s %10s | %s\n", "config", "T", "x B", "rows B", "floor", "all",
                "persist", "host", "host-ns", "exp.persist", "exp.host", "exp.hostns", "wait@rows p50/p90 us (persist)");
    enum Form { FLOOR = 0, ALL, PERSIST, HOST, HOSTNS, NFORM };
    for (const Cfg& c : cfgs) {
        const bool summed = std::strstr(c.what, "summed") != nullptr;
        const int xb = c.T * 2816 + 128;                       // q8_1 rows + ids
        const int n16_x = (xb + 15) / 16;
        const int rows_back = summed ? c.T : c.ent_srv;        // rows the server sends back
        const int row16 = c.row_bytes / 16;
        const int n16_r = rows_back * row16;
        double res[NFORM] = {0, 0, 0, 0, 0};
        double wp50 = 0, wp90 = 0;
        bool timeout = false;
        for (int form = 0; form < NFORM; ++form) {
            const bool exchange = form >= PERSIST;
            hflag[0] = hflag[32] = hflag[64] = 0;
            for (Dev* d : {&O, &S}) {
                CK(cudaSetDevice(d->id));
                CK(cudaMemset(d->epoch, 0, 4)); CK(cudaMemset(d->done, 0, 4)); CK(cudaMemset(d->xready, 0, 4));
                CK(cudaDeviceSynchronize());
            }
            // the owner's graph: one window
            cudaGraphExec_t ge = nullptr;
            CK(cudaSetDevice(O.id));
            {
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(O.s, cudaStreamCaptureModeThreadLocal));
                k_bump<<<1, 1, 0, O.s>>>(O.epoch);
                for (int l = 0; l < L; ++l) {
                    const size_t off_d = (size_t) l * dense_chunks * CHUNK;
                    const size_t off_e = (size_t) 48 * dense_chunks * CHUNK + (size_t) l * 32 * chunks * CHUNK;
                    k_dense<<<own_blocks, 256, 0, O.s>>>(O.pool, O.pool_bytes, off_d, dense_chunks, O.sink);
                    if (exchange) k_push<<<1, 256, 0, O.s>>>(dbox_x, O.d_x, n16_x, dflag + 0, O.epoch, l);
                    const int n_own = form == ALL ? c.n_loc + c.n_srv : c.n_loc;
                    if (n_own > 0)
                        k_experts<<<own_blocks, 256, 0, O.s>>>(O.pool, O.pool_bytes, off_e, n_own, chunks, c.T, O.rows, row16, O.sink);
                    if (exchange)
                        k_wait_copy<<<std::max(1, std::min(64, (n16_r + 255) / 256)), 256, 0, O.s>>>(
                            (uint4*) (O.rows + 2 * kRowsMax / 16), (const volatile uint4*) dbox_r, n16_r, dflag + 32, O.xready, O.epoch, l, O.stamps, derr);
                }
                CK(cudaStreamEndCapture(O.s, &g));
                CK(cudaGraphInstantiate(&ge, g, 0));
                CK(cudaGraphDestroy(g));
            }
            std::vector<double> per;
            cudaEvent_t e0, e1;
            CK(cudaSetDevice(O.id));
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            for (int w = 0; w < W && !timeout; ++w) {
                const uint32_t base = (uint32_t) (w + 1) << 8;
                if (form == PERSIST) {
                    CK(cudaSetDevice(S.id));
                    k_server<<<srv_blocks, 256, 0, S.s>>>(dflag + 0, dflag + 32, dbox_x, n16_x, S.d_x, S.xready, S.done, S.pool, S.pool_bytes,
                                                          c.n_srv, chunks, summed ? 0 : c.T, summed ? c.T : 0, dbox_r, row16, base, derr,
                                                          S.sink);
                    CK(cudaGetLastError());
                    cudaStreamQuery(S.s);   // WDDM: submit now
                }
                CK(cudaSetDevice(O.id));
                CK(cudaEventRecord(e0, O.s));
                CK(cudaGraphLaunch(ge, O.s));
                CK(cudaEventRecord(e1, O.s));
                cudaStreamQuery(O.s);
                if (form == HOST || form == HOSTNS) {   // today's peer path, on this thread: per layer
                    for (int l = 0; l < L; ++l) {
                        const uint32_t v = base | (uint32_t) (l + 1);
                        const auto t0 = std::chrono::steady_clock::now();
                        while (hflag[0] < v) {
                            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > 0.2) { hflag[64] = 1; break; }
                        }
                        if (hflag[64]) break;
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        // launch(): On(device_), the input and the plan copied in, hadamard + quantize + grouped + scatter
                        CK(cudaSetDevice(S.id));
                        CK(cudaMemcpyAsync(S.d_x, box_x, (size_t) n16_x * 16, cudaMemcpyHostToDevice, S.s));
                        CK(cudaMemcpyAsync(S.d_x + 2048, box_x, 4096, cudaMemcpyHostToDevice, S.s));
                        k_bump<<<1, 1, 0, S.s>>>(S.epoch);
                        k_bump<<<1, 1, 0, S.s>>>(S.epoch);
                        if (c.n_srv > 0)
                            k_experts_host<<<srv_blocks, 256, 0, S.s>>>(S.pool, S.pool_bytes, (size_t) l * c.n_srv * chunks * CHUNK, c.n_srv,
                                                                        chunks, summed ? 0 : c.T, summed ? c.T : 0, dbox_r, row16, S.sink);
                        k_bump<<<1, 1, 0, S.s>>>(S.epoch);
                        CK(cudaGetLastError());
                        CK(cudaSetDevice(O.id));
                        if (form == HOST) CK(cudaSetDevice(S.id));     // finish(): On(device_) around the spin
                        cudaError_t q;
                        while ((q = cudaStreamQuery(S.s)) == cudaErrorNotReady) {}
                        CK(q);
                        if (form == HOST) CK(cudaSetDevice(O.id));
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        hflag[32] = v;                                  // the pool's flag for the graph's wait
                    }
                }
                CK(cudaSetDevice(O.id));
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                if (w >= 3) per.push_back(ms * 1000.0 / L);
                if (form == PERSIST) { CK(cudaSetDevice(S.id)); CK(cudaStreamSynchronize(S.s)); }
                if (hflag[64]) timeout = true;
            }
            CK(cudaSetDevice(O.id));
            CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
            res[form] = pct(per, 0.5);
            if (form == PERSIST) {   // the wait at the rows' flag per layer (what the server + the hop exposed)
                std::vector<uint64_t> st((size_t) (W + 1) * L);
                CK(cudaMemcpy(st.data(), O.stamps, sizeof(uint64_t) * st.size(), cudaMemcpyDeviceToHost));
                std::vector<double> us;
                for (int w = 3; w < W; ++w)
                    for (int l = 0; l < L; ++l) us.push_back((double) st[(size_t) w * L + l] / 1000.0);
                wp50 = pct(us, 0.5);
                wp90 = pct(us, 0.9);
            }
            cudaGraphExecDestroy(ge);
        }
        std::printf("%-32s %4d %7d %7d | %8.1f %8.1f %8.1f %8.1f %8.1f | %10.1f %10.1f %10.1f | %.1f / %.1f%s\n", c.what, c.T, xb, n16_r * 16,
                    res[FLOOR], res[ALL], res[PERSIST], res[HOST], res[HOSTNS], res[PERSIST] - res[FLOOR], res[HOST] - res[FLOOR],
                    res[HOSTNS] - res[FLOOR], wp50, wp90, timeout ? "  [TIMEOUT]" : "");
    }
    std::printf("\nerr flag %u\n", hflag[64]);
    return 0;
}
