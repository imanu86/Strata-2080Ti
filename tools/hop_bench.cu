// lab/pipeline2 (docs/PIPELINE2.md): what does one hop between the two cards cost when nothing but the GPUs is on the
// path?  No peer access between a 3060 and a 2080 Ti under WDDM, so every hop goes through pinned, mapped host memory:
// the writer stores the payload over its link, fences (__threadfence_system) and raises a flag; the reader polls the
// flag over its link and reads the payload.  Three measures:
//
//   1. ping-pong between two persistent one-block kernels (no launches, no host): round trip per payload size;
//   2. one "layer" of expert-parallel as CUDA graphs on both cards (the engine's form: wait / copy / busy kernels as
//      graph nodes, a device-side epoch so the graphs replay): exposed cost per layer against the same graph alone;
//   3. the copy engines (cudaMemcpyAsync D2H on one card, H2D on the other, cross-device event): round trip.
//
//   nvcc -O2 -arch=sm_75 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -o hop_bench tools/hop_bench.cu
//   hop_bench [iters]        (run it under bench_lock.py: it owns both GPUs for ~10-20 s)
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)
using Clock = std::chrono::steady_clock;

__device__ __forceinline__ uint64_t gtime() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// bounded wait (200 ms): a stuck hop must not trip the WDDM watchdog of the display card
__device__ __forceinline__ bool wait_ge(const volatile uint32_t* f, uint32_t v, int sleep_ns, volatile uint32_t* err) {
    const uint64_t t0 = gtime();
    while (*f < v) {
        if (sleep_ns > 0) __nanosleep(sleep_ns);
        if (gtime() - t0 > 200000000ull) { *err = 1; return false; }
    }
    __threadfence_system();
    return true;
}

// one block copies n4 float4 with 8 loads in flight per thread (volatile: every load crosses the link)
__device__ __forceinline__ void block_copy(float4* dst, const volatile float4* src, int n4) {
    for (int base = 0; base < n4; base += blockDim.x * 8) {
        float4 v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = base + u * blockDim.x + threadIdx.x;
            if (i < n4) v[u] = __ldcv((const float4*) src + i);   // 16 B, not cached
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int i = base + u * blockDim.x + threadIdx.x;
            if (i < n4) dst[i] = v[u];
        }
    }
}
__device__ __forceinline__ void block_copy_out(volatile float4* dst, const float4* src, int n4) {
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        __stwt((float4*) dst + i, src[i]);   // 16 B, write-through
    }
}

// ---------------------------------------------------------------- 1. persistent ping-pong
// initiator: for i: write fwd payload, flag_f = i; wait flag_b >= i; read back payload.  responder: the mirror.
__global__ void pp_init(volatile uint32_t* flag_f, const volatile uint32_t* flag_b, volatile float4* out, const volatile float4* in,
                        float4* dloc, int n4_f, int n4_b, int iters, int sleep_ns, uint64_t* stamps, volatile uint32_t* err) {
    __shared__ int ok;
    for (int i = 1; i <= iters; ++i) {
        const uint64_t t0 = gtime();
        block_copy_out(out, dloc, n4_f);
        __syncthreads();
        if (threadIdx.x == 0) { __threadfence_system(); *flag_f = (uint32_t) i; ok = wait_ge(flag_b, (uint32_t) i, sleep_ns, err); }
        __syncthreads();
        if (!ok) return;
        block_copy(dloc + n4_f, in, n4_b);
        __syncthreads();
        if (threadIdx.x == 0) stamps[i - 1] = gtime() - t0;
    }
}
__global__ void pp_resp(const volatile uint32_t* flag_f, volatile uint32_t* flag_b, const volatile float4* in, volatile float4* out,
                        float4* dloc, int n4_f, int n4_b, int iters, int sleep_ns, volatile uint32_t* err) {
    __shared__ int ok;
    for (int i = 1; i <= iters; ++i) {
        if (threadIdx.x == 0) ok = wait_ge(flag_f, (uint32_t) i, sleep_ns, err);
        __syncthreads();
        if (!ok) return;
        block_copy(dloc, in, n4_f);
        __syncthreads();
        block_copy_out(out, dloc + n4_f, n4_b);
        __syncthreads();
        if (threadIdx.x == 0) { __threadfence_system(); *flag_b = (uint32_t) i; }
    }
}

// ---------------------------------------------------------------- 2. graph nodes (the engine's form)
__global__ void k_bump(uint32_t* epoch) { *epoch += 1; }
__global__ void k_busy(int ns) {   // one block holding an SM for ns (a stand-in for a kernel of known length)
    if (ns <= 0) return;
    const uint64_t t0 = gtime();
    while (gtime() - t0 < (uint64_t) ns) {}
}
// value of layer l in epoch e: monotonic over the replays
__device__ __forceinline__ uint32_t val(const uint32_t* epoch, int l) { return (*epoch << 8) | (uint32_t) (l + 1); }
// push: copy n4 float4 to the mapped inbox, then raise the flag (one block: the payload is small)
__global__ void k_push(volatile float4* out, const float4* src, int n4, volatile uint32_t* flag, const uint32_t* epoch, int l) {
    block_copy_out(out, src, n4);
    __syncthreads();
    if (threadIdx.x == 0) { __threadfence_system(); *flag = val(epoch, l); }
}
// rows: many blocks write, the last block to finish raises the flag (one launch: a separate flag kernel races, PR #1798)
__global__ void k_rows(volatile float4* out, const float4* src, int n4, volatile uint32_t* flag, unsigned* done, const uint32_t* epoch, int l) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        __stwt((float4*) out + i, src[i]);
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned d = atomicAdd(done, 1u);
        if (d == gridDim.x - 1) { *done = 0; __threadfence_system(); *flag = val(epoch, l); }
    }
}
__global__ void k_wait(const volatile uint32_t* flag, const uint32_t* epoch, int l, int sleep_ns, volatile uint32_t* err) {
    wait_ge(flag, val(epoch, l), sleep_ns, err);
}
// the engine's copy_from_mapped_kernel (up to 64 blocks x 256)
__global__ void k_copy_in(float4* dst, const volatile float4* src, int n4) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        dst[i] = __ldcv((const float4*) src + i);
    }
}
// wait folded into the copy (fuse_wait): every block polls the flag, then copies
__global__ void k_wait_copy(float4* dst, const volatile float4* src, int n4, const volatile uint32_t* flag, const uint32_t* epoch,
                            int l, int sleep_ns, volatile uint32_t* err) {
    __shared__ int ok;
    if (threadIdx.x == 0) ok = wait_ge(flag, val(epoch, l), sleep_ns, err);
    __syncthreads();
    if (!ok) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) {
        dst[i] = __ldcv((const float4*) src + i);
    }
}

struct Dev {
    int id = -1;
    cudaStream_t s = nullptr;
    float4* dloc = nullptr;
    uint32_t* epoch = nullptr;
    unsigned* done = nullptr;
};

static int blocks_for(int n4) { return std::max(1, std::min(64, (n4 + 255) / 256)); }

static double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (p * (v.size() - 1) + 0.5))];
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 2000;
    int nd = 0;
    CK(cudaGetDeviceCount(&nd));
    int i3060 = -1, i2080 = -1;
    for (int d = 0; d < nd; ++d) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, d));
        int tcc = 0;
        cudaDeviceGetAttribute(&tcc, cudaDevAttrTccDriver, d);
        std::printf("CUDA%d %s sm_%d%d pci %02x:%02x tcc=%d canMapHost=%d uva=%d\n", d, p.name, p.major, p.minor, p.pciBusID,
                    p.pciDeviceID, tcc, p.canMapHostMemory, p.unifiedAddressing);
        if (std::strstr(p.name, "3060")) i3060 = d;
        if (std::strstr(p.name, "2080")) i2080 = d;
    }
    if (i3060 < 0 || i2080 < 0) { std::printf("need a 3060 and a 2080 Ti\n"); return 1; }
    int p01 = 0, p10 = 0;
    cudaDeviceCanAccessPeer(&p01, i3060, i2080);
    cudaDeviceCanAccessPeer(&p10, i2080, i3060);
    std::printf("peer access 3060->2080Ti %d, 2080Ti->3060 %d\n", p01, p10);

    // host side: flags and payloads, pinned + mapped + portable (one cache line per flag)
    const size_t kMaxBytes = 512 * 1024;
    uint8_t* h = nullptr;
    CK(cudaHostAlloc((void**) &h, 4096 + 4 * kMaxBytes, cudaHostAllocPortable | cudaHostAllocMapped));
    std::memset(h, 0, 4096 + 4 * kMaxBytes);
    uint32_t* hflag = (uint32_t*) h;   // [0] fwd, [32] back, [64] err, [96] fwd2, [128] back2
    float4* box_f = (float4*) (h + 4096);
    float4* box_b = (float4*) (h + 4096 + kMaxBytes);
    uint32_t *dflag = nullptr;
    float4 *dbox_f = nullptr, *dbox_b = nullptr;
    CK(cudaHostGetDevicePointer((void**) &dflag, hflag, 0));
    CK(cudaHostGetDevicePointer((void**) &dbox_f, box_f, 0));
    CK(cudaHostGetDevicePointer((void**) &dbox_b, box_b, 0));
    volatile uint32_t* derr = dflag + 64;

    Dev dv[2];
    dv[0].id = i2080; dv[1].id = i3060;
    for (auto& d : dv) {
        CK(cudaSetDevice(d.id));
        CK(cudaStreamCreateWithFlags(&d.s, cudaStreamNonBlocking));
        CK(cudaMalloc(&d.dloc, 2 * kMaxBytes));
        CK(cudaMemset(d.dloc, 0, 2 * kMaxBytes));
        CK(cudaMalloc(&d.epoch, 4));
        CK(cudaMemset(d.epoch, 0, 4));
        CK(cudaMalloc(&d.done, 4));
        CK(cudaMemset(d.done, 0, 4));
        CK(cudaDeviceSynchronize());
    }
    const char* nm[2] = {"2080Ti", "3060"};

    // ------------------------------------------------ 1. persistent ping-pong
    std::printf("\n1. persistent ping-pong (one block of 512 threads each side, %d round trips; us)\n", iters);
    std::printf("%-8s %-10s %8s %8s %8s | %7s %7s %7s\n", "initiator", "sleep", "fwd B", "back B", "", "p50", "p90", "p99");
    struct Sz { int f, b; const char* what; };
    const Sz sizes[] = {{0, 0, "flag only"},
                        {6 * 1024, 0, "xm q8_1 T=2 one way"},
                        {0, 80 * 1024, "8 expert rows fp32 back"},
                        {6 * 1024, 80 * 1024, "EP T=2: xm out, 8 rows back"},
                        {12 * 1024, 200 * 1024, "EP T=4: xm out, 20 rows back"},
                        {20 * 1024, 20 * 1024, "TP T=2: 2560x2 fp32 each way"},
                        {100 * 1024, 100 * 1024, "hand-off T=2 (51 KB x T) each way"}};
    for (int who = 0; who < 2; ++who)
        for (int sl : {0, 100})
            for (const Sz& z : sizes) {
                Dev& I = dv[who];
                Dev& R = dv[1 - who];
                hflag[0] = hflag[32] = hflag[64] = 0;
                uint64_t* stamps = nullptr;
                CK(cudaSetDevice(I.id));
                CK(cudaMalloc(&stamps, sizeof(uint64_t) * iters));
                const int n4f = z.f / 16, n4b = z.b / 16;
                CK(cudaSetDevice(R.id));
                pp_resp<<<1, 512, 0, R.s>>>(dflag + 0, dflag + 32, dbox_f, dbox_b, R.dloc, n4f, n4b, iters, sl, derr);
                CK(cudaGetLastError());
                cudaStreamQuery(R.s);   // WDDM: flush the submission now
                CK(cudaSetDevice(I.id));
                pp_init<<<1, 512, 0, I.s>>>(dflag + 0, dflag + 32, dbox_f, dbox_b, I.dloc, n4f, n4b, iters, sl, stamps, derr);
                CK(cudaGetLastError());
                cudaStreamQuery(I.s);
                CK(cudaStreamSynchronize(I.s));
                CK(cudaSetDevice(R.id));
                CK(cudaStreamSynchronize(R.s));
                std::vector<uint64_t> st(iters);
                CK(cudaSetDevice(I.id));
                CK(cudaMemcpy(st.data(), stamps, sizeof(uint64_t) * iters, cudaMemcpyDeviceToHost));
                CK(cudaFree(stamps));
                std::vector<double> us;
                for (int i = iters / 10; i < iters; ++i) us.push_back(st[i] / 1000.0);   // skip the warm-up tenth
                std::printf("%-8s %-10s %8d %8d %8s | %7.1f %7.1f %7.1f  %s%s\n", nm[who], sl ? "nanosleep" : "spin", z.f, z.b, "",
                            pct(us, 0.5), pct(us, 0.9), pct(us, 0.99), z.what, hflag[64] ? "  [TIMEOUT]" : "");
            }

    // ------------------------------------------------ 2. EP layer as graph nodes
    // owner O = 2080 Ti (dense, router, combine), helper H = 3060 (a share of the experts).  Per layer:
    //   O: busy(dense) -> push(xm) -> busy(own experts) -> wait(rows) -> copy rows in      [exchange]
    //   H: wait(xm) -> copy xm in -> busy(helper experts) -> rows (+flag)
    //   O alone: busy(dense) -> busy(own experts)                                          [baseline]
    // The exposed cost of the exchange per layer = (exchange - baseline) / L.
    std::printf("\n2. expert-parallel layer as CUDA graphs (owner 2080 Ti, helper 3060), 48 layers per graph; us per layer\n");
    std::printf("%-6s %-6s %-6s %-7s %-7s %-6s | %9s %9s %9s\n", "dense", "own", "help", "xm B", "rows B", "fused", "baseline", "exchange",
                "exposed");
    struct Cfg { int dense, own, help, xb, rb, fused; };
    const Cfg cfgs[] = {
        {0, 0, 0, 6 * 1024, 80 * 1024, 0},  {0, 0, 0, 6 * 1024, 80 * 1024, 1},   // pure exchange chain
        {0, 0, 0, 64, 64, 1},                                                    // flags only
        {200, 50, 0, 6 * 1024, 80 * 1024, 1},                                    // helper computes nothing: hop cost hidden?
        {200, 50, 80, 6 * 1024, 80 * 1024, 1},                                   // T=2 balanced shares (S: 2080 Ti 74 us, 3060 93 us)
        {200, 80, 80, 6 * 1024, 80 * 1024, 1},
        {200, 50, 80, 12 * 1024, 200 * 1024, 1},                                 // T=4
        {200, 130, 0, 6 * 1024, 80 * 1024, 1},                                   // baseline with all experts on the owner
        {0, 0, 0, 6 * 1024, 20 * 1024, 1},                                       // rows summed on the helper (T=2: 2 rows)
        {200, 50, 80, 6 * 1024, 20 * 1024, 1},
        {200, 50, 80, 6 * 1024, 40 * 1024, 1},                                   // 8 rows in bf16
        {200, 50, 80, 12 * 1024, 40 * 1024, 1},                                  // T=4 summed on the helper
    };
    const int L = 48, reps = 20;
    for (const Cfg& c : cfgs) {
        double res[2] = {0, 0};
        for (int mode = 0; mode < 2; ++mode) {   // 0 baseline, 1 exchange
            Dev& O = dv[0];
            Dev& H = dv[1];
            hflag[0] = hflag[32] = hflag[64] = 0;
            CK(cudaSetDevice(O.id)); CK(cudaMemset(O.epoch, 0, 4)); CK(cudaDeviceSynchronize());
            CK(cudaSetDevice(H.id)); CK(cudaMemset(H.epoch, 0, 4)); CK(cudaDeviceSynchronize());
            const int n4x = c.xb / 16, n4r = c.rb / 16;
            cudaGraphExec_t ge[2] = {nullptr, nullptr};
            // owner graph
            CK(cudaSetDevice(O.id));
            {
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(O.s, cudaStreamCaptureModeThreadLocal));
                k_bump<<<1, 1, 0, O.s>>>(O.epoch);
                for (int l = 0; l < L; ++l) {
                    k_busy<<<1, 32, 0, O.s>>>(c.dense * 1000);
                    if (mode == 1) k_push<<<1, 256, 0, O.s>>>(dbox_f, O.dloc, n4x, dflag + 0, O.epoch, l);
                    k_busy<<<1, 32, 0, O.s>>>((mode == 1 ? c.own : c.own + c.help) * 1000);
                    if (mode == 1) {
                        if (c.fused) k_wait_copy<<<blocks_for(n4r), 256, 0, O.s>>>(O.dloc + kMaxBytes / 16, dbox_b, n4r, dflag + 32, O.epoch, l, 100, derr);
                        else {
                            k_wait<<<1, 1, 0, O.s>>>(dflag + 32, O.epoch, l, 100, derr);
                            k_copy_in<<<blocks_for(n4r), 256, 0, O.s>>>(O.dloc + kMaxBytes / 16, dbox_b, n4r);
                        }
                    }
                }
                CK(cudaStreamEndCapture(O.s, &g));
                CK(cudaGraphInstantiate(&ge[0], g, 0));
                CK(cudaGraphDestroy(g));
            }
            if (mode == 1) {
                CK(cudaSetDevice(H.id));
                cudaGraph_t g;
                CK(cudaStreamBeginCapture(H.s, cudaStreamCaptureModeThreadLocal));
                k_bump<<<1, 1, 0, H.s>>>(H.epoch);
                for (int l = 0; l < L; ++l) {
                    if (c.fused) k_wait_copy<<<blocks_for(n4x), 256, 0, H.s>>>(H.dloc, dbox_f, n4x, dflag + 0, H.epoch, l, 100, derr);
                    else {
                        k_wait<<<1, 1, 0, H.s>>>(dflag + 0, H.epoch, l, 100, derr);
                        k_copy_in<<<blocks_for(n4x), 256, 0, H.s>>>(H.dloc, dbox_f, n4x);
                    }
                    k_busy<<<1, 32, 0, H.s>>>(c.help * 1000);
                    k_rows<<<blocks_for(n4r), 256, 0, H.s>>>(dbox_b, H.dloc + kMaxBytes / 16, n4r, dflag + 32, H.done, H.epoch, l);
                }
                CK(cudaStreamEndCapture(H.s, &g));
                CK(cudaGraphInstantiate(&ge[1], g, 0));
                CK(cudaGraphDestroy(g));
            }
            std::vector<double> per;
            for (int r = 0; r < reps; ++r) {
                if (mode == 1) { CK(cudaSetDevice(H.id)); CK(cudaGraphLaunch(ge[1], H.s)); cudaStreamQuery(H.s); }
                CK(cudaSetDevice(O.id));
                cudaEvent_t e0, e1;
                CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                CK(cudaEventRecord(e0, O.s));
                CK(cudaGraphLaunch(ge[0], O.s));
                CK(cudaEventRecord(e1, O.s));
                cudaStreamQuery(O.s);
                CK(cudaEventSynchronize(e1));
                float ms = 0;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                if (r >= 3) per.push_back(ms * 1000.0 / L);
                CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
                if (mode == 1) { CK(cudaSetDevice(H.id)); CK(cudaStreamSynchronize(H.s)); }
            }
            res[mode] = pct(per, 0.5);
            CK(cudaSetDevice(O.id)); cudaGraphExecDestroy(ge[0]);
            if (ge[1]) { CK(cudaSetDevice(H.id)); cudaGraphExecDestroy(ge[1]); }
        }
        std::printf("%-6d %-6d %-6d %-7d %-7d %-6d | %9.1f %9.1f %9.1f%s\n", c.dense, c.own, c.help, c.xb, c.rb, c.fused, res[0], res[1],
                    res[1] - res[0], hflag[64] ? "  [TIMEOUT]" : "");
    }

    // ------------------------------------------------ 3. copy engines + cross-device events
    std::printf("\n3. copy engines: D2H on the sender, event, H2D on the receiver, and back (%d round trips; us)\n", iters / 4);
    for (int who = 0; who < 2; ++who)
        for (int bytes : {64, 20 * 1024, 80 * 1024}) {
            Dev& I = dv[who];
            Dev& R = dv[1 - who];
            cudaEvent_t ei, er;
            CK(cudaSetDevice(I.id)); CK(cudaEventCreateWithFlags(&ei, cudaEventDisableTiming));
            CK(cudaSetDevice(R.id)); CK(cudaEventCreateWithFlags(&er, cudaEventDisableTiming));
            const int n = iters / 4;
            auto t0 = Clock::now();
            for (int i = 0; i < n; ++i) {
                CK(cudaSetDevice(I.id));
                CK(cudaMemcpyAsync(box_f, I.dloc, bytes, cudaMemcpyDeviceToHost, I.s));
                CK(cudaEventRecord(ei, I.s));
                CK(cudaSetDevice(R.id));
                CK(cudaStreamWaitEvent(R.s, ei, 0));
                CK(cudaMemcpyAsync(R.dloc, box_f, bytes, cudaMemcpyHostToDevice, R.s));
                CK(cudaMemcpyAsync(box_b, R.dloc, bytes, cudaMemcpyDeviceToHost, R.s));
                CK(cudaEventRecord(er, R.s));
                CK(cudaSetDevice(I.id));
                CK(cudaStreamWaitEvent(I.s, er, 0));
                CK(cudaMemcpyAsync(I.dloc, box_b, bytes, cudaMemcpyHostToDevice, I.s));
            }
            CK(cudaSetDevice(I.id));
            CK(cudaStreamSynchronize(I.s));
            const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / n;
            std::printf("initiator %-7s %7d B: %7.1f us per round trip (4 copies)\n", nm[who], bytes, us);
            cudaEventDestroy(ei);
            CK(cudaSetDevice(R.id)); cudaEventDestroy(er);
        }
    std::printf("\nerr flag %u\n", hflag[64]);
    return 0;
}
