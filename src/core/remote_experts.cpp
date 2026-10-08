#include "strata/core/remote_experts.hpp"
#include "strata/core/remote_expert_opt.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <map>
#include <cstdio>
#if defined(_WIN32) && !defined(STRATA_USE_HIP)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#ifdef small
#undef small
#endif
#endif

namespace strata::core {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;

// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) + " experts: " +
               (failed_step ? failed_step : "device switch") +
               "(" + std::to_string(device) + ") failed: " + cudaGetErrorString(status) +
               " (CUDA error " + std::to_string((int) status) + ")";
    }
};

bool check(cudaError_t result, const char* what, std::string& err, int device) {
    if (result == cudaSuccess) return true;
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " + cudaGetErrorString(result);
    return false;
}
} // namespace


bool RemoteExperts::query_headroom(int device, Headroom& out, std::string& err) {
    const auto start = std::chrono::steady_clock::now();
    out = {}; DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_b = 0, total_b = 0;
    if (!check(cudaMemGetInfo(&free_b,&total_b),"helper headroom CUDA",err,device)) return false;
    out.cuda_free=free_b; out.cuda_total=total_b; out.available=free_b;
#if defined(_WIN32) && !defined(STRATA_USE_HIP)
    out.windows=true;
    cudaDeviceProp prop{};
    if (!check(cudaGetDeviceProperties(&prop,device),"helper headroom properties",err,device)) return false;
    const unsigned char* u=(const unsigned char*)prop.uuid.bytes; char uuid[41]{};
    std::snprintf(uuid,sizeof uuid,"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
    out.uuid=uuid;
    struct NvMem { unsigned long long total,free,used; };
    using Init=int(*)(); using ByUuid=int(*)(const char*,void**); using Mem=int(*)(void*,NvMem*);
    using Factory=HRESULT(WINAPI*)(REFIID,void**);
    struct Api {
        HMODULE nv=nullptr,dx=nullptr; ByUuid by=nullptr; Mem mem=nullptr; IDXGIFactory4* factory=nullptr; int status=-1;
        Api() { nv=LoadLibraryW(L"nvml.dll");dx=LoadLibraryW(L"dxgi.dll");
            auto init=nv?(Init)(void*)GetProcAddress(nv,"nvmlInit_v2"):nullptr;
            by=nv?(ByUuid)(void*)GetProcAddress(nv,"nvmlDeviceGetHandleByUUID"):nullptr;
            mem=nv?(Mem)(void*)GetProcAddress(nv,"nvmlDeviceGetMemoryInfo"):nullptr;
            auto create=dx?(Factory)(void*)GetProcAddress(dx,"CreateDXGIFactory1"):nullptr;
            if(init&&by&&mem&&create) {status=init();if(status==0&&FAILED(create(__uuidof(IDXGIFactory4),(void**)&factory)))status=-2;}
        }
        ~Api(){if(factory)factory->Release();} // DLLs remain loaded for cached handles through process teardown.
    };
    struct Binding { void* nvdev=nullptr; IDXGIAdapter3* adapter=nullptr; LUID luid{};
        ~Binding(){if(adapter)adapter->Release();}
    };
    static Api api;
    static std::map<std::string,std::unique_ptr<Binding>> bindings; // serialized helper safe points only
    if(api.status!=0||!api.factory) {err="remote elastic Windows headroom API init failed: "+std::to_string(api.status);return false;}
    LUID luid{};std::memcpy(&luid,prop.luid,sizeof luid);
    auto it=bindings.find(out.uuid);
    if(it==bindings.end()) {
        auto b=std::make_unique<Binding>();b->luid=luid;
        const int ns=api.by(uuid,&b->nvdev);
        const HRESULT ds=api.factory->EnumAdapterByLuid(luid,__uuidof(IDXGIAdapter3),(void**)&b->adapter);
        if(ns!=0||FAILED(ds)){err="remote elastic Windows UUID/LUID binding failed: "+std::to_string(ns)+":"+std::to_string((long)ds);return false;}
        it=bindings.emplace(out.uuid,std::move(b)).first;
    }
    if(std::memcmp(&it->second->luid,&luid,sizeof luid)!=0){err="remote elastic UUID/LUID changed";return false;}
    NvMem n{};DXGI_QUERY_VIDEO_MEMORY_INFO d{};
    const int ns=api.mem(it->second->nvdev,&n);
    const HRESULT ds=it->second->adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&d);
    if(ns!=0||FAILED(ds)||n.free>n.total||n.used>n.total||d.Budget>(uint64_t)INT64_MAX||d.CurrentUsage>(uint64_t)INT64_MAX){err="remote elastic Windows headroom query invalid: "+std::to_string(ns)+":"+std::to_string((long)ds);return false;}
    out.nvml_free=n.free;out.nvml_total=n.total;out.nvml_used=n.used;out.dxgi_budget=d.Budget;out.dxgi_usage=d.CurrentUsage;
    out.dxgi_available=(int64_t)d.Budget-(int64_t)d.CurrentUsage;
    out.available=std::min<uint64_t>(out.cuda_free,std::min<uint64_t>(out.nvml_free,(uint64_t)std::max<int64_t>(0,out.dxgi_available)));
#endif
    out.query_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return true;
}

RemoteExperts::~RemoteExperts() { close(); }

bool RemoteExperts::preflight(int device, double& free_gib, std::string& err) {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 1 || device >= count) {
        err = "CUDA" + std::to_string(device) + " experts: CUDA device is not visible";
        return false;
    }
    // The layer waits for this GPU on the CPU pool's critical path: spin instead of sleeping, whose wake-up
    // costs more than a small expert batch takes (measured: ~0.3 ms per round trip on Windows).  Only possible
    // before the device's context exists, so first thing; STRATA_REMOTE_SPIN=0 keeps the driver's default.
#if !defined(STRATA_USE_HIP)
    const char* spin = std::getenv("STRATA_REMOTE_SPIN");
    if (!(spin && spin[0] == '0')) cudaInitDevice(device, cudaDeviceScheduleSpin | cudaDeviceMapHost, 0);
    cudaGetLastError();
#endif
    // HIP has no cudaInitDevice equivalent; retain its default scheduling policy.
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "preflight free memory", err, device)) return false;
    free_gib = (double) free_bytes / 1073741824.0;
    return true;
}

void RemoteExperts::close() {
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (!scope.ok && cache_.elastic()) {
        std::fprintf(stderr, "FATAL helper elastic cleanup: device switch failed; bookkeeping retained\n"); return;
    }
    if (scope.ok) {
        if (stream_ && cudaStreamSynchronize(stream_) != cudaSuccess && cache_.elastic()) {
            std::fprintf(stderr, "FATAL helper elastic cleanup: stream synchronization failed; device/bookkeeping retained\n"); return;
        }
        cache_.close();
        if (cache_.elastic()) {
            std::fprintf(stderr, "FATAL helper elastic cleanup: cache driver state retained on device%d\n", device_); return;
        }
        if (d_x_) cudaFree(d_x_);
        if (d_out_) cudaFree(d_out_);
        if (d_q8_) cudaFree(d_q8_);
        if (d_scales_) cudaFree(d_scales_);
        if (d_scratch_) cudaFree(d_scratch_);
        if (d_meta_) cudaFree(d_meta_);
        if (h_x_) cudaFreeHost(h_x_);
        if (h_out_) cudaFreeHost(h_out_);
        if (h_meta_) cudaFreeHost(h_meta_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    device_ = -1;
    stream_ = nullptr;
    h_x_ = h_out_ = d_x_ = d_out_ = nullptr;
    h_meta_ = d_meta_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_scratch_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = nullptr;
    d_ptr_ = nullptr;
    original_row_.clear();
    layers_present_.clear();
}

bool RemoteExperts::open(int device, int slots, int64_t layers, int64_t experts,
                         const std::vector<std::pair<int32_t, int32_t>>& ranked,
                         const ExpertCache& primary, ExpertSource& source,
                         std::vector<uint8_t>& claimed, std::string& err, bool auto_size, bool elastic) {
    close();
    if (device_ >= 0 || !cache_.elastic_healthy()) { err = "remote experts: previous cleanup failed"; return false; }
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err, device)) return false;
    if (device < 1 || device >= count || slots <= 0 || ranked.empty() ||
        layers <= 0 || experts <= 0 || claimed.size() != (size_t) layers * (size_t) experts) {
        err = "CUDA" + std::to_string(device) + " experts: need the device, ranked experts and positive slot count";
        return false;
    }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    device_ = device;
    n_expert_ = experts;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (elastic && (!auto_size || !lay.native || !remote_opt_)) {
        err = "remote elastic requires auto/native/remote-expert-opt"; close(); return false;
    }
    const size_t elastic_scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    const uint64_t reserve = elastic ? ((500ull + 192 + 64) << 20) + elastic_scratch +
        CAP * (2 * H * sizeof(float) + (H / 32) * (36 + sizeof(float))) +
        sizeof(RemoteMeta) + remote_opt_->metadata_bytes() + strata::kernels::cpu::MAXT * H * sizeof(float) : 512ull << 20;
    size_t free_bytes = 0, total_bytes = 0;
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "free memory", err, device)) { close(); return false; }
    if (elastic) { Headroom h; if(!query_headroom(device,h,err)){close();return false;} free_bytes=h.available; elastic_stats_.query_ms+=h.query_ms;
        std::fprintf(stderr,"strata remote elastic: INITIAL_HEADROOM policy=windows-min3-v4 device=%d uuid=%s cuda_free_bytes=%llu nvml_free_bytes=%llu dxgi_available_bytes=%lld effective_free_bytes=%llu query_ms=%.3f\n",device,h.uuid.c_str(),(unsigned long long)h.cuda_free,(unsigned long long)h.nvml_free,(long long)h.dxgi_available,(unsigned long long)h.available,h.query_ms); }
    uint64_t needed = 0;
    std::vector<std::pair<int32_t, int32_t>> selected;
    selected.reserve((size_t) std::min<int64_t>(slots, layers * experts));
    std::vector<uint8_t> picked(claimed.size(), 0);
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (primary.slot_of(pair.first, pair.second) < 0 && !claimed[index] && !picked[index]) {
            const uint64_t bytes = lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : lay.max_blob;
            if (auto_size && needed + bytes + reserve > free_bytes) break;
            selected.push_back(pair);
            needed += bytes;
            picked[index] = 1;
            if ((int) selected.size() >= slots) break;
        }
    }
    if (selected.empty()) {
        err = "CUDA" + std::to_string(device) + (auto_size ? " experts: no unclaimed expert fits with 512 MiB free"
                                                          : " experts: no unclaimed experts remain");
        close(); return false;
    }
    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(selected.size());
        for (const auto& pair : selected) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    // Leave room for the CUDA context, staging and later driver allocations, especially under WDDM.
    if (needed + reserve > free_bytes) {
        err = "CUDA" + std::to_string(device) + " experts: slots leave less than 512 MiB free; reduce --expert-cache-device" + std::to_string(device);
        close(); return false;
    }
    cache_.set_elastic_strict(elastic);
    uint64_t va_bytes = 0;
    if (elastic) {
        for (int64_t l = 0; l < layers; ++l) va_bytes += ((lay.blob_bytes(l) + 255) / 256 * 256) * experts;
        elastic_ranked_.clear();
        for (const auto& p : ranked)
            if (p.first >= 0 && p.first < layers && p.second >= 0 && p.second < experts) elastic_ranked_.push_back(p);
        elastic_stats_ = {}; elastic_last_ms_ = elastic_stable_ms_ = 0;
    }
    const bool cache_ok = elastic ? cache_.open_sized_elastic(sizes, layers, experts, va_bytes, 64ull << 20, err) :
                         lay.native ? cache_.open_sized(sizes, layers, experts, err) :
                         cache_.open((int64_t) selected.size(), layers, experts, (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "CUDA" + std::to_string(device) + " experts: " + err; close(); return false; }
    for (const auto& pair : selected) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "CUDA" + std::to_string(device) + " experts: " +
                  (err.empty() ? "cache fill failed" : err);
            close(); return false;
        }
    }
    const auto& first = selected.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second),
                            err, (int64_t) lay.blob_bytes(first.first))) {
        err = "CUDA" + std::to_string(device) + " experts: " + err;
        close(); return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    const size_t meta_bytes = sizeof(RemoteMeta) + (remote_opt_ ? remote_opt_->metadata_bytes() : 0);
    const bool allocated =
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err, device) &&
        check(cudaHostAlloc((void**) &h_x_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "input staging", err, device) &&
        check(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped), "result staging", err, device) &&
        check(cudaHostAlloc(&h_meta_, meta_bytes, cudaHostAllocPortable), "metadata staging", err, device) &&
        check(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err, device) &&
        check(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "result", err, device) &&
        check(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activation", err, device) &&
        check(cudaMalloc((void**) &d_scales_, (size_t) CAP * (H / 32) * sizeof(float)), "activation scales", err, device) &&
        check(cudaMalloc(&d_scratch_, scratch), "scratch", err, device) &&
        check(cudaMalloc(&d_meta_, meta_bytes), "group metadata", err, device);
    if (!allocated) { close(); return false; }
    // Zero-copy: the helper reads its input from, and writes its compact rows into, the pinned host buffers
    // directly - two copies fewer per layer, each of which is a PCIe round trip.  STRATA_REMOTE_ZEROCOPY=0 copies.
    const char* zc = std::getenv("STRATA_REMOTE_ZEROCOPY");
    zero_copy_ = !(zc && zc[0] == '0') &&
                 cudaHostGetDevicePointer((void**) &z_x_, h_x_, 0) == cudaSuccess &&
                 cudaHostGetDevicePointer((void**) &z_out_, h_out_, 0) == cudaSuccess;
    cudaGetLastError();
    auto* meta = (RemoteMeta*) d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    owned_.resize(CAP);
    layers_present_.assign((size_t) layers, 0);
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    original_row_.reserve(CAP);
    computed_ = 0;
    launched_layers_ = 0;
    returned_bytes_ = full_row_bytes_ = 0;
    for (const auto& pair : selected) {
        layers_present_[(size_t) pair.first] = 1;
        claimed[(size_t) pair.first * (size_t) experts + (size_t) pair.second] = 1;
    }
    return true;
}

void RemoteExperts::refresh_layers() {
    std::fill(layers_present_.begin(), layers_present_.end(), 0);
    for (int32_t l = 0; l < (int32_t) layers_present_.size(); ++l)
        for (int32_t e = 0; e < n_expert_; ++e)
            if (holds(l, e)) layers_present_[(size_t) l] = 1;
}

bool RemoteExperts::elastic_audit(std::string& err) const {
    if (!cache_.elastic() || !cache_.elastic_healthy() || cache_.resident() != cache_.slots()) {
        err = "remote elastic: unhealthy mapping or unfilled slots"; return false;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<uint8_t> seen((size_t) cache_.slots(), 0);
    for (int32_t l = 0; l < (int32_t) layers_present_.size(); ++l)
        for (int32_t e = 0; e < n_expert_; ++e) {
            const int32_t slot = cache_.slot_of(l, e);
            if (slot < 0) continue;
            if (slot >= cache_.slots() || seen[(size_t) slot] ||
                cache_.slot_offset(slot + 1) - cache_.slot_offset(slot) < lay.blob_bytes(l)) {
                err = "remote elastic: duplicate/out-of-bounds/undersized ownership"; return false;
            }
            seen[(size_t) slot] = 1;
        }
    if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
        err = "remote elastic: missing slot ownership"; return false;
    }
    return true;
}

bool RemoteExperts::elastic_append(const std::vector<std::pair<int32_t, int32_t>>& selected,
                                  ExpertSource& source, std::string& err, bool verify) {
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<int64_t> sizes;
    for (const auto& p : selected) {
        if (p.first < 0 || p.first >= (int32_t) layers_present_.size() || p.second < 0 || p.second >= n_expert_ || holds(p.first, p.second)) {
            err = "remote elastic: invalid growth candidate"; return false;
        }
        sizes.push_back((int64_t) lay.blob_bytes(p.first));
    }
    const int64_t first = cache_.slots();
    if (!cache_.elastic_grow(sizes, err)) return false;
    // No ownership is visible until ALL copies have completed. A failed copy is fatal to the engine.
    for (size_t i = 0; i < selected.size(); ++i) {
        const auto& p = selected[i]; const uint8_t* blob = source.blob_stable(p.first, p.second);
        if (!blob || !cache_.fill_slot((int32_t) first + (int32_t) i, blob, stream_, err, sizes[i])) return false;
    }
    if (!check(cudaStreamSynchronize(stream_), "elastic fill synchronization", err, device_)) return false;
    if (verify)
        for (size_t i = 0; i < selected.size(); ++i)
            if (!cache_.verify_slot((int32_t) first + (int32_t) i, source.blob(selected[i].first, selected[i].second), err, sizes[i])) return false;
    for (size_t i = 0; i < selected.size(); ++i) {
        if (cache_.admit(selected[i].first, selected[i].second) != first + (int64_t) i) {
            err = "remote elastic: admission cursor mismatch"; return false;
        }
        layers_present_[(size_t) selected[i].first] = 1;
        elastic_stats_.copy_bytes += (uint64_t) sizes[i];
    }
    return true;
}

bool RemoteExperts::elastic_step(const std::vector<float>& usage, const std::vector<uint8_t>& excluded,
                                ExpertSource& source, bool request_start, std::string& err) {
    if (!cache_.elastic()) return true;
    const auto start = std::chrono::steady_clock::now();
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count();
    if (!request_start && now - elastic_last_ms_ < 1000) return true;
    elastic_last_ms_ = now;
    struct Cost { ElasticStats& stats; std::chrono::steady_clock::time_point start;
        ~Cost() { stats.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); }
    } cost{elastic_stats_, start};
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    if (!cache_.elastic_healthy() || excluded.size() != layers_present_.size() * (size_t) n_expert_ ||
        (!usage.empty() && usage.size() != excluded.size())) { err = "remote elastic: unhealthy arena/usage geometry"; return false; }
    for (float u : usage) if (!std::isfinite(u) || u < 0) { err = "remote elastic: invalid routing usage"; return false; }
    size_t free_bytes = 0, total = 0;
    Headroom headroom; if(!query_headroom(device_,headroom,err))return false;
    free_bytes=headroom.available; total=headroom.cuda_total; elastic_stats_.query_ms+=headroom.query_ms;
    elastic_stats_.cuda_free_min=std::min(elastic_stats_.cuda_free_min,headroom.cuda_free);
    if(headroom.windows){elastic_stats_.nvml_free_min=std::min(elastic_stats_.nvml_free_min,headroom.nvml_free);elastic_stats_.dxgi_available_min=std::min(elastic_stats_.dxgi_available_min,headroom.dxgi_available);}
    ++elastic_stats_.checks;
    elastic_stats_.free_min = std::min<uint64_t>(elastic_stats_.free_min, free_bytes);
    elastic_stats_.free_max = std::max<uint64_t>(elastic_stats_.free_max, free_bytes);
    constexpr uint64_t reserve = 500ull << 20, noise = 192ull << 20;
    if (free_bytes < reserve) {
        elastic_stable_ms_ = 0;
        if (!check(cudaStreamSynchronize(stream_), "elastic shrink quiescence", err, device_)) return false;
        const uint64_t dxgi_deficit = headroom.windows && headroom.dxgi_available < 0 ? (uint64_t)(-headroom.dxgi_available) : 0;
        const uint64_t give = std::min<uint64_t>(1024ull << 20, reserve + noise - free_bytes + std::min<uint64_t>(dxgi_deficit,1024ull << 20));
        const uint64_t mapped = cache_.elastic_mapped_bytes();
        const uint64_t rounded = (give + cache_.chunk_bytes() - 1) / cache_.chunk_bytes() * cache_.chunk_bytes();
        const int64_t keep = cache_.slots_within((int64_t) (mapped > rounded ? mapped - rounded : 0));
        if (keep == cache_.slots()) { err = "remote elastic: pressure shrink made no progress"; return false; }
        if (!cache_.elastic_shrink(keep, err)) return false;
        refresh_layers(); ++elastic_stats_.shrinks;
        return true;
    }
    if (free_bytes <= reserve + noise + cache_.chunk_bytes()) { elastic_stable_ms_ = 0; return true; }
    if (!elastic_stable_ms_) elastic_stable_ms_ = now;
    if (now - elastic_stable_ms_ < 5000) return true; // request boundaries never bypass post-shrink stability
    const uint64_t room = std::min<uint64_t>(512ull << 20, free_bytes - reserve - noise - cache_.chunk_bytes());
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::vector<std::pair<int32_t, int32_t>> order = elastic_ranked_, selected;
    if (!usage.empty()) std::stable_sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
        return usage[(size_t) a.first * n_expert_ + a.second] > usage[(size_t) b.first * n_expert_ + b.second];
    });
    std::vector<uint8_t> picked(excluded.size(), 0);
    uint64_t wanted = 0;
    for (const auto& p : order) {
        const size_t index = (size_t) p.first * n_expert_ + p.second;
        if (excluded[index] || picked[index] || holds(p.first, p.second)) continue;
        const uint64_t bytes = (lay.blob_bytes(p.first) + 255) / 256 * 256;
        if (bytes > room - wanted) continue;
        wanted += bytes; picked[index] = 1; selected.push_back(p);
    }
    if (selected.empty()) return true;
    if (!check(cudaStreamSynchronize(stream_), "elastic growth quiescence", err, device_)) return false;
    if (!elastic_append(selected, source, err)) return false;
    ++elastic_stats_.grows; elastic_stable_ms_ = 0;
    return true;
}

bool RemoteExperts::elastic_smoke(ExpertSource& source, std::string& err) {
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    if (!elastic_audit(err) || !check(cudaDeviceSynchronize(), "smoke quiescence", err, device_)) return false;
    const auto start = std::chrono::steady_clock::now();
    const uint64_t mapped = cache_.elastic_mapped_bytes();
    if (mapped <= cache_.chunk_bytes()) { err = "remote elastic smoke: insufficient mapped capacity"; return false; }
    const int64_t original = cache_.slots(), keep = cache_.slots_within((int64_t) (mapped - cache_.chunk_bytes()));
    std::vector<std::pair<int32_t, int32_t>> tail((size_t) (original - keep));
    for (int32_t l = 0; l < (int32_t) layers_present_.size(); ++l)
        for (int32_t e = 0; e < n_expert_; ++e) {
            const int32_t slot = cache_.slot_of(l, e);
            if (slot >= keep) tail[(size_t) (slot - keep)] = {l, e};
        }
    std::fprintf(stderr, "strata remote elastic: SMOKE_BEGIN mapped=%llu slots=%lld timeout_ms=5000\n", (unsigned long long) mapped, (long long) original);
    std::fflush(stderr); // runner must enforce 5s PID+birth watchdog; CUDA calls are not cancellable
    if (!cache_.elastic_shrink(keep, err)) return false;
    const uint64_t shrunk = cache_.elastic_mapped_bytes();
    if (mapped - shrunk < (64ull << 20) || mapped - shrunk > (128ull << 20)) {
        err = "remote elastic smoke: physical delta outside64..128MiB"; return false;
    }
    refresh_layers();
    if (!elastic_append(tail, source, err, true) || !elastic_audit(err)) return false;
    if (cache_.slots() != original || cache_.elastic_mapped_bytes() != mapped) {
        err = "remote elastic smoke: capacity restoration mismatch"; return false;
    }
    for (size_t i = 0; i < tail.size(); ++i)
        if (cache_.slot_of(tail[i].first, tail[i].second) != keep + (int64_t) i) {
            err = "remote elastic smoke: ownership restoration mismatch"; return false;
        }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (ms > 5000) { err = "remote elastic smoke: exceeded5s"; return false; }
    std::fprintf(stderr, "strata remote elastic: SMOKE_PASS before=%llu shrunk=%llu restored=%llu slots=%lld byteverify=1 ms=%.3f scope=SAFETY_NOT_RESULT\n",
                 (unsigned long long) mapped, (unsigned long long) shrunk, (unsigned long long) cache_.elastic_mapped_bytes(), (long long) original, ms);
    return true;
}

bool RemoteExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok,
                          int64_t k, const int32_t* kind, const int32_t* primary_res,
                          std::string& err) {
    // cumulative host time in here (staging and launches), reported per request by the driver
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != 10 || layer < 0 ||
        (size_t) layer >= layers_present_.size() || device_ < 0) {
        err = "CUDA" + std::to_string(device_) + " experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); original_row_.clear();
    if (!layers_present_[(size_t) layer]) return true;
    std::fill(group_of_.begin(), group_of_.begin() + n, -1);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) continue;
        owned_[(size_t) i] = 1;
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g)
            if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) i] = group;
        ++computed_;
    }
    if (group_id_.empty()) return true;
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int64_t i = 0; i < n; ++i) if (group_of_[(size_t) i] == (int32_t) g) {
            // The grouped kernels read the activation from `tok`, so their
            // output row can instead be packed densely for the USB4 return.
            original_row_.push_back((int32_t) i);
            dst_.push_back((int32_t) dst_.size());
            tok_.push_back((int32_t) (i / k));
        }
    }
    start_.push_back((int32_t) dst_.size());
    // Private pinned buffers survive until this GPU has consumed them. The CPU
    // pool can write other output rows without a cross-device race.
    std::memcpy(h_x_, x, (size_t) n_tok * H * sizeof(float));
    auto* meta = (RemoteMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();
    const bool reduce = remote_opt_ && remote_opt_->active();
    if (reduce) remote_opt_->prepare(*this, meta + 1);
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    groups_ = (int32_t) group_id_.size();
    const cudaStream_t s = stream_;
    const bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(d_x_, h_x_, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err, device_)) &&
        check(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(RemoteMeta) + (reduce ? remote_opt_->metadata_bytes() : 0),
                              cudaMemcpyHostToDevice, s), "copy group metadata", err, device_);
    if (!staged) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (lay.native) {
        strata::kernels::quantize_q8_1_rows(zero_copy_ ? z_x_ : d_x_, n_tok, H, d_q8_, s);
        const auto& fmt = lay.fmt[(size_t) layer];
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                               groups_, (int64_t) dst_.size(), d_q8_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(zero_copy_ ? z_x_ : d_x_, d_q8_, d_scales_, n_tok * H, s);
        strata::kernels::moe_grouped_s2(d_ptr_, d_start_, d_count_, d_dst_, d_tok_,
                                        groups_, (int64_t) dst_.size(), d_q8_, d_scales_, d_scratch_, zero_copy_ && !reduce ? z_out_ : d_out_, s);
    }
    const uint64_t compact_bytes = (uint64_t) (reduce ? n_tok : dst_.size()) * H * sizeof(float);
    if (reduce) {
        if (!remote_opt_->reduce(*this, (RemoteMeta*) d_meta_ + 1, err)) return false;
    } else if (!zero_copy_ && !check(cudaMemcpyAsync(h_out_, d_out_, (size_t) compact_bytes, cudaMemcpyDeviceToHost, s),
               "copy results", err, device_)) return false;
    ++launched_layers_;
    returned_bytes_ += compact_bytes;
    full_row_bytes_ += (uint64_t) n * H * sizeof(float);
    return true;
}

bool RemoteExperts::finish(float* out, std::string& err) {
    if (group_id_.empty()) return true;
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    const auto w0 = std::chrono::steady_clock::now();
    if (!check(cudaStreamSynchronize(stream_), "finish", err, device_)) return false;
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    if (remote_opt_ && remote_opt_->active()) { remote_opt_->accumulate(*this); return true; }
    for (size_t i = 0; i < original_row_.size(); ++i)
        std::memcpy(out + (size_t) original_row_[i] * H, h_out_ + i * H, (size_t) H * sizeof(float));
    return true;
}

} // namespace strata::core
