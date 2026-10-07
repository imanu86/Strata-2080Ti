// src/core/ram_dedup.cu - STRATA_RAM_DEDUP=1: see include/strata/core/ram_dedup.hpp.
#include "strata/core/ram_dedup.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::core {

namespace {

constexpr uint64_t kPage = 4096;

using Us = std::chrono::steady_clock;
inline uint64_t since_us(Us::time_point t0) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(Us::now() - t0).count();
}

// Multi-GPU: make `dev` current for a scope (as OnDevice, without the header dependency); -1 = leave it.
struct UseDevice {
    int prev = -1;
    explicit UseDevice(int dev) {
        int cur = 0;
        if (dev >= 0 && cudaGetDevice(&cur) == cudaSuccess && cur != dev && cudaSetDevice(dev) == cudaSuccess) prev = cur;
    }
    ~UseDevice() { if (prev >= 0) cudaSetDevice(prev); }
};

/// One unbuffered read of [offset, offset + bytes) of `path` into `dst` (any alignment): the 4 KiB-aligned window is
/// read into `tmp` (grown as needed, page-aligned) and the asked bytes copied out.  Synchronous: the last resort.
bool read_file_range(const std::string& path, uint64_t offset, uint64_t bytes, uint8_t* dst, std::vector<uint8_t>& tmp) {
    const uint64_t a0 = offset / kPage * kPage, a1 = (offset + bytes + kPage - 1) / kPage * kPage, span = a1 - a0;
    if (tmp.size() < span + kPage) tmp.resize((size_t) (span + kPage));
    uint8_t* buf = (uint8_t*) (((uintptr_t) tmp.data() + kPage - 1) & ~(uintptr_t) (kPage - 1));
#if defined(_WIN32)
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> w((size_t) (wide > 0 ? wide : 1), L'\0');
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wide);
    HANDLE h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    uint64_t got = 0;
    bool ok = true;
    while (ok && got < span) {   // a read past the end of the file is short: the blob's bytes are inside it
        OVERLAPPED ov{};
        ov.Offset = (DWORD) (a0 + got);
        ov.OffsetHigh = (DWORD) ((a0 + got) >> 32);
        DWORD n = 0;
        const DWORD want = (DWORD) std::min<uint64_t>(span - got, 64ull << 20);
        ok = ReadFile(h, buf + got, want, &n, &ov) && n > 0;
        got += n;
        if (n < want) break;
    }
    CloseHandle(h);
    if (got < offset + bytes - a0) return false;
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    uint64_t got = 0;
    while (got < span) {
        const ssize_t n = pread(fd, buf + got, (size_t) (span - got), (off_t) (a0 + got));
        if (n <= 0) break;
        got += (uint64_t) n;
    }
    ::close(fd);
    if (got < offset + bytes - a0) return false;
#endif
    std::memcpy(dst, buf + (offset - a0), (size_t) bytes);
    return true;
}

/// The GGUF file that holds role `r` of layer `l` (as expert_source.cpp's expert_gguf_file).
std::string gguf_role_file(const std::string& gguf, const strata::kernels::cpu::ExpertLayout& lay, int64_t l, int r) {
    const size_t i = (size_t) (3 * l + r);
    if (lay.gguf_file.size() <= i || lay.gguf_file[i].empty()) return gguf;
    const size_t cut = gguf.find_last_of("/\\");
    return (cut == std::string::npos ? std::string() : gguf.substr(0, cut + 1)) + lay.gguf_file[i];
}

}  // namespace

bool RamDedup::requested() {
    const char* v = std::getenv("STRATA_RAM_DEDUP");
    return v != nullptr && v[0] == '1';
}

uint64_t RamDedup::keep_quota_env() {
    const char* v = std::getenv("STRATA_RAM_DEDUP_KEEP_GIB");
    if (v == nullptr || v[0] == '\0') return ~0ull;
    const double gib = std::atof(v);
    return gib <= 0.0 ? 0 : (uint64_t) (gib * 1073741824.0);
}

uint64_t RamDedup::bounce_bytes_env() {
    const char* v = std::getenv("STRATA_RAM_DEDUP_BOUNCE_MIB");
    const long long mib = v != nullptr && v[0] != '\0' ? std::atoll(v) : 512;
    return (uint64_t) std::max<long long>(mib, 16) << 20;
}

RamDedup::~RamDedup() {
    for (Slot& s : ring_) {
        if (s.ev_before) cudaEventDestroy((cudaEvent_t) s.ev_before);
        if (s.ev_after) cudaEventDestroy((cudaEvent_t) s.ev_after);
    }
    for (size_t d = 0; d < devs_.size(); ++d)
        if (devs_[d].stream) {
            UseDevice on((int) d);
            cudaStreamDestroy((cudaStream_t) devs_[d].stream);
        }
    if (bounce_) cudaFreeHost(bounce_);
    (void) cudaGetLastError();
}

bool RamDedup::arm(uint8_t* base, uint64_t arena_bytes, uint64_t registered, bool large_pages, int64_t n_layers,
                   int64_t n_expert, const std::string& experts_bin, const std::string& gguf, uint64_t bounce_bytes,
                   std::string& err) {
    if (armed_) return true;
    if (base == nullptr || arena_bytes == 0 || n_layers <= 0 || n_expert <= 0) { err = "no arena"; return false; }
    if (large_pages) { err = "the arena is on large pages, which cannot be decommitted per expert (STRATA_NO_LARGEPAGES=1)"; return false; }
    if (((uintptr_t) base) % kPage != 0) { err = "the arena is not page aligned"; return false; }
    lay_ = &strata::kernels::cpu::expert_layout();
    if (lay_->n_layers != n_layers || lay_->n_expert != n_expert) { err = "the expert layout is of another geometry"; return false; }
    base_ = base;
    arena_bytes_ = arena_bytes;
    registered_ = registered;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    experts_bin_ = experts_bin;
    gguf_ = gguf;
    const size_t n = (size_t) (n_layers * n_expert);
    std::vector<std::atomic<uint8_t>>(n).swap(state_);
    for (auto& s : state_) s.store(kPresent, std::memory_order_relaxed);
    slot_of_.assign(n, -1);
    bounce_blob_ = lay_->max_blob;
    const size_t slots = (size_t) std::max<uint64_t>(4, bounce_bytes / std::max<uint64_t>(bounce_blob_, 1));
    bounce_bytes_ = (uint64_t) slots * bounce_blob_;
    if (cudaHostAlloc((void**) &bounce_, (size_t) bounce_bytes_, cudaHostAllocPortable) != cudaSuccess) {
        (void) cudaGetLastError();
        bounce_ = nullptr;
        err = "the bounce ring (" + std::to_string(bounce_bytes_ >> 20) + " MiB, page-locked) could not be allocated";
        return false;
    }
    ring_.assign(slots, Slot{});
    head_ = tail_ = in_ring_ = 0;
    armed_ = true;
    return true;
}

void RamDedup::page_span(int64_t idx, uint64_t& start, uint64_t& end) const {
    const int64_t l = idx / n_expert_, e = idx % n_expert_;
    const uint64_t off = lay_->blob_offset(l, e), bytes = lay_->blob_bytes(l);
    start = (off + kPage - 1) / kPage * kPage;
    end = (off + bytes) / kPage * kPage;
    if (end < start) end = start;
}

bool RamDedup::decommit(int64_t idx, uint64_t& bytes) {
    uint64_t s = 0, e = 0;
    page_span(idx, s, e);
    bytes = e - s;
    if (bytes == 0) return false;
#if defined(_WIN32)
    (void) VirtualUnlock(base_ + s, (SIZE_T) bytes);   // ERROR_NOT_LOCKED when the lock stopped before: harmless
    return VirtualFree(base_ + s, (SIZE_T) bytes, MEM_DECOMMIT) != 0;
#else
    (void) munlock(base_ + s, (size_t) bytes);
    return madvise(base_ + s, (size_t) bytes, MADV_DONTNEED) == 0;
#endif
}

bool RamDedup::recommit(int64_t idx) {
    uint64_t s = 0, e = 0;
    page_span(idx, s, e);
    if (e == s) return true;
#if defined(_WIN32)
    void* p = VirtualAlloc(base_ + s, (SIZE_T) (e - s), MEM_COMMIT, PAGE_READWRITE);
    if (p != base_ + s) return false;
    (void) VirtualLock(base_ + s, (SIZE_T) (e - s));   // best effort: the working-set minimum was raised at startup
    return true;
#else
    (void) mlock(base_ + s, (size_t) (e - s));
    return true;
#endif
}

uint64_t RamDedup::drop(int64_t layer, int64_t expert) {
    if (!armed_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return 0;
    const int64_t idx = layer * n_expert_ + expert;
    if (lay_->blob_offset(layer, expert) < registered_) {   // inside the CUDA-registered prefix: never decommitted
        kept_registered_.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (state_[(size_t) idx].load(std::memory_order_acquire) != kPresent) return 0;
    uint64_t bytes = 0;
    if (!decommit(idx, bytes)) return 0;
    state_[(size_t) idx].store(kAbsent, std::memory_order_release);
    absent_.fetch_add(1, std::memory_order_relaxed);
    decommitted_.fetch_add(bytes, std::memory_order_relaxed);
    return bytes;
}

bool RamDedup::device_stream(int dev, void*& stream) {
    if (dev < 0 && cudaGetDevice(&dev) != cudaSuccess) return false;
    if ((size_t) dev >= devs_.size()) devs_.resize((size_t) dev + 1);
    if (devs_[(size_t) dev].stream == nullptr) {
        UseDevice on(dev);
        cudaStream_t s = nullptr;
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) { (void) cudaGetLastError(); return false; }
        devs_[(size_t) dev].stream = s;
    }
    stream = devs_[(size_t) dev].stream;
    return true;
}

bool RamDedup::evict(int64_t layer, int64_t expert, const void* dev_slot, void* caller_stream, int dev) {
    if (!armed_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return true;
    const int64_t idx = layer * n_expert_ + expert;
    if (state_[(size_t) idx].load(std::memory_order_acquire) == kPresent) return true;
    const auto t0 = Us::now();
    std::lock_guard<std::mutex> lk(mu_);
    const uint8_t st = state_[(size_t) idx].load(std::memory_order_acquire);
    if (st == kPresent) return true;
    if (st == kInFlight) return true;   // an earlier copy back (the same bytes) is still landing: nothing more to do
    if (dev_slot == nullptr) return false;
    if (dev < 0 && cudaGetDevice(&dev) != cudaSuccess) return false;
    UseDevice on(dev);
    void* d2h = nullptr;
    if (!device_stream(dev, d2h)) return false;
    if (in_ring_ == ring_.size()) {   // full: the oldest lands now (a host wait on its copy)
        bounce_waits_.fetch_add(1, std::memory_order_relaxed);
        land_locked(tail_);
        tail_ = (tail_ + 1) % ring_.size();
        --in_ring_;
    }
    const size_t q = head_;
    Slot& s = ring_[q];
    if (s.ev_dev != dev) {
        if (s.ev_before) cudaEventDestroy((cudaEvent_t) s.ev_before);
        if (s.ev_after) cudaEventDestroy((cudaEvent_t) s.ev_after);
        s.ev_before = s.ev_after = nullptr;
        cudaEvent_t a = nullptr, b = nullptr;
        if (cudaEventCreateWithFlags(&a, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&b, cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            if (a) cudaEventDestroy(a);
            return false;
        }
        s.ev_before = a;
        s.ev_after = b;
        s.ev_dev = dev;
    }
    const size_t bytes = (size_t) lay_->blob_bytes(layer);
    cudaStream_t cs = (cudaStream_t) caller_stream, ds = (cudaStream_t) d2h;
    cudaError_t e = cudaEventRecord((cudaEvent_t) s.ev_before, cs);   // after the work the caller's stream holds now
    if (e == cudaSuccess) e = cudaStreamWaitEvent(ds, (cudaEvent_t) s.ev_before, 0);
    if (e == cudaSuccess) e = cudaMemcpyAsync(bounce_ + q * bounce_blob_, dev_slot, bytes, cudaMemcpyDeviceToHost, ds);
    if (e == cudaSuccess) e = cudaEventRecord((cudaEvent_t) s.ev_after, ds);
    if (e == cudaSuccess) e = cudaStreamWaitEvent(cs, (cudaEvent_t) s.ev_after, 0);   // the overwrite waits for the copy
    (void) cudaStreamQuery(ds);   // WDDM: submit now
    if (e != cudaSuccess) {
        (void) cudaGetLastError();
        std::fprintf(stderr, "strata ram dedup: copy back of layer %lld expert %lld failed (%s): it will be re-read from the "
                             "model file when needed\n", (long long) layer, (long long) expert, cudaGetErrorString(e));
        return false;
    }
    s.idx = idx;
    s.live = true;
    slot_of_[(size_t) idx] = (int32_t) q;
    state_[(size_t) idx].store(kInFlight, std::memory_order_release);
    head_ = (head_ + 1) % ring_.size();
    ++in_ring_;
    d2h_.fetch_add(1, std::memory_order_relaxed);
    d2h_us_.fetch_add(since_us(t0), std::memory_order_relaxed);
    return true;
}

bool RamDedup::land_locked(size_t q) {
    Slot& s = ring_[q];
    if (!s.live) return true;
    const auto t0 = Us::now();
    s.live = false;
    const int64_t idx = s.idx;
    s.idx = -1;
    bool ok = true;
    {
        UseDevice on(s.ev_dev);
        ok = cudaEventSynchronize((cudaEvent_t) s.ev_after) == cudaSuccess;
        if (!ok) (void) cudaGetLastError();
    }
    if (idx < 0 || slot_of_[(size_t) idx] != (int32_t) q || state_[(size_t) idx].load(std::memory_order_acquire) != kInFlight)
        return ok;   // superseded (the expert was read back another way meanwhile)
    slot_of_[(size_t) idx] = -1;
    if (!ok || !recommit(idx)) {
        state_[(size_t) idx].store(kAbsent, std::memory_order_release);   // `ensure` re-reads it from the file
        return false;
    }
    const int64_t l = idx / n_expert_, e = idx % n_expert_;
    std::memcpy(base_ + lay_->blob_offset(l, e), bounce_ + q * bounce_blob_, (size_t) lay_->blob_bytes(l));
    uint64_t ps = 0, pe = 0;
    page_span(idx, ps, pe);
    state_[(size_t) idx].store(kPresent, std::memory_order_release);
    absent_.fetch_sub(1, std::memory_order_relaxed);
    decommitted_.fetch_sub(pe - ps, std::memory_order_relaxed);
    d2h_us_.fetch_add(since_us(t0), std::memory_order_relaxed);
    return true;
}

void RamDedup::drain(bool wait) {
    if (!armed_) return;
    std::lock_guard<std::mutex> lk(mu_);
    while (in_ring_ > 0) {
        Slot& s = ring_[tail_];
        if (s.live && !wait) {
            UseDevice on(s.ev_dev);
            const cudaError_t q = cudaEventQuery((cudaEvent_t) s.ev_after);
            if (q == cudaErrorNotReady) break;
            if (q != cudaSuccess) (void) cudaGetLastError();
        }
        land_locked(tail_);
        tail_ = (tail_ + 1) % ring_.size();
        --in_ring_;
    }
}

void RamDedup::sync(int dev) {
    if (!armed_) return;
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t d = 0; d < devs_.size(); ++d) {
        if (devs_[d].stream == nullptr || (dev >= 0 && (int) d != dev)) continue;
        UseDevice on((int) d);
        if (cudaStreamSynchronize((cudaStream_t) devs_[d].stream) != cudaSuccess) (void) cudaGetLastError();
    }
}

bool RamDedup::reread_locked(int64_t idx) {
    const int64_t l = idx / n_expert_, e = idx % n_expert_;
    if (!recommit(idx)) return false;
    uint8_t* dst = base_ + lay_->blob_offset(l, e);
    bool ok = false;
    if (!experts_bin_.empty()) {
        ok = read_file_range(experts_bin_, lay_->blob_offset(l, e), lay_->blob_bytes(l), dst, aligned_tmp_);
    } else if (!gguf_.empty() && lay_->native && lay_->gguf_off.size() >= (size_t) (3 * n_layers_)) {
        const auto& fm = lay_->fmt[(size_t) l];
        const uint64_t per[3] = {fm.up_off, fm.up_off, lay_->bytes[(size_t) l] - fm.down_off};
        uint64_t at = 0;
        ok = true;
        for (int r = 0; r < 3 && ok; ++r) {
            const size_t i = (size_t) (3 * l + r);
            ok = read_file_range(gguf_role_file(gguf_, *lay_, l, r), lay_->gguf_off[i] + (uint64_t) e * per[r], per[r],
                                 dst + at, aligned_tmp_);
            at += per[r];
        }
    }
    if (!ok) return false;
    uint64_t ps = 0, pe = 0;
    page_span(idx, ps, pe);
    state_[(size_t) idx].store(kPresent, std::memory_order_release);
    absent_.fetch_sub(1, std::memory_order_relaxed);
    decommitted_.fetch_sub(pe - ps, std::memory_order_relaxed);
    reread_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool RamDedup::ensure(int64_t layer, int64_t expert) {
    if (!armed_ || layer < 0 || expert < 0 || layer >= n_layers_ || expert >= n_expert_) return false;
    const int64_t idx = layer * n_expert_ + expert;
    if (state_[(size_t) idx].load(std::memory_order_acquire) == kPresent) return true;
    std::lock_guard<std::mutex> lk(mu_);
    const uint8_t st = state_[(size_t) idx].load(std::memory_order_acquire);
    if (st == kPresent) return true;
    if (st == kInFlight && slot_of_[(size_t) idx] >= 0 && land_locked((size_t) slot_of_[(size_t) idx])) return true;
    return reread_locked(idx);
}

}  // namespace strata::core
