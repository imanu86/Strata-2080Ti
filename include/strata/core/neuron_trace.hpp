// Default-off, bounded expert replay inputs. Host buffers outlive every captured graph.
#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::core {

class NeuronTrace {
public:
    static constexpr int H = 2560, FF = 640, K = 10, L = 4, MaxT = 8, MaxWindows = 64;
    inline static constexpr int32_t Layers[L] = {0, 1, 35, 36};
    NeuronTrace() = default;
    NeuronTrace(const NeuronTrace&) = delete;
    NeuronTrace& operator=(const NeuronTrace&) = delete;
    // The owner must drain/destroy the graphs before destroying this object.
    ~NeuronTrace() {
        if (file_ && std::fclose(file_) != 0)
            std::fprintf(stderr, "strata neuron trace: close failed: %s\n", std::strerror(errno));
        if (host_) cudaFreeHost(host_);
    }

    bool init(const char* path, int max_t, std::string& err) {
        const uint16_t endian = 1;
        if (*(const uint8_t*) &endian != 1 || sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559)
            return fail(err, "SNTv0001 requires little-endian IEEE-754 float32");
        if (max_t < 1 || max_t > MaxT) return fail(err, "invalid window capacity");
        max_t_ = max_t;
        if (const char* limit = std::getenv("STRATA_NEURON_TRACE_WINDOWS")) {
            char* end = nullptr;
            const long value = std::strtol(limit, &end, 10);
            if (!*limit || *end || value < 1 || value > 512) return fail(err, "trace windows must be 1..512");
            max_windows_ = (int) value;
        }
        cap_ = max_t * K;
        // Separate layer slots, with token-group slices at their original token offsets.
        size_t bytes = 0;
        auto carve = [&](bool assign) {
            size_t at = 0;
            auto take = [&](size_t n) -> uint8_t* {
                uint8_t* p = assign ? host_ + at : nullptr;
                at += (n + 63) & ~(size_t) 63;
                return p;
            };
            for (auto& s : snapshots_) {
                s.x = (float*) take((size_t) max_t * H * 4);
                s.ids = (int32_t*) take((size_t) cap_ * 4);
                s.w = (float*) take((size_t) cap_ * 4);
                s.parts = (float*) take((size_t) cap_ * H * 4);
                for (auto& p : s.plan) {
                    p.counts = (int32_t*) take(3 * 4);
                    p.start = (int32_t*) take((size_t) (cap_ + 1) * 4);
                    p.dst = (int32_t*) take((size_t) cap_ * 4);
                    p.start2 = (int32_t*) take((size_t) (cap_ + 1) * 4);
                }
            }
            return at;
        };
        bytes = carve(false);
        const cudaError_t ce = cudaHostAlloc((void**) &host_, bytes, cudaHostAllocDefault);
        if (ce != cudaSuccess) return fail(err, std::string("pinned allocation: ") + cudaGetErrorString(ce));
        std::memset(host_, 0, bytes);
        carve(true);
        record_.reserve(max_record_bytes());
#if defined(_WIN32)
        const int fd = _open(path, _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
        const int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
#endif
        if (fd < 0) return fail(err, std::string("exclusive creation failed (existing files are never overwritten): ") + std::strerror(errno));
#if defined(_WIN32)
        file_ = _fdopen(fd, "wb");
#else
        file_ = fdopen(fd, "wb");
#endif
        if (!file_) {
            const int saved = errno;
#if defined(_WIN32)
            _close(fd);
#else
            ::close(fd);
#endif
            return fail(err, std::string("fdopen failed: ") + std::strerror(saved));
        }
        // A failed write must not leave stdio bytes that could reappear after truncating the partial append.
        if (std::setvbuf(file_, nullptr, _IONBF, 0) != 0) return fail(err, "cannot select unbuffered diagnostic output");
        static constexpr char magic[8] = {'S', 'N', 'T', 'v', '0', '0', '0', '1'};
        append(magic, sizeof magic);
        u32(H); u32(FF); u32(K); u32(L);
        append(Layers, sizeof Layers);
        if (!write_record(err)) return false;
        std::fprintf(stderr, "strata neuron trace: %s; layers 0,1,35,36; at most %d decode windows; diagnostic timing only\n", path, max_windows_);
        return true;
    }

    // Called while recording each graph, never allocates or synchronizes.
    bool capture(int64_t layer, int grp, int tb, int n, const float* x, const int32_t* ids,
                 const float* w, const float* parts, const int32_t* plan, cudaStream_t cs, std::string& err) {
        int index = -1;
        for (int i = 0; i < L; ++i) if (Layers[i] == layer) index = i;
        if (index < 0) return true;
        if (grp < 0 || grp > 1 || tb < 0 || n < 1 || tb + n > max_t_) return fail(err, "invalid captured slice");
        auto& s = snapshots_[(size_t) index];
        auto& p = s.plan[grp];
        auto copy = [&](void* dst, const void* src, size_t bytes) {
            const cudaError_t ce = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, cs);
            return ce == cudaSuccess || fail(err, std::string("captured copy: ") + cudaGetErrorString(ce));
        };
        const size_t ptr_off = ((4 + (size_t) (cap_ + 1) + 2 * cap_) + 1) & ~(size_t) 1;
        return copy(s.x + (size_t) tb * H, x + (size_t) tb * H, (size_t) n * H * 4) &&
               copy(s.ids + tb * K, ids + tb * K, (size_t) n * K * 4) &&
               copy(s.w + tb * K, w + tb * K, (size_t) n * K * 4) &&
               copy(s.parts + (size_t) tb * K * H, parts + (size_t) tb * K * H, (size_t) n * K * H * 4) &&
               copy(p.counts, plan, 3 * 4) &&
               copy(p.start, plan + 4, (size_t) (cap_ + 1) * 4) &&
               copy(p.dst, plan + 4 + cap_ + 1, (size_t) cap_ * 4) &&
               copy(p.start2, plan + ptr_off + 4 * cap_, (size_t) (cap_ + 1) * 4);
    }

    void begin_run() {
        // A caller can abandon a verified window (cancellation/retry). It must not become a sample or
        // prevent a subsequent prompt. Successful commits are the only operation that writes records.
        if (pending_)
            std::fprintf(stderr, "strata neuron trace: discarded uncommitted window %llu at %lld\n",
                         (unsigned long long) serial_, (long long) pos0_);
        pending_ = false;
    }
    void complete_run(int t, const int32_t* tokens, int64_t pos0, uint64_t serial, int groups, bool decode) {
        pending_ = decode && written_ < max_windows_;
        if (!pending_) return;
        t_ = t; pos0_ = pos0; serial_ = serial; groups_ = groups;
        std::memcpy(tokens_.data(), tokens, (size_t) t * 4);
    }
    bool pending() const { return pending_; }

    // The run and its commit have both completed successfully before this is called.
    bool committed(int n_keep, std::string& err) {
        if (!pending_) return true;
        if (n_keep < 1 || n_keep > t_ || !file_) return fail(err, "invalid committed snapshot");
        record_.clear();
        u32((uint32_t) t_); u32((uint32_t) n_keep);
        u64((uint64_t) pos0_); u64(serial_);
        append(tokens_.data(), (size_t) t_ * 4);
        for (const auto& s : snapshots_) {
            std::array<int32_t, MaxT * K> tier{};
            for (int grp = 0; grp < groups_; ++grp) {
                const int tb = grp == 0 ? 0 : (t_ + 1) / 2;
                const int te = groups_ == 1 || grp == 1 ? t_ : (t_ + 1) / 2;
                const int entries_max = (te - tb) * K;
                const auto& p = s.plan[grp];
                const int nv = p.counts[0], ne = p.counts[1], np = p.counts[2];
                if (nv < 0 || nv > entries_max || ne < 0 || ne > entries_max || np < 0 || np > entries_max)
                    return fail(err, "invalid plan counts");
                const int vram_end = p.start[nv];
                if (p.start[0] != 0 || vram_end < 0 || vram_end > ne || p.start2[0] != vram_end || p.start2[np] != ne)
                    return fail(err, "invalid plan boundaries");
                for (int j = 0; j < nv; ++j)
                    if (p.start[j] < 0 || p.start[j] >= p.start[j + 1] || p.start[j + 1] > vram_end)
                        return fail(err, "invalid resident group");
                for (int j = 0; j < np; ++j)
                    if (p.start2[j] < vram_end || p.start2[j] >= p.start2[j + 1] || p.start2[j + 1] > ne)
                        return fail(err, "invalid PCIe group");
                for (int j = 0; j < ne; ++j) {
                    const int dst = p.dst[j];
                    if (dst < 0 || dst >= entries_max || tier[(size_t) tb * K + dst] != 0)
                        return fail(err, "invalid or duplicate plan destination");
                    tier[(size_t) tb * K + dst] = j < vram_end ? 1 : 2;
                }
            }
            for (int j = 0; j < t_ * K; ++j)
                if (s.ids[j] < 0 || s.ids[j] >= 512) return fail(err, "expert id outside the supported geometry");
            append(s.x, (size_t) t_ * H * 4);
            append(s.ids, (size_t) t_ * K * 4);
            append(s.w, (size_t) t_ * K * 4);
            append(s.parts, (size_t) t_ * K * H * 4);
            append(tier.data(), (size_t) t_ * K * 4);
        }
        if (record_.size() > max_record_bytes()) return fail(err, "record exceeds its fixed bound");
        if (!write_record(err)) return false;
        pending_ = false;
        ++written_;
        if (written_ == max_windows_) {
            const int status = std::fclose(file_);
            file_ = nullptr;
            if (status != 0) return fail(err, std::string("close failed: ") + std::strerror(errno));
            std::fprintf(stderr, "strata neuron trace: completed %d decode windows; capture buffers retained for graph lifetime\n", max_windows_);
        }
        return true;
    }

private:
    struct Plan { int32_t *counts = nullptr, *start = nullptr, *dst = nullptr, *start2 = nullptr; };
    struct Snapshot {
        float *x = nullptr, *w = nullptr, *parts = nullptr;
        int32_t* ids = nullptr;
        Plan plan[2];
    };
    std::array<Snapshot, L> snapshots_{};
    uint8_t* host_ = nullptr;
    std::FILE* file_ = nullptr;
    std::vector<uint8_t> record_;
    std::array<int32_t, MaxT> tokens_{};
    int max_t_ = 0, cap_ = 0, written_ = 0, t_ = 0, groups_ = 1;
    int max_windows_ = MaxWindows;
    int64_t pos0_ = 0;
    uint64_t serial_ = 0;
    bool pending_ = false;
    static constexpr size_t max_record_bytes() {
        // 3,608,376 B at T=8; the 40-byte header plus 64 records is at most 230,936,104 B.
        return 24 + MaxT * 4 + L * MaxT * (H * 4 + K * 12 + K * H * 4);
    }
    static bool fail(std::string& err, const std::string& why) { err = "neuron trace: " + why; return false; }
    void append(const void* data, size_t bytes) {
        const uint8_t* p = (const uint8_t*) data;
        record_.insert(record_.end(), p, p + bytes);
    }
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) record_.push_back((uint8_t) (v >> (8 * i))); }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) record_.push_back((uint8_t) (v >> (8 * i))); }
    bool write_record(std::string& err) {
#if defined(_WIN32)
        const auto start = _ftelli64(file_);
#else
        const auto start = ftello(file_);
#endif
        if (start < 0) return fail(err, "cannot determine the output offset");
        if (std::fwrite(record_.data(), 1, record_.size(), file_) == record_.size() && std::fflush(file_) == 0) return true;
        const int saved = errno ? errno : EIO;
        // Roll back a failed append so a parser cannot mistake a partial record for a sample.
        std::clearerr(file_);
        std::fflush(file_);
#if defined(_WIN32)
        const bool truncated = _chsize_s(_fileno(file_), (uint64_t) start) == 0;
        _fseeki64(file_, start, SEEK_SET);
#else
        const bool truncated = ftruncate(fileno(file_), start) == 0;
        fseeko(file_, start, SEEK_SET);
#endif
        return fail(err, std::string("write failed: ") + std::strerror(saved) +
                         (truncated ? "; incomplete append removed" : "; rollback failed, discard this file"));
    }
};

}  // namespace strata::core
