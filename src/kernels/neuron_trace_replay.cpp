// Offline replay for SNTv0001 neuron traces. This is deliberately a separate executable: it does not
// alter dispatch, cache policy, or any online expert kernel.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace k = strata::kernels;

namespace {
constexpr int kH = 2560;
constexpr int kFF = 640;
constexpr int kK = 10;
constexpr int kLayers[] = {0, 1, 35, 36};
constexpr uint32_t kMaxWindows = 64;
constexpr uint64_t kMaxEntries = 7680;
constexpr uint64_t kMaxBlobBytes = 16ull * 1024 * 1024;
constexpr uint32_t kKeeps[] = {160, 320, 480};

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little,
              "SNTv0001 currently requires a little-endian host");

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

void read_exact(std::istream& in, void* dst, size_t bytes, const char* what) {
    if (bytes == 0) return;
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(bytes));
    if (!in || static_cast<size_t>(in.gcount()) != bytes) fail(std::string("truncated trace while reading ") + what);
}

template<class T> T read_le(std::istream& in, const char* what) {
    static_assert(std::is_unsigned_v<T>);
    std::array<uint8_t, sizeof(T)> b{};
    read_exact(in, b.data(), b.size(), what);
    T v = 0;
    for (size_t i = 0; i < b.size(); ++i) v |= static_cast<T>(b[i]) << (8 * i);
    return v;
}

int32_t read_i32(std::istream& in, const char* what) { return std::bit_cast<int32_t>(read_le<uint32_t>(in, what)); }
int64_t read_i64(std::istream& in, const char* what) { return std::bit_cast<int64_t>(read_le<uint64_t>(in, what)); }

template<class T> size_t checked_count(uint64_t n, const char* what) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) fail(std::string("trace size overflow: ") + what);
    return static_cast<size_t>(n);
}

bool all_finite(const std::vector<float>& v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}
template<size_t N> bool all_finite(const std::array<float, N>& v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}

struct LayerWindow {
    std::vector<float> x;       // [T,H]
    std::vector<int32_t> ids;   // [T,K]
    std::vector<float> weights; // [T,K]
    std::vector<float> parts;   // [T,K,H]
    std::vector<int32_t> tier;  // [T,K]
};

struct Window {
    uint32_t t = 0, nkeep = 0;
    int64_t pos0 = 0;
    uint64_t serial = 0;
    std::vector<int32_t> tokens;
    std::array<LayerWindow, 4> layer;
};

class TraceReader {
public:
    explicit TraceReader(const fs::path& path) : in_(path, std::ios::binary), path_(path) {
        if (!in_) fail("cannot open trace: " + path.string());
        char magic[8]{};
        read_exact(in_, magic, sizeof(magic), "magic");
        if (std::memcmp(magic, "SNTv0001", 8) != 0) fail("trace magic is not SNTv0001");
        const uint32_t h = read_le<uint32_t>(in_, "H"), ff = read_le<uint32_t>(in_, "FF");
        const uint32_t kk = read_le<uint32_t>(in_, "K"), ll = read_le<uint32_t>(in_, "L");
        if (h != kH || ff != kFF || kk != kK || ll != 4) fail("trace dimensions do not match H=2560 FF=640 K=10 L=4");
        for (int i = 0; i < 4; ++i)
            if (read_i32(in_, "layer ids") != kLayers[i]) fail("trace layer list must be [0,1,35,36]");
    }

    bool next(Window& w) {
        const int peek = in_.peek();
        if (peek == std::char_traits<char>::eof()) {
            if (in_.eof()) return false;
            fail("I/O error while reading trace: " + path_.string());
        }
        if (++windows_ > kMaxWindows) fail("trace exceeds the 64-window bound");
        w = Window{};
        w.t = read_le<uint32_t>(in_, "T");
        w.nkeep = read_le<uint32_t>(in_, "nkeep");
        w.pos0 = read_i64(in_, "pos0");
        w.serial = read_le<uint64_t>(in_, "serial");
        if (w.t < 1 || w.t > 8) fail("trace T is outside 1..8");
        if (w.nkeep == 0 || w.nkeep > w.t) fail("trace nkeep must be in 1..T");
        if (w.pos0 < 0 || w.pos0 > std::numeric_limits<int64_t>::max() - static_cast<int64_t>(w.t - 1))
            fail("trace position range is invalid");
        w.tokens.resize(w.t);
        for (auto& token : w.tokens) {
            token = read_i32(in_, "tokens");
            if (token < 0) fail("negative token id in trace");
        }
        for (auto& l : w.layer) {
            l.x.resize(checked_count<float>(uint64_t(w.t) * kH, "x"));
            l.ids.resize(checked_count<int32_t>(uint64_t(w.t) * kK, "ids"));
            l.weights.resize(checked_count<float>(uint64_t(w.t) * kK, "weights"));
            l.parts.resize(checked_count<float>(uint64_t(w.t) * kK * kH, "parts"));
            l.tier.resize(checked_count<int32_t>(uint64_t(w.t) * kK, "tier"));
            read_exact(in_, l.x.data(), l.x.size() * sizeof(float), "x");
            read_exact(in_, l.ids.data(), l.ids.size() * sizeof(int32_t), "ids");
            read_exact(in_, l.weights.data(), l.weights.size() * sizeof(float), "weights");
            read_exact(in_, l.parts.data(), l.parts.size() * sizeof(float), "parts");
            read_exact(in_, l.tier.data(), l.tier.size() * sizeof(int32_t), "tier");
        }
        return true;
    }

    uint32_t windows() const { return windows_; }

private:
    std::ifstream in_;
    fs::path path_;
    uint32_t windows_ = 0;
};

struct ExpertLayer {
    const strata::TensorInfo* gate = nullptr;
    const strata::TensorInfo* up = nullptr;
    const strata::TensorInfo* down = nullptr;
    const uint8_t* gate_data = nullptr;
    const uint8_t* up_data = nullptr;
    const uint8_t* down_data = nullptr;
    uint64_t experts = 0;
    k::NativeExpertLayout layout{};
};

std::array<ExpertLayer, 4> load_layers(const strata::GgufModel& model, size_t& max_blob) {
    std::array<ExpertLayer, 4> result{};
    max_blob = 0;
    for (size_t li = 0; li < result.size(); ++li) {
        auto& e = result[li];
        const std::string base = "blk." + std::to_string(kLayers[li]) + ".ffn_";
        size_t sg = 0, su = 0, sd = 0;
        e.gate = model.find(base + "gate_exps.weight", &sg);
        e.up = model.find(base + "up_exps.weight", &su);
        e.down = model.find(base + "down_exps.weight", &sd);
        if (!e.gate || !e.up || !e.down) fail("missing expert tensor at layer " + std::to_string(kLayers[li]));
        if (e.gate->shape.size() != 3 || e.up->shape != e.gate->shape || e.down->shape.size() != 3)
            fail("unexpected expert tensor rank/shape at layer " + std::to_string(kLayers[li]));
        if (e.gate->shape[0] != kH || e.gate->shape[1] != kFF || e.down->shape[0] != kFF ||
            e.down->shape[1] != kH || e.down->shape[2] != e.gate->shape[2] || !e.gate->shape[2] ||
            e.gate->type != e.up->type)
            fail("expert tensor dimensions/types do not match H=2560, FF=640 at layer " + std::to_string(kLayers[li]));
        if (!model.in_bounds(*e.gate, sg) || !model.in_bounds(*e.up, su) || !model.in_bounds(*e.down, sd))
            fail("expert tensor payload is out of bounds at layer " + std::to_string(kLayers[li]));
        e.experts = e.gate->shape[2];
        if (e.experts > 4096) fail("expert count exceeds the bounded trace replay limit");
        e.layout = k::native_expert_layout(static_cast<int>(e.gate->type), static_cast<int>(e.down->type), kH, kFF);
        if (!k::native_expert_supported(e.layout.gu_type, e.layout.d_type, kH, kFF) || !e.layout.bytes ||
            e.layout.bytes > kMaxBlobBytes)
            fail("native expert format is unsupported or exceeds the one-blob allocation bound at layer " +
                 std::to_string(kLayers[li]));
        if (strata::tensor_payload_bytes(*e.gate) != e.experts * e.layout.up_off ||
            strata::tensor_payload_bytes(*e.up) != e.experts * e.layout.up_off ||
            strata::tensor_payload_bytes(*e.down) != e.experts * (e.layout.bytes - e.layout.down_off))
            fail("GGUF expert byte layout disagrees with native kernel layout at layer " + std::to_string(kLayers[li]));
        e.gate_data = model.shard(sg).tensor_data(*e.gate);
        e.up_data = model.shard(su).tensor_data(*e.up);
        e.down_data = model.shard(sd).tensor_data(*e.down);
        max_blob = std::max(max_blob, e.layout.bytes);
    }
    return result;
}

uint64_t validate_trace(const fs::path& path, const std::array<ExpertLayer, 4>& layers) {
    TraceReader reader(path);
    std::array<int64_t, 4> last_pos{};
    last_pos.fill(-1);
    uint64_t entries = 0;
    Window w;
    while (reader.next(w)) {
        for (size_t li = 0; li < layers.size(); ++li) {
            const auto& data = w.layer[li];
            if (!all_finite(data.x) || !all_finite(data.weights) || !all_finite(data.parts))
                fail("non-finite trace float at layer " + std::to_string(kLayers[li]));
            for (uint32_t t = 0; t < w.t; ++t) {
                if (t < w.nkeep) {
                    const int64_t pos = w.pos0 + t;
                    if (pos <= last_pos[li]) fail("accepted trace positions are not strictly increasing at layer " + std::to_string(kLayers[li]));
                    last_pos[li] = pos;
                }
                std::set<int32_t> seen;
                for (int j = 0; j < kK; ++j) {
                    const size_t at = size_t(t) * kK + j;
                    const int32_t id = data.ids[at];
                    if (id < 0 || static_cast<uint64_t>(id) >= layers[li].experts)
                        fail("expert id outside the tensor's expert count at layer " + std::to_string(kLayers[li]));
                    if (!seen.insert(id).second) fail("duplicate expert id in a trace token's top-10 list");
                    if (data.tier[at] < 0 || data.tier[at] > 2) fail("tier must be 0 (CPU), 1 (VRAM), or 2 (PCIe)");
                }
            }
        }
        const uint64_t add = uint64_t(w.nkeep) * 4 * kK;
        if (add > kMaxEntries - entries) fail("trace exceeds the 7,680 emitted-entry bound");
        entries += add;
    }
    if (!reader.windows()) fail("trace contains no windows");
    return entries;
}

void validate_replay_window(const Window& w, const std::array<ExpertLayer, 4>& layers,
                            std::array<int64_t, 4>& last_pos) {
    for (size_t li = 0; li < layers.size(); ++li) {
        const auto& data = w.layer[li];
        if (!all_finite(data.x) || !all_finite(data.weights) || !all_finite(data.parts))
            fail("trace changed or contains non-finite floats before replay");
        for (uint32_t t = 0; t < w.t; ++t) {
            if (t < w.nkeep) {
                const int64_t pos = w.pos0 + t;
                if (pos <= last_pos[li]) fail("accepted trace positions changed or are not strictly increasing during replay");
                last_pos[li] = pos;
            }
            std::set<int32_t> seen;
            for (int j = 0; j < kK; ++j) {
                const size_t at = size_t(t) * kK + j;
                const int32_t id = data.ids[at];
                if (id < 0 || static_cast<uint64_t>(id) >= layers[li].experts)
                    fail("expert id changed or is outside the tensor's expert count during replay");
                if (!seen.insert(id).second) fail("duplicate expert id in replay window");
                if (data.tier[at] < 0 || data.tier[at] > 2) fail("invalid tier during replay");
            }
        }
    }
}

class OutputFile {
public:
    explicit OutputFile(const fs::path& final_path) : final_(final_path) {
        const auto parent = final_.parent_path();
        if (!parent.empty() && !fs::exists(parent)) fail("output directory does not exist");
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
#ifdef _WIN32
        const auto pid = static_cast<unsigned long long>(_getpid());
#else
        const auto pid = static_cast<unsigned long long>(getpid());
#endif
        for (unsigned attempt = 0; attempt < 32; ++attempt) {
            temp_ = final_;
            temp_ += ".tmp." + std::to_string(pid) + "." + std::to_string(stamp) + "." + std::to_string(attempt);
            file_ = open_exclusive(temp_);
            if (file_) break;
            if (errno != EEXIST) fail("cannot create temporary output: " + temp_.string());
        }
        if (!file_) fail("cannot reserve a unique temporary output path");
    }
    ~OutputFile() {
        if (file_) std::fclose(file_);
        if (!published_ && !temp_.empty()) { std::error_code ec; fs::remove(temp_, ec); }
    }
    OutputFile(const OutputFile&) = delete;
    OutputFile& operator=(const OutputFile&) = delete;

    void bytes(const void* data, size_t n) {
        if (n && std::fwrite(data, 1, n, file_) != n) fail("failed writing temporary replay output");
    }
    void u32(uint32_t v) { write_le(v); }
    void i32(int32_t v) { write_le(std::bit_cast<uint32_t>(v)); }
    void u64(uint64_t v) { write_le(v); }
    void i64(int64_t v) { write_le(std::bit_cast<uint64_t>(v)); }
    void f32(float v) { write_le(std::bit_cast<uint32_t>(v)); }

    void publish() {
        if (std::fflush(file_) != 0) fail("flush failed for replay output");
#ifdef _WIN32
        if (_commit(_fileno(file_)) != 0) fail("flush-to-disk failed for replay output");
#else
        if (fsync(fileno(file_)) != 0) fail("flush-to-disk failed for replay output");
#endif
        if (std::fclose(file_) != 0) { file_ = nullptr; fail("close failed for replay output"); }
        file_ = nullptr;
#ifdef _WIN32
        if (!MoveFileW(temp_.c_str(), final_.c_str())) {
            const DWORD e = GetLastError();
            fail(e == ERROR_ALREADY_EXISTS || e == ERROR_FILE_EXISTS ? "output already exists" :
                 "atomic no-overwrite output publish failed (Windows error " + std::to_string(e) + ")");
        }
#else
        if (::link(temp_.c_str(), final_.c_str()) != 0) {
            if (errno == EEXIST) fail("output already exists");
            fail("atomic no-overwrite output publish failed: " + std::string(std::strerror(errno)));
        }
        if (::unlink(temp_.c_str()) != 0) fail("published output but could not remove its temporary link");
#endif
        published_ = true;
    }

private:
    static std::FILE* open_exclusive(const fs::path& p) {
#ifdef _WIN32
        std::FILE* f = nullptr;
        const errno_t e = _wfopen_s(&f, p.c_str(), L"wbx");
        if (e) { errno = e; return nullptr; }
        return f;
#else
        return std::fopen(p.c_str(), "wbx");
#endif
    }
    template<class T> void write_le(T v) {
        static_assert(std::is_unsigned_v<T>);
        std::array<uint8_t, sizeof(T)> b{};
        for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint8_t>(v >> (8 * i));
        bytes(b.data(), b.size());
    }
    fs::path final_, temp_;
    std::FILE* file_ = nullptr;
    bool published_ = false;
};

#define CUDA_CHECK(expr) do { const cudaError_t e_ = (expr); if (e_ != cudaSuccess) \
    fail(std::string(#expr) + ": " + cudaGetErrorString(e_)); } while (0)

class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreate(&s_)); }
    ~Stream() { if (s_) cudaStreamDestroy(s_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return s_; }
private:
    cudaStream_t s_ = nullptr;
};

template<class T = uint8_t> class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
        if (!bytes_) fail("attempted zero-byte CUDA allocation");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p_), bytes_));
    }
    ~DeviceBuffer() { if (p_) cudaFree(p_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() { return p_; }
    const T* get() const { return p_; }
private:
    T* p_ = nullptr;
    size_t bytes_ = 0;
};

void copy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t s) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s));
}
void copy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t s) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
}

using KeepMask = std::array<uint8_t, kFF>;
KeepMask top_mask(const float* h, int keep) {
    std::array<int, kFF> order{};
    for (int i = 0; i < kFF; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + keep, order.end(), [h](int a, int b) {
        const float aa = std::fabs(h[a]), ab = std::fabs(h[b]);
        return aa == ab ? a < b : aa > ab;
    });
    KeepMask mask{};
    for (int i = 0; i < keep; ++i) mask[order[i]] = 1;
    return mask;
}

void down_project(const uint8_t* d_blob, const k::NativeExpertLayout& L, const float* h,
                  const KeepMask* mask, DeviceBuffer<float>& d_values, DeviceBuffer<uint8_t>& d_q8,
                  DeviceBuffer<float>& d_out, std::vector<float>& out, cudaStream_t stream) {
    std::array<float, kFF> pruned{};
    for (int i = 0; i < kFF; ++i) pruned[i] = (!mask || (*mask)[i]) ? h[i] : 0.0f;
    copy_h2d(d_values.get(), pruned.data(), sizeof(pruned), stream);
    k::native_quantize_q8_1(d_values.get(), d_q8.get(), kFF, 1, stream);
    k::native_mmvq(L.d_type, d_blob + L.down_off, d_q8.get(), d_out.get(), kFF, kH, 1, stream);
    copy_d2h(out.data(), d_out.get(), out.size() * sizeof(float), stream);
    if (!all_finite(out)) fail("non-finite down-projection output during replay");
}

void down_project_q8(const uint8_t* d_blob, const k::NativeExpertLayout& L, const void* d_hq,
                     DeviceBuffer<float>& d_out, std::vector<float>& out, cudaStream_t stream) {
    k::native_mmvq(L.d_type, d_blob + L.down_off, d_hq, d_out.get(), kFF, kH, 1, stream);
    copy_d2h(out.data(), d_out.get(), out.size() * sizeof(float), stream);
    if (!all_finite(out)) fail("non-finite original-Q8_1 down-projection output during replay");
}

uint64_t pair_key(int layer, int expert) {
    return (uint64_t(static_cast<uint32_t>(layer)) << 32) | static_cast<uint32_t>(expert);
}

void write_header(OutputFile& out) {
    out.bytes("SNRv0001", 8);
    out.u32(kH); out.u32(kFF); out.u32(kK); out.u32(4);
    for (int l : kLayers) out.i32(l);
}

void replay_trace(const fs::path& trace_path, const fs::path& output_path, const strata::GgufModel& model,
                  const std::array<ExpertLayer, 4>& layers, uint64_t expected_entries, size_t max_blob) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    if (prop.major * 10 + prop.minor < 75) fail("neuron replay requires an NVIDIA device with CC 7.5 or newer");

    size_t scratch_bytes = k::native_expert_scratch_bytes(1, kFF);
    const size_t fa = (size_t(kFF) * sizeof(float) + 255) & ~size_t(255);
    const size_t qbytes = k::native_q8_1_bytes(kFF);
    if (scratch_bytes < 3 * fa + qbytes) fail("native expert scratch layout is smaller than expected");

    Stream stream;
    DeviceBuffer<uint8_t> d_blob(max_blob), d_scratch(scratch_bytes), d_xq(k::native_q8_1_bytes(kH));
    DeviceBuffer<float> d_x(kH * sizeof(float)), d_group_out(kH * sizeof(float));
    DeviceBuffer<uint8_t> d_down_q8(qbytes);
    DeviceBuffer<float> d_down_in(kFF * sizeof(float)), d_down_out(kH * sizeof(float));
    DeviceBuffer<unsigned long long> d_ptr(sizeof(unsigned long long));
    DeviceBuffer<int32_t> d_start(2 * sizeof(int32_t)), d_ngroup(sizeof(int32_t));
    DeviceBuffer<int32_t> d_dst(sizeof(int32_t)), d_tok(sizeof(int32_t));
    const int32_t start[2] = {0, 1}, one = 1, zero = 0;
    copy_h2d(d_start.get(), start, sizeof(start), stream.get());
    copy_h2d(d_ngroup.get(), &one, sizeof(one), stream.get());
    copy_h2d(d_dst.get(), &zero, sizeof(zero), stream.get());
    copy_h2d(d_tok.get(), &zero, sizeof(zero), stream.get());
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));

    OutputFile out(output_path);
    write_header(out);
    TraceReader reader(trace_path);
    std::map<uint64_t, std::array<float, kFF>> last_h;
    std::set<std::pair<int, int>> v2_checked;
    uint64_t emitted = 0;
    std::array<int64_t, 4> replay_last_pos{};
    replay_last_pos.fill(-1);
    Window w;
    while (reader.next(w)) {
        validate_replay_window(w, layers, replay_last_pos);
        for (uint32_t t = 0; t < w.nkeep; ++t) {
            const int64_t pos = w.pos0 + t;
            for (size_t li = 0; li < layers.size(); ++li) {
                const auto& el = layers[li];
                const auto& wl = w.layer[li];
                const float* x = wl.x.data() + size_t(t) * kH;
                copy_h2d(d_x.get(), x, kH * sizeof(float), stream.get());
                k::quantize_q8_1_rows(d_x.get(), 1, kH, d_xq.get(), stream.get());
                for (int j = 0; j < kK; ++j) {
                    if (++emitted > expected_entries || emitted > kMaxEntries) fail("replay entry count exceeded validated bound");
                    const size_t at = size_t(t) * kK + j;
                    const int expert = wl.ids[at];
                    std::vector<uint8_t> blob(el.layout.bytes);
                    const size_t e = static_cast<size_t>(expert);
                    std::memcpy(blob.data(), el.gate_data + e * el.layout.up_off, el.layout.up_off);
                    std::memcpy(blob.data() + el.layout.up_off, el.up_data + e * el.layout.up_off, el.layout.up_off);
                    const size_t down_bytes = el.layout.bytes - el.layout.down_off;
                    std::memcpy(blob.data() + el.layout.down_off, el.down_data + e * down_bytes, down_bytes);
                    copy_h2d(d_blob.get(), blob.data(), blob.size(), stream.get());
                    const unsigned long long blob_ptr = reinterpret_cast<unsigned long long>(d_blob.get());
                    copy_h2d(d_ptr.get(), &blob_ptr, sizeof(blob_ptr), stream.get());

                    const bool had_last = last_h.find(pair_key(kLayers[li], expert)) != last_h.end();
                    const auto prev = had_last ? last_h.at(pair_key(kLayers[li], expert)) : std::array<float, kFF>{};
                    const bool v1_before = true;
                    k::native_grouped_set_v1(v1_before);
                    k::native_expert_grouped(el.layout, d_ptr.get(), d_start.get(), d_ngroup.get(), d_dst.get(), d_tok.get(),
                                             1, 1, d_xq.get(), d_scratch.get(), d_group_out.get(), stream.get());
                    std::array<float, kH> grouped{};
                    std::array<float, kFF> h{};
                    std::vector<uint8_t> hq_v1(qbytes);
                    copy_d2h(grouped.data(), d_group_out.get(), sizeof(grouped), stream.get());
                    copy_d2h(h.data(), reinterpret_cast<uint8_t*>(d_scratch.get()) + 2 * fa, sizeof(h), stream.get());
                    copy_d2h(hq_v1.data(), reinterpret_cast<uint8_t*>(d_scratch.get()) + 3 * fa, qbytes, stream.get());
                    if (!all_finite(grouped) || !all_finite(h))
                        fail("non-finite grouped replay output or hidden activation");

                    const auto fmt = std::make_pair(el.layout.gu_type, el.layout.d_type);
                    if (!v2_checked.count(fmt)) {
                        k::native_grouped_set_v1(false);
                        k::native_expert_grouped(el.layout, d_ptr.get(), d_start.get(), d_ngroup.get(), d_dst.get(), d_tok.get(),
                                                 1, 1, d_xq.get(), d_scratch.get(), d_group_out.get(), stream.get());
                        std::array<float, kH> grouped_v2{};
                        std::vector<uint8_t> hq_v2(qbytes);
                        copy_d2h(grouped_v2.data(), d_group_out.get(), sizeof(grouped_v2), stream.get());
                        copy_d2h(hq_v2.data(), reinterpret_cast<uint8_t*>(d_scratch.get()) + 3 * fa, qbytes, stream.get());
                        if (std::memcmp(grouped.data(), grouped_v2.data(), sizeof(grouped)) != 0 ||
                            std::memcmp(hq_v1.data(), hq_v2.data(), qbytes) != 0)
                            fail("native grouped v1/v2 output or Q8_1 differs for format " + std::to_string(fmt.first) +
                                 "/" + std::to_string(fmt.second));
                        v2_checked.insert(fmt);
                        std::printf("v1/v2 bitwise PASS for format %d/%d\n", fmt.first, fmt.second);
                    }

                    std::vector<float> mmvqfull(kH);
                    // Null mask means retain every h value; this is the original GPU-replayed activation, not trace.parts.
                    copy_h2d(d_down_q8.get(), hq_v1.data(), qbytes, stream.get());
                    down_project_q8(d_blob.get(), el.layout, d_down_q8.get(), d_down_out, mmvqfull, stream.get());
                    std::array<std::vector<float>, 3> oracle;
                    for (size_t m = 0; m < 3; ++m) {
                        const KeepMask mask = top_mask(h.data(), static_cast<int>(kKeeps[m]));
                        oracle[m].resize(kH);
                        down_project(d_blob.get(), el.layout, h.data(), &mask, d_down_in, d_down_q8, d_down_out,
                                     oracle[m], stream.get());
                    }
                    std::array<std::vector<float>, 3> last;
                    if (had_last) {
                        for (size_t m = 0; m < 3; ++m) {
                            const KeepMask mask = top_mask(prev.data(), static_cast<int>(kKeeps[m]));
                            last[m].resize(kH);
                            down_project(d_blob.get(), el.layout, h.data(), &mask, d_down_in, d_down_q8, d_down_out,
                                         last[m], stream.get());
                        }
                    } else {
                        for (auto& v : last) v = mmvqfull;
                    }

                    out.u64(w.serial); out.i64(pos); out.i32(kLayers[li]); out.i32(expert);
                    out.i32(wl.tier[at]); out.i32(el.layout.gu_type); out.i32(el.layout.d_type);
                    out.i32(had_last ? 0 : 1); out.f32(wl.weights[at]);
                    for (float v : h) out.f32(v);
                    for (float v : grouped) out.f32(v);
                    for (float v : mmvqfull) out.f32(v);
                    for (const auto& v : oracle) for (float z : v) out.f32(z);
                    for (const auto& v : last) for (float z : v) out.f32(z);
                    last_h[pair_key(kLayers[li], expert)] = h;
                }
            }
        }
    }
    if (emitted != expected_entries) fail("replay entry count disagrees with validated trace");
    if (v2_checked.size() != 0) std::printf("checked grouped v1/v2 bitwise for %zu native format pair(s)\n", v2_checked.size());
    out.publish();
    std::printf("replayed %llu accepted expert entries into %s\n",
                static_cast<unsigned long long>(emitted), output_path.string().c_str());
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: neuron_trace_replay <trace.snt> <model-shard.gguf> <output.snr>\n");
        return 2;
    }
    try {
        const fs::path trace_path = fs::u8path(argv[1]);
        const fs::path shard_path = fs::u8path(argv[2]);
        const fs::path output_path = fs::u8path(argv[3]);
        if (fs::exists(output_path)) fail("output already exists; refusing to overwrite it");
        const strata::GgufModel model(strata::gguf_split_paths(shard_path.string()));
        size_t max_blob = 0;
        const auto layers = load_layers(model, max_blob);
        const uint64_t entries = validate_trace(trace_path, layers); // validate the complete input before any GPU work
        replay_trace(trace_path, output_path, model, layers, entries, max_blob);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "neuron_trace_replay: %s\n", e.what());
        return 1;
    }
}
