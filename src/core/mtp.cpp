// src/core/mtp.cpp - see include/strata/core/mtp.hpp.
#include "strata/core/mtp.hpp"
#include "strata/core/coupled_draft.hpp"
#include "strata/core/spec_prob.hpp"
#include "strata/core/on_device.hpp"

#include "strata/core/native_head.hpp"
#include "strata/core/peer_experts.hpp"
#include "strata/core/prefix_cache_file.hpp"
#include <filesystem>
#include <cmath>
#include <cstddef>
#include "strata/platform/memory.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/rope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <vector>
#if defined(_WIN32)
#include <intrin.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif

#ifdef STRATA_NATIVE_EXPERTS
#include "strata/kernels/iq_kernels.hpp"
#include "ggml.h"   // --mtp-q4: the head rows' and projections' formats, dequantized and requantized at load
#endif

namespace strata::core {
namespace {

constexpr float EPS = 1e-6f;

// #783 PR-i: STRATA_MTP_CATCHUP_ALL=1 catches the drafter's K/V up for the whole verified window, rejected rows included
bool mtp_catchup_all() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_MTP_CATCHUP_ALL");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}
bool mtp_prefill_per_group_sync() {
    static const bool on = [] {
        if (const char* v = std::getenv("STRATA_MTP_PREFILL_SYNC")) return std::atoi(v) != 0;
#if defined(STRATA_USE_HIP)
        return true;
#else
        return false;
#endif
    }();
    return on;
}
constexpr int GGML_Q8_0 = 8;
constexpr int GGML_Q4_0 = 2;
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

bool mapped(size_t bytes, void** h, void** d) {
    if (cudaHostAlloc(h, bytes, cudaHostAllocMapped | (peer_portable() ? cudaHostAllocPortable : 0)) != cudaSuccess) return false;
    std::memset(*h, 0, bytes);
    return cudaHostGetDevicePointer(d, *h, 0) == cudaSuccess;
}

strata::kernels::QsaShapes shapes_of(const ModelGeometry& g) {
    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
    s.n_head = g.n_head;
    s.n_head_kv = g.n_head_kv;
    s.head_dim = g.head_dim;
    s.idx_n_head = g.idx_q_heads;
    s.idx_dim = g.idx_key_dim;
    return s;
}

// 64-bit seek/tell on a `FILE*`: `fseek`/`ftell` take a 32-bit `long` on Windows and would wrap past 2 GiB.
#if defined(_WIN32)
#define STRATA_FILE_SEEK64(f, o, w) _fseeki64((f), (long long) (o), (w))
#define STRATA_FILE_TELL64(f) _ftelli64(f)
#else
#define STRATA_FILE_SEEK64(f, o, w) fseeko((f), (off_t) (o), (w))
#define STRATA_FILE_TELL64(f) ftello(f)
#endif

bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    // Loader fix (0.1.15+loaderfix.2): `ifstream::read` reaches the disk as 4095-byte reads on MSVC - the same
    // split `load_experts_ranges` had - so the drafter's 111 MiB dense blob paid 4 KiB per operation on a cold
    // start.  `fread` passes a request bigger than the stream buffer straight to `ReadFile()`.
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    // A `FILE*` has no destructor and the short-read path below returns early: the guard closes on every path.
    struct Closer {
        FILE* f;
        ~Closer() { if (f != nullptr) std::fclose(f); }
    } closer{f};
    if (STRATA_FILE_SEEK64(f, 0, SEEK_END) != 0) return false;
    const long long n = (long long) STRATA_FILE_TELL64(f);
    if (n < 0 || STRATA_FILE_SEEK64(f, 0, SEEK_SET) != 0) return false;
#if !defined(_WIN32)
    strata::platform::advise_willneed(fileno(f), 0, (uint64_t) n);
#endif
    out.resize((size_t) n);
    return n == 0 || std::fread(out.data(), 1, (size_t) n, f) == (size_t) n;
}

}  // namespace

MtpDrafter::~MtpDrafter() {
    const OnDevice on_device(device_);
    if (cs_) cudaStreamSynchronize(cs_);
    if (source_cs_) {
        const OnDevice on_source(source_device_);
        cudaStreamSynchronize(source_cs_);
        cudaStreamDestroy(source_cs_);
    }
    for (auto& slot : source_slots_) {
        if (slot.consumed) cudaEventDestroy(slot.consumed);
        if (slot.host) cudaFreeHost(slot.host);
    }
    if (anchor_) cudaFree(anchor_);
    if (anchor_valid_) cudaFree(anchor_valid_);
    if (anchor_stats_) cudaFree(anchor_stats_);
    if (anchor_h_meta_) cudaFreeHost(anchor_h_meta_);
    if (observer_markers_) cudaFree(observer_markers_);
    if (observer_host_markers_) cudaFreeHost(observer_host_markers_);
    if (observer_native_) cudaFree(observer_native_);
    if (observer_target_) cudaFree(observer_target_);
    if (observer_host_) cudaFreeHost(observer_host_);
    if (observer_h_meta_) cudaFreeHost(observer_h_meta_);
    if (feature_bank_) cudaFree(feature_bank_);
    if (feature_fixture_) cudaFree(feature_fixture_);
    if (feature_mask_) cudaFree(feature_mask_);
    if (feature_stats_) cudaFree(feature_stats_);
    if (feature_host_) cudaFreeHost(feature_host_);
    if (feature_h_meta_) cudaFreeHost(feature_h_meta_);
    if (first_top2_host_) cudaFreeHost(first_top2_host_);
    if (prefill_host_R_) cudaFreeHost(prefill_host_R_);
    if (local_window_R_) cudaFree(local_window_R_);
    if (private_gr_arena_) cudaFree(private_gr_arena_);
    for (auto& family : hq_round_) for (auto& e : family) if (e) cudaGraphExecDestroy(e);
    for (auto& family : hq_step_) for (auto& e : family) if (e) cudaGraphExecDestroy(e);
    for (auto& e : selective_round_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : selective_step_) if (e) cudaGraphExecDestroy(e);
    if (selective_arena_) cudaFree(selective_arena_);
    if (selective_h_ids_) cudaFreeHost(selective_h_ids_);
    if (selective_h_proxy_) cudaFreeHost(selective_h_proxy_);
    for (auto& e : prefill_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : prefill_dev_exec_) if (e) cudaGraphExecDestroy(e);
    if (pf_dev_) cudaFree(pf_dev_);
    if (dense4_) cudaFree(dense4_);
    for (auto& e : round_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : round_exec_c_) if (e) cudaGraphExecDestroy(e);
    for (auto& e : step_exec_c_) if (e) cudaGraphExecDestroy(e);
    if (cparams_) cudaFree(cparams_);
    if (cring_) cudaFree(cring_);
    if (dinv_) cudaFree(dinv_);
    if (cscratch_) cudaFree(cscratch_);
    if (h_cparams_) cudaFreeHost(h_cparams_);
    if (h_chist_) cudaFreeHost(h_chist_);
    if (h_q_) cudaFreeHost(h_q_);
    if (cs_) cudaStreamDestroy(cs_);
    if (side_) cudaStreamDestroy(side_);
    if (sh_fork_) cudaEventDestroy(sh_fork_);
    if (sh_join_) cudaEventDestroy(sh_join_);
    if (owns_weights_ && dense_) cudaFree(dense_);
    if (owns_weights_ && experts_) cudaFree(experts_);
    if (state_arena_) cudaFree(state_arena_);
    if (arena_) cudaFree(arena_);
    if (head_logits_) cudaFree(head_logits_);
    if (owns_draft_head_ && dhead_) { strata::kernels::native_q6_k_unpack(dhead_); cudaFree(dhead_); }
    if (owns_draft_head_ && dvocab_) cudaFree(dvocab_);
    if (ev_chain_) cudaEventDestroy(ev_chain_);
    for (cudaEvent_t e : ev_step_) if (e) cudaEventDestroy(e);
    void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_, h_force_};
    for (void* h : hosts) if (h) cudaFreeHost(h);
}

namespace {
constexpr uint64_t hq_slab_bytes = 1415577600ull;
const char* hq_names[3] = {"q2legacy", "q2native", "q4native"};
uint64_t hq_bytes(int mode) { return mode == 2 ? hq_slab_bytes : hq_slab_bytes / 2; }
bool hq_file_size(const std::string& path, uint64_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    const bool ok = STRATA_FILE_SEEK64(f, 0, SEEK_END) == 0 &&
                    STRATA_FILE_TELL64(f) == (int64_t) bytes;
    std::fclose(f); return ok;
}
}

bool MtpDrafter::hq_validate(const ModelGeometry& g, const MtpDrafter* shared, std::string& err) {
#if !defined(STRATA_NATIVE_EXPERTS) || defined(STRATA_USE_HIP)
    err = "mtp hq: CUDA native expert build required"; return false;
#else
    if (shared || selective_config_ || !force_on_ || g.n_embd != 2560 || g.n_ff != 640 || g.n_expert != 512) {
        err = "mtp hq: unsupported geometry, sharing or selective configuration"; return false;
    }
    std::ifstream in(hq_pack_ + "/experts.txt");
    std::string magic, extra; int ne = 0, h = 0, ff = 0;
    if (!(in >> magic >> ne >> h >> ff) || magic != "STRATA_MTP_HQ_V1" || ne != 512 || h != 2560 || ff != 640) {
        err = "mtp hq: invalid versioned experts.txt geometry"; return false;
    }
    for (int i = 0; i < 3; ++i) {
        std::string name, layout, sha; int gu = -1, down = -1; uint64_t stride = 0, bytes = 0;
        const int type = i == 2 ? 2 : 42;
        const std::string path = i == 0 ? rt_dir_ + "/experts.bin" : hq_pack_ + "/" + hq_names[i] + ".bin";
        if (!(in >> name >> layout >> gu >> down >> stride >> bytes >> sha) || name != hq_names[i] ||
            layout != (i == 0 ? "legacy-planar" : "native-gguf") || gu != type || down != type ||
            stride != hq_bytes(i) / 512 || bytes != hq_bytes(i) || sha.size() != 64 ||
            !std::all_of(sha.begin(), sha.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }) ||
            !hq_file_size(path, bytes) || !strata::kernels::native_expert_supported(type, type, h, ff) ||
            strata::kernels::native_expert_layout(type, type, h, ff).bytes != stride) {
            err = "mtp hq: invalid layout/type/stride/hash declaration/file size for " + std::string(hq_names[i]); return false;
        }
        hq_sha_[i] = sha;
    }
    if (in >> extra) { err = "mtp hq: trailing experts.txt data"; return false; }
    std::fprintf(stderr, "strata lab mtp hq: allocation=retained slab_bytes=%llu families=q2legacy,q2native,q4native "
                         "prefill=front_only q2native=bit_repack native_activation=q8_1 legacy_activation=q8_scaled "
                         "hash_validation=manifest_declared\n", (unsigned long long) hq_slab_bytes);
    return true;
#endif
}

bool MtpDrafter::hq_switch(int mode, std::string& err) {
    const OnDevice on_device(device_);
    if (hq_pack_.empty() || mode < 0 || mode > 2 || chain_live_ || coupled_active_ || selective_config_) {
        err = "mtp hq: invalid mode or draft chain not idle"; return false;
    }
    // Every possible producer/reader is idle before overwriting the fixed slab. No mode change during decode.
    if ((cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) ||
        (side_ && cudaStreamSynchronize(side_) != cudaSuccess)) {
        err = "mtp hq: stream synchronization failed"; return false;
    }
    for (auto& count : hq_chains_) count = 0;
    hq_upload_bytes_ = 0; hq_upload_ms_ = 0;
    if (mode == hq_mode_) return true;
    const auto start = Clock::now();
    const uint64_t bytes = hq_bytes(mode);
    const std::string path = mode == 0 ? rt_dir_ + "/experts.bin" : hq_pack_ + "/" + hq_names[mode] + ".bin";
    if (!hq_file_size(path, bytes)) { err = "mtp hq: pack size changed"; return false; }
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "mtp hq: cannot reopen expert pack"; return false; }
    struct Closer { FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    std::vector<uint8_t> chunk(64u << 20); // GEN boundary only, before prompt/decode timers
    for (uint64_t off = 0; off < bytes;) {
        const size_t n = (size_t) std::min<uint64_t>(chunk.size(), bytes - off);
        if (std::fread(chunk.data(), 1, n, f) != n ||
            cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "mtp hq: expert upload failed; mixed slab cannot be used"; return false;
        }
        off += n;
    }
    if (std::fgetc(f) != EOF || std::ferror(f) || cudaStreamSynchronize(cs_) != cudaSuccess ||
        (side_ && cudaStreamSynchronize(side_) != cudaSuccess)) {
        err = "mtp hq: expert upload completion failed"; return false;
    }
    hq_mode_ = mode; hq_upload_bytes_ = bytes; hq_upload_ms_ = ms_since(start);
    return true;
}

bool MtpDrafter::hq_report(std::FILE* out, int64_t request, std::string& err) const {
    if (hq_pack_.empty()) return true;
    if (chain_live_) { err = "mtp hq: report while chain live"; return false; }
    std::fprintf(out, "strata lab mtp hq: request %lld mode=%s slab_bytes=%llu upload_bytes=%llu upload_ms=%.17g "
                      "q2legacy_chains=%lld q2native_chains=%lld q4native_chains=%lld expert_path=%s ggml_gu=%d ggml_down=%d weights_sha256_declared=%s "
                      "hash_validation=manifest_declared\n", (long long) request, hq_names[hq_mode_],
                 (unsigned long long) hq_slab_bytes, (unsigned long long) hq_upload_bytes_, hq_upload_ms_,
                 (long long) hq_chains_[0], (long long) hq_chains_[1], (long long) hq_chains_[2],
                 hq_mode_ == 0 ? "legacy_planar" : "native_gguf", hq_mode_ == 2 ? 2 : 42,
                 hq_mode_ == 2 ? 2 : 42, hq_sha_[hq_mode_].c_str());
    return std::ferror(out) == 0;
}

// Lab selective MTP uses only its OWN authoritative K/V. The target provides logical positions, never values.

bool MtpDrafter::anchor_allocate(std::string& err) {
    if (!force_on_ || feature_config_ || selective_config_ || !hq_pack_.empty() || hnorm_stream_ ||
        g_->hc != 4 || g_->n_embd != 2560) {
        err = "mtp anchor: unsupported geometry or graph configuration"; return false;
    }
    if (cudaMalloc((void**) &anchor_, 10240 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &anchor_valid_, sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc((void**) &anchor_stats_, 6 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaHostAlloc((void**) &anchor_h_meta_, 2 * sizeof(int32_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &anchor_m_meta_, anchor_h_meta_, 0) != cudaSuccess) {
        err = "mtp anchor: common buffers do not fit"; return false;
    }
    std::memset(anchor_h_meta_, 0, 2 * sizeof(int32_t));
    anchor_bytes_ = 10240 * sizeof(float) + sizeof(int32_t) + 6 * sizeof(unsigned long long);
    vram_ += anchor_bytes_;
    std::fprintf(stderr, "strata lab mtp anchor: allocation=retained row_floats=10240 vram_bytes=%llu "
                         "mapped_bytes=8 graph_gate=all_modes alpha_set=0,0.25 future_inputs=0\n",
                 (unsigned long long) anchor_bytes_);
    return true;
}

bool MtpDrafter::anchor_begin(bool on, std::string& err) {
    const OnDevice device(device_);
    if (!anchor_config_ || chain_live_) { err = "mtp anchor: request boundary is not idle"; return false; }
    if (cudaStreamSynchronize(cs_) != cudaSuccess || (side_ && cudaStreamSynchronize(side_) != cudaSuccess)) {
        err = "mtp anchor: request drain failed"; return false;
    }
    anchor_on_ = on; anchor_fresh_ = anchor_forced_ = 0;
    anchor_h_meta_[0] = on ? 1 : 0; anchor_h_meta_[1] = 0;
    if (cudaMemsetAsync(anchor_valid_, 0, sizeof(int32_t), cs_) != cudaSuccess ||
        cudaMemsetAsync(anchor_stats_, 0, 6 * sizeof(unsigned long long), cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) {
        err = "mtp anchor: request reset failed"; return false;
    }
    return true;
}

bool MtpDrafter::anchor_report(std::FILE* out, int64_t request, std::string& err) {
    const OnDevice device(device_);
    if (!anchor_config_ || chain_live_) { err = "mtp anchor: report while not idle"; return false; }
    unsigned long long s[6] = {};
    if (cudaStreamSynchronize(cs_) != cudaSuccess ||
        cudaMemcpy(s, anchor_stats_, sizeof s, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "mtp anchor: counter read failed"; return false;
    }
    std::fprintf(out, "strata lab mtp anchor: request %lld alpha=%s fresh_chains=%llu forced_chains=%llu "
                     "seed_valid=%llu seed_invalid=%llu applied_steps=%llu off_steps=%llu excluded_steps=%llu "
                     "invalid_steps=%llu vram_bytes=%llu row_floats=10240 first_head_direct=unchanged future_inputs=0\n",
                 (long long) request, anchor_on_ ? "0.25" : "0", (unsigned long long) anchor_fresh_,
                 (unsigned long long) anchor_forced_, s[0], s[1], s[2], s[3], s[4], s[5],
                 (unsigned long long) anchor_bytes_);
    return std::ferror(out) == 0;
}

bool MtpDrafter::feature_allocate(std::string& err) {
    if (!force_on_ || selective_config_ || !hq_pack_.empty() || g_->hc != 4 || g_->n_embd != 2560) {
        err = "mtp feature: unsupported geometry or graph configuration"; return false;
    }
    const size_t bytes = (size_t) 1024 * 4 * 2560 * sizeof(float);
    if (cudaMalloc((void**) &feature_bank_, bytes) != cudaSuccess ||
        cudaMalloc((void**) &feature_fixture_, 1024 * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc((void**) &feature_mask_, sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc((void**) &feature_stats_, 6 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaHostAlloc((void**) &feature_host_, bytes, cudaHostAllocPortable) != cudaSuccess ||
        cudaHostAlloc((void**) &feature_h_meta_, 4 * sizeof(int32_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &feature_m_meta_, feature_h_meta_, 0) != cudaSuccess) {
        err = "mtp feature: equal-allocation buffers do not fit"; return false;
    }
    std::memset(feature_h_meta_, 0, 4 * sizeof(int32_t));
    feature_bytes_ = bytes + 1024 * sizeof(int32_t) + sizeof(int32_t) + 6 * sizeof(unsigned long long);
    vram_ += feature_bytes_;
    std::fprintf(stderr, "strata lab mtp feature: allocation=retained table_rows=1024 row_floats=10240 "
                         "vram_bytes=%llu graph_gate=all_modes intervention=Rin_only fresh_only=1\n",
                 (unsigned long long) feature_bytes_);
    return true;
}

bool MtpDrafter::feature_begin(int mode, const std::string& path, const uint8_t* identity,
                              const std::vector<int64_t>& prompt, const std::vector<int32_t>& fixture,
                              int64_t requested, std::string& err) {
    const OnDevice on_device(device_);
    if (!feature_config_ || mode < 0 || mode > 2 || prompt.empty() || fixture.size() != 1024 ||
        requested < 1 || requested > 1024 || (mode == 1 && requested != 1024)) {
        err = "mtp feature: invalid request or fixture (exactly1024 required)"; return false;
    }
    while (chain_live_) if (chain_poll(err) < 0) return false;
    if (cudaStreamSynchronize(cs_) != cudaSuccess || (side_ && cudaStreamSynchronize(side_) != cudaSuccess)) {
        err = "mtp feature: request drain failed"; return false;
    }
    const auto started = Clock::now();
    feature_mode_ = mode; feature_path_ = path; feature_requested_ = requested;
    feature_prompt_last_ = (int32_t) prompt.back();
    feature_captured_ = feature_bank_rows_ = 0;
    feature_fresh_ = feature_forced_ = feature_root_mismatch_ = 0;
    FeatureHeader expected{};
    std::memcpy(expected.magic, "SMTPR1\0", 8);
    expected.version = 1; expected.hc = 4; expected.n_embd = 2560; expected.rows = 1024;
    expected.prompt_tokens = (int64_t) prompt.size(); expected.first_pos = expected.prompt_tokens - 1;
    std::memcpy(expected.identity, identity, 32);
    try {
        const auto ph = prefix_cache_digest(prompt.data(), prompt.size() * sizeof(int64_t));
        const auto fh = prefix_cache_digest(fixture.data(), fixture.size() * sizeof(int32_t));
        std::memcpy(expected.prompt, ph.data(), 32); std::memcpy(expected.fixture, fh.data(), 32);
        feature_header_ = expected;
        const bool exists = std::filesystem::exists(path);
        if (mode == 1 && exists) { err = "mtp feature: refusing to overwrite an existing bank"; return false; }
        if (mode != 1 && exists) {
            std::ifstream in(path, std::ios::binary);
            FeatureHeader got{};
            if (!in.read((char*) &got, sizeof(got)) || std::memcmp(&got, &expected, offsetof(FeatureHeader, payload)) != 0) {
                err = "mtp feature: bank identity/shape/prompt/fixture mismatch"; return false;
            }
            const size_t bytes = (size_t) 1024 * 10240 * sizeof(float);
            if (!in.read((char*) feature_host_, bytes) || in.peek() != std::char_traits<char>::eof()) {
                err = "mtp feature: incomplete or trailing bank payload"; return false;
            }
            const auto payload = prefix_cache_digest(feature_host_, bytes);
            if (std::memcmp(got.payload, payload.data(), 32) != 0) { err = "mtp feature: payload SHA256 mismatch"; return false; }
            for (size_t i = 0; i < bytes / sizeof(float); ++i) if (!std::isfinite(feature_host_[i])) {
                err = "mtp feature: nonfinite bank residual"; return false;
            }
            if (cudaMemcpy(feature_bank_, feature_host_, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = "mtp feature: bank upload failed"; return false;
            }
            feature_header_ = got; feature_bank_rows_ = 1024;
        } else if (mode == 2 || (mode == 0 && requested != 15)) {
            err = "mtp feature: bank missing (only off15 bootstrap may omit it)"; return false;
        }
    } catch (const std::exception& e) { err = std::string("mtp feature: ") + e.what(); return false; }
    std::memcpy(feature_ids_, fixture.data(), sizeof(feature_ids_));
    if (cudaMemcpy(feature_fixture_, feature_ids_, sizeof(feature_ids_), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(feature_stats_, 0, 6 * sizeof(unsigned long long)) != cudaSuccess) {
        err = "mtp feature: request metadata upload failed"; return false;
    }
    feature_load_ms_ = ms_since(started);
    return true;
}

bool MtpDrafter::feature_record(const float* R, int64_t pos, const int32_t* input, int keep, std::string& err) {
    if (!feature_config_ || feature_mode_ != 1) return true;
    const OnDevice on_device(device_);
    if (keep < 1 || pos != feature_header_.first_pos + feature_captured_ || feature_captured_ >= 1024) {
        err = "mtp feature: noncontiguous committed residual capture"; return false;
    }
    const int n = std::min(keep, 1024 - feature_captured_);
    for (int i = 0; i < n; ++i) {
        const int at = feature_captured_ + i;
        // The first row is the prompt's last input; all subsequent inputs are earlier forced outputs.
        if (input[i] != (at == 0 ? feature_prompt_last_ : feature_ids_[at - 1])) {
            err = "mtp feature: captured input row does not follow the fixture"; return false;
        }
    }
    // Instrumented record GEN only. Source producer has finished; copy completes before verifier parity reuse.
    if (cudaMemcpy(feature_host_ + (size_t) feature_captured_ * 10240, R,
                   (size_t) n * 10240 * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "mtp feature: committed residual readback failed"; return false;
    }
    feature_captured_ += n;
    return true;
}

bool MtpDrafter::feature_finish(std::FILE* out, int64_t request, int64_t produced, const char* finish, std::string& err) {
    if (!feature_config_) return true;
    const OnDevice on_device(device_);
    while (chain_live_) if (chain_poll(err) < 0) return false;
    if (produced != feature_requested_ || std::strcmp(finish, "length") != 0) {
        err = "mtp feature: incomplete fixed-continuation request"; return false;
    }
    if (feature_mode_ == 1) {
        if (feature_captured_ != 1024) { err = "mtp feature: incomplete bank capture"; return false; }
        try {
            const size_t bytes = (size_t) 1024 * 10240 * sizeof(float);
            for (size_t i = 0; i < bytes / sizeof(float); ++i) if (!std::isfinite(feature_host_[i])) {
                err = "mtp feature: nonfinite captured residual"; return false;
            }
            const auto hash = prefix_cache_digest(feature_host_, bytes);
            std::memcpy(feature_header_.payload, hash.data(), 32);
            if (std::filesystem::exists(feature_path_)) { err = "mtp feature: bank appeared during capture"; return false; }
            std::ofstream file(feature_path_, std::ios::binary);
            file.write((const char*) &feature_header_, sizeof(feature_header_));
            file.write((const char*) feature_host_, bytes); file.flush();
            if (!file.good()) { err = "mtp feature: bank write failed"; return false; }
            feature_bank_rows_ = 1024;
        } catch (const std::exception& e) { err = std::string("mtp feature: ") + e.what(); return false; }
    }
    unsigned long long stats[6]{};
    if (cudaMemcpy(stats, feature_stats_, sizeof(stats), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "mtp feature: counter readback failed"; return false;
    }
    char digest[65]{};
    for (int i = 0; i < 32; ++i) std::snprintf(digest + 2 * i, 3, "%02x", feature_header_.payload[i]);
    std::fprintf(out, "strata lab mtp feature: request %lld mode=%s bank_rows=%d captured_rows=%d "
                      "eligible_steps=%llu injected_steps=%llu mismatch_steps=%llu bounds_steps=%llu latched_steps=%llu excluded_steps=%llu "
                      "fresh_chains=%lld forced_chains=%lld root_mismatch_chains=%lld vram_bytes=%llu bank_load_ms=%.3f "
                      "record_instrumented=%d target_features_synthetic=%d draft_token_substitutions=0 draft_logit_substitutions=0 target_forced=1 payload_sha256=%s\n",
                 (long long) request, feature_mode_ == 1 ? "record" : feature_mode_ == 2 ? "feature" : "off",
                 feature_bank_rows_, feature_captured_, stats[0], stats[1], stats[2], stats[3], stats[4], stats[5],
                 (long long) feature_fresh_, (long long) feature_forced_, (long long) feature_root_mismatch_,
                 (unsigned long long) feature_bytes_, feature_load_ms_, feature_mode_ == 1, feature_mode_ == 2, digest);
    return true;
}

bool MtpDrafter::selective_allocate(std::string& err) {
    using namespace strata::kernels;
#if defined(STRATA_USE_HIP)
    err = "mtp selective: CUDA only"; return false;
#endif
    const QsaShapes shape = shapes_of(*g_);
    selective_supported_ = st_.kv_mode == 2 && st_.kv_int8 && !st_.kv_q4 && !st_.kv_hybrid &&
        st_.host.k_q && st_.host.v_q && st_.host.k_scale && st_.host.v_scale && window_ == 32768 &&
        shape.page_size == 4 && shape.head_dim == 256 && shape.n_head_kv == 2 && max_t_ <= 8;
    if (!selective_supported_) {
        std::fprintf(stderr, "strata lab selective: unsupported storage; dense fallback (requires int8 ring32768)\n");
        return true;
    }
    const int64_t blocks = (st_.max_cells + shape.page_size - 1) / shape.page_size;
    const uint64_t rows = (uint64_t) kSelectiveSlots * shape.n_head_kv * shape.page_size;
    auto carve = [&](Bump& b) {
        selective_pools_.k_q = b.take<int8_t>(rows * shape.head_dim);
        selective_pools_.v_q = b.take<int8_t>(rows * shape.head_dim);
        selective_pools_.k_scale = b.take<uint16_t>(rows * (shape.head_dim / KV_Q8_GROUP));
        selective_pools_.v_scale = b.take<uint16_t>(rows * (shape.head_dim / KV_Q8_GROUP));
        selective_map_.page_table = b.take<int32_t>(blocks);
        selective_map_.slot_block = b.take<int32_t>(kSelectiveSlots);
        selective_map_.slot_stamp = b.take<int32_t>(kSelectiveSlots);
        selective_map_.slot_ref = b.take<int32_t>(kSelectiveSlots);
        selective_map_.ctl = b.take<int32_t>(kKvCtlInts);
        selective_map_.miss_block = b.take<int32_t>(kSelectiveSlots);
        selective_map_.miss_slot = b.take<int32_t>(kSelectiveSlots);
        selective_d_ids_ = b.take<int32_t>(kSelectiveCap);
    };
    Bump count; carve(count);
    if (cudaMalloc(&selective_arena_, count.used) != cudaSuccess) { err = "mtp selective: private pool allocation"; return false; }
    Bump real; real.base = (uint8_t*) selective_arena_; carve(real);
    selective_map_.n_blocks = blocks; selective_map_.n_slots = kSelectiveSlots;
    selective_pools_.page_table = selective_map_.page_table;
    if (!mapped((size_t) 8 * kSelectiveCap * sizeof(int32_t), (void**) &selective_h_ids_, (void**) &selective_m_ids_) ||
        !mapped((size_t) kSelectiveProxy * sizeof(int32_t), (void**) &selective_h_proxy_, (void**) &selective_m_proxy_)) {
        err = "mtp selective: fixed host buffers"; return false;
    }
    kv_stream_reset(selective_map_, cs_);
    if (cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp selective: initial map reset"; return false; }
    vram_ += count.used; selective_bytes_ = count.used;
    std::fprintf(stderr, "strata lab selective: allocation=retained variants=dense,8192 pool_slots=%d pool_bytes=%llu "
                         "selection_cap=%d attention_chunks=161 original_chunks=%lld\n", kSelectiveSlots,
                 (unsigned long long) count.used, kSelectiveCap, (long long) ((cap_ + 63) / 64));
    return true;
}

bool MtpDrafter::selective_begin_request(uint64_t epoch, int64_t full_until, bool on, std::string& err) {
    if (!selective_config_) { err = "mtp selective: not configured before load"; return false; }
    if (chain_live_) { err = "mtp selective: request changed with a live chain"; return false; }
    if (!idle(err)) return false;
    selective_on_ = on; selective_epoch_ = epoch;
    selective_full_until_ = full_until >= 0 && full_until <= st_.max_cells ? full_until : -1;
    selective_source_ok_ = false; selective_active_ = false; selective_reset_ = true; selective_last_valid_ = false;
    selective_source_reason_ = 3; selective_proxy_width_ = selective_last_n_ = 0;
    selective_chains_ = selective_dense_ = selective_resets_ = selective_ids_ = selective_mirror_cells_ = 0;
    selective_check_cells_ = selective_check_distant_ = 0; selective_source_ms_ = 0;
    selective_far8192_ = selective_far32768_ = selective_max_union_ = 0;
    std::fill(std::begin(selective_fallbacks_), std::end(selective_fallbacks_), 0);
    return true;
}

bool MtpDrafter::selective_source(const int32_t* ids, int width, int64_t pos, int source_device, uint64_t epoch,
                                  std::string& err) {
    selective_source_ok_ = false; selective_source_reason_ = 3;
    if (!selective_config_ || !selective_on_ || !selective_supported_ || selective_full_until_ < 0) return true;
    if (chain_live_) { err = "mtp selective: replacing a live chain's fixed inputs"; return false; }
    if (!ids || width < 1 || width > kSelectiveProxy || pos < 0 || pos >= st_.max_cells ||
        width != std::min<int64_t>(pos + 1, kSelectiveProxy) || source_device != device_ || epoch != selective_epoch_)
        return true;
    const OnDevice on_device(device_);
    const auto t0 = Clock::now();
    if (cudaMemcpyAsync(selective_h_proxy_, ids, (size_t) width * sizeof(int32_t), cudaMemcpyDeviceToHost, cs_) != cudaSuccess ||
        cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp selective: copying target selection"; return false; }
    selective_source_ms_ += ms_since(t0);
    // The verifier's block top-k emits ascending, unique, causal IDs. Reject unexpected metadata before GPU resolve.
    for (int i = 0; i < width; ++i)
        if (selective_h_proxy_[i] < 0 || selective_h_proxy_[i] > pos ||
            (i > 0 && selective_h_proxy_[i] <= selective_h_proxy_[i - 1])) {
            selective_source_reason_ = 4; return true;
        }
    selective_source_pos_ = pos; selective_proxy_width_ = width; selective_source_ok_ = true;
    return true;
}

void MtpDrafter::selective_fallback(int reason) {
    selective_active_ = false; selective_source_ok_ = false; selective_reset_ = true; selective_last_valid_ = false;
    ++selective_dense_; ++selective_fallbacks_[std::max(0, std::min(reason, 7))];
}

bool MtpDrafter::selective_stage(int T, int64_t p, int a, int n_out, std::string& err) {
    using namespace strata::kernels;
    selective_active_ = false;
    const bool full_history = selective_full_until_ >= 0 && p >= 0 && p <= selective_full_until_;
    // Only the accepted catch-up becomes authoritative. Future recursive outputs remain private branch state.
    selective_full_until_ = full_history ? p + a + 1 : -1;
    if (!selective_on_) { selective_fallback(0); return true; }
    if (!selective_supported_) { selective_fallback(1); return true; }
    if (!full_history) { selective_fallback(2); return true; }
    if (!selective_source_ok_ || selective_source_pos_ != p + a) { selective_fallback(selective_source_reason_); return true; }
    if (p + T > st_.max_cells || p + a + n_out > st_.max_cells || n_out > 7) { selective_fallback(4); return true; }
    if (!selective_round_[T]) { selective_fallback(7); return true; }
    for (int j = 1; j < n_out; ++j) if (!selective_step_[j]) { selective_fallback(7); return true; }
    // A cell named by the last query of the previous chain is certainly resident until the next resolve.
    // Count catch-up overwrites for the diagnostic, before replacing the immutable host selection rows.
    if (selective_last_valid_ && !selective_reset_) {
        const int32_t* prev = selective_h_ids_ + (size_t) (selective_last_n_ - 1) * kSelectiveCap;
        const int width = selective_widths_[selective_last_n_ - 1];
        for (int t = 0; t < T; ++t)
            if (std::binary_search(prev, prev + width, (int32_t) (p + t))) ++selective_mirror_cells_;
    }
    int64_t total_ids = 0, far8192 = 0, far32768 = 0, max_union = 0;
    for (int j = 0; j < n_out; ++j) {
        const int64_t pos = p + a + j;
        const int64_t begin = std::max<int64_t>(0, pos - kSelectiveRecent + 1);
        int32_t* out = selective_h_ids_ + (size_t) j * kSelectiveCap;
        int q = 0, n = 0, pages = 0, last_page = -1;
        int64_t recent = begin;
        while (q < selective_proxy_width_ || recent <= pos) {
            const int64_t proxy = q < selective_proxy_width_ ? selective_h_proxy_[q] : st_.max_cells;
            const int64_t id = recent <= pos ? std::min(proxy, recent) : proxy;
            if (q < selective_proxy_width_ && proxy == id) ++q;
            if (recent <= pos && recent == id) ++recent;
            if (id < 0 || id > pos || id >= st_.max_cells || n >= kSelectiveCap) { selective_fallback(4); return true; }
            const int page = (int) (id / 4);
            if (page != last_page) { last_page = page; ++pages; }
            if (pages > kSelectiveSlots) { selective_fallback(5); return true; } // before resolve's atomic miss writes
            if (id < begin) ++far8192;
            if (id < pos - 32768 + 1) ++far32768;
            out[n++] = (int32_t) id;
        }
        std::fill(out + n, out + kSelectiveCap, 0); // unused graph padding is never attended
        selective_widths_[j] = n; total_ids += n; max_union = std::max<int64_t>(max_union, n);
    }
    if (selective_reset_) {
        kv_stream_reset(selective_map_, cs_);
        if (cudaGetLastError() != cudaSuccess) { err = "mtp selective: resetting private map"; return false; }
        ++selective_resets_; selective_reset_ = false;
    }
    h_step_[(2 * max_t_ - 1) * 4 + kStepWidth] = selective_widths_[0];
    for (int j = 1; j < n_out; ++j) h_step_[(max_t_ + j - 1) * 4 + kStepWidth] = selective_widths_[j];
    selective_active_ = true; selective_source_ok_ = false; selective_last_valid_ = true;
    selective_last_n_ = n_out; selective_last_pos_ = p + a + n_out - 1;
    ++selective_chains_; selective_ids_ += total_ids;
    selective_far8192_ += far8192; selective_far32768_ += far32768;
    selective_max_union_ = std::max(selective_max_union_, max_union);
    return true;
}

bool MtpDrafter::selective_check(std::string& err) {
    using namespace strata::kernels;
    if (!selective_supported_ || !selective_last_valid_ || selective_chains_ == 0 || chain_live_) {
        err = "mtp selective: parity gate has no completed selective chain"; return false;
    }
    const OnDevice on_device(device_);
    if (!idle(err)) return false;
    const QsaShapes shape = shapes_of(*g_);
    const size_t row_bytes = (size_t) shape.head_dim, scale_bytes = (size_t) (shape.head_dim / KV_Q8_GROUP) * 2;
    const size_t pool_rows = (size_t) kSelectiveSlots * shape.n_head_kv * shape.page_size;
    const size_t host_rows = (size_t) selective_map_.n_blocks * shape.n_head_kv * shape.page_size;
    std::vector<int32_t> pages((size_t) selective_map_.n_blocks);
    int32_t ctl[kKvCtlInts] = {};
    if (cudaMemcpy(pages.data(), selective_map_.page_table, pages.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(ctl, selective_map_.ctl, sizeof ctl, cudaMemcpyDeviceToHost) != cudaSuccess || ctl[3] != 0) {
        err = "mtp selective: parity map/overflow failure"; return false;
    }
    const int32_t* ids = selective_h_ids_ + (size_t) (selective_last_n_ - 1) * kSelectiveCap;
    const int n = selective_widths_[selective_last_n_ - 1];
    const void* pools[4] = {selective_pools_.k_q, selective_pools_.v_q, selective_pools_.k_scale, selective_pools_.v_scale};
    const void* hosts[4] = {st_.host.k_q, st_.host.v_q, st_.host.k_scale, st_.host.v_scale};
    // Diagnostic only, after decode_ms was frozen. Bulk reads avoid thousands of tiny WDDM transactions.
    for (int run = 0; run < 4; ++run) {
        const size_t stride = run < 2 ? row_bytes : scale_bytes;
        std::vector<uint8_t> got(pool_rows * stride), expected(host_rows * stride);
        if (cudaMemcpy(got.data(), pools[run], got.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(expected.data(), hosts[run], expected.size(), cudaMemcpyDefault) != cudaSuccess) {
            err = "mtp selective: parity readback failure"; return false;
        }
        for (int i = 0; i < n; ++i) {
            const int id = ids[i], block = id / (int) shape.page_size, slot = pages[(size_t) block];
            if (slot < 0 || slot >= kSelectiveSlots) { err = "mtp selective: selected parity cell is not resident"; return false; }
            for (int h = 0; h < shape.n_head_kv; ++h) {
                const size_t in_page = (size_t) h * shape.page_size + id % shape.page_size;
                const size_t actual = ((size_t) slot * shape.n_head_kv * shape.page_size + in_page) * stride;
                const size_t reference = ((size_t) block * shape.n_head_kv * shape.page_size + in_page) * stride;
                if (std::memcmp(got.data() + actual, expected.data() + reference, stride) != 0) {
                    err = "mtp selective: private/host mismatch at cell " + std::to_string(id) + " run " + std::to_string(run);
                    return false;
                }
            }
        }
    }
    selective_check_cells_ = n; selective_check_distant_ = 0;
    for (int i = 0; i < n; ++i) if (ids[i] < selective_last_pos_ - kSelectiveRecent + 1) ++selective_check_distant_;
    return true;
}

bool MtpDrafter::selective_report(std::FILE* out, int64_t request, std::string& err) const {
    const OnDevice on_device(device_);
    int32_t ctl[strata::kernels::kKvCtlInts] = {};
    const bool counter_ok = selective_supported_ && selective_resets_ > 0 &&
        cudaMemcpy(ctl, selective_map_.ctl, sizeof ctl, cudaMemcpyDeviceToHost) == cudaSuccess;
    if (selective_supported_ && selective_resets_ > 0 && (!counter_ok || ctl[3] != 0)) {
        err = "mtp selective: final map readback or overflow failure"; return false;
    }
    uint64_t misses = 0, lookups = 0, calls = 0;
    if (counter_ok) { std::memcpy(&misses, ctl + 4, 8); std::memcpy(&lookups, ctl + 6, 8); std::memcpy(&calls, ctl + 8, 8); }
    std::fprintf(out, "strata lab selective: request %lld mode=%d selective_chains=%lld dense_chains=%lld resets=%lld "
                     "pool_pages=%d pool_bytes=%llu ids=%lld ids_recent=%lld ids_beyond8192=%lld ids_beyond32768=%lld "
                     "max_union=%lld mirror_overlap_cells=%lld full_until=%lld history_epoch=%llu source_ms=%.3f "
                     "fallback_off=%lld fallback_storage=%lld fallback_history=%lld fallback_source=%lld "
                     "fallback_bounds=%lld fallback_pages=%lld fallback_reserved=%lld fallback_graph=%lld "
                     "map_counters_ok=%d map_counter_epoch=%lld map_miss_pages=%llu map_upload_bytes=%llu "
                     "map_lookups=%llu map_calls=%llu map_overflow=%d "
                     "check_pass=%d check_mismatches=0 check_cells=%lld check_distant=%lld check_recent=%lld check_outside_decode=1\n",
                 (long long) request, selective_on_ ? kSelectiveRecent : 0, (long long) selective_chains_,
                 (long long) selective_dense_, (long long) selective_resets_, selective_supported_ ? kSelectiveSlots : 0,
                 (unsigned long long) selective_bytes_, (long long) selective_ids_,
                 (long long) (selective_ids_ - selective_far8192_), (long long) selective_far8192_,
                 (long long) selective_far32768_, (long long) selective_max_union_,
                 (long long) selective_mirror_cells_, (long long) selective_full_until_, (unsigned long long) selective_epoch_,
                 selective_source_ms_, (long long) selective_fallbacks_[0], (long long) selective_fallbacks_[1],
                 (long long) selective_fallbacks_[2], (long long) selective_fallbacks_[3], (long long) selective_fallbacks_[4],
                 (long long) selective_fallbacks_[5], (long long) selective_fallbacks_[6], (long long) selective_fallbacks_[7],
                 counter_ok ? 1 : 0, (long long) selective_resets_, (unsigned long long) misses,
                 (unsigned long long) (misses * 4224), (unsigned long long) lookups, (unsigned long long) calls,
                 counter_ok ? ctl[3] : -1, selective_check_cells_ > 0 ? 1 : 0,
                 (long long) selective_check_cells_, (long long) selective_check_distant_,
                 (long long) (selective_check_cells_ - selective_check_distant_));
    return true;
}

bool MtpDrafter::idle(std::string& err) {
    const OnDevice on_device(device_);
    if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: its stream failed"; return false; }
    return true;
}

const float* MtpDrafter::f32(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "f32") return (const float*) (dense_ + t.off);
    return nullptr;
}
const uint16_t* MtpDrafter::bf16(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "bf16") return (const uint16_t*) (dense_ + t.off);
    return nullptr;
}
const void* MtpDrafter::q8(const char* name) const {
    for (const auto& t : tensors_) if (t.name == name && t.kind == "q8_0") return dense_ + t.off;
    return nullptr;
}
const void* MtpDrafter::wq(const char* name, int& type) const {
    if (dense4_ != nullptr)
        for (const auto& e : q4_off_)
            if (e.first == name) { type = GGML_Q4_0; return dense4_ + e.second; }
    type = GGML_Q8_0;
    return q8(name);
}

// --mtp-q4: every Q8_0 tensor of dense.bin as Q4_0 (ggml's reference quantizer), in one device buffer
bool MtpDrafter::make_q4_dense(const std::vector<uint8_t>& blob, std::string& err) {
#ifdef STRATA_NATIVE_EXPERTS
    uint64_t total = 0;
    std::vector<std::pair<const Tensor*, uint64_t>> plan;
    for (const auto& t : tensors_) {
        if (t.kind != "q8_0" || t.cols % 32 != 0) continue;
        plan.push_back({&t, total});
        total += ((uint64_t) t.rows * (uint64_t) (t.cols / 32) * 18 + 255) & ~255ull;
    }
    std::vector<uint8_t> host((size_t) total);
    std::vector<float> row;
    const auto* q8t = ggml_get_type_traits(GGML_TYPE_Q8_0);
    for (const auto& [t, off] : plan) {
        row.resize((size_t) t->cols);
        const uint64_t in_row = (uint64_t) (t->cols / 32) * 34, out_row = (uint64_t) (t->cols / 32) * 18;
        for (int64_t r = 0; r < t->rows; ++r) {
            q8t->to_float(blob.data() + t->off + (uint64_t) r * in_row, row.data(), t->cols);
            ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), host.data() + off + (uint64_t) r * out_row, 0, 1, t->cols,
                                nullptr);
        }
        q4_off_.push_back({t->name, off});
    }
    if (cudaMalloc((void**) &dense4_, (size_t) total) != cudaSuccess) { err = "mtp: the Q4_0 projections do not fit"; return false; }
    cudaMemcpy(dense4_, host.data(), (size_t) total, cudaMemcpyHostToDevice);
    vram_ += total;
    std::fprintf(stderr, "strata mtp: --mtp-q4: %zu projections as Q4_0 (%.1f MiB)\n", plan.size(), (double) total / 1048576.0);
    return true;
#else
    (void) blob;
    err = "mtp: --mtp-q4 needs a build with native experts (ggml)";
    return false;
#endif
}

// --mtp-q4: the draft head's subset rows as Q4_0 (dequantized from the main head's format)
bool MtpDrafter::make_q4_head(std::string& err) {
#ifdef STRATA_NATIVE_EXPERTS
    const int64_t N = g_->n_embd;
    const auto* tt = ggml_get_type_traits((ggml_type) head_->type());
    if (tt == nullptr || tt->to_float == nullptr || N % 32 != 0) { err = "mtp: --mtp-q4 cannot read the head's format"; return false; }
    const size_t in_row = head_->row_bytes(), out_row = (size_t) (N / 32) * 18;
    std::vector<uint8_t> src((size_t) n_dvocab_ * in_row), dst((size_t) n_dvocab_ * out_row);
    if (cudaMemcpy(src.data(), dhead_, src.size(), cudaMemcpyDeviceToHost) != cudaSuccess) { err = "mtp: reading the draft head"; return false; }
    std::vector<float> row((size_t) N);
    for (int64_t r = 0; r < n_dvocab_; ++r) {
        tt->to_float(src.data() + (size_t) r * in_row, row.data(), N);
        ggml_quantize_chunk(GGML_TYPE_Q4_0, row.data(), dst.data() + (size_t) r * out_row, 0, 1, N, nullptr);
    }
    // the Q4_0 rows replace the subset in place (they are smaller)
    if (cudaMemcpy(dhead_, dst.data(), dst.size(), cudaMemcpyHostToDevice) != cudaSuccess) { err = "mtp: writing the draft head"; return false; }
    dhead_type_ = GGML_Q4_0;
    std::fprintf(stderr, "strata mtp: --mtp-q4: draft head over %lld tokens as Q4_0 (%.1f MiB read per step, was %.1f)\n",
                 (long long) n_dvocab_, (double) dst.size() / 1048576.0, (double) src.size() / 1048576.0);
    return true;
#else
    err = "mtp: --mtp-q4 needs a build with native experts (ggml)";
    return false;
#endif
}

bool MtpDrafter::load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
                      int64_t window, const MtpDrafter* shared) {
    cudaGetDevice(&device_);   // all draft weights/state live on the device selected by the caller
    g_ = &g;
    ss_ = &ss;
    if (ple_ss_ == nullptr) ple_ss_ = &ss;
    max_t_ = max_t;
    rt_dir_ = rt_dir;
    if (!hq_pack_.empty() && !hq_validate(g, shared, err)) return false;
    if (max_t < 1 || max_t > strata::kernels::kVerifyMaxT) { err = "mtp: max_t out of range"; return false; }
    if (shared != nullptr) {
        if (shared->device_ != device_ || shared->g_ != &g || shared->dense_ == nullptr ||
            shared->experts_ == nullptr || shared->rt_dir_ != rt_dir) {
            err = "mtp: incompatible shared weights";
            return false;
        }
        dense_ = shared->dense_;
        experts_ = shared->experts_;
        tensors_ = shared->tensors_;
        owns_weights_ = false;
        owns_draft_head_ = false;
    }
    // Loader fix (0.1.15+loaderfix.2): the two reads below are the whole “drafter files” cost; reporting
    // them apart from the rest of the stage is what makes the next regression visible.
    const auto t_files = std::chrono::steady_clock::now();
    // ---- the index and the dense weights
    if (shared == nullptr) {
        std::ifstream idx(rt_dir + "/dense.txt");
        if (!idx) { err = "mtp: cannot open " + rt_dir + "/dense.txt (run tools/mtp_rt.py)"; return false; }
        std::string line;
        while (std::getline(idx, line)) {
            if (line.empty()) continue;
            std::istringstream is(line);
            Tensor t;
            is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;
            if (!is) { err = "mtp: malformed dense.txt line: " + line; return false; }
            tensors_.push_back(t);
        }
        std::vector<uint8_t> blob;
        if (!read_file(rt_dir + "/dense.bin", blob)) { err = "mtp: cannot read dense.bin"; return false; }
        const cudaError_t alloc = cudaMalloc((void**) &dense_, blob.size());
        if (alloc != cudaSuccess) {
            size_t free_bytes = 0, total_bytes = 0;
            const cudaError_t info = cudaMemGetInfo(&free_bytes, &total_bytes);
            err = "mtp: dense weights allocation failed (" + std::string(cudaGetErrorString(alloc)) +
                  "), requested " + std::to_string(blob.size() >> 20) + " MiB, CUDA0 free " +
                  (info == cudaSuccess ? std::to_string(free_bytes >> 20) + " MiB" : "unknown");
            return false;
        }
        cudaMemcpy(dense_, blob.data(), blob.size(), cudaMemcpyHostToDevice);
        vram_ += blob.size();
        if (q4_ && !make_q4_dense(blob, err)) return false;
    }
    // ---- the 512 routed experts, one blob each
    if (shared == nullptr) {
        const uint64_t bytes = (uint64_t) g.n_expert * strata::kernels::cpu::BLOB;
        // Loader fix (0.1.15+loaderfix.2): each 64 MiB read below reached the disk as ~16k 4095-byte reads under
        // MSVC's `basic_filebuf::xsgetn`, which is what made 675 MiB of drafter experts take minutes.
        FILE* f = std::fopen((rt_dir + "/experts.bin").c_str(), "rb");
        if (f == nullptr) { err = "mtp: cannot open experts.bin"; return false; }
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        const uint64_t allocated = hq_pack_.empty() ? bytes : 1415577600ull;
        if (cudaMalloc((void**) &experts_, allocated) != cudaSuccess) { err = "mtp: the 512 experts do not fit in VRAM"; return false; }
#if !defined(_WIN32)
        strata::platform::advise_willneed(fileno(f), 0, bytes);
#endif
        std::vector<uint8_t> chunk(64u << 20);
        for (uint64_t off = 0; off < bytes;) {
            const uint64_t n = std::min<uint64_t>(chunk.size(), bytes - off);
            if (std::fread(chunk.data(), 1, (size_t) n, f) != (size_t) n) { err = "mtp: experts.bin is truncated"; return false; }
            cudaMemcpy(experts_ + off, chunk.data(), n, cudaMemcpyHostToDevice);
            off += n;
        }
        vram_ += allocated;
    }
    const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                              "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.shared_expert.gate_proj.weight",
                              "mlp.shared_expert.up_proj.weight", "mlp.shared_expert.down_proj.weight"};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }

    // ---- the layer's own K/V (dense attention: no indexer state is read)
    const strata::kernels::QsaShapes s = shapes_of(g);
    const int64_t max_cells = ss.qsa_states[ss.qsa_primary()].max_cells;
    // KV streaming: the drafter only reads its last `window` cells, so with streaming on its K/V is a ring of the
    // window (plus the cells a round writes ahead of its queries) over a host copy, refilled on a resume. The host copy
    // is pinned after the expert arena has pinned what it could: if it does not fit, the K/V stays whole in VRAM.
    int64_t ring = (window > 0 && window < max_cells) ? window + 4 * (int64_t) max_t + 64 : 0;
    // K8V4 never applies to the drafter: its own attention paths (below, and verify.cpp) handle whole formats
    // only, whatever ring shape it takes (0, a window, or the -1 fully-resident fallback).
    const bool kv_hybrid_was = qsa_kv_hybrid();
    const bool kv_int8_was = qsa_kv_int8();
    const bool kv_q4_was = qsa_kv_q4();
    qsa_set_kv_hybrid(false);
    if (kv_hybrid_was) qsa_set_kv_int8(true);   // the drafter under --kv k8v4: plain INT8
    // STRATA_MTP_KV=q4|int8|f16 (opt-in): the draft layer's own K/V format; unset, it follows the main layers' as
    // before.  Under --kv q4_0 the drafter's rotated 4-bit K/V costs draft acceptance, and only the drafts change, never
    // the verified output: every drafter path reads st_'s own format (the K/V append and attention below, the prompt
    // path's draft-layer batch, the conversation snapshot's format check).  It is one layer of K/V on the drafter's card,
    // taken before `--expert-cache auto` sizes the slots: fully resident at --max-context 163840, q4_0 90 MiB, int8 165,
    // f16 320.
    std::string mtp_kv;   // "" = the main layers' format
    if (const char* kv = std::getenv("STRATA_MTP_KV"); kv != nullptr && *kv != '\0') {
        const std::string want(kv);
        if (want == "q4" || want == "q4_0") mtp_kv = "q4";
        else if (want == "int8" || want == "q8") mtp_kv = "int8";
        else if (want == "f16" || want == "fp16") mtp_kv = "f16";
        else std::fprintf(stderr, "strata mtp: STRATA_MTP_KV=%s is not q4, int8 or f16; the draft layer keeps the main "
                                  "layers' K/V format\n", kv);
    }
    if (!mtp_kv.empty()) {
        qsa_set_kv_q4(mtp_kv == "q4");
        qsa_set_kv_int8(mtp_kv == "int8");
    }
    uint64_t sb = qsa_state_bytes(g, max_cells, false, ring);
    if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
    if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) {
        if (st_.kv_mode == 0) { err = "mtp: state init failed"; return false; }
        std::fprintf(stderr, "strata mtp: no pinned RAM left for the draft layer's K/V copy; keeping it in VRAM\n");
        cudaGetLastError();
        cudaFree(state_arena_);
        st_ = QsaState{};
        ring = -1;   // fully resident
        sb = qsa_state_bytes(g, max_cells, false, ring);
        if (cudaMalloc(&state_arena_, sb) != cudaSuccess) { err = "mtp: the K/V state does not fit"; return false; }
        if (qsa_state_init(g, max_cells, state_arena_, st_, &ss.qsa_states[ss.qsa_primary()], ring) == 0) { err = "mtp: state init failed"; return false; }
    }
    qsa_set_kv_int8(kv_int8_was);
    qsa_set_kv_q4(kv_q4_was);
    qsa_set_kv_hybrid(kv_hybrid_was);
    if (!mtp_kv.empty())
        std::fprintf(stderr, "strata mtp: draft layer K/V format %s (STRATA_MTP_KV=%s; main layers %s), rotated %d, %s, "
                             "state %.1f MiB of VRAM\n", st_.kv_q4 ? "q4_0" : (st_.kv_int8 ? "int8" : "f16"), mtp_kv.c_str(),
                     kv_hybrid_was ? "k8v4" : (kv_q4_was ? "q4_0" : (kv_int8_was ? "int8" : "f16")), st_.kv_rot ? 1 : 0,
                     st_.kv_mode == 2 ? "a window ring over pinned RAM"
                                      : (st_.kv_mode == 1 ? "streamed over pinned RAM (--kv-resident)" : "fully resident"),
                     (double) sb / 1048576.0);
    qsa_state_zero(st_, g, nullptr);
    cudaDeviceSynchronize();
    vram_ += sb;

    // ---- buffers
    window_ = (window > 0 && window < max_cells) ? window : 0;
    cap_ = (((window_ > 0 ? window_ : max_cells) + 63) / 64) * 64;
    attn_scratch_floats_ = (int64_t) strata::kernels::qsa_decode_attn_scratch_floats(cap_, s);
    const uint64_t T = (uint64_t) max_t, N = (uint64_t) g.n_embd, HC = (uint64_t) g.hc, K = (uint64_t) ss.k;
    const uint64_t NH = (uint64_t) g.n_head, HD = (uint64_t) g.head_dim, NKV = (uint64_t) g.n_head_kv;
    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_) &&
              (!force_on_ || mapped(64, (void**) &h_force_, (void**) &m_force_));
    if (!ok) { err = "mtp: mapped staging failed"; return false; }
    if (first_top2_config_ && !mapped(sizeof(strata::kernels::MtpFirstTop2),
            (void**) &first_top2_host_, (void**) &first_top2_mapped_)) {
        err = "mtp: first-head probe staging failed"; return false;
    }
    if (h_force_ != nullptr) for (int j = 0; j < 16; ++j) h_force_[j] = -1;
    auto carve = [&](Bump& b) {
        tok_ = b.take<int32_t>(T); step_ = b.take<int32_t>(R2 * 4); pos_ = b.take<int32_t>(R2 * NH); row_ = b.take<int32_t>(4);
        ident_ = b.take<int32_t>(T * (uint64_t) cap_);
        Rin_ = b.take<float>(T * HC * N); R_ = b.take<float>(T * HC * N);
        emb_ = b.take<float>(T * N); en_ = b.take<float>(T * N); e2_ = b.take<float>(T * N);
        hn_ = b.take<float>(T * HC * N); h2_ = b.take<float>(T * HC * N);
        mixed_ = b.take<float>(T * N); inj_ = b.take<float>(T * HC); inj2_ = b.take<float>(T * HC);
        lo_ = b.take<float>(T * (uint64_t) g.hc_lr); rs_ = b.take<float>(T * HC); bo_ = b.take<float>(T * N);
        xn_ = b.take<float>(T * HC * N);
        xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) (NH * HD), 8));
        qfull_ = b.take<float>(T * NH * 2 * HD); qcur_ = b.take<float>(T * NH * HD);
        kcur_ = b.take<float>(T * NKV * HD); vcur_ = b.take<float>(T * NKV * HD);
        attn_ = b.take<float>(T * NH * HD); attn32_ = b.take<float>(T * NH * HD);
        attn_scratch_ = b.take<float>((uint64_t) attn_scratch_floats_);   // the full layer runs one row at a time
        logits_ = b.take<float>(T * (uint64_t) g.n_expert); w_ = b.take<float>(T * K); ids_ = b.take<int32_t>(T * K);
        shared_ = b.take<float>(T * N); parts_ = b.take<float>(T * K * N); y_ = b.take<float>(T * N);
        sample_ = b.take<float>(T * N);
        hit_slot_ = b.take<int32_t>(T * K); hit_dst_ = b.take<int32_t>(T * K); hit_count_ = b.take<int32_t>(4);
        grp_ptr_ = b.take<unsigned long long>(T * K); grp_start_ = b.take<int32_t>(T * K + 1);
        grp_counts_ = b.take<int32_t>(4);
        hit_xq_ = b.take<uint8_t>(T * (N / 32) * 34); hit_xs_ = b.take<float>(T * (N / 32));
        hit_scratch_ = b.take<uint8_t>(strata::kernels::moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
#ifdef STRATA_NATIVE_EXPERTS
        if (!hq_pack_.empty()) {
            hq_xq_ = b.take<uint8_t>(strata::kernels::native_q8_1_bytes((int) N, (int) T));
            hq_scratch_ = b.take<uint8_t>(strata::kernels::native_expert_scratch_bytes((int64_t) (T * K), g.n_ff));
        }
#endif
        sh_scratch_ = (float*) b.take<uint8_t>(strata::kernels::shared_expert_scratch_bytes(g.n_ff));
        x_bf16_ = b.take<uint16_t>(N);
        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        top_scratch_ = b.take<uint8_t>(strata::kernels::row_top_prob_scratch_bytes((int) T));
        dummy_inj_ = b.take<float>(HC);
    };
    Bump count;
    carve(count);
    if (cudaMalloc(&arena_, count.used) != cudaSuccess) { err = "mtp: buffers do not fit"; return false; }
    cudaMemset(arena_, 0, count.used);
    Bump real;
    real.base = (uint8_t*) arena_;
    carve(real);
    vram_ += count.used;
    {
        std::vector<int32_t> id((size_t) (T * (uint64_t) cap_));
        for (uint64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < cap_; ++i) id[(size_t) (t * (uint64_t) cap_ + (uint64_t) i)] = (int32_t) i;
        cudaMemcpy(ident_, id.data(), id.size() * 4, cudaMemcpyHostToDevice);
    }
    // --pipeline-windows 2 (the forcing graphs): the chain shares its card with stage 1's windows and the next launch
    // on stage 0 waits for it, so its stream gets the highest priority there (STRATA_MTP_PRIORITY=0: the default)
    bool prio = false;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)   // (HIP: the default priority)
    static const bool hi_prio = [] { const char* v = std::getenv("STRATA_MTP_PRIORITY"); return v == nullptr || std::atoi(v) != 0; }();
    int prio_lo = 0, prio_hi = 0;
    if (force_on_ && hi_prio && cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi) == cudaSuccess)
        prio = cudaStreamCreateWithPriority(&cs_, cudaStreamNonBlocking, prio_hi) == cudaSuccess;
    cudaGetLastError();
#endif
    if (!prio && cudaStreamCreateWithFlags(&cs_, cudaStreamNonBlocking) != cudaSuccess) { err = "mtp: stream"; return false; }
#if !defined(STRATA_USE_HIP)
    // HIP keeps the shared expert on cs_; only CUDA needs this branch stream.
    if (cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&sh_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&sh_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: streams";
        return false;
    }
#endif
    if (observer_config_ && !observer_allocate(err)) return false;
    if (selective_config_ && !selective_allocate(err)) return false;
    if (anchor_config_ && !anchor_allocate(err)) return false;
    if (feature_config_ && !feature_allocate(err)) return false;
    const double files_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_files).count();
    if (shared != nullptr) {
        std::fprintf(stderr, "strata mtp: shared draft weights, %.0f MiB of private state and buffers\n",
                     (double) vram_ / 1048576.0);
    } else {
        std::fprintf(stderr, "strata mtp: draft layer loaded, %.0f MiB of VRAM (experts %.0f, dense %.0f), files read in %.2f s (%.0f MiB/s)\n",
                     (double) vram_ / 1048576.0, (double) g.n_expert * strata::kernels::cpu::BLOB / 1048576.0,
                     (double) tensors_.back().off / 1048576.0, files_s,
                     files_s > 0 ? ((double) g.n_expert * strata::kernels::cpu::BLOB + (double) tensors_.back().off) /
                                       1048576.0 / files_s : 0.0);
    }
    return true;
}

namespace {
// STRATA_MTP_FULL_HEAD=1 (a diagnostic): draft over the whole vocabulary (the main head itself) instead of
// rt/draft_vocab.bin's subset - the reference head, to measure what the subset costs in acceptance
bool full_head_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_FULL_HEAD"); return v && v[0] == '1'; }();
    return on;
}
}  // namespace

bool MtpDrafter::top2_env() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_TOP2"); return v && v[0] == '1'; }();
    return on;
}

// STRATA_MTP_TOP2=1 (a diagnostic for tree drafts): draft j's runner-up under the draft layer, from the head logits
// the step just wrote (row 0), on the host.  Slow; only for measuring what a second branch would have caught.
void MtpDrafter::record_top2(int j) {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    if ((int64_t) lg_host_.size() < nv) lg_host_.resize((size_t) nv);
    if ((int) top2_.size() < max_t_) top2_.assign((size_t) max_t_, -1);
    if (j < 0 || j >= max_t_) return;
    if (cudaMemcpy(lg_host_.data(), head_logits_, (size_t) nv * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        top2_[(size_t) j] = -1;
        return;
    }
    int64_t b1 = -1, b2 = -1;
    for (int64_t i = 0; i < nv; ++i) {
        const float v = lg_host_[(size_t) i];
        if (b1 < 0 || v > lg_host_[(size_t) b1]) { b2 = b1; b1 = i; }
        else if (b2 < 0 || v > lg_host_[(size_t) b2]) b2 = i;
    }
    top2_[(size_t) j] = b2 < 0 ? -1 : dhead_ != nullptr ? dvocab_host_[(size_t) b2] : (int32_t) b2;
}

uint64_t MtpDrafter::bind_bytes(uint64_t head_row_bytes, int64_t n_vocab, int source_device) const {
    uint64_t bytes = head_logits_ ? 0 : (uint64_t) max_t_ * (uint64_t) n_vocab * sizeof(float);
    if (source_device >= 0 && source_device != device_ && g_ != nullptr) {
        if (local_window_R_ == nullptr)
            bytes += (uint64_t) max_t_ * (uint64_t) g_->hc * (uint64_t) g_->n_embd * sizeof(float);
        if (private_gr_arena_ == nullptr)
            bytes += strata::kernels::gr_workspace_bytes({g_->n_embd, g_->hc, g_->hc_lr});
    }
    if (dhead_ == nullptr && owns_draft_head_ && !full_head_env()) {
        if (FILE* f = std::fopen(vocab_file().c_str(), "rb")) {
            std::fseek(f, 0, SEEK_END);
            const long size = std::ftell(f);
            std::fclose(f);
            if (size >= 4 && size % 4 == 0) bytes += (uint64_t) (size / 4) * head_row_bytes + (uint64_t) size;
        }
    }
    // coupled draft sampling (STRATA_SPEC_COUPLED=1 only): an upper bound - the id -> subset map, the penalty ring,
    // the split scratch (64 lists of 64 entries at most) and the parameters
    if (coupled_draft_env() && cparams_ == nullptr)
        bytes += (uint64_t) n_vocab * 4 + (uint64_t) (kCoupledHistCap + max_t_) * 4 + 64ull * 64ull * 8ull + 4096;
    return bytes;
}

bool MtpDrafter::setup_coupled(std::string& err) {
    const int64_t nv = dhead_ != nullptr ? n_dvocab_ : n_vocab_;
    const size_t scratch = strata::kernels::coupled_draft_scratch_bytes((int) nv);
    if (scratch == 0) {
        std::fprintf(stderr, "strata mtp: STRATA_SPEC_COUPLED: %lld draft logits are too wide for the coupled sampler; "
                             "argmax drafts\n", (long long) nv);
        return true;
    }
    const size_t ring = (size_t) (kCoupledHistCap + max_t_) * sizeof(int32_t);
    if (cudaMalloc((void**) &cparams_, sizeof(strata::kernels::SamplerParams)) != cudaSuccess ||
        cudaMalloc((void**) &cring_, ring) != cudaSuccess || cudaMalloc(&cscratch_, scratch) != cudaSuccess ||
        !mapped(sizeof(strata::kernels::SamplerParams), (void**) &h_cparams_, (void**) &m_cparams_) ||
        !mapped((size_t) kCoupledHistCap * sizeof(int32_t), (void**) &h_chist_, (void**) &m_chist_)) {
        err = "mtp: the coupled draft sampler's buffers do not fit";
        return false;
    }
    if (spec_prob_env() &&
        !mapped((size_t) max_t_ * (size_t) kSpecQStride * sizeof(int32_t), (void**) &h_q_, (void**) &m_q_)) {
        err = "mtp: the draft distributions' buffer does not fit";
        return false;
    }
    cudaMemset(cring_, 0xff, ring);   // -1: no token
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
    vram_ += sizeof(strata::kernels::SamplerParams) + ring + scratch;
    if (dhead_ != nullptr) {   // token id -> subset index (-1: not in the draft head), for the penalties
        std::vector<int32_t> sub((size_t) n_dvocab_), inv((size_t) n_vocab_, -1);
        cudaMemcpy(sub.data(), dvocab_, sub.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < sub.size(); ++i)
            if (sub[i] >= 0 && sub[i] < n_vocab_ && inv[(size_t) sub[i]] < 0) inv[(size_t) sub[i]] = (int32_t) i;
        if (cudaMalloc((void**) &dinv_, inv.size() * sizeof(int32_t)) != cudaSuccess) {
            err = "mtp: the coupled draft sampler's token map does not fit";
            return false;
        }
        cudaMemcpy(dinv_, inv.data(), inv.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        vram_ += inv.size() * sizeof(int32_t);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = std::string("mtp: coupled setup: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    coupled_ok_ = true;
    if (spec_prob_env())
        std::fprintf(stderr, "strata mtp: probabilistic draft acceptance on (STRATA_SPEC_PROB): sampled requests draft by "
                             "sampling the draft head's distribution over %lld tokens; the verifier accepts with "
                             "min(1, p/q) and resamples the residual\n", (long long) nv);
    else
    std::fprintf(stderr, "strata mtp: coupled draft sampling on (STRATA_SPEC_COUPLED): sampled requests draft with the "
                         "target's chain and Philox draw over %lld tokens\n", (long long) nv);
    return true;
}

void MtpDrafter::set_draft_sampling(const strata::kernels::SamplerParams& sp) {
    coupled_active_ = coupled_ok_ && !sp.greedy && sp.temperature > 0.0f && sp.temperature >= spec_min_temp();
    if (!coupled_active_) return;
    *h_cparams_ = sp;   // read by the next round graph (after the previous one has synced)
    std::fill(h_chist_, h_chist_ + kCoupledHistCap, -1);
}

void MtpDrafter::set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next) {
    if (!coupled_active_) return;
    const int h = coupled_hist_len(h_cparams_->penalty_last_n, kCoupledHistCap);
    coupled_hist_base(tail, n_tail, next, h, h_chist_ + (kCoupledHistCap - h));
}

namespace {
// #474: what to change when the draft head's token subset does not fit the VRAM left - a smaller subset that setup
// ships (data/draft_vocab_*.bin; their token counts below), and how to pick it.  The start used to stop at "the draft
// head does not fit" with no hint.  Text only, after the failure: nothing changes for a start that fits.
void draft_head_hint(int64_t n_tokens, int64_t row_bytes) {
    size_t free_b = 0, total_b = 0;
    const bool have_free = cudaMemGetInfo(&free_b, &total_b) == cudaSuccess;
    auto mib = [&](int64_t n) { return (double) (n * row_bytes) / 1048576.0; };
    std::string free_s;
    if (have_free) {
        char b[64];
        std::snprintf(b, sizeof(b), " and %.0f MiB is free", (double) free_b / 1048576.0);
        free_s = b;
    }
    std::fprintf(stderr, "strata mtp: the draft head over %lld tokens needs %.0f MiB of VRAM%s.\n",
                 (long long) n_tokens, mib(n_tokens), free_s.c_str());
    struct Subset { const char* name; int64_t tokens; const char* what; };
    static const Subset smaller[] = {{"cyrillic", 58963, "English, code and the Cyrillic script"},
                                     {"en", 40525, "English and code"}};
    std::string opts;
    for (const Subset& s : smaller)
        if (s.tokens < n_tokens) {
            char b[160];
            std::snprintf(b, sizeof(b), "%s--draft-vocab %s (%s, ~%.0f MiB)", opts.empty() ? "" : " or ", s.name,
                          s.what, mib(s.tokens));
            opts += b;
        }
    if (!opts.empty())
        std::fprintf(stderr, "strata mtp: hint: a smaller draft vocabulary needs less VRAM: %s. Start once with it - "
                             "START-HERE.bat --draft-vocab en (Windows) or ./setup.sh --draft-vocab en - and the model "
                             "keeps it (\"draft_vocab\" in its strata-*.json config); or a smaller --context in "
                             "setup.\n",
                     opts.c_str());
    else
        std::fprintf(stderr, "strata mtp: hint: this is already the smallest shipped draft vocabulary: a smaller "
                             "--context (or closing what else uses the GPU) leaves it room.\n");
}
}  // namespace

bool MtpDrafter::bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err,
                      const MtpDrafter* shared, int source_device) {
    const OnDevice on_device(device_);
    const int src_device = source_device < 0 ? device_ : source_device;
    if (source_device_ >= 0 && source_device_ != src_device) {
        err = "mtp: the source device cannot change after bind";
        return false;
    }
    source_device_ = src_device;
    if (cross_device()) {
#if defined(STRATA_USE_HIP)
        err = "mtp: cross-device drafting currently requires CUDA";
        return false;
#endif
        if (shared != nullptr || full_head_env()) {
            err = "mtp: cross-device drafting requires a private draft head subset";
            return false;
        }
    }
    wt_ = &wt;
    head_ = head;
    bound_source_R_ = window_R;
    window_R_ = window_R;
    const WeightRef* wo = wt.find("output.weight");
    if (!wo) { err = "mtp: output.weight is missing"; return false; }
    n_vocab_ = wo->ne1;
    if (head == nullptr || !head->loaded()) { err = "mtp: the draft layer needs the native head (--native)"; return false; }
    if (cross_device()) {
        const size_t residual_bytes = (size_t) max_t_ * (size_t) g_->hc * (size_t) g_->n_embd * sizeof(float);
        if (local_window_R_ == nullptr) {
            if (cudaMalloc((void**) &local_window_R_, residual_bytes) != cudaSuccess) {
                err = "mtp: cross-device residual buffer does not fit";
                return false;
            }
            vram_ += residual_bytes;
        }
        window_R_ = local_window_R_;
        const strata::kernels::GrShapes gs{g_->n_embd, g_->hc, g_->hc_lr};
        if (private_gr_arena_ == nullptr) {
            const size_t bytes = strata::kernels::gr_workspace_bytes(gs);
            if (cudaMalloc(&private_gr_arena_, bytes) != cudaSuccess) {
                err = "mtp: private residual-mixer scratch does not fit";
                return false;
            }
            strata::kernels::gr_workspace_init(gs, private_gr_arena_, private_gr_);
            vram_ += bytes;
        }
        for (auto& slot : source_slots_) {
            if (slot.host == nullptr &&
                cudaHostAlloc((void**) &slot.host, residual_bytes, cudaHostAllocPortable) != cudaSuccess) {
                err = "mtp: pinned cross-device staging allocation failed";
                return false;
            }
            if (slot.consumed == nullptr &&
                cudaEventCreateWithFlags(&slot.consumed, cudaEventDisableTiming) != cudaSuccess) {
                err = "mtp: cross-device staging event creation failed";
                return false;
            }
        }
        if (source_cs_ == nullptr) {
            const OnDevice on_source(source_device_);
            if (cudaStreamCreateWithFlags(&source_cs_, cudaStreamNonBlocking) != cudaSuccess) {
                err = "mtp: residual source stream creation failed";
                return false;
            }
        }
    }
    if (head_logits_ == nullptr &&
        cudaMalloc((void**) &head_logits_, (size_t) max_t_ * (size_t) n_vocab_ * sizeof(float)) != cudaSuccess) {
        err = "mtp: the draft logits do not fit";
        return false;
    }
    if (shared != nullptr) {
        if (shared->head_ != head || shared->device_ != device_ ||
            shared->n_vocab_ != n_vocab_) {
            err = "mtp: incompatible shared draft head";
            return false;
        }
        dhead_ = shared->dhead_;
        dhead_type_ = shared->dhead_type_;   // the subset's type (the main head's, or Q4_0): -1 failed every --batch-mtp step
        dvocab_ = shared->dvocab_;
        n_dvocab_ = shared->n_dvocab_;
        owns_draft_head_ = false;
    }
    // the draft head's token subset, when tools/draft_vocab.py wrote one
    if (dhead_ == nullptr && shared == nullptr && !full_head_env()) {
        std::vector<uint8_t> raw;
        if (read_file(vocab_file(), raw) && raw.size() >= 4 && raw.size() % 4 == 0) {
            n_dvocab_ = (int64_t) (raw.size() / 4);
            const int64_t row_bytes = (int64_t) head->row_bytes();   // a vocabulary row of the native head
            if (cudaMalloc((void**) &dvocab_, raw.size()) != cudaSuccess ||
                cudaMalloc((void**) &dhead_, (size_t) (n_dvocab_ * row_bytes)) != cudaSuccess) {
                err = "mtp: the draft head does not fit";
                draft_head_hint(n_dvocab_, row_bytes);
                return false;
            }
            cudaMemcpy(dvocab_, raw.data(), raw.size(), cudaMemcpyHostToDevice);
            dvocab_host_.resize((size_t) n_dvocab_);
            std::memcpy(dvocab_host_.data(), raw.data(), raw.size());
            if (cross_device()) {
                // Read unchanged native blocks once at startup. The source card needs no temporary allocation
                // after its expert cache has been sized, and this path needs neither peer access nor remote loads.
                for (int32_t id : dvocab_host_) {
                    if (id < 0 || (int64_t) id >= n_vocab_) {
                        err = "mtp: a draft vocabulary token is outside the native head";
                        return false;
                    }
                }
                if (row_bytes <= 0 || head->weight_bytes() != (uint64_t) n_vocab_ * (uint64_t) row_bytes) {
                    err = "mtp: inconsistent native head dimensions for cross-device copy";
                    return false;
                }
                std::vector<uint8_t> source_head((size_t) head->weight_bytes());
                {
                    const OnDevice on_source(source_device_);
                    if (cudaMemcpy(source_head.data(), head->weights(), source_head.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
                        err = "mtp: reading the source device's native head failed";
                        return false;
                    }
                }
                std::vector<uint8_t> subset((size_t) n_dvocab_ * (size_t) row_bytes);
                for (int64_t i = 0; i < n_dvocab_; ++i)
                    std::memcpy(subset.data() + (size_t) i * (size_t) row_bytes,
                                source_head.data() + (size_t) dvocab_host_[(size_t) i] * (size_t) row_bytes,
                                (size_t) row_bytes);
                if (cudaMemcpy(dhead_, subset.data(), subset.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
                    err = "mtp: uploading the unchanged draft head subset failed";
                    return false;
                }
            } else {
                strata::kernels::gather_rows((const uint8_t*) head->weights(), row_bytes, dvocab_, n_dvocab_, dhead_, nullptr);
                cudaDeviceSynchronize();
            }
            vram_ += (uint64_t) (n_dvocab_ * row_bytes) + raw.size();
            std::fprintf(stderr, "strata mtp: draft head over %lld tokens (%.1f MiB)\n", (long long) n_dvocab_,
                         (double) (n_dvocab_ * row_bytes) / 1048576.0);
            dhead_type_ = head->type();
            if (q4_head_ && !make_q4_head(err)) return false;
            if (dhead_type_ == 14 && strata::kernels::native_q6_k_packed_enabled())   // STRATA_Q6_PACKED=1
                strata::kernels::native_q6_k_pack(dhead_, (int) (row_bytes / 210) * 256, (int) n_dvocab_,
                                                  "MTP draft head");
        }
    }
    if (cross_device() && dhead_ == nullptr) {
        err = "mtp: cross-device drafting needs a valid draft_vocab.bin or --mtp-draft-vocab subset";
        return false;
    }
    if (coupled_draft_env() && cparams_ == nullptr && !setup_coupled(err)) return false;
    return true;
}

// The layer for T rows.  full = false stops after the K/V append (the prompt only needs the cache).
bool MtpDrafter::record_forward(int T, int step_row0, cudaStream_t cs, std::string& err) {
    // step_row0 >= 0: the full layer on step rows [step_row0, +T); step_row0 < 0: K/V only on rows [-1 - step_row0, +T)
    const bool full = step_row0 >= 0;
    if (full && T != 1) { err = "mtp: the full layer runs one row at a time (its attention scratch is sized for one)"; return false; }
    if (!record_front(T, full ? step_row0 : -1 - step_row0, cs, err)) return false;
    return !full || record_rest(step_row0, cs, err);
}

bool MtpDrafter::record_front(int T, int row0, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    const int64_t N = g.n_embd, HC = g.hc, NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const QsaShapes s = shapes_of(g);
    const int32_t* step = step_ + row0 * 4;
    const int32_t* pos = pos_ + row0 * NH;
    // --mtp-q4: each projection's format and weights for this pass (Q8_0 unless the Q4_0 copies exist)
    int wt_fc_embedding, wt_fc_hidden, wt_k_proj, wt_v_proj;
    const void* wp_fc_embedding = wq("fc_embedding.weight", wt_fc_embedding);
    const void* wp_fc_hidden = wq("fc_hidden.weight", wt_fc_hidden);
    const void* wp_k_proj = wq("self_attn.k_proj.weight", wt_k_proj);
    const void* wp_v_proj = wq("self_attn.v_proj.weight", wt_v_proj);
    try {
        // ---- the two input branches
        const WeightRef* we = wt_->find("token_embd.weight");
        if (!we) { err = "mtp: token_embd.weight is missing"; return false; }
        if (const NativeEmbed* ne = native_embed()) {   // plan v0.3 P6: the GGUF-form table
            ne->gather_dev(tok_, T, emb_, cs);
        } else {
            const auto* codes = (const uint8_t*) we->data;
            const auto* scales = (const float*) (codes + we->codes_bytes);
            const auto* offsets = we->has_offset ? (const float*) (codes + we->codes_bytes + we->scales_bytes) : nullptr;
            embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                 (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
        }
        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
        native_mmvq(wt_fc_embedding, wp_fc_embedding, xq_, e2_, (int) N, (int) N, T, cs);
        if (hnorm_stream_)   // --mtp-hnorm stream: one RMS per stream, each scaled by its slice of the weight
            native_qsa_rms_norm_grouped(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) N, (int) HC, (int) (T * HC), EPS, cs);
        else
            native_qsa_rms_norm_weighted(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) (HC * N), T, EPS, cs);
        for (int c0 = 0; c0 < T * HC; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T * HC - c0);
            native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            native_mmvq(wt_fc_hidden, wp_fc_hidden, xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
        }
        add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
        // ---- the attention hyper-connection
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("attn_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("attn_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("attn_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("attn_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- attention: K/V into the layer's own cache
        native_quantize_q8_1(mixed_, xq_, (int) N, T, cs);
        native_mmvq(wt_k_proj, wp_k_proj, xq_, kcur_, (int) N, (int) (NKV * HD), T, cs);
        native_mmvq(wt_v_proj, wp_v_proj, xq_, vcur_, (int) N, (int) (NKV * HD), T, cs);
        // #783 PR-d (stuchapin909): the T tokens' K norm, K/V rotation and K/V append each run once over all T rows
        // (every row is independent; the rope keeps its per-token positions). STRATA_NO_BATCH_KV_STEP=1 appends per token.
        static const bool no_batch_kv = [] {
            const char* v = std::getenv("STRATA_NO_BATCH_KV_STEP");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        // #783 PR-f (stuchapin909): the per-head RMSNorm and the rope fused (bit-identical to the pair, rope_parity
        // check 6; STRATA_NO_NORM_ROPE=1 keeps the two; off on HIP until its parity check passes). K rows of one token
        // are consecutive in `pos`, tokens are NH apart, so a one-token window is the fusable case.
        const bool fuse_nr = native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);
        if (T == 1 && fuse_nr) {
            native_qsa_rms_norm_rope(kcur_, (int) HD, f32("self_attn.k_norm.weight"), kcur_, (int) NKV, (int) HD,
                                     (int) s.n_rot, EPS, rope_scaling(), pos, cs);
        } else {
            native_qsa_rms_norm_weighted(kcur_, f32("self_attn.k_norm.weight"), kcur_, (int) HD, (int) (T * NKV), EPS, cs);
            for (int t = 0; t < T; ++t) {
                float* kc = kcur_ + t * NKV * HD;
                if (native_rope_enabled()) native_rope_apply(kc, kc, (int) NKV, (int) HD, (int) s.n_rot, rope_scaling(), pos + t * NH, cs);
                else rope_neox_apply(kc, kc, (int) NKV, (int) HD, (int) s.n_rot, st_.cos_tab, st_.sin_tab, pos + t * NH, cs);
            }
        }
        if (st_.kv_rot) {   // rotated K and V (kv_q4.hpp): Q4_0, and INT8 with STRATA_KV_ROT=1
            fwht256_inplace_cuda(kcur_, (int64_t) T * NKV, cs);
            fwht256_inplace_cuda(vcur_, (int64_t) T * NKV, cs);
        }
        // stored in the state's own format (#293 appended rotated INT8 K/V as Q4_0, into pools INT8 never has)
        if (!no_batch_kv && st_.kv_q4)
            kv_append_q4_steps(st_.k_q4, st_.v_q4, st_.page_table, step, 4, T, kcur_, vcur_, s, cs, &st_.host);
        else if (!no_batch_kv && st_.kv_int8)
            kv_append_q8_steps(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step, 4, kcur_, vcur_,
                               (int) (NKV * HD), T, s, cs, &st_.host);
        else
            for (int t = 0; t < T; ++t) {
                if (st_.kv_q4)
                    kv_append_q4_step(st_.k_q4, st_.v_q4, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                      vcur_ + t * NKV * HD, s, cs, &st_.host);
                else if (st_.kv_int8)
                    kv_append_q8_step(st_.k_q, st_.v_q, st_.k_scale, st_.v_scale, st_.page_table, step + t * 4,
                                      kcur_ + t * NKV * HD, vcur_ + t * NKV * HD, s, cs, &st_.host);
                else
                    kv_append_step(st_.k_pool, st_.v_pool, st_.page_table, step + t * 4, kcur_ + t * NKV * HD,
                                   vcur_ + t * NKV * HD, s, cs, &st_.host);
            }
        if (selective_record_)
            kv_append_q8_steps(const_cast<int8_t*>(selective_pools_.k_q), const_cast<int8_t*>(selective_pools_.v_q),
                               const_cast<uint16_t*>(selective_pools_.k_scale), const_cast<uint16_t*>(selective_pools_.v_scale),
                               selective_map_.page_table, step, 4, kcur_, vcur_, (int) (NKV * HD), T, s, cs, nullptr);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

void MtpDrafter::norm_rope(float* data, const float* gamma, int rows, int cols, const int32_t* p, cudaStream_t cs) {
    using namespace strata::kernels;
    const QsaShapes s = shapes_of(*g_);
    native_qsa_rms_norm_weighted(data, gamma, data, cols, rows, EPS, cs);
    if (native_rope_enabled()) native_rope_apply(data, data, rows, cols, (int) s.n_rot, rope_scaling(), p, cs);
    else rope_neox_apply(data, data, rows, cols, (int) s.n_rot, st_.cos_tab, st_.sin_tab, p, cs);
}

bool MtpDrafter::record_rest(int step_row, cudaStream_t cs, std::string& err) {
    using namespace strata::kernels;
    const ModelGeometry& g = *g_;
    SessionState& ss = *ss_;
    const int T = 1;
    const int64_t N = g.n_embd, HC = g.hc, K = ss.k, NH = g.n_head, HD = g.head_dim;
    const QsaShapes s = shapes_of(g);
    const GrShapes gs{g.n_embd, g.hc, g.hc_lr};
    const int32_t* step = step_ + step_row * 4;
    const int32_t* pos = pos_ + step_row * NH;
    // --mtp-q4: the q and o projections' format and weights (Q8_0 unless the Q4_0 copies exist)
    int wt_q_proj, wt_o_proj;
    const void* wp_q_proj = wq("self_attn.q_proj.weight", wt_q_proj);
    const void* wp_o_proj = wq("self_attn.o_proj.weight", wt_o_proj);
    const bool fuse_nr = native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);   // #783 PR-f
    try {
        // ---- dense attention over every cell
        native_mmvq(wt_q_proj, wp_q_proj, xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        // all T tokens' q rows at once (the rows of token t sit at pos[t * NH ..], so row r reads pos[r])
        if (fuse_nr) {
            native_qsa_rms_norm_rope(qfull_, (int) (2 * HD), f32("self_attn.q_norm.weight"), qcur_, (int) (T * NH),
                                     (int) HD, (int) s.n_rot, EPS, rope_scaling(), pos, cs);
        } else {
            if (cudaMemcpy2DAsync(qcur_, (size_t) HD * 4, qfull_, (size_t) HD * 2 * 4, (size_t) HD * 4,
                                  (size_t) (T * NH), cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
                err = "mtp: q split failed";
                return false;
            }
            norm_rope(qcur_, f32("self_attn.q_norm.weight"), (int) (T * NH), (int) HD, pos, cs);
        }
        if (st_.kv_rot) fwht256_inplace_cuda(qcur_, (int64_t) T * NH, cs);
        if (selective_record_) {
            // Fixed mapped rows survive every graph until chain_poll reports the complete chain idle.
            const int j = step_row == 2 * max_t_ - 1 ? 0 : step_row - max_t_ + 1;
            copy_i32_from_mapped(selective_d_ids_, selective_m_ids_ + (size_t) j * kSelectiveCap, kSelectiveCap, cs);
            kv_stream_resolve(selective_map_, selective_pools_, st_.host, kKvInt8, selective_d_ids_, step,
                               1, kSelectiveCap, s, cs);
            qsa_decode_attn_batch(qcur_, selective_pools_, selective_d_ids_, step, kSelectiveCap, s,
                                  attn_scratch_, attn_, T, cs);
        } else {
            const QsaAttnPools pools = qsa_attn_pools(st_);
            if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
            qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
        }
        if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
        native_mmvq(wt_o_proj, wp_o_proj, xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        // ---- the MLP hyper-connection (the attention write folded in)
        {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = bo_ + t * N; fa[t].inj_prev = inj_ + t * HC;
                fa[t].w_norm = f32("mlp_hyper_connection.hc_norm.weight");
                fa[t].w_down = bf16("mlp_hyper_connection.input_mix_weight_down.weight");
                fa[t].w_up = bf16("mlp_hyper_connection.input_mix_weight_up.weight");
                fa[t].w_inject = bf16("mlp_hyper_connection.block_inject_weight.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = inj2_ + t * HC; fa[t].mixed = mixed_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        }
        // ---- MoE: router, the 512 resident experts, the shared expert (on a branch beside them on CUDA), the
        // combine, the write
        NativeSharedWeights nsw;
        nsw.gate_data = wq("mlp.shared_expert.gate_proj.weight", nsw.gate_type);
        nsw.up_data = wq("mlp.shared_expert.up_proj.weight", nsw.up_type);
        nsw.down_data = wq("mlp.shared_expert.down_proj.weight", nsw.down_type);
        nsw.q8_1 = xq_;   // the routed experts read hit_xq_
        const SForm none{};
        const bool need_bf16_x = !shared_expert_native_bf16_enabled();
        static const bool fuse_head_gr = [] {
            const char* v = std::getenv("STRATA_FUSE_HEAD_GR");
            return v != nullptr && std::atoi(v) != 0;
        }();
        static const bool head_mix_multi_on = [] {
#if defined(STRATA_USE_HIP)
            return false;
#else
            const char* v = std::getenv("STRATA_HEAD_MIX_MULTI");
            return v == nullptr || std::atoi(v) != 0;
#endif
        }();
        // the shared expert only reads mixed_ / xq_ and writes shared_: a branch beside the router and the routed
        // experts, joined before the combine (CUDA; HIP keeps one stream: STRATA_MTP_SHARED_BRANCH=0 does too)
        static const bool branch_on = [] {
#if defined(STRATA_USE_HIP)
            return false;
#else
            const char* v = std::getenv("STRATA_MTP_SHARED_BRANCH");
            return v == nullptr || std::atoi(v) != 0;
#endif
        }();
        cudaStream_t sh_cs = cs;
        if (branch_on) {
            if (cudaEventRecord(sh_fork_, cs) != cudaSuccess || cudaStreamWaitEvent(side_, sh_fork_, 0) != cudaSuccess) {
                err = "mtp: the shared expert's branch";
                return false;
            }
            sh_cs = side_;
        }
        auto shared_rows = [&]() {
            for (int t = 0; t < T; ++t) {
                if (need_bf16_x) f32_to_bf16_bulk(mixed_ + t * N, x_bf16_, N, sh_cs);
                shared_expert(nullptr, nullptr, x_bf16_, none, nullptr, nullptr, nullptr, none, nullptr, nullptr, nullptr,
                              none, nullptr, nullptr, nullptr, bf16("mlp.shared_expert_gate.weight"), sh_scratch_,
                              shared_ + t * N, N, g.n_ff, 32, sh_cs, mixed_ + t * N, &nsw);
            }
        };
        if (branch_on) {
            shared_rows();
            if (cudaEventRecord(sh_join_, side_) != cudaSuccess) { err = "mtp: the shared expert's branch"; return false; }
        }
        // #783 PR-c (stuchapin909): a multi-token window routes in two launches (one router GEMV reading the weight once,
        // one top-10 over all rows - each row bitwise the single-token call) and combines in one
        if (T > 1 && native_router_enabled() && g.n_expert == 512 && K == 10) {
            bf16_gemv_fp32_mmvf_multi(mixed_, N, bf16("mlp.gate.weight"), logits_, g.n_expert, (int) N, (int) g.n_expert, T, cs);
            native_router_top10_multi(logits_, ids_, w_, T, cs);
        } else {
        for (int t = 0; t < T; ++t) {
            bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, (int) N, (int) g.n_expert, cs);
            if (native_router_enabled() && g.n_expert == 512 && K == 10) native_router_top10(logits_ + t * g.n_expert, ids_ + t * K, w_ + t * K, cs);
            else router_top10(logits_ + t * g.n_expert, 1, (int) g.n_expert, (int) K, ids_ + t * K, w_ + t * K, cs);
        }
        }
        if (hq_record_mode_ != 0) {
#ifdef STRATA_NATIVE_EXPERTS
            const int type = hq_record_mode_ == 1 ? 42 : 2;
            const auto layout = native_expert_layout(type, type, N, g.n_ff);
            moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) layout.bytes, grp_ptr_,
                               grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
            native_quantize_q8_1(mixed_, hq_xq_, (int) N, T, cs);
            native_expert_grouped(layout, grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_,
                                  (int64_t) T * K, (int64_t) T * K, hq_xq_, hq_scratch_, parts_, cs);
#else
            err = "mtp hq: native expert kernels unavailable"; return false;
#endif
        } else {
        moe_group_resident(ids_, (int) (T * K), (int) K, experts_, (int64_t) strata::kernels::cpu::BLOB, grp_ptr_,
                           grp_start_, grp_counts_, hit_dst_, hit_slot_, cs);
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, (int64_t) T * K, (int64_t) T * K, hit_xq_,
                       hit_xs_, hit_scratch_, parts_, cs);
        }
        if (branch_on) {
            if (cudaStreamWaitEvent(cs, sh_join_, 0) != cudaSuccess) { err = "mtp: the shared expert's join"; return false; }
        } else {
            shared_rows();
        }
        static const bool no_multi_gr = [] {   // #783 PR-g: STRATA_NO_MULTI_GR=1 keeps the per-token GR calls
            const char* v = std::getenv("STRATA_NO_MULTI_GR");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        for (int t = 0; t < T; ++t) {
            if (!(T > 1 && native_moe_combine_enabled())) {
                if (native_moe_combine_enabled())
                    native_moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
                else
                    moe_combine(parts_ + (size_t) t * K * N, w_ + t * K, shared_ + t * N, y_ + t * N, N, K, cs);
            }
            if (!fuse_head_gr && no_multi_gr)
                gr_write(R_ + (size_t) t * HC * N, y_ + t * N, inj2_ + t * HC, gs, R_ + (size_t) t * HC * N, cs);
        }
        if (T > 1 && native_moe_combine_enabled())
            native_moe_combine_multi(parts_, w_, shared_, y_, N, K, T, cs);
        // #783 PR-g (stuchapin909): the T residual writes in one launch (each token's R, block output and injection are
        // its own, so doing them after the loop changes nothing)
        if (!fuse_head_gr && !no_multi_gr)
            gr_write_multi(R_, y_, inj2_, gs, R_, T, cs);
        // ---- the final mixer and the main model's head
        if (fuse_head_gr) {
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = true;
                fa[t].bo_prev = y_ + t * N; fa[t].inj_prev = inj2_ + t * HC;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].w_inject = nullptr;
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC;
                fa[t].inject_out = dummy_inj_; fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else if (head_mix_multi_on) {
            // the window's rows in one read (CUDA; HIP keeps gr_read per row): bitwise the same sums
            FusedGrArgs fa[kFusedGrMaxT];
            for (int t = 0; t < T; ++t) {
                fa[t].R = R_ + (size_t) t * HC * N; fa[t].R_out = R_ + (size_t) t * HC * N; fa[t].apply = false;
                fa[t].w_norm = f32("hyper_connection_mixer.hc_norm.weight");
                fa[t].w_down = bf16("hyper_connection_mixer.input_mix_weight_down.weight");
                fa[t].w_up = bf16("hyper_connection_mixer.input_mix_weight_up.weight");
                fa[t].eps = EPS; fa[t].lo = lo_ + t * g.hc_lr; fa[t].rs = rs_ + t * HC; fa[t].mixed = sample_ + t * N;
            }
            fused_gr_read_multi(fa, T, xn_, cs);
        } else {
            for (int t = 0; t < T; ++t)
                gr_read(R_ + (size_t) t * HC * N, f32("hyper_connection_mixer.hc_norm.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_down.weight"),
                        bf16("hyper_connection_mixer.input_mix_weight_up.weight"), nullptr, EPS, gs,
                        private_gr_arena_ ? private_gr_ : ss.block.gr,
                        sample_ + t * N, dummy_inj_, cs);
        }
        native_quantize_q8_1(sample_, xq_, (int) N, T, cs);
        const bool sub = dhead_ != nullptr;
        const int64_t nv = sub ? n_dvocab_ : n_vocab_;
        native_mmvq(sub ? dhead_type_ : head_->type(), sub ? dhead_ : head_->weights(), xq_, head_logits_, (int) N,
                    (int) nv, T, cs);
        if (coupled_rec_) {
            // coupled draft sampling: the target's chain and Philox draw for the row that will verify this draft
            // (counter = this cell + 1, from its step record), penalties over the ring; T == 1 (full layer)
            if (m_q_ != nullptr)   // STRATA_SPEC_PROB: a draw from q with its own stream, and q's list for the verifier
                spec_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                  cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, m_q_,
                                  spec_gate_pick(), spec_draft_temp_scale(), cs);
            else
            coupled_draft_sample(head_logits_, (int) nv, sub ? dvocab_ : nullptr, sub ? dinv_ : nullptr, (int) n_vocab_,
                                 cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, cs);
            return true;
        }
        if (argmax_rows_wanted()) {
            argmax_rows(head_logits_, T, (int) nv, arg_scratch_, out_ids_, cs);
        } else {
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(head_logits_, T, (int) nv, nullptr, 0, sp, out_ids_, cs);
        }
        if (multi_block_head_ops()) row_top_prob_split(head_logits_, T, (int) nv, out_ids_, probs_, top_scratch_, cs);
        else row_top_prob(head_logits_, T, (int) nv, out_ids_, probs_, cs);
        if (sub) map_ids(out_ids_, dvocab_, T, cs);
    } catch (const std::exception& e) {
        err = std::string("mtp: ") + e.what();
        return false;
    }
    return true;
}

namespace {
bool finish_capture(cudaStream_t cs, bool ok, cudaGraphExec_t& exec, const char* what, std::string& err) {
    cudaGraph_t graph = nullptr;
    const cudaError_t ce = cudaStreamEndCapture(cs, &graph);
    if (!ok) {
        if (graph) cudaGraphDestroy(graph);
        return false;
    }
    if (ce != cudaSuccess || cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        err = std::string("mtp: ") + what + " capture: " + cudaGetErrorString(ce);
        return false;
    }
    cudaGraphDestroy(graph);
    // an explicit upload: the first launch's implicit one blocked behind a device-side spin (verify.cpp)
    cudaGraphUpload(exec, cs);
    cudaStreamSynchronize(cs);
    return true;
}
}  // namespace

bool MtpDrafter::capture_prefill(int T, std::string& err) {
    if (prefill_exec_[T]) return true;
    using namespace strata::kernels;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) T * g_->n_head, cs_);
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T)
    return finish_capture(cs_, ok, prefill_exec_[T], "prefill", err);
}

bool MtpDrafter::capture_prefill_dev(int T, std::string& err) {
    if (prefill_dev_exec_[T]) return true;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    const bool ok = record_forward(T, -1, cs_, err);   // K/V only, rows [0, T); tok_/step_/pos_ filled before launch
    return finish_capture(cs_, ok, prefill_dev_exec_[T], "prefill (device inputs)", err);
}

bool MtpDrafter::capture_round(int T, bool coupled, std::string& err) {
    cudaGraphExec_t& exec = hq_record_mode_ ? hq_round_[hq_record_mode_ - 1][T] :
        (selective_record_ ? selective_round_[T] : (coupled ? round_exec_c_[T] : round_exec_[T]));
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
    bool ok = true;
    // coupled: the request's chain and the penalty history's base, for this round's drafts
    if (coupled) coupled_draft_stage(m_cparams_, m_chist_, cparams_, cring_, kCoupledHistCap, cs_);
    // the catch-up: the layer's front for the window's T cells (their K/V), then its rest for row a only, on row
    // a's intermediates copied to row 0, at the cell the host staged in step row 2*max_t - 1; the draft chain is
    // one graph per step (`capture_step`) so the host can stop it when a draft is unlikely
    const int ra = 2 * max_t_ - 1;
    const int64_t N = g_->n_embd, NH = g_->n_head;
#if defined(STRATA_USE_HIP)   // AMD keeps its separate copies
    copy_i32_from_mapped(tok_, m_tok_, T, cs_);
    copy_i32_from_mapped(step_, m_step_, (int64_t) 2 * T * 4, cs_);
    copy_i32_from_mapped(pos_, m_pos_, (int64_t) 2 * T * NH, cs_);
    copy_i32_from_mapped(row_, m_row_, 2, cs_);
    copy_from_mapped(Rin_, window_R_, (int64_t) T * HCN, cs_);
    copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + ra * NH, m_pos_ + ra * NH, NH, cs_);
#else   // the draft round's seven inputs in one launch
    const MappedCopy in[7] = {{tok_, m_tok_, T}, {step_, m_step_, (int64_t) 2 * T * 4}, {pos_, m_pos_, 2 * T * NH},
                              {row_, m_row_, 2}, {Rin_, window_R_, T * HCN}, {step_ + ra * 4, m_step_ + ra * 4, 4},
                              {pos_ + ra * NH, m_pos_ + ra * NH, NH}};
    copy_from_mapped_multi(in, 7, cs_);
#endif
    // At this point Rin contains the unchanged target rows; row_[0] is accepted row a.
    // Capture the anchor before record_front/rest/mtp_select can overwrite or transform these inputs.
    if (anchor_config_)
        mtp_anchor_seed(Rin_, HCN, T, row_, anchor_m_meta_, anchor_, anchor_valid_, anchor_stats_, cs_);
    ok = record_front(T, 0, cs_, err);
    if (ok) {
        if (T > 1) {
            copy_row_to_first(row_, R_, HCN, inj_, g_->hc, mixed_, N, cs_);
            native_quantize_q8_1(mixed_, xq_, (int) N, 1, cs_);
        }
        coupled_rec_ = coupled;
        coupled_j_ = 0;
        ok = record_rest(ra, cs_, err);
        coupled_rec_ = false;
    }
    // record_rest has selected/mapped its greedy output; row_[1] is the exact row used by mtp_select.
    // The later recursive steps overwrite head_logits_, so observe it here on the same captured stream.
    if (ok && first_top2_config_)
        mtp_first_top2(head_logits_, (int) (dhead_ ? n_dvocab_ : n_vocab_), 1, row_ + 1,
                       dhead_ ? dvocab_ : nullptr, out_ids_, first_top2_mapped_, cs_);
    if (ok && observer_config_)
        mtp_observer_copy(R_, HCN, sample_, N, row_ + 1, observer_m_meta_, observer_native_, observer_markers_, 0, kObserverChains, cs_);
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, 0, cs_);   // chain_launch: step 1's input
    return finish_capture(cs_, ok, exec, coupled ? "round (coupled)" : "round", err);
}

// Chain step j (1..max_t-2): one row at the cell staged in step row `max_t + j - 1`, from the previous step's
// residual and token (left in Rin_[0] / tok_[0] by mtp_select); draft j and its probability to the mapped outputs.
bool MtpDrafter::capture_step(int j, bool coupled, std::string& err) {
    cudaGraphExec_t& exec = hq_record_mode_ ? hq_step_[hq_record_mode_ - 1][j] :
        (selective_record_ ? selective_step_[j] : (coupled ? step_exec_c_[j] : step_exec_[j]));
    if (exec) return true;
    using namespace strata::kernels;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int row = max_t_ + j - 1;
    if (cudaStreamBeginCapture(cs_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { err = "mtp: begin capture"; return false; }
#if defined(STRATA_USE_HIP)
    copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
#else
    const MappedCopy in[2] = {{step_ + row * 4, m_step_ + row * 4, 4},
                              {pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head}};
    copy_from_mapped_multi(in, 2, cs_);
#endif
    coupled_rec_ = coupled;
    coupled_j_ = j;
    if (anchor_config_)
        strata::kernels::mtp_anchor_step(Rin_, HCN, anchor_m_meta_, anchor_, anchor_valid_, anchor_stats_, cs_);
    if (feature_config_)
        strata::kernels::mtp_feature_step(Rin_, HCN, tok_, feature_bank_, feature_fixture_, feature_m_meta_,
                                         feature_mask_, feature_stats_, j, cs_);
    bool ok = record_forward(1, row, cs_, err);
    coupled_rec_ = false;
    if (ok && observer_config_ && j < kObserverDepth)
        mtp_observer_copy(R_, HCN, sample_, g_->n_embd, row_ + 1, observer_m_meta_, observer_native_, observer_markers_, j, kObserverChains, cs_);
    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, j, cs_);   // chain_launch: step j+1's input
    return finish_capture(cs_, ok, exec, coupled ? "step (coupled)" : "step", err);
}

void MtpDrafter::kv_restore(int64_t upto) {
    const OnDevice on_device(device_);
    if (st_.kv_mode != 2 || upto <= 0) return;
    // the ring's blocks below `upto`, from the host copy: a checkpoint resume may have left later cells in them
    const strata::kernels::QsaShapes s = shapes_of(*g_);
    const int64_t b1 = (upto + s.page_size - 1) / s.page_size, b0 = std::max<int64_t>(0, b1 - st_.n_slots);
    strata::kernels::kv_ring_restore(qsa_attn_pools(st_), st_.host, qsa_kv_format(st_), b0, b1, st_.n_slots, s, cs_);
    cudaStreamSynchronize(cs_);
}

bool MtpDrafter::prepare_prefill(std::string& err) {
    const OnDevice on_device(device_);
    const int64_t per_row = 1 + 4 + g_->n_head;
    if (pf_cap_ < (int64_t) max_t_ * per_row) {
        if (pf_dev_) cudaFree(pf_dev_);
        pf_dev_ = nullptr;
        pf_cap_ = 0;
        if (cudaMalloc((void**) &pf_dev_, (size_t) (max_t_ * per_row) * sizeof(int32_t)) != cudaSuccess) {
            err = "mtp prefill: the input records do not fit";
            return false;
        }
        pf_cap_ = max_t_ * per_row;
    }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture_prefill_dev(T, err)) return false;   // (STRATA_MTP_PREFILL_SYNC's loop syncs anyway)
    return true;
}

bool MtpDrafter::acquire_source_slot(bool wait, int& slot, std::string& err) {
    const OnDevice on_device(device_);
    for (int i = 0; i < 2; ++i) {
        auto& s = source_slots_[i];
        if (s.host == nullptr || s.consumed == nullptr) {
            err = "mtp: cross-device residual staging was not bound";
            return false;
        }
        if (s.live) {
            const cudaError_t ce = cudaEventQuery(s.consumed);
            if (ce == cudaErrorNotReady) continue;
            if (ce != cudaSuccess) {
                err = std::string("mtp: residual staging event: ") + cudaGetErrorString(ce);
                return false;
            }
            s.live = false;
        }
        slot = i;
        return true;
    }
    if (!wait) {
        err = "mtp: both asynchronous residual staging slots are still in use";
        return false;
    }
    const cudaError_t ce = cudaEventSynchronize(source_slots_[0].consumed);
    if (ce != cudaSuccess) {
        err = std::string("mtp: waiting for residual staging: ") + cudaGetErrorString(ce);
        return false;
    }
    source_slots_[0].live = false;
    slot = 0;
    return true;
}

bool MtpDrafter::read_source_rows(float* host, const float* rows, int64_t n, std::string& err) {
    if (!host || !rows || n < 1 || source_cs_ == nullptr) {
        err = "mtp: invalid cross-device residual source";
        return false;
    }
    const OnDevice on_source(source_device_);
    const size_t bytes = (size_t) n * (size_t) g_->hc * (size_t) g_->n_embd * sizeof(float);
    // The caller has completed the producer. Finish this one D2H per window/chunk before returning, so it
    // can reuse that source window without adding another cross-device event to the verifier's ring.
    cudaError_t ce = cudaMemcpyAsync(host, rows, bytes, cudaMemcpyDeviceToHost, source_cs_);
    if (ce == cudaSuccess) ce = cudaStreamSynchronize(source_cs_);
    if (ce != cudaSuccess) {
        err = std::string("mtp: reading residuals from the source device: ") + cudaGetErrorString(ce);
        return false;
    }
    return true;
}

bool MtpDrafter::finish_source_slot(int slot, std::string& err) {
    auto& s = source_slots_[slot];
    const cudaError_t ce = cudaEventRecord(s.consumed, cs_);
    if (ce != cudaSuccess) {
        // Even an error after a successful H2D enqueue must not let the host buffer be reused while in flight.
        cudaStreamSynchronize(cs_);
        s.live = false;
        err = std::string("mtp: protecting residual staging: ") + cudaGetErrorString(ce);
        return false;
    }
    s.live = true;
    return true;
}

bool MtpDrafter::stage_remote_rows(float* local, const float* rows, int n, std::string& err) {
    if (!local || n < 1 || n > max_t_) { err = "mtp: residual window out of range"; return false; }
    int slot = -1;
    if (!acquire_source_slot(true, slot, err) || !read_source_rows(source_slots_[slot].host, rows, n, err)) return false;
    const size_t bytes = (size_t) n * (size_t) g_->hc * (size_t) g_->n_embd * sizeof(float);
    const cudaError_t ce = cudaMemcpyAsync(local, source_slots_[slot].host, bytes, cudaMemcpyHostToDevice, cs_);
    if (ce != cudaSuccess) {
        cudaStreamSynchronize(cs_);
        err = std::string("mtp: uploading residuals to the draft device: ") + cudaGetErrorString(ce);
        return false;
    }
    return finish_source_slot(slot, err);
}

bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                         std::string& err, bool sync) {
    if (!cross_device()) return prefill_impl(R_rows, next_tokens, n, cell0, err, sync, cudaMemcpyDeviceToDevice);
    const OnDevice on_device(device_);
    if (chain_live_) { err = "mtp: prefill cannot overlap a live draft chain"; return false; }
    if (n < 0 || (n > 0 && (!R_rows || !next_tokens))) { err = "mtp: invalid prefill inputs"; return false; }
    if (n == 0 || cell0 + n <= first_needed()) return true;
    const Clock::time_point t0 = Clock::now();
    const double prior_ms = ms_prefill;
    const int64_t HCN = g_->hc * g_->n_embd;
    const int64_t per_row = 1 + 4 + g_->n_head;
    int slot = -1;
    float* host_rows = nullptr;
    if (!sync) {
        // Pipeline callers reserve/capture up front. Never resize, capture, or wait for the drafter here.
        if (n > max_t_ || pf_cap_ < n * per_row || prefill_dev_exec_[(int) n] == nullptr || mtp_prefill_per_group_sync()) {
            err = "mtp: async cross-device prefill needs prepare_prefill, n <= max_t and STRATA_MTP_PREFILL_SYNC=0";
            return false;
        }
        if (!acquire_source_slot(false, slot, err)) return false;
        host_rows = source_slots_[slot].host;
    } else {
        // Skip only whole original groups: the first group that crosses first_needed still has the same rows.
        const int64_t skip = first_needed() > cell0 ? ((first_needed() - cell0) / max_t_) * max_t_ : 0;
        if (skip > 0) {
            R_rows += (size_t) skip * (size_t) HCN;
            next_tokens += skip;
            cell0 += skip;
            n -= skip;
        }
        if (prefill_host_cap_ < n) {
            if (prefill_host_R_) cudaFreeHost(prefill_host_R_);
            prefill_host_R_ = nullptr;
            prefill_host_cap_ = 0;
            if (cudaHostAlloc((void**) &prefill_host_R_, (size_t) n * (size_t) HCN * sizeof(float),
                              cudaHostAllocPortable) != cudaSuccess) {
                err = "mtp: pinned prompt residual staging allocation failed";
                return false;
            }
            prefill_host_cap_ = n;
        }
        host_rows = prefill_host_R_;
    }
    if (!read_source_rows(host_rows, R_rows, n, err)) return false;
    const bool ok = prefill_impl(host_rows, next_tokens, n, cell0, err, sync, cudaMemcpyHostToDevice);
    // Record even on failure: the helper may already have enqueued an upload from this slot.
    const bool protected_slot = slot < 0 || finish_source_slot(slot, err);
    if (!ok || !protected_slot) cudaStreamSynchronize(cs_);
    ms_prefill = prior_ms + ms_since(t0);
    return ok && protected_slot;
}

bool MtpDrafter::prefill_impl(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                              std::string& err, bool sync, cudaMemcpyKind residual_kind) {
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
    // cells the window can never reach again need no K/V
    const int64_t first_needed = this->first_needed();
    // E-4: every group's token / step / position records uploaded at once; each group is then device copies and a
    // graph on the one stream, with a single sync at the end (a group of <= max_t rows used to be staged in mapped
    // memory and synced before the next: ~5,500 host round trips on a 32K prompt).  The same work in the same
    // order.  STRATA_MTP_PREFILL_SYNC=1 keeps the old loop, =0 forces E-4.
    // HIP defaults to the old loop.  This pass runs whenever E-9 (Prefill::draft_kv) declines, which it does for the
    // drafter's ring (KV streaming, --kv-resident), and on gfx1201 / ROCm 7.2.4 the E-4 queue (a graph launch and four
    // device copies per group, ~2,000 groups per 8192-token chunk, no sync) sometimes never completes: the prompt hangs
    // in hipGraphLaunch or the final sync until the watchdog ends the engine.  R9700, full IQ3_XXS, --kv-resident
    // 32768, mixed load (chats, 32K and 7K prompts, deep follow-ups): E-4 hung in the first round on 2 of 2 tries,
    // the old loop ran 30 of 30 rounds clean, prompt speed unchanged.
    const bool per_group_sync = mtp_prefill_per_group_sync();
    const int64_t NHp = g_->n_head, per_row = 1 + 4 + NHp;
    if (!per_group_sync && n > 0) {
        if (pf_cap_ < n * per_row) {
            if (pf_dev_) cudaFree(pf_dev_);
            pf_dev_ = nullptr;
            pf_cap_ = 0;
            if (cudaMalloc((void**) &pf_dev_, (size_t) (n * per_row) * sizeof(int32_t)) != cudaSuccess) {
                err = "mtp prefill: the input records do not fit";
                return false;
            }
            pf_cap_ = n * per_row;
        }
        std::vector<int32_t> rec((size_t) (n * per_row));
        int32_t* tk = rec.data();
        int32_t* stp = tk + n;
        int32_t* ps = stp + 4 * n;
        for (int64_t i = 0; i < n; ++i) {
            const int64_t cell = cell0 + i;
            tk[i] = next_tokens[i];
            stp[i * 4 + 0] = (int32_t) cell;
            stp[i * 4 + 1] = (int32_t) (cell + 1);
            stp[i * 4 + 2] = (int32_t) ((cell + 1) / 4);
            stp[i * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < NHp; ++h) ps[i * NHp + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(pf_dev_, rec.data(), rec.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs_) != cudaSuccess) {
            err = "mtp prefill: the input records upload failed";
            return false;
        }
        const int32_t* d_tk = pf_dev_;
        const int32_t* d_stp = d_tk + n;
        const int32_t* d_ps = d_stp + 4 * n;
        for (int64_t c = 0; c < n; c += max_t_) {
            const int T = (int) std::min<int64_t>(max_t_, n - c);
            if (cell0 + c + T <= first_needed) continue;
            if (!capture_prefill_dev(T, err)) return false;
            if (cudaMemcpyAsync(tok_, d_tk + c, (size_t) T * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(step_, d_stp + c * 4, (size_t) T * 16, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(pos_, d_ps + c * NHp, (size_t) (T * NHp) * 4, cudaMemcpyDeviceToDevice, cs_) != cudaSuccess ||
                cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), residual_kind,
                                cs_) != cudaSuccess ||
                cudaGraphLaunch(prefill_dev_exec_[T], cs_) != cudaSuccess) {
                err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        if (!sync) {   // the caller orders on stream(); the records are staged (a pageable copy returns once taken)
            (void) cudaStreamQuery(cs_);
            ms_prefill += ms_since(t0);
            return true;
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        ms_prefill += ms_since(t0);
        return true;
    }
    for (int64_t c = 0; c < n; c += max_t_) {
        const int T = (int) std::min<int64_t>(max_t_, n - c);
        if (cell0 + c + T <= first_needed) continue;
        if (!capture_prefill(T, err)) return false;
        for (int t = 0; t < T; ++t) {
            const int64_t cell = cell0 + c + t;
            h_tok_[t] = next_tokens[c + t];
            h_step_[t * 4 + 0] = (int32_t) cell;
            h_step_[t * 4 + 1] = (int32_t) (cell + 1);
            h_step_[t * 4 + 2] = (int32_t) ((cell + 1) / 4);
            h_step_[t * 4 + 3] = (int32_t) (cell + 1);
            for (int64_t h = 0; h < g_->n_head; ++h) h_pos_[t * g_->n_head + h] = (int32_t) cell;
        }
        if (cudaMemcpyAsync(Rin_, R_rows + (size_t) c * HCN, (size_t) T * HCN * sizeof(float), residual_kind,
                            cs_) != cudaSuccess ||
            cudaGraphLaunch(prefill_exec_[T], cs_) != cudaSuccess ||
            cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp prefill: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    ms_prefill += ms_since(t0);
    return true;
}

bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                       float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
    if (chain_live_) { err = "mtp: a pipelined chain is still in flight"; return false; }
    // #783 PR-i (stuchapin909): the round runs for the cells up to the accepted row a (T = a + 1): the K/V of the rejected
    // rows is not caught up - no later read reaches a cell past the one being drafted and the next round writes it first
    if (!mtp_catchup_all()) T = a + 1;
    const bool cp = coupled_active_;   // coupled draft sampling for this request: its own graphs
    if (!capture_round(T, cp, err)) return false;
    const int max_steps = std::min(max_t_ - 1, max_drafts_);
    for (int j = 1; j < max_steps; ++j)
        if (!capture_step(j, cp, err)) return false;
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, coupled_draft_cell(p, a, 0));   // p + a: draft 0's cell
    for (int j = 1; j < max_steps; ++j)
        put(max_t_ + j - 1, coupled_draft_cell(p, a, j));
    for (int j = 0; j < max_steps; ++j) ((volatile int32_t*) h_out_)[j] = -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    if (h_force_ != nullptr) for (int j = 0; j < 16; ++j) h_force_[j] = -1;   // the forcing kernels: no-ops here
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!stage_source_R(T, err)) return false;

    SessionState* const pss = ple_ss_ ? ple_ss_ : ss_;
    const bool do_ple = pss != nullptr && pss->ple.ready() && pss->ple.table != nullptr;
    int32_t ple_prev[2] = {do_ple ? pss->ple_prev[0] : 0, do_ple ? pss->ple_prev[1] : 0};
    auto prefetch_ple = [&](int32_t tok) {
        if (!do_ple || tok < 0) return;
        uint32_t rows16[strata::kernels::PLE_N_HEADS];
        strata::kernels::ngram_rows(&tok, ple_prev, 1, pss->ple.consts, rows16);
        ple_prev[0] = ple_prev[1];
        ple_prev[1] = tok;
        pss->ple.table->prefetch_rows(rows16);
    };

    int n = 0;
    if (min_p <= 0.0f && max_steps > 0) {
        if (cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        for (int j = 1; j < max_steps; ++j) {
            if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
        }
        (void) cudaStreamQuery(cs_);
        prefetch_ple(tokens[a]);
        for (int j = 0; j + 1 < max_steps; ++j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                _mm_pause();
#endif
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            if (probs) probs[j] = ((volatile float*) h_prob_)[j];
            prefetch_ple(drafts[j]);
        }
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        const int last = max_steps - 1;
        drafts[last] = ((volatile int32_t*) h_out_)[last];
        if (probs) probs[last] = ((volatile float*) h_prob_)[last];
        prefetch_ple(drafts[last]);
        n = max_steps;
    } else if (max_steps > 0) {
        if (cudaGraphLaunch(cp ? round_exec_c_[T] : round_exec_[T], cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        (void) cudaStreamQuery(cs_);
        prefetch_ple(tokens[a]);
        auto wait_step = [&](int j) {
            uint32_t spins = 0;
            while (((volatile int32_t*) h_out_)[j] < 0) {
#if defined(_WIN32) || defined(__x86_64__)
                _mm_pause();
#endif
                if ((++spins & 1023u) == 0 && cudaStreamQuery(cs_) != cudaErrorNotReady) break;
            }
        };
        wait_step(0);
        drafts[0] = ((volatile int32_t*) h_out_)[0];
        float pj = ((volatile float*) h_prob_)[0];
        if (probs) probs[0] = pj;
        n = 1;
        for (int j = 1; j < max_steps && pj >= min_p; ++j) {
            if (cudaGraphLaunch(cp ? step_exec_c_[j] : step_exec_[j], cs_) != cudaSuccess) {
                err = std::string("mtp draft step: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            (void) cudaStreamQuery(cs_);
            prefetch_ple(drafts[j - 1]);
            wait_step(j);
            drafts[j] = ((volatile int32_t*) h_out_)[j];
            pj = ((volatile float*) h_prob_)[j];
            if (probs) probs[j] = pj;
            ++n;
        }
        prefetch_ple(drafts[n - 1]);
        if (cudaStreamSynchronize(cs_) != cudaSuccess) {
            err = std::string("mtp draft: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
    }
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }
    if (n_drafts) *n_drafts = n;
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::neuron_probe_100(const float* R_row, const int32_t* roots, int64_t cell,
                                 std::vector<int32_t>& candidates, double& prediction_ms, std::string& err) {
    const OnDevice on_device(device_);
    if (cross_device()) { err = "neuron probe: cross-device drafting is not supported by this diagnostic"; return false; }
    if (!R_row || !roots || cell < 0 || cell > 2048 || max_t_ < 4 || coupled_active_ ||
        (window_ > 0 && window_ < cell + 16)) {
        err = "neuron probe: requires short prefix, greedy MTP, window >= prefix+16, max_t >= 4";
        return false;
    }
    const size_t bytes = (size_t) g_->hc * g_->n_embd * sizeof(float);
    auto check = [&](cudaError_t ce) {
        if (ce == cudaSuccess) return true;
        err = std::string("neuron probe: ") + cudaGetErrorString(ce); return false;
    };
    candidates.assign(100, 0);
    const auto prediction_start = Clock::now();
    // Recompute in reverse order too: any dependence on a preceding branch's stale KV is an error.
    for (int pass = 0; pass < 2; ++pass) {
        for (int b0 = 0; b0 < 10; ++b0) {
            const int b = pass ? 9 - b0 : b0;
            int32_t token = roots[b];
            if (!check(cudaMemcpyAsync(Rin_, R_row, bytes, cudaMemcpyDeviceToDevice, cs_))) return false;
            if (!pass) candidates[b * 10] = token;
            for (int j = 0; j < 9; ++j) {
                const int32_t c = (int32_t) (cell + j);
                const int32_t rec[4] = {c, c + 1, (c + 1) / 4, c + 1};
                std::vector<int32_t> positions((size_t) g_->n_head, c);
                if (!check(cudaMemcpyAsync(tok_, &token, sizeof(token), cudaMemcpyHostToDevice, cs_)) ||
                    !check(cudaMemcpyAsync(step_, rec, sizeof(rec), cudaMemcpyHostToDevice, cs_)) ||
                    !check(cudaMemcpyAsync(pos_, positions.data(), positions.size() * sizeof(int32_t), cudaMemcpyHostToDevice, cs_)) ||
                    !record_forward(1, 0, cs_, err) ||
                    !check(cudaMemcpyAsync(&token, out_ids_, sizeof(token), cudaMemcpyDeviceToHost, cs_)) ||
                    !check(cudaStreamSynchronize(cs_))) return false;
                if (token < 0 || token >= n_vocab_) { err = "neuron probe: invalid draft token"; return false; }
                if (!pass) candidates[b * 10 + j + 1] = token;
                else if (candidates[b * 10 + j + 1] != token) {
                    err = "neuron probe: branch-order parity failed"; return false;
                }
                if (!check(cudaMemcpyAsync(Rin_, R_, bytes, cudaMemcpyDeviceToDevice, cs_))) return false;
            }
        }
        if (!pass) {
            if (!check(cudaStreamSynchronize(cs_))) return false;
            prediction_ms = ms_since(prediction_start);
        }
    }
    if (!check(cudaStreamSynchronize(cs_))) return false;
    std::vector<int32_t> baseline((size_t) max_t_);
    if (!draft(1, roots, cell, 0, baseline.data(), err)) return false;
    for (int j = 0; j < 3; ++j) if (baseline[j] != candidates[j + 1]) {
        err = "neuron probe: ordinary MTP chain parity failed"; return false;
    }
    std::fprintf(stderr, "neuron probe: 100 candidates; reverse-branch and ordinary-chain parity PASS\n");
    return true;
}

bool MtpDrafter::stage_source_R(int T, std::string& err) {
    if (source_already_staged_) return true;
    if (cross_device())
        return stage_remote_rows(local_window_R_, src_R_ ? src_R_ : bound_source_R_, T, err);
    if (src_R_ == nullptr || src_R_ == window_R_) return true;
    if (cudaMemcpyAsync((void*) window_R_, src_R_, (size_t) T * (size_t) (g_->hc * g_->n_embd) * sizeof(float),
                        cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
        err = "mtp: staging the window's residual rows failed";
        return false;
    }
    return true;
}

bool MtpDrafter::prepare_chain(std::string& err) {
    const OnDevice on_device(device_);
    if (!force_on_ || h_force_ == nullptr) { err = "mtp: chains need set_force_capture before load"; return false; }
    for (int T = 1; T <= max_t_; ++T)
        if (!capture_round(T, false, err)) return false;
    for (int j = 1; j <= max_t_ - 2; ++j)
        if (!capture_step(j, false, err)) return false;
    if (selective_supported_) {
        selective_record_ = true;
        bool ok = true;
        for (int T = 1; ok && T <= max_t_; ++T) ok = capture_round(T, false, err);
        for (int j = 1; ok && j <= max_t_ - 2; ++j) ok = capture_step(j, false, err);
        selective_record_ = false;
        if (!ok) return false;
    }
    if (!hq_pack_.empty()) {
        bool ok = true;
        for (int mode = 1; ok && mode <= 2; ++mode) {
            hq_record_mode_ = mode;
            for (int T = 1; ok && T <= max_t_; ++T) ok = capture_round(T, false, err);
            for (int j = 1; ok && j <= max_t_ - 2; ++j) ok = capture_step(j, false, err);
        }
        hq_record_mode_ = 0;
        if (!ok) return false;
    }
    if (ev_chain_ == nullptr && cudaEventCreateWithFlags(&ev_chain_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: event creation failed";
        return false;
    }
    for (cudaEvent_t& e : ev_step_)
        if (e == nullptr && cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
            err = "mtp: event creation failed";
            return false;
        }
    return true;
}

bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force,
                              int n_out, int n_early, std::string& err) {
    const OnDevice on_device(device_);
    if (chain_live_) { err = "mtp: a chain is already in flight"; return false; }
    if (!force_on_ || ev_chain_ == nullptr) { err = "mtp: chains need prepare_chain"; return false; }
    if (T < 1 || T > max_t_ || a < 0 || a >= T || n_out < 1 || n_out > max_t_ - 1 || n_force < 0 || n_force > n_out ||
        n_force > 16) {
        err = "mtp: chain arguments out of range";
        return false;
    }
    if ((hq_mode_ ? hq_round_[hq_mode_ - 1][T] : round_exec_[T]) == nullptr) {
        err = "mtp: chain graphs not prepared"; return false;
    }
    for (int j = 1; j < n_out; ++j)
        if ((hq_mode_ ? hq_step_[hq_mode_ - 1][j] : step_exec_[j]) == nullptr) {
            err = "mtp: chain graphs not prepared"; return false;
        }
    // chain_live_ was checked above: ev_chain was consumed by chain_poll, never merely ev_step.
    if (observer_config_) {
        const bool detailed = observer_on_ && observer_window_ >= 0 && observer_window_ < kObserverWindows;
        const int64_t slot = detailed ? observer_chains_++ : -1;
        observer_h_meta_[0] = detailed && slot < kObserverChains ? 1 : 0;
        observer_h_meta_[1] = observer_h_meta_[0] ? (int32_t) slot : -1;
        if (detailed && slot >= kObserverChains) ++observer_overflow_;
    }
    const Clock::time_point t0 = Clock::now();
    const int64_t NH = g_->n_head;
    auto put = [&](int row, int64_t cell) {
        h_step_[row * 4 + 0] = (int32_t) cell;
        h_step_[row * 4 + 1] = (int32_t) (cell + 1);
        h_step_[row * 4 + 2] = (int32_t) ((cell + 1) / 4);
        h_step_[row * 4 + 3] = (int32_t) (cell + 1);
        for (int64_t h = 0; h < NH; ++h) h_pos_[row * NH + h] = (int32_t) cell;
    };
    for (int t = 0; t < T; ++t) {
        h_tok_[t] = tokens[t];
        put(t, p + t);
    }
    put(2 * max_t_ - 1, p + a);                                      // output 0's cell
    for (int j = 1; j < n_out; ++j) put(max_t_ + j - 1, p + a + j);  // every step's cell, staged at once
    for (int j = 0; j < 16; ++j) h_force_[j] = j < n_force ? force[j] : -1;
    h_row_[0] = a;
    h_row_[1] = 0;
    if (selective_config_ && !selective_stage(T, p, a, n_out, err)) return false;
    if (feature_config_) {
        const int64_t base = p + a - feature_header_.first_pos;
        if (base < 0 || base > 1024) { err = "mtp feature: chain base outside bank request"; return false; }
        const bool fresh = force == nullptr && n_force == 0;
        if (fresh) ++feature_fresh_; else ++feature_forced_;
        const bool root_ok = base < 1024 && tokens[a] == feature_ids_[base];
        if (fresh && !root_ok) ++feature_root_mismatch_;
        feature_h_meta_[0] = feature_mode_;
        feature_h_meta_[1] = (int32_t) base;
        feature_h_meta_[2] = feature_bank_rows_;
        feature_h_meta_[3] = fresh && root_ok && feature_bank_rows_ == 1024;
        if (cudaMemsetAsync(feature_mask_, 0, sizeof(int32_t), cs_) != cudaSuccess) {
            err = "mtp feature: chain mask reset failed"; return false;
        }
    }
    if (anchor_config_) {
        const bool fresh = force == nullptr && n_force == 0;
        anchor_h_meta_[1] = fresh ? 1 : 0;
        if (fresh) ++anchor_fresh_; else ++anchor_forced_;
    }
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!stage_source_R(T, err)) return false;
    if (first_top2_config_) first_top2_observed_ = {};
    chain_early_ = std::max(0, std::min(n_early, n_out));
    bool ok = cudaGraphLaunch(hq_mode_ ? hq_round_[hq_mode_ - 1][T] : (selective_active_ ? selective_round_[T] : round_exec_[T]), cs_) == cudaSuccess;
    if (ok && chain_early_ >= 1) ok = cudaEventRecord(ev_step_[0], cs_) == cudaSuccess;
    for (int j = 1; ok && j < n_out; ++j) {
        ok = cudaGraphLaunch(hq_mode_ ? hq_step_[hq_mode_ - 1][j] : (selective_active_ ? selective_step_[j] : step_exec_[j]), cs_) == cudaSuccess;
        if (ok && j < chain_early_) ok = cudaEventRecord(ev_step_[j], cs_) == cudaSuccess;
    }
    if (!ok || cudaEventRecord(ev_chain_, cs_) != cudaSuccess) {
        err = std::string("mtp chain: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }
    (void) cudaStreamQuery(cs_);   // WDDM: submit now
    steps_seen_ = 0;
    chain_live_ = true;
    chain_n_ = n_out;
    if (!hq_pack_.empty()) ++hq_chains_[hq_mode_];
    ms_draft += ms_since(t0);
    ++rounds;
    return true;
}

bool MtpDrafter::observer_allocate(std::string& err) {
    const uint64_t native = (uint64_t) kObserverChains * 4 * (g_->hc * g_->n_embd + g_->n_embd) * sizeof(float);
    const uint64_t target = (uint64_t) kObserverWindows * 4 * g_->hc * g_->n_embd * sizeof(float);
    observer_bytes_ = native + target;
    const size_t markers = kObserverChains * 4 * sizeof(int32_t);
    if (cudaMalloc((void**) &observer_native_, native) != cudaSuccess ||
        cudaMalloc((void**) &observer_markers_, markers) != cudaSuccess ||
        cudaHostAlloc((void**) &observer_host_markers_, markers, cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc((void**) &observer_target_, target) != cudaSuccess ||
        cudaHostAlloc((void**) &observer_host_, observer_bytes_, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void**) &observer_h_meta_, 2 * sizeof(int32_t), cudaHostAllocMapped) != cudaSuccess ||
        cudaHostGetDevicePointer((void**) &observer_m_meta_, observer_h_meta_, 0) != cudaSuccess) {
        err = "mtp observer: common private allocation failed before expert-cache sizing"; return false;
    }
    observer_h_meta_[0] = 0; observer_h_meta_[1] = -1;
    vram_ += observer_bytes_ + markers;
    return true;
}
bool MtpDrafter::observer_begin(bool on, std::string& err) {
    const OnDevice device(device_);
    if (!observer_config_ || chain_live_ || !idle(err)) { err = "mtp observer: GEN boundary must be fully idle"; return false; }
    observer_on_ = on; observer_chains_ = observer_overflow_ = 0; observer_window_ = -1;
    observer_h_meta_[0] = 0; observer_h_meta_[1] = -1;
    // Clear equal slabs in BOTH arms before decode; unread unused rows cannot leak old GEN data.
    const size_t nb = (size_t) kObserverChains * 4 * (g_->hc * g_->n_embd + g_->n_embd) * sizeof(float);
    if (cudaMemset(observer_markers_, 0, kObserverChains * 4 * sizeof(int32_t)) != cudaSuccess) {
        err = "mtp observer: marker reset failed"; return false;
    }
    if (cudaMemset(observer_native_, 0, nb) != cudaSuccess ||
        cudaMemset(observer_target_, 0, observer_bytes_ - nb) != cudaSuccess) {
        err = "mtp observer: boundary reset failed"; return false;
    }
    return true;
}
float* MtpDrafter::observer_target_slot(int seq) const {
    return observer_on_ && seq >= 0 && seq < kObserverWindows
        ? observer_target_ + (int64_t) seq * 4 * g_->hc * g_->n_embd : nullptr;
}
bool MtpDrafter::observer_finish(const std::string& path, int64_t request, std::FILE* out, std::string& err) {
    const OnDevice device(device_);
    const Clock::time_point drain_start = Clock::now();
    if (chain_live_ || !idle(err) || cudaDeviceSynchronize() != cudaSuccess) {
        err = "mtp observer: untimed dump requires full completion"; return false;
    }
    const double drain_ms = ms_since(drain_start);
    const size_t nb = (size_t) kObserverChains * 4 * (g_->hc * g_->n_embd + g_->n_embd) * sizeof(float);
    const size_t used = (size_t) std::min<int64_t>(observer_chains_, kObserverChains) * 4 *
                         (g_->hc * g_->n_embd + g_->n_embd) * sizeof(float);
    if (observer_on_) {
        if (cudaMemcpy(observer_host_, observer_native_, used, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy((char*) observer_host_ + nb, observer_target_, observer_bytes_ - nb, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "mtp observer: untimed payload copy failed"; return false;
        }
        const size_t marker_bytes = kObserverChains * 4 * sizeof(int32_t);
        if (cudaMemcpy(observer_host_markers_, observer_markers_, marker_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "mtp observer: marker copy failed"; return false;
        }
        const std::string marker_path = path + "." + std::to_string(request) + ".markers.i32";
        std::FILE* mf = std::fopen(marker_path.c_str(), "wb");
        if (!mf) { err = "mtp observer: marker open failed"; return false; }
        const bool marker_ok = std::fwrite(observer_host_markers_, 1, marker_bytes, mf) == marker_bytes;
        if (std::fclose(mf) != 0 || !marker_ok) { err = "mtp observer: marker write failed"; return false; }
        const std::string native_path = path + "." + std::to_string(request) + ".mtp.f32";
        const std::string target_path = path + "." + std::to_string(request) + ".target.f32";
        std::FILE* fp = std::fopen(native_path.c_str(), "wb");
        if (!fp) { err = "mtp observer: native payload open failed"; return false; }
        bool ok = std::fwrite(observer_host_, 1, used, fp) == used; ok = std::fclose(fp) == 0 && ok;
        fp = std::fopen(target_path.c_str(), "wb");
        if (!fp) { err = "mtp observer: target payload open failed"; return false; }
        ok = std::fwrite((char*) observer_host_ + nb, 1, observer_bytes_ - nb, fp) == observer_bytes_ - nb && ok;
        ok = std::fclose(fp) == 0 && ok;
        if (!ok) { err = "mtp observer: payload short write"; return false; }
    }
    std::fprintf(out, "strata lab observer: request %lld on=%d chain_slots=%lld chain_cap=%d overflow=%lld gpu_bytes=%llu pinned_bytes=%llu A0=deferred target_cap=%d depth=4 drain_ms=%.6f\n",
        (long long) request, observer_on_, (long long) observer_chains_, kObserverChains,
        (long long) observer_overflow_, (unsigned long long) (observer_bytes_ + kObserverChains*4*sizeof(int32_t)), (unsigned long long) (observer_bytes_ + kObserverChains*4*sizeof(int32_t) + 2*sizeof(int32_t)), kObserverWindows, drain_ms);
    return true;
}

void MtpDrafter::observe_first_top2() {
    if (!first_top2_config_ || first_top2_observed_.status != 0) return;
    const volatile auto& r = *first_top2_host_;
    first_top2_observed_.row = r.row; first_top2_observed_.first = r.first;
    first_top2_observed_.second = r.second; first_top2_observed_.pick = r.pick;
    first_top2_observed_.nonfinite = r.nonfinite; first_top2_observed_.gap = r.gap;
    first_top2_observed_.status = r.status;
}

int MtpDrafter::chain_outputs_ready(std::string& err) {
    if (!chain_live_) return chain_n_;
    const OnDevice on_device(device_);
    while (steps_seen_ < chain_early_) {
        const cudaError_t q = cudaEventQuery(ev_step_[steps_seen_]);
        if (q == cudaErrorNotReady) break;
        if (q != cudaSuccess) { err = std::string("mtp chain: ") + cudaGetErrorString(q); return -1; }
        chain_tok_[steps_seen_] = ((volatile int32_t*) h_out_)[steps_seen_];
        chain_prob_[steps_seen_] = ((volatile float*) h_prob_)[steps_seen_];
        ++steps_seen_;
    }
    if (steps_seen_ > 0) observe_first_top2();
    return steps_seen_;
}

int MtpDrafter::chain_poll(std::string& err) {
    if (!chain_live_) return 1;
    const OnDevice on_device(device_);
    const cudaError_t q = cudaEventQuery(ev_chain_);
    if (q == cudaErrorNotReady) return 0;
    chain_live_ = false;
    if (q != cudaSuccess) { err = std::string("mtp chain: ") + cudaGetErrorString(q); return -1; }
    observe_first_top2();
    for (int j = 0; j < chain_n_; ++j) {
        chain_tok_[j] = ((volatile int32_t*) h_out_)[j];
        chain_prob_[j] = ((volatile float*) h_prob_)[j];
    }
    for (int j = chain_n_; j < 8; ++j) { chain_tok_[j] = 0; chain_prob_[j] = 0.0f; }
    return 1;
}

bool MtpDrafter::draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                             float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    // row 0 is the real pair; rows 1.. repeat it and only write cells the next round overwrites
    const int64_t HCN = g_->hc * g_->n_embd;
    if (cross_device()) {
        if (T < 1 || T > max_t_ || chain_live_) { err = "mtp: invalid first draft or a chain is still live"; return false; }
        if (!stage_remote_rows(local_window_R_, R_row, 1, err)) return false;
        for (int t = 1; t < T; ++t) {
            if (cudaMemcpyAsync(local_window_R_ + (size_t) t * (size_t) HCN, local_window_R_,
                                (size_t) HCN * sizeof(float), cudaMemcpyDeviceToDevice, cs_) != cudaSuccess) {
                err = "mtp: repeating the first local residual failed";
                return false;
            }
        }
        std::vector<int32_t> toks((size_t) T, token);
        // draft() normally stages the verifier's bound window. This call has just supplied its own first row.
        source_already_staged_ = true;
        const bool ok = draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
        source_already_staged_ = false;
        return ok;
    }
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row, (size_t) HCN * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            err = "mtp: staging the first residual failed";
            return false;
        }
    std::vector<int32_t> toks((size_t) T, token);
    return draft(T, toks.data(), cell, 0, drafts, err, probs, min_p, n_drafts);
}

}  // namespace strata::core
