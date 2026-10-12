// src/core/mtp_capture.cpp - see include/strata/core/mtp_capture.hpp.
#include "strata/core/mtp_capture.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/verify.hpp"
#include "strata/kernels/topk_lse.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core {
namespace {

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t) ((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));   // inf / nan
    const uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t) ((u + 0x7fffu + lsb) >> 16);
}

struct Job {
    enum Kind { Open, Rows } kind = Rows;
    int64_t k = 0;            // Open: the request number
    int64_t n_prompt = 0;     // Open
    int64_t pos0 = 0;         // Rows
    int T = 0;                // Rows
    std::vector<int32_t> tok;
    std::vector<float> R;         // T * hcn (f32, converted by the worker)
    std::vector<int32_t> top_ids; // T * K
    std::vector<float> top_lp;    // T * K
    std::vector<float> nll;       // T
};

/// Device scratch for the top-k of up to `rows` rows (on the head stage's device).
struct DevScratch {
    int device = -1;
    int rows = 0, K = 0;
    int32_t* next = nullptr;
    int32_t* ids = nullptr;
    float* lps = nullptr;
    float* nll = nullptr;
    bool ensure(int dev, int n_rows, int topk, std::string& err) {
        if (device == dev && n_rows <= rows && K == topk) return true;
        release();
        const OnDevice on(dev);
        device = dev; rows = std::max(n_rows, 8); K = topk;
        if (cudaMalloc((void**) &next, (size_t) rows * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &ids, (size_t) rows * K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &lps, (size_t) rows * K * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &nll, (size_t) rows * sizeof(float)) != cudaSuccess) {
            err = "mtp capture: the device scratch could not be allocated";
            release();
            return false;
        }
        return true;
    }
    void release() {
        if (device < 0) return;
        const OnDevice on(device);
        if (next) cudaFree(next);
        if (ids) cudaFree(ids);
        if (lps) cudaFree(lps);
        if (nll) cudaFree(nll);
        next = nullptr; ids = nullptr; lps = nullptr; nll = nullptr;
        device = -1; rows = 0;
    }
};

}  // namespace

struct MtpCapture::Impl {
    std::string dir;
    int topk = 20;
    int64_t hcn = 0, n_vocab = 0;
    DevScratch scratch;
    // the queue
    std::mutex mu;
    std::condition_variable cv_push, cv_pop;
    std::deque<std::unique_ptr<Job>> q;
    size_t max_queue = 4;
    bool stop = false;
    int64_t queued = 0, done = 0;
    std::thread worker;
    std::string worker_err;
    // the worker's files
    std::FILE *f_tok = nullptr, *f_r = nullptr, *f_top = nullptr, *f_nll = nullptr;
    int64_t cur_k = -1, cur_n = 0, cur_first = -1;
    std::vector<uint16_t> bf;

    void close_files() {
        if (f_tok) { std::fclose(f_tok); f_tok = nullptr; }
        if (f_r) { std::fclose(f_r); f_r = nullptr; }
        if (f_top) { std::fclose(f_top); f_top = nullptr; }
        if (f_nll) { std::fclose(f_nll); f_nll = nullptr; }
    }
    std::string base() const { return dir + "/seq_" + std::to_string(cur_k); }
    void write_meta() {
        if (cur_k < 0) return;
        std::FILE* f = std::fopen((base() + ".json").c_str(), "wb");
        if (!f) return;
        std::fprintf(f, "{\"n\": %lld, \"hcn\": %lld, \"topk\": %d, \"n_vocab\": %lld, \"first_pos\": %lld}\n",
                     (long long) cur_n, (long long) hcn, topk, (long long) n_vocab, (long long) cur_first);
        std::fclose(f);
    }
    void open(int64_t k) {
        close_files();
        if (cur_k >= 0) write_meta();
        cur_k = k;
        cur_n = 0;
        cur_first = -1;
        f_tok = std::fopen((base() + ".tok.i32").c_str(), "wb");
        f_r = std::fopen((base() + ".r.bf16").c_str(), "wb");
        f_top = std::fopen((base() + ".top.bin").c_str(), "wb");
        f_nll = std::fopen((base() + ".nll.f32").c_str(), "wb");
        if (!f_tok || !f_r || !f_top || !f_nll) worker_err = "mtp capture: cannot open " + base() + ".*";
    }
    void rows(const Job& j) {
        if (!f_tok || !worker_err.empty()) return;
        if (cur_first < 0) cur_first = j.pos0;
        const int K = topk;
        bf.resize((size_t) j.T * (size_t) hcn);
        for (size_t i = 0; i < bf.size(); ++i) bf[i] = f32_to_bf16_rne(j.R[i]);
        std::fwrite(j.tok.data(), sizeof(int32_t), (size_t) j.T, f_tok);
        std::fwrite(bf.data(), sizeof(uint16_t), bf.size(), f_r);
        for (int t = 0; t < j.T; ++t) {
            std::fwrite(j.top_ids.data() + (size_t) t * K, sizeof(int32_t), (size_t) K, f_top);
            std::fwrite(j.top_lp.data() + (size_t) t * K, sizeof(float), (size_t) K, f_top);
        }
        std::fwrite(j.nll.data(), sizeof(float), (size_t) j.T, f_nll);
        cur_n += j.T;
        std::fflush(f_tok); std::fflush(f_r); std::fflush(f_top); std::fflush(f_nll);
        write_meta();
    }
    void run() {
        for (;;) {
            std::unique_ptr<Job> j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv_pop.wait(lk, [&] { return stop || !q.empty(); });
                if (q.empty()) return;
                j = std::move(q.front());
                q.pop_front();
            }
            if (j->kind == Job::Open) open(j->k);
            else rows(*j);
            {
                std::lock_guard<std::mutex> lk(mu);
                ++done;
            }
            cv_push.notify_all();
        }
    }
    void push(std::unique_ptr<Job> j) {
        std::unique_lock<std::mutex> lk(mu);
        cv_push.wait(lk, [&] { return q.size() < max_queue; });
        q.push_back(std::move(j));
        ++queued;
        cv_pop.notify_one();
    }
    void drain() {
        std::unique_lock<std::mutex> lk(mu);
        cv_push.wait(lk, [&] { return done >= queued; });
    }
};

MtpCapture::MtpCapture(std::string dir, int topk) : impl_(new Impl) {
    impl_->dir = std::move(dir);
    impl_->topk = topk;
    impl_->worker = std::thread([this] { impl_->run(); });
}

MtpCapture::~MtpCapture() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->stop = true;
    }
    impl_->cv_pop.notify_all();
    if (impl_->worker.joinable()) impl_->worker.join();
    impl_->close_files();
    impl_->write_meta();
    impl_->scratch.release();
    delete impl_;
}

MtpCapture* MtpCapture::instance() {
    static MtpCapture* cap = [] () -> MtpCapture* {
        const char* d = std::getenv("STRATA_MTP_CAPTURE");
        if (d == nullptr || d[0] == '\0') return nullptr;
        const char* k = std::getenv("STRATA_MTP_CAPTURE_TOPK");
        const int topk = k != nullptr ? std::max(1, std::min(32, std::atoi(k))) : 20;
        static MtpCapture holder(d, topk);   // destroyed at exit: the worker is joined and the files closed
        std::fprintf(stderr, "strata mtp capture: on, %s, top-%d (the prompt read's chunks and windows write the final "
                             "residuals and the head's top-k; opt-in lab hook)\n", d, topk);
        return &holder;
    }();
    return cap;
}

void MtpCapture::begin_request(int64_t n_prompt) {
    static int64_t req = 0;
    auto j = std::make_unique<Job>();
    j->kind = Job::Open;
    j->k = req++;
    j->n_prompt = n_prompt;
    impl_->push(std::move(j));
}

// T rows whose final residuals are at R_dev and whose head logits are at logits_dev (both on the head stage's
// device, complete on head_stream()): the top-k on the device, then one job for the worker.
bool MtpCapture::rows(const Verifier& ver, const float* R_dev, const float* logits_dev, const int32_t* tokens,
                      const int32_t* next, int T, int64_t pos0, std::string& err) {
    if (T <= 0) return true;
    if (impl_->n_vocab == 0) impl_->n_vocab = ver.vocab();
    if (impl_->hcn == 0) impl_->hcn = ver.hc_n_embd();
    const int64_t nv = impl_->n_vocab, hcn = impl_->hcn;
    const int K = impl_->topk;
    if (nv <= 0 || hcn <= 0) { err = "mtp capture: the verifier has no head or no residual rows"; return false; }
    const int dev = ver.head_device();
    const OnDevice on(dev);
    DevScratch& s = impl_->scratch;
    if (!s.ensure(dev, T, K, err)) return false;
    cudaStream_t cs = ver.head_stream();
    auto j = std::make_unique<Job>();
    j->pos0 = pos0;
    j->T = T;
    j->tok.assign(tokens, tokens + T);
    j->R.resize((size_t) T * (size_t) hcn);
    j->top_ids.resize((size_t) T * K);
    j->top_lp.resize((size_t) T * K);
    j->nll.resize((size_t) T);
    try {
        if (cudaMemcpyAsync(s.next, next, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, cs) != cudaSuccess)
            throw std::runtime_error("the next-token copy failed");
        strata::kernels::row_topk_lse(logits_dev, T, (int) nv, K, s.next, s.ids, s.lps, s.nll, cs);
        if (cudaMemcpyAsync(j->top_ids.data(), s.ids, j->top_ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
            cudaMemcpyAsync(j->top_lp.data(), s.lps, j->top_lp.size() * sizeof(float), cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
            cudaMemcpyAsync(j->nll.data(), s.nll, j->nll.size() * sizeof(float), cudaMemcpyDeviceToHost, cs) != cudaSuccess ||
            cudaMemcpyAsync(j->R.data(), R_dev, j->R.size() * sizeof(float), cudaMemcpyDeviceToHost, cs) != cudaSuccess)
            throw std::runtime_error("a copy to the host failed");
        if (cudaStreamSynchronize(cs) != cudaSuccess)
            throw std::runtime_error(std::string("the stream failed: ") + cudaGetErrorString(cudaGetLastError()));
    } catch (const std::exception& e) {
        err = std::string("mtp capture: ") + e.what();
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        if (!impl_->worker_err.empty()) { err = impl_->worker_err; return false; }
    }
    written_ += T;
    impl_->push(std::move(j));
    return true;
}

bool MtpCapture::window(const Verifier& ver, const int32_t* tokens, const int32_t* next, int T, int64_t pos0,
                        std::string& err) {
    // run() synchronized the verifier's stream: the final residuals and the head logits of this window are complete
    return rows(ver, ver.final_R_all(), ver.head_logits(), tokens, next, T, pos0, err);
}

bool MtpCapture::chunk(Verifier& ver, const float* R_rows, const int32_t* tokens, const int32_t* next, int T,
                       int64_t pos0, std::string& err) {
    if (T <= 0) return true;
    const int64_t hcn = ver.hc_n_embd();
    const int step = std::min(ver.max_t(), 8);
    if (step < 1 || hcn <= 0) { err = "mtp capture: the verifier is not initialized"; return false; }
    for (int done = 0; done < T; done += step) {
        const int n = std::min(step, T - done);
        const float* R = R_rows + (size_t) done * (size_t) hcn;
        if (!ver.head_rows(R, n, err)) return false;
        if (!rows(ver, R, ver.head_logits(), tokens + done, next + done, n, pos0 + done, err)) return false;
    }
    return true;
}

void MtpCapture::flush() { impl_->drain(); }

}  // namespace strata::core
