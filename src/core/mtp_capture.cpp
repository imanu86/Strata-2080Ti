// src/core/mtp_capture.cpp - see include/strata/core/mtp_capture.hpp.
#include "strata/core/mtp_capture.hpp"
#include "strata/core/verify.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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
    std::vector<int32_t> tok, nxt;
    std::vector<float> R;       // T * hcn
    std::vector<float> logits;  // T * n_vocab
};

}  // namespace

struct MtpCapture::Impl {
    std::string dir;
    int topk = 20;
    int64_t hcn = 0, n_vocab = 0;
    // the queue
    std::mutex mu;
    std::condition_variable cv_push, cv_pop;
    std::deque<std::unique_ptr<Job>> q;
    size_t max_queue = 3;
    bool stop = false;
    int64_t queued = 0, done = 0;
    std::thread worker;
    std::string worker_err;
    // the worker's files
    std::FILE *f_tok = nullptr, *f_r = nullptr, *f_top = nullptr, *f_nll = nullptr;
    int64_t cur_k = -1, cur_n = 0, cur_first = -1;
    std::vector<uint16_t> bf;
    std::vector<int32_t> ids;
    std::vector<float> lps;

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
        ids.resize((size_t) K);
        lps.resize((size_t) K);
        bf.resize((size_t) hcn);
        for (int t = 0; t < j.T; ++t) {
            const float* row = j.logits.data() + (size_t) t * (size_t) n_vocab;
            // the K largest: a min-heap of K (ids, value) pairs; ties keep the lower id first (stable enough for a trainer)
            struct E { float v; int32_t i; };
            std::vector<E> heap;
            heap.reserve((size_t) K);
            auto cmp = [](const E& a, const E& b) { return a.v > b.v || (a.v == b.v && a.i < b.i); };   // min at front
            for (int64_t v = 0; v < n_vocab; ++v) {
                const float x = row[v];
                if ((int) heap.size() < K) {
                    heap.push_back({x, (int32_t) v});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (x > heap.front().v) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {x, (int32_t) v};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
            std::sort_heap(heap.begin(), heap.end(), cmp);   // ascending by cmp = descending by value
            std::reverse(heap.begin(), heap.end());
            const double maxv = heap.empty() ? 0.0 : (double) heap.front().v;
            double sum = 0.0;
            for (int64_t v = 0; v < n_vocab; ++v) sum += std::exp((double) row[v] - maxv);
            const double lse = maxv + std::log(sum);
            for (int i = 0; i < K; ++i) {
                ids[(size_t) i] = i < (int) heap.size() ? heap[(size_t) i].i : -1;
                lps[(size_t) i] = i < (int) heap.size() ? (float) ((double) heap[(size_t) i].v - lse) : -INFINITY;
            }
            const int32_t nx = j.nxt[(size_t) t];
            const float nll = nx >= 0 && nx < n_vocab ? (float) ((double) row[nx] - lse) : NAN;
            const float* R = j.R.data() + (size_t) t * (size_t) hcn;
            for (int64_t i = 0; i < hcn; ++i) bf[(size_t) i] = f32_to_bf16_rne(R[i]);
            std::fwrite(&j.tok[(size_t) t], sizeof(int32_t), 1, f_tok);
            std::fwrite(bf.data(), sizeof(uint16_t), (size_t) hcn, f_r);
            std::fwrite(ids.data(), sizeof(int32_t), (size_t) K, f_top);
            std::fwrite(lps.data(), sizeof(float), (size_t) K, f_top);
            std::fwrite(&nll, sizeof(float), 1, f_nll);
            ++cur_n;
        }
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
    delete impl_;
}

MtpCapture* MtpCapture::instance() {
    static MtpCapture* cap = [] () -> MtpCapture* {
        const char* d = std::getenv("STRATA_MTP_CAPTURE");
        if (d == nullptr || d[0] == '\0') return nullptr;
        const char* k = std::getenv("STRATA_MTP_CAPTURE_TOPK");
        const int topk = k != nullptr ? std::max(1, std::min(256, std::atoi(k))) : 20;
        static MtpCapture holder(d, topk);   // destroyed at exit: the worker is joined and the files closed
        std::fprintf(stderr, "strata mtp capture: on, %s, top-%d (the prompt read's windows write the final residuals "
                             "and the head's top-k; opt-in lab hook)\n", d, topk);
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

bool MtpCapture::window(const Verifier& ver, const int32_t* tokens, const int32_t* next, int T, int64_t pos0,
                        std::string& err) {
    if (T <= 0) return true;
    if (impl_->n_vocab == 0) impl_->n_vocab = ver.vocab();
    if (impl_->hcn == 0) impl_->hcn = ver.hc_n_embd();
    const int64_t nv = impl_->n_vocab, hcn = impl_->hcn;
    if (nv <= 0 || hcn <= 0) { err = "mtp capture: the verifier has no head or no residual rows"; return false; }
    auto j = std::make_unique<Job>();
    j->pos0 = pos0;
    j->T = T;
    j->tok.assign(tokens, tokens + T);
    j->nxt.assign(next, next + T);
    j->R.resize((size_t) T * (size_t) hcn);
    j->logits.resize((size_t) T * (size_t) nv);
    // run() synchronized the verifier's stream: the final residuals and the head logits of this window are complete
    if (cudaMemcpy(j->R.data(), ver.final_R_all(), j->R.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = "mtp capture: the final residual copy failed";
        return false;
    }
    for (int t = 0; t < T; ++t)
        if (!ver.copy_logits(t, j->logits.data() + (size_t) t * (size_t) nv)) {
            err = "mtp capture: the head logits copy failed";
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

void MtpCapture::flush() { impl_->drain(); }

}  // namespace strata::core
