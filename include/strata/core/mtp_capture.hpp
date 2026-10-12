// include/strata/core/mtp_capture.hpp - lab (opt-in): the MTP distillation capture.
//
// STRATA_MTP_CAPTURE=<dir> makes the --serve prompt read write, for every prompt position it processes, what a
// draft-layer trainer needs:
//
//   * the main model's FINAL multi-stream residual at that position (hc * n_embd floats, the MTP's `Rin`), as BF16;
//   * the head's top-K token ids and their log-probabilities (K = STRATA_MTP_CAPTURE_TOPK, default 20; log-softmax
//     over the whole vocabulary);
//   * the token at the position and the log-probability of the prompt's own next token (teacher-forced NLL).
//
// Two prompt paths feed it, both with the verifier's own head kernels:
//
//   * the batched prompt path (Prefill::on_chunk): `chunk()` runs the head on the chunk's final residual rows, eight
//     at a time (Verifier::head_rows), so a 4k-token sequence is captured at the prefill's speed (~1700 tok/s on the
//     two local cards) instead of at the teacher-forced windows' (~160 tok/s);
//   * the windows of `read_windows` (the prompt's tail under --short-read, or everything with --short-read <huge>):
//     `window()` takes the window's head logits as they are.
//
// The top-K and the log-sum-exp run on the GPU (kernels/topk_lse.hpp): on the host they were ~2 ms a token, a cap of
// ~500 tokens/s.  The request thread copies the rows to the host; ONE worker thread converts to BF16 and writes,
// behind a bounded queue.  Every GEN/GENI request of the process is one sequence, numbered from 0 (`seq_<k>.*`):
//
//   seq_<k>.tok.i32   int32[n]            the token at each captured position (positions are consecutive from 0)
//   seq_<k>.r.bf16    uint16[n][hcn]      the final residual rows, BF16 (round to nearest even)
//   seq_<k>.top.bin   {int32[K]; f32[K]}  per position: ids by descending probability, then their log-probabilities
//   seq_<k>.nll.f32   f32[n]              log-probability of the prompt's next token (NaN when it is not known)
//   seq_<k>.json      {"n", "hcn", "topk", "n_vocab", "first_pos"}, rewritten after every job
//
// Start the engine with --prompt-cache 0 so every prompt position is read (from 0) in this request; the driver
// (docs/porto/strata_adattivo/mtp_distill/capture_driver.py in the lab repo) does this.  Nothing changes when the
// variable is unset.
#pragma once

#include <cstdint>
#include <string>

namespace strata::core {

class Verifier;

class MtpCapture {
public:
    /// The process's capture, or null when STRATA_MTP_CAPTURE is unset.
    static MtpCapture* instance();
    /// A request starts: its files are `<dir>/seq_<k>.*` (k counts the requests that reached here, from 0).
    void begin_request(int64_t n_prompt);
    /// After a prompt window's Verifier::run (rows t = 0..T-1 at positions pos0 + t): `tokens[t]` is the token at
    /// pos0 + t, `next[t]` the prompt's token at pos0 + t + 1 (or -1).  Reads the final residuals and the head logits.
    bool window(const Verifier& ver, const int32_t* tokens, const int32_t* next, int T, int64_t pos0, std::string& err);
    /// After a batched prompt chunk (Prefill::on_chunk, its stream synchronized): `R_rows` = the chunk's T final
    /// residual rows on the head stage's device; tokens / next as window().  Runs the head on the rows.
    bool chunk(Verifier& ver, const float* R_rows, const int32_t* tokens, const int32_t* next, int T, int64_t pos0,
               std::string& err);
    /// Waits for the worker to write everything queued so far (the request is over).
    void flush();
    int64_t positions() const { return written_; }

private:
    MtpCapture(std::string dir, int topk);
    ~MtpCapture();
    bool rows(const Verifier& ver, const float* R_dev, const float* logits_dev, const int32_t* tokens,
              const int32_t* next, int T, int64_t pos0, std::string& err);
    struct Impl;
    Impl* impl_;
    int64_t written_ = 0;
};

}  // namespace strata::core
