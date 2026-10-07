// include/strata/core/mtp.hpp - plan v0.3 P6: the MTP draft layer on the GPU.
//
// The model ships one multi-token-prediction layer (vLLM 0.30.0 `qwen4_exp` MTP; transcribed and validated in
// `tools/mtp_probe.py`, where it accepts 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3).  Cell i of its
// sequence pairs the main model's final multi-stream residual at position i with the token at position i+1, at
// rope position i (vLLM's convention); its output predicts the token at position i+2, and its own residual feeds
// the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1+w))           h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e  ->  attention hyper-connection  ->  QSA attention (own K/V)  ->  MLP hyper-connection  ->
//   MoE (512 experts, top-10, shared expert)  ->  R += y * 2 sigmoid(inject / hc)  ->  final mixer  ->  head
//
// Runtime choices, each a trade of exactness for simplicity that only affects DRAFT quality, never the output
// (the verify window decides every emitted token):
//   * the large projections run as Q8_0 through the multi-column MMVQ (`tools/mtp_rt.py` quantizes them);
//   * the attention is DENSE over every cell the layer has seen - identical to the model's sparse selection
//     below 2,051 cells - so speculative cells (draft steps, rejected window rows) never touch indexer state and
//     are simply overwritten when their positions are processed again;
//   * all 512 routed experts live in VRAM (708 MB) and run through the grouped hit kernels.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::core {

class NativeHead;

class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    /// Loads `rt_dir` (from tools/mtp_rt.py) and allocates the layer's K/V and buffers for up to `max_t` rows.
    /// Call before the VRAM expert tier is sized: this takes ~0.9 GB.
    bool load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
              int64_t window = 32768, const MtpDrafter* shared = nullptr);
    /// The prompt's length: prefill() skips the cells the attention window can never reach again.
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    /// At most this many drafts per round (below max_t - 1): a window longer than the MTP's comes from elsewhere.
    void set_max_drafts(int k) { max_drafts_ = k; }
    /// --mtp-hnorm stream (opt-in; before the first draft or prefill): pre_fc_norm_hidden normalizes each
    /// hyper-connection stream on its own, as llama.cpp's qwen4exp MTP graph does, instead of one RMS over all four.
    void set_hnorm_per_stream(bool on) { hnorm_stream_ = on; }
    bool hnorm_per_stream() const { return hnorm_stream_; }
    /// --mtp-q4 (opt-in; before load): the draft layer's Q8_0 projections and its draft head run from 4-bit (Q4_0)
    /// copies made at load/bind - fewer bytes per draft step; drafts only, the verify window decides every token.
    /// (The prompt path's batched K/V pass keeps the Q8_0 originals.)
    void set_q4(bool proj, bool head) { q4_ = proj; q4_head_ = head; }
    /// --mtp-draft-vocab FILE (opt-in; before bind): the draft head's token subset from FILE instead of
    /// rt/draft_vocab.bin (e.g. data/draft_vocab_en.bin, 40K tokens)
    void set_draft_vocab(const std::string& path) { dvocab_path_ = path; }
    /// STRATA_MTP_TOP2=1 (diagnostic): draft j's runner-up token in the last draft() (-1 = unknown)
    static bool top2_env();
    int32_t top2(int j) const { return j >= 0 && j < (int) top2_.size() ? top2_[(size_t) j] : -1; }
    int max_t() const { return max_t_; }
    uint64_t vram_bytes() const { return vram_; }
    /// Lab v1: optional private int8 KV pool and a second pre-captured chain graph family. Before load.
    void set_selective_capture(bool on) { selective_config_ = on; }
    /// Lab-only equal-allocation comparison. Configure before load; switch only at an idle GEN boundary.
    void set_hq_pack(const std::string& dir) { hq_pack_ = dir; }
    bool hq_switch(int mode, std::string& err);
    bool hq_report(std::FILE* out, int64_t request, std::string& err) const;
    /// After an idle, completed prefill/validated SSD restore. `full_until` is exclusive and -1 means unproven.
    bool selective_begin_request(uint64_t epoch, int64_t full_until, bool on, std::string& err);
    /// Once per verdict, while the old MTP chain is idle and the verifier's selection is still stable.
    /// Bad metadata selects the exact original dense graphs; CUDA failures return false.
    bool selective_source(const int32_t* ids, int width, int64_t pos, int source_device, uint64_t epoch,
                           std::string& err);
    /// Untimed diagnostic AFTER decode: compare private codes/scales with the authoritative MTP host copy.
    bool selective_check(std::string& err);
    bool selective_report(std::FILE* out, int64_t request, std::string& err) const;
    /// The draft layer's K/V state (read-only: --serve's STRATA_STATE_HASH check hashes it)
    const QsaState& kv_state() const { return st_; }
    /// KV streaming: refill the ring of the drafter's window from its host copy for a sequence that continues at
    /// `upto` (a conversation-cache resume). No-op unless the drafter's K/V is a ring.
    void kv_restore(int64_t upto);
    /// The VRAM bind() will allocate for a native head of `head_row_bytes` per vocabulary row: the draft logits and
    /// the draft head over rt/draft_vocab.bin's subset.  The expert cache is sized before bind(), so it reserves this.
    uint64_t bind_bytes(uint64_t head_row_bytes, int64_t n_vocab, int source_device = -1) const;
    /// The embedding weights must be local to the drafter (or its mapped native embedding). `source_device`
    /// owns the native head and every residual pointer passed to draft/prefill/set_source_R; -1 keeps them local.
    /// CUDA cross-device drafting copies the head subset unchanged and stages residual windows through pinned RAM.
    /// The caller must complete the source producer before passing residuals; the source rows may be reused on return.
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err,
              const MtpDrafter* shared = nullptr, int source_device = -1);

    /// Prompt cells [cell0, cell0 + n): residual rows `R_rows` (device, hc*n_embd each) and `next_tokens` (host,
    /// the token at position cell+1). Runs in batches of up to max_t rows. `sync` false: returns without waiting
    /// for the drafter's stream (the caller orders on it: `stream()`). Cross-device async calls require
    /// prepare_prefill(), n <= max_t and a free pinned staging slot; the source copy itself completes on return.
    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err,
                 bool sync = true);
    /// --pipeline-windows: capture the prefill graphs and size its input records for windows of up to max_t rows now,
    /// so a later prefill(sync = false) of a window neither captures nor allocates (both wait for the device).
    bool prepare_prefill(std::string& err);
    cudaStream_t stream() const { return cs_; }

    /// One round: catch-up over T cells from `p` (rows = the window's final residuals, `tokens` = the window's
    /// argmaxes: row t pairs R_{p+t} with the token at p+t+1), then the draft chain from row `a` (the last
    /// accepted row) for T-1 drafts at cells p+a+1 ...  `drafts` gets T-1 tokens.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    /// The first round: one cell (`cell`) from `R_row` (device) and `token` -> T-1 drafts.
    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                     float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr);

    // Offline neuron-cache probe only: ten distinct roots, nine greedy MTP steps each.
    // Writes speculative MTP cells at/after cell; caller must end the request immediately.
    // Short prefixes ensure branches never overwrite resident prefix KV through ring wrap.
    bool neuron_probe_100(const float* R_row, const int32_t* roots, int64_t cell,
                          std::vector<int32_t>& candidates, double& prediction_ms, std::string& err);

    /// COUPLED DRAFT SAMPLING (core/coupled_draft.hpp; STRATA_SPEC_COUPLED=1, set up by bind()): the request's
    /// sampling.  A sampled request (temperature > 0, not greedy) then drafts by SAMPLING with the target's chain and
    /// the target's Philox draw for the verifying row, and `probs` is the draft's probability under that chain; a
    /// greedy one keeps the argmax drafts and their graphs.  A no-op when the switch is off.
    void set_draft_sampling(const strata::kernels::SamplerParams& sp);
    /// Coupled mode with penalties: the history the next draft() chain starts from - what the session holds after the
    /// window's commit (`tail`) and the window's pick at its last accepted row (`next`, the next window's row 0).
    void set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next);
    void set_ple_session(SessionState* s) { ple_ss_ = s; }
    bool coupled() const { return coupled_active_; }
    /// PROBABILISTIC DRAFT ACCEPTANCE (core/spec_prob.hpp, STRATA_SPEC_PROB=1): this request's drafts are SAMPLED from
    /// q and each draft j's distribution is in spec_q() + j * kSpecQStride (ids, -1 terminated; then float bits),
    /// valid on the host once draft() has returned.  False for greedy requests and where the mode is off.
    bool prob() const { return coupled_active_ && h_q_ != nullptr; }
    const int32_t* spec_q() const { return h_q_; }

    // ---- --pipeline-windows 2: the chain as one asynchronous launch (the round and its steps back to back, no host
    // wait), its first `n_force` steps fed the given tokens instead of their own picks (teacher forcing: the drafts of
    // a window already in flight), each output read once the event after its step has completed.
    /// Before `load`: the round/step graphs get the forcing kernel (a no-op while its token is -1), and the drafter's
    /// stream the highest priority (STRATA_MTP_PRIORITY=0: the default priority).
    void set_force_capture(bool on) { force_on_ = on; }
    /// The source-device rows the next round reads (null: the bound `window_R`), staged before the round.
    void set_source_R(const float* rows) { src_R_ = rows; }
    /// Capture the round graphs for T = 1..max_t and every step graph now (a capture syncs the drafter's stream).
    bool prepare_chain(std::string& err);
    /// The catch-up over T cells from `p` (as draft()), then `n_out` outputs from row `a`: output j is the chain's
    /// own pick, and step j (1 <= j <= n_force) takes force[j - 1] as its input token instead of output j - 1.
    /// `n_early`: an event after each of the first `n_early` outputs (chain_outputs_ready).
    bool chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force, int n_out,
                      int n_early, std::string& err);
    /// 1: every output is in chain_tok()/chain_prob(); 0: still running; -1: an error.
    int chain_poll(std::string& err);
    /// How many of the chain's first outputs (up to `n_early`; all of them once the chain is done) have landed in
    /// chain_tok()/chain_prob(); -1: an error.  A window's size is often decided by its first outputs (a draft below
    /// spec_min_p ends it), so the next launch need not wait for all of them.
    int chain_outputs_ready(std::string& err);
    bool chain_live() const { return chain_live_; }
    const int32_t* chain_tok() const { return chain_tok_; }
    const float* chain_prob() const { return chain_prob_; }

    double ms_draft = 0, ms_prefill = 0;
    int64_t rounds = 0;

    /// E-9: the prompt path computes this layer's prompt K/V in batches (Prefill::draft_kv): its tensors, its K/V
    /// state, the first cell a round can still read, its device, and a wait for its own stream.
    const float* tensor_f32(const char* name) const { return f32(name); }
    const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
    const void* tensor_q8(const char* name) const { return q8(name); }
    QsaState& kv_state_rw() { return st_; }
    int64_t first_needed() const {
        return (window_ > 0 && prompt_len_ > 0 && !full_prefix_) ? prompt_len_ - window_ - 64 : 0;
    }
    /// this prompt's root goes to the prefix file: K/V for every cell, also those the window will not reach again
    /// (a saved prefix must serve later prompts of any length)
    void set_full_prefix(bool on) { full_prefix_ = on; }
    int device() const { return device_; }
    bool idle(std::string& err);

private:
    static constexpr int kSelectiveRecent = 8192, kSelectiveProxy = 2051, kSelectiveCap = 10304;
    static constexpr int kSelectiveSlots = 3072;
    bool selective_allocate(std::string& err);
    bool selective_stage(int T, int64_t p, int a, int n_out, std::string& err);
    void selective_fallback(int reason);
    bool selective_config_ = false, selective_on_ = false, selective_record_ = false;
    bool selective_active_ = false, selective_source_ok_ = false, selective_reset_ = true;
    bool selective_supported_ = false, selective_last_valid_ = false;
    uint64_t selective_epoch_ = 0, selective_bytes_ = 0;
    int64_t selective_full_until_ = -1, selective_source_pos_ = -1;
    int selective_proxy_width_ = 0, selective_last_n_ = 0, selective_source_reason_ = 3;
    int64_t selective_last_pos_ = -1;
    void* selective_arena_ = nullptr;
    strata::kernels::QsaAttnPools selective_pools_;
    strata::kernels::KvStreamMap selective_map_;
    int32_t *selective_h_ids_ = nullptr, *selective_m_ids_ = nullptr, *selective_d_ids_ = nullptr;
    int32_t *selective_h_proxy_ = nullptr, *selective_m_proxy_ = nullptr;
    int32_t selective_widths_[8] = {};
    cudaGraphExec_t selective_round_[9] = {}, selective_step_[9] = {};
    int64_t selective_chains_ = 0, selective_dense_ = 0, selective_resets_ = 0, selective_ids_ = 0;
    int64_t selective_far8192_ = 0, selective_far32768_ = 0, selective_max_union_ = 0;
    int64_t selective_mirror_cells_ = 0, selective_check_cells_ = 0, selective_check_distant_ = 0;
    int64_t selective_fallbacks_[8] = {};
    double selective_source_ms_ = 0;
    bool prefill_impl(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0,
                      std::string& err, bool sync, cudaMemcpyKind residual_kind);
    bool cross_device() const { return source_device_ >= 0 && source_device_ != device_; }
    bool acquire_source_slot(bool wait, int& slot, std::string& err);
    bool read_source_rows(float* host, const float* rows, int64_t n, std::string& err);
    bool finish_source_slot(int slot, std::string& err);
    bool stage_remote_rows(float* local, const float* rows, int n, std::string& err);
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    /// The layer's front for T rows at step rows [row0, +T): the embedding, the fc projections, the attention
    /// hyper-connection read (R_, inj_, mixed_) and the K/V appended.
    bool record_front(int T, int row0, cudaStream_t cs, std::string& err);
    /// The rest for row 0 at step row `step_row`: the attention from the q projection, the MLP, the final mixer,
    /// the head, the draft and its probability.  Needs row 0's R_, inj_ and mixed_ (its q8_1 rows in xq_).
    bool record_rest(int step_row, cudaStream_t cs, std::string& err);
    void norm_rope(float* data, const float* gamma, int rows, int cols, const int32_t* p, cudaStream_t cs);
    bool capture_prefill(int T, std::string& err);
    bool capture_prefill_dev(int T, std::string& err);   ///< E-4: without the mapped staging (inputs copied on device)
    bool capture_round(int T, bool coupled, std::string& err);
    bool capture_step(int j, bool coupled, std::string& err);
    cudaGraphExec_t step_exec_[9] = {};
    // coupled draft sampling: its own round/step graphs (the argmax ones stay as they were), the request's
    // parameters and the penalty ring (mapped staging + device copies), the split scratch, token id -> subset index
    bool setup_coupled(std::string& err);
    cudaGraphExec_t round_exec_c_[9] = {};
    cudaGraphExec_t step_exec_c_[9] = {};
    bool coupled_ok_ = false, coupled_active_ = false;
    bool coupled_rec_ = false;   ///< record_forward: the full layer ends in the coupled sampler (draft coupled_j_)
    int coupled_j_ = 0;
    strata::kernels::SamplerParams *h_cparams_ = nullptr, *m_cparams_ = nullptr, *cparams_ = nullptr;
    int32_t *h_chist_ = nullptr, *m_chist_ = nullptr, *cring_ = nullptr, *dinv_ = nullptr;
    int32_t *h_q_ = nullptr, *m_q_ = nullptr;   ///< STRATA_SPEC_PROB: the drafts' q lists (mapped, max_t_ rows)
    void* cscratch_ = nullptr;
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    SessionState* ple_ss_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const NativeHead* head_ = nullptr;
    const float* window_R_ = nullptr;
    int max_t_ = 0;
    int device_ = -1;   ///< the device `load` ran on: the public calls switch to it (layer split)
    int source_device_ = -1;   ///< head and residual input owner, fixed by the first bind
    const float* bound_source_R_ = nullptr;
    float* local_window_R_ = nullptr;   ///< stable graph input when the residual source is another device
    cudaStream_t source_cs_ = nullptr; ///< D2H only, on source_device_; each source copy completes before return
    struct SourceSlot {
        float* host = nullptr;
        cudaEvent_t consumed = nullptr;   ///< on device_, after the last H2D read of host
        bool live = false;
    };
    SourceSlot source_slots_[2];
    float* prefill_host_R_ = nullptr;   ///< large prompt chunks, reused only by synchronous prefill
    int64_t prefill_host_cap_ = 0;
    void* private_gr_arena_ = nullptr;
    strata::kernels::GrWorkspace private_gr_;
    bool source_already_staged_ = false;   ///< draft_first filled local_window_R_ itself
    int max_drafts_ = 1 << 30;
    bool hnorm_stream_ = false;
    bool q4_ = false, q4_head_ = false;
    uint8_t* dense4_ = nullptr;              ///< --mtp-q4: Q4_0 copies of the Q8_0 tensors
    std::vector<std::pair<std::string, uint64_t>> q4_off_;
    int dhead_type_ = -1;                    ///< the draft head subset's ggml type (the main head's, or Q4_0)
    std::string dvocab_path_;
    std::string vocab_file() const { return dvocab_path_.empty() ? rt_dir_ + "/draft_vocab.bin" : dvocab_path_; }
    /// a projection's weights for the draft layer's own pass: the Q4_0 copy under --mtp-q4, else the Q8_0 original
    const void* wq(const char* name, int& type) const;
    bool hq_validate(const ModelGeometry& g, const MtpDrafter* shared, std::string& err);
    std::string hq_pack_, hq_sha_[3];
    int hq_mode_ = 0, hq_record_mode_ = 0;
    uint64_t hq_upload_bytes_ = 0;
    double hq_upload_ms_ = 0.0;
    int64_t hq_chains_[3] = {};
    cudaGraphExec_t hq_round_[2][9] = {}, hq_step_[2][9] = {};
    uint8_t* hq_xq_ = nullptr; // private: shared_expert concurrently consumes xq_
    void* hq_scratch_ = nullptr; // both carved from the ordinary arena before expert-cache sizing
    bool make_q4_dense(const std::vector<uint8_t>& blob, std::string& err);
    bool make_q4_head(std::string& err);
    void record_top2(int j);
    std::vector<int32_t> top2_, dvocab_host_;
    std::vector<float> lg_host_;
    // --pipeline-windows 2 (chain_launch)
    bool stage_source_R(int T, std::string& err);   ///< set_source_R's rows into the bound buffer
    bool force_on_ = false, chain_live_ = false;
    const float* src_R_ = nullptr;
    int32_t *h_force_ = nullptr, *m_force_ = nullptr;   ///< the forced tokens (mapped), -1 = the step's own pick
    cudaEvent_t ev_chain_ = nullptr;
    cudaEvent_t ev_step_[8] = {};                       ///< after each of the first n_early outputs
    int steps_seen_ = 0, chain_n_ = 0, chain_early_ = 0;
    int32_t chain_tok_[8] = {};
    float chain_prob_[8] = {};
    int64_t n_vocab_ = 0;
    uint64_t vram_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaGraphExec_t prefill_exec_[9] = {};
    cudaGraphExec_t prefill_dev_exec_[9] = {};
    int32_t* pf_dev_ = nullptr;   ///< E-4: a prompt's rows' token / step / position records, uploaded at once
    int64_t pf_cap_ = 0;          ///< its capacity in ints
    cudaStream_t side_ = nullptr;                    // the shared expert's branch of the draft graphs (CUDA)
    cudaEvent_t sh_fork_ = nullptr, sh_join_ = nullptr;
    cudaGraphExec_t round_exec_[9] = {};

    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    std::vector<Tensor> tensors_;
    uint8_t* dense_ = nullptr;
    uint8_t* experts_ = nullptr;
    // Slot drafters borrow immutable weights and head; each still owns its state and scratch.
    bool owns_weights_ = true;
    bool owns_draft_head_ = true;
    void* state_arena_ = nullptr;
    QsaState st_;
    void* arena_ = nullptr;

    // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
    // the drafts out
    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
    // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
    uint8_t* dhead_ = nullptr;
    int32_t* dvocab_ = nullptr;
    int64_t n_dvocab_ = 0;
    std::string rt_dir_;
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t prompt_len_ = 0;
    bool full_prefix_ = false;
    float* probs_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' and row_top_prob_split's partials and counters
    uint8_t* top_scratch_ = nullptr;
    // device
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
    float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
    float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
    float* xn_ = nullptr;
    uint8_t* xq_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr;
    float* attn_scratch_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr, *out_ids_ = nullptr;
    uint8_t* hit_xq_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *grp_counts_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float* sh_scratch_ = nullptr;
    uint16_t* x_bf16_ = nullptr;
    float* head_logits_ = nullptr;
    float* dummy_inj_ = nullptr;
    int64_t cap_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
