// include/strata/kernels/resident_mask.hpp - STRATA_RESIDENT_MASK=1 (opt-in, changes the output): the MoE router
// may only pick experts that are resident on some GPU.  See src/kernels/cuda/resident_mask.cu.
//
// The mask is the complement of the resident set: an expert no GPU holds (the primary's VRAM cache, a layer split's
// stage caches, the expert servers of --peer-device) is taken out of the top-10 SELECTION, so the next-best resident
// expert is picked in its place (a re-route, not STRATA_ROUTE_TAIL_SKIP's zeroing).  The WEIGHTS keep the router's
// formula on the selected set: the softmax over all 512 unbiased logits, renormalised over the 10 picked.
//
// Unset (or 0) nothing here is called: every call site is behind `requested()`.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::kernels::resident_mask {

/// STRATA_RESIDENT_MASK is set and not 0 (read once).
bool requested();
/// 0: off.  1: every routing of the model is masked (verify windows and the prompt path).  2: the verify windows only
/// (decode, speculative verify, --batch slots, a prompt's short reads through windows) - the chunked prompt path
/// routes as without the mask and keeps streaming the experts no GPU holds from RAM.
int mode();
/// The chunked prompt path masks too (mode 1).
bool prefill_masked();

/// This device's mask table, counters and demand counters, allocated once (512 experts only).  Call OUTSIDE a graph
/// capture, before the first routing kernel is recorded on `device` (Verifier::init, Prefill::init, the program's
/// start-up do).  Until a source is added the table lets every expert through.  False (with `err`): the allocation
/// failed or the geometry is not the 512-expert one.
bool prepare_device(int device, int64_t n_layers, int64_t n_expert, std::string& err);

/// A host residency table ([n_layers * n_expert], an entry >= 0: resident) the mask is the union of: the primary's
/// `host_res` (a layer split's stages share it), each expert server's `res_table()`.  The pointer must stay valid;
/// its entries are read on every refresh.  `what` names it in the start-up line.
void add_source(const int32_t* table, const char* what);

/// The adaptive tier's usage counts ([n_layers * n_expert], +1 per routed entry): a DECODE pick the mask took out is
/// added there as demand, so a masked expert that keeps being asked for heats up and is swapped in.  The demand is
/// weighted and bounded: a pick adds min(1, 10 x its router top-10 weight) - 1 for a pick at or above the uniform
/// share, a few hundredths for a tail pick - times STRATA_RESIDENT_MASK_DEMAND_GAIN (default 1).  The prompt path's
/// picks never count.  STRATA_RESIDENT_MASK_DEMAND=0: nothing is counted (the tier then only moves on what is routed).
/// nullptr: not counted.
void set_demand_sink(std::vector<float>* usage);

/// A prompt read borrows cache slots (their pairs leave the residency tables until the refill): while a loan is out
/// the mask keeps the set it had before, so the prompt is read - and other slots keep decoding - with the same mask
/// as the reply.  `hold_enter/leave` bracket the code that edits the tables; `set_lent` says a loan is out.
void hold_enter();
void hold_leave();
void set_lent(bool lent);

/// Rebuilds the mask from the sources and uploads the layers that changed to every prepared device.  Cheap when
/// nothing moved (one pass over the tables, no CUDA call); called before each verify window is launched.  `force`:
/// no rate limit and no hold (the start-up call).
void refresh(bool force = false);

/// A residency table was just edited and uploaded (the program's own upload hook): refresh now, unless a loan is out.
void residency_changed();

/// The routing site's call, right after the router wrote `ids` / `weights` for `n_tok` tokens of `layer` and before
/// anything reads them: re-picks the tokens whose top-10 holds a masked expert, on the CURRENT device's table.
/// False (with `err`): not 512 x 10, or no table was prepared on this device (the feature never runs partially).
/// `prefill`: the chunked prompt path's call - counted apart by the sensor, never counted as demand.
bool apply(const float* logits, int32_t* ids, float* weights, int64_t layer, int64_t n_expert, int64_t k, int n_tok,
           void* stream, std::string& err, bool prefill = false);

/// The sensor, cumulative since the start and summed over the devices: the verify windows (decode) and the chunked
/// prompt path apart.
struct Part {
    unsigned long long choices = 0;    ///< routing choices seen (10 per token per layer)
    unsigned long long rerouted = 0;   ///< choices the mask replaced (a masked expert was in the router's top-10)
    unsigned long long tokens = 0;     ///< token x layer routings seen
    double masked_mass = 0.0;          ///< sum over those routings of the router's top-10 weight on replaced experts
};
struct Counters {
    Part decode, prefill;
};
bool read(Counters& c);
/// `strata resident-mask: request N, rerouted X of Y choices (Z%), masked weight mass W%` for the DECODE routings since
/// the last report (one per request), then the prompt path's own figures.  Nothing is printed when the feature is off.
void report(std::FILE* f);

/// Routing kernel (one warp per token; `mask_layer` is the layer's 512 entries, >= 0: resident).  `stats` (4 x
/// uint64: choices, rerouted, mass x 2^24, tokens) and `demand_layer` (512 x uint32, in 1/256 of a pick) may be null.
void resident_mask_top10(const float* logits, const int32_t* mask_layer, int32_t* ids, float* weights, int n_tok,
                         unsigned long long* stats, unsigned int* demand_layer, void* stream);

}  // namespace strata::kernels::resident_mask
