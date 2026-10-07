// include/strata/core/prefill_seed.hpp - the VRAM expert tier seeded from the prompt's routing mass (docs/PREFILL_SEED.md).
//
// At the end of a prefill the prompt has told us, for every (layer, expert), how often the router chose it and with
// what total gate weight (the MASS).  Before the decode starts, the experts the prompt leaned on are swapped into
// the layer's own cache in place of the resident ones the prompt barely touched, so the adaptive tier does not have
// to discover them one round at a time.  This header holds what needs no CUDA: the configuration (the environment,
// then per request the `lab=` key) and the swap plan, a pure function of the mass table and the residency table.
//
// Ablation on the DeepSeek port this comes from: ranking by mass beat ranking by count (+7.3% vs -9.2%); the plan
// ranks by mass, ties by (layer, expert) ascending.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace strata::core {

struct PrefillSeedCfg {
    bool armed = false;         ///< STRATA_PREFILL_SEED is set at all: the buffers exist, a request may switch the seed
    bool enabled = false;       ///< STRATA_PREFILL_SEED=1 (lab=PREFILL_SEED:1): the seed runs after the prompt
    int swaps = 512;            ///< STRATA_PREFILL_SEED_SWAPS: swaps per request over all layers (0 = no cap)
    int floor = 0;              ///< STRATA_PREFILL_SEED_FLOOR: swaps every layer is granted before the global cap
    float gain = 1.5f;          ///< STRATA_PREFILL_SEED_GAIN: a candidate enters when its mass >= the victim's x gain
    int64_t min_tokens = 256;   ///< STRATA_PREFILL_SEED_MIN_TOKENS: the seed runs only past this many NEW prompt tokens
    float prior = 32.0f;        ///< STRATA_PREFILL_SEED_PRIOR: the usage prior's horizon in tokens (0 = leave usage)
    bool carry = true;          ///< STRATA_PREFILL_SEED_CARRY: add the live conversation's earlier prompts' mass
};

inline const char* prefill_seed_env(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? v : nullptr;
}

/// The defaults: STRATA_PREFILL_SEED (unset: nothing is armed and nothing below runs) and its companions.
inline PrefillSeedCfg prefill_seed_from_env() {
    PrefillSeedCfg c;
    const char* on = prefill_seed_env("STRATA_PREFILL_SEED");
    if (on == nullptr) return c;
    c.armed = true;
    c.enabled = std::atoi(on) != 0;
    if (const char* v = prefill_seed_env("STRATA_PREFILL_SEED_SWAPS")) c.swaps = std::max(0, std::atoi(v));
    if (const char* v = prefill_seed_env("STRATA_PREFILL_SEED_FLOOR")) c.floor = std::max(0, std::atoi(v));
    if (const char* v = prefill_seed_env("STRATA_PREFILL_SEED_GAIN")) c.gain = std::max(1.0f, (float) std::atof(v));
    if (const char* v = prefill_seed_env("STRATA_PREFILL_SEED_MIN_TOKENS")) c.min_tokens = std::max(0ll, (long long) std::atoll(v));
    if (const char* v = prefill_seed_env("STRATA_PREFILL_SEED_PRIOR")) c.prior = std::max(0.0f, (float) std::atof(v));
    if (const char* v = prefill_seed_env("STRATA_PREFILL_SEED_CARRY")) c.carry = std::atoi(v) != 0;
    return c;
}

/// The request's `lab=NAME:VALUE,NAME:VALUE,...` key applied over `out` (names case-insensitive, with or without the
/// STRATA_ prefix).  Only the PREFILL_SEED levers are this header's: any other name is left alone for whoever owns
/// it (the lab plan's own levers share the key), so the serve loop can hand the same spec to several parsers.
/// False, with `err`, only on a malformed item; `out` then holds the items before it.
inline bool prefill_seed_lab_apply(const std::string& spec, PrefillSeedCfg& out, std::string& err) {
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string item = spec.substr(i, j - i);
        i = j + 1;
        if (item.empty()) continue;
        const size_t c = item.find(':');
        if (c == std::string::npos || c == 0 || c + 1 >= item.size()) { err = "lab: '" + item + "' is not NAME:VALUE"; return false; }
        std::string name = item.substr(0, c);
        for (char& ch : name) ch = (char) std::toupper((unsigned char) ch);
        if (name.rfind("STRATA_", 0) == 0) name = name.substr(7);
        if (name.rfind("PREFILL_SEED", 0) != 0) continue;   // another lab's lever
        if (!out.armed) { err = "lab: " + name + " needs STRATA_PREFILL_SEED set at start (the mass buffers)"; return false; }
        char* end = nullptr;
        const double v = std::strtod(item.c_str() + c + 1, &end);
        if (end == nullptr || *end != '\0') { err = "lab: '" + item + "' has no numeric value"; return false; }
        if (name == "PREFILL_SEED") out.enabled = v != 0.0;
        else if (name == "PREFILL_SEED_SWAPS") out.swaps = std::max(0, (int) v);
        else if (name == "PREFILL_SEED_FLOOR") out.floor = std::max(0, (int) v);
        else if (name == "PREFILL_SEED_GAIN") out.gain = std::max(1.0f, (float) v);
        else if (name == "PREFILL_SEED_MIN_TOKENS") out.min_tokens = std::max<int64_t>(0, (int64_t) v);
        else if (name == "PREFILL_SEED_PRIOR") out.prior = std::max(0.0f, (float) v);
        else if (name == "PREFILL_SEED_CARRY") out.carry = v != 0.0;
        else { err = "lab: unknown lever '" + name + "'"; return false; }
    }
    return true;
}

inline std::string prefill_seed_describe(const PrefillSeedCfg& c) {
    return std::string("PREFILL_SEED=") + (c.enabled ? "1" : "0") + " SWAPS=" + std::to_string(c.swaps) +
           " FLOOR=" + std::to_string(c.floor) + " GAIN=" + std::to_string(c.gain) +
           " MIN_TOKENS=" + std::to_string(c.min_tokens) + " PRIOR=" + std::to_string(c.prior) +
           " CARRY=" + (c.carry ? "1" : "0");
}

/// One swap of the plan: `in` (not resident, mass `gain` above the victim's) takes `out`'s slot in `layer`.
struct PrefillSeedSwap {
    float gain;
    int32_t layer, in, out;
};

struct PrefillSeedPlan {
    std::vector<PrefillSeedSwap> swaps;   ///< best gain first (ties: layer, then `in`, ascending)
    double mass_total = 0.0;              ///< the prompt's whole mass
    double mass_resident = 0.0;           ///< the mass of the experts resident before the plan
    double mass_after = 0.0;              ///< ... and after every swap of the plan
    int64_t experts_seen = 0;             ///< (layer, expert) pairs the prompt routed to at all
    int64_t candidates = 0;               ///< pairs that could have entered (not resident, admitted, mass > 0)
};

/// The plan.  `mass` and `host_res` are [n_layers x n_expert] (slot or -1).  `may_evict(l, e)` says whether a
/// RESIDENT expert may be a victim (the adaptive tier's rule: the elastic core stays, a later stage's layers ...);
/// `may_enter(l, e)` whether a NON-resident one may come in (not held by the peer, a remote or a helper tier).
///
/// Per layer the non-resident experts are ranked by mass (ties: lower id first) and the eligible residents by mass
/// (lowest first; ties: lower id first); the i-th candidate takes the i-th victim's slot while its mass is at least
/// `gain` times the victim's and strictly above it, so a candidate the prompt never picked never enters.  Then the
/// budget: every layer keeps its best `floor` swaps; the rest compete globally by gain until `swaps` are chosen
/// (0 = all).  Swaps stay inside their layer: the layer's own cache, on the card that owns it.
template <class MayEvict, class MayEnter>
PrefillSeedPlan prefill_seed_plan(const float* mass, const int32_t* host_res, int64_t n_layers, int64_t n_expert,
                                  const PrefillSeedCfg& cfg, MayEvict may_evict, MayEnter may_enter) {
    PrefillSeedPlan plan;
    std::vector<std::vector<PrefillSeedSwap>> per_layer((size_t) n_layers);
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers; ++l) {
        cand.clear();
        vict.clear();
        const float* m = mass + l * n_expert;
        const int32_t* r = host_res + l * n_expert;
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            plan.mass_total += (double) m[e];
            plan.experts_seen += m[e] > 0.0f;
            if (r[e] >= 0) {
                plan.mass_resident += (double) m[e];
                if (may_evict(l, e)) vict.emplace_back(m[e], e);
            } else if (m[e] > 0.0f && may_enter(l, e)) {
                cand.emplace_back(m[e], e);
                ++plan.candidates;
            }
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        std::sort(vict.begin(), vict.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first < b.first : a.second < b.second;
        });
        const size_t nc = std::min(cand.size(), vict.size());
        for (size_t i = 0; i < nc; ++i) {
            if (!(cand[i].first > vict[i].first) || cand[i].first < vict[i].first * cfg.gain) break;
            per_layer[(size_t) l].push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    // the budget: each layer's floor first (its swaps are already in gain order), then the rest by gain
    std::vector<PrefillSeedSwap> rest;
    for (auto& v : per_layer) {
        const size_t keep = std::min(v.size(), (size_t) std::max(0, cfg.floor));
        plan.swaps.insert(plan.swaps.end(), v.begin(), v.begin() + (ptrdiff_t) keep);
        rest.insert(rest.end(), v.begin() + (ptrdiff_t) keep, v.end());
    }
    auto by_gain = [](const PrefillSeedSwap& a, const PrefillSeedSwap& b) {
        if (a.gain != b.gain) return a.gain > b.gain;
        return a.layer != b.layer ? a.layer < b.layer : a.in < b.in;
    };
    std::sort(rest.begin(), rest.end(), by_gain);
    const size_t cap = cfg.swaps > 0 ? (size_t) cfg.swaps : plan.swaps.size() + rest.size();
    if (plan.swaps.size() > cap) plan.swaps.resize(cap);   // floors past the cap: the best of them, in layer order
    const size_t room = cap - plan.swaps.size();
    plan.swaps.insert(plan.swaps.end(), rest.begin(), rest.begin() + (ptrdiff_t) std::min(room, rest.size()));
    std::sort(plan.swaps.begin(), plan.swaps.end(), by_gain);
    plan.mass_after = plan.mass_resident;
    for (const PrefillSeedSwap& s : plan.swaps) plan.mass_after += (double) s.gain;
    return plan;
}

/// The usage prior (docs/PREFILL_SEED.md): the decode's adaptive tier counts choices per (layer, expert) and decays
/// them; a seeded expert with a usage of zero would be the first victim of the next round.  The prompt's mass is
/// turned into "one horizon of synthetic choices": prior[i] = horizon x mass[i] x (choices / mass over the whole
/// table) / tokens, i.e. the choices the expert would collect over `horizon` tokens if the decode routed as the
/// prompt did, weighted by gate mass.  `usage[i] = max(usage[i], prior[i])`; nothing is lowered.  Returns how many
/// cells were raised.
inline int64_t prefill_seed_prior(std::vector<float>& usage, const float* mass, const float* count, size_t cells,
                                  int64_t tokens, float horizon) {
    if (usage.size() != cells || tokens <= 0 || !(horizon > 0.0f)) return 0;
    double mass_sum = 0.0, count_sum = 0.0;
    for (size_t i = 0; i < cells; ++i) { mass_sum += (double) mass[i]; count_sum += (double) count[i]; }
    if (!(mass_sum > 0.0) || !(count_sum > 0.0)) return 0;
    const double scale = (double) horizon * (count_sum / mass_sum) / (double) tokens;
    int64_t raised = 0;
    for (size_t i = 0; i < cells; ++i) {
        if (!(mass[i] > 0.0f)) continue;
        const float p = (float) ((double) mass[i] * scale);
        if (p > usage[i]) { usage[i] = p; ++raised; }
    }
    return raised;
}

}  // namespace strata::core
