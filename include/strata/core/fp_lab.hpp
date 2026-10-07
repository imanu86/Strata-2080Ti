// include/strata/core/fp_lab.hpp - FABLE (docs/FABLE_PLAN.md): the lab levers of the two-card decode as ONE runtime
// struct instead of env-cached statics, so a running engine switches them per request (`lab=NAME:V,...` in the serve
// protocol, "strata_lab" in the OpenAI request) and an A/B needs no restart: the prefix stays cached, the two arms
// alternate on the same process.
//
//   fp_lab_env()  the defaults, read once from STRATA_FP_* / STRATA_MTP_PRIORITY;
//   fp_lab()      the live values: set by the serve loop between requests (nothing in flight), read by the engine at
//                 the points that consume them.  A request without `lab=` runs the defaults.
//
// Levers that are baked into captured CUDA graphs (the kernel variants EXPERT_V / HC_FUSE_NORM / SHEXP_FUSE and the
// device plan DEVPLAN) make the serve loop drop every window graph when they change (Verifier::drop_graphs,
// MtpDrafter::drop_graphs: recaptured at the next window); the others are read per request or per layer.
#pragma once

#include <cctype>
#include <cstdlib>
#include <string>

namespace strata::core {

struct FpLab {
    int early_l1 = 0;          ///< STRATA_FP_EARLY_L1: stage 1 launched behind stage 0's event (fresh windows)
    int devplan = 0;           ///< STRATA_FP_DEVPLAN: device-planned all-resident groups in the pipelined windows (graphs)
    int probe_waita = 0;       ///< STRATA_FP_PROBE_WAITA: lab, WRONG text - flag A up before the CPU jobs
    int el_async = 0;          ///< STRATA_FP_EL_ASYNC: the elastic growth's blobs through a pinned bounce on a worker
    int chain_trim = 0;        ///< STRATA_FP_CHAIN_TRIM: the chain's tail steps launched as outputs land, none past B
    int chain_trim_ahead = 2;  ///< STRATA_FP_CHAIN_TRIM_AHEAD: steps queued beyond the landed outputs
    int expert_v = 0;          ///< STRATA_FP_EXPERT_V: the VRAM experts' kernel variant 0..4 (graphs)
    int expert_v4_tmax = 0;    ///< STRATA_FP_EXPERT_V4_TMAX: variant 4 only for windows of T <= N tokens, 0 = all (graphs)
    int hc_fuse_norm = 0;      ///< STRATA_FP_HC_FUSE_NORM: the hc read with the norm folded in, 0/1/2 (graphs)
    int shexp_fuse = 0;        ///< STRATA_FP_SHEXP_FUSE: the shared expert's fused launches, 0/1/2 (graphs)
    int mtp_priority = 1;      ///< STRATA_MTP_PRIORITY: the chain stream, 1 highest, 0 default, -1 lowest
};

inline int fp_lab_env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
}

inline FpLab fp_lab_from_env() {
    FpLab l;
    l.early_l1 = fp_lab_env_int("STRATA_FP_EARLY_L1", 0) != 0;
    l.devplan = fp_lab_env_int("STRATA_FP_DEVPLAN", 0) != 0;
    l.probe_waita = fp_lab_env_int("STRATA_FP_PROBE_WAITA", 0) != 0;
    l.el_async = fp_lab_env_int("STRATA_FP_EL_ASYNC", 0) != 0;
    l.chain_trim = fp_lab_env_int("STRATA_FP_CHAIN_TRIM", 0) != 0;
    l.chain_trim_ahead = fp_lab_env_int("STRATA_FP_CHAIN_TRIM_AHEAD", 2);
    if (l.chain_trim_ahead < 1) l.chain_trim_ahead = 1;
    l.expert_v = fp_lab_env_int("STRATA_FP_EXPERT_V", 0);
    if (l.expert_v < 0 || l.expert_v > 4) l.expert_v = 0;
    l.expert_v4_tmax = fp_lab_env_int("STRATA_FP_EXPERT_V4_TMAX", 0);
    if (l.expert_v4_tmax < 0) l.expert_v4_tmax = 0;
    l.hc_fuse_norm = fp_lab_env_int("STRATA_FP_HC_FUSE_NORM", 0);
    if (l.hc_fuse_norm < 0 || l.hc_fuse_norm > 2) l.hc_fuse_norm = l.hc_fuse_norm > 2 ? 2 : 0;
    l.shexp_fuse = fp_lab_env_int("STRATA_FP_SHEXP_FUSE", 0);
    if (l.shexp_fuse < 0 || l.shexp_fuse > 2) l.shexp_fuse = l.shexp_fuse > 2 ? 2 : 0;
    l.mtp_priority = fp_lab_env_int("STRATA_MTP_PRIORITY", 1);
    l.mtp_priority = l.mtp_priority > 0 ? 1 : l.mtp_priority < 0 ? -1 : 0;
    return l;
}

/// the defaults (the environment, read once)
inline const FpLab& fp_lab_env() {
    static const FpLab env = fp_lab_from_env();
    return env;
}

/// the live levers (the serve loop writes them between requests; everything else reads)
inline FpLab& fp_lab() {
    static FpLab live = fp_lab_env();
    return live;
}

inline bool fp_lab_same(const FpLab& a, const FpLab& b) {
    return a.early_l1 == b.early_l1 && a.devplan == b.devplan && a.probe_waita == b.probe_waita &&
           a.el_async == b.el_async && a.chain_trim == b.chain_trim && a.chain_trim_ahead == b.chain_trim_ahead &&
           a.expert_v == b.expert_v && a.expert_v4_tmax == b.expert_v4_tmax && a.hc_fuse_norm == b.hc_fuse_norm &&
           a.shexp_fuse == b.shexp_fuse &&
           a.mtp_priority == b.mtp_priority;
}

/// `spec` = "NAME:VALUE,NAME:VALUE,..." applied over `out` (names case-insensitive, with or without the STRATA_FP_ /
/// STRATA_ prefix; the value an integer).  False with `err` on an unknown name or a malformed item; `out` then holds
/// the items before it.
inline bool fp_lab_parse(const std::string& spec, FpLab& out, std::string& err) {
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
        if (name.rfind("STRATA_FP_", 0) == 0) name = name.substr(10);
        else if (name.rfind("STRATA_", 0) == 0) name = name.substr(7);
        char* end = nullptr;
        const long v = std::strtol(item.c_str() + c + 1, &end, 10);
        if (end == nullptr || *end != '\0') { err = "lab: '" + item + "' has no integer value"; return false; }
        const int x = (int) v;
        if (name == "EARLY_L1") out.early_l1 = x != 0;
        else if (name == "DEVPLAN") out.devplan = x != 0;
        else if (name == "PROBE_WAITA") out.probe_waita = x != 0;
        else if (name == "EL_ASYNC") out.el_async = x != 0;
        else if (name == "CHAIN_TRIM") out.chain_trim = x != 0;
        else if (name == "CHAIN_TRIM_AHEAD") out.chain_trim_ahead = x < 1 ? 1 : x;
        else if (name == "EXPERT_V") out.expert_v = x < 0 || x > 4 ? 0 : x;
        else if (name == "EXPERT_V4_TMAX") out.expert_v4_tmax = x < 0 ? 0 : x;
        else if (name == "HC_FUSE_NORM") out.hc_fuse_norm = x <= 0 ? 0 : x >= 2 ? 2 : 1;
        else if (name == "SHEXP_FUSE") out.shexp_fuse = x <= 0 ? 0 : x >= 2 ? 2 : 1;
        else if (name == "MTP_PRIORITY") out.mtp_priority = x > 0 ? 1 : x < 0 ? -1 : 0;
        else { err = "lab: unknown lever '" + name + "'"; return false; }
    }
    return true;
}

inline std::string fp_lab_describe(const FpLab& l) {
    return "EARLY_L1=" + std::to_string(l.early_l1) + " DEVPLAN=" + std::to_string(l.devplan) +
           " PROBE_WAITA=" + std::to_string(l.probe_waita) + " EL_ASYNC=" + std::to_string(l.el_async) +
           " CHAIN_TRIM=" + std::to_string(l.chain_trim) + " CHAIN_TRIM_AHEAD=" + std::to_string(l.chain_trim_ahead) +
           " EXPERT_V=" + std::to_string(l.expert_v) + " EXPERT_V4_TMAX=" + std::to_string(l.expert_v4_tmax) +
           " HC_FUSE_NORM=" + std::to_string(l.hc_fuse_norm) +
           " SHEXP_FUSE=" + std::to_string(l.shexp_fuse) + " MTP_PRIORITY=" + std::to_string(l.mtp_priority);
}

}  // namespace strata::core
