#pragma once
// Multi-GPU: make `device` current for a scope and restore the caller's device after it.  An object that owns CUDA
// streams, graphs and buffers on one device (a verify stage, the drafter) wraps its public calls in this, so the
// caller's thread may be on any device.  A negative device, or the device already current, does nothing.

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdlib>

namespace strata::core {

// lab/pipeline2 tappa 0b (docs/PIPELINE2.md): STRATA_P2_SWITCH_STAT=1 counts the device switches of OnDevice and the
// host time spent in them (cudaGetDevice + cudaSetDevice there and back); STRATA_P2_FLUSH_NOSWITCH=1 lets the pipelined
// polls (Verifier::service's 2 ms flush, Verifier::done, MtpDrafter::chain_outputs_ready / chain_poll) query their
// events and streams without switching (port of the NTRYourWaifu fork's STRATA_FLUSH_NO_SWITCH, two-pc-042 v7/v8).
// Both off by default; neither changes any GPU work.
inline bool p2_env(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::atoi(v) != 0;
}
inline bool p2_switch_stat() { static const bool on = p2_env("STRATA_P2_SWITCH_STAT"); return on; }
inline bool p2_flush_noswitch() { static const bool on = p2_env("STRATA_P2_FLUSH_NOSWITCH"); return on; }
// lab/pipeline2 (docs/PIPELINE2.md, tappa 1): STRATA_P2_FLUSH_US=<us> sets the period of the WDDM flush
// (cudaStreamQuery) in the pipelined polls (Verifier::service, done, pl_head_done); unset = 2000 us as before.  nsys
// shows the 2080 Ti's stage-1 graph idle ~2 ms in most windows between gr_down_staged and gr_up_multi of its first layer,
// until that flush.  No GPU work changes.
inline double p2_flush_ms() {
    static const double ms = [] {
        const char* v = std::getenv("STRATA_P2_FLUSH_US");
        const int us = v != nullptr ? std::atoi(v) : 0;
        return us > 0 ? us / 1000.0 : 2.0;
    }();
    return ms;
}
struct P2SwitchStat {
    std::atomic<long long> calls{0};      // OnDevice scopes that switched
    std::atomic<long long> ns{0};         // host time inside the switch calls (both ways)
    std::atomic<long long> max_ns{0};     // the longest single switch (one way)
    std::atomic<long long> slow{0};       // switches over 100 us (one way)
    std::atomic<long long> noswitch{0};   // polls answered without a switch (STRATA_P2_FLUSH_NOSWITCH)
    std::atomic<long long> refused{0};    // polls that fell back to the switch (the query refused the foreign device)
    // tappa 1: host time inside cudaGraphLaunch of the pipelined window graphs (pl_launch, the head and the tail), per
    // card slot (device & 1): nsys shows the 2080 Ti's 1300-node graph start its first nodes and then wait for the call
    std::atomic<long long> launches[2]{{0}, {0}};
    std::atomic<long long> launch_ns[2]{{0}, {0}};
    std::atomic<long long> launch_max_ns[2]{{0}, {0}};
};
inline P2SwitchStat& p2_switch() { static P2SwitchStat s; return s; }
// a query that refuses the foreign device turns the switch back on for the rest of the process, said once
inline std::atomic<bool>& p2_noswitch_off() { static std::atomic<bool> off{false}; return off; }
inline bool p2_noswitch_refused() { return p2_noswitch_off().load(std::memory_order_relaxed); }
void p2_noswitch_refuse(cudaError_t e);   // verify.cpp
inline void p2_launch_note(int device, std::chrono::steady_clock::time_point t0) {
    const long long d = (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count();
    P2SwitchStat& s = p2_switch();
    const int k = device & 1;
    s.launches[k].fetch_add(1, std::memory_order_relaxed);
    s.launch_ns[k].fetch_add(d, std::memory_order_relaxed);
    long long m = s.launch_max_ns[k].load(std::memory_order_relaxed);
    while (d > m && !s.launch_max_ns[k].compare_exchange_weak(m, d, std::memory_order_relaxed)) {}
}
inline void p2_switch_note(std::chrono::steady_clock::time_point t0) {
    const long long d = (long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count();
    P2SwitchStat& s = p2_switch();
    s.ns.fetch_add(d, std::memory_order_relaxed);
    if (d > 100000) s.slow.fetch_add(1, std::memory_order_relaxed);
    long long m = s.max_ns.load(std::memory_order_relaxed);
    while (d > m && !s.max_ns.compare_exchange_weak(m, d, std::memory_order_relaxed)) {}
}

struct OnDevice {
    int previous = -1;
    explicit OnDevice(int device) {
        if (device < 0) return;
        const bool stat = p2_switch_stat();
        const auto t0 = stat ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        int cur = 0;
        if (cudaGetDevice(&cur) == cudaSuccess && cur != device && cudaSetDevice(device) == cudaSuccess) {
            previous = cur;
            if (stat) {
                p2_switch().calls.fetch_add(1, std::memory_order_relaxed);
                p2_switch_note(t0);
            }
        }
    }
    ~OnDevice() {
        if (previous < 0) return;
        if (p2_switch_stat()) {
            const auto t0 = std::chrono::steady_clock::now();
            cudaSetDevice(previous);
            p2_switch_note(t0);
        } else {
            cudaSetDevice(previous);
        }
    }
    OnDevice(const OnDevice&) = delete;
    OnDevice& operator=(const OnDevice&) = delete;
};

}  // namespace strata::core
