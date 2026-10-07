// include/strata/core/ram_dedup.hpp - STRATA_RAM_DEDUP=1: the expert arena without the RAM copies of the experts the
// GPUs hold.
//
// The resident arena (ArenaExpertSource / PinnedArena) holds EVERY expert, and the VRAM tiers hold ~half of them
// again: on a 96 GB PC with a 12 GB + 22 GB pair the arena is ~40 GiB of which ~21 GiB are duplicates of what sits
// in the two caches.  This object gives those pages back to the OS - decommitted in place (VirtualFree MEM_DECOMMIT
// on Windows, MADV_DONTNEED on Linux), so no pointer moves and `blob()` keeps returning base + offset - and brings an
// expert's bytes back into the arena BEFORE its slot is overwritten or unmapped, by a device-to-host copy on a
// dedicated stream into a page-locked bounce ring, ordered by events against the stream that will overwrite the slot.
//
// WHAT STAYS IN RAM (the caller decides per expert, see `drop`): the experts in slots the prompt path may lend or
// the elastic tail may shrink keep their RAM copy (a loan streams them from RAM; a shrink hands them to the CPU),
// as does anything inside the arena's CUDA-registered prefix (decommitting registered pages is not allowed), and
// an optional quota of the hottest slots (STRATA_RAM_DEDUP_KEEP_GIB).
//
// WHAT NEVER CHANGES: the bytes.  A restored expert is the slot's bytes, which are the arena's bytes (a slot is only
// ever filled from the arena), so the tokens are the same bit for bit; only where the bytes sit changes.  When a
// copy back is missing (a path that evicted without telling this object, or a failed copy), the expert is read
// again from the model file, unbuffered, and counted (`reread`).
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace strata::kernels::cpu {
struct ExpertLayout;
}

namespace strata::core {

class RamDedup {
public:
    RamDedup() = default;
    ~RamDedup();
    RamDedup(const RamDedup&) = delete;
    RamDedup& operator=(const RamDedup&) = delete;

    /// STRATA_RAM_DEDUP=1 asked for.  Off by default.
    static bool requested();
    /// STRATA_RAM_DEDUP_KEEP_GIB: the redundant quota in bytes, or ~0 when unset (the caller's default: the lend
    /// regions plus the elastic tail).
    static uint64_t keep_quota_env();
    /// STRATA_RAM_DEDUP_BOUNCE_MIB: the page-locked bounce ring's size (default 512).
    static uint64_t bounce_bytes_env();

    /// Arms the object on an arena of `arena_bytes` at `base` (anonymous, committed, 4 KiB pages).  `registered`:
    /// the CUDA-registered prefix (never decommitted).  `large_pages`: refused (a large-page region cannot be
    /// decommitted page by page).  `experts_bin`: the pack's experts.bin, or empty with `gguf` the --native shard
    /// (the re-read fallback reads whichever the arena was loaded from).  Creates the bounce ring (cudaHostAlloc)
    /// and nothing per device yet (streams are made on first use).  False with `err` when it cannot run.
    bool arm(uint8_t* base, uint64_t arena_bytes, uint64_t registered, bool large_pages, int64_t n_layers,
             int64_t n_expert, const std::string& experts_bin, const std::string& gguf, uint64_t bounce_bytes,
             std::string& err);
    bool armed() const { return armed_; }

    /// (layer, expert) is in VRAM and nothing reads its RAM copy any more: decommit the whole pages inside its blob.
    /// Returns the bytes given back (0: kept - inside the registered prefix, already absent, a copy in flight).
    uint64_t drop(int64_t layer, int64_t expert);
    /// Whether the arena holds (layer, expert)'s bytes right now (false: absent, or a copy back in flight).
    bool present(int64_t layer, int64_t expert) const {
        return !armed_ || state_[(size_t) (layer * n_expert_ + expert)].load(std::memory_order_acquire) == kPresent;
    }
    /// (layer, expert) is about to leave VRAM: its slot `dev_slot` (on device `dev`, -1 = the current one) will be
    /// overwritten or unmapped by work queued on `caller_stream` after this call.  When the arena lacks the expert, a
    /// device-to-host copy of the slot into the bounce ring is queued on this object's stream for that device,
    /// after the work `caller_stream` holds now, and `caller_stream` is made to wait for it (events, no host wait).
    /// A full ring lands its oldest entry first (a host wait on that copy).  Nothing happens for a present expert.
    /// False only when CUDA refused (the expert is then read from the file when the CPU needs it).
    bool evict(int64_t layer, int64_t expert, const void* dev_slot, void* caller_stream, int dev);
    /// Lands the copies that have completed (bounce -> arena); `wait` = all of them, with a host wait.
    void drain(bool wait);
    /// Waits for every copy queued on this object's stream of device `dev` (-1: every device): before a slot's
    /// chunks are unmapped.  The landing itself stays with `drain` / `ensure`.
    void sync(int dev);
    /// Makes (layer, expert) readable in the arena: lands its copy back (a wait on its event), or re-reads it from
    /// the model file.  Called by the arena's `blob` for a non-present expert; any thread.  False: unreadable.
    bool ensure(int64_t layer, int64_t expert);

    // ---- the numbers for the log
    uint64_t arena_bytes() const { return arena_bytes_; }
    uint64_t decommitted_bytes() const { return decommitted_.load(std::memory_order_relaxed); }
    int64_t absent() const { return absent_.load(std::memory_order_relaxed); }
    int64_t d2h_experts() const { return d2h_.load(std::memory_order_relaxed); }
    double d2h_ms() const { return (double) d2h_us_.load(std::memory_order_relaxed) / 1000.0; }
    int64_t reread() const { return reread_.load(std::memory_order_relaxed); }
    int64_t kept_registered() const { return kept_registered_.load(std::memory_order_relaxed); }
    uint64_t bounce_bytes() const { return bounce_bytes_; }
    int64_t bounce_waits() const { return bounce_waits_.load(std::memory_order_relaxed); }

private:
    static constexpr uint8_t kPresent = 0, kAbsent = 1, kInFlight = 2;
    struct Dev { void* stream = nullptr; };
    struct Slot { int64_t idx = -1; void* ev_before = nullptr; void* ev_after = nullptr; int ev_dev = -1; bool live = false; };
    bool decommit(int64_t idx, uint64_t& bytes);
    bool recommit(int64_t idx);
    void page_span(int64_t idx, uint64_t& start, uint64_t& end) const;
    bool land_locked(size_t q);           ///< mu_ held: bounce slot q -> the arena (its event complete)
    bool reread_locked(int64_t idx);      ///< mu_ held: the model file -> the arena
    bool device_stream(int dev, void*& stream);
    bool armed_ = false;
    uint8_t* base_ = nullptr;
    uint64_t arena_bytes_ = 0, registered_ = 0, bounce_bytes_ = 0;
    int64_t n_layers_ = 0, n_expert_ = 0;
    std::string experts_bin_, gguf_;
    const strata::kernels::cpu::ExpertLayout* lay_ = nullptr;
    std::vector<std::atomic<uint8_t>> state_;
    std::vector<int32_t> slot_of_;        ///< per expert: its bounce slot while in flight, else -1
    std::vector<Slot> ring_;
    uint8_t* bounce_ = nullptr;
    uint64_t bounce_blob_ = 0;
    size_t head_ = 0, tail_ = 0, in_ring_ = 0;
    std::vector<Dev> devs_;
    mutable std::mutex mu_;
    std::atomic<uint64_t> decommitted_{0}, d2h_us_{0};
    std::atomic<int64_t> absent_{0}, d2h_{0}, reread_{0}, kept_registered_{0}, bounce_waits_{0};
    std::vector<uint8_t> aligned_tmp_;    ///< the re-read's 4 KiB-aligned window (mu_ held)
};

}  // namespace strata::core
