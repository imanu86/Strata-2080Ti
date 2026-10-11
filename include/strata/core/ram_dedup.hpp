// Derived from FABLE cd7bdce/0dc44cea/01385d7; GPT6.1 checked port.
// Opt-in arena decommit: only interior pages, registered pages never removed.
// No disk recovery: a lost/failed GPU backing is fatal before reuse/unmap.
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
namespace strata::kernels::cpu { struct ExpertLayout; }
namespace strata::core {
class RamDedup {
public:
    RamDedup() = default;
    ~RamDedup();
    RamDedup(const RamDedup&) = delete;
    RamDedup& operator=(const RamDedup&) = delete;
    static bool requested();
    static uint64_t keep_quota_env();
    static uint64_t bounce_bytes_env();
    bool arm(uint8_t* base, uint64_t arena_bytes,
             const std::vector<std::pair<uint64_t,uint64_t>>& registered_ranges,
             bool normal_anonymous_pages, int64_t n_layers, int64_t n_expert,
             uint64_t bounce_bytes, std::string& err);
    bool armed() const { return armed_; }
    // Only caller-proved quiescent resident experts may be dropped.
    bool drop(int64_t layer, int64_t expert, uint64_t& released, std::string& err);
    bool present(int64_t layer, int64_t expert) const;
    bool evict(int64_t layer, int64_t expert, const void* device_slot,
               void* caller_stream, int device, std::string& err);
    bool drain(bool wait, std::string& err);
    bool sync(int device, std::string& err);
    bool ensure(int64_t layer, int64_t expert, std::string& err);
    // Explicit checked close must precede freeing the arena; false retains handles.
    bool close(std::string& err);
    std::string error() const;
    uint64_t arena_bytes() const { return arena_bytes_; }
    uint64_t decommitted_bytes() const { return decommitted_.load(); }
    uint64_t bounce_bytes() const { return bounce_bytes_; }
    uint64_t metadata_bytes() const {
        return sizeof(*this)+registered_.capacity()*sizeof(std::pair<uint64_t,uint64_t>)+
               state_.capacity()*sizeof(std::atomic<uint8_t>)+slot_of_.capacity()*sizeof(int32_t)+
               ring_.capacity()*sizeof(Slot)+devices_.capacity()*sizeof(Device);
    }
    int64_t absent() const { return absent_.load(); }
    int64_t d2h_experts() const { return d2h_.load(); }
    int64_t bounce_waits() const { return bounce_waits_.load(); }
    int64_t kept_registered() const { return kept_registered_.load(); }
    double d2h_ms() const { return d2h_us_.load()/1000.0; }
    static bool interior_span(uint64_t offset, uint64_t bytes, uint64_t capacity,
                              uint64_t& start, uint64_t& end) {
        constexpr uint64_t page=4096;
        start=end=0;
        if(offset>capacity || bytes>capacity-offset || offset>UINT64_MAX-(page-1)) return false;
        start=(offset+page-1)/page*page; end=(offset+bytes)/page*page;
        if(end<start)end=start;
        return true;
    }
private:
    enum : uint8_t { Present, Absent, InFlight };
    struct Device { void* stream=nullptr; };
    struct Slot { int64_t idx=-1; void* before=nullptr; void* after=nullptr;
                  int device=-1; bool live=false; };
    bool fail(const std::string& reason, std::string& err);
    bool healthy(std::string& err) const;
    bool index(int64_t layer, int64_t expert, size_t& idx) const;
    bool span(size_t idx, uint64_t& start, uint64_t& end) const;
    bool protected_span(uint64_t start, uint64_t end) const;
    bool select(int device, int& previous, std::string& err);
    bool restore_device(int previous, std::string& err);
    bool device_stream(int device, void*& stream, std::string& err);
    bool land_locked(size_t q, std::string& err);
    bool drain_locked(bool wait, std::string& err);
    bool sync_locked(int device, std::string& err);
    bool armed_=false;
    uint8_t* base_=nullptr;
    uint64_t arena_bytes_=0, bounce_bytes_=0, bounce_blob_=0;
    int64_t n_layers_=0, n_expert_=0;
    const strata::kernels::cpu::ExpertLayout* layout_=nullptr;
    std::vector<std::pair<uint64_t,uint64_t>> registered_;
    std::vector<std::atomic<uint8_t>> state_;
    std::vector<int32_t> slot_of_;
    std::vector<Slot> ring_;
    std::vector<Device> devices_;
    uint8_t* bounce_=nullptr;
    size_t head_=0, tail_=0, in_ring_=0;
    std::string fatal_;
    mutable std::mutex mu_;
    std::atomic<uint64_t> decommitted_{0},d2h_us_{0};
    std::atomic<int64_t> absent_{0},d2h_{0},bounce_waits_{0},kept_registered_{0};
};
} // namespace strata::core
