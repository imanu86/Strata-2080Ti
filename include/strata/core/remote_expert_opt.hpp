#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
class RemoteExperts;
class ExpertSource;

// Remote Expert Decode Optimization v1 on existing helper caches. One instance per serial engine.
class RemoteExpertOpt {
public:
    RemoteExpertOpt() = default;
    ~RemoteExpertOpt();
    RemoteExpertOpt(const RemoteExpertOpt&) = delete;
    RemoteExpertOpt& operator=(const RemoteExpertOpt&) = delete;

    void attach(RemoteExperts& remote); // before RemoteExperts::open
    bool init(std::string& err);        // after all helper caches are open
    bool owns(int64_t layer, int32_t expert) const;
    bool adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary,
               const std::vector<std::pair<int32_t, int32_t>>& pending, int max_swaps,
               ExpertSource& source);

    bool elastic_due(bool request_start) const;
    bool audit_disjoint(const std::vector<int32_t>& primary,
                        const std::vector<std::pair<int32_t, int32_t>>& pending, std::string& err) const;
    bool elastic_step(const std::vector<float>& usage, const std::vector<int32_t>& primary,
                      const std::vector<std::pair<int32_t, int32_t>>& pending,
                      ExpertSource& source, bool request_start, std::string& err);
    uint64_t adapt_rounds() const { return adapt_rounds_; }
    uint64_t adapt_swaps() const { return adapt_swaps_; }

    // Bracket the existing pool callback; its signature and dispatch remain unchanged.
    void begin(const float* weights, int token_begin, int tokens);
    void end() { weights_ = nullptr; }
    void copy_rows(float* dst, const float* src, int token_begin, int tokens,
                   const int32_t* hit_rows, const int32_t* count, void* stream) const;
    void combine(float* dst, float* scratch, int token_begin, int tokens,
                 const uint32_t* skip, uint32_t ring, void* stream) const;

private:
    friend class RemoteExperts;
    static size_t metadata_bytes();
    bool active() const { return weights_ != nullptr; }
    void prepare(const RemoteExperts& remote, void* metadata) const;
    bool reduce(RemoteExperts& remote, const void* metadata, std::string& err);
    void accumulate(const RemoteExperts& remote);

    struct Peer { RemoteExperts* remote; float* sum = nullptr; };
    std::vector<Peer> peers_;
    float* h_sum_ = nullptr;
    float* m_sum_ = nullptr;
    int32_t* h_mask_ = nullptr;
    int32_t* m_mask_ = nullptr;
    uint64_t adapt_rounds_ = 0, adapt_swaps_ = 0;
    const float* weights_ = nullptr;
    int token_begin_ = 0, tokens_ = 0;
};
} // namespace strata::core
