// Optional, single text-prefix persistence. The codec has no CUDA dependency.
#pragma once
#include "strata/core/conversation_cache.hpp"
#include <array>
#include <filesystem>
#include <string>

namespace strata::core {
using PrefixDigest = std::array<uint8_t, 32>;
constexpr uint64_t prefix_cache_max_bytes = 512ULL * 1024 * 1024;
constexpr size_t prefix_cache_max_tokens = 32768;

// SHA-256 payload integrity and cache-key hashing (CNG on Windows).
PrefixDigest prefix_cache_digest(const void *bytes, size_t size);
// Local artifact identity: canonical paths, size/mtime, complete small files,
// and three 64 KiB samples of large files. This is NOT a full weights hash.
// Replacing weights while preserving metadata and unsampled bytes requires
// removing the cache; ordinary model/config/prompt edits invalidate it.
bool prefix_cache_identity(const std::vector<std::filesystem::path> &assets,
                           const std::vector<std::string> &settings,
                           PrefixDigest &identity, std::string &error);

// One minimal snapshot: no images, stage parts or duplicate checkpoints.
// next_token also keys MTP's one-token lookahead at the saved frontier.
bool prefix_cache_write(const std::filesystem::path &path,
                        const PrefixDigest &identity, int32_t next_token,
                        const SavedConversation &image, std::string &error);
// Bounded decode into a temporary CPU object; out is untouched on any failure.
// Exact expected prefix is checked before allocating the large state payload.
// Caller must still run conversation_snapshot_validate before any GPU writes.
bool prefix_cache_read(const std::filesystem::path &path,
                       const PrefixDigest &identity,
                       const std::vector<int32_t> &expected_prefix,
                       int32_t next_token, bool cvec, SavedConversation &out,
                       std::string &error);
} // namespace strata::core
