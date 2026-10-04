#include "strata/core/prefix_cache_file.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char *message) {
  ++checks;
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

std::string hex(const PrefixDigest &digest) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(digest.size() * 2);
  for (uint8_t byte : digest) {
    result.push_back(digits[byte >> 4]);
    result.push_back(digits[byte & 15]);
  }
  return result;
}

struct TempDirectory {
  fs::path path;
  TempDirectory() {
    const auto root = fs::current_path();
    for (unsigned attempt = 0; attempt < 32; ++attempt) {
      const auto stamp =
          std::chrono::steady_clock::now().time_since_epoch().count();
      auto candidate =
          root / ("prefix-cache-file-test-" + std::to_string(stamp) + "-" +
                  std::to_string(attempt));
      std::error_code ec;
      if (fs::create_directory(candidate, ec)) {
        path = std::move(candidate);
        return;
      }
      if (ec && ec != std::errc::file_exists) {
        std::fprintf(stderr, "cannot create test directory: %s\n",
                     ec.message().c_str());
        std::exit(1);
      }
    }
    std::fprintf(stderr, "cannot allocate a unique test directory\n");
    std::exit(1);
  }
  ~TempDirectory() {
    std::error_code ec;
    fs::remove_all(path, ec);
    if (ec)
      std::fprintf(stderr, "warning: test directory cleanup failed: %s\n",
                   ec.message().c_str());
  }
};

void fill_pattern(ConversationBuffer &buffer, uint8_t salt);

SavedConversation make_image() {
  SavedConversation image;
  image.geometry = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                    10, 11, 12, 13, 14, 15, 16, 17, 18};
  image.layer_lo = 2;
  image.layer_hi = 5;
  image.cvec = true;
  image.live.ids = {17, 23, 41, 59};
  image.live.gdn = {1, 2, 3};
  image.live.ple = {4, 5};
  image.live.tails = {6};
  image.live.dead = {7, 8, 9};
  image.live.block_pos = {10, 11};
  ConversationKv kv;
  kv.format = 3;
  kv.cells = 64;
  kv.heads = 8;
  kv.head_dim = 128;
  kv.page_size = 16;
  kv.pooled_rows = 12;
  kv.idx_dim = 32;
  kv.k.resize(ConversationBuffer::segment_bytes + 257);
  kv.v.resize(37);
  kv.k_scale = {12, 13, 14};
  kv.v_scale = {15, 16};
  kv.pooled.resize(19);
  fill_pattern(kv.k, 29);
  fill_pattern(kv.v, 47);
  fill_pattern(kv.pooled, 83);
  image.kv.push_back(std::move(kv));
  return image;
}

void fill_pattern(ConversationBuffer &buffer, uint8_t salt) {
  const bool ok = buffer.visit(
      0, buffer.size(), [=](uint8_t *data, size_t size, size_t offset) {
        for (size_t i = 0; i < size; ++i)
          data[i] = static_cast<uint8_t>(salt + (offset + i) * 17 +
                                         (offset + i) / 251);
        return true;
      });
  check(ok, "fill segmented conversation buffer");
}

bool equal_image(const SavedConversation &a, const SavedConversation &b) {
  if (a.geometry != b.geometry || a.layer_lo != b.layer_lo ||
      a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
      a.live.ids != b.live.ids || a.live.imgs != b.live.imgs ||
      a.live.gdn != b.live.gdn || a.live.ple != b.live.ple ||
      a.live.tails != b.live.tails || a.live.dead != b.live.dead ||
      a.live.block_pos != b.live.block_pos ||
      a.live.stage_parts.size() != b.live.stage_parts.size() ||
      a.checkpoints.size() != b.checkpoints.size() ||
      a.kv.size() != b.kv.size())
    return false;
  for (size_t i = 0; i < a.kv.size(); ++i) {
    const auto &x = a.kv[i];
    const auto &y = b.kv[i];
    if (x.format != y.format || x.cells != y.cells || x.heads != y.heads ||
        x.head_dim != y.head_dim || x.page_size != y.page_size ||
        x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
        !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) ||
        !(x.v_scale == y.v_scale) || !(x.pooled == y.pooled))
      return false;
  }
  return true;
}

std::vector<uint8_t> read_file(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input),
                              std::istreambuf_iterator<char>());
}
void write_file(const fs::path &path, const std::vector<uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  check(bool(output), "open mutation fixture");
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  check(bool(output), "write mutation fixture");
}
void put_u64_le(std::vector<uint8_t> &data, size_t offset, uint64_t value) {
  check(offset <= data.size() && data.size() - offset >= 8,
        "u64 mutation offset in bounds");
  for (size_t i = 0; i < 8; ++i)
    data[offset + i] = static_cast<uint8_t>(value >> (i * 8));
}

void preserved_failure(const fs::path &path, const PrefixDigest &identity,
                       const std::vector<int32_t> &prefix, int32_t frontier,
                       bool cvec, const SavedConversation &sentinel,
                       const char *message) {
  SavedConversation out = sentinel;
  std::string error;
  check(!prefix_cache_read(path, identity, prefix, frontier, cvec, out, error),
        message);
  check(!error.empty(), "failed read reports a diagnostic");
  check(equal_image(out, sentinel),
        "failed read leaves caller output unchanged");
}
} // namespace

int main() {
  check(hex(prefix_cache_digest(nullptr, 0)) ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA-256 empty vector");
  const char abc[] = "abc";
  check(hex(prefix_cache_digest(abc, 3)) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 abc vector");
  const char multi[] =
      "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  check(hex(prefix_cache_digest(multi, sizeof(multi) - 1)) ==
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "SHA-256 multiblock vector");

  TempDirectory temp;
  const fs::path weights = temp.path / "weights.bin";
  {
    std::ofstream file(weights, std::ios::binary);
    const char contents[] = "prefix-cache model bytes v1";
    file.write(contents, sizeof(contents) - 1);
    check(bool(file), "create identity asset");
  }
  const std::vector<std::string> settings = {"format=iq3", "context=262144",
                                             "prompt-hash=abc"};
  PrefixDigest identity{};
  std::string error;
  check(prefix_cache_identity({weights}, settings, identity, error),
        "compute model/config identity");
  check(error.empty(), "identity success clears diagnostic");
  PrefixDigest config_identity{};
  check(prefix_cache_identity(
            {weights}, {"format=iq3", "context=262144", "prompt-hash=def"},
            config_identity, error),
        "compute changed-prompt identity");
  check(config_identity != identity,
        "prompt/configuration participates in identity");
  PrefixDigest asset_identity{};
  {
    std::ofstream file(weights, std::ios::binary | std::ios::trunc);
    const char contents[] = "prefix-cache model bytes v2";
    file.write(contents, sizeof(contents) - 1);
  }
  check(prefix_cache_identity({weights}, settings, asset_identity, error),
        "recompute changed-asset identity");
  check(asset_identity != identity,
        "model asset content participates in identity");
  {
    std::ofstream file(weights, std::ios::binary | std::ios::trunc);
    const char contents[] = "prefix-cache model bytes v1";
    file.write(contents, sizeof(contents) - 1);
  }
  check(prefix_cache_identity({weights}, settings, identity, error),
        "restore original identity");

  const fs::path cache = temp.path / "saved-prefix.bin";
  const SavedConversation source = make_image();
  const std::vector<int32_t> prefix = source.live.ids;
  constexpr int32_t frontier = 101;
  check(prefix_cache_write(cache, identity, frontier, source, error),
        "write minimal text prefix cache");
  check(error.empty(), "write success clears diagnostic");
  SavedConversation restored;
  check(prefix_cache_read(cache, identity, prefix, frontier, source.cvec,
                          restored, error),
        "read valid cache");
  check(error.empty() && equal_image(source, restored),
        "roundtrip preserves metadata and all segmented KV bytes");
  check(restored.kv[0].k.size() > ConversationBuffer::segment_bytes,
        "roundtrip exercised multiple KV segments");

  const SavedConversation sentinel = make_image();
  const auto valid_bytes = read_file(cache);
  check(valid_bytes.size() > 88 + 208,
        "serialized fixture includes fixed header and first field");
  check(valid_bytes.size() - 88 <= prefix_cache_max_bytes,
        "valid fixture below file cap");
  check(valid_bytes[0] == 'S' && valid_bytes[1] == 'T',
        "serialized fixture magic sanity");

  check(!prefix_cache_read(cache, config_identity, prefix, frontier,
                           source.cvec, restored, error) &&
            equal_image(restored, source),
        "identity mismatch rejects without replacing output");
  check(!prefix_cache_read(cache, asset_identity, prefix, frontier, source.cvec,
                           restored, error) &&
            equal_image(restored, source),
        "changed model asset identity rejects without replacing output");
  auto changed_prefix = prefix;
  changed_prefix[1]++;
  check(!prefix_cache_read(cache, identity, changed_prefix, frontier,
                           source.cvec, restored, error) &&
            equal_image(restored, source),
        "prompt token mismatch rejects without replacing output");
  check(!prefix_cache_read(cache, identity, prefix, frontier + 1, source.cvec,
                           restored, error) &&
            equal_image(restored, source),
        "next-token frontier mismatch rejects without replacing output");
  check(!prefix_cache_read(cache, identity, prefix, frontier, false, restored,
                           error) &&
            equal_image(restored, source),
        "cvec mode mismatch rejects without replacing output");

  const auto reject_mutation = [&](const char *name, auto mutate) {
    auto bytes = valid_bytes;
    mutate(bytes);
    const fs::path bad = temp.path / (std::string(name) + ".bin");
    write_file(bad, bytes);
    preserved_failure(bad, identity, prefix, frontier, source.cvec, sentinel,
                      name);
  };
  reject_mutation("bad-magic", [](auto &b) { b[0] ^= 0x20; });
  reject_mutation("bad-version", [](auto &b) { b[8] ^= 1; });
  reject_mutation("truncated", [](auto &b) { b.pop_back(); });
  reject_mutation("bad-checksum", [](auto &b) { b[56] ^= 0x80; });
  reject_mutation("trailing-data", [](auto &b) {
    b.push_back(0x5a);
    put_u64_le(b, 16, static_cast<uint64_t>(b.size() - 88));
  });
  reject_mutation("oversized-field-length", [](auto &b) {
    // Still below the global 512 MiB cap, but larger than all remaining file
    // bytes.
    put_u64_le(b, 88 + 216, static_cast<uint64_t>(b.size()));
  });
  reject_mutation("field-over-512mib", [](auto &b) {
    // Header(88) + 18 geometry + layer bounds + cvec/frontier + token count +
    // IDs.
    put_u64_le(b, 88 + 216, prefix_cache_max_bytes + 1);
  });

  const fs::path atomic = temp.path / "atomic-preserve.bin";
  check(prefix_cache_write(atomic, identity, frontier, source, error),
        "seed atomic-write destination");
  const auto before_invalid_write = read_file(atomic);
  auto invalid = source;
  invalid.live.ids[0] = -1; // Fail after the temporary file has already
                            // received part of the payload.
  check(!prefix_cache_write(atomic, identity, frontier, invalid, error),
        "reject invalid token during atomic write");
  check(read_file(atomic) == before_invalid_write,
        "invalid write preserves prior file byte-for-byte");
  for (const auto &entry : fs::directory_iterator(temp.path))
    check(entry.path().filename().string().find(".tmp-") == std::string::npos,
          "atomic writer leaves no temp file");

  auto too_many_tokens = source;
  too_many_tokens.live.ids.resize(prefix_cache_max_tokens + 1, 1);
  const fs::path too_many = temp.path / "too-many-tokens.bin";
  check(
      !prefix_cache_write(too_many, identity, frontier, too_many_tokens, error),
      "reject token count above 32768");
  check(!fs::exists(too_many), "oversized token write publishes no file");
  check(!prefix_cache_read(cache, identity,
                           std::vector<int32_t>(prefix_cache_max_tokens + 1, 1),
                           frontier, source.cvec, restored, error) &&
            equal_image(restored, source),
        "reject expected prefix above 32768 without changing output");

  std::printf("prefix_cache_file_test: %d checks passed\n", checks);
  return 0;
}
