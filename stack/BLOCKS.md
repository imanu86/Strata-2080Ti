# Strata-2080Ti patch stack: the inserted blocks

Generated from the patches by `check_anchors.py --build`. One section per insertion point of `INDEX.md`, the anchor (unpatched lines) first, then the hunk as `git diff` shows it. Hunks longer than 80 lines are cut after 24: the `.patch` has them whole.

## 01 elastic-cache

`01-elastic-cache.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la cache elastica con nucleo fisso (--elastic, PR #1117)

### 01-1

`CMakeLists.txt` · if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)

Anchor:
```
    add_test(NAME host_graph_launch_test COMMAND host_graph_launch_test)
    set_tests_properties(host_graph_launch_test PROPERTIES TIMEOUT 30)
  endif()
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 10 lines):
```diff
     add_test(NAME host_graph_launch_test COMMAND host_graph_launch_test)
     set_tests_properties(host_graph_launch_test PROPERTIES TIMEOUT 30)
   endif()
+  if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
+    # --elastic: the automatic elastic expert cache (CUDA virtual memory management through the driver API)
+    target_link_libraries(strata_engine PUBLIC CUDA::cuda_driver)
+  endif()
   if(STRATA_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core/exchange_storage_test.cpp")
     add_executable(exchange_storage_test tests/core/exchange_storage_test.cpp)
     target_include_directories(exchange_storage_test PRIVATE include)
```

### 01-2

`CMakeLists.txt` · if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)

Anchor:
```
    target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine)
    add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
    set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
```
Block (hunk 2 of `CMakeLists.txt` in the patch, 12 lines):
```diff
     target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine)
     add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
     set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
+    if(STRATA_ENABLE_CUDA)
+      add_executable(expert_cache_elastic_test tests/core/expert_cache_elastic_test.cpp)
+      target_link_libraries(expert_cache_elastic_test PRIVATE strata_engine)
+      add_test(NAME expert_cache_elastic_test COMMAND expert_cache_elastic_test)
+      set_tests_properties(expert_cache_elastic_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
+    endif()
     add_executable(expert_cache_segmented_test tests/core/expert_cache_segmented_test.cpp)
     target_link_libraries(expert_cache_segmented_test PRIVATE strata_engine)
     add_test(NAME expert_cache_segmented_test COMMAND expert_cache_segmented_test)
```

### 01-3

`docs/ELASTIC_CACHE.md` · (new file)

New file, 72 lines: see the patch.

### 01-4

`include/strata/core/expert_cache.hpp` · public:

Anchor:
```
    const uint64_t* slot_offsets() const { return off_.empty() ? nullptr : off_.data(); }
    void close();
```
Block (hunk 1 of `include/strata/core/expert_cache.hpp` in the patch, 26 lines):
```diff
     const uint64_t* slot_offsets() const { return off_.empty() ? nullptr : off_.data(); }
     void close();

+    /// ELASTIC ARENA (--elastic).  The same sized slots as `open_sized`, but the arena is a reserved
+    /// virtual range of `va_bytes`, backed by physical chunks of `chunk_bytes` (CUDA virtual memory management),
+    /// so the cache can GROW past its first size and SHRINK below it while the VRAM other programs use comes and
+    /// goes - and **the arena's address never moves**: the verify plan, the prompt path and the device residency
+    /// table keep the pointers they were handed at startup, and `slot_offsets()` has room reserved for every
+    /// (layer, expert) so it is never reallocated either.  False (err set) when the driver has no VMM; the caller
+    /// then opens the plain arena.
+    bool open_sized_elastic(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
+                            uint64_t va_bytes, uint64_t chunk_bytes, std::string& err);
+    bool elastic() const { return elastic_vmm_; }
+    /// Appends slots of the given sizes after the last one, mapping chunks as needed and zeroing them.  All or
+    /// nothing: on failure the cache is exactly as before.  (Named apart from #533's byte-sized `grow`/`shrink`
+    /// of the segmented arena below: the two resizable arenas are separate and never open together.)
+    bool elastic_grow(const std::vector<int64_t>& slot_bytes, std::string& err);
+    /// Drops the slots from `n_slots` on and unmaps every chunk wholly above the new end.  The caller has taken
+    /// every expert out of those slots and synchronized the device: nothing may read them any more.
+    bool elastic_shrink(int64_t n_slots, std::string& err);
+    uint64_t elastic_mapped_bytes() const { return mapped_; }
+    uint64_t chunk_bytes() const { return chunk_; }
+
     bool valid() const { return base_ != nullptr; }
     /// The slots usable now: all of them, unless `shrink` gave the tail's VRAM back (#533) - then the ones wholly
     /// inside the VRAM still backed, always a prefix.  The prompt path's loan (the last slots) and every report
```

### 01-5

`include/strata/core/expert_cache.hpp` · private:

Anchor:
```
    std::vector<int32_t> layer_next_;   ///< [n_layers] -> that layer's next free slot
    std::vector<uint64_t> off_;         ///< plan v0.3 P6: slot offsets (slots + 1 entries) when sized
    int64_t admitted_ = 0;
```
Block (hunk 2 of `include/strata/core/expert_cache.hpp` in the patch, 16 lines):
```diff
     std::vector<int32_t> layer_next_;   ///< [n_layers] -> that layer's next free slot
     std::vector<uint64_t> off_;         ///< plan v0.3 P6: slot offsets (slots + 1 entries) when sized
     int64_t admitted_ = 0;
+    // the elastic arena (CUdeviceptr and CUmemGenericAllocationHandle are both unsigned long long)
+    bool map_to(uint64_t end, std::string& err);   ///< maps chunks until `end` bytes are backed
+    void unmap_above(uint64_t keep);               ///< releases every chunk wholly above `keep` bytes
+    bool elastic_vmm_ = false;
+    unsigned long long va_ = 0;
+    uint64_t va_bytes_ = 0;
+    uint64_t chunk_ = 0;
+    uint64_t mapped_ = 0;
+    int device_ = 0;
+    std::vector<unsigned long long> handles_;     ///< one per mapped chunk, in address order
 };

 }  // namespace strata::core
```

### 01-6

`include/strata/prefill/prefill.hpp` · public:

Anchor:
```
    /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
    /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
    /// lent for the prompt and refilled after it); null = allocate normally.
```
Block (hunk 1 of `include/strata/prefill/prefill.hpp` in the patch, 13 lines):
```diff
     /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
     /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
     /// lent for the prompt and refilled after it); null = allocate normally.
+    /// `chunk_max` (--elastic): the largest chunk a later `relayout` may ask for - the host buffers are
+    /// sized for it, the device buffers for `chunk`; 0 = `chunk` (a cache that cannot grow never lends a bigger one).
     bool init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
               core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
               void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0,
-              bool borrow_spill = false);
+              bool borrow_spill = false, int64_t chunk_max = 0);

     /// With borrowed buffers: lay them out again for chunks of `chunk` tokens (at most `init`'s) in `borrow` - a
     /// request lends only the slots its prompt needs.  The stream must be idle (between prompts).
```

### 01-7

`src/core/expert_cache.cpp` · (file scope)

Anchor:
```
#include <cstdio>
#include <filesystem>
#include <cstdlib>
```
Block (hunk 1 of `src/core/expert_cache.cpp` in the patch, 7 lines):
```diff
 #include <cstdio>
 #include <filesystem>
 #include <cstdlib>
+#include <limits>
 #include <utility>
 #include <cstring>

```

### 01-8

`src/core/expert_cache.cpp` · bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_l

Anchor:
```
void ExpertCache::close() {
#if defined(STRATA_USE_HIP)
    if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
```
Block (hunk 2 of `src/core/expert_cache.cpp` in the patch, 174 lines):
```diff
     return true;
 }

+#if !defined(STRATA_EC_NO_VMM)   // --elastic arena: CUDA VMM, linked against the driver library
+namespace {
+bool drv_ok(CUresult r, const char* what, std::string& err) {
+    if (r == CUDA_SUCCESS) return true;
+    const char* s = nullptr;
+    cuGetErrorString(r, &s);
+    err = std::string("ExpertCache (elastic): ") + what + ": " + (s ? s : "unknown driver error");
+    return false;
+}
+CUmemAllocationProp chunk_prop(int device) {
+    CUmemAllocationProp p = {};
+    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
+    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
+    p.location.id = device;
+    return p;
+}
+}  // namespace
+
+bool ExpertCache::map_to(uint64_t end, std::string& err) {
+    const CUmemAllocationProp prop = chunk_prop(device_);
+    CUmemAccessDesc acc = {};
... 150 more lines: see 01-elastic-cache.patch
```

### 01-9

`src/core/expert_cache.cpp` · void ExpertCache::close() {

Anchor:
```
#endif
    off_.clear();
    guard_ = 0;
```
Block (hunk 3 of `src/core/expert_cache.cpp` in the patch, 22 lines):
```diff
 #endif
     off_.clear();
     guard_ = 0;
-    if (!segs_.empty()) {
+    if (elastic_vmm_) {   // the --elastic arena (open_sized_elastic)
+        unmap_above(0);
+#if !defined(STRATA_EC_NO_VMM)
+        if (va_ != 0) cuMemAddressFree((CUdeviceptr) va_, va_bytes_);
+#endif
+        va_ = 0;
+        va_bytes_ = 0;
+        chunk_ = 0;
+        mapped_ = 0;
+        elastic_vmm_ = false;
+        base_ = nullptr;   // never cudaMalloc'd
+    } else if (!segs_.empty()) {
         release_segmented();
-    } else if (vmm_) {
+    } else if (vmm_) {   // the elastic K/V's VmmRange (set_vmm)
         vmm_.reset();   // unmaps and frees every chunk it still holds
         base_ = nullptr;
     } else if (base_ != nullptr) {
```

### 01-10

`src/prefill/prefill.cpp` · uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused) {

Anchor:
```

bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                   core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
```
Block (hunk 1 of `src/prefill/prefill.cpp` in the patch, 8 lines):
```diff

 bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                    core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
-                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes, bool borrow_spill) {
+                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes, bool borrow_spill, int64_t chunk_max) {
     Impl& m = *impl_;
 #if defined(STRATA_GPU_METAL)
     (void) wt; (void) src; (void) cache; (void) host_res; (void) stream; (void) borrow;
```

### 01-11

`src/prefill/prefill.cpp` · bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co

Anchor:
```
    m.T = chunk > 0 ? chunk : 1;
    return strata::metal_engine::window_ensure(g, ss, ss.qsa_states[ss.qsa_primary()].max_cells, err);
#endif
```
Block (hunk 2 of `src/prefill/prefill.cpp` in the patch, 9 lines):
```diff
     m.T = chunk > 0 ? chunk : 1;
     return strata::metal_engine::window_ensure(g, ss, ss.qsa_states[ss.qsa_primary()].max_cells, err);
 #endif
+    // the host side is sized for the largest chunk a relayout may ask for; the device side (carved below from the
+    // borrowed slots) for this one
+    const int64_t cap_T = std::max<int64_t>(chunk, chunk_max);
     m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
     m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
     if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
```

### 01-12

`src/prefill/prefill.cpp` · bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co

Anchor:
```
    // the CPU share's staged-chunk limit, before the ring is sized below (a caller that sizes loans first armed it)
    arm_cpu_share(cpu_pool_ != nullptr);
    for (int b = 0; next_ != nullptr && b < 2; ++b)
```
Block (hunk 3 of `src/prefill/prefill.cpp` in the patch, 8 lines):
```diff
     // the CPU share's staged-chunk limit, before the ring is sized below (a caller that sizes loans first armed it)
     arm_cpu_share(cpu_pool_ != nullptr);
     for (int b = 0; next_ != nullptr && b < 2; ++b)
-        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4, cudaHostAllocPortable) != cudaSuccess) {
+        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) cap_T * D * 4, cudaHostAllocPortable) != cudaSuccess) {
             err = "prefill: the layer split's hand-off buffers";
             return false;
         }
```

### 01-13

`src/prefill/prefill.cpp` · bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co

Anchor:
```
        m.hand_pieces = handoff_pipe_pieces();
    }
    if (m.tok_dev == nullptr) {
```
Block (hunk 4 of `src/prefill/prefill.cpp` in the patch, 22 lines):
```diff
         m.hand_pieces = handoff_pipe_pieces();
     }
     if (m.tok_dev == nullptr) {
-        if (const cudaError_t e = cudaMalloc((void**) &m.tok_dev, (size_t) chunk * sizeof(int32_t)); e != cudaSuccess) {
+        if (const cudaError_t e = cudaMalloc((void**) &m.tok_dev, (size_t) cap_T * sizeof(int32_t)); e != cudaSuccess) {
             err = std::string("prefill: the token id buffer (") + cudaGetErrorString(e) + ")";
             return false;
         }
         m.owned.push_back(m.tok_dev);
-        m.tok_host.resize((size_t) chunk);
-        m.tok_up.init((size_t) chunk);
+        m.tok_host.resize((size_t) cap_T);
+        m.tok_up.init((size_t) cap_T);
     }
     if (cudaStreamCreateWithFlags(&m.copy, cudaStreamNonBlocking) != cudaSuccess) { err = "prefill: copy stream"; return false; }
-    const size_t T = (size_t) chunk;
-    m.T_max = chunk;
+    const size_t T = (size_t) cap_T;   // the host buffers below; `carve` gets this chunk's own size
+    m.T_max = cap_T;
     m.borrowed = borrow != nullptr;
     m.borrow_spill = m.borrowed && borrow_spill;
     bool ok = true;
```

### 01-14

`src/prefill/prefill.cpp` · bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co

Anchor:
```
        static const bool xn_env = [] { const char* e = std::getenv("STRATA_XN_F16"); return e == nullptr || e[0] != '0'; }();
        m.xn_f16 = xn_env && !m.f16_io && m.gemm.bf16_converts_x();
    }
```
Block (hunk 5 of `src/prefill/prefill.cpp` in the patch, 8 lines):
```diff
         static const bool xn_env = [] { const char* e = std::getenv("STRATA_XN_F16"); return e == nullptr || e[0] != '0'; }();
         m.xn_f16 = xn_env && !m.f16_io && m.gemm.bf16_converts_x();
     }
-    if (!carve(T, &o)) {
+    if (!carve((size_t) chunk, &o)) {
         size_t fb = 0, tb = 0;
         cudaMemGetInfo(&fb, &tb);
         (void) cudaGetLastError();
```

### 01-15

`src/program/generate.cpp` · struct Options {

Anchor:
```
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 23 lines):
```diff
     /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
     /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
     int vram_reserve_mib = 700;
+    /// --elastic (opt-in): an AUTOMATIC elastic expert cache (`--serve`, one GPU).  The startup sizes the cache with
+    /// `--vram-reserve-mib` as before, then the cache follows the free VRAM while it runs: it keeps
+    /// `elastic_reserve_mib` free, grows with the most-used missing experts (then the profile's order) once the
+    /// VRAM has stayed free for `elastic_stable_ms`, and shrinks at once when it has not - the tail's hottest
+    /// experts moving into colder slots first, never a flush.  Checked every `elastic_period_ms` at the decode's
+    /// safe point and at every request's start.
+    bool elastic = false;
+    int elastic_reserve_mib = 500;
+    int elastic_chunk_mib = 64;
+    int elastic_period_ms = 1000;
+    int elastic_stable_ms = 5000;
+    /// The TWO-ZONE cache.  The first slots (filled from the profile at startup) are the CORE: the adaptive swaps
+    /// never pick their experts to leave, the elastic cache never shrinks below them and the prompt path never
+    /// borrows them.  Only the TAIL - at most this many MiB of the startup cache, -1 = min(6144, half of it), 0 = no
+    /// core - rotates, shrinks, grows and is lent.  A shrink gives back at most `elastic_shrink_step_mib` per check.
+    int elastic_tail_mib = -1;
+    int elastic_shrink_step_mib = 1024;
     bool vram_reserve_given = false;   ///< --vram-reserve-mib on the command line (#496: no smaller automatic reserve)
     /// #533 (opt-in): the expert cache in physical segments (CUDA virtual memory management), so the serve loop's
     /// `VRAM <reserve_mib>` command can give part of it back to other programs and take it again.  Off: one cudaMalloc.
```

### 01-16

`src/program/generate.cpp` · void usage() {

Anchor:
```
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                 "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
                  "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                  "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                  "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
+                 "  --elastic           --serve, single CUDA GPU: resize the expert-cache tail (default off)\n"
+                 "  --elastic-reserve-mib N  free-memory target (default 500, >=0)\n"
+                 "  --elastic-chunk-mib N    VMM mapping chunk (default 64, >=1)\n"
+                 "  --elastic-period-ms N    minimum policy-check interval (default 1000, >=0)\n"
+                 "  --elastic-stable-ms N    sustained room before growth (default 5000, >=0)\n"
+                 "  --elastic-tail-mib N     initial tail limit (-1 auto, 0 no fixed core, >0 MiB)\n"
+                 "  --elastic-shrink-step-mib N  maximum reduction per check (default 1024, >=1)\n"
                  "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =\n"
                  "                       the largest chunk up to 8192 whose buffers the expert cache can lend;\n"
                  "                       auto:16384 / auto:32768 (or STRATA_PREFILL_AUTO_MAX) allow bigger ones\n"
```

### 01-17

`src/program/generate.cpp` · void usage() {

Anchor:
```
                 "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                 "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                 "                       ms/token, -2.7 ms/token end to end.\n"
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 18 lines):
```diff
                  "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                  "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                  "                       ms/token, -2.7 ms/token end to end.\n"
+                 "  --elastic            (serve, one GPU) the expert cache follows the free VRAM while it runs:\n"
+                 "                       grows with the most-used missing experts, shrinks the least-used first;\n"
+                 "                       automatic, never together with --vram-elastic (#533, resized on request)\n"
+                 "  --elastic-reserve-mib N  VRAM it keeps free (default 500)\n"
+                 "  --elastic-chunk-mib N    granularity of growing and shrinking (default 64)\n"
+                 "  --elastic-period-ms N    how often it checks (default 1000)\n"
+                 "  --elastic-stable-ms N    how long the room must last before it grows (default 5000; doubles after a\n"
+                 "                           shrink that follows a growth, back to N after 5 minutes without one)\n"
+                 "  --elastic-tail-mib N     two zones: only this much of the startup cache (its tail) rotates, shrinks\n"
+                 "                           and is lent to the prompt path; the rest is a fixed core (default\n"
+                 "                           min(6144, half the cache); 0 = the whole cache is tail)\n"
+                 "  --elastic-shrink-step-mib N  at most this much given back per check (default 1024)\n"
                  "  --expert-cache-device1 N|auto  fill CUDA1 with N experts or as many as fit\n"
                  "  --expert-cache-device2 N|auto  fill CUDA2 with more experts\n"
                  "  --expert-cache-device3 N|auto  fill CUDA3 with more experts\n"
```

### 01-18

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
        else if (a == "--vram-reserve-later-mib")
            o.vram_reserve_later_mib = std::atoi(next("--vram-reserve-later-mib"));
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 26 lines):
```diff
         else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
         else if (a == "--vram-reserve-later-mib")
             o.vram_reserve_later_mib = std::atoi(next("--vram-reserve-later-mib"));
+        else if (a == "--elastic") o.elastic = true;
+        else if (a == "--elastic-reserve-mib" || a == "--elastic-chunk-mib" ||
+                 a == "--elastic-period-ms" || a == "--elastic-stable-ms" ||
+                 a == "--elastic-tail-mib" || a == "--elastic-shrink-step-mib") {
+            const std::string value = next(a.c_str());
+            int number = 0;
+            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
+            const int minimum = a == "--elastic-tail-mib" ? -1 :
+                (a == "--elastic-chunk-mib" || a == "--elastic-shrink-step-mib") ? 1 : 0;
+            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || number < minimum) {
+                std::fprintf(stderr, "%s needs an integer >= %d within int32 range\n", a.c_str(), minimum);
+                return 2;
+            }
+            if (a == "--elastic-reserve-mib") o.elastic_reserve_mib = number;
+            else if (a == "--elastic-chunk-mib") o.elastic_chunk_mib = number;
+            else if (a == "--elastic-period-ms") o.elastic_period_ms = number;
+            else if (a == "--elastic-stable-ms") o.elastic_stable_ms = number;
+            else if (a == "--elastic-tail-mib") o.elastic_tail_mib = number;
+            else o.elastic_shrink_step_mib = number;
+        }
         else if (a == "--prefill") {
             const std::string v = next("--prefill");
             o.prefill_auto = v == "auto" || v.rfind("auto:", 0) == 0;
```

### 01-19

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
    strata::core::set_peer_portable(o.peer_device >= 1);   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
    {   // the RAM guard (parking, session save/restore): --memory-limit-mib, else STRATA_MEMORY_LIMIT_MIB, else none
        int64_t mib = o.memory_limit_mib;
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 17 lines):
```diff
             return 2;
         }
     }
+    if (o.elastic && (o.peer_device >= 1 || std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
+                                                     [](int slots) { return slots > 0; }))) {
+        std::fprintf(stderr, "strata: elastic cache does not support peer or remote expert tiers; using fixed cache\n");
+        o.elastic = false;
+    }
+    // --batch: the elastic step runs between two batch windows and at a request's start, before its prompt lends any
+    // slot; the pipelined groups (--batch-groups > 1) keep windows of several stages in flight: not with it
+    if (o.elastic && o.batch != 0 && o.batch_groups > 1) {
+        std::fprintf(stderr, "strata: elastic cache is not validated with --batch-groups; using fixed cache\n");
+        o.elastic = false;
+    }
     strata::core::set_peer_portable(o.peer_device >= 1);   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
     {   // the RAM guard (parking, session save/restore): --memory-limit-mib, else STRATA_MEMORY_LIMIT_MIB, else none
         int64_t mib = o.memory_limit_mib;
```

### 01-20

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            !o.expert_profile.empty() && !o.resident_cpu_experts && o.expert_cache != 0 && !remote &&
                            strata::core::vmm_available() &&
                            // the batch slots carve their own K/V and --vram-elastic's cache is not one VMM range
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 14 lines):
```diff
                             !o.expert_profile.empty() && !o.resident_cpu_experts && o.expert_cache != 0 && !remote &&
                             strata::core::vmm_available() &&
                             // the batch slots carve their own K/V and --vram-elastic's cache is not one VMM range
-                            o.batch == 0 && !o.vram_elastic && o.peer_device < 0;
-            if (asked && !on)
+                            o.batch == 0 && !o.vram_elastic && o.peer_device < 0 &&
+                            // --elastic's arena is not the VmmRange the K/V takes its room from: never together
+                            !(o.elastic && o.serve);
+            if (asked && !on && o.elastic && o.serve)
+                std::fprintf(stderr, "strata generate: --kv-grow is off: --elastic resizes the expert cache itself\n");
+            else if (asked && !on)
                 std::fprintf(stderr, "strata generate: --kv-grow is off (one GPU, a profile, the whole K/V in VRAM, "
                                      "every expert in RAM, no --batch, --vram-elastic or --peer-device)\n");
             const char* iv = std::getenv("STRATA_KV_GROW_INIT");
```

### 01-21

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        : multi_gpu ? "a layer split has a cache per GPU"
                        : std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                      [](int n) { return n > 0; }) ? "the helper caches on other GPUs"
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                         : multi_gpu ? "a layer split has a cache per GPU"
                         : std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                       [](int n) { return n > 0; }) ? "the helper caches on other GPUs"
-                        : o.vram_segment_mib < 64 ? "--vram-segment-mib is below 64" : nullptr;
+                        : o.vram_segment_mib < 64 ? "--vram-segment-mib is below 64"
+                        : (o.elastic && !o.resident_cpu_experts) ? "--elastic (the automatic elastic cache) resizes it"
+                        : nullptr;
 #if defined(STRATA_USE_HIP)
         if (why == nullptr) why = "it is NVIDIA-only for now";
 #endif
```

### 01-22

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
        // free figure read again; while it is short of the reserve the cache is reopened smaller.
        // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 46 lines):
```diff
         // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
         // free figure read again; while it is short of the reserve the cache is reopened smaller.
         // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
+        if (o.elastic && (!o.serve || multi_gpu || o.resident_cpu_experts))
+            std::fprintf(stderr, "strata: elastic cache requires single-GPU serve without resident CPU complement; using fixed cache\n");
         int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL") ? std::atoi(std::getenv("STRATA_TEST_CACHE_FAIL")) : 0;
         int failed = 0;
         int zero_reads = 0;
+        int el_oom_tries = 0;   // --elastic: smaller arenas tried after an out-of-memory map
         for (int attempt = 0;; ++attempt) {
             bool ok = false;
             if (fake_fails > 0) {
                 --fake_fails;
                 err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
             } else {
-                ok = sized_slots.empty()
-                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) strata::kernels::cpu::expert_layout().max_blob, err)
-                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
+                if (!sized_slots.empty() && o.elastic && o.serve && !multi_gpu && !o.resident_cpu_experts) {
+                    // --elastic: the elastic arena reserves the whole card's address range; without VMM, the plain one
+                    size_t fb = 0, tb = 0;
+                    cudaMemGetInfo(&fb, &tb);
+                    ok = xcache.open_sized_elastic(sized_slots, g.n_layers, g.n_expert, (uint64_t) tb,
+                                                   (uint64_t) o.elastic_chunk_mib << 20, err);
+                    // the arena maps whole chunks: a card a little tighter than the sizing read (the --batch slots,
+                    // a desktop program) fails its last ones; a smaller arena stays elastic, the fixed cache is last
+                    if (!ok && auto_cache && el_oom_tries < 4 && err.find("out of memory") != std::string::npos &&
+                        shrink_to(cache_bytes() - (512ll << 20))) {
+                        ++el_oom_tries;
+                        std::fprintf(stderr, "strata generate: %s; trying a smaller elastic cache: %d slots\n",
+                                     err.c_str(), o.expert_cache);
+                        continue;
+                    }
+                    if (!ok) {
+                        std::fprintf(stderr, "strata generate: %s - the cache stays fixed\n", err.c_str());
+                        o.elastic = false;
+                        ok = xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
+                    }
+                } else {
+                    ok = sized_slots.empty()
+                        ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) strata::kernels::cpu::expert_layout().max_blob, err)
+                        : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
+                }
             }
             if (!ok) {
                 // Issue #60: on Windows a device allocation is also charged to the system commit (RAM + page file),
```

### 01-23

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 18 lines):
```diff
         const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
         return std::min(max_chunk, rounded);
     };
+    // --elastic: the TWO-ZONE cache.  Slots [0, el_core) are the core - filled from the profile at startup, never
+    // picked by the adaptive swaps to leave, never shrunk, never lent; the tail after it is the elastic zone.
+    int64_t el_core = 0;
+    if (o.elastic && o.serve && xcache.elastic() && !multi_gpu && xcache.slot_offsets() != nullptr) {
+        const uint64_t* off = xcache.slot_offsets();
+        const int64_t S = xcache.slots();
+        const uint64_t tail = o.elastic_tail_mib >= 0 ? (uint64_t) o.elastic_tail_mib << 20
+                                                      : std::min<uint64_t>(6144ull << 20, off[S] / 2);
+        int64_t c = S;
+        while (c > 0 && off[S] - off[c - 1] <= tail) --c;   // the smallest core whose tail fits in `tail`
+        el_core = tail == 0 ? 0 : c;
+    }
     // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits,
     // or with --prefill auto the largest chunk whose buffers take at most kAutoLendPct % of the slots (a
     // lent slot's expert is streamed during the prompt and refilled after it; measured on a 12 GB card, 32K Q2_0
```

### 01-24

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
    };
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        auto slots_for = lend_slots;
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
     };
     auto plan_lend = [&](int64_t& chunk) -> int64_t {
         auto slots_for = lend_slots;
+        const int64_t tail_slots = xcache.slots() - el_core;   // --elastic: only the tail is lent (all of it without a core)
         // 0.1.39's list with its ring (also the size up to which a prompt keeps that ring under #583)
         auto old_rule = [&]() -> int64_t {
             strata::prefill::Prefill::set_ring_budget(0, 0);
```

### 01-25

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;
                if (o.prefill_auto_cap > 0 && c > o.prefill_auto_cap) continue;   // #1630
                const int64_t k = slots_for(c);
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                 if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;
                 if (o.prefill_auto_cap > 0 && c > o.prefill_auto_cap) continue;   // #1630
                 const int64_t k = slots_for(c);
-                if (k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots()) return c;
+                if (k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots() && k <= tail_slots) return c;
             }
             return 0;
         };
```

### 01-26

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            constexpr int64_t kRingMin = 16;
            const int64_t small = old_rule();   // 0.1.39's chunk (and its ring for the prompts that fit it)
            const int64_t ring_max = strata::prefill::Prefill::ring_cap_for(small);
```
Block (hunk 12 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             constexpr int64_t kRingMin = 16;
             const int64_t small = old_rule();   // 0.1.39's chunk (and its ring for the prompts that fit it)
             const int64_t ring_max = strata::prefill::Prefill::ring_cap_for(small);
-            const int64_t budget = std::min(xcache.slots() - 128, kAutoLendPct * xcache.slots() / 100);
+            const int64_t budget = std::min({xcache.slots() - 128, kAutoLendPct * xcache.slots() / 100, tail_slots});
             // The room is a BYTE budget.  A ring slot is max_blob bytes (`carve` lays out one whole blob each),
             // while these cache slots hold their own layer's blob, which is smaller than max_blob unless the cache
             // happens to hold the pack's biggest layer - on the 4-way IQ3_S rig that is 2.15 MiB a slot against a
```

### 01-27

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which case the
                    // chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                    const int64_t k = slots_for(t);
```
Block (hunk 13 of `src/program/generate.cpp` in the patch, 9 lines):
```diff
                     // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which case the
                     // chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                     const int64_t k = slots_for(t);
-                    return k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots();
+                    return k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots() &&
+                           k <= tail_slots;
                 });
             };
             int64_t c = scan(ring_max);
```

### 01-28

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        }
        for (int64_t c = chunk; c >= 256; c /= 2) {
            const int64_t k = slots_for(c);
```
Block (hunk 14 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
         }
         for (int64_t c = chunk; c >= 256; c /= 2) {
             const int64_t k = slots_for(c);
-            if (k + 128 <= xcache.slots()) { chunk = c; return k; }
+            if (k + 128 <= xcache.slots() && k <= tail_slots) { chunk = c; return k; }
         }
         return 0;
     };
```

### 01-29

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
            auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                const int64_t k = part_slots(p, c);
```
Block (hunk 15 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
             // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
             auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                 const int64_t k = part_slots(p, c);
+                if (p.cache == &xcache && k > xcache.slots() - el_core) return false;
                 if (k <= 0 || k + 128 > p.cache->slots()) return false;
                 return !(cap && k * 100 > kAutoLendPct * p.cache->slots());
             };
```

### 01-30

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    // clamped at 0: a cache under 128 slots has a negative lend budget, and `slots - budget` is
                    // then `slots`, which is off_[slots] - the end of the prefix sum, so that cache lends 0 bytes
                    // rather than reading one entry past its offsets
```
Block (hunk 16 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
                     // clamped at 0: a cache under 128 slots has a negative lend budget, and `slots - budget` is
                     // then `slots`, which is off_[slots] - the end of the prefix sum, so that cache lends 0 bytes
                     // rather than reading one entry past its offsets
-                    const int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - 128,
-                                                                        kAutoLendPct * p.cache->slots() / 100));
+                    int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - 128,
+                                                                  kAutoLendPct * p.cache->slots() / 100));
+                    // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
+                    if (p.cache == &xcache) budget = std::min(budget, std::max<int64_t>(0, xcache.slots() - el_core));
                     const uint64_t avail = part_bytes(p, (int32_t) (p.cache->slots() - budget));
                     const uint64_t nonring = strata::prefill::Prefill::bytes_needed_no_ring(g, *p.ses, c, srcp != nullptr);
                     room = std::min(room, (int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) kBlob));
```

### 01-31

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
        // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
        // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
```
Block (hunk 17 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
         // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
         // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
         // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
+        const int64_t sp_chunk_max = (o.elastic && xcache.elastic() && o.prefill_auto && borrow != nullptr) ? o.prefill_auto_max : 0;
         auto init_prompt_paths = [&]() -> int {   // 0: ready; 1: failed (err set); 2: failed with "do not fit"
             for (size_t i = 0; i < stages.size(); ++i) {
                 GpuStage& st = *stages[i];
```

### 01-32

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
            if (share_pool) sp.set_cpu_pool(&pool);
            if (!sp.init(wt, g, ss, prefill_src, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes,
```
Block (hunk 18 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
             if (share_pool) sp.set_cpu_pool(&pool);
             if (!sp.init(wt, g, ss, prefill_src, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes,
-                         o.prefill_head_alias))
+                         o.prefill_head_alias, sp_chunk_max))
                 return err.find("do not fit") != std::string::npos ? 2 : 1;
             return 0;
         };
```

### 01-33

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
```
Block (hunk 19 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                 const int32_t* r = host_res.data() + l * g.n_expert;
                 for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                     if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
-                    else vict.emplace_back(u[e], e);
+                    else if (r[e] >= el_core) vict.emplace_back(u[e], e);   // --elastic: the core never leaves
                 }
                 if (cand.empty() || vict.empty()) continue;
                 std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
```

### 01-34

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            for (float& v : drive.d.usage) v *= o.adapt_decay;
            return true;
        };
```
Block (hunk 20 of `src/program/generate.cpp` in the patch, 239 lines):
```diff
             for (float& v : drive.d.usage) v *= o.adapt_decay;
             return true;
         };
+        // ---- --elastic: THE AUTOMATIC ELASTIC EXPERT CACHE.  The arena is a reserved address range backed by
+        // chunks (ExpertCache::open_sized_elastic), so its slots can be added or dropped at the TAIL without any
+        // pointer moving.  `host_res` stays the one truth about which expert sits where (the adaptive tier writes it
+        // too); the prompt path borrows the tail, so after the tail moved its buffers are laid out again (and a cache
+        // that shrank too far lends a smaller chunk, the same rule as plan_lend).  Run only where nothing is in
+        // flight: a request's start (nothing lent yet) and the decode's safe point (window, commit, draft and the
+        // adaptive swaps all finished).
+        const bool el_on = o.elastic && xcache.elastic() && d_res != nullptr && !multi_gpu && !host_res.empty();
+        const int64_t el_reserve = (int64_t) o.elastic_reserve_mib << 20;
+        const int64_t el_hyst_min = 192ll << 20;   // growth only past the reserve plus max(this, the measured noise)
+        static const bool el_trace = std::getenv("STRATA_ELASTIC_TRACE") != nullptr;
+        const int64_t el_step_max = 512ll << 20;   // at most this much per growth step (its copies are async)
+        const int64_t el_shrink_step = std::max<int64_t>(64ll << 20, (int64_t) o.elastic_shrink_step_mib << 20);
+        const int64_t el_min_slots = std::max<int64_t>(256, el_core);   // the core is never shrunk
+        const int64_t el_chunk_max = sp_chunk_max > 0 ? sp_chunk_max : o.prefill_chunk;
+        Clock::time_point el_last{};
+        Clock::time_point el_stable_since = Clock::now();
+        // growth against noise: the free readings since the last resize (a resize changes the level, so it clears
+        // them); a growth must hold on the MINIMUM of the stable window and leaves reserve + margin free, where the
+        // margin is the spread the readings have shown.  A shrink soon after a growth doubles the wait (at most 8x),
+        // five quiet minutes restore it.
... 215 more lines: see 01-elastic-cache.patch
```

### 01-35

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            if (batch_on() || (piped && pipe_inflight())) {
                if (!try_next_line(line)) {
                    if (!(piped ? pump(true) : batch_step())) return 1;
```
Block (hunk 21 of `src/program/generate.cpp` in the patch, 15 lines):
```diff
             if (batch_on() || (piped && pipe_inflight())) {
                 if (!try_next_line(line)) {
                     if (!(piped ? pump(true) : batch_step())) return 1;
+                    // --elastic: the cache's step between two batch windows (nothing lent, no window in flight), as
+                    // after a round of the single-request decode
+                    if (!piped) {
+                        std::string ee;
+                        if (!elastic_step(false, ee)) {
+                            std::printf("ERR %s\n", ee.c_str());
+                            return 1;
+                        }
+                    }
                     continue;
                 }
                 // a request reads its prompt through every stage: the groups in flight finish first
```

### 01-36

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            auto lend = [&](int64_t tokens, std::string& e, int64_t exact) -> bool {
                seg_lend = 0;
                if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
```
Block (hunk 22 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
             auto lend = [&](int64_t tokens, std::string& e, int64_t exact) -> bool {
                 seg_lend = 0;
                 if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
+                if (el_on) apply_pending(true);   // elastic fills must be published before the tail is lent
                 const auto t_ln = Clock::now();
                 // what this segment needs, capped by the configured chunk: a request lends only what its own
                 // prompt needs, so a large chunk costs a short prompt nothing
```

### 01-37

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                return true;
            };
            apply_pending(true);
```
Block (hunk 23 of `src/program/generate.cpp` in the patch, 14 lines):
```diff
                 return true;
             };
             apply_pending(true);
+            // --elastic: the cache follows the VRAM before the prompt borrows its tail (nothing is lent yet)
+            {
+                std::string ee;
+                if (!elastic_step(true, ee)) {
+                    std::printf("ERR %s\n", ee.c_str());
+                    return 1;
+                }
+            }
             if (!adapt_tick(true)) {   // --adapt-async: the round in flight lands before the prompt lends any slot
                 std::printf("ERR an adaptive refill failed\n");
                 return 1;
```

### 01-38

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                         std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                if (eos) { finish = "stop"; break; }
                if (stop_req.load()) { finish = "cancel"; break; }
```
Block (hunk 24 of `src/program/generate.cpp` in the patch, 14 lines):
```diff
                                          std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                 if (eos) { finish = "stop"; break; }
                 if (stop_req.load()) { finish = "cancel"; break; }
+                // --elastic: the cache's step, after the round's timing is recorded (the draft policy reads it)
+                {
+                    std::string ee;
+                    if (!elastic_step(false, ee)) {
+                        std::printf("ERR %s\n", ee.c_str());
+                        return 1;
+                    }
+                }
                 x = outv[(size_t) a];
                 p += a + 1;
             }
```

### 01-39

`tests/core/expert_cache_elastic_test.cpp` · (new file)

New file, 71 lines: see the patch.

## 02 split-prompt-borrow

`02-split-prompt-borrow.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la stima del prestito del prompt sul device del proprio blocco (PR #1417)

### 02-1

`src/prefill/prefill.cpp` · bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co

Anchor:
```
        err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
              std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
              std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
```
Block (hunk 1 of `src/prefill/prefill.cpp` in the patch, 9 lines):
```diff
         err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
               std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
               std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
-              std::to_string(o.failed_bytes >> 20) + " MiB)";
+              std::to_string(o.failed_bytes >> 20) + " MiB" +
+              (o.base != nullptr ? "; the loan holds " + std::to_string(o.cap >> 20) + " MiB" : std::string()) + ")";
         return false;
     }
     return true;
```

### 02-2

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                                     (uint64_t) strata::kernels::cpu::expert_layout().max_blob, need);
        };
        auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 9 lines):
```diff
                                                      (uint64_t) strata::kernels::cpu::expert_layout().max_blob, need);
         };
         auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
+            // counted on the part's own device, as `init` and the other `bytes_needed` callers in this block are: the
+            // count follows `init`, whose per-device choices (prompt_f16) are the stage device's, not CUDA0's
+            const strata::core::OnDevice on(p.dev);
             return cache_slots_for(*p.cache, strata::prefill::Prefill::bytes_needed(g, *p.ses, c, srcp != nullptr));
         };
         auto part_bytes = [&](const PfPart& p, int32_t first) -> uint64_t {
```

## 03 forced-bench

`03-forced-bench.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il banco forzato STRATA_FORCE_IDS/STRATA_FORCE_WINDOWS/STRATA_FORCE_TRACE (PR #1419)

### 03-1

`src/program/generate.cpp` · static const char* pipe_dbg_env(const char* name) {

Anchor:
```
    return on ? std::getenv(name) : nullptr;
}
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 79 lines):
```diff
     return on ? std::getenv(name) : nullptr;
 }

+// lab bench (deterministic work), STRATA_FORCE_IDS=<file>[;<file>...] (opt-in): the k-th generation request of the
+// process (GEN/GENI, counted from 0) generates exactly the token ids of the k-th file - int32 little-endian, or a text
+// list (decimal ids separated by spaces, commas, newlines, brackets) - whatever the target picks.  Requests past the
+// last file are not forced.  STRATA_FORCE_TRACE=<file>: one line per verified window appended ("W <request> <pos>
+// <rows> <accepted> <row tokens...>") to compare two runs.
+static const std::vector<std::vector<int32_t>>& force_id_lists() {
+    static const std::vector<std::vector<int32_t>> lists = [] {
+        std::vector<std::vector<int32_t>> out;
+        const char* v = std::getenv("STRATA_FORCE_IDS");
+        if (v == nullptr || v[0] == 0) return out;
+        std::string all(v);
+        size_t at = 0;
+        while (at <= all.size()) {
+            const size_t semi = all.find(';', at);
+            const std::string path = all.substr(at, semi == std::string::npos ? std::string::npos : semi - at);
+            at = semi == std::string::npos ? all.size() + 1 : semi + 1;
+            if (path.empty()) continue;
+            std::ifstream f(path, std::ios::binary);
+            const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
+            std::vector<int32_t> ids;
+            const bool text = !bytes.empty() && bytes.find_first_not_of("0123456789 ,;\r\n\t[]") == std::string::npos;
+            if (text) {
+                long long cur = -1;
+                for (const char c : bytes) {
+                    if (c >= '0' && c <= '9') cur = (cur < 0 ? 0 : cur * 10) + (c - '0');
+                    else if (cur >= 0) { ids.push_back((int32_t) cur); cur = -1; }
+                }
+                if (cur >= 0) ids.push_back((int32_t) cur);
+            } else {
+                ids.resize(bytes.size() / 4);
+                if (!ids.empty()) std::memcpy(ids.data(), bytes.data(), ids.size() * 4);
+            }
+            std::fprintf(stderr, "strata force: list %zu: %s, %zu ids (%s)\n", out.size(), path.c_str(), ids.size(),
+                         text ? "text" : "int32");
+            out.push_back(std::move(ids));
+        }
+        return out;
+    }();
+    return lists;
+}
+
+// lab bench, STRATA_FORCE_WINDOWS=<file> (opt-in, with STRATA_FORCE_IDS): a reference run's STRATA_FORCE_TRACE replayed -
+// a verified window that starts at a position the trace has takes the trace's rows (size and draft tokens) instead
+// of the drafter's, so the verified work is the reference's by construction.  The drafter still runs (its cost stays).
+// Per request (the trace's request index), position -> rows.
+static const std::vector<std::map<int64_t, std::vector<int32_t>>>& force_window_lists() {
+    static const std::vector<std::map<int64_t, std::vector<int32_t>>> lists = [] {
+        std::vector<std::map<int64_t, std::vector<int32_t>>> out;
+        const char* v = std::getenv("STRATA_FORCE_WINDOWS");
+        if (v == nullptr || v[0] == 0) return out;
+        std::ifstream f(v);
+        std::string line;
+        int64_t n = 0;
+        while (std::getline(f, line)) {
+            std::istringstream ls(line);
+            std::string tag;
+            long long req = -1, pos = -1;
+            int T = 0, acc = 0;
+            if (!(ls >> tag >> req >> pos >> T >> acc) || tag != "W" || req < 0 || T < 1) continue;
+            std::vector<int32_t> rows((size_t) T);
+            bool ok = true;
+            for (int i = 0; i < T && ok; ++i) ok = (bool) (ls >> rows[(size_t) i]);
+            if (!ok) continue;
+            if ((size_t) req >= out.size()) out.resize((size_t) req + 1);
+            out[(size_t) req][pos] = std::move(rows);
+            ++n;
+        }
+        std::fprintf(stderr, "strata force: windows of %s: %lld windows, %zu requests\n", v, (long long) n, out.size());
+        return out;
+    }();
+    return lists;
+}
+
 // The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
 // by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
 // the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
```

### 03-2

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                continue;
            }
            char* endp = nullptr;
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                 continue;
             }
             char* endp = nullptr;
-            const long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);
+            long long max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10);   // (STRATA_FORCE_IDS may lower it)
             // optional sampling keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F,
             // penalty_last_n=N, penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N (text requests
             // only).  Absent keys keep today's behavior: greedy, no penalties.
```

### 03-3

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
            std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
            std::fflush(stdout);
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 70 lines):
```diff
             const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
             std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
             std::fflush(stdout);
+            // lab bench: STRATA_FORCE_IDS - this request's forced ids (see force_id_lists), its windows' rows and acceptance
+            static int64_t force_req = 0;
+            const int64_t force_k = force_req++;
+            const std::vector<int32_t>* forced = nullptr;
+            if (force_k < (int64_t) force_id_lists().size() && !force_id_lists()[(size_t) force_k].empty()) {
+                forced = &force_id_lists()[(size_t) force_k];
+                if ((long long) forced->size() < max_new) max_new = (long long) forced->size();
+                std::fprintf(stderr, "strata force: request %lld generates the %zu ids of list %lld (max_new %lld)\n",
+                             (long long) force_k, forced->size(), (long long) force_k, max_new);
+            }
+            const std::map<int64_t, std::vector<int32_t>>* force_win =
+                force_k < (int64_t) force_window_lists().size() && !force_window_lists()[(size_t) force_k].empty()
+                    ? &force_window_lists()[(size_t) force_k] : nullptr;
+            int64_t force_win_hit = 0, force_win_miss = 0;
+            // STRATA_FORCE_WINDOWS: the reference's rows for a window at `pos` whose row 0 is `row0`: false when the
+            // trace has no window there (or another row 0: a speculative guess the reference never made)
+            auto force_window = [&](int64_t pos, int32_t row0, int& T, int32_t* rows, int max_t) -> bool {
+                if (force_win == nullptr) return false;
+                const auto it = force_win->find(pos);
+                if (it == force_win->end() || it->second.empty() || it->second[0] != row0 ||
+                    (int) it->second.size() > max_t) {
+                    ++force_win_miss;
+                    return false;
+                }
+                T = (int) it->second.size();
+                for (int i = 0; i < T; ++i) rows[i] = it->second[(size_t) i];
+                ++force_win_hit;
+                return true;
+            };
+            std::vector<int64_t> force_rows(16, 0);   // windows by rows
+            int64_t force_windows = 0, force_acc = 0, force_off = 0, force_over = 0;
+            // the last forced window's rows: the target's own picks and the forced ids that replaced them
+            std::vector<int32_t> force_orig(64, -1), force_val(64, -1);
+            static std::FILE* force_trace = [] {
+                const char* v = std::getenv("STRATA_FORCE_TRACE");
+                return v != nullptr && v[0] != 0 ? std::fopen(v, "a") : nullptr;
+            }();
+            // the target's picks for the rows of a window starting at generated index `at` replaced by the forced ids
+            // (`over`: how many differed); then a verified window's rows and acceptance counted and traced
+            auto force_rows_out = [&](int32_t* out, int T, int64_t at) {
+                if (forced == nullptr) return;
+                for (int i = 0; i < T; ++i) {
+                    force_orig[(size_t) i] = force_val[(size_t) i] = -1;
+                    const int64_t k = at + i;
+                    if (k >= (int64_t) forced->size()) break;
+                    force_orig[(size_t) i] = out[i];
+                    force_val[(size_t) i] = (*forced)[(size_t) k];
+                    out[i] = (*forced)[(size_t) k];
+                }
+            };
+            auto force_note = [&](int64_t pos, int T, int acc, const int32_t* rows) {
+                if (forced != nullptr)   // a verified row (on the forced path) where the target picked another id
+                    for (int i = 0; i <= acc && i < T; ++i)
+                        if (force_val[(size_t) i] >= 0 && force_orig[(size_t) i] != force_val[(size_t) i]) ++force_over;
+                force_rows[(size_t) std::min<int>(T, 15)] += 1;
+                ++force_windows;
+                force_acc += acc;
+                force_off += T - 1;
+                if (force_trace != nullptr) {
+                    std::fprintf(force_trace, "W %lld %lld %d %d", (long long) force_k, (long long) pos, T, acc);
+                    for (int i = 0; i < T; ++i) std::fprintf(force_trace, " %d", (int) rows[i]);
+                    std::fputc('\n', force_trace);
+                }
+            };
             // the verify windows: the first holds the last prompt token alone (STRATA_ADMIT_MERGE_FIRST: the prompt
             // tokens from merge_at on, the last one's output its first token)
             const int merge_T = merge_at >= 0 && !cancelled ? (int) (n - merge_at) : 0;
```

### 03-4

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        if (v.service(&drive_pool_split, SDf(A), err) < 0) return die(err);
                        if (v.done(err)) {
                            if (!v.pl_finish(outp.data(), err)) return die(err);
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                         if (v.service(&drive_pool_split, SDf(A), err) < 0) return die(err);
                         if (v.done(err)) {
                             if (!v.pl_finish(outp.data(), err)) return die(err);
+                            force_rows_out(outp.data(), A.T, produced_n);   // lab bench: STRATA_FORCE_IDS
                             A.s1_done = true;
                             tre("F1", A.seq, A.T);
                         } else if (!err.empty()) return die(err);
```

### 03-5

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                // stage B now (its PLE rows from the tokens before it as they will be once A is
                                // committed whole), so its launch behind A is only the graph launch (never while a
                                // rollback is pending: B's verifier is the one the undo commit reads)
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                                 // stage B now (its PLE rows from the tokens before it as they will be once A is
                                 // committed whole), so its launch behind A is only the graph launch (never while a
                                 // rollback is pending: B's verifier is the one the undo commit reads)
+                                if (B.ready && force_win != nullptr) force_window(B.p, B.tok[0], B.T, B.tok, S);
                                 if (B.ready && B.p_on >= theta && pl_prestage && !VF(0, B).in_flight() && !doomed) {
                                     int32_t prev[2] = {ss.ple_prev[0], ss.ple_prev[1]};
                                     for (int i = 0; i < A.T; ++i) { prev[0] = prev[1]; prev[1] = A.tok[i]; }
```

### 03-6

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    // state)
                    if (A.ready && !A.fl && A.fr < pl_nf && !doomed) {
                        const int st = A.fr;
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     // state)
                     if (A.ready && !A.fl && A.fr < pl_nf && !doomed) {
                         const int st = A.fr;
+                        if (st == 0 && force_win != nullptr) force_window(A.p, A.tok[0], A.T, A.tok, S);   // lab bench replay
                         if (st == 0 && A.p + A.T > o.max_context) { ending = true; continue; }
                         if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
                         if (!snap_take(st, A.seq)) return die("the GDN snapshot failed");
```

### 03-7

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    }();
                    if (B.ready && !B.fl && B.fr < pl_nf && B.fr < pl_spec_depth && A.fr > B.fr && !doomed) {
                        const int st = B.fr;
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     }();
                     if (B.ready && !B.fl && B.fr < pl_nf && B.fr < pl_spec_depth && A.fr > B.fr && !doomed) {
                         const int st = B.fr;
+                        if (st == 0 && force_win != nullptr) force_window(B.p, B.tok[0], B.T, B.tok, S);   // lab bench replay
                         if (st > 0 || (B.p_on >= theta && B.p + B.T <= o.max_context)) {
                             if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
                             PlSnap& sn = pl_snaps[(size_t) st];
```

### 03-8

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    chain_kind = 0;
                    int a = 0;
                    while (a < A.T - 1 && A.tok[a + 1] == outp[(size_t) a]) ++a;
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     chain_kind = 0;
                     int a = 0;
                     while (a < A.T - 1 && A.tok[a + 1] == outp[(size_t) a]) ++a;
+                    force_note(A.p, A.T, a, A.tok);
                     if (!V1(A).pl_commit_async(a + 1, err)) return die(err);
                     for (int i = 0; i <= a; ++i) consumed.push_back(A.tok[i]);
                     draft_offered += A.T - 1;
```

### 03-9

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                         : from_sfx ? sbuf[(size_t) i - 1]
                                                    : drafts[(size_t) i - 1];
                for (int i = 0; i < chain_n; ++i) window[(size_t) (T_mtp + i)] = cbuf[(size_t) i];
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                                          : from_sfx ? sbuf[(size_t) i - 1]
                                                     : drafts[(size_t) i - 1];
                 for (int i = 0; i < chain_n; ++i) window[(size_t) (T_mtp + i)] = cbuf[(size_t) i];
+                if (force_win != nullptr && !merged) force_window(p, window[0], T, window.data(), S);   // lab bench replay
                 drive.d.layers = 0;
                 drive.d.experts = 0;
                 drive.d.failed = false;
```

### 03-10

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    }
                }
                int a = merged ? T - 1 : 0;   // a merged window commits every row, as the prompt windows do
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 9 lines):
```diff
                     }
                 }
                 int a = merged ? T - 1 : 0;   // a merged window commits every row, as the prompt windows do
+                if (!merged) force_rows_out(outv.data(), T, produced_n);   // lab bench: STRATA_FORCE_IDS (a merged window holds prompt rows: not forced)
                 while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
+                force_note(p, T, a, window.data());
                 if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
                 if (chain_n > 0) { ++chain_windows; chain_drafts += chain_n; chain_ok += std::max(0, a - (T_mtp - 1)); }
                 if (!from_sfx && !first_window)
```

### 03-11

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                p += a + 1;
            }
            const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 21 lines):
```diff
                 p += a + 1;
             }
             const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
+            {   // lab bench: the verified windows' work, to compare two runs (forced or not)
+                std::string h;
+                char hb[32];
+                for (int r = 1; r < 16; ++r)
+                    if (force_rows[(size_t) r] > 0) {
+                        std::snprintf(hb, sizeof hb, " %d:%lld", r, (long long) force_rows[(size_t) r]);
+                        h += hb;
+                    }
+                std::fprintf(stderr, "strata work: request %lld%s, %lld windows, rows/window%s, accepted %lld of %lld drafts, "
+                                     "%lld tokens; target picks replaced on verified rows %lld; trace windows %lld, misses %lld\n", (long long) force_k,
+                             forced != nullptr ? " (forced)" : "", (long long) force_windows, h.c_str(),
+                             (long long) force_acc, (long long) force_off, (long long) produced_n, (long long) force_over, (long long) force_win_hit,
+                             (long long) force_win_miss);
+                if (force_trace != nullptr) std::fflush(force_trace);
+            }
             // the last commit (set_commit_async): the session is complete before anything reads or copies it
             if (!ver.wait_commit(err)) {
                 std::printf("ERR %s\n", err.c_str());
```

## 04 elastic-split-pipeline

`04-elastic-split-pipeline.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la cache elastica di CUDA0 con --layer-split e accanto a --pipeline-windows

### 04-1

`src/program/generate.cpp` · static const char* pipe_dbg_env(const char* name) {

Anchor:
```
    return on ? std::getenv(name) : nullptr;
}
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 20 lines):
```diff
     return on ? std::getenv(name) : nullptr;
 }

+// fork (Daily dd8eba52 + 1541e340 + a0ebf745): the elastic cache of CUDA0's stage beside --pipeline-windows across GPUs.
+// With two cards and --pipeline-windows it is ON by default (main sets g_pipe_elastic_auto before the first call);
+// STRATA_PIPELINE_ELASTIC=0 turns it off, =1 forces it on.  Its tail moves only at a gap of stage 0 (see el_gap).
+static bool g_pipe_elastic_auto = false;
+static bool pipe_elastic_env() {
+    static const bool on = [] {
+        const char* v = std::getenv("STRATA_PIPELINE_ELASTIC");
+        if (v != nullptr && v[0] == '0') return false;
+        if (v != nullptr && v[0] == '1') return true;
+        return g_pipe_elastic_auto;
+    }();
+    return on;
+}
+
 // lab bench (deterministic work), STRATA_FORCE_IDS=<file>[;<file>...] (opt-in): the k-th generation request of the
 // process (GEN/GENI, counted from 0) generates exactly the token ids of the k-th file - int32 little-endian, or a text
 // list (decimal ids separated by spaces, commas, newlines, brackets) - whatever the target picks.  Requests past the
```

### 04-2

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
        // free figure read again; while it is short of the reserve the cache is reopened smaller.
        // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 24 lines):
```diff
         // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
         // free figure read again; while it is short of the reserve the cache is reopened smaller.
         // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
-        if (o.elastic && (!o.serve || multi_gpu || o.resident_cpu_experts))
-            std::fprintf(stderr, "strata: elastic cache requires single-GPU serve without resident CPU complement; using fixed cache\n");
+        // fork: under a layer split only CUDA0's stage (the card with the display) is elastic; the later stages keep
+        // fixed caches.  Not with windows pipelined across the stages.
+        // upsync 0.1.42: a layer split with --batch now pipelines one group per stage by default, resolved after the
+        // check where --elastic is read: the same rule as there, on the resolved number
+        if (o.elastic && o.batch != 0 && o.batch_groups > 1) {
+            std::fprintf(stderr, "strata: elastic cache is not validated with --batch-groups; using fixed cache\n");
+            o.elastic = false;
+        }
+        // (fork: beside --pipeline-windows across the cards too, by default - see pipe_elastic_env; then the resident RAM
+        // mode keeps it as well: an expert that leaves the tail is computed by the CPU from the RAM copy, or read from
+        // the model files when the copy does not hold it)
+        g_pipe_elastic_auto = multi_gpu && o.pipeline_windows > 0;   // before the first pipe_elastic_env()
+        const bool el_split_ok = !multi_gpu || o.pipeline_windows == 0 || pipe_elastic_env();
+        const bool el_res_ok = !o.resident_cpu_experts || pipe_elastic_env();
+        if (o.elastic && (!o.serve || !el_split_ok || !el_res_ok))
+            std::fprintf(stderr, "strata: elastic cache requires serve, no --pipeline-windows across GPUs and no resident CPU complement; using fixed cache\n");
         int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL") ? std::atoi(std::getenv("STRATA_TEST_CACHE_FAIL")) : 0;
         int failed = 0;
         int zero_reads = 0;
```

### 04-3

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                --fake_fails;
                err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
            } else {
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                 --fake_fails;
                 err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
             } else {
-                if (!sized_slots.empty() && o.elastic && o.serve && !multi_gpu && !o.resident_cpu_experts) {
+                if (!sized_slots.empty() && o.elastic && o.serve && el_split_ok && el_res_ok) {
                     // --elastic: the elastic arena reserves the whole card's address range; without VMM, the plain one
                     size_t fb = 0, tb = 0;
                     cudaMemGetInfo(&fb, &tb);
```

### 04-4

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
    // --elastic: the TWO-ZONE cache.  Slots [0, el_core) are the core - filled from the profile at startup, never
    // picked by the adaptive swaps to leave, never shrunk, never lent; the tail after it is the elastic zone.
    int64_t el_core = 0;
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 29 lines):
```diff
     // --elastic: the TWO-ZONE cache.  Slots [0, el_core) are the core - filled from the profile at startup, never
     // picked by the adaptive swaps to leave, never shrunk, never lent; the tail after it is the elastic zone.
     int64_t el_core = 0;
-    if (o.elastic && o.serve && xcache.elastic() && !multi_gpu && xcache.slot_offsets() != nullptr) {
+    if (o.elastic && o.serve && xcache.elastic() && (!multi_gpu || o.pipeline_windows == 0 || pipe_elastic_env()) &&
+        xcache.slot_offsets() != nullptr) {
         const uint64_t* off = xcache.slot_offsets();
         const int64_t S = xcache.slots();
+        // fork: beside the pipeline (pipe_elastic_env) the tail defaults to STRATA_PIPELINE_ELASTIC_TAIL_MIB (1536): only a
+        // small part of CUDA0's cache moves, the core stays mapped
+        const int64_t pe_tail = [] {
+            const char* v = std::getenv("STRATA_PIPELINE_ELASTIC_TAIL_MIB");
+            return v != nullptr ? (int64_t) std::max(0, std::atoi(v)) : (int64_t) 1536;
+        }();
         const uint64_t tail = o.elastic_tail_mib >= 0 ? (uint64_t) o.elastic_tail_mib << 20
-                                                      : std::min<uint64_t>(6144ull << 20, off[S] / 2);
+                            : pipe_elastic_env() ? std::min<uint64_t>((uint64_t) pe_tail << 20, off[S] / 2)
+                                                 : std::min<uint64_t>(6144ull << 20, off[S] / 2);
         int64_t c = S;
         while (c > 0 && off[S] - off[c - 1] <= tail) --c;   // the smallest core whose tail fits in `tail`
         el_core = tail == 0 ? 0 : c;
     }
+    // fork (Daily 12a6f553): beside the pipeline the prompt path may borrow the core too - nothing decodes while a prompt
+    // is read, and the lent slots are refilled before the decode as the tail's are (with only the 1.5 GiB tail lent the
+    // 131k prompt read fell from ~85 s to 134 s)
+    const int64_t el_lend_floor = pipe_elastic_env() ? 0 : el_core;
     // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits,
     // or with --prefill auto the largest chunk whose buffers take at most kAutoLendPct % of the slots (a
     // lent slot's expert is streamed during the prompt and refilled after it; measured on a 12 GB card, 32K Q2_0
```

### 04-5

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
    };
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        auto slots_for = lend_slots;
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
     };
     auto plan_lend = [&](int64_t& chunk) -> int64_t {
         auto slots_for = lend_slots;
-        const int64_t tail_slots = xcache.slots() - el_core;   // --elastic: only the tail is lent (all of it without a core)
+        const int64_t tail_slots = xcache.slots() - el_lend_floor;   // --elastic: only the tail is lent (all of it without a core)
         // 0.1.39's list with its ring (also the size up to which a prompt keeps that ring under #583)
         auto old_rule = [&]() -> int64_t {
             strata::prefill::Prefill::set_ring_budget(0, 0);
```

### 04-6

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
            auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                const int64_t k = part_slots(p, c);
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
             auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                 const int64_t k = part_slots(p, c);
-                if (p.cache == &xcache && k > xcache.slots() - el_core) return false;
+                if (p.cache == &xcache && k > xcache.slots() - el_lend_floor) return false;
                 if (k <= 0 || k + 128 > p.cache->slots()) return false;
                 return !(cap && k * 100 > kAutoLendPct * p.cache->slots());
             };
```

### 04-7

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - 128,
                                                                  kAutoLendPct * p.cache->slots() / 100));
                    // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                     int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - 128,
                                                                   kAutoLendPct * p.cache->slots() / 100));
                     // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
-                    if (p.cache == &xcache) budget = std::min(budget, std::max<int64_t>(0, xcache.slots() - el_core));
+                    if (p.cache == &xcache) budget = std::min(budget, std::max<int64_t>(0, xcache.slots() - el_lend_floor));
                     const uint64_t avail = part_bytes(p, (int32_t) (p.cache->slots() - budget));
                     const uint64_t nonring = strata::prefill::Prefill::bytes_needed_no_ring(g, *p.ses, c, srcp != nullptr);
                     room = std::min(room, (int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) kBlob));
```

### 04-8

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
        // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
        // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
         // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
         // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
         // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
-        const int64_t sp_chunk_max = (o.elastic && xcache.elastic() && o.prefill_auto && borrow != nullptr) ? o.prefill_auto_max : 0;
+        // fork: the largest chunk the elastic cache may later lend CUDA0's prompt path.  Under a layer split the later
+        // stages' prompt paths are laid out for the start's chunk only, so the chunk may shrink but never grow past it
+        // (a growth to more than the start's 6656 failed a later stage's relayout: "a chunk of at most 6656")
+        const int64_t sp_chunk_max = (o.elastic && xcache.elastic() && o.prefill_auto && borrow != nullptr && !multi_gpu)
+                                         ? o.prefill_auto_max : 0;
         auto init_prompt_paths = [&]() -> int {   // 0: ready; 1: failed (err set); 2: failed with "do not fit"
             for (size_t i = 0; i < stages.size(); ++i) {
                 GpuStage& st = *stages[i];
```

### 04-9

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        std::function<void()> adapt_fence;
        int64_t adapt_nswaps = 0;   // the blocking tier's swaps chosen, cumulative (the decode timing line's swaps/window)
        // the VRAM tier follows the conversation (the same rule as the speculative loop below)
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 14 lines):
```diff
         std::function<void()> adapt_fence;
         int64_t adapt_nswaps = 0;   // the blocking tier's swaps chosen, cumulative (the decode timing line's swaps/window)
         // the VRAM tier follows the conversation (the same rule as the speculative loop below)
+        // fork (Daily 8b9c342b, 1541e340): STRATA_ELASTIC_CORE_ADAPT=1, and beside the pipeline by default - the core keeps
+        // its size and mapping (never shrunk, never lent to the decode) but its experts are swap victims like the tail's
+        static const bool core_adapt = [] {
+            const char* v = std::getenv("STRATA_ELASTIC_CORE_ADAPT");
+            const bool on = (v != nullptr && v[0] == '1') || pipe_elastic_env();
+            if (on) std::fprintf(stderr, "strata serve: STRATA_ELASTIC_CORE_ADAPT: the core's experts are swappable\n");
+            return on;
+        }();
         auto adapt = [&]() -> bool {
             if (!pending.empty()) return true;   // the previous swaps are still in flight
             if (remote_opt && !remote_opt->adapt(drive.d.usage, host_res, pending, o.adapt_swaps, *srcp)) return false;
```

### 04-10

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 11 lines):
```diff
                 const int32_t* r = host_res.data() + l * g.n_expert;
                 for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                     if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
-                    else if (r[e] >= el_core) vict.emplace_back(u[e], e);   // --elastic: the core never leaves
+                    // --elastic: the core never leaves - unless core_adapt; beside the pipeline el_core counts CUDA0's
+                    // slots and a later stage's layers hold slots of their own cache
+                    else if (r[e] >= el_core || core_adapt || (pipe_elastic_env() && multi_gpu && stage_of(l) != 0))
+                        vict.emplace_back(u[e], e);
                 }
                 if (cand.empty() || vict.empty()) continue;
                 std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
```

### 04-11

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // that shrank too far lends a smaller chunk, the same rule as plan_lend).  Run only where nothing is in
        // flight: a request's start (nothing lent yet) and the decode's safe point (window, commit, draft and the
        // adaptive swaps all finished).
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
         // that shrank too far lends a smaller chunk, the same rule as plan_lend).  Run only where nothing is in
         // flight: a request's start (nothing lent yet) and the decode's safe point (window, commit, draft and the
         // adaptive swaps all finished).
-        const bool el_on = o.elastic && xcache.elastic() && d_res != nullptr && !multi_gpu && !host_res.empty();
+        const bool el_on = o.elastic && xcache.elastic() && d_res != nullptr &&
+                           (!multi_gpu || o.pipeline_windows == 0 || pipe_elastic_env()) &&
+                           !host_res.empty();
+        // under a split host_res holds slot numbers of whichever cache owns the layer: the elastic cache is CUDA0's,
+        // so it shrinks and grows over the layers of the first stage only
+        auto el_layer = [&](int64_t l) { return !multi_gpu || stage_of(l) == 0; };
         const int64_t el_reserve = (int64_t) o.elastic_reserve_mib << 20;
         const int64_t el_hyst_min = 192ll << 20;   // growth only past the reserve plus max(this, the measured noise)
         static const bool el_trace = std::getenv("STRATA_ELASTIC_TRACE") != nullptr;
```

### 04-12

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        Clock::time_point el_last_grow{}, el_last_shrink{};
        int el_backoff = 1;
        auto el_free = []() -> int64_t {
```
Block (hunk 12 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
         Clock::time_point el_last_grow{}, el_last_shrink{};
         int el_backoff = 1;
         auto el_free = []() -> int64_t {
+            const strata::core::OnDevice on(0);   // the elastic cache's card (CUDA0)
             size_t f = 0, t = 0;
             return cudaMemGetInfo(&f, &t) == cudaSuccess ? (int64_t) f : -1;
         };
```

### 04-13

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            const bool have_usage = !drive.d.usage.empty();
            std::vector<std::vector<std::pair<float, int32_t>>> keep((size_t) g.n_layers), tail((size_t) g.n_layers);
            for (int64_t l = 0; l < g.n_layers; ++l)
```
Block (hunk 13 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             const bool have_usage = !drive.d.usage.empty();
             std::vector<std::vector<std::pair<float, int32_t>>> keep((size_t) g.n_layers), tail((size_t) g.n_layers);
             for (int64_t l = 0; l < g.n_layers; ++l)
-                for (int32_t ex = 0; ex < (int32_t) g.n_expert; ++ex) {
+                for (int32_t ex = 0; ex < (int32_t) g.n_expert && el_layer(l); ++ex) {
                     const size_t i = (size_t) (l * g.n_expert + ex);
                     if (host_res[i] < el_core) continue;   // not resident, or in the core (never displaced)
                     const float u = have_usage ? drive.d.usage[i] : 0.0f;
```

### 04-14

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // GROW by at most `room` bytes: the missing experts the conversation routes most (usage >= 2, as the adaptive
        // tier's candidates), then the profile's order for what it has not asked for yet; filled on the refill
        // stream and published like the adaptive swaps (apply_pending), so the windows never wait for them
```
Block (hunk 14 of `src/program/generate.cpp` in the patch, 22 lines):
```diff
         // GROW by at most `room` bytes: the missing experts the conversation routes most (usage >= 2, as the adaptive
         // tier's candidates), then the profile's order for what it has not asked for yet; filled on the refill
         // stream and published like the adaptive swaps (apply_pending), so the windows never wait for them
-        auto el_grow = [&](int64_t room, std::string& e) -> bool {
+        // fork (Daily 1541e340): the pipelined decode's growth, at a gap of stage 0 (no window of CUDA0 in flight): `el_fill`
+        // gets the fills (residency index, slot), published by the loop once `el_fill_ev` has passed; never a device-wide wait
+        std::vector<std::pair<int32_t, int32_t>> el_fill;
+        cudaEvent_t el_fill_ev = nullptr;
+        // the asynchronous tier's round (set where it is built): the elastic cache acts only between rounds
+        std::function<bool()> el_tier_busy;
+        bool el_hold_tier = false;   // no round of either tier starts while a growth's fills are unpublished
+        auto el_grow = [&](int64_t room, std::string& e, bool pipe_gap = false) -> bool {
             const auto& lay = strata::kernels::cpu::expert_layout();
             std::vector<std::pair<float, int32_t>> cand;
             if (!drive.d.usage.empty())
                 for (size_t i = 0; i < host_res.size(); ++i)
-                    if (host_res[i] < 0 && drive.d.usage[i] >= 2.0f) cand.emplace_back(drive.d.usage[i], (int32_t) i);
+                    if (host_res[i] < 0 && drive.d.usage[i] >= 2.0f && el_layer((int64_t) i / g.n_expert))
+                        cand.emplace_back(drive.d.usage[i], (int32_t) i);
             std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
             std::vector<char> chosen(host_res.size(), 0);
             std::vector<int32_t> pick;
```

### 04-15

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            for (const auto& pr : profile) {
                if (used + (int64_t) lay.max_blob > room) break;
                const int32_t i = (int32_t) (pr.first * g.n_expert + pr.second);
```
Block (hunk 15 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             for (const auto& pr : profile) {
                 if (used + (int64_t) lay.max_blob > room) break;
                 const int32_t i = (int32_t) (pr.first * g.n_expert + pr.second);
-                if (host_res[(size_t) i] >= 0 || chosen[(size_t) i]) continue;
+                if (host_res[(size_t) i] >= 0 || chosen[(size_t) i] || !el_layer(pr.first)) continue;
                 if (!take(i)) break;
             }
             if (pick.empty()) return true;
```

### 04-16

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            // under WDDM the free figure is only trustworthy once the pages are written: `grow` zeroed them
            cudaDeviceSynchronize();
```
Block (hunk 16 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
                 e.clear();
                 return true;
             }
-            // under WDDM the free figure is only trustworthy once the pages are written: `grow` zeroed them
-            cudaDeviceSynchronize();
+            // under WDDM the free figure is only trustworthy once the pages are written: `grow` zeroed them (the
+            // pipelined decode: the legacy stream's memset only - the stages' streams hold windows and commits)
+            if (pipe_gap) cudaStreamSynchronize(nullptr);
+            else cudaDeviceSynchronize();
             if (el_free() < el_reserve) {
                 xcache.elastic_shrink(first, e);
                 std::fprintf(stderr, "strata elastic: growth undone, the VRAM was not really free\n");
```

### 04-17

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    e = "elastic cache: a fill copy failed";
                    return false;
                }
```
Block (hunk 17 of `src/program/generate.cpp` in the patch, 22 lines):
```diff
                     e = "elastic cache: a fill copy failed";
                     return false;
                 }
-                pending.emplace_back(i, (int32_t) (first + (int64_t) j));
+                if (pipe_gap) el_fill.emplace_back(i, (int32_t) (first + (int64_t) j));
+                else pending.emplace_back(i, (int32_t) (first + (int64_t) j));
+            }
+            if (pipe_gap) {
+                if (el_fill_ev == nullptr && cudaEventCreateWithFlags(&el_fill_ev, cudaEventDisableTiming) != cudaSuccess) {
+                    e = "elastic cache: event creation failed";
+                    return false;
+                }
+                cudaEventRecord(el_fill_ev, adapt_stream);
+                (void) cudaStreamQuery(adapt_stream);   // WDDM: submit now
+                el_hold_tier = true;
+            } else {
+                cudaEventRecord(adapt_ev, adapt_stream);
             }
-            cudaEventRecord(adapt_ev, adapt_stream);
             // a bigger cache may lend a bigger chunk (the prompt reads faster)
             el_replan_lend();
             std::fprintf(stderr, "strata elastic: +%lld experts (%.0f MiB) -> %lld slots, %.2f GiB; %lld MiB free\n",
```

### 04-18

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // one step of the policy: shrink at once below the reserve; grow once the room has stayed for stable_ms.  At a
        // request's start it only shrinks: a growth's fills land asynchronously in the tail, and the prompt is about
        // to borrow the tail - the decode's safe point grows, and apply_pending publishes the fills before any lend
```
Block (hunk 18 of `src/program/generate.cpp` in the patch, 18 lines):
```diff
         // one step of the policy: shrink at once below the reserve; grow once the room has stayed for stable_ms.  At a
         // request's start it only shrinks: a growth's fills land asynchronously in the tail, and the prompt is about
         // to borrow the tail - the decode's safe point grows, and apply_pending publishes the fills before any lend
-        auto elastic_step = [&](bool request_start, std::string& e) -> bool {
+        // fork (Daily dd8eba52): with `defer` the step only decides - the same bookkeeping, no device work and no
+        // apply_pending - and leaves its action in *defer; the pipelined decode runs it at a gap of stage 0 (el_gap)
+        struct ElPlan { int kind = 0; int64_t arg = 0; };   // kind 1: shrink to `arg` slots; 2: grow by `arg` bytes
+        auto elastic_step = [&](bool request_start, std::string& e, ElPlan* defer = nullptr) -> bool {
             if (!el_on) return true;
+            if (el_tier_busy && el_tier_busy()) return true;   // between the asynchronous tier's rounds only
             const Clock::time_point t = Clock::now();
             if (!request_start && t - el_last < std::chrono::milliseconds(o.elastic_period_ms)) return true;
             el_last = t;
-            apply_pending(true);
+            if (defer == nullptr) apply_pending(true);
             const int64_t fr = el_free();
             if (fr < 0) return true;
             if (el_trace)
```

### 04-19

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    el_backoff = std::min(el_backoff * 2, 8);   // a shrink right after a growth: it grew too far
                el_last_shrink = t;
                el_seen.clear();
```
Block (hunk 19 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     el_backoff = std::min(el_backoff * 2, 8);   // a shrink right after a growth: it grew too far
                 el_last_shrink = t;
                 el_seen.clear();
+                if (defer != nullptr) { *defer = ElPlan{1, n}; return true; }
                 return el_shrink(n, e);
             }
             if (el_last_shrink.time_since_epoch().count() != 0 && t - el_last_shrink > std::chrono::minutes(5))
```

### 04-20

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            el_stable_since = t;
            el_last_grow = t;
            el_seen.clear();
```
Block (hunk 20 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
             el_stable_since = t;
             el_last_grow = t;
             el_seen.clear();
+            if (defer != nullptr) { *defer = ElPlan{2, std::min(spare, el_step_max)}; return true; }
             return el_grow(std::min(spare, el_step_max), e);
         };
         if (el_on) {
```

### 04-21

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                switch (astate) {
                case AState::Idle:
                    if (drain || a_win - a_last < (int64_t) o.adapt_every) return true;
```
Block (hunk 21 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                 switch (astate) {
                 case AState::Idle:
                     if (drain || a_win - a_last < (int64_t) o.adapt_every) return true;
+                    if (el_hold_tier) return true;   // fork: a growth of the elastic tail is being published
                     a_last = a_win;
                     ++a_rounds;
                     a_t0 = Clock::now();
```

### 04-22

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // --pipeline-windows 2: the loop has drained (nothing in flight on either card): what waits for a gap goes now,
        // and the round in flight goes on unfenced (a_pl_idle unset) from the next tick, the next request's drain.  The
        // job's work is the host's alone, so the wait for it ends.
```
Block (hunk 22 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
         // --pipeline-windows 2: the loop has drained (nothing in flight on either card): what waits for a gap goes now,
         // and the round in flight goes on unfenced (a_pl_idle unset) from the next tick, the next request's drain.  The
         // job's work is the host's alone, so the wait for it ends.
+        if (ajob) el_tier_busy = [&]() { return astate != AState::Idle || !ajob->idle(); };
         auto a_pl_settle = [&]() {
             if (!ajob || !a_pl_idle) return;
             ajob->wait();
```

### 04-23

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                bool b_done = false;     // B has been made from the chain in flight
                int chain_n = 0;
                int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
```
Block (hunk 23 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
                 bool b_done = false;     // B has been made from the chain in flight
                 int chain_n = 0;
                 int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
+                // fork (Daily 1541e340): the elastic tail beside the windows, at the gaps of stage 0 (see el_gap)
+                int64_t pl_el_shrinks = 0, pl_el_grows = 0;
+                double pl_el_ms = 0.0;
+                static const int64_t pl_el_step = [] {   // a growth at a gap moves at most this much (MiB)
+                    const char* v = std::getenv("STRATA_PIPELINE_ELASTIC_STEP_MIB");
+                    return (int64_t) (v != nullptr ? std::max(64, std::atoi(v)) : 128) << 20;
+                }();
                 std::vector<int32_t> outp(8, 0);
                 bool ending = false;
                 A.T = 1;
```

### 04-24

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                // --adapt-async: the asynchronous tier beside the windows (see adapt_tick).  A stage is idle when neither
                // of its verifiers has a window in flight (a commit or a snapshot copy waits on nothing of the host's)
                if (ajob) a_pl_idle = [&](int st) { return !PV[st][0]->in_flight() && !PV[st][1]->in_flight(); };
```
Block (hunk 24 of `src/program/generate.cpp` in the patch, 77 lines):
```diff
                 // --adapt-async: the asynchronous tier beside the windows (see adapt_tick).  A stage is idle when neither
                 // of its verifiers has a window in flight (a commit or a snapshot copy waits on nothing of the host's)
                 if (ajob) a_pl_idle = [&](int st) { return !PV[st][0]->in_flight() && !PV[st][1]->in_flight(); };
+                // fork (Daily 1541e340): the elastic tail of CUDA0's cache beside the windows, without a drain.  Every CUDA
+                // call of it reaches CUDA0 at a GAP of stage 0 - neither of its verifiers has a window in flight (a commit
+                // or a snapshot copy waits on nothing of the host's) - so no driver call meets a window that spins on this
+                // thread, and every window planned before has completed.  Windows plan each layer from host_res when that
+                // layer is served, and the later stages never read CUDA0's slots.  So:
+                //   - SHRINK: the tail's experts leave host_res (the CPU computes them) and CUDA0's residency table is
+                //     rewritten, then the tail is unmapped - all at one gap;
+                //   - GROW: mapped, zeroed and filled at a gap (at most STRATA_PIPELINE_ELASTIC_STEP_MIB, 128), published
+                //     into host_res once the fills have landed: a window never sees a slot before its expert is there.
+                // The core is never remapped.  Either tier acts only between its rounds, and starts none while a growth
+                // is unpublished (el_hold_tier).  False (with err) when a step failed.
+                bool el_res_dirty = false;   // every residency table uploaded again once the loop has drained
+                auto el_gap = [&]() -> bool {
+                    if (!el_on || PV[0][0]->in_flight() || PV[0][1]->in_flight()) return true;
+                    if (!el_fill.empty()) {   // a growth's fills: published once landed
+                        const cudaError_t q = cudaEventQuery(el_fill_ev);
+                        if (q == cudaErrorNotReady) return true;
+                        if (q != cudaSuccess) { err = "elastic cache: a fill copy failed"; return false; }
+                        for (const auto& [i, slot] : el_fill) {
+                            host_res[(size_t) i] = slot;
+                            srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
+                        }
+                        el_fill.clear();
+                        el_hold_tier = false;
+                        el_res_dirty = true;
+                        const strata::core::OnDevice on(0);   // the elastic cache's card (CUDA0)
+                        if (res_put(d_res) != cudaSuccess) { err = "elastic cache: the residency table upload failed"; return false; }
+                        return true;
+                    }
+                    // between the tier's rounds only: the blocking one (its thread joined by pl_adapt, nothing pending or
+                    // waiting for its exchange fence), or the asynchronous one (el_tier_busy, inside elastic_step)
+                    if (!ajob && (pl_adapt_thr.joinable() || !pending.empty() || exch_wait)) return true;
+                    ElPlan plan;
+                    if (!elastic_step(false, err, &plan)) return false;
+                    if (plan.kind == 0) return true;
+                    const Clock::time_point t0 = Clock::now();
+                    const strata::core::OnDevice on(0);   // the elastic cache's card (CUDA0)
+                    if (plan.kind == 1) {
+                        const int64_t S = xcache.slots(), n = plan.arg;
+                        if (n >= S) return true;
+                        int dropped = 0;
+                        for (size_t i = 0; i < host_res.size(); ++i)
+                            if (host_res[i] >= n && el_layer((int64_t) i / g.n_expert)) {
+                                host_res[i] = strata::core::kNotResident;
+                                ++dropped;
+                            }
+                        std::atomic_thread_fence(std::memory_order_seq_cst);
+                        el_res_dirty = true;
+                        if (res_put(d_res) != cudaSuccess) { err = "elastic cache: the residency table upload failed"; return false; }
+                        if (!xcache.elastic_shrink(n, err)) return false;
+                        el_replan_lend();
+                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
+                        std::fprintf(stderr, "strata elastic: -%lld slots at a gap of stage 0 (%d experts to the CPU) -> %lld "
+                                             "slots, %.2f GiB; %lld MiB free; %.2f ms\n", (long long) (S - n), dropped,
+                                     (long long) xcache.slots(), xcache.gib(), (long long) (el_free() >> 20), ms);
+                        ++pl_el_shrinks;
+                        pl_el_ms += ms;
+                    } else {
+                        const int64_t before = xcache.slots();
+                        if (!el_grow(std::min<int64_t>(plan.arg, pl_el_step), err, true)) return false;
+                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
+                        if (xcache.slots() != before) {
+                            std::fprintf(stderr, "strata elastic: the growth above at a gap of stage 0 took %.2f ms (published "
+                                                 "once its fills land)\n", ms);
+                            ++pl_el_grows;
+                        }
+                        pl_el_ms += ms;
+                    }
+                    tre("EL", A.seq, plan.kind);
+                    return true;
+                };
                 double a_poll_ms = 0.0;
                 auto pl_adapt = [&]() -> bool {   // the blocking tier (--adapt-async 0)
                     if (pl_adapt_thr.joinable() && pl_adapt_done.load()) {
```

### 04-25

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    if (pl_adapt_thr.joinable()) return true;   // still adapting
                    pl_apply();
                    if (!drive.d.usage.empty() && o.adapt_every > 0 && (rounds % o.adapt_every) == 0 && pending.empty() &&
```
Block (hunk 25 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                     if (pl_adapt_thr.joinable()) return true;   // still adapting
                     pl_apply();
                     if (!drive.d.usage.empty() && o.adapt_every > 0 && (rounds % o.adapt_every) == 0 && pending.empty() &&
-                        !exch_wait) {
+                        !el_hold_tier && !exch_wait) {
                         pl_adapt_done = false;
                         pl_adapt_thr = std::thread([&] { strata::aux_cpus::pin_current_thread(); pl_adapt_ok = adapt(); pl_adapt_done = true; });
                     }
```

### 04-26

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        } else if (!err.empty()) return die(err);
                    }
                    if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
```
Block (hunk 26 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                         } else if (!err.empty()) return die(err);
                     }
                     if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
+                    if (!el_gap()) return die(err);   // fork: the elastic tail at a gap of stage 0
                     // ---- a wrong speculative window has stopped (it is launched on no further stage once doomed):
                     // every front stage it ran on back to the verified window's real commit
                     if (doomed && !D.fl) {
```

### 04-27

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        if (st == 0 && force_win != nullptr) force_window(A.p, A.tok[0], A.T, A.tok, S);   // lab bench replay
                        if (st == 0 && A.p + A.T > o.max_context) { ending = true; continue; }
                        if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
```
Block (hunk 27 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                         if (st == 0 && force_win != nullptr) force_window(A.p, A.tok[0], A.T, A.tok, S);   // lab bench replay
                         if (st == 0 && A.p + A.T > o.max_context) { ending = true; continue; }
                         if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
+                        if (st == 0 && !el_gap()) return die(err);   // fork: the elastic tail
                         if (!snap_take(st, A.seq)) return die("the GDN snapshot failed");
                         if (!VF(st, A).pl_launch(A.T, A.tok, A.p, err)) return die(err);
                         A.fl = true;
```

### 04-28

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        if (st == 0 && force_win != nullptr) force_window(B.p, B.tok[0], B.T, B.tok, S);   // lab bench replay
                        if (st > 0 || (B.p_on >= theta && B.p + B.T <= o.max_context)) {
                            if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
```
Block (hunk 28 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                         if (st == 0 && force_win != nullptr) force_window(B.p, B.tok[0], B.T, B.tok, S);   // lab bench replay
                         if (st > 0 || (B.p_on >= theta && B.p + B.T <= o.max_context)) {
                             if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
+                            if (st == 0 && !el_gap()) return die(err);   // fork: the elastic tail
                             PlSnap& sn = pl_snaps[(size_t) st];
                             cudaStream_t cs = pl_stream[(size_t) st];
                             {
```

### 04-29

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                // unfenced (the INVARIANT at pl_drain: a checkpoint, a park, the serial loop and the next request find
                // nothing pipelined in flight) at the next tick, the next request's drain
                a_pl_settle();
```
Block (hunk 29 of `src/program/generate.cpp` in the patch, 20 lines):
```diff
                 // unfenced (the INVARIANT at pl_drain: a checkpoint, a park, the serial loop and the next request find
                 // nothing pipelined in flight) at the next tick, the next request's drain
                 a_pl_settle();
+                if (!el_fill.empty()) {   // fork: nothing in flight - the growth's fills land and are published
+                    if (cudaEventSynchronize(el_fill_ev) != cudaSuccess) {
+                        std::printf("ERR elastic cache: a fill copy failed\n");
+                        return 1;
+                    }
+                    for (const auto& [i, slot] : el_fill) {
+                        host_res[(size_t) i] = slot;
+                        srcp->release((int64_t) i / g.n_expert, (int64_t) i % g.n_expert);
+                    }
+                    el_fill.clear();
+                    el_hold_tier = false;
+                    el_res_dirty = true;
+                }
+                if (el_res_dirty) res_upload();
                 for (cudaEvent_t ev : fence_ev) if (ev) cudaEventDestroy(ev);
                 for (cudaEvent_t ev : exch_ev) if (ev) cudaEventDestroy(ev);
                 if (pl_trace_path != nullptr && !pl_trace.empty()) {
```

### 04-30

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                         "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                 (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                 (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
```
Block (hunk 30 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                                          "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                  (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                  (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
+                    if (el_on)
+                        std::fprintf(stderr, "strata pipeline: elastic tail beside the windows: %lld shrinks, %lld growths at "
+                                             "gaps of stage 0, %.1f ms on the loop's thread\n",
+                                     (long long) pl_el_shrinks, (long long) pl_el_grows, pl_el_ms);
                     std::fprintf(stderr, "strata pipeline classes: fresh %.0f windows %.2f ms %.2f tok | speculative %.0f "
                                          "windows %.2f ms %.2f tok | forced-chain disagreements %lld, chain late %lld\n",
                                  cls_n[0], avg(cls_ms[0], cls_n[0]), avg(cls_tok[0], cls_n[0]), cls_n[1],
```

## 05 prefix-disk

`05-prefix-disk.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il prefisso su disco con due schede (--prompt-cache-file, formato v2)

### 05-1

`CMakeLists.txt` · option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

Anchor:
```
# native-experts section far below, so a first configure registered four tests fewer than the second)
option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 23 lines):
```diff
 # native-experts section far below, so a first configure registered four tests fewer than the second)
 option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)

+# fork: the system prefix persisted on disk (--prompt-cache-file)
+add_library(strata_prefix_cache STATIC src/core/prefix_cache_file.cpp)
+target_include_directories(strata_prefix_cache PUBLIC include)
+if(WIN32)
+  target_link_libraries(strata_prefix_cache PRIVATE bcrypt)
+endif()
+
 if(STRATA_BUILD_CONVERSATION_TESTS)
   enable_testing()
+  add_executable(prefix_cache_file_test src/core/prefix_cache_file_test.cpp)
+  target_link_libraries(prefix_cache_file_test PRIVATE strata_prefix_cache)
+  add_test(NAME prefix_cache_file_test COMMAND prefix_cache_file_test)
+  add_executable(prefix_cache_portable_test src/core/prefix_cache_file_test.cpp src/core/prefix_cache_file.cpp)
+  target_include_directories(prefix_cache_portable_test PRIVATE include)
+  target_compile_definitions(prefix_cache_portable_test PRIVATE STRATA_PREFIX_PORTABLE_SHA=1)
+  add_test(NAME prefix_cache_portable_test COMMAND prefix_cache_portable_test)
+  set_tests_properties(prefix_cache_file_test prefix_cache_portable_test PROPERTIES TIMEOUT 60)
   add_executable(conversation_cache_test src/core/conversation_cache_test.cpp)
   target_include_directories(conversation_cache_test PRIVATE include)
   add_test(NAME conversation_cache_test COMMAND conversation_cache_test)
```

### 05-2

`CMakeLists.txt` · if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)

Anchor:
```
    src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp
    src/core/arch_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
```
Block (hunk 2 of `CMakeLists.txt` in the patch, 8 lines):
```diff
     src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp
     src/core/arch_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
   target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
-  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
+  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu strata_prefix_cache ${_strata_gpu_runtime_target})
   if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
     add_executable(host_graph_launch_test tests/core/host_graph_launch_test.cu)
     target_include_directories(host_graph_launch_test PRIVATE include)
```

### 05-3

`include/strata/core/mtp.hpp` · public:

Anchor:
```
    const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
    const void* tensor_q8(const char* name) const { return q8(name); }
    QsaState& kv_state_rw() { return st_; }
```
Block (hunk 1 of `include/strata/core/mtp.hpp` in the patch, 13 lines):
```diff
     const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
     const void* tensor_q8(const char* name) const { return q8(name); }
     QsaState& kv_state_rw() { return st_; }
-    int64_t first_needed() const { return (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0; }
+    int64_t first_needed() const {
+        return (window_ > 0 && prompt_len_ > 0 && !full_prefix_) ? prompt_len_ - window_ - 64 : 0;
+    }
+    /// this prompt's root goes to the prefix file: K/V for every cell, also those the window will not reach again
+    /// (a saved prefix must serve later prompts of any length)
+    void set_full_prefix(bool on) { full_prefix_ = on; }
     int device() const { return device_; }
     bool idle(std::string& err) {
         if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: its stream failed"; return false; }
```

### 05-4

`include/strata/core/mtp.hpp` · private:

Anchor:
```
    std::string rt_dir_;
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t prompt_len_ = 0;
```
Block (hunk 2 of `include/strata/core/mtp.hpp` in the patch, 7 lines):
```diff
     std::string rt_dir_;
     int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
     int64_t prompt_len_ = 0;
+    bool full_prefix_ = false;
     const int32_t* rr_res_ = nullptr;
     float rr_margin_ = 0.0f;
     int rr_lo_ = 6, rr_hi_ = 9;
```

### 05-5

`include/strata/core/prefix_cache_file.hpp` · (new file)

New file, 64 lines: see the patch.

### 05-6

`src/core/mtp.cpp` · bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_

Anchor:
```
    const OnDevice on_device(device_);
    const Clock::time_point t0 = Clock::now();
    const int64_t HCN = g_->hc * g_->n_embd;
```
Block (hunk 1 of `src/core/mtp.cpp` in the patch, 11 lines):
```diff
     const OnDevice on_device(device_);
     const Clock::time_point t0 = Clock::now();
     const int64_t HCN = g_->hc * g_->n_embd;
-    // cells the window can never reach again need no K/V
-    const int64_t first_needed = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
+    // cells the window can never reach again need no K/V (all of them do when the root goes to the prefix file:
+    // first_needed() is 0 then, see set_full_prefix)
+    const int64_t first_needed = this->first_needed();
     // E-4: every group's token / step / position records uploaded at once; each group is then device copies and a
     // graph on the one stream, with a single sync at the end (a group of <= max_t rows used to be staged in mapped
     // memory and synced before the next: ~5,500 host round trips on a 32K prompt).  The same work in the same
```

### 05-7

`src/core/prefix_cache_file.cpp` · (new file)

New file, 599 lines: see the patch.

### 05-8

`src/core/prefix_cache_file_test.cpp` · (new file)

New file, 587 lines: see the patch.

### 05-9

`src/program/generate.cpp` · (file scope)

Anchor:
```
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/session_save_reclaim.hpp"
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
 #include "strata/core/conversation_snapshot.hpp"
 #include "strata/core/conversation_file.hpp"
 #include "strata/core/session_save_reclaim.hpp"
+#include "strata/core/prefix_cache_file.hpp"
 #include "strata/core/conversation_memory.hpp"
 #include "strata/core/coupled_draft.hpp"
 #include "strata/core/expert_source.hpp"
```

### 05-10

`src/program/generate.cpp` · namespace strata::metal_engine { void set_model_shard(const char* shard1); }   /

Anchor:
```
#include <cstdio>
#include <cstdlib>
#include <cstring>
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
 #include <cstdio>
 #include <cstdlib>
 #include <cstring>
+#include <cwchar>
 #include <filesystem>
 #include <fstream>
 #include <sstream>
```

### 05-11

`src/program/generate.cpp` · struct Options {

Anchor:
```
    /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
    /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
     /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
     /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
     int prompt_cache = 6;
+    std::string prompt_cache_file; // opt-in single text-root snapshot, persisted across engine restarts
     int64_t conversation_cache_mib = 0; // opt-in host RAM for independent conversations
     int conversation_cache_slots = 4;
     int64_t conversation_cache_min_free_mib = 2560;
```

### 05-12

`src/program/generate.cpp` · void usage() {

Anchor:
```
                 "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 9 lines):
```diff
                  "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                  "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                  "                       of RAM each; 0 = read every prompt from the start)\n"
+                 "  --prompt-cache-file PATH  persist one text system-prefix across restarts (default off; one GPU or a\n"
+                 "                       --layer-split, up to 262144 tokens/8 GiB; replaces its prior prefix when\n"
+                 "                       instructions change)\n"
                  "  --conversation-cache-mib N  --serve: RAM budget for parked conversations (default 0 = off)\n"
                  "  --conversation-cache-slots N  --serve: at most N parked conversations (default 4)\n"
                  "  --conversation-cache-min-free-mib N  --serve: RAM floor (under MemAvailable, the cgroup limits and\n"
```

### 05-13

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        else if (a == "--vision") o.vision = true;
        else if (a == "--session-save-reclaim") o.session_save_reclaim = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
         else if (a == "--vision") o.vision = true;
         else if (a == "--session-save-reclaim") o.session_save_reclaim = true;
         else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
+        else if (a == "--prompt-cache-file") o.prompt_cache_file = next("--prompt-cache-file");
         else if (a == "--conversation-cache-mib" || a == "--conversation-cache-slots" ||
                  a == "--conversation-cache-min-free-mib" || a == "--conversation-cache-min-tokens" ||
                  a == "--conversation-cache-park-gain" ||
```

### 05-14

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
    }();
    strata::core::set_primary_device(primary_device);
    // parking with --layer-split saves every stage (SavedConversation::stage_images)
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
     }();
     strata::core::set_primary_device(primary_device);
     // parking with --layer-split saves every stage (SavedConversation::stage_images)
+    if (!o.prompt_cache_file.empty() && (!o.serve || o.prompt_cache <= 0 || o.prompt_cache_root <= 0 ||
+            o.turn_token < 0 || o.peer_device >= 1 ||
+            std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(), [](int slots) { return slots > 0; }))) {
+        std::fprintf(stderr, "strata: --prompt-cache-file requires --serve, enabled root checkpoints, no peer device "
+                             "and no remote expert caches\n");
+        return 2;
+    }
     // Layer split (multi-GPU): the later stages run layers [K_i, K_i+1) on their own GPUs (--split-device, default
     // the next visible ones); "auto" places the K from each GPU's free VRAM once the weights are in (below).  KV
     // streaming and images (the server's GPU encoder) run across a split, also with parking and --batch.  Across
```

### 05-15

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
        // without --mtp the conversation cache stays on: a parked image carries the draft layer's K/V only when there is one
        // (the snapshots accept a null draft)
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 103 lines):
```diff
         bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
         // without --mtp the conversation cache stays on: a parked image carries the draft layer's K/V only when there is one
         // (the snapshots accept a null draft)
+        strata::core::PrefixDigest prefix_identity{};
+        bool prefix_disk_enabled = !o.prompt_cache_file.empty();
+        if (prefix_disk_enabled) {
+            const auto identity_start = Clock::now();
+            std::vector<std::filesystem::path> assets{argv[0], o.pack, o.mtp, o.ple_gguf, o.embd_gguf,
+                o.native_preset, o.native_head_gguf, o.expert_profile};
+            for (const auto& p : o.native_shards) assets.emplace_back(p);
+            for (const auto& p : o.native_head_shards) assets.emplace_back(p);
+            for (const auto& p : o.native_dense_gguf) assets.emplace_back(p);
+            for (const auto& item : o.cvec_files) assets.emplace_back(item.first);
+            std::vector<std::string> settings;
+            // Activations are local to this device/runtime, not portable model files.
+            int device = 0, driver = 0, runtime = 0;
+            cudaDeviceProp device_properties{};
+            if (cudaGetDevice(&device) != cudaSuccess ||
+                    cudaGetDeviceProperties(&device_properties, device) != cudaSuccess ||
+                    cudaDriverGetVersion(&driver) != cudaSuccess || cudaRuntimeGetVersion(&runtime) != cudaSuccess) {
+                cudaGetLastError(); prefix_disk_enabled = false; err = "device/runtime identity unavailable";
+            } else {
+                settings.emplace_back(device_properties.name);
+                settings.push_back(std::to_string(device_properties.major) + ":" +
... 79 more lines: see 05-prefix-disk.patch
```

### 05-16

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            // Preserve the outgoing branch before any checkpoint rewind, reset,
            // or incoming restore overwrites the positional state it requires.
            // pin=N: this request resumes exactly at the pinned shared prefix of the conversation the session holds
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 60 lines):
```diff
                     err.clear();
                 }
             }
+            int64_t incoming_tokens = parked.tokens;
+            bool incoming_live = parked.live, incoming_disk = false;
+            double disk_read_ms = 0;
+            // Only the same first-turn boundary as the ordinary root checkpoint is eligible.
+            // Do not replace a longer in-RAM match or read a disk image for a vision request.
+            if (prefix_disk_enabled && req_imgs.empty() && std::getenv("STRATA_CKPT_REREAD") == nullptr) {
+                int64_t disk_root = -1, last_turn = -1;
+                for (int64_t i = n - 1; i > 0; --i) if (ids[size_t(i)] == o.turn_token) { last_turn = i; break; }
+                for (int64_t i = 1; i < last_turn; ++i) if (ids[size_t(i)] == o.turn_token) { disk_root = i; break; }
+                if (disk_root >= o.prompt_cache_root && disk_root <= int64_t(strata::core::prefix_cache_max_tokens) &&
+                        disk_root > std::max(resume, (incoming || borrow) ? incoming_tokens : int64_t(0))) {
+                    const auto t0 = Clock::now();
+                    std::error_code ec;
+                    const auto bytes = std::filesystem::file_size(o.prompt_cache_file, ec);
+                    const uint64_t floor = uint64_t(o.conversation_cache_min_free_mib) * 1024 * 1024;
+                    const uint64_t held = incoming ? incoming->bytes() : 0;
+                    if (!ec && bytes <= strata::core::prefix_cache_max_bytes + 88 &&
+                            strata::core::conversation_memory_admit(strata::core::available_host_bytes(), bytes + held + 1024 * 1024, floor)) {
+                        std::vector<int32_t> prefix(ids.begin(), ids.begin() + disk_root);
+                        strata::core::SavedConversation image;
+                        // One GPU: the whole-session form, as before.  A layer split: the same checks as a parked
+                        // image - each later stage's image against its own session (the draft K/V with the last
+                        // stage's), then the first stage's without the draft - all before any GPU write.
+                        auto disk_image_valid = [&](const strata::core::SavedConversation& im) -> bool {
+                            if (stages.empty()) return strata::core::conversation_snapshot_validate(im, ss, g, use_mtp ? &mtp.kv_state() : nullptr, err);
+                            if (im.stage_images.size() != stages.size()) { err = "stage count differs"; return false; }
+                            for (size_t i = 0; i < stages.size(); ++i) {
+                                const strata::core::OnDevice on(stages[i]->dev);
+                                if (!strata::core::conversation_snapshot_validate(im.stage_images[i], stages[i]->ss, g,
+                                        use_mtp && i + 1 == stages.size() ? &mtp.kv_state() : nullptr, err)) {
+                                    err = "stage CUDA" + std::to_string(stages[i]->dev) + ": " + err;
+                                    return false;
+                                }
+                            }
+                            return strata::core::conversation_snapshot_validate(im, ss, g,
+                                       static_cast<const strata::core::QsaState*>(nullptr), err);
+                        };
+                        if (strata::core::prefix_cache_read(o.prompt_cache_file, prefix_identity, prefix,
+                                int32_t(ids[size_t(disk_root)]), want_cvec, image, err) &&
+                                disk_image_valid(image)) {
+                            incoming.emplace(std::move(image)); incoming_tokens = disk_root;
+                            incoming_live = true; incoming_disk = true;
+                            borrow = false;   // the disk root is the longer match: a donor stays parked, untouched
+                            slot_source = -1;   // batch slots: the disk root is the longer match
+                        } else {
+                            std::fprintf(stderr, "strata serve: prefix disk cache: miss (%s)\n", err.c_str()); err.clear();
+                        }
+                    } else {
+                        std::fprintf(stderr, "strata serve: prefix disk cache: miss (%s)\n",
+                                     ec ? "file unavailable" : "file size or physical RAM admission");
+                    }
+                    disk_read_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
+                }
+            }
             // Preserve the outgoing branch before any checkpoint rewind, reset,
             // or incoming restore overwrites the positional state it requires.
             // pin=N: this request resumes exactly at the pinned shared prefix of the conversation the session holds
```

### 05-17

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                 mtp.kv_state().kv_mode, "ram",
                                 (long long) (mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size));
                }
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 40 lines):
```diff
                                  mtp.kv_state().kv_mode, "ram",
                                  (long long) (mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size));
                 }
-                live = std::move(incoming->live.ids);
-                live_imgs = std::move(incoming->live.imgs);
-                if (stages.empty()) {
-                    checks = std::move(incoming->checkpoints);
-                } else {   // give each checkpoint back its stage parts (moved, not copied); none if they do not line up
-                    strata::core::ConversationCheckpointSplit cs;
-                    cs.stage0 = std::move(incoming->checkpoints);
-                    for (auto& si : incoming->stage_images) cs.parts.push_back(std::move(si.checkpoints));
-                    if (!strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear();
+                if (incoming_disk) {
+                    live = incoming->live.ids;
+                    live_imgs.clear(); checks.clear();
+                    incoming->live.used = ++check_clock;
+                    // a layer split: the root checkpoint carries each later stage's running state in its stage
+                    // parts, as checkpoint_at saves one (the stage images hold the same ids; restored above)
+                    for (auto& si : incoming->stage_images) incoming->live.stage_parts.push_back(std::move(si.live));
+                    checks.push_back(std::move(incoming->live)); // keep the restored root pinned in RAM
+                } else {
+                    live = std::move(incoming->live.ids);
+                    live_imgs = std::move(incoming->live.imgs);
+                    if (stages.empty()) {
+                        checks = std::move(incoming->checkpoints);
+                    } else {   // give each checkpoint back its stage parts (moved, not copied); none if they do not line up
+                        strata::core::ConversationCheckpointSplit cs;
+                        cs.stage0 = std::move(incoming->checkpoints);
+                        for (auto& si : incoming->stage_images) cs.parts.push_back(std::move(si.checkpoints));
+                        if (!strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear();
+                    }
                 }
                 cvec_cached = incoming->cvec;
-                resume = parked.tokens;
-                from_live = parked.live;
+                resume = incoming_tokens;
+                from_live = incoming_live;
                 if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                     std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                     for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
```

### 05-18

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                }
                incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                lent_slot = -1;   // (the main session holds the parked conversation now)
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                 }
                 incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                 lent_slot = -1;   // (the main session holds the parked conversation now)
+                if (incoming_disk)
+                    std::fprintf(stderr, "strata serve: prefix disk cache: restored %lld tokens; read/validate %.1f ms, upload %.1f ms\n",
+                                 (long long) resume, disk_read_ms,
+                                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                 std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu\n",
                              (long long) resume, from_live ? "live" : "checkpoint",
                              std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
```

### 05-19

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        if (i >= o.prompt_cache_root) root_at = i;
                        break;
                    }
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 11 lines):
```diff
                         if (i >= o.prompt_cache_root) root_at = i;
                         break;
                     }
+            // a root past the drafter's window (Claude Code's ~40K system turn): its draft K/V are computed in full, or
+            // the prefix file below would skip it as incomplete; costs the draft layer's K/V of the extra cells once
+            if (use_mtp)
+                mtp.set_full_prefix(prefix_disk_enabled && root_at > 0 && req_imgs.empty() &&
+                                    root_at <= int64_t(strata::core::prefix_cache_max_tokens));
             static const bool message_checkpoint = [] {
                 const char* e = std::getenv("STRATA_CACHE_MESSAGE_BOUNDARY");
                 return e != nullptr && std::atoi(e) != 0;
```

### 05-20

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                if (trace && to == message_at)
                    std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                                 (long long) message_at, (long long) (turn_at - message_at));
```
Block (hunk 12 of `src/program/generate.cpp` in the patch, 72 lines):
```diff
                 if (trace && to == message_at)
                     std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                                  (long long) message_at, (long long) (turn_at - message_at));
+                // A long first request may skip old draft cells outside MTP's window.
+                // Persist only complete draft prefixes, usable by later shorter requests too.
+                if (to == root_at && prefix_disk_enabled && req_imgs.empty() && mtp.first_needed() > 0)
+                    std::fprintf(stderr, "strata serve: prefix disk cache: skip saving (incomplete windowed draft prefix)\n");
+                if (to == root_at && prefix_disk_enabled && req_imgs.empty() && mtp.first_needed() <= 0 &&
+                        to <= int64_t(strata::core::prefix_cache_max_tokens)) {
+                    const auto save_start = Clock::now();
+                    try {
+                        const std::vector<int32_t> prefix(ids.begin(), ids.begin() + to);
+                        const std::vector<ImgKey> no_images;
+                        const std::vector<ConvCheckpoint> no_checkpoints;
+                        const strata::core::ConversationView view{prefix, no_images, no_checkpoints, want_cvec};
+                        size_t estimate = 0;
+                        const uint64_t floor = uint64_t(o.conversation_cache_min_free_mib) * 1024 * 1024;
+                        // A layer split saves one image per stage, as parking does: the first stage's without the
+                        // draft, each later stage's on its own device, the draft K/V with the last one; the size is
+                        // the sum of the stages'.  Every stage is at `to` here (checkpoint_at just read them all).
+                        const size_t n_st = stages.size();
+                        const strata::core::QsaState* const no_draft = nullptr;
+                        auto disk_draft_of = [&](size_t k) -> const strata::core::QsaState* {
+                            return use_mtp && k + 1 == n_st ? &mtp.kv_state() : nullptr;
+                        };
+                        bool sized = n_st == 0
+                            ? strata::core::conversation_snapshot_bytes(view, ss, g, use_mtp ? &mtp.kv_state() : nullptr, estimate, err)
+                            : strata::core::conversation_snapshot_bytes(view, ss, g, no_draft, estimate, err);
+                        for (size_t k = 0; sized && k < n_st; ++k) {
+                            const strata::core::OnDevice on(stages[k]->dev);
+                            size_t stage_bytes = 0;
+                            sized = strata::core::conversation_snapshot_bytes(view, stages[k]->ss, g, disk_draft_of(k),
+                                                                              stage_bytes, err) &&
+                                    stage_bytes <= SIZE_MAX - estimate;
+                            if (sized) estimate += stage_bytes;
+                        }
+                        if (sized &&
+                                estimate <= strata::core::prefix_cache_max_bytes &&
+                                strata::core::conversation_memory_admit(strata::core::available_host_bytes(), estimate, floor)) {
+                            strata::core::SavedConversation image;
+                            bool captured = n_st == 0
+                                ? strata::core::conversation_snapshot_save(image, view, ss, g, use_mtp ? &mtp.kv_state() : nullptr, err)
+                                : strata::core::conversation_snapshot_save(image, view, ss, g, no_draft, err);
+                            for (size_t k = 0; captured && k < n_st; ++k) {   // the later stages, in stage order
+                                const strata::core::OnDevice on(stages[k]->dev);
+                                strata::core::SavedConversation part;
+                                captured = strata::core::conversation_snapshot_save(part, view, stages[k]->ss, g,
+                                                                                    disk_draft_of(k), err);
+                                if (!captured) err = "stage CUDA" + std::to_string(stages[k]->dev) + ": " + err;
+                                else image.stage_images.push_back(std::move(part));
+                            }
+                            if (!captured) {
+                                // A CUDA copy/synchronization error can poison the live session.
+                                std::printf("ERR capturing prefix disk cache: %s\n", err.c_str()); return 1;
+                            }
+                            const bool stored = strata::core::prefix_cache_write(o.prompt_cache_file, prefix_identity,
+                                                    int32_t(ids[size_t(to)]), image, err);
+                            std::fprintf(stderr, "strata serve: prefix disk cache: %s %lld tokens, %zu bytes in %.1f ms%s%s\n",
+                                         stored ? "saved" : "skip saving", (long long) to, image.bytes(),
+                                         std::chrono::duration<double, std::milli>(Clock::now() - save_start).count(),
+                                         stored ? "" : ": ", stored ? "" : err.c_str());
+                        } else {
+                            std::fprintf(stderr, "strata serve: prefix disk cache: skip saving (size, geometry or physical RAM admission)\n");
+                        }
+                    } catch (const std::bad_alloc&) {
+                        std::fprintf(stderr, "strata serve: prefix disk cache: skip saving (host allocation failed)\n");
+                    }
+                    err.clear(); // A disk miss/write failure never changes the normal prefill error channel.
+                }
             }
             if (!refill(err)) {
                 std::printf("ERR refilling a lent slot failed: %s\n", err.c_str());
```

## 06 vram-free-nvml

`06-vram-free-nvml.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la PR upstream #1911: VRAM libera della scheda intera via NVML (STRATA_VRAM_FREE=device), letta anche dalla cache elastica

### 06-1

`docs/DETAILS.md` · IQ3_XXS -39%, IQ2_XXS -34%, IQ2_XS -24%, IQ2_S -3% on a P-core.  End to end, wit

Anchor:
```
window falls 5% on a Xeon E5-2643 v3 + RTX A4000 (22.2 -> 21.1 ms of a 46.9 ms window, decode +0.5% over 10 interleaved pairs,
+2.5% on the story and code prompts) and on a Core Ultra 7 265 + RTX 3060 (8.2 -> 7.8 ms of 39.5 ms, decode +0.4%), and does not
move on a Threadripper PRO 3975WX + R9700 (14.5 ms, +0.4%, 5 pairs); no case reached +3% end to end, so the switch stays off.
```
Block (hunk 1 of `docs/DETAILS.md` in the patch, 17 lines):
```diff
 window falls 5% on a Xeon E5-2643 v3 + RTX A4000 (22.2 -> 21.1 ms of a 46.9 ms window, decode +0.5% over 10 interleaved pairs,
 +2.5% on the story and code prompts) and on a Core Ultra 7 265 + RTX 3060 (8.2 -> 7.8 ms of 39.5 ms, decode +0.4%), and does not
 move on a Threadripper PRO 3975WX + R9700 (14.5 ms, +0.4%, 5 pairs); no case reached +3% end to end, so the switch stays off.
+- `STRATA_VRAM_FREE=device` (Windows, NVIDIA): `--expert-cache auto` reads the free VRAM of the whole card through NVML.
+  Under WDDM CUDA's own figure counts this process only: with 6 GiB held by another process it still read 30,994 MiB
+  free where NVML read 24,348. With the image encoder loaded (1,547 MiB) the cache was therefore sized too large and
+  shrunk by guesses: six starts of one config (engine 0.1.40.3 and 0.1.41, the default 700 MiB reserve) ended with
+  9,002 to 14,454 slots and 93 MiB to 10 GiB free. With the switch and `--vram-reserve-mib 1500` (0.1.41 built from
+  source): one pass, 13,007 to 13,188 slots and 973 to 1,392 MiB free over seven starts. On 0.1.42 with this switch
+  built in, three interleaved starts each: the default sized 13,543 slots after two shrink steps and ended with 336,
+  602 and 602 MiB free; the switch with a 1,500 MiB reserve 13,102 to 13,130 slots after one step and 1,107 to 1,288
+  MiB free. A run that filled the card to
+  0 MiB free decoded 80-115 tok/s where the same text ran at 205-230 with 900 MiB or more free. Not measured: AMD
+  cards, Linux (the switch is compiled for Windows CUDA builds only).

 **Switches added in 0.1.40 (all off unless noted; none changes the default output):**
 - `--kv-grow` (or `STRATA_KV_GROW=1`; `--no-kv-grow` turns it off): the K/V takes VRAM only for the cells the requests
```

### 06-2

`include/strata/core/expert_cache.hpp` · bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_e

Anchor:
```
/// Free device memory for the expert cache.  On a unified memory system - one memory for the CPU and the GPU, such as
/// the DGX Spark (GB10); CUDA reports it as cudaDevAttrIntegrated - cudaMemGetInfo counts only MemFree, not the page cache - which, with the experts read from the GGUF in place, is mostly those
/// files' clean pages and gives way to a device allocation.  There the figure is MemAvailable less 6 GiB for the OS
```
Block (hunk 1 of `include/strata/core/expert_cache.hpp` in the patch, 10 lines):
```diff
 /// Free device memory for the expert cache.  On a unified memory system - one memory for the CPU and the GPU, such as
 /// the DGX Spark (GB10); CUDA reports it as cudaDevAttrIntegrated - cudaMemGetInfo counts only MemFree, not the page cache - which, with the experts read from the GGUF in place, is mostly those
 /// files' clean pages and gives way to a device allocation.  There the figure is MemAvailable less 6 GiB for the OS
-/// and the engine's host side (STRATA_UMA_HEADROOM_GIB).  Elsewhere it is cudaMemGetInfo's.
+/// and the engine's host side (STRATA_UMA_HEADROOM_GIB).  Elsewhere it is cudaMemGetInfo's.  On Windows with an NVIDIA
+/// card STRATA_VRAM_FREE=device (opt-in) takes the smaller of that and NVML's figure for the whole card: under WDDM
+/// cudaMemGetInfo does not count what other processes hold.
 size_t device_free_bytes();

 /// One piece of an expert blob that lives contiguously in a source's own memory: `n` bytes at `p`, to land `at` bytes into the blob.
```

### 06-3

`src/core/expert_cache.cpp` · (file scope)

Anchor:
```
#include <utility>
#include <cstring>
```
Block (hunk 1 of `src/core/expert_cache.cpp` in the patch, 61 lines):
```diff
 #include <utility>
 #include <cstring>

+#if defined(_WIN32) && !defined(STRATA_EC_NO_VMM)
+#ifndef WIN32_LEAN_AND_MEAN
+#define WIN32_LEAN_AND_MEAN
+#endif
+#ifndef NOMINMAX
+#define NOMINMAX
+#endif
+#include <windows.h>
+#endif
+
 namespace strata::core {

+namespace {
+
+// STRATA_VRAM_FREE=device (opt-in, Windows + NVIDIA).  Under WDDM cudaMemGetInfo answers for THIS process: memory
+// another process holds on the card (the image encoder, a browser, another engine) is not in its free figure.
+// Measured on an RTX 5090, driver 610.88: with 6 GiB written by another process, cudaMemGetInfo in a second process
+// still read 30,994 MiB free where NVML read 24,348.  NVML (nvml.dll, installed with every driver) counts the whole
+// card.  The library is loaded from the system directory only, at first use; false when it cannot answer.
+bool nvml_device_free(size_t& out) {
+    struct Memory { unsigned long long total, free, used; };
+    typedef int (*InitFn)(void);
+    typedef int (*ByBusFn)(const char*, void**);
+    typedef int (*MemFn)(void*, Memory*);
+    struct Lib {
+        ByBusFn by_bus = nullptr;
+        MemFn mem = nullptr;
+        Lib() {
+            HMODULE h = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
+            if (h == nullptr) return;
+            const auto init = (InitFn) (void*) GetProcAddress(h, "nvmlInit_v2");
+            const auto bus = (ByBusFn) (void*) GetProcAddress(h, "nvmlDeviceGetHandleByPciBusId_v2");
+            const auto m = (MemFn) (void*) GetProcAddress(h, "nvmlDeviceGetMemoryInfo");
+            if (init == nullptr || bus == nullptr || m == nullptr || init() != 0) return;
+            by_bus = bus;
+            mem = m;
+        }
+    };
+    static const Lib lib;
+    if (lib.mem == nullptr) return false;
+    int dev = 0;
+    cudaDeviceProp prop{};
+    if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
+    char bus[32];
+    std::snprintf(bus, sizeof bus, "%08X:%02X:%02X.0", (unsigned) prop.pciDomainID, (unsigned) prop.pciBusID,
+                  (unsigned) prop.pciDeviceID);
+    void* handle = nullptr;
+    Memory m{};
+    if (lib.by_bus(bus, &handle) != 0 || lib.mem(handle, &m) != 0) return false;
+    out = (size_t) m.free;
+    return true;
+}
+
+}  // namespace
+
 // See the header.  NVIDIA's DGX Spark Porting Guide (section 5.5, "Memory reporting on UMA systems") recommends the same:
 // not to rely on cudaMemGetInfo alone but to count the memory the OS can reclaim.  Swap is not counted here (unlike
 // NVIDIA's reference snippet): an expert cache that pushes the system into swap would be far slower than a smaller one.
```

### 06-4

`src/core/expert_cache.cpp` · size_t device_free_bytes() {

Anchor:
```
        }
    }
#endif
```
Block (hunk 2 of `src/core/expert_cache.cpp` in the patch, 25 lines):
```diff
         }
     }
 #endif
+    static const bool device_wide = [] {
+        const char* v = std::getenv("STRATA_VRAM_FREE");
+        return v != nullptr && std::strcmp(v, "device") == 0;
+    }();
+    if (device_wide) {
+        size_t nvml_b = 0;
+        const bool got = nvml_device_free(nvml_b);
+        static std::atomic<bool> said{false};
+        if (!said.exchange(true)) {
+            if (got)
+                std::fprintf(stderr, "strata generate: STRATA_VRAM_FREE=device: free VRAM is read for the whole card "
+                                     "(NVML: %lld MiB now; CUDA's figure for this process alone: %lld MiB)\n",
+                             (long long) (nvml_b >> 20), (long long) (free_b >> 20));
+            else
+                std::fprintf(stderr, "strata generate: STRATA_VRAM_FREE=device: NVML did not answer (nvml.dll); free "
+                                     "VRAM stays CUDA's figure for this process\n");
+        }
+        if (got) free_b = std::min(free_b, nvml_b);
+    }
     return free_b;
 }

```

### 06-5

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        std::deque<std::pair<Clock::time_point, int64_t>> el_seen;
        Clock::time_point el_last_grow{}, el_last_shrink{};
        int el_backoff = 1;
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 15 lines):
```diff
         std::deque<std::pair<Clock::time_point, int64_t>> el_seen;
         Clock::time_point el_last_grow{}, el_last_shrink{};
         int el_backoff = 1;
+        // upsync: with STRATA_VRAM_FREE=device (#1911) the whole card's free VRAM (NVML), the smaller of the two: under
+        // WDDM cudaMemGetInfo counts this process only, so the desktop and other programs on CUDA0 (the card with the
+        // display) were invisible to the policy and the cache grew until Windows paged.  Without it, as before.
         auto el_free = []() -> int64_t {
             const strata::core::OnDevice on(0);   // the elastic cache's card (CUDA0)
             size_t f = 0, t = 0;
-            return cudaMemGetInfo(&f, &t) == cudaSuccess ? (int64_t) f : -1;
+            if (cudaMemGetInfo(&f, &t) != cudaSuccess) return -1;
+            return (int64_t) std::min(f, strata::core::device_free_bytes());
         };
         // the tail moved: the prompt path's chunk and first lendable slot are planned again by the startup's own rule
         // (plan_lend: auto takes the largest chunk that lends at most its share of the slots), and the next lend lays
```

### 06-6

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            // and a page-in while the verify graph spins on a host flag stalls the request for good
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             // and a page-in while the verify graph spins on a host flag stalls the request for good
             size_t free_b = 0, total_b = 0;
             cudaMemGetInfo(&free_b, &total_b);
+            // STRATA_VRAM_FREE=device: the whole card's figure when it is the smaller one (the same value otherwise)
+            free_b = std::min(free_b, strata::core::device_free_bytes());
             // below ~256 MiB a later allocation (a first-used window's buffers, the desktop, another program) can make
             // the driver page GPU memory, and a verify graph spinning on a host flag then never finishes
             const int64_t free_mib = (int64_t) (free_b >> 20);
```

## 07 park-no-foreign-kv

`07-park-no-foreign-kv.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la PR upstream #1503: mai parcheggiare una conversazione col K/V di un'altra

### 07-1

`include/strata/core/conversation_cache.hpp` · struct ConversationKvReuse {

Anchor:
```
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
```
Block (hunk 1 of `include/strata/core/conversation_cache.hpp` in the patch, 15 lines):
```diff
     std::vector<ConversationKv> kv;
     // Original image extent for validation, and the earliest subsequent rewrite.
     int64_t captured_tokens = 0, unchanged_tokens = 0;
+    // The conversation the image was captured from: a park reuses it only for the prefix it shares with the
+    // conversation then in the main session (a slot may have replaced that session's K/V since).
+    std::vector<int32_t> ids;
     // With a layer split: the later stages' own retained K/V, one per stage, same extents (empty: none)
     std::vector<ConversationKvReuse> stages;
     size_t bytes() const {
-        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse);
+        size_t n = kv.capacity() * sizeof(ConversationKv) + stages.capacity() * sizeof(ConversationKvReuse) +
+                   ids.capacity() * sizeof(int32_t);
         for (const auto& layer : kv) n += layer.bytes();
         for (const auto& s : stages) n += s.bytes();
         return n;
```

### 07-2

`include/strata/core/conversation_cache.hpp` · public:

Anchor:
```
    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
```
Block (hunk 2 of `include/strata/core/conversation_cache.hpp` in the patch, 15 lines):
```diff
     // Retain only the restored K/V buffers, not duplicate running checkpoints.
     // This optimization never evicts a parked conversation to make itself fit.
     // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
+    // `ids`: the conversation the K/V belongs to (take_reuse_for refuses it for any other; none: never reused).
     void retain(std::vector<ConversationKv>&& kv, int64_t tokens,
-                std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
+                std::vector<std::vector<ConversationKv>>&& stage_kv = {}, std::vector<int32_t> ids = {}) {
         reuse_ = {};
-        ConversationKvReuse candidate{std::move(kv), tokens, tokens, {}};
-        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}});
+        ConversationKvReuse candidate{std::move(kv), tokens, tokens, std::move(ids), {}};
+        for (auto& k : stage_kv) candidate.stages.push_back(ConversationKvReuse{std::move(k), tokens, tokens, {}, {}});
         if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
     }
     void limit_reuse(int64_t first_dirty) {
```

### 07-3

`include/strata/core/conversation_cache.hpp` · public:

Anchor:
```
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
```
Block (hunk 3 of `include/strata/core/conversation_cache.hpp` in the patch, 22 lines):
```diff
         if (reuse_.unchanged_tokens <= 0) reuse_ = {};
     }
     ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
+    // The retained K/V for parking `live`: limited to the prefix `live` shares with the conversation it was captured
+    // from, and nothing when they share none.  Without this a park after a slot gave its conversation back to the main
+    // session wrote the RESTORED conversation's K/V into the new one's image (2026-10-08 cross-conversation leak).
+    template<class Token>
+    ConversationKvReuse take_reuse_for(const std::vector<Token>& live) {
+        ConversationKvReuse r = std::exchange(reuse_, {});
+        const int64_t n = std::min<int64_t>((int64_t) r.ids.size(), (int64_t) live.size());
+        int64_t same = 0;
+        while (same < n && (int64_t) r.ids[(size_t) same] == (int64_t) live[(size_t) same]) ++same;
+        r.unchanged_tokens = std::min(r.unchanged_tokens, same);
+        for (auto& st : r.stages) st.unchanged_tokens = std::min(st.unchanged_tokens, same);
+        if (r.unchanged_tokens <= 0) return {};
+        return r;
+    }
     size_t retained_bytes() const { return reuse_.bytes(); }
+    int64_t retained_tokens() const { return reuse_.kv.empty() ? 0 : reuse_.unchanged_tokens; }
     bool can_fit(size_t incoming, size_t held = 0) const {
         return enabled() && held <= budget_ && incoming <= budget_ - held &&
                entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
```

### 07-4

`src/core/conversation_cache_test.cpp` · int main() {

Anchor:
```
        auto none = conversation_checkpoints_split({}, 3);
        check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
    }
```
Block (hunk 1 of `src/core/conversation_cache_test.cpp` in the patch, 41 lines):
```diff
         auto none = conversation_checkpoints_split({}, 3);
         check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
     }
+    {
+        // 2026-10-08 cross-conversation leak: the retained K/V of a restored conversation must never be reused to
+        // park ANOTHER conversation the main session later holds (a slot gave its conversation back)
+        auto kv = [](size_t n) {
+            std::vector<ConversationKv> v(2);
+            for (auto& l : v) l.k.resize(n, 1);
+            return v;
+        };
+        ConversationCache cache(1 << 20, 4);
+        const std::vector<int32_t> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
+        std::vector<std::vector<ConversationKv>> st;
+        st.push_back(kv(100));
+        cache.retain(kv(500), 16, std::move(st), a);
+        std::vector<int32_t> b(16, 99);
+        auto other = cache.take_reuse_for(b);
+        check(other.kv.empty() && other.stages.empty() && other.unchanged_tokens == 0,
+              "another conversation never gets the retained K/V");
+        check(cache.retained_bytes() == 0, "the retained K/V is dropped once refused");
+        cache.retain(kv(500), 16, {}, a);
+        check(cache.retained_tokens() == 16, "the retained extent is reported");
+        std::vector<int32_t> cont(a.begin(), a.begin() + 10);
+        cont.push_back(42);
+        cont.push_back(43);
+        auto partly = cache.take_reuse_for(cont);
+        check(partly.unchanged_tokens == 10, "a conversation that diverges reuses only the shared prefix");
+        std::vector<std::vector<ConversationKv>> st2;
+        st2.push_back(kv(100));
+        cache.retain(kv(500), 16, std::move(st2), a);
+        cache.limit_reuse(9);
+        auto same = cache.take_reuse_for(a);
+        check(same.unchanged_tokens == 9 && same.stages.size() == 1 && same.stages[0].unchanged_tokens == 9,
+              "the same conversation keeps the first rewrite's limit, on every stage");
+        cache.retain(kv(500), 16);
+        check(cache.take_reuse_for(a).kv.empty(), "retained K/V of unknown identity is never reused");
+    }
     {
         // the later stages' retained K/V, kept with the first stage's and limited together
         auto kv = [](size_t n) {
```

### 07-5

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
            } merge_back{cs, checks, n_st > 0};
            const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
                 ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
             } merge_back{cs, checks, n_st > 0};
             const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
-            auto reuse = conversations.take_reuse();
+            const int64_t retained_n = conversations.retained_tokens();
+            auto reuse = conversations.take_reuse_for(live);   // never another conversation's K/V
+            if (retained_n > 0 && reuse.unchanged_tokens < retained_n)
+                std::fprintf(stderr, "strata serve: conversation cache: retained K/V reused for %lld of its %lld tokens "
+                                     "(the rest is not this conversation's)\n", (long long) reuse.unchanged_tokens,
+                             (long long) retained_n);
             std::vector<strata::core::ConversationKvReuse> stage_reuse = std::move(reuse.stages);
             stage_reuse.resize(n_st);
             reuse.stages.clear();
```

### 07-6

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // conversation an idle slot holds reads on from there.  Every stage, the same calls.
        // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
        auto copy_from_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
         // conversation an idle slot holds reads on from there.  Every stage, the same calls.
         // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
         auto copy_from_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
+            // the main session's K/V is replaced by the slot's: K/V retained from a restore described the old one
+            conversations.take_reuse();
             const int64_t upto = at != nullptr ? (int64_t) at->ids.size() : (int64_t) bs[(size_t) b].ids.size();
             if (at != nullptr && at->stage_parts.size() != stages.size()) { e = "a checkpoint without its stage parts"; return false; }
             if (slot_copy_direct && at == nullptr) {
```

### 07-7

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                    std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                    for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                 if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                     std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                     for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
-                    conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv));
+                    conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv), live);
                 }
                 incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                 lent_slot = -1;   // (the main session holds the parked conversation now)
```

### 07-8

`sycl/src/program/generate.cpp` · int main(int argc, char **argv) try {

Anchor:
```
                ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
            } merge_back{cs, checks, n_st > 0};
            const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
```
Block (hunk 1 of `sycl/src/program/generate.cpp` in the patch, 13 lines):
```diff
                 ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
             } merge_back{cs, checks, n_st > 0};
             const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
-            auto reuse = conversations.take_reuse();
+            const int64_t retained_n = conversations.retained_tokens();
+            auto reuse = conversations.take_reuse_for(live);   // never another conversation's K/V
+            if (retained_n > 0 && reuse.unchanged_tokens < retained_n)
+                std::fprintf(stderr, "strata serve: conversation cache: retained K/V reused for %lld of its %lld tokens "
+                                     "(the rest is not this conversation's)\n", (long long) reuse.unchanged_tokens,
+                             (long long) retained_n);
             std::vector<strata::core::ConversationKvReuse> stage_reuse = std::move(reuse.stages);
             stage_reuse.resize(n_st);
             reuse.stages.clear();
```

### 07-9

`sycl/src/program/generate.cpp` · int main(int argc, char **argv) try {

Anchor:
```
        // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
        auto copy_from_slot = [&](int b, const ConvCheckpoint *at,
                                  std::string &e) -> bool {
```
Block (hunk 2 of `sycl/src/program/generate.cpp` in the patch, 8 lines):
```diff
         // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
         auto copy_from_slot = [&](int b, const ConvCheckpoint *at,
                                   std::string &e) -> bool {
+            // the main session's K/V is replaced by the slot's: K/V retained from a restore described the old one
+            conversations.take_reuse();
             try {
         const int64_t upto = at != nullptr ? (int64_t)at->ids.size()
                                            : (int64_t)bs[(size_t)b].ids.size();
```

### 07-10

`sycl/src/program/generate.cpp` · int main(int argc, char **argv) try {

Anchor:
```
                if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                    std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                    for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
```
Block (hunk 3 of `sycl/src/program/generate.cpp` in the patch, 8 lines):
```diff
                 if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                     std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                     for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
-                    conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv));
+                    conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv), live);
                 }
                 incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                 std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu\n",
```

## 08 pipeline-architectds

`08-pipeline-architectds.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il profilo architectds della pipeline (STRATA_PIPELINE_ELASTIC) e i log di rollback e latenza della catena

### 08-1

`include/strata/core/verify.hpp` · public:

Anchor:
```
    /// window as they WILL be, their pages prefetched).  A later `pl_launch` of the same window (T, pos0, tokens, and
    /// `ss.ple_prev` equal to `ple_prev` by then) skips the staging; anything else stages again.
    bool prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
```
Block (hunk 1 of `include/strata/core/verify.hpp` in the patch, 12 lines):
```diff
     /// window as they WILL be, their pages prefetched).  A later `pl_launch` of the same window (T, pos0, tokens, and
     /// `ss.ple_prev` equal to `ple_prev` by then) skips the staging; anything else stages again.
     bool prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
-    /// 1: every layer served; 0: the GPU has not reached the next layer yet; -1: an error (`err`).
-    int service(PoolMultiFn pool, void* user, std::string& err);
+    /// 1: every layer served; 0: the GPU has not reached the next layer yet (or `max_layers` were served this call);
+    /// -1: an error (`err`).  `max_layers` > 0: serve at most that many layers, so the caller can turn to another
+    /// window in between (the pipelined decode serves a speculative window one layer at a time; architectds 6175807a).
+    int service(PoolMultiFn pool, void* user, std::string& err, int max_layers = 0);
     bool in_flight() const { return fl_active_; }
     /// The window's graph (and its profile copy) completed; false while it runs.  An error sets `err`.
     bool done(std::string& err);
```

### 08-2

`src/core/verify.cpp` · bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string

Anchor:
```
int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {
```
Block (hunk 1 of `src/core/verify.cpp` in the patch, 8 lines):
```diff
     return true;
 }

-int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {
+int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_layers) {
     if (!fl_active_) return 1;
     if (fl_k_ >= fl_total_) return 1;
     // Nothing rang yet and neither the WDDM flush nor the timeout below is due: return before any CUDA call.  The
```

### 08-3

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {

Anchor:
```
    }
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
```
Block (hunk 2 of `src/core/verify.cpp` in the patch, 9 lines):
```diff
     }
     const int G = groups_[T] > 0 ? groups_[T] : 1;
     const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
+    int served = 0;   // this call's layers (a layer's halves count apart when its window is split in two groups)
     while (fl_k_ < fl_total_) {
+        if (max_layers > 0 && served >= max_layers) return 0;
         const int64_t l = lb_ + fl_k_ / G;
         const uint32_t want = (uint32_t) (fl_k_ + 1);
         if (*(volatile uint32_t*) h_seq_ < want) {
```

### 08-4

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {

Anchor:
```
        if (fl_k_ == 0 && !gather_ple()) return -1;
        *(volatile uint32_t*) h_flag_ = want;
        ++fl_k_;
```
Block (hunk 3 of `src/core/verify.cpp` in the patch, 7 lines):
```diff
         if (fl_k_ == 0 && !gather_ple()) return -1;
         *(volatile uint32_t*) h_flag_ = want;
         ++fl_k_;
+        ++served;
         ms_pool += ms_since(b);
         fl_since_ms_ = fl_flush_ms_ = now_ms();
     }
```

### 08-5

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            int pl_pw = pipe ? o.pipeline_windows : 0;
            float pl_theta = [] {   // a speculative window is launched only when its estimated chance is at least this
                const char* v = pipe_dbg_env("STRATA_PIPELINE_THETA");
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
             int pl_pw = pipe ? o.pipeline_windows : 0;
             float pl_theta = [] {   // a speculative window is launched only when its estimated chance is at least this
                 const char* v = pipe_dbg_env("STRATA_PIPELINE_THETA");
-                return v ? (float) std::atof(v) : 0.2f;
+                // fork (Daily 5140210d + 83c27bbd), architectds d6f62341: 0.10 once the verified path is served first (a
+                // wrong guess costs little then); measured there, RTX 3060 + RTX 5070 Ti, against 0.20: 150K +2.4%, short
+                // chats -0.1%.  Only in the STRATA_PIPELINE_ELASTIC profile (on by default with two cards and
+                // --pipeline-windows, see pipe_elastic_env), where the verified path is served first; else 0.20
+                return v ? (float) std::atof(v) : pipe_elastic_env() ? 0.10f : 0.2f;
             }();
             int pl_force_miss = [] {   // exactness test: every k-th speculative window gets a wrong first token
                 const char* v = pipe_dbg_env("STRATA_PIPELINE_FORCE_MISS");
```

### 08-6

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    bool spec = false;     // launched speculatively (behind the window before, ahead of its verdict)
                    bool made = false;     // made from a chain (its outcome can be scored even when the gate held it)
                    bool sfx = false;      // a lookup window (the suffix drafter's drafts)
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     bool spec = false;     // launched speculatively (behind the window before, ahead of its verdict)
                     bool made = false;     // made from a chain (its outcome can be scored even when the gate held it)
                     bool sfx = false;      // a lookup window (the suffix drafter's drafts)
+                    bool lookup = false;   // fork: a B taken from the lookup's continuation (pick_lookup, lookup_next)
                     int sfx_match = 0;
                 };
                 // front stage `st`'s verifier of w's parity, and the last stage's
```

### 08-7

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                bool b_done = false;     // B has been made from the chain in flight
                int chain_n = 0;
                int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                 bool b_done = false;     // B has been made from the chain in flight
                 int chain_n = 0;
                 int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
+                double pl_chain_t0 = 0.0, pl_chain_ms[2] = {0, 0}, pl_chain_n[2] = {0, 0};   // fork: chain -> B latency
                 // fork (Daily 1541e340): the elastic tail beside the windows, at the gaps of stage 0 (see el_gap)
                 int64_t pl_el_shrinks = 0, pl_el_grows = 0;
                 double pl_el_ms = 0.0;
```

### 08-8

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                  std::chrono::duration<double, std::milli>(Clock::now() - pl_tr0).count(), ev, seq, x0, x1);
                    pl_trace += b;
                };
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                                   std::chrono::duration<double, std::milli>(Clock::now() - pl_tr0).count(), ev, seq, x0, x1);
                     pl_trace += b;
                 };
-                auto pump0 = [&](PW& w) -> bool {   // w's front stage in flight
+                // w's front stage in flight.  `max_layers` > 0: serve at most that many of w's layers now (fork, Daily
+                // 5140210d: see the loop's order below)
+                auto pump0 = [&](PW& w, int max_layers = 0) -> bool {
                     if (!w.fl) return true;
                     strata::core::Verifier& v = VF(w.fr, w);
                     // a doomed window (its guess was wrong) gets empty plans: no expert work on the CPU or the GPU for
```

### 08-9

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    // undo restores everything it wrote).  STRATA_PIPELINE_DOOM_SKIP=0: served in full.
                    static const bool doom_skip = [] { const char* e = pipe_dbg_env("STRATA_PIPELINE_DOOM_SKIP"); return e == nullptr || std::atoi(e) != 0; }();
                    const bool skip = doom_skip && doomed && &w == &D;
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                     // undo restores everything it wrote).  STRATA_PIPELINE_DOOM_SKIP=0: served in full.
                     static const bool doom_skip = [] { const char* e = pipe_dbg_env("STRATA_PIPELINE_DOOM_SKIP"); return e == nullptr || std::atoi(e) != 0; }();
                     const bool skip = doom_skip && doomed && &w == &D;
-                    if (v.service(skip ? nullptr : &drive_pool_split, SDf(w), err) < 0) return false;
+                    if (v.service(skip ? nullptr : &drive_pool_split, SDf(w), err, max_layers) < 0) return false;
                     if (v.done(err)) {
                         if (!v.pl_finish(nullptr, err)) return false;
                         w.fl = false;
```

### 08-10

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        B.p_on = pl_lookup_pon;
                        B.ready = true;
                        B.made = true;
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 65 lines):
```diff
                         B.p_on = pl_lookup_pon;
                         B.ready = true;
                         B.made = true;
+                        B.lookup = true;
                         b_done = true;   // the chain in flight makes no B for this A
                         tre("CD", A.seq + 1, 3, (int) (1000.0f * B.p_on));
                     }
                 };
+                // fork (Daily 5140210d + 83c27bbd), architectds 6175807a / d6f62341 / f45102d4: the next B of a copy.  Once
+                // a B taken from the lookup was on the path (it is A now), the lookup goes on past it, so the window after
+                // it is taken from the lookup too, not from the drafter's chain.  LOOKUP_ANY: also after a window the
+                // chain made, when the lookup runs through its drafts with a match of 12+ tokens and the policy's rate for
+                // such matches is >= 0.85 (that rate is B's p_on).  The chain still runs; an off-path verdict starts
+                // afresh.  On by default only in the STRATA_PIPELINE_ELASTIC profile (on with two cards and
+                // --pipeline-windows); STRATA_PIPELINE_LOOKUP_NEXT / LOOKUP_ANY = 0/1 (read under STRATA_PIPELINE_DEBUG=1)
+                // set them either way.
+                static const bool pl_lookup_next = [] {
+                    const char* v = pipe_dbg_env("STRATA_PIPELINE_LOOKUP_NEXT");
+                    return v != nullptr ? std::atoi(v) != 0 : pipe_elastic_env();
+                }();
+                static const bool pl_lookup_any = [] {
+                    const char* v = pipe_dbg_env("STRATA_PIPELINE_LOOKUP_ANY");
+                    return v != nullptr ? std::atoi(v) != 0 : pipe_elastic_env();
+                }();
+                int64_t pl_lk_next = 0, pl_lk_any = 0;
+                auto lookup_next = [&]() {
+                    if (!pl_lookup_next || o.suffix_draft <= 0 || pl_lookup_pon <= 0.0f || b_done) return;
+                    if (!A.lookup && !pl_lookup_any) return;
+                    const int k = sfx.propose(2 * S - 1, pl_sbuf.data());   // after A's row 0 (its predecessor's bonus)
+                    if (k < A.T) return;                                     // A's drafts and B's row 0 at least
+                    for (int i = 1; i < A.T; ++i)
+                        if (pl_sbuf[(size_t) i - 1] != A.tok[i]) return;     // the lookup no longer runs through A
+                    float p_on = pl_lookup_pon;
+                    if (!A.lookup) {
+                        const int match = sfx.last_match();
+                        const double q = policy.lookup_rate(match);
+                        if (match < 12 || q < 0.85) return;
+                        p_on = (float) q;
+                        ++pl_lk_any;
+                    }
+                    const int rest = k - (A.T - 1);
+                    B = PW{};
+                    B.seq = A.seq + 1;
+                    B.p = A.p + A.T;
+                    B.T = std::min(S, rest);
+                    for (int i = 0; i < B.T; ++i) B.tok[i] = pl_sbuf[(size_t) (A.T - 1 + i)];
+                    for (int i = 1; i < B.T; ++i) B.prob[i - 1] = 1.0f;
+                    B.p_on = p_on;
+                    B.ready = true;
+                    B.made = true;
+                    B.lookup = true;
+                    b_done = true;   // the chain in flight makes no B for this A
+                    ++pl_lk_next;
+                    tre("CD", A.seq + 1, 4, (int) (1000.0f * B.p_on));
+                };
+                // fork (Daily 5140210d + 83c27bbd), architectds 6175807a: the verified path served first (see the loop's
+                // order).  On by default only in the STRATA_PIPELINE_ELASTIC profile; STRATA_PIPELINE_YIELD=0/1 (read
+                // under STRATA_PIPELINE_DEBUG=1) sets it either way
+                static const bool pl_yield = [] {
+                    const char* v = pipe_dbg_env("STRATA_PIPELINE_YIELD");
+                    return v != nullptr ? std::atoi(v) != 0 : pipe_elastic_env();
+                }();
                 const Clock::time_point pl_t0 = Clock::now();
                 for (int st = 0; st < 2; ++st)
                     for (int par = 0; par < 2; ++par) g_pl_diag.v[st][par] = PV[st][par];
```

### 08-11

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                // per window class (0: fresh, 1: run speculatively ahead of its predecessor's verdict): the time from
                // the previous verdict, the tokens it emitted, how many
                double cls_ms[2] = {0, 0}, cls_tok[2] = {0, 0}, cls_n[2] = {0, 0};
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
                 // per window class (0: fresh, 1: run speculatively ahead of its predecessor's verdict): the time from
                 // the previous verdict, the tokens it emitted, how many
                 double cls_ms[2] = {0, 0}, cls_tok[2] = {0, 0}, cls_n[2] = {0, 0};
+                // fork (Daily ae6a4561): the cost of a rollback on the verified path.  Verdict intervals by [A run ahead]
+                // [the verdict before rolled a window back][a B ran beside A]: a fresh A after a rollback against a fresh
+                // A after a verdict that launched nothing is the time the wrong window held stage 0; B beside A or not is
+                // the contention a speculative window puts on the window being verified.  Log only.
+                double rb_ms[2][2][2] = {}, rb_n[2][2][2] = {};
+                bool pl_prev_rolled = false;
                 int64_t cal_n[10] = {}, cal_on[10] = {};   // per p_on decile: windows scored, on the path
                 double last_verdict = 0.0;
                 int64_t pl_disagree = 0, pl_late = 0;
```

### 08-12

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            if (!adapt_tick(false, false)) return die("an adaptive refill failed");
                        }
                    }
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 15 lines):
```diff
                             if (!adapt_tick(false, false)) return die("an adaptive refill failed");
                         }
                     }
-                    // ---- every window in flight served
-                    if ((doomed && !pump0(D)) || !pump0(A) || !pump0(B)) return die(err);
+                    // ---- every window in flight served.  Every stage's CPU experts run from this one loop, a layer at a
+                    // time, so a stage whose GPU has reached a layer waits while the loop serves another window's.
+                    // fork (Daily 5140210d), architectds 6175807a: the verified path goes first - A's front stage, then
+                    // A's last stage; the windows off it after (D, a wrong guess being undone, and B, run ahead of A's
+                    // verdict), and while A's last stage runs, B one layer per pass
+                    if (!pl_yield && ((doomed && !pump0(D)) || !pump0(A) || !pump0(B))) return die(err);
+                    if (pl_yield && !pump0(A)) return die(err);
                     if (A.s1 && !A.s1_done) {
                         strata::core::Verifier& v = V1(A);
                         if (v.service(&drive_pool_split, SDf(A), err) < 0) return die(err);
```

### 08-13

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            tre("F1", A.seq, A.T);
                        } else if (!err.empty()) return die(err);
                    }
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                             tre("F1", A.seq, A.T);
                         } else if (!err.empty()) return die(err);
                     }
+                    if (pl_yield && ((doomed && !pump0(D)) || !pump0(B, A.s1 && !A.s1_done ? 1 : 0))) return die(err);
                     if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
                     if (!el_gap()) return die(err);   // fork: the elastic tail at a gap of stage 0
                     // ---- a wrong speculative window has stopped (it is launched on no further stage once doomed):
```

### 08-14

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                    if (!VF(0, B).prestage(B.T, B.tok, B.p, prev, err)) return die(err);
                                }
                                b_done = true;
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                                     if (!VF(0, B).prestage(B.T, B.tok, B.p, prev, err)) return die(err);
                                 }
                                 b_done = true;
+                                if (chain_kind == 1 || chain_kind == 2) {   // fork (Daily 12a6f553): the chain's latency to B
+                                    pl_chain_ms[chain_kind - 1] += ms_now() - pl_chain_t0;
+                                    pl_chain_n[chain_kind - 1] += 1;
+                                }
                                 tre("CD", A.seq, chain_kind, (int) (1000.0f * B.p_on));
                                 if (pl_log)
                                     std::fprintf(stderr, "strata pipeline: chain %d for p=%lld T=%d: B T=%d p_on %.3f (%d "
```

### 08-15

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        if (A.sfx) { ++sfx_windows; sfx_drafts += A.T - 1; sfx_ok += a; }
                        if (A.seq > 0 && !last) policy.observe(A.sfx, A.T, a, A.sfx_match, now - last_verdict);
                        cls_ms[c] += now - last_verdict;
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                         if (A.sfx) { ++sfx_windows; sfx_drafts += A.T - 1; sfx_ok += a; }
                         if (A.seq > 0 && !last) policy.observe(A.sfx, A.T, a, A.sfx_match, now - last_verdict);
                         cls_ms[c] += now - last_verdict;
+                        rb_ms[c][pl_prev_rolled ? 1 : 0][B.launched ? 1 : 0] += now - last_verdict;
+                        rb_n[c][pl_prev_rolled ? 1 : 0][B.launched ? 1 : 0] += 1;
                         cls_tok[c] += a + 1;
                         cls_n[c] += 1;
                         last_verdict = now;
```

### 08-16

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    if (!last && !ajob && !pl_adapt()) return die("an adaptive refill failed");
                    if (on && !last) {
                        ++pl_on;
```
Block (hunk 12 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     if (!last && !ajob && !pl_adapt()) return die("an adaptive refill failed");
                     if (on && !last) {
                         ++pl_on;
+                        pl_prev_rolled = false;   // fork (rb_ms)
                         // the chain over A, forced through B's drafts: B's bonus guess and the next window's drafts
                         mtp.set_source_R(V1(A).final_R(0));
                         chain_n = std::min(strata::kernels::kVerifyMaxT - 1, B.T + S_mtp - 1);
```

### 08-17

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            return die(err);
                        chain_kind = 1;
                        tre("CL", A.seq, 1);
```
Block (hunk 13 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                             return die(err);
                         chain_kind = 1;
                         tre("CL", A.seq, 1);
+                        pl_chain_t0 = ms_now();
                         early_used = false;
                         b_done = false;
                         // A accepted whole: its commit on the front stages B has not reached yet (B, now A, runs there
```

### 08-18

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            if ((A.com & (1u << st)) == 0 && !VF(st, A).pl_commit_async(A.T, err)) return die(err);
                        A = B;
                        B = PW{};
```
Block (hunk 14 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
                             if ((A.com & (1u << st)) == 0 && !VF(st, A).pl_commit_async(A.T, err)) return die(err);
                         A = B;
                         B = PW{};
+                        lookup_next();   // fork: a copy goes on - the next B from the lookup (else the chain makes it)
                     } else {
                         // the front stages B never ran on take A's real commit now; those it did run on are undone
                         // (snapshot restored, the commit replayed) once it has stopped
                         for (int st = 0; st < pl_nf; ++st)
                             if ((A.com & (1u << st)) == 0 && !VF(st, A).pl_commit_async(a + 1, err)) return die(err);
+                        pl_prev_rolled = B.launched;   // fork (rb_ms)
                         if (B.launched) {   // a wrong guess (or the end): undo once it has stopped
                             doomed = true;
                             D = B;
```

### 08-19

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                return die(err);
                            chain_kind = 2;
                            tre("CL", A.seq, 2);
```
Block (hunk 15 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                                 return die(err);
                             chain_kind = 2;
                             tre("CL", A.seq, 2);
+                            pl_chain_t0 = ms_now();
                             early_used = false;
                             b_done = false;
                             PW nA;
```

### 08-20

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                         "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                 (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                 (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
```
Block (hunk 16 of `src/program/generate.cpp` in the patch, 29 lines):
```diff
                                          "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                  (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                  (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
+                    std::fprintf(stderr, "strata pipeline: the chain from its launch to B: forced %.2f ms (%.0f), fresh %.2f ms "
+                                         "(%.0f)\n", avg(pl_chain_ms[0], pl_chain_n[0]), pl_chain_n[0],
+                                 avg(pl_chain_ms[1], pl_chain_n[1]), pl_chain_n[1]);
+                    std::fprintf(stderr, "strata pipeline: %lld B from a running lookup (%lld after a chain window); yield %s\n",
+                                 (long long) pl_lk_next, (long long) pl_lk_any, pl_yield ? "on" : "off");
                     if (el_on)
                         std::fprintf(stderr, "strata pipeline: elastic tail beside the windows: %lld shrinks, %lld growths at "
                                              "gaps of stage 0, %.1f ms on the loop's thread\n",
                                      (long long) pl_el_shrinks, (long long) pl_el_grows, pl_el_ms);
+                    {   // fork (Daily ae6a4561): verdict intervals, [A ahead][after a rollback][B beside]
+                        std::string t;
+                        char cb[96];
+                        for (int i = 0; i < 2; ++i)
+                            for (int j = 0; j < 2; ++j)
+                                for (int k = 0; k < 2; ++k)
+                                    if (rb_n[i][j][k] > 0) {
+                                        std::snprintf(cb, sizeof cb, " %s%s%s %.0fx%.2f", i ? "spec" : "fresh",
+                                                      j ? "+afterRB" : "", k ? "+Bbeside" : "", rb_n[i][j][k],
+                                                      rb_ms[i][j][k] / rb_n[i][j][k]);
+                                        t += cb;
+                                    }
+                        std::fprintf(stderr, "strata pipeline rollback cost (windows x ms):%s\n", t.c_str());
+                    }
                     std::fprintf(stderr, "strata pipeline classes: fresh %.0f windows %.2f ms %.2f tok | speculative %.0f "
                                          "windows %.2f ms %.2f tok | forced-chain disagreements %lld, chain late %lld\n",
                                  cls_n[0], avg(cls_ms[0], cls_n[0]), avg(cls_tok[0], cls_n[0]), cls_n[1],
```

## 09 df-defaults

`09-df-defaults.patch` — SONNET-STRATA-Rimetti STRATA_DF_BRANCH (CUDA su Windows) e STRATA_DF_PLE (CUDA) accesi di default come nel Daily

### 09-1

`src/core/verify.cpp` · bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&

Anchor:
```
    {   // STRATA_DF_BRANCH=1 (opt-in): the mixer's side branches.  Off by default: on Linux (open modules, GSP) a
        // stage's window with the branches stopped inside its graph when an NVML query (nvidia-smi) came between
        // requests (2x RTX 3090 and 2x RTX 2080 Ti, #905); never seen on Windows.  0 / unset: one stream, as before
```
Block (hunk 1 of `src/core/verify.cpp` in the patch, 14 lines):
```diff
     {   // STRATA_DF_BRANCH=1 (opt-in): the mixer's side branches.  Off by default: on Linux (open modules, GSP) a
         // stage's window with the branches stopped inside its graph when an NVML query (nvidia-smi) came between
         // requests (2x RTX 3090 and 2x RTX 2080 Ti, #905); never seen on Windows.  0 / unset: one stream, as before
+        // Fork (lab/upsync): on by default with CUDA on Windows, as in the Daily (be4cf1c, +2.4% at 131k on the two
+        // cards); STRATA_DF_BRANCH=0 turns it off.  Linux and HIP keep upstream's opt-in.
         const char* v = std::getenv("STRATA_DF_BRANCH");
+#if defined(_WIN32) && !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
+        df_branch_ = v == nullptr || std::atoi(v) != 0;
+#else
         df_branch_ = v != nullptr && std::atoi(v) != 0;
+#endif
         if (df_branch_ && df_fork_ == nullptr) {
             bool okb = cudaEventCreateWithFlags(&df_fork_, cudaEventDisableTiming) == cudaSuccess;
             for (int i = 0; i < 2 && okb; ++i)
```

### 09-2

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
            // below (the norms divide at run time as its norms do; the history advances token by token), which also
            // writes the commit's history snapshots.  Scratch: layer 0's expert rows, combined by now, and the
            // hyper-connection read's `lo_`, which the read below rewrites.
```
Block (hunk 2 of `src/core/verify.cpp` in the patch, 15 lines):
```diff
             // below (the norms divide at run time as its norms do; the history advances token by token), which also
             // writes the commit's history snapshots.  Scratch: layer 0's expert rows, combined by now, and the
             // hyper-connection read's `lo_`, which the read below rewrites.
+            // Fork (lab/upsync): on by default with CUDA, as in the Daily (be4cf1c, +3.3% at 131k on the two cards);
+            // STRATA_DF_PLE=0 turns it off.  HIP keeps upstream's opt-in.
             static const bool ple_post_env = [] {
                 const char* e = std::getenv("STRATA_DF_PLE");
+#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
+                return e == nullptr || std::atoi(e) != 0;
+#else
                 return e != nullptr && std::atoi(e) != 0;
+#endif
             }();
             const bool ple_post = ple_batch_kv && ple_post_env && !batch_rec_;
             // #783 PR-g (stuchapin909): the window's residual writes in one launch (STRATA_NO_MULTI_GR=1: per token)
```

## 10 ple-diag

`10-ple-diag.patch` — SONNET-STRATA-Rimetti la diagnostica PLE di prefill e decode (letture lente, gather del decode con STRATA_DECODE_TIMING)

### 10-1

`include/strata/core/verify.hpp` · public:

Anchor:
```
    bool pcie_balance_on() const { return bal_ != nullptr && m_bal_ != nullptr; }

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
```
Block (hunk 1 of `include/strata/core/verify.hpp` in the patch, 8 lines):
```diff
     bool pcie_balance_on() const { return bal_ != nullptr && m_bal_ != nullptr; }

     double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
+    /// Fork (Daily 93af2041), STRATA_DECODE_TIMING: the PLE gather subset of ms_host (submit, wait, dequantization).
+    double ms_ple_gather = 0;
     int64_t windows = 0;
     /// STRATA_VERIFY_PROFILE=1 - GPU stage times of the windows since the last call (ms per
     /// window), as one line; empty when off.
```

### 10-2

`include/strata/ngram/ple_reader.hpp` · struct ReaderStats {

Anchor:
```
    double wait_us = 0;           ///< time `collect` spent blocked
    double submit_us = 0;         ///< time spent inside the read submission call (non-zero = it blocks)
    double read_us_sum = 0;       ///< sum of per-read latencies (issue to completion)
```
Block (hunk 1 of `include/strata/ngram/ple_reader.hpp` in the patch, 8 lines):
```diff
     double wait_us = 0;           ///< time `collect` spent blocked
     double submit_us = 0;         ///< time spent inside the read submission call (non-zero = it blocks)
     double read_us_sum = 0;       ///< sum of per-read latencies (issue to completion)
+    double read_us_max = 0;       ///< the slowest read (all of them, not only the percentile ring)
+    uint64_t slow_reads = 0;      ///< reads that took more than 100 ms
     uint64_t late_injected = 0;   ///< reads delayed by fault injection
     uint64_t keepalive_reads = 0; ///< pages read only to keep the SSD awake (not in `reads`, `bytes` or latencies)
     double keepalive_us_max = 0;  ///< the slowest of them
```

### 10-3

`src/core/verify.cpp` · constexpr float EPS = 1e-6f;

Anchor:
```
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
```
Block (hunk 1 of `src/core/verify.cpp` in the patch, 7 lines):
```diff
 using Clock = std::chrono::steady_clock;
 double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
 const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
+const bool g_ple_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;   // fork: Verifier::ms_ple_gather
 inline bool g_lfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_LFUSE"); return v != nullptr && v[0] == '1'; }(); return on; }
 // STRATA_ROUTE_RESIDENT=<margin logits> (+ STRATA_ROUTE_RESIDENT_RANKS=lo-hi, default 6-9): residency-biased routing.
 struct RouteResidentCfg;
```

### 10-4

`src/core/verify.cpp` · bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool,

Anchor:
```
            _mm_sfence();
            *flag = 1;
            ms_host += ms_since(tp);
```
Block (hunk 2 of `src/core/verify.cpp` in the patch, 7 lines):
```diff
             _mm_sfence();
             *flag = 1;
             ms_host += ms_since(tp);
+            if (g_ple_timing) ms_ple_gather += ms_since(tp);
         } else {
             *flag = 1;
         }
```

### 10-5

`src/core/verify.cpp` · bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool,

Anchor:
```
            std::atomic_thread_fence(std::memory_order_seq_cst);
            _mm_sfence();
            ms_host += ms_since(tp);
```
Block (hunk 3 of `src/core/verify.cpp` in the patch, 7 lines):
```diff
             std::atomic_thread_fence(std::memory_order_seq_cst);
             _mm_sfence();
             ms_host += ms_since(tp);
+            if (g_ple_timing) ms_ple_gather += ms_since(tp);
         }
         if (!(test_stall && k + 1 == steps)) *flag = want;
         if (h_bal_ != nullptr) bal_p_[(size_t) ((k / G) * 2 + grp)] = ms_since(b);
```

### 10-6

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la

Anchor:
```
        std::atomic_thread_fence(std::memory_order_seq_cst);
        _mm_sfence();
        ms_host += ms_since(tp);
```
Block (hunk 4 of `src/core/verify.cpp` in the patch, 7 lines):
```diff
         std::atomic_thread_fence(std::memory_order_seq_cst);
         _mm_sfence();
         ms_host += ms_since(tp);
+        if (g_ple_timing) ms_ple_gather += ms_since(tp);
         return true;
     };
     if (ar_on()) {   // the zero-doorbell graph never rings: it waits only for flag 1 (stage 0's PLE rows).
```

### 10-7

`src/kernels/ngram.cpp` · void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_del

Anchor:
```
std::string PleTable::io_report() const {
    if (impl_->mode != PleIo::Direct || !impl_->reader.is_open()) return {};
    const strata::ngram::ReaderStats s = impl_->reader.snapshot();
```
Block (hunk 1 of `src/kernels/ngram.cpp` in the patch, 8 lines):
```diff
 std::string PleTable::io_report() const {
     if (impl_->mode != PleIo::Direct || !impl_->reader.is_open()) return {};
     const strata::ngram::ReaderStats s = impl_->reader.snapshot();
-    char buf[400];
+    char buf[512];
     int n = std::snprintf(buf, sizeof buf,
                   "ple io: %llu rows, %.1f%% row-cache hits, %llu SSD reads (%.1f MB), read p50 %.0f us p99 %.0f us, "
                   "blocked %.3f ms total (submit %.3f ms), cache %llu/%llu rows",
```

### 10-8

`src/kernels/ngram.cpp` · std::string PleTable::io_report() const {

Anchor:
```
                  (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                  s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                  (unsigned long long) impl_->reader.cache_capacity());
```
Block (hunk 2 of `src/kernels/ngram.cpp` in the patch, 10 lines):
```diff
                   (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                   s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                   (unsigned long long) impl_->reader.cache_capacity());
+    // p50/p99 cover only the last 65,536 reads; the mean, the maximum and the slow count cover all of them
+    if (s.reads > 0 && n > 0 && n < (int) sizeof buf)
+        n += std::snprintf(buf + n, sizeof buf - (size_t) n, ", read mean %.0f us max %.1f ms, %llu over 100 ms",
+                           s.read_us_sum / (double) s.reads, s.read_us_max / 1000.0, (unsigned long long) s.slow_reads);
     if (s.keepalive_reads > 0 && n > 0 && n < (int) sizeof buf)
         std::snprintf(buf + n, sizeof buf - (size_t) n, ", SSD kept awake by %llu reads (slowest %.1f ms)",
                       (unsigned long long) s.keepalive_reads, s.keepalive_us_max / 1000.0);
```

### 10-9

`src/ngram/ple_reader.cpp` · struct PleReader::Impl {

Anchor:
```

    void record_latency(double us) {
        stats.read_us_sum += us;
```
Block (hunk 1 of `src/ngram/ple_reader.cpp` in the patch, 8 lines):
```diff

     void record_latency(double us) {
         stats.read_us_sum += us;
+        stats.read_us_max = std::max(stats.read_us_max, us);
+        if (us > 100e3) ++stats.slow_reads;
         if (stats.read_us.size() < LATENCY_RING) stats.read_us.push_back((float) us);
         else stats.read_us[ring_pos++ % LATENCY_RING] = (float) us;
     }
```

### 10-10

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                });
            }
            ple_buf ^= 1;
```
Block (hunk 1 of `src/prefill/prefill.cpp` in the patch, 13 lines):
```diff
                 });
             }
             ple_buf ^= 1;
-            stats_.ms_ple += ms_since(tp);
+            const double waited = ms_since(tp);
+            stats_.ms_ple += waited;
+            // a chunk that waited long for its rows: the SSD stalled (the read-ahead normally hides the reads)
+            if (waited > 1000.0)
+                std::fprintf(stderr, "strata prefill: the chunk at token %lld waited %.0f ms for its PLE rows\n",
+                             (long long) c0, waited);
             return true;
         };
         for (int64_t t = 0; t < T; ++t) { prev[0] = prev[1]; prev[1] = (int32_t) tokens[c0 + t]; }
```

### 10-11

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            const Clock::time_point d0 = Clock::now();
            // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
            static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 17 lines):
```diff
             const Clock::time_point d0 = Clock::now();
             // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
             static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
+            // fork (Daily 93af2041): the PLE reader's cumulative counters before and after this request's decode
+            const std::string decode_io_before = dec_timing ? ple_table.io_report() : std::string{};
             struct DecSnap {
-                double wait, pool, host, plan, actq, jobs, run;
+                double wait, pool, host, ple, plan, actq, jobs, run;
                 int64_t misses, entries, hits, pcie;
             };
             auto dec_snap = [&]() {
-                return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
+                return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, ver.ms_ple_gather,
+                               drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                                drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,
                                drive.d.pcie_experts};
             };
```

### 10-12

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    if (st == 0) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());   // text as before
                    else std::fprintf(stderr, "strata decode GPU stages, stage %d (ms/window):%s\n", st, pr.c_str());
                }
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 17 lines):
```diff
                     if (st == 0) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());   // text as before
                     else std::fprintf(stderr, "strata decode GPU stages, stage %d (ms/window):%s\n", st, pr.c_str());
                 }
+                // fork (Daily 93af2041): the primary verifier's PLE gather (a subset of its host staging; with
+                // --pipeline-windows the second verifier's gathers are not in it)
+                const double ple_ms = d1.ple - ds0.ple;
+                std::fprintf(stderr, "strata decode PLE primary: gather %.3f ms total (%.3f ms/window), other staging %.3f ms/window\n",
+                             ple_ms, ple_ms / w, std::max(0.0, d1.host - ds0.host - ple_ms) / w);
+                // reader snapshots are cumulative: compare the before/after counters, not their percentiles
+                if (!decode_io_before.empty()) {
+                    std::fprintf(stderr, "strata decode PLE before: %s\n", decode_io_before.c_str());
+                    const std::string after = ple_table.io_report();
+                    if (!after.empty()) std::fprintf(stderr, "strata decode PLE after: %s\n", after.c_str());
+                }
             }
             if (drive.d.fs) std::fprintf(stderr, "strata serve: %s\n", drive.d.fs->report().c_str());
             if (!cancelled) {
```

## 11 prefill-sm75

`11-prefill-sm75.patch` — SONNET-STRATA-Tieni accesi di default su sm_75 STRATA_SELECT_SIMT e il dequant a 16 B come nel Daily

### 11-1

`include/strata/core/tc_sm75.hpp` · (file scope)

Anchor:
```
//
// 0.1.45: the Turing paths are opt-in (nobody here has an sm_75 card to confirm them): STRATA_SM75_PERF=1
// (sm75_series.hpp) turns them on, STRATA_TC_SM75=0 / 1 forces this one off / on (still only on a 7.5 card).
```
Block (hunk 1 of `include/strata/core/tc_sm75.hpp` in the patch, 21 lines):
```diff
 //
 // 0.1.45: the Turing paths are opt-in (nobody here has an sm_75 card to confirm them): STRATA_SM75_PERF=1
 // (sm75_series.hpp) turns them on, STRATA_TC_SM75=0 / 1 forces this one off / on (still only on a 7.5 card).
+// Fork (RTX 2080 Ti, the Daily): this one (the 16-byte dequant of the prompt projections, bitwise) is on by default on a
+// 7.5 card, as before 0.1.45; STRATA_TC_SM75=0 turns it off.
 #pragma once

 #include "strata/core/sm75_series.hpp"

+#include <cstdlib>
+
 namespace strata::core {

 inline bool tc_sm75_enabled(int cc_major, int cc_minor) {
-    return strata::sm75_switch("STRATA_TC_SM75", cc_major, cc_minor);   // 7.5 and only 7.5 - never another SM model
+    if (!strata::is_sm75(cc_major, cc_minor)) return false;   // 7.5 and only 7.5 - never another SM model
+    const char* v = std::getenv("STRATA_TC_SM75");
+    return v == nullptr || std::atoi(v) != 0;
 }

 }  // namespace strata::core
```

### 11-2

`include/strata/kernels/qsa_select.hpp` · bool qsa_block_scores_rows(const float* pooled, const float* dead, const float*

Anchor:
```
/// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
/// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
/// On AMD it is the gfx12 (RDNA4) WMMA kernel (a three-way bf16 split, six products); false on any other AMD target.
```
Block (hunk 1 of `include/strata/kernels/qsa_select.hpp` in the patch, 8 lines):
```diff
 /// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
 /// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
 /// On AMD it is the gfx12 (RDNA4) WMMA kernel (a three-way bf16 split, six products); false on any other AMD target.
-/// On CUDA below sm_80 (no TF32 tensor cores) it is an FP32 tiled kernel, also not bitwise, and only with STRATA_SELECT_SIMT=1 (otherwise false).
+/// On CUDA below sm_80 (no TF32 tensor cores) it is an FP32 tiled kernel, also not bitwise; fork: on by default (STRATA_SELECT_SIMT=0: false).
 bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                          int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks);

```

### 11-3

`src/kernels/cuda/qsa_select.cu` · bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_

Anchor:
```
            cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
        }
        if (cc_major[dev] < 8) {
```
Block (hunk 1 of `src/kernels/cuda/qsa_select.cu` in the patch, 17 lines):
```diff
             cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
         }
         if (cc_major[dev] < 8) {
-            // no TF32 tensor cores: the FP32 tiled kernel, opt-in (STRATA_SELECT_SIMT=1): its sum order differs from the
-            // warp kernel's, so the default stays the caller's warp kernel and the output bytes stay the same
-            // STRATA_SELECT_SIMT=2: block_scores_simt2_kernel (the same bits as =1, larger tiles)
+            // no TF32 tensor cores: the FP32 tiled kernel (its sum order differs from the warp kernel's: not bitwise).
+            // Fork (Daily dab24920, PR #1091): on by default, as in the Daily (-16% of a 2080 Ti's prompt time);
+            // STRATA_SELECT_SIMT=0 keeps the caller's warp kernel (upstream's default), =2 is block_scores_simt2_kernel
+            // (the same bits as =1, larger tiles)
             static const int simt = [] {
                 const char* v = std::getenv("STRATA_SELECT_SIMT");
-                return v == nullptr ? 0 : v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
+                return v == nullptr ? 1 : v[0] == '1' ? 1 : v[0] == '2' ? 2 : 0;
             }();
             if (!simt) return false;
             const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
```

### 11-4

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
```
Block (hunk 1 of `src/prefill/prefill.cpp` in the patch, 8 lines):
```diff
                         // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                         const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                 strata::kernels::kStepNBid)] + 1;
-                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise), below sm_80 the FP32 tiled kernel with STRATA_SELECT_SIMT=1;
+                        // the scores on tensor cores (3xTF32: FP32-level, not bitwise), below sm_80 the FP32 tiled kernel (fork: by default, STRATA_SELECT_SIMT=0 off);
                         // STRATA_SELECT_OLD=1: the warp kernel
                         static const bool old_sel = std::getenv("STRATA_SELECT_OLD") != nullptr;
                         if (old_sel || !strata::kernels::qsa_block_scores_tc(st.idx_pooled, st.idx_dead, m.q_idx + t0 * 512,
```

## 12 ram-core

`12-ram-core.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il nucleo fisso per scheda senza copia RAM (STRATA_RAM_CORE=1, STRATA_RAM_CORE_SLOTS_DEV<n>)

### 12-1

`docs/RAM_CORE.md` · (new file)

New file, 114 lines: see the patch.

### 12-2

`include/strata/core/expert_source.hpp` · private:

Anchor:
```
// Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
class ArenaExpertSource : public ExpertSource {
public:
```
Block (hunk 1 of `include/strata/core/expert_source.hpp` in the patch, 8 lines):
```diff
 // Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
 class ArenaExpertSource : public ExpertSource {
 public:
-    ArenaExpertSource() = default;
+    ArenaExpertSource();   // out of line: the STRATA_RAM_CORE state (rc_) is complete only in expert_source.cpp
     ~ArenaExpertSource() override;
     ArenaExpertSource(const ArenaExpertSource&) = delete;
     ArenaExpertSource& operator=(const ArenaExpertSource&) = delete;
```

### 12-3

`include/strata/core/expert_source.hpp` · public:

Anchor:
```
    double load_read_seconds() const { return load_read_s_; }
    double load_copy_seconds() const { return load_copy_s_; }
```
Block (hunk 2 of `include/strata/core/expert_source.hpp` in the patch, 38 lines):
```diff
     double load_read_seconds() const { return load_read_s_; }
     double load_copy_seconds() const { return load_copy_s_; }

+    // ---- STRATA_RAM_CORE=1 (docs/RAM_CORE.md): the RAM copy of the experts of the caches' FIXED CORE goes away.
+    // The driver picks, per cache, a core of slots that never changes hands (no adaptive swap, no loan to the prompt
+    // path, no elastic shrink, no K/V growth), and hands each of their experts to `ram_core_drop` once: the whole
+    // pages inside its blob are decommitted (VirtualUnlock + VirtualFree MEM_DECOMMIT; munlock + MADV_DONTNEED).  The
+    // address range stays reserved, so `blob` is still base + offset.  Nothing copies those bytes back: a core expert
+    // never leaves VRAM.  If a path asks for one anyway, `blob` reads it again from the model file first and counts
+    // it (`ram_core_reread`, which must stay 0).  Never inside the CUDA-registered prefix (decommitting registered
+    // pages is not allowed: those experts keep their copy and are counted apart).
+    /// STRATA_RAM_CORE=1 asked for (off by default).
+    static bool ram_core_requested();
+    /// Arms the bookkeeping.  False with `err` when the arena cannot do it (not loaded, mapped read-only, large
+    /// pages, a shared file).
+    bool ram_core_arm(std::string& err);
+    bool ram_core_armed() const { return rc_ != nullptr; }
+    /// Gives back the RAM of (layer, expert).  Returns the bytes decommitted; 0 when kept (inside the registered
+    /// prefix: *registered set; already dropped; refused by the OS).
+    uint64_t ram_core_drop(int64_t layer, int64_t expert, bool* registered = nullptr);
+    /// After the drops: lowers the working-set minimum the arena lock raised by what was given back (best effort).
+    void ram_core_settle();
+    bool ram_core_dropped(int64_t layer, int64_t expert) const;
+    uint64_t ram_core_bytes() const;     ///< bytes decommitted now
+    int64_t ram_core_experts() const;    ///< experts without a RAM copy now
+    int64_t ram_core_reread() const;     ///< experts read again from the model file (must stay 0)
+    uint64_t arena_capacity() const { return arena_capacity_; }
+    uint64_t registered_bytes() const { return pinned_bytes_; }
+
 private:
+    struct RamCore;
+    std::unique_ptr<RamCore> rc_;
+    bool rc_recover(int64_t layer, int64_t expert);   ///< `blob` of a dropped expert: the model file -> the arena
+    uint64_t arena_capacity_ = 0;
+    std::string experts_bin_;        ///< the pack's experts.bin when the arena was loaded from it (else empty: the GGUF)
     void* arena_ = nullptr;          ///< the PinnedArena, owned
     void* map_ = nullptr;            ///< STRATA_ARENA_MMAP: the arena file, mapped read-only (not the PinnedArena)
     uint64_t map_bytes_ = 0;
```

### 12-4

`src/core/expert_source.cpp` · LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata:

Anchor:
```
    return st;
}
```
Block (hunk 1 of `src/core/expert_source.cpp` in the patch, 164 lines):
```diff
     return st;
 }

+// ---- STRATA_RAM_CORE=1 (see the header and docs/RAM_CORE.md) ------------------------------------------------------
+struct ArenaExpertSource::RamCore {
+    std::unique_ptr<std::atomic<uint8_t>[]> dropped;   ///< per blob: 1 = its interior pages are decommitted
+    size_t n = 0;
+    std::mutex mu;                                       ///< drops and recoveries (never on a hot path)
+    std::atomic<uint64_t> bytes{0};
+    std::atomic<int64_t> experts{0}, reread{0};
+    uint64_t settled = 0;                                ///< bytes the working-set minimum was already lowered by
+};
+
+namespace {
+constexpr uint64_t kRcPage = 4096;
+/// The whole pages of the blob [off, off + bytes) that may be given back: past its first 256 bytes (a copy of the
+/// neighbour before it rounded up to 256 B, as a native slot is, may read into them) and before its end (the page
+/// it shares with the next blob stays).
+void rc_span(uint64_t off, uint64_t bytes, uint64_t& s, uint64_t& e) {
+    s = (off + 256 + kRcPage - 1) / kRcPage * kRcPage;
+    e = (off + bytes) / kRcPage * kRcPage;
+    if (e < s) e = s;
+}
+bool rc_read(const std::string& path, uint64_t at, uint64_t n, uint8_t* dst) {
... 140 more lines: see 12-ram-core.patch
```

### 12-5

`src/core/expert_source.cpp` · bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int6

Anchor:
```
    arena_ = a;
    base_ = a->data();
    pinned_bytes_ = a->registered_bytes;
```
Block (hunk 2 of `src/core/expert_source.cpp` in the patch, 8 lines):
```diff
     arena_ = a;
     base_ = a->data();
     pinned_bytes_ = a->registered_bytes;
+    arena_capacity_ = a->capacity;
+    experts_bin_ = from_gguf ? std::string() : path;   // STRATA_RAM_CORE: where a core expert would be read again from
     // plan v0.3 P6: device aliases of the mapped registration, for the PCIe share of the misses
     dev_slice_.clear();
     slice_bytes_ = a->slice_bytes;
```

### 12-6

`src/core/expert_source.cpp` · void ArenaExpertSource::close() {

Anchor:
```
    map_ = nullptr;
    map_bytes_ = 0;
    inputs_.clear();
```
Block (hunk 3 of `src/core/expert_source.cpp` in the patch, 7 lines):
```diff
     map_ = nullptr;
     map_bytes_ = 0;
     inputs_.clear();
+    rc_.reset();
     if (arena_ != nullptr) {
         delete (PinnedArena*) arena_;
         arena_ = nullptr;
```

### 12-7

`src/core/expert_source.cpp` · void ArenaExpertSource::close() {

Anchor:
```
    base_ = nullptr;
    blobs_ = 0;
    n_expert_ = 0;
```
Block (hunk 4 of `src/core/expert_source.cpp` in the patch, 8 lines):
```diff
     base_ = nullptr;
     blobs_ = 0;
     n_expert_ = 0;
+    arena_capacity_ = 0;
+    experts_bin_.clear();
 }

 void ArenaExpertSource::prefetch(int64_t layer, int64_t expert) {
```

### 12-8

`src/core/expert_source.cpp` · const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {

Anchor:
```
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    reads_.fetch_add(1, std::memory_order_relaxed);
```
Block (hunk 5 of `src/core/expert_source.cpp` in the patch, 9 lines):
```diff
     const int64_t idx = layer * n_expert_ + expert;
     if (idx < 0 || idx >= blobs_) return nullptr;
     reads_.fetch_add(1, std::memory_order_relaxed);
+    // STRATA_RAM_CORE: a core expert has no RAM copy (it never leaves VRAM); asked for anyway, it is read from the
+    // model file first (counted).  One atomic load otherwise.
+    if (rc_ && rc_->dropped[(size_t) idx].load(std::memory_order_acquire) != 0 && !rc_recover(layer, expert)) return nullptr;
     // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
     // this class over `FileExpertSource`.
     return base_ + arena_off(layer, expert);
```

### 12-9

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
    // fork (Daily 12a6f553): beside the pipeline the prompt path may borrow the core too - nothing decodes while a prompt
    // is read, and the lent slots are refilled before the decode as the tail's are (with only the 1.5 GiB tail lent the
    // 131k prompt read fell from ~85 s to 134 s)
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 9 lines):
```diff
     // fork (Daily 12a6f553): beside the pipeline the prompt path may borrow the core too - nothing decodes while a prompt
     // is read, and the lent slots are refilled before the decode as the tail's are (with only the 1.5 GiB tail lent the
     // 131k prompt read fell from ~85 s to 134 s)
-    const int64_t el_lend_floor = pipe_elastic_env() ? 0 : el_core;
+    // (not const: STRATA_RAM_CORE raises it to CUDA0's fixed core in the serve path, so no later replan lends the core)
+    int64_t el_lend_floor = pipe_elastic_env() ? 0 : el_core;
     // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits,
     // or with --prefill auto the largest chunk whose buffers take at most kAutoLendPct % of the slots (a
     // lent slot's expert is streamed during the prompt and refilled after it; measured on a 12 GB card, 32K Q2_0
```

### 12-10

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            int32_t first_now = -1;        // where its buffers are laid out now
            int64_t lent_chunk = 0;
            std::vector<std::pair<int32_t, int32_t>> lent;
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
             int32_t first_now = -1;        // where its buffers are laid out now
             int64_t lent_chunk = 0;
             std::vector<std::pair<int32_t, int32_t>> lent;
+            int64_t rc_floor = 0;          // STRATA_RAM_CORE: the fixed core's slots [0, rc_floor) are never lent
         };
         // bytes -> slots for one cache: exact when it knows its per-slot offsets (a native pack's blobs differ
         // per layer), otherwise max_blob each.
```

### 12-11

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // keeps the pinned-share rule.  STRATA_SPLIT_RING=N: N slots on a split; 0: the pinned-share rule.
        // (decided above, before the lend regions were sized: a no-op here unless the resident RAM mode skipped it)
        decide_split_ring();
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 45 lines):
```diff
         // keeps the pinned-share rule.  STRATA_SPLIT_RING=N: N slots on a split; 0: the pinned-share rule.
         // (decided above, before the lend regions were sized: a no-op here unless the resident RAM mode skipped it)
         decide_split_ring();
+        // ---- STRATA_RAM_CORE=1 (docs/RAM_CORE.md): per cache (0: CUDA0's, i: stages[i-1]'s) a FIXED CORE of its lowest
+        // slots - the profile's hottest experts, which the startup fill put there first - that never changes hands: no
+        // adaptive swap picks it (core_adapt or not), the prompt path never borrows it, the elastic tail never shrinks
+        // into it, the elastic K/V never takes it.  Its experts then need no RAM copy and the arena gives those pages
+        // back (rc_arm, before the first request).  rc_floor[ci]: the slots a loan must leave, asked for BEFORE the loan
+        // is planned (STRATA_RAM_CORE_SLOTS_DEV<n>), so the loan takes only what lies above; rc_core[ci]: the core
+        // itself, settled once the loans are (CUDA0's default: the part of the elastic core the prompt does not borrow).
+        bool rc_req = false, rc_auto0 = false;
+        std::vector<int64_t> rc_core(1 + stages.size(), 0), rc_floor(1 + stages.size(), 0);
+        auto rc_env = [](int dev, int64_t def) -> int64_t {
+            const std::string name = "STRATA_RAM_CORE_SLOTS_DEV" + std::to_string(dev);
+            const char* v = std::getenv(name.c_str());
+            return v != nullptr && v[0] != '\0' ? (int64_t) std::atoll(v) : def;
+        };
+        constexpr int64_t kRcDefault = 3500;   // ~6 GB of a 2080 Ti's 1.78 MB slots
+        if (strata::core::ArenaExpertSource::ram_core_requested()) {
+            std::string why;
+            if (srcp != &arena_src || !arena_src.mapped()) why = "the experts are not in the resident arena (--mmap-experts / file tier)";
+            else if (host_res.empty() || d_res == nullptr) why = "no residency table (a fillable --expert-cache with --expert-profile is needed)";
+            else if (peer.valid()) why = "a peer GPU's expert tier (--peer-device) is on";
+            else if (drive.d.remote_count > 0 || remote_opt) why = "remote experts are on";
+            else if (!arena_src.ram_core_arm(why)) {}
+            if (!why.empty()) {
+                std::fprintf(stderr, "strata ram core: NOT armed: %s\n", why.c_str());
+            } else {
+                rc_req = true;
+                const int64_t r0 = rc_env(0, -1);
+                rc_auto0 = r0 < 0;
+                if (!rc_auto0) {
+                    int64_t c = std::min<int64_t>(r0, std::max<int64_t>(0, xcache.slots() - 128));
+                    if (el_core > 0) c = std::min(c, el_core);   // above the elastic core the tail may shrink
+                    rc_floor[0] = c;
+                    el_lend_floor = std::max(el_lend_floor, c);
+                }
+                for (size_t i = 0; i < stages.size(); ++i)
+                    rc_floor[i + 1] = std::min<int64_t>(std::max<int64_t>(0, rc_env(stages[i]->dev, kRcDefault)),
+                                                        std::max<int64_t>(0, stages[i]->cache.slots() - 128));
+            }
+        }
         std::vector<PfPart> pf_parts;
         // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card
         // whose cache already filled its reserve, that is the over-subscription the auto sizing avoids - so the
```

### 12-12

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
            for (auto& st : stages)
                pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
             pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
             for (auto& st : stages)
                 pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
+            for (size_t i = 0; i < pf_parts.size() && i < rc_floor.size(); ++i) pf_parts[i].rc_floor = rc_floor[i];
             // The two tests plan_lend makes for CUDA0 alone, one participant at a time: a loan must leave the
             // 128-slot floor.  The percentage cap is an AUTO-chunk rule and only the auto scan applies it - an
             // explicit --prefill is the operator's number, and a loan of it only has to fit.  With one participant
```

### 12-13

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                const int64_t k = part_slots(p, c);
                if (p.cache == &xcache && k > xcache.slots() - el_lend_floor) return false;
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
             auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                 const int64_t k = part_slots(p, c);
                 if (p.cache == &xcache && k > xcache.slots() - el_lend_floor) return false;
+                if (k > p.cache->slots() - p.rc_floor) return false;   // STRATA_RAM_CORE: the fixed core is never lent
                 if (k <= 0 || k + 128 > p.cache->slots()) return false;
                 return !(cap && k * 100 > kAutoLendPct * p.cache->slots());
             };
```

### 12-14

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                                                  kAutoLendPct * p.cache->slots() / 100));
                    // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
                    if (p.cache == &xcache) budget = std::min(budget, std::max<int64_t>(0, xcache.slots() - el_lend_floor));
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                                                                   kAutoLendPct * p.cache->slots() / 100));
                     // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
                     if (p.cache == &xcache) budget = std::min(budget, std::max<int64_t>(0, xcache.slots() - el_lend_floor));
+                    budget = std::min(budget, std::max<int64_t>(0, p.cache->slots() - p.rc_floor));   // STRATA_RAM_CORE
                     const uint64_t avail = part_bytes(p, (int32_t) (p.cache->slots() - budget));
                     const uint64_t nonring = strata::prefill::Prefill::bytes_needed_no_ring(g, *p.ses, c, srcp != nullptr);
                     room = std::min(room, (int64_t) ((avail - std::min(avail, nonring)) / (uint64_t) kBlob));
```

### 12-15

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
        // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
        // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 31 lines):
```diff
                 }
             }
         }
+        // STRATA_RAM_CORE: the loans are settled - each fixed core is what its floor asked for, never past its first
+        // lendable slot; CUDA0's default is the elastic core below the prompt path's loan (without the elastic cache,
+        // kRcDefault below it).  From here on el_lend_floor (CUDA0's replans) and rc_floor (the stages) keep it unlent.
+        if (rc_req) {
+            const int64_t first0 = !pf_parts.empty() && pf_parts[0].first >= 0 ? (int64_t) pf_parts[0].first : xcache.slots();
+            rc_core[0] = std::min<int64_t>(rc_auto0 ? (el_core > 0 ? el_core : kRcDefault) : rc_floor[0], first0);
+            rc_core[0] = std::max<int64_t>(0, std::min<int64_t>(rc_core[0], xcache.slots()));
+            el_lend_floor = std::max(el_lend_floor, rc_core[0]);
+            for (size_t i = 0; i < stages.size(); ++i) {
+                const bool lends = i + 1 < pf_parts.size() && pf_parts[i + 1].first >= 0;
+                const int64_t fi = lends ? (int64_t) pf_parts[i + 1].first : stages[i]->cache.slots();
+                rc_core[i + 1] = std::max<int64_t>(0, std::min(rc_floor[i + 1], fi));
+            }
+            for (size_t i = 0; i < pf_parts.size() && i < rc_core.size(); ++i) pf_parts[i].rc_floor = rc_core[i];
+        }
         kvg_start(borrow != nullptr ? (int64_t) lend_first : xcache.slots());
+        if (rc_req) kvg.floor = std::max<int64_t>(kvg.floor, rc_core[0]);   // the elastic K/V never takes the core
+        // STRATA_RAM_CORE: whether the expert of layer l in slot `slot` (of the layer's own cache) is in a fixed core
+        auto rc_fixed = [&](int64_t l, int32_t slot) -> bool {
+            if (!rc_req || slot < 0) return false;
+            const size_t ci = multi_gpu ? (size_t) stage_of(l) : 0;
+            return ci < rc_core.size() && (int64_t) slot < rc_core[ci];
+        };
+        std::atomic<int64_t> rc_swaps{0};      // the adaptive tier's swaps (adapt() may run on the pipeline's thread)
+        int64_t rc_swaps0 = 0;                 // ... at the previous request-end line
         // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
         // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
         // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
```

### 12-16

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
                    // --elastic: the core never leaves - unless core_adapt; beside the pipeline el_core counts CUDA0's
                    // slots and a later stage's layers hold slots of their own cache
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                     if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
                     // --elastic: the core never leaves - unless core_adapt; beside the pipeline el_core counts CUDA0's
                     // slots and a later stage's layers hold slots of their own cache
-                    else if (r[e] >= el_core || core_adapt || (pipe_elastic_env() && multi_gpu && stage_of(l) != 0))
+                    // STRATA_RAM_CORE: a fixed core's expert is never a victim, core_adapt or not
+                    else if (!rc_fixed(l, r[e]) &&
+                             (r[e] >= el_core || core_adapt || (pipe_elastic_env() && multi_gpu && stage_of(l) != 0)))
                         vict.emplace_back(u[e], e);
                 }
                 if (cand.empty() || vict.empty()) continue;
```

### 12-17

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                agate.has.assign((size_t) g.n_layers, 0);
                for (const Swap& s : swaps) agate.has[(size_t) s.layer] = 1;
            }
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                 agate.has.assign((size_t) g.n_layers, 0);
                 for (const Swap& s : swaps) agate.has[(size_t) s.layer] = 1;
             }
+            rc_swaps.fetch_add((int64_t) swaps.size(), std::memory_order_relaxed);
             // the copies back read the cache of the card that owns each layer (see resident_stage_swaps)
             auto home_of = [&](int64_t layer) -> SwapHome {
                 const int stn = multi_gpu ? stage_of(layer) : 0;
```

### 12-18

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        static const bool el_trace = std::getenv("STRATA_ELASTIC_TRACE") != nullptr;
        const int64_t el_step_max = 512ll << 20;   // at most this much per growth step (its copies are async)
        const int64_t el_shrink_step = std::max<int64_t>(64ll << 20, (int64_t) o.elastic_shrink_step_mib << 20);
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
         static const bool el_trace = std::getenv("STRATA_ELASTIC_TRACE") != nullptr;
         const int64_t el_step_max = 512ll << 20;   // at most this much per growth step (its copies are async)
         const int64_t el_shrink_step = std::max<int64_t>(64ll << 20, (int64_t) o.elastic_shrink_step_mib << 20);
-        const int64_t el_min_slots = std::max<int64_t>(256, el_core);   // the core is never shrunk
+        // the core is never shrunk (nor STRATA_RAM_CORE's fixed core, which lies inside it when the elastic core exists)
+        const int64_t el_fixed = std::max<int64_t>(el_core, rc_core[0]);
+        const int64_t el_min_slots = std::max<int64_t>(256, el_fixed);
         const int64_t el_chunk_max = sp_chunk_max > 0 ? sp_chunk_max : o.prefill_chunk;
         Clock::time_point el_last{};
         Clock::time_point el_stable_since = Clock::now();
```

### 12-19

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            int64_t c = el_chunk_max;
            const int64_t k = plan_lend(c);
            if (k <= 0 || c > el_chunk_max) return;   // nothing fits (not above el_min_slots): keep the plan
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             int64_t c = el_chunk_max;
             const int64_t k = plan_lend(c);
             if (k <= 0 || c > el_chunk_max) return;   // nothing fits (not above el_min_slots): keep the plan
+            // STRATA_RAM_CORE: plan_lend kept its count above el_lend_floor; the participant's own count must too
+            if (xcache.slots() - part_slots(pf_parts[0], c) < el_lend_floor) return;
             o.prefill_chunk = c;
             // 0.1.30: the loan lives in the participant (CUDA0 is the first; the elastic cache is single-GPU)
             pf_parts[0].first = (int32_t) (xcache.slots() - part_slots(pf_parts[0], c));
```

### 12-20

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            for (int64_t l = 0; l < g.n_layers; ++l)
                for (int32_t ex = 0; ex < (int32_t) g.n_expert && el_layer(l); ++ex) {
                    const size_t i = (size_t) (l * g.n_expert + ex);
```
Block (hunk 12 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             for (int64_t l = 0; l < g.n_layers; ++l)
                 for (int32_t ex = 0; ex < (int32_t) g.n_expert && el_layer(l); ++ex) {
                     const size_t i = (size_t) (l * g.n_expert + ex);
-                    if (host_res[i] < el_core) continue;   // not resident, or in the core (never displaced)
+                    if (host_res[i] < el_fixed) continue;   // not resident, or in the core (never displaced)
                     const float u = have_usage ? drive.d.usage[i] : 0.0f;
                     (host_res[i] >= n ? tail : keep)[(size_t) l].emplace_back(u, ex);
                 }
```

### 12-21

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                const int32_t* r = a_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
```
Block (hunk 13 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                 const int32_t* r = a_res.data() + l * g.n_expert;
                 for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                     if (r[e] < 0) { if (u[e] >= 2.0f && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
-                    else vict.emplace_back(u[e], e);
+                    else if (!rc_fixed(l, r[e])) vict.emplace_back(u[e], e);   // STRATA_RAM_CORE: the core stays
                 }
                 if (cand.empty() || vict.empty()) continue;
                 std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
```

### 12-22

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
            std::fflush(stderr);
        };
```
Block (hunk 14 of `src/program/generate.cpp` in the patch, 78 lines):
```diff
             std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
             std::fflush(stderr);
         };
+        // ---- STRATA_RAM_CORE: everything is laid out and nothing is lent - the fixed cores' RAM copies go now.  rc_slot_of:
+        // per residency index, the core slot its expert was given back in (-1: not in a core); every request's end checks
+        // the experts are still there (`fuori posto`, must be 0) and how many were read again from the file (must be 0).
+        std::vector<int32_t> rc_slot_of;
+        std::vector<int64_t> rc_n(rc_core.size(), 0), rc_kept_reg(rc_core.size(), 0);
+        std::vector<uint64_t> rc_bytes(rc_core.size(), 0);
+        auto rc_dev = [&](size_t ci) -> int { return ci == 0 ? 0 : stages[ci - 1]->dev; };
+        auto rc_log = [&](const char* when) {
+            if (!rc_req) return;
+            const double G = 1073741824.0;
+            int64_t misplaced = 0;
+            for (size_t i = 0; i < rc_slot_of.size(); ++i)
+                if (rc_slot_of[i] >= 0 && host_res[i] != rc_slot_of[i]) ++misplaced;
+            std::string per;
+            for (size_t ci = 0; ci < rc_core.size(); ++ci) {
+                char b[160];
+                if (ci == 0)
+                    std::snprintf(b, sizeof b, "dev%d %lld esperti (%.2f GiB) senza copia RAM", rc_dev(ci), (long long) rc_n[ci],
+                                  (double) rc_bytes[ci] / G);
+                else
+                    std::snprintf(b, sizeof b, ", dev%d %lld (%.2f GiB)", rc_dev(ci), (long long) rc_n[ci], (double) rc_bytes[ci] / G);
+                per += b;
+            }
+            const uint64_t cap = arena_src.arena_capacity();
+            const MemSample ms = mem_sample();
+            std::fprintf(stderr, "strata ram core: %s; arena da %.2f a %.2f GiB; riletti %lld (%s; fuori posto %lld; scambi "
+                                 "dell'adattamento %lld; privata %.2f GiB, working set %.2f GiB)\n", per.c_str(), (double) cap / G,
+                         (double) (cap - std::min(cap, arena_src.ram_core_bytes())) / G, (long long) arena_src.ram_core_reread(),
+                         when, (long long) misplaced, (long long) (rc_swaps.load() - rc_swaps0), (double) ms.commit_mib / 1024.0,
+                         (double) ms.rss_mib / 1024.0);
+            rc_swaps0 = rc_swaps.load();
+            std::fflush(stderr);
+        };
+        if (rc_req) {
+            const auto t0 = Clock::now();
+            const MemSample ms0 = mem_sample();
+            const auto& lay = strata::kernels::cpu::expert_layout();
+            rc_slot_of.assign(host_res.size(), -1);
+            std::vector<int64_t> held(rc_core.size(), 0);
+            for (size_t i = 0; i < host_res.size(); ++i) {
+                const int64_t l = (int64_t) i / g.n_expert;
+                if (!rc_fixed(l, host_res[i])) continue;
+                const size_t ci = multi_gpu ? (size_t) stage_of(l) : 0;
+                ++held[ci];
+                rc_slot_of[i] = host_res[i];
+                bool reg = false;
+                const uint64_t b = arena_src.ram_core_drop(l, (int64_t) i % g.n_expert, &reg);
+                if (b > 0) {
+                    ++rc_n[ci];
+                    rc_bytes[ci] += b;
+                } else if (reg) {
+                    ++rc_kept_reg[ci];
+                }
+            }
+            arena_src.ram_core_settle();
+            const MemSample ms1 = mem_sample();
+            for (size_t ci = 0; ci < rc_core.size(); ++ci) {
+                const strata::core::ExpertCache& c = ci == 0 ? xcache : stages[ci - 1]->cache;
+                std::fprintf(stderr, "strata ram core:   dev%d: nucleo fisso slot [0, %lld) di %lld (%lld esperti, %.2f GiB in "
+                                     "VRAM); %lld senza copia RAM, %lld tenuti nel prefisso registrato (%.2f GiB), mai "
+                                     "prestato, mai scambiato, mai ristretto\n", rc_dev(ci), (long long) rc_core[ci],
+                             (long long) c.slots(), (long long) held[ci],
+                             c.slot_offsets() != nullptr ? (double) c.slot_offsets()[rc_core[ci]] / 1073741824.0
+                                                         : (double) rc_core[ci] * (double) lay.max_blob / 1073741824.0,
+                             (long long) rc_n[ci], (long long) rc_kept_reg[ci],
+                             (double) arena_src.registered_bytes() / 1073741824.0);
+            }
+            std::fprintf(stderr, "strata ram core:   pagine restituite in %.0f ms; privata %.2f -> %.2f GiB, working set %.2f -> %.2f GiB\n",
+                         std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), (double) ms0.commit_mib / 1024.0,
+                         (double) ms1.commit_mib / 1024.0, (double) ms0.rss_mib / 1024.0, (double) ms1.rss_mib / 1024.0);
+            rc_log("avvio");
+        }
         {
             // what is left once everything is allocated: under WDDM a GPU filled to the brim does not fail, it pages -
             // and a page-in while the verify graph spins on a host flag stalls the request for good
```

### 12-23

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
            const int64_t before = xcache.slots();
            // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
```
Block (hunk 15 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
             const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
             const int64_t before = xcache.slots();
             // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
-            const int64_t floor_slots = std::min<int64_t>(xcache.full_slots(),
-                128 + (pf_parts.empty() || pf_parts[0].first < 0 ? 0 : part_slots(pf_parts[0], 256)));
+            const int64_t floor_slots = std::max<int64_t>(rc_core[0], std::min<int64_t>(xcache.full_slots(),   // + STRATA_RAM_CORE
+                128 + (pf_parts.empty() || pf_parts[0].first < 0 ? 0 : part_slots(pf_parts[0], 256))));
             std::string note;
             if ((int64_t) free_b < want_free) {
                 const int64_t floor_b = xcache.bytes_of(floor_slots);
```

### 12-24

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        (long long) req_offload, chain_txt, (long long) req_chunk,
                        (long long) (sp.chain_experts_streamed() - streamed0));
            std::fflush(stdout);
```
Block (hunk 16 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                         (long long) req_offload, chain_txt, (long long) req_chunk,
                         (long long) (sp.chain_experts_streamed() - streamed0));
             std::fflush(stdout);
+            rc_log("fine richiesta");   // STRATA_RAM_CORE
             if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
                 bool cont = !cancelled && produced_n == 1 && admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                             (int64_t) live.size() == p;
```

## 13 sa-bet

`13-sa-bet.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la scommessa sulla testa dello stadio 0 (STRATA_SA_BET, STRATA_SA_CHAIN, opt-in)

### 13-1

`docs/STRUCT_A.md` · (new file)

New file, 155 lines: see the patch.

### 13-2

`include/strata/core/expert_source.hpp` · struct ExpertDispatch {

Anchor:
```
    /// whose routed ids all look resident there publishes no activation (doorbell_publish_res) - counted
    const int32_t* dres_check = nullptr;
    int64_t stale_x = 0;
```
Block (hunk 1 of `include/strata/core/expert_source.hpp` in the patch, 9 lines):
```diff
     /// whose routed ids all look resident there publishes no activation (doorbell_publish_res) - counted
     const int32_t* dres_check = nullptr;
     int64_t stale_x = 0;
+    /// STRUCT-A (STRATA_SA_BET): the call serves a head bet - its routes are not counted in `usage` (the caller
+    /// counts them once the bet is confirmed)
+    bool no_usage = false;
     int64_t multi_misses = 0;      ///< distinct (layer, expert) pairs the CPU computed in verify windows
     int64_t multi_entries = 0;     ///< routed (token, expert) entries the CPU served in verify windows
     /// Multi-GPU: the second GPU's tier.  Its experts are computed there instead of on the CPU (kind 2).
```

### 13-3

`include/strata/core/mtp.hpp` · public:

Anchor:
```
    /// own pick, and step j (1 <= j <= n_force) takes force[j - 1] as its input token instead of output j - 1.
    /// `n_early`: an event after each of the first `n_early` outputs (chain_outputs_ready).
    bool chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force, int n_out,
```
Block (hunk 1 of `include/strata/core/mtp.hpp` in the patch, 14 lines):
```diff
     /// own pick, and step j (1 <= j <= n_force) takes force[j - 1] as its input token instead of output j - 1.
     /// `n_early`: an event after each of the first `n_early` outputs (chain_outputs_ready).
     bool chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force, int n_out,
-                      int n_early, std::string& err);
+                      int n_early, std::string& err, int n_now = 0);
+    /// STRUCT-A (STRATA_SA_CHAIN, opt-in): a chain launched with `n_now` > 0 queues only its first `n_now` outputs;
+    /// `chain_extend` queues more (up to `n_out`), `chain_cut` gives up the rest (the chain ends with the outputs
+    /// queued so far).  A chain not extended to `n_out` nor cut stays live: the caller does one or the other.
+    bool chain_extend(int want, std::string& err);
+    void chain_cut();
+    int chain_queued() const { return chain_launched_; }
     /// 1: every output is in chain_tok()/chain_prob(); 0: still running; -1: an error.
     int chain_poll(std::string& err);
     /// How many of the chain's first outputs (up to `n_early`; all of them once the chain is done) have landed in
```

### 13-4

`include/strata/core/mtp.hpp` · private:

Anchor:
```
    cudaEvent_t ev_chain_ = nullptr;
    cudaEvent_t ev_step_[8] = {};                       ///< after each of the first n_early outputs
    int steps_seen_ = 0, chain_n_ = 0, chain_early_ = 0;
```
Block (hunk 2 of `include/strata/core/mtp.hpp` in the patch, 7 lines):
```diff
     cudaEvent_t ev_chain_ = nullptr;
     cudaEvent_t ev_step_[8] = {};                       ///< after each of the first n_early outputs
     int steps_seen_ = 0, chain_n_ = 0, chain_early_ = 0;
+    int chain_launched_ = 0;   ///< STRUCT-A: the outputs queued so far (chain_n_ once the chain is whole or cut)
     int32_t chain_tok_[8] = {};
     float chain_prob_[8] = {};
     int64_t n_vocab_ = 0;
```

### 13-5

`include/strata/core/verify.hpp` · public:

Anchor:
```
    /// The commit without a host sync; `ss.ple_prev` advances now (host side).  A second call for the same window
    /// (after its state was restored) replays it with another count.
    bool pl_commit_async(int n_keep, std::string& err);
```
Block (hunk 1 of `include/strata/core/verify.hpp` in the patch, 27 lines):
```diff
     /// The commit without a host sync; `ss.ple_prev` advances now (host side).  A second call for the same window
     /// (after its state was restored) replays it with another count.
     bool pl_commit_async(int n_keep, std::string& err);
+    // ---- STRUCT-A (lab, STRATA_SA_BET, opt-in; docs/STRUCT_A.md): a pipelined window's HEAD - its inputs and its
+    // first `c` layers, all GDN layers before the stage's first QSA layer - launched as a bet before its rows are
+    // decided, then either its TAIL (the remaining layers and the hand-off: head + tail are the window `pl_launch`
+    // runs, the same kernels in the same order) or the head thrown away.  A pipelined window leaves the GDN state to
+    // the commit and writes no K/V in those layers; the PLE history it advances in place is saved before the head and
+    // restored by `pl_drop_head`.  Stage 0 of a split only (`lb_` == 0 is not required, but the head must hold no QSA).
+    /// Capture the head and tail graphs for T = 1 .. `max_t` with `c` head layers (clamped to the GDN layers before the
+    /// first QSA one; nothing in flight).  False with `err` when it cannot (the caller turns the bet off).
+    bool sa_prepare(int c, int max_t, std::string& err);
+    bool sa_ready(int T) const { return sa_c_ > 0 && T >= 1 && T <= sa_max_t_ && exec_head_[T] && exec_tail_[T]; }
+    int sa_layers() const { return sa_c_; }
+    /// Stage and launch window (T, tokens, pos0)'s head; `service` then serves its `c` layers and stops there.
+    bool pl_launch_head(int T, const int32_t* tokens, int64_t pos0, std::string& err);
+    bool sa_head_live() const { return fl_active_ && sa_head_; }
+    /// Every head layer served and the head graph done (the head can then be thrown away).
+    bool pl_head_done(std::string& err);
+    /// The bet confirmed: the window goes on from its head (its tail launched; `service` / `done` as after pl_launch).
+    bool pl_launch_tail(std::string& err);
+    /// After `pl_head_done`: the PLE history restored (queued on the stage's stream), nothing in flight.
+    bool pl_drop_head(std::string& err);
+
     /// Fold another verifier's counters and GPU profile into this one's (the two verifiers of one stage report once).
     void absorb_stats(Verifier& o);
     /// The watchdog's line for a pipelined verifier: in flight, layers served, the GPU's ring and flags, its events.
```

### 13-6

`include/strata/core/verify.hpp` · private:

Anchor:
```
    bool fl_active_ = false, fl_prof_ = false, fl_ple_ = false, commit_live_ = false, pl_prestaged_ = false;
    int fl_T_ = 0;
    int64_t fl_k_ = 0, fl_total_ = 0;
```
Block (hunk 2 of `include/strata/core/verify.hpp` in the patch, 16 lines):
```diff
     bool fl_active_ = false, fl_prof_ = false, fl_ple_ = false, commit_live_ = false, pl_prestaged_ = false;
     int fl_T_ = 0;
     int64_t fl_k_ = 0, fl_total_ = 0;
+    // STRUCT-A (STRATA_SA_BET): the head / tail graphs, the layers `service` may serve (a head's: sa_c_ * groups; else
+    // all), the head's completion, the PLE history saved before it; record_window's layer range (-1: the stage's)
+    cudaGraphExec_t exec_head_[9] = {}, exec_tail_[9] = {};
+    cudaEvent_t ev_head_ = nullptr;
+    int sa_c_ = 0, sa_max_t_ = 0;
+    int64_t fl_cap_ = 0;
+    bool sa_head_ = false;
+    float* sa_hist_ = nullptr;
+    int64_t sa_hist_n_ = 0;
+    int64_t rec_from_ = -1, rec_to_ = -1;
     double fl_since_ms_ = 0, fl_flush_ms_ = 0, fl_launch_ms_ = 0;
     int32_t pl_prev_[2] = {-1, -1};
     std::vector<uint32_t> pl_ple_rows_;   ///< the window's PLE rows (T x PLE_N_HEADS), gathered when layer 0 is served
```

### 13-7

`src/core/expert_source.cpp` · void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32

Anchor:
```
    pt("begin");
    d.src->begin_layer(d.layers, ids, n_tok * k);
    pt("begun");
```
Block (hunk 1 of `src/core/expert_source.cpp` in the patch, 8 lines):
```diff
     pt("begin");
     d.src->begin_layer(d.layers, ids, n_tok * k);
     pt("begun");
-    if (!d.usage.empty())
+    if (!d.usage.empty() && !d.no_usage)   // STRUCT-A: a head bet's routes are counted when it is confirmed
         for (int64_t i = 0; i < n_tok * k; ++i)
             if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) d.layers * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
     // ---- plan v0.3 P6: the GPU's share, decided and published FIRST so the GPU starts while the CPU works.
```

### 13-8

`src/core/mtp.cpp` · bool MtpDrafter::prepare_chain(std::string& err) {

Anchor:
```
}

bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force,
```
Block (hunk 1 of `src/core/mtp.cpp` in the patch, 8 lines):
```diff
 }

 bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force,
-                              int n_out, int n_early, std::string& err) {
+                              int n_out, int n_early, std::string& err, int n_now) {
     const OnDevice on_device(device_);
     if (chain_live_) { err = "mtp: a chain is already in flight"; return false; }
     if (!force_on_ || ev_chain_ == nullptr) { err = "mtp: chains need prepare_chain"; return false; }
```

### 13-9

`src/core/mtp.cpp` · bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, co

Anchor:
```
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (!stage_source_R(T, err)) return false;
    chain_early_ = std::max(0, std::min(n_early, n_out));
```
Block (hunk 2 of `src/core/mtp.cpp` in the patch, 12 lines):
```diff
     std::atomic_thread_fence(std::memory_order_seq_cst);
     if (!stage_source_R(T, err)) return false;
     chain_early_ = std::max(0, std::min(n_early, n_out));
+    // STRUCT-A: n_now > 0 queues the first n_now outputs only (chain_extend / chain_cut do the rest)
+    const int n_q = n_now > 0 ? std::min(n_now, n_out) : n_out;
     bool ok = cudaGraphLaunch(round_exec_[T], cs_) == cudaSuccess;
     if (ok && chain_early_ >= 1) ok = cudaEventRecord(ev_step_[0], cs_) == cudaSuccess;
-    for (int j = 1; ok && j < n_out; ++j) {
+    for (int j = 1; ok && j < n_q; ++j) {
         ok = cudaGraphLaunch(step_exec_[j], cs_) == cudaSuccess;
         if (ok && j < chain_early_) ok = cudaEventRecord(ev_step_[j], cs_) == cudaSuccess;
     }
```

### 13-10

`src/core/mtp.cpp` · bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, co

Anchor:
```
    steps_seen_ = 0;
    chain_live_ = true;
    chain_n_ = n_out;
```
Block (hunk 3 of `src/core/mtp.cpp` in the patch, 41 lines):
```diff
     steps_seen_ = 0;
     chain_live_ = true;
     chain_n_ = n_out;
+    chain_launched_ = n_q;
     ms_draft += ms_since(t0);
     ++rounds;
     return true;
 }

+bool MtpDrafter::chain_extend(int want, std::string& err) {
+    if (!chain_live_ || want <= chain_launched_ || chain_launched_ >= chain_n_) return true;
+    const OnDevice on_device(device_);
+    const int to = std::min(want, chain_n_);
+    bool ok = true;
+    for (int j = chain_launched_; ok && j < to; ++j) {
+        ok = cudaGraphLaunch(step_exec_[j], cs_) == cudaSuccess;
+        if (ok && j < chain_early_) ok = cudaEventRecord(ev_step_[j], cs_) == cudaSuccess;
+    }
+    if (!ok || cudaEventRecord(ev_chain_, cs_) != cudaSuccess) {   // the chain's end moves with its last step
+        err = std::string("mtp chain: ") + cudaGetErrorString(cudaGetLastError());
+        return false;
+    }
+    (void) cudaStreamQuery(cs_);
+    chain_launched_ = to;
+    return true;
+}
+
+void MtpDrafter::chain_cut() {
+    if (!chain_live_) return;
+    chain_n_ = chain_launched_;
+    chain_early_ = std::min(chain_early_, chain_launched_);
+}
+
 int MtpDrafter::chain_outputs_ready(std::string& err) {
     if (!chain_live_) return chain_n_;
     const OnDevice on_device(device_);
-    while (steps_seen_ < chain_early_) {
+    while (steps_seen_ < std::min(chain_early_, chain_launched_)) {
         const cudaError_t q = cudaEventQuery(ev_step_[steps_seen_]);
         if (q == cudaErrorNotReady) break;
         if (q != cudaSuccess) { err = std::string("mtp chain: ") + cudaGetErrorString(q); return -1; }
```

### 13-11

`src/core/mtp.cpp` · int MtpDrafter::chain_outputs_ready(std::string& err) {

Anchor:
```

int MtpDrafter::chain_poll(std::string& err) {
    if (!chain_live_) return 1;
```
Block (hunk 4 of `src/core/mtp.cpp` in the patch, 7 lines):
```diff

 int MtpDrafter::chain_poll(std::string& err) {
     if (!chain_live_) return 1;
+    if (chain_launched_ < chain_n_) return 0;   // STRUCT-A: not whole yet (chain_extend or chain_cut first)
     const OnDevice on_device(device_);
     const cudaError_t q = cudaEventQuery(ev_chain_);
     if (q == cudaErrorNotReady) return 0;
```

### 13-12

`src/core/verify.cpp` · Verifier::~Verifier() {

Anchor:
```
        if (e) cudaGraphExecDestroy(e);
    for (auto& e : exec_nr_)
        if (e) cudaGraphExecDestroy(e);
```
Block (hunk 1 of `src/core/verify.cpp` in the patch, 11 lines):
```diff
         if (e) cudaGraphExecDestroy(e);
     for (auto& e : exec_nr_)
         if (e) cudaGraphExecDestroy(e);
+    for (auto* set : {exec_head_, exec_tail_})   // STRUCT-A
+        for (int T = 0; T < 9; ++T)
+            if (set[T]) cudaGraphExecDestroy(set[T]);
+    if (ev_head_) cudaEventDestroy(ev_head_);
+    if (sa_hist_) cudaFree(sa_hist_);
     if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
     for (auto& kv : exec_bm_)
         if (kv.second) cudaGraphExecDestroy(kv.second);
```

### 13-13

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```

    // ---- the window's inputs, from mapped staging.  With branches the steps and positions (first read by the first
    // layer's mixer) cross PCIe on side 1 beside the embedding and the first hyper-connection read.
```
Block (hunk 2 of `src/core/verify.cpp` in the patch, 26 lines):
```diff

     // ---- the window's inputs, from mapped staging.  With branches the steps and positions (first read by the first
     // layer's mixer) cross PCIe on side 1 beside the embedding and the first hyper-connection read.
-    copy_i32_from_mapped(tok_, m_tok_, T, cs);
+    // STRUCT-A (STRATA_SA_BET): a window recorded in two graphs, the head [lb_, rec_to_) and the tail [rec_from_, le_);
+    // the tail records no inputs (its head left them on the device) and the head no hand-off
+    const int64_t rec_l0 = rec_from_ >= 0 ? rec_from_ : lb_, rec_l1 = rec_to_ >= 0 ? rec_to_ : le_;
+    const bool rec_in = rec_l0 == lb_;
+    if (rec_in) copy_i32_from_mapped(tok_, m_tok_, T, cs);
     bool inputs_pending = false;
     cudaStream_t in_s = cs;
-    if (br) {
+    if (br && rec_in) {
         fork(1);
         in_s = df_side_[1];
         inputs_pending = true;
     }
-    copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, in_s);
-    copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), in_s);
+    if (rec_in) {
+        copy_i32_from_mapped(step_, m_step_, (int64_t) T * kStepCount, in_s);
+        copy_i32_from_mapped(pos_, m_pos_, (int64_t) MT * (NH + NKV + IQ), in_s);
+    }
     // per-ROW positions of the K rows [t][NKV] and the indexer query rows [t][IQ] (for batched RoPE)
     const int32_t* pos_k = pos_ + MT * NH;
     const int32_t* pos_i = pos_ + MT * (NH + NKV);
```

### 13-14

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
    // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
    // residual, pending write and inject (see set_stage)
    const int64_t HB = Verifier::handoff_floats(g);
```
Block (hunk 3 of `src/core/verify.cpp` in the patch, 10 lines):
```diff
     // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
     // residual, pending write and inject (see set_stage)
     const int64_t HB = Verifier::handoff_floats(g);
-    if (lb_ > 0) {
+    if (!rec_in) {
+        // STRUCT-A: a tail - the residual rows are its head's
+    } else if (lb_ > 0) {
         const float* hin = hand_in_ + (size_t) hrow0 * HB;   // the group's rows: [R][bo][inj] contiguous
         // the overlapped split: this graph was launched before the previous stage finished - wait for its hand-off
         if (hflag_in_d_ != nullptr && !batch_rec_) wait_handoff(hflag_in_d_, 1u, herr_d_, g_split_wait_ns, cs);
```

### 13-15

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
    };

    for (int grp = 0; grp < G; ++grp)
```
Block (hunk 4 of `src/core/verify.cpp` in the patch, 16 lines):
```diff
     };

     for (int grp = 0; grp < G; ++grp)
-        if (!pre(lb_, grp)) return false;
-    for (int64_t l = lb_; l < le_; ++l)
+        if (!pre(rec_l0, grp)) return false;
+    for (int64_t l = rec_l0; l < rec_l1; ++l)
         for (int grp = 0; grp < G; ++grp) {
             if (!post(l, grp)) return false;
-            if (l + 1 < le_ && !pre(l + 1, grp)) return false;
+            if (l + 1 < rec_l1 && !pre(l + 1, grp)) return false;
         }
+    if (rec_l1 < le_) return true;   // STRUCT-A: a head ends after its last layer
     if (le_ < g.n_layers) {   // a layer split's earlier stage: hand the residual on, no head
         float* hout = hand_out_ + (size_t) hrow0 * HB;
         if (hflag_out_d_ != nullptr && !batch_rec_) {   // the overlapped split: payload + the next stage's flag
```

### 13-16

`src/core/verify.cpp` · bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string

Anchor:
```

int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_layers) {
    if (!fl_active_) return 1;
```
Block (hunk 5 of `src/core/verify.cpp` in the patch, 9 lines):
```diff

 int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_layers) {
     if (!fl_active_) return 1;
-    if (fl_k_ >= fl_total_) return 1;
+    const int64_t fl_cap = fl_cap_ > 0 ? fl_cap_ : fl_total_;   // STRUCT-A: a head bet serves its own layers only
+    if (fl_k_ >= fl_cap) return 1;
     // Nothing rang yet and neither the WDDM flush nor the timeout below is due: return before any CUDA call.  The
     // pipelined loop asks both stages' windows this millions of times a second while they wait on their doorbells,
     // and the device switch (OnDevice: cudaGetDevice and two cudaSetDevice) made that a few driver calls each time:
```

### 13-17

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la

Anchor:
```
    const int G = groups_[T] > 0 ? groups_[T] : 1;
    const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
    int served = 0;   // this call's layers (a layer's halves count apart when its window is split in two groups)
```
Block (hunk 6 of `src/core/verify.cpp` in the patch, 8 lines):
```diff
     const int G = groups_[T] > 0 ? groups_[T] : 1;
     const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
     int served = 0;   // this call's layers (a layer's halves count apart when its window is split in two groups)
-    while (fl_k_ < fl_total_) {
+    while (fl_k_ < fl_cap) {
         if (max_layers > 0 && served >= max_layers) return 0;
         const int64_t l = lb_ + fl_k_ / G;
         const uint32_t want = (uint32_t) (fl_k_ + 1);
```

### 13-18

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la

Anchor:
```
            const double now = now_ms();
            if (now - fl_flush_ms_ > 2.0) {   // flush WDDM and notice a dead graph, as run() does
                fl_flush_ms_ = now;
```
Block (hunk 7 of `src/core/verify.cpp` in the patch, 8 lines):
```diff
             const double now = now_ms();
             if (now - fl_flush_ms_ > 2.0) {   // flush WDDM and notice a dead graph, as run() does
                 fl_flush_ms_ = now;
-                const cudaError_t q = cudaEventQuery(ev_done_);
+                const cudaError_t q = cudaEventQuery(sa_head_ ? ev_head_ : ev_done_);   // STRUCT-A: a head bet has no done event yet
                 if (q != cudaErrorNotReady && *(volatile uint32_t*) h_seq_ < want) {
                     trace_ev("NEVER-RANG", fl_k_, l, (int64_t) q);
                     err = "verify: layer " + std::to_string(l) + " never rang (" +
```

### 13-19

`src/core/verify.cpp` · bool Verifier::pl_commit_async(int n_keep, std::string& err) {

Anchor:
```
void Verifier::absorb_stats(Verifier& o) {
    ms_wait += o.ms_wait; ms_pool += o.ms_pool; ms_host += o.ms_host; ms_commit += o.ms_commit;
    windows += o.windows;
```
Block (hunk 8 of `src/core/verify.cpp` in the patch, 153 lines):
```diff
     return true;
 }

+// ---- STRUCT-A (STRATA_SA_BET, docs/STRUCT_A.md): the head bet of a pipelined window
+bool Verifier::sa_prepare(int c, int max_t, std::string& err) {
+    using namespace strata::kernels;
+    const OnDevice on_device(device_);
+    if (g_ == nullptr || ev_done_ == nullptr || pl_ple_rows_.empty()) { err = "verify: the head bet before capture_all"; return false; }
+    if (fl_active_) { err = "verify: the head bet's capture with a window in flight"; return false; }
+    if (ar_on() || remote_opt_ != nullptr || batch_rec_) { err = "verify: the head bet needs the stage's doorbell graph"; return false; }
+    const ModelGeometry& g = *g_;
+    int gdn = 0;   // the stage's GDN layers before its first QSA layer: no K/V, no indexer
+    while (lb_ + gdn < le_ && !is_qsa_layer(g, lb_ + gdn)) ++gdn;
+    c = std::min<int>(c, gdn);
+    c = (int) std::min<int64_t>(c, le_ - lb_ - 1);
+    if (c < 1) { err = "verify: no GDN layer before the stage's first QSA layer"; return false; }
+    max_t = std::max(1, std::min(max_t, max_t_));
+    SessionState& ss = *ss_;
+    if (ss.ple.ready() && ple_stage() && sa_hist_ == nullptr) {   // the PLE history the head's layer 1 advances in place
+        sa_hist_n_ = (int64_t) NG_HIST * NG_HC_DIM;
+        if (cudaMalloc((void**) &sa_hist_, (size_t) sa_hist_n_ * sizeof(float)) != cudaSuccess) {
+            sa_hist_ = nullptr;
+            cudaGetLastError();
+            err = "verify: the head bet's PLE history copy does not fit";
... 129 more lines: see 13-sa-bet.patch
```

### 13-20

`src/program/generate.cpp` · void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t

Anchor:
```
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 25 lines):
```diff
     drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
 }

+/// STRUCT-A (STRATA_SA_BET, docs/STRUCT_A.md): the pool of a head bet's layers - the stage's split pool without counting
+/// the routes in the adaptive tier's usage (`ExpertDispatch::no_usage`); the bet counts them once it is confirmed, so a
+/// thrown-away bet leaves the counts as they were
+struct SaBetPool {
+    SplitDrive* sd = nullptr;
+    int64_t n_expert = 0;
+    std::vector<int64_t> routes;   ///< layer * n_expert + id, as the dispatch counts them
+};
+void sa_bet_pool(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out, int64_t layer) {
+    SaBetPool* b = (SaBetPool*) user;
+    strata::core::ExpertDispatch& d = b->sd->base->d;
+    d.no_usage = true;
+    drive_pool_split(b->sd, x_f, ids, n_tok, k, out, layer);
+    d.no_usage = false;
+    if (!d.usage.empty() && !d.failed)
+        for (int64_t i = 0; i < n_tok * k; ++i)
+            if (ids[i] >= 0 && ids[i] < b->n_expert) b->routes.push_back(layer * b->n_expert + ids[i]);
+}
+
 /// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
 /// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
 /// lives on its device too).
```

### 13-21

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                bool early_used = false;
                bool b_done = false;     // B has been made from the chain in flight
                int chain_n = 0;
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 63 lines):
```diff
                 bool early_used = false;
                 bool b_done = false;     // B has been made from the chain in flight
                 int chain_n = 0;
+                // ---- STRUCT-A (docs/STRUCT_A.md), opt-in.  STRATA_SA_BET=1: a fresh window's HEAD (its first
+                // STRATA_SA_BET_LAYERS GDN layers, default 2) goes to stage 0 as a bet while the drafter's chain still
+                // decides its rows - first [x] (T = 1), then [x, d1] once d1 clears spec_min_p, up to T =
+                // STRATA_SA_BET_MAXT (2); where A launches today the bet's tail runs if A is the bet, else the head is
+                // thrown away (Verifier::pl_drop_head).  STRATA_SA_CHAIN=1: the forced chain (after an on-path verdict)
+                // queued a step at a time (STRATA_SA_CHAIN_AHEAD more, default 0) and cut once B is decided.
+                static const int sa_bet_c = [] {
+                    const char* v = std::getenv("STRATA_SA_BET");
+                    if (v == nullptr || std::atoi(v) == 0) return 0;
+                    const char* c = std::getenv("STRATA_SA_BET_LAYERS");
+                    return c != nullptr ? std::max(1, std::atoi(c)) : 2;
+                }();
+                static const int sa_bet_maxt = [] {
+                    const char* v = std::getenv("STRATA_SA_BET_MAXT");
+                    return v != nullptr ? std::max(1, std::min(3, std::atoi(v))) : 2;
+                }();
+                // STRATA_SA_BET_MINT (2): the smallest bet.  After a rollback (most fresh windows) stage 0 replays the
+                // commit until about when the chain decides T = 1, so a T = 1 head gains nothing there and, lost, holds
+                // stage 0 when the T = 2 bet should start (docs/STRUCT_A.md, 5)
+                static const int sa_bet_mint = [] {
+                    const char* v = std::getenv("STRATA_SA_BET_MINT");
+                    return v != nullptr ? std::max(1, std::min(3, std::atoi(v))) : 2;
+                }();
+                static const bool sa_chain = [] { const char* v = std::getenv("STRATA_SA_CHAIN"); return v != nullptr && std::atoi(v) != 0; }();
+                static const int sa_chain_ahead = [] {
+                    const char* v = std::getenv("STRATA_SA_CHAIN_AHEAD");
+                    return v != nullptr ? std::max(0, std::atoi(v)) : 0;
+                }();
+                static bool sa_bet_failed = false;   // the head graphs could not be made: the bet stays off
+                bool sa_bet_on = sa_bet_c > 0 && !sa_bet_failed && pl_nf == 1 && req_spec_min_p > 0.0;
+                if (sa_bet_on)
+                    for (int par = 0; par < 2 && sa_bet_on; ++par) {
+                        strata::core::Verifier& v = stage_ver_p(0, par);
+                        if (v.sa_ready(1)) continue;
+                        const strata::core::OnDevice on(v.device());
+                        std::string e2;
+                        if (!v.sa_prepare(sa_bet_c, std::min(sa_bet_maxt, S_mtp - 1), e2)) {
+                            std::fprintf(stderr, "strata struct-a: STRATA_SA_BET off: %s\n", e2.c_str());
+                            sa_bet_failed = true;
+                            sa_bet_on = false;
+                        } else if (par == 1)
+                            std::fprintf(stderr, "strata struct-a: STRATA_SA_BET: heads of %d layers on CUDA%d, bets T=%d..%d\n",
+                                         v.sa_layers(), v.device(), sa_bet_mint, std::min(sa_bet_maxt, S_mtp - 1));
+                    }
+                struct SaBet {
+                    bool live = false, dead = false;   // dead: known lost, served on empty plans until its head is done
+                    int T = 0, seq = -1;
+                    int64_t p = 0;
+                    int32_t tok[8] = {};
+                };
+                SaBet bet;
+                SaBetPool sa_pool;
+                sa_pool.n_expert = g.n_expert;
+                int64_t sa_want = -1, sa_forced = -1;   // the fresh window a bet is for; A's forced rows read (once per A)
+                int64_t sa_snapped = -1;                // the window whose GDN snapshot a bet took
+                int sa_last_T = 0;                      // the last bet's size for sa_want
+                int64_t sa_placed[4] = {}, sa_tails[4] = {}, sa_drops[4] = {}, sa_chains = 0, sa_cut = 0;
                 int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
                 double pl_chain_t0 = 0.0, pl_chain_ms[2] = {0, 0}, pl_chain_n[2] = {0, 0};   // fork: chain -> B latency
                 // fork (Daily 1541e340): the elastic tail beside the windows, at the gaps of stage 0 (see el_gap)
```

### 13-22

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    return v == nullptr || std::atoi(v) != 0;
                }();
                auto ms_now = [&]() { return std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count(); };
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 40 lines):
```diff
                     return v == nullptr || std::atoi(v) != 0;
                 }();
                 auto ms_now = [&]() { return std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count(); };
+                // STRUCT-A: A's head launched as a bet of size T (its rows: A's row 0 and the chain's first T - 1 drafts),
+                // with the gap work and the GDN snapshot A's launch on stage 0 does (nothing commits on stage 0 before
+                // A's launch, so the snapshot is the same)
+                auto sa_place = [&](int T) -> bool {
+                    bet = SaBet{};
+                    bet.T = T;
+                    bet.seq = A.seq;
+                    bet.p = A.p;
+                    bet.tok[0] = A.tok[0];
+                    for (int i = 1; i < T; ++i) bet.tok[i] = mtp.chain_tok()[i - 1];
+                    if (ajob) a_gap(0);
+                    if (!el_gap()) return false;
+                    if (sa_snapped != A.seq && !snap_take(0, A.seq)) { err = "the GDN snapshot failed"; return false; }
+                    sa_snapped = A.seq;
+                    if (!VF(0, A).pl_launch_head(T, bet.tok, bet.p, err)) return false;
+                    bet.live = true;
+                    sa_pool.routes.clear();
+                    sa_last_T = T;
+                    ++sa_placed[T];
+                    tre("SH", A.seq, T);
+                    return true;
+                };
+                // a lost bet thrown away once its head is done: 1 thrown away (or no bet), 0 its head still runs, -1 error
+                auto sa_try_drop = [&]() -> int {
+                    if (!bet.live) return 1;
+                    strata::core::Verifier& v = stage_ver_p(0, bet.seq & 1);
+                    if (!v.pl_head_done(err)) return err.empty() ? 0 : -1;
+                    if (!v.pl_drop_head(err)) return -1;
+                    sa_pool.routes.clear();   // never counted (sa_bet_pool)
+                    bet.live = false;
+                    ++sa_drops[bet.T];
+                    tre("SX", bet.seq, bet.T);
+                    return 1;
+                };
                 while (true) {
                     g_pl_diag.iters.fetch_add(1, std::memory_order_relaxed);
                     g_pl_diag.a_seq.store(A.seq, std::memory_order_relaxed);
```

### 13-23

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        } else if (!err.empty()) return die(err);
                    }
                    if (pl_yield && ((doomed && !pump0(D)) || !pump0(B, A.s1 && !A.s1_done ? 1 : 0))) return die(err);
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 11 lines):
```diff
                         } else if (!err.empty()) return die(err);
                     }
                     if (pl_yield && ((doomed && !pump0(D)) || !pump0(B, A.s1 && !A.s1_done ? 1 : 0))) return die(err);
+                    if (bet.live) {   // STRUCT-A: the head bet's layers (a lost one on empty plans)
+                        sa_pool.sd = PSD[bet.seq & 1];
+                        if (stage_ver_p(0, bet.seq & 1).service(bet.dead ? nullptr : &sa_bet_pool, &sa_pool, err) < 0)
+                            return die(err);
+                    }
                     if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
                     if (!el_gap()) return die(err);   // fork: the elastic tail at a gap of stage 0
                     // ---- a wrong speculative window has stopped (it is launched on no further stage once doomed):
```

### 13-24

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        tre("U", undo_w.seq, undo_keep);
                    }
                    if (ending) {
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
                         tre("U", undo_w.seq, undo_keep);
                     }
                     if (ending) {
+                        if (bet.live) {   // STRUCT-A: no head left in flight (a drain cannot finish it)
+                            bet.dead = true;
+                            const int d = sa_try_drop();
+                            if (d < 0) return die(err);
+                            if (d == 0) continue;
+                        }
                         if (!doomed) break;
                         continue;
                     }
```

### 13-25

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    if (chain_kind != 0) {
                        const int k_ready = mtp.chain_outputs_ready(err);
                        if (k_ready < 0) return die(err);
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                     if (chain_kind != 0) {
                         const int k_ready = mtp.chain_outputs_ready(err);
                         if (k_ready < 0) return die(err);
+                        // STRUCT-A: a bet of size T is lost once the output after its rows clears spec_min_p (A is bigger)
+                        if (bet.live && !bet.dead && chain_kind == 2 && k_ready >= bet.T &&
+                            mtp.chain_prob()[bet.T - 1] >= (float) req_spec_min_p)
+                            bet.dead = true;
                         if (chain_kind == 2 && !early_used) {
                             // A's size is decided by its first outputs: a draft below spec_min_p ends it (the serial
                             // loop's rule), so A goes to stage 0 as soon as that is known, not after all S_mtp - 1
```

### 13-26

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                                         "outputs)\n", chain_kind, (long long) A.p, A.T, B.T, B.p_on, k);
                            }
                        }
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 71 lines):
```diff
                                                          "outputs)\n", chain_kind, (long long) A.p, A.T, B.T, B.p_on, k);
                             }
                         }
+                        // STRUCT-A (STRATA_SA_CHAIN): the forced chain one step past what is seen, cut once B is decided
+                        if (r != 1 && chain_kind == 1 && sa_chain && mtp.chain_live()) {
+                            if (b_done) {
+                                sa_cut += chain_n - mtp.chain_queued();
+                                mtp.chain_cut();
+                            } else if (!mtp.chain_extend(k_ready + 1 + sa_chain_ahead, err))
+                                return die(err);
+                        }
                         if (r == 1) chain_kind = 0;
                     }
+                    // ---- STRUCT-A: a lost bet thrown away once its head is done; then a bet on the next size (the first
+                    // one, T = 1, as soon as the fresh A exists and stage 0 is free)
+                    if (bet.live && bet.dead) {
+                        if (sa_try_drop() < 0) return die(err);
+                    }
+                    if (sa_bet_on && !bet.live && A.seq == sa_want && chain_kind == 2 && !early_used && !A.ready && !A.fl &&
+                        A.fr == 0 && !doomed && !VF(0, A).in_flight()) {
+                        const int k = mtp.chain_outputs_ready(err);
+                        if (k < 0) return die(err);
+                        int T = std::max(sa_last_T + 1, sa_bet_mint);   // the smallest size A can still have and is bet on
+                        while (k >= T && mtp.chain_prob()[T - 1] >= (float) req_spec_min_p) ++T;
+                        if (T > std::min(sa_bet_maxt, S_mtp - 1) || A.p + T > o.max_context) {
+                            sa_want = -1;   // no bet left for this window
+                        } else if (k >= T - 1) {   // its rows are out (else: wait for them)
+                            bool ok = true;
+                            for (int j = 0; ok && j < T - 1; ++j) ok = mtp.chain_prob()[j] >= (float) req_spec_min_p;
+                            if (!ok) sa_want = -1;   // A is smaller (and decided by now)
+                            else if (!sa_place(T)) return die(err);
+                        }
+                    }
                     // ---- the front stages: A on its next one (never while a wrong window still holds a front stage's
                     // state)
                     if (A.ready && !A.fl && A.fr < pl_nf && !doomed) {
                         const int st = A.fr;
-                        if (st == 0 && force_win != nullptr) force_window(A.p, A.tok[0], A.T, A.tok, S);   // lab bench replay
+                        if (st == 0 && force_win != nullptr && sa_forced != A.seq)   // lab bench replay
+                            force_window(A.p, A.tok[0], A.T, A.tok, S);
+                        if (st == 0) sa_forced = A.seq;   // STRUCT-A: read once, even when a head below makes A wait
+                        if (st == 0 && bet.live) {   // STRUCT-A: A is the bet - its tail; else the head is thrown away
+                            const bool match = !bet.dead && bet.seq == A.seq && bet.T == A.T && bet.p == A.p &&
+                                               std::equal(bet.tok, bet.tok + bet.T, A.tok);
+                            if (match) {
+                                if (!VF(0, A).pl_launch_tail(err)) return die(err);
+                                if (!sa_pool.routes.empty() && !drive.d.usage.empty())   // the bet's routes, counted now
+                                    for (const int64_t i : sa_pool.routes) drive.d.usage[(size_t) i] += 1.0f;
+                                sa_pool.routes.clear();
+                                bet.live = false;
+                                ++sa_tails[bet.T];
+                                A.fl = true;
+                                A.launched = true;
+                                tre("ST", A.seq, A.T);
+                                tre("L0", A.seq, A.T, st);
+                                continue;
+                            }
+                            bet.dead = true;
+                            const int d = sa_try_drop();
+                            if (d < 0) return die(err);
+                            if (d == 0) continue;   // its head still runs: A launches behind it at a later pass
+                        }
                         if (st == 0 && A.p + A.T > o.max_context) { ending = true; continue; }
                         if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
                         if (st == 0 && !el_gap()) return die(err);   // fork: the elastic tail
-                        if (!snap_take(st, A.seq)) return die("the GDN snapshot failed");
+                        // STRUCT-A: a thrown-away bet took this snapshot already (a head leaves the GDN state alone)
+                        if (!(st == 0 && sa_snapped == A.seq) && !snap_take(st, A.seq)) return die("the GDN snapshot failed");
                         if (!VF(st, A).pl_launch(A.T, A.tok, A.p, err)) return die(err);
                         A.fl = true;
                         if (st == 0) A.launched = true;
```

### 13-27

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    // the drafter must be idle before the next chain.  The chain needs nothing from the host, so this
                    // wait is bounded even with B (made from its first outputs) in flight on stage 0; in practice the
                    // chain ended long before a verdict that needs a whole stage-1 window.
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                     // the drafter must be idle before the next chain.  The chain needs nothing from the host, so this
                     // wait is bounded even with B (made from its first outputs) in flight on stage 0; in practice the
                     // chain ended long before a verdict that needs a whole stage-1 window.
+                    if (sa_chain && chain_kind != 0 && mtp.chain_live()) {   // STRUCT-A: a chain fed a step at a time ends here
+                        sa_cut += chain_n - mtp.chain_queued();
+                        mtp.chain_cut();
+                    }
                     while (chain_kind != 0 && mtp.chain_live())
                         if (mtp.chain_poll(err) < 0) return die(err);
                     if (chain_kind != 0 && !B.launched && !B.made) B = PW{};
```

### 13-28

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        // the chain over A, forced through B's drafts: B's bonus guess and the next window's drafts
                        mtp.set_source_R(V1(A).final_R(0));
                        chain_n = std::min(strata::kernels::kVerifyMaxT - 1, B.T + S_mtp - 1);
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
                         // the chain over A, forced through B's drafts: B's bonus guess and the next window's drafts
                         mtp.set_source_R(V1(A).final_R(0));
                         chain_n = std::min(strata::kernels::kVerifyMaxT - 1, B.T + S_mtp - 1);
-                        if (!mtp.chain_launch(A.T, outp.data(), A.p, a, B.tok + 1, B.T - 1, chain_n, chain_n, err))
+                        // STRUCT-A (STRATA_SA_CHAIN): only the outputs B's size needs first - B's row 0 and one draft
+                        // past the forced ones - then a step at a time from the loop
+                        const int sa_now = sa_chain ? std::min(chain_n, B.T + 1 + sa_chain_ahead) : 0;
+                        if (sa_chain) ++sa_chains;
+                        if (!mtp.chain_launch(A.T, outp.data(), A.p, a, B.tok + 1, B.T - 1, chain_n, chain_n, err, sa_now))
                             return die(err);
                         chain_kind = 1;
                         tre("CL", A.seq, 1);
```

### 13-29

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            nA.p = A.p + a + 1;
                            nA.tok[0] = outp[(size_t) a];
                            A = nA;
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                             nA.p = A.p + a + 1;
                             nA.tok[0] = outp[(size_t) a];
                             A = nA;
+                            if (sa_bet_on) {   // STRUCT-A: this fresh window's head may go to stage 0 as a bet
+                                sa_want = A.seq;
+                                sa_last_T = 0;
+                            }
                         }
                     }
                     // --adapt-async: the verdict's tick, once the drafter's chain is launched (off its critical path)
```

### 13-30

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                while (mtp.chain_live() && mtp.chain_poll(err) == 0) {}
                mtp.set_source_R(ver.final_R_all());   // the serial loop's rows again
                const double pl_ms = std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count();
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     std::printf("ERR an adaptive refill failed\n");
                     return 1;
                 }
+                if (mtp.chain_live()) mtp.chain_cut();   // STRUCT-A: a chain fed a step at a time ends with what it has
                 while (mtp.chain_live() && mtp.chain_poll(err) == 0) {}
                 mtp.set_source_R(ver.final_R_all());   // the serial loop's rows again
                 const double pl_ms = std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count();
```

### 13-31

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    std::fprintf(stderr, "strata pipeline: drafter on CUDA%d: %.3f ms of host time per chain launch (%.3f ms of it "
                                         "staging the source rows, %lld times so far)\n", mtp.device(),
                                 avg(mtp.ms_draft, (double) mtp.rounds), avg(mtp.ms_src, (double) mtp.n_src), (long long) mtp.n_src);
```
Block (hunk 12 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
                     std::fprintf(stderr, "strata pipeline: drafter on CUDA%d: %.3f ms of host time per chain launch (%.3f ms of it "
                                          "staging the source rows, %lld times so far)\n", mtp.device(),
                                  avg(mtp.ms_draft, (double) mtp.rounds), avg(mtp.ms_src, (double) mtp.n_src), (long long) mtp.n_src);
+                    if (sa_bet_on || sa_chain)   // STRUCT-A
+                        std::fprintf(stderr, "strata struct-a: head bets (placed/tail/dropped) T1 %lld/%lld/%lld, T2 %lld/%lld/%lld, "
+                                             "T3 %lld/%lld/%lld; forced chains fed a step at a time %lld, outputs not run %lld\n",
+                                     (long long) sa_placed[1], (long long) sa_tails[1], (long long) sa_drops[1],
+                                     (long long) sa_placed[2], (long long) sa_tails[2], (long long) sa_drops[2],
+                                     (long long) sa_placed[3], (long long) sa_tails[3], (long long) sa_drops[3],
+                                     (long long) sa_chains, (long long) sa_cut);
                     std::string cal;
                     char b[48];
                     for (int i = 0; i < 10; ++i)
```

## 14 prefix-disk-multi

`14-prefix-disk-multi.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il prefisso su disco su un thread e un file per prefisso (STRATA_PREFIX_DISK_MULTI)

### 14-1

`include/strata/core/prefix_cache_file.hpp` · bool prefix_cache_read(const std::filesystem::path &path,

Anchor:
```
                       const std::vector<int32_t> &expected_prefix,
                       int32_t next_token, bool cvec, SavedConversation &out,
                       std::string &error);
```
Block (hunk 1 of `include/strata/core/prefix_cache_file.hpp` in the patch, 31 lines):
```diff
                        const std::vector<int32_t> &expected_prefix,
                        int32_t next_token, bool cvec, SavedConversation &out,
                        std::string &error);
+
+// Several prefixes beside one another (fork, STRATA_PREFIX_DISK_MULTI): one
+// file per prefix, named `<base>.<16 hex digits>` from a 64-bit FNV-1a of the
+// prefix ids and the frontier token, so two clients with different system
+// prompts no longer overwrite each other's file.  The name is only the
+// lookup: the file still holds its full prefix and prefix_cache_read checks
+// it token by token (a hash collision reads as a miss).
+std::filesystem::path prefix_cache_keyed_path(
+    const std::filesystem::path &base, const std::vector<int32_t> &prefix,
+    int32_t next_token);
+// The keyed files of `base` (its own directory, `<base filename>.` + 16 hex
+// digits; temporary files excluded), with their sizes and write times.
+struct PrefixCacheEntry {
+  std::filesystem::path path;
+  uint64_t bytes = 0;
+  std::filesystem::file_time_type written{};
+};
+std::vector<PrefixCacheEntry>
+prefix_cache_keyed_files(const std::filesystem::path &base);
+// Least recently written first: removes keyed files of `base` until they add
+// up to at most `cap_bytes`, never `keep`.  Returns how many were removed.
+// A read hit refreshes its file's write time (prefix_cache_touch), so the
+// order is the order of use.
+size_t prefix_cache_evict(const std::filesystem::path &base,
+                          uint64_t cap_bytes,
+                          const std::filesystem::path &keep);
+void prefix_cache_touch(const std::filesystem::path &path);
 } // namespace strata::core
```

### 14-2

`src/core/prefix_cache_file.cpp` · (file scope)

Anchor:
```
#include "strata/core/prefix_cache_file.hpp"
```
Block (hunk 1 of `src/core/prefix_cache_file.cpp` in the patch, 6 lines):
```diff
 #include "strata/core/prefix_cache_file.hpp"

+#include <algorithm>
 #include <bit>
 #include <chrono>
 #include <climits>
```

### 14-3

`src/core/prefix_cache_file.cpp` · (file scope)

Anchor:
```
#include <memory>
#include <set>
#include <stdexcept>
```
Block (hunk 2 of `src/core/prefix_cache_file.cpp` in the patch, 7 lines):
```diff
 #include <memory>
 #include <set>
 #include <stdexcept>
+#include <system_error>
 #ifdef _WIN32
 #ifndef NOMINMAX
 #define NOMINMAX
```

### 14-4

`src/core/prefix_cache_file.cpp` · bool prefix_cache_read(const std::filesystem::path &path,

Anchor:
```
} // namespace strata::core
```
Block (hunk 3 of `src/core/prefix_cache_file.cpp` in the patch, 93 lines):
```diff
     return false;
   }
 }
+
+std::filesystem::path prefix_cache_keyed_path(
+    const std::filesystem::path &base, const std::vector<int32_t> &prefix,
+    int32_t next_token) {
+  uint64_t h = 1469598103934665603ull;   // FNV-1a 64
+  auto mix = [&h](const void *data, size_t n) {
+    const auto *p = static_cast<const uint8_t *>(data);
+    for (size_t i = 0; i < n; ++i) {
+      h ^= p[i];
+      h *= 1099511628211ull;
+    }
+  };
+  const uint64_t count = prefix.size();
+  mix(&count, sizeof(count));
+  if (!prefix.empty())
+    mix(prefix.data(), prefix.size() * sizeof(int32_t));
+  mix(&next_token, sizeof(next_token));
+  char suffix[18];
+  std::snprintf(suffix, sizeof(suffix), ".%016llx", (unsigned long long)h);
+  auto path = base;
+  path += suffix;
... 69 more lines: see 14-prefix-disk-multi.patch
```

### 14-5

`src/core/prefix_cache_file_test.cpp` · int main() {

Anchor:
```
    reject_split_write("split-above-token-cap", over_cap);
  }
```
Block (hunk 1 of `src/core/prefix_cache_file_test.cpp` in the patch, 44 lines):
```diff
     reject_split_write("split-above-token-cap", over_cap);
   }

+  {   // fork (TTFT): one file per prefix, least recently used out past the cap
+    const fs::path base = temp.path / "multi" / "prefix.bin";
+    fs::create_directories(base.parent_path());
+    const std::vector<int32_t> a = {1, 2, 3}, b = {1, 2, 4};
+    const auto ka = prefix_cache_keyed_path(base, a, 7);
+    check(ka == prefix_cache_keyed_path(base, a, 7), "keyed path is stable");
+    check(ka != prefix_cache_keyed_path(base, b, 7) &&
+              ka != prefix_cache_keyed_path(base, a, 8),
+          "keyed path follows the prefix and the frontier");
+    check(ka.parent_path() == base.parent_path() &&
+              ka.filename().string().size() == base.filename().string().size() + 17,
+          "keyed path is <base>.<16 hex>");
+    auto put = [](const fs::path &p, size_t bytes) {
+      std::ofstream f(p, std::ios::binary);
+      const std::string data(bytes, 'x');
+      f.write(data.data(), (std::streamsize)data.size());
+    };
+    const auto kb = prefix_cache_keyed_path(base, b, 7);
+    const auto kc = prefix_cache_keyed_path(base, {9}, 7);
+    put(ka, 100);
+    put(kb, 100);
+    put(kc, 100);
+    put(base, 1000);                       // the single file is not a keyed one
+    put(fs::path(ka.string() + ".tmp-1-2"), 1000);   // nor is a temporary
+    const auto now = fs::file_time_type::clock::now();
+    fs::last_write_time(ka, now - std::chrono::hours(3));
+    fs::last_write_time(kb, now - std::chrono::hours(2));
+    fs::last_write_time(kc, now - std::chrono::hours(1));
+    check(prefix_cache_keyed_files(base).size() == 3, "three keyed files");
+    prefix_cache_touch(ka);                // a read hit: ka is the newest now
+    check(prefix_cache_evict(base, 250, kc) == 1 && !fs::exists(kb) &&
+              fs::exists(ka) && fs::exists(kc) && fs::exists(base),
+          "eviction removes the least recently used keyed file only");
+    check(prefix_cache_evict(base, 0, kc) == 1 && !fs::exists(ka) &&
+              fs::exists(kc),
+          "eviction never removes the file just written");
+  }
+
   std::printf("prefix_cache_file_test: %d checks passed\n", checks);
   return 0;
 }
```

### 14-6

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            if (wchar_t* block = GetEnvironmentStringsW()) {
                for (const wchar_t* p = block; *p; p += std::wcslen(p) + 1) {
                    if (_wcsnicmp(p, L"STRATA_", 7) && _wcsnicmp(p, L"CUDA_", 5)) continue;
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 11 lines):
```diff
             if (wchar_t* block = GetEnvironmentStringsW()) {
                 for (const wchar_t* p = block; *p; p += std::wcslen(p) + 1) {
                     if (_wcsnicmp(p, L"STRATA_", 7) && _wcsnicmp(p, L"CUDA_", 5)) continue;
+                    // fork (TTFT): the switches of the disk store and of the PLE read-ahead change no computed
+                    // value, so they do not make the saved prefixes another engine's
+                    if (!_wcsnicmp(p, L"STRATA_PREFIX_DISK_MULTI=", 25) || !_wcsnicmp(p, L"STRATA_PREFIX_DISK_SYNC=", 24) ||
+                        !_wcsnicmp(p, L"STRATA_PLE_LOOKAHEAD=", 21))
+                        continue;
                     const int count = WideCharToMultiByte(CP_UTF8, 0, p, -1, nullptr, 0, nullptr, nullptr);
                     std::string entry(size_t(count > 0 ? count : 1), '\0');
                     if (count > 0 && WideCharToMultiByte(CP_UTF8, 0, p, -1, entry.data(), count, nullptr, nullptr)) {
```

### 14-7

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            } else { prefix_disk_enabled = false; err = "environment identity unavailable"; }
#else
            for (char** p = environ; p && *p; ++p)
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 11 lines):
```diff
             } else { prefix_disk_enabled = false; err = "environment identity unavailable"; }
 #else
             for (char** p = environ; p && *p; ++p)
-                if (!std::strncmp(*p, "STRATA_", 7) || !std::strncmp(*p, "CUDA_", 5)) environment.emplace_back(*p);
+                if ((!std::strncmp(*p, "STRATA_", 7) || !std::strncmp(*p, "CUDA_", 5)) &&
+                    std::strncmp(*p, "STRATA_PREFIX_DISK_MULTI=", 25) && std::strncmp(*p, "STRATA_PREFIX_DISK_SYNC=", 24) &&
+                    std::strncmp(*p, "STRATA_PLE_LOOKAHEAD=", 21))
+                    environment.emplace_back(*p);
 #endif
             std::sort(environment.begin(), environment.end());
             settings.insert(settings.end(), environment.begin(), environment.end());
```

### 14-8

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                         prefix_disk_enabled ? "" : ": ", prefix_disk_enabled ? "" : err.c_str());
            err.clear();
        }
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 26 lines):
```diff
                          prefix_disk_enabled ? "" : ": ", prefix_disk_enabled ? "" : err.c_str());
             err.clear();
         }
+        // fork (TTFT): STRATA_PREFIX_DISK_MULTI=<GiB> keeps one file per prefix beside --prompt-cache-file
+        // (`<file>.<16 hex>`, prefix_cache_keyed_path), least recently used out past the cap, so two clients (Claude
+        // Code and DSH) no longer overwrite each other's system turn; the single file is still read when no keyed one
+        // matches.  Unset or 0: the one file, as before.
+        const uint64_t prefix_multi_cap = [] {
+            const char* v = std::getenv("STRATA_PREFIX_DISK_MULTI");
+            const double gib = v != nullptr ? std::atof(v) : 0.0;
+            return gib > 0.0 ? (uint64_t) (gib * 1024.0 * 1024.0 * 1024.0) : (uint64_t) 0;
+        }();
+        if (prefix_disk_enabled && prefix_multi_cap > 0)
+            std::fprintf(stderr, "strata serve: prefix disk cache: one file per prefix (%s.<key>), at most %.1f GiB\n",
+                         o.prompt_cache_file.c_str(), (double) prefix_multi_cap / (1024.0 * 1024.0 * 1024.0));
+        // fork (TTFT): the file is written on a thread once the image is on the host (the SHA-256 and the write were
+        // ~0.5 s of the 0.76 s a 26K system turn added to the first request's prompt); STRATA_PREFIX_DISK_SYNC=1
+        // writes it inline as before.  A read waits for a write in flight; the engine's exit waits for it too.
+        static const bool prefix_write_sync = [] {
+            const char* v = std::getenv("STRATA_PREFIX_DISK_SYNC");
+            return v != nullptr && std::atoi(v) != 0;
+        }();
+        std::future<void> prefix_write_job;
         strata::core::ConversationCache conversations(
             o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
             (size_t) o.conversation_cache_slots, o.conversation_cache_min_tokens);
```

### 14-9

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                if (disk_root >= o.prompt_cache_root && disk_root <= int64_t(strata::core::prefix_cache_max_tokens) &&
                        disk_root > std::max(resume, (incoming || borrow) ? incoming_tokens : int64_t(0))) {
                    const auto t0 = Clock::now();
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 24 lines):
```diff
                 if (disk_root >= o.prompt_cache_root && disk_root <= int64_t(strata::core::prefix_cache_max_tokens) &&
                         disk_root > std::max(resume, (incoming || borrow) ? incoming_tokens : int64_t(0))) {
                     const auto t0 = Clock::now();
+                    if (prefix_write_job.valid()) prefix_write_job.wait();   // a save in flight: this read sees it
+                    std::vector<int32_t> prefix(ids.begin(), ids.begin() + disk_root);
+                    // STRATA_PREFIX_DISK_MULTI: this prefix's own file when there is one, else the single file
+                    std::filesystem::path disk_path = o.prompt_cache_file;
+                    if (prefix_multi_cap > 0) {
+                        std::error_code kec;
+                        auto keyed = strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix,
+                                                                           int32_t(ids[size_t(disk_root)]));
+                        if (std::filesystem::is_regular_file(keyed, kec) && !kec) disk_path = std::move(keyed);
+                    }
                     std::error_code ec;
-                    const auto bytes = std::filesystem::file_size(o.prompt_cache_file, ec);
+                    const auto bytes = std::filesystem::file_size(disk_path, ec);
                     const uint64_t floor = uint64_t(o.conversation_cache_min_free_mib) * 1024 * 1024;
                     const uint64_t held = incoming ? incoming->bytes() : 0;
                     if (!ec && bytes <= strata::core::prefix_cache_max_bytes + 88 &&
                             strata::core::conversation_memory_admit(strata::core::available_host_bytes(), bytes + held + 1024 * 1024, floor)) {
-                        std::vector<int32_t> prefix(ids.begin(), ids.begin() + disk_root);
                         strata::core::SavedConversation image;
                         // One GPU: the whole-session form, as before.  A layer split: the same checks as a parked
                         // image - each later stage's image against its own session (the draft K/V with the last
```

### 14-10

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            return strata::core::conversation_snapshot_validate(im, ss, g,
                                       static_cast<const strata::core::QsaState*>(nullptr), err);
                        };
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
                             return strata::core::conversation_snapshot_validate(im, ss, g,
                                        static_cast<const strata::core::QsaState*>(nullptr), err);
                         };
-                        if (strata::core::prefix_cache_read(o.prompt_cache_file, prefix_identity, prefix,
+                        if (strata::core::prefix_cache_read(disk_path, prefix_identity, prefix,
                                 int32_t(ids[size_t(disk_root)]), want_cvec, image, err) &&
                                 disk_image_valid(image)) {
+                            if (prefix_multi_cap > 0 && disk_path != o.prompt_cache_file)
+                                strata::core::prefix_cache_touch(disk_path);   // the LRU order is the order of use
                             incoming.emplace(std::move(image)); incoming_tokens = disk_root;
                             incoming_live = true; incoming_disk = true;
                             borrow = false;   // the disk root is the longer match: a donor stays parked, untouched
```

### 14-11

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                // A CUDA copy/synchronization error can poison the live session.
                                std::printf("ERR capturing prefix disk cache: %s\n", err.c_str()); return 1;
                            }
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 50 lines):
```diff
                                 // A CUDA copy/synchronization error can poison the live session.
                                 std::printf("ERR capturing prefix disk cache: %s\n", err.c_str()); return 1;
                             }
-                            const bool stored = strata::core::prefix_cache_write(o.prompt_cache_file, prefix_identity,
-                                                    int32_t(ids[size_t(to)]), image, err);
-                            std::fprintf(stderr, "strata serve: prefix disk cache: %s %lld tokens, %zu bytes in %.1f ms%s%s\n",
-                                         stored ? "saved" : "skip saving", (long long) to, image.bytes(),
-                                         std::chrono::duration<double, std::milli>(Clock::now() - save_start).count(),
-                                         stored ? "" : ": ", stored ? "" : err.c_str());
+                            // fork (TTFT): the image is on the host, the SHA-256 and the file write need nothing of
+                            // the GPU: a thread does them (STRATA_PREFIX_DISK_SYNC=1: inline, as before)
+                            const double capture_ms = std::chrono::duration<double, std::milli>(Clock::now() - save_start).count();
+                            const std::filesystem::path disk_target = prefix_multi_cap > 0
+                                ? strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix, int32_t(ids[size_t(to)]))
+                                : std::filesystem::path(o.prompt_cache_file);
+                            auto write_job = [img = std::make_shared<strata::core::SavedConversation>(std::move(image)),
+                                              base = std::filesystem::path(o.prompt_cache_file), disk_target,
+                                              identity = prefix_identity, next = int32_t(ids[size_t(to)]),
+                                              tokens = (long long) to, cap = prefix_multi_cap, save_start, capture_ms,
+                                              background = !prefix_write_sync]() {
+                                if (background) strata::aux_cpus::pin_current_thread();
+                                std::string we;
+                                const bool stored = strata::core::prefix_cache_write(disk_target, identity, next, *img, we);
+                                const size_t evicted = stored && cap > 0 ? strata::core::prefix_cache_evict(base, cap, disk_target) : 0;
+                                std::string tail;
+                                char buf[160];
+                                if (background) {
+                                    std::snprintf(buf, sizeof buf, " (on a thread; the request waited %.1f ms for the capture)",
+                                                  capture_ms);
+                                    tail += buf;
+                                }
+                                if (cap > 0) {
+                                    std::snprintf(buf, sizeof buf, " -> %s, %zu older removed",
+                                                  disk_target.filename().string().c_str(), evicted);
+                                    tail += buf;
+                                }
+                                std::fprintf(stderr, "strata serve: prefix disk cache: %s %lld tokens, %zu bytes in %.1f ms%s%s%s\n",
+                                             stored ? "saved" : "skip saving", tokens, img->bytes(),
+                                             std::chrono::duration<double, std::milli>(Clock::now() - save_start).count(),
+                                             stored ? "" : ": ", stored ? "" : we.c_str(), tail.c_str());
+                            };
+                            if (prefix_write_sync) {
+                                write_job();
+                            } else {
+                                if (prefix_write_job.valid()) prefix_write_job.wait();
+                                prefix_write_job = std::async(std::launch::async, std::move(write_job));
+                            }
                         } else {
                             std::fprintf(stderr, "strata serve: prefix disk cache: skip saving (size, geometry or physical RAM admission)\n");
                         }
```

## 15 ple-lookahead

`15-ple-lookahead.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la lettura in anticipo delle righe PLE del primo chunk della parte dopo (STRATA_PLE_LOOKAHEAD)

### 15-1

`include/strata/kernels/ngram.hpp` · public:

Anchor:
```
    /// as ONE reader request - page dedupe and sort across the whole batch, the reader's full queue depth.  Not
    /// while a single-token `issue` is pending.  The mapped mode gathers row by row.
    bool gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err);
```
Block (hunk 1 of `include/strata/kernels/ngram.hpp` in the patch, 12 lines):
```diff
     /// as ONE reader request - page dedupe and sort across the whole batch, the reader's full queue depth.  Not
     /// while a single-token `issue` is pending.  The mapped mode gathers row by row.
     bool gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err);
+    /// Fork (TTFT): read the rows of `n_tokens` tokens into the reader's row cache and nothing else, so a later
+    /// gather_batch of them is served from RAM (the same bytes: the cache holds what the file holds).  Only in Direct
+    /// mode with the reader's I/O thread, where it may run on another thread beside every other call (it issues and
+    /// collects its own reader ticket, nothing of this object's state); returns false having done nothing otherwise,
+    /// or when the row cache is too small to keep them.
+    bool warm_rows(const uint32_t* rows, size_t n_tokens, std::string& err);

     /// Fault injection (Direct mode): every row read completes no earlier than `us` after issue.
     void set_injected_delay_us(double us);
```

### 15-2

`include/strata/ngram/ple_reader.hpp` · public:

Anchor:
```
    void reset_stats();
    uint64_t cache_capacity() const;
    uint64_t cache_size() const;
```
Block (hunk 1 of `include/strata/ngram/ple_reader.hpp` in the patch, 9 lines):
```diff
     void reset_stats();
     uint64_t cache_capacity() const;
     uint64_t cache_size() const;
+    /// True when a worker thread owns the reads (`open` with io_thread): `issue`/`collect` may then be called
+    /// from several threads (fork, TTFT: PleTable::warm_rows).
+    bool threaded() const;

 private:
     struct Impl;
```

### 15-3

`include/strata/prefill/prefill.hpp` · (file scope)

Anchor:
```
#include <future>
#include <memory>
#include <string>
```
Block (hunk 1 of `include/strata/prefill/prefill.hpp` in the patch, 7 lines):
```diff
 #include <future>
 #include <memory>
 #include <string>
+#include <vector>

 namespace strata::core { class PeerExperts; }
 namespace strata::kernels::cpu { class ExpertPool; }
```

### 15-4

`include/strata/prefill/prefill.hpp` · public:

Anchor:
```

    const PrefillStats& stats() const { return stats_; }
```
Block (hunk 2 of `include/strata/prefill/prefill.hpp` in the patch, 17 lines):
```diff

     const PrefillStats& stats() const { return stats_; }

+    /// Fork (TTFT): the PLE rows of the NEXT run's first tokens (`tokens`, `n` of them, `prev0`/`prev1` the two before
+    /// them, oldest first).  The first chunk of every run waits for its whole chunk's rows (the read-ahead only covers
+    /// the chunks after it): 5.7 s of SSD reads for the part after the system turn of Claude Code's first request.
+    /// Set before a run, they are read into the PLE row cache on a thread once this run has its last chunk's rows (the
+    /// SSD is idle then), and the next run's first gather finds them there: the same bytes, earlier.  A run waits for
+    /// a warm still in flight before its first gather.  No-op on a stage without the PLE, in mmap mode, without the
+    /// reader's I/O thread or a row cache that can hold them.  STRATA_PLE_LOOKAHEAD=0 turns both calls off.
+    void set_ple_lookahead(const int64_t* tokens, int64_t n, int32_t prev0, int32_t prev1);
+    /// The same read, started now (e.g. beside a prefix restore, before the run that needs them).
+    void ple_warm_now(const int64_t* tokens, int64_t n, int32_t prev0, int32_t prev1);
+
     /// multi-GPU: the experts the peer GPU holds are computed THERE for every prompt chunk (up to `cap_rows` routed
     /// rows per layer; the rest of the peer's experts are read by this GPU over P2P).  Allocates the peer's buffers for
     /// chunks of up to init's `chunk` tokens.  Needs P2P between the two cards.
```

### 15-5

`include/strata/prefill/prefill.hpp` · private:

Anchor:
```
    std::future<bool> next_run_;
    int hand_buf_ = 0;
```
Block (hunk 3 of `include/strata/prefill/prefill.hpp` in the patch, 13 lines):
```diff
     std::future<bool> next_run_;
     int hand_buf_ = 0;

+    // fork (TTFT): set_ple_lookahead / ple_warm_now
+    std::vector<int64_t> ple_ahead_;
+    int32_t ple_ahead_prev_[2] = {-1, -1};
+    std::future<bool> ple_warm_;
+    void ple_warm_launch(const int64_t* tokens, int64_t n, int32_t prev0, int32_t prev1);
+    void ple_warm_wait();
+
     bool carve(std::size_t T, void* alloc);   // the device buffers of a chunk (prefill.cpp's Alloc)
     void release();                          // the destructor's cleanup (also `reset`'s)
     struct Impl;
```

### 15-6

`src/kernels/ngram.cpp` · bool PleTable::gather_batch(const uint32_t* rows, size_t n_tokens, float* out, s

Anchor:
```
    return true;
}
```
Block (hunk 1 of `src/kernels/ngram.cpp` in the patch, 16 lines):
```diff
     return true;
 }

+bool PleTable::warm_rows(const uint32_t* rows, size_t n_tokens, std::string& err) {
+    const size_t n = n_tokens * (size_t) PLE_N_HEADS;
+    if (n == 0 || impl_->mode != PleIo::Direct || !impl_->reader.is_open() || !impl_->reader.threaded() ||
+        impl_->reader.cache_capacity() < 2 * (uint64_t) n)
+        return false;
+    std::vector<uint8_t> raw(n * impl_->rb);
+    const auto ticket = impl_->reader.issue(rows, n, raw.data());
+    return impl_->reader.collect(ticket, err);
+}
+
 void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_delay_us(us); }

 std::string PleTable::io_report() const {
```

### 15-7

`src/ngram/ple_reader.cpp` · void PleReader::reset_stats() {

Anchor:
```
}
uint64_t PleReader::cache_capacity() const { return impl_->cache.sets * WAYS; }
uint64_t PleReader::cache_size() const { return impl_->cache.used; }
```
Block (hunk 1 of `src/ngram/ple_reader.cpp` in the patch, 6 lines):
```diff
 }
 uint64_t PleReader::cache_capacity() const { return impl_->cache.sets * WAYS; }
 uint64_t PleReader::cache_size() const { return impl_->cache.used; }
+bool PleReader::threaded() const { return impl_->threaded; }

 }  // namespace strata::ngram
```

### 15-8

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
        return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
    };
    std::string ple_next_err;
```
Block (hunk 1 of `src/prefill/prefill.cpp` in the patch, 10 lines):
```diff
         return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
     };
     std::string ple_next_err;
+    struct AheadOnce {   // fork (TTFT): a lookahead belongs to the run after this one only
+        std::vector<int64_t>& v;
+        ~AheadOnce() { v.clear(); }
+    } ahead_once{ple_ahead_};
     std::future<bool> ple_next;             // declared after everything it reads: an early return waits for it
     int ple_buf = 0;

```

### 15-9

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
        // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
        // layer 1 on, and gathering them here first left the GPU idle for the whole read (~0.4 s of a 32K prompt)
        if (ple_on && !ple_next.valid())
```
Block (hunk 2 of `src/prefill/prefill.cpp` in the patch, 10 lines):
```diff
         // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
         // layer 1 on, and gathering them here first left the GPU idle for the whole read (~0.4 s of a 32K prompt)
         if (ple_on && !ple_next.valid())
-            ple_next = std::async(std::launch::async, [&ple_gather, &ple_next_err, c0, b = ple_buf] {
+            ple_next = std::async(std::launch::async, [this, &ple_gather, &ple_next_err, c0, b = ple_buf] {
                 strata::aux_cpus::pin_current_thread();
+                ple_warm_wait();   // fork (TTFT): rows a lookahead is still reading land in the row cache first
                 return ple_gather(c0, b, ple_next_err);
             });
         bool ple_pending = ple_on;
```

### 15-10

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                    strata::aux_cpus::pin_current_thread();
                    return ple_gather(c1, b, ple_next_err);
                });
```
Block (hunk 3 of `src/prefill/prefill.cpp` in the patch, 10 lines):
```diff
                     strata::aux_cpus::pin_current_thread();
                     return ple_gather(c1, b, ple_next_err);
                 });
+            } else if (!ple_ahead_.empty()) {
+                // fork (TTFT): this run's last rows are here and the SSD is idle: the next run's first chunk now
+                ple_warm_launch(ple_ahead_.data(), (int64_t) ple_ahead_.size(), ple_ahead_prev_[0], ple_ahead_prev_[1]);
+                ple_ahead_.clear();
             }
             ple_buf ^= 1;
             const double waited = ms_since(tp);
```

### 15-11

`src/prefill/prefill.cpp` · bool Prefill::reserve_uploads(int64_t T) {

Anchor:
```
bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
#if defined(STRATA_GPU_METAL)
    {
```
Block (hunk 4 of `src/prefill/prefill.cpp` in the patch, 58 lines):
```diff
     return ok;
 }

+// fork (TTFT): STRATA_PLE_LOOKAHEAD=0 turns set_ple_lookahead / ple_warm_now off
+namespace {
+bool ple_lookahead_on() {
+    static const bool on = [] {
+        const char* e = std::getenv("STRATA_PLE_LOOKAHEAD");
+        return e == nullptr || std::atoi(e) != 0;
+    }();
+    return on;
+}
+constexpr int64_t kPleWarmMaxTokens = 8192;   // a chunk of the auto prefill at most: what the first gather asks for
+}  // namespace
+
+void Prefill::ple_warm_wait() {
+    if (ple_warm_.valid()) (void) ple_warm_.get();   // a failed warm only means the gather reads the rows itself
+}
+
+void Prefill::ple_warm_launch(const int64_t* tokens, int64_t n, int32_t prev0, int32_t prev1) {
+    if (!ple_lookahead_on() || impl_ == nullptr || impl_->ss == nullptr || tokens == nullptr || n <= 0) return;
+    core::SessionState& ss = *impl_->ss;
+    if (!ss.ple.ready() || !(stage_lb_ <= 1 && 1 < stage_le_) || ss.ple.table->mode() != kernels::PleIo::Direct) return;
+    ple_warm_wait();   // one at a time
+    const int64_t T = std::min<int64_t>(n, kPleWarmMaxTokens);
+    std::vector<int32_t> toks((size_t) T);
+    for (int64_t t = 0; t < T; ++t) toks[(size_t) t] = (int32_t) tokens[t];
+    ple_warm_ = std::async(std::launch::async, [table = ss.ple.table, consts = &ss.ple.consts, toks = std::move(toks),
+                                                prev0, prev1]() -> bool {
+        strata::aux_cpus::pin_current_thread();
+        // the rows exactly as the prefill's ple_gather hashes them: each token after the two before it
+        std::vector<uint32_t> rows(toks.size() * (size_t) strata::kernels::PLE_N_HEADS);
+        int32_t pv[2] = {prev0, prev1};
+        for (size_t t = 0; t < toks.size(); ++t) {
+            strata::kernels::ngram_rows(&toks[t], pv, 1, *consts, rows.data() + t * strata::kernels::PLE_N_HEADS);
+            pv[0] = pv[1];
+            pv[1] = toks[t];
+        }
+        std::string e;
+        return table->warm_rows(rows.data(), toks.size(), e);
+    });
+}
+
+void Prefill::set_ple_lookahead(const int64_t* tokens, int64_t n, int32_t prev0, int32_t prev1) {
+    ple_ahead_.clear();
+    if (!ple_lookahead_on() || tokens == nullptr || n <= 0) return;
+    ple_ahead_.assign(tokens, tokens + std::min<int64_t>(n, kPleWarmMaxTokens));
+    ple_ahead_prev_[0] = prev0;
+    ple_ahead_prev_[1] = prev1;
+}
+
+void Prefill::ple_warm_now(const int64_t* tokens, int64_t n, int32_t prev0, int32_t prev1) {
+    ple_warm_launch(tokens, n, prev0, prev1);
+}
+
 bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
 #if defined(STRATA_GPU_METAL)
     {
```

### 15-12

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        std::error_code kec;
                        auto keyed = strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix,
                                                                           int32_t(ids[size_t(disk_root)]));
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 15 lines):
```diff
                         std::error_code kec;
                         auto keyed = strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix,
                                                                            int32_t(ids[size_t(disk_root)]));
-                        if (std::filesystem::is_regular_file(keyed, kec) && !kec) disk_path = std::move(keyed);
+                        if (std::filesystem::is_regular_file(keyed, kec) && !kec) {
+                            disk_path = std::move(keyed);
+                            // fork (TTFT): this prefix's own file (a hit unless the engine's identity changed): the
+                            // PLE rows of the part read after it are read beside the restore (~0.7 s of file read)
+                            if (o.batch <= 0 && disk_root >= 2 && last_turn - disk_root > o.short_read)
+                                sp.ple_warm_now(ids.data() + disk_root, last_turn - disk_root,
+                                                (int32_t) ids[size_t(disk_root - 2)], (int32_t) ids[size_t(disk_root - 1)]);
+                        }
                     }
                     std::error_code ec;
                     const auto bytes = std::filesystem::file_size(disk_path, ec);
```

### 15-13

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                }
                if (win) merge_at = merge_from(at, to);   // (after the refill: the loans are back)
                const int64_t read_to = merge_at >= 0 ? merge_at : to;   // (merge_at > at: the part's windows before it)
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 19 lines):
```diff
                 }
                 if (win) merge_at = merge_from(at, to);   // (after the refill: the loans are back)
                 const int64_t read_to = merge_at >= 0 ? merge_at : to;   // (merge_at > at: the part's windows before it)
+                // fork (TTFT): a batched part followed by another batched part (Claude Code's system turn, then its
+                // first message): the next part's first PLE rows are read once this part has its last chunk's rows,
+                // so the next run's first chunk does not wait for the SSD (Prefill::set_ple_lookahead)
+                if (!win && o.batch <= 0) {
+                    int64_t next_to = INT64_MAX;
+                    for (const int64_t c : cuts)
+                        if (c > to) next_to = std::min(next_to, c);
+                    if (next_to != INT64_MAX && to >= 2 && !windows_ok(to, next_to))
+                        sp.set_ple_lookahead(ids.data() + to, next_to - to, (int32_t) ids[(size_t) (to - 2)],
+                                             (int32_t) ids[(size_t) (to - 1)]);
+                    else
+                        sp.set_ple_lookahead(nullptr, 0, -1, -1);
+                }
                 const auto tsp = Clock::now();
                 pp_exact = false;   // a batched or pipelined part is in flight: a cancel may leave the state past pp_reached
                 const bool sp_ok = read_to <= at || (win ? read_windows(at, read_to, err) : read_part(at, to, err));
```

## 16 conv-disk

`16-conv-disk.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il salvataggio su disco della conversazione al confine di turno a motore fermo (STRATA_CONV_DISK)

### 16-1

`include/strata/core/prefix_cache_file.hpp` · bool prefix_cache_read(const std::filesystem::path &path,

Anchor:
```
std::filesystem::path prefix_cache_keyed_path(
    const std::filesystem::path &base, const std::vector<int32_t> &prefix,
    int32_t next_token);
```
Block (hunk 1 of `include/strata/core/prefix_cache_file.hpp` in the patch, 13 lines):
```diff
 std::filesystem::path prefix_cache_keyed_path(
     const std::filesystem::path &base, const std::vector<int32_t> &prefix,
     int32_t next_token);
+// The same names for several prefixes of one prompt in one pass (fork, STRATA_CONV_DISK): for each position a
+// in `at` (ascending, each < ids.size()) the keyed path of ids[0, a) with the frontier ids[a].  The result is
+// shorter than `at` when a position is out of order or out of range.
+std::vector<std::filesystem::path>
+prefix_cache_keyed_paths(const std::filesystem::path &base,
+                         const std::vector<int32_t> &ids,
+                         const std::vector<int64_t> &at);
 // The keyed files of `base` (its own directory, `<base filename>.` + 16 hex
 // digits; temporary files excluded), with their sizes and write times.
 struct PrefixCacheEntry {
```

### 16-2

`src/core/prefix_cache_file.cpp` · bool prefix_cache_read(const std::filesystem::path &path,

Anchor:
```
std::filesystem::path prefix_cache_keyed_path(
    const std::filesystem::path &base, const std::vector<int32_t> &prefix,
    int32_t next_token) {
```
Block (hunk 1 of `src/core/prefix_cache_file.cpp` in the patch, 71 lines):
```diff
   }
 }

-std::filesystem::path prefix_cache_keyed_path(
-    const std::filesystem::path &base, const std::vector<int32_t> &prefix,
-    int32_t next_token) {
-  uint64_t h = 1469598103934665603ull;   // FNV-1a 64
-  auto mix = [&h](const void *data, size_t n) {
-    const auto *p = static_cast<const uint8_t *>(data);
-    for (size_t i = 0; i < n; ++i) {
-      h ^= p[i];
-      h *= 1099511628211ull;
-    }
-  };
-  const uint64_t count = prefix.size();
-  mix(&count, sizeof(count));
-  if (!prefix.empty())
-    mix(prefix.data(), prefix.size() * sizeof(int32_t));
-  mix(&next_token, sizeof(next_token));
+namespace {
+// FNV-1a 64 over the ids, then the count and the frontier token: the ids come first so one pass over a prompt gives
+// the key of every prefix of it (prefix_cache_keyed_paths)
+constexpr uint64_t kFnvBasis = 1469598103934665603ull;
+inline void fnv_mix(uint64_t &h, const void *data, size_t n) {
+  const auto *p = static_cast<const uint8_t *>(data);
+  for (size_t i = 0; i < n; ++i) {
+    h ^= p[i];
+    h *= 1099511628211ull;
+  }
+}
+std::filesystem::path keyed_name(const std::filesystem::path &base, uint64_t h,
+                                 uint64_t count, int32_t next_token) {
+  fnv_mix(h, &count, sizeof(count));
+  fnv_mix(h, &next_token, sizeof(next_token));
   char suffix[18];
   std::snprintf(suffix, sizeof(suffix), ".%016llx", (unsigned long long)h);
   auto path = base;
   path += suffix;
   return path;
 }
+} // namespace
+
+std::filesystem::path prefix_cache_keyed_path(
+    const std::filesystem::path &base, const std::vector<int32_t> &prefix,
+    int32_t next_token) {
+  uint64_t h = kFnvBasis;
+  if (!prefix.empty())
+    fnv_mix(h, prefix.data(), prefix.size() * sizeof(int32_t));
+  return keyed_name(base, h, prefix.size(), next_token);
+}
+
+std::vector<std::filesystem::path>
+prefix_cache_keyed_paths(const std::filesystem::path &base,
+                         const std::vector<int32_t> &ids,
+                         const std::vector<int64_t> &at) {
+  std::vector<std::filesystem::path> out;
+  out.reserve(at.size());
+  uint64_t h = kFnvBasis;
+  int64_t done = 0;
+  for (const int64_t a : at) {
+    if (a < done || a >= (int64_t)ids.size())
+      break;   // ascending and inside ids, else the rest is not computed
+    fnv_mix(h, ids.data() + done, size_t(a - done) * sizeof(int32_t));
+    done = a;
+    out.push_back(keyed_name(base, h, (uint64_t)a, ids[size_t(a)]));
+  }
+  return out;
+}

 std::vector<PrefixCacheEntry>
 prefix_cache_keyed_files(const std::filesystem::path &base) {
```

### 16-3

`src/core/prefix_cache_file_test.cpp` · int main() {

Anchor:
```
    check(ka.parent_path() == base.parent_path() &&
              ka.filename().string().size() == base.filename().string().size() + 17,
          "keyed path is <base>.<16 hex>");
```
Block (hunk 1 of `src/core/prefix_cache_file_test.cpp` in the patch, 18 lines):
```diff
     check(ka.parent_path() == base.parent_path() &&
               ka.filename().string().size() == base.filename().string().size() + 17,
           "keyed path is <base>.<16 hex>");
+    {   // STRATA_CONV_DISK: every prefix of one prompt in one pass, the same names
+      const std::vector<int32_t> ids = {5, 6, 7, 8, 9, 10};
+      const auto many = prefix_cache_keyed_paths(base, ids, {1, 3, 5});
+      check(many.size() == 3 &&
+                many[0] == prefix_cache_keyed_path(base, {5}, 6) &&
+                many[1] == prefix_cache_keyed_path(base, {5, 6, 7}, 8) &&
+                many[2] == prefix_cache_keyed_path(base, {5, 6, 7, 8, 9}, 10),
+            "one-pass keys equal the single keys");
+      check(prefix_cache_keyed_paths(base, ids, {3, 1}).size() == 1 &&
+                prefix_cache_keyed_paths(base, ids, {6}).empty(),
+            "out of order or out of range positions stop the pass");
+    }
     auto put = [](const fs::path &p, size_t bytes) {
       std::ofstream f(p, std::ios::binary);
       const std::string data(bytes, 'x');
```

### 16-4

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    // fork (TTFT): the switches of the disk store and of the PLE read-ahead change no computed
                    // value, so they do not make the saved prefixes another engine's
                    if (!_wcsnicmp(p, L"STRATA_PREFIX_DISK_MULTI=", 25) || !_wcsnicmp(p, L"STRATA_PREFIX_DISK_SYNC=", 24) ||
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
                     // fork (TTFT): the switches of the disk store and of the PLE read-ahead change no computed
                     // value, so they do not make the saved prefixes another engine's
                     if (!_wcsnicmp(p, L"STRATA_PREFIX_DISK_MULTI=", 25) || !_wcsnicmp(p, L"STRATA_PREFIX_DISK_SYNC=", 24) ||
-                        !_wcsnicmp(p, L"STRATA_PLE_LOOKAHEAD=", 21))
+                        !_wcsnicmp(p, L"STRATA_PLE_LOOKAHEAD=", 21) || !_wcsnicmp(p, L"STRATA_CONV_DISK", 16))
                         continue;
                     const int count = WideCharToMultiByte(CP_UTF8, 0, p, -1, nullptr, 0, nullptr, nullptr);
                     std::string entry(size_t(count > 0 ? count : 1), '\0');
```

### 16-5

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            for (char** p = environ; p && *p; ++p)
                if ((!std::strncmp(*p, "STRATA_", 7) || !std::strncmp(*p, "CUDA_", 5)) &&
                    std::strncmp(*p, "STRATA_PREFIX_DISK_MULTI=", 25) && std::strncmp(*p, "STRATA_PREFIX_DISK_SYNC=", 24) &&
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             for (char** p = environ; p && *p; ++p)
                 if ((!std::strncmp(*p, "STRATA_", 7) || !std::strncmp(*p, "CUDA_", 5)) &&
                     std::strncmp(*p, "STRATA_PREFIX_DISK_MULTI=", 25) && std::strncmp(*p, "STRATA_PREFIX_DISK_SYNC=", 24) &&
-                    std::strncmp(*p, "STRATA_PLE_LOOKAHEAD=", 21))
+                    std::strncmp(*p, "STRATA_PLE_LOOKAHEAD=", 21) && std::strncmp(*p, "STRATA_CONV_DISK", 16))
                     environment.emplace_back(*p);
 #endif
             std::sort(environment.begin(), environment.end());
```

### 16-6

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            return v != nullptr && std::atoi(v) != 0;
        }();
        std::future<void> prefix_write_job;
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 64 lines):
```diff
             return v != nullptr && std::atoi(v) != 0;
         }();
         std::future<void> prefix_write_job;
+        // fork (TTFT): a captured prefix image to `disk_target`, on a thread unless STRATA_PREFIX_DISK_SYNC=1; with
+        // STRATA_PREFIX_DISK_MULTI the least recently used keyed files past the cap go.  `what` names the cache in the
+        // log line ("prefix disk cache" for the system turn, "conversation disk cache" for STRATA_CONV_DISK).
+        auto store_prefix = [&](strata::core::SavedConversation&& image, std::filesystem::path disk_target, int32_t next,
+                                long long tokens, Clock::time_point save_start, double capture_ms, const char* what,
+                                const char* waited) {
+            auto write_job = [img = std::make_shared<strata::core::SavedConversation>(std::move(image)),
+                              base = std::filesystem::path(o.prompt_cache_file), disk_target = std::move(disk_target),
+                              identity = prefix_identity, next, tokens, cap = prefix_multi_cap, save_start, capture_ms,
+                              background = !prefix_write_sync, what, waited]() {
+                if (background) strata::aux_cpus::pin_current_thread();
+                std::string we;
+                const bool stored = strata::core::prefix_cache_write(disk_target, identity, next, *img, we);
+                const size_t evicted = stored && cap > 0 ? strata::core::prefix_cache_evict(base, cap, disk_target) : 0;
+                std::string tail;
+                char buf[200];
+                if (background) {
+                    std::snprintf(buf, sizeof buf, " (on a thread; %s %.1f ms for the capture)", waited, capture_ms);
+                    tail += buf;
+                }
+                if (cap > 0) {
+                    std::snprintf(buf, sizeof buf, " -> %s, %zu older removed", disk_target.filename().string().c_str(),
+                                  evicted);
+                    tail += buf;
+                }
+                std::fprintf(stderr, "strata serve: %s: %s %lld tokens, %zu bytes in %.1f ms%s%s%s\n", what,
+                             stored ? "saved" : "skip saving", tokens, img->bytes(),
+                             std::chrono::duration<double, std::milli>(Clock::now() - save_start).count(),
+                             stored ? "" : ": ", stored ? "" : we.c_str(), tail.c_str());
+            };
+            if (prefix_write_sync) {
+                write_job();
+            } else {
+                if (prefix_write_job.valid()) prefix_write_job.wait();
+                prefix_write_job = std::async(std::launch::async, std::move(write_job));
+            }
+        };
+        // fork (TTFT): STRATA_CONV_DISK=<idle seconds> (opt-in, needs --prompt-cache-file and STRATA_PREFIX_DISK_MULTI):
+        // after a request, once the engine has been idle that long, the live conversation at its last turn boundary
+        // (the checkpoint the next turn resumes from) goes to a keyed file like the system turn's; a request looks for
+        // the longest such file among its own turn boundaries.  A conversation survives a restart (or the RAM cache
+        // evicting it) for the cost of reading the file.  STRATA_CONV_DISK_MIN (default 4096): only a boundary at least
+        // that many tokens past the system turn is saved.
+        const double conv_disk_idle_s = [] {
+            const char* v = std::getenv("STRATA_CONV_DISK");
+            return v != nullptr ? std::max(0.0, std::atof(v)) : 0.0;
+        }();
+        const int64_t conv_disk_min = [] {
+            const char* v = std::getenv("STRATA_CONV_DISK_MIN");
+            return v != nullptr ? std::max<int64_t>(1, std::atoll(v)) : int64_t(4096);
+        }();
+        const bool conv_disk_on = conv_disk_idle_s > 0.0 && prefix_disk_enabled && prefix_multi_cap > 0 && o.batch <= 0;
+        if (conv_disk_idle_s > 0.0)
+            std::fprintf(stderr, "strata serve: conversation disk cache: %s\n",
+                         conv_disk_on ? "on (the live conversation at its last turn boundary after the idle time)"
+                                      : "off (needs --prompt-cache-file, STRATA_PREFIX_DISK_MULTI and no --batch)");
+        int64_t conv_pending_len = -1;   // the last request's turn boundary, saved at the next idle
+        int64_t conv_root_len = -1;      // that request's system turn (a saved boundary must be past it)
         strata::core::ConversationCache conversations(
             o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0,
             (size_t) o.conversation_cache_slots, o.conversation_cache_min_tokens);
```

### 16-7

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                if (!pump(false)) return false;
            return true;
        };
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 111 lines):
```diff
                 if (!pump(false)) return false;
             return true;
         };
+        // STRATA_CONV_DISK: the next line, or false with `idle` set after `ms` without one
+        auto next_line_for = [&](std::string& out, int64_t ms, bool& idle) -> bool {
+            idle = false;
+            std::unique_lock<std::mutex> lk(in_mu);
+            if (!in_cv.wait_for(lk, std::chrono::milliseconds(ms), [&] { return !in_lines.empty() || in_eof; })) {
+                idle = true;
+                return false;
+            }
+            if (in_lines.empty()) return false;
+            out = std::move(in_lines.front());
+            in_lines.pop_front();
+            return true;
+        };
+        // STRATA_CONV_DISK: the live conversation at the last request's turn boundary B to a keyed file.  The image is
+        // the one the system turn's save takes, built from what the next turn would resume from: the K/V cells [0, B)
+        // of the sessions (the cells past B were written after them and are not taken) and the running state of the
+        // checkpoint at B, stage by stage.  Returns false only for a CUDA failure (the engine stops, as for a request).
+        auto conv_idle_save = [&]() -> bool {
+            const int64_t B = conv_pending_len;
+            conv_pending_len = -1;
+            if (!live_ok || B < 1 || B > int64_t(strata::core::prefix_cache_max_tokens) || B - std::max<int64_t>(conv_root_len, 0) < conv_disk_min)
... 87 more lines: see 16-conv-disk.patch
```

### 16-8

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            } else if (!next_line(line)) {
                break;
            }
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 14 lines):
```diff
                         std::fflush(stderr);
                     }
                 }
+            } else if (conv_disk_on && conv_pending_len > 0) {   // STRATA_CONV_DISK: the idle save comes first
+                bool idle = false;
+                if (!next_line_for(line, (int64_t) (conv_disk_idle_s * 1000.0), idle)) {
+                    if (!idle) break;
+                    if (!conv_idle_save()) return 1;
+                    std::fflush(stdout);
+                    continue;
+                }
             } else if (!next_line(line)) {
                 break;
             }
```

### 16-9

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                int64_t disk_root = -1, last_turn = -1;
                for (int64_t i = n - 1; i > 0; --i) if (ids[size_t(i)] == o.turn_token) { last_turn = i; break; }
                for (int64_t i = 1; i < last_turn; ++i) if (ids[size_t(i)] == o.turn_token) { disk_root = i; break; }
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 45 lines):
```diff
                 int64_t disk_root = -1, last_turn = -1;
                 for (int64_t i = n - 1; i > 0; --i) if (ids[size_t(i)] == o.turn_token) { last_turn = i; break; }
                 for (int64_t i = 1; i < last_turn; ++i) if (ids[size_t(i)] == o.turn_token) { disk_root = i; break; }
-                if (disk_root >= o.prompt_cache_root && disk_root <= int64_t(strata::core::prefix_cache_max_tokens) &&
-                        disk_root > std::max(resume, (incoming || borrow) ? incoming_tokens : int64_t(0))) {
+                // STRATA_CONV_DISK: the longest saved conversation among this prompt's turn boundaries past what RAM
+                // and the system turn already give (one pass for the keys, one directory listing for the names)
+                const int64_t have = std::max(resume, (incoming || borrow) ? incoming_tokens : int64_t(0));
+                int64_t conv_at = -1;
+                std::filesystem::path conv_path;
+                if (conv_disk_on && last_turn > 0) {
+                    std::vector<int64_t> at;
+                    const int64_t lo = std::max(have, disk_root);
+                    for (int64_t i = lo + 1; i <= last_turn && i <= int64_t(strata::core::prefix_cache_max_tokens); ++i)
+                        if (ids[size_t(i)] == o.turn_token) at.push_back(i);
+                    if (!at.empty()) {
+                        const std::vector<int32_t> all(ids.begin(), ids.begin() + (at.back() + 1));
+                        const auto keys = strata::core::prefix_cache_keyed_paths(o.prompt_cache_file, all, at);
+                        std::set<std::string> on_disk;
+                        for (const auto& e : strata::core::prefix_cache_keyed_files(o.prompt_cache_file))
+                            on_disk.insert(e.path.filename().string());
+                        for (size_t k = keys.size(); k-- > 0;)
+                            if (on_disk.count(keys[k].filename().string())) { conv_at = at[k]; conv_path = keys[k]; break; }
+                    }
+                }
+                const int64_t disk_at = conv_at > 0 ? conv_at : disk_root;
+                if ((conv_at > 0 || disk_root >= o.prompt_cache_root) && disk_at <= int64_t(strata::core::prefix_cache_max_tokens) &&
+                        disk_at > have) {
                     const auto t0 = Clock::now();
                     if (prefix_write_job.valid()) prefix_write_job.wait();   // a save in flight: this read sees it
-                    std::vector<int32_t> prefix(ids.begin(), ids.begin() + disk_root);
+                    std::vector<int32_t> prefix(ids.begin(), ids.begin() + disk_at);
                     // STRATA_PREFIX_DISK_MULTI: this prefix's own file when there is one, else the single file
                     std::filesystem::path disk_path = o.prompt_cache_file;
-                    if (prefix_multi_cap > 0) {
+                    if (conv_at > 0) {
+                        disk_path = conv_path;
+                    } else if (prefix_multi_cap > 0) {
                         std::error_code kec;
                         auto keyed = strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix,
-                                                                           int32_t(ids[size_t(disk_root)]));
+                                                                           int32_t(ids[size_t(disk_at)]));
                         if (std::filesystem::is_regular_file(keyed, kec) && !kec) {
                             disk_path = std::move(keyed);
                             // fork (TTFT): this prefix's own file (a hit unless the engine's identity changed): the
```

### 16-10

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                       static_cast<const strata::core::QsaState*>(nullptr), err);
                        };
                        if (strata::core::prefix_cache_read(disk_path, prefix_identity, prefix,
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 24 lines):
```diff
                                        static_cast<const strata::core::QsaState*>(nullptr), err);
                         };
                         if (strata::core::prefix_cache_read(disk_path, prefix_identity, prefix,
-                                int32_t(ids[size_t(disk_root)]), want_cvec, image, err) &&
+                                int32_t(ids[size_t(disk_at)]), want_cvec, image, err) &&
                                 disk_image_valid(image)) {
                             if (prefix_multi_cap > 0 && disk_path != o.prompt_cache_file)
                                 strata::core::prefix_cache_touch(disk_path);   // the LRU order is the order of use
-                            incoming.emplace(std::move(image)); incoming_tokens = disk_root;
+                            if (conv_at > 0)
+                                std::fprintf(stderr, "strata serve: conversation disk cache: hit %lld tokens (%s)\n",
+                                             (long long) conv_at, disk_path.filename().string().c_str());
+                            incoming.emplace(std::move(image)); incoming_tokens = disk_at;
                             incoming_live = true; incoming_disk = true;
                             borrow = false;   // the disk root is the longer match: a donor stays parked, untouched
                             slot_source = -1;   // batch slots: the disk root is the longer match
                         } else {
-                            std::fprintf(stderr, "strata serve: prefix disk cache: miss (%s)\n", err.c_str()); err.clear();
+                            std::fprintf(stderr, "strata serve: %s: miss (%s)\n",
+                                         conv_at > 0 ? "conversation disk cache" : "prefix disk cache", err.c_str());
+                            err.clear();
                         }
                     } else {
                         std::fprintf(stderr, "strata serve: prefix disk cache: miss (%s)\n",
```

### 16-11

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    std::printf("ERR saving a conversation checkpoint failed%s\n", ckpt_why.c_str());
                    return 1;
                }
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 11 lines):
```diff
                     std::printf("ERR saving a conversation checkpoint failed%s\n", ckpt_why.c_str());
                     return 1;
                 }
+                if (conv_disk_on && to == turn_at && req_imgs.empty()) {   // STRATA_CONV_DISK: saved at the next idle
+                    conv_pending_len = to;
+                    conv_root_len = -1;
+                    for (int64_t i = 1; i < to; ++i) if (ids[size_t(i)] == o.turn_token) { conv_root_len = i; break; }
+                }
                 if (to == pin_at && !pin_checkpoint(pin_at))
                     std::fprintf(stderr, "strata serve: pin=%lld: its checkpoint was not kept\n", (long long) pin_at);
                 if (trace && to == message_at)
```

### 16-12

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            const std::filesystem::path disk_target = prefix_multi_cap > 0
                                ? strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix, int32_t(ids[size_t(to)]))
                                : std::filesystem::path(o.prompt_cache_file);
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 40 lines):
```diff
                             const std::filesystem::path disk_target = prefix_multi_cap > 0
                                 ? strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix, int32_t(ids[size_t(to)]))
                                 : std::filesystem::path(o.prompt_cache_file);
-                            auto write_job = [img = std::make_shared<strata::core::SavedConversation>(std::move(image)),
-                                              base = std::filesystem::path(o.prompt_cache_file), disk_target,
-                                              identity = prefix_identity, next = int32_t(ids[size_t(to)]),
-                                              tokens = (long long) to, cap = prefix_multi_cap, save_start, capture_ms,
-                                              background = !prefix_write_sync]() {
-                                if (background) strata::aux_cpus::pin_current_thread();
-                                std::string we;
-                                const bool stored = strata::core::prefix_cache_write(disk_target, identity, next, *img, we);
-                                const size_t evicted = stored && cap > 0 ? strata::core::prefix_cache_evict(base, cap, disk_target) : 0;
-                                std::string tail;
-                                char buf[160];
-                                if (background) {
-                                    std::snprintf(buf, sizeof buf, " (on a thread; the request waited %.1f ms for the capture)",
-                                                  capture_ms);
-                                    tail += buf;
-                                }
-                                if (cap > 0) {
-                                    std::snprintf(buf, sizeof buf, " -> %s, %zu older removed",
-                                                  disk_target.filename().string().c_str(), evicted);
-                                    tail += buf;
-                                }
-                                std::fprintf(stderr, "strata serve: prefix disk cache: %s %lld tokens, %zu bytes in %.1f ms%s%s%s\n",
-                                             stored ? "saved" : "skip saving", tokens, img->bytes(),
-                                             std::chrono::duration<double, std::milli>(Clock::now() - save_start).count(),
-                                             stored ? "" : ": ", stored ? "" : we.c_str(), tail.c_str());
-                            };
-                            if (prefix_write_sync) {
-                                write_job();
-                            } else {
-                                if (prefix_write_job.valid()) prefix_write_job.wait();
-                                prefix_write_job = std::async(std::launch::async, std::move(write_job));
-                            }
+                            store_prefix(std::move(image), disk_target, int32_t(ids[size_t(to)]), (long long) to, save_start,
+                                         capture_ms, "prefix disk cache", "the request waited");
                         } else {
                             std::fprintf(stderr, "strata serve: prefix disk cache: skip saving (size, geometry or physical RAM admission)\n");
                         }
```

## 17 p2-flush-noswitch

`17-p2-flush-noswitch.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il polling della pipeline senza cambio di scheda (STRATA_P2_FLUSH_NOSWITCH, STRATA_P2_SWITCH_STAT, STRATA_P2_SMEM, STRATA_P2_FLUSH_US)

### 17-1

`docs/PIPELINE2.md` · (new file)

New file, 727 lines: see the patch.

### 17-2

`include/strata/core/on_device.hpp` · (file scope)

Anchor:
```

#include <cuda_runtime.h>
```
Block (hunk 1 of `include/strata/core/on_device.hpp` in the patch, 100 lines):
```diff

 #include <cuda_runtime.h>

+#include <atomic>
+#include <chrono>
+#include <cstdlib>
+
 namespace strata::core {

+// lab/pipeline2 tappa 0b (docs/PIPELINE2.md): STRATA_P2_SWITCH_STAT=1 counts the device switches of OnDevice and the
+// host time spent in them (cudaGetDevice + cudaSetDevice there and back); STRATA_P2_FLUSH_NOSWITCH=1 lets the pipelined
+// polls (Verifier::service's 2 ms flush, Verifier::done, MtpDrafter::chain_outputs_ready / chain_poll) query their
+// events and streams without switching (port of the NTRYourWaifu fork's STRATA_FLUSH_NO_SWITCH, two-pc-042 v7/v8).
+// Both off by default; neither changes any GPU work.
+inline bool p2_env(const char* name) {
+    const char* v = std::getenv(name);
+    return v != nullptr && std::atoi(v) != 0;
+}
+inline bool p2_switch_stat() { static const bool on = p2_env("STRATA_P2_SWITCH_STAT"); return on; }
+inline bool p2_flush_noswitch() { static const bool on = p2_env("STRATA_P2_FLUSH_NOSWITCH"); return on; }
+// lab/pipeline2 (docs/PIPELINE2.md, tappa 1): STRATA_P2_FLUSH_US=<us> sets the period of the WDDM flush
+// (cudaStreamQuery) in the pipelined polls (Verifier::service, done, pl_head_done); unset = 2000 us as before.  nsys
+// shows the 2080 Ti's stage-1 graph idle ~2 ms in most windows between gr_down_staged and gr_up_multi of its first layer,
+// until that flush.  No GPU work changes.
... 76 more lines: see 17-p2-flush-noswitch.patch
```

### 17-3

`src/core/mtp.cpp` · void MtpDrafter::chain_cut() {

Anchor:
```

int MtpDrafter::chain_outputs_ready(std::string& err) {
    if (!chain_live_) return chain_n_;
```
Block (hunk 1 of `src/core/mtp.cpp` in the patch, 19 lines):
```diff

 int MtpDrafter::chain_outputs_ready(std::string& err) {
     if (!chain_live_) return chain_n_;
-    const OnDevice on_device(device_);
+    // lab/pipeline2 0b (STRATA_P2_FLUSH_NOSWITCH=1): only event queries and mapped reads here: no device switch
+    const bool nosw = p2_flush_noswitch() && !p2_noswitch_refused();
+    if (nosw && p2_switch_stat()) p2_switch().noswitch.fetch_add(1, std::memory_order_relaxed);
+    const OnDevice on_device(nosw ? -1 : device_);
     while (steps_seen_ < std::min(chain_early_, chain_launched_)) {
-        const cudaError_t q = cudaEventQuery(ev_step_[steps_seen_]);
+        cudaError_t q = cudaEventQuery(ev_step_[steps_seen_]);
+        if (nosw && q != cudaSuccess && q != cudaErrorNotReady) {   // refused without the device: ask again under it
+            p2_noswitch_refuse(q);
+            const OnDevice again(device_);
+            q = cudaEventQuery(ev_step_[steps_seen_]);
+        }
         if (q == cudaErrorNotReady) break;
         if (q != cudaSuccess) { err = std::string("mtp chain: ") + cudaGetErrorString(q); return -1; }
         chain_tok_[steps_seen_] = ((volatile int32_t*) h_out_)[steps_seen_];
```

### 17-4

`src/core/mtp.cpp` · int MtpDrafter::chain_outputs_ready(std::string& err) {

Anchor:
```
int MtpDrafter::chain_poll(std::string& err) {
    if (!chain_live_) return 1;
    if (chain_launched_ < chain_n_) return 0;   // STRUCT-A: not whole yet (chain_extend or chain_cut first)
```
Block (hunk 2 of `src/core/mtp.cpp` in the patch, 17 lines):
```diff
 int MtpDrafter::chain_poll(std::string& err) {
     if (!chain_live_) return 1;
     if (chain_launched_ < chain_n_) return 0;   // STRUCT-A: not whole yet (chain_extend or chain_cut first)
-    const OnDevice on_device(device_);
-    const cudaError_t q = cudaEventQuery(ev_chain_);
+    const bool nosw = p2_flush_noswitch() && !p2_noswitch_refused();   // lab/pipeline2 0b, as chain_outputs_ready
+    if (nosw && p2_switch_stat()) p2_switch().noswitch.fetch_add(1, std::memory_order_relaxed);
+    const OnDevice on_device(nosw ? -1 : device_);
+    cudaError_t q = cudaEventQuery(ev_chain_);
+    if (nosw && q != cudaSuccess && q != cudaErrorNotReady) {   // refused without the device: ask again under it
+        p2_noswitch_refuse(q);
+        const OnDevice again(device_);
+        q = cudaEventQuery(ev_chain_);
+    }
     if (q == cudaErrorNotReady) return 0;
     chain_live_ = false;
     if (q != cudaSuccess) { err = std::string("mtp chain: ") + cudaGetErrorString(q); return -1; }
```

### 17-5

`src/core/verify.cpp` · bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&

Anchor:
```
    }
    release_gpu_fn().store(&release_live_verifiers);
    cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
```
Block (hunk 1 of `src/core/verify.cpp` in the patch, 18 lines):
```diff
     }
     release_gpu_fn().store(&release_live_verifiers);
     cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
+    {   // lab/pipeline2 (docs/PIPELINE2.md, tappa 2a): STRATA_P2_SMEM=1 asks every kernel of this card for the shared
+        // side of the L1/shared split, before any graph is captured.  On the 2080 Ti the hyper-connection read flips the
+        // SM between 64 KiB (gr_down_staged, carveout 100) and 32 KiB (the rest) twice a layer, and nsys shows 11 us of
+        // idle before each gr_up_multi (65 a window).  A cache split moves no value: bitwise.
+        static const bool p2_smem = std::getenv("STRATA_P2_SMEM") != nullptr && std::atoi(std::getenv("STRATA_P2_SMEM")) != 0;
+        if (p2_smem) {
+            const cudaError_t ce = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
+            std::fprintf(stderr, "strata pipeline2: STRATA_P2_SMEM: CUDA%d prefers shared memory (%s)\n", device_,
+                         ce == cudaSuccess ? "ok" : cudaGetErrorString(ce));
+            (void) cudaGetLastError();
+        }
+    }
     rr_stats_ = route_resident_cfg().stats_on(device_);   // STRATA_ROUTE_RESIDENT: this GPU's counters, outside graph capture
     strata::kernels::fused_gr_check();   // once per card: which bitwise-equal hyper-connection read runs there
     strata::kernels::fused_gr_persist_init();   // opt-in STRATA_HC_PERSIST=1 (HIP)
```

### 17-6

`src/core/verify.cpp` · namespace {

Anchor:
```
double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
}  // namespace
```
Block (hunk 2 of `src/core/verify.cpp` in the patch, 15 lines):
```diff
 double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
 }  // namespace

+// lab/pipeline2 0b (on_device.hpp): a poll without the device switch was refused: switch again from now on
+void p2_noswitch_refuse(cudaError_t e) {
+    if (!p2_noswitch_off().exchange(true))
+        std::fprintf(stderr, "strata pipeline2: the poll without the device switch was refused (%s): switching again\n",
+                     cudaGetErrorString(e));
+    p2_switch().refused.fetch_add(1, std::memory_order_relaxed);
+    (void) cudaGetLastError();
+}
+
 void Verifier::diag_pipelined(std::FILE* f, const char* name) const {
     auto rd = [](const uint32_t* p) { return p ? *(const volatile uint32_t*) p : 0u; };
     auto ev = [](cudaEvent_t e) {
```

### 17-7

`src/core/verify.cpp` · bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string

Anchor:
```
    std::atomic_thread_fence(std::memory_order_seq_cst);
    trace_ev("WINDOW (pipelined)", -1, -1, pos0 * 16 + T);
    ms_host += ms_since(t0);
```
Block (hunk 3 of `src/core/verify.cpp` in the patch, 14 lines):
```diff
     std::atomic_thread_fence(std::memory_order_seq_cst);
     trace_ev("WINDOW (pipelined)", -1, -1, pos0 * 16 + T);
     ms_host += ms_since(t0);
+    const auto p2_t0 = std::chrono::steady_clock::now();
     const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);
     trace_ev("LAUNCHED", -1, -1, (int64_t) le);
     if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
     if (fl_prof_) cudaMemcpyAsync(prof_pin_, prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost, cs_);
     if (cudaEventRecord(ev_done_, cs_) != cudaSuccess) { err = "verify: event record failed"; return false; }
     (void) cudaStreamQuery(cs_);   // WDDM: submit now
+    if (p2_switch_stat()) p2_launch_note(device_, p2_t0);   // lab/pipeline2 tappa 1: launch + WDDM submit
     fl_active_ = true;
     fl_since_ms_ = fl_flush_ms_ = fl_launch_ms_ = now_ms();
     return true;
```

### 17-8

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la

Anchor:
```
    // on it (0.1.41, Windows, an RTX 3060 + RTX 5070 Ti split).  The same answer as the loop below gives.
    if (!ar_on() && *(volatile uint32_t*) h_seq_ < (uint32_t) (fl_k_ + 1)) {
        const double now = now_ms();
```
Block (hunk 4 of `src/core/verify.cpp` in the patch, 26 lines):
```diff
     // on it (0.1.41, Windows, an RTX 3060 + RTX 5070 Ti split).  The same answer as the loop below gives.
     if (!ar_on() && *(volatile uint32_t*) h_seq_ < (uint32_t) (fl_k_ + 1)) {
         const double now = now_ms();
-        if (now - fl_flush_ms_ <= 2.0 && now - fl_since_ms_ <= 20000.0) return 0;
+        if (now - fl_flush_ms_ <= p2_flush_ms() && now - fl_since_ms_ <= 20000.0) return 0;
+        // lab/pipeline2 0b (STRATA_P2_FLUSH_NOSWITCH=1): the due flush without the device switch while the graph still
+        // runs - cudaEventQuery works on any device's event, the stream carries its own context (NTRYourWaifu v7)
+        if (now - fl_since_ms_ <= 20000.0 && p2_flush_noswitch() && !p2_noswitch_refused()) {
+            const double was = fl_flush_ms_;
+            fl_flush_ms_ = now;
+            const cudaError_t q = cudaEventQuery(sa_head_ ? ev_head_ : ev_done_);
+            if (q == cudaErrorNotReady) {
+                const cudaError_t s = cudaStreamQuery(cs_);
+                if (s == cudaSuccess || s == cudaErrorNotReady) {
+                    if (p2_switch_stat()) p2_switch().noswitch.fetch_add(1, std::memory_order_relaxed);
+                    return 0;
+                }
+                p2_noswitch_refuse(s);
+            } else if (q != cudaSuccess) {
+                p2_noswitch_refuse(q);
+            }
+            fl_flush_ms_ = was;   // finished, failed or refused: the loop below asks again under the device
+        }
     }
     const OnDevice on_device(device_);
     const ModelGeometry& g = *g_;
```

### 17-9

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la

Anchor:
```
        const uint32_t want = (uint32_t) (fl_k_ + 1);
        if (*(volatile uint32_t*) h_seq_ < want) {
            const double now = now_ms();
```
Block (hunk 5 of `src/core/verify.cpp` in the patch, 8 lines):
```diff
         const uint32_t want = (uint32_t) (fl_k_ + 1);
         if (*(volatile uint32_t*) h_seq_ < want) {
             const double now = now_ms();
-            if (now - fl_flush_ms_ > 2.0) {   // flush WDDM and notice a dead graph, as run() does
+            if (now - fl_flush_ms_ > p2_flush_ms()) {   // flush WDDM and notice a dead graph, as run() does
                 fl_flush_ms_ = now;
                 const cudaError_t q = cudaEventQuery(sa_head_ ? ev_head_ : ev_done_);   // STRUCT-A: a head bet has no done event yet
                 if (q != cudaErrorNotReady && *(volatile uint32_t*) h_seq_ < want) {
```

### 17-10

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la

Anchor:
```

bool Verifier::done(std::string& err) {
    if (!fl_active_ || fl_k_ < fl_total_) return false;
```
Block (hunk 6 of `src/core/verify.cpp` in the patch, 29 lines):
```diff

 bool Verifier::done(std::string& err) {
     if (!fl_active_ || fl_k_ < fl_total_) return false;
+    if (p2_flush_noswitch() && !p2_noswitch_refused()) {   // lab/pipeline2 0b: the poll while the tail runs, no switch
+        const cudaError_t q = cudaEventQuery(ev_done_);
+        if (q == cudaErrorNotReady) {
+            const double now = now_ms();
+            if (now - fl_since_ms_ <= 20000.0) {
+                if (now - fl_flush_ms_ > p2_flush_ms()) {
+                    fl_flush_ms_ = now;
+                    const cudaError_t s = cudaStreamQuery(cs_);
+                    if (s != cudaSuccess && s != cudaErrorNotReady) p2_noswitch_refuse(s);
+                }
+                if (p2_switch_stat()) p2_switch().noswitch.fetch_add(1, std::memory_order_relaxed);
+                return false;
+            }
+        } else if (q != cudaSuccess) {
+            p2_noswitch_refuse(q);
+        }
+    }
     const OnDevice on_device(device_);
     const cudaError_t q = cudaEventQuery(ev_done_);
     if (q == cudaErrorNotReady) {
         const double now = now_ms();
-        if (now - fl_flush_ms_ > 2.0) { fl_flush_ms_ = now; (void) cudaStreamQuery(cs_); }
+        if (now - fl_flush_ms_ > p2_flush_ms()) { fl_flush_ms_ = now; (void) cudaStreamQuery(cs_); }
         if (now - fl_since_ms_ > 20000.0) {   // every layer served, and the graph still runs: #267 as above
             err = "verify: the window never finished" + released_note(release_gpu_waits(5000));
         }
```

### 17-11

`src/core/verify.cpp` · bool Verifier::pl_launch_head(int T, const int32_t* tokens, int64_t pos0, std::s

Anchor:
```
        err = "verify: saving the PLE history failed";
        return false;
    }
```
Block (hunk 7 of `src/core/verify.cpp` in the patch, 12 lines):
```diff
         err = "verify: saving the PLE history failed";
         return false;
     }
+    const auto p2_t0 = std::chrono::steady_clock::now();
     const cudaError_t le = cudaGraphLaunch(exec_head_[T], cs_);
     if (le != cudaSuccess) { err = std::string("verify: head launch: ") + cudaGetErrorString(le); return false; }
     if (cudaEventRecord(ev_head_, cs_) != cudaSuccess) { err = "verify: event record failed"; return false; }
     (void) cudaStreamQuery(cs_);   // WDDM: submit now
+    if (p2_switch_stat()) p2_launch_note(device_, p2_t0);   // lab/pipeline2 tappa 1: launch + WDDM submit
     fl_active_ = true;
     sa_head_ = true;
     fl_since_ms_ = fl_flush_ms_ = fl_launch_ms_ = now_ms();
```

### 17-12

`src/core/verify.cpp` · bool Verifier::pl_head_done(std::string& err) {

Anchor:
```
    const cudaError_t q = cudaEventQuery(ev_head_);
    if (q == cudaErrorNotReady) {
        const double now = now_ms();
```
Block (hunk 8 of `src/core/verify.cpp` in the patch, 8 lines):
```diff
     const cudaError_t q = cudaEventQuery(ev_head_);
     if (q == cudaErrorNotReady) {
         const double now = now_ms();
-        if (now - fl_flush_ms_ > 2.0) { fl_flush_ms_ = now; (void) cudaStreamQuery(cs_); }
+        if (now - fl_flush_ms_ > p2_flush_ms()) { fl_flush_ms_ = now; (void) cudaStreamQuery(cs_); }
         return false;
     }
     if (q != cudaSuccess) { err = std::string("verify: head: ") + cudaGetErrorString(q); return false; }
```

### 17-13

`src/core/verify.cpp` · bool Verifier::pl_launch_tail(std::string& err) {

Anchor:
```
    fl_cap_ = 0;
    sa_head_ = false;
    fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
```
Block (hunk 9 of `src/core/verify.cpp` in the patch, 14 lines):
```diff
     fl_cap_ = 0;
     sa_head_ = false;
     fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
+    const auto p2_t0 = std::chrono::steady_clock::now();
     const cudaError_t le = cudaGraphLaunch(exec_tail_[T], cs_);
     trace_ev("LAUNCHED TAIL", -1, -1, (int64_t) le);
     if (le != cudaSuccess) { err = std::string("verify: tail launch: ") + cudaGetErrorString(le); return false; }
     if (fl_prof_) cudaMemcpyAsync(prof_pin_, prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost, cs_);
     if (cudaEventRecord(ev_done_, cs_) != cudaSuccess) { err = "verify: event record failed"; return false; }
     (void) cudaStreamQuery(cs_);
+    if (p2_switch_stat()) p2_launch_note(device_, p2_t0);   // lab/pipeline2 tappa 1: launch + WDDM submit
     fl_since_ms_ = fl_flush_ms_ = now_ms();   // GPU-reach wait counts from here, as from a launch
     return true;
 }
```

### 17-14

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    return v != nullptr ? std::atoi(v) != 0 : pipe_elastic_env();
                }();
                const Clock::time_point pl_t0 = Clock::now();
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 11 lines):
```diff
                     return v != nullptr ? std::atoi(v) != 0 : pipe_elastic_env();
                 }();
                 const Clock::time_point pl_t0 = Clock::now();
+                if (strata::core::p2_switch_stat()) {   // lab/pipeline2 0b: count this decode's switches only
+                    auto& s0 = strata::core::p2_switch();
+                    s0.calls = 0; s0.ns = 0; s0.max_ns = 0; s0.slow = 0; s0.noswitch = 0; s0.refused = 0;
+                    for (int k = 0; k < 2; ++k) { s0.launches[k] = 0; s0.launch_ns[k] = 0; s0.launch_max_ns[k] = 0; }
+                }
                 for (int st = 0; st < 2; ++st)
                     for (int par = 0; par < 2; ++par) g_pl_diag.v[st][par] = PV[st][par];
                 strata::core::diag_pipeline_fn().store(&pl_diag_print);
```

### 17-15

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                     (long long) sa_placed[2], (long long) sa_tails[2], (long long) sa_drops[2],
                                     (long long) sa_placed[3], (long long) sa_tails[3], (long long) sa_drops[3],
                                     (long long) sa_chains, (long long) sa_cut);
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 25 lines):
```diff
                                      (long long) sa_placed[2], (long long) sa_tails[2], (long long) sa_drops[2],
                                      (long long) sa_placed[3], (long long) sa_tails[3], (long long) sa_drops[3],
                                      (long long) sa_chains, (long long) sa_cut);
+                    if (strata::core::p2_switch_stat()) {   // lab/pipeline2 0b: the host's device switches this request
+                        auto& s = strata::core::p2_switch();
+                        const long long c = s.calls.exchange(0), ns = s.ns.exchange(0), mx = s.max_ns.exchange(0),
+                                        sl = s.slow.exchange(0), nw = s.noswitch.exchange(0), rf = s.refused.exchange(0);
+                        const double nw_ = dec_windows > 0 ? (double) dec_windows : 1.0;
+                        std::fprintf(stderr, "strata pipeline2 switches: %lld device switches (%.1f per window), %.3f ms "
+                                             "host in them per window, longest %.3f ms, %lld over 0.1 ms; polls without "
+                                             "a switch %lld (%s), refused %lld\n",
+                                     c, c / nw_, ns / 1e6 / nw_, mx / 1e6, sl, nw,
+                                     strata::core::p2_flush_noswitch() ? "STRATA_P2_FLUSH_NOSWITCH=1" : "off", rf);
+                        for (int k = 0; k < 2; ++k) {   // tappa 1: host time in the window graphs' cudaGraphLaunch
+                            const long long n = s.launches[k].exchange(0), lns = s.launch_ns[k].exchange(0),
+                                            lmx = s.launch_max_ns[k].exchange(0);
+                            if (n > 0)
+                                std::fprintf(stderr, "strata pipeline2 launches: CUDA%d %lld window graph launches, %.3f ms "
+                                                     "host each in launch + submit (%.3f ms per window), longest %.3f ms\n",
+                                             k, n, lns / 1e6 / (double) n, lns / 1e6 / nw_, lmx / 1e6);
+                        }
+                    }
                     std::string cal;
                     char b[48];
                     for (int i = 0; i < 10; ++i)
```

## 18 hc16-sm75

`18-hc16-sm75.patch` — SONNET-STRATA-Porta su upstream 0.1.45 l'hc16 del prefill su sm_75 come leva opt-in (STRATA_HC16_SM75)

### 18-1

`include/strata/core/tc_sm75.hpp` · inline bool tc_sm75_enabled(int cc_major, int cc_minor) {

Anchor:
```
    return v == nullptr || std::atoi(v) != 0;
}
```
Block (hunk 1 of `include/strata/core/tc_sm75.hpp` in the patch, 13 lines):
```diff
     return v == nullptr || std::atoi(v) != 0;
 }

+/// Fork (hc16): the hyper-connection projections and ssm_alpha / ssm_beta of the prompt on FP16 tensor cores, with their
+/// own small staged weight - opt-in, STRATA_HC16_SM75=1, and only on a 7.5 card (STRATA_EMULATE_CC=75 counts).  Not part
+/// of STRATA_SM75_PERF (it is not bitwise).  Beside STRATA_BF16_TC=1 it wins on those call sites, see prefill.cpp.
+inline bool hc16_sm75_enabled(int cc_major, int cc_minor) {
+    if (!strata::is_sm75(cc_major, cc_minor)) return false;
+    const char* v = std::getenv("STRATA_HC16_SM75");
+    return v != nullptr && std::atoi(v) != 0;
+}
+
 }  // namespace strata::core
```

### 18-2

`include/strata/prefill/kernels.hpp` · namespace strata::prefill {

Anchor:
```

// ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
/// xn[t, c*2560 + d] = R[t,c,d] * rsqrt(mean_d R[t,c,:]^2 + eps) * w_norm[c*2560 + d]; also its BF16 image.
```
Block (hunk 1 of `include/strata/prefill/kernels.hpp` in the patch, 17 lines):
```diff

 // ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
 /// xn[t, c*2560 + d] = R[t,c,d] * rsqrt(mean_d R[t,c,:]^2 + eps) * w_norm[c*2560 + d]; also its BF16 image.
+/// `f16img` (CUDA only; the gr_norm* / gr_write_*norm_rs writers and gr_silu): what the 16-bit image holds - 0 BF16,
+/// 1 the FP16 value of the BF16 image (upstream's STRATA_BF16_TC xn_f16), 2 the FP16 value of the FP32 one (fork hc16,
+/// STRATA_HC16_SM75, sm_75: the staged FP16 hyper-connection GEMMs read it).
 /// `xn16_lo` (null: none) takes bf16(x - xn16): W.xn16 + W.xn16_lo is the product with ~16 mantissa bits of x.
 void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
-             uint16_t* xn16_lo = nullptr, bool f16img = false);
+             uint16_t* xn16_lo = nullptr, int f16img = 0);
 /// F-1: gr_norm without its FP32 output: the row scales rs[t*4 + c] and the BF16 image; gr_mix_r then reads R.
 void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
-                uint16_t* xn16_lo = nullptr, int64_t ldx = 0, bool f16img = false);   // ldx: xn16's token stride (0 = 10240)
+                uint16_t* xn16_lo = nullptr, int64_t ldx = 0, int f16img = 0);   // ldx: xn16's token stride (0 = 10240)
 /// gr_mix with xn recomputed from R, rs and w_norm exactly as gr_norm computes it (the same bits).
 void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
               int64_t T, void* stream, uint16_t* mixed_h = nullptr, uint16_t* mixed16_lo = nullptr);
```

### 18-3

`include/strata/prefill/kernels.hpp` · bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const

Anchor:
```
              float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream);
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr,
```
Block (hunk 2 of `include/strata/prefill/kernels.hpp` in the patch, 18 lines):
```diff
               float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream);
 void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                       float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr,
-                      int64_t ldx = 0, bool f16img = false);
+                      int64_t ldx = 0, int f16img = 0);
 /// S23 (STRATA_CVEC_FUSE=1): gr_write, then the control vector (v_l = layer l's direction row, *s_l its scale,
 /// *on the request flag, mode 0 project / 1 add; cvec_kernel's arithmetic), then gr_norm_rs with the next half's
 /// norm - one pass over R, the same bits as the three kernels.
 void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                            const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
-                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr, int64_t ldx = 0, bool f16img = false);
+                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr, int64_t ldx = 0, int f16img = 0);
 /// lo16[t, k] = bf16(silu(lo[t, k] / hc))
-void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo = nullptr);
+void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo = nullptr, int f16img = 0);
 /// The BF16-weight GEMMs' activation image (gr_* kernels, to_bf16) in FP16 instead of BF16, on the current device.
 void set_act_f16(bool on);
 /// STRATA_F16_RANGE: count what the activation images convert to FP16 on this device (hf_sat): the largest |value| and
```

### 18-4

`include/strata/prefill/kernels.hpp` · void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const in

Anchor:
```

/// fp32 -> fp16 bits and fp32 -> bf16, n elements (the two activation images of the prompt GEMMs).
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
```
Block (hunk 3 of `include/strata/prefill/kernels.hpp` in the patch, 9 lines):
```diff

 /// fp32 -> fp16 bits and fp32 -> bf16, n elements (the two activation images of the prompt GEMMs).
 void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
+/// Fork hc16 (sm_75): bf16 bits -> fp16 bits, n elements (a resident BF16 weight re-encoded for the FP16 tensor cores
+/// into a small staging buffer; the value is unchanged wherever FP16 holds it, beyond +-65504 it saturates).
+void hc16_w_to_f16(const uint16_t* x, uint16_t* y, int64_t n, void* stream);
 void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo = nullptr);
 /// y = fp32(fp16(x)): what an FP16 store of x would read back (the FP16 indexer-key experiment)
 void round_f16(const float* x, float* y, int64_t n, void* stream);
```

### 18-5

`src/prefill/kernels.cu` · __global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restr

Anchor:
```
        const float v = r[d] * rs * w[c * N + d];
        xn[row * N + d] = v;
        const uint16_t h = act16(v);
```
Block (hunk 1 of `src/prefill/kernels.cu` in the patch, 8 lines):
```diff
         const float v = r[d] * rs * w[c * N + d];
         xn[row * N + d] = v;
         const uint16_t h = act16(v);
-        xn16[row * N + d] = f16img ? bf_to_hf_conv(h) : h;
+        xn16[row * N + d] = f16img == 2 ? hf_sat(v) : f16img ? bf_to_hf_conv(h) : h;
         if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
     }
 }
```

### 18-6

`src/prefill/kernels.cu` · __global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __re

Anchor:
```
    for (int d = threadIdx.x; d < N; d += blockDim.x) {
        const float v = r[d] * rs * w[c * N + d];
        const uint16_t h = act16(v);
```
Block (hunk 2 of `src/prefill/kernels.cu` in the patch, 8 lines):
```diff
     for (int d = threadIdx.x; d < N; d += blockDim.x) {
         const float v = r[d] * rs * w[c * N + d];
         const uint16_t h = act16(v);
-        xn16[xo + d] = f16img ? bf_to_hf_conv(h) : h;
+        xn16[xo + d] = f16img == 2 ? hf_sat(v) : f16img ? bf_to_hf_conv(h) : h;
         if (xn16_lo) xn16_lo[xo + d] = bf_lo(v, h);
     }
 }
```

### 18-7

`src/prefill/kernels.cu` · __global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict

Anchor:
```
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float x = v[k] * rs * w[c * N + d];
        const uint16_t h = act16(x);
```
Block (hunk 3 of `src/prefill/kernels.cu` in the patch, 8 lines):
```diff
     for (int d = threadIdx.x; d < N; d += 256, ++k) {
         const float x = v[k] * rs * w[c * N + d];
         const uint16_t h = act16(x);
-        xn16[xo + d] = f16img ? bf_to_hf_conv(h) : h;
+        xn16[xo + d] = f16img == 2 ? hf_sat(x) : f16img ? bf_to_hf_conv(h) : h;
         if (xn16_lo) xn16_lo[xo + d] = bf_lo(x, h);
     }
 }
```

### 18-8

`src/prefill/kernels.cu` · __global__ void __launch_bounds__(256) gr_write_cvec_norm_rs_kernel(

Anchor:
```
    for (int d = threadIdx.x; d < N; d += 256, ++k) {
        const float y = x[k] * rs * w[c * N + d];
        const uint16_t h = act16(y);
```
Block (hunk 4 of `src/prefill/kernels.cu` in the patch, 21 lines):
```diff
     for (int d = threadIdx.x; d < N; d += 256, ++k) {
         const float y = x[k] * rs * w[c * N + d];
         const uint16_t h = act16(y);
-        xn16[t * ldx + (int64_t) c * N + d] = f16img ? bf_to_hf_conv(h) : h;
+        xn16[t * ldx + (int64_t) c * N + d] = f16img == 2 ? hf_sat(y) : f16img ? bf_to_hf_conv(h) : h;
         if (xn16_lo) xn16_lo[t * ldx + (int64_t) c * N + d] = bf_lo(y, h);
     }
 }
 __global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, uint16_t* __restrict__ lo16_lo,
-                               int64_t n) {
+                               int64_t n, int f16img) {
     const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
     const float x = lo[i] / (float) HC;
     const float v = x / (1.0f + __expf(-x));
     const uint16_t h = act16(v);
-    lo16[i] = h;
+    lo16[i] = f16img ? hf_sat(v) : h;   // f16img (fork hc16, sm_75): the FP16 image of the FP32 value, not of the BF16 one
     if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
 }
 __global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
```

### 18-9

`src/prefill/kernels.cu` · __global__ void kv_append_kernel(const float* __restrict__ K, const float* __res

Anchor:
```
        if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
    }
}
```
Block (hunk 5 of `src/prefill/kernels.cu` in the patch, 12 lines):
```diff
         if (threadIdx.x == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
     }
 }
+// Fork hc16 (sm_75): a resident BF16 weight re-encoded as FP16 bits (the value is unchanged wherever FP16 holds it;
+// beyond +-65504 it saturates like gemm.cu's own conversion).  One matrix at a time into a small staging buffer.
+__global__ void hc16_w_to_f16_kernel(const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
+    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
+        y[i] = hf_sat(__uint_as_float((uint32_t) x[i] << 16));
+}
 __global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
     for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
 #if defined(__HIPCC__)
```

### 18-10

`src/prefill/kernels.cu` · bool f16_range_read(float& max_abs, unsigned long long& over, bool reset) {

Anchor:
```
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
```
Block (hunk 6 of `src/prefill/kernels.cu` in the patch, 11 lines):
```diff
     return true;
 #endif
 }
+void hc16_w_to_f16(const uint16_t* x, uint16_t* y, int64_t n, void* stream) {
+    if (n <= 0) return;
+    hc16_w_to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
+    check("hc16_w_to_f16");
+}
 void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
     if (n <= 0) return;
     to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0, (cudaStream_t) stream>>>(x, y, n);
```

### 18-11

`src/prefill/kernels.cu` · void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo

Anchor:
```
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
```
Block (hunk 7 of `src/prefill/kernels.cu` in the patch, 18 lines):
```diff
 }

 void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
-             uint16_t* xn16_lo, bool f16img) {
-    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16, xn16_lo, f16img ? 1 : 0);
+             uint16_t* xn16_lo, int f16img) {
+    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16, xn16_lo, f16img);
     check("gr_norm");
 }
 void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
-                uint16_t* xn16_lo, int64_t ldx, bool f16img) {
+                uint16_t* xn16_lo, int64_t ldx, int f16img) {
     gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16, xn16_lo,
-                                                                                ldx > 0 ? ldx : D, f16img ? 1 : 0);
+                                                                                ldx > 0 ? ldx : D, f16img);
     check("gr_norm_rs");
 }
 void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
```

### 18-12

`src/prefill/kernels.cu` · bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const

Anchor:
```
#endif
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
```
Block (hunk 8 of `src/prefill/kernels.cu` in the patch, 27 lines):
```diff
 #endif
 }
 void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
-                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, bool f16img) {
+                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, int f16img) {
     gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm_next, eps,
                                                                                       rs, xn16, xn16_lo,
-                                                                                      ldx > 0 ? ldx : D, f16img ? 1 : 0);
+                                                                                      ldx > 0 ? ldx : D, f16img);
     check("gr_write_norm_rs");
 }
 void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                            const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
-                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, bool f16img) {
+                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, int f16img) {
     gr_write_cvec_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(
-        R, bo, inj, inj_ld, v_l, s_l, on, mode, w_norm_next, eps, rs, xn16, xn16_lo, ldx > 0 ? ldx : D, f16img ? 1 : 0);
+        R, bo, inj, inj_ld, v_l, s_l, on, mode, w_norm_next, eps, rs, xn16, xn16_lo, ldx > 0 ? ldx : D, f16img);
     check("gr_write_cvec_norm_rs");
 }
-void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
-    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, lo16_lo, T * LR);
+void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo, int f16img) {
+    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, lo16_lo, T * LR, f16img);
     check("gr_silu");
 }
 void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
```

### 18-13

`src/prefill/prefill.cpp` · (file scope)

Anchor:
```
#include "strata/platform/aux_cpus.hpp"

#include "strata/core/layout.hpp"
```
Block (hunk 1 of `src/prefill/prefill.cpp` in the patch, 7 lines):
```diff
 #include "strata/platform/aux_cpus.hpp"

 #include "strata/core/layout.hpp"
+#include "strata/core/tc_sm75.hpp"
 #include "strata/kernels/cpu/expert.hpp"
 #include "strata/kernels/native_qsa.hpp"
 #include "strata/kernels/cpu/native_expert.hpp"
```

### 18-14

`src/prefill/prefill.cpp` · inline bool hcd_exact() {

Anchor:
```
    return v;
}
inline bool hc_pad() { return pf_hcdown() || hcd_exact(); }
```
Block (hunk 2 of `src/prefill/prefill.cpp` in the patch, 31 lines):
```diff
     return v;
 }
 inline bool hc_pad() { return pf_hcdown() || hcd_exact(); }
+
+/// Fork hc16 (STRATA_HC16_SM75=1, sm_75 only): is the FP16 tensor-core path of the hyper-connection projections (down, up,
+/// inject) and of ssm_alpha / ssm_beta on for the device this prompt runs on?  Cached per thread and device: a mixed-device
+/// layer split must not inherit the first card's capability.
+bool hc16_dev() {
+    int dev = 0;
+    if (cudaGetDevice(&dev) != cudaSuccess) { cudaGetLastError(); return false; }
+    static thread_local int cached_device = -1;
+    static thread_local bool on = false;
+    if (cached_device != dev) {
+        int major = 0, minor = 0;
+        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
+            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
+            cudaGetLastError();
+            return false;
+        }
+        on = core::hc16_sm75_enabled(major, minor);
+        cached_device = dev;
+    }
+    return on;
+}
+/// hc16 owns a staging buffer for one re-encoded weight (the largest is hc_down: 4 * 2560 * 320 elements, 6.55 MB).
+/// BF16X2's low parts (an explicit accuracy option) keep the BF16 path.  `carve` and `bytes_needed_impl` both ask this.
+inline bool hc16_alloc(bool f16_io) { return hc16_dev() && !bf16x2_hc(f16_io); }
+constexpr int64_t HC16_W_ELEMS = HC * N * LR;
 inline bool cvec_fuse() {
     static const bool v = [] { const char* e = std::getenv("STRATA_CVEC_FUSE"); return e && e[0] == '1'; }();
     return v;
```

### 18-15

`src/prefill/prefill.cpp` · struct Prefill::Impl {

Anchor:
```
    uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
    bool f16_io = false;   // HIP, STRATA_HIP_PROMPT_F16=1: this stage's 16-bit GEMMs run FP16 in and out (set in init)
    bool xn_f16 = false;   // Turing, BF16 on FP16 tensor cores: xn16 written as the FP16 values the GEMM would convert
```
Block (hunk 3 of `src/prefill/prefill.cpp` in the patch, 7 lines):
```diff
     uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
     bool f16_io = false;   // HIP, STRATA_HIP_PROMPT_F16=1: this stage's 16-bit GEMMs run FP16 in and out (set in init)
     bool xn_f16 = false;   // Turing, BF16 on FP16 tensor cores: xn16 written as the FP16 values the GEMM would convert
+    uint16_t* hc16_w = nullptr;   // fork hc16 (sm_75): staging for one BF16 weight re-encoded as FP16 (HC16_W_ELEMS)
     uint16_t *xn16_lo = nullptr, *lo16_lo = nullptr, *mixed_bf_lo = nullptr;   // bf16x2(): the BF16 GEMMs' low parts
     float* bo = nullptr;
     // GDN
```

### 18-16

`src/prefill/prefill.cpp` · bool Prefill::carve(size_t T, void* alloc) {

Anchor:
```
    m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
    m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
    m.mixed_h = o.take<uint16_t>(T * N, ok);
```
Block (hunk 4 of `src/prefill/prefill.cpp` in the patch, 7 lines):
```diff
     m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
     m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
     m.mixed_h = o.take<uint16_t>(T * N, ok);
+    m.hc16_w = hc16_alloc(m.f16_io) ? o.take<uint16_t>((size_t) HC16_W_ELEMS, ok) : nullptr;
     // Embedding rows are dead after gr_broadcast on m.cs; subsequent half outputs use the same stream.
     m.bo = m.emb;
     if (bf16x2_hc(m.f16_io)) { m.xn16_lo = o.take<uint16_t>(T * D, ok); m.lo16_lo = o.take<uint16_t>(T * LR, ok); }
```

### 18-17

`src/prefill/prefill.cpp` · uint64_t Prefill::bytes_needed_impl(const core::ModelGeometry& g, const core::Se

Anchor:
```
    f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
    if (!emb_reuse_account()) f(T * N);   // bo: aliases emb in carve; still counted unless STRATA_EMB_REUSE_ACCOUNT=1
    const bool f16_io = prompt_f16();   // the current device's mode (the stage's), as Prefill::init will decide it
```
Block (hunk 5 of `src/prefill/prefill.cpp` in the patch, 9 lines):
```diff
     f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
     if (!emb_reuse_account()) f(T * N);   // bo: aliases emb in carve; still counted unless STRATA_EMB_REUSE_ACCOUNT=1
     const bool f16_io = prompt_f16();   // the current device's mode (the stage's), as Prefill::init will decide it
+    // fork hc16: the weight staging `carve` takes after mixed_h.  THIS MUST STAY IN STEP WITH `carve`: the region is sized
+    // from this list, and a missing entry is not a smaller buffer but an allocation past the end of the arena.
+    if (hc16_alloc(f16_io)) o.take<uint16_t>((size_t) HC16_W_ELEMS, ok);
     if (bf16x2_hc(f16_io)) { o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok); }
     if (bf16x2(f16_io)) o.take<uint16_t>(T * N, ok);
     o.take<int32_t>(T * strata::kernels::kStepCount, ok);
```

### 18-18

`src/prefill/prefill.cpp` · bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y,

Anchor:
```
    return true;
}
```
Block (hunk 6 of `src/prefill/prefill.cpp` in the patch, 29 lines):
```diff
     return true;
 }

+/// hc16 can take this weight: a plain resident BF16 matrix that fits the staging buffer
+inline bool hc16_weight_ok(const core::WeightRef* w) {
+    return w->kind == core::WeightKind::Bf16InF32 && w->data && w->q8_scale == nullptr &&
+           w->ne0 * (w->ne1 > 0 ? w->ne1 : 1) <= HC16_W_ELEMS;
+}
+/// Fork hc16 (sm_75): a projection whose weight is a resident BF16 matrix, re-encoded as FP16 into `w16` (one matrix at a
+/// time, HC16_W_ELEMS at most) and multiplied on the FP16 tensor cores against `X16`, an FP16 image of the activations.
+/// The re-encoding is value-preserving - a speed path, not a precision one - and shares the compute stream with the GEMM,
+/// so the staging buffer cannot be overwritten while cuBLAS still reads it.
+bool f16_proj_staged(Gemm& gm, const core::WeightRef* w, const uint16_t* X16, uint16_t* w16, float* Y, int64_t T,
+                     const std::string& name, std::string& err, int64_t ldy, void* stream) {
+    if (w->kind != core::WeightKind::Bf16InF32 || !w->data || w->q8_scale != nullptr || w16 == nullptr) {
+        err = "prefill: " + name + " is not a resident BF16 tensor (STRATA_HC16_SM75 needs plain BF16 hyper-connection "
+              "projections; it is off with STRATA_HC_REQ8 / Q8_0 ones, unset it)";
+        return false;
+    }
+    const int64_t Nn = w->ne1 > 0 ? w->ne1 : 1, Kk = w->ne0;
+    if (Nn * Kk > HC16_W_ELEMS) { err = "prefill: " + name + " is larger than the STRATA_HC16_SM75 staging buffer"; return false; }
+    hc16_w_to_f16((const uint16_t*) w->data, w16, Kk * Nn, stream);
+    gm.f16(X16, w16, Y, T, Nn, Kk, ldy);
+    return true;
+}
+
 }  // namespace

 namespace {
```

### 18-19

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                }
                pt.mark(kPfHc, cs);
                // the xn16 token stride of this chunk (the writers of both halves and of the fused write-backs use it)
```
Block (hunk 7 of `src/prefill/prefill.cpp` in the patch, 23 lines):
```diff
                 }
                 pt.mark(kPfHc, cs);
                 // the xn16 token stride of this chunk (the writers of both halves and of the fused write-backs use it)
-                const int64_t ldx = hc_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) && !gr_unfused() &&
+                // fork hc16 (STRATA_HC16_SM75=1, sm_75): xn16 and lo16 hold the FP16 images of the FP32 values and the
+                // hyper-connection projections run staged on the FP16 tensor cores.  PRECEDENCE with STRATA_BF16_TC=1: hc16
+                // wins on its call sites (down / up / inject here, ssm_alpha / beta below): xn16 is then written by hc16's
+                // rule (xn_img 2), never by BF16_TC's FP16-of-BF16 one (xf16 off), and the staged weight replaces BF16_TC's
+                // per-call conversion.  Every other BF16 GEMM (router, indexer, the draft layer) keeps what BF16_TC says.
+                // hc16 keeps xn16 unpadded (ldx = D), so STRATA_PF_HCDOWN / STRATA_HCD_EXACT / STRATA_HC_UPMIX stay off.
+                const bool hc16 = m.hc16_w != nullptr;
+                const int64_t ldx = !hc16 && hc_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) && !gr_unfused() &&
                                             !m.xn16_lo ? D + XN_PAD : D;
-                const bool xf16 = m.xn_f16 && !m.xn16_lo && ldx == D;   // xn16 holds FP16 (its writers and readers below)
-                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo, xf16);
-                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs, m.xn16_lo, ldx, xf16);
+                const bool xf16 = !hc16 && m.xn_f16 && !m.xn16_lo && ldx == D;   // xn16 holds FP16 (its writers and readers below)
+                const int xn_img = hc16 ? 2 : xf16 ? 1 : 0;   // the f16img mode of the gr_* writers (kernels.hpp)
+                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo, xn_img);
+                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs, m.xn16_lo, ldx, xn_img);
                 normed = false;
                 bool hcd = false;
                 bool hdown = false;
```

### 18-20

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                    }
                    hcd = true;
                }
```
Block (hunk 8 of `src/prefill/prefill.cpp` in the patch, 15 lines):
```diff
                     }
                     hcd = true;
                 }
-                if (!hcd && !hdown && !bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo, ldx != D ? ldx : 0, m.cs, xf16)) return false;
-                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo);
+                if (hc16) {
+                    if (!f16_proj_staged(m.gemm, wd, m.xn16, m.hc16_w, m.lo, T, sd, err, 0, m.cs)) return false;
+                } else if (!hcd && !hdown && !bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo, ldx != D ? ldx : 0, m.cs, xf16)) return false;
+                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo, hc16 ? 1 : 0);
                 bool upmixed = false;
-                if (wu->q8_scale == nullptr && hc_upmix() && T >= std::max(pf_switch_min_t(), hc_upmix_min_t()) && !gr_unfused() && !m.lo16_lo && !m.mixed_bf_lo &&
+                if (!hc16 && wu->q8_scale == nullptr && hc_upmix() && T >= std::max(pf_switch_min_t(), hc_upmix_min_t()) && !gr_unfused() && !m.lo16_lo && !m.mixed_bf_lo &&
                     wu->kind == core::WeightKind::Bf16InF32 && wu->data && wu->ne0 == LR && wu->ne1 == D) {
                     static int checks = [] { const char* e = std::getenv("STRATA_HC_UPMIX_CHECK"); return e ? std::atoi(e) : 0; }();
                     if (checks > 0) {   // the default pair first, kept for the comparison
```

### 18-21

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                                           m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                    }
                }
```
Block (hunk 9 of `src/prefill/prefill.cpp` in the patch, 14 lines):
```diff
                                            m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                     }
                 }
-                if (!upmixed && !bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo, 0, m.cs)) return false;
-                if (!hcd && !bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo, ldx != D ? ldx : 0, m.cs, xf16)) return false;
+                if (hc16) {
+                    if (!f16_proj_staged(m.gemm, wu, m.lo16, m.hc16_w, m.gated, T, su, err, 0, m.cs)) return false;
+                } else if (!upmixed && !bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo, 0, m.cs)) return false;
+                if (hc16) {
+                    if (!f16_proj_staged(m.gemm, wi, m.xn16, m.hc16_w, m.inj, T, si, err, 0, m.cs)) return false;
+                } else if (!hcd && !bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo, ldx != D ? ldx : 0, m.cs, xf16)) return false;
                 if (upmixed) {
                 } else if (gr_unfused()) {
                     gr_mix(m.xn, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h, m.mixed_bf_lo);
```

### 18-22

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
```
Block (hunk 10 of `src/prefill/prefill.cpp` in the patch, 16 lines):
```diff
                     float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                     if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                     if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
-                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
-                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
+                    // fork hc16: these two are BF16 GEMMs too (FFMA on Turing) and `mixed_h` is already the FP16 image of
+                    // their input, so they take the same staged tensor-core path
+                    if (hc16 && hc16_weight_ok(wa)) {
+                        if (!f16_proj_staged(m.gemm, wa, m.mixed_h, m.hc16_w, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.cs)) return false;
+                    } else if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
+                    if (hc16 && hc16_weight_ok(wb)) {
+                        if (!f16_proj_staged(m.gemm, wb, m.mixed_h, m.hc16_w, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.cs)) return false;
+                    } else if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                     pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                     gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                     gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
```

### 18-23

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                        cudaMemcpyAsync(Rc, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, m.cs);
                        gr_write(Rc, m.bo, m.inj, HC, T, m.cs);
                        strata::kernels::cvec_apply(Rc, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
```
Block (hunk 11 of `src/prefill/prefill.cpp` in the patch, 12 lines):
```diff
                         cudaMemcpyAsync(Rc, m.R, (size_t) T * D * 4, cudaMemcpyDeviceToDevice, m.cs);
                         gr_write(Rc, m.bo, m.inj, HC, T, m.cs);
                         strata::kernels::cvec_apply(Rc, l, T, D, nullptr, 0, nullptr, 0, false, m.cs);
-                        gr_norm_rs(Rc, (const float*) wnn->data, EPS, rc, xc, T, m.cs, nullptr, D, xf16);
+                        gr_norm_rs(Rc, (const float*) wnn->data, EPS, rc, xc, T, m.cs, nullptr, D, xn_img);
                         gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                               strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs,
-                                              m.xn16, T, m.cs, m.xn16_lo, ldx, xf16);
+                                              m.xn16, T, m.cs, m.xn16_lo, ldx, xn_img);
                         std::vector<float> a((size_t) T * D), b((size_t) T * D), ra((size_t) T * HC), rb((size_t) T * HC);
                         std::vector<uint16_t> xa((size_t) T * D), xb((size_t) T * D);
                         cudaMemcpyAsync(a.data(), m.R, a.size() * 4, cudaMemcpyDeviceToHost, m.cs);
```

### 18-24

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                    } else
                    gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                          strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs, m.xn16,
```
Block (hunk 12 of `src/prefill/prefill.cpp` in the patch, 13 lines):
```diff
                     } else
                     gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                           strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs, m.xn16,
-                                          T, m.cs, m.xn16_lo, ldx, xf16);
+                                          T, m.cs, m.xn16_lo, ldx, xn_img);
                     normed = true;
                 } else if (wnn) {
                     gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.grs, m.xn16, T, m.cs,
-                                     m.xn16_lo, ldx, xf16);
+                                     m.xn16_lo, ldx, xn_img);
                     normed = true;
                 } else {
                     gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
```

### 18-25

`sycl/src/prefill/kernels.dp.cpp` · void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo

Anchor:
```
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
```
Block (hunk 1 of `sycl/src/prefill/kernels.dp.cpp` in the patch, 8 lines):
```diff
 }

 void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
-             uint16_t* xn16_lo, bool /*f16img: the Turing FP16 image of xn16 (CUDA only)*/) {
+             uint16_t* xn16_lo, int /*f16img: the Turing FP16 image of xn16 (CUDA only)*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

### 18-26

`sycl/src/prefill/kernels.dp.cpp` · void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t

Anchor:
```
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
```
Block (hunk 2 of `sycl/src/prefill/kernels.dp.cpp` in the patch, 8 lines):
```diff
     check("gr_norm");
 }
 void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
-                uint16_t* xn16_lo, int64_t ldx, bool /*f16img*/) {
+                uint16_t* xn16_lo, int64_t ldx, int /*f16img*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

### 18-27

`sycl/src/prefill/kernels.dp.cpp` · bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const

Anchor:
```
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx,
```
Block (hunk 3 of `sycl/src/prefill/kernels.dp.cpp` in the patch, 8 lines):
```diff
 }
 void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                       float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx,
-                      bool /*f16img*/) {
+                      int /*f16img*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

### 18-28

`sycl/src/prefill/kernels.dp.cpp` · void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_l

Anchor:
```
}
void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                           const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
```
Block (hunk 4 of `sycl/src/prefill/kernels.dp.cpp` in the patch, 8 lines):
```diff
 }
 void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                            const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
-                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, bool /*f16img*/) {
+                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, int /*f16img*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

### 18-29

`sycl/src/prefill/kernels.dp.cpp` · void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t

Anchor:
```
    }
    check("gr_write_cvec_norm_rs");
}
```
Block (hunk 5 of `sycl/src/prefill/kernels.dp.cpp` in the patch, 8 lines):
```diff
     }
     check("gr_write_cvec_norm_rs");
 }
-void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
+void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo, int /*f16img: the Turing FP16 image (CUDA only)*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```
