# Strata-2080Ti patch stack: the inserted blocks

Generated from the patches by `check_anchors.py --build`. One section per insertion point of `INDEX.md`, the anchor (unpatched lines) first, then the hunk as `git diff` shows it. Hunks longer than 80 lines are cut after 24: the `.patch` has them whole.

## 01 elastic-cache

`01-elastic-cache.patch` — FABLE-STRATA-Porta su 0.1.43 la cache elastica con nucleo fisso (--elastic, PR #1117)

### 01-1

`CMakeLists.txt` · if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)

Anchor:
```
    src/core/arch_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
  target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 10 lines):
```diff
     src/core/arch_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
   target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
   target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
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
-              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0);
+              void* stream, std::string& err, void* borrow = nullptr, uint64_t borrow_bytes = 0,
+              int64_t chunk_max = 0);

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
    blocking_staging_bytes_ = 0;
#endif
    off_.clear();
```
Block (hunk 3 of `src/core/expert_cache.cpp` in the patch, 22 lines):
```diff
     blocking_staging_bytes_ = 0;
 #endif
     off_.clear();
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
Block (hunk 1 of `src/prefill/prefill.cpp` in the patch, 12 lines):
```diff

 bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                    core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
-                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes) {
+                   void* stream, std::string& err, void* borrow, uint64_t borrow_bytes, int64_t chunk_max) {
     Impl& m = *impl_;
+    // the host side is sized for the largest chunk a relayout may ask for; the device side (carved below from the
+    // borrowed slots) for this one
+    const int64_t cap_T = std::max<int64_t>(chunk, chunk_max);
     m.wt = &wt; m.g = &g; m.ss = &ss; m.src = src; m.cache = cache; m.host_res = host_res;
     m.T = chunk; m.cs = (cudaStream_t) stream; m.stats = &stats_;
     if (g.n_embd != N || g.hc != HC || g.hc_lr != LR || g.n_expert < 1 || ss.k != K) {
```

### 01-11

`src/prefill/prefill.cpp` · bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co

Anchor:
```
    // the CPU share's staged-chunk limit, before the ring is sized below (a caller that sizes loans first armed it)
    arm_cpu_share(cpu_pool_ != nullptr);
    for (int b = 0; next_ != nullptr && b < 2; ++b)
```
Block (hunk 2 of `src/prefill/prefill.cpp` in the patch, 28 lines):
```diff
     // the CPU share's staged-chunk limit, before the ring is sized below (a caller that sizes loans first armed it)
     arm_cpu_share(cpu_pool_ != nullptr);
     for (int b = 0; next_ != nullptr && b < 2; ++b)
-        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) chunk * D * 4, cudaHostAllocPortable) != cudaSuccess) {
+        if (!m.hand[b] && cudaHostAlloc((void**) &m.hand[b], (size_t) cap_T * D * 4, cudaHostAllocPortable) != cudaSuccess) {
             err = "prefill: the layer split's hand-off buffers";
             return false;
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
     bool ok = true;
     // one-time: events, the stager, the host buffers (for the largest chunk), the identity page table
```

### 01-12

`src/prefill/prefill.cpp` · bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co

Anchor:
```
        m.gemm.set_f16_io(m.f16_io);
        set_act_f16(m.f16_io);
    }
```
Block (hunk 3 of `src/prefill/prefill.cpp` in the patch, 8 lines):
```diff
         m.gemm.set_f16_io(m.f16_io);
         set_act_f16(m.f16_io);
     }
-    if (!carve(T, &o)) {
+    if (!carve((size_t) chunk, &o)) {
         size_t fb = 0, tb = 0;
         cudaMemGetInfo(&fb, &tb);
         (void) cudaGetLastError();
```

### 01-13

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

### 01-14

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

### 01-15

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

### 01-16

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

### 01-17

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

### 01-18

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

### 01-19

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

### 01-20

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

### 01-21

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

### 01-22

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

### 01-23

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

### 01-24

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

### 01-25

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

### 01-26

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

### 01-27

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

### 01-28

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

### 01-29

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

### 01-30

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            }
            if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
            if (share_pool) sp.set_cpu_pool(&pool);
```
Block (hunk 18 of `src/program/generate.cpp` in the patch, 8 lines):
```diff
             }
             if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
             if (share_pool) sp.set_cpu_pool(&pool);
-            if (!sp.init(wt, g, ss, prefill_src, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes))
+            if (!sp.init(wt, g, ss, prefill_src, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes, sp_chunk_max))
                 return err.find("do not fit") != std::string::npos ? 2 : 1;
             return 0;
         };
```

### 01-31

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

### 01-32

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

### 01-33

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

### 01-34

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            // participant's own cache, and marking only that participant's own layers
            auto lend = [&](int64_t tokens, std::string& e) -> bool {
                if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
```
Block (hunk 22 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
             // participant's own cache, and marking only that participant's own layers
             auto lend = [&](int64_t tokens, std::string& e) -> bool {
                 if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
+                if (el_on) apply_pending(true);   // elastic fills must be published before the tail is lent
                 const auto t_ln = Clock::now();
                 // what this segment needs, capped by the configured chunk: a request lends only what its own
                 // prompt needs, so a large chunk costs a short prompt nothing
```

### 01-35

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

### 01-36

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

### 01-37

`tests/core/expert_cache_elastic_test.cpp` · (new file)

New file, 71 lines: see the patch.

## 02 split-prompt-borrow

`02-split-prompt-borrow.patch` — FABLE-STRATA-Porta su 0.1.43 la stima del prestito del prompt sul device del proprio blocco (PR #1417)

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

`03-forced-bench.patch` — FABLE-STRATA-Porta su 0.1.43 il banco forzato STRATA_FORCE_IDS/STRATA_FORCE_WINDOWS/STRATA_FORCE_TRACE (PR #1419)

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
             // the verify windows: the first holds the last prompt token alone
             int64_t p = n - 1;
             int32_t x = (int32_t) ids[(size_t) (n - 1)];
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
                window[0] = x;
                for (int i = 1; i < T_mtp; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
                for (int i = 0; i < chain_n; ++i) window[(size_t) (T_mtp + i)] = cbuf[(size_t) i];
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                 window[0] = x;
                 for (int i = 1; i < T_mtp; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
                 for (int i = 0; i < chain_n; ++i) window[(size_t) (T_mtp + i)] = cbuf[(size_t) i];
+                if (force_win != nullptr) force_window(p, window[0], T, window.data(), S);   // lab bench replay
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
                int a = 0;
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 9 lines):
```diff
                     }
                 }
                 int a = 0;
+                force_rows_out(outv.data(), T, produced_n);   // lab bench: STRATA_FORCE_IDS
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

## 04 b6-mmvq-rows

`04-b6-mmvq-rows.patch` — FABLE-STRATA-Porta su 0.1.43 l'MMVQ multi-colonna di B6 opt-in (STRATA_B6_MMVQ_ROWS=1, PR #1418)

### 04-1

`CMakeLists.txt` · if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)

Anchor:
```
  add_executable(router_chain_bench src/kernels/router_chain_bench.cpp)
  target_include_directories(router_chain_bench PRIVATE ${_strata_gpu_include_directories})
  target_link_libraries(router_chain_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 12 lines):
```diff
   add_executable(router_chain_bench src/kernels/router_chain_bench.cpp)
   target_include_directories(router_chain_bench PRIVATE ${_strata_gpu_include_directories})
   target_link_libraries(router_chain_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
+  # B6: the dense projections' time per verify row at the model's shapes, default vs STRATA_B6_MMVQ_ROWS, bitwise
+  if(STRATA_ENABLE_CUDA)
+    add_executable(b6_mmvq_bench src/kernels/b6_mmvq_bench.cpp)
+    target_include_directories(b6_mmvq_bench PRIVATE ${_strata_gpu_include_directories})
+    target_link_libraries(b6_mmvq_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
+  endif()
   add_executable(gr_parity src/kernels/gr_parity.cpp)
   target_include_directories(gr_parity PRIVATE ${_strata_gpu_include_directories})
   target_link_libraries(gr_parity PRIVATE strata_kernels ${_strata_gpu_runtime_target})
```

### 04-2

`include/strata/kernels/native_mmvq.hpp` · std::size_t native_q8_1_bytes(int n_in, int ncols = 1);

Anchor:
```
// graph capture; captured graphs keep the kernels they captured.
void native_mmvq_set_multi_exact(bool exact);
bool native_mmvq_multi_exact();
```
Block (hunk 1 of `include/strata/kernels/native_mmvq.hpp` in the patch, 12 lines):
```diff
 // graph capture; captured graphs keep the kernels they captured.
 void native_mmvq_set_multi_exact(bool exact);
 bool native_mmvq_multi_exact();
+// B6 (lab, opt-in STRATA_B6_MMVQ_ROWS=1): the exact multi-column layout with the activation blocks of every column
+// loaded once per weight block and reused across the block's rows (bitwise the same sums as the default exact
+// layout). 0 = off (default), 1 = 2 rows per block, 2 / 4 / 8 = that many rows. Q4_K, Q5_K, Q6_K and IQ4_XS only.
+// Set before graph capture.
+void native_mmvq_set_b6_rows(int mode);
+int native_mmvq_b6_rows();

 // One quantization may serve multiple weight matrices sharing the same input.
 // Q8_1 stores FP16 scale and FP16 warp sum of the ORIGINAL float inputs; it does
```

### 04-3

`src/kernels/b6_mmvq_bench.cpp` · (new file)

New file, 218 lines: see the patch.

### 04-4

`src/kernels/cuda/native_mmvq.cu` · struct SmallTraits<IQ4NLBlock, 4> {

Anchor:
```
// llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
// equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)
```
Block (hunk 1 of `src/kernels/cuda/native_mmvq.cu` in the patch, 8 lines):
```diff
 // llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
 // equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
 bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)
+// B6 STRATA_B6_MMVQ_ROWS (lab, default 0 = off): see native_mmvq_set_b6_rows
+int g_b6_rows = [] { const char* v = std::getenv("STRATA_B6_MMVQ_ROWS"); return v ? std::atoi(v) : 0; }();

 // S26 STRATA_TSUM=1 (TS): warp 0's NCOLS x ROWS sums as one transposed butterfly (s26_tsum.cuh), bitwise the same
 static bool s26_tsum_on() {
```

### 04-5

`src/kernels/cuda/native_mmvq.cu` · __global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w

Anchor:
```
    if (!g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
```
Block (hunk 2 of `src/kernels/cuda/native_mmvq.cu` in the patch, 174 lines):
```diff
     }
 }

+// ---- B6 STRATA_B6_MMVQ_ROWS (lab, opt-in): the exact multi-column layout with each column's activation fragment
+// loaded ONCE per weight block and reused across ROWS rows of the block (the default kernel reloads it for every row
+// it applies it to, and the K-quant scales once per column).  A thread keeps the same blocks (kbx = tid / T, stride
+// BPI with WARPS warps), each (row, column) sum adds the same per-block values in the same order, and the cross-warp
+// and warp reductions are the default kernel's: every output is bitwise the default exact layout's, so bitwise a
+// single-column call.  Only the formats with a B6X below; the others keep the default kernel.
+template<typename F> struct B6X { static constexpr bool ok = false; };
+template<> struct B6X<Q6KTraits> {
+    static constexpr bool ok = true;
+    struct X { int u[2]; float d8[2]; };
+    struct W { int vl, vh, sc[2]; float d; };
+    __device__ static X xload(const Q81Block* x, int iqs) {
+        X r;
+        const int off = 4 * (iqs / 16) + (iqs % 16) / 8;   // Q6KTraits::load's bq8_offset
+#pragma unroll
+        for (int i = 0; i < 2; ++i) {
+            r.u[i] = reinterpret_cast<const int*>(x[off + 2 * i].qs)[iqs % 8];
+            r.d8[i] = __low2float(x[off + 2 * i].ds);
+        }
+        return r;
+    }
... 150 more lines: see 04-b6-mmvq-rows.patch
```

### 04-6

`src/kernels/cuda/native_mmvq.cu` · void native_q6_k_unpack(const void* weights) {

Anchor:
```

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }
```
Block (hunk 3 of `src/kernels/cuda/native_mmvq.cu` in the patch, 8 lines):
```diff

 void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
 bool native_mmvq_multi_exact() { return g_multi_exact; }
+void native_mmvq_set_b6_rows(int mode) { g_b6_rows = mode; }
+int native_mmvq_b6_rows() { return g_b6_rows; }

 std::size_t native_q8_1_bytes(int n_in, int ncols) {
     validate_shape(n_in, ncols);
```

## 05 elastic-split-pipeline

`05-elastic-split-pipeline.patch` — FABLE-STRATA-Porta su 0.1.43 la cache elastica di CUDA0 con --layer-split e accanto a --pipeline-windows (di default con due schede)

### 05-1

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

### 05-2

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

### 05-3

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

### 05-4

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

### 05-5

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

### 05-6

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

### 05-7

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

### 05-8

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

### 05-9

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        // --pipeline-windows: set while the tier adapts beside windows in flight (see its use in adapt)
        std::function<void()> adapt_fence;
        // the VRAM tier follows the conversation (the same rule as the speculative loop below)
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 14 lines):
```diff
         // --pipeline-windows: set while the tier adapts beside windows in flight (see its use in adapt)
         std::function<void()> adapt_fence;
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

### 05-10

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

### 05-11

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

### 05-12

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

### 05-13

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

### 05-14

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

### 05-15

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

### 05-16

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

### 05-17

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

### 05-18

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

### 05-19

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

### 05-20

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

### 05-21

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

### 05-22

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

### 05-23

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

### 05-24

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

### 05-25

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

### 05-26

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

### 05-27

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

### 05-28

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

### 05-29

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

### 05-30

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

## 06 prefix-disk

`06-prefix-disk.patch` — FABLE-STRATA-Porta su 0.1.43 il prefisso su disco con due schede (--prompt-cache-file, formato v2)

### 06-1

`CMakeLists.txt` · option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})

Anchor:
```
option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 23 lines):
```diff
 option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
 option(STRATA_WERROR "Treat compiler warnings as errors" OFF)

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

### 06-2

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
   if(STRATA_ENABLE_CUDA AND NOT STRATA_HIP_GFX906)
     # --elastic: the automatic elastic expert cache (CUDA virtual memory management through the driver API)
     target_link_libraries(strata_engine PUBLIC CUDA::cuda_driver)
```

### 06-3

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

### 06-4

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

### 06-5

`include/strata/core/prefix_cache_file.hpp` · (new file)

New file, 64 lines: see the patch.

### 06-6

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

### 06-7

`src/core/prefix_cache_file.cpp` · (new file)

New file, 599 lines: see the patch.

### 06-8

`src/core/prefix_cache_file_test.cpp` · (new file)

New file, 587 lines: see the patch.

### 06-9

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

### 06-10

`src/program/generate.cpp` · (file scope)

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

### 06-11

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

### 06-12

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

### 06-13

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
                  a == "--session-min-free-mib" || a == "--memory-limit-mib") {
```

### 06-14

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    // parking with --layer-split saves every stage (SavedConversation::stage_images)
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
         std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                      o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
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

### 06-15

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
... 79 more lines: see 06-prefix-disk.patch
```

### 06-16

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                incoming.reset();
                err.clear();
            }
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 59 lines):
```diff
                 incoming.reset();
                 err.clear();
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
+                        disk_root > std::max(resume, incoming ? incoming_tokens : int64_t(0))) {
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

### 06-17

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                                 mtp.kv_state().kv_mode, "ram",
                                 (long long) (mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size));
                }
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 50 lines):
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
                     conversations.retain(std::move(incoming->kv), int64_t(live.size()), std::move(stage_kv));
                 }
                 incoming.reset(); // Running-state/checkpoint copies are no longer needed.
+                if (incoming_disk)
+                    std::fprintf(stderr, "strata serve: prefix disk cache: restored %lld tokens; read/validate %.1f ms, upload %.1f ms\n",
+                                 (long long) resume, disk_read_ms,
+                                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
                 std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu\n",
                              (long long) resume, from_live ? "live" : "checkpoint",
                              std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
```

### 06-18

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        if (i >= o.prompt_cache_root) root_at = i;
                        break;
                    }
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 11 lines):
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

### 06-19

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                if (trace && to == message_at)
                    std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                                 (long long) message_at, (long long) (turn_at - message_at));
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 72 lines):
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

## 07 vram-free-nvml

`07-vram-free-nvml.patch` — FABLE-STRATA-Porta su 0.1.43 la PR upstream #1911: VRAM libera della scheda intera via NVML (STRATA_VRAM_FREE=device), letta anche dalla cache elastica

### 07-1

`docs/DETAILS.md` · GPU nor the RAM budget holds (only pages - the experts computed are the same; `S

Anchor:
```
is what runs [Unsloth's UD-Q4_K_XL](UNSLOTH_Q4.md) (72 GiB of experts) on a 64 GB PC: 7-8.5 tokens/s at N = 40 on an
RTX 5070, against ~3 tokens/s before these changes.
```
Block (hunk 1 of `docs/DETAILS.md` in the patch, 20 lines):
```diff
 is what runs [Unsloth's UD-Q4_K_XL](UNSLOTH_Q4.md) (72 GiB of experts) on a 64 GB PC: 7-8.5 tokens/s at N = 40 on an
 RTX 5070, against ~3 tokens/s before these changes.

+**Opt-in switches measured on a hybrid Intel CPU under Windows (Core i7-13700K, 8 P + 8 E cores, RTX 5090, 64 GB RAM,
+Windows 11, Swift 1.5 IQ3_XXS at 128K):**
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
+
 **Switches added in 0.1.40 (all off unless noted; none changes the default output):**
 - `--kv-grow` (or `STRATA_KV_GROW=1`; `--no-kv-grow` turns it off): the K/V takes VRAM only for the cells the requests
   reach, and the expert cache holds the rest, giving slots back as the context grows. It needs one GPU, an expert
```

### 07-2

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

 class ExpertCache {
```

### 07-3

`src/core/expert_cache.cpp` · (file scope)

Anchor:
```
#include <utility>
#include <cstring>
```
Block (hunk 1 of `src/core/expert_cache.cpp` in the patch, 64 lines):
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
+#define STRATA_EC_NVML_FREE 1
+#endif
+
 namespace strata::core {

+#if defined(STRATA_EC_NVML_FREE)
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
+#endif
+
 // See the header.  NVIDIA's DGX Spark Porting Guide (section 5.5, "Memory reporting on UMA systems") recommends the same:
 // not to rely on cudaMemGetInfo alone but to count the memory the OS can reclaim.  Swap is not counted here (unlike
 // NVIDIA's reference snippet): an expert cache that pushes the system into swap would be far slower than a smaller one.
```

### 07-4

`src/core/expert_cache.cpp` · size_t device_free_bytes() {

Anchor:
```
            if (avail > head && avail - head > free_b) free_b = (size_t) (avail - head);
        }
    }
```
Block (hunk 2 of `src/core/expert_cache.cpp` in the patch, 27 lines):
```diff
             if (avail > head && avail - head > free_b) free_b = (size_t) (avail - head);
         }
     }
+#endif
+#if defined(STRATA_EC_NVML_FREE)
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
 #endif
     return free_b;
 }
```

### 07-5

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

### 07-6

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

## 08 park-no-foreign-kv

`08-park-no-foreign-kv.patch` — FABLE-STRATA-Porta su 0.1.43 la PR upstream #1503: mai parcheggiare una conversazione col K/V di un'altra

### 08-1

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

### 08-2

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

### 08-3

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

### 08-4

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

### 08-5

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

### 08-6

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
             for (size_t k = 0; k < bslot_ss.size(); ++k) {
```

### 08-7

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
                 if (incoming_disk)
```

### 08-8

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

### 08-9

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

### 08-10

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

## 09 pipeline-architectds

`09-pipeline-architectds.patch` — FABLE-STRATA-Porta su 0.1.43 il profilo architectds della pipeline (STRATA_PIPELINE_ELASTIC: verificata servita per prima, lookup oltre una finestra, theta 0,10) e i log di rollback e latenza della catena

### 09-1

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

### 09-2

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

### 09-3

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

### 09-4

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

### 09-5

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

### 09-6

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

### 09-7

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

### 09-8

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

### 09-9

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

### 09-10

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

### 09-11

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

### 09-12

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

### 09-13

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

### 09-14

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

### 09-15

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

### 09-16

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

### 09-17

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

### 09-18

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

### 09-19

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

### 09-20

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

## 10 attn-daily

`10-attn-daily.patch` — FABLE-STRATA-Porta su 0.1.43 l'attenzione QSA del decode del Daily (STRATA_ATTN_DAILY=1: chunk pre75 su ogni scheda e merge v2) col test di parita

### 10-1

`CMakeLists.txt` · if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)

Anchor:
```
    target_link_libraries(qsa_topk_active_parity PRIVATE strata_kernels CUDA::cudart)
    add_test(NAME qsa_topk_active_parity COMMAND qsa_topk_active_parity)
    set_tests_properties(qsa_topk_active_parity PROPERTIES SKIP_RETURN_CODE 77)
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 17 lines):
```diff
     target_link_libraries(qsa_topk_active_parity PRIVATE strata_kernels CUDA::cudart)
     add_test(NAME qsa_topk_active_parity COMMAND qsa_topk_active_parity)
     set_tests_properties(qsa_topk_active_parity PROPERTIES SKIP_RETURN_CODE 77)
+    # fork: the Daily's decode attention (STRATA_ATTN_DAILY=1: PR #540's chunk kernel on every card and the bounded
+    # merge) against upstream's kernels, bitwise; the production entries checked with the variable off and on
+    add_executable(qsa_merge_daily_parity src/kernels/qsa_merge_daily_parity.cpp)
+    target_include_directories(qsa_merge_daily_parity PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
+    target_link_libraries(qsa_merge_daily_parity PRIVATE strata_kernels CUDA::cudart)
+    add_test(NAME qsa_merge_daily_parity_default COMMAND qsa_merge_daily_parity)
+    add_test(NAME qsa_merge_daily_parity_daily COMMAND qsa_merge_daily_parity)
+    set_tests_properties(qsa_merge_daily_parity_default PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=0"
+                         SKIP_RETURN_CODE 77)
+    set_tests_properties(qsa_merge_daily_parity_daily PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=1"
+                         SKIP_RETURN_CODE 77)
   endif()
   # the QSA selection (block scores on the warp kernel and the tensor-core one, top-k): a GPU bench with a
   # scorer-vs-scorer comparison and an FP64 accuracy gate (synthetic, no model); the ctest runs a 32K context under a
```

### 10-2

`include/strata/kernels/qsa_decode_attn.hpp` · void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32

Anchor:
```
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);
```
Block (hunk 1 of `include/strata/kernels/qsa_decode_attn.hpp` in the patch, 16 lines):
```diff
 void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                            int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

+/// For src/kernels/qsa_merge_daily_parity.cpp only: qsa_decode_attn_batch's two passes apart, whatever the environment
+/// says.  `chunks_only` runs the chunk pass into `scratch` with `daily_chunk` 0: attn_chunk_kernel (upstream's default),
+/// 1: the Daily's (attn_chunk_kernel_pre75, what STRATA_ATTN_DAILY=1 picks on CUDA).  `merge_only` merges those
+/// partials with `v2` 0: attn_merge_kernel, 1: attn_merge_v2_kernel (STRATA_ATTN_MERGE_V2, on with STRATA_ATTN_DAILY).
+/// Width must not exceed an unaligned cap (the chunk pass keeps its upstream contract); for aligned caps the grid
+/// bounds widths above cap.  One input can go through every pair and the outputs be compared bit for bit.
+void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
+                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, int daily_chunk,
+                                 void* stream);
+void qsa_decode_attn_merge_only(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch,
+                                float* attn, int64_t n_q, int v2, void* stream);
+
 }  // namespace strata::kernels
```

### 10-3

`src/kernels/cuda/qsa_decode_attn.cu` · __global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __rest

Anchor:
```
// run on gfx103x (RDNA2), with 8 cells per step and the butterfly's lane exchanges in registers (DPP / permlanex16)
// instead of ds_bpermute: there the score loop is bound by the LDS pipe (the q reads and ds_bpermute share it), not by
// occupancy - 8 cells per step at 7 waves/SIMD beat 1 cell at 9.  Every other card keeps exactly the kernel above.
```
Block (hunk 1 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 12 lines):
```diff
 // run on gfx103x (RDNA2), with 8 cells per step and the butterfly's lane exchanges in registers (DPP / permlanex16)
 // instead of ds_bpermute: there the score loop is bound by the LDS pipe (the q reads and ds_bpermute share it), not by
 // occupancy - 8 cells per step at 7 waves/SIMD beat 1 cell at 9.  Every other card keeps exactly the kernel above.
-#if defined(STRATA_EXPERIMENTAL_SM60) || (defined(__HIPCC__) && !defined(STRATA_HIP_GFX906))
+// Fork (lab/upsync): built on every CUDA build too, for STRATA_ATTN_DAILY=1 (see daily_attn below).  With NC 2 and
+// no DPP it is the Daily's (be4cf1c) attn_chunk_kernel line for line - the same reduce12 scores, the same cell-major
+// probabilities and the same four-cell value fold - except that `spt` shares `sq`'s storage here (no arithmetic
+// change).  Without the variable nothing selects it on a card at or above sm_75.
+#if defined(STRATA_EXPERIMENTAL_SM60) || !defined(__HIPCC__) || !defined(STRATA_HIP_GFX906)
 #define STRATA_ATTN_PRE75_BUILT 1
 #endif
 #if defined(STRATA_ATTN_PRE75_BUILT)
```

### 10-4

`src/kernels/cuda/qsa_decode_attn.cu` · __global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict_

Anchor:
```
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}
```
Block (hunk 2 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 86 lines):
```diff
     attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
 }

+// attn_merge_kernel's arithmetic in the same chunk order, with two changes that leave every output bit as it was
+// (the Daily's 3effa9d5, from Hardin22; qsa_merge_daily_parity compares the two kernels on the same partials, all four
+// KV formats):
+//
+// (1) Bounded by the width.  The chunk kernel writes m = -FLT_MAX, l = 0 for every chunk with c0 >= step[kStepWidth]
+//     (its `n_here <= 0` exit) and the merge skips exactly those, so it can stop at n_act = ceil(width / CHUNK):
+//     fmaxf with -FLT_MAX never changes M, and a skipped chunk never touches L or acc.  The drafter's merge walks its
+//     whole capacity (--mtp-window 32768: 512 chunks) at every context, where a short context fills one or two.
+// (2) Loads batched.  attn_merge_kernel loads part_l / part_acc behind its `m == -FLT_MAX` branch, so the compiler
+//     issues each chunk's l and acc loads inside that chunk's branch: a few dependent L2 round trips per chunk.  Here
+//     the m, l and acc of U chunks are loaded first, guarded only by the chunk index, then consumed in chunk order.
+//     The skip is a select on the loaded m instead of a branch, so the loads cannot move back behind it.  The values
+//     selected are the very fmaf's attn_merge_kernel computes (same operands, same order), and a skipped chunk leaves
+//     L and acc as they were, as its `continue` does.
+constexpr int kMergeU = 16;
+
+template <int U, bool TAIL>
+__device__ __forceinline__ void merge_max_block(const float* __restrict__ pm, int c0, int n_act, float& M) {
+    float mm[U];
+#pragma unroll
+    for (int u = 0; u < U; ++u) mm[u] = (!TAIL || c0 + u < n_act) ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
... 62 more lines: see 10-attn-daily.patch
```

### 10-5

`src/kernels/cuda/qsa_decode_attn.cu` · bool query_swizzle_gfx906_device() {

Anchor:
```
}
#endif
```
Block (hunk 3 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 42 lines):
```diff
 }
 #endif

+// Fork (lab/upsync): STRATA_ATTN_DAILY=1 runs the Daily's (be4cf1c) decode attention on every CUDA card: the chunk
+// pass on PR #540's kernel (attn_chunk_kernel_pre75<M, 2, false>, the Daily's attn_chunk_kernel), bit for bit the
+// default kernel's partials.  CUDA only; unset or anything but "1" keeps upstream 0.1.42's kernels.
+// STRATA_ATTN_PRE75=0 still turns the chunk kernel off (A/B).
+[[maybe_unused]] bool daily_attn() {
+#if defined(__HIPCC__)
+    return false;
+#else
+    static const bool on = [] {
+        const char* v = std::getenv("STRATA_ATTN_DAILY");
+        return v != nullptr && v[0] == '1' && v[1] == '\0';
+    }();
+    return on;
+#endif
+}
+
 #if defined(STRATA_ATTN_PRE75_BUILT)
-// CUDA: the current device is below sm_75; HIP: it is gfx103x (per device: a layer split can mix cards).
+// CUDA: the current device is below sm_75 (experimental build) or STRATA_ATTN_DAILY=1 (any card); HIP: it is gfx103x
+// (per device: a layer split can mix cards).
 // STRATA_ATTN_PRE75=0 turns PR #540's kernel off (A/B); on HIP =1 also runs it on another wave32 card.
 bool pre75_attn() {
     static const int env = [] {
         const char* e = std::getenv("STRATA_ATTN_PRE75");
         return e == nullptr ? -1 : e[0] == '0' ? 0 : e[0] == '1' ? 1 : -1;
     }();
+    if (env == 0) return false;
+#if !defined(__HIPCC__) && !defined(STRATA_EXPERIMENTAL_SM60)
+    return daily_attn();   // the ready-made engine: upstream's attn_chunk_kernel unless STRATA_ATTN_DAILY=1
+#else
+#if !defined(__HIPCC__)
+    if (daily_attn()) return true;
+#endif
     static int cc[64] = {};
     int dev = 0;
-    if (env == 0) return false;
     if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
 #if defined(__HIPCC__)
     if (cc[dev] == 0) {   // 2: gfx103x, 1: another card
```

### 10-6

`src/kernels/cuda/qsa_decode_attn.cu` · bool pre75_attn() {

Anchor:
```
    }
    return cc[dev] < 75;
#endif
```
Block (hunk 4 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 58 lines):
```diff
     }
     return cc[dev] < 75;
 #endif
+#endif  // ready-made CUDA engine
 }
+// `PRE` picks PR #540's kernel; the environment's choice is STRATA_ATTN_CHUNK.
 #if defined(__HIPCC__)
-#define STRATA_ATTN_CHUNK(M) (pre75_attn() ? attn_chunk_kernel_pre75<M, 8, true> : attn_chunk_kernel<M>)
+#define STRATA_ATTN_CHUNK_SEL(M, PRE) ((PRE) ? attn_chunk_kernel_pre75<M, 8, true> : attn_chunk_kernel<M>)
 #else
-#define STRATA_ATTN_CHUNK(M) (pre75_attn() ? attn_chunk_kernel_pre75<M> : attn_chunk_kernel<M>)
+#define STRATA_ATTN_CHUNK_SEL(M, PRE) ((PRE) ? attn_chunk_kernel_pre75<M> : attn_chunk_kernel<M>)
 #endif
+#define STRATA_ATTN_CHUNK(M) STRATA_ATTN_CHUNK_SEL(M, pre75_attn())
 #else
+#define STRATA_ATTN_CHUNK_SEL(M, PRE) attn_chunk_kernel<M>
 #define STRATA_ATTN_CHUNK(M) attn_chunk_kernel<M>
 #endif

-}  // namespace
+// STRATA_ATTN_MERGE_V2: 1 merges with attn_merge_v2_kernel, 0 with attn_merge_kernel.  Unset it follows
+// STRATA_ATTN_DAILY (the Daily had it on by default on NVIDIA), so upstream's merger stays the default.
+int attn_merge_v2() {
+    static const int on = [] {
+        int r = daily_attn() ? 1 : 0;
+        const char* v = std::getenv("STRATA_ATTN_MERGE_V2");
+        if (v != nullptr && *v != '\0') r = std::atoi(v) != 0 ? 1 : 0;
+        return r;
+    }();
+    return on;
+}

-void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
-                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
-    if (n_q <= 0) return;
-    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
-        !pools.page_table || n_q > 65535) {
-        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
-        std::exit(1);
-    }
+// the merge of `n_q` queries' partials: v2 = 0 attn_merge_kernel, 1 attn_merge_v2_kernel (bitwise the same output)
+void launch_merge(int v2, const float* part_acc, const float* part_m, const float* part_l, int n_chunks,
+                  const int32_t* steps, float* attn, long long stride, unsigned n_head, unsigned n_q, cudaStream_t st) {
+    if (v2)
+        attn_merge_v2_kernel<kMergeU><<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, steps,
+                                                                         attn, stride);
+    else
+        attn_merge_kernel<<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn, stride);
+}
+
+// qsa_decode_attn_batch's chunk pass: `n_q` queries' partials into `scratch` (one stride each).  `kernel` < 0 is the
+// environment's choice (lane cell, gfx906 swizzle, STRATA_ATTN_DAILY / PR #540); 0 is attn_chunk_kernel and 1 PR #540's
+// kernel where it is built (the parity harness: never the environment).
+void launch_chunks_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
+                         int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, cudaStream_t st, int kernel) {
     const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                         : (pools.k_q != nullptr ? 1 : 0));
     const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
```

### 10-7

`src/kernels/cuda/qsa_decode_attn.cu` · void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int3

Anchor:
```
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
```
Block (hunk 5 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 21 lines):
```diff
     float* part_l = part_m + (size_t) n_chunks * s.n_head;
     const float scale = 1.0f / sqrtf((float) HD);
     const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
-    cudaStream_t st = (cudaStream_t) stream;
+#if defined(STRATA_ATTN_PRE75_BUILT)
+    const bool pre = kernel < 0 ? pre75_attn() : kernel == 1;
+#endif
     // S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
-    static const bool lane_cell = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
+    static const bool lane_cell_env = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
+    const bool lane_cell = kernel < 0 && lane_cell_env;
 #if defined(STRATA_HIP_GFX906)
     static const bool query_swizzle = [] {
         const char* v = std::getenv("STRATA_GFX906_ATTN_QUERY_SWIZZLE"); return v && v[0] == '1';
     }();
-    const bool use_query_swizzle = query_swizzle && !lane_cell && kv_mode == 1 && query_swizzle_gfx906_device();
+    const bool use_query_swizzle = kernel < 0 && query_swizzle && !lane_cell && kv_mode == 1 &&
+                                   query_swizzle_gfx906_device();
     static const bool query_swizzle_trace = std::getenv("STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE") != nullptr;
     static thread_local bool query_swizzle_said = false;
     if (query_swizzle_trace && !query_swizzle_said) {
```

### 10-8

`src/kernels/cuda/qsa_decode_attn.cu` · void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int3

Anchor:
```
    } else
#endif
    if (kv_mode == 3)
```
Block (hunk 6 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 57 lines):
```diff
     } else
 #endif
     if (kv_mode == 3)
-        STRATA_ATTN_CHUNK(3)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
-                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
+        STRATA_ATTN_CHUNK_SEL(3, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
+                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
+                                                                n_chunks, (int) cap, stride);
     else if (kv_mode == 2)
-        STRATA_ATTN_CHUNK(2)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
-                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
+        STRATA_ATTN_CHUNK_SEL(2, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
+                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
+                                                                n_chunks, (int) cap, stride);
     else if (kv_mode == 1)
-        STRATA_ATTN_CHUNK(1)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
-                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
+        STRATA_ATTN_CHUNK_SEL(1, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
+                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
+                                                                n_chunks, (int) cap, stride);
     else
-        STRATA_ATTN_CHUNK(0)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
-                                                        scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
-    attn_merge_kernel<<<dim3((unsigned) s.n_head, (unsigned) n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks,
-                                                                                  attn, stride);
+        STRATA_ATTN_CHUNK_SEL(0, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
+                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
+                                                                n_chunks, (int) cap, stride);
+}
+
+// qsa_decode_attn_batch's merge pass over the partials launch_chunks_batch wrote; `v2` names the merge kernel
+void launch_merge_batch(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch, float* attn,
+                        int64_t n_q, int v2, cudaStream_t st) {
+    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
+    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
+    const float* part_acc = scratch;
+    const float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
+    const float* part_l = part_m + (size_t) n_chunks * s.n_head;
+    launch_merge(v2, part_acc, part_m, part_l, n_chunks, steps, attn, stride, (unsigned) s.n_head, (unsigned) n_q, st);
+}
+
+}  // namespace
+
+void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
+                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
+    if (n_q <= 0) return;
+    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
+        !pools.page_table || n_q > 65535) {
+        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
+        std::exit(1);
+    }
+    cudaStream_t st = (cudaStream_t) stream;
+    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, st, -1);
+    launch_merge_batch(steps, cap, s, scratch, attn, n_q, attn_merge_v2(), st);
     const cudaError_t e = cudaGetLastError();
     if (e != cudaSuccess) {
         std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
```

### 10-9

`src/kernels/cuda/qsa_decode_attn.cu` · void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32

Anchor:
```
    else
        STRATA_ATTN_CHUNK(0)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
```
Block (hunk 7 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 8 lines):
```diff
     else
         STRATA_ATTN_CHUNK(0)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                         scale, part_acc, part_m, part_l, n_chunks, 0, 0);
-    attn_merge_kernel<<<(unsigned) s.n_head, HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn);
+    launch_merge(attn_merge_v2(), part_acc, part_m, part_l, n_chunks, step, attn, 0, (unsigned) s.n_head, 1u, st);
     const cudaError_t e = cudaGetLastError();
     if (e != cudaSuccess) {
         std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
```

### 10-10

`src/kernels/cuda/qsa_decode_attn.cu` · void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32

Anchor:
```
}  // namespace strata::kernels
```
Block (hunk 8 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 38 lines):
```diff
     }
 }

+// qsa_merge_daily_parity: qsa_decode_attn_batch's two passes apart, the chunk kernel and the merge named by the
+// arguments (never the environment), so the Daily's kernels and upstream's can be run on the same inputs and the
+// outputs compared bit for bit.  The engine never calls these.
+void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
+                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, int daily_chunk,
+                                 void* stream) {
+    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
+        !q || !ids || !steps || !pools.page_table || (daily_chunk != 0 && daily_chunk != 1)) {
+        std::fprintf(stderr, "qsa_decode_attn_chunks_only: unsupported geometry or missing buffers\n");
+        std::exit(1);
+    }
+    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, (cudaStream_t) stream, daily_chunk);
+    const cudaError_t e = cudaGetLastError();
+    if (e != cudaSuccess) {
+        std::fprintf(stderr, "qsa_decode_attn_chunks_only: %s\n", cudaGetErrorString(e));
+        std::exit(1);
+    }
+}
+
+void qsa_decode_attn_merge_only(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch,
+                                float* attn, int64_t n_q, int v2, void* stream) {
+    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
+        (v2 != 0 && v2 != 1) || !steps || !attn) {
+        std::fprintf(stderr, "qsa_decode_attn_merge_only: unsupported geometry or missing buffers\n");
+        std::exit(1);
+    }
+    launch_merge_batch(steps, cap, s, scratch, attn, n_q, v2, (cudaStream_t) stream);
+    const cudaError_t e = cudaGetLastError();
+    if (e != cudaSuccess) {
+        std::fprintf(stderr, "qsa_decode_attn_merge_only: %s\n", cudaGetErrorString(e));
+        std::exit(1);
+    }
+}
+
 }  // namespace strata::kernels
```

### 10-11

`src/kernels/qsa_merge_daily_parity.cpp` · (new file)

New file, 418 lines: see the patch.

## 11 df-defaults

`11-df-defaults.patch` — FABLE-STRATA-Rimetti STRATA_DF_BRANCH (CUDA su Windows) e STRATA_DF_PLE (CUDA) accesi di default come nel Daily

### 11-1

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

### 11-2

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

## 12 critpath-levers

`12-critpath-levers.patch` — FABLE-STRATA-Porta su 0.1.43 le leve CRITPATH del percorso critico (STRATA_CP_L1FIRST/S1FIRST/CHAIN_PRE/CHAIN_FUSE/RESPLAN, tutte opt-in)

### 12-1

`docs/CRITPATH.md` · (new file)

New file, 200 lines: see the patch.

### 12-2

`include/strata/core/mtp.hpp` · (file scope)

Anchor:
```
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"
```
Block (hunk 1 of `include/strata/core/mtp.hpp` in the patch, 7 lines):
```diff
 #include "strata/core/layer.hpp"
 #include "strata/core/session.hpp"
 #include "strata/kernels/sampler.hpp"
+#include "strata/kernels/verify_kernels.hpp"   // critpath: MtpChainVerdict

 #include <cuda_runtime.h>

```

### 12-3

`include/strata/core/mtp.hpp` · public:

Anchor:
```
    /// chain_tok()/chain_prob(); -1: an error.  A window's size is often decided by its first outputs (a draft below
    /// spec_min_p ends it), so the next launch need not wait for all of them.
    int chain_outputs_ready(std::string& err);
```
Block (hunk 2 of `include/strata/core/mtp.hpp` in the patch, 23 lines):
```diff
     /// chain_tok()/chain_prob(); -1: an error.  A window's size is often decided by its first outputs (a draft below
     /// spec_min_p ends it), so the next launch need not wait for all of them.
     int chain_outputs_ready(std::string& err);
-    bool chain_live() const { return chain_live_; }
+    /// A chain in flight, or a pre-launched one not yet adopted (critpath STRATA_CP_CHAIN_PRE).
+    bool chain_live() const { return chain_live_ || pre_live_; }
+
+    // ---- critpath STRATA_CP_CHAIN_PRE (docs/CRITPATH.md): the next chain's round and first `pre_steps` steps queued
+    // on the drafter's stream behind the last stage's window (`after`), their inputs derived on the device from the
+    // window's picks (`out_dev`, mtp_chain_verdict) - no host turn between the window's end and the round.  At the
+    // verdict chain_adopt takes it as the chain the host would launch (the same inputs: then only the remaining steps
+    // are launched), or lets it finish and returns 0 so the caller launches the chain as before.
+    /// `out_host`: the host view of `out_dev` (chain_adopt checks the rows the device read against the host's).
+    bool chain_prelaunch(const strata::kernels::MtpChainVerdict& v, const int32_t* out_dev, const int32_t* out_host,
+                         cudaEvent_t after, const float* src_R, int pre_steps, std::string& err);
+    /// chain_launch's arguments: 1 = adopted (the chain is live, as after chain_launch), 0 = not (nothing in flight
+    /// any more: launch it), -1 = an error.  No pre-launched chain: 0 at once.
+    int chain_adopt(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force, int n_out,
+                    int n_early, std::string& err);
+    int64_t pre_launched = 0, pre_adopted = 0, pre_redone = 0;
     const int32_t* chain_tok() const { return chain_tok_; }
     const float* chain_prob() const { return chain_prob_; }

```

### 12-4

`include/strata/core/mtp.hpp` · private:

Anchor:
```
    cudaEvent_t ev_chain_ = nullptr;
    cudaEvent_t ev_step_[8] = {};                       ///< after each of the first n_early outputs
    int steps_seen_ = 0, chain_n_ = 0, chain_early_ = 0;
```
Block (hunk 3 of `include/strata/core/mtp.hpp` in the patch, 11 lines):
```diff
     cudaEvent_t ev_chain_ = nullptr;
     cudaEvent_t ev_step_[8] = {};                       ///< after each of the first n_early outputs
     int steps_seen_ = 0, chain_n_ = 0, chain_early_ = 0;
+    bool pre_live_ = false;                             ///< critpath STRATA_CP_CHAIN_PRE: queued, not adopted
+    int pre_steps_ = 0;
+    cudaEvent_t ev_pre_ = nullptr;                      ///< after the pre-launched round and steps
+    strata::kernels::MtpChainVerdict pre_v_;
+    const int32_t* pre_out_host_ = nullptr;             ///< the window's picks as the host reads them
     int32_t chain_tok_[8] = {};
     float chain_prob_[8] = {};
     int64_t n_vocab_ = 0;
```

### 12-5

`include/strata/core/verify.hpp` · public:

Anchor:
```
    /// The watchdog's line for a pipelined verifier: in flight, layers served, the GPU's ring and flags, its events.
    void diag_pipelined(std::FILE* f, const char* name) const;
```
Block (hunk 1 of `include/strata/core/verify.hpp` in the patch, 33 lines):
```diff
     /// The watchdog's line for a pipelined verifier: in flight, layers served, the GPU's ring and flags, its events.
     void diag_pipelined(std::FILE* f, const char* name) const;

+    // ---- critical-path levers of the pipelined windows (docs/CRITPATH.md); all opt-in, the default graphs unchanged.
+    /// STRATA_CP_S1FIRST: the window in flight has a ring the host has not served yet (its GPU waits on it now).
+    bool pl_rang() const {
+        return fl_active_ && fl_k_ < fl_total_ && !ar_on() && *(volatile const uint32_t*) h_seq_ >= (uint32_t) (fl_k_ + 1);
+    }
+    /// ... has layers left for the host to serve (its GPU will ring again).
+    bool pl_serving() const { return fl_active_ && fl_k_ < fl_total_ && !ar_on(); }
+    int64_t pl_served() const { return fl_k_; }
+    /// STRATA_CP_CHAIN_PRE: the event after the last pipelined window's graph, and its picks on the device (the mapped
+    /// argmax rows), for a chain launched behind it on the device.
+    cudaEvent_t done_event() const { return ev_done_; }
+    const int32_t* out_dev() const { return m_out_; }
+    const int32_t* out_host() const { return h_out_; }
+    /// pl_finish samples on the host (a sampled request, penalties, a logit bias): the graph's rows are not the picks.
+    bool host_sampling() const {
+        const bool sampled = !sampling_.greedy && sampling_.temperature > 0.0f;
+        return head_sampling_ && (sampled || hist_d_ != nullptr || sampling_.logit_bias != nullptr);
+    }
+    /// STRATA_CP_RESPLAN: the routes of a group the device planned (no host service) are still counted - the adaptive
+    /// tier ranks its victims by them.  Called with the pool's `user` instead of the pool.
+    using CpNoteFn = void (*)(void* user, const int32_t* ids, int64_t n, int64_t layer);
+    void set_cp_note(CpNoteFn fn) { cp_note_ = fn; }
+    int64_t cp_planned = 0;      ///< STRATA_CP_RESPLAN: layer-groups the device planned (no host service)
+    int64_t cp_host = 0;         ///< ... layer-groups of those layers the host served (a route outside VRAM)
+    int64_t cp_uploads = 0;      ///< ... residency table uploads before a launch (the host table had moved)
+    int cp_layers() const { int n = 0; for (uint8_t m : cp_mask_) n += m; return n; }
+
     /// Measurement hook (STRATA_LOGPOS): after run(), write one line per row t of the last window's head -
     /// "pos target logprob top top_logprob hit extra_logprob target_logprob_without_extra" - where row t is the
     /// distribution at pos0 + t, targets[t] is the token at pos0 + t + 1, extra_logprob is the log-probability of
```

### 12-6

`include/strata/core/verify.hpp` · private:

Anchor:
```
    double fl_since_ms_ = 0, fl_flush_ms_ = 0, fl_launch_ms_ = 0;
    int32_t pl_prev_[2] = {-1, -1};
    std::vector<uint32_t> pl_ple_rows_;   ///< the window's PLE rows (T x PLE_N_HEADS), gathered when layer 0 is served
```
Block (hunk 2 of `include/strata/core/verify.hpp` in the patch, 24 lines):
```diff
     double fl_since_ms_ = 0, fl_flush_ms_ = 0, fl_launch_ms_ = 0;
     int32_t pl_prev_[2] = {-1, -1};
     std::vector<uint32_t> pl_ple_rows_;   ///< the window's PLE rows (T x PLE_N_HEADS), gathered when layer 0 is served
+    // critpath STRATA_CP_RESPLAN (capture_all): the pipelined graphs plan on the device the groups of the layers in
+    // cp_mask_ whose routes are all in VRAM, and the host never serves those (see record_window, service)
+    bool cp_rec_ = false;                 ///< record_window is capturing the RESPLAN variant
+    bool cp_win_ = false;                 ///< the window in flight runs it
+    std::vector<uint8_t> cp_mask_;        ///< per layer of [lb_, le_): planned on the device when every route hits
+    cudaGraphExec_t exec_cp_[9] = {};
+    uint32_t* h_cpmap_ = nullptr; uint32_t* m_cpmap_ = nullptr;   ///< per ring: 1 device-planned, 2 served by the host
+    int32_t* h_cpids_ = nullptr; int32_t* m_cpids_ = nullptr;     ///< per ring: the planned group's routed ids
+    static constexpr int kCpRings = 2 * 64 + 2;
+    CpNoteFn cp_note_ = nullptr;
+    std::vector<int32_t> cp_shadow_;      ///< the host table's [lb_, le_) rows as last queued to the device
+    std::vector<uint64_t> cp_off_shadow_; ///< the slot offsets as last queued (an elastic arena moves them)
+    unsigned long long* cp_off_d_ = nullptr;   ///< the device copy, room for every (layer, expert) slot
+    uint8_t* cp_pin_ = nullptr;           ///< pinned copies for the uploads: the rows, then the offsets
+    cudaEvent_t cp_up_ev_ = nullptr;      ///< after the last upload (the pinned copies are free again)
+    bool cp_sync_res(std::string& err);   ///< the stage's residency rows to the device when the host's moved
+    bool cp_prepare(std::string& err);    ///< capture_all: the mask and the RESPLAN graphs (once)
+    bool capture_into(int T, cudaGraphExec_t& exec_t, std::string& err);
     bool capture_commit_batch(const int* rows, int S, int hbase, std::string& err);
     bool stage_batch(const int* rows, int S, int hbase, const int32_t* tokens, const int64_t* pos, std::string& err);
     strata::kernels::SamplerParams sampling_ = [] {
```

### 12-7

`include/strata/kernels/elementwise.hpp` · void doorbell_publish_value(const float* x, const int32_t* ids, const float* wei

Anchor:
```
                            void* stream);
void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k,
                          float* x_out, int32_t* ids_out, uint32_t* d_seq, void* stream);
```
Block (hunk 1 of `include/strata/kernels/elementwise.hpp` in the patch, 13 lines):
```diff
                             void* stream);
 void doorbell_publish_res(const float* x, const int32_t* ids, const int32_t* d_res, int n_expert, int64_t n, int64_t k,
                           float* x_out, int32_t* ids_out, uint32_t* d_seq, void* stream);
+/// critpath STRATA_CP_RESPLAN (docs/CRITPATH.md): the pipelined window's doorbell, ring STORED as `value`.  `skip` null:
+/// doorbell_publish_value.  Else, when *skip == value (resident_plan planned the group on the device: every routed
+/// expert in VRAM) only the k routed ids go to `ids_log` and map[value] = 1 - the host has nothing to serve; otherwise
+/// the payload as doorbell_publish_value and map[value] = 2.  The map word is visible before the ring.
+void doorbell_publish_cp(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
+                         int32_t* ids_out, float* weights_out, uint32_t* d_seq, uint32_t value, const uint32_t* skip,
+                         uint32_t* map, int32_t* ids_log, void* stream);
 /// SYCL: publishes launched or recorded while `on` store the payload with plain stores instead of uncached ones, for
 /// a caller that reads the payload only after the publishing kernel has ended (the stepped verify window).
 void doorbell_plain_payload(bool on);
```

### 12-8

`include/strata/kernels/verify_kernels.hpp` · void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_e

Anchor:
```
void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream);
/// The draft chain's next input: R_dst[:] = R_src[row], tok_dst[0] = ids[row], out[j] = ids[row], with
/// row = *row_dev (device memory).  `out` may be mapped host memory.
```
Block (hunk 1 of `include/strata/kernels/verify_kernels.hpp` in the patch, 29 lines):
```diff
 void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream);
 /// The draft chain's next input: R_dst[:] = R_src[row], tok_dst[0] = ids[row], out[j] = ids[row], with
 /// row = *row_dev (device memory).  `out` may be mapped host memory.
+/// `force` (critpath STRATA_CP_CHAIN_FUSE, mapped host memory): force_token folded in - *tok_dst = force[j] when it is
+/// >= 0, the pick otherwise; out[j] stays the pick.  The same words as mtp_select then force_token.
 void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                 int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs = nullptr,
-                float* out_p = nullptr);
+                float* out_p = nullptr, const int32_t* force = nullptr);
 /// --pipeline-windows, the drafter's chain teacher forced: `*tok = force[j]` when force[j] >= 0 (`force` is mapped
 /// host memory, read when the kernel runs), else `*tok` is left as it is.
 void force_token(int32_t* tok, const int32_t* force, int j, void* stream);
+/// critpath STRATA_CP_CHAIN_PRE (docs/CRITPATH.md): the next chain's inputs from the verified window's picks, on the
+/// device, so the round can follow the last stage's graph with no host turn.  `out` = the window's picks (the last
+/// stage's mapped argmax rows); row t's token is repl[t] when >= 0 (the forced bench), else out[t].  a = the accepted
+/// drafts (a_tok[t + 1] == token t), on = b_launched && a == T - 1 && b_tok[0] == token T - 1.  Writes what
+/// MtpDrafter::chain_launch stages on the host: tok[0, T), the step/pos rows of cells p + t (t < T), of p + a (row
+/// 2 max_t - 1) and of p + a + j (row max_t + j - 1, 1 <= j <= max_t - 2), row = {a, 0}, force[16] = b_tok[1 + j]
+/// for j < b_T - 1 when on, else -1.  All outputs mapped host memory (device aliases).
+struct MtpChainVerdict {
+    int T = 0, max_t = 0, nh = 0, b_launched = 0, b_T = 0;
+    long long p = 0;
+    int32_t a_tok[8] = {}, repl[8] = {}, b_tok[8] = {};
+};
+void mtp_chain_verdict(const MtpChainVerdict& v, const int32_t* out, int32_t* tok, int32_t* step, int32_t* pos,
+                       int32_t* row, int32_t* force, void* stream);
 /// Row *row_dev of a, b and c (a_n, b_n, c_n floats a row) copied to their row 0 (the draft layer's rest runs on
 /// row 0).  Graph-capturable: the row is read on the device.
 void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b, int64_t b_n, float* c, int64_t c_n,
```

### 12-9

`src/core/mtp.cpp` · bool mtp_catchup_all() {

Anchor:
```
    }();
    return on;
}
```
Block (hunk 1 of `src/core/mtp.cpp` in the patch, 20 lines):
```diff
     }();
     return on;
 }
+// critpath STRATA_CP_CHAIN_FUSE=1 (opt-in; docs/CRITPATH.md): the chain's graphs with fewer launches, every word the
+// same - the round copies every step's staged row at once (the host stages them all before the round, in chain_launch
+// and draft() alike, and chain_prelaunch's device verdict writes them all), so a step no longer copies its own; and
+// force_token rides in mtp_select.  The argmax graphs only; coupled sampling keeps its own.
+bool cp_chain_fuse() {
+    static const bool on = [] {
+        const char* v = std::getenv("STRATA_CP_CHAIN_FUSE");
+        const bool f = v != nullptr && v[0] != '\0' && v[0] != '0';
+        if (f) std::fprintf(stderr, "strata critpath: STRATA_CP_CHAIN_FUSE: the round copies every step row, mtp_select "
+                                    "forces the token (2 launches fewer a step)\n");
+        return f;
+    }();
+    return on;
+}
 constexpr int GGML_Q8_0 = 8;
 constexpr int GGML_Q4_0 = 2;
 using Clock = std::chrono::steady_clock;
```

### 12-10

`src/core/mtp.cpp` · MtpDrafter::~MtpDrafter() {

Anchor:
```
    if (owns_draft_head_ && dhead_) { strata::kernels::native_q6_k_unpack(dhead_); cudaFree(dhead_); }
    if (owns_draft_head_ && dvocab_) cudaFree(dvocab_);
    if (ev_chain_) cudaEventDestroy(ev_chain_);
```
Block (hunk 2 of `src/core/mtp.cpp` in the patch, 7 lines):
```diff
     if (owns_draft_head_ && dhead_) { strata::kernels::native_q6_k_unpack(dhead_); cudaFree(dhead_); }
     if (owns_draft_head_ && dvocab_) cudaFree(dvocab_);
     if (ev_chain_) cudaEventDestroy(ev_chain_);
+    if (ev_pre_) cudaEventDestroy(ev_pre_);
     for (cudaEvent_t e : ev_step_) if (e) cudaEventDestroy(e);
     void* hosts[] = {h_tok_, h_step_, h_pos_, h_row_, h_out_, h_prob_, h_force_};
     for (void* h : hosts) if (h) cudaFreeHost(h);
```

### 12-11

`src/core/mtp.cpp` · bool MtpDrafter::capture_round(int T, bool coupled, std::string& err) {

Anchor:
```
    copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + ra * NH, m_pos_ + ra * NH, NH, cs_);
#else   // the draft round's seven inputs in one launch
```
Block (hunk 3 of `src/core/mtp.cpp` in the patch, 14 lines):
```diff
     copy_i32_from_mapped(step_ + ra * 4, m_step_ + ra * 4, 4, cs_);
     copy_i32_from_mapped(pos_ + ra * NH, m_pos_ + ra * NH, NH, cs_);
 #else   // the draft round's seven inputs in one launch
+    // critpath STRATA_CP_CHAIN_FUSE: rows max_t .. 2 max_t - 1 (every step's and output 0's, ra the last of them)
+    const bool fuse = cp_chain_fuse() && !coupled;
+    const int64_t r0 = fuse ? max_t_ : ra, nr = fuse ? max_t_ : 1;
     const MappedCopy in[7] = {{tok_, m_tok_, T}, {step_, m_step_, (int64_t) 2 * T * 4}, {pos_, m_pos_, 2 * T * NH},
-                              {row_, m_row_, 2}, {Rin_, window_R_, T * HCN}, {step_ + ra * 4, m_step_ + ra * 4, 4},
-                              {pos_ + ra * NH, m_pos_ + ra * NH, NH}};
+                              {row_, m_row_, 2}, {Rin_, window_R_, T * HCN}, {step_ + r0 * 4, m_step_ + r0 * 4, nr * 4},
+                              {pos_ + r0 * NH, m_pos_ + r0 * NH, nr * NH}};
     copy_from_mapped_multi(in, 7, cs_);
 #endif
     ok = record_front(T, 0, cs_, err);
```

### 12-12

`src/core/mtp.cpp` · bool MtpDrafter::capture_round(int T, bool coupled, std::string& err) {

Anchor:
```
        ok = record_rest(ra, cs_, err);
        coupled_rec_ = false;
    }
```
Block (hunk 4 of `src/core/mtp.cpp` in the patch, 11 lines):
```diff
         ok = record_rest(ra, cs_, err);
         coupled_rec_ = false;
     }
-    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_);
-    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, 0, cs_);   // chain_launch: step 1's input
+    const bool fold = cp_chain_fuse() && force_on_ && !coupled;   // critpath: force_token inside mtp_select
+    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_, fold ? m_force_ : nullptr);
+    if (ok && force_on_ && !coupled && !fold) force_token(tok_, m_force_, 0, cs_);   // chain_launch: step 1's input
     return finish_capture(cs_, ok, exec, coupled ? "round (coupled)" : "round", err);
 }

```

### 12-13

`src/core/mtp.cpp` · bool MtpDrafter::capture_step(int j, bool coupled, std::string& err) {

Anchor:
```
    copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
    copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
#else
```
Block (hunk 5 of `src/core/mtp.cpp` in the patch, 24 lines):
```diff
     copy_i32_from_mapped(step_ + row * 4, m_step_ + row * 4, 4, cs_);
     copy_i32_from_mapped(pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head, cs_);
 #else
-    const MappedCopy in[2] = {{step_ + row * 4, m_step_ + row * 4, 4},
-                              {pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head}};
-    copy_from_mapped_multi(in, 2, cs_);
+    if (!(cp_chain_fuse() && !coupled)) {   // critpath STRATA_CP_CHAIN_FUSE: the round copied this row already
+        const MappedCopy in[2] = {{step_ + row * 4, m_step_ + row * 4, 4},
+                                  {pos_ + row * g_->n_head, m_pos_ + row * g_->n_head, g_->n_head}};
+        copy_from_mapped_multi(in, 2, cs_);
+    }
 #endif
     coupled_rec_ = coupled;
     coupled_j_ = j;
     bool ok = record_forward(1, row, cs_, err);
     coupled_rec_ = false;
-    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_);
-    if (ok && force_on_ && !coupled) force_token(tok_, m_force_, j, cs_);   // chain_launch: step j+1's input
+    const bool fold = cp_chain_fuse() && force_on_ && !coupled;   // critpath: force_token inside mtp_select
+    if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_, fold ? m_force_ : nullptr);
+    if (ok && force_on_ && !coupled && !fold) force_token(tok_, m_force_, j, cs_);   // chain_launch: step j+1's input
     return finish_capture(cs_, ok, exec, coupled ? "step (coupled)" : "step", err);
 }

```

### 12-14

`src/core/mtp.cpp` · bool MtpDrafter::draft(int T, const int32_t* tokens, int64_t p, int a, int32_t*

Anchor:
```
                       float* probs, float min_p, int* n_drafts) {
    const OnDevice on_device(device_);
    if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
```
Block (hunk 6 of `src/core/mtp.cpp` in the patch, 8 lines):
```diff
                        float* probs, float min_p, int* n_drafts) {
     const OnDevice on_device(device_);
     if (T < 1 || T > max_t_ || a < 0 || a >= T) { err = "mtp: draft arguments out of range"; return false; }
-    if (chain_live_) { err = "mtp: a pipelined chain is still in flight"; return false; }
+    if (chain_live_ || pre_live_) { err = "mtp: a pipelined chain is still in flight"; return false; }
     // #783 PR-i (stuchapin909): the round runs for the cells up to the accepted row a (T = a + 1): the K/V of the rejected
     // rows is not caught up - no later read reaches a cell past the one being drafted and the next round writes it first
     if (!mtp_catchup_all()) T = a + 1;
```

### 12-15

`src/core/mtp.cpp` · int MtpDrafter::chain_outputs_ready(std::string& err) {

Anchor:
```
}

int MtpDrafter::chain_poll(std::string& err) {
```
Block (hunk 7 of `src/core/mtp.cpp` in the patch, 16 lines):
```diff
 }

 int MtpDrafter::chain_poll(std::string& err) {
-    if (!chain_live_) return 1;
+    if (!chain_live_) {
+        if (!pre_live_) return 1;
+        const OnDevice on_device(device_);   // critpath: a pre-launched chain nobody adopted (the request ended)
+        const cudaError_t q = cudaEventQuery(ev_pre_);
+        if (q == cudaErrorNotReady) return 0;
+        pre_live_ = false;
+        if (q != cudaSuccess) { err = std::string("mtp chain: ") + cudaGetErrorString(q); return -1; }
+        return 1;
+    }
     const OnDevice on_device(device_);
     const cudaError_t q = cudaEventQuery(ev_chain_);
     if (q == cudaErrorNotReady) return 0;
```

### 12-16

`src/core/mtp.cpp` · int MtpDrafter::chain_poll(std::string& err) {

Anchor:
```
    return 1;
}
```
Block (hunk 8 of `src/core/mtp.cpp` in the patch, 104 lines):
```diff
     return 1;
 }

+bool MtpDrafter::chain_prelaunch(const strata::kernels::MtpChainVerdict& v, const int32_t* out_dev,
+                                 const int32_t* out_host, cudaEvent_t after, const float* src_R, int pre_steps,
+                                 std::string& err) {
+    const OnDevice on_device(device_);
+    if (chain_live_ || pre_live_) { err = "mtp: a chain is already in flight"; return false; }
+    if (!force_on_ || ev_chain_ == nullptr || h_force_ == nullptr) { err = "mtp: chains need prepare_chain"; return false; }
+    const int T = v.T;
+    if (T < 1 || T > max_t_ || pre_steps < 0 || pre_steps > max_t_ - 2 || v.b_T < 0 || v.b_T > 8 || out_dev == nullptr ||
+        out_host == nullptr || after == nullptr || round_exec_[T] == nullptr) {
+        err = "mtp: pre-launched chain arguments out of range";
+        return false;
+    }
+    for (int j = 1; j <= pre_steps; ++j)
+        if (step_exec_[j] == nullptr) { err = "mtp: chain graphs not prepared"; return false; }
+    if (ev_pre_ == nullptr && cudaEventCreateWithFlags(&ev_pre_, cudaEventDisableTiming) != cudaSuccess) {
+        err = "mtp: event creation failed";
+        return false;
+    }
+    const Clock::time_point t0 = Clock::now();
+    strata::kernels::MtpChainVerdict w = v;
+    w.max_t = max_t_;
... 80 more lines: see 12-critpath-levers.patch
```

### 12-17

`src/core/verify.cpp` · Verifier::~Verifier() {

Anchor:
```
        if (e) cudaGraphExecDestroy(e);
    for (auto& e : exec_nr_)
        if (e) cudaGraphExecDestroy(e);
```
Block (hunk 1 of `src/core/verify.cpp` in the patch, 13 lines):
```diff
         if (e) cudaGraphExecDestroy(e);
     for (auto& e : exec_nr_)
         if (e) cudaGraphExecDestroy(e);
+    for (auto& e : exec_cp_)
+        if (e) cudaGraphExecDestroy(e);
+    if (cp_up_ev_) { cudaEventSynchronize(cp_up_ev_); cudaEventDestroy(cp_up_ev_); }
+    if (cp_pin_) cudaFreeHost(cp_pin_);
+    if (cp_off_d_) cudaFree(cp_off_d_);
+    if (h_cpmap_) cudaFreeHost(h_cpmap_);
+    if (h_cpids_) cudaFreeHost(h_cpids_);
     if (commit_exec_) cudaGraphExecDestroy(commit_exec_);
     for (auto& kv : exec_bm_)
         if (kv.second) cudaGraphExecDestroy(kv.second);
```

### 12-18

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
            resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                          hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                          plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, nullptr, 0, cs, m_plan_err_);
```
Block (hunk 2 of `src/core/verify.cpp` in the patch, 25 lines):
```diff
             resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
                           hits_.cache_base, slot_off_d_, (long long) hits_.blob,
                           plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, nullptr, 0, cs, m_plan_err_);
+        } else if (cp_rec_) {
+            // critpath STRATA_CP_RESPLAN: a layer of cp_mask_ plans its group on the device when every route is in VRAM
+            // (resident_plan writes skip_[grp] = ring); its doorbell then carries only the routed ids (the host counts
+            // their usage and serves nothing) and its waits below return at once.  Every ring is STORED (a planned
+            // group does not ring the host), so the host reads the ring's map word before it serves a masked layer.
+            const uint32_t ring = (uint32_t) ((l - lb_) * G + grp + 1);
+            const bool cpl = cp_mask_[(size_t) (l - lb_)] != 0;
+            if (cpl)
+                resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
+                              hits_.cache_base, hits_.slot_off != nullptr ? cp_off_d_ : nullptr, (long long) hits_.blob,
+                              plan_ + (size_t) grp * (size_t) (plan_i32_ + 16), (long long) max_t_ * K, skip_ + grp,
+                              ring, cs);
+            doorbell_publish_cp(xm, ids_ + tb * K, w_ + tb * K, (int64_t) n * N, (int64_t) n * K, m_x_ + tb * N,
+                                m_ids_ + tb * K, m_w_ + tb * K, m_seq_, ring, cpl ? skip_ + grp : nullptr,
+                                cpl ? m_cpmap_ : nullptr, m_cpids_ + (size_t) ring * (size_t) (max_t_ * K), cs);
+            if (sh_fork_late) {
+                cudaEventRecord(ev_fork_, cs);
+                cudaStreamWaitEvent(sh_cs_, ev_fork_, 0);
+            }
         } else {
             if (device_plan_)   // E-6: every routed expert resident: this group's plan without the host
                 resident_plan(ids_ + tb * K, n * (int) K, (int) K, hits_.d_res + l * g.n_expert, (int) g.n_expert,
```

### 12-19

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
            }
        };
        bool hits_in_combine = false;
```
Block (hunk 3 of `src/core/verify.cpp` in the patch, 36 lines):
```diff
             }
         };
         bool hits_in_combine = false;
-        if (ar_on()) {
+        // critpath STRATA_CP_RESPLAN: a masked layer's group waits like E-6's (returns at once when the device planned
+        // it) but copies only the CPU's rows, as the host-planned path does: a planned group's plan lists every entry as
+        // a GPU row, so copy_rows_from_mapped writes zeros and reads nothing over PCIe (E-6 copied all n*k rows).
+        const bool cpl = cp_rec_ && !ar_on() && cp_mask_[(size_t) (l - lb_)] != 0;
+        if (cpl) {
+            wait_flag_ge_or(m_flagA_, ring, skip_ + grp, cs);
+            copy_i32_from_mapped_unless(pl, m_plan_ + (size_t) grp * (size_t) plan_i32_, plan_i32_, skip_ + grp, ring, cs);
+            stamp(l, 19, grp);
+            grouped(p_ptr, p_start, p_counts, 0, hit_out);
+            stamp(l, 20, grp);
+            wait_flag_ge_or(m_flagB_, ring, skip_ + grp, cs);
+            if (sink_.pcie_mode == 2) {
+                const int64_t per = G == 2 ? kStagingBlobs / 2 : kStagingBlobs;
+                uint8_t* stage = staging_ + (size_t) (grp * per) * lay.max_blob;
+                fetch_blobs(p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), (int) per, cs);
+                rebase_ptrs((unsigned long long*) p_ptr2, p_counts + 2, stage, (int64_t) lay.blob_bytes(l), cs);
+            }
+            stamp(l, 21, grp);
+            grouped(p_ptr2, p_start2, p_counts + 2, kPcieGroupRows, hit_out);
+            stamp(l, 22, grp);
+            wait_flag_ge_or(m_flag_, ring, skip_ + grp, cs);
+            stamp(l, 23, grp);
+            if (dec_batch)
+                copy_rows_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (int64_t) n * K, N, p_dst, p_counts + 1, cs);
+            else
+                copy_or_zero_from_mapped(parts_out, m_ymiss_ + (size_t) tb * K * N, (long long) n * K * N, skip_ + grp,
+                                         ring, cs);
+            moe_hit_add(parts_out, hit_out, p_dst, p_counts + 1, cap, N, cs);
+        } else if (ar_on()) {
             stamp(l, 19, grp);
             grouped(p_ptr, p_start, p_counts, 0, parts_out);
             stamp(l, 20, grp);
```

### 12-20

`src/core/verify.cpp` · void Verifier::refresh_ar() {

Anchor:
```
}

bool Verifier::capture(int T, std::string& err) {
```
Block (hunk 4 of `src/core/verify.cpp` in the patch, 12 lines):
```diff
 }

 bool Verifier::capture(int T, std::string& err) {
-    cudaGraphExec_t& exec_t = ar_off_ ? exec_nr_[T] : exec_[T];
+    return capture_into(T, ar_off_ ? exec_nr_[T] : exec_[T], err);
+}
+
+// the window graph into `exec_t` (exec_ / exec_nr_ by ar_off_, or exec_cp_ while cp_rec_ is set)
+bool Verifier::capture_into(int T, cudaGraphExec_t& exec_t, std::string& err) {
     if (exec_t != nullptr) return true;
     {   // said before the capture: a process that exits inside it (#1275: Windows, 313 MiB free) leaves this line as the trace
         size_t free_b = 0, total_b = 0;
```

### 12-21

`src/core/verify.cpp` · void Verifier::stage_inputs(int T, const int32_t* tokens, int64_t pos0) {

Anchor:
```
    std::atomic_thread_fence(std::memory_order_seq_cst);
    last_t_ = T;
    last_pos0_ = pos0;
```
Block (hunk 5 of `src/core/verify.cpp` in the patch, 7 lines):
```diff
     *(volatile uint32_t*) h_flag_ = 0;
     *(volatile uint32_t*) h_flagA_ = 0;
     *(volatile uint32_t*) h_flagB_ = 0;
+    if (h_cpmap_ != nullptr) std::memset(h_cpmap_, 0, (size_t) kCpRings * 4);   // critpath STRATA_CP_RESPLAN
     std::atomic_thread_fence(std::memory_order_seq_cst);
     last_t_ = T;
     last_pos0_ = pos0;
```

### 12-22

`src/core/verify.cpp` · bool Verifier::capture_all(std::string& err) {

Anchor:
```
        cudaGetLastError();
    }
    pl_ple_rows_.assign((size_t) strata::kernels::kVerifyMaxT * strata::kernels::PLE_N_HEADS, 0u);
```
Block (hunk 6 of `src/core/verify.cpp` in the patch, 125 lines):
```diff
         cudaGetLastError();
     }
     pl_ple_rows_.assign((size_t) strata::kernels::kVerifyMaxT * strata::kernels::PLE_N_HEADS, 0u);
+    if (!cp_prepare(err)) return false;
+    return true;
+}
+
+// critpath STRATA_CP_RESPLAN=1 (opt-in; docs/CRITPATH.md): the pipelined windows' second graph set.  A layer of this
+// stage whose experts are resident at least STRATA_CP_RESPLAN_MIN (default 1.0: all 512) when the graphs are captured
+// plans each group on the device when every route of the group is in VRAM (resident_plan, E-6), and then neither rings
+// the host nor waits for it; the host skips it (service) and only counts its routes' usage (set_cp_note).  A group with
+// a route outside VRAM rings and is served as always.  The device table follows the host's before every launch
+// (cp_sync_res): an expert the tier evicts is -1 on the host before its slot is reused, so a later window never plans
+// a stale slot; a window launched before is covered by the tier's fence, as the doorbell path is.
+namespace {
+double cp_resplan_min() {
+    static const double v = [] {
+        const char* e = std::getenv("STRATA_CP_RESPLAN");
+        if (e == nullptr || e[0] == 0 || e[0] == '0') return -1.0;
+        const char* m = std::getenv("STRATA_CP_RESPLAN_MIN");
+        const double x = m != nullptr ? std::atof(m) : 1.0;
+        return x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x;
+    }();
+    return v;
... 101 more lines: see 12-critpath-levers.patch
```

### 12-23

`src/core/verify.cpp` · bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string

Anchor:
```
    fl_total_ = (le_ - lb_) * G;
    fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
    if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
```
Block (hunk 7 of `src/core/verify.cpp` in the patch, 14 lines):
```diff
     fl_total_ = (le_ - lb_) * G;
     fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
     if (trace_h_ != nullptr) std::memset(trace_h_, 0, trace_n_ * 8);   // #649: this window's breadcrumbs only
+    // critpath STRATA_CP_RESPLAN: the RESPLAN graph, behind the residency rows the host holds now
+    cp_win_ = exec_cp_[T] != nullptr;
+    if (cp_win_ && !cp_sync_res(err)) return false;
     std::atomic_thread_fence(std::memory_order_seq_cst);
     trace_ev("WINDOW (pipelined)", -1, -1, pos0 * 16 + T);
     ms_host += ms_since(t0);
-    const cudaError_t le = cudaGraphLaunch(exec_[T], cs_);
+    const cudaError_t le = cudaGraphLaunch(cp_win_ ? exec_cp_[T] : exec_[T], cs_);
     trace_ev("LAUNCHED", -1, -1, (int64_t) le);
     if (le != cudaSuccess) { err = std::string("verify: launch: ") + cudaGetErrorString(le); return false; }
     if (fl_prof_) cudaMemcpyAsync(prof_pin_, prof_, prof_h_.size() * 8, cudaMemcpyDeviceToHost, cs_);
```

### 12-24

`src/core/verify.cpp` · int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la

Anchor:
```
        const Clock::time_point b = Clock::now();
        ms_wait += now_ms() - fl_since_ms_;
        const int grp = (int) (fl_k_ % G);
```
Block (hunk 8 of `src/core/verify.cpp` in the patch, 22 lines):
```diff
             }
             return 0;
         }
+        if (cp_win_ && cp_mask_[(size_t) (l - lb_)]) {   // critpath STRATA_CP_RESPLAN: did the device plan this group?
+            const uint32_t m = *(volatile uint32_t*) (h_cpmap_ + want);
+            if (m == 1) {   // yes: its GPU waits for nothing; its routes still count for the adaptive tier
+                const int grp = (int) (fl_k_ % G);
+                const int n = gte[grp] - gtb[grp];
+                if (cp_note_ != nullptr && pool != nullptr)
+                    cp_note_(user, h_cpids_ + (size_t) want * (size_t) (max_t_ * ss.k), (int64_t) n * ss.k, l);
+                ++cp_planned;
+                ++fl_k_;
+                const double now = now_ms();
+                ms_wait += now - fl_since_ms_;
+                fl_since_ms_ = fl_flush_ms_ = now;
+                continue;
+            }
+            ++cp_host;
+        }
         const Clock::time_point b = Clock::now();
         ms_wait += now_ms() - fl_since_ms_;
         const int grp = (int) (fl_k_ % G);
```

### 12-25

`src/core/verify.cpp` · bool Verifier::pl_commit_async(int n_keep, std::string& err) {

Anchor:
```
void Verifier::absorb_stats(Verifier& o) {
    ms_wait += o.ms_wait; ms_pool += o.ms_pool; ms_host += o.ms_host; ms_commit += o.ms_commit;
    windows += o.windows;
```
Block (hunk 9 of `src/core/verify.cpp` in the patch, 8 lines):
```diff
 void Verifier::absorb_stats(Verifier& o) {
     ms_wait += o.ms_wait; ms_pool += o.ms_pool; ms_host += o.ms_host; ms_commit += o.ms_commit;
     windows += o.windows;
+    cp_planned += o.cp_planned; cp_host += o.cp_host; cp_uploads += o.cp_uploads;   // critpath STRATA_CP_RESPLAN
+    o.cp_planned = o.cp_host = o.cp_uploads = 0;
     o.ms_wait = o.ms_pool = o.ms_host = o.ms_commit = 0;
     o.windows = 0;
     for (int k = 0; k < 2; ++k)
```

### 12-26

`src/kernels/cuda/elementwise.cu` · void doorbell_publish_value(const float* x, const int32_t* ids, const float* wei

Anchor:
```
                                                                          ids_out, weights_out, d_seq, value);
}
```
Block (hunk 1 of `src/kernels/cuda/elementwise.cu` in the patch, 40 lines):
```diff
                                                                           ids_out, weights_out, d_seq, value);
 }

+// critpath STRATA_CP_RESPLAN: see elementwise.hpp.  The unplanned branch is doorbell_publish_value_kernel's body.
+__global__ void doorbell_publish_cp_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
+                                           const float* __restrict__ w, int n, int k, float* x_out, int32_t* ids_out,
+                                           float* w_out, uint32_t* seq, uint32_t value, const uint32_t* skip,
+                                           uint32_t* map, int32_t* ids_log) {
+    const bool planned = skip != nullptr && *(const volatile uint32_t*) skip == value;
+    if (planned) {
+        if ((int) threadIdx.x < k) ids_log[threadIdx.x] = ids[threadIdx.x];
+    } else {
+        for (int i = threadIdx.x; i < n; i += blockDim.x) x_out[i] = x[i];
+        if ((int) threadIdx.x < k) { ids_out[threadIdx.x] = ids[threadIdx.x]; w_out[threadIdx.x] = w[threadIdx.x]; }
+    }
+    __threadfence_system();
+    __syncthreads();
+    if (threadIdx.x == 0) {
+        __threadfence_system();
+        if (map != nullptr) {
+            *(volatile uint32_t*) (map + value) = planned ? 1u : 2u;
+            __threadfence_system();
+        }
+        *(volatile uint32_t*) seq = value;
+        __threadfence_system();
+    }
+}
+
+void doorbell_publish_cp(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
+                         int32_t* ids_out, float* weights_out, uint32_t* d_seq, uint32_t value, const uint32_t* skip,
+                         uint32_t* map, int32_t* ids_log, void* stream) {
+    if (k > 1024) { std::fprintf(stderr, "doorbell_publish_cp: k too large\n"); std::exit(1); }
+    doorbell_publish_cp_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, ids, weights, (int) n, (int) k, x_out, ids_out,
+                                                                       weights_out, d_seq, value, skip, map, ids_log);
+    check_launch("doorbell_publish_cp");
+}
+
 void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                       int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
     if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
```

### 12-27

`src/kernels/cuda/verify_kernels.cu` · __global__ void row_top_prob_kernel(const float* __restrict__ logits, int n_voca

Anchor:
```

__global__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                  const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
```
Block (hunk 1 of `src/kernels/cuda/verify_kernels.cu` in the patch, 18 lines):
```diff

 __global__ void mtp_select_kernel(const float* __restrict__ R_src, int64_t stride, const int32_t* __restrict__ ids,
                                   const int32_t* __restrict__ row_dev, float* __restrict__ R_dst,
-                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p) {
+                                  int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p,
+                                  const int32_t* force) {
     const int row = *row_dev;
     for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < stride; i += (int64_t) gridDim.x * blockDim.x)
         R_dst[i] = R_src[(size_t) row * stride + i];
     __syncthreads();
     if (blockIdx.x == 0 && threadIdx.x == 0) {
         const int32_t tok = ids[row];
-        *tok_dst = tok;
+        const int32_t f = force != nullptr ? ((const volatile int32_t*) force)[j] : -1;   // critpath: force_token folded in
+        *tok_dst = f >= 0 ? f : tok;
         if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
         __threadfence_system();
         if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
```

### 12-28

`src/kernels/cuda/verify_kernels.cu` · __global__ void force_token_kernel(int32_t* tok, const int32_t* force, int j) {

Anchor:
```
    if (f >= 0) *tok = f;
}
```
Block (hunk 2 of `src/kernels/cuda/verify_kernels.cu` in the patch, 41 lines):
```diff
     if (f >= 0) *tok = f;
 }

+// critpath STRATA_CP_CHAIN_PRE: see mtp_chain_verdict in verify_kernels.hpp.  The rows are MtpDrafter::chain_launch's
+// put(): {cell, cell + 1, (cell + 1) / 4, cell + 1} and nh copies of cell.
+__global__ void mtp_chain_verdict_kernel(MtpChainVerdict v, const int32_t* out, int32_t* tok, int32_t* step,
+                                         int32_t* pos, int32_t* row, int32_t* force) {
+    __shared__ int32_t s_o[8];
+    __shared__ int s_a, s_on;
+    const int i = threadIdx.x;
+    if (i == 0) {
+        for (int t = 0; t < v.T; ++t) s_o[t] = v.repl[t] >= 0 ? v.repl[t] : ((const volatile int32_t*) out)[t];
+        int a = 0;
+        while (a < v.T - 1 && v.a_tok[a + 1] == s_o[a]) ++a;
+        s_a = a;
+        s_on = (v.b_launched != 0 && a == v.T - 1 && v.b_tok[0] == s_o[v.T - 1]) ? 1 : 0;
+    }
+    __syncthreads();
+    const int a = s_a;
+    const int n_rows = v.T + 1 + (v.max_t - 2);   // the window's cells, output 0's cell, every step's cell
+    for (int r = i; r < n_rows; r += blockDim.x) {
+        int rw;
+        long long cell;
+        if (r < v.T) { rw = r; cell = v.p + r; }
+        else if (r == v.T) { rw = 2 * v.max_t - 1; cell = v.p + a; }
+        else { const int j = r - v.T; rw = v.max_t + j - 1; cell = v.p + a + j; }
+        step[rw * 4 + 0] = (int32_t) cell;
+        step[rw * 4 + 1] = (int32_t) (cell + 1);
+        step[rw * 4 + 2] = (int32_t) ((cell + 1) / 4);
+        step[rw * 4 + 3] = (int32_t) (cell + 1);
+        for (int h = 0; h < v.nh; ++h) pos[rw * v.nh + h] = (int32_t) cell;
+    }
+    if (i < v.T) tok[i] = s_o[i];
+    if (i < 16) force[i] = (s_on && i < v.b_T - 1) ? v.b_tok[1 + i] : -1;
+    if (i == 0) { row[0] = a; row[1] = 0; }
+    __threadfence_system();
+}
+
 }  // namespace

 void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
```

### 12-29

`src/kernels/cuda/verify_kernels.cu` · void copy_row_to_first(const int32_t* row_dev, float* a, int64_t a_n, float* b,

Anchor:
```
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
```
Block (hunk 3 of `src/kernels/cuda/verify_kernels.cu` in the patch, 25 lines):
```diff
 }

 void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
-                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
+                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p,
+                const int32_t* force) {
     mtp_select_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(R_src, R_stride, ids, row_dev, R_dst, tok_dst, out, j,
-                                                            probs, out_p);
+                                                            probs, out_p, force);
     check("mtp_select");
 }

+void mtp_chain_verdict(const MtpChainVerdict& v, const int32_t* out, int32_t* tok, int32_t* step, int32_t* pos,
+                       int32_t* row, int32_t* force, void* stream) {
+    if (v.T < 1 || v.T > 8 || v.max_t < 2 || v.max_t > 8 || v.b_T > 8) {
+        std::fprintf(stderr, "mtp_chain_verdict: window size out of range\n");
+        std::exit(1);
+    }
+    mtp_chain_verdict_kernel<<<1, 64, 0, (cudaStream_t) stream>>>(v, out, tok, step, pos, row, force);
+    check("mtp_chain_verdict");
+}
+
 void force_token(int32_t* tok, const int32_t* force, int j, void* stream) {
     force_token_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(tok, force, j);
     check("force_token");
```

### 12-30

`src/program/generate.cpp` · void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t

Anchor:
```
    d.d.pcie_num = s->pcie_num[st];
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 23 lines):
```diff
     d.d.pcie_num = s->pcie_num[st];
     drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
 }
+/// critpath STRATA_CP_RESPLAN (Verifier::set_cp_note): a group the device planned is never served, but its routes
+/// count as the pool would count them (usage for the adaptive tier, every entry a VRAM hit).
+void drive_note_split(void* user, const int32_t* ids, int64_t n, int64_t layer) {
+    SplitDrive* s = (SplitDrive*) user;
+    strata::core::ExpertDispatch& d = s->base->d;
+    if (!d.usage.empty())
+        for (int64_t i = 0; i < n; ++i)
+            if (ids[i] >= 0 && ids[i] < d.n_expert) d.usage[(size_t) layer * (size_t) d.n_expert + (size_t) ids[i]] += 1.0f;
+    d.cache_hits += n;     // (the multi dispatch's counts for a window whose entries are all VRAM hits)
+    d.experts += n;
+    d.layers = layer + 1;
+}
+/// critpath (docs/CRITPATH.md): an integer lever, `dflt` when unset or empty.
+int cp_env_int(const char* name, int dflt) {
+    const char* v = std::getenv(name);
+    return v != nullptr && v[0] != 0 ? std::atoi(v) : dflt;
+}

 /// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
 /// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
```

### 12-31

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                for (int par = 0; par < 2; ++par) {
                    stage_ver_p(st, par).set_stream(pl_stream[(size_t) st]);
                    stage_ver_p(st, par).set_always_publish(true);
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                 for (int par = 0; par < 2; ++par) {
                     stage_ver_p(st, par).set_stream(pl_stream[(size_t) st]);
                     stage_ver_p(st, par).set_always_publish(true);
+                    stage_ver_p(st, par).set_cp_note(&drive_note_split);   // critpath STRATA_CP_RESPLAN (opt-in)
                 }
         }
         if (n_stages > 1) {
```

### 12-32

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                std::thread pl_adapt_thr;
                std::atomic<bool> pl_adapt_done{true};
                bool pl_adapt_ok = true;
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
                 std::thread pl_adapt_thr;
                 std::atomic<bool> pl_adapt_done{true};
                 bool pl_adapt_ok = true;
+                // critpath STRATA_CP_RESPLAN (docs/CRITPATH.md): the blocking tier's fence waits until this loop has passed a
+                // point between two launches after the tier marked its evictions on the host (cp_fence_ack, at the top of
+                // every pass): a later launch then uploads the marks before its window plans on the device (Verifier::
+                // cp_sync_res), an earlier one is behind the fence's events.  Open once the loop ends (no launch follows).
+                static const bool cp_resplan = cp_env_int("STRATA_CP_RESPLAN", 0) != 0;
+                std::atomic<int64_t> cp_fence_req{0}, cp_fence_ack{0};
+                std::atomic<bool> cp_fence_open{false};
                 // an error ends the engine, as the serial loop's do: every GPU wait is released first (#267), so no
                 // window in flight keeps spinning on a flag nobody raises, and the tier's thread can finish (the
                 // asynchronous tier's job does host work alone: it ends)
```

### 12-33

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    std::fprintf(stderr, "strata serve: pipelined decode: %s\n", m.c_str());
                    strata::core::diag_pipeline_fn().store(nullptr);
                    strata::core::release_gpu_waits(stderr);
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     std::fprintf(stderr, "strata serve: pipelined decode: %s\n", m.c_str());
                     strata::core::diag_pipeline_fn().store(nullptr);
                     strata::core::release_gpu_waits(stderr);
+                    cp_fence_open.store(true);   // critpath: the tier's fence never waits for this loop again
                     if (pl_adapt_thr.joinable()) pl_adapt_thr.join();
                     if (ajob) {
                         ajob->wait();
```

### 12-34

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                // adapt()'s fence, on the tier's thread: it waits on the HOST.  A device-side wait (the refill streams
                // waiting for the stages' events) would queue copies behind windows that need this loop, beside it.
                adapt_fence = [&]() {
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                 // adapt()'s fence, on the tier's thread: it waits on the HOST.  A device-side wait (the refill streams
                 // waiting for the stages' events) would queue copies behind windows that need this loop, beside it.
                 adapt_fence = [&]() {
+                    if (cp_resplan) {   // critpath STRATA_CP_RESPLAN: the loop past a point between launches first
+                        const int64_t r = cp_fence_req.fetch_add(1) + 1;
+                        while (cp_fence_ack.load() < r && !cp_fence_open.load()) std::this_thread::yield();
+                    }
                     stage_record(fence_ev);
                     while (!stage_passed(fence_ev)) std::this_thread::yield();
                 };
```

### 12-35

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    return v == nullptr || std::atoi(v) != 0;
                }();
                auto ms_now = [&]() { return std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count(); };
```
Block (hunk 6 of `src/program/generate.cpp` in the patch, 38 lines):
```diff
                     return v == nullptr || std::atoi(v) != 0;
                 }();
                 auto ms_now = [&]() { return std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count(); };
+                // ---- critpath levers (docs/CRITPATH.md), each opt-in; unset, the loop is the one above
+                // STRATA_CP_L1FIRST=1: A's last stage is launched in the pass that sees its front stage finish, right after
+                // the pumps - before the chain read, the elastic gap and B's speculative launch (its commit wait, snapshot
+                // and staging), which held the stage-1 launch ~0.45 ms in the fresh windows.
+                static const bool cp_l1first = cp_env_int("STRATA_CP_L1FIRST", 0) != 0;
+                // STRATA_CP_S1FIRST=1: while A's last stage runs, a rung layer of the speculative B is served only when its
+                // pool call fits before stage 1's next ring (the GPU's time from a served ring to the next one and B's layer
+                // time, both measured: EMAs) or right after stage 1 was served; else B waits a pass.
+                // STRATA_CP_S1FIRST_MAX_US (1500) bounds how long one B layer may wait.  B has slack: stage 1 takes ~17 ms,
+                // B's stage 0 ~7-11 ms.
+                static const bool cp_s1first = cp_env_int("STRATA_CP_S1FIRST", 0) != 0;
+                static const double cp_s1_max = std::max(0, cp_env_int("STRATA_CP_S1FIRST_MAX_US", 1500)) / 1000.0;
+                // STRATA_CP_CHAIN_PRE=n (1..S_mtp-1): the next chain's round and first n steps queued on the drafter's
+                // stream behind A's stage-1 window (MtpDrafter::chain_prelaunch), adopted at the verdict
+                static const int cp_chain_pre = cp_env_int("STRATA_CP_CHAIN_PRE", 0);
+                int64_t cp_l1_early = 0, cp_b_held = 0, cp_b_gap = 0, cp_b_forced = 0, cp_pre_seq = -1;
+                double cp_b_hold_ms = 0.0, cp_b_wait0 = -1.0;
+                double cp_s1_last = -1.0, cp_s1_gap = 0.35, cp_b_cost = 0.15;   // ms: stage 1's ring gap, B's layer
+                int64_t cp_s1_k = -1, cp_s1_seq = -1;
+                bool cp_prev_b = false;   // the pass before served a B layer (its ring gap sample is late)
+                const int64_t cp_pre0 = mtp.pre_launched, cp_ado0 = mtp.pre_adopted, cp_redo0 = mtp.pre_redone;
+                // A's last stage, once every front stage is done
+                auto launch_s1 = [&]() -> bool {
+                    if (chain_kind == 1 && !B.ready) ++pl_late;   // stage 0 idles until the chain has B
+                    if (ajob) a_gap(1);   // --adapt-async: stage 1 is idle until this launch
+                    if (!V1(A).pl_launch(A.T, A.tok, A.p, err)) return false;
+                    A.s1 = true;
+                    tre("L1", A.seq, A.T);
+                    return true;
+                };
                 while (true) {
+                    if (cp_resplan) cp_fence_ack.store(cp_fence_req.load());   // critpath: between two launches
                     g_pl_diag.iters.fetch_add(1, std::memory_order_relaxed);
                     g_pl_diag.a_seq.store(A.seq, std::memory_order_relaxed);
                     g_pl_diag.a_bits.store(pw_bits(A), std::memory_order_relaxed);
```

### 12-36

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    // verdict), and while A's last stage runs, B one layer per pass
                    if (!pl_yield && ((doomed && !pump0(D)) || !pump0(A) || !pump0(B))) return die(err);
                    if (pl_yield && !pump0(A)) return die(err);
```
Block (hunk 7 of `src/program/generate.cpp` in the patch, 23 lines):
```diff
                     // verdict), and while A's last stage runs, B one layer per pass
                     if (!pl_yield && ((doomed && !pump0(D)) || !pump0(A) || !pump0(B))) return die(err);
                     if (pl_yield && !pump0(A)) return die(err);
+                    if (cp_l1first && A.finished && !A.s1) {   // critpath STRATA_CP_L1FIRST: stage 1 before anything else
+                        if (!launch_s1()) return die(err);
+                        ++cp_l1_early;
+                    }
                     if (A.s1 && !A.s1_done) {
                         strata::core::Verifier& v = V1(A);
+                        const double cp_t_in = cp_s1first ? ms_now() : 0.0;
+                        const int64_t cp_k0 = v.pl_served();
+                        const bool cp_rang = cp_s1first && v.pl_rang();
                         if (v.service(&drive_pool_split, SDf(A), err) < 0) return die(err);
+                        if (cp_s1first && v.pl_served() > cp_k0) {   // critpath STRATA_CP_S1FIRST: stage 1's ring gap
+                            if (cp_rang && cp_s1_seq == A.seq && cp_s1_last >= 0.0 && !cp_prev_b && cp_k0 == cp_s1_k)
+                                cp_s1_gap += 0.125 * ((cp_t_in - cp_s1_last) - cp_s1_gap);
+                            cp_s1_last = ms_now();
+                            cp_s1_k = v.pl_served();
+                            cp_s1_seq = A.seq;
+                        }
                         if (v.done(err)) {
                             if (!v.pl_finish(outp.data(), err)) return die(err);
                             force_rows_out(outp.data(), A.T, produced_n);   // lab bench: STRATA_FORCE_IDS
```

### 12-37

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                            tre("F1", A.seq, A.T);
                        } else if (!err.empty()) return die(err);
                    }
```
Block (hunk 8 of `src/program/generate.cpp` in the patch, 51 lines):
```diff
                             tre("F1", A.seq, A.T);
                         } else if (!err.empty()) return die(err);
                     }
-                    if (pl_yield && ((doomed && !pump0(D)) || !pump0(B, A.s1 && !A.s1_done ? 1 : 0))) return die(err);
+                    if (pl_yield) {
+                        if (doomed && !pump0(D)) return die(err);
+                        // critpath STRATA_CP_S1FIRST: B's rung layer waits while stage 1 is about to ring (see above)
+                        bool serve_b = true;
+                        const double cp_now = cp_s1first ? ms_now() : 0.0;
+                        if (cp_s1first && A.s1 && !A.s1_done && B.fl && VF(B.fr, B).pl_rang()) {
+                            strata::core::Verifier& v1 = V1(A);
+                            bool hold = v1.pl_rang();   // stage 1 rang meanwhile: the next pass serves it first
+                            // else B goes when its layer fits before the predicted ring, or right after stage 1 was
+                            // served (the start of its gap: the least B can delay it when the layer does not fit)
+                            if (!hold && v1.pl_serving() && cp_s1_seq == A.seq && cp_s1_last >= 0.0)
+                                hold = cp_now + cp_b_cost > cp_s1_last + cp_s1_gap - 0.03 && cp_now - cp_s1_last > 0.05;
+                            bool bound = false;
+                            if (hold) {
+                                if (cp_b_wait0 < 0.0) cp_b_wait0 = cp_now;
+                                if (cp_now - cp_b_wait0 > cp_s1_max) {   // the bound: B's layer goes now
+                                    hold = false;
+                                    bound = true;
+                                    ++cp_b_forced;
+                                }
+                            }
+                            if (hold) {
+                                ++cp_b_held;
+                                serve_b = false;
+                            } else if (!bound) {
+                                ++cp_b_gap;
+                            }
+                        }
+                        if (serve_b && cp_b_wait0 >= 0.0) {
+                            cp_b_hold_ms += cp_now - cp_b_wait0;
+                            cp_b_wait0 = -1.0;
+                        }
+                        cp_prev_b = false;
+                        if (serve_b) {
+                            strata::core::Verifier* vb = B.fl ? &VF(B.fr, B) : nullptr;
+                            const int64_t kb0 = vb != nullptr ? vb->pl_served() : 0;
+                            if (!pump0(B, A.s1 && !A.s1_done ? 1 : 0)) return die(err);
+                            if (cp_s1first && vb != nullptr && vb->pl_served() > kb0) {   // B's layer time
+                                const double dt = ms_now() - cp_now;
+                                cp_b_cost += 0.125 * (dt - cp_b_cost);
+                                cp_prev_b = true;
+                            }
+                        }
+                    }
                     if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
                     if (!el_gap()) return die(err);   // fork: the elastic tail at a gap of stage 0
                     // ---- a wrong speculative window has stopped (it is launched on no further stage once doomed):
```

### 12-38

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        }
                    }
                    // ---- the last stage: A, once every front stage is done
```
Block (hunk 9 of `src/program/generate.cpp` in the patch, 44 lines):
```diff
                         }
                     }
                     // ---- the last stage: A, once every front stage is done
-                    if (A.finished && !A.s1) {
-                        if (chain_kind == 1 && !B.ready) ++pl_late;   // stage 0 idles until the chain has B
-                        if (ajob) a_gap(1);   // --adapt-async: stage 1 is idle until this launch
-                        if (!V1(A).pl_launch(A.T, A.tok, A.p, err)) return die(err);
-                        A.s1 = true;
-                        tre("L1", A.seq, A.T);
+                    if (A.finished && !A.s1 && !launch_s1()) return die(err);
+                    // ---- critpath STRATA_CP_CHAIN_PRE: the chain after A's verdict, queued behind A's stage-1 window on the
+                    // drafter's stream, its inputs derived on the device from A's picks (MtpDrafter::chain_prelaunch).  Once
+                    // the chain that made B is done and B's launch is settled (B is part of the verdict); greedy rows only
+                    if (cp_chain_pre > 0 && A.s1 && !A.s1_done && cp_pre_seq != A.seq && chain_kind == 0 &&
+                        !mtp.chain_live() && !(B.ready && !B.launched) && !ending && S_mtp >= 2 && mtp.max_t() >= 3 &&
+                        A.T <= mtp.max_t()) {
+                        strata::core::Verifier& v1 = V1(A);
+                        if (!v1.host_sampling() && mtp.device() == v1.device()) {
+                            strata::kernels::MtpChainVerdict cv;
+                            cv.T = A.T;
+                            cv.p = (long long) A.p;
+                            cv.b_launched = B.launched ? 1 : 0;
+                            cv.b_T = B.launched ? B.T : 0;
+                            for (int i = 0; i < 8; ++i) {
+                                cv.a_tok[i] = A.tok[i];
+                                cv.b_tok[i] = B.tok[i];
+                                cv.repl[i] = -1;
+                            }
+                            if (forced != nullptr)   // the forced bench: the rows force_rows_out replaces at the verdict
+                                for (int i = 0; i < A.T; ++i) {
+                                    const int64_t k = produced_n + i;
+                                    if (k >= (int64_t) forced->size()) break;
+                                    cv.repl[i] = (*forced)[(size_t) k];
+                                }
+                            const int pre = std::max(1, std::min(cp_chain_pre, std::min(S_mtp - 1, mtp.max_t() - 2)));
+                            if (!mtp.chain_prelaunch(cv, v1.out_dev(), v1.out_host(), v1.done_event(), v1.final_R(0), pre,
+                                                     err))
+                                return die(err);
+                            tre("CP", A.seq, pre);
+                        }
+                        cp_pre_seq = A.seq;   // once per A (also when it could not be queued)
                     }
                     // ---- A's verdict
                     if (!A.s1_done) continue;
```

### 12-39

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        // the chain over A, forced through B's drafts: B's bonus guess and the next window's drafts
                        mtp.set_source_R(V1(A).final_R(0));
                        chain_n = std::min(strata::kernels::kVerifyMaxT - 1, B.T + S_mtp - 1);
```
Block (hunk 10 of `src/program/generate.cpp` in the patch, 13 lines):
```diff
                         // the chain over A, forced through B's drafts: B's bonus guess and the next window's drafts
                         mtp.set_source_R(V1(A).final_R(0));
                         chain_n = std::min(strata::kernels::kVerifyMaxT - 1, B.T + S_mtp - 1);
-                        if (!mtp.chain_launch(A.T, outp.data(), A.p, a, B.tok + 1, B.T - 1, chain_n, chain_n, err))
+                        int cp_ad = 0;   // critpath STRATA_CP_CHAIN_PRE: the chain queued behind A's window, if it fits
+                        if (cp_pre_seq == A.seq &&
+                            (cp_ad = mtp.chain_adopt(A.T, outp.data(), A.p, a, B.tok + 1, B.T - 1, chain_n, chain_n, err)) < 0)
+                            return die(err);
+                        if (cp_ad == 0 &&
+                            !mtp.chain_launch(A.T, outp.data(), A.p, a, B.tok + 1, B.T - 1, chain_n, chain_n, err))
                             return die(err);
                         chain_kind = 1;
                         tre("CL", A.seq, 1);
```

### 12-40

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        if (!last) {
                            mtp.set_source_R(V1(A).final_R(0));
                            chain_n = strata::kernels::kVerifyMaxT - 1;
```
Block (hunk 11 of `src/program/generate.cpp` in the patch, 12 lines):
```diff
                         if (!last) {
                             mtp.set_source_R(V1(A).final_R(0));
                             chain_n = strata::kernels::kVerifyMaxT - 1;
-                            if (!mtp.chain_launch(A.T, outp.data(), A.p, a, nullptr, 0, chain_n, chain_n, err))
+                            int cp_ad = 0;   // critpath STRATA_CP_CHAIN_PRE (see above)
+                            if (cp_pre_seq == A.seq &&
+                                (cp_ad = mtp.chain_adopt(A.T, outp.data(), A.p, a, nullptr, 0, chain_n, chain_n, err)) < 0)
+                                return die(err);
+                            if (cp_ad == 0 && !mtp.chain_launch(A.T, outp.data(), A.p, a, nullptr, 0, chain_n, chain_n, err))
                                 return die(err);
                             chain_kind = 2;
                             tre("CL", A.seq, 2);
```

### 12-41

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    if (!last && ajob && !adapt_tick(false)) return die("an adaptive refill failed");
                    if (last) ending = true;
                }
```
Block (hunk 12 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                     if (!last && ajob && !adapt_tick(false)) return die("an adaptive refill failed");
                     if (last) ending = true;
                 }
+                cp_fence_open.store(true);   // critpath: no launch follows, the tier's fence never waits for the loop
                 pl_drain();   // nothing in flight: both stages' commits and the drafter have landed
                 strata::core::diag_pipeline_fn().store(nullptr);
                 if (pl_adapt_thr.joinable()) pl_adapt_thr.join();   // its copies waited only for finished work
```

### 12-42

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                }
                // the odd windows' counters and GPU profiles into the verifiers that report them
                for (int st = 0; st < n_stages; ++st) stage_ver_p(st, 0).absorb_stats(stage_ver_p(st, 1));
```
Block (hunk 13 of `src/program/generate.cpp` in the patch, 28 lines):
```diff
                 }
                 // the odd windows' counters and GPU profiles into the verifiers that report them
                 for (int st = 0; st < n_stages; ++st) stage_ver_p(st, 0).absorb_stats(stage_ver_p(st, 1));
+                // critpath (docs/CRITPATH.md): one line per request with every lever's counters while any is on
+                static const bool cp_any = cp_l1first || cp_s1first || cp_chain_pre > 0 || cp_resplan ||
+                                           cp_env_int("STRATA_CP_CHAIN_FUSE", 0) != 0;
+                if (cp_any) {
+                    std::string rp;
+                    for (int st = 0; st < n_stages; ++st) {
+                        const strata::core::Verifier& v = stage_ver_p(st, 0);
+                        char b[160];
+                        std::snprintf(b, sizeof b, " stage %d: %d layers, %lld planned / %lld host, %lld uploads;", st,
+                                      v.cp_layers(), (long long) v.cp_planned, (long long) v.cp_host,
+                                      (long long) v.cp_uploads);
+                        rp += b;
+                    }
+                    std::fprintf(stderr, "strata critpath: L1FIRST %lld early stage-1 launches | S1FIRST B held %lld passes "
+                                         "(%.1f ms), %lld B layers in a stage-1 gap, %lld past the bound; stage-1 ring gap "
+                                         "%.3f ms, B layer %.3f ms | CHAIN_PRE %lld queued, %lld adopted, %lld redone | "
+                                         "RESPLAN (all requests):%s\n",
+                                 (long long) cp_l1_early, (long long) cp_b_held, cp_b_hold_ms, (long long) cp_b_gap,
+                                 (long long) cp_b_forced, cp_s1_gap, cp_b_cost, (long long) (mtp.pre_launched - cp_pre0),
+                                 (long long) (mtp.pre_adopted - cp_ado0), (long long) (mtp.pre_redone - cp_redo0),
+                                 rp.c_str());
+                }
                 if (!pl_adapt_ok) {
                     std::printf("ERR an adaptive refill failed\n");
                     return 1;
```

## 13 chain-fuse

`13-chain-fuse.patch` — FABLE-STRATA-Porta su 0.1.43 le fusioni della catena MTP del drafter (STRATA_CHAIN_FUSE2, STRATA_CHAIN_FUSE2_MASK, STRATA_CHAIN_FUSE2_ATTN, opt-in) e chain_step_bench

### 13-1

`CMakeLists.txt` · if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)

Anchor:
```
    set_tests_properties(qsa_merge_daily_parity_daily PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=1"
                         SKIP_RETURN_CODE 77)
  endif()
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 14 lines):
```diff
     set_tests_properties(qsa_merge_daily_parity_daily PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=1"
                          SKIP_RETURN_CODE 77)
   endif()
+  # chainfuse (docs/CHAINFUSE.md): the MTP chain step's fused segments (STRATA_CHAIN_FUSE2) against the default kernel
+  # sequences, bit for bit, with the microseconds of each (GPU, synthetic, no model).  Not in the default build:
+  # `cmake --build <dir> --target chain_step_bench`.
+  if(STRATA_ENABLE_CUDA)
+    add_executable(chain_step_bench EXCLUDE_FROM_ALL src/kernels/chain_step_bench.cpp)
+    target_include_directories(chain_step_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
+    target_link_libraries(chain_step_bench PRIVATE strata_kernels CUDA::cudart)
+  endif()
   # the QSA selection (block scores on the warp kernel and the tensor-core one, top-k): a GPU bench with a
   # scorer-vs-scorer comparison and an FP64 accuracy gate (synthetic, no model); the ctest runs a 32K context under a
   # 262144-cell capacity (the top-k dispatch follows the capacity) and exits non-zero on a failed gate
```

### 13-2

`docs/CHAINFUSE.md` · (new file)

New file, 209 lines: see the patch.

### 13-3

`include/strata/core/mtp.hpp` · private:

Anchor:
```
    float* probs_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' and row_top_prob_split's partials and counters
    uint8_t* top_scratch_ = nullptr;
```
Block (hunk 1 of `include/strata/core/mtp.hpp` in the patch, 7 lines):
```diff
     float* probs_ = nullptr;
     uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' and row_top_prob_split's partials and counters
     uint8_t* top_scratch_ = nullptr;
+    uint8_t* tail_scratch_ = nullptr;  ///< chainfuse: mtp_head_tail's partials and counter (docs/CHAINFUSE.md)
     // device
     int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
     float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
```

### 13-4

`include/strata/kernels/native_gr_postops.hpp` · void native_gr_post_multi(const float* residual, const float* block_out, const f

Anchor:
```
void native_gr_post_multi(const float* residual, const float* block_out, const float* inject,
                          float* output, int n_embd, int hc, int n_tok, void* stream);
```
Block (hunk 1 of `include/strata/kernels/native_gr_postops.hpp` in the patch, 14 lines):
```diff
 void native_gr_post_multi(const float* residual, const float* block_out, const float* inject,
                           float* output, int n_embd, int hc, int n_tok, void* stream);

+// chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1): native_moe_combine (k = 10, shared row, the vec4 kernel's
+// expression: native_moe.cu compiles with the same --use_fast_math) and native_gr_post in one launch for one token:
+// y = sum_k parts[k] * weights[k] + shared (the first product rounded, then one fmaf per expert in order, then the
+// shared row), output[c][:] = fmaf(y, 2 sigmoid(inject[c] / hc), residual[c][:]) for the hc streams.  `y_out`
+// (nullable) also receives y.  parts (10 x n_embd), shared, residual, output and y_out 16-byte aligned, n_embd a
+// multiple of 4; output may equal residual exactly.
+void native_moe_combine_k10_post(const float* parts, const float* weights, const float* shared, float* y_out,
+                                 const float* residual, const float* inject, float* output, int n_embd, int hc,
+                                 void* stream);
+
 } // namespace strata::kernels
```

### 13-5

`include/strata/kernels/native_qsa.hpp` · void native_qsa_rms_norm_grouped(const float* input, const float* gamma, float*

Anchor:
```
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                           int n_head, int head_dim, void* stream);
```
Block (hunk 1 of `include/strata/kernels/native_qsa.hpp` in the patch, 16 lines):
```diff
 void native_qsa_gate_apply(const float* attn, const float* q_full, float* output,
                            int n_head, int head_dim, void* stream);

+// chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1): native_qsa_rms_norm_weighted followed by native_quantize_q8_1
+// of its output (n_cols * n_rows contiguous values, n_cols a multiple of 32) in ONE launch: the norm kernel, then the
+// quantizer's expressions in its epilogue on the warp's 32 consecutive columns (the quantizer's own block and lane
+// order).  `output` may be null (the normalized row is not kept); `x_q8_1` is native_q8_1_bytes(n_cols, n_rows).
+// Same file and flags as the norm, the quantizer's helpers (q8_1_finite.hpp) under those flags: the same bytes.
+void native_qsa_rms_norm_q8_1(const float* input, const float* gamma, float* output, void* x_q8_1,
+                              int n_cols, int n_rows, float epsilon, void* stream);
+// native_qsa_gate_apply followed by native_quantize_q8_1 of its output (n_head * head_dim a multiple of 32) in one
+// launch; `output` may be null.
+void native_qsa_gate_q8_1(const float* attn, const float* q_full, float* output, void* x_q8_1,
+                          int n_head, int head_dim, void* stream);
+
 } // namespace strata::kernels
```

### 13-6

`include/strata/kernels/qsa_decode_attn.hpp` · void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32

Anchor:
```
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);
```
Block (hunk 1 of `include/strata/kernels/qsa_decode_attn.hpp` in the patch, 29 lines):
```diff
 void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                            int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);

-/// For src/kernels/qsa_merge_daily_parity.cpp only: qsa_decode_attn_batch's two passes apart, whatever the environment
-/// says.  `chunks_only` runs the chunk pass into `scratch` with `daily_chunk` 0: attn_chunk_kernel (upstream's default),
-/// 1: the Daily's (attn_chunk_kernel_pre75, what STRATA_ATTN_DAILY=1 picks on CUDA).  `merge_only` merges those
-/// partials with `v2` 0: attn_merge_kernel, 1: attn_merge_v2_kernel (STRATA_ATTN_MERGE_V2, on with STRATA_ATTN_DAILY).
+/// chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1; the MTP drafter's dense attention): qsa_decode_attn_batch with
+/// the staged chunk kernel (`variant` 1..4 = attn_chunk_kernel_stage<M, KPF, EARLY>: the values of a stage of cells
+/// through shared memory with one 16-byte load a thread, 1: plain, 2: the next pair of keys prefetched, 3: the first
+/// stage of values fetched before the scores, 4: both; 0: the environment's chunk kernel; INT8 and FP16 pools, other
+/// formats take the environment's chunk kernel) and, with `merge` 1, the staged merge (attn_merge_v3_kernel).  Bit for
+/// bit the output of attn_chunk_kernel_pre75 + attn_merge_v2_kernel (the same expressions in the same order; only how
+/// the loads are issued changes).  CUDA only: elsewhere, or where the pre75 kernel is not built, it is
+/// qsa_decode_attn_batch.  `qsa_decode_attn_fuse2_built()` says which.
+void qsa_decode_attn_batch_fuse2(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
+                                 int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, int variant,
+                                 int merge, void* stream);
+bool qsa_decode_attn_fuse2_built();
+
+/// For src/kernels/qsa_merge_daily_parity.cpp and src/kernels/chain_step_bench.cpp only: qsa_decode_attn_batch's two
+/// passes apart, whatever the environment says.  `chunks_only` runs the chunk pass into `scratch` with `daily_chunk`
+/// 0: attn_chunk_kernel (upstream's default), 1: the Daily's (attn_chunk_kernel_pre75, what STRATA_ATTN_DAILY=1 picks
+/// on CUDA), 2 / 3 / 4 / 5: chainfuse's attn_chunk_kernel_stage variants 1 / 2 / 3 / 4 (INT8 and FP16 pools; a Q4_0
+/// pool takes the environment's kernel).  `merge_only` merges those partials with `v2` 0: attn_merge_kernel,
+/// 1: attn_merge_v2_kernel (STRATA_ATTN_MERGE_V2, on with STRATA_ATTN_DAILY), 2: chainfuse's attn_merge_v3_kernel.
 /// Width must not exceed an unaligned cap (the chunk pass keeps its upstream contract); for aligned caps the grid
 /// bounds widths above cap.  One input can go through every pair and the outputs be compared bit for bit.
 void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
```

### 13-7

`include/strata/kernels/verify_kernels.hpp` · bool multi_block_head_ops();

Anchor:
```
/// True where the verify head's and the draft's greedy pick should be argmax_rows: sm_80 to sm_89 (the one-block
/// kernel is slow there; sm_90+ has sample_tokens' cluster kernel, other cards keep sample_tokens).
bool argmax_rows_wanted();
```
Block (hunk 1 of `include/strata/kernels/verify_kernels.hpp` in the patch, 16 lines):
```diff
 /// True where the verify head's and the draft's greedy pick should be argmax_rows: sm_80 to sm_89 (the one-block
 /// kernel is slow there; sm_90+ has sample_tokens' cluster kernel, other cards keep sample_tokens).
 bool argmax_rows_wanted();
+/// chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1): a draft step's tail in one launch, the same words as
+/// sample_tokens (greedy) + row_top_prob(_split) + map_ids (`table`, null = none) + mtp_select (+ force_token via
+/// `force`) on row 0 of `logits` (n_vocab values): out_ids[0] and probs_dev[0] get the mapped pick and its probability,
+/// then R_dst[:] = R_src[*row_dev], *tok_dst = force[j] >= 0 ? force[j] : out_ids[*row_dev], out_p[j] =
+/// probs_dev[*row_dev], out[j] = out_ids[*row_dev] (out / out_p / force may be mapped host memory).  `scratch`:
+/// mtp_head_tail_scratch_bytes(), zero before the first launch (left so).  Graph-capturable.  CUDA only.
+uint64_t mtp_head_tail_scratch_bytes();
+void mtp_head_tail(const float* logits, int n_vocab, const int32_t* table, const float* R_src, int64_t R_stride,
+                   const int32_t* row_dev, float* R_dst, int32_t* out_ids, float* probs_dev, int32_t* tok_dst,
+                   int32_t* out, int j, float* out_p, const int32_t* force, void* scratch, void* stream);
 /// Dense-attention step records for `n` cells: [cell, cell+1, (cell+1)/4, cell+1] from cells[i] (device memory).
 void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream);
 /// A sliding attention window: for `n` step records (kStepCount ints each, n_kv at [1]) the selection becomes the
```

### 13-8

`src/core/mtp.cpp` · (file scope)

Anchor:
```
#include "strata/kernels/gr.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
```
Block (hunk 1 of `src/core/mtp.cpp` in the patch, 7 lines):
```diff
 #include "strata/kernels/gr.hpp"
 #include "strata/kernels/kv_q4.hpp"
 #include "strata/kernels/kv_q8.hpp"
+#include "strata/kernels/native_gr_postops.hpp"   // chainfuse: native_moe_combine_k10_post
 #include "strata/kernels/native_moe.hpp"
 #include "strata/kernels/native_mmvq.hpp"
 #include "strata/kernels/native_qsa.hpp"
```

### 13-9

`src/core/mtp.cpp` · bool cp_chain_fuse() {

Anchor:
```
constexpr int GGML_Q8_0 = 8;
constexpr int GGML_Q4_0 = 2;
using Clock = std::chrono::steady_clock;
```
Block (hunk 2 of `src/core/mtp.cpp` in the patch, 46 lines):
```diff
     }();
     return on;
 }
+// chainfuse (docs/CHAINFUSE.md): STRATA_CHAIN_FUSE2=1 (opt-in) - the drafter's step and round graphs with fused kernels,
+// every word the same (the same expressions in the same order; only launches and loads in flight change).
+// STRATA_CHAIN_FUSE2_MASK (default 62) picks the fusions for an A/B one at a time: 2 the staged attention merge, 4 norm +
+// q8_1, 8 gate + q8_1, 16 the head tail (argmax + top prob + map_ids + mtp_select + force in one launch), 32 combine +
+// residual write.  The attention CHUNK kernel is its own variable, so it can be measured apart (the first version lost):
+// STRATA_CHAIN_FUSE2_ATTN = 1..4 picks attn_chunk_kernel_stage's variant (1 plain staged values, 2 + the next keys
+// prefetched, 3 + the first values before the scores, 4 both; 0 / unset: pre75 as before).  The argmax graphs only
+// (coupled sampling keeps its own); CUDA only.
+int chain_fuse2_attn_env() {
+    const char* a = std::getenv("STRATA_CHAIN_FUSE2_ATTN");
+    const int n = a != nullptr ? std::atoi(a) : 0;
+    return n >= 1 && n <= 4 ? n : 0;
+}
+int chain_fuse2_mask() {
+    static const int mask = [] {
+        const char* v = std::getenv("STRATA_CHAIN_FUSE2");
+        const bool on = v != nullptr && v[0] != '\0' && v[0] != '0';
+        if (!on) return 0;
+        int m = 62;
+        const char* mv = std::getenv("STRATA_CHAIN_FUSE2_MASK");
+        if (mv != nullptr && mv[0] != '\0') m = std::atoi(mv) & 62;
+#if defined(STRATA_USE_HIP)
+        std::fprintf(stderr, "strata chainfuse: STRATA_CHAIN_FUSE2 is CUDA only: off\n");
+        return 0;
+#else
+        int attn = chain_fuse2_attn_env();
+        if (!strata::kernels::qsa_decode_attn_fuse2_built()) { m &= ~2; attn = 0; }
+        if (attn) m |= 1;
+        std::fprintf(stderr, "strata chainfuse: STRATA_CHAIN_FUSE2 mask %d (%s%s%s%s%s), attention chunk variant %d\n", m,
+                     (m & 2) ? "attn-merge " : "", (m & 4) ? "norm+q8 " : "", (m & 8) ? "gate+q8 " : "",
+                     (m & 16) ? "head-tail " : "", (m & 32) ? "combine+post" : "", attn);
+        return m;
+#endif
+    }();
+    return mask;
+}
+bool chain_fuse2(int bit) { return (chain_fuse2_mask() & bit) != 0; }
+int chain_fuse2_attn() { return chain_fuse2(1) ? chain_fuse2_attn_env() : 0; }
+constexpr int kFuse2AttnChunk = 1, kFuse2AttnMerge = 2, kFuse2NormQ8 = 4, kFuse2GateQ8 = 8, kFuse2HeadTail = 16,
+              kFuse2CombinePost = 32;
 constexpr int GGML_Q8_0 = 8;
 constexpr int GGML_Q4_0 = 2;
 using Clock = std::chrono::steady_clock;
```

### 13-10

`src/core/mtp.cpp` · bool MtpDrafter::load(const std::string& rt_dir, const ModelGeometry& g, Session

Anchor:
```
        probs_ = b.take<float>(T + 4);
        arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
        top_scratch_ = b.take<uint8_t>(strata::kernels::row_top_prob_scratch_bytes((int) T));
```
Block (hunk 3 of `src/core/mtp.cpp` in the patch, 7 lines):
```diff
         probs_ = b.take<float>(T + 4);
         arg_scratch_ = b.take<uint8_t>(strata::kernels::argmax_rows_scratch_bytes((int) T));
         top_scratch_ = b.take<uint8_t>(strata::kernels::row_top_prob_scratch_bytes((int) T));
+        tail_scratch_ = b.take<uint8_t>(std::max<uint64_t>(64, strata::kernels::mtp_head_tail_scratch_bytes()));
         dummy_inj_ = b.take<float>(HC);
     };
     Bump count;
```

### 13-11

`src/core/mtp.cpp` · bool MtpDrafter::record_front(int T, int row0, cudaStream_t cs, std::string& err

Anchor:
```
            embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                 (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
        }
```
Block (hunk 4 of `src/core/mtp.cpp` in the patch, 23 lines):
```diff
             embedding_gather_dev(codes, scales, offsets, tok_, T, we->ne0, we->code_bits, we->code_bias, we->group_elems,
                                  (uint64_t) (we->ne0 / (8 / we->code_bits)), (uint64_t) (we->ne0 / we->group_elems), emb_, cs);
         }
-        native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
-        native_quantize_q8_1(en_, xq_, (int) N, T, cs);
+        // chainfuse (docs/CHAINFUSE.md): one row (a chain step, a T = 1 round) - the norm and its q8_1 quantization in
+        // one launch (the same bytes in xq_; en_ / hn_ are not kept: nothing else reads them)
+        const bool fuse_nq = T == 1 && chain_fuse2(kFuse2NormQ8);
+        if (fuse_nq) {
+            native_qsa_rms_norm_q8_1(emb_, f32("pre_fc_norm_embedding.weight"), nullptr, xq_, (int) N, T, EPS, cs);
+        } else {
+            native_qsa_rms_norm_weighted(emb_, f32("pre_fc_norm_embedding.weight"), en_, (int) N, T, EPS, cs);
+            native_quantize_q8_1(en_, xq_, (int) N, T, cs);
+        }
         native_mmvq(wt_fc_embedding, wp_fc_embedding, xq_, e2_, (int) N, (int) N, T, cs);
+        if (fuse_nq && !hnorm_stream_ && HC <= 8) {
+            // one RMS over the hc * N values, quantized as the HC columns the loop below quantizes in one chunk
+            native_qsa_rms_norm_q8_1(Rin_, f32("pre_fc_norm_hidden.weight"), nullptr, xq_, (int) (HC * N), T, EPS, cs);
+            native_mmvq(wt_fc_hidden, wp_fc_hidden, xq_, h2_, (int) N, (int) N, (int) HC, cs);
+        } else {
         if (hnorm_stream_)   // --mtp-hnorm stream: one RMS per stream, each scaled by its slice of the weight
             native_qsa_rms_norm_grouped(Rin_, f32("pre_fc_norm_hidden.weight"), hn_, (int) N, (int) HC, (int) (T * HC), EPS, cs);
         else
```

### 13-12

`src/core/mtp.cpp` · bool MtpDrafter::record_front(int T, int row0, cudaStream_t cs, std::string& err

Anchor:
```
            native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
            native_mmvq(wt_fc_hidden, wp_fc_hidden, xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
        }
```
Block (hunk 5 of `src/core/mtp.cpp` in the patch, 7 lines):
```diff
             native_quantize_q8_1(hn_ + (size_t) c0 * N, xq_, (int) N, nc, cs);
             native_mmvq(wt_fc_hidden, wp_fc_hidden, xq_, h2_ + (size_t) c0 * N, (int) N, (int) N, nc, cs);
         }
+        }
         add_streams_broadcast(h2_, e2_, R_, N, (int) HC, T, cs);
         // ---- the attention hyper-connection
         {
```

### 13-13

`src/core/mtp.cpp` · bool MtpDrafter::record_rest(int step_row, cudaStream_t cs, std::string& err) {

Anchor:
```
        if (st_.kv_rot) fwht256_inplace_cuda(qcur_, (int64_t) T * NH, cs);
        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
```
Block (hunk 6 of `src/core/mtp.cpp` in the patch, 23 lines):
```diff
         if (st_.kv_rot) fwht256_inplace_cuda(qcur_, (int64_t) T * NH, cs);
         const QsaAttnPools pools = qsa_attn_pools(st_);
         if (window_ > 0) window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, cs);
-        qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
+        // chainfuse (docs/CHAINFUSE.md): the deeper-pipelined chunk kernel and / or the staged merge, the same partials
+        // and the same output (bit for bit) with more loads in flight
+        if (chain_fuse2(kFuse2AttnChunk) || chain_fuse2(kFuse2AttnMerge))
+            qsa_decode_attn_batch_fuse2(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, chain_fuse2_attn(),
+                                        chain_fuse2(kFuse2AttnMerge) ? 1 : 0, cs);
+        else
+            qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_, s, attn_scratch_, attn_, T, cs);
         if (st_.kv_rot) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
-        native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
-        native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
+        if (chain_fuse2(kFuse2GateQ8)) {   // chainfuse: the gate and its q8_1 quantization in one launch (attn32_ not kept)
+            native_qsa_gate_q8_1(attn_, qfull_, nullptr, xq_, (int) (T * NH), (int) HD, cs);
+        } else {
+            native_qsa_gate_apply(attn_, qfull_, attn32_, (int) (T * NH), (int) HD, cs);
+            native_quantize_q8_1(attn32_, xq_, (int) (NH * HD), T, cs);
+        }
         native_mmvq(wt_o_proj, wp_o_proj, xq_, bo_, (int) (NH * HD), (int) N, T, cs);
         // ---- the MLP hyper-connection (the attention write folded in)
         {
```

### 13-14

`src/core/mtp.cpp` · bool MtpDrafter::record_rest(int step_row, cudaStream_t cs, std::string& err) {

Anchor:
```
            const char* v = std::getenv("STRATA_NO_MULTI_GR");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
```
Block (hunk 7 of `src/core/mtp.cpp` in the patch, 14 lines):
```diff
             const char* v = std::getenv("STRATA_NO_MULTI_GR");
             return v != nullptr && v[0] != '\0' && v[0] != '0';
         }();
+        // chainfuse (docs/CHAINFUSE.md): the combine and the residual write in one launch (T = 1, the native combine's
+        // k = 10 vec4 kernel and gr_write's native post: the same expressions)
+        const bool fuse_cp = T == 1 && !fuse_head_gr && chain_fuse2(kFuse2CombinePost) && native_moe_combine_enabled() &&
+                             K == 10 && gr_native_mmvf_enabled() && (N & 3) == 0 &&
+                             (((uintptr_t) parts_ | (uintptr_t) shared_ | (uintptr_t) R_ | (uintptr_t) y_) & 15u) == 0;
+        if (fuse_cp) {
+            native_moe_combine_k10_post(parts_, w_, shared_, y_, R_, inj2_, R_, (int) N, (int) HC, cs);
+        } else {
         for (int t = 0; t < T; ++t) {
             if (!(T > 1 && native_moe_combine_enabled())) {
                 if (native_moe_combine_enabled())
```

### 13-15

`src/core/mtp.cpp` · bool MtpDrafter::record_rest(int step_row, cudaStream_t cs, std::string& err) {

Anchor:
```
        // its own, so doing them after the loop changes nothing)
        if (!fuse_head_gr && !no_multi_gr)
            gr_write_multi(R_, y_, inj2_, gs, R_, T, cs);
```
Block (hunk 8 of `src/core/mtp.cpp` in the patch, 7 lines):
```diff
         // its own, so doing them after the loop changes nothing)
         if (!fuse_head_gr && !no_multi_gr)
             gr_write_multi(R_, y_, inj2_, gs, R_, T, cs);
+        }
         // ---- the final mixer and the main model's head
         if (fuse_head_gr) {
             FusedGrArgs fa[kFusedGrMaxT];
```

### 13-16

`src/core/mtp.cpp` · bool MtpDrafter::record_rest(int step_row, cudaStream_t cs, std::string& err) {

Anchor:
```
                                     cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, cs);
            return true;
        }
```
Block (hunk 9 of `src/core/mtp.cpp` in the patch, 9 lines):
```diff
                                      cparams_, cring_, kCoupledHistCap, coupled_j_, step, cscratch_, out_ids_, probs_, cs);
             return true;
         }
+        // chainfuse (docs/CHAINFUSE.md): the pick, its probability, map_ids and mtp_select run as one launch recorded by
+        // the caller (capture_round / capture_step: mtp_head_tail) - the same words
+        if (chain_fuse2(kFuse2HeadTail)) return true;
         if (argmax_rows_wanted()) {
             argmax_rows(head_logits_, T, (int) nv, arg_scratch_, out_ids_, cs);
         } else {
```

### 13-17

`src/core/mtp.cpp` · bool MtpDrafter::capture_round(int T, bool coupled, std::string& err) {

Anchor:
```
        coupled_rec_ = false;
    }
    const bool fold = cp_chain_fuse() && force_on_ && !coupled;   // critpath: force_token inside mtp_select
```
Block (hunk 10 of `src/core/mtp.cpp` in the patch, 16 lines):
```diff
         coupled_rec_ = false;
     }
     const bool fold = cp_chain_fuse() && force_on_ && !coupled;   // critpath: force_token inside mtp_select
+    if (ok && !coupled && chain_fuse2(kFuse2HeadTail)) {
+        // chainfuse: record_rest left the head logits; the pick, its probability, map_ids, mtp_select and the forcing in
+        // one launch
+        mtp_head_tail(head_logits_, (int) (dhead_ != nullptr ? n_dvocab_ : n_vocab_), dhead_ != nullptr ? dvocab_ : nullptr,
+                      R_, HCN, row_ + 1, Rin_, out_ids_, probs_, tok_, m_out_, 0, m_prob_, force_on_ ? m_force_ : nullptr,
+                      tail_scratch_, cs_);
+    } else {
     if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, 0, cs_, probs_, m_prob_, fold ? m_force_ : nullptr);
     if (ok && force_on_ && !coupled && !fold) force_token(tok_, m_force_, 0, cs_);   // chain_launch: step 1's input
+    }
     return finish_capture(cs_, ok, exec, coupled ? "round (coupled)" : "round", err);
 }

```

### 13-18

`src/core/mtp.cpp` · bool MtpDrafter::capture_step(int j, bool coupled, std::string& err) {

Anchor:
```
    bool ok = record_forward(1, row, cs_, err);
    coupled_rec_ = false;
    const bool fold = cp_chain_fuse() && force_on_ && !coupled;   // critpath: force_token inside mtp_select
```
Block (hunk 11 of `src/core/mtp.cpp` in the patch, 14 lines):
```diff
     bool ok = record_forward(1, row, cs_, err);
     coupled_rec_ = false;
     const bool fold = cp_chain_fuse() && force_on_ && !coupled;   // critpath: force_token inside mtp_select
+    if (ok && !coupled && chain_fuse2(kFuse2HeadTail)) {   // chainfuse: the step's tail in one launch (capture_round)
+        mtp_head_tail(head_logits_, (int) (dhead_ != nullptr ? n_dvocab_ : n_vocab_), dhead_ != nullptr ? dvocab_ : nullptr,
+                      R_, HCN, row_ + 1, Rin_, out_ids_, probs_, tok_, m_out_, j, m_prob_, force_on_ ? m_force_ : nullptr,
+                      tail_scratch_, cs_);
+    } else {
     if (ok) mtp_select(R_, HCN, out_ids_, row_ + 1, Rin_, tok_, m_out_, j, cs_, probs_, m_prob_, fold ? m_force_ : nullptr);
     if (ok && force_on_ && !coupled && !fold) force_token(tok_, m_force_, j, cs_);   // chain_launch: step j+1's input
+    }
     return finish_capture(cs_, ok, exec, coupled ? "step (coupled)" : "step", err);
 }

```

### 13-19

`src/kernels/chain_step_bench.cpp` · (new file)

New file, 568 lines: see the patch.

### 13-20

`src/kernels/cuda/native_gr_postops.cu` · void native_gr_post_multi(const float* residual, const float* block_out, const f

Anchor:
```
        residual, block_out, inject, output, n_embd, hc, 1.0f / float(hc), r_stride, b_stride, i_stride);
    check_launch();
}
```
Block (hunk 1 of `src/kernels/cuda/native_gr_postops.cu` in the patch, 67 lines):
```diff
         residual, block_out, inject, output, n_embd, hc, 1.0f / float(hc), r_stride, b_stride, i_stride);
     check_launch();
 }
+
+// ======================= chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1) ==========================================
+// native_moe_combine's combine_k10_vec4 (native_moe.cu, compiled with the same --use_fast_math as this file) and `post`
+// above in ONE launch for one token: a thread owns 4 columns; y = the vec4 kernel's expression (the first product
+// rounded on its own, one fmaf per expert in order, the shared row added), then post's expression per stream,
+// `__fmaf_rn(y, weight_c, residual)`.  `y_out` (nullable) also keeps the combined row.  output may equal residual
+// exactly (each thread reads and writes only its own elements).
+namespace {
+__global__ void combine_k10_post(const float4* __restrict__ parts4, const float* __restrict__ weights,
+                                 const float4* __restrict__ shared4, float4* __restrict__ y4, const float* residual,
+                                 const float* __restrict__ inject, float* output, int n_embd, int hc, float scale,
+                                 long long n4) {
+    const long long c4 = (long long) blockIdx.x * blockDim.x + threadIdx.x;
+    if (c4 >= n4) return;
+    const float w0 = __ldg(weights + 0);
+    const float4 p0 = parts4[c4];
+    float4 sum = make_float4(p0.x * w0, p0.y * w0, p0.z * w0, p0.w * w0);
+#pragma unroll
+    for (int expert = 1; expert < 10; ++expert) {
+        const float w = __ldg(weights + expert);
+        const float4 p = parts4[(long long) expert * n4 + c4];
+        sum.x = fmaf(p.x, w, sum.x);
+        sum.y = fmaf(p.y, w, sum.y);
+        sum.z = fmaf(p.z, w, sum.z);
+        sum.w = fmaf(p.w, w, sum.w);
+    }
+    const float4 sh = shared4[c4];
+    sum.x += sh.x;
+    sum.y += sh.y;
+    sum.z += sh.z;
+    sum.w += sh.w;
+    if (y4 != nullptr) y4[c4] = sum;
+    for (int c = 0; c < hc; ++c) {
+        const float weight = scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
+        const std::size_t o = std::size_t(c) * n_embd + std::size_t(c4) * 4;
+        const float4 r = *reinterpret_cast<const float4*>(residual + o);
+        float4 v;
+        v.x = __fmaf_rn(sum.x, weight, r.x);
+        v.y = __fmaf_rn(sum.y, weight, r.y);
+        v.z = __fmaf_rn(sum.z, weight, r.z);
+        v.w = __fmaf_rn(sum.w, weight, r.w);
+        *reinterpret_cast<float4*>(output + o) = v;
+    }
+}
+} // namespace
+
+void native_moe_combine_k10_post(const float* parts, const float* weights, const float* shared, float* y_out,
+                                 const float* residual, const float* inject, float* output, int n_embd, int hc,
+                                 void* stream) {
+    check_shape(n_embd, hc);
+    check_pointer(parts); check_pointer(weights); check_pointer(shared); check_pointer(residual);
+    check_pointer(inject); check_pointer(output);
+    if (n_embd % 4 != 0 ||
+        ((reinterpret_cast<std::uintptr_t>(parts) | reinterpret_cast<std::uintptr_t>(shared) |
+          reinterpret_cast<std::uintptr_t>(residual) | reinterpret_cast<std::uintptr_t>(output) |
+          reinterpret_cast<std::uintptr_t>(y_out)) & 15u) != 0)
+        throw std::invalid_argument("native combine+post requires n_embd a multiple of 4 and 16-byte aligned rows");
+    const long long n4 = n_embd / 4;
+    combine_k10_post<<<unsigned((n4 + 127) / 128), 128, 0, static_cast<cudaStream_t>(stream)>>>(
+        reinterpret_cast<const float4*>(parts), weights, reinterpret_cast<const float4*>(shared),
+        reinterpret_cast<float4*>(y_out), residual, inject, output, n_embd, hc, 1.0f / float(hc), n4);
+    check_launch();
+}
 } // namespace strata::kernels
```

### 13-21

`src/kernels/cuda/native_qsa.cu` · (file scope)

Anchor:
```
// SOFTWARE.

#include "strata/kernels/native_qsa.hpp"
```
Block (hunk 1 of `src/kernels/cuda/native_qsa.cu` in the patch, 8 lines):
```diff
 // SOFTWARE.

 #include "strata/kernels/native_qsa.hpp"
+#include "strata/kernels/q8_1_finite.hpp"   // chainfuse: the q8_1 quantizer's helpers (native_mmvq.cu's)
+#include <cuda_fp16.h>
 #include <cuda_runtime.h>
 #include <atomic>
 #include <cmath>
```

### 13-22

`src/kernels/cuda/native_qsa.cu` · void native_qsa_gate_apply(const float* attn, const float* q_full, float* output

Anchor:
```
    gate<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(attn, q_full, output, n_head, head_dim);
    check_launch();
}
```
Block (hunk 2 of `src/kernels/cuda/native_qsa.cu` in the patch, 99 lines):
```diff
     gate<<<unsigned((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(attn, q_full, output, n_head, head_dim);
     check_launch();
 }
+
+// ======================= chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1) ==========================================
+// The MTP drafter's chain quantizes the norm's and the gate's outputs to q8_1 right away (native_quantize_q8_1_kernel,
+// native_mmvq.cu: thread i is element i, so a warp is one 32-element block; amax = warp_max, sum = warp_sum of the raw
+// values, d = q8_1_finite(amax / 127), q = roundf(x / d) clamped, ds = (d, sum) as fp16).  Here the quantizer's
+// expressions run in the producing kernel's epilogue on the same lanes: in `norm` lane l of warp w holds column
+// 32 w + l + BlockSize k, so a warp's 32 columns are exactly one block in the quantizer's lane order (n_cols is a
+// multiple of 32, so a warp is whole or absent in every iteration); in `gate` thread i is element i as there.
+// This file compiles with --use_fast_math like native_mmvq.cu, and q8_1_finite.hpp's helpers inline under the same
+// flags: the same bytes as the two launches.  Parity: src/kernels/chain_step_bench.cpp.
+namespace {
+struct Q81Out { __half2 ds; int8_t qs[32]; };   // native_mmvq.cu's Q81Block layout (36 bytes)
+static_assert(sizeof(Q81Out) == 36 && alignof(Q81Out) == 4, "q8_1 block layout");
+__device__ __forceinline__ float warp_max(float value) {   // native_mmvq.cu's warp_max: the same pairing order
+#pragma unroll
+    for (int offset = 16; offset; offset >>= 1)
+        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset, 32));
+    return value;
+}
+// the quantizer's block on the warp's 32 lanes: `v` is this lane's value, `i` its element index
+__device__ __forceinline__ void q8_1_block(float v, std::size_t i, Q81Out* y) {
... 75 more lines: see 13-chain-fuse.patch
```

### 13-23

`src/kernels/cuda/qsa_decode_attn.cu` · __global__ void __launch_bounds__(HD) attn_merge_v2_kernel(const float* __restri

Anchor:
```
#if defined(STRATA_HIP_GFX906)
// The compat backend can target other architectures. Check the calling thread's current device,
// caching only its last ordinal so a layer split or another thread cannot reuse the wrong answer.
```
Block (hunk 1 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 365 lines):
```diff
     attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
 }

+// ======================= chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1): the drafter's dense attention =========
+// attn_chunk_kernel_pre75's block is a chain of dependent memory round trips: a warp scores its 8 cells two at a time
+// (4 trips), a thread folds the 64 cells' values four at a time (16 trips).  On the 2080 Ti over 32k cells (1024
+// blocks, 3 a SM, 5 waves) a wave takes ~48 us: 240 us for one query (nsys, DET-cp-nsys5).  attn_merge_v2_kernel
+// (24 blocks, 32 batches of 16 dependent loads a thread) takes another 50 us.
+// The kernels below keep EVERY expression and EVERY fold order of attn_chunk_kernel_pre75<M, 2, false> and
+// attn_merge_v2_kernel<16> - the same scores (the same dot expression per cell, reduce12), the same chunk max / sum /
+// cell-major probabilities, the same ascending fold of the values, the same merge of the chunks in chunk order - and
+// change only how the loads are issued: the values of a stage of cells come through shared memory with one 16-byte
+// load a thread (the next stage fetched while this one folds), the next pair of keys' raw words may be fetched before
+// the current pair's scores (KPF), the first stage of values before the score phase (EARLY); the merge stages each
+// chunk's m, l and weight in shared memory once per block (every dimension needs the same three numbers) and keeps U
+// accumulator loads in flight.  Raw words are converted with load8 / load_v1's own expressions at use.
+// A first attempt (all 8 keys of a warp prefetched, 8-32 single-byte values a thread in flight, one cell at a time)
+// was 1.5-2.5x SLOWER than pre75 on the 2080 Ti: 80 registers with spills (cuobjdump: STACK 40-224 B against pre75's
+// 73 / 0) and twice the shared-memory traffic for q.  Hence the pre75 structure kept here.
+// CUDA only, INT8 and FP16 pools (the drafter's: --kv int8 / f16); Q4_0 pools keep attn_chunk_kernel_pre75.
+// Parity: src/kernels/chain_step_bench.cpp (memcmp of the partials and of the output against the pre75 + v2 pair).
+#if !defined(__HIPCC__) && defined(STRATA_ATTN_PRE75_BUILT)
+template <int KV_MODE> struct RawK;   // one lane's 8 key values, undecoded
+template <> struct RawK<0> { uint4 w; };
... 341 more lines: see 13-chain-fuse.patch
```

### 13-24

`src/kernels/cuda/qsa_decode_attn.cu` · int attn_merge_v2() {

Anchor:
```
// the merge of `n_q` queries' partials: v2 = 0 attn_merge_kernel, 1 attn_merge_v2_kernel (bitwise the same output)
void launch_merge(int v2, const float* part_acc, const float* part_m, const float* part_l, int n_chunks,
                  const int32_t* steps, float* attn, long long stride, unsigned n_head, unsigned n_q, cudaStream_t st) {
```
Block (hunk 2 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 12 lines):
```diff
 // the merge of `n_q` queries' partials: v2 = 0 attn_merge_kernel, 1 attn_merge_v2_kernel (bitwise the same output)
 void launch_merge(int v2, const float* part_acc, const float* part_m, const float* part_l, int n_chunks,
                   const int32_t* steps, float* attn, long long stride, unsigned n_head, unsigned n_q, cudaStream_t st) {
+#if defined(STRATA_ATTN_FUSE2_BUILT)
+    if (v2 == 2)   // chainfuse: the staged merge (bit for bit v2's output)
+        attn_merge_v3_kernel<kMergeV3U><<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, steps,
+                                                                           attn, stride);
+    else
+#endif
     if (v2)
         attn_merge_v2_kernel<kMergeU><<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, steps,
                                                                          attn, stride);
```

### 13-25

`src/kernels/cuda/qsa_decode_attn.cu` · void launch_chunks_batch(const float* q, const QsaAttnPools& pools, const int32_

Anchor:
```
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
#if defined(STRATA_ATTN_PRE75_BUILT)
```
Block (hunk 3 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 29 lines):
```diff
     const float scale = 1.0f / sqrtf((float) HD);
     const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
 #if defined(STRATA_ATTN_PRE75_BUILT)
-    const bool pre = kernel < 0 ? pre75_attn() : kernel == 1;
+    // kernel >= 2 (chainfuse) on a Q4_0 pool: the environment's chunk kernel, as the ready-made engine
+    const bool pre = (kernel < 0 || kernel >= 2) ? pre75_attn() : kernel == 1;
+#endif
+#if defined(STRATA_ATTN_FUSE2_BUILT)
+    // chainfuse (docs/CHAINFUSE.md): 2..5 = attn_chunk_kernel_stage<M, KPF, EARLY> - 2: (0, 0), 3: (1, 0), 4: (0, 1),
+    // 5: (1, 1); INT8 and FP16 pools only (the drafter's), bit for bit attn_chunk_kernel_pre75's partials
+    if (kernel >= 2 && (kv_mode == 0 || kv_mode == 1)) {
+#define STRATA_ATTN_STAGE(M, KPF, EARLY) attn_chunk_kernel_stage<M, KPF, EARLY><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
+        if (kv_mode == 1) {
+            if (kernel == 2) STRATA_ATTN_STAGE(1, false, false);
+            else if (kernel == 3) STRATA_ATTN_STAGE(1, true, false);
+            else if (kernel == 4) STRATA_ATTN_STAGE(1, false, true);
+            else STRATA_ATTN_STAGE(1, true, true);
+        } else {
+            if (kernel == 2) STRATA_ATTN_STAGE(0, false, false);
+            else if (kernel == 3) STRATA_ATTN_STAGE(0, true, false);
+            else if (kernel == 4) STRATA_ATTN_STAGE(0, false, true);
+            else STRATA_ATTN_STAGE(0, true, true);
+        }
+#undef STRATA_ATTN_STAGE
+        return;
+    }
 #endif
     // S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
     static const bool lane_cell_env = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
```

### 13-26

`src/kernels/cuda/qsa_decode_attn.cu` · void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, cons

Anchor:
```
                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, int daily_chunk,
                                 void* stream) {
    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
```
Block (hunk 4 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 17 lines):
```diff
                                  int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, int daily_chunk,
                                  void* stream) {
     if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
-        !q || !ids || !steps || !pools.page_table || (daily_chunk != 0 && daily_chunk != 1)) {
+        !q || !ids || !steps || !pools.page_table || daily_chunk < 0 || daily_chunk > 5) {
         std::fprintf(stderr, "qsa_decode_attn_chunks_only: unsupported geometry or missing buffers\n");
         std::exit(1);
     }
+#if !defined(STRATA_ATTN_FUSE2_BUILT)
+    if (daily_chunk >= 2) {
+        std::fprintf(stderr, "qsa_decode_attn_chunks_only: the chainfuse chunk kernel is not built here\n");
+        std::exit(1);
+    }
+#endif
     launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, (cudaStream_t) stream, daily_chunk);
     const cudaError_t e = cudaGetLastError();
     if (e != cudaSuccess) {
```

### 13-27

`src/kernels/cuda/qsa_decode_attn.cu` · void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, cons

Anchor:
```
        (v2 != 0 && v2 != 1) || !steps || !attn) {
```
Block (hunk 5 of `src/kernels/cuda/qsa_decode_attn.cu` in the patch, 50 lines):
```diff
     }
 }

+bool qsa_decode_attn_fuse2_built() {
+#if defined(STRATA_ATTN_FUSE2_BUILT)
+    return true;
+#else
+    return false;
+#endif
+}
+
+void qsa_decode_attn_batch_fuse2(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
+                                 int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, int variant,
+                                 int merge, void* stream) {
+    if (n_q <= 0) return;
+    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
+        !pools.page_table || n_q > 65535 || variant < 0 || variant > 4 || (merge != 0 && merge != 1)) {
+        std::fprintf(stderr, "qsa_decode_attn_batch_fuse2: unsupported geometry or missing buffers\n");
+        std::exit(1);
+    }
+#if defined(STRATA_ATTN_FUSE2_BUILT)
+    cudaStream_t st = (cudaStream_t) stream;
+    // variant 0: the environment's chunk kernel (STRATA_ATTN_DAILY), only the merge changes; 1..4: the staged kernel
+    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, st, variant == 0 ? -1 : variant + 1);
+    launch_merge_batch(steps, cap, s, scratch, attn, n_q, merge ? 2 : attn_merge_v2(), st);
+    const cudaError_t e = cudaGetLastError();
+    if (e != cudaSuccess) {
+        std::fprintf(stderr, "qsa_decode_attn_batch_fuse2: %s\n", cudaGetErrorString(e));
+        std::exit(1);
+    }
+#else
+    (void) variant; (void) merge;
+    qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scratch, attn, n_q, stream);
+#endif
+}
+
 void qsa_decode_attn_merge_only(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch,
                                 float* attn, int64_t n_q, int v2, void* stream) {
+#if !defined(STRATA_ATTN_FUSE2_BUILT)
+    if (v2 == 2) {
+        std::fprintf(stderr, "qsa_decode_attn_merge_only: the chainfuse merge is not built here\n");
+        std::exit(1);
+    }
+#endif
     if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
-        (v2 != 0 && v2 != 1) || !steps || !attn) {
+        v2 < 0 || v2 > 2 || !steps || !attn) {
         std::fprintf(stderr, "qsa_decode_attn_merge_only: unsupported geometry or missing buffers\n");
         std::exit(1);
     }
```

### 13-28

`src/kernels/cuda/verify_kernels.cu` · __global__ void row_top_prob_split_kernel(const float* __restrict__ logits, int

Anchor:
```
    rows[row].counter = 0;
}
```
Block (hunk 1 of `src/kernels/cuda/verify_kernels.cu` in the patch, 104 lines):
```diff
     rows[row].counter = 0;
 }

+// chainfuse (docs/CHAINFUSE.md, STRATA_CHAIN_FUSE2=1): a draft step's tail in ONE launch - sample_tokens' greedy pick
+// over the draft head's row 0 (argmax_rows' slices and rule: the lowest index of the largest value above -inf, NaN
+// never, 0 when there is none), row_top_prob's probability of it (row_top_prob_split's tree: virtual thread vt of 1024
+// sums i = vt, vt + 1024, ... with __expf(l[i] - m), xor warp sums, the 32 warp sums added in order, 1 / tot), map_ids
+// (`table`, null for the whole vocabulary) and mtp_select with force_token folded in (the same words).  Grid (nb),
+// 1024 threads: every block copies its share of mtp_select's R row and scans its slice of the logits; the last block
+// to finish merges the picks and does the rest.  Replaces five launches (sampler_greedy / row_top_prob_split /
+// map_ids / mtp_select / force_token), ~34 us of one-block kernels on the 2080 Ti.  `scratch`: an ArgRow (zero before
+// the first launch, left so).  Row 0 only: the drafter's full layer runs one row at a time (record_rest).
+__global__ void __launch_bounds__(1024) mtp_head_tail_kernel(const float* __restrict__ logits, int n, int per_block,
+                                                            const int32_t* __restrict__ table,
+                                                            const float* __restrict__ R_src, int64_t stride,
+                                                            const int32_t* __restrict__ row_dev,
+                                                            float* __restrict__ R_dst, int32_t* out_ids,
+                                                            float* probs_dev, int32_t* tok_dst, int32_t* out, int j,
+                                                            float* out_p, const int32_t* force, ArgRow* __restrict__ sc) {
+    const int b = (int) blockIdx.x, nb = (int) gridDim.x;
+    const int row = *row_dev;
+    // mtp_select's copy of row `row`, spread over every block
+    for (int64_t i = (int64_t) b * blockDim.x + threadIdx.x; i < stride; i += (int64_t) nb * blockDim.x)
+        R_dst[i] = R_src[(size_t) row * stride + i];
... 80 more lines: see 13-chain-fuse.patch
```

### 13-29

`src/kernels/cuda/verify_kernels.cu` · void argmax_rows(const float*, int, int, void*, int32_t*, void*) {

Anchor:
```
    std::fprintf(stderr, "argmax_rows: not built for this backend\n");
    std::exit(1);
}
```
Block (hunk 2 of `src/kernels/cuda/verify_kernels.cu` in the patch, 12 lines):
```diff
     std::fprintf(stderr, "argmax_rows: not built for this backend\n");
     std::exit(1);
 }
+uint64_t mtp_head_tail_scratch_bytes() { return 0; }
+void mtp_head_tail(const float*, int, const int32_t*, const float*, int64_t, const int32_t*, float*, int32_t*, float*,
+                   int32_t*, int32_t*, int, float*, const int32_t*, void*, void*) {
+    std::fprintf(stderr, "mtp_head_tail: not built for this backend (STRATA_CHAIN_FUSE2 is CUDA only)\n");
+    std::exit(1);
+}
 uint64_t row_top_prob_scratch_bytes(int) { return 0; }
 void row_top_prob_split(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void*,
                         void* stream) {
```

## 14 side-mixer

`14-side-mixer.patch` — FABLE-STRATA-Porta su 0.1.43 STRATA_SIDE_MIXER=1/2 (PR #1889): il lavoro del mixer che legge solo l'input su rami del grafo

### 14-1

`include/strata/core/verify.hpp` · private:

Anchor:
```
    cudaEvent_t df_join_[2] = {};
    cudaGraphExec_t exec_[9] = {};
    cudaGraphExec_t exec_nr_[9] = {};   // #871: the doorbell variant of a stage that is all-resident otherwise
```
Block (hunk 1 of `include/strata/core/verify.hpp` in the patch, 13 lines):
```diff
     cudaEvent_t df_join_[2] = {};
     cudaGraphExec_t exec_[9] = {};
     cudaGraphExec_t exec_nr_[9] = {};   // #871: the doorbell variant of a stage that is all-resident otherwise
+    // fork 34a35db (STRATA_SIDE_MIXER=1): the GDN gates and the QSA indexer on a branch of the window's graph beside the
+    // mixer's projections, joined before the recurrence / the attention
+    cudaStream_t side_ = nullptr;
+    cudaEvent_t mfork_ = nullptr, mjoin_ = nullptr;
+    // fork b063c13 (STRATA_SIDE_MIXER=2): a second branch for the GDN gate projection and the QSA queries' chain
+    cudaStream_t side2_ = nullptr;
+    cudaEvent_t pfork_ = nullptr, pjoin_ = nullptr;
     cudaGraphExec_t commit_exec_ = nullptr;

     // mapped staging (host pointer, device alias)
```

### 14-2

`src/core/verify.cpp` · inline bool g_qfuse_hc() {

Anchor:
```
inline bool g_qfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_QFUSE"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
#define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)
```
Block (hunk 1 of `src/core/verify.cpp` in the patch, 9 lines):
```diff
     return false;
 #endif
 }
+// fork 34a35db + b063c13 (Eddoursul), STRATA_SIDE_MIXER=1 / =2 (opt-in): the mixer's work that reads only the layer's
+// input on branches of the window's graph (bitwise the same outputs); 0 / unset = in line, and no branch stream is created
+inline int g_side_mixer() { static const int m = [] { const char* v = std::getenv("STRATA_SIDE_MIXER"); return v ? std::max(0, std::min(2, std::atoi(v))) : 0; }(); return m; }
 inline bool g_qfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_QFUSE"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
 #define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)

```

### 14-3

`src/core/verify.cpp` · Verifier::~Verifier() {

Anchor:
```
    if (cs_ && cs_ != ext_stream_) cudaStreamDestroy(cs_);   // set_stream: the stage's stream, shared, not ours
    if (sh_cs_) cudaStreamDestroy(sh_cs_);
    if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
```
Block (hunk 2 of `src/core/verify.cpp` in the patch, 10 lines):
```diff
     if (cs_ && cs_ != ext_stream_) cudaStreamDestroy(cs_);   // set_stream: the stage's stream, shared, not ours
     if (sh_cs_) cudaStreamDestroy(sh_cs_);
     if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
+    if (side_) cudaStreamDestroy(side_);
+    if (side2_) cudaStreamDestroy(side2_);
+    for (cudaEvent_t e : {mfork_, mjoin_, pfork_, pjoin_})
+        if (e) cudaEventDestroy(e);
     if (commit_done_) cudaEventDestroy(commit_done_);
     if (ev_done_) cudaEventDestroy(ev_done_);
     if (ev_commit_) cudaEventDestroy(ev_commit_);
```

### 14-4

`src/core/verify.cpp` · bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&

Anchor:
```
#else
        df_branch_ = v != nullptr && std::atoi(v) != 0;
#endif
```
Block (hunk 3 of `src/core/verify.cpp` in the patch, 19 lines):
```diff
 #else
         df_branch_ = v != nullptr && std::atoi(v) != 0;
 #endif
+        // PR #1889 (fork lab/ports1): STRATA_SIDE_MIXER=1/2 branches a superset of this switch's steps (the GDN a/b and
+        // z, the QSA queries and the indexer's query; plus the indexer's key, snapshot, append and top-k), so the two
+        // never run together: STRATA_SIDE_MIXER wins, and this one's streams are not even created.  Said once when both
+        // were asked for explicitly; the Windows default of this switch gives way silently.
+        if (df_branch_ && g_side_mixer() > 0) {
+            static bool said = false;
+            if (v != nullptr && !said) {
+                std::fprintf(stderr, "strata: STRATA_SIDE_MIXER=%d is set: STRATA_DF_BRANCH is off (the side mixer includes "
+                                     "its branches)\n", g_side_mixer());
+                said = true;
+            }
+            df_branch_ = false;
+        }
         if (df_branch_ && df_fork_ == nullptr) {
             bool okb = cudaEventCreateWithFlags(&df_fork_, cudaEventDisableTiming) == cudaSuccess;
             for (int i = 0; i < 2 && okb; ++i)
```

### 14-5

`src/core/verify.cpp` · bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&

Anchor:
```
            all_resident_ = all_ok;
        }
    }
```
Block (hunk 4 of `src/core/verify.cpp` in the patch, 16 lines):
```diff
             all_resident_ = all_ok;
         }
     }
+    if (g_side_mixer() > 0 &&
+        (cudaStreamCreateWithFlags(&side_, cudaStreamNonBlocking) != cudaSuccess ||
+        cudaEventCreateWithFlags(&mfork_, cudaEventDisableTiming) != cudaSuccess ||
+        cudaEventCreateWithFlags(&mjoin_, cudaEventDisableTiming) != cudaSuccess ||
+        cudaStreamCreateWithFlags(&side2_, cudaStreamNonBlocking) != cudaSuccess ||
+        cudaEventCreateWithFlags(&pfork_, cudaEventDisableTiming) != cudaSuccess ||
+        cudaEventCreateWithFlags(&pjoin_, cudaEventDisableTiming) != cudaSuccess)) {
+        err = "verify: the mixer branch's stream / events could not be created";
+        return false;
+    }
     // E-6: a layer whose routed experts are all resident is planned on the device (STRATA_VERIFY_DEVICE_PLAN=1: on;
     // exact, but neutral on RIBPC 1-2 GPUs: off by default)
     {
```

### 14-6

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
    // Pipelined windows must wait for the host verdict/snapshot fence before mutating state.
    const bool self_commit = T == 1 && !always_publish_ && !batch_rec_ && !g_qfuse() && one_token_self_commit();   // see Verifier::commit
    static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
```
Block (hunk 5 of `src/core/verify.cpp` in the patch, 17 lines):
```diff
     // Pipelined windows must wait for the host verdict/snapshot fence before mutating state.
     const bool self_commit = T == 1 && !always_publish_ && !batch_rec_ && !g_qfuse() && one_token_self_commit();   // see Verifier::commit
     static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
+    // fork 34a35db: the GDN gates and the QSA indexer beside the projections (STRATA_SIDE_MIXER=1; 0 / unset = in line)
+    const int side_mixer_mode = g_side_mixer();
+    const bool side_mixer = side_mixer_mode >= 1;
+    const bool side_proj = side_mixer_mode >= 2 && !batch_rec_;   // fork b063c13: the GDN gate projection, the QSA queries
+    auto branch = [&](cudaEvent_t e, cudaStream_t from, cudaStream_t to, const char* what) {
+        if (cudaEventRecord(e, from) == cudaSuccess && cudaStreamWaitEvent(to, e, 0) == cudaSuccess) return true;
+        err = std::string("verify: the mixer branch could not ") + what;
+        return false;
+    };
+    // a batch window's rows each read their own slot's indexer: in line there
+    const bool sidem = side_mixer && !batch_rec_;
     auto stamp = [&](int64_t l, int i, int grp) {
         if (prof_on_ && grp == 0) gpu_stamp(prof_, (int) (l * kProfPer + i), cs);
         if (trace_m_ != nullptr) gpu_stamp(trace_m_, (int) ((l * kProfPer + i) * 2 + grp), cs);   // #649
```

### 14-7

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
    // makes side i depend on everything captured on `cs` so far, join(i) makes `cs` depend on everything captured on
    // side i: a branch never races with what came before its fork or after its join.  Not in a split or batch window,
    // nor under the stage profiler (whose stamps are on `cs`).
```
Block (hunk 6 of `src/core/verify.cpp` in the patch, 8 lines):
```diff
     // makes side i depend on everything captured on `cs` so far, join(i) makes `cs` depend on everything captured on
     // side i: a branch never races with what came before its fork or after its join.  Not in a split or batch window,
     // nor under the stage profiler (whose stamps are on `cs`).
-    const bool br = df_branch_ && G == 1 && !batch_rec_ && !prof_on_;
+    const bool br = df_branch_ && G == 1 && !batch_rec_ && !prof_on_ && !side_mixer;   // STRATA_SIDE_MIXER wins
     auto fork = [&](int i) {
         cudaEventRecord(df_fork_, cs);
         cudaStreamWaitEvent(df_side_[i], df_fork_, 0);
```

### 14-8

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
                float* hb = h_L_ + (size_t) gi * MT * C;
                float* gate = gate_L_ + (size_t) gi * MT * HV;
                float* beta = beta_L_ + (size_t) gi * MT * HV;
```
Block (hunk 7 of `src/core/verify.cpp` in the patch, 30 lines):
```diff
                 float* hb = h_L_ + (size_t) gi * MT * C;
                 float* gate = gate_L_ + (size_t) gi * MT * HV;
                 float* beta = beta_L_ + (size_t) gi * MT * HV;
+                if (sidem) {   // the gates (a few blocks, latency-bound) beside the projections, joined below
+                    if (!branch(mfork_, cs, side_, "fork")) return false;
+                    gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
+                                 (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N,
+                                 (int) HV, n, side_);
+                    if (cudaEventRecord(mjoin_, side_) != cudaSuccess) { err = "verify: the GDN gates' branch could not join"; return false; }
+                }
                 if (!q8_attn) native_quantize_q8_1(xm, xq_, (int) N, n, cs);
+                if (side_proj) {   // the gate projection (compute-bound for i-quants) beside qkv and the conv
+                    // the branch reads F4's interleaved copy too (written on cs before the fork; bitwise native_mmvq's
+                    // output), and qkv is recorded before the branch's launch, so its grid is dispatched first and the
+                    // conv / recurrence that wait for it start sooner (z is read only after the recurrence)
+                    const bool g_il = g_mmvq_il() && strata::kernels::native_mmvq_il_supported(wg->native_type, n, (int) ZV);
+                    if (g_il && !il_ready && n >= 2) {   // one column reads the plain q8_1 blocks
+                        strata::kernels::native_q8_1_interleave(xq_, xil_, (int) N, n, cs);
+                        il_ready = true;
+                    }
+                    if (!branch(pfork_, cs, side2_, "fork")) return false;
+                    mm(wqkv, qkv + (size_t) tb * C, (int) N, (int) C);
+                    if (g_il) strata::kernels::native_mmvq_il(wg->native_type, wg->native_data, xq_, xil_, z_ + (size_t) tb * ZV,
+                                                              (int) N, (int) ZV, n, side2_);
+                    else native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, side2_);
+                    if (cudaEventRecord(pjoin_, side2_) != cudaSuccess) { err = "verify: the gate projection's branch could not join"; return false; }
+                }
                 if (br) {   // STRATA_DF_BRANCH: a/b and z beside q/k/v and the conv; all of them read only this layer's input
                     fork(0);
                     gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
```

### 14-9

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
                    native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,
                                df_side_[1]);
                }
```
Block (hunk 8 of `src/core/verify.cpp` in the patch, 10 lines):
```diff
                     native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,
                                 df_side_[1]);
                 }
-                const bool fuse_ab = fuse_convab_on() && !br && !batch_rec_ && !prof_on_;
-                mm(wqkv, qkv + (size_t) tb * C, (int) N, (int) C);
+                const bool fuse_ab = fuse_convab_on() && !br && !sidem && !batch_rec_ && !prof_on_;   // a/b fused in the conv: not with a/b on a branch
+                if (!side_proj) mm(wqkv, qkv + (size_t) tb * C, (int) N, (int) C);   // with side_proj: above, first
                 stamp(l, 2, grp);
                 if (batch_rec_) {   // contiguous rows may be proposals for the same slot
                     for (int t = tb; t < te;) {
```

### 14-10

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
                    join(0);
                    join(1);
                } else {
```
Block (hunk 9 of `src/core/verify.cpp` in the patch, 23 lines):
```diff
                     join(0);
                     join(1);
                 } else {
-                if (!fuse_ab)
+                if (!fuse_ab && !sidem)
                 gdn_ab_multi(xm, (const uint16_t*) wa->data, (const uint16_t*) wb->data, (const float*) wdt->data,
                              (const float*) wsa->data, gate + (size_t) tb * HV, beta + (size_t) tb * HV, (int) N, (int) HV,
                              n, cs);
                 stamp(l, 4, grp);
-                mm(wg, z_ + (size_t) tb * ZV, (int) N, (int) ZV);
+                if (!side_proj) mm(wg, z_ + (size_t) tb * ZV, (int) N, (int) ZV);
                 stamp(l, 5, grp);
+                if (side_proj && cudaStreamWaitEvent(cs, pjoin_, 0) != cudaSuccess) {
+                    err = "verify: the gate projection's branch could not join";
+                    return false;
+                }
+                if (sidem && cudaStreamWaitEvent(cs, mjoin_, 0) != cudaSuccess) {
+                    err = "verify: the GDN gates' branch could not join";
+                    return false;
+                }
                 }
                 // the recurrence from the untouched state over tokens [0, te); outputs only for this group's
                 if (batch_rec_) {   // each slot's recurrence over its own proposed-token group
```

### 14-11

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
                                              IQ * ID, N, IQ * ID, n, df_side_[1]);
                    norm_rope_on(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, df_side_[1]);
                }
```
Block (hunk 10 of `src/core/verify.cpp` in the patch, 129 lines):
```diff
                                               IQ * ID, N, IQ * ID, n, df_side_[1]);
                     norm_rope_on(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, df_side_[1]);
                 }
-                if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, cs);
-                else for (int t = tb; t < te; ++t)
-                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, cs);
+                // STRATA_SIDE_MIXER=1 / =2 (fork Eddoursul 34a35db + b063c13): the indexer's key projection,
+                // the tail's snapshot, its keys appended, its query projection, norm and RoPE, the block scores and the
+                // top-k read only the mixer's input and the indexer's own state - =1 runs them on a branch beside the
+                // attention's projections; =2 also the queries' chain on a second branch.  Each step below is
+                // the in-line code with its stream as a parameter; with the switch off they run on cs in the same order
+                // as before (STRATA_DF_BRANCH, the same idea for fewer of these steps, is off whenever this is on).
+                auto norm_rope_s = [&](float* data, const WeightRef* norm, int rows, int cols, const int32_t* pos, cudaStream_t ns) {
+                    if (fuse_nr && native_norm_rope_usable(cols, (int) s.n_rot)) {
+                        native_qsa_rms_norm_rope(data, cols, (const float*) norm->data, data, rows, cols, (int) s.n_rot, EPS,
+                                                 rope_scaling(), pos, ns);
+                        return;
+                    }
+                    norm_rope_on(data, norm, rows, cols, pos, ns);
+                };
+                auto idx_key = [&](cudaStream_t ns) {
+                    if (qb) bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wik->data, idx_raw + tb * ID, ID, N, ID, n, ns);
+                    else for (int t = tb; t < te; ++t)
+                        bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wik->data, idx_raw + t * ID, (int) N, (int) ID, ns);
... 105 more lines: see 14-side-mixer.patch
```

### 14-12

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
                    fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                }
                stamp(l, 8, grp);
```
Block (hunk 11 of `src/core/verify.cpp` in the patch, 14 lines):
```diff
                     fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                 }
                 stamp(l, 8, grp);
-                if (batch_rec_) {   // snapshot each slot's indexer tail before its first proposed row
-                    for (int t = tb; t < te; ++t)
-                        if (t == tb || brow_[t] != brow_[t - 1])
-                            copy_from_mapped(tail_snap_b_ + ((size_t) brow_[t] * nQall + qi) * TS,
-                                             slot_ss(t).qsa_states[qi].idx_tail, TS, cs);
-                } else
-                if (grp == 0) copy_from_mapped(tail_snap_ + (size_t) qi * TS, st.idx_tail, TS, cs);
+                if (!sidem) idx_snap(cs);
                 // #783 PR-d (stuchapin909): the window's K/V cells in one launch per pool instead of one per token (a
                 // batch's rows each have their own K/V, so that case keeps the loop)
                 if (dec_batch && !g_no_batch_kv && !batch_rec_) {
```

### 14-13

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
                        kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                       vcur_ + t * NKV * HD, s, cs, &st.host);
                }
```
Block (hunk 12 of `src/core/verify.cpp` in the patch, 64 lines):
```diff
                         kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                        vcur_ + t * NKV * HD, s, cs, &st.host);
                 }
-                if (dec_batch && !g_no_batch_kv && !batch_rec_) {
-                    const QsaIndexerBuffers ib{st.idx_tail, st.idx_dead, st.idx_pooled, st.idx_block_pos};
-                    native_qsa_indexer_append_steps(idx_raw + tb * ID, step_ + tb * kStepCount + kStepPos, kStepCount,
-                                                    n, 0, (const float*) wikn->data, EPS, ib, s, st.max_cells,
-                                                    rope_scaling(), cs);
-                } else {
-                    for (int t = tb; t < te; ++t) {
-                        const QsaState& sx = slot_ss(t).qsa_states[qi];
-                        const QsaIndexerBuffers ib{sx.idx_tail, sx.idx_dead, sx.idx_pooled, sx.idx_block_pos};
-                        native_qsa_indexer_append(idx_raw + t * ID, step_ + t * kStepCount + kStepPos, 0,
-                                                  (const float*) wikn->data, EPS, ib, s, sx.max_cells,
-                                                  rope_scaling(), cs);
-                    }
-                }
+                if (!sidem) idx_append(cs);
                 stamp(l, 9, grp);
                 if (qbr) {
                     join(1);   // the indexer query, for the block scores
                 } else {
-                mm(wq, qfull_ + tb * NH * 2 * HD, (int) N, (int) (NH * 2 * HD));
-                if (qb) {
-                    if (fuse_nr) {   // the q/gate split reads q straight out of the q|gate rows (stride 2 * HD)
-                        native_qsa_rms_norm_rope(qfull_ + tb * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data,
-                                                 qcur_ + tb * NH * HD, (int) (n * NH), (int) HD, (int) s.n_rot, EPS,
-                                                 rope_scaling(), pos_ + tb * NH, cs);
-                    } else {
-                    // the q/gate split as a copy kernel: the window's chain stays kernel to kernel (PDL)
-                    copy_rows_strided(qcur_ + tb * NH * HD, qfull_ + tb * NH * 2 * HD, (int64_t) n * NH, HD, 2 * HD, cs);
-                    norm_rope(qcur_ + tb * NH * HD, wqn, (int) (n * NH), (int) HD, pos_ + tb * NH);
-                    }
-                    if (st.kv_rot) fwht256_inplace_cuda(qcur_ + tb * NH * HD, (int64_t) n * NH, cs);   // <Hq, Hk> = <q, k>
-                    bf16_gemv_fp32_mmvf_multi(mixed_ + tb * N, N, (const uint16_t*) wiq->data, qidx_ + tb * IQ * ID, IQ * ID,
-                                              N, IQ * ID, n, cs);
-                    norm_rope(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ);
-                } else {
-                for (int t = tb; t < te; ++t) {
-                    float* qc = qcur_ + t * NH * HD;
-                    if (fuse_nr) {
-                        native_qsa_rms_norm_rope(qfull_ + t * NH * 2 * HD, (int) (2 * HD), (const float*) wqn->data, qc,
-                                                 (int) NH, (int) HD, (int) s.n_rot, EPS, rope_scaling(), pos_ + t * NH, cs);
-                    } else {
-                    if (cudaMemcpy2DAsync(qc, (size_t) HD * 4, qfull_ + t * NH * 2 * HD, (size_t) HD * 2 * 4,
-                                          (size_t) HD * 4, (size_t) NH, cudaMemcpyDeviceToDevice, cs) != cudaSuccess) {
-                        err = "verify: the q/gate split failed";
-                        return false;
-                    }
-                    norm_rope(qc, wqn, (int) NH, (int) HD, pos_ + t * NH);
-                    }
-                    if (st.kv_rot) fwht256_inplace_cuda(qc, NH, cs);   // <Hq, Hk> = <q, k>
-                }
-                for (int t = tb; t < te; ++t) {
-                    float* qx = qidx_ + t * IQ * ID;
-                    bf16_gemv_fp32_mmvf(mixed_ + t * N, (const uint16_t*) wiq->data, qx, (int) N, (int) (IQ * ID), cs);
-                    norm_rope(qx, wiqn, (int) IQ, (int) ID, pos_ + t * NH);
-                }
-                }
+                    if (!side_proj && !q_chain(cs)) return false;
+                    if (!sidem) idx_query(cs);
                 }
                 stamp(l, 10, grp);
                 if (batch_rec_) {   // each row selects and attends over its own slot's K/V
```

### 14-14

`src/core/verify.cpp` · bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {

Anchor:
```
                                              attn_ + t * NH * HD, 1, cs);
                    }
                } else {
```
Block (hunk 13 of `src/core/verify.cpp` in the patch, 19 lines):
```diff
                                               attn_ + t * NH * HD, 1, cs);
                     }
                 } else {
-                qsa_block_scores(st.idx_pooled, st.idx_dead, qidx_ + tb * IQ * ID, step_ + tb * kStepCount, n, max_blocks_,
-                                 s, scores_ + (size_t) tb * max_blocks_, cs);
-                qsa_block_topk(scores_ + (size_t) tb * max_blocks_, step_ + tb * kStepCount, n, max_blocks_, cap_, s,
-                               sel_ + (size_t) tb * cap_, cs);
+                if (!sidem) idx_select(cs);
+                else if (cudaStreamWaitEvent(cs, mjoin_, 0) != cudaSuccess) {
+                    err = "verify: the indexer's branch could not join";
+                    return false;
+                }
+                if (side_proj && cudaStreamWaitEvent(cs, pjoin_, 0) != cudaSuccess) {
+                    err = "verify: the queries' branch could not join";
+                    return false;
+                }
                 stamp(l, 11, grp);
                 // KV streaming: the n selections' blocks resident (device-side, inside the graph)
                 qsa_kv_resolve(st, *g_, sel_ + (size_t) tb * cap_, step_ + tb * kStepCount, n, cap_, cs);
```

## 15 ple-direct

`15-ple-direct.patch` — FABLE-STRATA-Porta su 0.1.43 la PLE in lettura diretta anche con la sonda lenta (STRATA_PLE_AUTO_MMAP=1 per il passaggio a mmap) e la diagnostica PLE di prefill e decode

### 15-1

`include/strata/core/verify.hpp` · public:

Anchor:
```
    /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.

    double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
```
Block (hunk 1 of `include/strata/core/verify.hpp` in the patch, 8 lines):
```diff
     /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.

     double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
+    /// Fork (Daily 93af2041), STRATA_DECODE_TIMING: the PLE gather subset of ms_host (submit, wait, dequantization).
+    double ms_ple_gather = 0;
     int64_t windows = 0;
     /// STRATA_VERIFY_PROFILE=1 - GPU stage times of the windows since the last call (ms per
     /// window), as one line; empty when off.
```

### 15-2

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

### 15-3

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

### 15-4

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

### 15-5

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
         ms_wait += std::chrono::duration<double, std::milli>(b - a).count();
```

### 15-6

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

### 15-7

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

### 15-8

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

### 15-9

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

### 15-10

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

### 15-11

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        }
        // Recommend, never force: when the default direct reads measure slow, switch to the mapped reads (40-100x
        // faster on the reporters' drives); an explicit --ple-io direct is kept and only warned about.
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 22 lines):
```diff
         }
         // Recommend, never force: when the default direct reads measure slow, switch to the mapped reads (40-100x
         // faster on the reporters' drives); an explicit --ple-io direct is kept and only warned about.
+        // Fork (Daily 0.1.40.1 behaviour): the switch is opt-in, STRATA_PLE_AUTO_MMAP=1.  The probe reads 64 rows a
+        // batch from a cold table, a much lower queue depth than a prompt chunk's batched read: on the owner's SATA
+        // SSD it measured 1,259 rows/s, switched to mmap, and a 131K prompt took 260 s instead of 75 s (mmap faults
+        // the rows in one page at a time; direct keeps --ple-inflight reads in flight and the read-ahead hides them).
+        static const bool ple_auto_mmap = [] {
+            const char* v = std::getenv("STRATA_PLE_AUTO_MMAP");
+            return v != nullptr && v[0] == '1';
+        }();
         if (pio.mode == strata::kernels::PleIo::Direct && std::getenv("STRATA_PLE_PROBE") == nullptr) {
             constexpr double kSlowRowsPerS = 15000.0;   // ~60 MB/s of 4 KiB pages; a healthy NVMe is >100k
             const double rps = ple_direct_rows_per_s(ple_table);
             if (rps >= kSlowRowsPerS)
                 std::fprintf(stderr, "strata generate: PLE direct read probe: %.0f rows/s (fine)\n", rps);
             if (rps > 0 && rps < kSlowRowsPerS) {
-                if (!o.ple_io_explicit) {
+                if (!o.ple_io_explicit && ple_auto_mmap) {
                     std::fprintf(stderr, "strata generate: --ple-io direct measures slow here (%.0f rows/s, about %.0f MB/s of "
                                          "random 4 KiB reads): using --ple-io mmap instead (the OS file cache holds the pages; "
                                          "--ple-io direct forces the old mode, --ple-io ram keeps the table in RAM)\n",
```

### 15-12

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    std::fprintf(stderr, "strata generate: WARNING: --ple-io direct measures slow here (%.0f rows/s, about %.0f "
                                         "MB/s of random 4 KiB reads): a prompt can spend minutes reading layer 1 and trip the "
                                         "stall watchdog (#1425, #1629). --ple-io mmap was 40-100x faster on such drives; "
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 10 lines):
```diff
                     std::fprintf(stderr, "strata generate: WARNING: --ple-io direct measures slow here (%.0f rows/s, about %.0f "
                                          "MB/s of random 4 KiB reads): a prompt can spend minutes reading layer 1 and trip the "
                                          "stall watchdog (#1425, #1629). --ple-io mmap was 40-100x faster on such drives; "
-                                         "--ple-io ram needs the RAM for the table\n", rps, rps * 4096.0 / 1e6);
+                                         "--ple-io ram needs the RAM for the table; kept direct (fork: the probe's batches are "
+                                         "smaller than a prompt's, STRATA_PLE_AUTO_MMAP=1 switches to mmap)\n",
+                                 rps, rps * 4096.0 / 1e6);
                 }
             }
         }
```

### 15-13

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
            const Clock::time_point d0 = Clock::now();
            // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
            static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 17 lines):
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

### 15-14

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                    if (st == 0) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());   // text as before
                    else std::fprintf(stderr, "strata decode GPU stages, stage %d (ms/window):%s\n", st, pr.c_str());
                }
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 17 lines):
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

## 16 prefill-sm75

`16-prefill-sm75.patch` — FABLE-STRATA-Porta su 0.1.43 il prefill del Daily su sm_75: STRATA_SELECT_SIMT di default sotto sm_80, dequant con store da 16 B e hc16 sui tensor core FP16 (STRATA_TC_SM75=0 li spegne)

### 16-1

`include/strata/core/tc_sm75.hpp` · (new file)

New file, 28 lines: see the patch.

### 16-2

`include/strata/kernels/qsa_select.hpp` · void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx

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

### 16-3

`include/strata/prefill/kernels.hpp` · namespace strata::prefill {

Anchor:
```
// ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
/// xn[t, c*2560 + d] = R[t,c,d] * rsqrt(mean_d R[t,c,:]^2 + eps) * w_norm[c*2560 + d]; also its BF16 image.
/// `xn16_lo` (null: none) takes bf16(x - xn16): W.xn16 + W.xn16_lo is the product with ~16 mantissa bits of x.
```
Block (hunk 1 of `include/strata/prefill/kernels.hpp` in the patch, 15 lines):
```diff
 // ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
 /// xn[t, c*2560 + d] = R[t,c,d] * rsqrt(mean_d R[t,c,:]^2 + eps) * w_norm[c*2560 + d]; also its BF16 image.
 /// `xn16_lo` (null: none) takes bf16(x - xn16): W.xn16 + W.xn16_lo is the product with ~16 mantissa bits of x.
+/// `xn16h` (fork sm_75, null: none): also the FP16 image, unpadded (token stride 10240), for the staged FP16 GEMMs.
 void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
-             uint16_t* xn16_lo = nullptr);
+             uint16_t* xn16_lo = nullptr, uint16_t* xn16h = nullptr);
 /// F-1: gr_norm without its FP32 output: the row scales rs[t*4 + c] and the BF16 image; gr_mix_r then reads R.
 void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
-                uint16_t* xn16_lo = nullptr, int64_t ldx = 0);   // ldx: xn16's token stride (0 = 10240)
+                uint16_t* xn16_lo = nullptr, int64_t ldx = 0,   // ldx: xn16's token stride (0 = 10240)
+                uint16_t* xn16h = nullptr);   // fork sm_75: also the FP16 image (unpadded, token stride 10240)
 /// gr_mix with xn recomputed from R, rs and w_norm exactly as gr_norm computes it (the same bits).
 void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
               int64_t T, void* stream, uint16_t* mixed_h = nullptr, uint16_t* mixed16_lo = nullptr);
```

### 16-4

`include/strata/prefill/kernels.hpp` · bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const

Anchor:
```
              float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream);
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr,
```
Block (hunk 2 of `include/strata/prefill/kernels.hpp` in the patch, 20 lines):
```diff
               float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream);
 void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                       float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr,
-                      int64_t ldx = 0);
+                      int64_t ldx = 0, uint16_t* xn16h = nullptr);
 /// S23 (STRATA_CVEC_FUSE=1): gr_write, then the control vector (v_l = layer l's direction row, *s_l its scale,
 /// *on the request flag, mode 0 project / 1 add; cvec_kernel's arithmetic), then gr_norm_rs with the next half's
 /// norm - one pass over R, the same bits as the three kernels.
 void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                            const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
-                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr, int64_t ldx = 0);
+                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr, int64_t ldx = 0,
+                           uint16_t* xn16h = nullptr);
 /// lo16[t, k] = bf16(silu(lo[t, k] / hc))
-void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo = nullptr);
+void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo = nullptr,
+             uint16_t* lo16h = nullptr);   // fork sm_75: also the FP16 image
 /// The BF16-weight GEMMs' activation image (gr_* kernels, to_bf16) in FP16 instead of BF16, on the current device.
 void set_act_f16(bool on);
 /// mixed[t, d] = mean_c xn[t, c, d] * sigmoid(gated[t, c, d]); FP32, BF16 and FP16 (either image may be null).
```

### 16-5

`include/strata/prefill/kernels.hpp` · void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const in

Anchor:
```
/// fp32 -> fp16 bits and fp32 -> bf16, n elements (the two activation images of the prompt GEMMs).
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo = nullptr);
```
Block (hunk 3 of `include/strata/prefill/kernels.hpp` in the patch, 9 lines):
```diff
 /// fp32 -> fp16 bits and fp32 -> bf16, n elements (the two activation images of the prompt GEMMs).
 void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
 void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo = nullptr);
+/// Fork sm_75: bf16 bits -> fp16 bits, n elements (a resident BF16 weight re-encoded for the FP16 tensor cores; the
+/// value is unchanged wherever FP16 holds it).
+void bf16_to_f16(const uint16_t* x, uint16_t* y, int64_t n, void* stream);
 /// y = fp32(fp16(x)): what an FP16 store of x would read back (the FP16 indexer-key experiment)
 void round_f16(const float* x, float* y, int64_t n, void* stream);
 /// Expert blob -> FP16 (Q2_0 values are exact in FP16).
```

### 16-6

`src/kernels/cuda/dequant_bf16.cu` · (file scope)

Anchor:
```
// Arithmetic transcribed from ggml/src/ggml-quants.c at the pinned llama.cpp (MIT License, Copyright (c) 2023-2026
// The ggml authors): dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs.
#include "strata/kernels/dequant_bf16.hpp"
```
Block (hunk 1 of `src/kernels/cuda/dequant_bf16.cu` in the patch, 7 lines):
```diff
 // Arithmetic transcribed from ggml/src/ggml-quants.c at the pinned llama.cpp (MIT License, Copyright (c) 2023-2026
 // The ggml authors): dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs.
 #include "strata/kernels/dequant_bf16.hpp"
+#include "strata/core/tc_sm75.hpp"
 #include "strata/kernels/iq_kernels.hpp"

 #include <cuda_fp16.h>
```

### 16-7

`src/kernels/cuda/dequant_bf16.cu` · __device__ __forceinline__ void group32(const uint8_t* row_blocks, int gi_in_row

Anchor:
```
    }
}
```
Block (hunk 2 of `src/kernels/cuda/dequant_bf16.cu` in the patch, 30 lines):
```diff
     }
 }

-template <int TYPE, typename T>
+/// `VEC` (fork, Daily be4cf1c, sm_75 only): the group is dequantised into a register buffer and flushed with 16-byte
+/// stores.  The plain form writes each element where it is computed, which for a warp means 32 store instructions
+/// whose lanes are 64 bytes apart - 32 two-byte transactions per instruction.  **The values are identical** (the
+/// same arithmetic, the same order), only the stores are wider; the Turing build measured this kernel at 18% of the
+/// row's bandwidth.  (0.1.42's Q8_0 kernel below already writes coalesced and is left as it is.)
+template <int TYPE, typename T, bool VEC>
 __global__ void dequant_kernel(const uint8_t* __restrict__ blocks, int64_t row_bytes, int64_t row0, int64_t rows,
                                int64_t groups_per_row, int64_t ld, T* __restrict__ out) {
     const int64_t g = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
     if (g >= rows * groups_per_row) return;
     const int64_t r = g / groups_per_row, gi = g % groups_per_row;
-    group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, out + r * ld + gi * 32);
+    T* dst = out + r * ld + gi * 32;
+    if constexpr (!VEC) {
+        group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, dst);
+    } else {
+        __align__(16) T buf[32];        // the uint4 flush below reads it as 16-byte vectors
+        group32<TYPE>(blocks + (row0 + r) * row_bytes, (int) gi, buf);
+        // 32 elements: 64 B for the 16-bit forms (4 x uint4), 128 B for the float one (8 x uint4)
+        constexpr int CHUNKS = (sizeof(T) == 2 ? 4 : 8);
+#pragma unroll
+        for (int c = 0; c < CHUNKS; ++c) reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<const uint4*>(buf)[c];
+    }
 }

 // Q8_0 with four threads per block, 8 values each: the warp reads 8 consecutive blocks and writes 512 consecutive
```

### 16-8

`src/kernels/cuda/dequant_bf16.cu` · void launch(int type, const void* blocks, int64_t row0, int64_t rows, int64_t co

Anchor:
```
        if (e != cudaSuccess) { std::fprintf(stderr, "dequant launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
        return;
    }
```
Block (hunk 3 of `src/kernels/cuda/dequant_bf16.cu` in the patch, 20 lines):
```diff
         if (e != cudaSuccess) { std::fprintf(stderr, "dequant launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
         return;
     }
-#define STRATA_DQ(TY) dequant_kernel<TY, T><<<grid, 256, 0, st>>>(p, row_bytes, row0, rows, gpr, ld, out); break
+    // sm_75 (and only cc 7.5) takes the vectorised-store form when every group's 32 outputs start on 16 bytes (an
+    // aligned `out` and a row stride of whole 16-byte units); every other SM model keeps the plain kernel
+    int dev = 0, cmaj = 0, cmin = 0;
+    const bool vec = (reinterpret_cast<uintptr_t>(out) & 15) == 0 && ((uint64_t) ld * sizeof(T)) % 16 == 0 &&
+                     cudaGetDevice(&dev) == cudaSuccess &&
+                     cudaDeviceGetAttribute(&cmaj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
+                     cudaDeviceGetAttribute(&cmin, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess &&
+                     strata::core::tc_sm75_enabled(cmaj, cmin);
+    if (!vec) cudaGetLastError();
+#define STRATA_DQ(TY)                                                                                     \
+    if (vec) dequant_kernel<TY, T, true><<<grid, 256, 0, st>>>(p, row_bytes, row0, rows, gpr, ld, out);    \
+    else dequant_kernel<TY, T, false><<<grid, 256, 0, st>>>(p, row_bytes, row0, rows, gpr, ld, out);       \
+    break
     switch (type) {
     case 2: STRATA_DQ(2);
     case 6: STRATA_DQ(6);
```

### 16-9

`src/kernels/cuda/qsa_select.cu` · bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_

Anchor:
```
            cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
        }
        if (cc_major[dev] < 8) {
```
Block (hunk 1 of `src/kernels/cuda/qsa_select.cu` in the patch, 15 lines):
```diff
             cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
         }
         if (cc_major[dev] < 8) {
-            // no TF32 tensor cores: the FP32 tiled kernel, opt-in (STRATA_SELECT_SIMT=1): its sum order differs from the
-            // warp kernel's, so the default stays the caller's warp kernel and the output bytes stay the same
+            // no TF32 tensor cores: the FP32 tiled kernel (its sum order differs from the warp kernel's: not bitwise).
+            // Fork (Daily dab24920, PR #1091): on by default, as in the Daily (-16% of a 2080 Ti's prompt time);
+            // STRATA_SELECT_SIMT=0 keeps upstream 0.1.42's default, the caller's warp kernel
             static const bool simt = [] {
                 const char* v = std::getenv("STRATA_SELECT_SIMT");
-                return v != nullptr && v[0] == '1';
+                return v == nullptr || v[0] != '0';
             }();
             if (!simt) return false;
             const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
```

### 16-10

`src/prefill/kernels.cu` · unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t

Anchor:
```

// ---------------------------------------------------------------- hyper-connection
__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
```
Block (hunk 1 of `src/prefill/kernels.cu` in the patch, 9 lines):
```diff

 // ---------------------------------------------------------------- hyper-connection
 __global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
-                               float* __restrict__ xn, uint16_t* __restrict__ xn16, uint16_t* __restrict__ xn16_lo) {
+                               float* __restrict__ xn, uint16_t* __restrict__ xn16, uint16_t* __restrict__ xn16_lo,
+                               uint16_t* __restrict__ xn16h) {
     __shared__ float sh[32];
     const int64_t row = blockIdx.x;                 // t * 4 + c
     const int c = (int) (row % HC);
```

### 16-11

`src/prefill/kernels.cu` · __global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restr

Anchor:
```
        const uint16_t h = act16(v);
        xn16[row * N + d] = h;
        if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
```
Block (hunk 2 of `src/prefill/kernels.cu` in the patch, 15 lines):
```diff
         const uint16_t h = act16(v);
         xn16[row * N + d] = h;
         if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
+        if (xn16h) xn16h[row * N + d] = hf(v);   // fork sm_75: the FP16 image
     }
 }
 // F-1: the row scale only (and the BF16 image); gr_mix_r_kernel recomputes r * rs * w itself, in the same order,
 // so the FP32 copy of the normalized rows (T x 10240 floats) is neither written nor read
 __global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
                                   float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
-                                  uint16_t* __restrict__ xn16_lo, int64_t ldx) {
+                                  uint16_t* __restrict__ xn16_lo, int64_t ldx, uint16_t* __restrict__ xn16h) {
     __shared__ float sh[32];
     const int64_t row = blockIdx.x;                 // t * 4 + c
     const int c = (int) (row % HC);
```

### 16-12

`src/prefill/kernels.cu` · __global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __re

Anchor:
```
        const uint16_t h = act16(v);
        xn16[xo + d] = h;
        if (xn16_lo) xn16_lo[xo + d] = bf_lo(v, h);
```
Block (hunk 3 of `src/prefill/kernels.cu` in the patch, 7 lines):
```diff
         const uint16_t h = act16(v);
         xn16[xo + d] = h;
         if (xn16_lo) xn16_lo[xo + d] = bf_lo(v, h);
+        if (xn16h) xn16h[row * N + d] = hf(v);   // fork sm_75: the FP16 image, unpadded (token stride D)
     }
 }
 __global__ void gr_mix_r_kernel(const float* __restrict__ R, const float* __restrict__ rs, const float* __restrict__ w,
```

### 16-13

`src/prefill/kernels.cu` · __global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict

Anchor:
```
                                                               const float* __restrict__ inj, int64_t inj_ld,
                                                               const float* __restrict__ w, float eps,
                                                               float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
```
Block (hunk 4 of `src/prefill/kernels.cu` in the patch, 9 lines):
```diff
                                                                const float* __restrict__ inj, int64_t inj_ld,
                                                                const float* __restrict__ w, float eps,
                                                                float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
-                                                               uint16_t* __restrict__ xn16_lo, int64_t ldx) {
+                                                               uint16_t* __restrict__ xn16_lo, int64_t ldx,
+                                                               uint16_t* __restrict__ xn16h) {
     __shared__ float sh[32];
     const int64_t row = blockIdx.x;                 // t * 4 + c
     const int64_t t = row / HC;
```

### 16-14

`src/prefill/kernels.cu` · __global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict

Anchor:
```
        const uint16_t h = act16(x);
        xn16[xo + d] = h;
        if (xn16_lo) xn16_lo[xo + d] = bf_lo(x, h);
```
Block (hunk 5 of `src/prefill/kernels.cu` in the patch, 7 lines):
```diff
         const uint16_t h = act16(x);
         xn16[xo + d] = h;
         if (xn16_lo) xn16_lo[xo + d] = bf_lo(x, h);
+        if (xn16h) xn16h[row * N + d] = hf(x);   // fork sm_75: the FP16 image, unpadded
     }
 }
 // S23 (opt-in STRATA_CVEC_FUSE=1): a steered layer's FFN write, its control vector and the next half's norm in one
```

### 16-15

`src/prefill/kernels.cu` · __global__ void __launch_bounds__(256) gr_write_cvec_norm_rs_kernel(

Anchor:
```
    float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj, int64_t inj_ld,
    const float* __restrict__ v_l, const float* __restrict__ s_l, const int* __restrict__ on, int mode,
    const float* __restrict__ w, float eps, float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
```
Block (hunk 6 of `src/prefill/kernels.cu` in the patch, 8 lines):
```diff
     float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj, int64_t inj_ld,
     const float* __restrict__ v_l, const float* __restrict__ s_l, const int* __restrict__ on, int mode,
     const float* __restrict__ w, float eps, float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
-    uint16_t* __restrict__ xn16_lo, int64_t ldx) {
+    uint16_t* __restrict__ xn16_lo, int64_t ldx, uint16_t* __restrict__ xn16h) {
     __shared__ float sh[32];
     __shared__ float part[8];
     const int64_t row = blockIdx.x;                 // t * 4 + c
```

### 16-16

`src/prefill/kernels.cu` · __global__ void __launch_bounds__(256) gr_write_cvec_norm_rs_kernel(

Anchor:
```
        const uint16_t h = act16(y);
        xn16[t * ldx + (int64_t) c * N + d] = h;
        if (xn16_lo) xn16_lo[t * ldx + (int64_t) c * N + d] = bf_lo(y, h);
```
Block (hunk 7 of `src/prefill/kernels.cu` in the patch, 12 lines):
```diff
         const uint16_t h = act16(y);
         xn16[t * ldx + (int64_t) c * N + d] = h;
         if (xn16_lo) xn16_lo[t * ldx + (int64_t) c * N + d] = bf_lo(y, h);
+        if (xn16h) xn16h[row * N + d] = hf(y);   // fork sm_75: the FP16 image, unpadded
     }
 }
 __global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restrict__ lo16, uint16_t* __restrict__ lo16_lo,
-                               int64_t n) {
+                               int64_t n, uint16_t* __restrict__ lo16h) {
     const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
     const float x = lo[i] / (float) HC;
```

### 16-17

`src/prefill/kernels.cu` · __global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restric

Anchor:
```
    const uint16_t h = act16(v);
    lo16[i] = h;
    if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
```
Block (hunk 8 of `src/prefill/kernels.cu` in the patch, 7 lines):
```diff
     const uint16_t h = act16(v);
     lo16[i] = h;
     if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
+    if (lo16h) lo16h[i] = hf(v);   // fork sm_75: the FP16 image
 }
 __global__ void gr_mix_kernel(const float* __restrict__ xn, const float* __restrict__ g, float* __restrict__ mixed,
                               uint16_t* __restrict__ mixed16, int64_t T, uint16_t* __restrict__ mixed_h,
```

### 16-18

`src/prefill/kernels.cu` · __global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict_

Anchor:
```
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
        y[i] = hf(x[i]);
}
```
Block (hunk 9 of `src/prefill/kernels.cu` in the patch, 13 lines):
```diff
     for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
         y[i] = hf(x[i]);
 }
+// Fork sm_75: bf16 bits -> fp16 bits.  The value is unchanged whenever fp16 can hold it, and this is what lets the
+// hyper-connection projections reach a tensor core on Turing: cuBLAS has no bf16 GEMM kernel below sm_80 and runs an
+// FFMA one, which measured 19.5% of the prompt's GPU time on a 2x2080 Ti (the Daily's snapshot).
+__global__ void bf16_to_f16_kernel(const uint16_t* __restrict__ x, uint16_t* __restrict__ y, int64_t n) {
+    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
+        y[i] = hf(__uint_as_float((uint32_t) x[i] << 16));
+}
 __global__ void round_f16_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
     for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
         y[i] = __half2float(__float2half_rn(x[i]));
```

### 16-19

`src/prefill/kernels.cu` · void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo

Anchor:
```
    check("to_bf16");
}
```
Block (hunk 10 of `src/prefill/kernels.cu` in the patch, 25 lines):
```diff
     check("to_bf16");
 }

+void bf16_to_f16(const uint16_t* x, uint16_t* y, int64_t n, void* stream) {
+    if (n <= 0) return;
+    bf16_to_f16_kernel<<<(unsigned) ((n + 255) / 256 < 4096 ? (n + 255) / 256 : 4096), 256, 0,
+                         (cudaStream_t) stream>>>(x, y, n);
+    check("bf16_to_f16");
+}
 void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
-             uint16_t* xn16_lo) {
-    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16, xn16_lo);
+             uint16_t* xn16_lo, uint16_t* xn16h) {
+    gr_norm_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, xn, xn16, xn16_lo, xn16h);
     check("gr_norm");
 }
 void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
-                uint16_t* xn16_lo, int64_t ldx) {
+                uint16_t* xn16_lo, int64_t ldx, uint16_t* xn16h) {
     gr_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, w_norm, eps, rs, xn16, xn16_lo,
-                                                                                ldx > 0 ? ldx : D);
+                                                                                ldx > 0 ? ldx : D, xn16h);
     check("gr_norm_rs");
 }
 void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
```

### 16-20

`src/prefill/kernels.cu` · bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const

Anchor:
```
#endif
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
```
Block (hunk 11 of `src/prefill/kernels.cu` in the patch, 28 lines):
```diff
 #endif
 }
 void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
-                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx) {
+                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx,
+                      uint16_t* xn16h) {
     gr_write_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(R, bo, inj, inj_ld, w_norm_next, eps,
                                                                                       rs, xn16, xn16_lo,
-                                                                                      ldx > 0 ? ldx : D);
+                                                                                      ldx > 0 ? ldx : D, xn16h);
     check("gr_write_norm_rs");
 }
 void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                            const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
-                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx) {
+                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, uint16_t* xn16h) {
     gr_write_cvec_norm_rs_kernel<<<(unsigned) (T * HC), 256, 0, (cudaStream_t) stream>>>(
-        R, bo, inj, inj_ld, v_l, s_l, on, mode, w_norm_next, eps, rs, xn16, xn16_lo, ldx > 0 ? ldx : D);
+        R, bo, inj, inj_ld, v_l, s_l, on, mode, w_norm_next, eps, rs, xn16, xn16_lo, ldx > 0 ? ldx : D, xn16h);
     check("gr_write_cvec_norm_rs");
 }
-void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
-    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, lo16_lo, T * LR);
+void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo, uint16_t* lo16h) {
+    gr_silu_kernel<<<blocks_for(T * LR), 256, 0, (cudaStream_t) stream>>>(lo, lo16, lo16_lo, T * LR, lo16h);
     check("gr_silu");
 }
 void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
```

### 16-21

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

### 16-22

`src/prefill/prefill.cpp` · inline bool hcd_exact() {

Anchor:
```
    return v;
}
inline bool hc_pad() { return pf_hcdown() || hcd_exact(); }
```
Block (hunk 2 of `src/prefill/prefill.cpp` in the patch, 27 lines):
```diff
     return v;
 }
 inline bool hc_pad() { return pf_hcdown() || hcd_exact(); }
+
+/// Fork sm_75 (Daily be4cf1c): is the Turing tensor-core path of the hyper-connection projections on for the device
+/// this prompt runs on?  Only cc 7.5 (tc_sm75.hpp; STRATA_TC_SM75=0 turns it off).  Cached per thread and device: a
+/// mixed-device layer split must not inherit the first card's capability.
+bool tc_sm75() {
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
+        on = core::tc_sm75_enabled(major, minor);
+        cached_device = dev;
+    }
+    return on;
+}
 inline bool cvec_fuse() {
     static const bool v = [] { const char* e = std::getenv("STRATA_CVEC_FUSE"); return e && e[0] == '1'; }();
     return v;
```

### 16-23

`src/prefill/prefill.cpp` · struct Prefill::Impl {

Anchor:
```
    float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
    float* grs = nullptr;                    // F-1: the hyper-connection read's row scales (T x 4)
    uint16_t *xn16 = nullptr, *lo16 = nullptr;
```
Block (hunk 3 of `src/prefill/prefill.cpp` in the patch, 9 lines):
```diff
     float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
     float* grs = nullptr;                    // F-1: the hyper-connection read's row scales (T x 4)
     uint16_t *xn16 = nullptr, *lo16 = nullptr;
+    // fork sm_75: the FP16 images the hyper-connection projections' tensor-core path reads, and the staging the BF16
+    // weights are re-encoded into, one matrix at a time (6.55 MB, not 1.9 GB)
+    uint16_t *xn16h = nullptr, *lo16h = nullptr, *hc_w16 = nullptr;
     float* mixed = nullptr;
     uint16_t *mixed_bf = nullptr, *mixed_h = nullptr;
     bool f16_io = false;   // HIP, STRATA_HIP_PROMPT_F16=1: this stage's 16-bit GEMMs run FP16 in and out (set in init)
```

### 16-24

`src/prefill/prefill.cpp` · bool Prefill::carve(size_t T, void* alloc) {

Anchor:
```
    m.xn = gr_unfused() ? o.take<float>(T * D, ok) : nullptr;   // F-1: not needed (gr_mix_r reads R)
    m.grs = o.take<float>(T * HC, ok);
    m.xn16 = o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
```
Block (hunk 4 of `src/prefill/prefill.cpp` in the patch, 10 lines):
```diff
     m.xn = gr_unfused() ? o.take<float>(T * D, ok) : nullptr;   // F-1: not needed (gr_mix_r reads R)
     m.grs = o.take<float>(T * HC, ok);
     m.xn16 = o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
+    if (tc_sm75()) {   // fork sm_75: the FP16 images (unpadded) and the one-matrix weight staging
+        m.xn16h = o.take<uint16_t>(T * D, ok); m.lo16h = o.take<uint16_t>(T * LR, ok);
+        m.hc_w16 = o.take<uint16_t>((size_t) g.hc * (size_t) g.n_embd * (size_t) LR, ok);
+    }
     m.gated = o.take<float>(T * D, ok); m.inj = o.take<float>(T * HC, ok);
     m.mixed = o.take<float>(T * N, ok); m.mixed_bf = o.take<uint16_t>(T * N, ok);
     m.mixed_h = o.take<uint16_t>(T * N, ok);
```

### 16-25

`src/prefill/prefill.cpp` · bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t

Anchor:
```
        proj(hn, hn16, w_fh, h2, nb * g.hc, Nn, Nn, 1);   // every stream through fc_hidden
        strata::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
        // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
```
Block (hunk 5 of `src/prefill/prefill.cpp` in the patch, 8 lines):
```diff
         proj(hn, hn16, w_fh, h2, nb * g.hc, Nn, Nn, 1);   // every stream through fc_hidden
         strata::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
         // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
+        // THE DRAFT LAYER STAYS BF16 (fork sm_75, measured in the Daily): moved to the FP16 tensor cores with the main
+        // layers it cost 27 points of draft acceptance (93% -> 66% on a 1K prompt); its GEMMs are a few tokens wide.
         gr_norm_rs(Rm, w_hn, EPS, grs, xn16, nb, m.cs);
         m.gemm.bf16(xn16, w_dn, lo, nb, LR, HCN);
         gr_silu(lo, lo16, nb, m.cs);
```

### 16-26

`src/prefill/prefill.cpp` · uint64_t Prefill::bytes_needed_impl(const core::ModelGeometry& g, const core::Se

Anchor:
```
        f(T * N); f(T * D); f(T * D);
    }
    o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
```
Block (hunk 6 of `src/prefill/prefill.cpp` in the patch, 13 lines):
```diff
         f(T * N); f(T * D); f(T * D);
     }
     o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
+    // fork sm_75: the FP16 images and the weight staging `carve` takes right here.  THIS LIST MUST STAY IN STEP WITH
+    // `carve`: the region is sized from it, and a missing entry is not a smaller buffer but an allocation past the end
+    // of the arena (measured in the Daily: "prefill copy_i32: an illegal memory access").
+    if (tc_sm75()) {
+        o.take<uint16_t>(T * D, ok); o.take<uint16_t>(T * LR, ok);
+        o.take<uint16_t>((size_t) g.hc * (size_t) g.n_embd * (size_t) LR, ok);
+    }
     f(T * D); f(T * HC); f(T * N); o.take<uint16_t>(T * N, ok); o.take<uint16_t>(T * N, ok);
     if (!emb_reuse_account()) f(T * N);   // bo: aliases emb in carve; still counted unless STRATA_EMB_REUSE_ACCOUNT=1
     const bool f16_io = prompt_f16();   // the current device's mode (the stage's), as Prefill::init will decide it
```

### 16-27

`src/prefill/prefill.cpp` · bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y,

Anchor:
```
    if (X_lo) gm.bf16(X_lo, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, 1.0f);
    return true;
}
```
Block (hunk 7 of `src/prefill/prefill.cpp` in the patch, 21 lines):
```diff
     if (X_lo) gm.bf16(X_lo, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, 1.0f);
     return true;
 }
+/// Fork sm_75: a projection whose weight is BF16 in the arena, re-encoded to FP16 into `w16` (one matrix at a time:
+/// 6.55 MB for the largest) and multiplied on the FP16 tensor cores.  The re-encoding is value-preserving - a speed
+/// path, not a precision one - and the conversion shares the compute stream with the GEMM, so the staging buffer
+/// cannot be overwritten while cuBLAS still reads it.
+bool f16_proj_staged(Gemm& gm, const core::WeightRef* w, const uint16_t* X16, uint16_t* w16, float* Y, int64_t T,
+                     const std::string& name, std::string& err, int64_t ldy = 0, void* stream = nullptr) {
+    if (w->kind != core::WeightKind::Bf16InF32 || !w->data || w16 == nullptr) {
+        err = "prefill: " + name + " is not a resident BF16 tensor (or has no fp16 staging)";
+        return false;
+    }
+    const int64_t N = w->ne1 > 0 ? w->ne1 : 1, K = w->ne0;
+    bf16_to_f16((const uint16_t*) w->data, w16, K * N, stream);
+    gm.f16(X16, w16, Y, T, N, K, ldy);
+    return true;
+}

 }  // namespace

```

### 16-28

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                                      *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                if (!wn || !wd || !wu || !wi) return false;
                pt.mark(kPfHc, cs);
```
Block (hunk 8 of `src/prefill/prefill.cpp` in the patch, 21 lines):
```diff
                                       *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                 if (!wn || !wd || !wu || !wi) return false;
                 pt.mark(kPfHc, cs);
+                // fork sm_75 (hc16, Daily be4cf1c): the hyper-connection projections on the FP16 tensor cores through
+                // the staged weight; BF16X2 is an explicit accuracy option: kept when requested.  hc16 keeps xn16
+                // unpadded (ldx = D), so the HIP-only padded paths (STRATA_PF_HCDOWN / STRATA_HCD_EXACT) and
+                // STRATA_HC_UPMIX stay off while it is on.
+                const bool hc16 = !bf16x2_hc(m.f16_io) && m.xn16h && m.lo16h && m.hc_w16;
                 // the xn16 token stride of this chunk (the writers of both halves and of the fused write-backs use it)
-                const int64_t ldx = hc_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) && !gr_unfused() &&
+                const int64_t ldx = !hc16 && hc_pad() && T >= std::max<int64_t>(pf_switch_min_t(), 64) && !gr_unfused() &&
                                             !m.xn16_lo ? D + XN_PAD : D;
-                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo);
-                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs, m.xn16_lo, ldx);
+                if (gr_unfused()) gr_norm(m.R, (const float*) wn->data, EPS, m.xn, m.xn16, T, m.cs, m.xn16_lo,
+                                          hc16 ? m.xn16h : nullptr);
+                else if (!normed) gr_norm_rs(m.R, (const float*) wn->data, EPS, m.grs, m.xn16, T, m.cs, m.xn16_lo, ldx,
+                                             hc16 ? m.xn16h : nullptr);
                 normed = false;
                 bool hcd = false;
                 bool hdown = false;
```

### 16-29

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                    }
                    hcd = true;
                }
```
Block (hunk 9 of `src/prefill/prefill.cpp` in the patch, 15 lines):
```diff
                     }
                     hcd = true;
                 }
-                if (!hcd && !hdown && !bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
-                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo);
+                if (hc16) {
+                    if (!f16_proj_staged(m.gemm, wd, m.xn16h, m.hc_w16, m.lo, T, sd, err, 0, m.cs)) return false;
+                } else if (!hcd && !hdown && !bf16_proj(m.gemm, wd, m.xn16, m.lo, T, sd, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
+                gr_silu(m.lo, m.lo16, T, m.cs, m.lo16_lo, hc16 ? m.lo16h : nullptr);
                 bool upmixed = false;
-                if (hc_upmix() && T >= pf_switch_min_t() && !gr_unfused() && !m.lo16_lo && !m.mixed_bf_lo &&
+                if (!hc16 && hc_upmix() && T >= pf_switch_min_t() && !gr_unfused() && !m.lo16_lo && !m.mixed_bf_lo &&
                     wu->kind == core::WeightKind::Bf16InF32 && wu->data && wu->ne0 == LR && wu->ne1 == D) {
                     static int checks = [] { const char* e = std::getenv("STRATA_HC_UPMIX_CHECK"); return e ? std::atoi(e) : 0; }();
                     if (checks > 0) {   // the default pair first, kept for the comparison
```

### 16-30

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                                           m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                    }
                }
```
Block (hunk 10 of `src/prefill/prefill.cpp` in the patch, 14 lines):
```diff
                                            m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                     }
                 }
-                if (!upmixed && !bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
-                if (!hcd && !bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
+                if (hc16) {
+                    if (!f16_proj_staged(m.gemm, wu, m.lo16h, m.hc_w16, m.gated, T, su, err, 0, m.cs)) return false;
+                } else if (!upmixed && !bf16_proj(m.gemm, wu, m.lo16, m.gated, T, su, err, 0, m.lo16_lo)) return false;
+                if (hc16) {
+                    if (!f16_proj_staged(m.gemm, wi, m.xn16h, m.hc_w16, m.inj, T, si, err, 0, m.cs)) return false;
+                } else if (!hcd && !bf16_proj(m.gemm, wi, m.xn16, m.inj, T, si, err, 0, m.xn16_lo, ldx != D ? ldx : 0)) return false;
                 if (upmixed) {
                 } else if (gr_unfused()) {
                     gr_mix(m.xn, m.gated, m.mixed, m.mixed_bf, T, m.cs, m.mixed_h, m.mixed_bf_lo);
```

### 16-31

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                    float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                    if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
```
Block (hunk 11 of `src/prefill/prefill.cpp` in the patch, 18 lines):
```diff
                     float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                     if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                     if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
-                    if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
-                    if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
+                    // fork sm_75: these two are BF16 GEMMs too (FFMA on Turing), and their input `mixed_h` is already
+                    // the FP16 image, so they take the same staged FP16 tensor-core path as the hyper-connection
+                    if (hc16 && !bf16x2(m.f16_io) && wa->kind == core::WeightKind::Bf16InF32) {
+                        if (!f16_proj_staged(m.gemm, wa, m.mixed_h, m.hc_w16, m.ab, T,
+                                             v.name("ssm_alpha.weight"), err, 2 * HV, m.cs)) return false;
+                    } else if (!bf16_proj(m.gemm, wa, m.mixed_bf, m.ab, T, v.name("ssm_alpha.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
+                    if (hc16 && !bf16x2(m.f16_io) && wb->kind == core::WeightKind::Bf16InF32) {
+                        if (!f16_proj_staged(m.gemm, wb, m.mixed_h, m.hc_w16, m.ab + HV, T,
+                                             v.name("ssm_beta.weight"), err, 2 * HV, m.cs)) return false;
+                    } else if (!bf16_proj(m.gemm, wb, m.mixed_bf, m.ab + HV, T, v.name("ssm_beta.weight"), err, 2 * HV, m.mixed_bf_lo)) return false;
                     pt.mark(kPfGdnConv, cs);   // "gdn" is the projections in; the rest on their own lines
                     gdn_gates(m.ab, (const float*) wdt->data, (const float*) wsa->data, m.gate, m.beta, T, m.cs);
                     gdn_conv(conv, m.qkv, (const float*) wc->data, m.hbuf, T, EPS, m.cs);
```

### 16-32

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                        // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                        const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                strata::kernels::kStepNBid)] + 1;
```
Block (hunk 12 of `src/prefill/prefill.cpp` in the patch, 8 lines):
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

### 16-33

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                        gr_norm_rs(Rc, (const float*) wnn->data, EPS, rc, xc, T, m.cs, nullptr, D);
                        gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                              strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs,
```
Block (hunk 13 of `src/prefill/prefill.cpp` in the patch, 8 lines):
```diff
                         gr_norm_rs(Rc, (const float*) wnn->data, EPS, rc, xc, T, m.cs, nullptr, D);
                         gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                               strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs,
-                                              m.xn16, T, m.cs, m.xn16_lo, ldx);
+                                              m.xn16, T, m.cs, m.xn16_lo, ldx, hc16 ? m.xn16h : nullptr);
                         std::vector<float> a((size_t) T * D), b((size_t) T * D), ra((size_t) T * HC), rb((size_t) T * HC);
                         std::vector<uint16_t> xa((size_t) T * D), xb((size_t) T * D);
                         cudaMemcpyAsync(a.data(), m.R, a.size() * 4, cudaMemcpyDeviceToHost, m.cs);
```

### 16-34

`src/prefill/prefill.cpp` · bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri

Anchor:
```
                    } else
                    gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                          strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs, m.xn16,
```
Block (hunk 14 of `src/prefill/prefill.cpp` in the patch, 13 lines):
```diff
                     } else
                     gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                           strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs, m.xn16,
-                                          T, m.cs, m.xn16_lo, ldx);
+                                          T, m.cs, m.xn16_lo, ldx, hc16 ? m.xn16h : nullptr);
                     normed = true;
                 } else if (wnn) {
                     gr_write_norm_rs(m.R, m.bo, m.inj, HC, (const float*) wnn->data, EPS, m.grs, m.xn16, T, m.cs,
-                                     m.xn16_lo, ldx);
+                                     m.xn16_lo, ldx, hc16 ? m.xn16h : nullptr);   // fork sm_75: + the FP16 image
                     normed = true;
                 } else {
                     gr_write(m.R, m.bo, m.inj, HC, T, m.cs);
```

### 16-35

`src/program/generate.cpp` · (file scope)

Anchor:
```
#include "strata/core/progress.hpp"
#include "strata/core/device.hpp"
#include "strata/core/emulate.hpp"
```
Block (hunk 1 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
 #include "strata/core/progress.hpp"
 #include "strata/core/device.hpp"
 #include "strata/core/emulate.hpp"
+#include "strata/core/tc_sm75.hpp"
 #ifndef NOMINMAX
 #define NOMINMAX   // gguf_reader.hpp includes windows.h
 #endif
```

### 16-36

`src/program/generate.cpp` · (file scope)

Anchor:
```
#include <set>
#include <vector>
```
Block (hunk 2 of `src/program/generate.cpp` in the patch, 23 lines):
```diff
 #include <set>
 #include <vector>

+/// Fork sm_75 (Daily be4cf1c): cc 7.5, and only 7.5, adds FP16 images and a weight staging to the prompt path
+/// (prefill.cpp, hc16).  Where the prompt path owns its buffers - no borrow - each one is its own cudaMalloc and the
+/// expert cache, sized from the empirical MiB reserve, is what leaves room for them: add their bytes to it.  Sizes
+/// from `Prefill::carve`: xn16h = chunk * hc * n_embd * 2, lo16h = chunk * hc_lr * 2, the staged weight =
+/// hc * n_embd * hc_lr * 2.  The callers add this only where the empirical reserve stands in for the prompt path
+/// (`prefill_mib` / `split_pf_mib`); a borrowing stage's loan is sized by `Prefill::bytes_needed`, which prices them.
+static int64_t sm75_prompt_extra_bytes(const strata::core::ModelGeometry& g, int64_t chunk, int dev) {
+    int maj = 0, min = 0;
+    if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
+        cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
+        cudaGetLastError();
+        return 0;
+    }
+    if (!strata::core::tc_sm75_enabled(maj, min) || chunk <= 0) return 0;
+    return chunk * g.hc * g.n_embd * 2 + chunk * g.hc_lr * 2 + g.hc * g.n_embd * g.hc_lr * 2;
+}
+
 namespace {
 // Windows' WDDM driver model: native Windows, or WSL2 (its GPU goes through /dev/dxg to the Windows driver).  There,
 // pinning a large arena into two CUDA contexts leaves WDDM refusing every later allocation (the 5080 + 3090 rig);
```

### 16-37

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        const int64_t pf = search && split_own_auto && !place_with_reserve ? 0 : split_pf_mib;
        const int64_t base_reserve =
            later && o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib;
```
Block (hunk 3 of `src/program/generate.cpp` in the patch, 11 lines):
```diff
         const int64_t pf = search && split_own_auto && !place_with_reserve ? 0 : split_pf_mib;
         const int64_t base_reserve =
             later && o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib;
-        const int64_t reserve = (base_reserve + pf + (later ? kWindowMib : kPipeSnapMib) +
-                                 kPipeWindowMib + (drafter ? kDrafterMib : 0)) << 20;
+        const int64_t reserve = ((base_reserve + pf + (later ? kWindowMib : kPipeSnapMib) +
+                                 kPipeWindowMib + (drafter ? kDrafterMib : 0)) << 20) +
+                                (pf > 0 ? sm75_prompt_extra_bytes(g, o.prefill_chunk, dev) : 0);
         return std::max<int64_t>((int64_t) fb - reserve, 0);
     };
     if (multi_gpu && split_auto) {
```

### 16-38

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        if (stages.empty())
            for (const auto& d : slot_mtp)
                mtp_bind += (int64_t) d->bind_bytes(native_head.row_bytes(), n_vocab);
```
Block (hunk 4 of `src/program/generate.cpp` in the patch, 9 lines):
```diff
         if (stages.empty())
             for (const auto& d : slot_mtp)
                 mtp_bind += (int64_t) d->bind_bytes(native_head.row_bytes(), n_vocab);
-        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind + pipe_first;
+        const int64_t sm75_extra = prefill_mib > 0 ? sm75_prompt_extra_bytes(g, o.prefill_chunk, 0) : 0;
+        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind + pipe_first + sm75_extra;
         const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
         int64_t slots = ((int64_t) free_b - reserve) / blob;
         if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
```

### 16-39

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
        size_t free_b = 0, total_b = 0;
        free_b = strata::core::device_free_bytes(); (void) total_b;
        const int64_t prefill_mib = owned_prefill_mib();
```
Block (hunk 5 of `src/program/generate.cpp` in the patch, 9 lines):
```diff
         size_t free_b = 0, total_b = 0;
         free_b = strata::core::device_free_bytes(); (void) total_b;
         const int64_t prefill_mib = owned_prefill_mib();
-        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + pipe_first;
+        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + pipe_first +
+                                (prefill_mib > 0 ? sm75_prompt_extra_bytes(g, o.prefill_chunk, 0) : 0);
         const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) strata::kernels::cpu::expert_layout().max_blob, 0);
         if (o.expert_cache > fit) {
             // a WARNING that names the knob: the user asked for this size, and gets fewer slots
```

### 16-40

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
-             uint16_t* xn16_lo) {
+             uint16_t* xn16_lo, uint16_t* /*xn16h: CUDA sm_75 only*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

### 16-41

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
-                uint16_t* xn16_lo, int64_t ldx) {
+                uint16_t* xn16_lo, int64_t ldx, uint16_t* /*xn16h*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

### 16-42

`sycl/src/prefill/kernels.dp.cpp` · bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const

Anchor:
```
#endif
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
```
Block (hunk 3 of `sycl/src/prefill/kernels.dp.cpp` in the patch, 8 lines):
```diff
 #endif
 }
 void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
-                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx) {
+                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, uint16_t* /*xn16h*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

### 16-43

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
-                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx) {
+                           uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo, int64_t ldx, uint16_t* /*xn16h*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

### 16-44

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
+void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo, uint16_t* /*lo16h*/) {
     {
         auto exp_props = sycl::ext::oneapi::experimental::properties{
             sycl::ext::oneapi::experimental::use_root_sync};
```

## 17 ram-core

`17-ram-core.patch` — FABLE-STRATA-Porta su 0.1.43 il nucleo fisso per scheda senza copia RAM (STRATA_RAM_CORE=1, STRATA_RAM_CORE_SLOTS_DEV<n>)

### 17-1

`docs/RAM_CORE.md` · (new file)

New file, 114 lines: see the patch.

### 17-2

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

### 17-3

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

### 17-4

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
... 140 more lines: see 17-ram-core.patch
```

### 17-5

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

### 17-6

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

### 17-7

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

### 17-8

`src/core/expert_source.cpp` · const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {

Anchor:
```
    const int64_t idx = layer * n_expert_ + expert;
    if (idx < 0 || idx >= blobs_) return nullptr;
    ++reads_;
```
Block (hunk 5 of `src/core/expert_source.cpp` in the patch, 9 lines):
```diff
     const int64_t idx = layer * n_expert_ + expert;
     if (idx < 0 || idx >= blobs_) return nullptr;
     ++reads_;
+    // STRATA_RAM_CORE: a core expert has no RAM copy (it never leaves VRAM); asked for anyway, it is read from the
+    // model file first (counted).  One atomic load otherwise.
+    if (rc_ && rc_->dropped[(size_t) idx].load(std::memory_order_acquire) != 0 && !rc_recover(layer, expert)) return nullptr;
     // Pointer arithmetic into resident memory.  No fault, no copy, no mapping - which is the entire point of
     // this class over `FileExpertSource`.
     return base_ + strata::kernels::cpu::expert_layout().blob_offset(layer, expert);
```

### 17-9

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

### 17-10

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

### 17-11

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

### 17-12

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

### 17-13

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

### 17-14

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

### 17-15

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

### 17-16

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

### 17-17

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

### 17-18

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

### 17-19

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

### 17-20

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

### 17-21

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

### 17-22

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

### 17-23

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

### 17-24

`src/program/generate.cpp` · int main(int argc, char** argv) {

Anchor:
```
                        (double) (src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                        (long long) req_offload, chain_txt);
            std::fflush(stdout);
```
Block (hunk 16 of `src/program/generate.cpp` in the patch, 7 lines):
```diff
                         (double) (src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                         (long long) req_offload, chain_txt);
             std::fflush(stdout);
+            rc_log("fine richiesta");   // STRATA_RAM_CORE
             if (admit_slot >= 0) {   // --batch: BADM <slot> <1 = continues in the batch windows | 0 = done>
                 bool cont = !cancelled && produced_n == 1 && admit_max_new > 1 && std::strcmp(finish, "length") == 0 &&
                             (int64_t) live.size() == p;
```

## 18 qsa-rows

`18-qsa-rows.patch` — FABLE-STRATA-Porta su 0.1.43 STRATA_QSA_ROWS=1: punteggi dell'indicizzatore QSA con le query in registri e le 4 teste ridotte in 6 shuffle (stessi bit) e qsa_rows_bench

### 18-1

`CMakeLists.txt` · if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)

Anchor:
```
    target_include_directories(chain_step_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(chain_step_bench PRIVATE strata_kernels CUDA::cudart)
  endif()
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 14 lines):
```diff
     target_include_directories(chain_step_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
     target_link_libraries(chain_step_bench PRIVATE strata_kernels CUDA::cudart)
   endif()
+  # QSAROWS (docs/QSAROWS.md): what the T rows of a verification window cost in a QSA layer's decode kernels at a long
+  # context (block scores, top-k, attention over each row's selection, indexer append), row by row; GPU, synthetic, no
+  # model.  Not in the default build: `cmake --build <dir> --target qsa_rows_bench`.
+  if(STRATA_ENABLE_CUDA)
+    add_executable(qsa_rows_bench EXCLUDE_FROM_ALL src/kernels/qsa_rows_bench.cpp)
+    target_include_directories(qsa_rows_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
+    target_link_libraries(qsa_rows_bench PRIVATE strata_kernels CUDA::cudart)
+  endif()
   # the QSA selection (block scores on the warp kernel and the tensor-core one, top-k): a GPU bench with a
   # scorer-vs-scorer comparison and an FP64 accuracy gate (synthetic, no model); the ctest runs a 32K context under a
   # 262144-cell capacity (the top-k dispatch follows the capacity) and exits non-zero on a failed gate
```

### 18-2

`docs/QSAROWS.md` · (new file)

New file, 300 lines: see the patch.

### 18-3

`docs/qsa_rows_bench_2026-10-12.txt` · (new file)

New file, 59 lines: see the patch.

### 18-4

`docs/qsa_rows_bench_rows_2026-10-12.txt` · (new file)

New file, 67 lines: see the patch.

### 18-5

`include/strata/kernels/qsa_select.hpp` · void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx

Anchor:
```
                      int64_t active_blocks = -1);   ///< perf-review C-1: > 0 launches only this many blocks (the
                                                     ///< batch's largest n_bid + 1; not for a captured graph)
```
Block (hunk 1 of `include/strata/kernels/qsa_select.hpp` in the patch, 13 lines):
```diff
                       int64_t active_blocks = -1);   ///< perf-review C-1: > 0 launches only this many blocks (the
                                                      ///< batch's largest n_bid + 1; not for a captured graph)

+/// QSAROWS (docs/QSAROWS.md): block_scores_multi_kernel's scores for a call of up to 4 queries, bit for bit, with the
+/// queries in registers and the four heads reduce-scattered (six shuffles instead of twenty a block and query).
+/// qsa_block_scores takes it with STRATA_QSA_ROWS=1; false (nothing launched) above 4 queries or on another geometry.
+/// For qsa_rows_bench's parity and timing it is callable on its own.
+bool qsa_block_scores_rows(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
+                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream);
+
 /// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
 /// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
 /// On AMD it is the gfx12 (RDNA4) WMMA kernel (a three-way bf16 split, six products); false on any other AMD target.
```

### 18-6

`src/kernels/cuda/qsa_select.cu` · __global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_multi_kernel(co

Anchor:
```
#if !defined(__HIPCC__)
// ---- the decode top-k on a thread-block cluster (sm_90+; S19).  One CTA per query (block_topk_reg_kernel, or
// block_topk_kernel above 33,792 blocks: a --max-context over ~135K) makes four radix passes and two scans over up to
```
Block (hunk 1 of `src/kernels/cuda/qsa_select.cu` in the patch, 82 lines):
```diff
     }
 }

+// QSAROWS (docs/QSAROWS.md, STRATA_QSA_ROWS=1): block_scores_multi_kernel's work for NQ <= 4 queries with less issue
+// pressure per (block, query).  nsys of a 131k decode window (DET-qsarows-nsys2): the multi kernel costs ~20 us more a
+// call for every row of the window on both cards - per (block, query) it reads the four query slices from shared memory
+// (4 x LDS.128) and reduces each head's lane products with its own five shuffles (20 a block and query), and those two
+// pipes bound it, not the key reads (the keys are read once for all the queries, as there).  Here (1) the lane's slices
+// of the NQ queries (4 heads x 4 floats each) live in registers for the whole kernel, (2) the four heads' lane products
+// are reduce-scattered with warp_sum's pairing (xor 16, 8, then 4, 2, 1: six shuffles for the four heads) and gathered
+// to lane 0 in head order (three more).  Every sum adds the same two operands as block_scores_kernel's warp_sum at the
+// same level (float add commutes), the dot expression is the same text in the same file, the relu-and-add runs over the
+// heads in the same order: the same score bits.  The tail block and the +1e9 are as in the multi kernel.
+template <int NQ>
+__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_rows_kernel(const float* __restrict__ pooled,
+                                                                             const float* __restrict__ dead,
+                                                                             const float* __restrict__ q_idx,
+                                                                             const int32_t* __restrict__ steps, int nq,
+                                                                             int64_t max_blocks, float* __restrict__ out) {
+    __shared__ int64_t s_nkv[NQ], s_nbid[NQ];
+    if (threadIdx.x < nq) {
+        s_nkv[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNKv];
+        s_nbid[threadIdx.x] = steps[threadIdx.x * kStepCount + kStepNBid];
+    }
... 58 more lines: see 18-qsa-rows.patch
```

### 18-7

`src/kernels/cuda/qsa_select.cu` · void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx

Anchor:
```
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
    if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
```
Block (hunk 2 of `src/kernels/cuda/qsa_select.cu` in the patch, 9 lines):
```diff
     // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
     static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
     if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
+        // QSAROWS: STRATA_QSA_ROWS=1 (exactly) takes block_scores_rows_kernel for calls of up to 4 queries - the same bits
+        static const bool rows_env = [] { const char* v = std::getenv("STRATA_QSA_ROWS"); return v != nullptr && v[0] == '1' && v[1] == '\0'; }();
+        if (rows_env && qsa_block_scores_rows(pooled, dead, q_idx, steps, nq, max_blocks, s, scores, stream)) return;
         // STRATA_QSA_EARLY_EXIT=0 keeps the old staging (every CTA loads the queries first); the scores are the same bits
         static const bool early = [] { const char* v = std::getenv("STRATA_QSA_EARLY_EXIT"); return v == nullptr || std::atoi(v) != 0; }();
         block_scores_multi_kernel<<<256, SCORE_WARPS * 32, 0, (cudaStream_t) stream>>>(pooled, dead, q_idx, steps, (int) nq,
```

### 18-8

`src/kernels/cuda/qsa_select.cu` · void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx

Anchor:
```
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
```
Block (hunk 3 of `src/kernels/cuda/qsa_select.cu` in the patch, 21 lines):
```diff
     if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
 }

+bool qsa_block_scores_rows(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
+                           int64_t max_blocks, const QsaShapes& s, float* scores, void* stream) {
+    if (nq <= 0) return true;
+    if (nq > 4 || s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R) return false;
+    cudaStream_t st = (cudaStream_t) stream;
+    const int n = (int) nq;
+    if (n == 1) block_scores_rows_kernel<1><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, n, max_blocks, scores);
+    else if (n == 2) block_scores_rows_kernel<2><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, n, max_blocks, scores);
+    else if (n == 3) block_scores_rows_kernel<3><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, n, max_blocks, scores);
+    else block_scores_rows_kernel<4><<<256, SCORE_WARPS * 32, 0, st>>>(pooled, dead, q_idx, steps, n, max_blocks, scores);
+    const cudaError_t e = cudaGetLastError();
+    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores rows: %s\n", cudaGetErrorString(e)); std::exit(1); }
+    return true;
+}
+
 bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                          int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
     if (nq <= 0) return true;
```

### 18-9

`src/kernels/qsa_rows_bench.cpp` · (new file)

New file, 354 lines: see the patch.

## 19 expdec

`19-expdec.patch` — FABLE-STRATA-Porta su 0.1.43 EXPDEC: attivazioni del gruppo in shared e R righe per lane per gli esperti VRAM (STRATA_EXPDEC=1, down dietro STRATA_EXPDEC_DOWN=1) e expdec_bench

### 19-1

`CMakeLists.txt` · if(STRATA_NATIVE_EXPERTS)

Anchor:
```
    add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
    target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
```
Block (hunk 1 of `CMakeLists.txt` in the patch, 13 lines):
```diff
     add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
     target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
     target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
+    if(STRATA_ENABLE_CUDA)
+      # EXPDEC (docs/EXPERTDEC.md): the grouped experts' default kernels against STRATA_EXPDEC=1 at T = 1..4, bit for bit
+      # and in microseconds, on synthetic blobs read cold from DRAM
+      add_executable(expdec_bench src/kernels/expdec_bench.cpp)
+      target_include_directories(expdec_bench PRIVATE ${_strata_gpu_include_directories})
+      target_link_libraries(expdec_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
+    endif()
     if(STRATA_BUILD_TESTS AND STRATA_ENABLE_CUDA)
       # Unsloth UD-Q4_K_XL's three expert pairs (and Q8_0 alone, its embedding) on synthetic ggml-quantized weights
       # at 2560/640: GPU and CPU experts against the float reference, GPU dequantizers against to_float.
```

### 19-2

`docs/EXPERTDEC.md` · (new file)

New file, 322 lines: see the patch.

### 19-3

`include/strata/kernels/iq_kernels.hpp` · void native_grouped_set_v1(bool v1);

Anchor:
```
/// The bench only: the AMD kernel layout (STRATA_EXP_MODE values; -1 = the environment's) and the phase
/// (0 all, 1 gate/up + SwiGLU + quantize, 2 down).
void native_expert_set_mode(int mode, int phase);
```
Block (hunk 1 of `include/strata/kernels/iq_kernels.hpp` in the patch, 28 lines):
```diff
 /// The bench only: the AMD kernel layout (STRATA_EXP_MODE values; -1 = the environment's) and the phase
 /// (0 all, 1 gate/up + SwiGLU + quantize, 2 down).
 void native_expert_set_mode(int mode, int phase);
+
+/// EXPDEC (STRATA_EXPDEC=1, docs/EXPERTDEC.md): `native_expert_grouped` stages the group's q8_1 activations in shared
+/// memory and computes several rows per lane, the same sums per (row, entry) as the default kernels (bitwise), for
+/// gate/up IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / IQ2_S at n_embd 2560 and down IQ4_NL / Q2_0 at n_ff 640; other layouts
+/// keep the default kernels.  Set before graph capture; captured graphs keep the kernels they captured.
+void native_expert_set_expdec(int on);
+int native_expert_expdec();
+/// Its knobs, read from the environment at startup (STRATA_EXPDEC_R 1 | 2, _DR 1 | 2 | 4, _MINB 2 | 4, _D20 0 | 1,
+/// _RB, _DRB, _MINBLK): gate/up and down rows per lane, the gate/up kernel's __launch_bounds__ minimum blocks per SM
+/// (2 = up to 128 registers, 4 = up to 64), EXPDEC for the IQ4_NL down too, gate/up and down rows per block (0 = auto
+/// from the launch's groups), the auto rule's minimum blocks.  In the setter a field <= 0 (< 0 for the two rows per
+/// block and down_iq4_nl) keeps the current value.
+/// `down` (STRATA_EXPDEC_DOWN, default 0): EXPDEC for the down too; by default only the gate/up (in the engine the
+/// Q2_0 down was ~10% slower with EXPDEC, docs/EXPERTDEC.md par. 7).
+struct NativeExpertExpdecParams {
+    int gu_rows_per_lane = 0, down_rows_per_lane = 0, gu_min_blocks_per_sm = 0, down_iq4_nl = -1, down = -1;
+    int gu_rows_per_block = -1, down_rows_per_block = -1, min_blocks = 0;
+};
+void native_expert_set_expdec_params(const NativeExpertExpdecParams& p);
+NativeExpertExpdecParams native_expert_expdec_params();
+/// Whether EXPDEC has kernels for this layout's gate/up and down (the bench's check).
+bool native_expert_expdec_takes(const NativeExpertLayout& L, bool& gu, bool& down);
 /// `iq_mmvq` and `native_expert_grouped` decode each weight part once and apply it to every column / entry;
 /// true selects the older kernels that decode it again per column (STRATA_OLD_IQ_MMVQ=1 at startup).  Both give
 /// bitwise the same results.  Set before graph capture; captured graphs keep the kernels they captured.
```

### 19-4

`src/kernels/cuda/iq_kernels.cu` · template<> struct Split<16> {   // IQ2_XXS

Anchor:
```
};
// IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
struct SplitLs2 {
```
Block (hunk 1 of `src/kernels/cuda/iq_kernels.cu` in the patch, 16 lines):
```diff
         const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
         return d * sumi;
     }
+    // `apply` with the q8_1 block already in registers: u = its 8 words, d8 = __low2float(ds).  Same operations in
+    // the same order (the EXPDEC kernels, see expdec below).
+    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
+        int sumi = 0;
+#pragma unroll
+        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], u[j], sumi);
+        sumi = sumi * r.ls / 8;
+        const float d = r.dw * d8;
+        return d * sumi;
+    }
 };
 // IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
 struct SplitLs2 {
```

### 19-5

`src/kernels/cuda/iq_kernels.cu` · struct SplitLs2 {

Anchor:
```
};
template<> inline constexpr bool kSplit<17> = true;
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
```
Block (hunk 2 of `src/kernels/cuda/iq_kernels.cu` in the patch, 16 lines):
```diff
         const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
         return d * sumi;
     }
+    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
+        int sumi0 = 0, sumi1 = 0;
+#pragma unroll
+        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], u[j], sumi0);
+#pragma unroll
+        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], u[j], sumi1);
+        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
+        const float d = r.dw * d8;
+        return d * sumi;
+    }
 };
 template<> inline constexpr bool kSplit<17> = true;
 template<> struct Split<17> : SplitLs2 {   // IQ2_XS
```

### 19-6

`src/kernels/cuda/iq_kernels.cu` · template<> inline constexpr bool kSplit<18> = true;

Anchor:
```
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    template<bool STAGE_GRID = false>
```
Block (hunk 3 of `src/kernels/cuda/iq_kernels.cu` in the patch, 20 lines):
```diff
 template<> struct Split<18> {   // IQ3_XXS
     struct W { int g[8]; int ls; float dw; };
     template<bool STAGE_GRID = false>
-    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
+    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
         const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
         const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs + 1));
         const uint8_t* q3 = (const uint8_t*) &q3_packed;
         const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
+        const uint32_t* grid_lut;   // the same 256 words either way (expdec stages them)
+        if constexpr (STAGE_GRID) grid_lut = s_grid;
+        else grid_lut = iq3xxs_grid;
         W r;
 #pragma unroll
         for (int l0 = 0; l0 < 8; l0 += 2) {
-            const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
+            const int2 grid_pos = make_int2(grid_lut[q3[l0 + 0]], grid_lut[q3[l0 + 1]]);
             const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
             const int signs0 = __vcmpne4(signs & 0x08040201, 0);
             r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
```

### 19-7

`src/kernels/cuda/iq_kernels.cu` · template<> struct Split<18> {   // IQ3_XXS

Anchor:
```
        const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
        return d * sumi;
    }
```
Block (hunk 4 of `src/kernels/cuda/iq_kernels.cu` in the patch, 37 lines):
```diff
         const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
         return d * sumi;
     }
+    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
+        int sumi = 0;
+#pragma unroll
+        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], u[j], sumi);
+        sumi = (r.ls * sumi + sumi / 2) / 2;
+        const float d = r.dw * d8;
+        return d * sumi;
+    }
 };
 template<> inline constexpr bool kSplit<21> = true;
 template<> struct Split<21> {   // IQ3_S
     struct W { int g[8]; int ls; float dw; };
     template<bool STAGE_GRID = false>
-    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ = nullptr) {
+    __device__ static W load(const void* __restrict__ vbq, int kbx, int iqs, const uint32_t* __restrict__ s_grid = nullptr) {
         const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
         const int2 qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
         const uint8_t* qs = (const uint8_t*) &qs_packed;
         const int qh = bq3->qh[iqs / 2];
         const int signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
         const uint8_t* signs_packed_8 = (const uint8_t*) &signs_packed_32;
+        const uint32_t* grid_lut;   // the same 512 words either way (expdec stages them)
+        if constexpr (STAGE_GRID) grid_lut = s_grid;
+        else grid_lut = iq3s_grid;
         W r;
 #pragma unroll
         for (int l0 = 0; l0 < 8; l0 += 2) {
-            const int2 grid_pos = make_int2(iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
-                                            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
+            const int2 grid_pos = make_int2(grid_lut[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
+                                            grid_lut[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
             const int signs0 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x03) << 7) | ((signs_packed_8[l0 / 2] & 0x0C) << 21), 0x00000000);
             const int signs1 = __vcmpne4(((signs_packed_8[l0 / 2] & 0x30) << 3) | ((signs_packed_8[l0 / 2] & 0xC0) << 17), 0x00000000);
             r.g[l0 + 0] = __vsub4(grid_pos.x ^ signs0, signs0);
```

### 19-8

`src/kernels/cuda/iq_kernels.cu` · template<> struct Split<21> {   // IQ3_S

Anchor:
```
};
template<> inline constexpr bool kSplit<29> = true;
template<> struct Split<29> {   // IQ1_M
```
Block (hunk 5 of `src/kernels/cuda/iq_kernels.cu` in the patch, 14 lines):
```diff
         const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
         return d * sumi;
     }
+    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
+        int sumi = 0;
+#pragma unroll
+        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], u[j], sumi);
+        sumi *= r.ls;
+        const float d = r.dw * d8;
+        return d * sumi;
+    }
 };
 template<> inline constexpr bool kSplit<29> = true;
 template<> struct Split<29> {   // IQ1_M
```

### 19-9

`src/kernels/cuda/iq_kernels.cu` · template<> struct Split<20> {   // IQ4_NL

Anchor:
```
        const float d = r.dw * __low2float(bq8_1->ds);
        return d * sumi;
    }
```
Block (hunk 6 of `src/kernels/cuda/iq_kernels.cu` in the patch, 16 lines):
```diff
         const float d = r.dw * __low2float(bq8_1->ds);
         return d * sumi;
     }
+    // `apply` with the block's words in registers: p0 = words iqs, iqs + 1; p1 = words iqs + 4, iqs + 5; d8 = __low2float(ds)
+    __device__ static float apply_u(const W& r, int2 p0, int2 p1, float d8) {
+        int sumi = 0;
+        sumi = ggml_cuda_dp4a(r.v[0].x, p0.x, sumi);
+        sumi = ggml_cuda_dp4a(r.v[0].y, p1.x, sumi);
+        sumi = ggml_cuda_dp4a(r.v[1].x, p0.y, sumi);
+        sumi = ggml_cuda_dp4a(r.v[1].y, p1.y, sumi);
+        const float d = r.dw * d8;
+        return d * sumi;
+    }
 };
 template<> inline constexpr bool kSplit<23> = true;
 template<> struct Split<23> {   // IQ4_XS
```

### 19-10

`src/kernels/cuda/iq_kernels.cu` · template<> struct Split<42> {   // Q2_0

Anchor:
```
        const float d8 = __low2float(bq8_1_chunk->ds);
        return r.d2 * d8 * sumi;
    }
```
Block (hunk 7 of `src/kernels/cuda/iq_kernels.cu` in the patch, 16 lines):
```diff
         const float d8 = __low2float(bq8_1_chunk->ds);
         return r.d2 * d8 * sumi;
     }
+    // `apply` with the q8_1 block (2 kbx + iqs) in registers: u = its 8 words, d8 = __low2float(ds)
+    __device__ static float apply_u(const W& r, const int (&u)[8], float d8) {
+        int sumi = 0;
+#pragma unroll
+        for (int j = 0; j < 4; ++j) {
+            sumi = ggml_cuda_dp4a(u[j * 2 + 0], r.qx[j], sumi);
+            sumi = ggml_cuda_dp4a(u[j * 2 + 1], r.qy[j], sumi);
+        }
+        return r.d2 * d8 * sumi;
+    }
 };

 // One row against the n <= NC activations x + off[0..n) (n >= 1, warp-uniform; offsets in q8_1 blocks, 32-bit to
```

### 19-11

`src/kernels/cuda/iq_kernels.cu` · __global__ void __launch_bounds__(256) native_down_multi_kernel(const unsigned l

Anchor:
```
// ---------------------------------------------------------------- q8_1 (quantize.cu)
// value i of a q8_1 row set (the warp holds block i / 32, lane = i % 32)
__device__ __forceinline__ void q8_1_store(const float xi, block_q8_1* __restrict__ y, const long long i) {
```
Block (hunk 8 of `src/kernels/cuda/iq_kernels.cu` in the patch, 369 lines):
```diff
     }
 }

+// ---------------------------------------------------------------- EXPDEC (STRATA_EXPDEC=1): the group's activations
+// in shared memory, R rows per lane
+//
+// native_gu_multi_kernel / native_down_multi_kernel decode each weight part once and apply it to every entry of the
+// group, but the gate/up apply re-reads the entry's q8_1 block from global memory with 9 scattered 4-byte loads
+// (36-byte blocks: a warp's load touches 32 sectors), so an entry costs about 60% of a whole expert (nsys, 2080 Ti,
+// L02 at 131k: ~2.6 us per expert and ~1.6 us per entry and layer) although it reads nothing from DRAM.  Here a
+// block stages its group's entries (ED_NC at a time, as the default) in shared memory laid out so that a call's
+// words are two 16-byte loads (gate/up, Q2_0 down: the block's two halves in separate planes, conflict-free) or two
+// 8-byte loads (IQ4_NL down), the fp16 block scale already a float; every lane computes R rows at once, so the
+// activation words are loaded once per R rows; the IQ3_XXS / IQ3_S codebooks are staged too (IQ2_* already were),
+// and a block takes `rb` rows (auto: the most that keep ~STRATA_EXPDEC_MINBLK blocks in flight) instead of 16.
+// Each (row, entry) sum is the default kernel's sum tree (row_dot_80_sub16 / row_dot_40_sub8 / row_dot_20_sub16: the
+// same lane does the same calls in the same order with the same operations, `apply_u` = `apply` on the block's
+// words), so the output is bitwise the same (expdec_bench checks it; docs/EXPERTDEC.md).
+#ifndef STRATA_EXPDEC_MINB
+#define STRATA_EXPDEC_MINB 4   // gate/up __launch_bounds__ min blocks per SM by default (STRATA_EXPDEC_MINB at runtime): 4 = up to 64
+                               // registers, as the default kernels (bench: equal on the 2080 Ti, 0.87-0.95 on the 3060); 2 = up to 128
+#endif
+constexpr int ED_NC = GRP_NC;   // entries staged at once
+constexpr int ED_XB = 80;       // q8_1 blocks of a gate/up row (n_embd 2560)
... 345 more lines: see 19-expdec.patch
```

### 19-12

`src/kernels/cuda/iq_kernels.cu` · void s27_launch_ts(bool ts, const NativeExpertLayout& L, int64_t cap_groups, cud

Anchor:
```
    else s27_launch<TG, TD, true, 4, 4, false>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
}
```
Block (hunk 9 of `src/kernels/cuda/iq_kernels.cu` in the patch, 144 lines):
```diff
     else s27_launch<TG, TD, true, 4, 4, false>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
 }

+// ---- EXPDEC launch: knobs, shapes, rows per block
+int env_int_or(const char* name, int dflt) {
+    const char* v = std::getenv(name);
+    return v != nullptr && v[0] != '\0' ? std::atoi(v) : dflt;
+}
+int g_expdec = env_on("STRATA_EXPDEC") ? 1 : 0;
+int g_expdec_r = env_int_or("STRATA_EXPDEC_R", 1);          // gate/up rows per lane: 1, 2 (bench: R=1 wins on the 2080 Ti)
+int g_expdec_dr = env_int_or("STRATA_EXPDEC_DR", 4);        // down rows per lane: 1, 2, 4 (Q2_0: R=4 wins on both cards)
+int g_expdec_minb = env_int_or("STRATA_EXPDEC_MINB", STRATA_EXPDEC_MINB);   // gate/up __launch_bounds__ min blocks: 2 (<= 128 regs) or 4 (<= 64)
+int g_expdec_d20 = env_int_or("STRATA_EXPDEC_D20", 0);      // 1: EXPDEC also for the IQ4_NL down (bench: no gain, default kernel)
+// 1: EXPDEC for the down too (Q2_0; IQ4_NL with _D20).  Default 0: in the engine (nsys DET-x2-expdec-nsysB/B2) the Q2_0
+// down with EXPDEC was ~10% slower than the default kernel although the bench alone had it at 0.78, so by default
+// EXPDEC takes the gate/up only (-20/-23% at T >= 2 on the 2080 Ti in the engine).
+int g_expdec_down = env_int_or("STRATA_EXPDEC_DOWN", 0);
+int g_expdec_rb = env_int_or("STRATA_EXPDEC_RB", 0);        // gate/up rows per block (0 = auto)
+int g_expdec_drb = env_int_or("STRATA_EXPDEC_DRB", 0);      // down rows per block (0 = auto)
+int g_expdec_minblk = env_int_or("STRATA_EXPDEC_MINBLK", 800);   // auto: the most rows per block that keep this many blocks
+int expdec_r() { return g_expdec_r == 2 ? 2 : 1; }
+int expdec_dr() { return g_expdec_dr == 1 || g_expdec_dr == 2 ? g_expdec_dr : 4; }
+
+bool expdec_gu_ok(const NativeExpertLayout& L) {
... 120 more lines: see 19-expdec.patch
```

### 19-13

`src/kernels/cuda/iq_kernels.cu` · bool g_grouped_v1 = env_on("STRATA_GROUPED_V1");

Anchor:
```

void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }
```
Block (hunk 10 of `src/kernels/cuda/iq_kernels.cu` in the patch, 36 lines):
```diff

 void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }

+void native_expert_set_expdec(int on) { g_expdec = on ? 1 : 0; }
+int native_expert_expdec() { return g_expdec; }
+void native_expert_set_expdec_params(const NativeExpertExpdecParams& p) {
+    if (p.gu_rows_per_lane > 0) g_expdec_r = p.gu_rows_per_lane;
+    if (p.down_rows_per_lane > 0) g_expdec_dr = p.down_rows_per_lane;
+    if (p.gu_min_blocks_per_sm > 0) g_expdec_minb = p.gu_min_blocks_per_sm;
+    if (p.down_iq4_nl >= 0) g_expdec_d20 = p.down_iq4_nl;
+    if (p.down >= 0) g_expdec_down = p.down;
+    if (p.gu_rows_per_block >= 0) g_expdec_rb = p.gu_rows_per_block;
+    if (p.down_rows_per_block >= 0) g_expdec_drb = p.down_rows_per_block;
+    if (p.min_blocks > 0) g_expdec_minblk = p.min_blocks;
+}
+NativeExpertExpdecParams native_expert_expdec_params() {
+    NativeExpertExpdecParams p;
+    p.gu_rows_per_lane = expdec_r();
+    p.down_rows_per_lane = expdec_dr();
+    p.gu_min_blocks_per_sm = g_expdec_minb >= 4 ? 4 : 2;
+    p.down_iq4_nl = g_expdec_d20 ? 1 : 0;
+    p.down = g_expdec_down ? 1 : 0;
+    p.gu_rows_per_block = g_expdec_rb;
+    p.down_rows_per_block = g_expdec_drb;
+    p.min_blocks = g_expdec_minblk;
+    return p;
+}
+bool native_expert_expdec_takes(const NativeExpertLayout& L, bool& gu, bool& down) {
+    gu = expdec_gu_ok(L);
+    down = expdec_down_ok(L);
+    return gu || down;
+}
+
 void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                            const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                            int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
```

### 19-14

`src/kernels/cuda/iq_kernels.cu` · void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long

Anchor:
```
    } else
#undef STRATA_GU_AMD
#endif
```
Block (hunk 11 of `src/kernels/cuda/iq_kernels.cu` in the patch, 7 lines):
```diff
     } else
 #undef STRATA_GU_AMD
 #endif
+    if (!expdec_gu(L, gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_tok, X, gate, up))   // STRATA_EXPDEC=1
     switch (L.gu_type) {
 #define STRATA_GU(T) case T: launch_gu<T>(ggu, s, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
         STRATA_GU_FMTS(STRATA_GU)
```

### 19-15

`src/kernels/cuda/iq_kernels.cu` · void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long

Anchor:
```
    } else
#undef STRATA_D_AMD
#endif
```
Block (hunk 12 of `src/kernels/cuda/iq_kernels.cu` in the patch, 7 lines):
```diff
     } else
 #undef STRATA_D_AMD
 #endif
+    if (!expdec_down(L, gy, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, hq, out))   // STRATA_EXPDEC=1
     switch (L.d_type) {
 #define STRATA_DOWN(T) case T: launch_down<T>(gd, s, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
         STRATA_D_FMTS(STRATA_DOWN)
```

### 19-16

`src/kernels/expdec_bench.cpp` · (new file)

New file, 480 lines: see the patch.
