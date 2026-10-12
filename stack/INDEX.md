# Strata-2080Ti patch stack

The patches this fork (imanu86/Strata-2080Ti) carries on top of upstream Strata, one lever each, in the order they apply. Written for a two-card Windows box (RTX 2080 Ti 22 GB as stage 1, RTX 3060 12 GB as stage 0, `--layer-split 16 --trim-stage-weights --pipeline-windows 2`), but every lever is a plain switch and most are bitwise with the default path. The registry exists so that the next upstream version can be checked and re-based mechanically: run `check_anchors.py --check` on the new tag, read what moved, re-apply.

Base: upstream `Niko1221/Strata` **v0.1.43** (`7695859a`). Stack head: `2705a5a3`. 17 patches, applied in this order with `git am stack/*.patch` (or one at a time with `git apply --3way`). Check a new upstream with `python stack/check_anchors.py --check <repo> --rev <tag>`.

## Reading this file

- **Switch / default**: how the lever is turned on, and what happens when nothing is set.
- **Origin**: `fork` = written for this fork; `PR #n` = an upstream pull request (ours or somebody else's) carried as a patch while it is open.
- **Upstream**: whether upstream already has it, has an open PR for it, or never saw it.
- **Measured**: the value on this fork's hardware (RTX 2080 Ti 22 GB + RTX 3060 12 GB, `--layer-split 16`, Windows 11, 131k-token forced bench, B-A-B unless said otherwise). Details in the linked docs.
- **Insertion points**: one per hunk: file, the enclosing symbol as git names it, and an anchor of 2-3 lines of the *unpatched* file, unique in that file at the time the patch was made. No line numbers. The inserted block is in `BLOCKS.md` under the same id; the applicable form is the `.patch`.

## Upstream defaults this fork forces

Upstream defaults that this stack or its reference configuration changes, so that a run on the stack behaves like the fork's Daily:

| default in upstream 0.1.43 | here | by |
|---|---|---|
| `STRATA_SELECT_SIMT` opt-in (QSA block scores below sm_80 on the warp kernel) | **on** below sm_80, `=0` turns it off | patch 14 |
| `STRATA_DF_BRANCH`, `STRATA_DF_PLE` opt-in (PR #904's graph branches / PLE post-ops) | **on** with CUDA (BRANCH: on Windows), `=0` turns them off; BRANCH yields to `STRATA_SIDE_MIXER` | patch 11, 12 |
| `--ple-io` auto: the start-up probe switches to mmap under 15 000 rows/s | **direct** stays; `STRATA_PLE_AUTO_MMAP=1` restores the switch | patch 13 |
| `STRATA_ROUTE_TAIL_SKIP=7` (off from 90% cached experts, #1884) | unchanged in code; the reference config sets `=0` (-1.6% with it at 96-98% hit) | config |
| `STRATA_MMVQ_IL` on (interleaved dense GEMV where the card's table has a cell) | unchanged in code; the reference config sets `STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1` (IL -3.0% / sm_75 table 0.0% against B6 on the 2080 Ti) | config, patch 04 |
| `STRATA_NAN_GUARD` on (#1892, new in 0.1.43) | unchanged (cost < 0.2%, bitwise otherwise); `=0` turns it off | - |
| `STRATA_ADAPT_OVERLAP`, `STRATA_PCIE_AUTO_CANON` opt-in on Windows (0.1.43) | unchanged | - |
| `STRATA_PIPELINE_ELASTIC` (not in upstream) | **on** with two cards and `--pipeline-windows`: theta 0.10, verified window served first; `=0` gives upstream's theta 0.20 and D-A-B order | patch 09 |

Reference configuration environment (the fork's "0.1.42 Daily r2"): `STRATA_VRAM_FREE=device STRATA_ROUTE_TAIL_SKIP=0 STRATA_ATTN_DAILY=1 STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1 STRATA_MTP_SHARED_BRANCH=0 STRATA_SIDE_MIXER=2 STRATA_RAM_CORE=1 STRATA_RAM_CORE_SLOTS_DEV1=5500 STRATA_EXPDEC=1 STRATA_QSA_ROWS=1`, options `--elastic --prompt-cache-tail --prompt-cache-every 16384 --prompt-cache-file <file> --layer-split 16 --trim-stage-weights --pipeline-windows 2 --vram-reserve-later-mib 384 --host-core last`.

## Patches

| id | name | what | switch (default) | origin | upstream | measured |
|---|---|---|---|---|---|---|
| 01 | [elastic-cache](#01-elastic-cache) | `--elastic`: the expert cache grows and shrinks at runtime around a fixed core, sized on the card's free VRAM. | `--elastic` (off) | fork (PR #1117, ours) | PR open since 0.1.40; not in 0.1.43 | Daily default since 3 Oct; with the two-card split see patch 05. Resubmitted PR: opt-in, exact. |
| 02 | [split-prompt-borrow](#02-split-prompt-borrow) | With `--layer-split`, each stage's prompt loan is estimated on its own device, not on CUDA0's. | always on with `--layer-split` (correctness) | fork (PR #1417, ours) | PR open; not in 0.1.43 | Correctness: the stage-1 card no longer refuses a loan that fits. No speed claim. |
| 03 | [forced-bench](#03-forced-bench) | Deterministic bench: replay token ids and verified windows (`STRATA_FORCE_IDS`, `STRATA_FORCE_WINDOWS`, trace with `STRATA_FORCE_TRACE`), adapted to the multi-stage pipeline. | `STRATA_FORCE_IDS=<file>[;<file>]`, `STRATA_FORCE_WINDOWS=<trace>`, `STRATA_FORCE_TRACE=<out>` (off) | fork (PR #1419, ours) | PR open; not in 0.1.43 | Tooling: every B-A-B number in this registry is read on this bench (131k context, L01 full prefill + L02 cached prefix, 1024 tokens, same text in every arm). |
| 04 | [b6-mmvq-rows](#04-b6-mmvq-rows) | Dense GEMV (MMVQ) that reuses the quantized activations across 2-4 output rows (Q4_K, Q5_K, Q6_K, IQ4_XS), bitwise. | `STRATA_B6_MMVQ_ROWS=1` (off) | fork (PR #1418, ours) | PR open; validated independently on a 2080 Ti/Windows (bitwise, kernels -17/-33%); upstream went its own way with the MMVQ_IL tables (0.1.42: sm_80+, 0.1.43: sm_86 and one column, Q3_K) | Reference config uses it with `STRATA_MMVQ_IL=0`: against it, upstream's IL path -3.0% and the sm_75 shape table of PR #1437 0.0% on the 2080 Ti (11-12 Oct). |
| 05 | [elastic-split-pipeline](#05-elastic-split-pipeline) | The elastic cache with `--layer-split` (CUDA0's block only) and beside `--pipeline-windows`: the queue moves only in stage-0 gaps, growth published when the fills landed, 1.5 GiB queue, 128 MiB step, the prompt borrows the core too; on by default with two cards. | `STRATA_PIPELINE_ELASTIC` (on with two cards and `--pipeline-windows`; `=0` off) | fork | never proposed (depends on 01) | Part of the Daily's two-card configuration (83.7 t/s at 40-71k on a real SWE case, 7 Oct). The 0.1.42 port with the whole stack measured +0.7% against the Daily (131k forced bench). |
| 06 | [prefix-disk](#06-prefix-disk) | `--prompt-cache-file`: the long prompt prefix saved to disk and restored across restarts, format v2 with one image per split block; the MTP drafter computes every cell of the saved root. | `--prompt-cache-file <path>` (off) | fork | never proposed (upstream has `--prompt-cache-tail/-every` in RAM only) | Restart of an agent session: the shared 40-130K prefix is restored from the file instead of being read again (a 40K first prompt took ~42 s to read here). Restore time: see the fork's smoke test of the PR branch. The file identity covers exe, args and STRATA_*/CUDA_* variables. |
| 07 | [vram-free-nvml](#07-vram-free-nvml) | `STRATA_VRAM_FREE=device`: free VRAM of the whole card through NVML (matched by PCI bus id, so `CUDA_VISIBLE_DEVICES` order does not matter); the elastic cache reads it too. | `STRATA_VRAM_FREE=device` (off: CUDA's per-process view) | upstream PR #1911 (Xiaomi2008) + fork glue for the elastic cache | PR open; not in 0.1.43 | Fixes the 7-36 t/s collapses under WDDM paging on the 3060 (CUDA saw 9.7 GB free, NVML 8.1): with it the stage holds 77 t/s. Daily default since 11 Oct. |
| 08 | [park-no-foreign-kv](#08-park-no-foreign-kv) | The conversation cache never parks one conversation with another's retained K/V. | always on (correctness) | upstream PR #1503 (sanastasiou) | PR open; not in 0.1.43 | Correctness only. |
| 09 | [pipeline-architectds](#09-pipeline-architectds) | The two-window pipeline's profile from the Daily: the verified window is served first (front stage of A, last stage of A, then D and B one layer per pass), lookup beyond one window, theta 0.10; plus two log lines (rollback cost per window class, chain latency). | `STRATA_PIPELINE_ELASTIC` (on with two cards and `--pipeline-windows`; `=0`: upstream's theta 0.20 and D-A-B order) | fork (from the architectds lab branch) | never proposed | 0.1.42 without it: -4.6/-7.3/-7.8% against the Daily (speculative windows 133/600 instead of 164/600, stage host 0.50 -> 1.69 ms). With it and patch 11: +0.7% (parity). |
| 10 | [attn-daily](#10-attn-daily) | The Daily's QSA decode attention on every CUDA card: the two-cells-per-step chunk kernel (upstream keeps it below sm_75 only) and the `attn_merge_v2` merge; bitwise with the default; parity harness `qsa_merge_daily_parity`. | `STRATA_ATTN_DAILY=1` (off); `STRATA_ATTN_MERGE_V2`, `STRATA_ATTN_PRE75=0` for the two halves | fork (kernel body from PR #540 sskver and the merge from Hardin22, both carried in the old Daily) | upstream has the chunk kernel as `attn_chunk_kernel_pre75` (sm_60 builds only); the merge v2 is not upstream | nsys on the 3060: `attn_chunk_kernel` 99.8 -> 46.4 us a call. In the engine the attention is ~8% of the per-row cost (docs/QSAROWS.md): part of the +0.7% parity, not separable. |
| 11 | [df-defaults](#11-df-defaults) | `STRATA_DF_BRANCH` (mixer side branches in the window graph; CUDA on Windows) and `STRATA_DF_PLE` (PLE post-projections in one pass; CUDA) on by default, as in the Daily. | `STRATA_DF_BRANCH=0`, `STRATA_DF_PLE=0` turn them off (upstream: opt-in `=1`) | fork (defaults of PR #904's switches) | upstream merged #904 opt-in and keeps BRANCH off after a Linux stall (#905, never seen on Windows) | Daily measurements: BRANCH +2.4% at 131k, PLE +3.3% (bitwise). On 0.1.42 their absence was the main part of ~1.1 ms of GPU per window. |
| 12 | [side-mixer](#12-side-mixer) | `STRATA_SIDE_MIXER=1/2`: the mixer's work that reads only the layer's input (GDN gates, the QSA indexer chain, with `=2` also z and the q chain) on branches of the window graph; bitwise. Turns `STRATA_DF_BRANCH` off when set. | `STRATA_SIDE_MIXER=1\|2` (off) | upstream PR #1889 (KarlGrier) | PR open; not in 0.1.43 | `=2`: +0.9% against DF_BRANCH on the 131k forced bench (11-12 Oct). Daily default since 12 Oct. |
| 13 | [ple-direct](#13-ple-direct) | The PLE table stays in direct I/O even when the start-up probe measures the disk slow (0.1.42+ switches to mmap under 15 000 rows/s); PLE diagnostics for prefill (mean/max/reads over 100 ms, waiting chunks) and decode (`STRATA_DECODE_TIMING`). | default direct; `STRATA_PLE_AUTO_MMAP=1` restores upstream's switch to mmap | fork | never proposed | PLE on a cold SATA SSD: 131k prefill 260 s (mmap) -> 75 s (direct), decode +3.9%. The probe read 1 259 rows/s; direct I/O with `--ple-inflight` reads and chunk read-ahead hides it. |
| 14 | [prefill-sm75](#14-prefill-sm75) | The Daily's prefill on Turing: `STRATA_SELECT_SIMT` on by default below sm_80 (FP32 tiled QSA block scores), dequant of the projections with 16-byte stores, and the hyper-connection projections plus ssm_alpha/beta on FP16 tensor cores (`hc16`; weights re-encoded once, activations in FP16: not bitwise with BF16, like the Daily). | `STRATA_SELECT_SIMT=0`, `STRATA_TC_SM75=0` turn them off (sm_75 only) | fork (SIMT scores from PR #1091 konijiwa110, merged upstream opt-in) | SIMT: upstream merged opt-in; 16-byte dequant and hc16: never proposed (upstream's `STRATA_BF16_TC` stays off on sm_75) | 131k prefill 100 -> 75 s on the 2080 Ti stage (with patch 13: 75.4 s against the Daily's 74.9). In the old Daily hc16 removed 19.5% of the prompt's GPU time (BF16 GEMM in FFMA below sm_80). |
| 15 | [ram-core](#15-ram-core) | A fixed core of expert slots per card, excluded from swaps, loans and shrink, whose RAM copy is given back (the resident arena no longer mirrors experts that never leave VRAM). | `STRATA_RAM_CORE=1` (off), `STRATA_RAM_CORE_SLOTS_DEV<n>=<slots>` | fork | never proposed | `STRATA_RAM_CORE_SLOTS_DEV1=5500`: arena 39.97 -> 29.62 GiB (-10.35), working set 48.7 -> 33.6 GB, t/s +0.1%, hit rate unchanged, 0 re-reads from file. Daily default since 12 Oct. |
| 16 | [qsa-rows](#16-qsa-rows) | QSA decode block scores with the queries in registers and the four heads reduced in six shuffles (T <= 4), bitwise; `qsa_rows_bench` (cost per window row of the QSA decode kernels). | `STRATA_QSA_ROWS=1` (off) | fork | PR branch prepared (`pr/qsa-rows-scores`, one commit on v0.1.43); not yet opened | Kernel 101 -> 55 us a call at T=4 on the 2080 Ti; engine B-A-B +3.2 / +0.1% (about +1-1.5%). Daily default since 12 Oct. |
| 17 | [expdec](#17-expdec) | Routed experts' gate/up in the verify window with the group's activations in shared memory and R rows per lane (the down behind `STRATA_EXPDEC_DOWN=1`), bitwise; `expdec_bench`. | `STRATA_EXPDEC=1` (off); `STRATA_EXPDEC_DOWN`, `_R`, `_DR`, `_MINB`, `_D20`, `_RB`, `_DRB` knobs | fork | PR branch prepared (`pr/verify-expdec`, one commit on v0.1.43); not yet opened | gate/up kernel -11/-23% at T 1-4 on the 2080 Ti; engine B-A-B +1.1% (B +0.0/+2.1). Daily default since 12 Oct. The Q2_0 down with EXPDEC was ~10% slower in the engine: off by default. |

## Retired

Levers that were in the stack and were dropped (code stays on the fork's lab branches):

| name | what | measured | why |
|---|---|---|---|
| critpath-levers | Critical-path levers for the two-stage pipeline (STRATA_CP_L1FIRST/S1FIRST/CHAIN_PRE/CHAIN_FUSE/RESPLAN, opt-in). | All within noise on the 131k forced bench (-0.4/-2.5/+0.1/-0.1%). | 12 Oct 2026: dropped from the stack (42 insertion points in generate.cpp, verify.cpp, mtp.cpp for no gain). Code: fork branch lab/critpath2. |
| chain-fuse | Fused kernels of the MTP drafter's step (STRATA_CHAIN_FUSE2, opt-in), chain_step_bench. | Kernels -37..-86% on the 2080 Ti, engine +0.4% (noise); the fused attention was slower. | 12 Oct 2026: dropped from the stack (29 insertion points). Code: fork branch lab/chainfuse. |
| mmvq-il-sm75 (PR #1437) | Exact-shape MMVQ_IL table for sm_75, opt-in STRATA_MMVQ_IL=1. | 0.0% against B6 on the 2080 Ti. | Not ported to 0.1.43: upstream rewrote the IL dispatch; the PR is somebody else's and still open. |

## 01 elastic-cache

`01-elastic-cache.patch` — FABLE-STRATA-Porta su 0.1.43 la cache elastica con nucleo fisso (--elastic, PR #1117)

- **What**: `--elastic`: the expert cache grows and shrinks at runtime around a fixed core, sized on the card's free VRAM.
- **Switch (default)**: `--elastic` (off)
- **Origin**: fork (PR #1117, ours)
- **Upstream**: PR open since 0.1.40; not in 0.1.43
- **Measured**: Daily default since 3 Oct; with the two-card split see patch 05. Resubmitted PR: opt-in, exact.
- **Notes**: Conflict on 0.1.43: the pinned token upload of #1712 (`tok_up`) is sized on `cap_T` (largest chunk) like the host buffer.
- **Docs**: docs/ELASTIC_CACHE.md
- **Files**: `CMakeLists.txt`, `docs/ELASTIC_CACHE.md`, `include/strata/core/expert_cache.hpp`, `include/strata/prefill/prefill.hpp`, `src/core/expert_cache.cpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`, `tests/core/expert_cache_elastic_test.cpp`

Insertion points (37):

- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [01-1](BLOCKS.md#01-1)
  ```
      src/core/arch_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
    target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})
  ```
- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [01-2](BLOCKS.md#01-2)
  ```
      target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine)
      add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
      set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  ```
- `docs/ELASTIC_CACHE.md` — (new file) → [01-3](BLOCKS.md#01-3)
- `include/strata/core/expert_cache.hpp` · `public:` → [01-4](BLOCKS.md#01-4)
  ```
      const uint64_t* slot_offsets() const { return off_.empty() ? nullptr : off_.data(); }
      void close();
  ```
- `include/strata/core/expert_cache.hpp` · `private:` → [01-5](BLOCKS.md#01-5)
  ```
      std::vector<int32_t> layer_next_;   ///< [n_layers] -> that layer's next free slot
      std::vector<uint64_t> off_;         ///< plan v0.3 P6: slot offsets (slots + 1 entries) when sized
      int64_t admitted_ = 0;
  ```
- `include/strata/prefill/prefill.hpp` · `public:` → [01-6](BLOCKS.md#01-6)
  ```
      /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
      /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
      /// lent for the prompt and refilled after it); null = allocate normally.
  ```
- `src/core/expert_cache.cpp` · `(file scope)` → [01-7](BLOCKS.md#01-7)
  ```
  #include <cstdio>
  #include <filesystem>
  #include <cstdlib>
  ```
- `src/core/expert_cache.cpp` · `bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_l` → [01-8](BLOCKS.md#01-8)
  ```
  void ExpertCache::close() {
  #if defined(STRATA_USE_HIP)
      if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
  ```
- `src/core/expert_cache.cpp` · `void ExpertCache::close() {` → [01-9](BLOCKS.md#01-9)
  ```
      blocking_staging_bytes_ = 0;
  #endif
      off_.clear();
  ```
- `src/prefill/prefill.cpp` · `uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused) {` → [01-10](BLOCKS.md#01-10)
  ```
  
  bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                     core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co` → [01-11](BLOCKS.md#01-11)
  ```
      // the CPU share's staged-chunk limit, before the ring is sized below (a caller that sizes loans first armed it)
      arm_cpu_share(cpu_pool_ != nullptr);
      for (int b = 0; next_ != nullptr && b < 2; ++b)
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co` → [01-12](BLOCKS.md#01-12)
  ```
          m.gemm.set_f16_io(m.f16_io);
          set_act_f16(m.f16_io);
      }
  ```
- `src/program/generate.cpp` · `struct Options {` → [01-13](BLOCKS.md#01-13)
  ```
      /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
      /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
      int vram_reserve_mib = 700;
  ```
- `src/program/generate.cpp` · `void usage() {` → [01-14](BLOCKS.md#01-14)
  ```
                   "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                   "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                   "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
  ```
- `src/program/generate.cpp` · `void usage() {` → [01-15](BLOCKS.md#01-15)
  ```
                   "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                   "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                   "                       ms/token, -2.7 ms/token end to end.\n"
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-16](BLOCKS.md#01-16)
  ```
          else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
          else if (a == "--vram-reserve-later-mib")
              o.vram_reserve_later_mib = std::atoi(next("--vram-reserve-later-mib"));
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-17](BLOCKS.md#01-17)
  ```
      strata::core::set_peer_portable(o.peer_device >= 1);   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
      {   // the RAM guard (parking, session save/restore): --memory-limit-mib, else STRATA_MEMORY_LIMIT_MIB, else none
          int64_t mib = o.memory_limit_mib;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-18](BLOCKS.md#01-18)
  ```
                              !o.expert_profile.empty() && !o.resident_cpu_experts && o.expert_cache != 0 && !remote &&
                              strata::core::vmm_available() &&
                              // the batch slots carve their own K/V and --vram-elastic's cache is not one VMM range
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-19](BLOCKS.md#01-19)
  ```
                          : multi_gpu ? "a layer split has a cache per GPU"
                          : std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                        [](int n) { return n > 0; }) ? "the helper caches on other GPUs"
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-20](BLOCKS.md#01-20)
  ```
          // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
          // free figure read again; while it is short of the reserve the cache is reopened smaller.
          // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-21](BLOCKS.md#01-21)
  ```
          const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
          return std::min(max_chunk, rounded);
      };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-22](BLOCKS.md#01-22)
  ```
      };
      auto plan_lend = [&](int64_t& chunk) -> int64_t {
          auto slots_for = lend_slots;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-23](BLOCKS.md#01-23)
  ```
                  if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;
                  if (o.prefill_auto_cap > 0 && c > o.prefill_auto_cap) continue;   // #1630
                  const int64_t k = slots_for(c);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-24](BLOCKS.md#01-24)
  ```
              constexpr int64_t kRingMin = 16;
              const int64_t small = old_rule();   // 0.1.39's chunk (and its ring for the prompts that fit it)
              const int64_t ring_max = strata::prefill::Prefill::ring_cap_for(small);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-25](BLOCKS.md#01-25)
  ```
                      // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which case the
                      // chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                      const int64_t k = slots_for(t);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-26](BLOCKS.md#01-26)
  ```
          }
          for (int64_t c = chunk; c >= 256; c /= 2) {
              const int64_t k = slots_for(c);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-27](BLOCKS.md#01-27)
  ```
              // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
              auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                  const int64_t k = part_slots(p, c);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-28](BLOCKS.md#01-28)
  ```
                      // clamped at 0: a cache under 128 slots has a negative lend budget, and `slots - budget` is
                      // then `slots`, which is off_[slots] - the end of the prefix sum, so that cache lends 0 bytes
                      // rather than reading one entry past its offsets
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-29](BLOCKS.md#01-29)
  ```
          // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
          // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
          // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-30](BLOCKS.md#01-30)
  ```
              }
              if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
              if (share_pool) sp.set_cpu_pool(&pool);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-31](BLOCKS.md#01-31)
  ```
                  const int32_t* r = host_res.data() + l * g.n_expert;
                  for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                      if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-32](BLOCKS.md#01-32)
  ```
              for (float& v : drive.d.usage) v *= o.adapt_decay;
              return true;
          };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-33](BLOCKS.md#01-33)
  ```
              if (batch_on() || (piped && pipe_inflight())) {
                  if (!try_next_line(line)) {
                      if (!(piped ? pump(true) : batch_step())) return 1;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-34](BLOCKS.md#01-34)
  ```
              // participant's own cache, and marking only that participant's own layers
              auto lend = [&](int64_t tokens, std::string& e) -> bool {
                  if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-35](BLOCKS.md#01-35)
  ```
                  return true;
              };
              apply_pending(true);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-36](BLOCKS.md#01-36)
  ```
                                           std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                  if (eos) { finish = "stop"; break; }
                  if (stop_req.load()) { finish = "cancel"; break; }
  ```
- `tests/core/expert_cache_elastic_test.cpp` — (new file) → [01-37](BLOCKS.md#01-37)

## 02 split-prompt-borrow

`02-split-prompt-borrow.patch` — FABLE-STRATA-Porta su 0.1.43 la stima del prestito del prompt sul device del proprio blocco (PR #1417)

- **What**: With `--layer-split`, each stage's prompt loan is estimated on its own device, not on CUDA0's.
- **Switch (default)**: always on with `--layer-split` (correctness)
- **Origin**: fork (PR #1417, ours)
- **Upstream**: PR open; not in 0.1.43
- **Measured**: Correctness: the stage-1 card no longer refuses a loan that fits. No speed claim.
- **Docs**: docs/UPSYNC.md
- **Files**: `src/prefill/prefill.cpp`, `src/program/generate.cpp`

Insertion points (2):

- `src/prefill/prefill.cpp` · `bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co` *(anchor ambiguous in the base file)* → [02-1](BLOCKS.md#02-1)
  ```
          err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
                std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
                std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [02-2](BLOCKS.md#02-2)
  ```
                                                       (uint64_t) strata::kernels::cpu::expert_layout().max_blob, need);
          };
          auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
  ```

## 03 forced-bench

`03-forced-bench.patch` — FABLE-STRATA-Porta su 0.1.43 il banco forzato STRATA_FORCE_IDS/STRATA_FORCE_WINDOWS/STRATA_FORCE_TRACE (PR #1419)

- **What**: Deterministic bench: replay token ids and verified windows (`STRATA_FORCE_IDS`, `STRATA_FORCE_WINDOWS`, trace with `STRATA_FORCE_TRACE`), adapted to the multi-stage pipeline.
- **Switch (default)**: `STRATA_FORCE_IDS=<file>[;<file>]`, `STRATA_FORCE_WINDOWS=<trace>`, `STRATA_FORCE_TRACE=<out>` (off)
- **Origin**: fork (PR #1419, ours)
- **Upstream**: PR open; not in 0.1.43
- **Measured**: Tooling: every B-A-B number in this registry is read on this bench (131k context, L01 full prefill + L02 cached prefix, 1024 tokens, same text in every arm).
- **Docs**: docs/UPSYNC.md
- **Files**: `src/program/generate.cpp`

Insertion points (11):

- `src/program/generate.cpp` · `static const char* pipe_dbg_env(const char* name) {` → [03-1](BLOCKS.md#03-1)
  ```
      return on ? std::getenv(name) : nullptr;
  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-2](BLOCKS.md#03-2)
  ```
                  continue;
              }
              char* endp = nullptr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-3](BLOCKS.md#03-3)
  ```
              const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
              std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
              std::fflush(stdout);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-4](BLOCKS.md#03-4)
  ```
                          if (v.service(&drive_pool_split, SDf(A), err) < 0) return die(err);
                          if (v.done(err)) {
                              if (!v.pl_finish(outp.data(), err)) return die(err);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-5](BLOCKS.md#03-5)
  ```
                                  // stage B now (its PLE rows from the tokens before it as they will be once A is
                                  // committed whole), so its launch behind A is only the graph launch (never while a
                                  // rollback is pending: B's verifier is the one the undo commit reads)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-6](BLOCKS.md#03-6)
  ```
                      // state)
                      if (A.ready && !A.fl && A.fr < pl_nf && !doomed) {
                          const int st = A.fr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-7](BLOCKS.md#03-7)
  ```
                      }();
                      if (B.ready && !B.fl && B.fr < pl_nf && B.fr < pl_spec_depth && A.fr > B.fr && !doomed) {
                          const int st = B.fr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-8](BLOCKS.md#03-8)
  ```
                      chain_kind = 0;
                      int a = 0;
                      while (a < A.T - 1 && A.tok[a + 1] == outp[(size_t) a]) ++a;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-9](BLOCKS.md#03-9)
  ```
                  window[0] = x;
                  for (int i = 1; i < T_mtp; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
                  for (int i = 0; i < chain_n; ++i) window[(size_t) (T_mtp + i)] = cbuf[(size_t) i];
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-10](BLOCKS.md#03-10)
  ```
                      }
                  }
                  int a = 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-11](BLOCKS.md#03-11)
  ```
                  p += a + 1;
              }
              const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
  ```

## 04 b6-mmvq-rows

`04-b6-mmvq-rows.patch` — FABLE-STRATA-Porta su 0.1.43 l'MMVQ multi-colonna di B6 opt-in (STRATA_B6_MMVQ_ROWS=1, PR #1418)

- **What**: Dense GEMV (MMVQ) that reuses the quantized activations across 2-4 output rows (Q4_K, Q5_K, Q6_K, IQ4_XS), bitwise.
- **Switch (default)**: `STRATA_B6_MMVQ_ROWS=1` (off)
- **Origin**: fork (PR #1418, ours)
- **Upstream**: PR open; validated independently on a 2080 Ti/Windows (bitwise, kernels -17/-33%); upstream went its own way with the MMVQ_IL tables (0.1.42: sm_80+, 0.1.43: sm_86 and one column, Q3_K)
- **Measured**: Reference config uses it with `STRATA_MMVQ_IL=0`: against it, upstream's IL path -3.0% and the sm_75 shape table of PR #1437 0.0% on the 2080 Ti (11-12 Oct).
- **Notes**: Adjacency conflict in CMakeLists.txt with 0.1.43's router_chain_bench: both kept. Also adds `b6_mmvq_bench`.
- **Docs**: docs/UPSYNC.md
- **Files**: `CMakeLists.txt`, `include/strata/kernels/native_mmvq.hpp`, `src/kernels/b6_mmvq_bench.cpp`, `src/kernels/cuda/native_mmvq.cu`

Insertion points (6):

- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [04-1](BLOCKS.md#04-1)
  ```
    add_executable(router_chain_bench src/kernels/router_chain_bench.cpp)
    target_include_directories(router_chain_bench PRIVATE ${_strata_gpu_include_directories})
    target_link_libraries(router_chain_bench PRIVATE strata_kernels ${_strata_gpu_runtime_target})
  ```
- `include/strata/kernels/native_mmvq.hpp` · `std::size_t native_q8_1_bytes(int n_in, int ncols = 1);` → [04-2](BLOCKS.md#04-2)
  ```
  // graph capture; captured graphs keep the kernels they captured.
  void native_mmvq_set_multi_exact(bool exact);
  bool native_mmvq_multi_exact();
  ```
- `src/kernels/b6_mmvq_bench.cpp` — (new file) → [04-3](BLOCKS.md#04-3)
- `src/kernels/cuda/native_mmvq.cu` · `struct SmallTraits<IQ4NLBlock, 4> {` → [04-4](BLOCKS.md#04-4)
  ```
  // llama.cpp's generic multi-column table (ncols 2-4: 4 warps; 5-8: 2 warps; always 2 rows per block): faster,
  // equal to ncols = 1 only to float rounding (the cross-warp reduction groups partial sums differently).
  bool g_multi_exact = true;   // until the upstream layout is timed on an idle GPU (plan rule: default only what is measured)
  ```
- `src/kernels/cuda/native_mmvq.cu` · `__global__ void native_mmvq_multi_kernel(const typename F::Block* __restrict__ w` → [04-5](BLOCKS.md#04-5)
  ```
      if (!g_multi_exact) {
          constexpr int NW = NCOLS <= 4 ? 4 : 2;
          const unsigned blocks = unsigned((std::size_t(n_out) + 1) / 2);
  ```
- `src/kernels/cuda/native_mmvq.cu` · `void native_q6_k_unpack(const void* weights) {` → [04-6](BLOCKS.md#04-6)
  ```
  
  void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
  bool native_mmvq_multi_exact() { return g_multi_exact; }
  ```

## 05 elastic-split-pipeline

`05-elastic-split-pipeline.patch` — FABLE-STRATA-Porta su 0.1.43 la cache elastica di CUDA0 con --layer-split e accanto a --pipeline-windows (di default con due schede)

- **What**: The elastic cache with `--layer-split` (CUDA0's block only) and beside `--pipeline-windows`: the queue moves only in stage-0 gaps, growth published when the fills landed, 1.5 GiB queue, 128 MiB step, the prompt borrows the core too; on by default with two cards.
- **Switch (default)**: `STRATA_PIPELINE_ELASTIC` (on with two cards and `--pipeline-windows`; `=0` off)
- **Origin**: fork
- **Upstream**: never proposed (depends on 01)
- **Measured**: Part of the Daily's two-card configuration (83.7 t/s at 40-71k on a real SWE case, 7 Oct). The 0.1.42 port with the whole stack measured +0.7% against the Daily (131k forced bench).
- **Depends on**: 01
- **Docs**: docs/UPSYNC.md
- **Files**: `src/program/generate.cpp`

Insertion points (30):

- `src/program/generate.cpp` · `static const char* pipe_dbg_env(const char* name) {` → [05-1](BLOCKS.md#05-1)
  ```
      return on ? std::getenv(name) : nullptr;
  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-2](BLOCKS.md#05-2)
  ```
          // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
          // free figure read again; while it is short of the reserve the cache is reopened smaller.
          // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-3](BLOCKS.md#05-3)
  ```
                  --fake_fails;
                  err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
              } else {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-4](BLOCKS.md#05-4)
  ```
      // --elastic: the TWO-ZONE cache.  Slots [0, el_core) are the core - filled from the profile at startup, never
      // picked by the adaptive swaps to leave, never shrunk, never lent; the tail after it is the elastic zone.
      int64_t el_core = 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-5](BLOCKS.md#05-5)
  ```
      };
      auto plan_lend = [&](int64_t& chunk) -> int64_t {
          auto slots_for = lend_slots;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-6](BLOCKS.md#05-6)
  ```
              // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
              auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                  const int64_t k = part_slots(p, c);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-7](BLOCKS.md#05-7)
  ```
                      int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - 128,
                                                                    kAutoLendPct * p.cache->slots() / 100));
                      // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-8](BLOCKS.md#05-8)
  ```
          // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
          // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
          // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-9](BLOCKS.md#05-9)
  ```
          // --pipeline-windows: set while the tier adapts beside windows in flight (see its use in adapt)
          std::function<void()> adapt_fence;
          // the VRAM tier follows the conversation (the same rule as the speculative loop below)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-10](BLOCKS.md#05-10)
  ```
                  const int32_t* r = host_res.data() + l * g.n_expert;
                  for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                      if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-11](BLOCKS.md#05-11)
  ```
          // that shrank too far lends a smaller chunk, the same rule as plan_lend).  Run only where nothing is in
          // flight: a request's start (nothing lent yet) and the decode's safe point (window, commit, draft and the
          // adaptive swaps all finished).
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-12](BLOCKS.md#05-12)
  ```
          Clock::time_point el_last_grow{}, el_last_shrink{};
          int el_backoff = 1;
          auto el_free = []() -> int64_t {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-13](BLOCKS.md#05-13)
  ```
              const bool have_usage = !drive.d.usage.empty();
              std::vector<std::vector<std::pair<float, int32_t>>> keep((size_t) g.n_layers), tail((size_t) g.n_layers);
              for (int64_t l = 0; l < g.n_layers; ++l)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-14](BLOCKS.md#05-14)
  ```
          // GROW by at most `room` bytes: the missing experts the conversation routes most (usage >= 2, as the adaptive
          // tier's candidates), then the profile's order for what it has not asked for yet; filled on the refill
          // stream and published like the adaptive swaps (apply_pending), so the windows never wait for them
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-15](BLOCKS.md#05-15)
  ```
              for (const auto& pr : profile) {
                  if (used + (int64_t) lay.max_blob > room) break;
                  const int32_t i = (int32_t) (pr.first * g.n_expert + pr.second);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-16](BLOCKS.md#05-16)
  ```
              // under WDDM the free figure is only trustworthy once the pages are written: `grow` zeroed them
              cudaDeviceSynchronize();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-17](BLOCKS.md#05-17)
  ```
                      e = "elastic cache: a fill copy failed";
                      return false;
                  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-18](BLOCKS.md#05-18)
  ```
          // one step of the policy: shrink at once below the reserve; grow once the room has stayed for stable_ms.  At a
          // request's start it only shrinks: a growth's fills land asynchronously in the tail, and the prompt is about
          // to borrow the tail - the decode's safe point grows, and apply_pending publishes the fills before any lend
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-19](BLOCKS.md#05-19)
  ```
                      el_backoff = std::min(el_backoff * 2, 8);   // a shrink right after a growth: it grew too far
                  el_last_shrink = t;
                  el_seen.clear();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-20](BLOCKS.md#05-20)
  ```
              el_stable_since = t;
              el_last_grow = t;
              el_seen.clear();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-21](BLOCKS.md#05-21)
  ```
                  switch (astate) {
                  case AState::Idle:
                      if (drain || a_win - a_last < (int64_t) o.adapt_every) return true;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-22](BLOCKS.md#05-22)
  ```
          // --pipeline-windows 2: the loop has drained (nothing in flight on either card): what waits for a gap goes now,
          // and the round in flight goes on unfenced (a_pl_idle unset) from the next tick, the next request's drain.  The
          // job's work is the host's alone, so the wait for it ends.
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-23](BLOCKS.md#05-23)
  ```
                  bool b_done = false;     // B has been made from the chain in flight
                  int chain_n = 0;
                  int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-24](BLOCKS.md#05-24)
  ```
                  // --adapt-async: the asynchronous tier beside the windows (see adapt_tick).  A stage is idle when neither
                  // of its verifiers has a window in flight (a commit or a snapshot copy waits on nothing of the host's)
                  if (ajob) a_pl_idle = [&](int st) { return !PV[st][0]->in_flight() && !PV[st][1]->in_flight(); };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-25](BLOCKS.md#05-25)
  ```
                      if (pl_adapt_thr.joinable()) return true;   // still adapting
                      pl_apply();
                      if (!drive.d.usage.empty() && o.adapt_every > 0 && (rounds % o.adapt_every) == 0 && pending.empty() &&
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-26](BLOCKS.md#05-26)
  ```
                          } else if (!err.empty()) return die(err);
                      }
                      if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-27](BLOCKS.md#05-27)
  ```
                          if (st == 0 && force_win != nullptr) force_window(A.p, A.tok[0], A.T, A.tok, S);   // lab bench replay
                          if (st == 0 && A.p + A.T > o.max_context) { ending = true; continue; }
                          if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-28](BLOCKS.md#05-28)
  ```
                          if (st == 0 && force_win != nullptr) force_window(B.p, B.tok[0], B.T, B.tok, S);   // lab bench replay
                          if (st > 0 || (B.p_on >= theta && B.p + B.T <= o.max_context)) {
                              if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-29](BLOCKS.md#05-29)
  ```
                  // unfenced (the INVARIANT at pl_drain: a checkpoint, a park, the serial loop and the next request find
                  // nothing pipelined in flight) at the next tick, the next request's drain
                  a_pl_settle();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-30](BLOCKS.md#05-30)
  ```
                                           "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                   (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                   (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
  ```

## 06 prefix-disk

`06-prefix-disk.patch` — FABLE-STRATA-Porta su 0.1.43 il prefisso su disco con due schede (--prompt-cache-file, formato v2)

- **What**: `--prompt-cache-file`: the long prompt prefix saved to disk and restored across restarts, format v2 with one image per split block; the MTP drafter computes every cell of the saved root.
- **Switch (default)**: `--prompt-cache-file <path>` (off)
- **Origin**: fork
- **Upstream**: never proposed (upstream has `--prompt-cache-tail/-every` in RAM only)
- **Measured**: Restart of an agent session: the shared 40-130K prefix is restored from the file instead of being read again (a 40K first prompt took ~42 s to read here). Restore time: see the fork's smoke test of the PR branch. The file identity covers exe, args and STRATA_*/CUDA_* variables.
- **Notes**: Uses upstream's `available_host_bytes()` (0.1.42 removed `conversation_available_memory`).
- **Docs**: docs/UPSYNC.md
- **Files**: `CMakeLists.txt`, `include/strata/core/mtp.hpp`, `include/strata/core/prefix_cache_file.hpp`, `src/core/mtp.cpp`, `src/core/prefix_cache_file.cpp`, `src/core/prefix_cache_file_test.cpp`, `src/program/generate.cpp`

Insertion points (19):

- `CMakeLists.txt` · `option(STRATA_BUILD_TESTS "Build the test suite" ${_strata_tests_default})` → [06-1](BLOCKS.md#06-1)
  ```
  option(STRATA_BUILD_CONVERSATION_TESTS "Build standalone conversation cache tests" OFF)
  option(STRATA_WERROR "Treat compiler warnings as errors" OFF)
  ```
- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [06-2](BLOCKS.md#06-2)
  ```
      src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp
      src/core/arch_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
    target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ```
- `include/strata/core/mtp.hpp` · `public:` → [06-3](BLOCKS.md#06-3)
  ```
      const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
      const void* tensor_q8(const char* name) const { return q8(name); }
      QsaState& kv_state_rw() { return st_; }
  ```
- `include/strata/core/mtp.hpp` · `private:` → [06-4](BLOCKS.md#06-4)
  ```
      std::string rt_dir_;
      int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
      int64_t prompt_len_ = 0;
  ```
- `include/strata/core/prefix_cache_file.hpp` — (new file) → [06-5](BLOCKS.md#06-5)
- `src/core/mtp.cpp` · `bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_` → [06-6](BLOCKS.md#06-6)
  ```
      const OnDevice on_device(device_);
      const Clock::time_point t0 = Clock::now();
      const int64_t HCN = g_->hc * g_->n_embd;
  ```
- `src/core/prefix_cache_file.cpp` — (new file) → [06-7](BLOCKS.md#06-7)
- `src/core/prefix_cache_file_test.cpp` — (new file) → [06-8](BLOCKS.md#06-8)
- `src/program/generate.cpp` · `(file scope)` → [06-9](BLOCKS.md#06-9)
  ```
  #include "strata/core/conversation_snapshot.hpp"
  #include "strata/core/conversation_file.hpp"
  #include "strata/core/session_save_reclaim.hpp"
  ```
- `src/program/generate.cpp` · `(file scope)` → [06-10](BLOCKS.md#06-10)
  ```
  #include <cstdio>
  #include <cstdlib>
  #include <cstring>
  ```
- `src/program/generate.cpp` · `struct Options {` → [06-11](BLOCKS.md#06-11)
  ```
      /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
      /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
      int prompt_cache = 6;
  ```
- `src/program/generate.cpp` · `void usage() {` → [06-12](BLOCKS.md#06-12)
  ```
                   "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                   "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                   "                       of RAM each; 0 = read every prompt from the start)\n"
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-13](BLOCKS.md#06-13)
  ```
          else if (a == "--vision") o.vision = true;
          else if (a == "--session-save-reclaim") o.session_save_reclaim = true;
          else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-14](BLOCKS.md#06-14)
  ```
          std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                       o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
      // parking with --layer-split saves every stage (SavedConversation::stage_images)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-15](BLOCKS.md#06-15)
  ```
          bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
          // without --mtp the conversation cache stays on: a parked image carries the draft layer's K/V only when there is one
          // (the snapshots accept a null draft)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-16](BLOCKS.md#06-16)
  ```
                  incoming.reset();
                  err.clear();
              }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-17](BLOCKS.md#06-17)
  ```
                                   mtp.kv_state().kv_mode, "ram",
                                   (long long) (mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size));
                  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-18](BLOCKS.md#06-18)
  ```
                          if (i >= o.prompt_cache_root) root_at = i;
                          break;
                      }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-19](BLOCKS.md#06-19)
  ```
                  if (trace && to == message_at)
                      std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                                   (long long) message_at, (long long) (turn_at - message_at));
  ```

## 07 vram-free-nvml

`07-vram-free-nvml.patch` — FABLE-STRATA-Porta su 0.1.43 la PR upstream #1911: VRAM libera della scheda intera via NVML (STRATA_VRAM_FREE=device), letta anche dalla cache elastica

- **What**: `STRATA_VRAM_FREE=device`: free VRAM of the whole card through NVML (matched by PCI bus id, so `CUDA_VISIBLE_DEVICES` order does not matter); the elastic cache reads it too.
- **Switch (default)**: `STRATA_VRAM_FREE=device` (off: CUDA's per-process view)
- **Origin**: upstream PR #1911 (Xiaomi2008) + fork glue for the elastic cache
- **Upstream**: PR open; not in 0.1.43
- **Measured**: Fixes the 7-36 t/s collapses under WDDM paging on the 3060 (CUDA saw 9.7 GB free, NVML 8.1): with it the stage holds 77 t/s. Daily default since 11 Oct.
- **Depends on**: 01
- **Notes**: The fork's always-defined build macro STRATA_EC_NVML_FREE around the NVML code was removed (follow-up commit, no behaviour change).
- **Docs**: docs/UPSYNC.md
- **Files**: `docs/DETAILS.md`, `include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`, `src/program/generate.cpp`

Insertion points (6):

- `docs/DETAILS.md` · `GPU nor the RAM budget holds (only pages - the experts computed are the same; `S` → [07-1](BLOCKS.md#07-1)
  ```
  is what runs [Unsloth's UD-Q4_K_XL](UNSLOTH_Q4.md) (72 GiB of experts) on a 64 GB PC: 7-8.5 tokens/s at N = 40 on an
  RTX 5070, against ~3 tokens/s before these changes.
  ```
- `include/strata/core/expert_cache.hpp` · `bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_e` → [07-2](BLOCKS.md#07-2)
  ```
  /// Free device memory for the expert cache.  On a unified memory system - one memory for the CPU and the GPU, such as
  /// the DGX Spark (GB10); CUDA reports it as cudaDevAttrIntegrated - cudaMemGetInfo counts only MemFree, not the page cache - which, with the experts read from the GGUF in place, is mostly those
  /// files' clean pages and gives way to a device allocation.  There the figure is MemAvailable less 6 GiB for the OS
  ```
- `src/core/expert_cache.cpp` · `(file scope)` → [07-3](BLOCKS.md#07-3)
  ```
  #include <utility>
  #include <cstring>
  ```
- `src/core/expert_cache.cpp` · `size_t device_free_bytes() {` → [07-4](BLOCKS.md#07-4)
  ```
          }
      }
  #endif
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [07-5](BLOCKS.md#07-5)
  ```
          std::deque<std::pair<Clock::time_point, int64_t>> el_seen;
          Clock::time_point el_last_grow{}, el_last_shrink{};
          int el_backoff = 1;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [07-6](BLOCKS.md#07-6)
  ```
              // and a page-in while the verify graph spins on a host flag stalls the request for good
              size_t free_b = 0, total_b = 0;
              cudaMemGetInfo(&free_b, &total_b);
  ```

## 08 park-no-foreign-kv

`08-park-no-foreign-kv.patch` — FABLE-STRATA-Porta su 0.1.43 la PR upstream #1503: mai parcheggiare una conversazione col K/V di un'altra

- **What**: The conversation cache never parks one conversation with another's retained K/V.
- **Switch (default)**: always on (correctness)
- **Origin**: upstream PR #1503 (sanastasiou)
- **Upstream**: PR open; not in 0.1.43
- **Measured**: Correctness only.
- **Docs**: docs/UPSYNC.md
- **Files**: `include/strata/core/conversation_cache.hpp`, `src/core/conversation_cache_test.cpp`, `src/program/generate.cpp`, `sycl/src/program/generate.cpp`

Insertion points (10):

- `include/strata/core/conversation_cache.hpp` · `struct ConversationKvReuse {` → [08-1](BLOCKS.md#08-1)
  ```
      std::vector<ConversationKv> kv;
      // Original image extent for validation, and the earliest subsequent rewrite.
      int64_t captured_tokens = 0, unchanged_tokens = 0;
  ```
- `include/strata/core/conversation_cache.hpp` · `public:` → [08-2](BLOCKS.md#08-2)
  ```
      // Retain only the restored K/V buffers, not duplicate running checkpoints.
      // This optimization never evicts a parked conversation to make itself fit.
      // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
  ```
- `include/strata/core/conversation_cache.hpp` · `public:` → [08-3](BLOCKS.md#08-3)
  ```
          if (reuse_.unchanged_tokens <= 0) reuse_ = {};
      }
      ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
  ```
- `src/core/conversation_cache_test.cpp` · `int main() {` → [08-4](BLOCKS.md#08-4)
  ```
          auto none = conversation_checkpoints_split({}, 3);
          check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
      }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-5](BLOCKS.md#08-5)
  ```
                  ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
              } merge_back{cs, checks, n_st > 0};
              const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-6](BLOCKS.md#08-6)
  ```
          // conversation an idle slot holds reads on from there.  Every stage, the same calls.
          // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
          auto copy_from_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-7](BLOCKS.md#08-7)
  ```
                  if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                      std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                      for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
  ```
- `sycl/src/program/generate.cpp` · `int main(int argc, char **argv) try {` → [08-8](BLOCKS.md#08-8)
  ```
                  ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
              } merge_back{cs, checks, n_st > 0};
              const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
  ```
- `sycl/src/program/generate.cpp` · `int main(int argc, char **argv) try {` → [08-9](BLOCKS.md#08-9)
  ```
          // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
          auto copy_from_slot = [&](int b, const ConvCheckpoint *at,
                                    std::string &e) -> bool {
  ```
- `sycl/src/program/generate.cpp` · `int main(int argc, char **argv) try {` → [08-10](BLOCKS.md#08-10)
  ```
                  if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                      std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                      for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
  ```

## 09 pipeline-architectds

`09-pipeline-architectds.patch` — FABLE-STRATA-Porta su 0.1.43 il profilo architectds della pipeline (STRATA_PIPELINE_ELASTIC: verificata servita per prima, lookup oltre una finestra, theta 0,10) e i log di rollback e latenza della catena

- **What**: The two-window pipeline's profile from the Daily: the verified window is served first (front stage of A, last stage of A, then D and B one layer per pass), lookup beyond one window, theta 0.10; plus two log lines (rollback cost per window class, chain latency).
- **Switch (default)**: `STRATA_PIPELINE_ELASTIC` (on with two cards and `--pipeline-windows`; `=0`: upstream's theta 0.20 and D-A-B order)
- **Origin**: fork (from the architectds lab branch)
- **Upstream**: never proposed
- **Measured**: 0.1.42 without it: -4.6/-7.3/-7.8% against the Daily (speculative windows 133/600 instead of 164/600, stage host 0.50 -> 1.69 ms). With it and patch 11: +0.7% (parity).
- **Depends on**: 05
- **Docs**: docs/UPSYNC.md
- **Files**: `include/strata/core/verify.hpp`, `src/core/verify.cpp`, `src/program/generate.cpp`

Insertion points (20):

- `include/strata/core/verify.hpp` · `public:` → [09-1](BLOCKS.md#09-1)
  ```
      /// window as they WILL be, their pages prefetched).  A later `pl_launch` of the same window (T, pos0, tokens, and
      /// `ss.ple_prev` equal to `ple_prev` by then) skips the staging; anything else stages again.
      bool prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
  ```
- `src/core/verify.cpp` · `bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string` → [09-2](BLOCKS.md#09-2)
  ```
  int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {` → [09-3](BLOCKS.md#09-3)
  ```
      }
      const int G = groups_[T] > 0 ? groups_[T] : 1;
      const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {` → [09-4](BLOCKS.md#09-4)
  ```
          if (fl_k_ == 0 && !gather_ple()) return -1;
          *(volatile uint32_t*) h_flag_ = want;
          ++fl_k_;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-5](BLOCKS.md#09-5)
  ```
              int pl_pw = pipe ? o.pipeline_windows : 0;
              float pl_theta = [] {   // a speculative window is launched only when its estimated chance is at least this
                  const char* v = pipe_dbg_env("STRATA_PIPELINE_THETA");
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-6](BLOCKS.md#09-6)
  ```
                      bool spec = false;     // launched speculatively (behind the window before, ahead of its verdict)
                      bool made = false;     // made from a chain (its outcome can be scored even when the gate held it)
                      bool sfx = false;      // a lookup window (the suffix drafter's drafts)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-7](BLOCKS.md#09-7)
  ```
                  bool b_done = false;     // B has been made from the chain in flight
                  int chain_n = 0;
                  int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-8](BLOCKS.md#09-8)
  ```
                                    std::chrono::duration<double, std::milli>(Clock::now() - pl_tr0).count(), ev, seq, x0, x1);
                      pl_trace += b;
                  };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-9](BLOCKS.md#09-9)
  ```
                      // undo restores everything it wrote).  STRATA_PIPELINE_DOOM_SKIP=0: served in full.
                      static const bool doom_skip = [] { const char* e = pipe_dbg_env("STRATA_PIPELINE_DOOM_SKIP"); return e == nullptr || std::atoi(e) != 0; }();
                      const bool skip = doom_skip && doomed && &w == &D;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-10](BLOCKS.md#09-10)
  ```
                          B.p_on = pl_lookup_pon;
                          B.ready = true;
                          B.made = true;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-11](BLOCKS.md#09-11)
  ```
                  // per window class (0: fresh, 1: run speculatively ahead of its predecessor's verdict): the time from
                  // the previous verdict, the tokens it emitted, how many
                  double cls_ms[2] = {0, 0}, cls_tok[2] = {0, 0}, cls_n[2] = {0, 0};
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-12](BLOCKS.md#09-12)
  ```
                              if (!adapt_tick(false, false)) return die("an adaptive refill failed");
                          }
                      }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-13](BLOCKS.md#09-13)
  ```
                              tre("F1", A.seq, A.T);
                          } else if (!err.empty()) return die(err);
                      }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-14](BLOCKS.md#09-14)
  ```
                                      if (!VF(0, B).prestage(B.T, B.tok, B.p, prev, err)) return die(err);
                                  }
                                  b_done = true;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-15](BLOCKS.md#09-15)
  ```
                          if (A.sfx) { ++sfx_windows; sfx_drafts += A.T - 1; sfx_ok += a; }
                          if (A.seq > 0 && !last) policy.observe(A.sfx, A.T, a, A.sfx_match, now - last_verdict);
                          cls_ms[c] += now - last_verdict;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-16](BLOCKS.md#09-16)
  ```
                      if (!last && !ajob && !pl_adapt()) return die("an adaptive refill failed");
                      if (on && !last) {
                          ++pl_on;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-17](BLOCKS.md#09-17)
  ```
                              return die(err);
                          chain_kind = 1;
                          tre("CL", A.seq, 1);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-18](BLOCKS.md#09-18)
  ```
                              if ((A.com & (1u << st)) == 0 && !VF(st, A).pl_commit_async(A.T, err)) return die(err);
                          A = B;
                          B = PW{};
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-19](BLOCKS.md#09-19)
  ```
                                  return die(err);
                              chain_kind = 2;
                              tre("CL", A.seq, 2);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [09-20](BLOCKS.md#09-20)
  ```
                                           "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                   (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                   (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
  ```

## 10 attn-daily

`10-attn-daily.patch` — FABLE-STRATA-Porta su 0.1.43 l'attenzione QSA del decode del Daily (STRATA_ATTN_DAILY=1: chunk pre75 su ogni scheda e merge v2) col test di parita

- **What**: The Daily's QSA decode attention on every CUDA card: the two-cells-per-step chunk kernel (upstream keeps it below sm_75 only) and the `attn_merge_v2` merge; bitwise with the default; parity harness `qsa_merge_daily_parity`.
- **Switch (default)**: `STRATA_ATTN_DAILY=1` (off); `STRATA_ATTN_MERGE_V2`, `STRATA_ATTN_PRE75=0` for the two halves
- **Origin**: fork (kernel body from PR #540 sskver and the merge from Hardin22, both carried in the old Daily)
- **Upstream**: upstream has the chunk kernel as `attn_chunk_kernel_pre75` (sm_60 builds only); the merge v2 is not upstream
- **Measured**: nsys on the 3060: `attn_chunk_kernel` 99.8 -> 46.4 us a call. In the engine the attention is ~8% of the per-row cost (docs/QSAROWS.md): part of the +0.7% parity, not separable.
- **Docs**: docs/UPSYNC.md
- **Files**: `CMakeLists.txt`, `include/strata/kernels/qsa_decode_attn.hpp`, `src/kernels/cuda/qsa_decode_attn.cu`, `src/kernels/qsa_merge_daily_parity.cpp`

Insertion points (11):

- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [10-1](BLOCKS.md#10-1)
  ```
      target_link_libraries(qsa_topk_active_parity PRIVATE strata_kernels CUDA::cudart)
      add_test(NAME qsa_topk_active_parity COMMAND qsa_topk_active_parity)
      set_tests_properties(qsa_topk_active_parity PROPERTIES SKIP_RETURN_CODE 77)
  ```
- `include/strata/kernels/qsa_decode_attn.hpp` · `void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32` → [10-2](BLOCKS.md#10-2)
  ```
  void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                             int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream);
  ```
- `src/kernels/cuda/qsa_decode_attn.cu` · `__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __rest` → [10-3](BLOCKS.md#10-3)
  ```
  // run on gfx103x (RDNA2), with 8 cells per step and the butterfly's lane exchanges in registers (DPP / permlanex16)
  // instead of ds_bpermute: there the score loop is bound by the LDS pipe (the q reads and ds_bpermute share it), not by
  // occupancy - 8 cells per step at 7 waves/SIMD beat 1 cell at 9.  Every other card keeps exactly the kernel above.
  ```
- `src/kernels/cuda/qsa_decode_attn.cu` · `__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict_` → [10-4](BLOCKS.md#10-4)
  ```
      attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
  }
  ```
- `src/kernels/cuda/qsa_decode_attn.cu` · `bool query_swizzle_gfx906_device() {` → [10-5](BLOCKS.md#10-5)
  ```
  }
  #endif
  ```
- `src/kernels/cuda/qsa_decode_attn.cu` · `bool pre75_attn() {` → [10-6](BLOCKS.md#10-6)
  ```
      }
      return cc[dev] < 75;
  #endif
  ```
- `src/kernels/cuda/qsa_decode_attn.cu` · `void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int3` → [10-7](BLOCKS.md#10-7)
  ```
      float* part_l = part_m + (size_t) n_chunks * s.n_head;
      const float scale = 1.0f / sqrtf((float) HD);
      const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
  ```
- `src/kernels/cuda/qsa_decode_attn.cu` · `void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int3` → [10-8](BLOCKS.md#10-8)
  ```
      } else
  #endif
      if (kv_mode == 3)
  ```
- `src/kernels/cuda/qsa_decode_attn.cu` · `void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32` → [10-9](BLOCKS.md#10-9)
  ```
      else
          STRATA_ATTN_CHUNK(0)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, 0, 0);
  ```
- `src/kernels/cuda/qsa_decode_attn.cu` · `void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32` → [10-10](BLOCKS.md#10-10)
  ```
  }  // namespace strata::kernels
  ```
- `src/kernels/qsa_merge_daily_parity.cpp` — (new file) → [10-11](BLOCKS.md#10-11)

## 11 df-defaults

`11-df-defaults.patch` — FABLE-STRATA-Rimetti STRATA_DF_BRANCH (CUDA su Windows) e STRATA_DF_PLE (CUDA) accesi di default come nel Daily

- **What**: `STRATA_DF_BRANCH` (mixer side branches in the window graph; CUDA on Windows) and `STRATA_DF_PLE` (PLE post-projections in one pass; CUDA) on by default, as in the Daily.
- **Switch (default)**: `STRATA_DF_BRANCH=0`, `STRATA_DF_PLE=0` turn them off (upstream: opt-in `=1`)
- **Origin**: fork (defaults of PR #904's switches)
- **Upstream**: upstream merged #904 opt-in and keeps BRANCH off after a Linux stall (#905, never seen on Windows)
- **Measured**: Daily measurements: BRANCH +2.4% at 131k, PLE +3.3% (bitwise). On 0.1.42 their absence was the main part of ~1.1 ms of GPU per window.
- **Upstream defaults it changes**: DF_BRANCH, DF_PLE default on (see the table above)
- **Docs**: docs/UPSYNC.md
- **Files**: `src/core/verify.cpp`

Insertion points (2):

- `src/core/verify.cpp` · `bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&` → [11-1](BLOCKS.md#11-1)
  ```
      {   // STRATA_DF_BRANCH=1 (opt-in): the mixer's side branches.  Off by default: on Linux (open modules, GSP) a
          // stage's window with the branches stopped inside its graph when an NVML query (nvidia-smi) came between
          // requests (2x RTX 3090 and 2x RTX 2080 Ti, #905); never seen on Windows.  0 / unset: one stream, as before
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [11-2](BLOCKS.md#11-2)
  ```
              // below (the norms divide at run time as its norms do; the history advances token by token), which also
              // writes the commit's history snapshots.  Scratch: layer 0's expert rows, combined by now, and the
              // hyper-connection read's `lo_`, which the read below rewrites.
  ```

## 12 side-mixer

`12-side-mixer.patch` — FABLE-STRATA-Porta su 0.1.43 STRATA_SIDE_MIXER=1/2 (PR #1889): il lavoro del mixer che legge solo l'input su rami del grafo

- **What**: `STRATA_SIDE_MIXER=1/2`: the mixer's work that reads only the layer's input (GDN gates, the QSA indexer chain, with `=2` also z and the q chain) on branches of the window graph; bitwise. Turns `STRATA_DF_BRANCH` off when set.
- **Switch (default)**: `STRATA_SIDE_MIXER=1|2` (off)
- **Origin**: upstream PR #1889 (KarlGrier)
- **Upstream**: PR open; not in 0.1.43
- **Measured**: `=2`: +0.9% against DF_BRANCH on the 131k forced bench (11-12 Oct). Daily default since 12 Oct.
- **Notes**: Ported without PR #1437's sm_75 table: the branches' MMVQ_IL calls use 0.1.43's signature and interleave only for 2+ columns. 0.1.43's `STRATA_FUSE_CONVAB` (HIP) is forced off when a/b run on a branch.
- **Docs**: docs/UPSYNC.md
- **Files**: `include/strata/core/verify.hpp`, `src/core/verify.cpp`

Insertion points (14):

- `include/strata/core/verify.hpp` · `private:` → [12-1](BLOCKS.md#12-1)
  ```
      cudaEvent_t df_join_[2] = {};
      cudaGraphExec_t exec_[9] = {};
      cudaGraphExec_t exec_nr_[9] = {};   // #871: the doorbell variant of a stage that is all-resident otherwise
  ```
- `src/core/verify.cpp` · `inline bool g_qfuse_hc() {` → [12-2](BLOCKS.md#12-2)
  ```
  inline bool g_qfuse() { static const bool on = [] { const char* v = std::getenv("STRATA_QFUSE"); return v != nullptr && std::atoi(v) != 0; }(); return on; }
  #define VDBG(...) do { if (g_dbg) { std::fprintf(stderr, "verify dbg: " __VA_ARGS__); std::fflush(stderr); } } while (0)
  ```
- `src/core/verify.cpp` · `Verifier::~Verifier() {` → [12-3](BLOCKS.md#12-3)
  ```
      if (cs_ && cs_ != ext_stream_) cudaStreamDestroy(cs_);   // set_stream: the stage's stream, shared, not ours
      if (sh_cs_) cudaStreamDestroy(sh_cs_);
      if (copy_) { cudaStreamSynchronize(copy_); cudaStreamDestroy(copy_); }
  ```
- `src/core/verify.cpp` · `bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&` → [12-4](BLOCKS.md#12-4)
  ```
  #else
          df_branch_ = v != nullptr && std::atoi(v) != 0;
  #endif
  ```
- `src/core/verify.cpp` · `bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&` → [12-5](BLOCKS.md#12-5)
  ```
              all_resident_ = all_ok;
          }
      }
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-6](BLOCKS.md#12-6)
  ```
      // Pipelined windows must wait for the host verdict/snapshot fence before mutating state.
      const bool self_commit = T == 1 && !always_publish_ && !batch_rec_ && !g_qfuse() && one_token_self_commit();   // see Verifier::commit
      static const bool dec_batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-7](BLOCKS.md#12-7)
  ```
      // makes side i depend on everything captured on `cs` so far, join(i) makes `cs` depend on everything captured on
      // side i: a branch never races with what came before its fork or after its join.  Not in a split or batch window,
      // nor under the stage profiler (whose stamps are on `cs`).
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-8](BLOCKS.md#12-8)
  ```
                  float* hb = h_L_ + (size_t) gi * MT * C;
                  float* gate = gate_L_ + (size_t) gi * MT * HV;
                  float* beta = beta_L_ + (size_t) gi * MT * HV;
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-9](BLOCKS.md#12-9)
  ```
                      native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,
                                  df_side_[1]);
                  }
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-10](BLOCKS.md#12-10)
  ```
                      join(0);
                      join(1);
                  } else {
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-11](BLOCKS.md#12-11)
  ```
                                                IQ * ID, N, IQ * ID, n, df_side_[1]);
                      norm_rope_on(qidx_ + tb * IQ * ID, wiqn, (int) (n * IQ), (int) ID, pos_i + tb * IQ, df_side_[1]);
                  }
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-12](BLOCKS.md#12-12)
  ```
                      fwht256_inplace_cuda(vcur_ + tb * NKV * HD, (int64_t) n * NKV, cs);
                  }
                  stamp(l, 8, grp);
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-13](BLOCKS.md#12-13)
  ```
                          kv_append_step(st.k_pool, st.v_pool, st.page_table, step_t, kcur_ + t * NKV * HD,
                                         vcur_ + t * NKV * HD, s, cs, &st.host);
                  }
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [12-14](BLOCKS.md#12-14)
  ```
                                                attn_ + t * NH * HD, 1, cs);
                      }
                  } else {
  ```

## 13 ple-direct

`13-ple-direct.patch` — FABLE-STRATA-Porta su 0.1.43 la PLE in lettura diretta anche con la sonda lenta (STRATA_PLE_AUTO_MMAP=1 per il passaggio a mmap) e la diagnostica PLE di prefill e decode

- **What**: The PLE table stays in direct I/O even when the start-up probe measures the disk slow (0.1.42+ switches to mmap under 15 000 rows/s); PLE diagnostics for prefill (mean/max/reads over 100 ms, waiting chunks) and decode (`STRATA_DECODE_TIMING`).
- **Switch (default)**: default direct; `STRATA_PLE_AUTO_MMAP=1` restores upstream's switch to mmap
- **Origin**: fork
- **Upstream**: never proposed
- **Measured**: PLE on a cold SATA SSD: 131k prefill 260 s (mmap) -> 75 s (direct), decode +3.9%. The probe read 1 259 rows/s; direct I/O with `--ple-inflight` reads and chunk read-ahead hides it.
- **Upstream defaults it changes**: `--ple-io` auto no longer switches to mmap
- **Docs**: docs/UPSYNC.md
- **Files**: `include/strata/core/verify.hpp`, `include/strata/ngram/ple_reader.hpp`, `src/core/verify.cpp`, `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`

Insertion points (14):

- `include/strata/core/verify.hpp` · `public:` → [13-1](BLOCKS.md#13-1)
  ```
      /// the pool never plans a PCIe share (--pcie-frac 0): the window skips that path.  Before the first run.
  
      double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
  ```
- `include/strata/ngram/ple_reader.hpp` · `struct ReaderStats {` → [13-2](BLOCKS.md#13-2)
  ```
      double wait_us = 0;           ///< time `collect` spent blocked
      double submit_us = 0;         ///< time spent inside the read submission call (non-zero = it blocks)
      double read_us_sum = 0;       ///< sum of per-read latencies (issue to completion)
  ```
- `src/core/verify.cpp` · `constexpr float EPS = 1e-6f;` → [13-3](BLOCKS.md#13-3)
  ```
  using Clock = std::chrono::steady_clock;
  double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
  const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
  ```
- `src/core/verify.cpp` · `bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool,` → [13-4](BLOCKS.md#13-4)
  ```
              _mm_sfence();
              *flag = 1;
              ms_host += ms_since(tp);
  ```
- `src/core/verify.cpp` · `bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool,` → [13-5](BLOCKS.md#13-5)
  ```
              std::atomic_thread_fence(std::memory_order_seq_cst);
              _mm_sfence();
              ms_host += ms_since(tp);
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la` → [13-6](BLOCKS.md#13-6)
  ```
          std::atomic_thread_fence(std::memory_order_seq_cst);
          _mm_sfence();
          ms_host += ms_since(tp);
  ```
- `src/kernels/ngram.cpp` · `void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_del` → [13-7](BLOCKS.md#13-7)
  ```
  std::string PleTable::io_report() const {
      if (impl_->mode != PleIo::Direct || !impl_->reader.is_open()) return {};
      const strata::ngram::ReaderStats s = impl_->reader.snapshot();
  ```
- `src/kernels/ngram.cpp` · `std::string PleTable::io_report() const {` → [13-8](BLOCKS.md#13-8)
  ```
                    (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                    s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                    (unsigned long long) impl_->reader.cache_capacity());
  ```
- `src/ngram/ple_reader.cpp` · `struct PleReader::Impl {` → [13-9](BLOCKS.md#13-9)
  ```
  
      void record_latency(double us) {
          stats.read_us_sum += us;
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [13-10](BLOCKS.md#13-10)
  ```
                  });
              }
              ple_buf ^= 1;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-11](BLOCKS.md#13-11)
  ```
          }
          // Recommend, never force: when the default direct reads measure slow, switch to the mapped reads (40-100x
          // faster on the reporters' drives); an explicit --ple-io direct is kept and only warned about.
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-12](BLOCKS.md#13-12)
  ```
                      std::fprintf(stderr, "strata generate: WARNING: --ple-io direct measures slow here (%.0f rows/s, about %.0f "
                                           "MB/s of random 4 KiB reads): a prompt can spend minutes reading layer 1 and trip the "
                                           "stall watchdog (#1425, #1629). --ple-io mmap was 40-100x faster on such drives; "
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-13](BLOCKS.md#13-13)
  ```
              const Clock::time_point d0 = Clock::now();
              // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
              static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-14](BLOCKS.md#13-14)
  ```
                      if (st == 0) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());   // text as before
                      else std::fprintf(stderr, "strata decode GPU stages, stage %d (ms/window):%s\n", st, pr.c_str());
                  }
  ```

## 14 prefill-sm75

`14-prefill-sm75.patch` — FABLE-STRATA-Porta su 0.1.43 il prefill del Daily su sm_75: STRATA_SELECT_SIMT di default sotto sm_80, dequant con store da 16 B e hc16 sui tensor core FP16 (STRATA_TC_SM75=0 li spegne)

- **What**: The Daily's prefill on Turing: `STRATA_SELECT_SIMT` on by default below sm_80 (FP32 tiled QSA block scores), dequant of the projections with 16-byte stores, and the hyper-connection projections plus ssm_alpha/beta on FP16 tensor cores (`hc16`; weights re-encoded once, activations in FP16: not bitwise with BF16, like the Daily).
- **Switch (default)**: `STRATA_SELECT_SIMT=0`, `STRATA_TC_SM75=0` turn them off (sm_75 only)
- **Origin**: fork (SIMT scores from PR #1091 konijiwa110, merged upstream opt-in)
- **Upstream**: SIMT: upstream merged opt-in; 16-byte dequant and hc16: never proposed (upstream's `STRATA_BF16_TC` stays off on sm_75)
- **Measured**: 131k prefill 100 -> 75 s on the 2080 Ti stage (with patch 13: 75.4 s against the Daily's 74.9). In the old Daily hc16 removed 19.5% of the prompt's GPU time (BF16 GEMM in FFMA below sm_80).
- **Upstream defaults it changes**: SELECT_SIMT default on below sm_80
- **Docs**: docs/UPSYNC.md
- **Files**: `include/strata/core/tc_sm75.hpp`, `include/strata/kernels/qsa_select.hpp`, `include/strata/prefill/kernels.hpp`, `src/kernels/cuda/dequant_bf16.cu`, `src/kernels/cuda/qsa_select.cu`, `src/prefill/kernels.cu`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`, `sycl/src/prefill/kernels.dp.cpp`

Insertion points (44):

- `include/strata/core/tc_sm75.hpp` — (new file) → [14-1](BLOCKS.md#14-1)
- `include/strata/kernels/qsa_select.hpp` · `void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx` → [14-2](BLOCKS.md#14-2)
  ```
  /// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
  /// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
  /// On AMD it is the gfx12 (RDNA4) WMMA kernel (a three-way bf16 split, six products); false on any other AMD target.
  ```
- `include/strata/prefill/kernels.hpp` · `namespace strata::prefill {` → [14-3](BLOCKS.md#14-3)
  ```
  // ---- hyper-connection (n_embd 2560, hc 4, hc_lr 320)
  /// xn[t, c*2560 + d] = R[t,c,d] * rsqrt(mean_d R[t,c,:]^2 + eps) * w_norm[c*2560 + d]; also its BF16 image.
  /// `xn16_lo` (null: none) takes bf16(x - xn16): W.xn16 + W.xn16_lo is the product with ~16 mantissa bits of x.
  ```
- `include/strata/prefill/kernels.hpp` · `bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const` → [14-4](BLOCKS.md#14-4)
  ```
                float* mixed, uint16_t* mixed16, uint16_t* mixed_h, int64_t T, void* stream);
  void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                        float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo = nullptr,
  ```
- `include/strata/prefill/kernels.hpp` · `void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const in` → [14-5](BLOCKS.md#14-5)
  ```
  /// fp32 -> fp16 bits and fp32 -> bf16, n elements (the two activation images of the prompt GEMMs).
  void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
  void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo = nullptr);
  ```
- `src/kernels/cuda/dequant_bf16.cu` · `(file scope)` → [14-6](BLOCKS.md#14-6)
  ```
  // Arithmetic transcribed from ggml/src/ggml-quants.c at the pinned llama.cpp (MIT License, Copyright (c) 2023-2026
  // The ggml authors): dequantize_row_q2_0/q4_0/q5_0/q8_0/q3_K/q4_K/q5_K/q6_K/iq4_nl/iq4_xs.
  #include "strata/kernels/dequant_bf16.hpp"
  ```
- `src/kernels/cuda/dequant_bf16.cu` · `__device__ __forceinline__ void group32(const uint8_t* row_blocks, int gi_in_row` *(anchor ambiguous in the base file)* → [14-7](BLOCKS.md#14-7)
  ```
      }
  }
  ```
- `src/kernels/cuda/dequant_bf16.cu` · `void launch(int type, const void* blocks, int64_t row0, int64_t rows, int64_t co` → [14-8](BLOCKS.md#14-8)
  ```
          if (e != cudaSuccess) { std::fprintf(stderr, "dequant launch: %s\n", cudaGetErrorString(e)); std::exit(1); }
          return;
      }
  ```
- `src/kernels/cuda/qsa_select.cu` · `bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_` → [14-9](BLOCKS.md#14-9)
  ```
              cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
          }
          if (cc_major[dev] < 8) {
  ```
- `src/prefill/kernels.cu` · `unsigned blocks_for(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t` → [14-10](BLOCKS.md#14-10)
  ```
  
  // ---------------------------------------------------------------- hyper-connection
  __global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restrict__ w, float eps,
  ```
- `src/prefill/kernels.cu` · `__global__ void gr_norm_kernel(const float* __restrict__ R, const float* __restr` → [14-11](BLOCKS.md#14-11)
  ```
          const uint16_t h = act16(v);
          xn16[row * N + d] = h;
          if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
  ```
- `src/prefill/kernels.cu` · `__global__ void gr_norm_rs_kernel(const float* __restrict__ R, const float* __re` → [14-12](BLOCKS.md#14-12)
  ```
          const uint16_t h = act16(v);
          xn16[xo + d] = h;
          if (xn16_lo) xn16_lo[xo + d] = bf_lo(v, h);
  ```
- `src/prefill/kernels.cu` · `__global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict` → [14-13](BLOCKS.md#14-13)
  ```
                                                                 const float* __restrict__ inj, int64_t inj_ld,
                                                                 const float* __restrict__ w, float eps,
                                                                 float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
  ```
- `src/prefill/kernels.cu` · `__global__ void __launch_bounds__(256) gr_write_norm_rs_kernel(float* __restrict` → [14-14](BLOCKS.md#14-14)
  ```
          const uint16_t h = act16(x);
          xn16[xo + d] = h;
          if (xn16_lo) xn16_lo[xo + d] = bf_lo(x, h);
  ```
- `src/prefill/kernels.cu` · `__global__ void __launch_bounds__(256) gr_write_cvec_norm_rs_kernel(` → [14-15](BLOCKS.md#14-15)
  ```
      float* __restrict__ R, const float* __restrict__ bo, const float* __restrict__ inj, int64_t inj_ld,
      const float* __restrict__ v_l, const float* __restrict__ s_l, const int* __restrict__ on, int mode,
      const float* __restrict__ w, float eps, float* __restrict__ rs_out, uint16_t* __restrict__ xn16,
  ```
- `src/prefill/kernels.cu` · `__global__ void __launch_bounds__(256) gr_write_cvec_norm_rs_kernel(` → [14-16](BLOCKS.md#14-16)
  ```
          const uint16_t h = act16(y);
          xn16[t * ldx + (int64_t) c * N + d] = h;
          if (xn16_lo) xn16_lo[t * ldx + (int64_t) c * N + d] = bf_lo(y, h);
  ```
- `src/prefill/kernels.cu` · `__global__ void gr_silu_kernel(const float* __restrict__ lo, uint16_t* __restric` → [14-17](BLOCKS.md#14-17)
  ```
      const uint16_t h = act16(v);
      lo16[i] = h;
      if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
  ```
- `src/prefill/kernels.cu` · `__global__ void to_f16_kernel(const float* __restrict__ x, uint16_t* __restrict_` → [14-18](BLOCKS.md#14-18)
  ```
      for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x)
          y[i] = hf(x[i]);
  }
  ```
- `src/prefill/kernels.cu` · `void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo` → [14-19](BLOCKS.md#14-19)
  ```
      check("to_bf16");
  }
  ```
- `src/prefill/kernels.cu` · `bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const` → [14-20](BLOCKS.md#14-20)
  ```
  #endif
  }
  void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
  ```
- `src/prefill/prefill.cpp` · `(file scope)` → [14-21](BLOCKS.md#14-21)
  ```
  #include "strata/platform/aux_cpus.hpp"
  
  #include "strata/core/layout.hpp"
  ```
- `src/prefill/prefill.cpp` · `inline bool hcd_exact() {` → [14-22](BLOCKS.md#14-22)
  ```
      return v;
  }
  inline bool hc_pad() { return pf_hcdown() || hcd_exact(); }
  ```
- `src/prefill/prefill.cpp` · `struct Prefill::Impl {` → [14-23](BLOCKS.md#14-23)
  ```
      float *emb = nullptr, *R = nullptr, *xn = nullptr, *lo = nullptr, *gated = nullptr, *inj = nullptr;
      float* grs = nullptr;                    // F-1: the hyper-connection read's row scales (T x 4)
      uint16_t *xn16 = nullptr, *lo16 = nullptr;
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::carve(size_t T, void* alloc) {` → [14-24](BLOCKS.md#14-24)
  ```
      m.xn = gr_unfused() ? o.take<float>(T * D, ok) : nullptr;   // F-1: not needed (gr_mix_r reads R)
      m.grs = o.take<float>(T * HC, ok);
      m.xn16 = o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); m.lo = o.take<float>(T * LR, ok); m.lo16 = o.take<uint16_t>(T * LR, ok);
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::draft_kv(core::MtpDrafter& mtp, const float* R_rows, const int32_t` → [14-25](BLOCKS.md#14-25)
  ```
          proj(hn, hn16, w_fh, h2, nb * g.hc, Nn, Nn, 1);   // every stream through fc_hidden
          strata::kernels::add_streams_broadcast(h2, e2, Rm, Nn, (int) g.hc, (int) nb, m.cs);
          // the attention hyper-connection's read (its mixed input only: this pass writes nothing back)
  ```
- `src/prefill/prefill.cpp` · `uint64_t Prefill::bytes_needed_impl(const core::ModelGeometry& g, const core::Se` → [14-26](BLOCKS.md#14-26)
  ```
          f(T * N); f(T * D); f(T * D);
      }
      o.take<uint16_t>(T * (D + (hc_pad() ? XN_PAD : 0)), ok); f(T * LR); o.take<uint16_t>(T * LR, ok);
  ```
- `src/prefill/prefill.cpp` · `bool bf16_proj(Gemm& gm, const core::WeightRef* w, const uint16_t* X, float* Y,` → [14-27](BLOCKS.md#14-27)
  ```
      if (X_lo) gm.bf16(X_lo, (const uint16_t*) w->data, Y, T, w->ne1 > 0 ? w->ne1 : 1, w->ne0, ldy, 1.0f);
      return true;
  }
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [14-28](BLOCKS.md#14-28)
  ```
                                        *wu = need(v, su.c_str(), err), *wi = need(v, si.c_str(), err);
                  if (!wn || !wd || !wu || !wi) return false;
                  pt.mark(kPfHc, cs);
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [14-29](BLOCKS.md#14-29)
  ```
                      }
                      hcd = true;
                  }
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [14-30](BLOCKS.md#14-30)
  ```
                                             m.mixed, m.mixed_bf, m.mixed_h, T, m.cs);
                      }
                  }
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [14-31](BLOCKS.md#14-31)
  ```
                      float* conv = state + (uint64_t) g.ssm_state_size * g.ssm_v_heads * g.ssm_state_size;
                      if (!native_proj(m.gemm, wqkv, m.mixed_h, m.qkv, T, v.name("attn_qkv.weight"), err)) return false;
                      if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [14-32](BLOCKS.md#14-32)
  ```
                          // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                          const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                  strata::kernels::kStepNBid)] + 1;
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [14-33](BLOCKS.md#14-33)
  ```
                          gr_norm_rs(Rc, (const float*) wnn->data, EPS, rc, xc, T, m.cs, nullptr, D);
                          gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                                strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs,
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [14-34](BLOCKS.md#14-34)
  ```
                      } else
                      gr_write_cvec_norm_rs(m.R, m.bo, m.inj, HC, cv_dir + l * N, cv_s + l, cv_on,
                                            strata::kernels::cvec().mode, (const float*) wnn->data, EPS, m.grs, m.xn16,
  ```
- `src/program/generate.cpp` · `(file scope)` → [14-35](BLOCKS.md#14-35)
  ```
  #include "strata/core/progress.hpp"
  #include "strata/core/device.hpp"
  #include "strata/core/emulate.hpp"
  ```
- `src/program/generate.cpp` · `(file scope)` → [14-36](BLOCKS.md#14-36)
  ```
  #include <set>
  #include <vector>
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-37](BLOCKS.md#14-37)
  ```
          const int64_t pf = search && split_own_auto && !place_with_reserve ? 0 : split_pf_mib;
          const int64_t base_reserve =
              later && o.vram_reserve_later_mib >= 0 ? o.vram_reserve_later_mib : o.vram_reserve_mib;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-38](BLOCKS.md#14-38)
  ```
          if (stages.empty())
              for (const auto& d : slot_mtp)
                  mtp_bind += (int64_t) d->bind_bytes(native_head.row_bytes(), n_vocab);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-39](BLOCKS.md#14-39)
  ```
          size_t free_b = 0, total_b = 0;
          free_b = strata::core::device_free_bytes(); (void) total_b;
          const int64_t prefill_mib = owned_prefill_mib();
  ```
- `sycl/src/prefill/kernels.dp.cpp` · `void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo` → [14-40](BLOCKS.md#14-40)
  ```
  }
  
  void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
  ```
- `sycl/src/prefill/kernels.dp.cpp` · `void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t` → [14-41](BLOCKS.md#14-41)
  ```
      check("gr_norm");
  }
  void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
  ```
- `sycl/src/prefill/kernels.dp.cpp` · `bool gr_upmix(const uint16_t* lo16, const uint16_t* w_up, const float* R, const` → [14-42](BLOCKS.md#14-42)
  ```
  #endif
  }
  void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
  ```
- `sycl/src/prefill/kernels.dp.cpp` · `void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_l` → [14-43](BLOCKS.md#14-43)
  ```
  }
  void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* v_l,
                             const float* s_l, const int* on, int mode, const float* w_norm_next, float eps, float* rs,
  ```
- `sycl/src/prefill/kernels.dp.cpp` · `void gr_write_cvec_norm_rs(float* R, const float* bo, const float* inj, int64_t` → [14-44](BLOCKS.md#14-44)
  ```
      }
      check("gr_write_cvec_norm_rs");
  }
  ```

## 15 ram-core

`15-ram-core.patch` — FABLE-STRATA-Porta su 0.1.43 il nucleo fisso per scheda senza copia RAM (STRATA_RAM_CORE=1, STRATA_RAM_CORE_SLOTS_DEV<n>)

- **What**: A fixed core of expert slots per card, excluded from swaps, loans and shrink, whose RAM copy is given back (the resident arena no longer mirrors experts that never leave VRAM).
- **Switch (default)**: `STRATA_RAM_CORE=1` (off), `STRATA_RAM_CORE_SLOTS_DEV<n>=<slots>`
- **Origin**: fork
- **Upstream**: never proposed
- **Measured**: `STRATA_RAM_CORE_SLOTS_DEV1=5500`: arena 39.97 -> 29.62 GiB (-10.35), working set 48.7 -> 33.6 GB, t/s +0.1%, hit rate unchanged, 0 re-reads from file. Daily default since 12 Oct.
- **Notes**: Adjacency conflict in generate.cpp with 0.1.43's `STRATA_ADAPT_OVERLAP` ordering of the swaps: both kept.
- **Docs**: docs/RAM_CORE.md
- **Files**: `docs/RAM_CORE.md`, `include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp`, `src/program/generate.cpp`

Insertion points (24):

- `docs/RAM_CORE.md` — (new file) → [15-1](BLOCKS.md#15-1)
- `include/strata/core/expert_source.hpp` · `private:` → [15-2](BLOCKS.md#15-2)
  ```
  // Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
  class ArenaExpertSource : public ExpertSource {
  public:
  ```
- `include/strata/core/expert_source.hpp` · `public:` → [15-3](BLOCKS.md#15-3)
  ```
      double load_read_seconds() const { return load_read_s_; }
      double load_copy_seconds() const { return load_copy_s_; }
  ```
- `src/core/expert_source.cpp` · `LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata:` → [15-4](BLOCKS.md#15-4)
  ```
      return st;
  }
  ```
- `src/core/expert_source.cpp` · `bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int6` → [15-5](BLOCKS.md#15-5)
  ```
      arena_ = a;
      base_ = a->data();
      pinned_bytes_ = a->registered_bytes;
  ```
- `src/core/expert_source.cpp` · `void ArenaExpertSource::close() {` → [15-6](BLOCKS.md#15-6)
  ```
      map_ = nullptr;
      map_bytes_ = 0;
      inputs_.clear();
  ```
- `src/core/expert_source.cpp` · `void ArenaExpertSource::close() {` → [15-7](BLOCKS.md#15-7)
  ```
      base_ = nullptr;
      blobs_ = 0;
      n_expert_ = 0;
  ```
- `src/core/expert_source.cpp` · `const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {` → [15-8](BLOCKS.md#15-8)
  ```
      const int64_t idx = layer * n_expert_ + expert;
      if (idx < 0 || idx >= blobs_) return nullptr;
      ++reads_;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-9](BLOCKS.md#15-9)
  ```
      // fork (Daily 12a6f553): beside the pipeline the prompt path may borrow the core too - nothing decodes while a prompt
      // is read, and the lent slots are refilled before the decode as the tail's are (with only the 1.5 GiB tail lent the
      // 131k prompt read fell from ~85 s to 134 s)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-10](BLOCKS.md#15-10)
  ```
              int32_t first_now = -1;        // where its buffers are laid out now
              int64_t lent_chunk = 0;
              std::vector<std::pair<int32_t, int32_t>> lent;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-11](BLOCKS.md#15-11)
  ```
          // keeps the pinned-share rule.  STRATA_SPLIT_RING=N: N slots on a split; 0: the pinned-share rule.
          // (decided above, before the lend regions were sized: a no-op here unless the resident RAM mode skipped it)
          decide_split_ring();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-12](BLOCKS.md#15-12)
  ```
              pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
              for (auto& st : stages)
                  pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-13](BLOCKS.md#15-13)
  ```
              auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                  const int64_t k = part_slots(p, c);
                  if (p.cache == &xcache && k > xcache.slots() - el_lend_floor) return false;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-14](BLOCKS.md#15-14)
  ```
                                                                    kAutoLendPct * p.cache->slots() / 100));
                      // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
                      if (p.cache == &xcache) budget = std::min(budget, std::max<int64_t>(0, xcache.slots() - el_lend_floor));
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-15](BLOCKS.md#15-15)
  ```
          // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
          // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
          // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-16](BLOCKS.md#15-16)
  ```
                      if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
                      // --elastic: the core never leaves - unless core_adapt; beside the pipeline el_core counts CUDA0's
                      // slots and a later stage's layers hold slots of their own cache
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-17](BLOCKS.md#15-17)
  ```
                  agate.has.assign((size_t) g.n_layers, 0);
                  for (const Swap& s : swaps) agate.has[(size_t) s.layer] = 1;
              }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-18](BLOCKS.md#15-18)
  ```
          static const bool el_trace = std::getenv("STRATA_ELASTIC_TRACE") != nullptr;
          const int64_t el_step_max = 512ll << 20;   // at most this much per growth step (its copies are async)
          const int64_t el_shrink_step = std::max<int64_t>(64ll << 20, (int64_t) o.elastic_shrink_step_mib << 20);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-19](BLOCKS.md#15-19)
  ```
              int64_t c = el_chunk_max;
              const int64_t k = plan_lend(c);
              if (k <= 0 || c > el_chunk_max) return;   // nothing fits (not above el_min_slots): keep the plan
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-20](BLOCKS.md#15-20)
  ```
              for (int64_t l = 0; l < g.n_layers; ++l)
                  for (int32_t ex = 0; ex < (int32_t) g.n_expert && el_layer(l); ++ex) {
                      const size_t i = (size_t) (l * g.n_expert + ex);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-21](BLOCKS.md#15-21)
  ```
                  const int32_t* r = a_res.data() + l * g.n_expert;
                  for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                      if (r[e] < 0) { if (u[e] >= 2.0f && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-22](BLOCKS.md#15-22)
  ```
              std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
              std::fflush(stderr);
          };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-23](BLOCKS.md#15-23)
  ```
              const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
              const int64_t before = xcache.slots();
              // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-24](BLOCKS.md#15-24)
  ```
                          (double) (src.file_read_bytes() - file_bytes0) / 1e6, (long long) read_n,
                          (long long) req_offload, chain_txt);
              std::fflush(stdout);
  ```

## 16 qsa-rows

`16-qsa-rows.patch` — FABLE-STRATA-Porta su 0.1.43 STRATA_QSA_ROWS=1: punteggi dell'indicizzatore QSA con le query in registri e le 4 teste ridotte in 6 shuffle (stessi bit) e qsa_rows_bench

- **What**: QSA decode block scores with the queries in registers and the four heads reduced in six shuffles (T <= 4), bitwise; `qsa_rows_bench` (cost per window row of the QSA decode kernels).
- **Switch (default)**: `STRATA_QSA_ROWS=1` (off)
- **Origin**: fork
- **Upstream**: PR branch prepared (`pr/qsa-rows-scores`, one commit on v0.1.43); not yet opened
- **Measured**: Kernel 101 -> 55 us a call at T=4 on the 2080 Ti; engine B-A-B +3.2 / +0.1% (about +1-1.5%). Daily default since 12 Oct.
- **Docs**: docs/QSAROWS.md
- **Files**: `CMakeLists.txt`, `docs/QSAROWS.md`, `docs/qsa_rows_bench_2026-10-12.txt`, `docs/qsa_rows_bench_rows_2026-10-12.txt`, `include/strata/kernels/qsa_select.hpp`, `src/kernels/cuda/qsa_select.cu`, `src/kernels/qsa_rows_bench.cpp`

Insertion points (9):

- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [16-1](BLOCKS.md#16-1)
  ```
      set_tests_properties(qsa_merge_daily_parity_daily PROPERTIES ENVIRONMENT "STRATA_ATTN_DAILY=1"
                           SKIP_RETURN_CODE 77)
    endif()
  ```
- `docs/QSAROWS.md` — (new file) → [16-2](BLOCKS.md#16-2)
- `docs/qsa_rows_bench_2026-10-12.txt` — (new file) → [16-3](BLOCKS.md#16-3)
- `docs/qsa_rows_bench_rows_2026-10-12.txt` — (new file) → [16-4](BLOCKS.md#16-4)
- `include/strata/kernels/qsa_select.hpp` · `void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx` → [16-5](BLOCKS.md#16-5)
  ```
                        int64_t active_blocks = -1);   ///< perf-review C-1: > 0 launches only this many blocks (the
                                                       ///< batch's largest n_bid + 1; not for a captured graph)
  ```
- `src/kernels/cuda/qsa_select.cu` · `__global__ void __launch_bounds__(SCORE_WARPS * 32) block_scores_multi_kernel(co` → [16-6](BLOCKS.md#16-6)
  ```
  #if !defined(__HIPCC__)
  // ---- the decode top-k on a thread-block cluster (sm_90+; S19).  One CTA per query (block_topk_reg_kernel, or
  // block_topk_kernel above 33,792 blocks: a --max-context over ~135K) makes four radix passes and two scans over up to
  ```
- `src/kernels/cuda/qsa_select.cu` · `void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx` → [16-7](BLOCKS.md#16-7)
  ```
      // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
      static const bool multi = [] { const char* v = std::getenv("STRATA_SCORES_MULTI"); return v == nullptr || std::atoi(v) != 0; }();
      if (multi && nq <= MQ && active_blocks <= 0) {   // no active count: decode (captured or not) and prefill's pooled16
  ```
- `src/kernels/cuda/qsa_select.cu` · `void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx` → [16-8](BLOCKS.md#16-8)
  ```
      if (e != cudaSuccess) { std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e)); std::exit(1); }
  }
  ```
- `src/kernels/qsa_rows_bench.cpp` — (new file) → [16-9](BLOCKS.md#16-9)

## 17 expdec

`17-expdec.patch` — FABLE-STRATA-Porta su 0.1.43 EXPDEC: attivazioni del gruppo in shared e R righe per lane per gli esperti VRAM (STRATA_EXPDEC=1, down dietro STRATA_EXPDEC_DOWN=1) e expdec_bench

- **What**: Routed experts' gate/up in the verify window with the group's activations in shared memory and R rows per lane (the down behind `STRATA_EXPDEC_DOWN=1`), bitwise; `expdec_bench`.
- **Switch (default)**: `STRATA_EXPDEC=1` (off); `STRATA_EXPDEC_DOWN`, `_R`, `_DR`, `_MINB`, `_D20`, `_RB`, `_DRB` knobs
- **Origin**: fork
- **Upstream**: PR branch prepared (`pr/verify-expdec`, one commit on v0.1.43); not yet opened
- **Measured**: gate/up kernel -11/-23% at T 1-4 on the 2080 Ti; engine B-A-B +1.1% (B +0.0/+2.1). Daily default since 12 Oct. The Q2_0 down with EXPDEC was ~10% slower in the engine: off by default.
- **Docs**: docs/EXPERTDEC.md
- **Files**: `CMakeLists.txt`, `docs/EXPERTDEC.md`, `include/strata/kernels/iq_kernels.hpp`, `src/kernels/cuda/iq_kernels.cu`, `src/kernels/expdec_bench.cpp`

Insertion points (16):

- `CMakeLists.txt` · `if(STRATA_NATIVE_EXPERTS)` → [17-1](BLOCKS.md#17-1)
  ```
      add_executable(native_expert_bench src/kernels/native_expert_bench.cpp)
      target_include_directories(native_expert_bench PRIVATE ${_strata_gpu_include_directories})
      target_link_libraries(native_expert_bench PRIVATE strata_kernels strata_kernels_cpu strata_artifact ${_strata_gpu_runtime_target})
  ```
- `docs/EXPERTDEC.md` — (new file) → [17-2](BLOCKS.md#17-2)
- `include/strata/kernels/iq_kernels.hpp` · `void native_grouped_set_v1(bool v1);` → [17-3](BLOCKS.md#17-3)
  ```
  /// The bench only: the AMD kernel layout (STRATA_EXP_MODE values; -1 = the environment's) and the phase
  /// (0 all, 1 gate/up + SwiGLU + quantize, 2 down).
  void native_expert_set_mode(int mode, int phase);
  ```
- `src/kernels/cuda/iq_kernels.cu` · `template<> struct Split<16> {   // IQ2_XXS` → [17-4](BLOCKS.md#17-4)
  ```
  };
  // IQ2_XS and IQ2_S share the apply: two half sums, two 4-bit scales
  struct SplitLs2 {
  ```
- `src/kernels/cuda/iq_kernels.cu` · `struct SplitLs2 {` → [17-5](BLOCKS.md#17-5)
  ```
  };
  template<> inline constexpr bool kSplit<17> = true;
  template<> struct Split<17> : SplitLs2 {   // IQ2_XS
  ```
- `src/kernels/cuda/iq_kernels.cu` · `template<> inline constexpr bool kSplit<18> = true;` → [17-6](BLOCKS.md#17-6)
  ```
  template<> struct Split<18> {   // IQ3_XXS
      struct W { int g[8]; int ls; float dw; };
      template<bool STAGE_GRID = false>
  ```
- `src/kernels/cuda/iq_kernels.cu` · `template<> struct Split<18> {   // IQ3_XXS` *(anchor ambiguous in the base file)* → [17-7](BLOCKS.md#17-7)
  ```
          const float d = r.dw * __low2float(bq8_1[iqs / 2].ds);
          return d * sumi;
      }
  ```
- `src/kernels/cuda/iq_kernels.cu` · `template<> struct Split<21> {   // IQ3_S` → [17-8](BLOCKS.md#17-8)
  ```
  };
  template<> inline constexpr bool kSplit<29> = true;
  template<> struct Split<29> {   // IQ1_M
  ```
- `src/kernels/cuda/iq_kernels.cu` · `template<> struct Split<20> {   // IQ4_NL` → [17-9](BLOCKS.md#17-9)
  ```
          const float d = r.dw * __low2float(bq8_1->ds);
          return d * sumi;
      }
  ```
- `src/kernels/cuda/iq_kernels.cu` · `template<> struct Split<42> {   // Q2_0` → [17-10](BLOCKS.md#17-10)
  ```
          const float d8 = __low2float(bq8_1_chunk->ds);
          return r.d2 * d8 * sumi;
      }
  ```
- `src/kernels/cuda/iq_kernels.cu` · `__global__ void __launch_bounds__(256) native_down_multi_kernel(const unsigned l` → [17-11](BLOCKS.md#17-11)
  ```
  // ---------------------------------------------------------------- q8_1 (quantize.cu)
  // value i of a q8_1 row set (the warp holds block i / 32, lane = i % 32)
  __device__ __forceinline__ void q8_1_store(const float xi, block_q8_1* __restrict__ y, const long long i) {
  ```
- `src/kernels/cuda/iq_kernels.cu` · `void s27_launch_ts(bool ts, const NativeExpertLayout& L, int64_t cap_groups, cud` → [17-12](BLOCKS.md#17-12)
  ```
      else s27_launch<TG, TD, true, 4, 4, false>(L, cap_groups, s, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, X, gate, up, h, hq, out, nh);
  }
  ```
- `src/kernels/cuda/iq_kernels.cu` · `bool g_grouped_v1 = env_on("STRATA_GROUPED_V1");` → [17-13](BLOCKS.md#17-13)
  ```
  
  void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }
  ```
- `src/kernels/cuda/iq_kernels.cu` · `void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long` → [17-14](BLOCKS.md#17-14)
  ```
      } else
  #undef STRATA_GU_AMD
  #endif
  ```
- `src/kernels/cuda/iq_kernels.cu` · `void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long` → [17-15](BLOCKS.md#17-15)
  ```
      } else
  #undef STRATA_D_AMD
  #endif
  ```
- `src/kernels/expdec_bench.cpp` — (new file) → [17-16](BLOCKS.md#17-16)
