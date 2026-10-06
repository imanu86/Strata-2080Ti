# Evidence for `--elastic` (automatic elastic expert cache with a fixed core)

Evidence for the upstream pull request from branch `contrib/elastic-core-0139` of
https://github.com/imanu86/Strata-2080Ti. Machine: RTX 2080 Ti modified to 22 GB (PCIe 3.0 x16), Ryzen 7 5800X,
96 GB DDR4, Windows 11, CUDA 12.6 / MSVC 14.44, model Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (native pack).

## 2026-10-06: the port on upstream 0.1.39 (`6f32ec0`)

Folder `2026-10-06-upstream-0.1.39/` (`build_elastic_pr.py`, `validate_elastic_pr.sh`):

- **Build and tests**: pure upstream and the PR head in the same Release sm_75 configuration; CTest
  `expert_cache_elastic_test`, `expert_cache_per_layer_test`, `expert_profile_save_test`,
  `expert_cache_segmented_test`: 4/4 (`ctest.log`). `expert_cache_elastic_test` uses real CUDA VMM.
- **Off = upstream, bit for bit**: teacher forcing with a fixed expert placement (fixed cache of 5000 slots, adaptive
  swaps off, fixed PCIe share) on a code and an explain task: the per-position log-probabilities of the PR built
  without `--elastic` are byte-identical to pure upstream's (`tf_fixed_elasticpr.txt`, `runs/TFX-*`).
- **On, 131k context** (`runs/LE-*`, one run each, PLE table on SATA, the same machine idle):

  | | startup cache | L01 (131,248-token prompt) | L02 (131,072 reused + 167) |
  |---|---|---|---|
  | PR, `--elastic` | 6781 slots (core 3357 / tail 3424), grew +300 experts (510 MiB) while VRAM was free | prefill 153.9 s, 49.4 tok/s | 45.8 tok/s |
  | upstream, fixed cache | 7054 slots | prefill 153.1 s, 47.3 tok/s | 46.3 tok/s |

  On an idle machine the two are within run-to-run noise; the point of the elastic cache is the other case - other
  programs taking and returning VRAM while the server runs - where the fixed cache can only be sized for the worst
  moment (or, with `--vram-elastic`, resized by hand).

## 2026-10-03: the first port, on upstream 0.1.38 (`99f3dbd`)

Folder `2026-10-03-upstream-0.1.38/` (its own README): patch, scripts, CLI and mode checks, the isolated numeric gate,
operational dialogues at 131k and 250k, and the checkpoint-tail contribution that became PR #614.

## History

The cache has been the default configuration of the fork's daily build on this machine since late September 2026
(fixed core + rotating tail since `50a37fa`), through upstream 0.1.30 - 0.1.39.
