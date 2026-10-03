# Strata-2080Ti

A validated Windows snapshot of [Strata](https://github.com/Niko1221/Strata)
for **RTX 2080 Ti / Turing SM75**, running Qwen3.8-Flash-Next IQ3_XXS.

Main is aligned to upstream **0.1.38**, preserving the automatic elastic expert
cache and validated Turing ports. The original checkpoint grid has been removed;
the conservative tail checkpoint is optional in the engine and enabled by this
fork's profile generator. The pinned wide top-k from PR603 is included after
SM75 parity and model validation. Serial intermediate quantization remains the
profile default. Historical0.1.31 and0.1.36 evidence is preserved separately.
[Credits](CREDITS.md) explain which changes come from upstream and community forks.

## Current0.1.38 validation

Six CTest checks,18 wide top-k cases,64 CPU quantization cases and118 server
tests pass. Paired model regression and real conversations at131k/250k pass;
see [current evidence and limits](docs/sm75/BENCHMARKS.md). KV capacity is262144.

| Actual fresh input | Fresh prefill t/s | Long decode t/s | Output tokens |
|---|---:|---:|---:|
| 131072 | 1027.01 | 36.913 | 1024 |
| 250000 | 833.50 | 39.065 | 1024 |

Single automatic-profile samples; cache/MTP vary. The separately controlled
PR603250k A/B/A and numerical parity are documented in the evidence above.

## Historical0.1.36 comparison

| Engine | Fresh 131k prefill t/s | Aggregate decode t/s | Initial elastic slots |
|---|---:|---:|---:|
| 0.1.31 reference | 974.05 | 41.255 | 6592 |
| 0.1.36 serial (daily) | 972.68 | 40.123 | 6786 |
| 0.1.36 parallel quant (opt-in) | 970.60 | 40.272 | 6034 |

Single samples with real desktop activity, elastic cache and varying MTP; no causal
speedup is claimed. See [current validation](docs/sm75/BENCHMARKS.md).

## Historical 0.1.31 performance

Windows, **modified RTX 2080 Ti 22 GB**, Ryzen 7 5800X, 96 GB RAM, PCIe 3.0,
CUDA 12.6, IQ3_XXS, MTP, int8 KV capacity **262144**, 32768 resident KV cells,
automatic elastic expert cache. A stock 11 GB 2080 Ti needs separate tuning and
validation; these measurements describe the 22 GB card.

| Actual prompt depth | Fresh prefill | Decode | Decode sample |
|---|---:|---:|---:|
|131248 / 131239 tokens|886.38 tokens/s on the first request|39.272 tokens/s aggregate|2 x 1024 tokens|
|250022, then 250091 tokens|638.67 tokens/s on the first request|39.218 tokens/s on the cached follow-up|1024 tokens|

The top-k fix improved fresh 131k prefill by **6.83%** in a same-executable
capacity/active/capacity A/B/A: **831.20 / 886.38 / 828.26 tokens/s**.
Reference spread 0.35%; no decode gain is attributed to this prefill change.
Retrieval passed at three insertion depths and at 250022 real tokens.
[Benchmarks and limitations](docs/sm75/BENCHMARKS.md) describe the settings,
numerical checks, cache variation and model quality coverage.

## Windows quick start

1. Clone this repository and create a Python 3.12 environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Download this fork's [release engine](https://github.com/imanu86/Strata-2080Ti/releases/latest),
   verify SHA256, and extract it into `engine/`. The measured executable requires
   **CUDA 12.6 runtime/cuBLAS on PATH and Microsoft Visual C++ x64 runtime**.
   It was tested on Ryzen 5800X/AVX2. Alternatively, [build these sources](docs/sm75/BUILD.md).

3. Supply your own existing pack, native GGUF pair and MTP directory:

   ```powershell
   .\.venv\Scripts\python.exe tools/sm75/configure.py `
     --pack D:/models/iq3_xxs `
     --native D:/models/model-00001-of-00002.gguf `
     --ple-gguf D:/models/model-00002-of-00002.gguf `
     --mtp D:/models/mtp/rt
   ```

   The examples are placeholders. This writes ignored `profiles/local-sm75.json`.
   Use `--help` to adjust capacity, resident KV, reserve and port.
   Model packs are supplied separately; [upstream](https://github.com/Niko1221/Strata)
   explains model/pack selection.

4. Run `START-HERE.bat`. It starts this checkout's server and your selected engine
   at `http://127.0.0.1:8100`, with an OpenAI-compatible API. Set `STRATA_API_KEY`
   in your environment for authentication. Credentials are never copied into
   generated JSON. Close the server window to stop it.

This launcher uses an engine you installed or built. The inherited `setup.py`
is preserved upstream source; its download flow obtains upstream engines.
The fork-specific steps above explicitly select this fork's validated engine.

## Stable contents

- Elastic expert cache with a fixed core and rotating tail.
- Validated HC/GDN FP16 and vector dequantization ports from the
  [2080Ti community fork](https://github.com/jklnvlink/Strata-2080ti).
- PR315 HC read with its existing numerical gate.
- Top-k prefill selection using the actual active block bound while preserving
  the KV-sized score stride. The wide fallback from PR603 also accelerates long selections and decode;
  selected IDs pass parity against the upstream reference.
- Opt-in first-logit diagnostics and numerical/benchmark evidence.

CH16, score BPW and PR439 gather experiments are **not enabled** in this release:
model gains were not demonstrated or model validation is pending.
See the [research record](https://github.com/imanu86/moe-aggressive-commit/tree/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo).

The current0.1.38 engine was built and validated for this integration.
Current and historical source manifests/validation remain separate.
Fresh builds may have different executable hashes because toolchains, dependencies
and linking vary. General model quality and physical DRAM bandwidth were not measured.

MIT license and upstream/contributor notices preserved: [LICENSE](LICENSE).

Model validation method and measured limits: [protocol](docs/sm75/VALIDATION.md), [0.1.36 quality results](docs/sm75/QUALITY-0136.md).
