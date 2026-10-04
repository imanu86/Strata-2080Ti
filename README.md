# Strata-2080Ti

An optional `--prompt-cache-file PATH` now persists a fixed text system prefix
across engine restarts. On the tested 4003-token prefix, three Daily-profile
pairs reduce median native first-token latency from 5.000 to 1.344 seconds after
model loading. Fixed-placement full-logit/output parity and changed-prompt
invalidation pass. Automatic elastic placement remains numerically variable;
all seven factual answers pass. [Configuration, evidence and limits](docs/sm75/BENCHMARKS.md#persistent-system-prefix-4-october-2026).

A validated Windows snapshot of [Strata](https://github.com/Niko1221/Strata)
for **RTX 2080 Ti / Turing SM75**, running Qwen3.8-Flash-Next IQ3_XXS.

Main is aligned to upstream **0.1.38**, preserving the automatic elastic expert
cache and validated Turing ports. The original checkpoint grid has been removed;
the conservative tail checkpoint is optional in the engine and enabled by this
fork's profile generator. The pinned wide top-k from PR603 is included after
SM75 parity and model validation. Serial intermediate quantization remains the
profile default. Historical0.1.31 and0.1.36 evidence is preserved separately.
[Credits](CREDITS.md) explain which changes come from upstream and community forks.

Chat prompt tokenization now reuses exact shared prefixes from community
[PR567](https://github.com/Niko1221/Strata/pull/567). On the local 187-turn trace,
all 25,306,163 input token IDs match full encoding. Around 250k context the CPU
encoding component falls from 932 to 10.6 ms; GPU prefill/decode numbers below
are unchanged. [Evidence and limits](docs/sm75/BENCHMARKS.md#incremental-prompt-tokenization-3-october-2026).

`STRATA_PREFILL_EXACT_SMALL=1` reduces the cache space borrowed for small routed
prefill batches. It remains off by default in the engine and is enabled in the
local Daily after short paired tests and a completed delayed long control.
Fixed-cache warm 130k improves about 3.9%; warm 249k is unchanged. HTTP checks also
exposed short-response and variable-startup behavior, documented without a broad
quality or new SOTA claim in the [follow-up evidence](docs/sm75/BENCHMARKS.md#completed-controls-and-http-follow-up).

A default-off grouped-query QSA prefill path, `STRATA_QSA_PREFILL_MULTI=1`,
now targets SM75. With exact-small already enabled, controlled warm requests gain
5.38% at 130380 and6.26% at 249322 input tokens; captured logits and output IDs match
bitwise. A separate 190-turn elastic conversation reaches 250130 tokens. Its complete
[curve, startup anomalies and quality limits](docs/sm75/BENCHMARKS.md#grouped-query-qsa-prefill-4-october-2026)
are documented. The earlier snapshot throughput table below is retained as history;
these changes do not establish a new cold-prefill or sustained-decode SOTA.

The local Daily now uses executable SHA256 `f2e371bf…`, with persistent prefix
caching enabled. The optional decode PLE diagnostic is included and remains off.
Its earlier automatic-profile answer failure is preserved; the completed repeat
with matched placement now agrees bitwise with the old Daily and answers correctly.
See [prefix validation and deployment](docs/sm75/BENCHMARKS.md#persistent-system-prefix-4-october-2026)
and the [earlier diagnostic results](docs/sm75/BENCHMARKS.md#decode-ple-timing-and-pcie-screening-4-october-2026).
The current download,
[`v0.1.38-sm75-20261004-r1`](https://github.com/imanu86/Strata-2080Ti/releases/tag/v0.1.38-sm75-20261004-r1),
contains that exact tested Daily executable, including QSA prefill, the optional
diagnostic and persistent-prefix support. The previous October 3 release remains
available for rollback. [Build instructions and provenance](docs/sm75/BUILD.md). Full binary hashes,
the partial extended teacher check and bounded 250k recall results are in
[the benchmark record](docs/sm75/BENCHMARKS.md#actual-daily-full-history-recall-and-arithmetic).

Main also includes a **disabled neuron-trace diagnostic and offline replay tool**.
The October 4 pilot covers 192 accepted positions on four layers; naive reuse of
the previous neuron mask introduces substantial local error. It is research,
not an installed pruning optimization or a speed improvement. The Daily and
download above remain unchanged. [Measured results and limits](docs/sm75/BENCHMARKS.md#neuron-cache-pilot-4-october-2026)
and [diagnostic build instructions](docs/sm75/BUILD.md#optional-neuron-trace-research-overlay).

The follow-up [100-candidate probe](docs/sm75/BUILD.md#optional-100-candidate-experiment)
constructs temporary neuron masks from ten real MTP continuations of ten tokens
and measures their lifetime with adaptive rebuilding in offline GPU replay.
It remains a disabled research tool; the released engine and Daily are unchanged.

## 0.1.38 baseline validation

Six CTest checks,18 wide top-k cases,64 CPU quantization cases and118 server
tests pass. Paired model regression and real conversations at131k/250k pass;
see [current evidence and limits](docs/sm75/BENCHMARKS.md). KV capacity is262144.
These checks and the throughput samples below belong to the base snapshot;
later patches have their separately documented gates above.

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
     --mtp D:/models/mtp/rt `
     --prompt-cache-file ./cache/system-prefix.bin
   ```

   The examples are placeholders. This writes ignored `profiles/local-sm75.json`.
   Use `--help` to adjust capacity, resident KV, reserve and port. The current
   profile enables exact-small and grouped-query prefill; use
   `--no-prefill-exact-small` or `--no-qsa-prefill-multi` to disable either.
   Omit `--prompt-cache-file` to disable disk persistence. These settings target
   the new October 4 executable. The default 262k context / 32k resident KV and
   1393 MiB reserve reproduce the tested 22 GB profile; tune them for your PC.
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
