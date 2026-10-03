# Validated0.1.38 SM75 source, 3 October2026

Upstream main `99f3dbd0b21d1401b3769e0c0d963913607f380b`, automatic elastic expert cache, SM75 HC/GDN FP16 and vector
dequantization ports, serial intermediate quantization, conservative optional
`--prompt-cache-tail` ([PR614](https://github.com/Niko1221/Strata/pull/614)), and wide
top-k ([PR603](https://github.com/Niko1221/Strata/pull/603), exact pinned source).
The old checkpoint grid is removed, including its chunk-boundary overrides.
The old failed grid comparisons remain in the research record; this replacement
does not turn those failures into passes.

Full fresh CMake/Ninja Release build, CUDA12.6/MSVC14.44, architecture75.
Six CTest checks pass; native CPU quantization64 cases bitwise; wide top-k18 cases,
continuous scores, ties and equal scores, prompt/decode through524288 cells;
118 upstream server mock tests pass. A fixed-placement old036/new038/old036
model comparison has2560 teacher rows (2557 scored positions) per arm, finite
logits, six complete recall oracles, and all local numerical gates pass.
Separate Daily auto/elastic conversations at131072 and250000 input tokens pass
six oracles, with1024 generated tokens at each depth and a new marker afterwards.

Controlled PR603 comparison at250000 actual tokens: same038 source,5357 expert
slots,8192 prefill chunk, spec4, KV262144/int8/resident32768, resize disabled by
real long policy intervals. A/B/A repeated baseline; first logits are bitwise
identical and every complete recall/marker oracle passes. Detailed timings and
the distinction from the automatic Daily profile are in `validation.json`.

Only the modified22GB RTX2080Ti, Ryzen5800X/AVX2 and96GB RAM were exercised.
No artificial system saturation, general quality score, physical DRAM-bandwidth
measurement, stock11GB or multiGPU performance claim.
Executable SHA256: `014d7773886a45b10072cf362cf01082ec048a9063b318e3cd1ff024b3beaf71`. Previous manifests and validation remain intact.

| Actual fresh input | Fresh prefill t/s | Long decode t/s | Output tokens |
|---|---:|---:|---:|
| 131072 | 1027.01 | 36.913 | 1024 |
| 250000 | 833.50 | 39.065 | 1024 |

Single automatic-profile samples: initial6339 expert slots, runtime spec6,
MTPmax4/lookup3/PCIe0.36. They do not establish a causal gain against old036.
Controlled PR603250k prefill A/B/A: 701.49/832.03/699.49t/s,
+18.78% relative to the baseline mean, reference spread0.29%.
This fixed5357slot/8192chunk/spec4 profile has MTPmax0/lookup0/PCIe0.28;
keep it distinct from the automatic Daily profile above. Only one A/B/A.

## Historical records (their original versions and limits apply)

# Quality follow-up, 2 October 2026

[Native IQ quality regression](QUALITY-0136.md): A/B/A full logits are
byte-identical on 2,557 scored positions, and all three 131k retrievals pass.
The combined answer gate fails on one arithmetic question in every arm;
it also fails without drafts, and passes with reasoning enabled. The failed
verdict is retained. These are controlled quality runs, not a new SOTA.

# Main aligned to upstream 0.1.36, 2 October 2026

Current source base: `36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca`. The tested SM75 executable SHA256 is
`c35505ca79b790bd37ed7073fcadd399dd30d1b88e85bb99a6eb24a52d945d0a`. Windows, modified RTX2080Ti22GB, Ryzen5800X/AVX2, 96GB RAM,
CUDA12.6, IQ3_XXS, KV262144/int8/resident32768, reserve1393MiB, automatic elastic
cache, MTP. Two requests per arm: fresh131248 tokens, then131239 with131072
reused; 1024 output tokens each. No artificial saturation.

| Engine | Fresh 131k prefill t/s | Aggregate decode t/s | Initial elastic slots |
|---|---:|---:|---:|
| 0.1.31 reference | 974.05 | 41.255 | 6592 |
| 0.1.36 serial (daily) | 972.68 | 40.123 | 6786 |
| 0.1.36 parallel quant (opt-in) | 970.60 | 40.272 | 6034 |

These are single sequential samples with real desktop activity, changing cache
and MTP acceptance. They do not establish a causal gain from either upstream
changes or parallel quantization. The daily keeps serial quantization; the
parallel implementation is available for explicit controlled comparisons.
16 top-k cases pass bitwise; the unchanged pool port retains 64 real-GGUF
serial/parallel cases; the CMake-built HC test has zero failures. All100 upstream
server mock tests pass. General model quality and a new250k retrieval are not
covered by this comparison. Historical 0.1.31 evidence follows, including its
separate top-k A/B/A and250k retrieval; those measurements do not describe the
new executable.

# Measured daily:2 October2026

Modified RTX2080Ti22GB, Ryzen7 5800X,96GB RAM, Windows/CUDA12.6, PCIe3.0,
IQ3_XXS, KV262144/int8/resident32768, reserve1393MiB, automatic elastic cache,
automatic prefill, spec4/min-p0.5, MTP. Normal desktop activity remained;
no artificial saturation. JumpConnect sampled CPU was zero in retained runs.

## Same-executable top-k A/B/A

Prompts131248 and131239 per run; the second reused131072. Each generated1024
tokens. Only the dispatch override changed.

| Run | Dispatch | Fresh prefill tokens/s | TTFT s | Aggregate decode tokens/s | Initial slots |
|---|---|---:|---:|---:|---:|
|A1|capacity|831.20|158.509|38.251|6537|
|B|active|886.38|148.664|39.272|6524|
|A2|capacity|828.26|159.092|39.293|6872|

Mean reference829.73, candidate886.38: **+6.83%**, reference spread0.35%.
Mean TTFT158.801→148.664s:10.137s saved. Previous exact daily independently
measured827.53prefill/39.048decode. This is one retained A/B/A; elastic slots
and MTP output vary. B/A2 decode is virtually identical. See `validation.json`.

## Retrieval and deep decode

Needles at10/50/90% were all FOUND, real tokens30742/30742/30743. The tool's
"32k" label estimates characters; use these actual counts.

The exact250k fixture: **250022 tokens**, zero reused, needle at50%, correct
13-token answer,391472.7ms prefill (**638.670tokens/s**). That short answer is
not the decode benchmark.

The follow-up:250091 prompt,250015 reused, **1024 generated to finish length**,
251115 total. Decode26110.5ms = **39.217939tokens/s**;76 fresh prompt tokens in
1317ms. MTP583/944 drafts accepted, expert cache88.3%, KV blocks95.09% fromVRAM.

131k/250k use different prompts and acceptance, so these are observed context
points rather than a controlled sweep of one prompt. The previous250k result
442.4prefill came from another time window: the difference from638.7 cannot
be attributed entirely to the top-k fix, which affects only the initial135k.

## Numerical limits

- 16top-k cases:stride65538, batch1/8/255/256/257, partial tails, zero/tied scores,
  register limit33792/fallback33793, context up to262144. IDs bitwise equal.
- First131k logits:248320 finite values, same argmax1596. Even A/A model logits
  differ: this does not validate general model quality.
- Retrieval/deep-decode logit captures finite. HC/GDN FP16 changes precision;
  numerical checks and retrieval coverage are limited validation.
- Physical DRAM throughput not measured: Nsight Compute reported
  ERR_NVGPUCTRPERM. NVML memory activity is not measured GB/s.

## Reproduction

Original commands, fixture hashes, native logs and tools:
[full research record](https://github.com/imanu86/moe-aggressive-commit/blob/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo/RISULTATI_TOPK_2_OTTOBRE.md).

`tests/sm75/topk_active.cpp` is the deterministic native bench. Link against
the fork's kernels/core and CUDA runtime; pass context, query count, score mode:

After an explicit engine build, from an x64 VS2022 Developer PowerShell:

```powershell
cl /nologo /std:c++20 /EHsc /O2 /MD /Iinclude /I"$env:CUDA_PATH/include" `
  tests/sm75/topk_active.cpp /Fo:build-sm75/topk_active.obj /Fe:build-sm75/topk_active.exe `
  /link build-sm75/strata_kernels.lib build-sm75/strata_core.lib `
  "$env:CUDA_PATH/lib/x64/cudart.lib" "$env:CUDA_PATH/lib/x64/cudadevrt.lib"
```

Run it from `build-sm75/` with the CUDA runtime on PATH:

```text
topk_active.exe 131072 256 0
topk_active.exe 135164 256 0
topk_active.exe 135168 256 0
topk_active.exe 131072 256 1
```

Mode0=random,1=zero,2=ties,3=bound0,4=bound above stride. Numerical/model tests
consume GPU time and are explicit actions. No new GPU test was run for
publication. The previously installed exact daily passed startup in42.84s at
KV262144, then stopped. Local publication checks cover source/config only;
this fork has no automated workflow.


## Incremental prompt tokenization, 3 October 2026

Community [PR567](https://github.com/Niko1221/Strata/pull/567), pinned at
`f29e527856b85e37bda90626ccc28e002b4989dd`, reuses tokens up to a safe special-token
boundary in a shared rendered prompt. Only the remaining text runs through BPE.
The two production Python files and the upstream tests are ported unchanged.

Validated on Windows, Ryzen 7 5800X, 96 GiB RAM, with OktoCGControl open. CPU only,
no artificial load. The real IQ3_XXS pack vocabulary, merges and chat template
were used on all 187 prompts of the 20,001 -> 250,012-token conversation.
Full encoding from the installed pre-patch server was compared with the new
incremental encoder on every prompt: **25,306,163 input token IDs, all equal**.

| Actual context window | Turns | Full encode, median | Incremental, median |
|---|---:|---:|---:|
| 125k-135k | 8 | 434.1 ms | 7.77 ms |
| 240k-250,012 | 9 | 931.9 ms | 10.57 ms |

This measures the tokenization component before the native engine. It does not
measure end-to-end chat TTFT, change GPU prefill/decode, or establish a new SOTA.
There is one chronological replay; each turn runs full encoding then incremental
encoding in the same process. First prompts and unrelated prefixes still need
full encoding. Rendering and HTTP transfer costs are outside this table.

`python -B -m unittest tools.test_strata_tokenizer serve.test_server -v`:
132 tests run, 131 passed, one skipped (llama.cpp reference vectors unavailable).
`STRATA_TOKENIZER` points to the actual pack, so real-vocabulary prompt tests ran.
Coverage includes shared/edited/shortened prompts, tools, images, overlapping
special tokens, mock-engine HTTP requests, and tokenizers without resume points.

[Per-turn counts, hashes, timings and validation metadata](incremental-prompts-20261003.json)
retain the evidence without publishing the full local transcript. The native
Daily executable remains SHA-256
`13ee1a89be497ad83c67f62553b1133fec2a8d9309230d43a9dae639d89008a5`.
The historical source manifest stays intact; the installed frontend has a
separate overlay manifest and a backup of the replaced files.


## Request-sized routed buffers, 3 October 2026 (experimental)

`STRATA_PREFILL_EXACT_SMALL=1` keeps the buffer layout below the streaming
threshold when the actual prompt segment is below it. The default is **off**.
A 1007-token batch previously borrowed buffers for 1024 tokens, including the
large streaming ring, but still executed the routed path (threshold1024).
The opt-in uses a1007-token layout: **517 cache slots lent instead of1033**,
leaving516 more experts resident during that batch. KV capacity is unchanged.
The rounding of other requests remains unchanged; the helper is shared by the
native serve and CLI paths and observes `STRATA_PREFILL_STREAM_MIN`.

Windows, modified22GB2080Ti, Ryzen5800X,96GB RAM, CUDA12.6/SM75, IQ3_XXS,
OktoCGControl open, no artificial load. Same executable, profiling off, native
pipe with trace and first-logits dumps. Fixed6167 resident experts (10218MiB),
KV int8 capacity262144/resident32768, spec4,7 workers, adaptive swaps and PCIe
fraction off for the paired laboratory checks. This fixed cache is not a Daily
configuration change.

The standalone C++ boundary/property test passed519268 checks. Eight real-token
slices exercise actual batched sizes256,768,769,1007,1023,1024,1025,1536.
The short comparison completed in order off/on/off (A1/E/A2):

| Batched tokens | Base A1 / A2 prefill ms | Exact buffers ms | t/s change vs mean base ms |
|---|---:|---:|---:|
| 256 | 1279.2 / 1295.9 | 1281.7 | +0.46% |
| 768 | 2006.4 / 1989.4 | 2026.0 | -1.39% |
| 769 | 2186.7 / 2201.7 | 2043.5 | +7.37% |
| 1007 | 2336.1 / 2324.9 | 2196.2 | +6.12% |
| 1023 | 2455.4 / 2448.5 | 2341.9 | +4.70% |
| 1024 | 2813.8 / 2821.5 | 2816.2 | +0.05% |
| 1025 | 2833.9 / 2848.5 | 2830.6 | +0.37% |
| 1536 | 2868.4 / 2924.6 | 2876.4 | +0.70% |

Times include the final input token (for example1008 input tokens for a1007
batch) and cache refill. These are short-context samples, not130k/250k chat
throughput or a new cold-prefill SOTA.

The completed long-context A1/E pair used real continuation fixtures:

| Total input | Fresh input | Base A1 ms | Exact ms | Interpretation |
|---|---:|---:|---:|---|
|130380|1014|3873.1|3713.9|Single pair, performance not confirmed|
|249322|1014|4357.9|4363.1|No observed gain|

Both final long control attempts became unusually slow during cold prefill and
were stopped before completing a request. The original result files remain
failed; their timings are excluded. A later smoke using the actual elastic
Daily configuration also slowed in its first cold chunk and was stopped.
Startup had6426 slots and1001MiB free; loading39.97GiB took about146s overall
(arena0.37GiB/s), compared with about32s overall in faster starts. One snapshot
of the first slow control found620MiB VRAM free and normal GPU clocks. Cause
is **unresolved**; this is not evidence that OktoCGControl or the patch caused it.
The operational smoke did not pass. **Native Daily and its configuration remain
unchanged; this feature is not promoted.**

Numerical evidence is separate from those inconclusive long timings. All248320
first-target logits match bit for bit for each of the8 short cases and4 completed
long requests (two seeds and two continuations). The generated greedy token IDs
also match:64 short and66 long,130 total per A1/E comparison. Maximum absolute
logit difference and KL are both zero. This checks allocation/copy equivalence
on these fixtures; it is not a broad model quality/PPL evaluation. Other GPUs,
stock11GB cards, and multi-GPU operation have not been validated.

An exploratory8-slot grouped-gather variant also matched numerically but showed
no gain in the usable short pairs; its final baseline was anomalously slow.
That variant was removed before the final build. It is not shipped here.

Reproduction: build `prefill_request_chunk_test` with `STRATA_BUILD_TESTS=ON`
and run `ctest -R prefill_request_chunk_test`, or compile the standalone test
with the repository include directory. For model comparisons, restart the same
binary with `STRATA_PREFILL_EXACT_SMALL=0` and1, keep inputs/residency/settings
fixed, and repeat the reference. `STRATA_DUMP_FIRST_LOGITS=<absolute-prefix>`
now works for native `--serve`: each request writes the first target row to
`<prefix>.pos<last-input-position>.f32` (float32). The existing CLI dump format
is unchanged. Dumps are opt-in and no raw transcript or weights are published.
[Hashes, per-case timings, parity and excluded runs](request-chunk-20261003.json)
retain the measured evidence. Resolve the long-run instability and repeat the
actual-config smoke before enabling the flag in the operational profile.
