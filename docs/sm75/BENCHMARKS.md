# MTP expert precision (laboratory only, 2026-10-07)

The opt-in `STRATA_LAB_MTP_HQ_SWITCH` and `STRATA_LAB_MTP_HQ_PACK` compare
the existing Q2 expert kernel, a lossless Q2 repack using the native kernel,
and Q4 experts quantized from the original local BF16 tensors. All three
keep the same 1,350 MiB expert allocation and pre-captured graph families;
weights change only at an idle GEN boundary, before prefix restore and decode.
The native paths also change activation quantization, so Q2-native is a
separate control. The default engine path is unchanged.

On the Ryzen 5800X / 96 GB Windows workstation, stage 0 is the RTX 3060 12 GB
on Oculink x4 with an 8K display; stage 1 and MTP use the modified RTX 2080 Ti
22 GB on PCIe x16. At 131,248 / 131,239 input tokens, 1,024 fixed output IDs,
spec4/min-p0.5, int8 KV/resident32768, split16/pipeline2 and automatic elastic
expert cache, the final hot Q4/legacy/Q4 B-A-B was:

| Case | Q4 B1 | Q2 legacy A | Q4 B2 |
|---|---:|---:|---:|
| L01 | 72.7603 | 71.0100 | 72.7004 |
| L02 | 64.9598 | 65.0692 | 65.3040 |

Values are decode tokens/s with identical policy telemetry in every arm.
All measured GENs restore 131,072 tokens from private SSD files. This is a
fixed-continuation diagnostic, not free-generation quality or a deployment
speedup: L01 improves about 2.4%, L02 is neutral, and the common allocation
already charges the Q2 controls the extra 675 MiB. Upload time is outside
decode; cache resize activity remains in decode and is recorded without
subtraction. No Daily setting was promoted.

The 32-request integrity/cleanup audit passed. All 512 Q2 experts pass the
exporter's lossless round-trip; a separate decoder verifies byte/float-bit
parity and lower Q4 weight error on 12 complete matrices from four fixed
experts. These numerical checks do not prove draft quality.
See [the frozen source, build, protocol, controls and numerical evidence](mtp-hq-20261007.json).

# Causal window diagnostics, 7 October 2026

`STRATA_LAB_POLICY_TRACE` enables default-off fixed buffers for MTP chain IDs,
probabilities, host-observed readiness and verifier-window verdicts. Allocation
precedes decode; JSONL is flushed after its timer. Raw target choices remain
distinct from forced fixture IDs. Labels beyond the valid prefix or final output
budget are censored; overflow, nonfinite values and incomplete chains invalidate
the record. This is diagnostic timing, not GPU kernel duration or a speed claim.

Build `6d033548e5a917f3f42795dce4016909ce4d2281d410cc055c79b5a098fd8aaa`
passes the original sm75/sm86 build. Eighteen trace/parser tests and seventeen
runner tests pass. `DET-policy-trace-01` completes fourteen GEN: two cold prefix
warmups, then min-p0/0.5/0 hot BAB smoke and1024-token pairs. Every measured
request restores131072 SSD tokens. JSONL, actual GEN wire, forced-window trace,
IDs and counters agree; isolation passes and no owned process remains.

Instrumented long BAB L01:62.22/71.39/62.53 t/s; L02:53.47/64.40/54.00.
These rates are diagnostic, with all arms instrumented; they do not measure
instrumentation overhead. The workload remains forced and is not a free-greedy
quality test. The telemetry supplies observations for future policy decisions.

See [the frozen protocol, source/build and complete validation record](policy-trace-20261007.json).

# Hot MTP probability thresholds, 7 October 2026

The same immutable oracle-capable executable runs with the oracle OFF, real MTP
drafts and only target IDs forced. All26 requests restore the original131072-token
prefix from SSD, including warmups; each GEN wire write and flush proves the
requested threshold. One process, split16, pipeline2, S4, seed73, greedy sampling,
INT8 KV/resident32768, auto expert cache and elasticity; no forced windows.

| Threshold B / control A / repeat B | L01 decode t/s | L02 decode t/s |
|---|---:|---:|
| 0 /0.5 /0 | 61.50 /71.80 /62.33 | 53.02 /65.56 /53.98 |
| 0.2 /0.5 /0.2 | 64.82 /69.93 /65.35 | 57.25 /64.77 /56.38 |
| 0.8 /0.5 /0.8 | 70.05 /71.46 /69.88 | 62.09 /64.88 /62.50 |

Every long B arm is slower than its interposed control. Retain0.5 for this setup;
this does not establish a global optimum or rule out asymmetric/adaptive policies.
All1024-output comparisons keep the same allocation and SSD contents. Fourteen
CPU checker tests pass, including actual Daily serialization and mutations.
Nine cache growths/six shrinks and PLE warming remain observed confounders; no
timers are subtracted. The small0.8 gap is particularly exposed to that drift.
Forced continuation equality controls workload and does not establish answer quality.
Daily unchanged; no compiler interference or remaining owned process.

The first attempt stopped after two warmups because the checker misread the
legacy `synthetic=1` marker as oracle draft delivery. It remains failed; raw
mode=off, real chains, zero bypasses and zero delivered oracle IDs prove the
checker defect. A regression covers the repaired condition in the new frozen suite.
See [complete protocol, histograms, wire proof, events and hashes](hot-minp-20261007.json).

# Selective MTP history, 7 October 2026

A default-off prototype combines the last8192 MTP cells with positions selected
by the target's final QSA. It retrieves **the drafter's own K/V**, using a private
13,341,184-byte pool and precaptured graphs. Target K/V values are never reused.
The original32k dense path and the selective path keep identical allocations
throughout one16-request engine run. Every measured request restores131072
tokens from SSD; actual prompts131248/131239, long outputs1024, split16/S4,
pipeline2, KVint8/resident32768, suffix/lookup0 and forced target IDs only.

| Treatment B / control A / repeat B | L01 decode t/s | L02 decode t/s |
|---|---:|---:|
| Selective8192 / dense32768 / selective8192 | 70.61 / 71.84 / 71.39 | 64.45 / 64.27 / 64.76 |

There is no convincing gain to promote. All selective chains actually used the
new path: no dense fallback, one private-map reset per request and no overflow.
Six GPU gate/smoke requests compare all final-query selected cells with their
own host codes/scales, including remote history and resident cells overwritten
by catch-up; every check passes. Long timing disables that postdecode diagnostic.
Thirteen CPU contract-model tests and six evidence-checker tests pass; those
CPU models do not substitute for CUDA validation. Daily and its files are unchanged.

The method both adds remote history and shortens the recent window; this result
does not isolate the value of remote retrieval alone. Cache/elastic events are
retained, and forced output equality is workload control, not answer quality.
No free-generation or multi-chat claim. New counters separate A rejection,
wrong B bonus, B not ready at stage0 completion and useful B work.
See [source/build hashes, all measurements and limits](selective-mtp-20261007.json).

# Synthetic draft oracle, 7 October 2026

An opt-in laboratory oracle separates the value of perfect draft proposals
from their execution cost. This is **not real model throughput**. One process
ran 26 requests, including smoke and long B/A/B triplets; every measured request
restored 131072 tokens from an unchanged SSD prefix and read the remaining
176/167 prompt tokens. Actual prompts are 131248/131239 tokens, with 1024 output
tokens in each long case. Draft windows are free to change; only output IDs are
forced. Suffix and lookup speculation are disabled.

| Proposals / execution | L01 B1 / A / B2 t/s | L02 B1 / A / B2 t/s |
|---|---:|---:|
| Perfect proposals, real MTP work | 141.12 / 71.25 / 144.58 | 143.34 / 65.04 / 144.75 |
| Perfect proposals, no MTP decode work | 175.63 / 71.89 / 173.85 | 175.08 / 64.44 / 174.81 |

Perfect proposals require 257 verified windows for 1024 outputs, instead of
533/618 windows in the first control pair. All 767 offered drafts are accepted;
quality mode retains real chain execution and readiness, while instant mode
bypasses decode launches. Both preserve MTP initialization, prefill and VRAM.
Instant invalidates decoded live state before reuse. The result motivates work
on draft accuracy; it does not promise this speed for real generated answers.

Windows, Ryzen5800X/96GiB, RTX3060 12GB plus modified RTX2080Ti22GB,
split16/pipeline2/spec4, KV int8/resident32768/capacity262144. Auto/elastic cache
starts at4434/9151 slots; early warmup/smoke growth reaches4699 on CUDA0.
Eleven long requests report zero shrink/growth events; the final instant L02
reports one growth costing21.0ms. Cache contents can evolve.
Ten CPU hot-suite tests, nine initial parser tests, six negative CLI guards,
all26 output/counter checks and Daily/isolation checks pass. An earlier cold
instant attempt failed in nvcuda64.dll before READY and executed no request;
that failure remains in the record. No multi-chat or free-generation quality
claim. Defaults and Daily remain unchanged. The laboratory controls require
explicit compatible forcing and fail closed on invalid modes or fixtures.
See [source hashes, protocol, measurements and limits](oracle-20261007.json).

# MTP placement experiment, 7 October 2026

Main was integrated with upstream 0.1.40.3 (`d5ea713`) at `ccd4f3c`.
The opt-in `STRATA_MTP_DEVICE=0` relocates the drafter to the first GPU;
unset or `last` preserves the existing placement. The exercised setup is
Windows, RTX3060 12GB + modified RTX2080Ti 22GB, split16/pipeline2,
actual 131248/131239-token prompts and 1024 generated tokens per case.
The existing deterministic bench ran B1/A/B2 with one executable.

| MTP device | L01 decode t/s | L02 decode t/s |
|---|---:|---:|
| 3060, B1 | 63.85 | 57.53 |
| 2080 Ti, control A | 71.39 | 65.37 |
| 3060, B2 | 64.77 | 57.24 |

The relocated configuration regresses in both repetitions and is not promoted.
Forced IDs/windows match and the separate unforced conversation/checkpoint
restore gate passes, including draft-KV verification on CUDA0. Forced timing
is not a free-generation quality test. Full free 131k BAB and multi-chat
performance were not run after this negative single-chat result. Daily is unchanged.
The implementation remains experimental and disabled by default; CUDA two-stage
single-chat serve only, with a native draft-head subset and pinned-host residual
transport. See [the complete validation record](mtp-device-20261007.json) for
build/source hashes, capacities, controls, failed first gate and limitations.

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
slots,8192 prefill chunk, spec4, KV 262144/int8/resident 32768, resize disabled by
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
CUDA12.6, IQ3_XXS, KV 262144/int8/resident 32768, reserve1393MiB, automatic elastic
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
IQ3_XXS, KV 262144/int8/resident 32768, reserve1393MiB, automatic elastic cache,
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


### Single B repeat after the anomalous runs

At the user's request, only the candidate was repeated, once. It completed all
four requests in **372.2 s including startup**, without the earlier slowdown.
Executable, CLI arguments, input fixtures and expert-profile hashes match the
interrupted operational candidate. Automatic elastic cache started at6342
experts /10512MiB, with1137MiB free. KV remains262144/int8/resident32768.

| Actual context | Fresh input | Prefill | First token, native pipe |
|---|---:|---:|---:|
|130380|1014|279.6 tokens/s|3.702 s|
|249322|1014|248.5 tokens/s|4.193 s|

The initial129366-token cold prompt ran at937.1 tokens/s. The later preparation
of248308 total input reused130373 tokens, so its639.4 tokens/s over117935 new
tokens must not be called a cold248k result. Each continuation generated32 tokens
(25.4 and33.6 tokens/s); these short samples are not long-decode SOTA measures.
All four captured logit rows contained248320 finite values. This single B does
not add a paired numerical comparison or establish a causal long-context gain.

As in the previous candidate test, the harness uses `STRATA_IQ_MT_MIN=1` for
consistent CPU rounding across draft widths (the engine default is2), alongside
trace/first-logits diagnostics. Thus these are the Daily CLI/cache settings with
the stated benchmark overrides, not a claim of unmodified Daily decode speed.
In64 lightweight process/CPU/I/O samples there were no detected compiler
processes; median CPU use was22%, minimum available RAM22.9GiB. This documents
the repeat's conditions, not the cause of the earlier anomalies. Those records
are preserved. No additional A or B was run; the engine was closed afterward.
The installed Daily and default-off flag were left unchanged in this test-only
step. [Repeat evidence and hashes](request-chunk-B-repeat-20261003.json).


### Completed controls and HTTP follow-up

The delayed fixed-cache A2 completed in 372.8 s with 6167 slots (10218 MiB).
All four first-logit rows (248320 floats each) and 66 generated IDs match A1/B
bit for bit. A2 followed intervening runs; this is not an uninterrupted A/B/A.
Original stalled attempts remain recorded, with their cause unresolved.
The numerical bench forces IQ_MT_MIN=1, unlike the normal Daily environment.

| Actual input | Fresh tokens | A1 ms | B ms | Delayed A2 ms | B vs control mean |
|---|---:|---:|---:|---:|---:|
|130380|1014|3873.1|3713.9|3840.9|+3.85%|
|249322|1014|4357.9|4363.1|4359.2|-0.10%|

An elastic A also completed after B: observed warm gains 4.10%/3.00%, with
6342/6347 startup slots. These are not fixed-placement causal gains. At 249k,
first logits differ (same argmax), and generated IDs first diverge at token 28.
Do not describe this as elastic bitwise parity.

The local Daily enables exact-small; the engine default stays off. The original
manifest is preserved, with separate binary/config/source backups and overlay.
Actual-environment HTTP checks used no IQ_MT_MIN, trace, profiler or logits dump:

- With cap 128, both exact-small on and off complete 11 requests to about 30k.
  After the first two length-capped responses, later outputs increasingly end
  mid-sentence despite native `stop`. This behavior is not specific to the patch;
  its cause is not established by these different sampled conversation histories.
- With cap 512, nine responses of 184–259 tokens finish with complete sentences.
  The harness then fails to size the next prompt (target 30008, minimum 30176).
  This is a partial run to 29854 input tokens, not a completed 30k test or a broad
  answer-quality pass. The original failed result is retained.
- First prefill varied from 18.24 s to 214.68 s and 196.03 s across these HTTP runs.
  The slow runs are excluded from speed claims, and their cause is under review.

No new cold-prefill or sustained-decode SOTA is claimed.
[Hashes, measured rows and limitations](request-chunk-final-20261003.json).


## 2026-10-03: main-model BF16 tensor-core experiment rejected

A local adaptation of [rafatxf's PR655](https://github.com/Niko1221/Strata/pull/655)
converted the remaining main-model BF16 products through FP16 on SM75, preserving
the MTP draft path and reusing existing scratch. It passed the bounded kernel tests
but failed the predeclared paired model gate at 20,001, 130,380 and 249,000 tokens
(256 teacher-forced positions each). The off/on/off baseline repeat was bitwise
identical at all three depths. Top-1 agreement with the candidate was respectively
97.27%, 98.44% and 94.92%; these are agreement rates, not semantic accuracy scores.

The experimental source was removed before the next build and was never enabled
in the Daily. No end-to-end speed claim is made. This is a result for our local
adaptation, not a verdict on the unmodified upstream PR. Configuration, thresholds,
raw hashes and per-depth metrics are in [the rejection report](bf16-main-rejected-20261003.json).


## Grouped-query QSA prefill (4 October 2026)

This default-off `STRATA_QSA_PREFILL_MULTI=1` path reuses each FP32 key block for
up to eight prompt queries on SM75. It extends q8atnight's
[PR187](https://github.com/Niko1221/Strata/pull/187) to bounded prompt batches;
per-query arithmetic and the decode path are unchanged. The measured card is a
modified 22 GB RTX 2080 Ti, with Ryzen 7 5800X, 96 GB RAM, CUDA 12.6 and PCIe 3.0.
Stock 11 GB cards and multi-GPU operation are unvalidated.

The 12-case synthetic suite checks complete score/selection buffers: query counts
7/8/9/256/1007/8192, small/large/tail/cap shapes, ties, non-finite inputs, and the
unspecified active-bound fallback(-1). All comparisons are byte-identical.
A same-binary off/on/off model gate also passes: all four 248320-value first-logit
rows are finite and bitwise identical, and all 258 output IDs per arm match.
These are bounded regression checks, not a broad semantic-quality evaluation.

The model gate uses KV 262144/int8/resident32768, fixed 6167 expert slots/10218 MiB,
spec 4, suffix drafts off, exact-small on, and default IQ_MT_MIN 2 (no override).
First-logit dumping occurs after native prompt_ms; it affects decode/wall time.

| Actual context | Fresh / reused | A1 prompt ms | B prompt ms | A2 prompt ms | Throughput gain vs control mean |
|---|---:|---:|---:|---:|---:|
|130380|1014 /129366|3722.0|3520.2|3697.0|5.38%|
|249322|1014 /248308|4233.5|4000.3|4268.0|6.26%|

The gain is `(mean(A1 prompt_ms, A2 prompt_ms) / B prompt_ms - 1) * 100`.

The full-fresh 129366 request took 325.298/130.794/417.148 s in A1/B/A2.
The large control spread remains unexplained and cannot establish a causal cold
speedup. The 248308 request reuses 130373 tokens, so it is not a cold 248k measure.
Additional group/grid tuning passed parity but showed no consistent winner;
one anomalous 8x64 case was repeated once and remained slower. It is not shipped.

A separate real HTTP conversation completed 190 turns, from 20002 to 250130 input
tokens, in 2017.73 s including startup/cleanup. It uses the Daily automatic/elastic
configuration, KV 262144/int8/resident32768, CLI spec 4 with adaptive policy,
exact-small and QSA enabled, and no IQ override, trace, logits dump or profiler.
Each user follow-up targets 1000 tokens, with a shorter final turn. Temperature 0.6,
top_p 0.95/top_k20, thinking off, output cap 512. All requests report stop.
The first cold 20k prefill took 214.342 s (93.3 t/s). Turns 2-5 were also slow, then the
same process recovered without changing flags. All warm observations are retained.

| Warm context band | Requests | Aggregate fresh prefill t/s | Aggregate decode t/s | Median first-token s |
|---|---:|---:|---:|---:|
|20-50k|24|230.6|31.5|2.99|
|50-100k|40|328.2|36.1|3.16|
|100-150k|41|307.5|39.8|3.42|
|150-200k|41|290.3|40.6|3.64|
|200k-end|43|269.6|41.0|3.86|

Across all 189 warm turns: 286.5 prefill/38.0 decode t/s. Initial slow turns remain
included. The final 151-fresh-token request has 73.7 prefill t/s and 2.218 s first
latency; its smaller batch is not comparable to a 1k turn. The nearest full-sized
turn and final turn are recorded separately. No causal comparison to the earlier
187-turn trace or new cold/decode SOTA is claimed.

The API cache count 6420 is a startup snapshot. Native-log counts at request end
range 6618-6878; they document resize decisions, not physical residency. The CSV
labels these fields separately. Manual reading of sampled outputs found complete
sentences but repeated stock reasoning and unsupported technical claims. This
conversation is continuity/performance evidence, not a semantic-quality pass.

![One completed chat curve; the final shorter turn is labeled](qsa-chat-20261004.png)

[SVG](qsa-chat-20261004.svg), [per-turn CSV](qsa-chat-20261004.csv), and
[hashes, controls, settings and limitations](qsa-prefill-multi-20261004.json).
No private conversation text or raw token IDs are published.

Reproduce the synthetic gate with a CUDA SM75 build and runtime libraries on PATH:

```text
cmake --build build --target qsa_prefill_multi_parity
python tests/cuda/qsa_prefill_multi_check.py --exe build/qsa_prefill_multi_parity.exe --out qsa-check --run
```

Omit `--run` for the CPU-only dry-run. The explicit GPU target is not added to
automatic CTest execution. The portable runner itself was executed successfully:
12 cases, 26 processes, full-buffer equality, unchanged executable hash. Its later
timings are only reproducibility evidence. Selected model executable SHA256:
`aefbde27309d3a16ca02ba1ab67ff62ea71c80275af993a2026db67ae66a0177`.

### Extended teacher check: partial, not a full pass

A later off/on/off check planned 256 teacher-forced positions at each of 130380
and 249000 tokens per arm, using the same QSA binary and fixed 6167-slot profile.
The work deadline interrupted A2; elapsed time was 1712.984 s. It captured 1456
of 1536 planned full rows: A1 and B have 512 each; A2 has 256 at 130380 and 176
at 249000. Independent byte, finiteness and per-row hash checks find A1/B equal
at all 512 positions, with A2 equal at all 432 common positions. The missing 80
A2 rows prevent a complete extended-gate PASS. These observations do not support
a semantic-quality or performance claim. [Coverage and hashes](qsa-teacher-partial-20261004.json).


## Optional HtoD and PLE diagnostics (4 October 2026)

Two four-arm checks at 20001 and 21015 input tokens pass full first-row F32
bitwise parity and generated-ID equality. The first compares the original
binary with counters disabled/enabled; the second adds PLE snapshots and one
`--ple-sync-submit` arm. Both use fixed6167 slots, KV262144/int8/resident32768,
spec4, QSA/exact-small enabled, default IQ_MT_MIN2 and timing events. These are
diagnostic runs, not pure throughput or broad quality tests.

The counted primary-ring expert copies are 110833920000 bytes for the cold
20001 request and 33418368000 bytes for the 1014-fresh-token follow-up, all direct
from pinned memory in this fixture. The latter uses two batched chunks across a
checkpoint/turn boundary; it is not a one-chunk elastic HTTP turn. Dividing by
prompt wall time is not an instantaneous PCIe or VRAM bandwidth measurement.
The counters cover 19994 and 1013 batched tokens respectively; native fresh-token
counts are 20001 and 1014. Final-token verification is outside these counters.

The first gate retained a 211.984-second cold outlier: host PLE time was193044ms,
against693ms in its19.087-second control. The same candidate binary subsequently
ran the cold request in19.208 seconds. This localizes waiting to the PLE path but
does not identify a disk, worker, GPU-clock or synchronization cause. CUDA phase
intervals spanning host gaps must not be described as kernel execution time.

PLE read counts and wait/submit totals are cumulative. Read p50/p99 reflect the
most recent up to65536 completed reads; subtracting percentiles is invalid.
`--ple-sync-submit` also disables the existing keepalive, so that arm changes
more than worker scheduling alone. The operational I/O policy is unchanged.
The diagnostic flag remains off by default. [All measured arms and hashes](prefill-diagnostics-20261004.json).

### Actual Daily: full-history recall and arithmetic

The installed diagnostics binary (SHA256 below), with statistics disabled and
the unchanged automatic/elastic Daily configuration, completed two HTTP requests
over the full 190-pair history. Both use temperature 0 and cap 1536. The first
answer is not appended to the second request; no history is truncated.

| Thinking | Rendered prompt tokens | Reused | Completion tokens | Native prompt s | Automatic JSON oracle |
|---|---:|---:|---:|---:|---|
| off | 250487 | 0 | 52 | 293.332 | FAIL: GPU-name suffix |
| on | 250523 | 0 | 306 | 407.485 | PASS |

Independent reading confirms all five requested facts in both final answers:
GPU, 22 GB VRAM, KV capacity 262144, resident KV 32768, and the margin calculation
`262144 - 250130 - 512 = 11502`. The first names the GPU `RTX 2080 Ti modificata`;
the predeclared literal validator rejects that suffix. Its original FAIL is
preserved separately from the manual factual assessment.

Elapsed time including startup and cleanup was 989.66 s; HTTP readiness took
274.74 s. Initial automatic cache: 7082 slots, runtime spec 6 (CLI spec 4 with
adaptive policy). Both requests read their full rendered prompts. These single
observations, with intermittent stalls and variable elastic cache, are not a
causal timing comparison between thinking modes or a new SOTA. Facts also occur
in earlier assistant responses: this is a five-field check, not a hidden-needle
or general semantic-quality evaluation. All pinned inputs stayed unchanged and
the owned processes closed. [Sanitized result and hashes](qsa-daily-recall-20261004.json).

Installed Daily executable SHA256:
`a17d0dd5362bc35420ca9ef3d1a8cf8c014dafbb3aea189c6c664fe7930e8674`.


## Decode PLE timing and PCIe screening (4 October 2026)

The default-off `STRATA_DECODE_TIMING` diagnostic now measures the primary
verifier's PLE gather (submit, wait and dequantization, excluding n-gram keys).
It is a subset of host staging. The residual is not separate source attribution,
and the primary-verifier counter is not an aggregate across multiple GPUs.
Cumulative reader counts/blocked time can be differenced; rolling p50/p99
must not be subtracted as request-local percentiles. Leave the flag unset to
disable it, including on Windows (setting it to 0 still enables it).

Old executable / candidate off / candidate on pass three full first-logit
rows at 4163, 9283 and 20480 input tokens (248320 finite F32 values per row)
and 384 generated IDs per arm, bitwise. These are 128-token prefixes, not
complete answers or a throughput comparison. PLE gather in the ON arm takes
0.849/0.758/0.457 ms per window, with 0.002 ms residual staging. The earlier
intermittent 30–33 ms staging cost was not reproduced here.
These numerical arms use 5080 actual cache slots / 8442 MiB, PCIe share 0,
spec 4/MTP max 4, with elastic/adaptive swaps, suffix and prompt caching off.
They do not establish full-answer parity under the operative elastic policy.

**Initial decision: not installed in Daily.** A separate smoke using the operative automatic,
elastic and tail settings finishes its 9283-token LRU task but fails the exact
answer oracle: hits 1 / misses 13 instead of hits 2 / misses 12. Valid JSON
and throughput do not establish correct output. The old Daily reference passes
the complete-answer oracle. Initial automatic cache allocations differ (5849
candidate slots, 6014 reference slots), and full generated token sequences
differ. This pair alone does not isolate a code regression.
The operational executable remains `a17d0dd5…`; config, launcher, source
snapshot and historical manifest are unchanged. No new SOTA or release.

The separate PCIe-share study warms three tasks and runs A/B/A in one process:
A=0.36, B=0.55; workers 7, spec 6/MTP max 4, suffix/adapt/elastic off.
Auto sizing at startup gives 5732 slots/9503 MiB cache, 1639 MiB free with
reserve 2048. Prefill still borrows slots: placement is not fully frozen,
and this is not an operational elastic-profile comparison.

| Rendered input | A1 decode t/s | B decode t/s | A2 decode t/s |
|---:|---:|---:|---:|
| 4163 | 38.422 | 28.656 | 36.875 |
| 9283 | 38.082 | 28.363 | 38.851 |
| 20480 | 38.362 | 31.905 | 36.821 |

Higher share is rejected: all three tasks slow down. First/third control
pairs also drift about 4.1%, beyond the predeclared 3% stability bound, so
these are screening observations, not precise general regression estimates.
All eight arithmetic/LRU requests pass exact final oracles. Four archive
answers, including warmup, finish within 200–300 words and retain the bounded
decision facts. This does not establish general quality or numerical parity
between CPU/GPU partitions, which change rounding and continuations.

Failed preflights, count/parser failures and incomplete answers remain
separate from completed gates. [Evidence, hashes and limitations](decode-ple-diagnostics-20261004.json).

## Persistent system prefix (4 October 2026)

`--prompt-cache-file PATH` persists one text root across engine restarts. It is
off by default. [Configuration and file contract](../DETAILS.md) describe exact
token/frontier matching, complete MTP state, integrity checks and invalidation.
The measured model is Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS on the modified RTX
2080 Ti 22 GB, Ryzen 7 5800X, 96 GiB RAM, Windows/CUDA 12.6/SM75.

The synthetic system prefix contains **4003 tokens**, followed by 49 fresh tokens
in a 4052-token request. The file is 179,213,800 bytes (170.91 MiB). Each restore
runs in a fresh engine process. KV remains INT8 with capacity 262144 and 32768
resident tokens. Seven complete factual answers pass exact JSON-content oracles.

With the actual Daily auto/elastic profile (PCIe 0.36, workers 7, CLI spec 4/runtime 6,
MTP 4, reserve 1393 MiB; exact-small and grouped QSA enabled), three pairs measure:

| Pair | Recompute prefix: first token (s) | Restore prefix: first token (s) | Saved (s) |
|---|---:|---:|---:|
| 1 | 5.328 | 1.875 | 3.453 |
| 2 | 5.000 | 1.344 | 3.656 |
| 3 | 4.594 | 1.266 | 3.328 |

The two medians are **5.000 and 1.344 seconds**, a 73.12% reduction in their ratio.
Median paired saving is 3.453 seconds; these are distinct statistics. The shared
first-logit diagnostic is enabled on both sides. This is native request-to-first
`T` latency after READY, not full HTTP/client TTFT. Model startup takes about
37–40 seconds and is excluded. File pages can be in the Windows filesystem cache;
there was no reboot or OS-cache purge, so this is not a cold-SSD bandwidth test.
Three pairs do not establish a population distribution, p99 or a new decode SOTA.

Restore read/validation takes 212.3–252.6 ms, plus 24.2–29.8 ms upload. The first
creation takes 9.625 seconds to the first token, with 477.9 ms recorded for capture
and write. Its full excess over separate cold samples is not isolated as write
overhead. A first use or changed prompt still requires prefill and file creation.

Correctness is checked separately with fixed placement: actual 5080 slots / 8442
MiB, PCIe 0, spec 4/MTP 4, workers 7, elastic/suffix/adaptive swaps disabled and
`STRATA_IQ_MT_MIN=1`. Reference, candidate off, capture and fresh-process restore
produce bitwise identical full first-logit rows (248320 finite F32 values) and
all 31 generated IDs. Changing the ticket in the system prompt forces a miss and
replacement; its following restore also matches bitwise and answers correctly.
The old Daily and candidate off also complete the 9283-token LRU task with 740
identical output IDs, identical first-logit rows and the correct 2 hits / 12 misses.
This resolves that repeat under matched placement; it does not erase the earlier
auto-profile failure or establish general model quality.

The actual auto/elastic timing arms have 7284–7448 initial slots. Their logits are
not bitwise equal, including cold-versus-cold samples. Two cold outputs use a
multiline JSON layout instead of compact JSON; all values are correct. Full IDs
differ with that formatting. These arms demonstrate bounded factual correctness
and latency under variable placement, not numerical equivalence or causal
attribution of every floating-point difference to a particular setting.

Both CPU codec suites pass (OS SHA-256 and portable SHA-256). They cover known
digests, segmented roundtrip, corrupt/truncated/oversized data, identity/prefix/
frontier/steering mismatch and failed-write preservation. All 13 model processes
from the two completed runs have exited; the public server was not started.

Candidate executable SHA256:
`f2e371bf631d13b40890b8efc8b35395844b1dfd592f8c5baab0cef60729621a`.
[Sanitized evidence and hashes](prefix-cache-20261004.json) and
[synthetic token fixture](prefix-cache-fixture-20261004.json) contain no user chats.
The implementation and CPU tests are in main; the published release binary
`v0.1.38-sm75-20261003-r1` predates this feature.

**Local Daily deployment completed.** Source commit `f01c0843daa32238e6194b5e54a294f5956ab555`
and the candidate hash above are installed in `build-0138-daily`. The config adds
only `--prompt-cache-file D:\ds4_work\strata\sota\prompt-prefix-v1.bin`;
its SHA256 is `db77e8364bf9221cfc846b8133a3c40fc8936196aa9939fa583cee3d539fe172`.
The first eligible user request will create that file; synthetic test snapshots
were not installed as user prefixes. The existing launcher is unchanged.

`build-0138-daily/backup-prefix-disk-20261004` preserves the old executable,
config, changed operational sources and all 48 untracked checkout files.
The historical `source-manifest.json` is unchanged; the separate
`prefix-cache-validation.json` records this installation and all 289 verified
source hashes. To disable persistence, remove the flag and its path from the
config while the engine is stopped. To restore the previous binary deployment,
restore the executable, config and sources from that backup according to
`recovery.json`. Post-install checks confirm the server remains off and port
8100 is free. This is a local Daily update, not a new published binary release.

### Larger persistent prefixes and the byte limit (4 October 2026)

The same installed engine and Daily configuration were tested with synthetic
ledgers containing three verified facts near the start, middle and end. The
prefix lengths below are exact; each request adds 48 fresh tokens. Every arm
starts a separate engine process. Hardware, native first-token metric, startup
exclusion and Windows filesystem-cache caveats are the same as above.

| Prefix tokens | Run | Recompute: first token (s) | Create file: first token (s) | Restore: first token (s) | File (MiB) |
|---:|---|---:|---:|---:|---:|
| 10000 | matrix | 10.656 | 14.656 | 1.375 | 258.25 |
| 20000 | earlier standalone | 18.765 | 18.375 | 1.907 | 403.89 |
| 20000 | matrix | 16.781 | 18.046 | 1.547 | 403.89 |
| 27000 | repeat 1 | 22.688 | 23.672 | 2.281 | 505.84 |
| 27000 | repeat 2 | 25.218 | 23.812 | 1.704 | 505.84 |

All completed arms return the correct three facts. These auto/elastic runs
are factual checks, not bitwise parity or broad model-quality evaluations.
At 27000 tokens, both fresh-process restores actually reuse all 27000 tokens;
the paired first-token reductions are 89.95% and 93.24%. Files are 530,414,864
bytes. Read/validation is 716.8 / 599.0 ms and upload is 73.6 / 63.1 ms.
This is not a cold-SSD test or a change in decode speed.

Both limits apply: **32768 tokens and 512 MiB of serialized payload**. For this
model and INT8 KV geometry the byte cap is tighter. **27000 is the largest
measured successful prefix**, not a universal limit for all models or KV modes.
A 28000-token image is estimated at about 520.4 MiB and was not run.

The original ten-trial plan (two each at 10/20/30/40/50k) was stopped at the
user's request to focus on supported sizes. It did not complete ten trials:

- At 30000, no file was saved (size/geometry/RAM admission log; the estimated
  image is about 549.5 MiB). The third arm recomputed the prompt in 23.906 s,
  versus 23.984 s cold. Both have zero prefix reuse; this is a fallback,
  not a cache speedup.
- At 40000, the engine reported an incomplete windowed MTP prefix and saved
  no file; this also exceeds the token cap. The fallback took 32.797 s,
  versus 33.422 s cold, with zero prefix reuse.
- At 50000, cold and create arms finished (40.375 / 40.797 s, correct facts,
  no file). The restore attempt was interrupted when the scope changed;
  it has no completed timing or correctness result.

Excluded attempts remain recorded: the initial runner's 1024 MiB READY floor
rejected 10k restore and 20k cold before generation (720 / 600 MiB free).
The floor was changed to 512 MiB, matching the engine's existing elastic
target of about 500 MiB; engine settings were unchanged. One subsequent 20k
startup timed out after 180 s with slow expert loading, before generation;
its cause is unresolved. Only the completed repeats appear in the table.
No competing application was stopped and no artificial saturation was added.

[Measured arms, exclusions and evidence hashes](prefix-cache-long-20261004.json)
and [synthetic messages and token fixtures](prefix-cache-long-fixtures-20261004.json)
are provided separately from the earlier controlled parity gate. All test
engines have exited. The production prefix file was not populated by these
tests; the Daily executable and configuration hashes are unchanged.

## Neuron cache pilot, 4 October 2026

A default-off diagnostic now records actual expert inputs, IDs, routing weights,
outputs and CPU/VRAM/PCIe placement. Its offline GPU replay tests neuron masks;
it does not prune the running model. Three synthetic prompts (86,85,102 input
tokens), each capped at 64 generated tokens, supply 192 accepted positions on
layers 0,1,35,36: 768 routed sums and 7680 expert invocations. This is short
context despite the configured 262144-cell capacity and 32768 resident cells.

Hardware: modified RTX2080Ti 22GB, Ryzen5800X/AVX2,96GiB RAM, Windows,
CUDA12.6/MSVC14.44, SM75 Release. Controls: spec4/MTPmax4, workers7,
PCIe0.25, IQ_MT_MIN1, suffix/adaptation/elasticity off. The requested cache3800
sets a max-blob budget, yielding exactly5080 variable-size slots/8442MiB with
the pinned profile. These are controlled diagnostic settings, not Daily auto.

Daily executable vs diagnostic-off, then diagnostic-off vs diagnostic-on:
all64 output IDs and all248320 finite first-logit float32 values are bitwise
equal on the code prompt. GPU replay reproduces every captured VRAM/PCIe
expert output with zero L2 difference. CPU replay-to-GPU differences are
reported separately. The alternate down projection's routed-sum numerical
floor is below4.9e-8 relative L2. Grouped v1/v2 output and Q8_1 checks pass for
four real native-format pairs in each of the three replays. Six executable
I/O cases pass, including overlapping rejected draft tails and overwrite refusal;
the Python parser/analyzer also rejects invalid and truncated records.

For each current activation, retain its largest-|h| neurons at their original
positions, and retain whole touched down-quantization blocks:

| Neurons retained | Hypothetical weight-byte saving | Mean routed-sum relative L2 | Per-prompt p99 relative L2 |
|---|---:|---:|---:|
|75% (480/640)|15.22%|2.06–2.20%|3.98–4.92%|
|50% (320/640)|30.43%|7.69–8.38%|14.63–16.07%|
|25% (160/640)|45.65%|19.51–21.44%|33.68–35.91%|

Errors compare against full GPU replay on the same input, using the FP64 sum
of the ten routing-weighted experts. Shared expert and residual are excluded.
They are not percentages of incorrect answers. Savings count selected gate/up
rows and original down blocks, normalized by total full bytes across these
invocations; selection, packing, alignment and metadata costs are excluded.
The replay still reads full blobs, so no bandwidth or speed gain is measured.

Reusing the previous invocation's top480 mask gives21.76–22.78% mean local
error, with p9954.94–61.11%, even though27–28% of calls fall back to full width
because no history exists. This history uses the *complete* previous h:
refreshing it is not free in a genuinely reduced execution. Its Q8_1 block
scales may also change. Top-|h| optimizes retained hidden energy, not necessarily
the error after the down matrix; neither experiment tests final answer quality.

As an optimistic packing calculation, unions of top480 masks over eight accepted
positions retain88.98–89.56% of the distinct experts' bytes per layer/window on
average; sixteen positions retain90.27–90.48%. These are observed accepted-path
unions, not predicted branches or a tested100-candidate speculation tree.

The simple previous-mask policy is not suitable for promotion. Draft-conditioned
prediction, output-aware selection and an exact fallback remain research tasks.
Daily, launcher and public ZIP are unchanged; no server or test engine remains.
[Diagnostic source manifest](neuron-trace-20261004.json) and
[measurements, analyzer and fixtures](https://github.com/imanu86/moe-aggressive-commit/tree/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo/corse_2080ti/20261004_neuron_cache)
preserve the evidence independently of earlier release validation.
## Real 100-candidate adaptive probe, 4 October 2026

Seven mask-bank epochs were measured on three synthetic prompts: **700 candidate
positions and 28,000 candidate expert calls on layers 0, 1, 35 and 36**. Each epoch
starts from an accepted prefix, takes ten distinct non-EOS top-logit target roots,
and extends each root by nine greedy MTP steps. These are ten coherent paths of
ten tokens, not a beam search or 100 distinct tree nodes: the epochs contain 38–88
distinct `(depth, token)` pairs. Reverse branch order and the ordinary MTP chain
pass token-level controls. Candidate masks are constructed before inspecting the
held-out continuation's errors.

For each `(layer, expert)`, union the top 320 absolute hidden activations from
every candidate invocation. Keep that bank until mean relative L2 across the four
routed-expert sums exceeds 5%, any layer exceeds 10%, or mean missing route weight
exceeds 10%. Missing experts use full fallback. A trigger row also calls for full
fallback; rebuild from the newly accepted prefix before the next position.
These are laboratory thresholds, **not answer-accuracy guarantees**.

| Synthetic task | Positions within threshold in successive epochs | Estimated bank weight saving | Ideal weight-read saving on those positions |
|---|---|---|---|
| Code | 5, 4, 11 | 14.60%, 14.72%, 16.36% | 3.46%, 3.72%, 8.81% |
| Explanation | 4, 2 | 13.94%, 15.19% | 3.37%, 5.86% |
| Reasoning | 6, 1 | 13.92%, 13.76% | 3.47%, 1.80% |

Storage estimates compare the bank with the full weights of its own selected
experts. The path estimate weights the bytes of actual expert calls before each
trigger, including full fallback for unseen experts. Both retain original down
quantization blocks. The bank includes many alternatives that the target never
uses; frequently used experts tend to retain more neurons. Neither number is a
measured bandwidth or t/s improvement: the replay still reads full blobs.

Predicting the 100 candidates took **118–128 ms** per epoch, excluding the extra
reverse-order and standard-chain checks. Sequential target scans took 15.64–17.95 s
in this diagnostic harness, including repeated prefix prefill; their decode
components totaled 3.90–5.47 s. Additional GPU activation replay cost 11.71–15.39 s,
CPU mask construction 0.328–0.390 s and target replay/validation 5.33–6.80 s.
These stages are recorded separately; they are not optimized online construction
or verification costs, and startup is excluded. No end-to-end acceleration was
implemented or demonstrated.

The masks were also replayed with top160/top480 candidate selections. The raw
curves retain all three variants; only top320 controlled the actual refresh
prefixes. Other variants' frozen-bank lifetimes are counterfactual, not separate
adaptive runs. Full-mask GPU results are bit-identical, zero masks produce exact
zero, and absent-expert fallback is bit-identical; four malformed bank inputs are
rejected. The portable analyzer reproduces all seven banks and curves exactly.

The code continuation retains the previous 64 greedy IDs. Explanation/reasoning
do not match the older run under different memory registration settings, so two
fresh-process controls used the same new runtime variant without any candidate
probe. Each matches all64 IDs and all256 accepted `(position,layer)` trace rows
byte-for-byte, including inputs, IDs, routing weights, expert outputs and tiers.
Free-VRAM telemetry differs; the numerical runtime settings match. No first-logit
or model-wide quality claim is inferred from these controls.

Hardware: modified 22 GB RTX 2080 Ti, Ryzen 7 5800X, 96 GiB RAM, Windows/CUDA12.6.
Initial prompts were 85–102 tokens; refresh prefixes reached 109. KV capacity was
262144/int8, resident32768; actual expert cache5080 slots/8442 MiB, spec4/MTP4,
workers7, PCIe0.25, IQ_MT_MIN1, no elasticity/swaps/suffix/persistent prefix cache.
Valid runs used `STRATA_NO_LARGEPAGES=1` and `STRATA_ARENA_PIN_GIB=8`. Three startup
attempts were excluded: a180 s timeout and two stopped repeats, including one
with the pin cap alone. This does not establish the cause of the startup stalls.

Errors apply to local routed-expert sums, excluding shared experts and residuals.
Target inputs/routing come from the full model: accumulated state error of an
online pruned model was not tested. This is a short-context measurement, not a
250k-depth validation. The Daily, launcher, config and public ZIP are unchanged.
Sources and validation are pinned in `neuron-probe100-20261004.json`; complete
synthetic candidates, curves, costs, exclusions and limits are in the research
repository's `20261004_neuron_cache/probe100-measurements.json`.


## Physical GPU/CPU expert blocks, 4 October 2026

A separate physical block probe preserves every weight and neuron contribution.
On 120 short-context, single-token cases, partition parity on each backend has
maximum L2 2.11e-7 (CPU) / 9.19e-8 (GPU). Mixed-backend output is not bitwise
identical to the all-GPU reference. Nine interleaved rounds per case compare
30% of ten experts on GPU with three complete GPU experts: mean case-median
latency is 859.35 versus 844.55 microseconds, with substantial variation and
no reproducible speedup. Packing and weight upload are excluded; no online
cache policy or end-to-end throughput was measured. Daily and release unchanged.
[Method, all controls, raw evidence and reproduction](HYBRID_BLOCKS.md).
