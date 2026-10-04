# Current Daily package, 4 October 2026

Release `v0.1.38-sm75-20261004-r1` packages the exact operational Daily engine:
SHA256 `f2e371bf631d13b40890b8efc8b35395844b1dfd592f8c5baab0cef60729621a`.
Its engine-source commit is `f01c0843daa32238e6194b5e54a294f5956ab555`, built with
CUDA 12.6, MSVC 14.44, Release, SM75, native CPU experts and shared CUDA runtime.
The executable was built and model-tested during development; publication only
copies it and checks bytes. No publication-time CUDA build or model run occurs.

The package adds the later exact-small and grouped-query prefill paths,
incremental frontend tokenization through the matching source checkout, and
opt-in persistent initial-prefix caching. It includes the disabled PLE diagnostic.
Use this release's source tag or current main for the Python server and profile
generator; the ZIP supplies the engine, notices, provenance and measured evidence,
not model weights or CUDA/VC++ runtime installers.

`release-20261004.json` records the tested engine/frontend source hashes plus
the current publication helpers. `check_source.py` verifies this overlay and
the hashes of all three historical manifests; those manifests are unchanged.
Historical validation below belongs to its stated engine, not automatically to
the new binary. See [the separate feature gates and limits](BENCHMARKS.md).

Publication CPU checks verify the archive contents, engine hash, profile generation
and preserved configurations; the two prefix-codec suites each pass 84 checks.
The tested hardware is a modified 22 GB RTX 2080 Ti, Ryzen 7 5800X/AVX2 and 96 GiB
RAM. Other memory sizes need their own settings and validation. Source builds
remain available with `python tools/sm75/build.py --jobs 2` as described below.

## Optional neuron trace research overlay

Later main adds `STRATA_NEURON_TRACE_FILE`, disabled unless explicitly set, and
the separate `neuron_trace_replay` executable. The published ZIP and operational
Daily remain the earlier executable above. `neuron-trace-20261004.json` records
the diagnostic build's exact sources and its narrow validation; it preserves
the release manifest and all three earlier manifests by hash. The newer
`neuron-probe100-20261004.json` pins the 100-candidate research overlay and
preserves this earlier trace manifest too. The default
`python tools/sm75/check_source.py` verifies this current research overlay.
`--manifest docs/sm75/release-20261004.json` explicitly checks the release source
instead and correctly reports mismatches on the later research sources.

In an already configured CUDA/native-expert build directory:

```powershell
cmake -S . -B build-sm75 -DSTRATA_BUILD_NEURON_REPLAY=ON
cmake --build build-sm75 --parallel 2 --target strata neuron_trace_replay
```

Set `STRATA_NEURON_TRACE_FILE` to a new file before starting the diagnostic
engine. It records at most 64 committed decode windows on layers 0,1,35,36,
with input vectors, expert IDs, true routing weights, expert outputs and tiers.
Only one visible GPU and the 2560/640/512/top-10 geometry are supported.
The extra graph copies and file writes invalidate timing comparisons.
After the engine exits, replay with:

```powershell
build-sm75/neuron_trace_replay trace.bin model-shard.gguf replay.bin
```

The replay selects accepted positions, reconstructs GPU hidden activations,
and compares full, current-top-|h| and previous-full-h masks. It always reads
full blobs; it does not implement a compact online expert. The public research
[analyzer and synthetic fixtures](https://github.com/imanu86/moe-aggressive-commit/tree/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo/corse_2080ti/20261004_neuron_cache)
provide the matched offline calculations. Source and model checks are described
in [the pilot record](BENCHMARKS.md#neuron-cache-pilot-4-october-2026).

## Historical 0.1.38 SM75 source, 3 October 2026

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

# Current SM75 build

Current main is upstream0.1.36 plus the documented port. The complete fresh
CMake/Ninja Release build passed with CUDA12.6, MSVC14.44, native CPU experts,
shared CUDA runtime and architecture75. The builder now defaults to75.
100 server mock tests and the CMake-built top-k/HC checks pass. Historical
publication notes below refer to the original0.1.31 release. The original
manifest is preserved as `source-manifest-0131.json`; `check_source.py` verifies
the current source manifest. No automated workflow is installed.

# Build the SM75 snapshot

The measured daily used Windows, CUDA **12.6**, MSVC **14.44.35207** (VS2022
Community), Python3.12, Ninja, Release, CUDA targets`75;86`, shared CUDA runtime
and `STRATA_PORTABLE=OFF`.

Open an **x64 Native Tools / Developer PowerShell for VS2022**, put CUDA12.6's
`bin` on PATH, install `requirements.txt` in `.venv`, then:

```powershell
.\.venv\Scripts\python.exe tools/sm75/build.py --jobs 2
```

This produces `build-sm75/strata.exe`. Generate your config with
`tools/sm75/configure.py --exe build-sm75/strata.exe ...`.
`--ggml-dir` selects an existing pinned llama.cpp source directory; otherwise
CMake fetches `3cf03257f219afbe7334045ff7c6a06ac68c627d`, recorded in
`third_party/ggml/VERSION.txt`. `--arch` changes the CUDA architecture list.

The builder never downloads model weights or starts the engine. Calling it
does compile CUDA code and use CPU/RAM; it is an explicit user action.
It was **not run for publication**. The release engine is the exact previously
tested daily, assembled from compatible cached base/HC/vector/PR315/top-k
objects. Their modified source is present here without publication-only CUDA
or C++ changes. Fresh toolchains/dependencies/linking can change the binary hash;
portability beyond the measured machine remains unvalidated.

## CPU-only source check

```powershell
python tools/sm75/check_source.py
```

It checks the source file hashes from the newest available research/release manifest and
the absence of CH16/BPW switches. Publication overrides cover README, launcher,
ignore rules and byte-preserving Git attributes; original README is kept in
`docs/UPSTREAM_README.md`. These checks never launch CUDA or the model.

See [BENCHMARKS.md](BENCHMARKS.md) for numerical targets and model evidence.
GPU checks are explicit. This fork has no automated build/test workflow.
## Optional 100-candidate experiment

The separate research engine supports `STRATA_NEURON_PROBE_CONTROL`, a path to
an ASCII control file reread for each pipe-server request. It is disabled by
default and is not present in the Daily profile or release ZIP. Only a single
GPU and prefixes of at most 2,049 tokens are supported by this probe.

- `draft C:\lab\candidates.txt`: request exactly one greedy output token.
  The probe takes ten distinct, non-EOS top-logit target roots and extends each
  with nine greedy MTP steps. It writes ten paths of ten tokens. The output path
  must contain no spaces and must not already exist. The first output line is
  `PROBE100 last_prefix_position total_probe_ms prediction_ms`; total time also
  includes reverse-order token parity and comparison with the ordinary MTP chain.
  The following ten lines contain token IDs; the final line has the ten root logits.
- `force 10 id0 id1 ... id9`: request eleven output rows. These are **teacher-forced
  diagnostic inputs**, not a generated answer or a quality test. They capture the
  last prefix row plus all ten candidate input rows, using one-token verifier calls.
  EOS inputs are rejected. Ignore the first trace row when collecting candidates.
- `normal`: ordinary generation, used for the held-out reference continuation.

Set `STRATA_NEURON_TRACE_FILE` as in the earlier trace instructions. For a probe
session with multiple requests, `STRATA_NEURON_TRACE_WINDOWS=512` raises the bounded
capture limit (default 64; accepted range 1..512; at most 1,847,488,552 bytes at T=8).
The harness splits the append-only stream at completed request boundaries; each
split retains the 40-byte header and must pass the existing per-trace validator.
Positions restart between requests: never feed the combined multi-request stream
to the ordinary replayer as though it were one continuation.

`neuron_trace_replay trace.snt model.gguf output.snr masks.snm` accepts an optional
frozen mask bank. `SNM10001` contains little-endian FF=640 and record count, followed
by unique `(layer, expert)` int32 keys and three arrays of 640 boolean bytes.
The output then uses magic `SNRv0002`: vectors 5..7 apply these three masks;
`cold=1` means an absent expert was evaluated in full. All other layout fields
match SNRv0001. Full expert blobs are still read; this tool does not implement
compact weight kernels or measure a bandwidth saving.

The public research analyzer builds unions from the 100 candidate positions and
evaluates target replay errors. Adaptive refresh decisions use only errors at
already-processed target positions. Reference inputs remain from the full model:
the local replay does not simulate accumulated state error of a pruned model.


## Optional physical GPU/CPU block experiment

The `hybrid_expert_probe` target is included only with
`STRATA_BUILD_NEURON_REPLAY=ON`. The accompanying pool test is enabled with
`STRATA_PARITY_POOL_QUANT=ON`. See [HYBRID_BLOCKS.md](HYBRID_BLOCKS.md) for the
input converter, explicit commands, measured outcomes and limitations. This
probe performs real complementary CPU/GPU computation; it does not enable an
online partial-expert cache or replace the Daily.

## Turn-local pruning replay correction

The current `neuron_trace_replay` uses `quantize_q8_1_rows` for masked down
projections, matching the grouped full-reference activation quantizer.
The earlier standalone quantizer produced a small full-mask discrepancy in
one of2560 calls. After correction, full/zero controls and the repeated10%
pruning case pass bitwise. Output formats and command-line arguments are unchanged;
the corrected executable SHA256 is
`64b55b1bef1b4ef4528ab8be9d7cc7f4465ed878e40ddf3b60bd00667d7a970c`.

`turn-pruning-20261004.json` records this diagnostic snapshot and preserves all
historical manifests. See [TURN_PRUNING.md](TURN_PRUNING.md) for the experiment,
public scripts/masks and the limits of the negative pruning result. Production
CUDA kernels, the Daily executable and release ZIP are unchanged.
