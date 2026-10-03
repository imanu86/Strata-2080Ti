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

It checks the source file hashes from `docs/sm75/source-manifest.json` and
the absence of CH16/BPW switches. Publication overrides cover README, launcher,
ignore rules and byte-preserving Git attributes; original README is kept in
`docs/UPSTREAM_README.md`. These checks never launch CUDA or the model.

See [BENCHMARKS.md](BENCHMARKS.md) for numerical targets and model evidence.
GPU checks are explicit. This fork has no automated build/test workflow.
