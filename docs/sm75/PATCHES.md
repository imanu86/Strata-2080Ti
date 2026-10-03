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

# Current 0.1.36 port

Upstream `36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca` plus the preserved SM75/elastic port. Original snapshot hashes
remain in `source-manifest-0131.json`; current hashes are in `source-manifest.json`.
PR315 and asynchronous commit are now upstream implementations. CUDA active-bound
top-k is restricted to SM75; new upstream SM90+ cluster dispatch is retained.
PR463 contributes the minimal pending-copy wait before serve/CLI verification.
PR500 contributes parallel intermediate quantization, default off and enabled by
`STRATA_PARALLEL_INTERMEDIATE_QUANT=1`. Its quantizers match the serial/native
AVX2 paths. New upstream fused prefill requires SM80+, cluster decode SM90+;
neither fast path is enabled on Turing. No CH16/BPW/PR439 or workflow is imported.

Current executable SHA256: `c35505ca79b790bd37ed7073fcadd399dd30d1b88e85bb99a6eb24a52d945d0a`.

## Historical 0.1.31 implementation and evidence

# Stable snapshot and experiments

Validated executable SHA256:

```text
f469e096e868e4579daaec71ceef63693790e300e7a4cc5b49a8853d075a1d34
```

Baseline0.1.31 plus nine local patches, HC/GDN FP16, vector dequantization,
PR315 and active-bound prefill top-k. Incremental patches and build records:
[research repository](https://github.com/imanu86/moe-aggressive-commit/tree/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo/patches).

| Component | Release status |
|---|---|
| Elastic expert cache, fixed core/rotating tail | Enabled by measured profile |
| Checkpoint-aligned grid, grouped experts, async commit | Included |
| HC/GDN FP16 and vector dequantization | Included, limited numerical/retrieval validation |
| PR315 HC read | Included with numerical gate |
| First logits | Opt-in `STRATA_DUMP_FIRST_LOGITS` |
| Active-bound prefill top-k | Included:16 bitwise-ID cases and model A/B/A |
| CH16 | Excluded: no demonstrated model gain |
| Score BPW | Excluded: model benefit unvalidated |
| PR439 gather | Excluded: A/B/A gain within noise |

## Active-bound dispatch

At KV262144, score rows have stride65538 blocks. The original dispatcher checks
that capacity against the register limit33792, disabling the fast kernel even
at131k actual context. Prefill already knows the batch's largest `n_bid+1`.

`qsa_block_topk(..., active_blocks)` applies the register limit to that bound while keeping
the score stride. Invalid/out-of-range bounds and contexts beyond135164 cells
fall back to the original API. Both kernel bodies and the scorer are unchanged.
Presence of `STRATA_TOPK_CAPACITY_GUARD` forces the former dispatcher; unset
it for the active path (setting it to0 still forces the capacity path).

Only prefill supplies the active bound. Captured decode graphs omit it:
their context can grow after capture, invalidating a fixed host bound. No decode
gain is attributed to this prefill change.


## Grouped-query QSA prefill, 4 October 2026

`STRATA_QSA_PREFILL_MULTI=1` enables a default-off SM75 scorer that shares each
pooled key across up to eight prompt queries. It applies only to valid positive
active bounds and more than eight queries; decode and other architectures keep
the prior path. Score/selection buffers and four captured model rows/outputs
match bitwise in bounded tests. Experimental group/grid tuning is not shipped.
See [paired measurements, the complete chat curve and limits](BENCHMARKS.md#grouped-query-qsa-prefill-4-october-2026).


## Optional prefill diagnostics, 4 October 2026

`STRATA_PREFILL_STATS=1` reports per-request primary/stage statistics and PLE I/O
snapshots on stderr. The default is off. Expert byte counters cover issued
non-peer primary-ring HtoD copies and distinguish direct pinned-arena copies.
They do not measure physical VRAM or PCIe bandwidth. Numerical comparisons and
the intermittent PLE stall are documented in
[the diagnostic evidence](prefill-diagnostics-20261004.json).
