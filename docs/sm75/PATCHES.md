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
