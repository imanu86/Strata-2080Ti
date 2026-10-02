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

`qsa_block_topk_active` applies the register limit to that bound while keeping
the score stride. Invalid/out-of-range bounds and contexts beyond135164 cells
fall back to the original API. Both kernel bodies and the scorer are unchanged.
Presence of `STRATA_TOPK_CAPACITY_GUARD` forces the former dispatcher; unset
it for the active path (setting it to0 still forces the capacity path).

Only prefill calls the new API. Captured decode graphs keep the original API:
their context can grow after capture, invalidating a fixed host bound. No decode
gain is attributed to this prefill change.
