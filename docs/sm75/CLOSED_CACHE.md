# Closed expert cache experiment

This opt-in approximation is a research probe, disabled by default. It is not
part of the released executable or the operational Daily. On the modified
22 GB RTX 2080 Ti, a 512-token prose continuation improved from 28.03 to
44.13 token/s including two cache refreshes. Structured answers deteriorated
into malformed JSON and repetition even with 97% MTP acceptance. These are
raw-token speeds, not a validated improvement in useful-answer throughput.

The full [protocol, answers, logs and source hashes](https://github.com/imanu86/moe-aggressive-commit/blob/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo/corse_2080ti/20261004_closed_cache/RESULTS.md)
record 11 completed runs, one corrected startup failure, and the limits.
Initial prompts were 85, 1680 and 2470 tokens. KV capacity was 262144/int8 with
32768 resident cells; this was not a 250k-depth experiment. The fixed expert
cache held 5080 slots / 8442 MiB. No result is promoted to a decode SOTA.

## What the probe does

The full prefill counts expert choices. At its end, the hottest experts fill
each layer's existing cache quota. During decode, no CPU expert pool callback
runs. Shared experts, dense layers, PLE I/O, orchestration and KV transfers
continue normally. CPU expert work is zero, not total host work or total PCIe.

Set `STRATA_CLOSED_ROUTING` only in a separate diagnostic engine environment:

- `resident`: select and normalize the top 10 among resident experts.
- `drop`: retain the original top 10 but zero absent contributions without
  renormalizing. Zero-weight resident dummy IDs satisfy the GPU planner contract.
- `observe`: collect original routing counts with full expert evaluation.
- Unset: ordinary inference, without the probe buffers or counting kernels.

Original routing choices are counted only on committed decode inputs. The
cache refreshes below 50% MTP acceptance after at least 64 offered drafts and
32 emitted tokens. Refresh time and transfer bytes are reported separately
and included in total decode time; the initial fill is included in prompt time.
On the prose run, initial fill took 613 ms; two refreshes took 758 ms total.
Counts accumulate until a refresh. Ties prefer current residents, then lower ID.
Only same-layer slot exchanges occur; the layer quotas are not optimized.

This implementation is restricted to the serve protocol, one GPU, the native
48-layer/512-expert/top-10 geometry, fixed cache with at least ten residents in
each layer, no ordinary adaptive swaps, no prompt cache, no suffix drafting,
and a positive prefill chunk. Tested with `--no-spec-split`. The compact RAM
complement exchange path is preserved but was not exercised by these arena runs.

## Validation and limitations

87 CUDA checks cover full-residency bitwise parity, an independent host reference
for restricted top-k and weights, ties/underflow, graph replay with a changed
policy, and exclusion of rejected rows from counts. The 152 output IDs of one
fixed-placement task match the old Daily, new default path and observer exactly.
This is a narrow output regression check, not full-logit or general-quality parity.

The two semantic oracles fail even on the complete model without reasoning.
The reduced outputs additionally become malformed; the LRU response repeats
D,E,F while one MTP block accepts 100%. The `drop` prose response also repeats
a character sequence. When confidence filtering proposes few drafts, a trigger
waiting for 64 proposals can stall. Neither policy is suitable for Daily use.
A further attempt needs an independent full-model check and a trigger that
accounts for missing drafts; MTP agreement alone cannot validate the approximation.

## Build the diagnostic

In an existing CUDA/native-expert SM75 build directory:

```powershell
cmake -S . -B build-sm75 -DSTRATA_BUILD_CLOSED_PROBE=ON
cmake --build build-sm75 --parallel 2 --target strata closed_router_parity
build-sm75/closed_router_parity
```

`closed-cache-20261004.json` pins the tested source and executable. All older
manifests remain preserved by hash. `python tools/sm75/check_source.py` verifies
the current research source. Use the published release for the validated Daily.
