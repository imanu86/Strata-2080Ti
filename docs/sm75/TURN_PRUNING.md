# Turn-local neuron pruning pilot, October 4, 2026

**Masks learned from earlier tokens do not pass the local error screen, even
when removing only10% of the neurons in eligible experts.** The Daily and release
remain unchanged. This experiment does not establish a throughput improvement.

## What was tested

Modified22GB RTX2080Ti, Ryzen5800X,96GiB RAM, Windows, CUDA12.6/SM75,
Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS. Three short reference texts (code, explanation,
reasoning), prefixes86/85/102 tokens and layers0/1/35/36. This is offline expert
replay on recorded full-model inputs, with no KV allocation or250k-depth test.

The first32 continuation input positions calibrate a frozen mask; the following
31 positions are held out. The selector receives only calibration rows. Experts
with fewer than four observations stay complete. Two scores rank neurons by the
mean or maximum past value of `route_weight^2 * h_i^2 * ||W_down[:,i]||_2^2`.
These estimate individual output contributions; they do not model cancellation
between neurons. Keep576/512/480 of640 neurons:10/20/25% removal.

The18 conditions share93 held-out input positions,372 routed layer sums and3720
expert calls. They are not18 independent conversations. Reference and masked
vectors use the same GPU backend; routing and all later-layer inputs remain
from the full model. This does not measure quality of generated pruned answers.

| Removed from eligible experts | Mean local relative L2, range over six text/score pairs | Ideal routed weight-byte saving, range over texts |
|---|---:|---:|
| 10% | 5.03–8.52% | 0.91–2.45% |
| 20% | 8.99–12.89% | 1.81–4.90% |
| 25% | 10.15–14.72% | 2.26–6.13% |

At10% removal, P99 relative L2 ranges23.51–79.53%. All variants already exceed10%
in at least one layer on the first held-out input. The declared engineering
screen (mean<=3%, P99<=10%, ideal weight saving>=3% in each text) rejects all of
them. This screen is not a model-quality guarantee.

Only37.34%/25.16%/13.63% of future expert calls are eligible in the respective
texts. Full fallback is included in the table, and covered-expert error is also
reported separately in the evidence. Byte estimates count retained gate/up rows
and touched original down blocks. They exclude metadata, mask refresh and checks.
The replay still loads complete blobs and zeros selected h values: the estimated
saving is neither measured DRAM/PCIe traffic nor measured t/s.

## Diagnostic correction and controls

The first full-mask control failed bitwise in one of2560 calls: reasoning,
position141, layer1, expert67. Its local relative L2 was0.039054%, and the routed
sum differed by0.001674%. The masked replay requantized h with
`native_quantize_q8_1`, whereas the reference retained the grouped Q8 values.

`src/kernels/neuron_trace_replay.cpp` now uses `quantize_q8_1_rows`, matching
the grouped activation quantizer. Only the separate diagnostic executable changes;
production kernels are untouched. All six measurements were repeated after this
correction. The full control is now bitwise equal across2560 calls, zero masks
produce exact zero, and the repeated10% pruning outputs are bitwise identical.
All four GU/down format pairs (17/20,16/42,18/20,18/42) are covered.

Additional checks reproduce all six mask banks, independently aggregate local
errors, check scalar/vectorized weight decoding and exercise known ranking,
rare-expert fallback and nested-mask fixtures. Full-reference outputs and h
match the older full-model replay bitwise. Historical oracle vectors2..4 are
allowed to change with the diagnostic quantizer correction.

## Evidence and use

The [complete report, runners, measurements, controls and mask banks](https://github.com/imanu86/moe-aggressive-commit/tree/8aff49f0d788f2b964bab16ecd1af9f85e4027de/docs/porto/strata_adattivo/corse_2080ti/20261004_turn_pruning)
are pinned to a research commit. The initial failed control is retained in its
audit record; the table above uses only the corrected replay. Large original
SNT/SNR captures and model files are not included; the runners require those
inputs and record their hashes alongside the result artifacts.

Build the existing `neuron_trace_replay` target with
`STRATA_BUILD_NEURON_REPLAY=ON`; see [BUILD.md](BUILD.md#turn-local-pruning-replay-correction).
No compact online kernel is promoted based on these masks. The result rejects
these two short-history heuristics, not every possible input-conditioned pruning
method. All test processes have exited.
