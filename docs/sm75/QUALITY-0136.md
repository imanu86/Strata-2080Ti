# Native IQ quality regression, 2 October 2026

The declared overall gate is **FAIL** because one answer oracle failed.
The **numerical regression gate passes**. Keep these outcomes separate.
Do not relabel the failed run after the follow-up succeeds with reasoning.

Modified RTX 2080 Ti 22 GB, Ryzen 5800X/AVX2, Windows/CUDA 12.6, IQ3_XXS.
Sequential A1 (0.1.31), B (0.1.36), A2 (0.1.31), exactly 6,167 resident
expert slots in every arm. KV capacity 262,144/int8/resident 32,768;
fixed lab budget 4,600, prefill chunks 2,048, reserve setting 1,393 MiB.
IQ multi-token minimum 1, adaptive swaps and missed-expert PCIe compute off,
suffix lookup off, MTP maximum 4. The operational elastic daily is unchanged.

## Frozen-text prediction

All three arms have byte-identical full logits dumps on code (512 input
tokens), document (1,024) and chat (1,024). There are 2,557 scored next-token
positions per arm. Final input rows are checked but have no target to score.
A1/A2 variation is zero on this corpus. No numerical degradation was observed
in the candidate under these laboratory settings.

| Text | Positions | Same top-1 A1/B | Mean KL | PPL A1 | PPL B |
|---|---:|---:|---:|---:|---:|
| code | 511 | 100.0% | 0.000000 | 1.834109 | 1.834109 |
| doc | 1023 | 100.0% | 0.000000 | 1.860732 | 1.860732 |
| chat | 1023 | 100.0% | 0.000000 | 2.290217 | 2.290217 |

## Complete-answer oracles

Each arm completes all seven requests with `finish=stop`. Six are correct:
MiB/GiB conversion, stable uniqueness, ledger arithmetic, dependency order,
short record lookup and three-record retrieval at 131,072 actual prompt
tokens (records near 10/50/90%, no reused prefix). This is six of seven
selected checks, repeated three times, not an accuracy estimate on 21
independent tasks or a general capability score.

All three arms answer `{"total":92}` to three items at 17 each, two at 23
each and shipping of 8. The oracle is `3*17 + 2*23 + 8 = 105`.
Reasoning is disabled in this short-answer suite. The error predates the
0.1.36 update in the measured reference; identical frozen-text logits do
not certify correct arithmetic.

## Targeted follow-up

On the same 0.1.36 executable, verifier capacity 2 / MTP maximum 1,
suffix lookup off, the request offers **zero drafts**:

- Reasoning disabled: `{"total":92}`, normal completion, failed oracle.
- Reasoning enabled: `{"total":105}`, normal completion, passed oracle.

The failed answer occurs without drafting; this case does not establish a
drafting defect. Limits with 1, 2 or more than 3 possible MTP drafts were
not swept. Enabling reasoning changes the rendered prompt as intended;
two modes on one problem are not a general quality assessment.

## Reproduction and scope

See [VALIDATION.md](VALIDATION.md) for the predeclared policy and runner.
Five CPU harness checks pass. Three initial startup-only attempts were
retained before correcting the admission criterion, before any scored token.
The original combined verdict remains FAIL. Raw logits (about 7.6 GB) are
kept privately; their hashes, frozen IDs, metrics, outputs and engine logs
are recorded in the [research evidence](https://github.com/imanu86/moe-aggressive-commit/tree/research/ds4-iq1-subbit-tier-planner/docs/porto/strata_adattivo/corse_2080ti/20261002_quality_0136).

This validates limited numerical continuity and 131k retrieval, not 250k
quality, elastic-cache parity, stock 11 GB hardware, or the isolated upstream
PR on its current base. PR #512 stays draft. No speed record or drafting
increase is promoted from these controlled quality runs.
