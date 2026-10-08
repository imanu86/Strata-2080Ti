# Experimental two-card MTP batching

This opt-in port restores the necessary local historical `be4f623` integration
onto current main. The normal defaults remain unchanged. Source attribution:
OPUS/Claude Opus 5.5 commits `a338706b`, `276761fb`, `1e4504a`, `bdd980f7`,
`792b1ec5`, `83d932e5`, `2e8a77a2`; FABLE commits `9f03e408`, `ba923ee7`,
`a418fe0c`, `1a2e42f4`. Current integration and laboratory provenance fix:
GPT-6.1 Sol. This is experimental qualification, not a Daily promotion.

`STRATA_BATCH_MTP_SPLIT=1` permits `--batch-mtp` on an actual layer split,
with per-slot draft K/V and residual buffers on the final-stage GPU. Slot weights,
draft head, optional Q4 projections and normalization mode are borrowed from the
shared drafter, whose ownership is preserved.

`STRATA_BATCH_PIPELINE=1` preserves the single-chat pipeline while batches remain
serial. `=2` pipelines two groups on two verifier parities: stage0(B) can overlap
stage1(A). The path uses deferred commits, nonblocking draft polling, sampling
completion on its own event/stream, and a drain before admissions, serial windows,
cache resizing and shutdown. It differs from target-only batch-groups. The tested
configuration has two slots, one per group, and one MTP draft: there is no
cross-slot expert sharing within a window. No multi-draft or split sweep was run.

## Native SSD laboratory bank and V2 continuations

`STRATA_LAB_MULTICHAT_SSD_SWITCH` is an absolute control-file path containing two
lines: `0`, then an absolute native prefix-file path. It requires split16 or22,
pipeline2, batch2, original MTP/spec4/minp0.5, greedy seed73 without penalties,
and no other laboratory/helper treatment. FORCE_IDS/FORCE_WINDOWS are forbidden.
The actual measured configuration uses split16, first16 layers on RTX3060,
last32 on RTX2080Ti, elastic/adaptive placement enabled and primary reserve2048MiB.
Split22 remains unmeasured; it needs restart and a compatible prefix identity.

A new HTTP request always clears only solo/RAM-prefix metadata after draining and
restores the native131072-token SSD prefix. Active/cached slots remain intact.
The unchanged native identity, checksum, full MTP prefill, true next-token lookahead
and per-stage snapshots are required. Root bounds are checked before indexing.
Cold creation is separate. Two files are needed if prompts differ in prefix or
lookahead; the tested L01/L02 share both, with next-token39737.

V1 incorrectly repeated the SSD reset on internal STOP-to-BGEN and BGEN-to-solo
legs. V2 adds internal LAB-only request metadata (ID, original prompt length,
total budget and emitted count). Public HTTP sampling fields cannot supply it.
An internal continuation may use normal live/cached RAM state only when the full
processed prefix and last actually emitted token match, the source is idle/full
(not an active or partial slot), request identity/budget agree and MTP is idle.
The BGEN budget is its full admission budget, not the transformed GEN1 budget.
A short checkpoint, wrong parent, changed prefix/token, or invalid budget is fatal.
Normal copy/drain/KV restoration and verification math are unchanged. These
predicates have CPU mutation coverage; that alone does not qualify every GPU branch.

## Evidence and limits (2026-10-08)

Private CUDA75/86 builds V1 and V2 passed. V1's existing exact-settings short
safety test passed two solo and two batched64-token generations with both token
sequences identical. The performance configuration intentionally differs from
that fixed-cache safety configuration. No new64-token test was run for V2.

V2 warmup passed two native SSD restores and an `exact_live` continuation.
B1 completed1024 FREE tokens per client:2048 /41.859s =48.9262 aggregate tokens/s,
including HTTP TTFT. Client TTFTs were6.250s and12.219s. It used two SSD initial
admissions and one exact-live RAM continuation. Native groups had588/589 windows,
stage0/1 averages15.1/24.0ms,62% overlap,62%/98% busy and73% MTP acceptance.
The native inner70.7 rows/s is a different scope from HTTP throughput.

The subsequent hot serial A2 completed L01, but L02 stopped progressing at691
received tokens while heartbeats continued from elapsed26s through138s. A foreign
`dotnet` guard then interrupted the run; its transient process command/identity
could not be recovered. The last native event was a completed169-slot elastic
shrink. Temporal proximity does not prove shrink caused the stall. The exact
blocking wait is unlocalized. The candidate commit/chain waits predate this port;
it is unknown whether new states expose an existing issue or the cause is elsewhere.
A2 has no valid aggregate TPS; B2 was not run.
`exact_slot` was not exercised on GPU, so demotion-branch qualification is pending.
Cleanup left no owned processes; Daily and the reused prefix/control were unchanged.

Use native completion usage and client full wall/TTFT for comparisons. SSE chunks
are not token receipts; steady token TPS and token-p95 are unknown. HTTP
`native_timings` describes the last internal leg, not the whole HTTP prompt:
individual native logs are needed for decomposition. Stage/host/copy timers overlap
and must not be added. The HTTP120s parameter is a socket-read timeout; heartbeat
traffic can keep a stalled stream alive, while the outer absolute deadline and
owned cleanup remain bounded. Adaptive cache placement and FREE outputs can vary.
There is no completed matched serial comparison and no causal gain claim versus
V1, historical59 full-wall TPS, or single-chat70 decode TPS.

See [the compact V2 report](sm75/multichat-split16-free-http-v2-20261008.json).
No full prompts, raw response texts or private configurations are published.
