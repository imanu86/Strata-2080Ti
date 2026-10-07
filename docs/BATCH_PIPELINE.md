# Batch slots and `--pipeline-windows` on a two-card layer split (lab/multichat-mtp)

Status: two opt-in steps, `STRATA_BATCH_PIPELINE=1` (phase A, a chat alone) and `=2` (phase B, the slot groups). Nothing here has been compiled or run on a
GPU yet. Target machine: RTX 3060 (sm_86, CUDA0, layers 0..K-1) + RTX 2080 Ti (sm_75, the rest and the head), 131k
context.

## Why `--batch` turns the pipeline off today

`generate.cpp` decides `--pipeline-windows` before any cache is sized and turns it off with `--batch`
("not with --batch slots"). What the pipeline assumes, and what `--batch` changes:

| Pipeline assumption (`--pipeline-windows 2`) | What `--batch` brings |
|---|---|
| Two verifiers per stage, by window parity (`ver`/`ver_b`, `stages[0]->ver`/`ver_b`), sharing the stage's one session `ss` and its stream; stage 1 never runs a speculative window. | Slot sessions (`bslot_ss`) given to the even verifiers only (`init_slots`); batch windows (`run_slot_rows`) chain through `ver -> stages[0]->ver`. |
| The drafter `mtp` is bound to its own row buffer `pl_mtp_R`, teacher-forced chain graphs (`set_force_capture`, max_t 8), its source rows copied from the parity that ran. | The slot drafters borrow `mtp`'s weights and head; admission copies `mtp`'s K/V into a slot drafter and starts it from `ver.final_R_all()`. |
| GDN snapshots for the speculative window's rollback (`pl_snap2`), one conversation in `ss`. | Admissions, parking, `copy_to_slot`/`copy_from_slot` read and write `ss` and every stage's session between requests. |
| INVARIANT: nothing pipelined is in flight outside the pipelined prompt read and decode loop (both drain at their end). | Batch windows run between requests and between the chunks of a prompt read, never inside the pipelined loops. |
| `--batch-groups G`: a different pipeline - slot groups flowing through the stages with `batch_launch`/`batch_poll`, every row kept (the commit is launched right behind the window). | |

The invariant is what makes phase A possible: when a batch window runs, both stages and the drafter are idle, and the
even verifiers hold no pipelined state.

## Phase A (implemented, opt-in): the pipeline for a chat alone

`STRATA_BATCH_PIPELINE=1` with `--batch N` and `--pipeline-windows 2`:

* the gate keeps `--pipeline-windows` on (still off with `--batch-groups > 1`: two pipelines over the same verifiers);
* the server's solo path (a chat alone, `GEN`, and a request that goes back to solo when the others end) decodes
  pipelined, as without `--batch`: the measured single-chat gain (~+12%) stays;
* batch windows (2+ chats) stay serial, on the even verifiers, exactly as with `--pipeline-windows 0`;
* an admission (`BGEN`) decodes its one solo token serially (`pl_want` requires `admit_slot < 0`), so
  `ver.final_R_all()` is the even verifier's residual of that window, which the slot drafter's `draft_first` reads.

Cost: the pipeline's second verifier on each card and the two GDN snapshots (~160 MiB + 2 x snapshot, see the
`pipe_first` reserve) beside the slot sessions. On the 11 GB 2080 Ti at 131k this comes out of the expert cache; if a
card no longer fits, the engine says so at start (`--pipeline-windows: ... does not fit`) and the decode stays serial.

Not verified by reading alone (to test): the pipelined prompt read (`read_windows_pl`, short re-reads) followed by a
slot admission; the elastic tail (`STRATA_PIPELINE_ELASTIC`, default on with two cards and the pipeline) beside the
batch windows' `elastic_step`; a solo request STOPped mid-pipeline and continued as `BGEN` (it relies on the drain at
the loop's end).

## Phase B (implemented, opt-in, NOT compiled or run): batch windows that overlap the two cards

`STRATA_BATCH_PIPELINE=2` with `--batch N` (N >= 2), `--pipeline-windows 2`, a two-stage layer split on two GPUs, the
MTP drafter, and no `--batch-groups` (`=2` implies everything `=1` gives: the solo path stays pipelined). With 2-3
chats, card 0 runs one group of slots while card 1 runs the other, each group carrying its slots' MTP draft rows.

### Shape (as designed, now in `generate.cpp` under `pb`)

Two slot groups, A and B, on the pipeline's two verifiers per stage: group A on `ver -> stages[0]->ver` (the first
hand-off), group B on `ver_b -> stages[0]->ver_b` (`hand_b`). A slot belongs to one group at a time. One window per
stage at a time. A group's round:

```
stage 0 window -> stage 1 window (the picks) -> verdict: batch_slot_keep per slot, BT/BDONE out,
batch_commit_async on both stages (queued on each stage's stream, no host wait),
the slot drafts launched (draft_launch, one stream each) -> drafts polled (draft_poll) -> ready for stage 0 again
```

```
card 0: [A_k s0][B_k s0][A_k commit][A_k+1 s0][B_k commit][B_k+1 s0] ...
card 1:         [A_k s1 + head][A_k commit][A drafts | B_k s1 + head][B_k commit][B drafts | A_k+1 s1] ...
```

No speculation between the groups (independent conversations): nothing to roll back. The per-slot output is the
serial batch window's: the same `batch_slot_keep` rule, the same rows at the same positions, each row the single-token
window's arithmetic (docs/BATCHING.md); which slots share a window differs, which matters only through the multi-row
CPU kernels (`STRATA_IQ_MT_MIN=1` for exact comparisons, as for plain batching).

### What is implemented

| Piece | Where |
|---|---|
| Two verifiers per stage hold the slots: `init_slots` on `ver_b` and `stages[0]->ver_b` with the same slot sessions; the odd verifiers are sized at the batch row capacity (8 with `--batch-mtp`, else `max(--spec, --batch)`) instead of `--spec`, and get the batch graph limit (`STRATA_BATCH_GRAPHS`). | `generate.cpp`: after the even verifiers' `init_slots` |
| Explicit row layout per launch: `Verifier::batch_launch_rows(rows, S, hbase, tok, pos, commit_behind)` (the pipeline's `batch_launch` is its `rows[t] = base + t`, commit-behind case). Each parity has its own hand-off, so both groups use `hbase = 0` and up to 8 rows. | `verify.hpp/.cpp` |
| Deferred commit per group: `Verifier::batch_commit_async(keep)` - `commit_slot_prefixes` without the host wait and without chaining to the next stage (the pump drives each stage); `ple_prev` advances host-side at the commit. `stage_batch` no longer writes the "keep all" words for a deferred window (the previous commit of that verifier may still be queued on the shared stream and reads them when it starts). | `verify.cpp` |
| Completion by event, not by stream: the two verifiers of a stage share its stream (`set_stream`), so `batch_poll` of a `batch_launch_rows` window polls an event recorded after it (`ev_done_`); the stream query only flushes WDDM. Sampled slots draw on a private stream (`samp_cs_`) after that event: a sync on the shared stream would wait for the other group's window, whose doorbells this thread serves (deadlock). | `verify.cpp` `batch_poll`, `sample_rows` |
| `MtpDrafter::draft_poll`: the launched draft's state without waiting (its own stream), so the drafts of one group run beside the other group's stage-1 window and are collected when they land. | `mtp.hpp/.cpp` |
| The pump (`pb_pump`): poll the stages, verdict on the last stage, start the waiting group on stage 1, start a ready group on stage 0 (the longer waiting first), poll the drafts. `pb_drain` finishes every round and syncs the four verifiers' streams (the deferred commits) before any request line but `BSTOP`, before a serial window, and before an elastic step. | `generate.cpp` |
| Groups balanced by rows (`core/batch_groups.hpp`, CPU test `batch_groups_test`): an admitted slot joins the group carrying fewer rows (`batch_group_join`, ties to A); when one group has emptied and the other holds two or more slots, the next stage-0 launch shares the slots out again by falling rows, greedy to the lighter group (`batch_group_split`), so two groups keep flowing with 3 chats (2+1) or 4 (2+2). | `batch_groups.hpp`, `pb_pump` step 4 |
| One slot alone: the pump still runs it (no overlap: stage 0 -> stage 1 -> drafts, serial through the poll); with nothing in flight and fewer than two live slots the loop takes the serial `batch_step` path, as phase A, and the server moves a request left alone to the solo path, which the pipeline runs. | main loop |
| Admission / exit mid-way: `BGEN` and every other line drain the pipeline first (as `--batch-groups`), then the usual admission (serial first window, `copy_to_slot`, `draft_first`); the slot's sampling is set on both parities and the pump assigns its group at the next turn. A slot that ends in a verdict keeps its conversation cache (no pad rows, unlike `--batch-groups`) and, with `_SPLIT` or `_DRAFTS`, gets the tail catch-up draft round. `BYIELD` and the interleaved prompt read (`read_part`) run serial windows between chunks, on a drained pipeline. | `pb_verdict`, BADM block |
| Elastic cache: only at gaps. Every `STRATA_BATCH_PIPELINE_GAP_MS` (2000; 0 = never) the loop drains the pipeline, runs `elastic_step(false)` (which keeps its own period) and the pump refills. `apply_pending(false)` runs before each stage-0 launch, as `--batch-groups`' pump does. | main loop |
| VRAM: the odd verifiers' slot buffers are small (commit staging, indexer tail snapshots); the batch graphs of each new row layout come later from the free VRAM. A floor on each card's free VRAM (`STRATA_BATCH_PIPELINE_MIN_FREE_MIB`, 192) or a failed carve turns phase B off at start with a message, and the windows stay serial (phase A). | `generate.cpp` |
| Log at idle: `strata batch pipeline: 2 groups, rows per window A x (n windows) / B y (m windows), ms per window stage 0 a / stage 1 b, drafts c ms per round, overlap p% (stage 0 busy q%, stage 1 busy r% of W ms); K rows kept = R rows/s`, then `strata batch MTP:` (drafts verified / accepted per window) and the GPU stage profiles per stage and group when `STRATA_VERIFY_PROFILE=1`. Overlap = time both cards were busy / time at least one was (`BatchOverlap`). | `pb_pump` |

The cost model of `STRATA_BATCH_MTP_DRAFTS` (base + row ms) is shared with the serial windows (`bp_cfg`); a pipelined
round measures it as stage 0's time + stage 1's time, the serial window's cost.

### Environment variables (phase B)

| Variable | Default | What it does |
|---|---|---|
| `STRATA_BATCH_PIPELINE=2` | off | phase A plus the two-group pipeline of the batch windows. `=1` phase A only. |
| `STRATA_BATCH_PIPELINE_GAP_MS=ms` | 2000 | with the elastic cache: how often the pump drains for an `elastic_step` (0 = never while slots decode). |
| `STRATA_BATCH_PIPELINE_MIN_FREE_MIB=n` | 192 | the free VRAM each card must have after everything else is allocated, else phase B stays off. |
| `STRATA_BATCH_GRAPHS=N` | 64 | (existing) captured batch graphs per verifier; with two verifiers per stage, twice the layouts are captured in all. |
| `STRATA_BATCH_MTP_DRAFTS`, `_ROWS`, `_PSCALE`, `_MIN_GAIN`, `STRATA_BATCH_ROW_MS`, `STRATA_BATCH_BASE_MS` | (see MULTICHAT_MTP.md) | apply per group window (each group plans its own rows within `_ROWS`). `STRATA_BATCH_MTP_PARALLEL` is irrelevant here: the groups' drafters are always launched together and polled. |

### Not verified by reading alone (nothing of this has been compiled)

* nvcc / MSVC accept it (new code in `verify.cpp`, `mtp.cpp`, `generate.cpp`); only `batch_groups_test` was meant to
  build on the CPU.
* the shared stream: a graph launch (window or commit) onto a stage's stream while the other parity's window with
  host-spinning doorbell kernels runs on it is an ordinary enqueue, as the solo pipeline relies on; the deferred
  commit's reading of `h_commitb_` through `copy_i32_from_mapped` happens when the commit graph starts, which is why
  `stage_batch` skips the staging for a deferred window;
* the first windows of every new row layout on the odd verifiers capture their graphs (a sync of the stage's stream,
  tens to hundreds of ms, during which the other card's window waits for the host): a stutter at the start of each
  multi-chat burst, worse with `_DRAFTS`;
* `capture_all` on the odd verifiers now captures 8 window sizes (not `--spec`): more VRAM and a longer first
  `pl_prepare`; the `pipe_first` reserve (kPipeWindowMib) was sized for `--spec` rows;
* the drafters on the 2080 Ti beside the other group's stage-1 window: separate streams, states and scratch, weights
  read-only; the gain assumes the card interleaves them;
* a slot that fails a draft after `draft_poll` returned an error keeps `draft_live_` set in its drafter until the next
  admission's `idle()`; it decodes without drafts meanwhile (as the serial path on a failed `draft_wait`);
* the elastic gap: a drain every 2 s costs about one pipeline fill (a window or two of overlap); with
  `STRATA_PIPELINE_ELASTIC` the solo pipeline's `el_gap` logic is NOT used here (the batch path drains instead).

### Test plan (owner's machine; `multi_chat_mc.py`)

Always `--layer-split 16 --pipeline-windows 2 --batch 3 --batch-mtp --spec 4 --mtp ... --conversation-cache-mib 16384
--conversation-cache-slots 4`, `STRATA_BATCH_MTP_SPLIT=1`. Bench: `20261006_due_schede/multi_chat_mc.py --tag --exe
--serve --extra --env --scen`.

| Tag | Env | Expect |
|---|---|---|
| pipeA | `STRATA_BATCH_PIPELINE=1` | today's numbers (39-45 t/s aggregate at 2-3 chats, 65-72 solo) |
| pipeB | `STRATA_BATCH_PIPELINE=2` | the `strata batch pipeline:` line with overlap > 0; aggregate above pipeA at 2-3 chats |
| pipeB-par | `+ STRATA_BATCH_MTP_PARALLEL=1` | same as pipeB (the flag is moot here): a control |
| pipeB-plan2 | `+ STRATA_BATCH_MTP_DRAFTS=2` | more rows per group window; acceptance per extra row above ~0.25 |
| pipeB-nogap | `+ STRATA_BATCH_PIPELINE_GAP_MS=0` | overlap a little higher; VRAM free stable (no elastic steps while decoding) |

Scenarios, each tag: (1) one chat alone at 131k - t/s and TTFT equal to pipeA (the solo pipeline); (2) 2 chats, the
same 131k context; (3) 2 chats, different 131k contexts; (4) 3 chats same context, 3 chats different contexts; (5) the
same requests one at a time, for the aggregate's reference. Record aggregate t/s, each chat's t/s and TTFT, the
`strata batch pipeline:` line (rows per group, ms per stage, overlap), the `strata batch MTP:` line, any `group X takes
n of group Y's slots`, any `decodes without MTP drafts`, VRAM free per card at start and at the end.

**Text checks** every reply: coherent, on topic, no loops, no truncation, no reply with another chat's content (a slot
mix-up would be a group/hand-off bug: the two groups write disjoint hand-offs, `hand` and `hand_b`). Exactness:
`tools/batch_test.py` with `--extra "--layer-split 16 --pipeline-windows 2 --pcie-frac 0 --adapt-every 1000000"` and
`STRATA_BATCH_PIPELINE=2 STRATA_BATCH_MTP_SPLIT=1`: every slot's greedy tokens equal its solo tokens, as for plain
batching; with draft rows a late numeric divergence is possible (multi-row kernels), an early one or another chat's
text is a bug - note the first differing position and whether the slot was in group A or B.

**What to look at first** (in this order): the start log (`STRATA_BATCH_PIPELINE=2: the batch windows of two slot
groups overlap` or the `is off` reason); then 2 chats, same context: do the first group windows complete at all (a
hang here = the shared-stream / event path or the deferred commit's staging), do both chats' texts stay their own,
and is the overlap above 0; then the aggregate against pipeA.

### Risks

* Never compiled: the first build may fail on the new `Verifier` members or the lambdas' captures in `generate.cpp`.
* A deadlock on the shared stream if some sync was missed: the symptoms are a `verify batch: timed out at layer` after
  20 s or a frozen pump; `STRATA_VERIFY_TRACE=1` and the watchdog's `diag` name the verifier.
* Wrong `keep` reaching a commit (a staging race) would show as a slot's text repeating or skipping a token right
  after a draft rejection; the exactness test catches it.
* VRAM: two verifiers per stage capture twice the batch graphs; on the 11 GB 2080 Ti at 131k the expert cache shrinks
  by that much (watch `strata elastic:` and the free MiB at the end of a run). Lower `STRATA_BATCH_GRAPHS` if it
  thrashes, or stay at `=1`.
* With one group alone much of the time (a chat that answers while the other reads a long prompt) phase B gains
  nothing over phase A and costs the captures; the server's solo path handles the one-chat case.
