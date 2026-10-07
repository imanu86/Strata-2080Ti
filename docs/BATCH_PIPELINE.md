# Batch slots and `--pipeline-windows` on a two-card layer split (lab/multichat-mtp)

Status: design, plus one opt-in step (`STRATA_BATCH_PIPELINE=1`, phase A). Nothing here has been compiled or run on a
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

## Phase B (design only): batch windows that overlap the two cards

Goal: with 2-3 chats, card 0 runs one group of slots while card 1 runs another, each group carrying its slots' MTP
drafts. Today with a split each batch window goes card 0 -> card 1 -> drafts, so each card idles about half the time.

### Shape

Two slot groups, A and B, by parity, on the pipeline's two verifiers per stage: group A on `ver`/`stages[0]->ver`,
group B on `ver_b`/`stages[0]->ver_b`. Steady state:

```
card 0: [A_k stage 0][B_k stage 0][A_k+1 stage 0][B_k+1 stage 0] ...
card 1:              [A_k stage 1 + head][drafts A][B_k stage 1 + head][drafts B] ...
```

No speculation across windows is needed (unlike the solo pipeline): the groups are independent, each one's next window
waits only for its own verdict and drafts, which land while card 0 runs the other group.

### What must change

1. **Two verifiers per stage hold slots.** `init_slots` on `ver_b` and `stages[0]->ver_b` too, each for its group's
   slots (the slot sessions are already per stage; a slot belongs to one parity). Reason: the batch commit graph reads
   the window's GDN intermediates (`qkv_L_`, `h_L_`, `gate_L_`, `beta_L_`, the indexer rows) from the verifier's
   scratch by window row. With one verifier per stage, group B's window overwrites them before group A's deferred
   commit. One verifier per group keeps them, as the solo pipeline keeps window K's commit inputs in its parity.
2. **Deferred commit per group.** `batch_launch` queues the commit right behind the window (every row kept). With
   drafts the kept prefix is known only after the last stage, so stage 0 commits group A after A's verdict, queued
   on card 0's stream behind group B's window (different slots: independent states). The verifier's commit staging is
   already per slot (`h_commitb_ + slot * CB`, `tail_snap_b_ + slot * nQ`); `last_rows_`/`last_t_`/`row_base_` are per
   verifier, which is fine once each group has its own verifier (point 1). The host side of the PLE history
   (`ple_prev`) moves to the commit, as `commit_slot_prefixes` does.
3. **Grouped rows in `batch_launch`.** It takes `base, S` and builds `rows[t] = base + t` (one row per slot). It needs
   the explicit `rows[]` of `run_slot_rows` (a slot's current row + its drafts), and `hbase` as a ROW offset in the
   hand-off. Two groups in flight write disjoint hand-off blocks: either rows(A) + rows(B) <= 8 (one 8-row hand-off,
   `batch_rows_error` already checks `hbase + S <= 8`) or one hand-off per parity (the pipeline already allocates
   `hand_b`; then each group may use all 8 rows).
4. **The drafts on card 1 beside the other group's window.** The slot drafters live on the 2080 Ti (the head's card).
   Group A's drafts run while card 1 starts group B's stage 1: the drafter streams at the highest priority (as
   `STRATA_MTP_PRIORITY` does for the solo chain) and `draft_launch`/`draft_wait` (this branch) so they overlap.
5. **Driver loop.** A `pump` like `--batch-groups`' (`batch_poll` per stage, start the waiting group on the free
   stage), plus per group: verdict -> `batch_slot_keep` -> commit on every stage -> emit -> drafts -> plan
   (`batch_plan_drafts`) -> launch on stage 0. Admissions and BYIELD drain the pump first (as `pipe_drain` does).
6. **Conversation cache.** `--batch-groups` today marks pipelined slots not reusable (`sl.cached = false`, pad rows of
   idle slots write their state). With explicit rows per group there are no pad rows (only active slots get rows), so
   a pipelined slot can stay a conversation cache.

### Expected gain and when not to bother

Per window both cards do the same work as now; the gain is the overlap. With stage times s0 (3060) and s1 (2080 Ti +
head + drafts), two groups give about `2 * tokens_per_group / max(2 * s0, 2 * s1)` per round against
`tokens / (s0 + s1)` serial: up to ~2x when s0 ~ s1, less when one card dominates. Two caveats: a group carries half
the rows, so the per-window fixed cost (expert reads, the layer chain) is paid twice per round; and with 2-3 chats one
group may have a single slot. First measure what exists: `--batch 4 --batch-groups 2` (plain rows, no MTP) against
`--batch 3 --batch-mtp` + `STRATA_BATCH_MTP_SPLIT=1` at 2 and 3 chats. If the groups do not beat the serial MTP windows
there, phase B will not either.

### Opt-in

Phase B would sit behind `STRATA_BATCH_PIPELINE=2` (phase A is `=1`), with `--batch-groups 2 --batch-mtp`, a layer
split of exactly two stages on two GPUs, and no helper caches.
