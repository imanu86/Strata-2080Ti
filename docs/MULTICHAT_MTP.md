# Concurrent chats with MTP drafts on the two-card split (lab/multichat-mtp)

Branch `lab/multichat-mtp`, from `lab/multichat` (9ef0792). Goal: the `--batch N` slots with MTP drafts on the layer
split RTX 3060 (CUDA0) + RTX 2080 Ti (head and drafters), 131k context.

**Nothing on this branch has been compiled with CUDA, installed or run on a GPU.** The header-only logic
(`batch_rows.hpp`, `batch_plan.hpp`) has CPU tests (`batch_rows_test`, `batch_plan_test`, built with
`STRATA_BUILD_TESTS`).

Baseline (build be4cf1c, split, `--batch 3 --batch-mtp --conversation-cache-mib 16384 --conversation-cache-slots 4`,
batch-MTP off because of the split): one chat 48-60 t/s (solo path); 3 chats 16-21 t/s each, ~45 t/s aggregate; two
131k contexts 39-45 t/s aggregate, 14-25 per chat; batch windows of 2.98 rows, ~47 ms each, no drafts.

## Environment variables

All opt-in; unset, every path is the one of `lab/multichat` / 0.1.40.

| Variable | Default | What it does |
|---|---|---|
| `STRATA_BATCH_MTP_SPLIT=1` | off | (lab/multichat, a338706) `--batch-mtp` on a layer split with each stage on its own GPU: slot drafters on the last stage's card, carved like that stage's slot sessions, drafting from its residual rows. Not with `--batch-groups > 1`, `--peer-device` or helper caches. |
| `STRATA_BATCH_SHARE_ACTIVE=1` | off | (lab/multichat, 9ef0792) a new chat copies its prefix (up to the turn checkpoint, e.g. a shared system prompt) from a slot that is still decoding. Not with `--batch-groups`. |
| `STRATA_BATCH_MTP_DRAFTS=D` | off (1 draft) | each slot's drafter proposes up to D drafts (D <= `--spec` - 1, <= 7) and every window plans how many each slot verifies (`core/batch_plan.hpp`): every slot gets its current row, the rows left go to the draft with the highest chance of acceptance (the product of the drafter's probabilities up to it) while it raises expected tokens per ms. A shaky drafter gets no draft row; a confident one up to D. |
| `STRATA_BATCH_MTP_ROWS=R` | 8 | with `_DRAFTS`: at most R rows per window (2..8). Also bounds how many slots share a window (then they rotate). |
| `STRATA_BATCH_ROW_MS=c` | 5.5 | with `_DRAFTS`: the cost of one more row in a window, ms (the measured ~6.5 ms at T~2.3, minus ~1 ms with the multi-column MMVQ). |
| `STRATA_BATCH_BASE_MS=b` | 30 | with `_DRAFTS`: the window's fixed cost before any is measured; then it follows the windows (an average of their ms less c per row), so it tracks the context length. 30 = 47 ms at 2.98 rows less 3 x 5.5. |
| `STRATA_BATCH_MTP_MIN_GAIN=g` | 0 | with `_DRAFTS`: never verify a draft whose chance is below g (on top of the cost model). |
| `STRATA_BATCH_MTP_PSCALE=s` | 1 | with `_DRAFTS`: multiplies the drafter's probabilities (calibration: < 1 when they are over-confident). |
| `STRATA_BATCH_MTP_PARALLEL=1` | off | the slot drafters run at once (`MtpDrafter::draft_launch` / `draft_wait`, each on its own stream) instead of one after the other. Matters more with `_DRAFTS` (D steps per slot). |
| `STRATA_BATCH_GRAPHS=N` | 64 | captured batch-window graphs kept per stage (LRU, one per row layout). Per-slot draft lengths make more layouts; each new one costs a capture. |
| `STRATA_BATCH_PIPELINE=1` | off | keeps `--pipeline-windows` with `--batch` for a chat alone (the solo path); batch windows stay serial. See `docs/BATCH_PIPELINE.md` (phase A). |
| `STRATA_BATCH_PIPELINE=2` | off | phase A plus the batch windows of two slot groups overlapped across the two cards on the pipeline's two verifiers per stage, with the slots' MTP draft rows, deferred commits per group and the drafts polled beside the other group's window (`docs/BATCH_PIPELINE.md`, phase B; `STRATA_BATCH_PIPELINE_GAP_MS`, `_MIN_FREE_MIB`). Not compiled. |

The `strata batch:` timing line printed when the slots go idle counts in "avg rows" the rows KEPT (every slot's
tokens); with `--batch-mtp` a second line, `strata batch MTP:`, gives the rows the windows carried, the drafts verified
and the drafts accepted per window.

## What changed on this branch (commits)

1. `276761f` slot drafters inherit the shared drafter's `--mtp-q4` Q4_0 projections (`dense4_`, borrowed, not freed)
   and `--mtp-hnorm stream`. Before, a slot drafter ran Q8_0 projections and the all-streams norm while the solo one
   ran Q4_0 / per-stream: the same bug class as `dhead_type_` (0cc421e). Drafts only, never output.
2. `1e4504a` `batch_plan.hpp` + `batch_plan_test`: the kept-prefix rule (with one draft it equals 0.1.40's, tested
   exhaustively) and the per-slot draft planner.
3. `bdd980f` `MtpDrafter::draft_launch` / `draft_wait`: `draft()` (min_p 0) cut at its host wait.
4. `792b1ec` `batch_step` rewritten around the plan; a failed slot draft or MTP admission no longer ends the engine
   (`return 1`): that slot decodes without drafts until its next admission (`mtp_off`).
5. `83d932e` `STRATA_BATCH_PIPELINE=1`.
6. `2e8a77a` (split or `_DRAFTS` only) a slot that just ended and stays a conversation cache runs one more draft round
   over its kept rows, so its drafter K/V has no stale tail when the next turn continues from it.
7. docs: this file and `docs/BATCH_PIPELINE.md`.

## Review of `STRATA_BATCH_MTP_SPLIT` (by reading the code)

Verified by reading (not run):

* **Draft K/V and positions per slot.** Each slot drafter has its own K/V state (`load` on the last stage's card from
  that stage's slot session: `qsa_states[primary].max_cells`, `ss.k`, and `ss.block.gr` as the `gr_read` scratch, all
  on that card) and its own residual buffer (`--spec` rows). Admission copies the solo drafter's K/V up to the prompt
  (`copy_to_slot`, on the drafter's card) and `draft_first` pairs `R(p-1)` with the first token at cell `p-1`. A window
  feeds `x` at `p` and drafts at `p+1..`; `draft(T, out, p, a)` catches up cells `p..p+a` with the target's picks and
  drafts from cell `p+a+1`, which is the next window's first draft row (`p' = p + keep`). Rejected rows are not caught
  up (`T = a + 1`) and are overwritten later.
* **Acceptance and rollback per slot.** `batch_slot_keep` keeps the accepted prefix cut where the slot ends (EOS,
  stop, `max_new`, context), so the kept rows are exactly the emitted ones (the slot's `ids` and its sessions agree).
  `commit_slot_prefixes(keep)` runs on every stage through `next_`: the GDN recurrence and conv state take `keep` rows,
  the QSA indexer appends `keep` cells, `ple_prev` (stage 0) advances by `keep` tokens. The rejected rows' K/V cells
  are past the slot's position and rewritten before any read.
* **Hidden rows between stages.** `stage_batch` on every stage takes the same `rows/S/pos` with `hbase = 0`; the
  hand-off buffers hold `kVerifyMaxT` (8) rows of `[R][bo][inj]` per window, written at `hand_out_ + hbase*HB` and read
  at `hand_in_ + hbase*HB`; `batch_rows_error` refuses `hbase + S > 8` on a stage that hands rows on or takes them
  (the old check compared with the slot count, which refused two rows per slot). Every stage's verifier is initialized
  with `max_t = 8` under `--batch-mtp`. `final_R_all()` follows `next_` to the last stage, whose rows the slot drafters
  copy on their own card with `cudaMemcpyAsync` on the drafter's stream after the window's and the commit's host syncs.
* **Parking and prefix copy.** `copy_from_slot` (an idle slot or, with `_SHARE_ACTIVE`, the checkpoint of a live one)
  copies the slot drafter's K/V up to the restored length into the solo drafter; draft cells below a live slot's
  checkpoint are never rewritten (rounds write from `p-1` on). The parking cache stores the main sessions and the solo
  drafter only; a slot reaches it through `copy_from_slot` first. Interleaved prompt reads (`read_part`) run batch
  windows between chunks: the solo drafter's prefill and the slot drafters use separate states and streams and only
  share read-only weights.
* **Draft-head type.** `bind(shared)` copies `dhead_`, `dhead_type_`, `dvocab_`, `n_dvocab_` (0cc421e) and, with this
  branch, `load(shared)` copies `dense4_`, `q4_off_`, the Q4 flags and `hnorm_stream_`. Per-request settings that
  stay per drafter by design: coupled sampling (`set_draft_sampling`, never set on slot drafters: they draft greedy)
  and `set_max_drafts`.
* **VRAM.** Slot drafters' state and buffers load before the caches are sized (on the last stage's card); their bind
  (draft logits) is subtracted from the last stage's room, not CUDA0's.

Not verified (needs the GPUs):

* any of it compiling with nvcc / MSVC (only the two header tests were built, on the CPU);
* that the slot drafter's draft K/V ring (`--mtp-window` < context: a ring over pinned RAM at 131k) is saved and
  restored correctly by `conversation_kv_save/restore` when a slot's K/V is copied (the code reads the host copy for a
  ring, which should hold every cell);
* the draft quality at the tail of a resumed conversation (a cell paired with the old next token; drafts only);
* graph capture cost with `_DRAFTS` (new row layouts) and the memory of `STRATA_BATCH_GRAPHS` > 64;
* concurrent slot drafters (`_PARALLEL`) on the 2080 Ti beside nothing else (the window is synced first): expected
  safe (separate streams, states, scratch; shared weights read-only), unmeasured;
* `STRATA_BATCH_PIPELINE=1`: see `docs/BATCH_PIPELINE.md`.

## Risks

* `STRATA_BATCH_MTP_SPLIT` itself has never run: the first run may fail at start (VRAM on the 2080 Ti: N slot
  drafters at ~100-200 MiB each plus the slot sessions) or at the first batch window.
* `_DRAFTS` multiplies row layouts: with 3 slots and D = 3 up to 64 layouts per slot order; each first use captures two
  graphs per stage (tens to hundreds of ms). If the windows stutter, lower D or raise `STRATA_BATCH_GRAPHS`.
* The planner trusts the drafter's probabilities; if they are over-confident it verifies too many rows (lower
  `STRATA_BATCH_MTP_PSCALE`, or set `_MIN_GAIN`).
* Phase A puts the pipeline's second verifiers and snapshots beside the slot sessions: less expert cache on both cards.

## Local test plan (owner's machine; only once batch + MTP + pipeline are built together)

Always with `--conversation-cache-mib 16384 --conversation-cache-slots 4` (else every different context re-reads
131k). Bench script: `20261006_due_schede/multi_chat_mc.py --tag --exe --serve --extra --env --scen`. For
exactness, `tools/batch_test.py` (greedy: every slot's tokens equal its solo tokens) and `tools/parking_test.py`.

Configurations (same build, same flags otherwise: split, `--batch 3 --batch-mtp --spec 4 --mtp ...`):

| Tag | Env |
|---|---|
| base | none (batch-MTP off on the split: today's numbers) |
| split | `STRATA_BATCH_MTP_SPLIT=1` |
| split-par | `STRATA_BATCH_MTP_SPLIT=1 STRATA_BATCH_MTP_PARALLEL=1` |
| plan2 | `+ STRATA_BATCH_MTP_DRAFTS=2` |
| plan3 | `+ STRATA_BATCH_MTP_DRAFTS=3` (needs `--spec 4`) |
| pipe | best of the above `+ STRATA_BATCH_PIPELINE=1` and `--pipeline-windows 2` |
| share | best `+ STRATA_BATCH_SHARE_ACTIVE=1` (scenario with a shared long system prompt) |

Scenarios, each for every configuration:

1. **One chat alone** at 131k: t/s and TTFT. With `pipe` it must recover the solo pipeline's ~+12% over `base`.
2. **2 chats, same 131k context** (same document, different questions): per-chat t/s, aggregate t/s, TTFT of the
   second (prefix from the first: slot / parking / `_SHARE_ACTIVE`).
3. **2 chats, different 131k contexts**.
4. **3 chats, same context** and **3 chats, different contexts**.
5. **One at a time**: the same 1-3 requests sequentially, for the aggregate's reference (sum of tokens / sum of wall
   time) and the per-chat ceiling.

Record per run: aggregate t/s, each chat's t/s and TTFT, the `strata batch:` line (windows, avg rows kept, ms/window)
and the `strata batch MTP:` line (rows carried, drafts verified/accepted per window), stage profiles, VRAM free per
card at start, and any `decodes without MTP drafts` line. Expected direction (from the 4090 report, +26% at 2 clients,
+10% at 4): `split` above `base` at 2-3 chats; `plan*` above `split` only if acceptance per extra row stays above the
cost model's break-even (~0.25 at 3 slots and 47 ms windows).

**Text checks**, every reply of every run: coherent, on topic, no loops or repeated paragraphs, no truncation before
the requested length, no reply that switches to another chat's content (a slot mix-up). With greedy settings, compare
each chat's text with the same prompt run alone (`tools/batch_test.py`): identical tokens are expected for plain
batching; with draft rows a window carries more rows, so a late divergence can be numeric (multi-row kernels), while
an early one, or text from another chat, is a bug - note the first differing position. With sampling, the row's draw
is Philox(seed, position), as solo, so the same holds.
