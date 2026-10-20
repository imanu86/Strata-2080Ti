# Overlap: the drafter's chain beside the windows (branch `lab/overlap`)

Three small levers, all opt-in and all leaving the default path exactly as it was when off: two on the pipelined
single-chat loop (`--pipeline-windows 2`, the MTP chain feeding the windows) and one diagnostic on the serial loop's drafts.
They come from the measurements in
[PIPELINE2.md](PIPELINE2.md) (and from the `lab/struct-b` diary, whose numbers are quoted here and not repeated): a
fresh window costs 25,9 ms for 1,7-2,0 tokens on this
fork's two cards, the chain's highest-priority stream takes stage 1 0,6-7,5 ms when the chain is the forced one
(kind 1), and what breaks the speculative window B is its row 0 - the drafter's guess of the window before's **bonus**
token, wrong on 69% of the windows B is made for.

**Status (20 Oct 2026): code committed, the levers still unmeasured.**  The numbers below are what the switches have to
be measured against; none of them is a result of these levers.

## Read this before quoting a number from the 20/10 forced-bench arms

Nine arms ran the night of 19-20 October (`corse_2080ti/20261004_routed_keep/DET-ov-*`, one build
`build-overlap-bc2e87a0`, engine 0.1.45, 3060 + 2080 Ti, Jump connected, 131k prompt + 1024 forced tokens, L01+L02).  **All of
them ran the serial decode loop, not the pipelined one**: the bench config (a copy of the Daily's, `det_force.sh`) does not ask
for `--pipeline-windows`, and in 0.1.45 the auto decision is CUDA-off (`on = false` in the HIP-only branch).  No arm's engine
log has a `strata pipeline classes:` line or the "`--pipeline-windows 2` is on by default" line.  So in those arms 19's branch
is never taken (the priority is only raised on a forced chain, which needs the pipeline), 20's counters never run and
`STRATA_P2_FLUSH_NOSWITCH` is never read: **three of the four levers were inert there**.  A verdict for them needs the same
B-A-B with `--pipeline-windows 2` added to the bench.

The metro the whole objective is compared against is good: `DET-ov-A0` 17,53 + 18,50 s and `DET-ov-A0b` 17,46 + 18,70 s on
L01+L02 → **0,4% dispersion** (criterion: under 3%).

## 19 - the chain's stream at the default priority (`STRATA_OV_CHAIN_PRIO=1`)

`MtpDrafter::load` gives the draft stream the highest priority whenever the chain graphs are in use, because stage 0's
next launch waits for the chain.  On a forced chain the chain runs on the 2080 Ti beside stage 1's window and costs it
0,6-7,5 ms (PIPELINE2.md), while stage 0 has ~5 ms of margin before its B is due.  With `STRATA_OV_CHAIN_PRIO=1` the
stream is created with `cudaStreamCreateWithFlags` - the default priority - so the chain arrives later and stage 1 keeps
the card.

Off (unset, or `=0`): the prioritized stream is asked for and used exactly as before.  What to measure: B-A-B on the
forced bench, and the `strata pipeline classes:` lines of the engine log - the speculative windows' ms must fall at the
same T, and stage 0's wait for B must not grow.

## 20 - the runner-up at the bonus position (`STRATA_OV_TOP2_STAT=1`, a diagnostic)

A tree draft's second branch would have to hit the drafter's *second* choice where its first one misses.  This switch
answers, per window accepted whole, what the bonus token was against the chain's top-1 and top-2 for that same position:
`outp[A.T-1]` against `oc[A.T-1]` (the row 0 a B made from this window would carry) and against the runner-up of the
draft head's logits for that step.  Each chain graph copies its own logits row to pinned host memory with a memcpy node
inside the graph, read once its step's event has passed; off, no staging is allocated, no node is in the graphs and no
line is printed - the round and step graphs are the old ones, so the tokens are the same.

At the end of a request (with `STRATA_DECODE_TIMING`):

```
strata overlap top2: bonus=top1 N1, bonus=top2 N2, altro N3 su N (runner-up non letto U, sua probabilita media p)
```

`N2/(N2+N3)` is the share a second branch would recover.  Below 0,20 the tree draft (two hypotheses on B's row 0, 8-12
days of work on the stage-0 GDN state and a second chain) is closed on this model; at or above 0,30 it is worth a real
attempt.  Deliberately slow - two host passes over 40.525 logits per chain output: nothing here belongs to a measured
path.

The 20/10 arm `DET-ov-top2` printed **no line at all** - the counters sit in the pipelined loop's window verdict, and that
bench ran the serial loop (see above).  No number yet.

## 21 - the same question on the loop this machine actually runs (`STRATA_MTP_TOP2=1`)

Upstream already ships this diagnostic: the switch, `MtpDrafter::record_top2`, and the counters in the serial loop's window
verdict - but nothing in the tree (cuda or sycl) ever calls `record_top2`, so `mtp.top2(j)` read -1 and the counters could
never count a hit.  `draft()` now calls it per draft step, at the point where that step's pick has landed in host memory and
no graph is in flight on the drafter's stream; with every step launched at once (`--spec-min-p 0`) the head's row 0 holds only
the last step's logits, so that one is recorded and the earlier ones stay -1; steps the round never produced go to -1 instead
of keeping the previous `draft()`'s value.  Off: no call, no graph changes, the same tokens.

At the end of a request:

```
strata serve: mtp top2: rejected/runner-up by depth N/N N/N N/N N/N of W windows; seconda scelta giusta N di M respinte (%)
```

The last pair answers the tree-draft question on the serial loop: of the positions where the drafter's first branch was
wrong, how often its runner-up was the token the model really produced.  Same thresholds as 20 (below 0,20 the tree draft is
closed on this model, at or above 0,30 it is worth a real attempt), on the path these two cards run by default.

## What the 20/10 serial-loop arms do say

| lever | arms (L01+L02, build-overlap-bc2e87a0) | reading |
|---|---|---|
| `STRATA_MTP_STAGE0=1` | B 17,56 + 19,16 s, A 17,53 + 19,38 and 17,42 + 18,51 s | B lost to one A by 2,2% (dispersion 0,4%) and beat the other by 0,5%: **no gain on the serial loop**, as the 19/10 single arm suggested (63-66 against 66-69 t/s).  Not a verdict for the pipelined two-stage split, where the engine asks for exactly this placement (`mtp_first`) - re-run it with `--pipeline-windows 2`. |
| `--spec-min-p 0.70` | B 17,89 + 19,23 s, A 17,47 + 19,30 and 17,51 + 19,13 s; 588/699 windows with 436 of 506 drafts accepted, against 519/604 with 505 of 705 | B was 1,0-1,3% slower than both A's, and the windows got smaller and more numerous.  The forced bench feeds the same 1024 tokens to every arm, so what an acceptance threshold really buys on free generation is not what this measures: **the community bench decides** (decode median at 128K, same tokens). |
| `STRATA_P2_FLUSH_NOSWITCH=1` | B 17,59 + 18,61 s, A 17,77 + 18,54 and 17,51 + 18,57 s (mean 0,0%) | inconclusive and inert: the switch is only read on the pipelined loop. |
| `STRATA_OV_CHAIN_PRIO=1` | B 17,53 + 18,69 s, A 17,83 + 18,64 and 17,37 + 18,54 s (mean 0,1%, that pair's own A spread 1,5%) | inconclusive and inert: with no chain in flight the priority branch never runs. |

The commands that would actually test 17, 19 and 20 - with the pipeline on - and the two diagnostic arms for 20 and 21 are
queued in the fork's work diary: `docs/porto/strata_adattivo/corse_2080ti/20261019_locale/CODA_MISURE.md`, section
"Coda 20 ottobre".

## 22 - the tree draft's second hypothesis (`STRATA_OV_TREE=1..3`, levels)

The 20/10 diagnostics said the drafter's runner-up is the token the model really produced **45,2%** of the time on the
serial loop and **29,5%** on the pipelined one - both above the 0,30 threshold that says "worth a real attempt".  So the
tree draft (two hypotheses for B's row 0 in the same pass of stage 0, keep the one whose row 0 is right) is being built
in levels, all of them behind `STRATA_OV_TREE` (0 off: today's path, byte for byte).

**Level 1 - `=1`: built and counted, nothing launched.**  Where `make_b` takes B's row 0 from the chain's pick at the
bonus position, a second window is built from `chain_top2`/`chain_top2_prob` for that same position, with the chain's
drafts after it and B's guards (no lookup or suffix windows).  Its drafts follow the *first* candidate, so its
acceptance is expected to be shorter - the model's own picks still decide at the verdict, so a wrong draft costs
acceptance, never a wrong token.  `pump0` and the launches know only A, B and D, so the windows served and the tokens
emitted stay today's.  At the verdict, per window accepted whole: which candidate is the model's own bonus token, how
many second hypotheses would clear `theta`, and both branches' mean `p_on`:

```
strata overlap tree (STRATA_OV_TREE=1): seconda ipotesi fatta N su M finestre B; finestre intere N -> prima scelta
giusta N, seconda giusta N, nessuna N; seconda sopra la soglia 0.50: N; p_on media: prima X, seconda Y
```

**Level 2 - `>=2`: the lane a launch needs exists.**  Two parities hold the two generations a pipelined loop has in
flight on a stage (A and B).  A second speculative window on stage 0 needs a lane of its own: the hand-off is per
parity (`ver -> hand[0] -> stage1.ver`), so a window on A's parity would write the hand-off stage 1 is still reading
for A.  With the switch at 2 or 3 a two-stage split gets a **third verifier per stage** (`ver_c` / `stages[].ver_c`),
its own pinned hand-off, its own `SplitDrive` and a **third GDN snapshot buffer** (`PlSnap::buf[2]`, not indexed by
`seq & 1`: the rollback of a wrong window reads its parity's slot, and overwriting it would restore the wrong state).
`capture_all`, `pl_drain`, `absorb_stats`, the watchdog (`stage 0 tree lane` / `stage 1 tree lane`) and the stage-0
occupancy tests (`a_pl_idle`, `el_gap`) cover the lane, so it is inert but consistent.  Priced up front: the reserve on
the first card goes from 160 + 96 MiB to **270 + 144 MiB** (a third arena and its graphs, a third copy of the recurrent
state), which is that much less expert cache on a 12 GB card.  If the hand-off or the third snapshot does not fit, the
lane turns itself off with its own line and the pipeline stays as it is today.

**Level 3 is not implemented** (launch, keep the winner, throw the loser away).  What is missing is a way to *give back*
a wrong speculative window: a pipelined window leaves the GDN state to the commit but does append its rows to the QSA
indexer (`tail_snap_` restores them at the commit) and advances the PLE history, and `Verifier::pl_commit_async` refuses
`n_keep < 1` - so the discard needs a new restore-only method on the Verifier (same restores, no rows appended, no GDN
step), and the two discards must run in the inverse order of their launches (each verifier's `tail_snap_` holds the state
as of *its* window's start).  Until that exists, levels 1 and 2 measure the idea and reserve its memory; they buy no
speed.  The counter-evidence is on the record too (PIPELINE2, option d2): a fork tried the tree with a 31% second branch
and gained nothing, because every speculative window added to stage 0 (~10 ms of a 3060) is paid always while the
bottleneck is stage 1 (~16 ms).
