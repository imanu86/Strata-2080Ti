# Overlap: the drafter's chain beside the windows (branch `lab/overlap`)

Two small levers on the pipelined single-chat loop (`--pipeline-windows 2`, the MTP chain feeding the windows), both
opt-in, both leaving the default path exactly as it was when off.  They come from the measurements in
[PIPELINE2.md](PIPELINE2.md) (and from the `lab/struct-b` diary, whose numbers are quoted here and not repeated): a
fresh window costs 25,9 ms for 1,7-2,0 tokens on this
fork's two cards, the chain's highest-priority stream takes stage 1 0,6-7,5 ms when the chain is the forced one
(kind 1), and what breaks the speculative window B is its row 0 - the drafter's guess of the window before's **bonus**
token, wrong on 69% of the windows B is made for.

**Status (19 Oct 2026): code committed, not measured on this fork's hardware yet.**  The numbers below are what the two
switches have to be measured against; none of them is a result of these two levers.

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
