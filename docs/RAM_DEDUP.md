# RAM de-duplication of the expert arena (`STRATA_RAM_DEDUP=1`)

Lab option, off by default, `--serve` only.  The resident expert arena (`ArenaExpertSource`,
`PinnedArena`) holds every expert of the model; the VRAM tiers hold about half of them again.
On the reference PC (Windows 11, 96 GB, RTX 3060 12 GB + RTX 2080 Ti 22 GB, `--layer-split 16
--trim-stage-weights --pipeline-windows 2`, elastic cache, `STRATA_ARENA_PIN_GIB=8`,
`STRATA_NO_LARGEPAGES=1`) the arena is 39.97 GiB, the two caches hold 12,944 of 24,576 experts
(~21.3 GiB), so ~21 GiB of RAM duplicate what the GPUs already have.  The PC is the owner's
desktop, not a server: the engine has to give at least 10 GB back.

With the option on, the RAM pages of an expert that sits in a VRAM slot are decommitted in
place, and brought back - from the slot, by a device-to-host copy - before the slot is
overwritten or unmapped.  The generated text is unchanged bit for bit: a slot was filled from
the arena, so the bytes that come back are the bytes that left.  Only where they sit changes.

## Who reads the RAM copy of a resident expert, and when (the map)

`host_res[layer * n_expert + expert]` is the one truth about residency (slot number, or
`kNotResident`).  The arena is read through `ExpertSource::blob()` = base + offset.

| Reader | Where (`src/program/generate.cpp`) | What it reads |
|---|---|---|
| Startup fill | cache fill from the profile, then `release` loop near "VRAM-held experts handed back" | every expert once (H2D); on Linux + `STRATA_ARENA_MMAP` the pages are then `MADV_DONTNEED`; **on Windows `release` was a no-op** - the whole arena stayed committed |
| CPU pool (verify windows, prompt path) | `expert_pool_dispatch*` through `drive.d.src` | **non-resident** experts only; a resident expert is computed on the GPU from its slot |
| PCIe share of the misses | `device_alias` (registered prefix only) | non-resident experts of the pinned prefix |
| Prompt path's loan (`lend` / `refill`) | `pf_parts[i].first ..` slots of each participant's cache | a lent slot's expert is **streamed from RAM during the prompt** and copied **from the arena** back into the same slot after it (`refill_issue` -> `blob`) |
| Adaptive tier (`adapt`, blocking; `closed_refill`) | swaps `in` (not resident, from RAM) into the slot of `out` | `out`'s RAM copy is **not read**: the slot is simply overwritten.  `out` is then a miss and the pool reads it from RAM |
| Adaptive tier, asynchronous (`--adapt-async`) | needs the file tier's RAM copy (`src.complement_ready()`) | never runs with the arena |
| Elastic cache shrink (`el_shrink`, `el_gap` kind 1) | tail experts hotter than core ones move by a device copy; the rest are dropped and the chunks unmapped | the dropped experts become misses: read from RAM |
| Elastic cache grow (`el_grow`) | fills new tail slots from `blob` | H2D from the arena |
| Elastic K/V (`kvg_ensure` / `kvg_trim`) | slots below the loan give their chunks to the K/V; later refilled from `blob` | the evicted experts become misses; the refill reads the arena |
| CPU experts (`--cpu-experts`, lab) | never resident | RAM always |

So the RAM copy of a resident expert is read in exactly two situations: a lent slot during a
prompt, and the moment an expert leaves VRAM (it becomes a miss).  Everything else reads the
slot.  That is what makes the de-duplication possible: keep the copy where a slot can be lent
or shrunk, and for every other slot copy the bytes back *before* the slot is reused.

## Design

`include/strata/core/ram_dedup.hpp`, `src/core/ram_dedup.cu` (class `RamDedup`), owned by
`ArenaExpertSource` (`dedup_arm`, `dedup()`); the driver's side is in the serve path of
`generate.cpp` (`dd_*` lambdas).

* **Drop** (`RamDedup::drop`): the whole 4 KiB pages inside the expert's blob are
  `VirtualUnlock` + `VirtualFree(MEM_DECOMMIT)` (Windows) or `munlock` + `madvise(MADV_DONTNEED)`
  (Linux).  The address range stays reserved: `blob()` is still base + offset.  The partial
  pages at the blob's edges, shared with its neighbours, stay.  Experts inside the arena's
  CUDA-registered prefix (`STRATA_ARENA_PIN_GIB`, 8 GiB on the reference PC) are never dropped:
  decommitting registered pages is not allowed.  A large-page arena refuses the whole option.
* **Who keeps a copy** (`dd_keep`): per cache, the slots from the prompt path's `first` lendable
  slot on and, on CUDA0 with the elastic cache, from `el_core` on (the tail that may shrink).
  Those are the structural minimum - the "quota ridondante".  `STRATA_RAM_DEDUP_KEEP_GIB` above
  it also keeps the lowest (hottest: the profile fills them first) slots of every cache, the extra
  split by the caches' sizes.  A value below the minimum is logged and the minimum applies.
* **Evict** (`RamDedup::evict`, called through `dd_evict` at every site that overwrites or
  unmaps a slot): when the arena lacks the expert, an event is recorded on the caller's stream
  (after the work it holds now), the dedup's own stream for that device waits for it, copies
  the slot into a page-locked bounce ring (`cudaHostAlloc`, `STRATA_RAM_DEDUP_BOUNCE_MIB`,
  default 512), records a second event, and the caller's stream waits for that one.  No host
  wait: the overwrite is queued behind the copy by the events.  Sites: `adapt` (both the
  per-copy and the `copy_blobs` batch), `closed_refill`, `el_shrink` (moved and dropped),
  the pipelined `el_gap` shrink (plus `sync` before the unmap), `kvg_ensure` (victims moved
  and slots given to the K/V), and `lend` (a lent slot whose expert lacks a copy, after the
  lend region moved with the elastic tail: copied back and waited for before the buffers are
  laid out).
* **Land** (`drain`, `ensure`): once a copy's event has completed, its pages are recommitted
  (`VirtualAlloc(MEM_COMMIT)`, re-locked best effort) and the bounce slot is copied into the
  arena.  `ArenaExpertSource::blob()` checks one atomic per call: a present expert costs
  nothing more; a non-present one is landed first (a wait on its event) - the pool thread
  that needs an evicted expert in the same window pays that latency once.
* **Last resort** (`reread`): an expert that is absent with no copy in flight (a path that
  did not call `evict`, or a copy CUDA refused) is read again from the model file,
  unbuffered (`experts.bin` at its offset, or the three GGUF role slices), and counted.  It
  should read 0; a non-zero count names a missed eviction site.
* **Sweep** (`dd_sweep`): at arming and at the end of every request (nothing in flight) every
  resident expert outside the kept regions is dropped - what the K/V, a shrink or a closed
  refill moved without going through a release site.

What is NOT covered: a peer GPU's expert tier (`--peer-device`), remote experts
(`--expert-cache-remote`), the file tier (`--mmap-experts`) and the non-serve `generate`
command.  The option refuses to arm in those cases and says why.

## Variables

| Variable | Meaning | Default |
|---|---|---|
| `STRATA_RAM_DEDUP=1` | arm the de-duplication in `--serve` | off |
| `STRATA_RAM_DEDUP_KEEP_GIB=N` | redundant quota (GiB of resident experts that keep a RAM copy); below the structural minimum the minimum applies | unset = lend regions + elastic tail |
| `STRATA_RAM_DEDUP_BOUNCE_MIB=N` | the page-locked bounce ring for the copies back | 512 |
| `STRATA_RAM_LOG=1` | the private-RAM breakdown line at every request even without the dedup (the "before" figure) | off |

## Log lines

```
strata ram dedup: armed: arena 39.97 GiB, prefisso registrato 8.00 GiB mai liberato, quota ridondante 7.42 GiB (prestito + coda elastica 7.42 GiB; STRATA_RAM_DEDUP_KEEP_GIB non impostata), ring D2H 512 MiB
strata ram dedup:   cache 0: 3794 slot, copia RAM tenuta per gli slot < 0 e >= 1217
strata ram dedup:   cache 1: 9150 slot, copia RAM tenuta per gli slot < 0 e >= 6980
strata ram dedup: prima passata in 180 ms: 1420 esperti nel prefisso registrato tenuti
strata ram dedup: avvio: arena 27.1 GiB (era 39.97), 7700 esperti senza copia RAM, D2H 0 esperti in 0.0 ms, riletti da file 0 (totali: ...)
strata ram dedup: inizio richiesta: ...
strata ram dedup: fine richiesta: arena 27.0 GiB (era 39.97), 7750 esperti senza copia RAM, D2H 96 esperti in 31.2 ms, riletti da file 0 (totali: D2H 96, riletti 0, attese ring 0)
strata ram: fine richiesta: privata 66.1 GiB (working set 36.4, non residente 29.7): arena esperti 27.0, ring dedup 0.5, cache righe PLE 0.1 (stima), checkpoint/prompt cache 0.3, parcheggio conversazioni 0.0, K/V pinned 0.0, pesi densi/embedding pinned 0.9, scambi file tier 0.0, altro 37.3
```

(The numbers above are the expected shape, not a measurement.)  The "D2H ... ms" figure is
host time spent in the dedup's own paths (issuing the copies, landing them, waiting in
`blob()`), not the copy engine's time.

The `strata ram:` line is the private-RAM breakdown the owner asked for.  Note what it does
not contain: file mappings (the PLE table in `direct` mode is not resident at all; a mapped
GGUF is working set, not private bytes).  On Windows the gap "privata - working set" is
commit charge that is not resident: under WDDM the driver charges the VRAM allocations'
paging reserve to the process, which is the first candidate for the ~46 GB of non-arena
private bytes measured on the reference PC (79.5 GB private, 49.6 GB working set, ~27 GiB
of VRAM in use across the two cards plus the pinned prefix).  "altro" is the remainder:
driver, CUDA contexts, heap, stacks, the prompt path's rings.  The breakdown prints at
serving start and at the end of every request.

## How to measure

Private RAM and tokens/s, both with the deterministic bench (same tokens, same windows):

1. Reference run, dedup off: `STRATA_RAM_LOG=1 STRATA_FORCE_IDS=<ids> STRATA_FORCE_TRACE=<trace-a>`,
   note the `strata ram:` line after the warm-up request (arena 39.97) and the DONE lines'
   decode ms; keep the generated ids.
2. Same command with `STRATA_RAM_DEDUP=1` and `STRATA_FORCE_WINDOWS=<trace-a>`: the verified
   work is the reference's by construction.  Compare: `strata ram:` private bytes (target:
   at least 10 GiB less), the `T <id>` lines (must be identical), decode ms per request, and
   the `strata ram dedup: fine richiesta` counters - `riletti da file` must stay 0; `D2H`
   should be a few hundred experts per request at most (what the adaptive tier swaps) and
   `attese ring` 0 (a ring too small for a shrink step: raise `STRATA_RAM_DEDUP_BOUNCE_MIB`).
3. Outside the engine: `Get-Process strata | Select PrivateMemorySize64, WorkingSet64`
   before and after arming, and the Task Manager's "Commit" column; the arena's own number
   is `arena X GiB (era Y)`.

Expected on the reference PC: ~21.3 GiB resident in VRAM, minus the kept lend regions
(3.63 + 3.79 GiB) and the experts of the registered prefix in non-protected slots (~1 GiB),
gives ~13 GiB given back.  If less is needed, `STRATA_ARENA_PIN_GIB` smaller frees more (the
prefix is never dropped) at the cost of the PCIe share and the DMA prompt copies.

## Risks

* **Latency of the exchanges.**  An expert leaving VRAM now costs a 1.4-1.7 MB D2H copy over
  the card's link (PCIe x4 on the 3060: ~0.5 ms each) before the slot can be refilled, and
  a window that routes that expert while the copy is in flight waits for it in `blob()`.
  The adaptive tier swaps ~96 experts per round and the reference log shows ~4,000
  residence changes per request: watch the DONE decode ms against the reference run.
* **Order of the copies.**  The guarantee is event-based: `evict` must be called on the same
  stream that will overwrite the slot, before the overwrite is queued, and before any unmap
  with a `sync` of that device.  A new eviction path that does not call `dd_evict` does not
  corrupt anything - the expert is re-read from the file (`riletti da file` > 0) - but it
  pays a disk read on the critical path.  That counter is the regression test.
* **The prompt path's loan.**  A lent slot's expert must be in RAM when the prompt streams it.
  The lend region keeps its copies, but it moves with the elastic tail (`el_replan_lend`);
  `lend` copies back whatever lacks one and waits before laying the buffers out.  The lab
  `STRATA_PIPELINE_ELASTIC` lets the prompt borrow the core too (`el_lend_floor = 0`): the
  protected region then follows `pf_parts[0].first`, which may be far below `el_core`, and
  less RAM is given back.
* **Decommit on locked pages.**  The arena beyond the registered prefix is `VirtualLock`ed in
  1 GiB chunks; a blob that straddles the end of what the lock reached fails to unlock and is
  kept (counted as not dropped, never wrong).
* **Working set minimum.**  Raised at startup for the whole arena, it is not lowered; the
  recommitted pages are re-locked best effort.
* **Batch mode (`--batch`) and the pipelined loop** share the same sites; the sweep runs only
  at `DONE`, when nothing is in flight.  Admitted batch slots keep decoding between requests
  through the same adaptive and elastic paths, so their evictions go through `dd_evict` too.

## Status

Written on the lab branch `lab/ram-dedup`.  Not compiled, not run on the reference PC yet
(the owner builds).  First thing to verify: a warm-up request with `STRATA_RAM_DEDUP=1`
prints `riletti da file 0` and identical `T <id>` lines against the reference trace.
