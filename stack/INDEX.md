# Strata-2080Ti patch stack

The patches this fork (imanu86/Strata-2080Ti) carries on top of upstream Strata, one lever each, in the order they apply. Written for a two-card Windows box (RTX 2080 Ti 22 GB as stage 1, RTX 3060 12 GB as stage 0, `--layer-split 16 --trim-stage-weights --pipeline-windows 2`), but every lever is a plain switch and most are bitwise with the default path. The registry exists so that the next upstream version can be checked and re-based mechanically: run `check_anchors.py --check` on the new tag, read what moved, re-apply.

Rebased on upstream v0.1.45 on 18 Oct 2026 (branch `lab/upsync-main`, diary `docs/UPSYNC_MAIN.md`): upstream took five of our levers as opt-in (B6 MMVQ rows, ATTN_DAILY, EXPDEC, QSA_ROWS, the 16-byte dequant) plus SIDE_MIXER, the direct PLE default and #1852; they are in the Retired table. The levers added after the 0.1.43 registry (13-17) come from lab/struct-a, lab/ttft43 and lab/pipeline2.

Patches 18-20 (19 Oct 2026, branch `lab/multichat45b`) are upstream pull requests ported opt-in on top of the multi-chat branch: #3095 (`STRATA_PW_WITH_BATCH`), #3104 (`STRATA_PF_FUSED_NATIVE_SM75`) and #2774 (`STRATA_PF_DIRETTO`). Their `.patch` files are the clean ports onto the stack head of patch 17 (d06429ee), so they apply with `git am` on `lab/upsync-main`; the commits named in the table are the ones on `lab/multichat45b`, where #3095 is merged with `STRATA_BATCH_PIPELINE` and `STRATA_PIPED_MTP` (three conflicts in `generate.cpp`). The upstream pull request #3085 is not carried (#3104 does the same with another gate); the sibnick fork's batched gather (97b83332) is not carried because 0.1.45 already groups the gather of the short prefill walk (653f0113, `STRATA_PREFILL_GROUP_GATHER`).

Patch 21 (19 Oct 2026, branch `lab/passata-unica` from `lab/multichat45b`) makes the packed prefill of the multi-chat branch run beside the concurrent read (`STRATA_PACK_CONCURRENT`); it depends on the pack commits of that branch, which are not entries of this stack.

Base: upstream `Niko1221/Strata` **v0.1.45** (`477ac13c`). Stack head: `4ee62d09`. 21 patches, applied in this order with `git am stack/*.patch` (or one at a time with `git apply --3way`). Check a new upstream with `python stack/check_anchors.py --check <repo> --rev <tag>`.

## Reading this file

- **Switch / default**: how the lever is turned on, and what happens when nothing is set.
- **Origin**: `fork` = written for this fork; `PR #n` = an upstream pull request (ours or somebody else's) carried as a patch while it is open.
- **Upstream**: whether upstream already has it, has an open PR for it, or never saw it.
- **Measured**: the value on this fork's hardware (RTX 2080 Ti 22 GB + RTX 3060 12 GB, `--layer-split 16`, Windows 11, 131k-token forced bench, B-A-B unless said otherwise). Details in the linked docs.
- **Insertion points**: one per hunk: file, the enclosing symbol as git names it, and an anchor of 2-3 lines of the *unpatched* file, unique in that file at the time the patch was made. No line numbers. The inserted block is in `BLOCKS.md` under the same id; the applicable form is the `.patch`.

## Upstream defaults this fork forces

Upstream defaults that this stack or its reference configuration changes, so that a run on the stack behaves like the fork's Daily:

| default in upstream 0.1.45 | here | by |
|---|---|---|
| `STRATA_SELECT_SIMT` opt-in (QSA block scores below sm_80 on the warp kernel) | **on** below sm_80, `=0` turns it off, `=2` larger tiles | patch 11 |
| `STRATA_TC_SM75` / `STRATA_SM75_PERF` opt-in (the bitwise Turing series, 16-byte dequant) | `STRATA_TC_SM75` **on** for a 7.5 card, `=0` off; the rest of the series stays behind `STRATA_SM75_PERF=1` | patch 11 |
| `STRATA_DF_BRANCH`, `STRATA_DF_PLE` opt-in (PR #904's graph branches / PLE post-ops) | **on** with CUDA (BRANCH: on Windows), `=0` turns them off; BRANCH yields to `STRATA_SIDE_MIXER` | patch 09 |
| `STRATA_PLE_LOOKAHEAD` (not in upstream) | **on**, `=0` off; direct PLE only | patch 15 |
| `STRATA_PIPELINE_ELASTIC` (not in upstream) | **on** with two cards and `--pipeline-windows`: theta 0.10, verified window served first; `=0` gives upstream's theta 0.20 and D-A-B order | patch 08 |
| `STRATA_ROUTE_TAIL_SKIP=7` (off from 90% cached experts, #1884) | unchanged in code; the reference config sets `=0` (-1.6% with it at 96-98% hit) | config |
| `STRATA_MMVQ_IL` on (interleaved dense GEMV where the card's table has a cell) | unchanged in code; the reference config sets `STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1` | config |
| `STRATA_NAN_GUARD` on (#1892) | unchanged (cost < 0.2%, bitwise otherwise); `=0` turns it off | - |

Reference configuration environment (the fork's 0.1.43 Daily r2, config `strata-0143-daily.json`): `STRATA_VRAM_FREE=device STRATA_ROUTE_TAIL_SKIP=0 STRATA_ATTN_DAILY=1 STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1 STRATA_MTP_SHARED_BRANCH=0 STRATA_SIDE_MIXER=2 STRATA_RAM_CORE=1 STRATA_RAM_CORE_SLOTS_DEV1=5500 STRATA_EXPDEC=1 STRATA_QSA_ROWS=1 STRATA_SA_BET=1 STRATA_SA_CHAIN=1 STRATA_PREFIX_DISK_MULTI=24`, options `--elastic --prompt-cache-tail --prompt-cache-every 16384 --prompt-cache-file <file> --layer-split 16 --trim-stage-weights --pipeline-windows 2 --vram-reserve-later-mib 384 --host-core last`. On 0.1.45 these variables mean the same, except that `STRATA_ATTN_DAILY`, `STRATA_B6_MMVQ_ROWS`, `STRATA_SIDE_MIXER`, `STRATA_EXPDEC`, `STRATA_QSA_ROWS` are now upstream's own. The same JSON also sets `STRATA_NODECUT=1 STRATA_TOPK_GRID=1 STRATA_B6_HEAD_MINB=6 STRATA_CONV_DISK=10`: the first three are lab levers of lab/struct-a that are NOT in this stack (neutral on the bench), they are ignored by a 0.1.45 build.

## Patches

| id | name | what | switch (default) | origin | upstream | measured |
|---|---|---|---|---|---|---|
| 01 | [elastic-cache](#01-elastic-cache) | `--elastic`: the expert cache grows and shrinks at runtime around a fixed core, sized on the card's free VRAM. | `--elastic` (off) | fork (PR #1117, ours) | PR #1117 open; Niko1221 (17 Oct): a feature for shared GPUs without a throughput claim, not part of the speed pass; not in 0.1.45 | Daily default since 3 Oct; with the two-card split see patch 05. Resubmitted PR: opt-in, exact. |
| 02 | [split-prompt-borrow](#02-split-prompt-borrow) | With `--layer-split`, each stage's prompt loan is estimated on its own device, not on CUDA0's. | always on with `--layer-split` (correctness) | fork (PR #1417, ours) | PR #1417 open; not ported (Niko1221: nothing changes on current upstream; to be sent with the fork's Turing buffers); applies clean on 0.1.45 | Correctness: the stage-1 card no longer refuses a loan that fits. No speed claim. |
| 03 | [forced-bench](#03-forced-bench) | Deterministic bench: replay token ids and verified windows (`STRATA_FORCE_IDS`, `STRATA_FORCE_WINDOWS`, trace with `STRATA_FORCE_TRACE`), adapted to the multi-stage pipeline. | `STRATA_FORCE_IDS=<file>[;<file>]`, `STRATA_FORCE_WINDOWS=<trace>`, `STRATA_FORCE_TRACE=<out>` (off) | fork (PR #1419, ours) | PR #1419 open; not in 0.1.45 | Tooling: every B-A-B number in this registry is read on this bench (131k context, L01 full prefill + L02 cached prefix, 1024 tokens, same text in every arm). |
| 04 | [elastic-split-pipeline](#04-elastic-split-pipeline) | The elastic cache with `--layer-split` (CUDA0's block only) and beside `--pipeline-windows`: the queue moves only in stage-0 gaps, growth published when the fills landed, 1.5 GiB queue, 128 MiB step, the prompt borrows the core too; on by default with two cards. | `STRATA_PIPELINE_ELASTIC` (on with two cards and `--pipeline-windows`; `=0` off) | fork | never proposed (depends on 01) | Part of the Daily's two-card configuration (83.7 t/s at 40-71k on a real SWE case, 7 Oct). The 0.1.42 port with the whole stack measured +0.7% against the Daily (131k forced bench). |
| 05 | [prefix-disk](#05-prefix-disk) | `--prompt-cache-file`: the long prompt prefix saved to disk and restored across restarts, format v2 with one image per split block; the MTP drafter computes every cell of the saved root. | `--prompt-cache-file <path>` (off) | fork | PR #2077 open (Niko1221: one disk design will be picked, #1090; not measured upstream); not in 0.1.45 | Restart of an agent session: the shared 40-130K prefix is restored from the file instead of being read again (a 40K first prompt took ~42 s to read here). Restore time: see the fork's smoke test of the PR branch. The file identity covers exe, args and STRATA_*/CUDA_* variables. |
| 06 | [vram-free-nvml](#06-vram-free-nvml) | `STRATA_VRAM_FREE=device`: free VRAM of the whole card through NVML (matched by PCI bus id, so `CUDA_VISIBLE_DEVICES` order does not matter); the elastic cache reads it too. | `STRATA_VRAM_FREE=device` (off: CUDA's per-process view) | upstream PR #1911 (Xiaomi2008) + fork glue for the elastic cache | upstream PR #1911 (Xiaomi2008), not in 0.1.45 (no STRATA_VRAM_FREE in the tree) | Fixes the 7-36 t/s collapses under WDDM paging on the 3060 (CUDA saw 9.7 GB free, NVML 8.1): with it the stage holds 77 t/s. Daily default since 11 Oct. |
| 07 | [park-no-foreign-kv](#07-park-no-foreign-kv) | The conversation cache never parks one conversation with another's retained K/V. | always on (correctness) | upstream PR #1503 (sanastasiou) | upstream PR #1503 (sanastasiou), not in 0.1.45; 3-way clean | Correctness only. |
| 08 | [pipeline-architectds](#08-pipeline-architectds) | The two-window pipeline's profile from the Daily: the verified window is served first (front stage of A, last stage of A, then D and B one layer per pass), lookup beyond one window, theta 0.10; plus two log lines (rollback cost per window class, chain latency). | `STRATA_PIPELINE_ELASTIC` (on with two cards and `--pipeline-windows`; `=0`: upstream's theta 0.20 and D-A-B order) | fork (from the architectds lab branch) | never proposed | 0.1.42 without it: -4.6/-7.3/-7.8% against the Daily (speculative windows 133/600 instead of 164/600, stage host 0.50 -> 1.69 ms). With it and patch 11: +0.7% (parity). |
| 09 | [df-defaults](#09-df-defaults) | `STRATA_DF_BRANCH` (mixer side branches in the window graph; CUDA on Windows) and `STRATA_DF_PLE` (PLE post-projections in one pass; CUDA) on by default, as in the Daily. | `STRATA_DF_BRANCH=0`, `STRATA_DF_PLE=0` turn them off (upstream: opt-in `=1`) | fork (defaults of PR #904's switches) | upstream merged #904 opt-in and keeps BRANCH off after a Linux stall (#905, never seen on Windows) | Daily measurements: BRANCH +2.4% at 131k, PLE +3.3% (bitwise). On 0.1.42 their absence was the main part of ~1.1 ms of GPU per window. |
| 10 | [ple-diag](#10-ple-diag) | PLE diagnostics: prefill reads (mean / max / over 100 ms) and the chunk that waits more than a second for its rows; decode gather time (`STRATA_DECODE_TIMING`) and the reader counters before/after a request. | log lines; `STRATA_DECODE_TIMING=1` for the decode ones | fork (ex patch 13 of the 0.1.43 stack, diagnostics half) | Niko1221 on PR #2073: the diagnostics are welcome as a separate small PR; the direct-read default itself is upstream since 0.1.43.1 (STRATA_PLE_AUTO_MMAP=1 for the old switch) | diagnostics only; found the 1 259 rows/s probe and the 260 s -> 75 s prefill of a cold SATA SSD |
| 11 | [prefill-sm75](#11-prefill-sm75) | The Daily's Turing prompt defaults: `STRATA_SELECT_SIMT` (FP32 tiled QSA block scores) on below sm_80, and the 16-byte dequant of the prompt projections (upstream's #2081) on for a 7.5 card. | `STRATA_SELECT_SIMT=0\|2`, `STRATA_TC_SM75=0` (sm_75 only) | fork (SIMT scores from PR #1091 konijiwa110, merged upstream opt-in; dequant from our PR #2081, merged upstream behind STRATA_SM75_PERF=1) | SIMT: PR #2080 (default on) declined (Niko1221, 17 Oct: changes the summation order, wants an independent sm_75 KL run); dequant 16 B: in 0.1.45 as `STRATA_TC_SM75` / `STRATA_SM75_PERF=1`. hc16 (FP16 tensor-core hyper-connection projections) NOT ported: 0.1.45 has the wider `STRATA_BF16_TC=1` + `STRATA_XN_F16` (#655 / #2091) | 131k prefill 100 -> 75 s on the 2080 Ti stage with the 0.1.43 stack (SIMT -16%, hc16 -19.5% of GPU prompt time). On 0.1.45 the hc16 share has to be re-measured with STRATA_BF16_TC=1 (timing of the prompt and a KL against the default). |
| 12 | [ram-core](#12-ram-core) | A fixed core of expert slots per card, excluded from swaps, loans and shrink, whose RAM copy is given back (the resident arena no longer mirrors experts that never leave VRAM). | `STRATA_RAM_CORE=1` (off), `STRATA_RAM_CORE_SLOTS_DEV<n>=<slots>` | fork | never proposed | `STRATA_RAM_CORE_SLOTS_DEV1=5500`: arena 39.97 -> 29.62 GiB (-10.35), working set 48.7 -> 33.6 GB, t/s +0.1%, hit rate unchanged, 0 re-reads from file. Daily default since 12 Oct. |
| 13 | [sa-bet](#13-sa-bet) | STRUCT-A: while the MTP chain runs, stage 0 serves the HEAD of the next window as a bet (`STRATA_SA_BET`, layers `_LAYERS`, widths `_MAXT`/`_MINT`); the forced chain is fed a step at a time and cut once B is decided (`STRATA_SA_CHAIN`, `_AHEAD`). | `STRATA_SA_BET=1`, `STRATA_SA_CHAIN=1` (off) | fork (lab/struct-a 075ce875) | never proposed | +2.5% against the 0.1.43 Daily on the 131k forced bench (SA_BET + SA_CHAIN, B-A-B, 12-13 Oct). Not re-measured on 0.1.45. |
| 14 | [prefix-disk-multi](#14-prefix-disk-multi) | `STRATA_PREFIX_DISK_MULTI=<GiB>`: one file per prefix beside `--prompt-cache-file` (`<file>.<16 hex>`), least recently used out past the cap; the file is written on a thread (`STRATA_PREFIX_DISK_SYNC=1` inline). | `STRATA_PREFIX_DISK_MULTI=<GiB>` (off: the single file) | fork (lab/ttft43 c00a8463) | never proposed (depends on 05) | TTFT bench 13 Oct with #1852 (now upstream): alternating clients 38.2 / 32.5 -> 16.2 / 4.9 s; after a restart 20.2 / 10.3 -> 6.4 / 5.1 s (with STRATA_CONV_DISK). |
| 15 | [ple-lookahead](#15-ple-lookahead) | The PLE rows of the first chunk of the next prompt part are read while the previous part finishes, and beside the prefix restore (`Prefill::set_ple_lookahead`, `ple_warm_now`). | `STRATA_PLE_LOOKAHEAD=0` turns it off (ON by default, direct PLE only) | fork (lab/ttft43 ea73f49e) | never proposed | part of the TTFT bench of patch 14 (about 0.7 s of file read hidden per restore). |
| 16 | [conv-disk](#16-conv-disk) | `STRATA_CONV_DISK=<s>`: after a request, once the engine has been idle that long, the live conversation at its last turn boundary goes to a keyed file; a request looks for the longest saved boundary among its own. | `STRATA_CONV_DISK=<idle seconds>`, `STRATA_CONV_DISK_MIN` (4096) (off; needs `--prompt-cache-file` and `STRATA_PREFIX_DISK_MULTI`, no `--batch`) | fork (lab/ttft43 abd685c5) | never proposed | a 130k conversation restarts from its file instead of the system turn alone (65-80 s of prefill); 13 Oct: a restart took 277 s with it on, later shown not to come from CONV_DISK (TTFT43.md). |
| 17 | [p2-flush-noswitch](#17-p2-flush-noswitch) | Pipeline polling without switching card: `STRATA_P2_FLUSH_NOSWITCH` (service, done and the chain read their mapped flags with no OnDevice), `STRATA_P2_FLUSH_US` (flush period), `STRATA_P2_SWITCH_STAT` (host time in the window launches), `STRATA_P2_SMEM` (config cache to shared). | `STRATA_P2_FLUSH_NOSWITCH=1`, `STRATA_P2_FLUSH_US=<us>`, `STRATA_P2_SWITCH_STAT=1`, `STRATA_P2_SMEM=1` (off) | fork (lab/pipeline2 f6a0b6b2; the polling idea from NTRYourWaifu's v7/v8) | never proposed | 13 Oct B-A-B (isolated bench): NOSWITCH removes ~8 100 card switches per request but the gain is below the spread; never promoted (docs/PIPELINE2.md par. 11). |
| 18 | [pw-with-batch](#18-pw-with-batch) | `--pipeline-windows` stays on for the solo path (one request alone) beside `--batch`, and the drafter stays on the first stage's card (`mtp_first`); the slot drafters of `--batch-mtp` follow it. Batch windows still run without the pipeline. | `STRATA_PW_WITH_BATCH=1` (off: `--batch` turns the pipeline off, as upstream) | upstream PR #3095 (18 Oct, +23/-5) | PR #3095 open, no maintainer answer yet; not in 0.1.45. Earlier precedent: fork noon-at-cgn d84da294. | Compiled and run 19 Oct (build 99fa6bd8, 2x solo:S with `--batch 4 --batch-groups 2`, 384 tokens, same reserves in both arms, A-P-A-P): decode 57.3 / 56.5 / 57.5 / 56.1 with the lever against 56.0 / 56.8 / 54.8 / 54.5 without (+2.4% mean), reserve 2300 MiB; the lever needs the larger reserve (the drafter, 837 MiB, and the windows, 234 MiB, move onto the 3060: with 1905 MiB the first request stalled for 240 s with 0 MiB free). It does NOT recover the single chat: the same config with reserve 1905 and no lever gives 60.8 (one run), the Daily without --batch 64.5. Installed no. |
| 19 | [pf-fused-native-sm75](#19-pf-fused-native-sm75) | The fused int8 prompt experts for the native i-quant packs (gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down Q2_0 or IQ4_NL) also on sm_75 (Turing), with the qualified 128-row shape. Pairs not covered stay on MMQ. Changes the bits of the prompt (rounding). | `STRATA_PF_FUSED=1 STRATA_PF_FUSED_SM75=1 STRATA_PF_FUSED_NATIVE_SM75=1` (off). `STRATA_PF_FUSED=1` also switches on the existing sm_86 native path of the first stage. | upstream PR #3104 (rebased from #1581; supersedes #3085, the same idea with another gate) | PR #3104 open (and #3085); in 0.1.45 `STRATA_PF_FUSED_SM75` only covers the Q2_0 pack, so on our IQ3_XXS pack it did nothing | Compiled and run 19 Oct (build ecaad907, fixed `--expert-cache 1500`, 3 prompts of ~300 and ~3000 tokens): TTFT median 1.83 s (control 1.77-1.84) at ~300 tokens and 6.55 s (control 5.72-5.89) at ~3000, i.e. +11% slower on the long prompts; first-token logits against the control KL 3e-5..6e-4, inside the control-to-control spread (1e-4..1.3e-3, the bench is not deterministic). The 300-token texts differ from a control as often as two controls differ. No gain on this pack and card pair; kept opt-in. Installed no. |
| 20 | [pf-diretto](#20-pf-diretto) | Short prompt chunks (below `stream_all_min`): resident experts used by at most `STRATA_PF_DIRETTO_MAX` rows (8) leave MMQ's list and run on the decode's grouped kernel (`native_expert_grouped` for native packs) straight from the compressed blob; also `STRATA_PF_ROUTE_PRIOR=lambda` (router push toward resident experts, off). Changes the bits of short prompts. | `STRATA_PF_DIRETTO=1`, `STRATA_PF_DIRETTO_MAX=<rows>` (off here; the PR has it on), `STRATA_PF_ROUTE_PRIOR=<lambda>` (off) | upstream PR #2774 (apatassini); default turned to opt-in in this fork | PR #2774 open; not in 0.1.45 | Compiled and run 19 Oct (build ecaad907): needed a fix (the PR's scratch was one static, allocated on the first stage's card and read by the second: stall at layer 16, then an illegal memory access); with one scratch per device it runs. TTFT median 1.81 s at ~300 tokens (control 1.77-1.84) and 5.81 s at ~3000 (control 5.72-5.89): no gain. Logits: bitwise equal to the control on the prompts above stream_all_min, KL 8e-5..1e-3 on the three ~300-token ones (the control-to-control spread is of the same size). Installed no. |
| 21 | [pack-concurrent](#21-pack-concurrent) | The packed prefill (STRATA_PACK_PREFILL: the prompts sent ahead by waiting requests read with this one in the same chunks, the experts streamed once for all of them) also beside STRATA_PIPED_CONCURRENT: the prefixes go into the pack slots on the host thread with nothing in flight, Prefill::run_packed runs on the read's thread while the slot groups keep decoding, the pack slots' checkpoints are taken after the drain. Without it the concurrent read returned before pk_read and never packed. | `STRATA_PACK_CONCURRENT=1` (off), with `STRATA_PACK_PREFILL=N` and `STRATA_PIPED_CONCURRENT=1`; a packed read takes no BYIELD | fork (lab/passata-unica, docs/porto/strata_adattivo/PASSATA_UNICA_2026-10-19.md in moe-aggressive-commit) | never proposed; 0.2.0-beta (upstream/main f57c0adc) still reads admissions one at a time and has no multi-sequence prefill | Compiled and run 19 Oct: see docs/MULTICHAT.md (passata unica) - to be filled from the letture_concorrenti bench (4 readers of 2-4k on 40-100k cached + 2 writers, B-A-B) and the replay. |

## Retired

Levers that were in the stack and were dropped (code stays on the fork's lab branches):

| name | what | measured | why |
|---|---|---|---|
| critpath-levers | Critical-path levers for the two-stage pipeline (STRATA_CP_L1FIRST/S1FIRST/CHAIN_PRE/CHAIN_FUSE/RESPLAN, opt-in). | All within noise on the 131k forced bench (-0.4/-2.5/+0.1/-0.1%). | 12 Oct 2026: dropped from the stack (42 insertion points in generate.cpp, verify.cpp, mtp.cpp for no gain). Code: fork branch lab/critpath2. |
| chain-fuse | Fused kernels of the MTP drafter's step (STRATA_CHAIN_FUSE2, opt-in), chain_step_bench. | Kernels -37..-86% on the 2080 Ti, engine +0.4% (noise); the fused attention was slower. | 12 Oct 2026: dropped from the stack (29 insertion points). Code: fork branch lab/chainfuse. |
| mmvq-il-sm75 (PR #1437) | Exact-shape MMVQ_IL table for sm_75, opt-in STRATA_MMVQ_IL=1. | 0.0% against B6 on the 2080 Ti. | Not ported to 0.1.43: upstream rewrote the IL dispatch; the PR is somebody else's and still open. |
| b6-mmvq-rows (PR #1418) | Dense GEMV (MMVQ) that reuses the quantized activations across 2-4 output rows (Q4_K, Q5_K, Q6_K, IQ4_XS), bitwise. | Reference config uses it with `STRATA_MMVQ_IL=0`: against it, upstream's IL path -3.0% and the sm_75 shape table of PR #1437 0.0% on the 2080 Ti (11-12 Oct). | 18 Oct 2026: in upstream 0.1.45 as opt-in `STRATA_B6_MMVQ_ROWS=1` (Niko1221 ported it on current main; +0.2..1% on an RTX 3060, +0.5..2% on an A4000, off by default). Same kernels, byte for byte; PR #1418 still shows open. |
| attn-daily (PR #2075) | The Daily's QSA decode attention on every CUDA card: the two-cells-per-step chunk kernel (upstream keeps it below sm_75 only) and the `attn_merge_v2` merge; bitwise with the default; parity harness `qsa_merge_daily_parity`. | nsys on the 3060: `attn_chunk_kernel` 99.8 -> 46.4 us a call. In the engine the attention is ~8% of the per-row cost (docs/QSAROWS.md): part of the +0.7% parity, not separable. | 18 Oct 2026: in upstream 0.1.45, commit 731528be, opt-in `STRATA_ATTN_DAILY=1` (only STRATA_ATTN_LANECELL, a lab knob, is missing). PR #2075 still shows open. |
| side-mixer (PR #1889) | `STRATA_SIDE_MIXER=1/2`: the mixer's work that reads only the layer's input (GDN gates, the QSA indexer chain, with `=2` also z and the q chain) on branches of the window graph; bitwise. Turns `STRATA_DF_BRANCH` off when set. | `=2`: +0.9% against DF_BRANCH on the 131k forced bench (11-12 Oct). Daily default since 12 Oct. | 18 Oct 2026: in upstream 0.1.45 (ffc30b8d, KarlGrier), opt-in `STRATA_SIDE_MIXER=1\|2`; it wins over STRATA_DF_BRANCH there as here. |
| qsa-rows (PR #2059) | QSA decode block scores with the queries in registers and the four heads reduced in six shuffles (T <= 4), bitwise; `qsa_rows_bench` (cost per window row of the QSA decode kernels). | Kernel 101 -> 55 us a call at T=4 on the 2080 Ti; engine B-A-B +3.2 / +0.1% (about +1-1.5%). Daily default since 12 Oct. | 18 Oct 2026: in upstream 0.1.45 (5b6c9398), opt-in `STRATA_QSA_ROWS=1`; level with the default on an RTX 3060 / A4000 (0.997 and 1.003 at 30K), so off upstream. PR #2059 still shows open. |
| expdec (PR #2058) | Routed experts' gate/up in the verify window with the group's activations in shared memory and R rows per lane (the down behind `STRATA_EXPDEC_DOWN=1`), bitwise; `expdec_bench`. | gate/up kernel -11/-23% at T 1-4 on the 2080 Ti; engine B-A-B +1.1% (B +0.0/+2.1). Daily default since 12 Oct. The Q2_0 down with EXPDEC was ~10% slower in the engine: off by default. | 18 Oct 2026: in upstream 0.1.45 (f7eaf2de), opt-in `STRATA_EXPDEC=1`, gate/up only, the down behind `STRATA_EXPDEC_DOWN=1` as here; +0.5% on an RTX 3060, +1..5% on an A4000. PR #2058 still shows open. |
| ple-direct (probe half) | The PLE table stays in direct I/O when the start-up probe measures the disk slow. | PLE on a cold SATA SSD: 131k prefill 260 s (mmap) -> 75 s (direct). | 18 Oct 2026: in upstream since 0.1.43.1 / 0.1.45 (62497bff): the slow probe warns and keeps direct, `STRATA_PLE_AUTO_MMAP=1` switches. The diagnostics half stays as patch 10. PR #2073 answered by Niko1221 as already contained. |
| prompt-encoder (#1852) | The server tokenizes the prompt from the last shared prefix (`STRATA_PROMPT_ENCODER`, server side). | TTFT bench of 13 Oct, see patch 14. | 18 Oct 2026: in upstream 0.1.45 (f13a870d, Niko1221, from gputier's #1852), unconditional there; the Daily bat variable is harmless. |
| hc16 | The hyper-connection projections and ssm_alpha/beta of the prompt on FP16 tensor cores (weights re-encoded once, FP16 activations; sm_75). | In the 0.1.43 Daily it removed 19.5% of the prompt GPU time. | 18 Oct 2026: not ported. 0.1.45 has its own, wider path (`STRATA_BF16_TC=1` with `STRATA_XN_F16`, #655/#2091) that overlaps the same call sites; two designs on one set of calls do not coexist. To decide after a prompt-time and KL A/B of STRATA_BF16_TC=1. Code: fork branches lab/expertdec 794268e4, lab/candidate2. |

## 01 elastic-cache

`01-elastic-cache.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la cache elastica con nucleo fisso (--elastic, PR #1117)

- **What**: `--elastic`: the expert cache grows and shrinks at runtime around a fixed core, sized on the card's free VRAM.
- **Switch (default)**: `--elastic` (off)
- **Origin**: fork (PR #1117, ours)
- **Upstream**: PR #1117 open; Niko1221 (17 Oct): a feature for shared GPUs without a throughput claim, not part of the speed pass; not in 0.1.45
- **Measured**: Daily default since 3 Oct; with the two-card split see patch 05. Resubmitted PR: opt-in, exact.
- **Notes**: Ported to 0.1.45: `Prefill::init` takes upstream's new `borrow_spill` and then `chunk_max`; `ExpertCache::close` zeroes upstream's `guard_` before the elastic branch; CMake keeps `host_graph_launch_test` and the `CUDA::cuda_driver` link.
- **Docs**: docs/ELASTIC_CACHE.md
- **Files**: `CMakeLists.txt`, `docs/ELASTIC_CACHE.md`, `include/strata/core/expert_cache.hpp`, `include/strata/prefill/prefill.hpp`, `src/core/expert_cache.cpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`, `tests/core/expert_cache_elastic_test.cpp`

Insertion points (39):

- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [01-1](BLOCKS.md#01-1)
  ```
      add_test(NAME host_graph_launch_test COMMAND host_graph_launch_test)
      set_tests_properties(host_graph_launch_test PROPERTIES TIMEOUT 30)
    endif()
  ```
- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [01-2](BLOCKS.md#01-2)
  ```
      target_link_libraries(expert_cache_per_layer_test PRIVATE strata_engine)
      add_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
      set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  ```
- `docs/ELASTIC_CACHE.md` — (new file) → [01-3](BLOCKS.md#01-3)
- `include/strata/core/expert_cache.hpp` · `public:` → [01-4](BLOCKS.md#01-4)
  ```
      const uint64_t* slot_offsets() const { return off_.empty() ? nullptr : off_.data(); }
      void close();
  ```
- `include/strata/core/expert_cache.hpp` · `private:` → [01-5](BLOCKS.md#01-5)
  ```
      std::vector<int32_t> layer_next_;   ///< [n_layers] -> that layer's next free slot
      std::vector<uint64_t> off_;         ///< plan v0.3 P6: slot offsets (slots + 1 entries) when sized
      int64_t admitted_ = 0;
  ```
- `include/strata/prefill/prefill.hpp` · `public:` → [01-6](BLOCKS.md#01-6)
  ```
      /// `host_res`: the static residency table (n_layers x n_expert, slot or -1) or null; `cache` its slots.
      /// `borrow`/`borrow_bytes`: device memory to carve every buffer from (the top slots of the expert cache,
      /// lent for the prompt and refilled after it); null = allocate normally.
  ```
- `src/core/expert_cache.cpp` · `(file scope)` → [01-7](BLOCKS.md#01-7)
  ```
  #include <cstdio>
  #include <filesystem>
  #include <cstdlib>
  ```
- `src/core/expert_cache.cpp` · `bool ExpertCache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_l` → [01-8](BLOCKS.md#01-8)
  ```
  void ExpertCache::close() {
  #if defined(STRATA_USE_HIP)
      if (blocking_staging_) (void) cudaFreeHost(blocking_staging_);
  ```
- `src/core/expert_cache.cpp` · `void ExpertCache::close() {` → [01-9](BLOCKS.md#01-9)
  ```
  #endif
      off_.clear();
      guard_ = 0;
  ```
- `src/prefill/prefill.cpp` · `uint64_t moe_set_bytes(size_t T, int64_t n_expert, bool fused) {` → [01-10](BLOCKS.md#01-10)
  ```
  
  bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, core::SessionState& ss,
                     core::ExpertSource* src, const core::ExpertCache* cache, const int32_t* host_res, int64_t chunk,
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co` → [01-11](BLOCKS.md#01-11)
  ```
      m.T = chunk > 0 ? chunk : 1;
      return strata::metal_engine::window_ensure(g, ss, ss.qsa_states[ss.qsa_primary()].max_cells, err);
  #endif
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co` → [01-12](BLOCKS.md#01-12)
  ```
      // the CPU share's staged-chunk limit, before the ring is sized below (a caller that sizes loans first armed it)
      arm_cpu_share(cpu_pool_ != nullptr);
      for (int b = 0; next_ != nullptr && b < 2; ++b)
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co` → [01-13](BLOCKS.md#01-13)
  ```
          m.hand_pieces = handoff_pipe_pieces();
      }
      if (m.tok_dev == nullptr) {
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co` → [01-14](BLOCKS.md#01-14)
  ```
          static const bool xn_env = [] { const char* e = std::getenv("STRATA_XN_F16"); return e == nullptr || e[0] != '0'; }();
          m.xn_f16 = xn_env && !m.f16_io && m.gemm.bf16_converts_x();
      }
  ```
- `src/program/generate.cpp` · `struct Options {` → [01-15](BLOCKS.md#01-15)
  ```
      /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
      /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
      int vram_reserve_mib = 700;
  ```
- `src/program/generate.cpp` · `void usage() {` → [01-16](BLOCKS.md#01-16)
  ```
                   "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                   "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                   "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
  ```
- `src/program/generate.cpp` · `void usage() {` → [01-17](BLOCKS.md#01-17)
  ```
                   "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                   "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                   "                       ms/token, -2.7 ms/token end to end.\n"
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-18](BLOCKS.md#01-18)
  ```
          else if (a == "--vram-reserve-mib") { o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib")); o.vram_reserve_given = true; }
          else if (a == "--vram-reserve-later-mib")
              o.vram_reserve_later_mib = std::atoi(next("--vram-reserve-later-mib"));
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-19](BLOCKS.md#01-19)
  ```
      strata::core::set_peer_portable(o.peer_device >= 1);   // multi-GPU: the Portable flag on mapped host buffers only with a peer device (before any allocation)
      {   // the RAM guard (parking, session save/restore): --memory-limit-mib, else STRATA_MEMORY_LIMIT_MIB, else none
          int64_t mib = o.memory_limit_mib;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-20](BLOCKS.md#01-20)
  ```
                              !o.expert_profile.empty() && !o.resident_cpu_experts && o.expert_cache != 0 && !remote &&
                              strata::core::vmm_available() &&
                              // the batch slots carve their own K/V and --vram-elastic's cache is not one VMM range
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-21](BLOCKS.md#01-21)
  ```
                          : multi_gpu ? "a layer split has a cache per GPU"
                          : std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                        [](int n) { return n > 0; }) ? "the helper caches on other GPUs"
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-22](BLOCKS.md#01-22)
  ```
          // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
          // free figure read again; while it is short of the reserve the cache is reopened smaller.
          // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-23](BLOCKS.md#01-23)
  ```
          const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
          return std::min(max_chunk, rounded);
      };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-24](BLOCKS.md#01-24)
  ```
      };
      auto plan_lend = [&](int64_t& chunk) -> int64_t {
          auto slots_for = lend_slots;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-25](BLOCKS.md#01-25)
  ```
                  if (c > 8192 && (c > o.prefill_auto_max || c > o.max_context)) continue;
                  if (o.prefill_auto_cap > 0 && c > o.prefill_auto_cap) continue;   // #1630
                  const int64_t k = slots_for(c);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-26](BLOCKS.md#01-26)
  ```
              constexpr int64_t kRingMin = 16;
              const int64_t small = old_rule();   // 0.1.39's chunk (and its ring for the prompts that fit it)
              const int64_t ring_max = strata::prefill::Prefill::ring_cap_for(small);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-27](BLOCKS.md#01-27)
  ```
                      // ring_slots() now returns `room` - unless STRATA_PREFILL_RING overrides it, in which case the
                      // chunk has to fit THAT ring, which is the pre-fix rule and the A/B arm
                      const int64_t k = slots_for(t);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-28](BLOCKS.md#01-28)
  ```
          }
          for (int64_t c = chunk; c >= 256; c /= 2) {
              const int64_t k = slots_for(c);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-29](BLOCKS.md#01-29)
  ```
              // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
              auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                  const int64_t k = part_slots(p, c);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-30](BLOCKS.md#01-30)
  ```
                      // clamped at 0: a cache under 128 slots has a negative lend budget, and `slots - budget` is
                      // then `slots`, which is off_[slots] - the end of the prefix sum, so that cache lends 0 bytes
                      // rather than reading one entry past its offsets
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-31](BLOCKS.md#01-31)
  ```
          // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
          // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
          // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-32](BLOCKS.md#01-32)
  ```
              if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
              if (share_pool) sp.set_cpu_pool(&pool);
              if (!sp.init(wt, g, ss, prefill_src, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes,
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-33](BLOCKS.md#01-33)
  ```
                  const int32_t* r = host_res.data() + l * g.n_expert;
                  for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                      if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-34](BLOCKS.md#01-34)
  ```
              for (float& v : drive.d.usage) v *= o.adapt_decay;
              return true;
          };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-35](BLOCKS.md#01-35)
  ```
              if (batch_on() || (piped && pipe_inflight())) {
                  if (!try_next_line(line)) {
                      if (!(piped ? pump(true) : batch_step())) return 1;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-36](BLOCKS.md#01-36)
  ```
              auto lend = [&](int64_t tokens, std::string& e, int64_t exact) -> bool {
                  seg_lend = 0;
                  if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-37](BLOCKS.md#01-37)
  ```
                  return true;
              };
              apply_pending(true);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [01-38](BLOCKS.md#01-38)
  ```
                                           std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
                  if (eos) { finish = "stop"; break; }
                  if (stop_req.load()) { finish = "cancel"; break; }
  ```
- `tests/core/expert_cache_elastic_test.cpp` — (new file) → [01-39](BLOCKS.md#01-39)

## 02 split-prompt-borrow

`02-split-prompt-borrow.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la stima del prestito del prompt sul device del proprio blocco (PR #1417)

- **What**: With `--layer-split`, each stage's prompt loan is estimated on its own device, not on CUDA0's.
- **Switch (default)**: always on with `--layer-split` (correctness)
- **Origin**: fork (PR #1417, ours)
- **Upstream**: PR #1417 open; not ported (Niko1221: nothing changes on current upstream; to be sent with the fork's Turing buffers); applies clean on 0.1.45
- **Measured**: Correctness: the stage-1 card no longer refuses a loan that fits. No speed claim.
- **Docs**: docs/UPSYNC.md
- **Files**: `src/prefill/prefill.cpp`, `src/program/generate.cpp`

Insertion points (2):

- `src/prefill/prefill.cpp` · `bool Prefill::init(const core::WeightTable& wt, const core::ModelGeometry& g, co` *(anchor ambiguous in the base file)* → [02-1](BLOCKS.md#02-1)
  ```
          err = "prefill: device buffers for a chunk of " + std::to_string(chunk) + " tokens do not fit (" +
                std::to_string(fb >> 20) + " of " + std::to_string(tb >> 20) + " MiB free at the failure; " +
                std::to_string(o.used >> 20) + " MiB taken, the failing buffer wanted " +
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [02-2](BLOCKS.md#02-2)
  ```
                                                       (uint64_t) strata::kernels::cpu::expert_layout().max_blob, need);
          };
          auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
  ```

## 03 forced-bench

`03-forced-bench.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il banco forzato STRATA_FORCE_IDS/STRATA_FORCE_WINDOWS/STRATA_FORCE_TRACE (PR #1419)

- **What**: Deterministic bench: replay token ids and verified windows (`STRATA_FORCE_IDS`, `STRATA_FORCE_WINDOWS`, trace with `STRATA_FORCE_TRACE`), adapted to the multi-stage pipeline.
- **Switch (default)**: `STRATA_FORCE_IDS=<file>[;<file>]`, `STRATA_FORCE_WINDOWS=<trace>`, `STRATA_FORCE_TRACE=<out>` (off)
- **Origin**: fork (PR #1419, ours)
- **Upstream**: PR #1419 open; not in 0.1.45
- **Measured**: Tooling: every B-A-B number in this registry is read on this bench (131k context, L01 full prefill + L02 cached prefix, 1024 tokens, same text in every arm).
- **Notes**: Ported to 0.1.45: upstream's `STRATA_ADMIT_MERGE_FIRST` merged first window (it carries prompt rows) is neither forced nor replayed; every other window is.
- **Docs**: docs/UPSYNC.md
- **Files**: `src/program/generate.cpp`

Insertion points (11):

- `src/program/generate.cpp` · `static const char* pipe_dbg_env(const char* name) {` → [03-1](BLOCKS.md#03-1)
  ```
      return on ? std::getenv(name) : nullptr;
  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-2](BLOCKS.md#03-2)
  ```
                  continue;
              }
              char* endp = nullptr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-3](BLOCKS.md#03-3)
  ```
              const double prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
              std::printf("REUSED %lld\n", (long long) resume);   // the prompt is read; the first window comes next
              std::fflush(stdout);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-4](BLOCKS.md#03-4)
  ```
                          if (v.service(&drive_pool_split, SDf(A), err) < 0) return die(err);
                          if (v.done(err)) {
                              if (!v.pl_finish(outp.data(), err)) return die(err);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-5](BLOCKS.md#03-5)
  ```
                                  // stage B now (its PLE rows from the tokens before it as they will be once A is
                                  // committed whole), so its launch behind A is only the graph launch (never while a
                                  // rollback is pending: B's verifier is the one the undo commit reads)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-6](BLOCKS.md#03-6)
  ```
                      // state)
                      if (A.ready && !A.fl && A.fr < pl_nf && !doomed) {
                          const int st = A.fr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-7](BLOCKS.md#03-7)
  ```
                      }();
                      if (B.ready && !B.fl && B.fr < pl_nf && B.fr < pl_spec_depth && A.fr > B.fr && !doomed) {
                          const int st = B.fr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-8](BLOCKS.md#03-8)
  ```
                      chain_kind = 0;
                      int a = 0;
                      while (a < A.T - 1 && A.tok[a + 1] == outp[(size_t) a]) ++a;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-9](BLOCKS.md#03-9)
  ```
                                           : from_sfx ? sbuf[(size_t) i - 1]
                                                      : drafts[(size_t) i - 1];
                  for (int i = 0; i < chain_n; ++i) window[(size_t) (T_mtp + i)] = cbuf[(size_t) i];
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-10](BLOCKS.md#03-10)
  ```
                      }
                  }
                  int a = merged ? T - 1 : 0;   // a merged window commits every row, as the prompt windows do
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [03-11](BLOCKS.md#03-11)
  ```
                  p += a + 1;
              }
              const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
  ```

## 04 elastic-split-pipeline

`04-elastic-split-pipeline.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la cache elastica di CUDA0 con --layer-split e accanto a --pipeline-windows

- **What**: The elastic cache with `--layer-split` (CUDA0's block only) and beside `--pipeline-windows`: the queue moves only in stage-0 gaps, growth published when the fills landed, 1.5 GiB queue, 128 MiB step, the prompt borrows the core too; on by default with two cards.
- **Switch (default)**: `STRATA_PIPELINE_ELASTIC` (on with two cards and `--pipeline-windows`; `=0` off)
- **Origin**: fork
- **Upstream**: never proposed (depends on 01)
- **Measured**: Part of the Daily's two-card configuration (83.7 t/s at 40-71k on a real SWE case, 7 Oct). The 0.1.42 port with the whole stack measured +0.7% against the Daily (131k forced bench).
- **Depends on**: 01
- **Docs**: docs/UPSYNC.md
- **Files**: `src/program/generate.cpp`

Insertion points (30):

- `src/program/generate.cpp` · `static const char* pipe_dbg_env(const char* name) {` → [04-1](BLOCKS.md#04-1)
  ```
      return on ? std::getenv(name) : nullptr;
  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-2](BLOCKS.md#04-2)
  ```
          // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
          // free figure read again; while it is short of the reserve the cache is reopened smaller.
          // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-3](BLOCKS.md#04-3)
  ```
                  --fake_fails;
                  err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
              } else {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-4](BLOCKS.md#04-4)
  ```
      // --elastic: the TWO-ZONE cache.  Slots [0, el_core) are the core - filled from the profile at startup, never
      // picked by the adaptive swaps to leave, never shrunk, never lent; the tail after it is the elastic zone.
      int64_t el_core = 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-5](BLOCKS.md#04-5)
  ```
      };
      auto plan_lend = [&](int64_t& chunk) -> int64_t {
          auto slots_for = lend_slots;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-6](BLOCKS.md#04-6)
  ```
              // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
              auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                  const int64_t k = part_slots(p, c);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-7](BLOCKS.md#04-7)
  ```
                      int64_t budget = std::max<int64_t>(0, std::min(p.cache->slots() - 128,
                                                                    kAutoLendPct * p.cache->slots() / 100));
                      // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-8](BLOCKS.md#04-8)
  ```
          // the prompt path's own buffers (no loan) are not priced into them: with the whole arena pinned (#253) a
          // `--prefill auto` split could stop at start with "device buffers for a chunk of 2048 tokens do not fit".  A
          // chunk that does not fit is tried again one size smaller, down to 512 tokens (a smaller chunk only reads slower).
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-9](BLOCKS.md#04-9)
  ```
          std::function<void()> adapt_fence;
          int64_t adapt_nswaps = 0;   // the blocking tier's swaps chosen, cumulative (the decode timing line's swaps/window)
          // the VRAM tier follows the conversation (the same rule as the speculative loop below)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-10](BLOCKS.md#04-10)
  ```
                  const int32_t* r = host_res.data() + l * g.n_expert;
                  for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                      if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-11](BLOCKS.md#04-11)
  ```
          // that shrank too far lends a smaller chunk, the same rule as plan_lend).  Run only where nothing is in
          // flight: a request's start (nothing lent yet) and the decode's safe point (window, commit, draft and the
          // adaptive swaps all finished).
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-12](BLOCKS.md#04-12)
  ```
          Clock::time_point el_last_grow{}, el_last_shrink{};
          int el_backoff = 1;
          auto el_free = []() -> int64_t {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-13](BLOCKS.md#04-13)
  ```
              const bool have_usage = !drive.d.usage.empty();
              std::vector<std::vector<std::pair<float, int32_t>>> keep((size_t) g.n_layers), tail((size_t) g.n_layers);
              for (int64_t l = 0; l < g.n_layers; ++l)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-14](BLOCKS.md#04-14)
  ```
          // GROW by at most `room` bytes: the missing experts the conversation routes most (usage >= 2, as the adaptive
          // tier's candidates), then the profile's order for what it has not asked for yet; filled on the refill
          // stream and published like the adaptive swaps (apply_pending), so the windows never wait for them
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-15](BLOCKS.md#04-15)
  ```
              for (const auto& pr : profile) {
                  if (used + (int64_t) lay.max_blob > room) break;
                  const int32_t i = (int32_t) (pr.first * g.n_expert + pr.second);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-16](BLOCKS.md#04-16)
  ```
              // under WDDM the free figure is only trustworthy once the pages are written: `grow` zeroed them
              cudaDeviceSynchronize();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-17](BLOCKS.md#04-17)
  ```
                      e = "elastic cache: a fill copy failed";
                      return false;
                  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-18](BLOCKS.md#04-18)
  ```
          // one step of the policy: shrink at once below the reserve; grow once the room has stayed for stable_ms.  At a
          // request's start it only shrinks: a growth's fills land asynchronously in the tail, and the prompt is about
          // to borrow the tail - the decode's safe point grows, and apply_pending publishes the fills before any lend
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-19](BLOCKS.md#04-19)
  ```
                      el_backoff = std::min(el_backoff * 2, 8);   // a shrink right after a growth: it grew too far
                  el_last_shrink = t;
                  el_seen.clear();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-20](BLOCKS.md#04-20)
  ```
              el_stable_since = t;
              el_last_grow = t;
              el_seen.clear();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-21](BLOCKS.md#04-21)
  ```
                  switch (astate) {
                  case AState::Idle:
                      if (drain || a_win - a_last < (int64_t) o.adapt_every) return true;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-22](BLOCKS.md#04-22)
  ```
          // --pipeline-windows 2: the loop has drained (nothing in flight on either card): what waits for a gap goes now,
          // and the round in flight goes on unfenced (a_pl_idle unset) from the next tick, the next request's drain.  The
          // job's work is the host's alone, so the wait for it ends.
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-23](BLOCKS.md#04-23)
  ```
                  bool b_done = false;     // B has been made from the chain in flight
                  int chain_n = 0;
                  int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-24](BLOCKS.md#04-24)
  ```
                  // --adapt-async: the asynchronous tier beside the windows (see adapt_tick).  A stage is idle when neither
                  // of its verifiers has a window in flight (a commit or a snapshot copy waits on nothing of the host's)
                  if (ajob) a_pl_idle = [&](int st) { return !PV[st][0]->in_flight() && !PV[st][1]->in_flight(); };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-25](BLOCKS.md#04-25)
  ```
                      if (pl_adapt_thr.joinable()) return true;   // still adapting
                      pl_apply();
                      if (!drive.d.usage.empty() && o.adapt_every > 0 && (rounds % o.adapt_every) == 0 && pending.empty() &&
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-26](BLOCKS.md#04-26)
  ```
                          } else if (!err.empty()) return die(err);
                      }
                      if (drive.d.failed) return die(drive.d.fail ? drive.d.fail : "the expert pool failed");
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-27](BLOCKS.md#04-27)
  ```
                          if (st == 0 && force_win != nullptr) force_window(A.p, A.tok[0], A.T, A.tok, S);   // lab bench replay
                          if (st == 0 && A.p + A.T > o.max_context) { ending = true; continue; }
                          if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-28](BLOCKS.md#04-28)
  ```
                          if (st == 0 && force_win != nullptr) force_window(B.p, B.tok[0], B.T, B.tok, S);   // lab bench replay
                          if (st > 0 || (B.p_on >= theta && B.p + B.T <= o.max_context)) {
                              if (ajob && st == 0) a_gap(0);   // --adapt-async: stage 0 is idle until this launch
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-29](BLOCKS.md#04-29)
  ```
                  // unfenced (the INVARIANT at pl_drain: a checkpoint, a park, the serial loop and the next request find
                  // nothing pipelined in flight) at the next tick, the next request's drain
                  a_pl_settle();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [04-30](BLOCKS.md#04-30)
  ```
                                           "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                   (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                   (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
  ```

## 05 prefix-disk

`05-prefix-disk.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il prefisso su disco con due schede (--prompt-cache-file, formato v2)

- **What**: `--prompt-cache-file`: the long prompt prefix saved to disk and restored across restarts, format v2 with one image per split block; the MTP drafter computes every cell of the saved root.
- **Switch (default)**: `--prompt-cache-file <path>` (off)
- **Origin**: fork
- **Upstream**: PR #2077 open (Niko1221: one disk design will be picked, #1090; not measured upstream); not in 0.1.45
- **Measured**: Restart of an agent session: the shared 40-130K prefix is restored from the file instead of being read again (a 40K first prompt took ~42 s to read here). Restore time: see the fork's smoke test of the PR branch. The file identity covers exe, args and STRATA_*/CUDA_* variables.
- **Notes**: Ported to 0.1.45: upstream's prefix borrowing between parked conversations (`ConversationCache::borrows`) coexists: a disk root longer than the in-RAM match wins and the donor stays parked (`borrow = false`); otherwise unchanged.
- **Docs**: docs/UPSYNC.md
- **Files**: `CMakeLists.txt`, `include/strata/core/mtp.hpp`, `include/strata/core/prefix_cache_file.hpp`, `src/core/mtp.cpp`, `src/core/prefix_cache_file.cpp`, `src/core/prefix_cache_file_test.cpp`, `src/program/generate.cpp`

Insertion points (20):

- `CMakeLists.txt` · `option(STRATA_WERROR "Treat compiler warnings as errors" OFF)` → [05-1](BLOCKS.md#05-1)
  ```
  # native-experts section far below, so a first configure registered four tests fewer than the second)
  option(STRATA_NATIVE_EXPERTS "Native (IQ) expert support through ggml-cpu" ON)
  ```
- `CMakeLists.txt` · `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` → [05-2](BLOCKS.md#05-2)
  ```
      src/core/conversation_snapshot.cpp src/core/conversation_state.cpp src/core/conversation_memory.cpp src/core/peer_experts.cpp
      src/core/arch_defaults.cpp src/core/vmm.cpp src/core/conversation_file.cpp)
    target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  ```
- `include/strata/core/mtp.hpp` · `public:` → [05-3](BLOCKS.md#05-3)
  ```
      const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
      const void* tensor_q8(const char* name) const { return q8(name); }
      QsaState& kv_state_rw() { return st_; }
  ```
- `include/strata/core/mtp.hpp` · `private:` → [05-4](BLOCKS.md#05-4)
  ```
      std::string rt_dir_;
      int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
      int64_t prompt_len_ = 0;
  ```
- `include/strata/core/prefix_cache_file.hpp` — (new file) → [05-5](BLOCKS.md#05-5)
- `src/core/mtp.cpp` · `bool MtpDrafter::prefill(const float* R_rows, const int32_t* next_tokens, int64_` → [05-6](BLOCKS.md#05-6)
  ```
      const OnDevice on_device(device_);
      const Clock::time_point t0 = Clock::now();
      const int64_t HCN = g_->hc * g_->n_embd;
  ```
- `src/core/prefix_cache_file.cpp` — (new file) → [05-7](BLOCKS.md#05-7)
- `src/core/prefix_cache_file_test.cpp` — (new file) → [05-8](BLOCKS.md#05-8)
- `src/program/generate.cpp` · `(file scope)` → [05-9](BLOCKS.md#05-9)
  ```
  #include "strata/core/conversation_snapshot.hpp"
  #include "strata/core/conversation_file.hpp"
  #include "strata/core/session_save_reclaim.hpp"
  ```
- `src/program/generate.cpp` · `namespace strata::metal_engine { void set_model_shard(const char* shard1); }   /` → [05-10](BLOCKS.md#05-10)
  ```
  #include <cstdio>
  #include <cstdlib>
  #include <cstring>
  ```
- `src/program/generate.cpp` · `struct Options {` → [05-11](BLOCKS.md#05-11)
  ```
      /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
      /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
      int prompt_cache = 6;
  ```
- `src/program/generate.cpp` · `void usage() {` → [05-12](BLOCKS.md#05-12)
  ```
                   "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                   "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                   "                       of RAM each; 0 = read every prompt from the start)\n"
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-13](BLOCKS.md#05-13)
  ```
          else if (a == "--vision") o.vision = true;
          else if (a == "--session-save-reclaim") o.session_save_reclaim = true;
          else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-14](BLOCKS.md#05-14)
  ```
      }();
      strata::core::set_primary_device(primary_device);
      // parking with --layer-split saves every stage (SavedConversation::stage_images)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-15](BLOCKS.md#05-15)
  ```
          bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
          // without --mtp the conversation cache stays on: a parked image carries the draft layer's K/V only when there is one
          // (the snapshots accept a null draft)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-16](BLOCKS.md#05-16)
  ```
              // Preserve the outgoing branch before any checkpoint rewind, reset,
              // or incoming restore overwrites the positional state it requires.
              // pin=N: this request resumes exactly at the pinned shared prefix of the conversation the session holds
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-17](BLOCKS.md#05-17)
  ```
                                   mtp.kv_state().kv_mode, "ram",
                                   (long long) (mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size));
                  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-18](BLOCKS.md#05-18)
  ```
                  }
                  incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                  lent_slot = -1;   // (the main session holds the parked conversation now)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-19](BLOCKS.md#05-19)
  ```
                          if (i >= o.prompt_cache_root) root_at = i;
                          break;
                      }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [05-20](BLOCKS.md#05-20)
  ```
                  if (trace && to == message_at)
                      std::fprintf(stderr, "strata serve: message boundary checkpoint: %lld tokens, %lld tail\n",
                                   (long long) message_at, (long long) (turn_at - message_at));
  ```

## 06 vram-free-nvml

`06-vram-free-nvml.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la PR upstream #1911: VRAM libera della scheda intera via NVML (STRATA_VRAM_FREE=device), letta anche dalla cache elastica

- **What**: `STRATA_VRAM_FREE=device`: free VRAM of the whole card through NVML (matched by PCI bus id, so `CUDA_VISIBLE_DEVICES` order does not matter); the elastic cache reads it too.
- **Switch (default)**: `STRATA_VRAM_FREE=device` (off: CUDA's per-process view)
- **Origin**: upstream PR #1911 (Xiaomi2008) + fork glue for the elastic cache
- **Upstream**: upstream PR #1911 (Xiaomi2008), not in 0.1.45 (no STRATA_VRAM_FREE in the tree)
- **Measured**: Fixes the 7-36 t/s collapses under WDDM paging on the 3060 (CUDA saw 9.7 GB free, NVML 8.1): with it the stage holds 77 t/s. Daily default since 11 Oct.
- **Depends on**: 01
- **Notes**: The fork's always-defined build macro STRATA_EC_NVML_FREE around the NVML code was removed (follow-up commit, no behaviour change).
- **Docs**: docs/UPSYNC.md
- **Files**: `docs/DETAILS.md`, `include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`, `src/program/generate.cpp`

Insertion points (6):

- `docs/DETAILS.md` · `IQ3_XXS -39%, IQ2_XXS -34%, IQ2_XS -24%, IQ2_S -3% on a P-core.  End to end, wit` → [06-1](BLOCKS.md#06-1)
  ```
  window falls 5% on a Xeon E5-2643 v3 + RTX A4000 (22.2 -> 21.1 ms of a 46.9 ms window, decode +0.5% over 10 interleaved pairs,
  +2.5% on the story and code prompts) and on a Core Ultra 7 265 + RTX 3060 (8.2 -> 7.8 ms of 39.5 ms, decode +0.4%), and does not
  move on a Threadripper PRO 3975WX + R9700 (14.5 ms, +0.4%, 5 pairs); no case reached +3% end to end, so the switch stays off.
  ```
- `include/strata/core/expert_cache.hpp` · `bool write_expert_profile(const std::string& path, int64_t n_layers, int64_t n_e` → [06-2](BLOCKS.md#06-2)
  ```
  /// Free device memory for the expert cache.  On a unified memory system - one memory for the CPU and the GPU, such as
  /// the DGX Spark (GB10); CUDA reports it as cudaDevAttrIntegrated - cudaMemGetInfo counts only MemFree, not the page cache - which, with the experts read from the GGUF in place, is mostly those
  /// files' clean pages and gives way to a device allocation.  There the figure is MemAvailable less 6 GiB for the OS
  ```
- `src/core/expert_cache.cpp` · `(file scope)` → [06-3](BLOCKS.md#06-3)
  ```
  #include <utility>
  #include <cstring>
  ```
- `src/core/expert_cache.cpp` · `size_t device_free_bytes() {` → [06-4](BLOCKS.md#06-4)
  ```
          }
      }
  #endif
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-5](BLOCKS.md#06-5)
  ```
          std::deque<std::pair<Clock::time_point, int64_t>> el_seen;
          Clock::time_point el_last_grow{}, el_last_shrink{};
          int el_backoff = 1;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [06-6](BLOCKS.md#06-6)
  ```
              // and a page-in while the verify graph spins on a host flag stalls the request for good
              size_t free_b = 0, total_b = 0;
              cudaMemGetInfo(&free_b, &total_b);
  ```

## 07 park-no-foreign-kv

`07-park-no-foreign-kv.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la PR upstream #1503: mai parcheggiare una conversazione col K/V di un'altra

- **What**: The conversation cache never parks one conversation with another's retained K/V.
- **Switch (default)**: always on (correctness)
- **Origin**: upstream PR #1503 (sanastasiou)
- **Upstream**: upstream PR #1503 (sanastasiou), not in 0.1.45; 3-way clean
- **Measured**: Correctness only.
- **Docs**: docs/UPSYNC.md
- **Files**: `include/strata/core/conversation_cache.hpp`, `src/core/conversation_cache_test.cpp`, `src/program/generate.cpp`, `sycl/src/program/generate.cpp`

Insertion points (10):

- `include/strata/core/conversation_cache.hpp` · `struct ConversationKvReuse {` → [07-1](BLOCKS.md#07-1)
  ```
      std::vector<ConversationKv> kv;
      // Original image extent for validation, and the earliest subsequent rewrite.
      int64_t captured_tokens = 0, unchanged_tokens = 0;
  ```
- `include/strata/core/conversation_cache.hpp` · `public:` → [07-2](BLOCKS.md#07-2)
  ```
      // Retain only the restored K/V buffers, not duplicate running checkpoints.
      // This optimization never evicts a parked conversation to make itself fit.
      // `stage_kv`: with a layer split, the later stages' restored K/V (one per stage), retained with the first's.
  ```
- `include/strata/core/conversation_cache.hpp` · `public:` → [07-3](BLOCKS.md#07-3)
  ```
          if (reuse_.unchanged_tokens <= 0) reuse_ = {};
      }
      ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
  ```
- `src/core/conversation_cache_test.cpp` · `int main() {` → [07-4](BLOCKS.md#07-4)
  ```
          auto none = conversation_checkpoints_split({}, 3);
          check(conversation_checkpoints_merge(std::move(none), back) && back.empty(), "no checkpoints: nothing to do");
      }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [07-5](BLOCKS.md#07-5)
  ```
                  ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
              } merge_back{cs, checks, n_st > 0};
              const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [07-6](BLOCKS.md#07-6)
  ```
          // conversation an idle slot holds reads on from there.  Every stage, the same calls.
          // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
          auto copy_from_slot = [&](int b, const ConvCheckpoint* at, std::string& e) -> bool {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [07-7](BLOCKS.md#07-7)
  ```
                  if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                      std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                      for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
  ```
- `sycl/src/program/generate.cpp` · `int main(int argc, char **argv) try {` → [07-8](BLOCKS.md#07-8)
  ```
                  ~MergeBack() { if (on && !strata::core::conversation_checkpoints_merge(std::move(cs), checks)) checks.clear(); }
              } merge_back{cs, checks, n_st > 0};
              const strata::core::ConversationView view{live, live_imgs, n_st > 0 ? cs.stage0 : checks, cvec_cached};
  ```
- `sycl/src/program/generate.cpp` · `int main(int argc, char **argv) try {` → [07-9](BLOCKS.md#07-9)
  ```
          // `at`: one of the slot's checkpoints - only the K/V up to it is copied and its state restored instead
          auto copy_from_slot = [&](int b, const ConvCheckpoint *at,
                                    std::string &e) -> bool {
  ```
- `sycl/src/program/generate.cpp` · `int main(int argc, char **argv) try {` → [07-10](BLOCKS.md#07-10)
  ```
                  if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr) {
                      std::vector<std::vector<strata::core::ConversationKv>> stage_kv;   // every stage's, for its next park
                      for (auto& si : incoming->stage_images) stage_kv.push_back(std::move(si.kv));
  ```

## 08 pipeline-architectds

`08-pipeline-architectds.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il profilo architectds della pipeline (STRATA_PIPELINE_ELASTIC) e i log di rollback e latenza della catena

- **What**: The two-window pipeline's profile from the Daily: the verified window is served first (front stage of A, last stage of A, then D and B one layer per pass), lookup beyond one window, theta 0.10; plus two log lines (rollback cost per window class, chain latency).
- **Switch (default)**: `STRATA_PIPELINE_ELASTIC` (on with two cards and `--pipeline-windows`; `=0`: upstream's theta 0.20 and D-A-B order)
- **Origin**: fork (from the architectds lab branch)
- **Upstream**: never proposed
- **Measured**: 0.1.42 without it: -4.6/-7.3/-7.8% against the Daily (speculative windows 133/600 instead of 164/600, stage host 0.50 -> 1.69 ms). With it and patch 11: +0.7% (parity).
- **Depends on**: 04
- **Docs**: docs/UPSYNC.md
- **Files**: `include/strata/core/verify.hpp`, `src/core/verify.cpp`, `src/program/generate.cpp`

Insertion points (20):

- `include/strata/core/verify.hpp` · `public:` → [08-1](BLOCKS.md#08-1)
  ```
      /// window as they WILL be, their pages prefetched).  A later `pl_launch` of the same window (T, pos0, tokens, and
      /// `ss.ple_prev` equal to `ple_prev` by then) skips the staging; anything else stages again.
      bool prestage(int T, const int32_t* tokens, int64_t pos0, const int32_t ple_prev[2], std::string& err);
  ```
- `src/core/verify.cpp` · `bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string` → [08-2](BLOCKS.md#08-2)
  ```
  int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {` → [08-3](BLOCKS.md#08-3)
  ```
      }
      const int G = groups_[T] > 0 ? groups_[T] : 1;
      const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err) {` → [08-4](BLOCKS.md#08-4)
  ```
          if (fl_k_ == 0 && !gather_ple()) return -1;
          *(volatile uint32_t*) h_flag_ = want;
          ++fl_k_;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-5](BLOCKS.md#08-5)
  ```
              int pl_pw = pipe ? o.pipeline_windows : 0;
              float pl_theta = [] {   // a speculative window is launched only when its estimated chance is at least this
                  const char* v = pipe_dbg_env("STRATA_PIPELINE_THETA");
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-6](BLOCKS.md#08-6)
  ```
                      bool spec = false;     // launched speculatively (behind the window before, ahead of its verdict)
                      bool made = false;     // made from a chain (its outcome can be scored even when the gate held it)
                      bool sfx = false;      // a lookup window (the suffix drafter's drafts)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-7](BLOCKS.md#08-7)
  ```
                  bool b_done = false;     // B has been made from the chain in flight
                  int chain_n = 0;
                  int64_t pl_spec = 0, pl_on = 0, pl_undo = 0, pl_gate = 0, pl_fm = 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-8](BLOCKS.md#08-8)
  ```
                                    std::chrono::duration<double, std::milli>(Clock::now() - pl_tr0).count(), ev, seq, x0, x1);
                      pl_trace += b;
                  };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-9](BLOCKS.md#08-9)
  ```
                      // undo restores everything it wrote).  STRATA_PIPELINE_DOOM_SKIP=0: served in full.
                      static const bool doom_skip = [] { const char* e = pipe_dbg_env("STRATA_PIPELINE_DOOM_SKIP"); return e == nullptr || std::atoi(e) != 0; }();
                      const bool skip = doom_skip && doomed && &w == &D;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-10](BLOCKS.md#08-10)
  ```
                          B.p_on = pl_lookup_pon;
                          B.ready = true;
                          B.made = true;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-11](BLOCKS.md#08-11)
  ```
                  // per window class (0: fresh, 1: run speculatively ahead of its predecessor's verdict): the time from
                  // the previous verdict, the tokens it emitted, how many
                  double cls_ms[2] = {0, 0}, cls_tok[2] = {0, 0}, cls_n[2] = {0, 0};
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-12](BLOCKS.md#08-12)
  ```
                              if (!adapt_tick(false, false)) return die("an adaptive refill failed");
                          }
                      }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-13](BLOCKS.md#08-13)
  ```
                              tre("F1", A.seq, A.T);
                          } else if (!err.empty()) return die(err);
                      }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-14](BLOCKS.md#08-14)
  ```
                                      if (!VF(0, B).prestage(B.T, B.tok, B.p, prev, err)) return die(err);
                                  }
                                  b_done = true;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-15](BLOCKS.md#08-15)
  ```
                          if (A.sfx) { ++sfx_windows; sfx_drafts += A.T - 1; sfx_ok += a; }
                          if (A.seq > 0 && !last) policy.observe(A.sfx, A.T, a, A.sfx_match, now - last_verdict);
                          cls_ms[c] += now - last_verdict;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-16](BLOCKS.md#08-16)
  ```
                      if (!last && !ajob && !pl_adapt()) return die("an adaptive refill failed");
                      if (on && !last) {
                          ++pl_on;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-17](BLOCKS.md#08-17)
  ```
                              return die(err);
                          chain_kind = 1;
                          tre("CL", A.seq, 1);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-18](BLOCKS.md#08-18)
  ```
                              if ((A.com & (1u << st)) == 0 && !VF(st, A).pl_commit_async(A.T, err)) return die(err);
                          A = B;
                          B = PW{};
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-19](BLOCKS.md#08-19)
  ```
                                  return die(err);
                              chain_kind = 2;
                              tre("CL", A.seq, 2);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [08-20](BLOCKS.md#08-20)
  ```
                                           "the path, %lld rolled back, %lld below the gate (theta %.2f)\n",
                                   (long long) dec_windows, pl_ms, avg(pl_ms, (double) dec_windows), (long long) pl_spec,
                                   (long long) pl_on, (long long) pl_undo, (long long) pl_gate, theta);
  ```

## 09 df-defaults

`09-df-defaults.patch` — SONNET-STRATA-Rimetti STRATA_DF_BRANCH (CUDA su Windows) e STRATA_DF_PLE (CUDA) accesi di default come nel Daily

- **What**: `STRATA_DF_BRANCH` (mixer side branches in the window graph; CUDA on Windows) and `STRATA_DF_PLE` (PLE post-projections in one pass; CUDA) on by default, as in the Daily.
- **Switch (default)**: `STRATA_DF_BRANCH=0`, `STRATA_DF_PLE=0` turn them off (upstream: opt-in `=1`)
- **Origin**: fork (defaults of PR #904's switches)
- **Upstream**: upstream merged #904 opt-in and keeps BRANCH off after a Linux stall (#905, never seen on Windows)
- **Measured**: Daily measurements: BRANCH +2.4% at 131k, PLE +3.3% (bitwise). On 0.1.42 their absence was the main part of ~1.1 ms of GPU per window.
- **Upstream defaults it changes**: DF_BRANCH, DF_PLE default on (see the table above); STRATA_SIDE_MIXER (upstream, 0.1.45) still wins over STRATA_DF_BRANCH
- **Docs**: docs/UPSYNC.md
- **Files**: `src/core/verify.cpp`

Insertion points (2):

- `src/core/verify.cpp` · `bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&` → [09-1](BLOCKS.md#09-1)
  ```
      {   // STRATA_DF_BRANCH=1 (opt-in): the mixer's side branches.  Off by default: on Linux (open modules, GSP) a
          // stage's window with the branches stopped inside its graph when an NVML query (nvidia-smi) came between
          // requests (2x RTX 3090 and 2x RTX 2080 Ti, #905); never seen on Windows.  0 / unset: one stream, as before
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [09-2](BLOCKS.md#09-2)
  ```
              // below (the norms divide at run time as its norms do; the history advances token by token), which also
              // writes the commit's history snapshots.  Scratch: layer 0's expert rows, combined by now, and the
              // hyper-connection read's `lo_`, which the read below rewrites.
  ```

## 10 ple-diag

`10-ple-diag.patch` — SONNET-STRATA-Rimetti la diagnostica PLE di prefill e decode (letture lente, gather del decode con STRATA_DECODE_TIMING)

- **What**: PLE diagnostics: prefill reads (mean / max / over 100 ms) and the chunk that waits more than a second for its rows; decode gather time (`STRATA_DECODE_TIMING`) and the reader counters before/after a request.
- **Switch (default)**: log lines; `STRATA_DECODE_TIMING=1` for the decode ones
- **Origin**: fork (ex patch 13 of the 0.1.43 stack, diagnostics half)
- **Upstream**: Niko1221 on PR #2073: the diagnostics are welcome as a separate small PR; the direct-read default itself is upstream since 0.1.43.1 (STRATA_PLE_AUTO_MMAP=1 for the old switch)
- **Measured**: diagnostics only; found the 1 259 rows/s probe and the 260 s -> 75 s prefill of a cold SATA SSD
- **Docs**: docs/UPSYNC_MAIN.md
- **Files**: `include/strata/core/verify.hpp`, `include/strata/ngram/ple_reader.hpp`, `src/core/verify.cpp`, `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`

Insertion points (12):

- `include/strata/core/verify.hpp` · `public:` → [10-1](BLOCKS.md#10-1)
  ```
      bool pcie_balance_on() const { return bal_ != nullptr && m_bal_ != nullptr; }
  
      double ms_wait = 0, ms_pool = 0, ms_host = 0, ms_commit = 0;
  ```
- `include/strata/ngram/ple_reader.hpp` · `struct ReaderStats {` → [10-2](BLOCKS.md#10-2)
  ```
      double wait_us = 0;           ///< time `collect` spent blocked
      double submit_us = 0;         ///< time spent inside the read submission call (non-zero = it blocks)
      double read_us_sum = 0;       ///< sum of per-read latencies (issue to completion)
  ```
- `src/core/verify.cpp` · `constexpr float EPS = 1e-6f;` → [10-3](BLOCKS.md#10-3)
  ```
  using Clock = std::chrono::steady_clock;
  double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
  const bool g_dbg = std::getenv("STRATA_VERIFY_DEBUG") != nullptr;
  ```
- `src/core/verify.cpp` · `bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool,` → [10-4](BLOCKS.md#10-4)
  ```
              _mm_sfence();
              *flag = 1;
              ms_host += ms_since(tp);
  ```
- `src/core/verify.cpp` · `bool Verifier::run(int T, const int32_t* tokens, int64_t pos0, PoolMultiFn pool,` → [10-5](BLOCKS.md#10-5)
  ```
              std::atomic_thread_fence(std::memory_order_seq_cst);
              _mm_sfence();
              ms_host += ms_since(tp);
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la` → [10-6](BLOCKS.md#10-6)
  ```
          std::atomic_thread_fence(std::memory_order_seq_cst);
          _mm_sfence();
          ms_host += ms_since(tp);
  ```
- `src/kernels/ngram.cpp` · `void PleTable::set_injected_delay_us(double us) { impl_->reader.set_injected_del` → [10-7](BLOCKS.md#10-7)
  ```
  std::string PleTable::io_report() const {
      if (impl_->mode != PleIo::Direct || !impl_->reader.is_open()) return {};
      const strata::ngram::ReaderStats s = impl_->reader.snapshot();
  ```
- `src/kernels/ngram.cpp` · `std::string PleTable::io_report() const {` → [10-8](BLOCKS.md#10-8)
  ```
                    (unsigned long long) s.reads, (double) s.bytes / 1e6, s.percentile(0.5), s.percentile(0.99),
                    s.wait_us / 1000.0, s.submit_us / 1000.0, (unsigned long long) impl_->reader.cache_size(),
                    (unsigned long long) impl_->reader.cache_capacity());
  ```
- `src/ngram/ple_reader.cpp` · `struct PleReader::Impl {` → [10-9](BLOCKS.md#10-9)
  ```
  
      void record_latency(double us) {
          stats.read_us_sum += us;
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [10-10](BLOCKS.md#10-10)
  ```
                  });
              }
              ple_buf ^= 1;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [10-11](BLOCKS.md#10-11)
  ```
              const Clock::time_point d0 = Clock::now();
              // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
              static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [10-12](BLOCKS.md#10-12)
  ```
                      if (st == 0) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());   // text as before
                      else std::fprintf(stderr, "strata decode GPU stages, stage %d (ms/window):%s\n", st, pr.c_str());
                  }
  ```

## 11 prefill-sm75

`11-prefill-sm75.patch` — SONNET-STRATA-Tieni accesi di default su sm_75 STRATA_SELECT_SIMT e il dequant a 16 B come nel Daily

- **What**: The Daily's Turing prompt defaults: `STRATA_SELECT_SIMT` (FP32 tiled QSA block scores) on below sm_80, and the 16-byte dequant of the prompt projections (upstream's #2081) on for a 7.5 card.
- **Switch (default)**: `STRATA_SELECT_SIMT=0|2`, `STRATA_TC_SM75=0` (sm_75 only)
- **Origin**: fork (SIMT scores from PR #1091 konijiwa110, merged upstream opt-in; dequant from our PR #2081, merged upstream behind STRATA_SM75_PERF=1)
- **Upstream**: SIMT: PR #2080 (default on) declined (Niko1221, 17 Oct: changes the summation order, wants an independent sm_75 KL run); dequant 16 B: in 0.1.45 as `STRATA_TC_SM75` / `STRATA_SM75_PERF=1`. hc16 (FP16 tensor-core hyper-connection projections) NOT ported: 0.1.45 has the wider `STRATA_BF16_TC=1` + `STRATA_XN_F16` (#655 / #2091)
- **Measured**: 131k prefill 100 -> 75 s on the 2080 Ti stage with the 0.1.43 stack (SIMT -16%, hc16 -19.5% of GPU prompt time). On 0.1.45 the hc16 share has to be re-measured with STRATA_BF16_TC=1 (timing of the prompt and a KL against the default).
- **Upstream defaults it changes**: SELECT_SIMT default on below sm_80; STRATA_TC_SM75 default on for a 7.5 card
- **Docs**: docs/UPSYNC_MAIN.md
- **Files**: `include/strata/core/tc_sm75.hpp`, `include/strata/kernels/qsa_select.hpp`, `src/kernels/cuda/qsa_select.cu`, `src/prefill/prefill.cpp`

Insertion points (4):

- `include/strata/core/tc_sm75.hpp` · `(file scope)` → [11-1](BLOCKS.md#11-1)
  ```
  //
  // 0.1.45: the Turing paths are opt-in (nobody here has an sm_75 card to confirm them): STRATA_SM75_PERF=1
  // (sm75_series.hpp) turns them on, STRATA_TC_SM75=0 / 1 forces this one off / on (still only on a 7.5 card).
  ```
- `include/strata/kernels/qsa_select.hpp` · `bool qsa_block_scores_rows(const float* pooled, const float* dead, const float*` → [11-2](BLOCKS.md#11-2)
  ```
  /// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
  /// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
  /// On AMD it is the gfx12 (RDNA4) WMMA kernel (a three-way bf16 split, six products); false on any other AMD target.
  ```
- `src/kernels/cuda/qsa_select.cu` · `bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_` → [11-3](BLOCKS.md#11-3)
  ```
              cc_major[dev] = w && (!std::strcmp(w, "1") || !std::strcmp(w, "select")) ? 7 : strata::cc_major_of(major);
          }
          if (cc_major[dev] < 8) {
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [11-4](BLOCKS.md#11-4)
  ```
                          // C-1: the grid reaches the batch's last query's n_bid (they rise with the position)
                          const int64_t active = (int64_t) m.steps_host[(size_t) ((t0 + nb - 1) * strata::kernels::kStepCount +
                                                                                  strata::kernels::kStepNBid)] + 1;
  ```

## 12 ram-core

`12-ram-core.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il nucleo fisso per scheda senza copia RAM (STRATA_RAM_CORE=1, STRATA_RAM_CORE_SLOTS_DEV<n>)

- **What**: A fixed core of expert slots per card, excluded from swaps, loans and shrink, whose RAM copy is given back (the resident arena no longer mirrors experts that never leave VRAM).
- **Switch (default)**: `STRATA_RAM_CORE=1` (off), `STRATA_RAM_CORE_SLOTS_DEV<n>=<slots>`
- **Origin**: fork
- **Upstream**: never proposed
- **Measured**: `STRATA_RAM_CORE_SLOTS_DEV1=5500`: arena 39.97 -> 29.62 GiB (-10.35), working set 48.7 -> 33.6 GB, t/s +0.1%, hit rate unchanged, 0 re-reads from file. Daily default since 12 Oct.
- **Notes**: Ported to 0.1.45: upstream made `reads_` atomic (`fetch_add`); the core's recovery read is counted the same way.
- **Docs**: docs/RAM_CORE.md
- **Files**: `docs/RAM_CORE.md`, `include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp`, `src/program/generate.cpp`

Insertion points (24):

- `docs/RAM_CORE.md` — (new file) → [12-1](BLOCKS.md#12-1)
- `include/strata/core/expert_source.hpp` · `private:` → [12-2](BLOCKS.md#12-2)
  ```
  // Phase 3's cache fills and the CPU/PCIe miss split - but registration is best-effort and reported, not assumed.
  class ArenaExpertSource : public ExpertSource {
  public:
  ```
- `include/strata/core/expert_source.hpp` · `public:` → [12-3](BLOCKS.md#12-3)
  ```
      double load_read_seconds() const { return load_read_s_; }
      double load_copy_seconds() const { return load_copy_s_; }
  ```
- `src/core/expert_source.cpp` · `LoadStats load_experts_gguf(const std::string& gguf, uint8_t* dst, const strata:` → [12-4](BLOCKS.md#12-4)
  ```
      return st;
  }
  ```
- `src/core/expert_source.cpp` · `bool ArenaExpertSource::open(const std::string& pack_dir, int64_t n_layers, int6` → [12-5](BLOCKS.md#12-5)
  ```
      arena_ = a;
      base_ = a->data();
      pinned_bytes_ = a->registered_bytes;
  ```
- `src/core/expert_source.cpp` · `void ArenaExpertSource::close() {` → [12-6](BLOCKS.md#12-6)
  ```
      map_ = nullptr;
      map_bytes_ = 0;
      inputs_.clear();
  ```
- `src/core/expert_source.cpp` · `void ArenaExpertSource::close() {` → [12-7](BLOCKS.md#12-7)
  ```
      base_ = nullptr;
      blobs_ = 0;
      n_expert_ = 0;
  ```
- `src/core/expert_source.cpp` · `const uint8_t* ArenaExpertSource::blob(int64_t layer, int64_t expert) {` → [12-8](BLOCKS.md#12-8)
  ```
      const int64_t idx = layer * n_expert_ + expert;
      if (idx < 0 || idx >= blobs_) return nullptr;
      reads_.fetch_add(1, std::memory_order_relaxed);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-9](BLOCKS.md#12-9)
  ```
      // fork (Daily 12a6f553): beside the pipeline the prompt path may borrow the core too - nothing decodes while a prompt
      // is read, and the lent slots are refilled before the decode as the tail's are (with only the 1.5 GiB tail lent the
      // 131k prompt read fell from ~85 s to 134 s)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-10](BLOCKS.md#12-10)
  ```
              int32_t first_now = -1;        // where its buffers are laid out now
              int64_t lent_chunk = 0;
              std::vector<std::pair<int32_t, int32_t>> lent;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-11](BLOCKS.md#12-11)
  ```
          // keeps the pinned-share rule.  STRATA_SPLIT_RING=N: N slots on a split; 0: the pinned-share rule.
          // (decided above, before the lend regions were sized: a no-op here unless the resident RAM mode skipped it)
          decide_split_ring();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-12](BLOCKS.md#12-12)
  ```
              pf_parts.push_back({&xcache, &ss, &sp, -1, 0, multi_gpu ? split_at[0] : g.n_layers, -1, -1, 0, {}});
              for (auto& st : stages)
                  pf_parts.push_back({&st->cache, &st->ss, &st->sp, st->dev, st->lb, st->le, -1, -1, 0, {}});
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-13](BLOCKS.md#12-13)
  ```
              auto fits_one = [&](const PfPart& p, int64_t c, bool cap) -> bool {
                  const int64_t k = part_slots(p, c);
                  if (p.cache == &xcache && k > xcache.slots() - el_lend_floor) return false;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-14](BLOCKS.md#12-14)
  ```
                                                                    kAutoLendPct * p.cache->slots() / 100));
                      // --elastic: the two-zone cache lends only its tail (fits_one refuses the core as well)
                      if (p.cache == &xcache) budget = std::min(budget, std::max<int64_t>(0, xcache.slots() - el_lend_floor));
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-15](BLOCKS.md#12-15)
  ```
          // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
          // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
          // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-16](BLOCKS.md#12-16)
  ```
                      if (r[e] < 0) { if (u[e] >= 2.0f && !(peer.valid() && peer.has(l, e)) && !(remote_opt && remote_opt->owns(l, e)) && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
                      // --elastic: the core never leaves - unless core_adapt; beside the pipeline el_core counts CUDA0's
                      // slots and a later stage's layers hold slots of their own cache
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-17](BLOCKS.md#12-17)
  ```
                  agate.has.assign((size_t) g.n_layers, 0);
                  for (const Swap& s : swaps) agate.has[(size_t) s.layer] = 1;
              }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-18](BLOCKS.md#12-18)
  ```
          static const bool el_trace = std::getenv("STRATA_ELASTIC_TRACE") != nullptr;
          const int64_t el_step_max = 512ll << 20;   // at most this much per growth step (its copies are async)
          const int64_t el_shrink_step = std::max<int64_t>(64ll << 20, (int64_t) o.elastic_shrink_step_mib << 20);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-19](BLOCKS.md#12-19)
  ```
              int64_t c = el_chunk_max;
              const int64_t k = plan_lend(c);
              if (k <= 0 || c > el_chunk_max) return;   // nothing fits (not above el_min_slots): keep the plan
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-20](BLOCKS.md#12-20)
  ```
              for (int64_t l = 0; l < g.n_layers; ++l)
                  for (int32_t ex = 0; ex < (int32_t) g.n_expert && el_layer(l); ++ex) {
                      const size_t i = (size_t) (l * g.n_expert + ex);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-21](BLOCKS.md#12-21)
  ```
                  const int32_t* r = a_res.data() + l * g.n_expert;
                  for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                      if (r[e] < 0) { if (u[e] >= 2.0f && !helper_holds(l, e)) cand.emplace_back(u[e], e); }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-22](BLOCKS.md#12-22)
  ```
              std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
              std::fflush(stderr);
          };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-23](BLOCKS.md#12-23)
  ```
              const int64_t want_free = (int64_t) reserve << 20, mapped = xcache.mapped_bytes();
              const int64_t before = xcache.slots();
              // what stays at least: the prompt path's smallest loan (256 tokens) and the 128 slots it must leave
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [12-24](BLOCKS.md#12-24)
  ```
                          (long long) req_offload, chain_txt, (long long) req_chunk,
                          (long long) (sp.chain_experts_streamed() - streamed0));
              std::fflush(stdout);
  ```

## 13 sa-bet

`13-sa-bet.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la scommessa sulla testa dello stadio 0 (STRATA_SA_BET, STRATA_SA_CHAIN, opt-in)

- **What**: STRUCT-A: while the MTP chain runs, stage 0 serves the HEAD of the next window as a bet (`STRATA_SA_BET`, layers `_LAYERS`, widths `_MAXT`/`_MINT`); the forced chain is fed a step at a time and cut once B is decided (`STRATA_SA_CHAIN`, `_AHEAD`).
- **Switch (default)**: `STRATA_SA_BET=1`, `STRATA_SA_CHAIN=1` (off)
- **Origin**: fork (lab/struct-a 075ce875)
- **Upstream**: never proposed
- **Measured**: +2.5% against the 0.1.43 Daily on the 131k forced bench (SA_BET + SA_CHAIN, B-A-B, 12-13 Oct). Not re-measured on 0.1.45.
- **Docs**: docs/STRUCT_A.md
- **Files**: `docs/STRUCT_A.md`, `include/strata/core/expert_source.hpp`, `include/strata/core/mtp.hpp`, `include/strata/core/verify.hpp`, `src/core/expert_source.cpp`, `src/core/mtp.cpp`, `src/core/verify.cpp`, `src/program/generate.cpp`

Insertion points (31):

- `docs/STRUCT_A.md` — (new file) → [13-1](BLOCKS.md#13-1)
- `include/strata/core/expert_source.hpp` · `struct ExpertDispatch {` → [13-2](BLOCKS.md#13-2)
  ```
      /// whose routed ids all look resident there publishes no activation (doorbell_publish_res) - counted
      const int32_t* dres_check = nullptr;
      int64_t stale_x = 0;
  ```
- `include/strata/core/mtp.hpp` · `public:` → [13-3](BLOCKS.md#13-3)
  ```
      /// own pick, and step j (1 <= j <= n_force) takes force[j - 1] as its input token instead of output j - 1.
      /// `n_early`: an event after each of the first `n_early` outputs (chain_outputs_ready).
      bool chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force, int n_out,
  ```
- `include/strata/core/mtp.hpp` · `private:` → [13-4](BLOCKS.md#13-4)
  ```
      cudaEvent_t ev_chain_ = nullptr;
      cudaEvent_t ev_step_[8] = {};                       ///< after each of the first n_early outputs
      int steps_seen_ = 0, chain_n_ = 0, chain_early_ = 0;
  ```
- `include/strata/core/verify.hpp` · `public:` → [13-5](BLOCKS.md#13-5)
  ```
      /// The commit without a host sync; `ss.ple_prev` advances now (host side).  A second call for the same window
      /// (after its state was restored) replays it with another count.
      bool pl_commit_async(int n_keep, std::string& err);
  ```
- `include/strata/core/verify.hpp` · `private:` → [13-6](BLOCKS.md#13-6)
  ```
      bool fl_active_ = false, fl_prof_ = false, fl_ple_ = false, commit_live_ = false, pl_prestaged_ = false;
      int fl_T_ = 0;
      int64_t fl_k_ = 0, fl_total_ = 0;
  ```
- `src/core/expert_source.cpp` · `void expert_pool_dispatch_multi(ExpertDispatch& d, const float* x_f, const int32` → [13-7](BLOCKS.md#13-7)
  ```
      pt("begin");
      d.src->begin_layer(d.layers, ids, n_tok * k);
      pt("begun");
  ```
- `src/core/mtp.cpp` · `bool MtpDrafter::prepare_chain(std::string& err) {` → [13-8](BLOCKS.md#13-8)
  ```
  }
  
  bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force,
  ```
- `src/core/mtp.cpp` · `bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, co` → [13-9](BLOCKS.md#13-9)
  ```
      std::atomic_thread_fence(std::memory_order_seq_cst);
      if (!stage_source_R(T, err)) return false;
      chain_early_ = std::max(0, std::min(n_early, n_out));
  ```
- `src/core/mtp.cpp` · `bool MtpDrafter::chain_launch(int T, const int32_t* tokens, int64_t p, int a, co` → [13-10](BLOCKS.md#13-10)
  ```
      steps_seen_ = 0;
      chain_live_ = true;
      chain_n_ = n_out;
  ```
- `src/core/mtp.cpp` · `int MtpDrafter::chain_outputs_ready(std::string& err) {` → [13-11](BLOCKS.md#13-11)
  ```
  
  int MtpDrafter::chain_poll(std::string& err) {
      if (!chain_live_) return 1;
  ```
- `src/core/verify.cpp` · `Verifier::~Verifier() {` → [13-12](BLOCKS.md#13-12)
  ```
          if (e) cudaGraphExecDestroy(e);
      for (auto& e : exec_nr_)
          if (e) cudaGraphExecDestroy(e);
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [13-13](BLOCKS.md#13-13)
  ```
  
      // ---- the window's inputs, from mapped staging.  With branches the steps and positions (first read by the first
      // layer's mixer) cross PCIe on side 1 beside the embedding and the first hyper-connection read.
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [13-14](BLOCKS.md#13-14)
  ```
      // ---- the embeddings, broadcast to the hc streams - or, in a later stage of a layer split, the previous stage's
      // residual, pending write and inject (see set_stage)
      const int64_t HB = Verifier::handoff_floats(g);
  ```
- `src/core/verify.cpp` · `bool Verifier::record_window(int T, cudaStream_t cs, std::string& err) {` → [13-15](BLOCKS.md#13-15)
  ```
      };
  
      for (int grp = 0; grp < G; ++grp)
  ```
- `src/core/verify.cpp` · `bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string` → [13-16](BLOCKS.md#13-16)
  ```
  
  int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_layers) {
      if (!fl_active_) return 1;
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la` → [13-17](BLOCKS.md#13-17)
  ```
      const int G = groups_[T] > 0 ? groups_[T] : 1;
      const int gtb[2] = {0, (T + 1) / 2}, gte[2] = {G == 2 ? (T + 1) / 2 : T, T};
      int served = 0;   // this call's layers (a layer's halves count apart when its window is split in two groups)
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la` → [13-18](BLOCKS.md#13-18)
  ```
              const double now = now_ms();
              if (now - fl_flush_ms_ > 2.0) {   // flush WDDM and notice a dead graph, as run() does
                  fl_flush_ms_ = now;
  ```
- `src/core/verify.cpp` · `bool Verifier::pl_commit_async(int n_keep, std::string& err) {` → [13-19](BLOCKS.md#13-19)
  ```
  void Verifier::absorb_stats(Verifier& o) {
      ms_wait += o.ms_wait; ms_pool += o.ms_pool; ms_host += o.ms_host; ms_commit += o.ms_commit;
      windows += o.windows;
  ```
- `src/program/generate.cpp` · `void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t` → [13-20](BLOCKS.md#13-20)
  ```
      drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-21](BLOCKS.md#13-21)
  ```
                  bool early_used = false;
                  bool b_done = false;     // B has been made from the chain in flight
                  int chain_n = 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-22](BLOCKS.md#13-22)
  ```
                      return v == nullptr || std::atoi(v) != 0;
                  }();
                  auto ms_now = [&]() { return std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count(); };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-23](BLOCKS.md#13-23)
  ```
                          } else if (!err.empty()) return die(err);
                      }
                      if (pl_yield && ((doomed && !pump0(D)) || !pump0(B, A.s1 && !A.s1_done ? 1 : 0))) return die(err);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-24](BLOCKS.md#13-24)
  ```
                          tre("U", undo_w.seq, undo_keep);
                      }
                      if (ending) {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-25](BLOCKS.md#13-25)
  ```
                      if (chain_kind != 0) {
                          const int k_ready = mtp.chain_outputs_ready(err);
                          if (k_ready < 0) return die(err);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-26](BLOCKS.md#13-26)
  ```
                                                           "outputs)\n", chain_kind, (long long) A.p, A.T, B.T, B.p_on, k);
                              }
                          }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-27](BLOCKS.md#13-27)
  ```
                      // the drafter must be idle before the next chain.  The chain needs nothing from the host, so this
                      // wait is bounded even with B (made from its first outputs) in flight on stage 0; in practice the
                      // chain ended long before a verdict that needs a whole stage-1 window.
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-28](BLOCKS.md#13-28)
  ```
                          // the chain over A, forced through B's drafts: B's bonus guess and the next window's drafts
                          mtp.set_source_R(V1(A).final_R(0));
                          chain_n = std::min(strata::kernels::kVerifyMaxT - 1, B.T + S_mtp - 1);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-29](BLOCKS.md#13-29)
  ```
                              nA.p = A.p + a + 1;
                              nA.tok[0] = outp[(size_t) a];
                              A = nA;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-30](BLOCKS.md#13-30)
  ```
                  while (mtp.chain_live() && mtp.chain_poll(err) == 0) {}
                  mtp.set_source_R(ver.final_R_all());   // the serial loop's rows again
                  const double pl_ms = std::chrono::duration<double, std::milli>(Clock::now() - pl_t0).count();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [13-31](BLOCKS.md#13-31)
  ```
                      std::fprintf(stderr, "strata pipeline: drafter on CUDA%d: %.3f ms of host time per chain launch (%.3f ms of it "
                                           "staging the source rows, %lld times so far)\n", mtp.device(),
                                   avg(mtp.ms_draft, (double) mtp.rounds), avg(mtp.ms_src, (double) mtp.n_src), (long long) mtp.n_src);
  ```

## 14 prefix-disk-multi

`14-prefix-disk-multi.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il prefisso su disco su un thread e un file per prefisso (STRATA_PREFIX_DISK_MULTI)

- **What**: `STRATA_PREFIX_DISK_MULTI=<GiB>`: one file per prefix beside `--prompt-cache-file` (`<file>.<16 hex>`), least recently used out past the cap; the file is written on a thread (`STRATA_PREFIX_DISK_SYNC=1` inline).
- **Switch (default)**: `STRATA_PREFIX_DISK_MULTI=<GiB>` (off: the single file)
- **Origin**: fork (lab/ttft43 c00a8463)
- **Upstream**: never proposed (depends on 05)
- **Measured**: TTFT bench 13 Oct with #1852 (now upstream): alternating clients 38.2 / 32.5 -> 16.2 / 4.9 s; after a restart 20.2 / 10.3 -> 6.4 / 5.1 s (with STRATA_CONV_DISK).
- **Depends on**: 05
- **Docs**: docs/TTFT43.md (fork branch lab/ttft43)
- **Files**: `include/strata/core/prefix_cache_file.hpp`, `src/core/prefix_cache_file.cpp`, `src/core/prefix_cache_file_test.cpp`, `src/program/generate.cpp`

Insertion points (11):

- `include/strata/core/prefix_cache_file.hpp` · `bool prefix_cache_read(const std::filesystem::path &path,` → [14-1](BLOCKS.md#14-1)
  ```
                         const std::vector<int32_t> &expected_prefix,
                         int32_t next_token, bool cvec, SavedConversation &out,
                         std::string &error);
  ```
- `src/core/prefix_cache_file.cpp` · `(file scope)` → [14-2](BLOCKS.md#14-2)
  ```
  #include "strata/core/prefix_cache_file.hpp"
  ```
- `src/core/prefix_cache_file.cpp` · `(file scope)` → [14-3](BLOCKS.md#14-3)
  ```
  #include <memory>
  #include <set>
  #include <stdexcept>
  ```
- `src/core/prefix_cache_file.cpp` · `bool prefix_cache_read(const std::filesystem::path &path,` → [14-4](BLOCKS.md#14-4)
  ```
  } // namespace strata::core
  ```
- `src/core/prefix_cache_file_test.cpp` · `int main() {` → [14-5](BLOCKS.md#14-5)
  ```
      reject_split_write("split-above-token-cap", over_cap);
    }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-6](BLOCKS.md#14-6)
  ```
              if (wchar_t* block = GetEnvironmentStringsW()) {
                  for (const wchar_t* p = block; *p; p += std::wcslen(p) + 1) {
                      if (_wcsnicmp(p, L"STRATA_", 7) && _wcsnicmp(p, L"CUDA_", 5)) continue;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-7](BLOCKS.md#14-7)
  ```
              } else { prefix_disk_enabled = false; err = "environment identity unavailable"; }
  #else
              for (char** p = environ; p && *p; ++p)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-8](BLOCKS.md#14-8)
  ```
                           prefix_disk_enabled ? "" : ": ", prefix_disk_enabled ? "" : err.c_str());
              err.clear();
          }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-9](BLOCKS.md#14-9)
  ```
                  if (disk_root >= o.prompt_cache_root && disk_root <= int64_t(strata::core::prefix_cache_max_tokens) &&
                          disk_root > std::max(resume, (incoming || borrow) ? incoming_tokens : int64_t(0))) {
                      const auto t0 = Clock::now();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-10](BLOCKS.md#14-10)
  ```
                              return strata::core::conversation_snapshot_validate(im, ss, g,
                                         static_cast<const strata::core::QsaState*>(nullptr), err);
                          };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [14-11](BLOCKS.md#14-11)
  ```
                                  // A CUDA copy/synchronization error can poison the live session.
                                  std::printf("ERR capturing prefix disk cache: %s\n", err.c_str()); return 1;
                              }
  ```

## 15 ple-lookahead

`15-ple-lookahead.patch` — SONNET-STRATA-Porta su upstream 0.1.45 la lettura in anticipo delle righe PLE del primo chunk della parte dopo (STRATA_PLE_LOOKAHEAD)

- **What**: The PLE rows of the first chunk of the next prompt part are read while the previous part finishes, and beside the prefix restore (`Prefill::set_ple_lookahead`, `ple_warm_now`).
- **Switch (default)**: `STRATA_PLE_LOOKAHEAD=0` turns it off (ON by default, direct PLE only)
- **Origin**: fork (lab/ttft43 ea73f49e)
- **Upstream**: never proposed
- **Measured**: part of the TTFT bench of patch 14 (about 0.7 s of file read hidden per restore).
- **Depends on**: 14
- **Upstream defaults it changes**: nothing in upstream: its own switch, on by default as in the 0.1.43 Daily r2
- **Files**: `include/strata/kernels/ngram.hpp`, `include/strata/ngram/ple_reader.hpp`, `include/strata/prefill/prefill.hpp`, `src/kernels/ngram.cpp`, `src/ngram/ple_reader.cpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`

Insertion points (13):

- `include/strata/kernels/ngram.hpp` · `public:` → [15-1](BLOCKS.md#15-1)
  ```
      /// as ONE reader request - page dedupe and sort across the whole batch, the reader's full queue depth.  Not
      /// while a single-token `issue` is pending.  The mapped mode gathers row by row.
      bool gather_batch(const uint32_t* rows, size_t n_tokens, float* out, std::string& err);
  ```
- `include/strata/ngram/ple_reader.hpp` · `public:` → [15-2](BLOCKS.md#15-2)
  ```
      void reset_stats();
      uint64_t cache_capacity() const;
      uint64_t cache_size() const;
  ```
- `include/strata/prefill/prefill.hpp` · `(file scope)` → [15-3](BLOCKS.md#15-3)
  ```
  #include <future>
  #include <memory>
  #include <string>
  ```
- `include/strata/prefill/prefill.hpp` · `public:` → [15-4](BLOCKS.md#15-4)
  ```
  
      const PrefillStats& stats() const { return stats_; }
  ```
- `include/strata/prefill/prefill.hpp` · `private:` → [15-5](BLOCKS.md#15-5)
  ```
      std::future<bool> next_run_;
      int hand_buf_ = 0;
  ```
- `src/kernels/ngram.cpp` · `bool PleTable::gather_batch(const uint32_t* rows, size_t n_tokens, float* out, s` *(anchor ambiguous in the base file)* → [15-6](BLOCKS.md#15-6)
  ```
      return true;
  }
  ```
- `src/ngram/ple_reader.cpp` · `void PleReader::reset_stats() {` → [15-7](BLOCKS.md#15-7)
  ```
  }
  uint64_t PleReader::cache_capacity() const { return impl_->cache.sets * WAYS; }
  uint64_t PleReader::cache_size() const { return impl_->cache.used; }
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [15-8](BLOCKS.md#15-8)
  ```
          return ss.ple.table->gather_batch(m.ple_rows[buf].data(), (size_t) T, m.ple_emb_host[buf], e);
      };
      std::string ple_next_err;
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [15-9](BLOCKS.md#15-9)
  ```
          // were read ahead during the previous chunk; the first chunk's are read beside layer 0 - they are needed from
          // layer 1 on, and gathering them here first left the GPU idle for the whole read (~0.4 s of a 32K prompt)
          if (ple_on && !ple_next.valid())
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [15-10](BLOCKS.md#15-10)
  ```
                      strata::aux_cpus::pin_current_thread();
                      return ple_gather(c1, b, ple_next_err);
                  });
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::reserve_uploads(int64_t T) {` → [15-11](BLOCKS.md#15-11)
  ```
  bool Prefill::run(const int64_t* tokens, int64_t n, int64_t pos0, std::string& err) {
  #if defined(STRATA_GPU_METAL)
      {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-12](BLOCKS.md#15-12)
  ```
                          std::error_code kec;
                          auto keyed = strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix,
                                                                             int32_t(ids[size_t(disk_root)]));
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [15-13](BLOCKS.md#15-13)
  ```
                  }
                  if (win) merge_at = merge_from(at, to);   // (after the refill: the loans are back)
                  const int64_t read_to = merge_at >= 0 ? merge_at : to;   // (merge_at > at: the part's windows before it)
  ```

## 16 conv-disk

`16-conv-disk.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il salvataggio su disco della conversazione al confine di turno a motore fermo (STRATA_CONV_DISK)

- **What**: `STRATA_CONV_DISK=<s>`: after a request, once the engine has been idle that long, the live conversation at its last turn boundary goes to a keyed file; a request looks for the longest saved boundary among its own.
- **Switch (default)**: `STRATA_CONV_DISK=<idle seconds>`, `STRATA_CONV_DISK_MIN` (4096) (off; needs `--prompt-cache-file` and `STRATA_PREFIX_DISK_MULTI`, no `--batch`)
- **Origin**: fork (lab/ttft43 abd685c5)
- **Upstream**: never proposed
- **Measured**: a 130k conversation restarts from its file instead of the system turn alone (65-80 s of prefill); 13 Oct: a restart took 277 s with it on, later shown not to come from CONV_DISK (TTFT43.md).
- **Depends on**: 14
- **Notes**: Ported to 0.1.45: the idle wait is a branch after upstream's pipeline drain-with-trace; the in-RAM comparison counts the prefix borrow (`incoming || borrow`).
- **Files**: `include/strata/core/prefix_cache_file.hpp`, `src/core/prefix_cache_file.cpp`, `src/core/prefix_cache_file_test.cpp`, `src/program/generate.cpp`

Insertion points (12):

- `include/strata/core/prefix_cache_file.hpp` · `bool prefix_cache_read(const std::filesystem::path &path,` → [16-1](BLOCKS.md#16-1)
  ```
  std::filesystem::path prefix_cache_keyed_path(
      const std::filesystem::path &base, const std::vector<int32_t> &prefix,
      int32_t next_token);
  ```
- `src/core/prefix_cache_file.cpp` · `bool prefix_cache_read(const std::filesystem::path &path,` → [16-2](BLOCKS.md#16-2)
  ```
  std::filesystem::path prefix_cache_keyed_path(
      const std::filesystem::path &base, const std::vector<int32_t> &prefix,
      int32_t next_token) {
  ```
- `src/core/prefix_cache_file_test.cpp` · `int main() {` → [16-3](BLOCKS.md#16-3)
  ```
      check(ka.parent_path() == base.parent_path() &&
                ka.filename().string().size() == base.filename().string().size() + 17,
            "keyed path is <base>.<16 hex>");
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-4](BLOCKS.md#16-4)
  ```
                      // fork (TTFT): the switches of the disk store and of the PLE read-ahead change no computed
                      // value, so they do not make the saved prefixes another engine's
                      if (!_wcsnicmp(p, L"STRATA_PREFIX_DISK_MULTI=", 25) || !_wcsnicmp(p, L"STRATA_PREFIX_DISK_SYNC=", 24) ||
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-5](BLOCKS.md#16-5)
  ```
              for (char** p = environ; p && *p; ++p)
                  if ((!std::strncmp(*p, "STRATA_", 7) || !std::strncmp(*p, "CUDA_", 5)) &&
                      std::strncmp(*p, "STRATA_PREFIX_DISK_MULTI=", 25) && std::strncmp(*p, "STRATA_PREFIX_DISK_SYNC=", 24) &&
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-6](BLOCKS.md#16-6)
  ```
              return v != nullptr && std::atoi(v) != 0;
          }();
          std::future<void> prefix_write_job;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-7](BLOCKS.md#16-7)
  ```
                  if (!pump(false)) return false;
              return true;
          };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-8](BLOCKS.md#16-8)
  ```
              } else if (!next_line(line)) {
                  break;
              }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-9](BLOCKS.md#16-9)
  ```
                  int64_t disk_root = -1, last_turn = -1;
                  for (int64_t i = n - 1; i > 0; --i) if (ids[size_t(i)] == o.turn_token) { last_turn = i; break; }
                  for (int64_t i = 1; i < last_turn; ++i) if (ids[size_t(i)] == o.turn_token) { disk_root = i; break; }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-10](BLOCKS.md#16-10)
  ```
                                         static_cast<const strata::core::QsaState*>(nullptr), err);
                          };
                          if (strata::core::prefix_cache_read(disk_path, prefix_identity, prefix,
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-11](BLOCKS.md#16-11)
  ```
                      std::printf("ERR saving a conversation checkpoint failed%s\n", ckpt_why.c_str());
                      return 1;
                  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [16-12](BLOCKS.md#16-12)
  ```
                              const std::filesystem::path disk_target = prefix_multi_cap > 0
                                  ? strata::core::prefix_cache_keyed_path(o.prompt_cache_file, prefix, int32_t(ids[size_t(to)]))
                                  : std::filesystem::path(o.prompt_cache_file);
  ```

## 17 p2-flush-noswitch

`17-p2-flush-noswitch.patch` — SONNET-STRATA-Porta su upstream 0.1.45 il polling della pipeline senza cambio di scheda (STRATA_P2_FLUSH_NOSWITCH, STRATA_P2_SWITCH_STAT, STRATA_P2_SMEM, STRATA_P2_FLUSH_US)

- **What**: Pipeline polling without switching card: `STRATA_P2_FLUSH_NOSWITCH` (service, done and the chain read their mapped flags with no OnDevice), `STRATA_P2_FLUSH_US` (flush period), `STRATA_P2_SWITCH_STAT` (host time in the window launches), `STRATA_P2_SMEM` (config cache to shared).
- **Switch (default)**: `STRATA_P2_FLUSH_NOSWITCH=1`, `STRATA_P2_FLUSH_US=<us>`, `STRATA_P2_SWITCH_STAT=1`, `STRATA_P2_SMEM=1` (off)
- **Origin**: fork (lab/pipeline2 f6a0b6b2; the polling idea from NTRYourWaifu's v7/v8)
- **Upstream**: never proposed
- **Measured**: 13 Oct B-A-B (isolated bench): NOSWITCH removes ~8 100 card switches per request but the gain is below the spread; never promoted (docs/PIPELINE2.md par. 11).
- **Docs**: docs/PIPELINE2.md
- **Files**: `docs/PIPELINE2.md`, `include/strata/core/on_device.hpp`, `src/core/mtp.cpp`, `src/core/verify.cpp`, `src/program/generate.cpp`

Insertion points (15):

- `docs/PIPELINE2.md` — (new file) → [17-1](BLOCKS.md#17-1)
- `include/strata/core/on_device.hpp` · `(file scope)` → [17-2](BLOCKS.md#17-2)
  ```
  
  #include <cuda_runtime.h>
  ```
- `src/core/mtp.cpp` · `void MtpDrafter::chain_cut() {` → [17-3](BLOCKS.md#17-3)
  ```
  
  int MtpDrafter::chain_outputs_ready(std::string& err) {
      if (!chain_live_) return chain_n_;
  ```
- `src/core/mtp.cpp` · `int MtpDrafter::chain_outputs_ready(std::string& err) {` → [17-4](BLOCKS.md#17-4)
  ```
  int MtpDrafter::chain_poll(std::string& err) {
      if (!chain_live_) return 1;
      if (chain_launched_ < chain_n_) return 0;   // STRUCT-A: not whole yet (chain_extend or chain_cut first)
  ```
- `src/core/verify.cpp` · `bool Verifier::init(const WeightTable& wt, const ModelGeometry& g, SessionState&` → [17-5](BLOCKS.md#17-5)
  ```
      }
      release_gpu_fn().store(&release_live_verifiers);
      cudaGetDevice(&device_);   // a layer split's stage on another GPU: its streams, graphs and buffers live there
  ```
- `src/core/verify.cpp` · `namespace {` → [17-6](BLOCKS.md#17-6)
  ```
  double now_ms() { return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count(); }
  }  // namespace
  ```
- `src/core/verify.cpp` · `bool Verifier::pl_launch(int T, const int32_t* tokens, int64_t pos0, std::string` → [17-7](BLOCKS.md#17-7)
  ```
      std::atomic_thread_fence(std::memory_order_seq_cst);
      trace_ev("WINDOW (pipelined)", -1, -1, pos0 * 16 + T);
      ms_host += ms_since(t0);
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la` → [17-8](BLOCKS.md#17-8)
  ```
      // on it (0.1.41, Windows, an RTX 3060 + RTX 5070 Ti split).  The same answer as the loop below gives.
      if (!ar_on() && *(volatile uint32_t*) h_seq_ < (uint32_t) (fl_k_ + 1)) {
          const double now = now_ms();
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la` → [17-9](BLOCKS.md#17-9)
  ```
          const uint32_t want = (uint32_t) (fl_k_ + 1);
          if (*(volatile uint32_t*) h_seq_ < want) {
              const double now = now_ms();
  ```
- `src/core/verify.cpp` · `int Verifier::service(PoolMultiFn pool, void* user, std::string& err, int max_la` → [17-10](BLOCKS.md#17-10)
  ```
  
  bool Verifier::done(std::string& err) {
      if (!fl_active_ || fl_k_ < fl_total_) return false;
  ```
- `src/core/verify.cpp` · `bool Verifier::pl_launch_head(int T, const int32_t* tokens, int64_t pos0, std::s` → [17-11](BLOCKS.md#17-11)
  ```
          err = "verify: saving the PLE history failed";
          return false;
      }
  ```
- `src/core/verify.cpp` · `bool Verifier::pl_head_done(std::string& err) {` → [17-12](BLOCKS.md#17-12)
  ```
      const cudaError_t q = cudaEventQuery(ev_head_);
      if (q == cudaErrorNotReady) {
          const double now = now_ms();
  ```
- `src/core/verify.cpp` · `bool Verifier::pl_launch_tail(std::string& err) {` → [17-13](BLOCKS.md#17-13)
  ```
      fl_cap_ = 0;
      sa_head_ = false;
      fl_prof_ = prof_on_ && G == 1 && prof_pin_ != nullptr;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [17-14](BLOCKS.md#17-14)
  ```
                      return v != nullptr ? std::atoi(v) != 0 : pipe_elastic_env();
                  }();
                  const Clock::time_point pl_t0 = Clock::now();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [17-15](BLOCKS.md#17-15)
  ```
                                       (long long) sa_placed[2], (long long) sa_tails[2], (long long) sa_drops[2],
                                       (long long) sa_placed[3], (long long) sa_tails[3], (long long) sa_drops[3],
                                       (long long) sa_chains, (long long) sa_cut);
  ```

## 18 pw-with-batch

`18-pw-with-batch.patch` — SONNET-STRATA-Porta da upstream #3095 STRATA_PW_WITH_BATCH: la chat singola tiene --pipeline-windows accanto a --batch (opt-in)

- **What**: `--pipeline-windows` stays on for the solo path (one request alone) beside `--batch`, and the drafter stays on the first stage's card (`mtp_first`); the slot drafters of `--batch-mtp` follow it. Batch windows still run without the pipeline.
- **Switch (default)**: `STRATA_PW_WITH_BATCH=1` (off: `--batch` turns the pipeline off, as upstream)
- **Origin**: upstream PR #3095 (18 Oct, +23/-5)
- **Upstream**: PR #3095 open, no maintainer answer yet; not in 0.1.45. Earlier precedent: fork noon-at-cgn d84da294.
- **Measured**: Compiled and run 19 Oct (build 99fa6bd8, 2x solo:S with `--batch 4 --batch-groups 2`, 384 tokens, same reserves in both arms, A-P-A-P): decode 57.3 / 56.5 / 57.5 / 56.1 with the lever against 56.0 / 56.8 / 54.8 / 54.5 without (+2.4% mean), reserve 2300 MiB; the lever needs the larger reserve (the drafter, 837 MiB, and the windows, 234 MiB, move onto the 3060: with 1905 MiB the first request stalled for 240 s with 0 MiB free). It does NOT recover the single chat: the same config with reserve 1905 and no lever gives 60.8 (one run), the Daily without --batch 64.5. Installed no.
- **Docs**: docs/MULTICHAT.md
- **Files**: `src/program/generate.cpp`

Insertion points (6):

- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [18-1](BLOCKS.md#18-1)
  ```
          return o.mtp_draft_vocab.empty() ? std::filesystem::exists(std::filesystem::path(o.mtp) / "draft_vocab.bin", ec)
                                           : std::filesystem::exists(o.mtp_draft_vocab, ec);
      }();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [18-2](BLOCKS.md#18-2)
  ```
          on = false;
  #endif
          const bool helpers_a = o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [18-3](BLOCKS.md#18-3)
  ```
                          : split_devs.size() > 1 && o.pipeline_windows == 1 ? "--pipeline-windows 1 needs exactly two stages"
                          : split_devs.size() > 1 && o.adapt_async ? "not with --adapt-async on three or more stages"
                          : o.mtp.empty() || o.spec < 2 ? "it needs the MTP drafter (--mtp, --spec)"
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [18-4](BLOCKS.md#18-4)
  ```
      // streams share the card and each takes ~2x as long (stage-1 window 11 -> 25 ms, chain 7 -> 21 ms, 2x R9700).  The chain's
      // source rows are copied over once per chain (stage_source_R), the draft head's rows once at load.
      const bool mtp_first = [&] {
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [18-5](BLOCKS.md#18-5)
  ```
              // runs on the first stage, as for the solo drafter (`mtp.set_ple_session(&ss)` above).
              for (int b = 0; b < o.batch; ++b) {
                  auto d = std::make_unique<strata::core::MtpDrafter>();
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [18-6](BLOCKS.md#18-6)
  ```
                      return 1;
                  }
                  slot_mtp_rows.emplace_back(r, free_slot_mtp_rows);
  ```

## 19 pf-fused-native-sm75

`19-pf-fused-native-sm75.patch` — SONNET-STRATA-Porta da upstream #3104 STRATA_PF_FUSED_NATIVE_SM75: esperti fusi int8 IQ del prefill anche su Turing (opt-in)

- **What**: The fused int8 prompt experts for the native i-quant packs (gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; down Q2_0 or IQ4_NL) also on sm_75 (Turing), with the qualified 128-row shape. Pairs not covered stay on MMQ. Changes the bits of the prompt (rounding).
- **Switch (default)**: `STRATA_PF_FUSED=1 STRATA_PF_FUSED_SM75=1 STRATA_PF_FUSED_NATIVE_SM75=1` (off). `STRATA_PF_FUSED=1` also switches on the existing sm_86 native path of the first stage.
- **Origin**: upstream PR #3104 (rebased from #1581; supersedes #3085, the same idea with another gate)
- **Upstream**: PR #3104 open (and #3085); in 0.1.45 `STRATA_PF_FUSED_SM75` only covers the Q2_0 pack, so on our IQ3_XXS pack it did nothing
- **Measured**: Compiled and run 19 Oct (build ecaad907, fixed `--expert-cache 1500`, 3 prompts of ~300 and ~3000 tokens): TTFT median 1.83 s (control 1.77-1.84) at ~300 tokens and 6.55 s (control 5.72-5.89) at ~3000, i.e. +11% slower on the long prompts; first-token logits against the control KL 3e-5..6e-4, inside the control-to-control spread (1e-4..1.3e-3, the bench is not deterministic). The 300-token texts differ from a control as often as two controls differ. No gain on this pack and card pair; kept opt-in. Installed no.
- **Docs**: docs/DETAILS.md
- **Files**: `docs/DETAILS.md`, `include/strata/core/sm75_series.hpp`, `include/strata/prefill/moe_fused_iq.hpp`, `src/prefill/moe_fused_iq.cu`, `tests/cuda/prefill_fused_iq_test.cpp`

Insertion points (13):

- `docs/DETAILS.md` · `argmax run on thread-block clusters (RTX 50, sm_90+; other cards keep the previo` → [19-1](BLOCKS.md#19-1)
  ```
  IQ3 packs about even; on HIP they run on gfx11 and on gfx1200 / gfx1201, see AMD_HIP.md); `STRATA_QSA_CLUSTER=0` / `STRATA_ARGMAX_MULTI=0` turn the decode kernels off. The tables
  below are 0.1.26's.
  ```
- `include/strata/core/sm75_series.hpp` · `(file scope)` → [19-2](BLOCKS.md#19-2)
  ```
  //
  // The numerics-changing Turing options are NOT part of this switch and keep their own explicit opt-ins:
  //   STRATA_SELECT_SIMT=1|2 (FP32 tiled block scorer), STRATA_PA_V2T=1 (v2 prompt attention), STRATA_PF_FUSED_SM75=1
  ```
- `include/strata/prefill/moe_fused_iq.hpp` · `(file scope)` → [19-3](BLOCKS.md#19-3)
  ```
  // include/strata/prefill/moe_fused_iq.hpp - #136: the fused int8 prompt experts (moe_fused.hpp) for the native GGUF
  // packs (tools/iq_pack.py: an expert is its raw GGUF slices [gate rows | up rows | down rows], formats per layer in
  ```
- `include/strata/prefill/moe_fused_iq.hpp` · `struct NativeGeom {` → [19-4](BLOCKS.md#19-4)
  ```
      size_t up_off = 0, down_off = 0;  ///< inside the blob
  };
  ```
- `src/prefill/moe_fused_iq.cu` · `native_f16_kernel(const Batch b, const NativeGeom geo, const Tables tb, const ui` → [19-5](BLOCKS.md#19-5)
  ```
  
  unsigned blocks(int64_t n, int per) { return (unsigned) ((n + per - 1) / per); }
  ```
- `src/prefill/moe_fused_iq.cu` · `template <int T, bool GU, int WW> bool setup_ww(int& occ) {` → [19-6](BLOCKS.md#19-6)
  ```
      return true;
  }
  template <int T, bool GU> bool setup_one(int& occ) { return setup_ww<T, GU, 4>(occ) && setup_ww<T, GU, 2>(occ); }
  ```
- `src/prefill/moe_fused_iq.cu` · `const DevInfo& dev_info() {` → [19-7](BLOCKS.md#19-7)
  ```
      static const bool sm75 = [] { const char* v = std::getenv("STRATA_PF_FUSED_SM75"); return v != nullptr && v[0] == '1'; }();
      if (major < 8 && !(major == 7 && minor == 5 && sm75)) return d;
      int occ = 1 << 20;
  ```
- `src/prefill/moe_fused_iq.cu` · `bool native_supported(int gu_type, int d_type) {` → [19-8](BLOCKS.md#19-8)
  ```
      if (env < 0 && gu_type != T_IQ3_S && gu_type != T_IQ3_XXS) return false;   // default: only these two
      const DevInfo& d = dev_info();
      if (!d.ok || !d_covered_on(d_type, d)) return false;
  ```
- `src/prefill/moe_fused_iq.cu` · `void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n` → [19-9](BLOCKS.md#19-9)
  ```
                      const int32_t* src, void* ha, float* dm, void* stream) {
      if (b.e1 <= b.e0 || n <= 0) return;
      const DevInfo& d = dev_info();
  ```
- `src/prefill/moe_fused_iq.cu` · `void experts_native(const Batch& b, const NativeGeom& g, int n_expert, int64_t n` → [19-10](BLOCKS.md#19-10)
  ```
          return;
      }
  #endif
  ```
- `tests/cuda/prefill_fused_iq_test.cpp` · `int main(int argc, char** argv) {` → [19-11](BLOCKS.md#19-11)
  ```
      try {
  #ifdef _WIN32
          _putenv_s("STRATA_PF_FUSED", "1");
  ```
- `tests/cuda/prefill_fused_iq_test.cpp` · `int main(int argc, char** argv) {` → [19-12](BLOCKS.md#19-12)
  ```
          for (const Pair& p : pairs) {
              if (only ? std::string(p.name).find(only) == std::string::npos : p.gu == GGML_TYPE_Q4_K || p.gu == GGML_TYPE_Q5_K) continue;
              if (!fused::native_supported((int) p.gu, (int) p.d)) {
  ```
- `tests/cuda/prefill_fused_iq_test.cpp` · `int main(int argc, char** argv) {` → [19-13](BLOCKS.md#19-13)
  ```
              // the IQ2_XS pack's layers (34 of 48 IQ2_S / Q2_0, 11 IQ2_XXS / Q2_0), the IQ3_S pack's most common, IQ3_XXS's
              for (const Pair& p : only ? pairs : std::vector<Pair>{pairs[0], pairs[1], pairs[4], pairs[3]}) {
                  if (only && std::string(p.name).find(only) == std::string::npos) continue;
  ```

## 20 pf-diretto

`20-pf-diretto.patch` — SONNET-STRATA-Porta da upstream #2774 STRATA_PF_DIRETTO: chunk corti, esperti residenti con al massimo 8 righe sul kernel di decode (opt-in)

- **What**: Short prompt chunks (below `stream_all_min`): resident experts used by at most `STRATA_PF_DIRETTO_MAX` rows (8) leave MMQ's list and run on the decode's grouped kernel (`native_expert_grouped` for native packs) straight from the compressed blob; also `STRATA_PF_ROUTE_PRIOR=lambda` (router push toward resident experts, off). Changes the bits of short prompts.
- **Switch (default)**: `STRATA_PF_DIRETTO=1`, `STRATA_PF_DIRETTO_MAX=<rows>` (off here; the PR has it on), `STRATA_PF_ROUTE_PRIOR=<lambda>` (off)
- **Origin**: upstream PR #2774 (apatassini); default turned to opt-in in this fork
- **Upstream**: PR #2774 open; not in 0.1.45
- **Measured**: Compiled and run 19 Oct (build ecaad907): needed a fix (the PR's scratch was one static, allocated on the first stage's card and read by the second: stall at layer 16, then an illegal memory access); with one scratch per device it runs. TTFT median 1.81 s at ~300 tokens (control 1.77-1.84) and 5.81 s at ~3000 (control 5.72-5.89): no gain. Logits: bitwise equal to the control on the prompts above stream_all_min, KL 8e-5..1e-3 on the three ~300-token ones (the control-to-control spread is of the same size). Installed no.
- **Notes**: Follow-up on lab/multichat45b: ecaad907 (one scratch per device); folded into this patch. The PR has the path on by default; here STRATA_PF_DIRETTO=1 is needed.
- **Docs**: docs/MULTICHAT.md
- **Files**: `src/prefill/prefill.cpp`

Insertion points (7):

- `src/prefill/prefill.cpp` · `(file scope)` → [20-1](BLOCKS.md#20-1)
  ```
  #include "strata/kernels/qsa_prompt_attn.hpp"
  #include "strata/kernels/qsa_select.hpp"
  #include "strata/prefill/gemm.hpp"
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [20-2](BLOCKS.md#20-2)
  ```
                      pt.mark(kPfRouter, cs);
                      if (!bf16_proj(m.gemm, wr, m.mixed_bf, m.logits, T, v.name("ffn_gate_inp.weight"), err, 0, m.mixed_bf_lo)) return false;
                      route(m.logits, m.ids, m.w, T, m.g->n_expert, m.cs);
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [20-3](BLOCKS.md#20-3)
  ```
                                  ++n_cpu;
                              }
                          }
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [20-4](BLOCKS.md#20-4)
  ```
                          {
                              int32_t r = 0;
                              for (int32_t e = 0; e < m.g->n_expert; ++e)
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [20-5](BLOCKS.md#20-5)
  ```
                                  if (on_cpu[(size_t) e]) { m.off[(size_t) e] = r; r += m.cnt[(size_t) e]; }
                              m.off[(size_t) m.g->n_expert] = r;
                              rows_local -= rows_cpu;
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [20-6](BLOCKS.md#20-6)
  ```
                          std::vector<int32_t> order, order_peer;
                          for (int32_t e = 0; e < m.g->n_expert; ++e)
                              if (m.cnt[(size_t) e] > 0 && (on_peer.empty() || on_peer[(size_t) e] != 2) &&
  ```
- `src/prefill/prefill.cpp` · `bool Prefill::run_impl(const int64_t* tokens, int64_t n, int64_t pos0, std::stri` → [20-7](BLOCKS.md#20-7)
  ```
                              }
                          }
                          n_order = order.size();
  ```

## 21 pack-concurrent

`21-pack-concurrent.patch` — FABLE-PASSATA-Pacco di lettura sul thread concorrente (STRATA_PACK_CONCURRENT, opt-in)

- **What**: The packed prefill (STRATA_PACK_PREFILL: the prompts sent ahead by waiting requests read with this one in the same chunks, the experts streamed once for all of them) also beside STRATA_PIPED_CONCURRENT: the prefixes go into the pack slots on the host thread with nothing in flight, Prefill::run_packed runs on the read's thread while the slot groups keep decoding, the pack slots' checkpoints are taken after the drain. Without it the concurrent read returned before pk_read and never packed.
- **Switch (default)**: `STRATA_PACK_CONCURRENT=1` (off), with `STRATA_PACK_PREFILL=N` and `STRATA_PIPED_CONCURRENT=1`; a packed read takes no BYIELD
- **Origin**: fork (lab/passata-unica, docs/porto/strata_adattivo/PASSATA_UNICA_2026-10-19.md in moe-aggressive-commit)
- **Upstream**: never proposed; 0.2.0-beta (upstream/main f57c0adc) still reads admissions one at a time and has no multi-sequence prefill
- **Measured**: Compiled and run 19 Oct: see docs/MULTICHAT.md (passata unica) - to be filled from the letture_concorrenti bench (4 readers of 2-4k on 40-100k cached + 2 writers, B-A-B) and the replay.
- **Depends on**: STRATA_PACK_PREFILL / Prefill::run_packed (lab/multichat43 -> lab/multichat45, not an entry of this stack: it is carried on the multi-chat branch) and STRATA_PIPED_CONCURRENT (#2326, in 0.1.45)
- **Notes**: pk_read is now pk_prepare + Prefill::run_packed + pk_finish; the path without the switch is the three in a row (behaviour unchanged).
- **Docs**: docs/MULTICHAT.md
- **Files**: `src/program/generate.cpp`

Insertion points (10):

- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-1](BLOCKS.md#21-1)
  ```
              // STRATA_PACK_PREFILL (docs/MULTICHAT.md): the prompts of waiting requests (BHINT) are read with this one, in the
              // same chunks, each into a pack slot from its longest cached prefix (a parked conversation, or a checkpoint of
              // this prompt below a0) up to its last turn boundary, where it gets the checkpoint its own read would take.
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-2](BLOCKS.md#21-2)
  ```
                      return v != nullptr ? std::max<int64_t>(1, std::atoll(v)) : (int64_t) 128;
                  }();
                  auto same = [](int32_t x, int64_t y) { return (int64_t) x == y; };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-3](BLOCKS.md#21-3)
  ```
                      plans.push_back(std::move(pl));
                  }
                  if (plans.empty()) return true;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-4](BLOCKS.md#21-4)
  ```
                          pl.L = 0;
                      }
                  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-5](BLOCKS.md#21-5)
  ```
                      sg.prev[1] = (int32_t) pl.h[(size_t) pl.m - 1];
                      for (size_t k = 0; k < pack_ss.size(); ++k) sg.ss.push_back(pack_ss[k][pl.slot].get());
                      segs.push_back(sg);
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-6](BLOCKS.md#21-6)
  ```
                      for (int64_t t = 0; t < T; ++t) q.nxt.push_back((int32_t) pl.h[(size_t) (p0 + t + 1)]);
                      return true;
                  };
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-7](BLOCKS.md#21-7)
  ```
                      q.used = ++pk_clock;
                      ++kept;
                  }
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-8](BLOCKS.md#21-8)
  ```
                      return true;
                  }();
                  if (piped && pr_conc && req_imgs.empty()) {   // (with no slot decoding it only watches for a BYIELD)
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-9](BLOCKS.md#21-9)
  ```
                          bool read_ok = true;
                          std::string read_err;
                          in_prompt_read = true;
  ```
- `src/program/generate.cpp` · `int main(int argc, char** argv) {` → [21-10](BLOCKS.md#21-10)
  ```
                          reader.join();   // (a failed window leaves the read to finish: it is not stopped mid-run)
                          sp.stop_at_chunk = nullptr;
                          in_prompt_read = false;
  ```
