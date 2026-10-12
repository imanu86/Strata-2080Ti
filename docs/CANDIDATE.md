# CANDIDATE: candidato Daily (`lab/candidate`)

Branch `lab/candidate` del fork `imanu86/Strata-2080Ti`, da `origin/lab/ports1` (`dc3bcf08`) più il nucleo RAM di
`lab/core-ram`. Serve a provare sul banco, come un unico motore, tutto quello che il lab ha portato finora, con la config
del Daily.

## Cosa contiene

| Pezzo | Origine | Dove si legge |
|---|---|---|
| Strata v0.1.42 di upstream + le patch del Daily | `lab/upsync` | `docs/UPSYNC.md` |
| critpath2 (`STRATA_CP_*`, tutti opt-in) | `lab/critpath2` | `docs/CRITPATH.md` |
| chainfuse del drafter (`STRATA_CHAIN_FUSE2`, opt-in) | `lab/chainfuse` | `docs/CHAINFUSE.md` |
| tabella MMVQ_IL sm_75 (#1437, opt-in `STRATA_MMVQ_IL=1`) e `STRATA_SIDE_MIXER=1/2` (#1889) | `lab/ports1` | `docs/UPSYNC.md`, `docs/OLDER_GPUS.md` |
| PLE in lettura diretta anche con la sonda lenta (`STRATA_PLE_AUTO_MMAP=1` per mmap) e diagnostica PLE del decode | `lab/ports1` (`aaf2e069`, `b83454c8`, `743b2d07`) | `docs/UPSYNC.md` |
| prefill del Daily su sm_75: `STRATA_SELECT_SIMT` di default sotto sm_80, dequant con store da 16 B, hc16 sui tensor core FP16 (`STRATA_TC_SM75=0` lo spegne) | `lab/ports1` (`83424a3b`, `7bdce0f2`, `794268e4`) | `docs/UPSYNC.md` |
| nucleo fisso in VRAM senza copia RAM (`STRATA_RAM_CORE=1`) | `lab/core-ram` `496b1c90`, `e1755272`, `deb72eb3` → qui `da00a0ce`, `2273e675`, `5dea5258` | `docs/RAM_CORE.md` |

I tre commit del nucleo sono entrati senza conflitti.

## Nucleo RAM e modifiche di ports1

Controllato che l'esclusione del nucleo da scambi, prestiti e shrink resti coerente:

- **critpath2** (`STRATA_CP_RESPLAN`): il device pianifica i gruppi dalle righe di residenza (`cp_sync_res`) e
  `drive_note_split` conta le rotte come hit in VRAM. Le vittime le sceglie sempre `adapt()`/`a_pick`, che saltano il
  nucleo (`rc_fixed`); la barriera `cp_fence_*` aspetta soltanto. Gli esperti del nucleo restano nel loro slot, le righe
  non cambiano: niente da adattare. `L1FIRST`, `S1FIRST`, `CHAIN_PRE`, `CHAIN_FUSE` cambiano l'ordine dei lanci, non la
  residenza.
- **SIDE_MIXER**: rami del grafo della finestra (gate GDN, indicizzatore QSA). Non tocca esperti né arena.
- **PLE diretta**: legge il file PLE con il suo lettore e la sua row cache, non l'arena degli esperti. hc16/dequant sm_75
  ripescano pesi densi BF16 (`WeightRef`), non i blob degli esperti, quindi non toccano pagine restituite.
- **riserva sm_75 del prompt** (`sm75_prompt_extra_bytes`): toglie qualche slot alla cache dove il prompt possiede i suoi
  buffer. Il nucleo resta comunque limitato a `slots - 128` e al primo slot prestato; se 5500 sulla 2080 Ti superasse la
  parte non prestata, il prestito si riduce (chunk più piccolo), non il nucleo. Da guardare nel log: `strata ram core:
  dev1: nucleo fisso slot [0, 5500) di S` e il chunk della prefill.

## Config del candidato Daily

Da aggiungere all'ambiente della config del Daily (le altre opzioni restano quelle del Daily):

```
STRATA_VRAM_FREE=device
STRATA_ROUTE_TAIL_SKIP=0
STRATA_ATTN_DAILY=1
STRATA_MMVQ_IL=0
STRATA_B6_MMVQ_ROWS=1
STRATA_MTP_SHARED_BRANCH=0
STRATA_SIDE_MIXER=2
STRATA_RAM_CORE=1
STRATA_RAM_CORE_SLOTS_DEV1=5500
```

- `STRATA_VRAM_FREE=device`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_MMVQ_IL=0`, `STRATA_B6_MMVQ_ROWS=1`: come nel Daily
  (memoria libera letta dal device, niente salto della coda delle rotte, MMVQ B6 a righe del Daily invece della tabella
  sm_75).
- `STRATA_ATTN_DAILY=1`: l'attenzione QSA del decode del Daily (chunk pre75 + merge v2).
- `STRATA_MTP_SHARED_BRANCH=0`: gli esperti condivisi del drafter sullo stesso stream, come nel Daily.
- `STRATA_SIDE_MIXER=2`: il lavoro del mixer che legge solo l'input su due rami. Vince su `STRATA_DF_BRANCH`, che con
  questa variabile si spegne da solo (riga `strata: STRATA_SIDE_MIXER=2 is set: STRATA_DF_BRANCH is off`).
- `STRATA_RAM_CORE=1` + `STRATA_RAM_CORE_SLOTS_DEV1=5500`: nucleo fisso di 5500 slot (~9,6 GiB) sulla 2080 Ti, sulla
  3060 il default (core elastico non prestato). Stima: ~10 GiB di RAM in meno (`docs/RAM_CORE.md`).

`STRATA_CP_*` e `STRATA_CHAIN_FUSE2` restano spenti: si accendono uno alla volta nei B-A-B.

## Prima verifica sul banco

1. Avvio: `strata ram core:` senza `NOT armed`, poi `riletti 0`, `fuori posto 0` a ogni `fine richiesta`.
2. Velocità contro il Daily, stesse richieste: decode a 40-71k e prefill L01 a 131k.
3. RAM: privata e working set di `strata.exe` dopo il riscaldamento, contro il Daily.
4. Se il decode cala: hit rate della cache e `scambi dell'adattamento` (il nucleo blocca la rotazione di 5500 slot).

## Stato

Build di `5dea5258` (il codice; questo file aggiunge solo documentazione): compilata con `build_lock.py` (Release,
CUDA 75;86, test spenti), rc 0. `strata.exe` sha256
`9ac34976a559a656cee90a9b2d889d0405fdf1b2dbd10c4026a1b848602a0f30`, copiata con `build.json` in
`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\build-candidate-5dea5258\`; accanto
`source\serve` e `source\tools` del worktree (server.py, tokenizer) e `strata-vision.exe` di `build-upsync-133e3a69`
(sha256 `93441f40…`). Installato no, provato no: nessuna GPU usata, nessun motore avviato.
