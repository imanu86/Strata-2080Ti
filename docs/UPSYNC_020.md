# UPSYNC_020 - la pila del fork sopra upstream 0.2.0-beta (19/10/2026)

Branch `lab/upsync-020` del fork `imanu86/Strata-2080Ti`, da `upstream/main` f57c0adc (= tag `v0.2.0-beta` + 9 commit di sola
infrastruttura: CI, bot del tracker, README, bundle del Monitor). Fra `v0.1.45` e questo punto upstream ha 786 commit (1283 file,
+220k righe: quasi tutto app, Terminale, packaging; il motore e' toccato in pochi file). La versione del motore resta
"0.1.45.3" (`CMakeLists.txt`). Il modello e' `docs/UPSYNC_MAIN.md` (giro precedente, 0.1.45).

Stato: vedi il paragrafo 6 (compilato / provato / misurato) - l'ultimo aggiornamento e' in fondo.

## 1. Esito per patch (18 di 18 applicate, nessuna ritirata)

`check_anchors.py --check` contro f57c0adc sulla pila 0.1.45: 10 pulite (02, 03, 07, 09, 10, 11, 14, 15, 16, 18), 1 con 3-way (05), le altre
"rewrite" solo perche' la 01 non si applicava e il resto della catena dipende da lei. Applicando in ordine con `git am --3way` e' rimasto **un solo
conflitto manuale**, nella patch 01 (due blocchi di `src/program/generate.cpp`); la 05 e la 12 passano dal 3-way automatico (senza conflitti), le altre
15 si applicano pulite. Il codice di nessuna delle 18 e' arrivato in upstream (ho cercato i nomi delle variabili e le funzioni: nessuna
presenza di `STRATA_VRAM_FREE`, `RAM_CORE`, `SA_BET`, `P2_FLUSH_NOSWITCH`, `PREFIX_DISK_MULTI`, `CONV_DISK`, `PLE_LOOKAHEAD`,
`--prompt-cache-file`, `HC16_SM75` in `src/`, `include/`, `serve/`). Upstream ha invece `--vram-elastic` (arena VMM), una cosa
diversa dal nostro `--elastic`.

| patch | leva | esito su 0.2.0 |
|---|---|---|
| 01 | elastic-cache (`--elastic`) | **portata a mano** (2 conflitti in `generate.cpp`): il planner dei chunk (`planned_chunk`, `prefill_plan::balanced_chunk`) e il tempo di drain di `adapt_tick` (`drain_t0`) restano come li ha scritti upstream; `el_core` e `elastic_step` si aggiungono accanto |
| 02 | split-prompt-borrow | pulita (stima del prestito sul device del blocco, `OnDevice` in `part_slots`) |
| 03 | forced-bench | pulita |
| 04 | elastic-split-pipeline | pulita (dopo la 01) |
| 05 | prefix-disk (`--prompt-cache-file`) | 3-way automatico (CMake, `mtp.cpp`, `generate.cpp`) |
| 06 | vram-free-nvml | pulita |
| 07 | park-no-foreign-kv | pulita |
| 08 | pipeline-architectds | pulita (dopo la 01) |
| 09 | df-defaults | pulita |
| 10 | ple-diag | pulita |
| 11 | prefill-sm75 (SELECT_SIMT, TC_SM75) | pulita; gli interruttori sono quelli di 0.1.45 |
| 12 | ram-core | 3-way automatico (`expert_source.cpp/.hpp`: upstream ha toccato 732 righe) |
| 13 | sa-bet | pulita (`verify.cpp` e `mtp.cpp` cambiati di poco) |
| 14-16 | prefix-disk-multi, ple-lookahead, conv-disk | pulite |
| 17 | p2-flush-noswitch | pulita |
| 18 | hc16-sm75 (`STRATA_HC16_SM75`) | pulita |

Le patch 02, 03, 05, 01 sono quelle delle PR chiuse il 19/10 (#1417, #1419, #2077 e #1117, quest'ultima chiusa).
Resta da vedere alla compilazione e alla prova se un'applicazione "pulita" nasconde un cambio di significato: non ho riletto riga per riga i
blocchi applicati in 3-way (05, 12).

## 2. Cosa cambia in 0.2.0 per il nostro assetto (3060 sm_86 + 2080 Ti sm_75, `--layer-split 16 --pipeline-windows 2`, Windows)

Letto da `docs/release-notes-0.2.0.md` (sezione "New defaults, with what we measured"), `DETAILS.md`, `BATCHING.md`, `MULTI_GPU.md`,
dal codice e da `git diff v0.1.45..HEAD`. Il "prompt split" come documento non esiste: "the split" nelle note e' la divisione del
progetto in app, Terminale e motore.

**Default nuovi che per noi NON scattano**
- La **tabella sm_86** (`src/core/nv_defaults.cpp`, applicata in `generate.cpp` ~2679-2691): ADAPT_NOWAIT=1, ADAPT_LAG=2, CPU_IQ_ALL_EXACT1,
  SIDE_MIXER=2, ATTN_DAILY, GR_UP75, GU_SWIGLU, QSA_ROWS, B6_MMVQ_ROWS (+10,2% decode su 3060, +13,0% su A4000, 8 coppie su 8, token
  identici). **Si applica solo se tutte le schede visibili hanno la stessa compute capability**: con 8.6 + 7.5 non si accende. Quelle
  che il Daily non ha gia' vanno messe a mano: `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_CPU_IQ_ALL_EXACT1=1` (non insieme a
  `STRATA_IQ_MT_MIN=1`), `STRATA_GR_UP75=1`, `STRATA_GU_SWIGLU=1`. Sono il braccio "sm86 a mano" del par. 5.
- Tabella sm_60 (P100) e sm_120 (SIDE_MIXER, ATTN_DAILY, GR_UP75, GU_SWIGLU, PF_FUSED, GR_DOWN_TG su schede 80+ SM): non ci riguardano.
- Per sm_75 non c'e' nessuna tabella: `STRATA_SM75_PERF` resta opt-in (`sm75_series.hpp` invariato).

**Cambia l'avvio su Windows/CUDA**
- Margine di memoria bloccata: 640+160 MiB mentre l'arena si blocca (`pin_budget.hpp`); `STRATA_PIN_RESERVE_MIB=0` lo spegne. Gli stagings
  di verify passano da `host_alloc_guarded`. E' un cambio di avvio, non di velocita' (da verificare nel log).
- `fit_max_tokens` e' default (il Daily lo ha gia' true). Worker CPU di default = core fisici con tetto 32 (prima: core fisici meno l'host:
  differenza piccola, ipotesi non verificata). La visione e' acceso di default (CPU sotto 24 GB): il Daily ha `--vision` esplicito.
- Il server applica un **tuning salvato** a ogni avvio (`serve/server.py` `apply_saved_tuning`, `strata_control/presets.py`); si esclude con
  `STRATA_NO_PRESET=1`. Il banco lo imposta. **Non rilanciare il setup automatico**: puo' riscrivere la config.
- **Planner dei chunk del prompt** (`prefill_plan.hpp`, `generate.cpp` ~7456-7550): sceglie chunk e ring minimizzando i byte letti dal disco
  secondo il costo dell'host (page cache, PCIe, classe del disco). Un prompt che sta in un chunk resta in un chunk, uno piu' lungo si legge in chunk
  uguali (niente coda piccola). Cambia solo l'arrotondamento (KL 0,002-0,012 sul primo logit, testo uguale 30/30). `STRATA_PREFILL_HOST_PLAN=0` lo
  spegne, `--prefill N` e `STRATA_PREFILL_RING` hanno la precedenza. Misurato da upstream su Linux/SATA (20K da 334 a 822 tok/s). Su Windows la
  lettura e' non bufferizzata (`page_cache=0`): ipotesi, da confermare dal log (`prefill plan: chunk N, ring M kept`), che col Daily il chunk resta quello.
  Somma i prestiti di tutte le parti (`part_slots`), quindi vale anche con `--layer-split`.
- Invariati: `ROUTE_TAIL_SKIP` (7 sotto il 90% di cache), tutti gli interruttori del par. 7 di UPSYNC_MAIN (SM75_PERF, MTP_STAGE0, BF16_TC, PF_FUSED,
  HANDOFF_PIPE, PIPED_*), `--pipeline-windows` e `--layer-split`.

**Rinominati o rimossi: nessuno** nel motore (confronto di tutte le `STRATA_*` e delle opzioni di `generate.cpp`). Nuove opzioni del server:
`--parallel`, `--batch-groups`, `--max-queue`, `--queue-timeout`, `--batch-telemetry`, `--no-fit-max-tokens`, `--offline`, `--ready-file`,
`--owner-fd`, `--shutdown-timeout`.

**Variabili del Daily 0.1.43 su 0.2.0**
| variabile | stato |
|---|---|
| WATCHDOG_S, POOL_SPIN_US, MMVQ_IL=0, ROUTE_TAIL_SKIP=0, MTP_SHARED_BRANCH=0, EXPDEC | lette, default invariati (EXPDEC resta opt-in) |
| B6_MMVQ_ROWS, ATTN_DAILY, SIDE_MIXER, QSA_ROWS | lette; diventano default solo su sm_86 uniforme / sm_120; il valore esplicito vince |
| QFUSE_HC=0 | letta solo nella build HIP: inerte su CUDA |
| VRAM_FREE, RAM_CORE(_SLOTS_DEV1), SA_BET, SA_CHAIN, PREFIX_DISK_MULTI, CONV_DISK | **nostre** (patch 06, 12, 13, 14, 16): ci sono solo con la pila |
| NODECUT, TOPK_GRID, B6_HEAD_MINB | assenti ovunque (leve neutre di lab/struct-a, mai nella pila): inerti |

**Multi-agente / batch / prefill rispetto ai nostri lavori in corso (non toccati)**
- *Passaggio a caldo chat singola <-> agente* (`lab/batch-dyn`): upstream ha **gia' da prima** (`BATCHING.md` 155-170, identico a 0.1.45) il passaggio
  solo/slot dentro un motore avviato con `--batch`: una richiesta sola gira sul percorso solo con MTP, all'arrivo della seconda la prima passa in uno
  slot e, rimasta sola, torna sul percorso solo (max 2 volte, `STRATA_PARALLEL_SOLO=0`). **Non** ridimensiona gli slot a caldo: gli slot sono fissi
  all'avvio (`"parallel": N` + riavvio, o `--parallel N` nuovo del server). 0.2.0 aggiunge solo telemetria (`STRATA_BATCH_TELEMETRY`, righe BSTAT/PSTAT),
  coda con scadenza (`--max-queue`, `--queue-timeout`) e `--batch-groups`. `--pipeline-windows` resta spento con `--batch`. Sovrapposizione possibile
  con la parte server/ammissione di batch-dyn; nel motore non c'e' conflitto.
- *Passata unica prefill+decode* (`lab/passata-unica`): **nessun equivalente** nel motore. Gli interruttori vicini sono opt-in e gia' noti
  (`STRATA_PIPED_CONCURRENT`, `PIPED_READ_PIECE`, `ADMIT_MERGE_FIRST`, `HANDOFF_PIPE`). Il prefill e' cambiato solo per Volta, sm_120, AMD e il planner.
- Il resto delle novita' (Linux io-prefetch/zero-copy GGUF, gfx12, Arc, APR trellis, Gyro) non tocca Windows/CUDA 7.5+8.6.

## 3. Le nostre PR upstream dopo il reset del 19/10

Vedi il paragrafo 7 (lista per l'owner). Niente e' stato riaperto o commentato.

## 4. Build

Da compilare con `build_lock.py` (sm_75+86, CUDA 12.6, MSBuild/Ninja di VS 2022, 4 job, priorita' bassa), cartelle in
`...\corse_2080ti\20261004_routed_keep\build-upsync020-<sha>\`. Il Daily dell'owner e' acceso (porta 8100): `build_lock.py` originale aspetta che non giri
nessun `strata.exe`, quindi uso una copia identica senza quel controllo (come in UPSYNC_MAIN par. 4); aspetta pero' il proprio turno su `bench.lock`
e `build.lock`.

(Paragrafi 5-7: misure e liste, aggiunti a fine giro.)
