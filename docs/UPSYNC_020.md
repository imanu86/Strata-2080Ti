# UPSYNC_020 - la pila del fork sopra upstream 0.2.0-beta (19/10/2026)

Branch `lab/upsync-020` del fork `imanu86/Strata-2080Ti`, da `upstream/main` f57c0adc (= tag `v0.2.0-beta` + 9 commit di sola
infrastruttura: CI, bot del tracker, README, bundle del Monitor). Fra `v0.1.45` e questo punto upstream ha 786 commit (1283 file,
+220k righe: quasi tutto app, Terminale, packaging; il motore e' toccato in pochi file). La versione del motore resta
"0.1.45.3" (`CMakeLists.txt`). Il modello e' `docs/UPSYNC_MAIN.md` (giro precedente, 0.1.45).

**Stato (19/10 19:35, fermato dall'owner per i token): patch portate si (18 di 18, committate e pushate), compilato no, installato no, provato no, misurato no.** La build non e' partita: la coda dei lock (`build.lock`/`bench.lock`) era tenuta dagli agenti batch-dyn e banco-bisezione e la mia attesa non ha mai ottenuto il turno. Fatti: punti 1, 2 e 5 del compito; restano 3 (build) e 4 (banco). Nessun lock tenuto, nessun processo mio in vita.

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

Vedi il paragrafo 6.

## 4. Build (DA FARE) e prossimi passi per l'agente locale

Stato git: `lab/upsync-020` = upstream/main f57c0adc + 18 commit di patch (uno per leva, `SONNET-UPSYNC-Porta su 0.2.0 ...`) + commit di `stack/`
(`python stack/check_anchors.py --verify-docs` = ok) + questo doc. Pushato su `origin/lab/upsync-020`. Worktree: `C:\Users\imanu\source\repos\Strata-wt\upsync-020`.
Conflitti aperti: nessuno (la 01 e' risolta). Se la build trova errori di compilazione, correggerli con un `git commit --fixup=<sha della patch>` e poi
`GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash upstream/main` (branch non ancora usato da altri), cosi' che `--build` ricrei 18 patch; poi
`python stack/check_anchors.py --build . --range upstream/main..HEAD` e `--verify-docs`. Rischi da guardare alla compilazione: i blocchi 3-way di 05 e 12
(`expert_source.cpp/.hpp`, 732 righe cambiate in upstream), 13 (`verify.cpp`), 17 (`mtp.cpp`, `verify.cpp`), 18 (kernel `gr_*` toccati da upstream).

1. **Build** (sm_75+86, 4 job, priorita' bassa; non usa la GPU). `build_lock.py` originale
   (`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261006_due_schede\build_lock.py`) aspetta che nessun `strata.exe` giri; il Daily
   dell'owner e' acceso, quindi usare una copia senza quel controllo (togliere `and not engine_running()` dalle due condizioni dell'attesa; come in
   UPSYNC_MAIN par. 4). Comando: `python -u build_lock_noeng.py C:\Users\imanu\source\repos\Strata-wt\upsync-020 --jobs 4` (aspetta `bench.lock`
   libero e prende `build.lock`; cartella di build `upsync-020\build-sm7586`). Poi copiare in
   `...\corse_2080ti\20261004_routed_keep\build-upsync020-<sha>\`: `strata.exe`, `strata-vision.exe` (da `D:\ds4_work\strata\Strata\build-0143-daily`;
   `tools/vision` cambia di 77 righe fra 0.1.45 e 0.2.0: per un confronto esatto ricompilarlo), `source\serve`, `source\tools` (come
   `build-upsmain-d9a26384`), piu' `build.json` con lo sha256 dell'exe.
2. **Config** (copie, mai modificare i JSON del Daily), in una cartella tua (es. `20261019_upsync020`), da
   `D:\ds4_work\strata\Strata\strata-0143-daily.json` (gia' senza `--batch`):
   - `exe` e `vision.exe` -> la nuova build; env tolte le inerti `STRATA_NODECUT`, `STRATA_TOPK_GRID`, `STRATA_B6_HEAD_MINB`;
   - **braccio A** "0.2.0+pila": env del Daily + `STRATA_HC16_SM75=1` (e' l'hc16 del Daily; su 0.2.0 e' opt-in);
   - **braccio C** "0.2.0+pila + sm_86 a mano": A + `STRATA_ADAPT_NOWAIT=1`, `STRATA_ADAPT_LAG=2`, `STRATA_CPU_IQ_ALL_EXACT1=1` (mai con
     `STRATA_IQ_MT_MIN=1`), `STRATA_GR_UP75=1`, `STRATA_GU_SWIGLU=1` (SIDE_MIXER=2, ATTN_DAILY, QSA_ROWS, B6_MMVQ_ROWS sono gia' nel Daily). Dichiarato
     +10,2% decode su 3060 da sola, bitwise; sulla 2080 Ti non misurato: verificare anche che i token restino identici ad A.
   - eventuale braccio con `STRATA_BF16_TC=1` al posto di HC16 (prompt a 131k: UPSYNC_MAIN par. 8).
   - Non rieseguire il setup automatico (puo' riscrivere la config); lanciare il runner con `STRATA_NO_PRESET=1` nell'ambiente.
3. **Banco** (GPU solo sotto `bench.lock`, mai priorita' alta, niente corsa se esiste `lab_locks\pause`; annotare se Jump e' connesso: il runner scrive
   `ambiente.txt`): `python ...\20261019_banco_comunita\community_bench.py <tag> <config.json assoluto> <build dir assoluta>`; ordine B-A-B
   (Daily 0.1.43 r2 con `D:\ds4_work\strata\Strata\strata-0143-daily.json` e build `D:\ds4_work\strata\Strata\build-0143-daily`, poi A, poi
   Daily), poi C. Riferimento: `RISULTATI.md` della stessa cartella (Daily 0.1.43 r2, mediane prompt/decode: 4K 701/75,9; 32K 1439/70,4;
   128K 1720/75,3; 0.1.45 senza batch 663/67,0; 1282/66,4; 1519/69,2). Ogni corsa dura ~7 minuti.
4. **Qualita'**: confrontare i `tokens-*-raw.json` fra Daily e A/C (stesso prompt, greedy, 256 token: dove il testo diverge e se resta coerente) e le righe
   `drafts accepted a of b` dell'`engine.log` (accettazione bozze). Per C anche l'identita' dei token con A. Nel log cercare `prefill plan: chunk N, ring M kept`
   (planner) e la riga di `pin_budget` all'avvio.
5. Se la build fallisce o il banco mostra un calo: bisezione per patch sulla pila (`stack/NN-*.patch`).

## 5. Misure

Nessuna (vedi il paragrafo 4).

## 6. Le nostre PR upstream dopo il reset del 19/10 (lista per l'owner; niente riaperto ne' commentato)

Condizioni di riapertura (commento di Niko del 19/10): rebase su `main` e verifiche rifatte su 0.2.0-beta; titolo con prefisso `[Engine]`, `[GUI]`,
`[TUI]`, `[Benchmark]`, `[Docs]` o `[Other]`; template compilato (Problem and result, Related discussion, Approach, Validation Passed/Failed/Not run,
Limits); commento `/reopen`. Le prestazioni vanno misurate prima/dopo sulla stessa macchina; un cambio di default richiede accordo sullo scopo. Tutte le
nostre PR aperte hanno il titolo senza prefisso e risultano in conflitto.

| PR | stato | in 0.2.0? | raccomandazione |
|---|---|---|---|
| #1419 banco forzato | chiusa | no | lasciare chiusa (strumento da banco, resta nel fork) |
| #1417 stima prestito | chiusa | no | lasciare chiusa (ha senso solo con i buffer Turing) |
| #1117 `--elastic` | chiusa | no (c'e' solo `--vram-elastic`) | lasciare chiusa; se si vuole, prima un issue per lo scopo, poi un PR piu' piccolo (solo arena VMM) |
| #2058 EXPDEC | aperta | si, opt-in | lasciare; valore residuo: A/B su 2080 Ti in 0.2.0 per chiedere il default |
| #2059 QSA_ROWS | aperta | si, opt-in | lasciare, come #2058 |
| #2073 PLE diretta + diagnostica | aperta | meta' (il default diretto si) | **aggiornare**: rebase tenendo solo la diagnostica (`ms_ple_gather`, `read_us_max`/`slow_reads`, avviso sul chunk fermo = patch 10), titolo `[Engine] ...`, template |
| #2075 ATTN_DAILY | aperta | si, opt-in | lasciare; seguire la risposta di Niko (confronto `=0`/`=1` sul binario 0.2.0 offerto da lui) |
| #2077 `--prompt-cache-file` | aperta | no | lasciare ferma finche' Niko non sceglie il design su disco (#1090); poi rebase con guard Metal (build Metal rotta, segnalata da Qiuzan2001) |
| #2080 SELECT_SIMT default | aperta | no (opt-in `=1` o `=2`) | ritirare, salvo produrre KL sm_75 indipendente, needle check e nota Volta/Pascal |
| #2081 dequant 16 B sm_75 | aperta | si (`STRATA_TC_SM75`, `SM75_PERF`) | **aggiornare**: sostituire con un commento di evidenza, A/B isolato su 2080 Ti in 0.2.0 (solo `STRATA_TC_SM75` 1 contro 0, almeno 10 coppie) |
| #1418 MMVQ righe riusate | chiusa | si, in main con credito | niente |

Altre PR nostre chiuse prima del reset (non riesaminate): #1110, #1063 (fix MTP `--batch-mtp`), #1030 (poi #1117), #614, #512.
Priorita' suggerita: un solo set di A/B sulla 2080 Ti su 0.2.0 con un binario e coppie interleaved per `STRATA_TC_SM75`, EXPDEC, QSA_ROWS, ATTN_DAILY,
da postare sui thread esistenti (Niko non ha una Turing: e' l'unica cosa che puo' spostare un default); poi #2073 solo-diagnostica; poi #2080.
