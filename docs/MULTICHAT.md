# Multi-chat su upstream 0.1.45 (lab/multichat45) - stato 18/10

Branch `lab/multichat45` da `lab/upsync-main` 09cba781 (upstream v0.1.45 + pila `stack/`). **Compilato si (sm_75+86), build
`build-multichat45-e42d8af8` (sha256 5dc49fe8...), installato no, provato solo con `mc2.py` (sotto).** Le sezioni dopo questa
(diagnosi del 13/10 su base 0.1.43) restano valide come storia; il codice del prefill impacchettato e' quello di
`lab/multichat43` portato qui.

## Stato della coda notturna (19/10 00:05)

Dalle 22:14 il `bench.lock` e' tenuto da una serie di `run_long.py` di un altro agente (`DET-x24-pf-*`, arm da 3000 s uno dopo l'altro,
lo stesso file di lock): la mia coda resta in attesa (`o-conc`, poi `coda_mc45r.sh notte`: r0-warm, r1-ctrlA, r2-cfg1, r3-conc, r4-kv16, r5-piece,
r6-help, r7-ctrlM, r8-pack3, r9-pack3+conc, r10-bpipe, r11/r12-pmtp2 vs b2, r13-bf16tc, r14-sm75perf, r16-ctrlB). Parte da sola quando il lock si libera e
ogni arm rigenera la tabella qui sopra (`publish.py`, commit e push). Lo scenario `agents:3:R` (banco agenti reale: prefisso 32k + storia 0-70k,
tool result 1-5k token, risposta 100-400, pausa 1-3 s, 6 turni; prefill share, TTFT p50/p90, righe/s in lettura, carico CPU altrui con flag
DISTURBATO) e' in `corse_2080ti/20261018_multichat45/mc2.py` (patch_R.py), provato solo come costruzione dei contesti, non ancora con il motore.
Misure valide finora: tabella sopra (finestra pulita 17:16-18:22) e `i-pmtp2`/`i0-b2`/`l-bpipe` (carico CPU 6-16%): PIPED_MTP vs gruppi senza MTP
`same:2` 51,6 / 50,6, `diff:2` 67,8 / 54,2, solo 50,2 / 51,5; BATCH_PIPELINE solo 55,0 (nessun recupero della chat singola).

## Uso reale (log del Daily 0.1.43, 13-17/10, solo conteggi)

Il Daily gira **senza `--batch`**: il server serializza (una richiesta alla volta), il log non ha timestamp per richiesta
(concorrenza e attese in coda non misurabili, dedotte). 1177 richieste (13 cancellate), prompt mediano 118k token (p90 158k).
Prefill = 46% del tempo attivo (7,9 h contro 9,2 h di decode); riuso cache 79%; letture >10 s = 402. 265 richieste (23%)
rileggono quasi tutto (~100k token, riusano solo il prefisso fisso da 9218): 6,3 h, l'83% del prefill, quasi tutte il 16/10
(alternanza fra conversazioni con `--conversation-cache-slots 2` / 6144 MiB). Decode mediano 55 t/s (p10 34, p90 73).
5 blocchi del watchdog a 240 s, 13 cancellazioni durante la lettura (988 s buttati). Tre colli: (1) cache conversazioni
troppo piccola per contesti da 100k, (2) prefill senza coda ne' preemption (FIFO), (3) stalli/cancellazioni che perdono la cache.

## Cosa e' gia' in 0.1.45 (opt-in, nessun codice da portare)

`STRATA_PIPED_CONCURRENT=1` (#2326: la lettura gira su un thread mentre i gruppi decodificano), `STRATA_PIPED_READ_PIECE`
(#2050), `STRATA_PIPE_BYIELD_NODRAIN`, `STRATA_ADMIT_SHORT_FIRST`, `STRATA_SLOT_COPY_DIRECT`/`COPY_BATCHSYNC`,
`--conversation-cache-park-gain`, gruppi senza `--batch-mtp` (default), piu' di 8 slot. Il prestito di prefisso (#1164)
c'e' ma `stages.empty()` lo spegne con lo split (da noi non agisce).

## Portato su lab/multichat45 (tutto opt-in, default invariati)

| cosa | origine | env |
|---|---|---|
| prefill impacchettato (+ drafter dei prompt impacchettati, K/V parcheggiato) | `lab/multichat43` a845d747, b6d8df2d, a6830a02, risolti 13 conflitti (admission `new=True`, `lend(...,exact)`, plan di `--prefill-pipe` escluso, `BYIELD` non scaricato) | `STRATA_PACK_PREFILL=N`, `STRATA_BATCH_PIPELINE=1` |
| #2589 slot libero preso alla conversazione meno costosa da perdere | upstream PR aperta | `STRATA_SLOT_EXPIRE_S` |
| #2384 checkpoint di turno che seguono la conversazione slot->solo->slot (solo generate.cpp; il server e' gia' #2518) | upstream PR aperta | - |
| #2985 bozze MTP nei gruppi in pipeline, uno slot per gruppo | upstream PR aperta | `--batch 2 --batch-mtp`, `STRATA_PIPED_MTP=1` |

Non portati: #1524 (rifiutata da upstream), #2738 (-22% misurato da terzi su 2080 Ti), spec208 (fork su 0.1.43, da riscrivere),
prestito di prefisso per stadio (#1164 esteso: i sotto-agenti funzionano gia' con la cache a 16 posti, costo park+restore
~0,3 s contro ~4 s di lettura).

## Leve del 19/10 (lab/multichat45b, da multichat45 34cf6351) - compilato no, installato no, provato no

Quattro leve dalla ricognizione `RICOGNIZIONE_PR_FORK_2026-10-19.md`; tutte opt-in, un commit per leva, registro in `stack/` (patch 18-20).
Scritte con il laboratorio in pausa (coordinatore, 19/10): **nessuna compilazione, nessuna misura**. La build (`build_lock.py`) e la serie di
verifica (`corse_2080ti/20261019_multichat45b/coda_mc45b.sh`, un solo `bench.lock`) partono quando la pausa viene tolta.

| leva | origine | env | commit | stato |
|---|---|---|---|---|
| la chat singola tiene `--pipeline-windows` accanto a `--batch` | upstream #3095 (+23/-5) | `STRATA_PW_WITH_BATCH=1` | 890bbca5 | portata; 3 conflitti in `generate.cpp` risolti (con `STRATA_BATCH_PIPELINE` e `STRATA_PIPED_MTP`, che restano indipendenti) |
| prefill delle richieste brevi, raccolta MMQ in blocchi | fork sibnick 97b83332 | (`STRATA_PREFILL_BATCH_GATHER`) | - | **non portata: gia' in 0.1.45** (vedi sotto) |
| fusione int8 degli esperti IQ anche su Turing | upstream #3104 (sostituisce #3085) | `STRATA_PF_FUSED=1 STRATA_PF_FUSED_SM75=1 STRATA_PF_FUSED_NATIVE_SM75=1` | 2ddc5e9f | portata, applicata pulita; cambia i bit (nb) |
| prefill di chunk molto corti sul kernel di decode | upstream #2774 | `STRATA_PF_DIRETTO=1` (la PR l'ha acceso di default: qui spento), `STRATA_PF_DIRETTO_MAX`, `STRATA_PF_ROUTE_PRIOR` (spento) | 0dd4dd95 | portata, applicata pulita; cambia i bit dei prompt corti (nb) |

**#3095.** Con `--batch` il motore spegneva le due finestre e teneva il drafter sull'ultima scheda (`mtp_first` richiedeva `batch == 0`). Con la
leva la richiesta sola mantiene le finestre e il drafter sulla prima scheda (3060); i drafter degli slot di `--batch-mtp` lo seguono (sessione
di carvatura `bslot_ss[0]`). `STRATA_BATCH_PIPELINE=1` (nostra, gia' misurata: solo 55,0) teneva le finestre ma non il drafter sulla prima
scheda: probabile motivo del mancato recupero. Non provato: `STRATA_PW_WITH_BATCH` insieme a `STRATA_PIPED_MTP` (il drafter degli slot e' sulla
prima scheda, le catene asincrone di PIPED_MTP presuppongono l'ultima). Atteso (S): chat singola da ~52 a ~62-65 t/s, aggregato invariato.

**Sibnick K37, gia' in base.** Il commit sibnick (raccolta MMQ degli esperti in blocchi sul percorso corto) si appoggia a un `main` precedente.
La 0.1.45 contiene `653f0113` (Breno Perucchi, 6/10: "gather experts in groups on short prompts too (resident and staged)"): ring da
`STAGE_GRP` = 32 slot, un solo wait e un solo evento per gruppo, spento da `STRATA_PREFILL_GROUP_GATHER=0`. E' lo stesso lavoro (108 078 -> 6 934 lanci
in sibnick = raggruppare), quindi il guadagno dichiarato (-12/-15%) e' gia' nel nostro exe; nella 0.1.43 del Daily non c'era. Portare il suo
codice avrebbe duplicato il raggruppamento (`group_gather` e `group_resident` coesisterebbero). Resta fuori solo il contatore opt-in
`STRATA_PREFILL_GATHER_COUNTS`. Verifica prevista: braccio `gg0` (`STRATA_PREFILL_GROUP_GATHER=0`) contro controllo, che misura quanto il
raggruppamento upstream gia' vale sui prompt da ~300 e ~3000 token.

**#3104.** Copre gate/up IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS e down Q2_0/IQ4_NL su sm_75; le coppie non coperte vanno su MMQ. Il pack nostro (gate/up
IQ3_XXS, down IQ4_NL sui layer 28-47) e' coperto dove il down e' IQ4_NL; gli altri layer restano su MMQ (da leggere nel log). `STRATA_PF_FUSED=1`
accende anche il percorso nativo gia' in upstream sulla 3060 (sm_86): l'effetto sul prompt riguarda i due stadi, non e' isolabile. Gli autori
danno errore relativo RMS 3,5e-5..2,6e-4 contro MMQ: i testi non sono garantiti identici, serve la KL.

**#2774.** Sotto `stream_all_min` gli esperti residenti con al massimo 8 righe escono da MMQ e girano su `native_expert_grouped` dal blob
compresso. Condizioni nel codice: `!stream_all`, senza peer, con cache e `host_res`. Scratch proprio (~9 MiB, `cudaMalloc` al primo uso: se non c'e'
VRAM il prefill fallisce con "PF_DIRETTO scratch"). Con `STRATA_PF_DIRETTO` non impostato il percorso e' quello di prima.

Verifica prevista sotto un solo `bench.lock` (`coda_mc45b.sh`): (a) `solo:S,solo:S` con `--batch 4 --batch-groups 2 ...` senza e con
`STRATA_PW_WITH_BATCH=1`, ordine C-P-C-P; (b) `mc45b_det.py` a cache fissa (`--expert-cache 1500`, senza `--elastic`, `--adapt-every 0`, senza cache
su disco ne' visione), 3 prompt da ~300 e 3 da ~3000 token a T=0, 96 token: controllo, `fused`, `diretto`, `gg0`, secondo controllo; confronto
con `cmp_det.py` (sha1 del testo contro il controllo, TTFT). Gli esiti vanno in una sezione apposta quando esistono.

## Misure mc2.py (corse singole, brevi; M = misurato)

Harness `corse_2080ti/20261018_multichat45/mc2.py` (copia di quello del 13/10), code `coda_mc45*.sh`, esiti `mc2_<tag>/res.json`.
`G2` = `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16`. t/s aggregati; solo = decode medio.
**Finestra pulita 17:16-18:22** (PC libero); dalle 18:22 altri processi dell'owner/agenti (node del Codex ~6 core, altri
lock del lab) hanno rallentato tutto di 1,5-2x: gli archi di quella finestra non sono confrontabili.

| braccio (exe) | solo | same:2 | diff:4 | cold:4 | agents:4 | agents:8 |
|---|---:|---:|---:|---:|---:|---:|
| a-daily: 0.1.45 puro, config Daily (upsmain) | 64,5 | 62,2 | 32,9 | - | 18,2 | 23,5 |
| b-g2cc: G2 (upsmain, solo config) | 56,2 | 60,7 | **74,8** | - | 26,7 | 28,7 |
| c-conc: G2 + `STRATA_PIPED_CONCURRENT=1` (m45) | 63,5 | - | - | 42,9 | **28,6** | **30,4** |

In c-conc le letture da ~1,6k token girano sul thread con 80-190 righe decodificate nel frattempo (25-40 righe/s).
Archi contaminati (non usare): d-pack3 (errore di stampa unicode dell'harness, rilancio con `PYTHONIOENCODING=utf-8`),
e-bpipe (avvio 436 s, solo 29,5), g-copy (agents 16,6/16,1), f-pmtp2 (solo 26,4, same:2 37,1, diff:2 41,6). Rilanci in coda
(`coda_mc45b.sh rerun`) con attesa di CPU libera e controllo b-g2cc ripetuto.

<!-- NOTTE:BEGIN -->
## Corse notturne 18-19/10 (aggiornate a ogni corsa da publish.py)

Aggregato = token generati / wall (t/s); dec = decode aggregato dal primo token. Una corsa per braccio, finestra con attesa di CPU libera sotto il lock. Obiettivo dell'owner: 180 t/s con 3 agenti.

| tag | ora | engine extra | env | scenari |
|---|---|---|---|---|
| `a-daily` | 18/10 17:25 | `config Daily` | `-` | solo:S **22.0** (dec 32.2, dec medio 64.5); same:2:S **62.2** (dec 63.3); diff:4:S **32.9** (dec 33.0); agents:4:P **18.2** (dec 25.2); agents:8:P **23.5** (dec 24.0) |
| `b-g2cc` | 18/10 17:33 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | solo:S **46.3** (dec 51.2, dec medio 56.2); same:2:S **60.7** (dec 61.7); diff:4:S **74.8** (dec 75.5); agents:4:P **26.7** (dec 28.0); agents:8:P **28.7** (dec 28.6) |
| `c-conc` | 18/10 18:22 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PIPED_CONCURRENT=1` | solo:S **51.8** (dec 57.4, dec medio 63.5); cold:4:S **42.9** (dec 43.0); agents:4:P **28.6** (dec 30.4); agents:8:P **30.4** (dec 30.3) |
| `e-bpipe` | 18/10 18:38 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_BATCH_PIPELINE=1` | solo:S **25.2** (dec 27.1, dec medio 29.5); same:4:S **29.4** (dec 29.6); diff:4:S **30.6** (dec 30.8) |
| `g-copy` | 18/10 18:47 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_SLOT_COPY_DIRECT=1 STRATA_COPY_BATCHSYNC=1` | agents:4:P **16.6** (dec 17.3); agents:8:P **16.1** (dec 16.1) |
| `f-pmtp2` | 18/10 18:53 | `--batch 2 --batch-mtp --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PIPED_MTP=1` | solo:S **22.7** (dec 24.4, dec medio 26.4); same:2:S **37.1** (dec 38.0); diff:2:S **41.6** (dec 43.5) |
| `f0-b2` | 18/10 19:26 | `--batch 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | solo:S **24.3** (dec 27.0, dec medio 29.1); same:2:S **30.7** (dec 31.1); diff:2:S **28.8** (dec 29.5) |
| `h-b-ctrl` | 18/10 20:05 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | agents:4:P **11.3** (dec 11.8); agents:8:P **15.8** (dec 15.7) |
| `i-pmtp2` | 18/10 20:34 | `--batch 2 --batch-mtp --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PIPED_MTP=1` | solo:S **41.8** (dec 45.7, dec medio 50.2); same:2:S **51.6** (dec 52.5); diff:2:S **67.8** (dec 69.9) |
| `i0-b2` | 18/10 20:37 | `--batch 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | solo:S **42.6** (dec 46.6, dec medio 51.5); same:2:S **50.6** (dec 51.3); diff:2:S **54.2** (dec 55.2) |
| `j-pack3` | 18/10 20:49 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PACK_PREFILL=3` | agents:4:P **15.7** (dec 18.8); agents:8:P **14.4** (dec 14.3) |
| `k-ctrl` | 18/10 20:58 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | agents:4:P **19.7** (dec 20.7); agents:8:P **22.3** (dec 22.3) |
| `l-bpipe` | 18/10 21:01 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_BATCH_PIPELINE=1` | solo:S **45.1** (dec 49.7, dec medio 55.0); same:4:S **61.9** (dec 62.5) |
| `m-pack3` | 18/10 21:38 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PACK_PREFILL=3` | agents:4:P **17.7** (dec 20.2); agents:8:P **19.8** (dec 19.8) |
| `n-ctrl` | 18/10 21:51 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | agents:4:P **15.3** (dec 16.0); agents:8:P **25.2** (dec 25.2) |
| `r0-warm` | 19/10 00:46 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | agents:3:R **20.4** (dec 21.1); [agents:3:R: prefill 66% del wall, TTFT p50/p90 6.83/17.42 s, lettura 404 tok/s, righe/s in lettura None, CPU altri 67.4% **DISTURBATO**] |
| `r1-ctrlA` | 19/10 00:55 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | solo:S **50.7** (dec 55.8, dec medio 61.5); agents:3:R **17.8** (dec 18.5); [agents:3:R: prefill 67% del wall, TTFT p50/p90 6.72/23.69 s, lettura 397 tok/s, righe/s in lettura None, CPU altri 66.6% **DISTURBATO**] |
| `r2-cfg1` | 19/10 01:02 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16 --short-read 256` | `STRATA_PIPED_CONCURRENT=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_SLOT_EXPIRE_S=300` | solo:S **52.3** (dec 57.1, dec medio 62.4); agents:3:R **21.6** (dec 22.6); [agents:3:R: prefill 78% del wall, TTFT p50/p90 8.61/29.98 s, lettura 325 tok/s, righe/s in lettura 18.7, CPU altri 52.1% **DISTURBATO**] |
| `r3-conc` | 19/10 01:09 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PIPED_CONCURRENT=1` | solo:S **49.1** (dec 53.8, dec medio 59.6); agents:3:R **19.6** (dec 20.4); [agents:3:R: prefill 78% del wall, TTFT p50/p90 8.84/30.01 s, lettura 327 tok/s, righe/s in lettura 16.5, CPU altri 54.0% **DISTURBATO**] |
| `r4-cfg1-kv16` | 19/10 01:16 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16 --short-read 256 --kv-resident 16384` | `STRATA_PIPED_CONCURRENT=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_SLOT_EXPIRE_S=300` | solo:S **55.1** (dec 60.4, dec medio 65.5); agents:3:R **20.1** (dec 21.0); [agents:3:R: prefill 83% del wall, TTFT p50/p90 9.47/27.14 s, lettura 303 tok/s, righe/s in lettura 18.3, CPU altri 50.6% **DISTURBATO**] |
| `r5-cfg1-piece` | 19/10 01:23 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16 --short-read 256` | `STRATA_PIPED_CONCURRENT=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_SLOT_EXPIRE_S=300 STRATA_PIPED_READ_PIECE=6656` | solo:S **54.6** (dec 59.9, dec medio 65.2); agents:3:R **20.6** (dec 21.5); [agents:3:R: prefill 78% del wall, TTFT p50/p90 8.88/30.09 s, lettura 315 tok/s, righe/s in lettura 17.3, CPU altri 52.7% **DISTURBATO**] |
| `r6-help` | 19/10 01:31 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16 --short-read 256` | `STRATA_PREFILL_HELP=1 STRATA_PREFILL_CPU_SHARE=auto` | solo:S **52.2** (dec 57.1, dec medio 62.4); agents:3:R **19.7** (dec 20.5); [agents:3:R: prefill 68% del wall, TTFT p50/p90 5.95/17.3 s, lettura 411 tok/s, righe/s in lettura None, CPU altri 66.2% **DISTURBATO**] |
| `r7-ctrlM` | 19/10 01:38 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | solo:S **50.6** (dec 55.8, dec medio 61.5); agents:3:R **18.1** (dec 18.8); [agents:3:R: prefill 68% del wall, TTFT p50/p90 6.38/23.78 s, lettura 391 tok/s, righe/s in lettura None, CPU altri 66.5% **DISTURBATO**] |
| `r8-pack3` | 19/10 01:46 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PACK_PREFILL=3` | solo:S **43.8** (dec 48.3, dec medio 53.2); agents:3:R **16.8** (dec 17.5); [agents:3:R: prefill 67% del wall, TTFT p50/p90 9.61/21.39 s, lettura 246 tok/s, righe/s in lettura None, CPU altri 63.2% **DISTURBATO**] |
| `r9-pack3-conc` | 19/10 01:54 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PACK_PREFILL=3 STRATA_PIPED_CONCURRENT=1` | solo:S **39.4** (dec 42.9, dec medio 46.8); agents:3:R **19.6** (dec 20.4); [agents:3:R: prefill 81% del wall, TTFT p50/p90 9.02/29.33 s, lettura 274 tok/s, righe/s in lettura 14.6, CPU altri 48.5% **DISTURBATO**] |
| `r10-bpipe` | 19/10 02:01 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_BATCH_PIPELINE=1` | solo:S **50.5** (dec 55.8, dec medio 61.5); agents:3:R **16.9** (dec 17.6); same:4:S **62.6** (dec 64.6); [agents:3:R: prefill 70% del wall, TTFT p50/p90 6.0/21.98 s, lettura 423 tok/s, righe/s in lettura None, CPU altri 64.7% **DISTURBATO**] |
| `r11-pmtp2` | 19/10 02:08 | `--batch 2 --batch-mtp --conversation-cache-mib 28672 --conversation-cache-slots 16` | `STRATA_PIPED_MTP=1` | solo:S **54.0** (dec 60.3, dec medio 66.6); same:2:S **82.6** (dec 84.2); agents:2:R **19.6** (dec 20.8); [agents:2:R: prefill 64% del wall, TTFT p50/p90 5.8/9.56 s, lettura 416 tok/s, righe/s in lettura None, CPU altri 71.4% **DISTURBATO**] |
| `r12-b2` | 19/10 02:14 | `--batch 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | solo:S **53.4** (dec 59.6, dec medio 65.6); same:2:S **64.5** (dec 65.5); agents:2:R **15.2** (dec 16.1); [agents:2:R: prefill 66% del wall, TTFT p50/p90 5.88/12.03 s, lettura 418 tok/s, righe/s in lettura None, CPU altri 71.0% **DISTURBATO**] |
| `r13-bf16tc` | 19/10 02:22 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16 --short-read 256` | `STRATA_PIPED_CONCURRENT=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_SLOT_EXPIRE_S=300 STRATA_BF16_TC=1` | pf:S **0.4** (dec 0.2); agents:3:R **21.3** (dec 22.3); [agents:3:R: prefill 79% del wall, TTFT p50/p90 10.3/28.3 s, lettura 331 tok/s, righe/s in lettura 16.3, CPU altri 53.4% **DISTURBATO**] |
| `r14-sm75perf` | 19/10 02:29 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16 --short-read 256` | `STRATA_PIPED_CONCURRENT=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_SLOT_EXPIRE_S=300 STRATA_SM75_PERF=1` | pf:S **0.4** (dec 0.2); agents:3:R **22.1** (dec 23.0); [agents:3:R: prefill 80% del wall, TTFT p50/p90 9.56/30.09 s, lettura 310 tok/s, righe/s in lettura 17.0, CPU altri 51.3% **DISTURBATO**] |
| `r15-cfg1-pf` | 19/10 02:32 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16 --short-read 256` | `STRATA_PIPED_CONCURRENT=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_SLOT_EXPIRE_S=300` | pf:S **0.3** (dec 0.2) |
| `r16-ctrlB` | 19/10 02:39 | `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` | `-` | solo:S **53.0** (dec 58.6, dec medio 64.8); agents:3:R **16.9** (dec 17.5); [agents:3:R: prefill 65% del wall, TTFT p50/p90 7.25/23.78 s, lettura 407 tok/s, righe/s in lettura None, CPU altri 67.3% **DISTURBATO**] |
<!-- NOTTE:END -->

# Multi-chat su 3060 + 2080 Ti (lab/multichat43)

Stato al 13/10, ripresa dopo la coda notturna. Obiettivo dell'owner: più chat (Claude Code / DSH, anche sotto-agenti della stessa chat madre)
in parallelo, con un aggregato vicino al doppio della chat singola (~77/70 t/s) e la chat singola che non peggiora.

- Base: `lab/struct-a` d21b2870; la build provata è `build-structa-075ce875` (sha256 9ef62e54..., **lo stesso exe
  della 0.1.43 Daily**). Il branch si chiama `lab/multichat43` perché `lab/multichat` esiste già su origin (lavoro del
  7/10 su base 0.1.39/40, superato da upstream: batch-mtp sullo split e gruppi sono in 0.1.42/43); spostarlo avrebbe
  richiesto un force-push.
- Harness: `moe-aggressive-commit/docs/porto/strata_adattivo/corse_2080ti/20261013_multichat/mc2.py` (copia in sola
  lettura la config del Daily `strata-0143-daily.json`, senza `--prompt-cache-file` e `--vision`; le opzioni date in
  `--extra` sostituiscono quelle del Daily). Code di prova `coda_mc2_notte*.sh`, tabella `tabella.py`.
- Legenda: **(M)** misurato, **(S)** stimato.

## Scenari

| scenario | cosa |
|---|---|
| `solo:S` | una chat alla volta, contesto 8k in cache: decode medio della chat singola con quella config |
| `pf:S` | una lettura alla volta di una coda nuova di 128/512/2048/8192 token su un prefisso 8k in cache (TTFT, s) |
| `same:N:S` | N chat insieme sullo stesso contesto 8k, domande diverse, 384 token ciascuna |
| `diff:N:S` | N chat insieme su contesti 8k diversi (scaldati all'avvio: se la cache non li tiene, si rileggono) |
| `cold:4:S` | 3 chat in decode e, 3 s dopo, una con un contesto nuovo da 16k (prefill accanto al decode) |
| `agents:N:P` | N sotto-agenti con lo stesso prefisso da 32k, 3 turni ciascuno: ogni turno aggiunge un tool result (~1,5k token) e la risposta precedente, 200 token per turno, partenze sfalsate di 0,7 s |
| `*:G` | come sopra con contesti da 64k |

Aggregato = token generati / wall dal primo invio all'ultima fine (TTFT e letture comprese).

## Prove di configurazione (M, 13/10 01:46-03:00, una corsa per braccio)

`CC` = `--conversation-cache-mib 20480 --conversation-cache-slots 8` (il Daily ha 6144/2).

| scenario | Daily | `--batch 4 --batch-mtp` g1 +CC | **`--batch 4 --batch-groups 2` +CC** | `--batch 8 --batch-mtp` g1 +CC | `--batch 8 --batch-groups 2` +CC |
|---|---|---|---|---|---|
| solo:S (decode, 2 richieste) | 55,8 / 75,2 | 66,6 / 70,0 | 66,4 / 68,3 | 62,7 / 62,5 | 65,4 / 67,5 |
| pf:S TTFT 128/512/2k/8k (s) | 1,19/1,92/4,45/9,97 | 1,34/2,09/4,78/9,22 | 1,34/2,17/4,77/10,47 | 1,38/2,20/4,86/10,59 | 1,39/2,20/4,91/9,61 |
| same:2 | 61,4 | 63,6 | 60,6 | 60,5 | 57,6 |
| same:4 | 70,8 | 73,6 | **84,7** | 69,7 | 76,0 |
| same:8 | 70,6 | 69,7 | 70,0 | 60,8 | 71,8 |
| diff:2 | 35,6 | 31,3 | 35,0 | 27,3 | 35,9 |
| diff:4 | 37,4 | 34,1 | 40,2 | 28,4 | 40,8 |
| diff:8 | 24,7 | 33,6 | 39,3 | 28,9 | 40,6 |
| cold:4 | 23,8 | 34,0 | **47,9** | 30,2 | 45,4 |
| agents:4:P | 18,4 | 20,2 | **23,3** | 19,0 | 19,9 |
| agents:8:P | 27,5 | 26,0 | **33,3** | 27,2 | 32,1 |

Letture:
- **La raccomandazione di upstream 0.1.43.1 (a0116b27, #2038) vale anche qui**: gruppi in pipeline senza
  `--batch-mtp` è il braccio migliore ovunque (same:4 +20%, cold:4 +101%, diff:8 +59%, agents:8 +21% sul Daily).
  `--batch-mtp` a un gruppo pareggia il Daily. Il porto di a0116b27 è solo il default (con `--batch-groups 2` esplicito
  non serve codice).
- **Siamo lontani dal 2x**: il massimo è 84,7 t/s (same:4), e sul caso reale (agenti) 33 t/s contro 27,5.
- 8 slot non rendono più di 4: ogni slot toglie VRAM agli esperti (sotto).
- La chat singola con `--batch` configurato: 66-68 t/s contro 75 del Daily (−10%, due sole richieste: vedi coda 3).

## Diagnosi

### 1. Perché `--batch` spegne la pipeline della chat singola (M dal codice, peso M)

`generate.cpp` (decisione di `--pipeline-windows`): `o.batch != 0 ? "not with --batch slots"`. Non è un'incompatibilità
dimostrata: il commit d'origine (bfa532e9) dice «off for anything it does not support: --batch slots ...; nothing is in
flight outside the two pipelined loops». Con `--batch` configurato la richiesta sola gira comunque sul percorso solo
(GEN con MTP), ma senza le due finestre in volo. In più con `--batch-groups > 1` si spegne la cache elastica («not
validated with --batch-groups; using fixed cache») e gli slot tolgono VRAM agli esperti: con `--batch 4 --batch-groups 2`
la cache passa da 4246 a 3382 slot sulla 3060 (−20%) e da 9148 a 7739 sulla 2080 Ti (−15%).
Peso sulla chat singola: −10% (M, 2 richieste per braccio; la coda 3 lo rimisura con 8). Il 47-49 t/s del 12/10 era
con `--batch 6` (2,18 GiB di slot sulla 3060).

### 2. Perché la finestra batch costa 75-94 ms (M)

`--batch 4 --batch-mtp`, same:4 (4 chat, 6,9 righe a finestra, 77 ms): stadio 0 = GPU 18,9 + esperti CPU 6,7 ms;
stadio 1 = GPU 26,1 + esperti CPU 9,8 ms; commit 3,3 + emissione 3,4 + adattivo 1,5 ms; il resto (~7 ms) passaggio e
staging. **Gli stadi corrono uno dopo l'altro** (`run_slot_rows` sincronizza lo stadio 0 e poi chiama lo stadio 1):
ogni scheda calcola il 25-34% della finestra e aspetta il resto. È il «lavorano un quarto del tempo» dell'owner.

Profilo GPU (`STRATA_VERIFY_PROFILE=1`, M, finestre miste solo+batch, i fusi spenti dal profilo): pesi densi e
hyper-connection ~35-40% del tempo GPU dello stadio, esperti in VRAM ~22%, attenzione QSA ~12% (ogni chat ha il suo
K/V: in batch non si condivide). Un riga in più nella stessa chat costa ~2,2/2,9 ms (stadio 0/1, da STRUCT_A); in batch
fra chat diverse ~3,7 ms a stadio (S, da 6,9 righe in 77 ms), perché ogni chat sceglie esperti suoi.

### 3. Esperti su CPU, attese, prefill che blocca (M)

- Esperti CPU: 6,7 + 9,8 = 16,5 ms su 77 (21%) a 4 chat; a 8 slot (`--batch 8 --batch-mtp`) salgono a 17 + 29 ms
  per finestra (meno VRAM, più chat) e la finestra arriva a 98 ms: per questo 8 slot rendono meno di 4.
- Attese GPU dell'host (wait): 45 ms su 77 (58%) = calcolo GPU, ma fatto da una scheda alla volta.
- **Prefill**: è il costo dominante nei casi reali.
  - Ogni lettura costa ~1 s fisso più ~1,0-1,8 ms a token: 128 token 1,2 s, 512 1,9 s, 2k 4,5 s, 8k 10 s con
    prefisso in cache (M, Daily). A 33k da zero 1452 tok/s (0,69 ms/token).
  - In agents:8 la lettura dei prompt è il **64% del wall** sia sul Daily (104 s su 161) sia coi gruppi (85 s su 134);
    il decode è la parte minore.
  - Coi gruppi in pipeline (`piped`) una lettura non lascia decodificare nessuno finché non finisce
    (`batch_read_piece` torna 0 se `piped`): il prompt ferma tutti (è il #2050 di upstream). Con un gruppo solo la
    lettura va a chunk e fra un chunk e l'altro gli slot decodificano metà del tempo del chunk: una lettura da 8k beside
    decode è durata 14,6 s (580 t/s, 2 parti) contro 11,1 s da sola, e gli slot hanno decodificato 2,6 s.
  - Ammissione: dopo la lettura la sessione viene **copiata** nello slot (`copy_to_slot`: K/V in RAM pinned salvato in
    un'immagine e ricaricato, su ogni stadio): ~13 ms per 1k token (8k 0,10 s; 131k 0,7-1,5 s, M 12/10), e la stessa
    copia all'indietro quando il turno dopo riparte dallo slot. Nel frattempo nessuno slot decodifica.
- **Cache**: col Daily (6144 MiB / 2 conversazioni) 8 contesti diversi non restano in cache: diff:4 e diff:8
  rileggono 8k a richiesta (11 s l'una). Con 8 posti si rileggono ancora i contesti più vecchi, perché ogni
  conversazione (contesto + domanda + risposta) occupa un posto. Il prefisso comune dei sotto-agenti invece funziona già:
  un posto parcheggiato che contiene il prefisso da 33380 token serve tutti (letti solo i tool result), ma ogni agente
  che perde la sua storia la rilegge (59k token nuovi letti sul Daily contro 35k coi gruppi e 8 posti).

### 4. Cosa fa diversamente blange48 (#2058: 4x 5080, 144/268/392 t/s a 1/4/8 client)

- 4 stadi: la sua chat singola (finestre seriali) usa una scheda su quattro alla volta; i gruppi in pipeline (4 gruppi
  da 2) le riempiono tutte. Il guadagno massimo dei gruppi è il numero di stadi (4x per lui, 2x per noi), e la nostra
  chat singola ha già la pipeline a 2 finestre, quindi il nostro punto di partenza è più alto.
- Esperti quasi tutti in VRAM (64 GB con `--trim-stage-weights`: 84-100% per scheda nella misura upstream): una
  riga in più costa solo GPU. Da noi il 45% degli esperti è in RAM e il batch fra chat diverse moltiplica gli esperti
  CPU.
- `--batch-groups 4` senza `--batch-mtp` (i gruppi non fanno bozze): esattamente a0116b27. Il suo «spec 4» vale per
  la sola chat singola.
- `--kv-resident 32768` come noi.

### 5. `--batch-groups 2` con 2 stadi fa lo stesso? (M)

Sì, è lo stesso meccanismo: 2 gruppi da 2 slot, ognuno sul suo stadio (`strata batch (pipelined, 2 groups of 2)`).
same:4: 1531 righe in 17,7 s = 86 righe/s; con `--batch-mtp` a un gruppo 6,95 token ogni 77 ms = 90 t/s di finestra
ma più tempo in ammissioni. Il guadagno dei gruppi è la sovrapposizione degli stadi; quello che perdono è l'MTP (1
token a riga invece di 1,74), l'adattivo e la cache elastica.

## Riferimenti esterni (dalla ricognizione 86ce57e)

- **a0116b27 / #2038** (0.1.43.1): solo il default; misurato sopra.
- **#1524**: finestre larghe condivise fra 2-4 stream su una scheda, 1,3-2,1x. Su una scheda: da noi il limite è la
  finestra a 8 righe (`kVerifyMaxT = 8`, 4 slot con MTP) e gli stadi seriali; idea utile per le finestre larghe.
- **ScrambledEggsOnToast/spec208** (`--batch-mtp --batch-mtp-pipeline --batch-groups 2`): bozze MTP dentro i gruppi in
  pipeline, a ondate di 2 slot, con un secondo verificatore per gruppo. Misura del fork (3 schede, tutti gli esperti
  in VRAM): +14% (prompt corti) e +6,6% (9,5k) sui gruppi senza MTP. Il codice tocca `generate.cpp` (+300 righe),
  `verify.cpp`, `resident_slots.cpp` su una base che ha 957 commit propri: da riscrivere sulla nostra, non da
  cherry-pick. Nota del fork: un gruppo finito aspetta che gli altri stadi abbiano servito gli esperti CPU prima del
  commit, quindi con esperti in RAM (noi) la sovrapposizione si riduce.

## Seconda coda (M, 13/10 02:54-03:19; interrotta per fine budget)

**Cache a 16 posti** (`--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16`):

| scenario | Daily | g2 + 8 posti | **g2 + 16 posti** |
|---|---|---|---|
| solo:S (decode) | 55,8 / 75,2 | 66,4 / 68,3 | 66,6 / 71,2 (media 68,9) |
| same:4 | 70,8 | 84,7 | 80,3 |
| diff:4 | 37,4 | 40,2 | **85,9** (nessuna rilettura) |
| diff:8 | 24,7 | 39,3 | **70,3** |
| cold:4 | 23,8 | 47,9 | 48,1 |
| agents:4:P | 18,4 | 23,3 | **26,6** |
| agents:8:P | 27,5 | 33,3 | 33,5 |

La cache più grande raddoppia le chat su contesti diversi (diff:4 2,3x il Daily). È config pura: RAM 96 GB, un
contesto da 8k parcheggiato pesa ~0,48 GB, da 33k ~1,1 GB, da 131k ~3,5-4 GB (S); 28 GiB bastano per ~8 agenti a 64k.

**Da dove viene il secondo fisso di ogni lettura** (`STRATA_PREFILL_TIMING=1`, Daily, prefisso 8k in cache, M):

| coda nuova | stadio 0 (GPU) | stadio 1 (GPU) | totale richiesta | dequant (st.0/st.1) |
|---|---|---|---|---|
| 139 token | 483 ms | 510 ms | 1,36 s | 46% / 43% |
| 529 token | 910 ms | 970 ms | 2,27 s | 39% / 39% |
| 2026 token | 2344 ms | 2245 ms | 5,31 s | 26% / 22% |
| 8192+399 token | 4653+1298 ms | 6989 ms | 13,1 s | 14% |

- Un pezzo di prompt tocca quasi tutti gli esperti dei suoi layer già da ~140 token (140 x 10 scelte su 512 esperti):
  **dequantizzare e caricare tutti gli esperti costa ~0,2-0,6 s a stadio, qualunque sia la lunghezza**. È questo il
  «secondo fisso», non la cache (il ripristino del checkpoint è 44 ms).
- **Con un pezzo solo gli stadi vanno in serie** (stadio 0 poi stadio 1): la pipeline fra stadi c'è solo fra i chunk
  di una lettura lunga. Un tool result da 2k tiene ferma ogni scheda metà del tempo.
- Il drafter dopo ogni chunk costa 89-365 ms sullo stadio 1 (`STRATA_SPLIT_MTP_BATCH=1` lo mette in batch: non provato).

Non misurati (coda fermata): `STRATA_PREFILL_HELP=1`, `STRATA_BATCH_DECODE_SHARE=2`, contesti 64k (`:G`),
`--kv-resident 16384`, la chat singola a 8 richieste (coda 3). Gli script sono pronti: `coda_mc2_notte2.sh` dalla riga
`b4g2-cc16-help`, `coda_mc2_notte3.sh`.

## Configurazione proposta oggi (M)

`--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` sopra la config del Daily.
Aggregato: 2 chat ~60 (come il Daily), 4 chat 80-86, 8 chat 70, agenti 27-34 t/s. **Costo: chat singola −8/−10%**
(66-71 contro 75 del Daily, campione piccolo) perché `--batch` spegne `--pipeline-windows 2` e la cache elastica e
toglie ~15-20% di slot esperti. Quindi **non** va nel Daily finché la leva L1 non riporta la chat singola al livello del
Daily: oggi è una config da usare solo nelle sessioni con più agenti.

## Disegno: una sola lettura dei pesi per più sessioni (direzione dell'owner)

### Cosa dicono i numeri

- Finestra di decode batch (4 chat, 7 righe): ~77 ms, cioè ~11 ms a riga; una riga in più nella stessa chat ~5 ms
  (2,2 + 2,9 ms dei due stadi).
- Un pezzo di prefill paga comunque tutti gli esperti (dequant + caricamento, ~0,5 s a stadio): una riga di decode
  agganciata costa solo il suo calcolo e la sua attenzione, ~0,57 ms a token a stadio al ritmo di un chunk da 8k,
  **~1,1-1,5 ms a riga sui due stadi contro ~11 ms in finestra (S)**.
- Ma un passo misto dura quanto il chunk (0,5-5 s): la chat in decode avanza di 1,7 token a passo. Agganciare righe di
  decode a un singolo prefill risparmia solo una finestra da 77 ms per chunk (−2…−15% del tempo di prefill):
  **il batching misto puro, da solo, rende poco qui**, perché il nostro prefill non è limitato dalla lettura dei pesi
  residenti (il caso di Sarathi/vLLM), ma dal toccare tutti gli esperti in ogni chunk.
- La forma che rende è il **prefill impacchettato di più sessioni**: le code di più agenti (tool result di 1-2k) nello
  stesso chunk. Quattro letture da 2k costano oggi 4 x 2,3 s a stadio = 9,2 s; un chunk da 8k ne costa 4,65 (M per i
  pezzi, S per l'impacchettamento): **−50% del tempo di prefill**, che negli agenti è il 64% del wall. Le righe di
  decode delle altre chat salgono sullo stesso chunk quasi gratis.
- E i chunk impacchettati consecutivi vanno in **pipeline fra i due stadi** (pacco k sullo stadio 1, pacco k+1 sullo
  stadio 0), come già fa la lettura lunga.

### Cosa serve nel motore

1. Un passo di prefill con **segmenti**: ogni riga ha la sua sessione (`SessionState` dello slot). Pesi densi, HC,
   esperto condiviso e esperti instradati sono per riga: si concatenano. Per sessione: GDN (scan a chunk con stato
   iniziale per segmento, oggi uno solo), attenzione QSA di prompt sul K/V del segmento (causale) e append K/V, PLE,
   indexer. Le righe di decode sono segmenti lunghi 1-2 (riga + bozza MTP) che usano l'attenzione di decode
   (`qsa_decode_attn_batch` esiste già).
2. Commit per segmento: accetta/rifiuta la bozza (`commit_slot_prefixes`), bozze MTP degli slot dopo il passo
   (come `batch_step`).
3. Scheduler (server + motore): a ogni passo prende tutte le letture in attesa (fino a un tetto di token) e le righe
   delle chat in decode; quando nessuno legge, finestre di decode (gruppi in pipeline). Con le sessioni in fasi diverse
   (una legge, una genera, una aspetta un tool) ogni passata si riempie con quello che c'è.
4. Ammissione senza copia: la lettura scrive direttamente nella sessione dello slot (oggi legge nella sessione
   principale e poi `copy_to_slot` copia tutto il K/V: ~13 ms per 1k token, 1-1,5 s a 131k, a ogni turno).

### Prefisso condiviso dei sotto-agenti

Funziona già, senza codice (M): la lettura dei sotto-agenti riprende dal checkpoint del turno parcheggiato che contiene
il prefisso comune (in agents:8 `cached 33380` su 35k: si leggono solo le code). Serve solo una cache con abbastanza
posti, uno per agente più il prefisso (16 posti sopra). Il K/V e lo stato GDN del prefisso però si **copiano** a ogni
ammissione (nessuna condivisione fisica): oltre 64k la copia pesa (0,7-1,5 s a 131k); la condivisione vera (pagine K/V
copy-on-write fra slot) è un passo successivo, utile solo coi contesti lunghi.

### Stima (S) degli aggregati, t/s

| | 2 chat | 4 chat | 8 chat |
|---|---|---|---|
| Daily oggi, chat in decode (same/diff) (M) | 61 / 36 | 71 / 37 | 71 / 25 |
| g2 + 16 posti oggi (M) | 61 / 35 | 80 / 86 | 70 / 70 |
| + L1 L2 (pipeline per il solo, gruppi con MTP) | 65-70 | 105-120 | 100-120 |
| Agenti, Daily oggi (M) | ~29 (S) | 18 | 28 |
| Agenti, g2 + 16 posti oggi (M) | — | 27 | 34 |
| + L3 prefill impacchettato e righe agganciate | 35-40 | 45-55 | 55-65 |
| + L2 | 40-45 | 50-60 | 60-75 |

Il 2x della chat singola (~150) non è raggiungibile con queste leve su due stadi con il 45% degli esperti in RAM:
il massimo realistico è ~1,5x (S) sulle chat in decode e ~2x il Daily attuale sugli agenti.

## L3 — prefill impacchettato: `STRATA_PACK_PREFILL` (scritto il 13/10, opt-in)

### Cosa dice il log degli agenti (M, `mc2_b4g2-cc16`, agents:8)

Ogni turno di un agente costa ~4,6 s di control lines: lettura della coda (1,6-2,1k token su 35k in cache) **~4,1 s**
(390-480 tok/s), park della conversazione uscente 65-180 ms, ripristino di quella entrante 70-110 ms, copia nello slot
210-250 ms. Il server dà le control lines a **una richiesta alla volta** (`_take_control`): il motore non vede mai due
prompt insieme, quindi l'impacchettamento ha bisogno anche del server.

### Disegno

1. **Server** (`_pack_hint`): una richiesta che sta per aspettare le control lines (lock preso da un'altra) manda
   avanti il suo prompt, `BHINT <id,...>`, senza risposta. Solo testo, solo se l'INFO del motore dice `pack_hints=N`.
2. **Motore, slot di pacco**: con `STRATA_PACK_PREFILL=N` (1-8) il motore crea N sessioni per stadio, come gli slot del
   batch ma **fuori dalle finestre** (coi gruppi in pipeline uno slot inattivo viene sporcato dalle righe di
   riempimento: non può fare da deposito). Costano VRAM come uno slot (0,36 GiB su CUDA0 e 0,66 GiB su CUDA1 ciascuna,
   M dal log; RAM pinned del K/V in streaming come uno slot).
3. **Lettura impacchettata** (`pk_read`, nella `read_part` della richiesta che ha le control lines, solo per una
   lettura in un pezzo, cioè coi gruppi in pipeline): per ogni prompt mandato avanti, fino a N, il motore prende il
   prefisso in cache più lungo (una conversazione parcheggiata, letta sul posto con `ConversationCache::peek`, o un
   checkpoint del prompt corrente sotto la sua posizione), lo ripristina nello slot di pacco su ogni stadio, e legge
   la coda fino all'**ultimo confine di turno** (lo stesso `turn_at` che la richiesta userebbe) **negli stessi chunk**
   della lettura corrente: `Prefill::run_packed`. Poi salva lì il checkpoint di confine.
4. **`Prefill::run_packed`**: le righe dei segmenti concatenate e lette a chunk; hyper-connection, proiezioni ed
   esperti sul chunk intero (una lettura e un dequant degli esperti per tutti), per segmento sulla sua sessione e alle
   sue posizioni: storia PLE, conv e ricorrenza GDN, append K/V (con lo staging per segmento se il K/V è in streaming),
   indexer, selezione, attenzione. `on_chunk`/`on_stage_chunk` vedono solo le righe della sessione propria (draft MTP,
   checkpoint periodici). Il percorso di default è un ramo `pk == nullptr`: stesso codice di prima.
5. **Ripresa**: quando arriva la `BGEN` della richiesta mandata avanti, lo slot di pacco è una sorgente come uno slot
   (`copy_from_slot`), più lunga del parcheggio; il checkpoint di confine passa nella catena principale (il turno
   dopo riparte da lì), la richiesta legge solo l'intestazione del turno. Lo slot di pacco si libera.
6. Opzioni: `STRATA_PACK_MIN` (default 128: code più corte non si impacchettano), `STRATA_PACK_CHUNK` (default 0 = un
   chunk come una lettura sola fino a 8192; N = chunk di ~N token, così un pacco grande va in pipeline fra i due stadi).

Limiti: niente immagini, niente `--batch-mtp`, solo con `--prompt-cache`; un `BHINT` di una richiesta poi annullata
legge per niente (lo slot si ricicla dal più vecchio); la richiesta che ha già le control lines non aspetta nessuno.

### Stima prima della misura (S)

Dalla tabella `STRATA_PREFILL_TIMING`: 4 code da 2k costano 4 x (2,34 + 2,25) = 18,4 s di GPU in serie, una da 8,6k
5,95 + 6,99 = 12,9 s (−30%); con `STRATA_PACK_CHUNK=4096` i due stadi si sovrappongono (S −40…−50%). Per ogni richiesta
impacchettata restano copia dallo slot di pacco (~0,2 s), intestazione, copia nello slot e park (~0,6 s). Sugli agenti
a 8: S 40-50 t/s contro 33,5 (la stima di 55-65 sopra contava anche le righe di decode agganciate, non fatte).

### Misure di costo e parità (M, 13/10 05:22-08:45, build `build-multichat-b6d8df2d`)

Scenario `packdet:4:P` di `mc2.py`: 4 sotto-agenti sul prefisso da 33k, turno 0 uno alla volta, poi il turno 1 (coda
~1,6k token a testa) mandato **insieme** con `max_tokens 1` (il wall di questa fase è il costo delle letture), poi il
turno 1 di nuovo uno alla volta da 200 token (decodificato da solo dal checkpoint di confine che la lettura concorrente
ha lasciato). Config: g2 + 16 posti, `--expert-cache 1500 --adapt-every 0` (cache fissa). Il braccio base ha gli stessi
slot di pacco allocati e mai usati (`STRATA_PACK_MIN=999999`), quindi la stessa VRAM.

| braccio | letture concorrenti di 4 code | TTFT a0/a1/a2/a3 (s) | testo t0 | testo t1 a0 / a1 a2 a3 |
|---|---|---|---|---|
| base (con MTP) | 18,80 s | 4,8 / 9,4 / 14,0 / 18,6 | = | = / = = = |
| `STRATA_PACK_PREFILL=3` | **11,12 s (−41%)** | 9,6 / 10,1 / 10,6 / 11,0 | identico | identico / diverso a 590, 47, 128 caratteri |
| `PACK=3 STRATA_PACK_CHUNK=4096` | 11,08 s | 9,3 / 10,4 / 11,0 / 9,7 | identico | identico / diverso a 207, 47, 128 |
| base senza MTP | 19,30 s | 5,1 / 9,7 / 14,4 / 19,2 | = | = / = = = |
| `PACK=3` senza MTP | **12,33 s (−36%)** | 10,7 / 12,3 / 11,1 / 11,6 | identico | identico / diverso a 381, 231, 135 |

- Il pacco legge 1569 + 3 x ~1610 token in **8,5 s** (un passo da 6,4k), contro ~4,6 s a coda letta da sola.
  La prima richiesta (quella che ha le control lines) aspetta il pacco: il suo TTFT sale da ~5 a ~10 s, le altre
  scendono; l'ultima da 18,6 a 11 s.
- `STRATA_PACK_CHUNK=4096` (due chunk in pipeline fra gli stadi) non cambia il wall: con 6,4k token il pacco è già
  dominato dal calcolo, non dal fisso degli esperti.
- **Parità**: il testo della richiesta che legge (a0) è identico in tutte le corse, con le sue righe dentro il pacco. Le
  tre richieste impacchettate divergono a metà testo (testo osservato coerente, contenuto simile: per
  esempio `-hf, -mu, -dr` contro `-hf, -dr`). **La divergenza esiste anche senza MTP**. È compatibile con la
  numerica della geometria del chunk: lo stesso pacco tagliato diversamente (`STRATA_PACK_CHUNK=4096`) sposta già la
  divergenza di a1 da 590 a 207 caratteri. Tra due corse base con VRAM diversa (`base` senza slot di pacco contro
  `base3`) divergono perfino i turni 0. **La parità bitwise non è dimostrata ed è fallita in questi confronti**.
  Senza confronto dei logits o margini misurati non si può dichiarare neppure parità numerica o attribuire ogni
  divergenza a token quasi pari. Il contenuto simile è un'osservazione qualitativa, non una verifica numerica.
- Il drafter MTP di una richiesta che riparte da uno slot di pacco ha il K/V di un'altra conversazione: le bozze
  rendono meno nel primo decode da solo, ma il testo non dipende da questo (la verifica è esatta). È così anche oggi
  per chi riparte da uno slot batch senza `--batch-mtp`. La patch incompleta
  `corse_2080ti/20261013_multichat/wip_pack_drafter_rows_b6d8df2d.patch` è stata completata nella ripresa del 13/10
  per gli slot di pacco; dettagli e limiti sotto. Queste misure precedono la correzione.

## Ultima coda consolidata (M, 13/10, conclusa alle 09:15)

Fonte: `corse_2080ti/20261013_multichat/mc2_m43d-{daily,base,bpipe,pack3,both}/res.json`.
RTX 3060 + 2080 Ti, stesso harness `mc2.py`, build `build-multichat-b6d8df2d`.
Base = `--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16`.

| braccio | solo:S decode medio, t/s | agents:4:P aggregato, t/s | agents:8:P aggregato, t/s |
|---|---:|---:|---:|
| Daily | 72,8 | 26,455 | 25,285 |
| base | 70,7 | 28,531 | 32,502 |
| base + `STRATA_BATCH_PIPELINE=1` | 74,0 | — | — |
| base + `STRATA_PACK_PREFILL=3` | — | 31,336 | 28,219 |
| base + entrambe le variabili | 69,7 | — | 23,744 |

**Una corsa per braccio, risposte diverse, solo due richieste nel campione solo:S: dati esplorativi**.
Packing guadagna con 4 agenti ma perde con 8 in questa coda; insieme alla pipeline perde ancora.
Non basta per attribuire la regressione a un bug dell'interazione: i tre slot di packing tolgono VRAM agli esperti,
il costo delle ammissioni e l'alternanza solo/batch cambiano, e le risposte non fanno lo stesso lavoro.
`STRATA_BATCH_PIPELINE=1` supera qui il Daily nel solo:S senza packing, ma non dimostra assenza di regressione generale.
Nessuna di queste opzioni viene promossa a default o al Daily.

## Correzione del 13/10: hints e drafter degli slot di pacco

La patch originaria aggiungeva soltanto un callback e campi non usati. Il completamento aggiunge:

- Validazione di `BHINT` **prima** che possa essere eseguito: almeno due token, tutti nel vocabolario e spazio
  per almeno un token generato più gli otto di margine del protocollo. Prima un hint invalido poteva raggiungere
  embedding e K/V senza passare dai controlli di `GEN`.
- Per ogni slot, immagine K/V del drafter della sorgente e residui FP32 del solo suffisso letto in pacco,
  associati ai token successivi della stessa sessione. Alla ripresa il drafter ripristina quell'immagine e
  ricostruisce il suffisso prima di leggere l'intestazione o lanciare le finestre speculative.
- Limite di **256 MiB di payload residui per slot** (6553 righe con larghezza 10240 float); suffissi più lunghi
  seguono la lettura ordinaria quando MTP è attivo. Si aggiungono l'immagine K/V del prefisso, metadati e token;
  non è un limite di 256 MiB all'intero slot. Replay con staging GPU di 64 righe, circa 2,5 MiB per questa geometria.
- Consumo, eviction, errore del packed prefill e invalidazione dopo non-finiti liberano immagine e vettori.
  Un `GEN` parsato ma respinto per contesto o vocabolario scarta gli slot pack corrispondenti.
  `BHINT` non contiene un request ID: una cancellazione prima del `GEN`, o una riga non parsabile, può lasciare
  il risultato anticipato fino alla prossima eviction; il numero di slot e il payload per slot restano limitati.
- I callback dei segmenti devono coprire il suffisso in ordine, senza buchi o sovrapposizioni. Il callback viene
  rimosso solo dopo che `run_packed` ha drenato tutti gli stadi, anche sul percorso di errore.

La ricostruzione riguarda gli slot di packing: **il caso preesistente di ritorno da uno slot batch senza
`--batch-mtp` rimane da risolvere**, così come l'eventuale storia approssimata già presente nell'immagine sorgente.
Il blocco `copy_from_slot` è identico alla build precedente `b6d8df2d`: copia sempre lo stato target, ma il K/V
del drafter solo quando `batch_mtp` è attivo; con più gruppi questa opzione è disabilitata. Ripristinare il ring
del drafter e aggiornare la lunghezza non ricrea le righe prodotte nel batch ordinario. Risolverlo richiederebbe
conservare quei residui, mantenere un drafter per slot o rileggere la storia: è un lavoro distinto dal replay dei
suffissi packed già disponibili. Non è stata introdotta qui una riscrittura L2.
Il drafter propone, il target verifica; ricostruire le bozze non è una prova di identità del testo fra geometrie diverse.
Non è stata individuata una violazione specifica del drain fra `STRATA_BATCH_PIPELINE` e packing: i percorsi batch
continuano dopo il drain della richiesta solo. Il peggioramento prestazionale della coda resta non attribuito.

Senza `STRATA_PACK_PREFILL` non si allocano sessioni, immagini o residui nuovi; `BHINT` non è pubblicizzato dal server.
Il percorso ordinario `pk == nullptr` conserva il calcolo precedente. I confronti della nuova build sotto
non dimostrano una parità globale di output: l'assenza di cambi al calcolo di default è una verifica del codice.

Verifiche CPU eseguite: `tools/test_pack_prefill.cpp` compilato con clang++ 19.1.5 (17 assert superati: limiti,
vocabolario, overflow del budget, chunk consecutivi/duplicati/incompleti); `python -B -m unittest serve.test_pack_hint -v`
(4 test superati), più `python -B -m unittest serve.test_parallel` (25 test superati in 43,35 s: scheduling,
cancellazione, ritorno al solo, errori e isolamento delle richieste con motore simulato).
Comandi, esiti e provenienza sono registrati nell'output esterno `multichat/cpu_validation/provenance.json`:
gli estratti sono trascritti dagli output tool conservati, non log grezzi ritrovati né nuove esecuzioni. Il codice
di uscita originale è conservato per i 25 test; per gli altri due comandi non era stato riportato separatamente.
Build **CUDA sm75 + sm86 verificata**, `cmake --build build-sm7586 --target strata -j 2` sotto lock: completata
con exit code 0 in 14 s. SHA256 dell'eseguibile:
`b1f3ccd05450293cea4c7bd985885977cf2d227fcddf4d57cd69d4c627db5162`.
I `.cpp` toccati sono condivisi da CUDA e HIP; **HIP non compilato** nell'ambiente corrente e rimane non verificato.
SYCL usa copie proprie di `generate.cpp` e `prefill.cpp`, non modificate; il campo nel header pubblico è additivo.

## Verifiche GPU della correzione (M, 13/10)

Hardware riconfermato via `nvidia-smi`: RTX 3060 **12288 MiB**, RTX 2080 Ti modificata **22528 MiB** (22 GiB).
Il desktop occupava circa 2412 MiB sulla 3060 all'avvio. Le prove correnti usano `--expert-cache 1500
--adapt-every 0`, cache elastica disabilitata: non confrontare i loro t/s assoluti con i 77/70 del profilo auto storico.
Output esterni al repository, nella workspace Codex `recupero-claude-strata-handoff`, sotto
`outputs/strata_completion_20261013/multichat/`.

**Default, senza PACK né BPIPE** (`mc2_default_old`, `mc2_default_new`, `mc2_default_old_repeat`): tutte le richieste
completate senza errore. Old contro new: testi identici **6/10**, cioè solo:S 2/2 e same:4:S 4/4, diff:4:S 0/4.
Old contro ripetizione old: **9/10**, differisce la seconda chat singola S1; repeat old contro new: **5/10**.
La variabilità della baseline è dimostrata, ma non spiega da sola tutte le divergenze old/new.
In diff:4 i primi caratteri diversi, indice da zero, sono 46/142/161/502 per S0/S1/S2/S3.
I log mostrano gruppi diversi: old e repeat accoppiano S0+S2 / S1+S3, new S0+S3 / S1+S2; il lock di ammissione
del server non è FIFO, quindi partenze sfalsate di 50 ms non fissano l'ordine delle righe. Prefissi riusati e
conteggi finali coincidono (200 token per risposta, 400 group-step e 795 righe). È una differenza osservata di
scheduling, **non una prova causale dell'intera divergenza**. Nessuna parità bitwise o numerica globale dichiarata.
Report: `default_regression_compare.json` e `default_repeat_compare.json`.

**Packing e replay** (`mc2_pack_parity_new`, `mc2_pack_bpipe_new`, `packdet:4:P`): tutte le richieste completate.
Con PACK=3, letture concorrenti **11,016 s**, tre replay da **1659/1656/1590** righe, ripristino degli slot
0/1/2 in 360,8/330,7/340,0 ms (copia target e ricostruzione drafter comprese).
Con PACK=3 e BPIPE=1, **11,594 s**, tre replay da 1660/1591/1656 righe e nessun errore osservato.
Sono controlli di funzionamento su una corsa per configurazione, non un confronto ripetuto di prestazioni:
le code lette differiscono leggermente e il nome `pack_parity_new` non certifica parità del testo.

**Ciclo di vita** (`tools/test_pack_lifecycle.py`, report `pack_lifecycle/result.json`): **PASS**, un solo processo
motore, circa 160 s inclusa apertura/chiusura; intervallo delle richieste 81,953 s. Prefisso comune di 36208 token,
19 richieste/cancellazioni, tre burst. Verificate cancellazione di una richiesta in attesa durante l'intervallo
di una lettura packed, ripresa con replay, riempimento e sostituzione di tutti e tre gli slot, ripresa dei nuovi
slot 0/1/2 e di una vecchia sessione espulsa. Tutte le risposte valide contengono il proprio marcatore SESSION e
nessun marcatore altrui. Il test usa l'Event che il server imposta dopo una disconnessione, non un socket HTTP;
l'intervallo è fra invio GEN e log di completamento packed, non una misura degli istanti dei singoli kernel.
Sul medesimo motore, gli hint con token negativo, fuori vocabolario e prompt di **299964 token** oltre il contesto
sono stati ignorati; la richiesta valida successiva è passata e il protocollo è rimasto sincronizzato.

## PACK a identica allocazione degli slot (M, 13/10)

Fonti esterne: `mc2_pack_perf_B1`, `mc2_pack_perf_A_resume1`, `mc2_pack_perf_B2_resume1`, ciascuna con
`res.json`, configurazione, engine/server log e telemetria `DET-mc-codex-pack_perf_*.environment.jsonl`.
Sorgenti `a6830a02`, eseguibile CUDA con SHA256 `b1f3ccd…` riportato sopra, invariati durante le prove.
B1/B2 usano `STRATA_PACK_PREFILL=3`; A mantiene i tre slot allocati ma disattiva la lettura anticipata con
`STRATA_PACK_MIN=999999`. Gli argomenti e le altre variabili sono identici: batch 4, due gruppi, cache conversazioni
28672 MiB/16 slot, expert cache 1500, adapt 0, senza elastic né BPIPE. Tutti riportano chunk automatico **3328 token,
ring di 96 slot**. Ogni motore esegue prima il warm-up P0, poi 4 e 8 agenti con tre turni e limite 200 token;
il caso a 8 agenti può quindi riusare la storia del caso a 4.

| agenti | braccio | token generati | wall scenario, s | aggregato, t/s | `dec_agg`, t/s | stop prima di 200 token |
|---:|---|---:|---:|---:|---:|---:|
| 4 | B1, PACK | 2104 | 76,703 | 27,4305 | 32,1767 | 4/12 |
| 4 | A, letture separate | 2089 | 89,578 | 23,3205 | 24,4399 | 3/12 |
| 4 | B2, PACK | 2266 | 78,891 | 28,7232 | 33,5322 | 1/12 |
| 8 | B1, PACK | 4185 | 162,000 | 25,8333 | 25,7422 | 5/24 |
| 8 | A, letture separate | 4160 | 153,829 | 27,0430 | 26,9473 | 6/24 |
| 8 | B2, PACK | 4470 | 160,219 | 27,8993 | 27,8120 | 4/24 |

Tutte le **36 richieste per braccio** sono terminate senza errori o cancellazioni. I server log confermano
`stop` per le risposte corte, `length` per quelle a 200 token: non sono tutti lavori di uguale lunghezza.
L'aggregato è token/wall dal primo invio all'ultima fine; `dec_agg` è la somma di `out-1` divisa per l'intervallo
dal primo token ricevuto all'ultima fine. **Include letture e ammissioni successive**, non misura il solo kernel decode.

Con 4 agenti il beneficio osservato mantiene il segno: **+17,6% / +23,2%** di throughput in B1/B2 rispetto ad A,
wall **−14,4% / −11,9%**. Con 8 agenti il throughput cambia segno (**−4,47% / +3,17%**) e il wall peggiora in
entrambi (**+5,31% / +4,15%**). B2 genera il 7,45% di token in più di A: il suo throughput maggiore non equivale
a concludere prima. Questa serie non dimostra un vantaggio stabile con 8 agenti.

Le risposte e l'ordine di ammissione variano. Nel caso a 4 agenti i primi gruppi sono a0+a3 / a2+a1 in B1,
a0+a1 / a2+a3 in A, a0+a1 / a3+a2 in B2; nel caso a 8 partono a0+a2 / a1+a3 in tutti, ma le ammissioni
successive e gli stop divergono. Testi identici B1/A: **0/12 e 2/24**; B2/A: **0/12 e 0/24**;
B1/B2: **1/12 e 1/24**. Non è dimostrata parità bitwise o numerica, né è isolato il contributo causale di
packing, scheduling e lunghezza dell'output.

La sequenza comprende una pausa richiesta dall'utente, non un B–A–B continuo: campioni ambiente B1
**11:08:59–11:14:45**, A ripreso **11:38:24–11:44:12**, B2 **11:44:22–11:52:21** (ora locale).
Il primo `mc2_pack_perf_A`, interrotto durante il caso a 8 dopo aver finito solo quello a 4, è preservato ed
**escluso** dalla tabella. Avvio motore 74/74/205 s e warm-up 32,5/32,6/32,8 s per B1/A/B2, fuori dal wall
degli scenari; durata totale dei job 353/358/485 s. Telemetria disponibile, ma nessuna equivalenza rigorosa di
carico CPU, disco, desktop o temperatura è dimostrata. Due campioni PACK e uno A non sono una stima statistica
del beneficio. Il confronto a uguale allocazione degli slot elimina quel fattore, non tutte le fonti di variabilità.

La variante PACK_CHUNK=4096 e la leva NOSWITCH non sono state rimisurate in questo confronto. Le prove CUDA e
funzionali superate autorizzano la correzione opt-in, **non una promozione al Daily né una promessa di throughput**.

L'harness originale scrive accanto a sé e contiene percorsi della macchina precedente: copiarlo/adattarlo **fuori dal
repository lab**, lasciando fixture e configurazione Daily in sola lettura. Le prove non autorizzano cambi al Daily.

## Leve di codice, in ordine

1. **L3 — prefill impacchettato di più sessioni con le righe di decode agganciate** (la direzione dell'owner, nella
   forma che rende qui). **Scritto il 13/10 senza le righe di decode** (sopra, `STRATA_PACK_PREFILL`). Grande: `Prefill::run` a segmenti (GDN e attenzione per segmento), scheduler. Primo passo
   misurabile: impacchettare solo le letture (senza righe di decode), 2 sessioni, parità di testo contro le letture
   separate.
2. **L1 — `--pipeline-windows 2` per la richiesta sola anche con `--batch`**: togliere il cancello «not with --batch
   slots» e scaricare la pipeline prima di promuovere la richiesta in uno slot; insieme, la cache elastica con i gruppi.
   Media. Condizione per mettere il batch nel Daily. **Si tocca con `lab/pipeline2`** (l'altro agente ridisegna la
   pipeline della chat singola): concordare prima chi tocca il cancello.
3. **L2 — gruppi in pipeline con le bozze MTP** (idea di spec208, da riscrivere sulla nostra base: il fork ha 957
   commit propri): S +30-40% sulle chat in decode (1,74 token a riga contro 1).
4. **#2050** di upstream: con i gruppi una lettura non ferma più gli altri (`batch_read_piece` torna 0 con `piped`).
   Media, cold:4 e agenti.
5. Ammissione senza copia (scrivere nello slot), utile coi contesti lunghi.
6. Config da provare (pronte negli script): `STRATA_PREFILL_HELP=1`, `STRATA_SPLIT_MTP_BATCH=1`,
   `STRATA_BATCH_DECODE_SHARE`, `--kv-resident 16384`, contesti 64k.

## Chiusura del filone

La prosecuzione autorizzata resta su `lab/multichat43`. L1 opt-in è già presente (`STRATA_BATCH_PIPELINE=1`),
L3 è implementato senza righe di decode agganciate e la patch drafter è completata e verificata su CUDA, con i
limiti di parità e prestazioni elencati sopra. Il confronto PACK a pari slot è concluso: beneficio osservato a
4 agenti, segno non stabile a 8; nessun cambio di default o del Daily. HIP rimane non compilato e il drafter
al ritorno dal batch ordinario senza `--batch-mtp` resta il limite preesistente descritto sopra.
L2 gruppi con MTP, ammissione senza copia, condivisione fisica del
prefisso e tuning esteso restano lavori futuri: non sono necessari per validare questa correzione.
