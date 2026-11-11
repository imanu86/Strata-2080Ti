# EP_SPLIT_PREFILL: il prompt a strati, la risposta expert-parallel

Ramo `lab/upsync-v0.2.9`, 3/11. Tappa 4 di `PIPELINE3_PIANO.md` (§3a, §4) e §3e di `PIPELINE3.md`. Tutto dietro
**`STRATA_EP_SPLIT_PREFILL=1`**, spenta di default: spenta, nessun oggetto, nessuna allocazione, nessuna riga cambia.
Le righe citate sono quelle lette il 3/11 su `8cb752a4`.

Contratto comune: `include/strata/core/ep_split.hpp` (knob, `Config`/`admit`, `StateMover`, `Sync`,
`checkpoint_merge_tail`, le righe di registro). Questo documento dice **come** le cose si incastrano; l'header dice
**che cosa** si chiama come. I pezzi (§9) scrivono file disgiunti e usano solo i simboli dell'header.

Convenzione: **M** misurato (`corse_2080ti\20261101_tre_schede`), **S** stimato da misure.

## 0. In una frase

Oggi con `--peer-device` la 3080 Ti legge il prompt **da sola** su 48 strati (il peer calcola solo i suoi esperti del
chunk, `prefill.cpp:2269` `set_peer`, via host): 1414-1476 t/s a 128K contro 2243 della divisione a strati (M, −35%).
La scrittura col mega-kernel (`STRATA_PEER_FUSED`) è invece la migliore: 22,7-22,9 ms a finestra, 85-87 t/s (M).
Con la variabile accesa l'engine **tiene la scrittura com'è** e legge il prompt **come la divisione a strati**: la
3080 Ti fa gli strati `[0, K)` sulla sua sessione, la 2080 Ti fa `[K, 48)` con **i suoi densi, la sua sessione (K/V,
GDN, indicizzatore)** e **la cache del peer come livello residente** dei suoi esperti, in pipeline chunk per chunk
come i `Prefill` a stadi di oggi. A prompt letto lo stato di `[K, 48)` passa dalla 2080 Ti alla sessione piena della
3080 Ti (niente P2P: pezzi bloccati D2H+H2D, oppure `memcpy` host→host per il K/V in streaming, la cui verità è già in
RAM bloccata); poi la scrittura gira identica a oggi. Prima del prompt del turno dopo, torna indietro ciò che la
scrittura ha scritto. K = 20 di default (`STRATA_EP_SPLIT_LAYER`).

Perché non il contrario (prompt tutto EP, oppure doppia modalità con pesi ricaricati): `PIPELINE3_PIANO` §3a lo
scarta (residenza EP della 3080 Ti troppo piccola; 1-2,5 s a cambio). Perché non chiedere alla 3080 Ti gli esperti
dello stadio 1: la 2080 Ti legge un blob dalla RAM bloccata sul suo x16 a 13 GB/s (M), mentre una riga chiesta alla
3080 Ti passa dall'Oculink x4 a 6,7 GB/s (M) **e** ruba tempo alla scheda che sta facendo lo stadio 0. Lo stadio 0
invece chiede al peer i suoi esperti di `[0, K)` (il percorso di `Vpeer`, oggi), perché per la 3080 Ti l'Oculink è il
collo e il peer calcola sul posto. «Simmetrico» quindi no: ogni scheda chiede all'altra solo quando le conviene.

## 1. Da dove partiamo: come gira oggi (righe lette)

| cosa | dove | oggi |
|---|---|---|
| cancello peer + divisione | `generate.cpp:4174` | `--peer-device` con `--layer-split` o `--expert-cache-device1..3` rifiutato |
| cancello prefissi su disco | `:3757` | `--prompt-cache-file` rifiutato con un peer |
| divisione a strati | `:3775-3824` | `split_at`, `split_devs`, `multi_gpu`; guida stadi, verificatori, checkpoint a parti, parcheggio a `stage_images` |
| stadio | `:1499 GpuStage` | `dev, lb, le, wt, dense, head, ss, stream, cache, d_res, ver, ver_b, sp, mrope` |
| pesi di uno stadio | `:5245 load_stage_weights(st, lb, le, trim)` | `trim`: solo i densi di `[lb, le)` (`NativeDense::set_layer_range`) |
| sessione CUDA0 | `:6170` `hi0 = multi_gpu ? split_at[0] : -1` | con `--peer-device` solo: carve pieno `[0, 48)` |
| sessioni degli stadi | `:6226-6249` | `session_bytes/init(g, max_context, K, buf, st.ss, st.lb, st.le)` |
| peer | `:7977-7994`, `peer_experts.cpp:214` | riempie `cache_` con le coppie che `xcache` non ha, fino a `free − reserve_mib` (`:306-321`) |
| tabella di residenza | `:8621-8694` | `host_res` (CUDA0, una per tutte le cache), `d_res`, `drive.d.host_res` |
| prompt path degli stadi | `:10157-10182` | `st.sp.set_stage(lb, le, next)`, `st.sp.init(wt, g, st.ss, srcp, &st.cache, host_res, chunk, stream, …)` |
| prompt path CUDA0 col peer | `:10261-10268` | `sp.set_peer(&peer, rows)`: il peer calcola i suoi esperti del chunk, via host (`prefill.cpp:2275`) |
| verificatore della scrittura | `:10361`, `:10578-10591` | `ver.init(wt, g, ss, vh, …)` + `set_peer_fused(peer.fused_hook())`; `verify.cpp:1262` rifiuta il hook con `next_`/`lb_ > 0` |
| on_chunk / on_stage_chunk | `:11581-11684` | con `multi_gpu` `on_chunk` va all'ultimo stadio (`:11661`); checkpoint a metà prompt dalle parti per stadio (`:11632-11642`, `:11425-11431`) |
| drafter | `mtp.cpp:1471`, `:1533/:1565` | `prefill(R_rows)` copia le righe D2D sul proprio device: le righe devono stare sulla scheda del drafter |
| stadi del prompt | `prefill.cpp:5730-5775` | lo stadio successivo gira su un filo, un chunk indietro, con `hand_in_` bloccata |
| azzeramento a inizio conversazione | `:15828-15835` | `session_zero(ss)` + ogni `stages[i]->ss` |
| ripristini | `:13615` (slot), `:15701` (prefisso prestato), `:15727` (parcheggio), `:15735` (parti stadi), `:15863` (checkpoint) | tutti scrivono le celle di `ss` (e degli stadi) |
| salvataggio prefisso su disco | `:17242-17265` | immagine di `ss` + `stage_images` |
| K/V in streaming | `layer.hpp:233-243`, `kv_stream.hpp:3-17` | `kv_mode 1`: la verità è la copia host bloccata (`QsaState::host`, `layer.cpp:585` Portable\|Mapped), `n_slots` pagine in VRAM, `map` |
| stato di una sessione | `session.hpp:50-62`, `session.cpp:38` | carve `[layer_lo, layer_hi)`, ordinali globali; `gdn_state_floats` = `S·H·S + C·(d−1)` |
| checkpoint | `conversation_state.cpp:386/392/431` | GDN rows + ple + tails/dead/block_pos **della carve**; una carve diversa è rifiutata (`:440-442`) |
| K/V | `conversation_snapshot.cpp:293/429/461`, `:413` | parti k/v/scale/pooled a `upto` celle; dopo una riscrittura della copia host `kv_stream_reset(map)` |

## 2. Disegno: oggetti e memoria per scheda

Con la variabile accesa **la divisione a strati non esiste** per il resto dell'engine: `multi_gpu` resta falso,
`stages` resta vuoto, `split_at` vuoto. Esiste un solo oggetto in più, **lo stadio EP**: un `GpuStage` tenuto a parte
(`ep_stage`, `std::unique_ptr<GpuStage>` fuori da `stages`), che usa solo `dev, lb, le, wt, dense, ss, stream, sp` e
lascia `cache, d_res, ver, ver_b, head, mrope` vuoti. Così verificatore, checkpoint, parcheggio, prefissi su disco,
slot e sessione-file vedono **una sola sessione piena su CUDA0**, esattamente come `--peer-device` oggi.

| scheda | oggi (`Vpeer`, M) | con EP (S) |
|---|---|---|
| **3080 Ti** (padrona) | densi 48 strati (~2,99 GiB), sessione piena, K/V residente 32768 celle × 12 QSA, testa, drafter, verificatore, prompt path (prestito 2119 slot), `xcache` 2561 slot | **un buffer in più, con il drafter**: le righe finali del chunk che lo stadio EP passa al drafter (`ep_R_dev0`: chunk × hc × n_embd × 4 B = **300 MiB a 7680 token**, 320 a 8192). È allocato dopo la cache, quindi il dimensionamento automatico della cache lo conta nella riserva (`ep_rows_bytes`, al chunk massimo dello scan): la cache ha ~180-190 slot in meno, la VRAM libera a fine avvio resta quella di oggi (~700 MiB). Senza drafter non esiste. Il prestito del prompt continua a prendere la coda di `xcache` (`:16218 lend`, parte 0 di `pf_parts`) |
| **2080 Ti** (server) | `PeerExperts::cache_` 10971 slot in 17,80 GiB (M: ~1,66 MiB a slot; riserva 3000 MiB, di cui ~2190 liberi per il ring del prompt di `set_peer`) | **più**: densi di `[20, 48)` (~1,75 GiB: `load_stage_weights(trim)`), sessione `[20, 48)` (7 QSA: 32768 celle residenti + copia host bloccata in RAM; 21 GDN; tabelle rope; a `--max-context 262144` ~0,6 GiB), buffer propri del prompt path (`Prefill::bytes_needed_owned(g, ep.ss, chunk)`, S 1,2-2,0 GiB a 2048 token col ring). **Tutto questo esce dalla cache del peer**: densi e sessione sono allocati prima di `peer.open`, che riempie fino a `free − reserve` (`peer_experts.cpp:306-317`), e la riserva sale dei buffer del prompt + 512 MiB (§4b). Conto (S): ~1450 slot per densi+sessione, ~800-1300 per i buffer e il margine → la cache del peer scende a **~8200-8700 slot** (da 10971). Le coppie che ne escono le calcola la CPU in scrittura, con arrotondamenti diversi (§10, §12). La riga `strata ep-split: CUDA1: the stage holds … MiB … (the same tier without the variable: --peer-reserve-mib N)` dice quanto e con quale riserva una corsa senza la variabile ha la stessa cache |
| RAM bloccata | arena 8 GiB + copie host K/V (3,09 GiB per 12 QSA a contesto pieno, M) | + la copia host dei 7 QSA della sessione EP a `max_context` (262144 celle INT8 × 1056 B × 7 ≈ **1,9 GB**, non 7/12 × 3,09) + i 2 buffer di passaggio di mano dello stadio 0 (`prefill.cpp:1611`: 2 × chunk × hc·n_embd × 4 = **600 MiB a 7680 token**, allocati perché ora `next_` è valorizzato) + 2 pezzi di staging (16 MiB): **≈ +2,5 GiB** rispetto a `--peer-device` di oggi |

Lo stadio EP **non ha una cache di esperti sua**: il suo prompt path riceve **la cache del peer e la tabella `res_`
del peer** (`Prefill::init(..., cache = &peer.cache(), host_res = peer.res_table(), ...)`), cioè la «seconda tabella
di residenza per il prompt» del PIANO §3a(a). `res_` è `n_layers × n_expert`, slot del peer o −1, lo stesso formato di
`host_res`; il prompt path la legge solo per gli strati `[K, 48)`. Niente prestito dalla cache del peer (i suoi slot
sono della scrittura: `own_win_`, `adapt`, `d_slot_ptr` del mega-kernel li tengono per indice), quindi lo stadio EP
**alloca i propri buffer** come uno stadio senza prestito (`:10164-10168`: `sb = nullptr`).

Il drafter resta su CUDA0 (`last_st == nullptr` → `mtp.bind(wt, &native_head, ver.final_R_all())`, `:10587`), come
in `Vpeer`.

## 3. Sequenza di una richiesta (primo turno, `resume == 0`)

1. **Avvio**: `admit` (§7) → riga `on`. Lo stadio EP carica i densi `[K, 48)` con `trim` (dopo CUDA0, come il ciclo
   `:5287`), crea `stream`, carva la sessione `[K, 48)` (`:6226-6249` senza testa), `session_zero`. **Prima** di
   `peer.open` (`:7985`) il reserve sale: `reserve_mib += ep_split::reserve_add_mib(Prefill::bytes_needed_owned(g,
   ep.ss, o.prefill_chunk))`. `peer.open` riempie la cache del peer con quel che resta. `ver.init(wt, g, ss, …)` e
   `set_peer_fused` **invariati**.
2. **Prompt path**: in `init_prompt_paths` (`:10157`), con EP: `ep.sp.set_stage(K, -1, nullptr)`; sotto
   `OnDevice(ep.dev)` `ep.sp.init(ep.wt, g, ep.ss, srcp, &peer.cache(), peer.res_table(), o.prefill_chunk, ep.stream,
   err, nullptr, 0)`; poi `sp.set_stage(0, K, &ep.sp)` e `sp.init(...)` come oggi; `sp.set_peer(&peer, rows)` come
   oggi (lo stadio 0 chiede al peer i suoi esperti di `[0, K)`; il `PeerPrefill` gira sulla 2080 Ti su stream propri,
   **in concorrenza** con lo stadio EP: VRAM separata, nessun dato condiviso). Il passo-giù del chunk (`own_fits`,
   `:10189`) conta anche la 2080 Ti (buffer propri: `bytes_needed(g, ep.ss, c)` contro la VRAM libera, che è la riserva).
   Niente `set_stage_helper` (`:10271`: solo `multi_gpu`).
3. **Richiesta**: `session_zero(ss)` (`:15829`) e **anche `session_zero(ep.ss)`** sotto `OnDevice(ep.dev)`; `sync.invalidate()`.
   `peer.quiesce()` (le copie di `refill_` di un `adapt` precedente sono atterrate: `res_` non cambia mentre lo
   stadio legge gli slot). Prestito del prompt su CUDA0 (`lend`, invariato), `sp.run(ids, n, 0)`:
   lo stadio 0 legge il chunk c su CUDA0, lo passa a `ep.sp` (filo, `hand_in_`), lo stadio EP legge c mentre lo
   stadio 0 legge c+1. Gli esperti dello stadio EP: in `peer.cache()` → hit sul posto; altrimenti stream dalla RAM
   sul x16 (il percorso di uno stadio senza prestito, `:10144`).
4. **`on_chunk`** sta sullo stadio EP (come `:11661` per `multi_gpu`), ma il drafter è su CUDA0: le righe finali
   del chunk (`R_rows`, T × hc·n_embd fp32 sulla 2080 Ti) passano con `mover.rows_to_primary(R_rows, ep_R_dev0, T,
   g.hc * g.n_embd)` in un buffer di CUDA0 allocato una volta per `chunk × hc × n_embd` float (300 MiB a 7680 token,
   hc = 4; **solo con un drafter**, e contato dal dimensionamento automatico della cache: §2); poi
   `mtp.prefill(ep_R_dev0, …)` come oggi. `draft_kv` batched (E-9) non si tenta (altro device).
   La riga `PP`, i checkpoint periodici (§6) e il resto di `on_chunk` invariati.
5. **A prompt letto** (dopo `run(q, r)` di tutti i segmenti, prima del checkpoint di coda / salvataggio prefisso /
   `refill`, `:17200-17275`): `mover.to_primary(ep.ss, ss, c0 = resume, c1 = n, running_state = true)`; poi
   `sync.note(ids, n)`. Da qui `ss` è la verità per tutti i 48 strati.
6. **Scrittura**: `ver.run` (`:18771`) con il mega-kernel, **invariata**: legge e scrive `ss` su 48 strati.
7. **Fine richiesta**: parcheggio, checkpoint, prefisso su disco, `refill` del prestito: tutti su `ss`, invariati.
   La riga `strata decode …` di fine richiesta aggiunge `ep-split: <handovers> passaggi, <MB>, <ms>`.

## 4. Turno successivo (`resume > 0`)

Dopo il primo turno le celle `[0, n)` di `[K, 48)` stanno su CUDA0 (prompt: calcolate dalla 2080 Ti e passate;
risposta: scritte da CUDA0). La sessione EP tiene ancora le celle del prompt `[0, n_prompt)` ma **non** quelle della
risposta, e il suo GDN è fermo a fine prompt. Il turno dopo legge `[resume, n')` con gli stadi (sempre, come fa la
divisione a strati per ogni segmento: niente soglia, così la lettura resta quella della divisione):

1. `c0 = sync.valid_for(ids, resume)`: le celle che lo stadio EP tiene **e** che coincidono con le celle di `ss`
   (stessa conversazione, nessuna riscrittura dal passaggio: §6). Tipico, continuazione: `c0 = n_prompt` del turno
   prima. Dopo un ripristino o una conversazione nuova: `c0 = 0`.
2. `mover.to_peer(ss, ep.ss, c0, resume, running_state = true)`: le celle `[c0, resume)` dei 7 QSA, le righe pooled,
   **sempre** le righe GDN dei 21 strati e tail/dead/block_pos (lo stato corrente a `resume`); `sync.note(ids, resume)`.
3. Il prompt come al §3 (punti 3-5), con `c0 = resume` nel passaggio in avanti.

Caso `resume == 0` con `sync` valida (stessa conversazione ricominciata): `session_zero(ep.ss)` + `invalidate`,
come per `ss`.

**Bit**: il primo turno dà **lo stesso testo della divisione a strati** a parità di stato (stessi stadi, stessi
kernel, stessa scheda per strato: lo stadio 0 con `set_peer` è quello di `Vpeer` per `[0, K)`, non quello di
`Vrif1`: identico a `Vrif1` solo con `--peer-prefill-rows 0`). La scrittura è **quella del mega-kernel** a parità di
stato (stesso `ver`, stesso `ss`). Dai turni successivi le celle della risposta di `[K, 48)` le ha scritte la 3080 Ti
(sm_86) e non la 2080 Ti (sm_75) come nella divisione: il prefill del turno 2 le legge → testo identico alla
divisione **solo a parità di stato**, non per costruzione. Va detto nel confronto, non nascosto.

## 5. Il passaggio di stato fra le schede (`StateMover`)

Per ogni strato `l ∈ [K, 48)`, due direzioni simmetriche (`to_primary` / `to_peer`), celle `[c0, c1)`:

| parte | dove sta | come passa | note |
|---|---|---|---|
| K/V dei 7 QSA, `kv_mode 0` (tutto in VRAM) | pool `k_q/v_q/k_scale/v_scale` (INT8), `k_q4/v_q4` (Q4), `k_pool/v_pool` (fp16), ibrido `k_q+k_scale+v_q4` | pagine intere `[⌊c0/page⌋·page, ⌈c1/page⌉·page)`: byte `[bytes(c0'), bytes(c1'))` di ogni parte da `conversation_kv_part_sizes` (le celle vengono prima, layout identità); D2H in un pezzo bloccato sullo stream della sorgente, H2D dal pezzo sullo stream della destinazione, due pezzi a staffetta (evento per pezzo) | il percorso «0,3 s a 128K» |
| K/V, `kv_mode 1` (`--kv-resident 32768`, la configurazione delle corse) | `QsaState::host.*` (RAM bloccata Portable, `layer.cpp:585`) | **`memcpy` host→host** delle stesse fasce, `env_threads()` fili, fette da 1 MiB con `progress_beat`; poi nella mappa di residenza della destinazione escono **solo i blocchi riscritti** `[c0/page, ⌈c1/page⌉)` (`page_table` a −1 e lo slot di ciascuno liberato: `slot_block` −1), perché gli slot VRAM possono tenere i byte vecchi di quei blocchi; il resto della mappa resta residente (un `kv_stream_reset` intero, come fa `conversation_snapshot.cpp:413` dopo aver riscritto una copia host intera, buttava a ogni turno il working set della scrittura: le prime finestre ricaricavano le selezioni dall'host sull'Oculink). Reset intero solo se l'intervallo è metà mappa o più, o ha più di `kResetAboveResident` (1024) blocchi residenti (primo turno, ripristino: lì la mappa era comunque vuota o da rifare) | niente PCIe per il K/V |
| righe pooled dell'indicizzatore | `idx_pooled` (device, `(max_cells/idx_block + 2) × idx_dim` float) | righe `[c0/idx_block, c1/idx_block]`, D2H+H2D | 16 MB a strato a 128K (S) |
| `idx_tail`, `idx_dead`, `idx_block_pos` | device, `conversation_state_sizes` → `tail/dead/block_pos` | D2H+H2D, solo con `running_state` | KB |
| GDN dei 21 strati | `gdn_state + (ord − gdn_ord0) × gdn_row_floats(g)` (ordinale globale, `is_qsa_layer`) | D2H+H2D a righe, solo con `running_state` | ~2 MiB a strato (S) |
| PLE, contatori per token, residuo | — | mai: la PLE è dello strato 1 (stadio 0); `step/pos` li riscrive il driver; `R` è scratch | |
| `ple_prev/ple_token` | `ss` | lo stadio 0 li avanza su `ss` (`Prefill::run`, `prefill.hpp:182`); lo stadio EP non ha la PLE | |

Vincoli controllati dal mover (errore, non testo diverso): stessi `kv_mode`, formato, `page_size`, `max_cells`,
`idx_pooled_rows` fra le due `QsaState`; `kv_mode 2` (ring) mai nel range; `c1 ≤ max_cells`; entrambe le sessioni
ferme (il chiamante ha sincronizzato `main_cs`/`ep.stream`, il mover sincronizza i propri stream prima di tornare).

Perché non `conversation_kv_save/restore` (`:293/:461`): copiano **tutte** le celle `[0, upto)` in un'immagine host
(a 128K: 1 GB in più di RAM a turno) e passano per `copy_nonblocking` sul device corrente; il mover copia solo le celle
nuove, host→host dove la verità è già in RAM, e non alloca. Perché non `conversation_checkpoint_copy` (`:431`): rifiuta
carve diverse (`:440-442`); qui le carve sono `[K, 48)` contro `[0, 48)` per costruzione.

## 6. Conversation cache, checkpoint, parcheggio, prefissi su disco, slot

Regola unica: **la sessione EP è scratch del prompt; la verità è sempre `ss`**. Quindi:

- **Parcheggio e ripristino** (`:11277-11327`, `:15727`): `stages` vuoto → immagine di `ss` sola, nessuna
  `stage_images`, nessuna `stage_parts`: come con `--peer-device` oggi. Al ripristino di un'altra conversazione (o
  della stessa da parcheggiata: `:15727`, prefisso prestato `:15701`, da disco `:15497-15520`, slot `:13615`,
  checkpoint montato `:15863`, `session_zero` `:15829`) **`sync.invalidate()`**: il turno dopo ricopia `[0, resume)`
  verso la 2080 Ti (§4, §8). Mai si legge uno stato EP vecchio: `valid_for` chiede la coincidenza degli id **e**
  nessuna riscrittura intermedia (ogni riscrittura invalida). Gli id uguali da soli non bastano: le celle di una
  risposta le scrive CUDA0, le stesse parole lette come prompt da un'altra conversazione le scriverebbe la 2080 Ti,
  byte diversi (§4).
- **Checkpoint a fine prompt / coda / radice** (`checkpoint_at`, `:11410`): dopo il passaggio in avanti → da `ss`,
  invariati.
- **Checkpoint a metà prompt** (`prompt_cache_every`, `:11623-11642`): a metà prompt le righe GDN e l'indicizzatore
  di `[K, 48)` in `ss` sono **vecchi** (lo stadio EP è avanti). Si fa come la divisione: `on_stage_chunk` su entrambi
  gli stadi (`:11666`, con `n_prompt_stages = 2`): parte 0 = `checkpoint_save(part, ss, g)` a `done` sul filo dello
  stadio 0 (sessione piena: le righe `[K, 48)` dentro sono vecchie), parte 1 = `checkpoint_save(part, ep.ss, g)` sul
  filo dello stadio EP (`OnDevice(ep.dev)`, stream sincronizzato per contratto del callback); in `checkpoint_at(done,
  &parts)` il ramo `:11425-11431` con EP fa `c = parts[0]; checkpoint_merge_tail(c, parts[1], ss, ep.ss, g)` invece
  di `stage_parts`. Il checkpoint risultante è **un checkpoint di sessione piena**, come lo vuole tutto il resto.
  **Non con `STRATA_CKPT_ASYNC=1` né `STRATA_CKPT_VRAM_SLOTS`** quando `--prompt-cache-every > 0` (§7): le due parti
  le salvano due fili su due schede, e `checkpoint_save` (`conversation_state.cpp:305-318`) tiene **uno** staging
  globale che libera e rialloca quando cambia la scheda, fuori dal suo mutex (protegge solo `pending` e la join): il
  filo del peer può liberare lo staging di CUDA0 sotto il D2H in volo dell'altra parte, o scriverci la sua GDN mentre
  l'altro la legge. La `conversation_checkpoint_join` di `checkpoint_merge_tail` arriva dopo e non lo copre. Lo stesso
  difetto c'è oggi con `--layer-split` e quelle variabili (da sistemare in `conversation_state.cpp`, non qui: spenta
  nulla cambia); la variabile non lo porta in `--peer-device`.
- **Prefissi su disco** (`--prompt-cache-file`): oggi rifiutati con un peer (`:3757`); **resta così** (spenta nulla
  cambia). Se un giorno il cancello si alza, l'immagine è quella di `ss` sola e il disegno già la rende corretta.
- **Slot (`--batch`)**: rifiutato (§7): le sessioni degli slot sarebbero anch'esse da carvare sulla 2080 Ti.
- **Sessioni-file**: rifiutate con un peer (`:14737`), invariato.
- **Prestito del prompt su CUDA0** (`:16218`) e `refill` (`:16191`): invariati (solo `pf_parts[0]`).
- **`adapt` del peer durante il prompt**: non avviene (gira fra le finestre di scrittura); le copie di `refill_` di
  un `adapt` **precedente** possono essere in volo a inizio richiesta → `peer.quiesce()` prima di ogni `sp.run`
  (pezzo B): `apply_pending(true)` + `cudaStreamSynchronize(stream_)` sotto `On(device_)`.
- **Mega-kernel e prompt**: nessuna finestra fused è aperta durante un prompt (`window_open_` falso); `has()` risponde
  da `res_` (`peer_experts.hpp:57`), la stessa tabella che `res_table()` espone.

## 7. Quando si spegne da sola (una riga, poi tutto come senza la variabile)

`ep_split::admit(Config)` a valle delle opzioni, prima di caricare. Riga: `strata: STRATA_EP_SPLIT_PREFILL off: <why>`.

| condizione | why |
|---|---|
| `--peer-device` assente | `needs --peer-device N` |
| `--layer-split` dato | `not with --layer-split (the EP prefill splits the prompt itself: STRATA_EP_SPLIT_LAYER=K)` |
| `--expert-cache-device1..3` | `not with a remote expert cache` |
| `--batch > 0` o `STRATA_BATCH_CONCURRENT_READ` | `not with --batch` |
| `--vision` | **ammesso**: lo stadio EP ha la sua tabella delle posizioni immagine sulla 2080 Ti (`ep.mrope`, caricata con i densi, ricaricata in `upload_mrope` a ogni richiesta con immagini, come gli stadi della divisione). La configurazione delle corse (`PEER`/`Gfus`) gira con `--vision`: un rifiuto avrebbe confrontato il motore con se stesso |
| `--prefill-head-alias` | `not with --prefill-head-alias` |
| `--prompt-cache-every > 0` con `STRATA_CKPT_ASYNC=1` o `STRATA_CKPT_VRAM_SLOTS` | `not with STRATA_CKPT_ASYNC / STRATA_CKPT_VRAM_SLOTS and --prompt-cache-every (…)` (§6) |
| `--pipeline-windows > 0` | `not with --pipeline-windows` |
| `STRATA_PREFILL_LAYER_MAJOR` | `not with STRATA_PREFILL_LAYER_MAJOR` |
| non `--serve` | `needs --serve` |
| pack non nativo o senza `--expert-profile` | `needs the native expert pack and --expert-profile (the peer tier)` |
| `STRATA_PRIMARY_DEVICE != 0` | `needs the primary on CUDA0` |
| `K < 2` o `K ≥ n_layers − 1` | `STRATA_EP_SPLIT_LAYER must be in [2, n_layers - 1)` |
| `--kv-grow`, `--elastic`, `--vram-elastic` | già rifiutati col peer (`:3699`, `:6182`): `not with the elastic K/V or cache` |

A runtime non si spegne mai «un po'»: un errore del mover o dello stadio EP **termina la richiesta con `ERR`**
(come un ripristino fallito, `:15732`), mai un testo diverso.

## 8. Stima dei tempi (S da M)

Numeri: 7 QSA e 21 GDN in `[20, 48)`; K/V INT8 1056 B a cella (`layer.hpp:318`), Q4 576; pooled `idx_dim` 128 float
ogni `idx_block` = 4 celle (`kv_stream.hpp:4`); GDN ~2 MiB a strato; righe finali hc·n_embd·4 = 40 KB a token
(`session.hpp:352`); link: 2080 Ti x16 13 GB/s, 3080 Ti Oculink 6,7 GB/s (M); `memcpy` host 8-10 GB/s a filo.

**Passaggio in avanti, primo turno (tutte le celle nuove)**

| contesto | K/V 7 strati INT8 | `kv_mode 1` (memcpy, 4 fili) | `kv_mode 0` (D2H 13 + H2D 6,7) | pooled 7 strati | GDN + piccoli | **totale mode 1 / mode 0** |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 30 MB | ~2 ms | ~7 ms | 2 MB, ~1 ms | 42 MiB, ~10 ms | **~15 / ~20 ms** |
| 32K | 242 MB | ~10 ms | ~55 ms | 29 MB, ~7 ms | ~10 ms | **~30 / ~75 ms** |
| 128K | 968 MB | ~35 ms | ~220 ms (a staffetta ~150) | 117 MB, ~27 ms | ~10 ms | **~75 / ~190-260 ms** |

Con Q4 il K/V pesa 0,55×. La stima «0,3 s a 128K» del compito è il tetto del `kv_mode 0`; la configurazione delle
corse (`--kv-resident 32768`) sta sotto 0,1 s. A 128K letti in ~58 s (2243 t/s), il passaggio vale **0,1-0,4 %**.

**Turno successivo**: indietro = GDN + piccoli (~10-15 ms) + le celle della risposta (256 token: ~2 MB, ~1 ms) ≈
**15 ms**; dopo un ripristino = tutto `[0, resume)` (la tabella sopra) + 15 ms. Avanti = celle nuove del turno.

**Righe del drafter** per chunk: 2048 × 40 KB = 80 MB → D2H x16 6 ms + H2D Oculink 12 ms ≈ **18 ms per chunk**,
sul filo dello stadio EP mentre lo stadio 0 legge il chunk dopo; un chunk da 2048 a 2243 t/s dura 0,91 s → ~2 %, in
sovrapposizione. Alternativa se pesa: `STRATA_SPLIT_MTP_BATCH` non si applica (altro device); le righe in bf16 (−50 %)
cambierebbero le bozze, non i token: non ora.

**Lettura**: la residenza dello stadio EP è **peggiore** di quella dello stadio 1 di `Vrif1`, per costruzione. Il suo
livello residente è la cache del peer, che contiene **solo le coppie che `xcache` non ha** (`peer_experts.cpp:314`),
distribuite su tutti i 48 strati: le coppie più calde di `[20, 48)` stanno in `xcache` su CUDA0 e lo stadio non le può
leggere, e solo ~28/48 degli slot del peer cadono in `[20, 48)`. Con ~8200-8700 slot (§2) ne restano **~5000** per
`[20, 48)`, senza le coppie di testa del profilo; `Vrif1` ne ha ~9090 con le coppie di testa. Per ogni chunk sopra
`stream_all_min` (1024) lo stadio EP manda in streaming sul x16 tutti gli esperti non residenti: ~14336 − 5000 =
**~9300 contro ~5250** di `Vrif1`, 1,8× i byte per chunk (a 13 GB/s, ~1,2 MiB a esperto: ~0,85 s per chunk da 7680
contro ~0,5; il chunk a 2243 t/s dura 3,4 s, quindi dentro, ma lo stadio EP è già il più lento). Lo stadio 0 ha ~1070
dei 2561 slot per `[0, 20)` (2119 in prestito al prompt: M in `Vpeer`) ma chiede al peer la maggioranza (`set_peer`, in
concorrenza con lo stadio EP sulla 2080 Ti). La pipeline va col più lento: lo stadio EP (28 strati, 2080 Ti: oggi 0,45
contro 0,325 ms a strato in scrittura, M) **più** lo streaming in più. Il vincolo −10 % su 900/1850/2210 è **a rischio
per costruzione, non provato**: la corsa 4a del PIANO (serie, `--layer-split 20`, cache forzate a ~1070 / **~5000**
slot per lo stadio 1 **senza le coppie di testa**, tre bersagli) la decide prima di compilare questo, e resta lavoro
della chat principale. Il passaggio (−0,4 %) e le righe del drafter (−2 % non sovrapposte, ~0 in sovrapposizione) sono
il resto.

**Scrittura**: identica al mega-kernel: nessun kernel, nodo o attesa in più nella finestra. La VRAM della 3080 Ti
non cambia (§2).

## 9. Pezzi (file disgiunti, solo scrittura; compila, prova e salva la chat principale)

| pezzo | file | simboli del contratto |
|---|---|---|
| **A wiring** | `src/program/generate.cpp` | `ep_split::Config/admit`, `env_*`, `reserve_add_mib`, `StateMover`, `Sync`, `checkpoint_merge_tail`, `line_*`; `PeerExperts::cache/res_table/quiesce` (B) |
| **B peer** | `include/strata/core/peer_experts.hpp`, `src/core/peer_experts.cpp` | `PeerExperts::cache()`, `res_table()`, `quiesce()`, `slot_guard` all'apertura |
| **C mover** | `src/core/ep_split.cpp` (nuovo) | tutto `ep_split.hpp` tranne quel che A chiama: `env_*`, `admit`, `StateMover`, `Sync`, `checkpoint_merge_tail`, `reserve_add_mib`, `line_*` |
| **D banco** | `tools/ep_split_test.cpp` (nuovo), `CMakeLists.txt` | `StateMover`, `Sync`, `checkpoint_merge_tail`, `gdn_row_floats` contro `session_init` |

L'ordine non conta per scrivere; per compilare prima C (il modulo), poi B, poi A, D in parallelo a A.

## 10. Rischi

- **La lettura perde più del 10 %**, per lo stadio 0 affamato (1070 slot per 20 strati) **e** per lo stadio EP con
  ~5000 slot senza le coppie di testa (§8): il rischio principale e la ragione della corsa 4a. Leve senza codice:
  `STRATA_PEER_HOT`, `--peer-prefill-rows`, `--vram-reserve-mib`; con codice: il prestito EP dalla coda di `xcache`
  oggi prende 2119 slot, un chunk più piccolo ne prende meno; oppure un ordine di riempimento del peer che privilegi
  `[20, 48)` (cambia la scrittura: non ora).
- **VRAM della 2080 Ti, e la scrittura non è più quella del mega-kernel a parità di cache**: densi + sessione
  (~2,35 GiB) + riserva dei buffer (1,2-2,0 GiB + 512 MiB) escono tutti dalla cache del peer → **~8200-8700 slot** da
  10971 (§2). Sotto ~8500 il mega-kernel paga più miss CPU (+0,3 voci a strato-finestra, S in `PIPELINE3` §3e), e le
  coppie uscite le calcola la CPU con altri arrotondamenti: **il testo di `Xeps` differisce da `Xfus` anche con
  `--adapt-swaps 0`**. Il confronto giusto è con un `Xfus` a cache uguale: la riga `strata ep-split: CUDA1: the stage
  holds …` stampa il `--peer-reserve-mib` che la dà (§12). `STRATA_EP_SPLIT_RESERVE_MIB` forza la sola riserva.
- **La riserva è dimensionata al chunk massimo dello scan** (`o.prefill_chunk` = `prefill_auto_max` a `peer.open`:
  8192/16384/32768 con `--prefill auto`), lo stadio alloca poi col chunk scelto (7680, anello da 96 slot): la differenza
  resta VRAM libera sulla 2080 Ti per tutto il processo, slot tolti al server. Piccola a `auto` (8192 → 7680), grande
  a `auto:32768`. La riga `strata ep-split: CUDA1: prompt buffers … MiB free on the card after them` la misura e dice
  lo `STRATA_EP_SPLIT_RESERVE_MIB` che restituisce il resto al server (secondo avvio). Il margine della riserva è
  512 MiB, lo stesso `kHeadroom` di `own_fits`: così il peer passa `own_fits` per ogni chunk ≤ massimo per costruzione,
  qualunque `--peer-reserve-mib`, e il chunk condiviso non scende per colpa del peer (con 256 MiB poteva).
- **Contesa sulla 2080 Ti**: `PeerPrefill` dello stadio 0 (`set_peer`) e lo stadio EP sulla stessa scheda: due
  stream, VRAM separata; il tempo si somma se i kernel non si sovrappongono. Misurabile con `--peer-prefill-rows 0`
  (stadio 0 senza peer) contro il default.
- **Eviction dei blocchi riscritti dopo il memcpy**: dimenticata, gli slot VRAM della destinazione tengono i blocchi
  vecchi → attenzione su celle sbagliate (testo diverso, non un errore). Il banco D copre il blocco riscritto: scrive
  un blocco residente, passa, legge (**non** copre ancora che un blocco residente fuori dall'intervallo resti
  residente: da aggiungere al banco). L'eviction è per blocco (§5): un reset intero a ogni turno costava alle prime
  finestre di scrittura le selezioni ricaricate dall'host sull'Oculink, lavoro che il mega-kernel di oggi non ha.
- **Formule locali duplicate**: `gdn_row_floats` è la copia di `session.cpp:38`; i byte per cella vengono da
  `conversation_kv_part_sizes`, non da una formula. Il banco D confronta con `session_bytes/session_init` e con
  `conversation_kv_save`.
- **Sincronizzazione**: il mover copia con sessioni ferme. Lo stadio EP finisce sul suo filo (`drain_pipeline`,
  `prefill.cpp:7280`) prima che `sp.run` torni; `main_cs` lo sincronizza A prima del passaggio; `ep.stream` idem.
- **Checkpoint a metà prompt**: la parte 0 è della sessione piena (più grande: 36 righe GDN + 12 QSA), la fusione
  sovrascrive `[K, 48)`; un `done` senza entrambe le parti salta il checkpoint come oggi (`:11640-11642`).
- **`MMQ_DIRECT` sulla cache del peer**: il prompt path legge un esperto residente nel suo slot solo se lo slot ha
  la guardia (`ExpertCache::set_slot_guard`, `prefill.hpp:172`); la cache del peer è aperta senza. Pezzo B la chiede
  come le cache degli stadi; se non si può, lo stadio EP fa il gather (più lento, stessi bit).
- **Drafter**: le righe del chunk passano per host (18 ms per chunk, S); un errore di copia termina la richiesta.
- **`Sync` troppo prudente**: ogni ripristino ricopia `[0, resume)` (≤ 0,1-0,3 s a 128K): corretto per costruzione,
  costo visibile nella riga `ep-split` di fine richiesta.

## 11. Non provato, e come si prova

Scritto e non compilato: compilato no, installato no, provato no. Punti dove la compilazione può fermarsi: le firme
`Prefill::init` con `const ExpertCache*` della cache del peer; `checkpoint_at`'s branch a parti; i `#include` di
`ep_split.hpp` dentro `generate.cpp` (ordine con `conversation_cache.hpp`).

## 12. Integrazione dei pezzi (3/11, letto il diff di tutti e quattro; compilato no)

Quadrature fatte sul diff (nessuna firma cambiata): `Prefill::init(const ExpertCache*, const int32_t*)` prende
`&peer.cache()` / `peer.res_table()`; `quiesce(std::string&)` è come A la chiama; `strata_engine` ha `ep_split.cpp`;
`ep_split_test` linka `strata_engine strata_kernels strata_core`. Correzioni dell'integrazione:

- **Cancellazione a metà parte** (`generate.cpp`, ramo `!sp_ok && stop_req`): **solo se la parte era batched**
  (`!win`), prima di `invalidate` le celle `[at, pp_reached)` dello stadio EP passano a `ss` (`ep_to_primary`): i
  checkpoint periodici fatti in quella parte restano in `checks` (posizioni ≤ `pp_reached`) e una ripresa da uno di
  essi legge il K/V di `[K, 48)` da `ss`, che senza il passaggio sarebbe fermo ad `at` (testo sbagliato in silenzio).
  Con una parte letta **a finestre** lo stadio EP non ha calcolato nulla (le celle le ha scritte `ver` su `ss`, 48
  strati) e il passaggio avrebbe messo in `ss` byte vecchi dello stadio per `[⌊at⌋, pp_reached)` mentre la
  conversazione resta viva a `pp_reached` (`pp_exact` vale dopo `read_windows`, `:19344`): guardia `!win` come nel
  ramo di successo (`:17418`). Il ramo `nf_hit` non passa nulla.
- **Quota CPU del prompt** (`arm_cpu_share`, `generate.cpp:9853`): con la variabile **non** si accende da sola sotto
  1024 token, come non si accende con la divisione a strati (`!multi_gpu` era vero con EP e la armava). Con la quota
  accesa lo stadio EP sceglierebbe gli esperti per la CPU fra quelli che manderebbe in streaming, insieme che dipende
  dalla residenza del peer, che `adapt` cambia fra una richiesta e l'altra: bit diversi da `Vrif1` e diversi da
  richiesta a richiesta. `STRATA_PREFILL_CPU_SHARE` dato a mano vale per entrambi gli stadi, come per la divisione.
- **`--vision`** ammesso (§7): tabella `mrope` dello stadio sulla 2080 Ti, ricaricata in `upload_mrope`.
- **`STRATA_CKPT_ASYNC` / `STRATA_CKPT_VRAM_SLOTS` con `--prompt-cache-every`** rifiutati (§6): la corsa sullo staging
  globale di `checkpoint_save` fra i due fili non si copre con la join della fusione.
- **`checkpoint_merge_tail`** chiama `conversation_checkpoint_join(nullptr)` prima di leggere i byte: con
  `STRATA_CKPT_ASYNC` le righe GDN di una parte atterrano su un filo dopo `conversation_checkpoint_save`.
- **Ripristino da file di sessione** (`:15099`): `invalidate` anche lì (oggi irraggiungibile col peer, `:14737`).
- **`STRATA_PREFILL_LAYER_MAJOR`** nell'ammissione si legge come la legge `prefill.cpp` (`split_env_on`): `=0` non
  rifiuta.

Da sapere per il confronto (non è codice): col peer `peer_portable()` è vero (`generate.cpp:3711`) e la catena
layer-major di `Prefill` declina («a peer GPU», `prefill.cpp:6068`), quindi con la variabile accesa la lettura a stadi
è **chunk-major** (il passaggio di mano `:5728`). La divisione a strati senza peer (`Vrif1`) ha invece il layer-major
acceso per default (`sm120_series.hpp:112`, 2 stadi, CUDA, non 026). «Lettura identica alla divisione» vale a parità di
modo: `Vrif1` con `STRATA_PREFILL_LAYER_MAJOR=0`, oppure si accetta la differenza e si misura.

Prova (chat principale, `PIPELINE3_PIANO` §5): build `build_lock_k.py`; coda con configurazione `PEER` +
`STRATA_PEER_FUSED=1` e la coppia `Xeps{n}` (`STRATA_EP_SPLIT_PREFILL=1`) / `Xfus{n}` (spenta, stessa build);
confronto lettura con `Xrif{n}` (serie, divisione 20). Criterio: lettura ≥ 90 % di 900/1850/2210; scrittura
22,7-22,9 ms e testo **identico a `Xfus'`** a cache fissa (`--adapt-swaps 0`) sul primo turno, dove `Xfus'` è `Xfus`
con il `--peer-reserve-mib` stampato dalla riga `strata ep-split: CUDA1: the stage holds … (the same tier without
the variable: --peer-reserve-mib N)`, cioè **la stessa cache del peer**: contro `Xfus` a riserva 3000 il testo
differisce per costruzione (§10: ~2300 slot in meno, coppie passate alla CPU). `Xeps` con `--vision` come le corse
(ammesso, §7), `STRATA_PREFILL_CPU_SHARE` non impostato in nessuna delle due. A più turni la ripresa del prefisso
(tempo del secondo turno come oggi) e testo identico a `Xfus'` **a parità di stato**; nessun `strata.exe` rimasto
acceso; la riga `ep-split` con passaggi e ms; la riga `prompt buffers … free on the card after them` per il secondo
avvio con `STRATA_EP_SPLIT_RESERVE_MIB`. Banco D: `ep_split_test` sulla stessa scheda e, se ce ne sono
due, fra due schede: byte identici in entrambe le direzioni, `kv_mode 0` e 1, fusione dei checkpoint contro un
checkpoint della sessione piena a stato uguale.

## 13. Tappa 4b: i buffer del prompt in prestito dal tier del peer, e la camminata per strato (3/11, letto `1fe8e46d`)

Scritto dopo le misure del 3/11 (`corse_2080ti\20261101_tre_schede\corse\Z4*`, build `1fe8e46d`). Righe lette su
`1fe8e46d` (quelle del §1-§12 sono di `8cb752a4`: alcune sono scivolate). Convenzione come sopra: **M** misurato,
**S** stimato.

### 13.0 In tre righe

1. **La divisione a strati di oggi NON legge per strato.** In tutte le corse del banco (oltre 160), Z4rif comprese, la
   camminata layer-major è accesa di default e poi **declinata a ogni prompt**: `strata prefill: layer-major off for
   this prompt: no VRAM for a super-chunk (596 MiB free)` (M, `Z4rif1/engine.log`). Nessun log ha mai la riga di
   riepilogo `strata prefill: layer-major (layers …)`. I 914/1898/2240 di Z4rif sono **chunk-major a 8192**. Quindi
   la causa (1) del compito («col peer il prompt va a blocchi, la divisione usa il layer-major») non spiega il
   divario: entrambe leggono a blocchi. La lettura per strato resta una leva **mai misurata su questo banco**, con
   bit diversi dal chunk-major (§13.3): si fa, ma dietro la sua variabile e con il suo riferimento.
2. **Il prestito dal tier del peer è il pezzo sicuro.** La tappa 4 perde 4928 MiB di riserva fissa per i buffer del
   prompt (M: `3000 -> 7928 MiB`) più 2692 MiB di densi+sessione: il tier scende da 10971 a 6372 slot (M) e la CPU
   calcola 2,2-5,0 esperti a strato-finestra contro 0,9-1,5 (M): ecco il −20/−30 % in scrittura. Con il prestito la
   riserva torna a 3000 e il tier a **~9350 slot** (S: 10971 − 2692 MiB / 1,66 MiB): scrittura stimata 85-96 t/s
   contro 94-100 del mega-kernel da solo e 61-87 della tappa 4.
3. **La sessione degli strati 20-47 non vale la pena di stringerla**: 604 MiB in tutto (M), di cui 231 MiB di K/V
   residente; scendere a 8192 celle libera ~105 slot (1 % del tier) e costa alla lettura lo streaming del K/V a ogni
   blocco (§13.4). Si lascia com'è.

### 13.1 Che cosa dicono i log (M) e dove sta nel codice

| cosa | Z4rif (divisione 20) | Z4fus (mega-kernel solo) | Z4ep (tappa 4) |
|---|---|---|---|
| lettura 4K / 32K / 128K t/s | 877-938 / 1892-1910 / 2208-2256 | 948-993 / 1373-1387 / 1448-1465 | 822-872 / 1525-1540 / 1703-1727 |
| scrittura t/s | 99-109 / 72-112 / 104-105 | 89-100 / 99-103 / 95-102 | 61-73 / 77-79 / 81-87 |
| chunk | 8192 | 7680 | 7168 |
| CUDA0: cache / prestito | 3556 / 2559 slot | 2559 / 2119 | 2365 / 2000 (−144 slot: `ep_R_dev0` 320 MiB, `generate.cpp:7331-7337`) |
| 2080 Ti: esperti residenti | 9088 slot, tutti di `[20, 48)` (prestito 2025 nel prompt → 7063) | tier 10971, 48 strati | tier 6372, 48 strati (≈ 3700 in `[20, 48)`) |
| 2080 Ti: riserva | — | 3000 MiB (2190 liberi dopo i buffer di `set_peer`) | 7928 MiB = 3000 + 4928 (buffer propri 3390 MiB a 7168 + 512; sizing a 8192) |
| 2080 Ti: lo stadio | densi `[20,48)` + sessione + testa; 252 MiB liberi a fine avvio | — | 2692 MiB (densi+flussi 2088, sessione 604) |
| layer-major | acceso di default, **declinato**: «no VRAM for a super-chunk (596 MiB free)» | n/a (un solo stadio) | **declinato**: «a peer GPU» |
| CPU a strato-finestra (scrittura) | 0,4-1,6 esperti | 0,9-1,5 | 2,2-5,0 |

Il codice che decide tutto questo:

- **Camminata per strato**: `prefill.cpp:7384-7398` (`run`: `prefill_layer_major_resolve` → `lm_prepare_chain` →
  `run_layer_major`), default `sm120_series.hpp:112-119` (acceso con 2+ stadi CUDA, «KL-checked, not bitwise»).
  `lm_prepare` (`prefill.cpp:6047-6207`): declina con `m.pp || helper_ || core::peer_portable()` (`:6068`, «a peer
  GPU»); vuole almeno **due chunk** (`min_s = 2·T`, `:6117`), punta a S = 32768 (`kLmDefaultOnTokens`, `:6026`) e
  **dimensiona i buffer del super-chunk dalla VRAM libera della scheda** (`cudaMemGetInfo` meno 512 MiB di margine
  quando è acceso di default, `:6132-6134`, `lm_margin_bytes` `:6017-6023`), poi li alloca con `cudaMalloc`
  (`:6176-6181`) e li libera a fine prompt (`lm_release`). Byte per token in modo host (`dev_bytes`, `:6127-6131`,
  costanti `:173`: N 2560, HC 4, D 10240, K 10): (2N + HC + 2K)·4 + N·2 = **25,7 KB**, più le tabelle (~0,2 KB) e
  `R2` = T·D·4 (40 KB a token di chunk). A T 8192 il minimo è 2·8192·25,9 KB + 8192·40 KB ≈ **760 MB + 512 MiB di
  margine = 1,27 GiB**: CUDA0 ne ha 596 MiB (M), lo stadio 1 di Z4rif 252 (M). Per questo non parte mai. Gli stadi
  dopo il primo tengono le righe in RAM bloccata (`mode = 2`, `:6142`, `R_host` S·D·4 = 1,34 GB a 32768). Gli esperti
  li legge come il chunk-major: residenti via `m.host_res`/`m.cache->device_slot` (`:6456`, `:6510`, `:6687`, gather
  → nessuna guardia MMQ_DIRECT richiesta), gli altri dall'anello di staging (`m.stage_dev`) riempito dallo `Stager`.
  Con `peer_portable()` vero `fused_ring()` è falso (`:300`) → buffer MMQ a misura piena → i requisiti di `lm_run`
  (`grp_gu/grp_d/Xq/H/Hq`, `:6096`; niente `fused_bufs`, `:6114`) sono soddisfatti: il rifiuto `:6068` è prudenziale,
  non tecnico, **tranne** per uno stadio che ha un `PeerPrefill` (`m.pp`, `set_peer` `:2269`): quello non ha forma
  layer-major.
- **Prestiti del prompt path** (`generate.cpp`): `PfPart` `:9921-9932` (cache, sessione, prompt path, device, strati
  `[lb, le)` che può prestare, `first`, `lent`); un `PfPart` per CUDA0 e uno per stadio (`:10001-10004`); `fits_one`
  `:10010-10016` (pavimento 128 slot, tetto `kAutoLendPct` nello scan auto); la scelta del chunk «il più grande che
  OGNI partecipante può prestare» (`scan`, `:10072-10095`); `first = slots − part_slots(chunk)` `:10188-10191`;
  `init_prompt_paths` `:10343-10383` passa `sb/sbb` (`device_slot(first)`, `part_bytes`) a `Prefill::init`; lo stadio
  EP oggi riceve `nullptr, 0` (`:10352-10353`, buffer propri). Per richiesta: `lend` `:16522-16605` (per ogni parte:
  `relayout(want_p, device_slot(first), part_bytes)`, poi **le coppie dei suoi strati con `host_res[i] >= first`**
  vanno in `p.lent` e la tabella a `kNotResident`, `res_upload`), `refill_issue/refill_wait/refill` `:16454-16517`
  (dall'arena `srcp->blob_stable` nello **stesso slot** con `fill_slot_queued`, tabella ripristinata, `res_upload`).
  Chiamate: `lend` prima di ogni parte a blocchi `:17472`, `refill` prima delle finestre `:17464` e a fine prompt
  `:17648`. `own_fits` `:10390-10412` salta chi ha un prestito e controlla lo stadio EP contro la sua VRAM libera
  (`:10404-10410`).
- **Riserva della tappa 4**: `generate.cpp:8135-8171` (`bytes_needed_owned` al chunk massimo dello scan →
  `ep_split::reserve_add_mib` = byte + 512 MiB, `ep_split.cpp:662-669`, `kReserveHeadroomBytes` `ep_split.hpp:54`) →
  `peer.open(..., o.peer_reserve_mib + ep_reserve_add, ...)`. Il tier si riempie da ciò che è libero meno la riserva
  (`peer_experts.cpp:305-322`), **in ordine di rango** (slot 0 la coppia più calda, la coda le più fredde), con slot a
  misura di strato (`open_sized`, `:328`, quindi `slot_offsets()` esiste e `split_lend_slots` funziona come per gli
  stadi). `adapt` (`:543-598`) scambia le vittime meno usate per strato sullo stream `refill_`, `apply_pending`
  (`:600-615`) pubblica in `res_`; `quiesce` (`:622-633`) fa atterrare tutto. `has()` fuori finestra risponde da
  `res_` (`peer_experts.hpp:57-60`); dentro la finestra da `own_win_`, costruita a `fused_begin` da `res_` e
  `host_res`; `d_slot_ptr_` (`:228`) è slot → indirizzo e non cambia mai.
- **Stadio EP**: `generate.cpp:5411-5457` (densi `[K, 48)` con `trim`, stream, mrope), sessione `:6361-6380`
  (`session_bytes/init(g, o.max_context, K, …)`), `set_peer` dello stadio 0 `:10509-10516` (`rows = chunk·K/2`,
  `--peer-prefill-rows 0` lo spegne), `ep_to_primary` `:17413-17431`, `peer.quiesce` prima del passaggio in avanti
  `:16740` (dentro la lettura, **dopo** il `lend`: va anticipato, §13.2).

### 13.2 Disegno (b): il tier del peer presta la coda al prompt path dello stadio EP

Il prestito esiste già, parametrizzato per partecipante: **lo stadio EP diventa un `PfPart` in più**, con il tier
del peer come cache e `res_` del peer come tabella. Niente riserva per i buffer del prompt: il tier è quello del
mega-kernel da solo meno densi e sessione dello stadio.

1. **Avvio** (`generate.cpp`): `ep_reserve_add = 0` (la riga `line_on` dice «prompt buffers borrowed from the tier»;
   `STRATA_EP_SPLIT_RESERVE_MIB` resta per forzare una riserva in più, default 0). Nuovo knob
   `STRATA_EP_SPLIT_LOAN` (default 1 con la variabile accesa; `0` = tappa 4 com'è oggi, l'A/B). `peer.open` a
   riserva `o.peer_reserve_mib` come senza la variabile.
2. **Parte in più**: `pf_parts.push_back({&peer.cache_mut(), &ep.ss, &ep.sp, ep.dev, 0, g.n_layers, …, res =
   peer.res_table_mut()})`: `PfPart` prende un campo `int32_t* res` (default `host_res.data()`), usato da `lend` e
   `refill_issue` al posto di `host_res`. `lb = 0, le = n_layers` perché il tier copre tutti gli strati: una coppia
   di qualunque strato che sta in uno slot prestato **esce dalla tabella** (`has()` → falso, `peer_experts.hpp:59`),
   così il `PeerPrefill` dello stadio 0 non la calcola (la manda in streaming, come fa per le coppie che il peer non
   ha) e lo stadio EP la manda in streaming dalla RAM sul x16. `res_upload` non tocca il peer (il suo `d_own_` nasce
   a ogni `fused_begin` da `res_`, dopo il `refill`).
3. **Scan del chunk**: `fits_one` e `ring_room` valgono invariati (`p.cache != &xcache`, `rc_floor 0`): il tier presta
   ~2000 dei ~9350 slot (21 % < `kAutoLendPct`), quindi **CUDA0 resta il partecipante che decide il chunk**, come
   oggi. `own_fits` salta lo stadio EP quando ha il prestito (`:10404`: `if (ep_on && !loan_ep)`).
4. **`init_prompt_paths`**: `ep.sp.init(..., sb = peer.cache().device_slot(first), sbb = part_bytes, ...)` al posto di
   `nullptr, 0` (`:10352`). Il resto identico.
5. **Per richiesta**: `lend` chiama **prima** `peer.quiesce(e)` per la parte del peer (le copie di `refill_` di un
   `adapt` precedente devono essere atterrate prima che il `relayout` scriva negli slot; oggi il `quiesce` sta a
   `:16740`, dopo il `lend`: va spostato nel `lend`, e `:16740` resta innocuo), poi `relayout` sotto `OnDevice(ep.dev)`
   e la marcatura in `res_`. `refill_issue` sul device del peer con `p.cache->fill_slot_queued` (il `ExpertCache` del
   peer, `expert_cache.hpp:228`; la coda è della cache, indipendente da `stream_/refill_` del peer) e `sync_queued`;
   su Windows `srcp->release` come per le altre parti. I `refill` ai punti `:17464` e `:17648` coprono ogni uscita
   della richiesta, errori compresi, perché la parte del peer segue la stessa lista. `adapt`, `apply_pending`,
   `fused_begin` girano solo fra le finestre, cioè **dopo** il `refill`: la coda è già tornata.
6. **Guardia nel peer** (pezzo B): `lend_guard(bool)`; con la guardia alzata `adapt()` torna subito (`:545`-stile),
   `apply_pending` non ha nulla in `pending_` (il `quiesce` l'ha svuotato), `fused_begin` rifiuta con errore (mai
   nel disegno). Accessori: `ExpertCache& cache_mut()`, `int32_t* res_table_mut()`, `int64_t lent_slots()` per la
   riga di log.
7. **Riga di log**: `strata serve:   CUDA1 prompt path borrows N of its M tier slots (x GiB), K of them pairs of
   layers [0, 20) the primary's chunk-major read would have had computed there` — la K misura il costo del punto 2.

**Che cosa cambia nei bit.** La scrittura: il tier ha più coppie (9350 contro 6372) e il mega-kernel ne serve di
più; lo stesso avvertimento del §10 vale verso `Xfus` a cache diversa (il confronto giusto è `Xfus'` con
`--peer-reserve-mib 3000 + 2692`, cioè la riga `the stage holds …` continua a stamparlo). La lettura: lo stadio EP
vede nel tier (durante il prompt, prestito escluso) ~7350·28/48 ≈ **4300** coppie di `[20, 48)` contro ~3700 della
tappa 4 (S: tier più grande, coda prestata), lo stadio 0 trova nel peer ~3060 coppie di `[0, 20)` contro ~2650 (S)
ma ne perde ~830 nella coda prestata (S: 2000·20/48), che vanno in streaming sull'Oculink: +1,0 GB a blocco a
6,7 GB/s ≈ +0,15 s su un blocco da 7168 che dura ~4,2 s (M a 1716 t/s): **≤ 4 %**, in sovrapposizione. Lettura
stimata: **come la tappa 4 ± 3 %** (830-880 / 1500-1580 / 1680-1780). Il divario verso Z4rif resta e ha altre
cause (M): la residenza di `[20, 48)` (7063 slot nel prompt contro ~4300: ~2700 esperti in più in streaming a
blocco, ≈ 3,3 GB a 13 GB/s ≈ +0,25 s ≈ 6 %), il chunk (8192 contro 7168: §13.2bis), e la contesa sulla 2080 Ti
(`PeerPrefill` dello stadio 0 con il suo 35 % di streaming + lo stadio EP sullo stesso x16; si misura con
`--peer-prefill-rows 0`, nessun codice). Il criterio −10 % su 900/1850/2210 **non lo decide questo pezzo**.

**13.2bis (facoltativo, pezzo A2): le righe del drafter nel prestito di CUDA0.** `ep_R_dev0` (320 MiB, allocato
dopo la cache, `generate.cpp:7331-7337`) toglie 144 slot a `xcache` **e** fa scendere il chunk da 7680 a 7168 (M:
Z4fus/Z4ep). Serve solo durante il prompt: può stare **in coda alla regione prestata di CUDA0** (gli ultimi
`T·hc·n_embd·4` byte della coda, `device_slot(slots) − rows_bytes`), con `part_slots` della parte 0 che conta
`bytes_needed(c) + rows_bytes(c)` e `relayout` che riceve `part_bytes − rows_bytes`. Il prestito di CUDA0 sale di
~185 slot (2119 + 185 = 2304 ≤ 2559 − 128), `xcache` torna a 2559 e il chunk a 7680; `ep_rows_bytes` esce dalla
riserva del sizing automatico. Il puntatore si ricalcola a ogni `lend` (il `first` cambia per richiesta). Decide la
chat principale: è un guadagno certo (M) ma allarga A.

### 13.3 Disegno (a): la camminata per strato con lo stadio EP (`STRATA_EP_SPLIT_LM=1`, spenta di default)

Tre cose la bloccano oggi, nessuna è un kernel: il rifiuto `peer_portable()` (`prefill.cpp:6068`), il `PeerPrefill`
dello stadio 0 (`m.pp`, senza forma layer-major), la VRAM libera per i buffer del super-chunk (596 MiB su CUDA0:
mai abbastanza, anche per Z4rif). Il disegno:

- **C1 (`prefill.cpp`)**: `lm_prepare` declina con `m.pp != nullptr`, con `helper_` e con `peer_portable()` **solo se
  nessuno ha dato allo stadio una regione** (`set_lm_region`, cioè senza `STRATA_EP_SPLIT_LM=1`): testo «a peer GPU»
  in tutti e tre i casi, come oggi. Con una regione il `peer_portable()` non declina più (il path è MMQ pieno,
  `fused_ring()` falso, `:300`: i buffer che `lm_run` vuole ci sono) e un `m.pp` residuo dice «this stage computes part
  of its experts on a peer GPU (set_peer)». Il flag `cudaHostAllocPortable` non c'entra con la camminata. **Nessun
  cambio con le variabili spente**: la regione è nulla, quindi anche un peer senza `set_peer` (`--peer-prefill-rows 0`,
  o un `set_peer` fallito) con `STRATA_PREFILL_LAYER_MAJOR=1` esplicito resta chunk-major con la stessa riga di
  prima (la prima stesura di C1 toglieva il rifiuto per tutti: lo avrebbe cambiato).
- **C2 (`prefill.cpp/.hpp`)**: i buffer del super-chunk da una **regione data dal chiamante** invece che da
  `cudaMalloc` sulla VRAM libera: `static uint64_t lm_bytes_needed(const ModelGeometry&, int64_t S, int64_t T)` (la
  formula di `dev_bytes(s, r_dev = false)`, modo host), `void set_lm_region(void* base, uint64_t bytes)` (null:
  oggi). In `lm_prepare` con regione: `room = bytes` (niente `cudaMemGetInfo`, niente margine), `mode` forzato a host
  (`r_dev = false`, le righe in `R_host` come già per gli stadi dopo il primo), `lb.dev = base`, `lm_release` non
  libera. `R_host`, `idsw_host`, `tab_host` restano in RAM bloccata (`host_grow`, tenuti fra un prompt e l'altro).
  `lm_prepare_chain` invariato. Generico: serve anche alla divisione a strati senza peer, che oggi non parte mai.
- **A-LM (`generate.cpp`, con `STRATA_EP_SPLIT_LM=1`)**: (i) `sp.set_peer` **non** si chiama (lo stadio 0 legge le
  coppie di `[0, 20)` che `xcache` non ha dalla RAM sull'Oculink, **una volta per super-chunk** invece che per blocco:
  è il punto della camminata; `--peer-prefill-rows 0` di oggi fa lo stesso senza la camminata); (ii) il chunk si
  ferma a **4096** (`o.prefill_auto_cap`-stile): la regione prestata deve tenere buffer del chunk + super-chunk:
  a T 4096 / S 32768 ≈ 1,8 GiB + 0,16 (anello) + 1,02 GiB ≈ **3,0 GiB ≤ 4,0 GiB** prestabili da CUDA0 (2431 slot ×
  1,66), a T 7168 non entra (3,25 + 1,14 = 4,4 GiB); (iii) `part_slots(p, c)` conta `bytes_needed(c) + lm_bytes_needed(
  min(S_target, n_prompt↑c), c)` per **ogni** parte (CUDA0 e peer; con la divisione a strati anche gli stadi: così
  Z4rif può finalmente girare layer-major ed essere il riferimento giusto); (iv) dopo ogni `lend`:
  `p.sp->set_lm_region(device_slot(first) + bytes_needed(c), lm_bytes)` e `relayout(c, device_slot(first),
  bytes_needed(c))` (misura esatta, non tutta la coda). Un prompt sotto 2·T (4K) legge chunk-major in un blocco da
  4096, come oggi; 32K = 1 super-chunk, 128K = 4 super-chunk pari (`lm_even`).
- **Bit**: la camminata somma gli esperti per id, non nell'ordine k del router (`prefill.cpp:5981-5983`,
  `sm120_series.hpp:112`: «KL-checked, not bitwise»). **Il testo differisce dal chunk-major per costruzione**: la
  regola del proprietario «testo identico» vale solo LM contro LM, con la divisione a strati anch'essa layer-major
  (iii). Va detto nel confronto, non nascosto.
- **Stima** (S, mai misurato qui): la camminata divide per S/T = 8 lo streaming a blocco dello stadio 0
  (≈ 6500-9200 esperti × 1,2 MiB ≈ 8-11 GB sull'Oculink a 6,7 GB/s ≈ 1,2-1,7 s di un blocco da ~4,2 s) e dello
  stadio EP (≈ 10000 × 1,2 MiB sul x16 ≈ 0,9 s). Se quello streaming oggi non si sovrappone al calcolo, il
  guadagno massimo è ~25-40 % sulla lettura a 32K/128K; se si sovrappone, ~0. Su 2× A4000 (solo streaming) pagava;
  qui lo dice solo la corsa. Costo certo: il chunk più piccolo (4096) rende più lento il chunk-major di ripiego e
  il 4K; e il sub-chunk più piccolo fa più passaggi di mano (`R_host`) per super-chunk.

### 13.4 La sessione degli strati 20-47 sulla 2080 Ti: lasciarla com'è

M: 604 MiB (`18.69 → 18.10 GiB free`). Dentro (S da `layer.hpp:318`, `kv_stream.hpp:4`): K/V residente 7 QSA ×
32768 celle × 1056 B = 231 MiB; righe pooled dell'indicizzatore 7 × (262144/4 + 2) × 128 × 4 = 224 MiB (a
`max_context`, non si tocca); GDN 21 × 2 MiB = 42; code, rope, passi il resto. Scendere a 8192 celle residenti per
la sola sessione EP: −173 MiB ≈ **105 slot (1,1 % del tier)**, cioè ~0,1 esperti CPU a strato-finestra: non
misurabile. Costo: sopra 8K la selezione dello stadio pesca le pagine K/V dalla copia host a ogni blocco (la mappa
di `kv_mode 1`), a 128K ≈ 968 MB a blocco sul **x16 che già porta gli esperti**: ~75 ms, 2 % del blocco, in contesa.
E il mover vuole le due `QsaState` uguali nel formato (§5): una variabile in più per niente. **No.** Densi (2088 MiB
con flussi) e pooled (224) sono il grosso e non si spostano finché lo stadio esiste.

### 13.5 Stime

| | tappa 4 (M) | 4b = A+B (S) | 4b + LM (S) |
|---|---|---|---|
| tier del peer | 6372 slot | ~9350 (`10971 − 2692 MiB / 1,66`) | ~9350 |
| scrittura 4K / 32K / 128K | 61-73 / 77-79 / 81-87 | **85-96** (finestra ~25,5-31 ms: CPU 1,4-2,7 esperti a strato-finestra, interpolato fra Z4fus e Z4ep) | uguale |
| lettura | 822-872 / 1525-1540 / 1703-1727 | ± 3 % (830-880 / 1500-1580 / 1680-1780) | 4K uguale (chunk-major a 4096); 32K/128K **tra +0 e +35 %**, bit diversi |
| refill a fine prompt | CUDA0 2000 slot sull'Oculink ≈ 0,5 s | + peer 2000 slot sul x16 ≈ 0,26 s **in parallelo** (`refill` emette tutte le parti, poi attende): ≈ 0,5 s | + le righe del drafter: 0 |
| RAM bloccata | +2,5 GiB (§2) | uguale | + `R_host` 1,34 GB per stadio |
| VRAM CUDA0 | −144 slot (`ep_R_dev0`) | uguale; con A2 torna a 2559 e chunk 7680 | chunk 4096 |

Lavoro (S): A ~200 righe in `generate.cpp` (parte del peer, `quiesce` nel `lend`, `own_fits`, riga di log, riserva
a 0; +60 con A2; +80 con A-LM), B ~80 righe, C ~120 righe. Compilato no, installato no, provato no.

### 13.6 Rischi

- **Coppie di `[0, 20)` nella coda prestata**: lo stadio 0 le manda in streaming (≤ 4 %, S). Se pesa: una coda
  «fredda» ordinata per strato cambierebbe il tier (bit della scrittura): non ora. La riga di log la conta.
- **`relayout` scrive negli slot mentre una copia di `adapt` è in volo**: coperto da `peer.quiesce` **dentro** il
  `lend` (oggi sta dopo, `:16740`). Il `fused_begin` con la guardia alzata è un errore, mai un testo diverso.
- **Refill dalla fonte giusta**: il tier è stato riempito da `src.blob` (`peer_experts.cpp:341`); il `refill_issue`
  usa `srcp->blob_stable` come per CUDA0: stessa arena. Un blob non stabile (`#4695`, Windows) passa per `blob_parts`
  come oggi.
- **Tabella condivisa**: `res_` è la stessa che `res_table()` ha dato allo stadio EP e che `has()` legge dal filo del
  pool. Durante il prompt non c'è pool né finestra; la scrittura in `lend`/`refill` avviene dal filo della
  richiesta, con gli stream fermi. Nessun lock in più.
- **Scan del chunk**: `part_slots` sul device del peer con `ep.ss` (28 strati: i buffer dipendono dalla sessione per
  la selezione QSA); se il tier non potesse prestare (mai, con 9350 slot) il ramo «allocates its own» di oggi
  resta e la riserva è 0: lo stadio fallirebbe l'`init` con «do not fit», mai un testo diverso. Con
  `STRATA_EP_SPLIT_LOAN=0` si torna alla riserva calcolata.
- **LM e checkpoint a metà prompt**: `layer_major_mid()` salta i checkpoint dentro un super-chunk (come per la
  divisione); con EP le parti del §6 arrivano da `on_stage_chunk` a fine super-chunk su entrambi gli stadi: da
  verificare nel banco, non per costruzione.
- **LM e bit**: §13.3. Mai confrontare LM con chunk-major chiedendo «identico».
- **Righe citate**: sono di `1fe8e46d`; `generate.cpp` cambia spesso. I pezzi cercano i simboli, non le righe.

### 13.7 Pezzi (file disgiunti, solo scrittura; compila, prova e salva la chat principale)

| pezzo | file | cosa |
|---|---|---|
| **A** | `src/program/generate.cpp` | la parte del peer in `pf_parts` (`PfPart::res`), `quiesce` nel `lend`, `init_prompt_paths` con `sb/sbb` per lo stadio EP, `own_fits`, riserva a 0 + `STRATA_EP_SPLIT_LOAN`, riga di log con le coppie di `[0, K)`; A2 facoltativo (righe del drafter nella coda di CUDA0); A-LM dietro `STRATA_EP_SPLIT_LM` (niente `set_peer`, chunk 4096, `part_slots` + `lm_bytes_needed`, `set_lm_region` dopo il `lend`). Usa solo: `PeerExperts::cache_mut/res_table_mut/lend_guard/lent_slots` (B) e `Prefill::lm_bytes_needed/set_lm_region` (C) |
| **B** | `include/strata/core/peer_experts.hpp`, `src/core/peer_experts.cpp` | `cache_mut()`, `res_table_mut()`, `lend_guard(bool)` + `lent_slots()`; `adapt` torna subito con la guardia; `fused_begin` rifiuta con errore; commenti del contratto (`:70-82`) aggiornati: il tier **presta** la coda |
| **C** | `include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp` | C1 (`lm_prepare` declina su `m.pp`/`helper_`, non su `peer_portable()`), C2 (`lm_bytes_needed`, `set_lm_region`, `lm_prepare`/`lm_release` con regione, modo host forzato) |

Ordine per compilare: B, C, poi A. A compila anche senza A-LM (C1/C2 restano inerti senza `set_lm_region`).

### 13.8 Prova (chat principale)

Stessa coda del §12, build nuova: `Xeps4b{n}` (variabile + prestito) contro `Xeps{n}` (`STRATA_EP_SPLIT_LOAN=0`) e
`Xfus'` a cache uguale (`--peer-reserve-mib 3000 + 2692`). Criteri: scrittura ≥ 85 t/s su tutte e tre le lunghezze
e testo **identico a `Xfus'`** sul primo turno con `--adapt-swaps 0`; lettura non sotto `Xeps` −3 %; la riga del
prestito del peer con K coppie di `[0, 20)`; il tempo di `refill` (riga `strata trace: refilled … on 2 stage(s)`)
≤ 0,6 s; nessun `strata.exe` rimasto acceso. Poi, a parte, `STRATA_EP_SPLIT_LM=1` contro un `Xrif` **anch'esso
layer-major** (la riga `strata prefill: layer-major (layers 0..20)` deve esistere in entrambi): lettura 32K/128K e
KL, non bit.

### 13.9 Integrazione (3/11, sul worktree `lab/upsync-v0.2.9` dopo `1fe8e46d`): com'è scritto davvero

Letto il diff dei tre pezzi contro HEAD e fatto quadrare. **Compilato no, provato no, commit no.** Dove il codice si
discosta dal disegno dei §13.2-13.3, vale il codice; qui la lista.

**Variabili e default** (`generate.cpp`, accanto a `ep_K`): `STRATA_EP_SPLIT_LOAN` (default 1, vale solo con
`ep_on`; `0` = riserva della tappa 4), `STRATA_EP_SPLIT_LM` (default 0; senza il prestito è ignorata con una riga),
`STRATA_EP_SPLIT_RESERVE_MIB` col prestito = riserva **in più** forzata (default 0), senza il prestito = come in
tappa 4 (`reserve_add_mib`). Il prestito nasce solo se `pf_borrow` (niente `--no-prefill-borrow`) e c'è la tabella
di residenza (`d_res`): la parte del peer è l'**ultima** di `pf_parts` (`ep_pi`; `ep_part()`, `ep_has_loan()`,
`is_ep_part(p)`), con `lb 0, le n_layers`, `rc_floor 0`, `res = peer.res_table_mut()`; tutte le altre parti hanno
`res = host_res.data()`, e `lend`/`refill_issue` scrivono `p.res` (per CUDA0 e stadi è la stessa memoria di prima).

**Per richiesta**: `lend` → per la parte del peer `peer.quiesce(e)` **prima** di `relayout` e della marcatura, poi
`peer.lend_guard(true, slots − first)` **dopo** la marcatura (anche se la coda non aveva coppie); `refill_wait`
abbassa la guardia dopo `sync_queued`; `refill` la abbassa anche quando non c'era nulla da rimettere. Il `quiesce`
dentro `run` resta e non trova nulla. Firma di B: `lend_guard(bool on, int64_t slots = 0)`, `lent()`,
`lent_slots()`. `fused_begin` con la guardia alzata = errore «tier slots are lent to the prompt path»; `adapt` torna
subito; `apply_pending` non controlla la guardia di proposito (vedi commento in `peer_experts.cpp`).

**A-LM** (`STRATA_EP_SPLIT_LM=1`): niente `sp.set_peer`; chunk fermo a 4096 (`prefill_auto_cap`, `prefill_auto_max`,
`prefill_chunk`); `part_slots` e `ring_room` contano `bytes_needed(c) + 256 + lm_bytes_needed(g, S, c)` per ogni parte,
S = `min(32768↓c, prompt↑c)` (32768 nello scan iniziale; `lm_tokens_hint` dal `lend`). **La regione del super-chunk
sta in FONDO al prestito**: `lm_off = (region − lm_bytes)` allineato in giù a 256, `relayout(c, device_slot(first),
lm_off)` e `set_lm_region(device_slot(first) + lm_off, lm_bytes)`, a ogni `lend` di ogni parte con prestito (non
«`device_slot(first) + bytes_needed(c)`» come scritto in §13.3 iv: stesso scopo, più tollerante). `init` e `relayout`
sono allocatori a bump dalla base (`prefill.cpp` `Alloc o; o.base = borrow`), quindi non toccano il fondo. Una parte
senza prestito (`first < 0`) non riceve mai la regione e `lm_prepare` declina con «a peer GPU»: la camminata non
parte e lo dice la riga `layer-major off for this prompt`.

**C1 corretto in integrazione**: `lm_prepare` declina con `peer_portable()` quando `m.lm.region == nullptr`
(§13.3); la stesura iniziale lo toglieva per tutti e cambiava il comportamento a variabili spente con
`--peer-prefill-rows 0` o `set_peer` fallito più `STRATA_PREFILL_LAYER_MAJOR=1`.

**Invarianza a variabili spente** (verificata riga per riga sul diff): `ep_on` falso → `ep_loan`, `ep_lm` falsi,
`ep_pi = SIZE_MAX`, nessuna parte del peer; `lm_extra_for` = 0 (`part_slots`, `ring_room` invariati); `region` in
`lend` = `part_bytes(p, first)` com'era; `p.res == host_res.data()` per tutte le parti, `any` come prima; le righe
di log del prestito stampano le stesse di prima (il `continue` sostituisce l'`if`); `prefill.cpp`: regione nulla →
`cudaMemGetInfo`, margine, `cudaMalloc`, `cudaFree`, stesso testo di `lm_run`; `dev_bytes(s, false)` =
`lm_bytes_needed` è la formula di prima al byte; `peer_experts`: nulla chiama `lend_guard`, `lent_` resta falso.
`host_plan_chunk` (#2824) vede CUDA0 e gli stadi della divisione a strati come prima: la parte del peer è esclusa
da `gc_bytes`/`gc_slots` e da `lend_of` (vedi le correzioni sotto); il fit in VRAM (`fits`) la include.

**Correzioni della revisione (3/11, stesso worktree; compilato no, provato no)**:
1. **Regione del super-chunk dimensionata sul primo segmento** (`lend`): `PfPart::lent_lm_S` tiene il super-chunk
   (`lm_S_for(want)`) per cui il prestito in corso è stato prezzato; il «`want <= lent_chunk` → continue» vale solo
   se anche `lm_S_for(want) <= lent_lm_S`, altrimenti `refill_one` e nuovo prestito con la regione giusta. Una
   richiesta tagliata (system 8K + messaggio 100K) non cammina più a super-chunk da 8192.
2. **`ep_lm` solo con un prestito vero**: spento con una riga (i) senza `pf_borrow` (`--no-prefill-borrow`), prima
   del tetto del chunk; (ii) prima di `peer.open`, se la proiezione del tier non regge il prestito (punto 4); (iii)
   dopo lo scan, se CUDA0 o la parte EP restano con `first < 0` (il chunk resta a 4096: detto nella riga). Spento,
   `set_peer` si chiama come senza la variabile. `ep_loan` ed `ep_lm` non sono più `const`.
3. **Piano #2824** (`host_plan_chunk`): la parte EP è esclusa da `gc_bytes`/`gc_slots` e da `lend_of`. Le coppie del
   tier del peer stanno nella copia in RAM (`stage_pairs` elenca solo stadi e tier helper), quindi contarle in
   `gpu_cache` le toglieva due volte dal `file_tier`, e il suo prestito finiva in `unbacked` pur tornando dalla RAM.
4. **Riserva tolta prima di sapere se il tier presta**: prima di `peer.open` si proietta il tier con la sua stessa
   aritmetica (VRAM libera − riserva, `--peer-slots`, slot = `max_blob`) contro il prestito che lo scan chiederà
   (`--prefill N`: N; auto: 1024, il chunk di ripiego quando nulla è prestabile), con il pavimento di 128 slot e, in
   auto, il tetto `STRATA_PREFILL_LEND_PCT`. Se non regge: riserva della tappa 4, `ep_loan` ed `ep_lm` spenti, una
   riga. (`d_res` non è ancora noto lì: il caso senza tabella resta come in §13.6.)
5. **WARNING #448 sulla parte EP**: niente consiglio «--expert-cache-device1» (incompatibile con `--peer-device`);
   dice che è il tier del peer e cosa gli lascia più slot (`--peer-reserve-mib`, `--peer-slots`,
   `STRATA_EP_SPLIT_LOAN=0`).
6. **`lm_release` con regione prestata** (`prefill.cpp`): oltre a `m.cs` sincronizza anche `m.copy` (prima lo faceva
   `cudaFree` per tutto il dispositivo): le H2D di `issue_up` in volo su un'uscita anticipata non possono più
   atterrare su uno slot che il `refill` ha già rimesso a esperto.
7. **`set_lm_region` azzerata**: in `refill_wait` (slot tornati esperti) e nel `lend` per una parte senza prestito
   (`first < 0`), come chiede il contratto in `prefill.hpp`.
8. **Prompt sotto due chunk**: `lm_S_for` restituisce 0 (nessuna regione) quando `lm_tokens_hint < 2·chunk`, perché
   `lm_prepare` declina comunque con «a short prompt» (`min_s = 2T`); con `STRATA_PREFILL_LM_TOKENS` o
   `STRATA_PREFILL_LM_FORCE` (pavimento T o 1) basta un chunk e la regione resta. Nel `lend` `lm_bytes == 0` non è
   più un errore: `relayout` con tutto il prestito, `set_lm_region(nullptr, 0)`. Il banco 4K non perde più slot.

**Punti probabili di errore in compilazione** (da guardare per primi):
1. `generate.cpp` `lend`: `p.cache->device_slot(first) + lm_off` (uint8_t* + uint64_t → void*: ok), `ep_lm ? lm_off :
   region` (entrambi uint64_t).
2. `generate.cpp` 10105: l'aggregato di `PfPart` con 10 elementi e due membri col default (`rc_floor`, `res`): già
   fatto così alle righe sopra.
3. `generate.cpp`: `if (const PfPart* pp = ep_part(); …)` (C++17, già usato a `:16025`); `SIZE_MAX` già usato.
4. `prefill.cpp` `dev_bytes`: lambda con due `return` entrambi `uint64_t` (`lm_bytes_needed` e l'espressione castata).
5. `prefill.cpp` `lm_bytes_needed` è `static` nell'hpp e usa `N, HC, D, K` (`:173`, file scope) e `lm_table_ints`
   (namespace anonimo sopra): deve stare **dopo** quel namespace, com'è.
6. `peer_experts.hpp`: `std::atomic<bool>` inline con `<atomic>` incluso; `PeerExperts` non copiabile per via
   dell'atomico (già aveva stream/eventi: nessun copy in giro, da verificare se qualcuno lo sposta).
7. Avvisi: `total_b` in `lm_prepare` ora usato solo nel ramo `else`; `(void) g` in `lm_bytes_needed`.

**Rischi funzionali da provare al banco** (non di sintassi):
- Un `ERR` fra `lend` e `refill` lascia la guardia alzata: i rami che ho visto finiscono con `return 1`, ma in
  `--serve` va verificato che ogni uscita della richiesta passi da `refill` (`:17644`, `:17828`).
- Il WARNING #448 sulla parte EP ora ha il suo testo (correzione 5); con ~9350 slot non dovrebbe scattare.
- `STRATA_EP_SPLIT_LOAN=1` senza `d_res` (nessuna tabella): riserva 0 e lo stadio fallisce l'`init` con «do not fit»
  (§13.6, accettato; la proiezione della correzione 4 non vede `d_res`, che nasce dopo).
- La proiezione del tier (correzione 4) conta gli slot a `max_blob`: su un pack nativo il tier reale ha più slot
  (blob per strato ≤ `max_blob`), quindi la proiezione è prudente e può tenere la riserva dove lo scan avrebbe
  prestato. Da leggere nella riga «will have about N MiB of slots» contro il «peer GPU 1: N experts» che segue.
- Il ri-prestito per segmento (correzione 1) paga un `refill_one` + `lend` in più sulla parte che cresce: 120-310
  ms per passaggio di stato, da contare nelle misure delle richieste tagliate.
- `--batch` e `--prefill-head-alias` sono rifiutati da `ep_split::admit`: la guardia su `fused_begin` non può
  scattare per le finestre fra i blocchi.

## 14. Tappa 4c: soglia, densi in prestito, passaggio sovrapposto (11/11, scritto su `8eb601e3`, ramo `lab/tappa4c`)

Scritto dopo le misure dell'11/11 (3080 Ti padrona su Oculink x4 Gen4, 2080 Ti primo server su x16 Gen3, 3060 secondo
server su Oculink x4 Gen3). Tutto dietro **`STRATA_EP_SPLIT_PREFILL=2`**: con `=1` il motore è la tappa 4b, bit per
bit (`ep_4c` falso: nessun ramo nuovo, `ep_staged` resta vero, la gamma di strati di `sp` è quella dell'`init`); con la
variabile assente nulla esiste. **Compilato no, installato no, provato no.** Convenzione: **M** misurato, **S** stimato.

### 14.0 In tre righe

1. **Da dove partiamo (M, 11/11).** Divisione a strati 20: lettura 897 / 1816-1821 / 2128-2133 (4K / 32K / 128K),
   scrittura 97-105, finestra 23,0 ms. Due server (`--peer-device 1,2`, mega-kernel, `STRATA_PEER2_PREFILL=1`): lettura
   969 / 1489 / 1567, scrittura 104,8-109,3, finestra 23,5 ms. Tappa 4b con un server sul riser (03/11): lettura
   853 / 1525-1538 / 1655-1740, scrittura 77-95, passaggio di stato 123-344 ms a richiesta; con due server sul riser
   scrittura 79-82. Il conto: la 4b legge come la divisione ma scrive peggio dei due server, perché la 2080 Ti tiene
   2,7 GiB di densi+sessione per tutta la sessione (~1600 posti esperti in meno) e paga il passaggio di stato.
2. **Che cosa fa la 4c.** (1) Sotto una soglia di token nuovi la richiesta si legge **come i due server di oggi** (la
   padrona su 48 strati, le quote di `set_peer` e `set_peer2`, la 3060 compresa; lo stadio dorme, niente passaggi);
   sopra, a stadi come la 4b. (2) I **densi dello stadio stanno negli ultimi slot del tier** del peer: caricati lì dopo
   `peer.open`, copiati una volta in RAM bloccata, e in VRAM **solo durante una lettura a stadi** (il `lend` li
   ricopia dallo specchio nei posti che ha appena preso, il `refill` ci rimette gli esperti): in scrittura il tier ha
   quei ~1300 posti. (3) Le celle di un chunk finito **passano alla padrona dal filo dello stadio** mentre lo stadio
   legge il chunk dopo; alla fine passa solo l'ultimo chunk con lo stato ricorrente.
3. **Numeri attesi (S).** Scrittura **~100-110 t/s** su tutte le lunghezze (il tier torna a ~10 900 meno i ~360 posti
   della sessione: i due server da soli fanno 105-109); lettura sotto la soglia **= due server** (969 / 1489 a 4K /
   32K), sopra **= 4b ± 3 %** meno il costo del prestito dei densi durante il prompt (circa −2 %, §14.2). Il passaggio
   di stato a fine prompt scende da 123-344 ms a **~20 ms** (S), più ~10 ms per chunk sovrapposti.

### 14.1 Soglia (`STRATA_EP_SPLIT_MIN_TOKENS`, default 16384)

Il pareggio fra «due server» e «a stadi» si stima dai numeri M interpolando la velocità fra i tre punti misurati e
aggiungendo alla lettura a stadi un costo fisso di **0,3-0,5 s** (passaggio di stato 4b 123-344 ms + ricarica dei densi
~175 ms, §14.2; con la sovrapposizione del §14.3 il primo scende, ma qui si tiene il valore misurato):

| token nuovi | due server (M interp.) | a stadi = divisione 20 (M interp.) + 0,4 s | a stadi = 4b (M interp.) + 0,4 s |
|---:|---:|---:|---:|
| 8K | 1043 t/s → 7,9 s | 1028 t/s → 8,4 s (+6 %) | ~980 → 8,8 s (+11 %) |
| 12K | 1118 → 11,0 s | 1160 → 11,0 s (pari) | ~1100 → 11,6 s (+5 %) |
| 16K | 1192 → 13,7 s | 1291 → 13,1 s (−5 %) | ~1230 → 13,7 s (pari) |
| 32K | 1489 → 22,0 s | 1816 → 18,4 s (−16 %) | 1525 → 21,9 s (pari) |
| 128K | 1567 → 83,6 s | 2128 → 62,0 s (−26 %) | ~1700 → 77,5 s (−7 %) |

Due cose si leggono nella tabella. **Il default 16384** è il primo punto dove la lettura a stadi può vincere **e** il
primo con due chunk da 8192, cioè dove la pipeline a due stadi si sovrappone almeno per un chunk (a un chunk solo i due
stadi girano in serie: ecco il 4K a 897 contro 969). E **il guadagno vero della 4c sta nella scrittura**, non nella
lettura: con `STRATA_PEER2_PREFILL=1` i due server leggono a 32K quanto la 4b misurata (1489 contro 1525); la lettura
a stadi vince solo se recupera la residenza della divisione (colonna di mezzo) e comunque dai 32K-128K. La soglia va
quindi **misurata**, non creduta: le corse del §14.8 con `STRATA_EP_SPLIT_MIN_TOKENS=0` (sempre a stadi) e con il
default dicono dove sta il pareggio su questa macchina; se la colonna di destra è quella vera, 32768 o 65536 è il
default giusto e si cambia il numero, non il codice.

**Come è fatto** (`generate.cpp`, per richiesta, dopo `read_from`): `fresh = n − read_from`;
`ep_staged = staged_read(fresh, soglia)`; `sp.set_stage(0, ep_staged ? K : n_layers, ep_staged ? &ep.sp : nullptr)`.
`Prefill::init` ha dimensionato i buffer di passaggio di mano una volta (con `next_`), e `run_impl` legge `stage_lb_`,
`stage_le_` e `next_` a ogni corsa (`prefill.cpp:3729`, `:7022`, `:7704`): la gamma si cambia fra una richiesta e
l'altra senza rifare l'`init`. Il callback `on_chunk` sta su **entrambi** i percorsi (`run_impl` lo chiama solo
sull'ultimo stadio, quello senza `next_`: `:7121`), così il `set_stage` decide chi riferisce; `on_stage_chunk` esce
subito quando la lettura non è a stadi, e i checkpoint a metà prompt prendono la via semplice (`checkpoint_at(done,
nullptr)`), come senza la variabile. Nel `lend` la parte del peer **non viene prestata** (il tier tiene ogni coppia per
la quota della padrona, i densi restano in RAM); `ep_to_primary`, il `to_peer` di `run` e le righe del drafter sono
saltati. La sessione dello stadio **resta indietro** di quel che la padrona ha letto da sola: la prossima lettura a
stadi copia prima ciò che manca (`Sync::valid_for` → `to_peer(c0, q)`), come dopo una risposta. Sempre a stadi con
`STRATA_EP_SPLIT_LM=1` (regioni e percorso senza `set_peer` sono della lettura a stadi) e con DFlash2 (`reserve_taps`
ha dimensionato i tap per gli strati dello stadio). Una riga per richiesta quando il modo cambia (sempre con
`STRATA_EP_SPLIT_TRACE=1`): `strata ep-split: N tokens to read: staged read … / the primary reads every layer …`.

### 14.2 Densi del secondo stadio non residenti in scrittura (`STRATA_EP_SPLIT_DENSE_LOAN`, default 1)

**Perché nel tier e non «strato per strato dalla RAM».** I pesi sono puntatori: `WeightTable` tiene `WeightRef::data`
nell'arena, `NativeDense` attacca `native_data` e registra le copie «packed» **per puntatore** nei kernel
(`native_q8_0_packed_register`), e il prompt path li legge così. Spostarli a ogni prompt vorrebbe un rebase di tutta la
tabella e delle registrazioni; trasmetterli strato per strato in un anello vorrebbe lo stesso rebase per strato, più
2,1 GiB a chunk sul x16 che già porta ~11 GB di esperti (+20 % di traffico per chunk). Invece **un indirizzo fisso**:
gli ultimi slot del tier del peer esistono per tutta la sessione, stanno in fondo alla coda che la 4b già presta al
prompt path dello stadio, e il tier li rivuole solo in scrittura. Quindi:

- **Avvio** (`generate.cpp`, dopo `peer.open`): `ep_dense_need` = arena del pack dei soli strati `[K, 48)` (lo stesso
  `pool_bytes` con lo `skip` di `add_foreign`) + `NativeDense::weight_bytes_for(K, 48)` (ogni matrice arrotondata al
  granulo di `cudaMalloc`: un tetto) + 512 B; `tail_region(slots, slot_offsets, max_blob, bytes, need, g_lend_keep)`
  dà l'offset allineato a 256 dove la regione comincia e il **primo slot che tocca** (`ep_dense_first`); la proiezione
  del tier prima dell'open (§13.9 correzione 4) conta `need` anche per i densi, così «non entra» si decide **prima**
  che il tier prenda la VRAM (allora i densi si caricano sulla scheda prima dell'open, come in 4b, con una riga). Il
  caricamento è quello di sempre (`load_stage_weights` con `place`): `WeightTable::load` scrive nei primi `pool_bytes`
  della regione; `NativeDense::set_placement` fa carvare le matrici (copie GGUF, packed, righe HC Q8) **a bump** nel
  resto invece che con un `cudaMalloc` ciascuna (gli scratch Q8_1 e `hc_bf16_view` restano `cudaMalloc`: li legge anche
  la scrittura, ~200 MiB dalla riserva del peer). Poi `DenseMirror::capture` (D2H una volta, RAM bloccata Portable,
  pageable se l'allocazione fallisce, detto in riga) e **gli esperti che l'open aveva messo in quegli slot tornano
  subito dall'arena** (`fill_slot_queued` + `sync_queued`, la via di `refill_issue`): il tier è intero prima della
  prima finestra. Riga: `strata ep-split: CUDAn: the stage's dense weights (M MiB) live in the peer tier's last S
  slots [first, slots) during a staged read only: loaded, mirrored … and E experts put back …`.
- **Per lettura a stadi** (`lend`): la parte del peer è prezzata per buffer + regione LM (se c'è) + **regione dei
  densi** (`dense_extra_for`: `kTailAlign + ep_dense_region`, i byte da `ep_dense_base` alla fine del tier), così `first
  ≤ ep_dense_first` per costruzione (un `first` oltre è errore, mai testo diverso); i buffer del prompt (`relayout`) e la
  regione LM ricevono i byte **prima** di `ep_dense_base`; le coppie dei posti prestati escono da `res_` come in 4b
  (quindi anche quelle del fondo); poi `DenseMirror::restore(ep_dense_base, ep.stream)`: **H2D in coda sullo stream di
  calcolo dello stadio**, prima del suo primo kernel della lettura (stesso stream: ordinato, nessuna attesa).
  `STRATA_EP_SPLIT_DENSE_SYNC=1` aspetta la copia nel `lend` e ne stampa i ms. Il `refill` (prima delle finestre e a
  fine prompt) rimette gli esperti su quei byte: in scrittura la regione **è esperti**.
- **Costo a richiesta (S).** 2088 MiB (M, densi+flussi di `[20, 48)`) sul x16 Gen3 a 12-13 GB/s: **160-175 ms**,
  sovrapposti al chunk 0 della padrona (2-4 s) quando la copia è in coda; ~175 ms seriali con `_DENSE_SYNC=1`; ~350 ms
  dalla RAM pageable. Durante il prompt il tier presta ~1300 posti in più (2,1 GiB / 1,66 MiB): lo stadio EP perde
  ~760 coppie di `[20, 48)` (+0,9 GB a chunk sul x16, circa +70 ms su ~4 s, 2 %), lo stadio 0 ne perde ~540 di
  `[0, 20)` nella coda (+0,65 GB sull'Oculink, circa +0,1 s a chunk, 2,5 %, in sovrapposizione). RAM: **+2,1 GiB
  bloccata** (lo specchio). Il tetto `weight_bytes_for` lascia fino a ~1 MiB a matrice di scarto nella regione
  (~170 MiB, ~100 posti prestati per niente durante il prompt; in scrittura sono esperti).
- **Guadagno (S).** In scrittura il tier ha ~1300 posti in più della 4b (restano fuori i ~360 della sessione dello
  stadio, 604 MiB, §13.4): la CPU calcola meno coppie a strato-finestra e la scrittura torna verso quella dei due
  server da soli (104,8-109,3 M). Sopra la soglia, una lettura da 32K (~20 s) paga ~2 % di streaming in più e 0-175 ms
  di ricarica; sotto la soglia non paga nulla.

### 14.2bis Le righe del drafter nella coda del prestito di CUDA0, e il costo permanente che resta (11/11 sera)

Prima misura della 4c nel motore (chat principale, build `497edc7e`, due server sull'Oculink, soglia 16K): lettura
911 / 1503 / 1724, scrittura 107 / 109 / 111; due server senza 4c 967 / 1492 / 1576, 107 / 109 / 105; 4b 845 / 1524 /
1728, 106 / 104 / 107. La scrittura è a posto (+9-10 % a 128K); **sotto soglia la 4K perde il 6 %** (911 contro 967).
Dal registro: `peer GPU 1: 10596 experts` contro 10969, `primary's 2360` contro 2554, prestito CUDA0 2000 slot
(chunk 7168) contro 2119 (7680). Cause, per scheda:

- **CUDA0, −194 slot e chunk 7168**: `ep_R_dev0` (320 MiB, le righe del drafter) era un `cudaMalloc` permanente dopo
  la cache, contato nella riserva del dimensionamento automatico (`ep_rows_bytes`). Serve solo durante una lettura a
  stadi. Ora (A2 del §13.2bis, fatto): con il prestito le righe stanno negli **ultimi slot di `xcache`** (indirizzo
  fisso, `tail_region`), prezzate nel prestito di CUDA0 **solo dentro `lend` e solo a stadi** (`ep_rows_priced`,
  `dense_extra_for`), mai nello scan: il chunk e il prestito sotto soglia sono quelli senza la variabile (2554 slot,
  7680, 2119); una `lend` a stadi prende ~185 slot in più (2304 ≤ 2554 − 128) e il `relayout` segue `first`. La
  riserva del dimensionamento non conta più le righe (`ep_rows_bytes = 0` con `ep_4c && pf_borrow`). Senza tabella di
  residenza (nessun prestito) resta il `cudaMalloc` di prima, detto in riga.
- **Il prestito su CUDA1 sotto soglia non scatta**: `lend` è chiamato solo nel ciclo dei tagli, dopo la decisione per
  richiesta, e salta la parte del peer quando `!ep_staged`; la riga `CUDA1 prompt path borrows 3274 of its 10596 tier
  slots` è dello **scan d'avvio** (il prezzo del prestito a stadi: buffer + densi), non di una richiesta.
- **CUDA1, −373 slot (620 MiB): la sessione dello stadio, e resta.** 604 MiB (M): K/V residente 7 QSA × 32768 celle ×
  1056 B = 231 MiB, righe pooled a `max_context` 262144 = 224 MiB, GDN 21 × 2 MiB = 42, buffer QSA/MoE/blocco, code e
  tabelle rope ~107; più ~16 MiB di stream, cvec, mrope e scratch Q8_1 di `NativeDense`. Metterla nella coda del tier
  come i densi (immagine «fresca» dopo `session_zero` specchiata in RAM, ripristinata a ogni `lend` a stadi,
  `ep_sync.invalidate()` al `refill`, `to_peer(0, q)` prima di ogni lettura a stadi: ~110 ms a 128K) richiede di
  **ricarvarla dopo `peer.open`**, perché `Prefill::bytes_needed_owned(g, ep.ss, …)` e la proiezione del tier la
  vogliono carvata prima; e una seconda `session_init` rialloca le copie host bloccate del K/V (1,9 GB) che
  `session_release` non libera (`layer.cpp` non ha un rilascio delle `QsaState::host`). Con quel rilascio il disegno è
  lo stesso dei densi e vale ~364 slot in più in scrittura; **senza, la sessione resta residente: 604 MiB, ~364 slot,
  l'unico costo permanente della 4c sulla 2080 Ti.** Non si stringe (§13.4: il mover vuole `max_cells` uguali).

Che cosa aspettarsi nelle stesse righe, dopo questa correzione: `primary's 2554` (come senza 4c), `the prompt path
borrows 2119 CUDA0 cache slots` e chunk 7680 sotto soglia; a stadi `lent … slots` di CUDA0 ≈ 2304 e la riga nuova
`CUDA0: the drafter's row buffer (320 MiB, 7680 rows) is the last ~190 cache slots [2364, 2554), lent with the prompt
buffers during a staged read only`; `peer GPU 1: ~10596 experts` **invariato** (la sessione). La 4K sotto soglia deve
tornare a 967 ± 2 % se la perdita era di CUDA0 (slot e chunk, quindi anello e quota `set_peer` a `chunk·K/2`); se
resta un divario, è la residenza del tier (−373 coppie → ~1,5 % S) e lo dice il confronto con `Xfus'` a
`--peer-reserve-mib 3000 + 620`.

### 14.2ter La sessione dello stadio nella coda del tier (`STRATA_EP_SPLIT_SESSION_LOAN`, default 1; 11/11 notte)

Dopo la correzione del §14.2bis (salvata in `da51cc27`, build `c2e31a79`): 4c 927-930 / 1491-1500 / 1757-1762 in
lettura e 103,9-104,6 / 107,4-111,6 / 111,5-113,0 in scrittura, due server 968-969 / 1489-1496 / 1565-1577 e
107,5-108,2 / 109,9-110,2 / 107,6-111,8 (M); `primary's 2554`, `peer GPU 1: 10599`. Resta il costo della sessione
dello stadio (604 MiB, ~370 posti). Il pezzo lo toglie, con lo stesso meccanismo dei densi.

**Che cosa deve sopravvivere fra una richiesta e l'altra nella sessione dello stadio: niente.** Verificato riga per
riga: la verità è sempre `ss` (§6). (a) Il K/V, le righe pooled, le righe GDN e `tail/dead/block_pos` delle celle
lette a stadi passano a `ss` con `ep_to_primary` **prima** del `refill` che segue (nel ciclo dei tagli il passaggio
viene subito dopo `read_part`, il `refill` all'inizio del taglio successivo o a fine prompt; anche nel ramo di
cancellazione). (b) La cache delle conversazioni (parcheggio, ripristino, prefisso prestato, slot) salva e rimette
**solo `ss`** (`stages` vuoto, nessuna `stage_images`) e ogni ripristino già chiama `ep_sync.invalidate()`. (c) I
checkpoint del prompt: quelli a fine prompt/coda/radice leggono `ss` dopo il passaggio; quelli a metà prompt prendono
la parte dello stadio **durante** la lettura (`on_stage_chunk`, filo dello stadio, regione prestata) e
`checkpoint_merge_tail` legge dalla sessione dello stadio solo ordinali e dimensioni (struttura host). (d) Le copie
host del K/V dello stadio (RAM bloccata, `QsaState::host`) **non** stanno nel tier: sopravvivono, ma non servono
(`to_peer(0, q)` le riscrive). (e) `upload_mrope` scrive `ep.mrope`, allocazione a parte. Quindi fra una lettura a
stadi e l'altra la VRAM della sessione può essere esperti.

**Come è fatto.**
- `layer.cpp`: `kv_host_alloc` registra (fuori Windows) i puntatori pinnati con `cudaHostRegister`;
  `kv_host_free(h, bytes)` fa `cudaHostUnregister` + `munmap` oppure `cudaFreeHost` via il bilancio dei pin
  (`host_free_guarded`, nuovo in `pin_budget.hpp`), e scala `g_kv_host_bytes`. `QsaState` tiene `host_base/host_bytes`
  (il puntatore **host**: in `host` stanno gli alias mappati). `qsa_state_release(st)`: tabella rope se propria, copia
  host del K/V, `host_step`/`host_pos`. `session_release` scorre gli stati **per ordinale** (la vecchia scansione
  `[0, qsa_alloc)` non rilasciava nulla di una sessione di stadio) e cancella l'array `qsa_states`: una seconda
  `session_init` allo stesso o a un altro indirizzo parte da zero e non raddoppia 1,9 GB di RAM bloccata.
- `generate.cpp`: la sessione carvata prima dell'open serve al dimensionamento (`bytes_needed_owned`, proiezione);
  **subito prima di `peer.open`** viene rilasciata e `ep_sbuf` liberato (il tier prende quella VRAM). Dopo l'open la
  regione di coda è `[sessione | densi]`: `tail_region` con `need = sessione (256-al.) + densi`; `session_init` a
  `ep_tail_base` (stessa carve: stessi `max_context` e strati, il mover trova gli stessi formati), `session_zero`,
  poi i densi a `ep_tail_base + sess_al`. **Il sigillo** (`ep_tail_seal`, dopo `init_prompt_paths`, prima di
  `arm_prompt_helpers`): specchia l'immagine fresca della sessione (`ep_sess_mirror`) e i densi (`ep_mirror`), rimette
  gli esperti della coda, `ep_sync.invalidate()`; così se l'`init` del prompt path avesse scritto nella sessione,
  l'immagine lo tiene. Nel `lend` della parte del peer: buffer fino a `ep_tail_base`, poi `restore` della sessione
  **aspettato** (604 MiB sul x16, ~50 ms: `run` fa subito dopo `to_peer` nella regione, con i suoi stream e il
  `memcpy` host), poi i densi in coda come prima. Al `refill_wait` della parte del peer e al `refill` che abbassa la
  guardia: `ep_sync.invalidate()`. `ep_zero` (richiesta con `resume == 0`) con la sessione in prestito **non scrive**
  (la regione in quel momento è esperti): invalida e basta; l'immagine fresca del `lend` è lo zero.
- Ogni lettura a stadi parte quindi con `to_peer(ss, ep.ss, 0, q)`: a 128K ≈ 75 ms di `memcpy` + 27 ms di pooled +
  10 ms di stato (S, `kv_mode 1`), su una lettura ≥ 10 s: ≤ 1 %. Prima lo pagavano solo i ripristini.

**Costo permanente che resta sulla 2080 Ti (S):** stream, tabelle cvec, `mrope`, lo scratch Q8_1 di `NativeDense`
e le piccole tabelle del prompt path: **decine di MiB, ≤ 20 posti**. RAM: +604 MiB bloccata (l'immagine). La coda
prestata a stadi sale a ~buffer + 1300 + 364 posti (≈ 3650, 34 % del tier).

**Nel registro:** `strata ep-split: CUDAn: the stage's session (604 MiB) is released before the tier opens; carved again
in the tier's tail after it …`; `peer GPU 1: ~10 960 experts` (contro 10599; i due server senza 4c hanno 10969);
la riga del sigillo `the stage's session (604 MiB, a zeroed image) and M MiB of dense weights live in the peer tier's
last S slots [first, slots) …` con S ≈ 3650 − buffer, cioè ~1660; `lent … slots` della parte del peer a stadi ≈
+364 rispetto a prima; nella riga di fine richiesta `densi ricaricati N volte, sessione N`; con `STRATA_EP_SPLIT_TRACE=1`
un `to peer: cells [0, q)` a ogni lettura a stadi (prima solo dopo un ripristino). Scrittura attesa ≥ due server
(il tier è il loro); lettura sopra soglia −0,5 % (S) per il `to_peer` pieno.

**Rischi.** (i) `Prefill::init` o una sua ricostruzione a richiesta (`arm_prompt_helpers` dopo un rebuild del prompt
path, `generate.cpp` ~18139) che scrivesse nella sessione mentre la regione è esperti: corromperebbe il tier; non ne ho
trovato traccia (`init` legge la struttura host di `ss`), ma non è provato per costruzione. (ii) `kv_host_free` su
Windows presume `cudaHostAlloc` (è l'unica via lì); fuori Windows il registro dei `cudaHostRegister` decide. (iii) La
sessione ricarvata **dopo** `peer.open` e `peer2.open`: nessun `cudaMalloc` (la carve è nel tier) ma le copie host da
1,9 GB sono riallocate dopo che l'arena da 8 GiB è mappata (WDDM): se il pin fallisce, `session_init` torna 0 ed è
errore all'avvio, non un testo diverso. (iv) Lo stesso `session_release` ora rilascia anche a fine processo le tabelle
rope e le copie host dello stadio (prima no): innocuo.

### 14.3 Passaggio di stato: sovrapposto e ridotto (`STRATA_EP_SPLIT_OVERLAP`, default 1)

**Cosa non si può saltare.** La scrittura (`ver.run` su `ss`, 48 strati) legge, per `[K, 48)`: il K/V di **tutte** le
celle (attenzione sull'intero contesto; in `kv_mode 1` la verità è la copia host e il `memcpy` host→host è il
passaggio), le righe pooled (selezione dell'indicizzatore), le righe GDN e `tail/dead/block_pos` (stato ricorrente).
«Solo le parti che la scrittura usa» sono quindi **tutte** quelle che il mover già muove: niente da togliere a bit
uguali. La riduzione viene da tre cose:

1. **Sovrapposizione** (`on_chunk`, filo dello stadio): finito il chunk `[p0, p0+T)` (stream dello stadio
   sincronizzato: le righe del drafter già ci contavano), `mover.to_primary(ep.ss, ss, p0, p0+T, running_state=false)`
   muove **solo K/V e pooled** di quelle celle mentre lo stadio legge il chunk dopo. La pagina e la riga pooled a
   cavallo di due chunk vengono scritte **due volte**: la prima con le celle del chunk precedente già definitive e il
   resto in corso, la seconda complete (il mover arrotonda a pagine intere e riga pooled inclusa, §5). La padrona nel
   frattempo scrive `[0, K)` di `ss` con lo stadio 0: celle di `QsaState` e righe GDN **diverse** da quelle che il mover
   scrive (`[K, 48)`), nessuna sovrapposizione di byte; lo stream dello stadio 0 non si aspetta. Il mover è usato da
   un filo solo per volta (il filo della richiesta aspetta in `run`). Costo: ~10 ms a chunk (S: 7 QSA × 8192 × 1056 B =
   60 MB di `memcpy` + 7 MB di pooled), sul filo dello stadio, cioè sullo stadio più lento: 0,25 % di un chunk.
2. **Passaggio finale ridotto** (`ep_to_primary`): se l'ultimo chunk passato in `on_chunk` finisce esattamente a `c1`,
   `c0` sale al suo inizio: passano **l'ultimo chunk** (pagina iniziale e riga pooled a cavallo, complete) **e lo stato
   ricorrente**. Da 123-344 ms (M, 4b) a **~20 ms** (S). La cancellazione a metà parte (§12) usa lo stesso cammino:
   `ep_inc_to == pp_reached`.
3. **La soglia** (§14.1) toglie i passaggi dove pesano di più in proporzione: un turno corto non ha né `to_peer` né
   `to_primary`; la sessione dello stadio si allinea alla prossima lettura lunga, in un colpo.

`ep_inc_from/ep_inc_to` si azzerano all'inizio di ogni `run`; `Sync::note` resta solo nel passaggio finale (prima,
`ss` non ha ancora lo stato ricorrente). **Bit identici**: stessi byte negli stessi posti; cambia solo quando arrivano.
Rischio del punto 1: se un giorno l'append del K/V in streaming passasse per uno stream diverso da quello di calcolo
dello stadio, la pagina a cavallo potrebbe essere letta prima che atterri; si manifesta come testo diverso, e
`STRATA_EP_SPLIT_OVERLAP=0` lo esclude; il banco 10 copre la convergenza delle pagine, non quel caso.

### 14.4 Variabili e default

| variabile | default | che cosa fa |
|---|---|---|
| `STRATA_EP_SPLIT_PREFILL` | 0 | **2** = tappa 4c (= 1 + i tre pezzi); 1 = tappa 4/4b com'è; 0 = spento |
| `STRATA_EP_SPLIT_MIN_TOKENS` | 16384 | token nuovi da leggere perché la lettura sia a stadi; 0 = sempre (4b) |
| `STRATA_EP_SPLIT_DENSE_LOAN` | 1 | densi dello stadio nella coda del tier solo durante le letture a stadi; 0 = residenti (4b). Vuole il prestito (`STRATA_EP_SPLIT_LOAN=1`, `pf_borrow`): senza, torna a 0 con una riga **prima** dell'open |
| `STRATA_EP_SPLIT_OVERLAP` | 1 | passaggio delle celle chunk per chunk da `on_chunk`; 0 = un passaggio a fine parte (4b) |
| `STRATA_EP_SPLIT_DENSE_SYNC` | 0 | 1: la ricarica dei densi è aspettata nel `lend` e cronometrata |
| `STRATA_EP_SPLIT_SESSION_LOAN` | 1 | la sessione dello stadio nella coda del tier solo durante le letture a stadi (§14.2ter); vuole `_DENSE_LOAN=1`; 0 = residente (604 MiB) |
| `STRATA_EP_SPLIT_TRACE` | 0 | come prima, più la riga per richiesta e una per passaggio di chunk |

Le variabili di 4/4b (`_LAYER`, `_LOAN`, `_LM`, `_RESERVE_MIB`, `_PIECE_MIB`, `_THREADS`) valgono come prima. Riga di
fine richiesta: `ep-split: N passaggi, MB, ms, lettura a stadi|sulla padrona, densi ricaricati R volte`.

### 14.5 Che cosa resta identico

- **Livello 1 e 0**: nessun ramo nuovo gira. Al livello 1 cambia una cosa sola nel codice che gira: `on_chunk` è
  tenuto anche da `sp` (oltre che da `ep.sp`), ma `sp` ha sempre `next_` e `run_impl` non lo chiama mai (`:7121`):
  stesse chiamate di prima. I tre chiamanti di `load_stage_weights` passano `nullptr, 0, nullptr`: stesso `cudaMalloc`.
  `NativeDense::load` senza `set_placement` fa gli stessi `cudaMalloc` (`place_alloc` li delega); `DeviceFree` e il
  distruttore saltano solo puntatori dentro una regione, che non esiste.
- **Livello 2, sotto la soglia**: la lettura è quella dei due server **a parità di cache**, ma la cache della padrona
  e il tier non sono quelli di `Xfus` (lo stadio tiene la sessione, 604 MiB, sul tier; `xcache` è intera dal
  §14.2bis): testo identico a `Xfus'` con `--peer-reserve-mib 3000 + 620` come in §12, non a `Xfus`.
- **Livello 2, sopra la soglia**: la lettura è quella della 4b a parità di residenza; la residenza durante il prompt è
  **minore** di ~1300 coppie (i densi nel tier). Lo streaming cambia **dove** si legge un esperto, non il kernel né
  l'ordine di somma del chunk-major; la sola differenza possibile è `MMQ_DIRECT`, che legge in posto solo gli slot
  con la guardia. Identico alla 4b va **misurato**, non dichiarato: criterio del §14.8.
- **Scrittura**: identica al mega-kernel a parità di tier; il tier è più grande della 4b (~1300 posti): testo diverso
  dalla 4b per costruzione (§10), confronto con `Xfus'` a cache uguale.

### 14.6 Banco (`tools/ep_split_test.cpp`, controlli 9-12)

9 `tail_region` (slot uniformi e dimensionati, pavimento `keep`, regione più grande dell'arena); 10 passaggio
sovrapposto: quattro chunk **non** allineati alle pagine senza stato ricorrente, poi l'ultimo con lo stato, contro un
passaggio intero in `ref` e contro il contratto host, byte identici; in `kv_mode 1` il blocco residente dentro
l'intervallo esce dalla mappa e quello fuori **resta** (il buco del §10, ora coperto); 11 `DenseMirror`: cattura,
sovrascrittura, ripristino aspettato e in coda, indirizzo diverso rifiutato, ripristino prima della cattura rifiutato;
12 `staged_read` e il default della soglia; 13 (§14.2ter) `session_release` + seconda `session_init` allo stesso
indirizzo (stessa carve, `host_base` segue il `kv_mode`), immagine fresca specchiata, sovrascritta e rimessa (le parti
nell'arena tornano fresche; in `kv_mode 1` i pool sono le copie host, fuori dall'arena, esclusi), poi un `to_peer`
nella sessione ricarvata contro il contratto. Esito (chat principale, 11/11): 9-12 OK in fp, `kv_mode 1`, int8, q4;
13 non provato.
Due correzioni della chat principale, da tenere: nel controllo 5 un `cudaDeviceSynchronize()` dopo l'upload e dopo il
memset (falliva su due schede anche sulla 0.2.13.1: gli stream del mover sono non-blocking), e in `on_chunk` un
`cudaDeviceSynchronize()` sotto `OnDevice(ep.dev)` prima di `rows_to_primary`. Comandi: `cmake --build build --target ep_split_test`, poi
`build\ep_split_test.exe 0 2 8192 0` (una scheda per ruolo o due), `build\ep_split_test.exe 0 2 40000 32768`
(`kv_mode 1`), `STRATA_EP_SPLIT_TEST_KV=int8` e `=q4` per i formati delle corse.

### 14.7 Rischi e punti probabili di errore in compilazione

- `generate.cpp`: `load_stage_weights` ha tre parametri in più (`place`, `place_bytes`, `used_out`); la struct locale
  `PlaceOff` con distruttore dentro la lambda; `(uint8_t*) ep_sb` su un `void*`; `tc.device_slot(0) + tr.offset`
  (`uint8_t* + uint64_t`); i `.c_str()` di temporanei dentro `fprintf`/`snprintf` (vivono fino alla fine dell'espressione).
- `native_dense.cpp`: `place_alloc` e `placed_ptr` devono stare **prima** di `DeviceFree` nel namespace anonimo (ci
  stanno, accanto a `g_layer_lb`); `placed_lo_/placed_hi_` sono membri nuovi nell'header.
- `ep_split.cpp`: `std::free`/`std::malloc` (`<cstdlib>` c'è); `DenseMirror::restore` con `cudaStream_t`.
- **Funzionali**: (a) `set_stage` fra una richiesta e l'altra: `run_impl` legge la gamma a ogni corsa, ma se un pezzo di
  `Prefill` tenesse per-strato qualcosa dimensionato all'`init` sulla gamma (`reserve_taps` lo fa: DFlash2 è escluso
  dalla soglia), una lettura piena fallirebbe o leggerebbe fuori: lo dice il banco motore. (b) Il `PeerPrefill` di
  `set_peer` pianifica per strato a ogni corsa (`peer_now`): in una lettura piena copre 48 strati con `cap_rows =
  chunk·K/2`, come oggi senza la variabile. (c) La regione dei densi è in **slot prestati**: un `adapt` o una finestra
  fused mentre è in uso sono esclusi dalla guardia del prestito (4b); un errore, non un testo. (d) La ricarica in coda
  sullo stream di calcolo vale se il primo kernel dello stadio che legge i densi è su `ep.stream` (`Prefill::init(...,
  stream)`); gli H2D degli esperti sul `copy` stream non leggono pesi. In dubbio, `STRATA_EP_SPLIT_DENSE_SYNC=1`
  (175 ms seriali). (e) Gli scratch HC di `NativeDense` (~200 MiB con `STRATA_HC_Q8`) restano `cudaMalloc` **dopo**
  l'open: escono dalla riserva del peer (3000 MiB, ~2190 liberi dopo i buffer di `set_peer`). (f) Il carico dei densi
  dal disco avviene dopo che l'arena host da 8 GiB è mappata (WDDM): nessun `cudaMalloc` grande, solo gli scratch; da
  guardare alla prima prova. (g) La soglia misura `n − read_from`: una richiesta tagliata in segmenti (system +
  messaggio) decide una volta per tutta la richiesta.
- **Non regge?** Nessuno dei tre punti è caduto in scrittura; il punto debole è il **guadagno in lettura** sopra la
  soglia (§14.1, colonna di destra): se la lettura a stadi non supera i due server nemmeno a 128K, la 4c resta utile per
  la sola scrittura con `STRATA_EP_SPLIT_MIN_TOKENS` molto alto, oppure si spegne e il numero da battere è quello dei
  due server.

### 14.8 Prova (chat principale)

Stessa coda del §13.8, build nuova. Corse: `X4c{n}` = configurazione due server (`--peer-device 2,1`,
`STRATA_PEER_FUSED=1`, `STRATA_PEER2_PREFILL=1`) + `STRATA_EP_SPLIT_PREFILL=2`; `X4b{n}` = stessa con `=1`;
`Xfus{n}` = senza. Bracci: `STRATA_EP_SPLIT_MIN_TOKENS=0` (sempre a stadi: isola i pezzi 2 e 3),
`STRATA_EP_SPLIT_DENSE_LOAN=0`, `STRATA_EP_SPLIT_OVERLAP=0`, `STRATA_EP_SPLIT_DENSE_SYNC=1` con `STRATA_EP_SPLIT_TRACE=1`
(stampa i ms della ricarica e di ogni passaggio). Criteri: scrittura ≥ 100 t/s a 4K/32K/128K e testo identico a
`Xfus'` (cache uguale, `--adapt-swaps 0`) sul primo turno; a 4K lettura = `Xfus` ± 2 % (sotto la soglia è lo stesso
percorso); a 32K/128K lettura ≥ `X4b` − 3 %; riga `ep-split` con ~20 ms di passaggio finale e `densi ricaricati 1
volte` per richiesta lunga; la riga `the stage's dense weights … live in the peer tier's last S slots` con S ≈ 1300;
nessun `strata.exe` rimasto acceso. Poi la soglia: `X4c` a 8K, 16K, 32K con `_MIN_TOKENS=0` e con il default: il
pareggio misurato fissa il default della prossima build.
