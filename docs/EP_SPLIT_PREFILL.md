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
