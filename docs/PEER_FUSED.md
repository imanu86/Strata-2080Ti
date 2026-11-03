# PEER_FUSED: il server di esperti della 2080 Ti come un solo kernel persistente per finestra

Ramo `lab/upsync-v0.2.9`, 3/11. Segue `PIPELINE3.md` §3a e `PIPELINE3_PIANO.md` §2-§3 (tappa 0: span GPU del peer a
strato p50 ~290-330 µs, `STRATA_PEER_STAMP`, corse `Xm*`/`Vst*` in `corse_2080ti\20261101_tre_schede`). Tutto dietro
**`STRATA_PEER_FUSED=1`**, spenta di default: spenta, nessun percorso cambia (né kernel, né grafo, né pool).

Contratto comune: `include/strata/core/peer_fused.hpp` (strutture condivise, firme, costanti). Questo documento dice
**come** le cose si incastrano; l'header dice **che cosa** si chiama come. I pezzi di lavoro (§10) scrivono file
disgiunti e usano solo i simboli dell'header.

**Due server** (`--peer-device A,B`: la 2080 Ti più la 3060 sul riser, ciascuna con il proprio kernel persistente):
`PEER_FUSED2.md`. Con un solo device tutto quello che segue vale invariato.

## 0. In una frase

Oggi (`peer_experts.cpp` `launch`/`finish`) ogni strato della finestra costa sul flusso della 2080 Ti: 1-2 copie H2D
(piano + righe x), `hadamard_rows` (se APR), `quantize_q8_1_rows`, `native_expert_grouped` (modo 8: gate/up fuso +
down LDS, 2 lanci), `scatter_rows_f32` verso le righe mappate, con 7-9 operazioni e i loro vuoti di sottomissione sotto
WDDM, e l'host che lancia e poi gira su `cudaStreamQuery`. Con `STRATA_PEER_FUSED=1` la 2080 Ti riceve **un lancio per
finestra**: un kernel persistente (tutti i blocchi residenti) che per ogni passo (strato × gruppo) **aspetta il
doorbell della padrona** (`m_seq_`, la stessa parola mappata che oggi aspetta il filo del pool), **legge x e id dalle
pagine mappate** del verificatore, **pianifica sul device** con la tabella di proprietà, quantizza, calcola gate/up e
down con **gli stessi corpi di kernel di oggi** (`native_gu_fused_body`, `native_down_lds_kernel` resi funzioni
`__device__`), scrive le righe compatte in VRAM e le **sparge nelle righe mappate** `m_ymiss_` con store a 16 byte,
fence di sistema, e alza **un flag d'uscita** che il grafo della padrona aspetta con un nodo `wait_flag_ge` in più.
Host fuori dal percorso: pianifica i miss CPU come oggi, conta le voci del peer per il controllo, fa i flush WDDM.

Righe **identiche bit per bit** al percorso attuale del peer: stesse funzioni, stessi ordini di somma, stessa
quantizzazione, stesso ordine dei gruppi e delle righe compatte (§8).

**Pack misti** (3/11, §5a): il kernel è **uno per tutti i tipi**. Il modello vero (Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS)
cambia tipo gate/up e down **strato per strato**; il kernel legge il layout dello strato da una tabella in VRAM
(`d_lay`, uno per strato) e sceglie il corpo a ogni passo. Prima versione: un'istanza per coppia `(TG, TD)`, e
`fused_open` rifiutava il pack al primo strato (`gate/up type 17`, corsa `Qfus1`).

## 1. Chi parla con chi (per finestra e per passo)

```
3080 Ti (grafo, strato l, gruppo grp)              host (filo del pool)                 2080 Ti (kernel persistente)
------------------------------------------------   ---------------------------------   ------------------------------------
doorbell_publish: x -> m_x_, ids -> m_ids_,
  fence, m_seq_ = ring                        ---> spin su h_seq_ >= ring         ---> blocco 0: spin su seq >= ring (mappata)
                                                   piano kind[] (0 VRAM, 2 peer,        legge ids, pianifica (own + slot_ptr)
                                                     -1 CPU), flag A/B                 lready = ring (VRAM): gli altri partono
wait A, copia piano, hit locali, PCIe              launch(): SOLO conta (groups,rows)   quantize (x mappata -> q8 VRAM)   [barriera]
                                                   CPU: i suoi miss -> m_ymiss_         gate/up fuso -> hq                [barriera]
                                                   finish(): niente (o HOSTWAIT)        down LDS -> out compatta           [barriera]
                                                   h_flag_ = ring                       scatter -> m_ymiss_ (__stwt 16 B)
wait_flag_ge(m_flagS_, ring)   <----------------------------------------------------- fence_system, cnt[s], flag_out = ring
wait_flag_ge(m_flag_, ring)    <--------------------'
copy_rows_from_mapped (salta le righe hit), combine
```

Per finestra: il verificatore, dopo `stage_inputs` (azzera `h_seq_`/`h_flag_`) e **prima** del lancio del grafo,
chiama `hook.begin` → `PeerExperts::fused_begin`: aggiorna la tabella di proprietà sul device, azzera le parole,
lancia il kernel, `cudaStreamQuery` (spinta WDDM). A fine finestra (dopo `cudaStreamSynchronize(cs_)`) chiama
`hook.end` → `fused_end`: attende l'uscita del kernel, controlla la parola d'errore e i conteggi passo per passo,
raccoglie gli stamp. Nel ciclo d'attesa di `run` (il flush di `cs_` ogni 2 ms) chiama `hook.flush` →
`cudaStreamQuery(stream_)` del peer. `release_gpu_waits` alza anche `flag_out` a `UINT32_MAX` e chiama `hook.abort`.

## 2. Memoria mappata (host, `cudaHostAllocPortable | cudaHostAllocMapped`)

Con `--peer-device` le pagine del verificatore sono già Portable (`set_peer_portable`): il peer ne prende i **propri**
puntatori device con `cudaHostGetDevicePointer` sotto `On(device_)`. Niente UVA implicita: ogni puntatore che entra
nel kernel è ottenuto esplicitamente per il device che lo usa.

### 2a. Pagine del verificatore (esistono già; il peer le legge o le scrive, mai le alloca)

| parola / pagina | chi scrive | chi legge | nota |
|---|---|---|---|
| `m_seq_` (uint32) | grafo padrona (`doorbell_publish`: +1 per passo, azzerata dall'host a ogni finestra) | host (pool), **kernel peer (blocco 0)** | ring = `(l - lb) * G + grp + 1` |
| `m_x_` (T × N fp32) | grafo padrona | host, **kernel peer (fase C, `__ldcv`)** | le righe del gruppo da `tb * N` |
| `m_ids_` (T × K int32) | grafo padrona | host, **kernel peer (fase B, `__ldcv`)** | da `tb * K` |
| `m_ymiss_` (T × K × N fp32) | host (righe CPU), **kernel peer (fase F, `__stwt`)** | grafo padrona (`copy_rows_from_mapped`) | riga `j` del gruppo a `(tb * K + j) * N` |
| `m_flag_` | host | grafo padrona | invariato |

### 2b. Pagina del peer (`PeerFusedPage`, `kPageBytes` = 8 KiB, allocata da `PeerExperts::open` se fused)

| offset | nome | tipo | chi scrive | chi legge |
|---:|---|---|---|---|
| 0 | `flag_out` | uint32 | kernel (ultimo blocco del passo: `= ring`), host (`fused_begin`: 0; `release_gpu_waits`: `UINT32_MAX`) | grafo padrona (`wait_flag_ge` con il puntatore device 0), host (HOSTWAIT) |
| 128 | `abort` | uint32 | host (`fused_abort`: 1; `fused_begin`: 0) | kernel (ogni attesa, ogni ~1024 giri) |
| 256 | `err` | uint32 (`peer_fused::Err`) | kernel (primo errore), host (0 a `fused_begin`) | host (`fused_end`, HOSTWAIT) |
| 384 | `exit` | uint32 | kernel (ultimo blocco a uscire: `= epoch`) | host (`fused_end`, diagnostica) |
| 512 | `cnt[kMaxSteps]` | uint32 | kernel (prima del flag: `groups << 16 \| rows`) | host (`fused_end`: contro il conteggio del pool) |
| 2048 | `stamp[2 * kMaxSteps]` | uint64 | kernel (`%globaltimer`: flag visto / flag alzato) | host (`fused_end` → `spans_us_`) |

Ogni parola di sincronizzazione sta su una riga di cache propria (128 B). Il grafo della padrona aspetta `flag_out`
con il **puntatore device 0** (`cudaHostGetDevicePointer` sotto `On(primary_device())`), ottenuto una volta a `open`.

## 3. Memoria del peer (VRAM, allocata da `PeerExperts::open` se fused)

| buffer | dimensione | contenuto | chi scrive |
|---|---|---|---|
| `d_own` | `n_layers × n_expert` int32 | slot del peer o −1: **l'esperto lo calcola il peer** (= `res_ ≥ 0 && host_res < 0`) | host (`fused_begin`: solo gli strati cambiati, da `h_own_` bloccata) |
| `d_slot_ptr` | `cache_.slots()` uint64 | indirizzo device del blob dello slot s (`cache_.device_slot(s)`) | host, a `open` (gli slot non si muovono: `adapt` riusa lo slot della vittima con lo stesso strato) |
| `d_plan[2]` | 2 × `Plan` | il piano del passo s in `d_plan[s & 1]` (doppio buffer: §5) | kernel, blocco 0 |
| `d_q8` | `kMaxTok × N/32` `block_q8_1` | x quantizzata (= `d_q8_` di oggi) | kernel, fase C |
| `d_scratch` | `native_expert_scratch_bytes(kCap, n_ff)` | solo `hq` a `peer_fused::hq_offset(kCap, n_ff)` = `3 × round256(kCap × n_ff × 4)`, **lo stesso offset di `native_expert_grouped`** | kernel, fase D |
| `d_out` | `kCap × N` fp32 | righe compatte (= `d_out_` di oggi) | kernel, fase E |
| `d_sync` | `Sync` (128 B) | contatore/generazione della barriera, `lready`, `done`, `quit`, `exited` | kernel; azzerata dall'host prima del lancio (`cudaMemsetAsync` su `stream_`) |
| `d_lay` | `n_layers × NativeExpertLayout` | il layout di ogni strato (`native_expert_layout(gu_type, d_type, n_embd, n_ff)` di `lay.fmt[l]`): tipi, passo delle righe, offset di up e down | host, a `open` (`fused_lays_`); il kernel legge `lay[l]` a ogni passo con voci (§5a) |

## 4. Tabella di proprietà e rotazioni fra finestre

- **Fonte unica della verità per una finestra**: `own_win_` (host, `std::vector<int32_t>`), calcolata in `fused_begin`
  da `res_` del peer e da `host_res` della padrona (`drive.d.host_res`, `generate.cpp:8694`): `own[i] = res_[i] >= 0 &&
  host_res[i] < 0 ? res_[i] : -1`. La stessa regola di `expert_pool_dispatch_multi` (`expert_source.cpp:4976-4984`:
  prima lo slot della padrona, poi `peer->has`). Con la finestra aperta `PeerExperts::has()` risponde da `own_win_`,
  non da `res_`: il piano dell'host e quello del device leggono la **stessa** tabella anche se `poll_pending` o
  `apply_pending` cambiano `res_` a metà finestra.
- **Invio sul device**: `fused_begin` confronta `own_win_` con l'ombra `own_shadow_` (quel che il device ha) strato per
  strato; gli strati diversi vanno in `h_own_` (bloccata) e con una `cudaMemcpyAsync` per strato (2 KiB) su `stream_`
  **prima** del lancio del kernel. Tipico: 0-17 strati a finestra (gli swap di `adapt`). Mai da `d_res` della padrona,
  che può restare indietro (PIANO §3f).
- **Rotazioni**: `adapt()` segna la vittima −1 subito e accoda la copia del nuovo blob su `refill_`; `apply_pending`
  rende residente il nuovo. Fra `adapt` e `apply_pending` lo slot è in scrittura: `own` lo esclude perché `res_[out]
  = -1`. Con fused, `adapt()` fa nulla (ritorna `true`) se una finestra è aperta (`window_open_`): le copie non
  toccano mai uno slot che il kernel in corsa può leggere. `apply_pending` resta com'è: l'effetto entra nella tabella
  alla `fused_begin` successiva.
- **Prestito del prompt / `res_put`**: cambiano `host_res` fra le finestre → la regola sopra li vede alla finestra dopo.
- **Fuori da una finestra fused** (prompt, `batch_serve`, un `run` senza hook): `has()` risponde da `res_` e
  `launch()`/`finish()` fanno il percorso di oggi (§6). Così nessuna via di chiamata resta senza righe.

## 5. Il kernel (`peer_fused_kernel<MINB>`, 256 thread, tutti i blocchi residenti)

Un ciclo `for (s = 0; s < w.steps; ++s)` identico in ogni blocco; `l = lb + s / G`, `grp = s % G`, `tb = gtb[grp]`,
`n_tok = gte[grp] - tb`, `n = n_tok * k`, `want = s + 1`, `plan = &w.plan[s & 1]`.

| fase | chi | che cosa | sincronizzazione dopo |
|---|---|---|---|
| **A** attesa | blocco 0, thread 0 | spin su `*w.seq >= want` (mappata, `volatile`, `__nanosleep(64)`); ogni 1024 giri controlla `*w.abort` e `%globaltimer - t0 > wait_ns` → `Err`; poi `__threadfence_system()`, `stamp[2s] = now` | — |
| **B** piano | blocco 0 (256 thread) | legge `ids[tb*k .. +n)` con `__ldcv`; `first(i)`, `owned(i)`, gruppo, riga (§8); scrive `plan->ptr/start/dst/tok/out_row/count`; `__threadfence()`; `__syncthreads()`; thread 0: `sync->lready = want` | gli altri blocchi: thread 0 spin su `sync->lready >= want` (VRAM, `__nanosleep(32)`, controllo `quit` ogni 1024 giri), `__syncthreads()` |
| **C** quantize | tutti | per il warp globale `wg` in `[blockIdx*8 + warp; wg < n_tok * N/32; wg += nblocks*8)`: `i = wg*32 + lane`, `q8_1_store(__ldcv(x + tb*N + i), q8, i)` — il corpo di `quantize_q8_1_kernel` (`iq_kernels.cu:2804-2822`) | barriera 1 |
| **D** gate/up | tutti | se `groups == 0` salta a F; `L = w.lay[l]` (VRAM, uniforme); item `it` in `[blockIdx; it < (n_ff/32) * groups; it += nblocks)`: `bx = it % (n_ff/32)`, `g = it / (n_ff/32)`; `switch (L.gu_type)` → `native_gu_fused_core<TG>(plan->ptr, plan->start, &plan->count[0], plan->tok, q8, L, hq, bx, g, griglia, tile, sx)` con TG ∈ {16, 17, 18, 21, 22}; tipo assente → `kErrBadType` | barriera 2 |
| **E** down | tutti | item in `[blockIdx; it < ceil(N/LDS_RB) * groups; ...)`: `bx = it % nbx`, `g = it / nbx`; `switch (L.d_type)` → `native_down_lds_body<TD>(plan->ptr, plan->start, &plan->count[0], plan->dst, hq, L, out, bx, g)` con TD ∈ {42, 20} | barriera 3 |
| **F** scatter + flag | tutti | `idx` in `[tid globale; idx < rows * N/4; idx += nblocks*256)`: `r = idx / (N/4)`, `c = idx % (N/4)`; `__stwt(ymiss4 + (tb*k + plan->out_row[r]) * N/4 + c, out4[r * N/4 + c])`; poi **ogni thread** `__threadfence_system()`, `__syncthreads()`, thread 0 `old = atomicAdd(&sync->done[s & 1], 1)` (**mai azzerato**: un contatore per parità); l'ultimo del passo (`old == ((s >> 1) + 1) * nblocks - 1`): `cnt[s] = groups << 16 \| rows`, `stamp[2s+1] = now`, `__threadfence_system()`, `*flag_out = want` (volatile), `__threadfence_system()` | nessuna: si va al passo s+1 |

**Barriera di griglia** (`grid_barrier`): una sola parola `sync->bar` (conteggio nei 16 bit bassi, generazione nei 16
alti): thread 0 fa `old = atomicAdd(&sync->bar, 1)` e prende la generazione `g = old >> 16` **dal valore restituito**
(un load separato della generazione potrebbe completarsi dopo l'atomic: schema load-buffering, lo stesso che
cooperative_groups evita così); se `(old & 0xFFFF) == nblocks - 1`: `__threadfence()`, `atomicAdd(&sync->bar, (1 <<
16) - nblocks)` (conteggio a 0 e generazione g+1 in un atomic solo); altrimenti spin su `bar >> 16 == g` con
`__nanosleep(32)`, controllando `sync->quit` ogni 256 giri, `*w.abort` (mappata) e il tempo (`wait_ns`) ogni 4096.
`__syncthreads()` prima e dopo. `nblocks < 65536` (controllo della finestra). Tre barriere a passo; nessuna fra F e il
passo successivo; **nessuna** nei passi con `groups == 0`.

**Perché basta** (nessuna corsa fra passi): un blocco veloce al passo s+1 scrive `d_q8` (fase C) mentre uno lento
finisce F(s), che legge solo `d_out` e `d_plan[s&1]`; `hq` e `d_out` del passo s+1 si scrivono dopo le barriere 1 e 2
di s+1, che i blocchi lenti raggiungono solo dopo F(s); `d_plan[(s+2)&1]` lo scrive il blocco 0 solo al passo s+2.
Con G=2 il grafo della padrona suona il doorbell di s+1 **prima** di aspettare flag(s) (`record_window`: `pre(l,0)`,
`pre(l,1)`, `post(l,0)`, `pre(l+1,0)`, `post(l,1)`, …): un blocco può quindi stare al passo s+1 mentre altri sono
ancora in F(s), e se s+1 non ha voci (nessuna barriera) i due passi si sovrappongono fino al conteggio di F. Per
questo i contatori `done` sono **due, uno per parità, e non si azzerano mai**: l'ultimo blocco del passo s è quello il
cui atomic restituisce `((s >> 1) + 1) * nblocks - 1`, e nessun incremento di s+1 può confondersi con quelli di s. Il
doorbell di s+2 arriva solo dopo flag(s), cioè dopo **tutti** gli incrementi di s: nessun blocco è mai due passi
avanti, quindi `done[s & 1]` al passo s+2 riparte dopo che s è completo e `d_plan[s&1]` è libero quando il blocco 0 lo
riscrive. Due passi consecutivi possono alzare `flag_out` in ordine inverso (s+2 prima di s+1): innocuo, ogni valore
scritto significa «tutti i passi sotto sono completi» e le attese sono `>=`.

**Gruppi vuoti**: `groups == 0` (il peer non ha voci in questo passo): niente C-E, F scrive `cnt[s] = 0` e alza
comunque `flag_out = want` (il grafo lo aspetta a ogni passo). Oggi `launch()` ritorna senza lavoro e il grafo non
aspetta nulla: con fused il nodo c'è sempre, il flag pure.

**Uscita** (`peer_fused::Err`): chi rileva l'errore (timeout su seq, su `lready`, su una barriera, `abort`) scrive
`*w.err` se ancora 0 (atomicCAS su parola mappata non serve: la prima scrittura vince, le altre sono uguali o
posteriori) e `sync->quit = 1`; ogni attesa del kernel controlla `quit` e `*w.abort` (la barriera e l'attesa di
`lready` ogni 4096 giri, il doorbell ogni 1024 e **una volta prima dello spin**, a ogni passo: una padrona rilasciata
corre il grafo fino in fondo e suona tutti i doorbell rimasti, il kernel non deve servirli) ed esce dal ciclo dei
passi; ogni blocco che esce fa `atomicAdd(&sync->exited, 1)`, l'ultimo scrive `*w.exit = epoch` con fence di sistema
e, se `quit != 0`, **`*flag_out = UINT32_MAX`**: i nodi `wait_flag_ge` della padrona (senza limite di tempo) passano
qualunque passo aspettino, l'host trova `err != 0` a `fused_end` e la finestra fallisce; senza questo un timeout di
barriera sugli ultimi passi terrebbe ferma la padrona (e l'host in `cudaStreamSynchronize`) fino a un release esterno.
Un passo **non** alza mai il proprio flag dopo un errore. `release_gpu_waits` alza `flag_out = UINT32_MAX`, chiama
`hook.abort` (che drena) e lo **rialza**: un passo in volo può aver scritto il suo ring sopra il primo MAX.

**Memoria condivisa** (sm_75, 64 KiB per SM): il corpo gate/up (`native_gu_fused_core`) riceve i suoi tre buffer come
argomenti e il kernel glieli ritaglia dal buffer dinamico: `[attivazioni LDS_NT × N/32 × 36 = 11 520 B | tile
res[4][64] 1 024 B | griglia del tipo]`; la griglia è dimensionata sulla **più grande fra i tipi del pack**
(`Shape::grid_bytes`: IQ2_XXS 2 KiB, IQ2_XS 4 KiB, IQ3_XXS 1 KiB, IQ3_S 2 KiB, IQ2_S 8 KiB). Pack misto di Flash-Next:
11 520 + 1 024 + 8 192 = 20 736 B a blocco (3 blocchi a SM per memoria, 2 per registri a MINB 2); pack solo IQ3_XXS:
13 568 B (4 a SM). Se ogni corpo tenesse la propria griglia statica, cinque griglie + cinque tile sarebbero ~22 KiB
sempre allocati per un corpo solo in uso: per questo il corpo è stato diviso in `native_gu_fused_body` (dichiara le
statiche come prima, per i kernel per-strato) e `native_gu_fused_core` (il lavoro, buffer dall'esterno). Il corpo down
usa la sua `extern __shared__` (2 880 B) sullo stesso buffer, dopo la barriera 2.

**Registri e blocchi** (`Shape`): `__launch_bounds__(256, MINB)`, MINB 2 di default (≤ 128 registri, senza spill: i
corpi stanno a 64-78), `STRATA_PEER_FUSED_MINB=3|4` sceglie le istanze più strette. `peer_fused_shape` chiede
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` per l'istanza scelta e la memoria dinamica; `blocks = min(per_sm,
STRATA_PEER_FUSED_BLOCKS o per_sm) × SM` (2080 Ti: 68 SM → 136 a MINB 2). Il lancio è `cudaLaunchCooperativeKernel`
quando `cudaDevAttrCooperativeLaunch` lo consente (co-residenza garantita o il lancio fallisce con un errore chiaro),
altrimenti un lancio normale con la stessa griglia; `STRATA_PEER_FUSED_COOP=0` forza il lancio normale. La barriera è
sempre la nostra (nessun `-rdc`, nessun `cooperative_groups::grid_group`).

**Istanze compilate**: una per MINB (`peer_fused_kernel<2>`; `<3>` e `<4>` solo con la macro di compilazione
`STRATA_PEER_FUSED_MINB_ALL`). L'istanza contiene i **cinque** corpi gate/up (IQ2_XXS 16, IQ2_XS 17, IQ3_XXS 18, IQ3_S
21, IQ2_S 22) e i **due** corpi down (Q2_0 42, IQ4_NL 20): il tipo lo decide lo strato, non l'istanza. Un pack con un
tipo fuori da questi (IQ4_XS 23, nibble, APR) è rifiutato da `peer_fused_shape` con lo strato nel messaggio e il peer
resta sul percorso di oggi con una riga su stderr.

### 5a. Pack misti: il tipo è dello strato

Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS (`native_experts.txt` del pack `iq3_xxs`, 48 strati di esperti, n_embd 2560,
n_ff 640, tutti con le stesse geometrie):

| gate/up \ down | Q2_0 (42) | IQ4_NL (20) | strati |
|---|---:|---:|---:|
| IQ2_XXS (16) | 8 | 1 | 9 |
| IQ2_XS (17) | 6 | 4 | 10 |
| IQ3_XXS (18) | 3 | 3 | 6 |
| IQ3_S (21) | 7 | 6 | 13 |
| IQ2_S (22) | 6 | 4 | 10 |
| **strati** | **30** | **18** | **48** |

Per strato: `0 17/20 · 1 16/42 · 2 17/20 · 3 16/42 · 4 17/20 · 5 17/42 · 6 16/42 · 7 16/42 · 8 21/42 · 9 17/42 · 10 17/42
· 11 16/42 · 12 22/42 · 13 16/42 · 14 17/42 · 15 22/42 · 16 16/42 · 17 22/42 · 18 21/42 · 19 17/42 · 20 21/42 · 21 22/42
· 22 21/42 · 23 21/42 · 24 21/42 · 25 17/42 · 26 22/42 · 27 16/42 · 28 21/20 · 29 22/20 · 30 21/20 · 31 16/20 · 32 21/20
· 33 21/42 · 34 21/20 · 35 18/20 · 36 18/42 · 37 21/20 · 38 22/20 · 39 17/20 · 40 22/42 · 41 18/42 · 42 22/20 · 43 21/20
· 44 18/20 · 45 18/42 · 46 22/20 · 47 18/20`. Dieci coppie diverse, **nessuna** copre più di 8 strati: un'istanza per
coppia non avrebbe mai acceso il kernel (al massimo 7-8 strati su 48 anche con una modalità mista kernel + percorso
per-strato nella stessa finestra). Con il tipo per strato il kernel copre **48 strati su 48**.

Che cosa serviva: (1) un prodotto IQ2_XS con la griglia in LDS (`DotG<17, SG>` e `GridOf<17>`, `iq_kernels.cu`: gli
stessi interi di `vec_dot_iq2_xs_q8_1`, codici a 9 bit, segni a 7 bit + parità come IQ2_XXS, scale a due metà come
IQ2_S); (2) il tipo 17 nella lista del gate/up fuso del **modo 8** (`fused_gu`, `iq_kernels.cu`), così il percorso
per-strato del peer **e** il kernel fanno la stessa cosa per quegli strati; (3) `Window::lay` / `n_layers`, `Shape`
con le maschere dei tipi (`gu_mask`, `d_mask`) e `grid_bytes`, `peer_fused_shape(lay, n_layers, …)` che controlla ogni
strato; (4) `fused_open` che costruisce `fused_lays_` e `d_lay_` invece di pretendere un'istanza sola.

**Attenzione (aritmetica)**: in modo 8 il percorso per-strato di IQ2_XS era `native_gu_multi_kernel<17>` (righe a 16
lane, `row_dot_80_sub16`) + `swiglu_q8_1_entries_kernel`; ora è il gate/up fuso (somme a 32 lane + `warp_sum`, SwiGLU e
q8_1 nel warp), come già per 16/18/21/22. Stessi prodotti interi, **altro ordine di somma in float**: i 10 strati IQ2_XS
in modo 8 non sono bit per bit quelli di prima — su **entrambe** le schede, perché `STRATA_EXP_MODE=8` è globale
(`config.json` della corsa `Qfus1`). Il modo 0 (default) non cambia. Il banco confronta kernel e percorso per-strato
**nello stesso modo**: per il 17 verifica l'impianto (layout per strato, griglia, dispatch), non il confronto con il
multi-kernel. Se si vuole l'identità con il vecchio percorso del 17, serve un corpo a 16 lane nel kernel (non fatto:
più codice, nessun guadagno).

## 6. Host: `PeerExperts` in modalità fused (pezzo B)

- `open()`: se `fused_env()` e `lay.native`: per ogni strato `native_expert_layout(f.gu_type, f.d_type, ...)` →
  `fused_lays_[l]`; poi **una** chiamata `peer_fused_shape(fused_lays_.data(), n_layers, minb, blocks_cap, coop, shape,
  why)` (sotto `On(device_)`) che controlla ogni strato (tipi fra quelli del kernel, stessa `n_embd`/`n_ff`, niente
  APR/nibble/tail) e il dispatch del device; rifiuta anche `had_gu`/`had_d` (APR: `hadamard_rows` non è nel kernel). Se
  qualcosa manca: stderr `strata: STRATA_PEER_FUSED: … (<why>): the peer keeps the per-layer path` e `fused_on_ =
  false`. Se tutto va: alloca pagina e buffer (§2b, §3), copia `fused_lays_` in `d_lay_`, riempie `d_slot_ptr`, calcola
  i due puntatori device di `flag_out` (peer e padrona), stampa una riga con i tipi del pack (`{16 17 18 21 22}` /
  `{20 42}`), blocchi, memoria dinamica e griglia, cooperative sì/no.
- `fused_hook()` → `peer_fused::Hook` (membro, puntatore stabile) con `ctx = this`, le quattro funzioni statiche,
  `flag_out_host`, `flag_out_dev0` (nullo con `STRATA_PEER_FUSED_HOSTWAIT=1`: il grafo non aspetta, aspetta
  `finish()`), da passare a `Verifier::set_peer_fused` in `generate.cpp` dove si costruisce il verificatore del decode
  (`ver`), dopo `drive.d.host_res` (`8694`) e prima del primo `ver.run`. `fused_set_host_res(host_res.data())`.
- `fused_begin(const WindowHost& w, err)`: (1) `cudaStreamQuery(stream_)` deve dire `cudaSuccess` (il kernel della
  finestra prima è uscito) altrimenti errore `peer fused: the previous window's kernel is still running`; (2) tabella
  §4 e invio delle righe cambiate; (3) azzera `flag_out`, `err`, `abort`, `exit`, `cnt[0..steps)`; `cudaMemsetAsync(d_sync)`;
  (4) risolve i puntatori device del peer per `w.seq_h`, `w.x_h`, `w.ids_h`, `w.ymiss_h` (cache per indirizzo: cambiano
  solo se il verificatore rialloca); (5) riempie `peer_fused::Window` (geometria, puntatori, `lay = d_lay_`,
  `n_layers`: il layout è **per strato**, il kernel legge `lay[l]` a ogni passo; `fused_begin` verifica che
  `lay.fmt[l]` sia ancora quello di `fused_lays_[l]` per `l ∈ [lb, le)`, altrimenti errore); (6) `++epoch_`,
  `peer_fused_launch(window, shape, stream_, err)`, `cudaStreamQuery(stream_)`; `window_open_ = true`, `step_ = 0`.
- `has(layer, e)`: `window_open_ ? own_win_[idx] >= 0 : res_[idx] >= 0`.
- `launch(...)` con `window_open_`: calcola `groups`/`rows` **esattamente** come oggi (stesso ciclo su `kind == 2`,
  stessi contatori `experts_`, `entries_`, `launches`, `row_of_`), salva `expect_[step_] = groups << 16 | rows`,
  `launched_direct_ = true`, `launched_rows_ = rows`, `++step_`; **nessuna** chiamata CUDA. `kind == 2` vale solo
  per voci con `has()` vero, quindi l'host e il device contano le stesse voci (§8).
- `finish(out, err)` con `window_open_`: se HOSTWAIT: spin su `flag_out >= step_` (mappata, volatile) con il limite
  `wait_ns` e controllo di `err`/`abort`; altrimenti ritorna subito. `launched_rows_ = 0`.
- `fused_flush()`: `cudaStreamQuery(stream_)` senza cambio di device (come `noswitch()`; se rifiutata, sotto `On`).
- `fused_end(err)`: attende `cudaStreamQuery(stream_) == cudaSuccess` (limite 5 s con `sleep 1 ms`); `window_open_ =
  false`; se `*err != 0` → messaggio con il codice e il passo; se `*exit != epoch_` → errore; per ogni passo
  `cnt[s] != expect_[s]` → `peer fused: plan mismatch at step s (layer l): device g×r, host g×r` (divergenza visibile,
  non testo diverso); stamp → `spans_us_` (µs a passo), `span_us_total`.
- `fused_abort()`: `*abort = 1`, poi drena `stream_` con `cudaStreamQuery` fino a 5 s; da `release_gpu_waits` (via hook)
  e da `close()`. Se il drain non ce la fa, `window_open_` **resta vero** (il kernel vivo legge ancora gli slot:
  `adapt` deve restarne fuori) e lo abbassa la `fused_begin` successiva quando trova lo stream vuoto; lo stesso in
  `fused_end`. `window_open_` è `std::atomic<bool>`: lo scrive anche il filo del watchdog di stallo.
- `adapt()`: con `window_open_` ritorna `true` senza fare nulla.
- Knob (tutti letti una volta): `STRATA_PEER_FUSED`, `STRATA_PEER_FUSED_HOSTWAIT`, `STRATA_PEER_FUSED_MINB`,
  `STRATA_PEER_FUSED_BLOCKS`, `STRATA_PEER_FUSED_COOP`, `STRATA_PEER_FUSED_WAIT_MS` (default 20 000).
- La riga `strata decode peer:` (`generate.cpp:18977`) stampa, con fused, `fused` e i passi serviti; gli span sono
  quelli degli stamp del kernel (flag visto → flag alzato: il lavoro vero, senza i vuoti di sottomissione).

## 7. Padrona: il grafo e `run` (pezzo C)

- `Verifier::set_peer_fused(const peer_fused::Hook*)` prima di `init` (come `set_handoff_flags`). Con `next_ != nullptr`
  (divisione a strati) o `device_plan_` il hook è rifiutato con un errore a `init` (il peer rifiuta già
  `--layer-split`).
- `capture` (lambda `post`, `verify.cpp:2255`, ramo non `device_plan_`): subito **prima** dell'attesa di `m_flag_`
  (sia `wait_flag_ge` che `copy_rows_from_mapped_after`): `if (hook && hook->flag_out_dev0) wait_flag_ge(hook->flag_out_dev0, ring, cs)`.
  Un nodo in più a passo; `copy_rows_from_mapped` resta quello di oggi (`p_dst` salta le righe hit della padrona).
- `run` (`3095`): dopo `stage_inputs` (`3143`, che azzera `h_seq_`) e prima di `launch_o.emplace` (`3190`): se `hook`
  e non `ar_on()`: `WindowHost w{T, G, lb_, le_, k, n_embd, gtb, gte, h_seq_, h_x_, h_ids_, h_ymiss_}` → `hook->begin`;
  errore → `return false` **dopo** aver chiamato `hook->abort` (niente kernel orfano). Nel ciclo d'attesa (`3246`,
  ogni 2 ms accanto al flush di `cs_`): `hook->flush`. Dopo `cudaStreamSynchronize(cs_)` (`3391`) e il controllo di
  `handoff_failed`: `hook->end`; errore → `return false`. Sulle uscite anticipate dopo il lancio (timeout, `NEVER-RANG`,
  errore di una layer): `release_gpu_waits` copre (sotto).
- `release_gpu_waits` (`609`): `hook->flag_out_host` nella lista delle parole alzate a `UINT32_MAX` (prima del
  drain), poi `hook->abort(ctx)` (che drena lo stream del peer dentro i 5 s), poi il drain di `cs_`/`copy_` come oggi.
- `ar_on()`/`batch_serve`/`run_slots`: nessuna chiamata al hook (il peer serve quelle vie come oggi, §4 ultimo punto).

## 8. Bit-identità: perché le righe sono le stesse

| passo | oggi (`peer_experts.cpp` `launch`) | fused | uguale perché |
|---|---|---|---|
| piano | ciclo su `i`: salta se `kind[i] != 2` o già visto (`kind[j]==2 && ids[j]==ids[i]`, j<i); gruppo = ordine di prima apparizione; righe del gruppo = le sue voci in ordine di `i`; `dst[r]=r`, `tok[r]=j/k`, `out_row[r]=j` | fase B: `owned(i)`, `first(i)` = min j ≤ i con `ids[j]==ids[i]`; gruppo di `i` = rango di `first(i)` fra i `first` posseduti; `start[g]` = somma delle taglie dei gruppi precedenti; riga di `i` = `start[g] + #{j<i : first(j)==first(i)}` | `kind[i]==2 ⇔ owned(i)` (§4, stessa tabella `own_win_`); `kind` è per esperto, quindi «già visto fra i kind 2» = «non è il primo del suo esperto» |
| x | `cudaMemcpyAsync` di `n_tok × N` fp32 da `h_x_ + tb*N` (o da `x`) → `d_x_` | letta con `__ldcv` dalla stessa pagina | gli stessi 4 byte |
| quantize | `quantize_q8_1_kernel`: thread `i` → `q8_1_store(x[i], y, i)`, un warp = un blocco q8_1 | fase C: il warp globale `wg` fa gli stessi 32 `i` con lo stesso `q8_1_store` | stesso codice, stesse 32 lane per blocco |
| gate/up | `native_gu_fused_kernel<TG>` (`<<<(n_ff/32, cap_groups), 256, sh>>>`): blocco `(x, y)` = righe `x*32..`, gruppo `y`, TG = `L.gu_type` dello strato (16/17/18/21/22 in modo 8) | fase D: `native_gu_fused_core<TG>(…, bx, g, griglia, tile, sx)` con TG = `lay[l].gu_type`, sugli stessi `(bx, g)`, ogni item fatto da un solo blocco | il corpo è lo stesso con `blockIdx` sostituito da parametri e i buffer condivisi passati invece che dichiarati; nessuna somma attraversa i blocchi |
| down | `native_down_lds_kernel<TD>` (`<<<(ceil(N/64), cap_groups), 256, sh>>>`), TD = `L.d_type` | fase E: `native_down_lds_body<TD>(…, bx, g)` con TD = `lay[l].d_type` | idem |
| scatter | `scatter_rows_kernel`: `dst[rows[r]*w4 + i] = src[r*w4 + i]` | fase F: `__stwt` dello stesso `float4` nello stesso posto | i bit non cambiano col tipo di store |
| riga vuota | il pool non azzera le righe `kind == 2`; il peer le scrive | idem; in più `cnt[s]` contro il conteggio dell'host (`fused_end`) | una voce non scritta è un errore, non un testo diverso |

Non c'è `hadamard_rows` (APR): fused rifiutata con `had_gu`/`had_d`. Non ci sono i rami `v1`/`v2`/`v2k`/`expdec`/
`gu_swiglu`/`nib`/`down_ws`: `peer_fused_shape` risolve la stessa decisione di `native_expert_grouped` (`iq_kernels.cu`
da `:6334`, `fused_gu` e `lds_down_ok`) e rifiuta ogni altro esito, così «fused» esiste solo dove il percorso di oggi è
esattamente «gate/up fuso modo 8 + down LDS».

Prova: `tools/peer_fused_test.cu` (pezzo D) sullo stesso device confronta `memcmp` le righe del kernel con quelle di
`quantize_q8_1_rows` + `native_expert_grouped` (con il layout **dello strato**) + `scatter_rows_f32` su blob sintetici
con struttura valida, su più passi, gruppi vuoti compresi, con un «doorbell» fatto dall'host **nell'ordine del
grafo** (i primi G passi suonati subito, poi flag(s) → doorbell di s+G) e, con G=2, passi vuoti forzati dietro passi
con voci; con due device opzionalmente la padrona finta è un kernel sull'altra scheda. Di default il modello del
banco è **misto**: le dieci coppie `(gate/up, down)` del pack vero, uno strato ciascuna nell'ordine di prima
apparizione (`17/20 16/42 17/42 21/42 22/42 21/20 22/20 16/20 18/20 18/42`), slot per strato nel layout dello strato:
una finestra attraversa ogni corpo e ogni cambio di tipo fra due passi consecutivi. `peer_fused_test 0 18 42` resta
il modello uniforme di 4 strati.

I corpi riusati (`native_gu_fused_core`, `native_down_lds_body`) **non** hanno `__restrict__` sui parametri: nel kernel
persistente piano, `q8` e `hq` vengono riscritti fra un passo e l'altro dello stesso lancio, e un ingresso `const
__restrict__` autorizza `ld.global.nc` (cache di sola lettura, non coerente dentro il lancio, che `__threadfence` non
invalida). I kernel `__global__` di oggi tengono il loro `__restrict__`: dopo l'inline le informazioni di non-alias
degli argomenti del kernel sono le stesse, il codice non cambia.

## 9. Tenuta

| caso | che cosa succede |
|---|---|
| la padrona non suona più (errore, timeout, finestra abbandonata) | il kernel sta nell'attesa A; `release_gpu_waits` alza `flag_out` (sblocca il grafo) e `hook.abort` alza `abort`: il kernel esce entro un giro di polling; `fused_abort` drena `stream_` (≤ 5 s). Senza release, il limite di `wait_ns` (20 s, come l'host: §3d del PIANO; i 200 ms del banco sono falsi positivi) fa uscire il kernel con `kErrTimeoutSeq`. |
| un blocco non arriva alla barriera (bug, SM occupato) | `kErrTimeoutBarrier` dopo `wait_ns`; tutti escono via `quit`; l'ultimo a uscire scrive `flag_out = UINT32_MAX`, il grafo della padrona passa da solo e `fused_end` riporta l'errore (righe scartate). Un `abort` dell'host arriva anche a chi è fermo alla barriera o su `lready`, non solo al blocco 0. |
| `fused_end` trova il kernel ancora in corsa | 5 s di `cudaStreamQuery`, poi errore; `fused_begin` della finestra dopo rifiuta finché lo stream non è vuoto. |
| conteggio diverso (`cnt[s] != expect_[s]`) | errore della finestra con strato e numeri: tabella o piano divergono. |
| WDDM | `cudaStreamQuery` subito dopo il lancio e a ogni flush del ciclo d'attesa (`hook.flush`): lo stream del peer non resta nel buffer del driver. |
| `close()` | `fused_abort()` prima di distruggere lo stream; mai `cudaStreamDestroy` con un kernel vivo. |
| spenta | `fused_on_ = false`: nessun hook, nessuna allocazione, nessun nodo nel grafo, `has/launch/finish` di oggi. |

## 10. Pezzi (file disgiunti, solo scrittura; compila, prova e salva la chat principale)

| pezzo | file | simboli del contratto |
|---|---|---|
| **A kernel** | `src/kernels/cuda/peer_fused.cuh` (nuovo), `src/kernels/cuda/iq_kernels.cu` | `peer_fused_kernel<MINB>`, `strata::kernels::peer_fused_shape/peer_fused_launch`, `native_gu_fused_core`/`native_down_lds_body` con `(bx, g)`, `DotG<17>`/`GridOf<17>` |
| **B host** | `src/core/peer_experts.cpp`, `include/strata/core/peer_experts.hpp`, `src/program/generate.cpp` | `PeerExperts::fused_*`, `peer_fused::Hook`, `Window`, `Err`, knob |
| **C padrona** | `src/core/verify.cpp`, `include/strata/core/verify.hpp` | `Verifier::set_peer_fused`, `peer_fused::Hook`, `WindowHost` |
| **D banco** | `tools/peer_fused_test.cu` (nuovo) | `Window`, `Plan`, `Sync`, `hq_offset`, `q8_bytes`, `peer_fused_shape/peer_fused_launch` |
| **E CMake** | `CMakeLists.txt` | il bersaglio `peer_fused_test` |

Il `.cuh` di A è incluso **dentro** `iq_kernels.cu` (una riga, prima di `}  // namespace strata::kernels` a
`:6524`): il kernel deve vedere `Fmt`, `DotG`, `GridOf`, `warp_sum`, `q8_1_store`, `LDS_NT`/`LDS_RB`, le griglie IQ e
i due corpi, tutti nello spazio anonimo di quel file; niente `-rdc`, nessun duplicato delle tabelle.

## 11. Non provato

Il kernel a istanza unica `(TG, TD)` è passato al banco bit per bit (3/11: 20 finestre, 970 voci identiche, ~85 µs a
strato a T=3-4, dati sintetici 18/42) ma nel motore vero non si è mai acceso (pack misto, §5a). La versione a tipo
per strato (§5a: `DotG<17>`, `native_gu_fused_core`, `Window::lay`, `Shape` a maschere, banco a modello misto) è
**scritta e non compilata**: compilato no, installato no, provato no. Punti dove la compilazione può fermarsi:
`native_gu_fused_core` (il parametro `float (*res)[64]` e `typename GridOf<TG>::T*`), la lambda generica `run` dentro
`peer_fused_kernel` (cattura `pf_dyn` e la `L` locale), `pf_sx_bytes` dichiarata `__host__ __device__ constexpr` con
`sizeof(block_q8_1)`, `kExpLds16` vista dal `.cuh`, `pf::mask_text` in `peer_experts.cpp`.

Le stime di guadagno restano quelle di `PIPELINE3_PIANO.md` §1:
il banco `ep_bench` dà 98-113 µs esposti per la forma persistente con letture di banda; i corpi veri costano quel che
costano (120-230 µs di attesa in `finish` oggi, stimato come limite inferiore del lavoro GPU puro). Il kernel toglie
copie, lanci, vuoti e l'host: quanto resta lo dice `spans_us_` di `fused_end` nella riga `strata decode peer:`.
