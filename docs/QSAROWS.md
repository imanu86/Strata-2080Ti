# QSAROWS: le T righe di una finestra nei kernel QSA del decode

Branch `lab/qsarows` del fork `imanu86/Strata-2080Ti`, da `lab/ports1` (`295d6e7f`: `lab/upsync` = upstream v0.1.42 più
le patch del Daily, `STRATA_CP_*` di `docs/CRITPATH.md`, `STRATA_CHAIN_FUSE2` di `docs/CHAINFUSE.md`, port #1437/#1889).
Configurazione del Daily: 3060 = stadio 0 (layer 0-15), 2080 Ti = stadio 1 (layer 16-47, testa, drafter MTP),
`--spec 4 --kv int8 --kv-resident 32768 --max-context 262144 --mtp rt_it48k`, chat singola a 131k.

Stato: vedi «Stato della build» in fondo. Nessun motore avviato; micro-bench sì (`qsa_rows_bench`, sotto).

## 0. L'ipotesi e il verdetto

Ipotesi (dal fronte `draftwin`): una riga in più nella finestra costa 5,7 ms nelle fresche (≈2,5 sullo stadio 0 e
2,5-2,9 sullo stadio 1); gli esperti in più spiegherebbero solo ~0,6 + 0,45 ms; se l'attenzione QSA del decode lavora
riga per riga rileggendo K/V per ogni riga, 8 layer × ~0,25 ms sullo stadio 1 e 4 × ~0,25 sulla 3060 spiegherebbero
quasi tutto, e anche il rallentamento da 40-70k a 131k.

**Verdetto: falsa.** La struttura «una riga = un lancio su tutte le celle» esiste solo nel drafter MTP, che però
attende sempre con **una** query (`record_rest`: `const int T = 1`), e i ~240 µs vengono dal suo window **denso** di
32 768 celle. Nella verifica (i 12 layer QSA del modello) ogni riga attende sulla **propria selezione di 2051 celle**
(`idx_top_k` 2048 + 3): 33 chunk × 2 teste K/V = 66 blocchi per riga, 2 MB di K/V per riga, già in L2 fra le righe. Il
kernel è legato alla latenza dei suoi blocchi, non alla banda, e le T righe sono blocchi in parallelo (`blockIdx.z`),
non lanci in sequenza. I numeri sono nel §2: l'intero layer QSA (append dell'indicizzatore, punteggi, top-k,
attenzione) costa per riga in più **29 µs sulla 2080 Ti** (22 con `STRATA_ATTN_DAILY=1`) e **88 µs sulla 3060** (70),
cioè **0,23 ms** (0,18) sugli 8 layer dello stadio 1 e **0,35 ms** (0,28) sui 4 dello stadio 0, contro i **3,0 e 2,8 ms**
misurati per riga sui due stadi (§3): l'attenzione QSA e la selezione spiegano **l'8% circa** del costo per riga, non il
grosso; l'ipotesi presupponeva 2,0 + 1,5 ms. Il costo per riga sta altrove (§3): esperti, GDN, hyper-connection, host.
Il kernel multi-riga (§4) non è stato portato: anche azzerando del tutto il termine per riga dell'attenzione si
guadagnerebbero al massimo 0,1-0,2 ms per riga e stadio.

## 1. Come il codice tratta le T righe (verifica nel sorgente)

Tutto in `src/core/verify.cpp` (layer QSA di una finestra di `n` = T righe, `tb..te`) e nei kernel richiamati:

| passo | chiamata | come tratta le T righe | K/V o chiavi letti |
|---|---|---|---|
| chiave dell'indicizzatore | `native_qsa_indexer_append_steps(raw, step+pos, 4, n, …)` (`native_qsa_indexer.cu`) | **un blocco**, le n celle in sequenza dentro il kernel (`for step_idx`); pooling al completamento del blocco di 4 | niente |
| punteggi dei blocchi | `qsa_block_scores(pooled, dead, qidx, steps, n, max_blocks_, …)` senza `active_blocks` → `block_scores_multi_kernel` (`qsa_select.cu` §810) | 256 CTA a passo fisso sui blocchi; **ogni chiave pooled letta una volta** per tutte le n query (`for qi < nq` sul `k4` già in registri); le query in shared | pooled `n_bid+1` × 128 × 4 B = **16,8 MB a 131k, una volta** |
| top-k | `qsa_block_topk(scores, steps, n, max_blocks_ (65538), cap_ 2051, …)` → `max_blocks > fit` (33 792) → `block_topk_wide_kernel` | **un CTA per query**, 1024 thread, 4 passate radix sui 32 769 punteggi; le n query in parallelo su n SM | i punteggi (128 KB per query) 4+2 volte |
| residenza | `qsa_kv_resolve(st, g, sel, steps, n, cap_)` | le n selezioni insieme | pagine mancanti |
| attenzione | `qsa_decode_attn_batch(qcur, pools, sel, steps, cap_ 2051, …, n)` → `launch_chunks_batch` griglia **(33 chunk, 2 teste K/V, n)** + merge (24, n) | **blocco per (chunk, testa, riga)**: `blockIdx.z` = riga con `ids + z*cap`, `step + z*4`; K/V della riga letti dal suo blocco | 2051 × 2 teste × 512 B = **2 MB per riga** (int8 K+V+scale); le righe condividono ~75-85% delle celle (§2) e stanno in L2 |

Quindi: nessun kernel per riga; il K/V viene letto da ogni blocco di ogni riga ma la quantità è di 2 MB per riga e per
layer e resta in L2 (5,5 MB sulla 2080 Ti, 3 MB sulla 3060); la selezione è già «multi-riga» (chiavi lette una volta).
Con `STRATA_ATTN_DAILY=1` il kernel dei chunk è `attn_chunk_kernel_pre75` (73-78 registri, 3 blocchi per SM): 66 × T
blocchi contro 204 slot sulla 2080 Ti (68 SM) e 84 sulla 3060 (28 SM) — sulla 3060 già T = 2 è una seconda ondata.
Senza la variabile il kernel di base (37-41 registri, 4 blocchi/SM per la shared da 15,9 KB: 272 e 112 slot).

**Il drafter** (`src/core/mtp.cpp`): il round di catch-up registra `record_front(T, …)` per i K/V delle T celle e
poi `record_rest` **per la sola riga a** (`copy_row_to_first`, `const int T = 1`): l'attenzione densa sul window di
32 768 celle (`window_ids` + `qsa_decode_attn_batch(…, T=1)`) gira una volta per round e una per passo. I 240 + 50 µs
di `docs/CHAINFUSE.md` sono questo kernel a una query; non si moltiplicano per le righe della finestra.

**Perché a 131k si va più piano che a 40-70k, allora.** Dalla parte QSA cresce solo il lavoro per *finestra* (non per
riga): i punteggi leggono `n_bid` chiavi (16,8 MB a 131k contro 5-9 MB) e il top-k passa 4 volte su 32 769 punteggi;
il resto (K/V 2051 celle) è costante. Il grosso del rallentamento è fuori dal QSA (esperti ed EV della cache, §3).

## 2. Misure: `qsa_rows_bench`

Target CMake fuori dal default (solo CUDA), stessa procedura di `chain_step_bench`:

```
python ...\20261006_due_schede\build_lock.py C:\Users\imanu\source\repos\Strata-wt\qsarows --target qsa_rows_bench
build-sm7586\qsa_rows_bench.exe --device 0            # la 2080 Ti (senza CUDA_VISIBLE_DEVICES)
build-sm7586\qsa_rows_bench.exe --device 1            # la 3060
STRATA_ATTN_DAILY=1 build-sm7586\qsa_rows_bench.exe …  # con i kernel del Daily
```

Dati sintetici alla geometria reale (24 teste × 256, 2 teste K/V, indicizzatore 4 × 128, blocchi di 4, top-k 2048),
pool int8 con tabella delle pagine identità, contesto 131 072 (righe alle celle 131 072..131 075), capacità 262 144 come
il Daily (`max_blocks` 65 538, `cap` 2051). Le query dell'indicizzatore delle 4 righe sono una query più un rumore
piccolo, così le selezioni si sovrappongono come in una finestra vera (il bench stampa l'unione). Per T = 1..4 misura
(eventi CUDA, mediana di tre giri da `--iters` chiamate) i quattro kernel del layer e la loro sequenza, più il caso
denso a 32 768 celle per riga (la forma da cui veniva il «~240 µs a query»). La GPU va presa con `bench.lock`.

Misurato il 12/10 (build `5dffa63f`, `--iters 100`, GPU libera, `bench.lock` preso). Unione delle selezioni delle righe:
2 righe 2474 celle (1,21× una riga), 3 righe 2759 (1,35×), 4 righe 2903 (1,42×). µs per chiamata:

### 2080 Ti (device 0)

| kernel | T=1 | T=2 | T=3 | T=4 | per riga |
|---|---|---|---|---|---|
| `qsa_block_scores` (multi) | 47,3 | 57,1 | 80,6 | 103,0 | +18,6 |
| `qsa_block_topk` (wide) | 110,9 | 81,0 | 82,3 | 82,9 | ~0 |
| `qsa_decode_attn_batch` cap 2051 | 67,0 | 74,5 | 106,2 | 138,6 | +23,9 |
| `native_qsa_indexer_append_steps` | 7,0 | 7,1 | 7,5 | 7,8 | +0,3 |
| **layer QSA** (i quattro in sequenza) | 245,9 | 223,7 | 276,1 | 333,3 | **+29,1** |
| denso 32 768 celle per riga (forma del drafter) | 651,3 | 1079,1 | 1525,4 | 1956,7 | +435 |

Con `STRATA_ATTN_DAILY=1` (pre75 + merge v2): attenzione 37,3 / 42,1 / 59,0 / 78,0 (+13,6 per riga); layer 209,2 / 191,1 /
230,1 / 275,4 (**+22,1**); denso 395,7 / 547,4 / 801,3 / 1060,9 (+222 per riga: il «~240 µs a query» dell'ipotesi).

### 3060 (device 1)

| kernel | T=1 | T=2 | T=3 | T=4 | per riga |
|---|---|---|---|---|---|
| `qsa_block_scores` (multi) | 64,4 | 106,1 | 152,6 | 194,2 | +43,3 |
| `qsa_block_topk` (wide) | 106,5 | 87,7 | 86,6 | 86,7 | ~0 |
| `qsa_decode_attn_batch` cap 2051 | 101,7 | 145,6 | 202,8 | 242,8 | +47,0 |
| `native_qsa_indexer_append_steps` | 6,9 | 7,1 | 8,3 | 7,0 | +0,1 |
| **layer QSA** | 263,7 | 340,6 | 449,6 | 527,3 | **+87,9** |
| denso 32 768 celle per riga | 1032,2 | 1826,6 | 2565,8 | 3311,4 | +760 |

Con `STRATA_ATTN_DAILY=1`: attenzione 41,3 / 71,0 / 110,3 / 131,4 (+30,0 per riga); layer 223,1 / 280,3 / 361,3 / 434,7
(**+70,5**); denso 475,3 / 869,1 / 1289,7 / 1728,8 (+418).

### Lettura

- **Per riga e per stadio**: 2080 Ti 8 layer × 29 µs = **0,23 ms** (0,18 col Daily); 3060 4 layer × 88 µs = **0,35 ms**
  (0,28). L'ipotesi prevedeva 2,0 e 1,5 ms: **l'attenzione QSA non è la leva**. Lo stadio 1 paga 3,0 ms per riga e lo
  stadio 0 2,8 (§3): il QSA ne è l'8% e il 12%.
- **L'attenzione a cap 2051** cresce di 14-24 µs per riga sulla 2080 Ti e 30-47 sulla 3060: non è la banda (2 MB per
  riga, in L2) ma l'ondata in più — 66 blocchi per riga, 204 slot (pre75) o 272 (kernel di base) sulla 2080 Ti, 84 /
  112 sulla 3060, dove già T = 2 è una seconda ondata. Il kernel del Daily dimezza il costo per riga su tutte e due.
- **I punteggi** (`block_scores_multi_kernel`) crescono di 19 µs (2080 Ti) e 43 (3060) per riga: le chiavi sono lette
  una volta ma il ciclo sulle query dentro il warp è sequenziale (12 istruzioni + 5 shuffle per testa e query su
  32 769 blocchi); è l'altra metà del termine per riga del layer QSA. Il **top-k** è piatto (un CTA per query, in
  parallelo) e **l'append** è trascurabile.
- **Il caso denso** conferma da dove veniva il numero dell'ipotesi: +222 µs per riga col pre75 sulla 2080 Ti (+418 sulla
  3060). Ma nessun percorso della verifica lo lancia, e il drafter attende con una riga sola (§1).
- Il T = 1 del top-k (111 contro 82 µs) e della sequenza è più lento di T = 2: è la prima misura della serie (clock della
  scheda non ancora al massimo), non un effetto del kernel.

## 3. Da dove viene davvero il costo per riga

Misure del motore (`DET-cp-clean/pipe_trace.txt`, 3 richieste L02 a 131k, banco forzato, mediane; script
`pipe_rows.py` ricostruibile: F0−L0 e F1−L1 per T):

| | T=1 | T=2 | T=3 | T=4 | per riga |
|---|---|---|---|---|---|
| stadio 0, `F0−L0` (3060, layer 0-15) | 22,05 | 24,64 | 29,16 | 30,53 | **+2,8 ms** |
| stadio 1, `F1−L1` (2080 Ti, layer 16-47 + testa) | 13,16 | 15,57 | 18,33 | 22,14 | **+3,0 ms** |
| passaggio `F0→L1` (host) | 0,17 | 0,19 | 0,20 | 0,20 | ~0 |

I kernel QSA (§2) ne coprono 0,23 ms (stadio 1) e 0,35 ms (stadio 0): l'8% e il 12%. Il resto, per quanto si legge nel codice e nei contatori del motore:

1. **Esperti.** `strata decode timing … per layer-window: VRAM hits 23.16 (avg T 2.02)` contro `35.12 (avg T 3.53)`
   (`DET-cp-clean/engine.log`, richieste 1 e 0): ~8-10 esperti distinti in più per riga e per layer (k = 10, con
   sovrapposizioni), ognuno 1,47 MB sulla 3060 (4433 slot in 6,21 GiB) e 1,83 MB sulla 2080 Ti (9150 slot in 15,95
   GiB, IQ3_XXS). A banda piena: stadio 1 32 layer × 9 × 1,83 MB = 0,53 GB → **0,86 ms per riga** (616 GB/s); stadio 0
   16 × 9 × 1,47 MB = 0,21 GB → **0,59 ms** (360 GB/s). I kernel `gu_grouped_t` / `down_grouped_t` non sono alla banda
   (nel drafter 24 + 15 µs per ~20 MB: ~50%), quindi il costo reale è probabilmente **1,2-1,7 ms** (stadio 1) e
   **0,9-1,2** (stadio 0). Più gli esperti **CPU**: `CPU experts 0.52 ms (0.58 entries)` a T 2,02 contro `2.98 ms (4.08
   entries)` a T 3,53 — ~0,75 ms l'uno, e il numero di mancati cresce con le righe (testi diversi: da ripetere sulla
   stessa lista).
2. **GDN** (36 layer): `gdn_step_norm_multi` esegue la ricorrenza delle T righe **in sequenza nello stesso kernel**
   (lo stato S×S per testa letto una volta, aggiornato T volte); costo per riga piccolo ma su 36 layer: da misurare
   (atteso 0,1-0,3 ms per stadio).
3. **Hyper-connection** (`fused_gr.cu`, tre chiamate per layer, 4 stream): nel drafter 111 µs per passo a T = 1;
   nel modello per layer e per riga la parte proporzionale a T è da misurare (kernel piccoli: più blocchi, non più
   tempo, se restano sotto un'ondata).
4. **Proiezioni**: MMVQ_IL / B6 multi-colonna, pesi letti una volta: ~0,3 ms per riga sulle due schede (B6, misurato).
5. **Host per layer**: `per-layer host 1.52 → 4.80 ms/window` fra avg T 2,02 e 3,53 (richieste diverse), `jobs`,
   `actq`, `plan`: il servizio degli esperti per layer cresce con gli esperti mancati, cioè con T.

Dove guardare per gli 80 t/s: esperti (riga per riga sono ~metà del costo) e il loro servizio host; non l'attenzione
QSA.

## 4. Kernel multi-riga: disegno e perché non è in questa build

Un kernel «K/V caricati una volta per tutte le righe che hanno scelto la cella» è possibile **bit per bit** solo se
ogni riga conserva la *propria* partizione in chunk (il massimo del chunk, la somma degli esponenziali e la piega dei
valori in ordine di cella sono alberi fissi sui 64 slot del chunk; il merge li piega in ordine di chunk). Il disegno
che lo rispetta: blocco per (chunk c, testa K/V), **tutte le righe** del chunk c insieme — `ids_t[64c .. 64c+63]` per
t = 0..T−1 — con l'unione delle celle del blocco caricata una volta (K in registri a coppie per warp come nel pre75,
V in scena in shared come `attn_chunk_kernel_stage`), i punteggi per (riga, cella) con la q della riga da shared/L1,
`sp[t][h][64]` e `spt[t]` per riga, la piega per riga nello stesso ordine. Griglia (33, 2) invece di (33, 2, T);
shared 15,9 KB + 3 KB × T + 4,2 KB; registri 12 × T accumulatori (T = 4: 48) → 2 blocchi per SM; le celle comuni fra
le righe (75-85%) lette una volta invece di T.

Perché non lo porto: a cap 2051 i blocchi per riga stanno in un'ondata fino a T = 3 sulla 2080 Ti (66 × T su 204
slot) e il kernel è legato alla latenza della catena del blocco, non alla banda; mettere T righe nello stesso blocco
allunga la catena (più celle nell'unione, T × il lavoro) e toglie parallelismo. Il guadagno possibile è sull'ondata in
più a T = 4 (2080 Ti con `STRATA_ATTN_DAILY=1`) e sulla 3060 da T = 2: azzerando del tutto il termine per riga
dell'attenzione misurato (§2) si guadagnerebbero al massimo 8 × 14-24 µs = **0,1-0,2 ms per riga** sullo stadio 1 e
4 × 30-47 = **0,12-0,19 ms** sullo stadio 0, su 3,0 e 2,8: sotto il rumore del banco (±5% in t/s). Sul **window denso** del drafter
(1024 blocchi, 5 ondate, 46 MB) il kernel multi-riga varrebbe ~200 µs per riga in più, ma il drafter attende con una
riga sola: niente da guadagnare lì. `STRATA_QSA_ROWS` resta quindi un nome riservato (il bench accetta `--rows` e lo
dice); se un giorno la verifica attende su T > 4 o su window densi, il disegno sopra è quello da portare, con
`qsa_rows_bench --rows` come parità (memcmp per riga contro `qsa_decode_attn_batch`).

## 5. Come misurarlo nel motore (banco forzato B-A-B)

Senza kernel nuovo non c'è un B-A-B da fare. Per attribuire il costo per riga nel motore, stesso exe e argomenti del
Daily (`docs/UPSYNC.md`, «Riprodurre il Daily»), `STRATA_DECODE_TIMING=1`, `STRATA_FORCE_IDS` + `STRATA_FORCE_WINDOWS`
della traccia di riferimento (le righe per finestra sono forzate, quindi uguali fra i bracci), `STRATA_PIPELINE_TRACE`:

- `pipe_rows.py` (F0−L0 e F1−L1 per T) sulla traccia: il costo per riga di ogni stadio;
- nsys di una richiesta (come `DET-cp-nsys5`) raggruppando i kernel per `cudaGraphLaunch` e per T della finestra:
  somma per famiglia (esperti `*_grouped_t`, `gdn_*`, `gr_*`, MMVQ, QSA `attn_chunk_*`/`block_*`) e differenza T=4 − T=1;
- `strata decode timing`: `VRAM hits` ed `entries` CPU per layer-window contro avg T, sulla stessa lista.

B-A-B utili sul QSA, già disponibili: `STRATA_ATTN_DAILY=1` / niente (kernel dei chunk e merge v2: cambia l'occupazione,
quindi le ondate a T = 4), `STRATA_SCORES_MULTI=0` (punteggi per query, griglia (max_blocks/8, n): per vedere quanto vale
la lettura unica delle chiavi).

## 6. Rischi e limiti

- Il bench è sintetico: selezioni da punteggi casuali con query perturbate (unione 75-85% di una riga); una finestra
  vera può sovrapporsi di più o di meno. Il costo dell'attenzione a cap 2051 non dipende dall'unione nel kernel attuale.
- La residenza (`qsa_kv_resolve`, pagine a −1) non è nel bench: le celle mascherate costano meno, non di più.
- I µs sono a GPU libera; nel motore i kernel QSA dello stadio 1 convivono con la catena del drafter e il merge legge
  parziali già in L2.
- L'attribuzione del §3 oltre il QSA è da contatori e da struttura del codice, non da nsys per T: è l'ordine di
  grandezza, non la misura. Il passo successivo è l'nsys per T del §5.

## Stato della build

Compilato sì (`build_lock.py`, CUDA 12.6, sm_75+86, test spenti): `strata.exe` dal codice di `5dffa63f` (= `lab/ports1`
`295d6e7f` nel motore: il branch aggiunge solo il bench, il target CMake e questo documento; **nessun kernel del motore è
cambiato**), sha256 `68c6413615cd06de854cb472e6a4bad51c949851d571e382d9b535a892e64b22`, copiato con `build.json` in
`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\build-qsarows-<sha>\`;
`qsa_rows_bench.exe` nello stesso `build-sm7586\` del worktree. Installato no, provato no (nessun motore avviato);
micro-bench sì, sulle due schede, sotto `bench.lock`. L'exe deve comportarsi come la build `cc94edf7` di `lab/ports1`.

## File toccati

- `src/kernels/qsa_rows_bench.cpp`, `CMakeLists.txt` (target `qsa_rows_bench`, `EXCLUDE_FROM_ALL`).
- `docs/QSAROWS.md`, `docs/qsa_rows_bench_2026-10-12.txt` (l'uscita grezza del bench sulle due schede, con e senza
  `STRATA_ATTN_DAILY=1`).
