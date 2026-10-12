# lab/speed4 — top-k QSA su più CTA, Q5_K «lento», testa Q5_K

Base: `lab/speed3` (0.1.43 + pila del fork + NODECUT di speed1 + EXPDEC_WT/MAGIC di speed3, tutti opt-in). Leve 8,
10 e 9 della classifica roofline (`ROOFLINE_2026-10-12.md` §5).

## 1. Top-k QSA su 8 CTA per query: `STRATA_TOPK_GRID=1` (opt-in)

**Il problema.** In decode a 131K con `--max-context 262144`, `qsa_block_topk` lancia `block_topk_wide_kernel`: una
CTA da 1024 thread per query, quattro passate radix e due scansioni su 32 769 blocchi. Una finestra ha da 1 a 5
query, quindi 1-5 SM su 68 per 80-110 µs. Il kernel a cluster di S19 (`block_topk_cluster_kernel`, 8 CTA per query
con la shared distribuita) gira solo da sm_90 in su.

**Cosa fa il ramo.** `block_topk_grid_kernel` (`src/kernels/cuda/qsa_select.cu`) usa l'algoritmo del kernel a
cluster con CTA normali:

- la CTA r di una query tiene in shared le chiavi dei blocchi `[r·per, (r+1)·per)` e scrive l'istogramma di ogni
  passata nello slot r di un'area di scambio in memoria globale;
- una barriera per query sostituisce `barrier.cluster`: contatore in globale con lo schema di `grid.sync`
  (barriera di CTA, fence, arrivo atomico, attesa con `ld.acquire`, barriera di CTA);
- cifre, soglia, `above`, conteggi per CTA e posizioni di scrittura sono quelli del kernel a cluster, quindi gli id
  sono **gli stessi per costruzione**: istogrammi interi sommati in qualunque ordine, pareggi alla soglia al blocco di
  indice più basso, uscita in ordine crescente.

Sicurezze:

- **Progresso.** Lo spin richiede le 8 CTA di una query residenti insieme. L'host controlla una volta per scheda e
  dimensione delle chiavi che la griglia più una query di margine stia sulla scheda (occupancy × SM); altrimenti
  restituisce false e restano i kernel a una CTA.
- **Riuso.** Il contatore cresce solo dentro una chiamata. L'ultima CTA che esce lo azzera, e la chiamata seguente
  è un kernel successivo sullo stream.
- **Concorrenza.** L'area di scambio è un pool `__device__` (nessuna allocazione durante la cattura) con 8 slot per
  scheda, assegnati per puntatore `ids`. I buffer del motore sono stabili (uno per verificatore e stadio), quindi due
  grafi in volo non condividono mai uno slot. Con il pool pieno si torna ai kernel a una CTA.

Gancio in `qsa_block_topk` per le chiamate di decode (nessun conteggio di blocchi attivi) fino a 8 query, dopo il
controllo del cluster. Riga di log `strata topk grid:` al primo lancio.

**Verifica (`topk_grid_parity`, target `EXCLUDE_FROM_ALL`).** Fixture di `decode_cluster_parity`: righe realistiche e
avversarie (tutte uguali, pochi valori, NaN, ±inf, zeri con segno, denormali, monotone, tutte NaN, quantizzate),
contesti da 7 a 262 144 celle, ogni peso della coda, 1/5/8 query, due capacità. Su entrambe le schede: **434 casi,
0 id diversi** dal kernel a una CTA e dal riferimento. 66 casi declinati: nq 8 con capacità 262K, dove la griglia non
sta con una query di margine. Inoltre: grafo catturato e rigiocato con input nuovi (anche 500 lanci di fila) e due
stream con buffer propri per 300 chiamate ciascuno, 0 differenze.

**Tempi** (`topk_grid_parity --bench`, µs a chiamata, mediana di 5 giri × 300, capacità = contesto + 272):

| contesto | nq | 2080 Ti una CTA → griglia | 3060 una CTA → griglia (rumorosa: monitor 8K) |
|---|---|---|---|
| 32K | 1-5 | 33-44 → 25-30 (1,3-1,6×) | 35-37 → 30-38 (0,9-1,2×) |
| **131K** | 1-5 | **77-83 → 27-30 (2,7-3,0×)** | 78-88 → 36-46 (1,9-2,4×) |
| 262K | 1-5 | 125-130 → 33-37 (3,4-3,8×) | 138-143 → 42-57 (2,5-3,3×) |

**Stima sulla finestra a 131K:** 8 layer QSA sullo stadio 1 × ~52 µs ≈ **−0,4 ms**, 4 sullo stadio 0 × ~45 µs ≈
**−0,18 ms**. È la stima della roofline (−0,5 / −0,25).

## 2. «Q5_K 2560×6144 al 20%» (leva 10): non esiste

Il tensore è `blk.{34,38,46}.attn_gate.weight`, Q5_K [2560, 6144], le uniche proiezioni Q5_K a 6144 uscite dello
stadio 1. La griglia 3072 di `b6_mmvq_rows` corrisponde a 6144 righe a due per blocco. Riletto dalle tracce nsys
(script `q5k_grid.py`, durata per griglia, tipo e T):

| traccia | Q5_K griglia 3072 sulla 2080 Ti, mediana per T |
|---|---|
| `DET-qsarows-nsys2` | T=2 27,8 µs · T=3 30,5 · T=4 36,3 · T=5 44,0 |
| `DET-x3-hostgap-nsysB2` | T=2 27,8 · T=3 30,7 · T=4 36,4 · T=5 44,0 |

`dense_bench` (SPEED2) dà 26-38 µs sulla stessa forma: 10,8 MB a 300-390 GB/s, il 49-63% del picco, come i gemelli
IQ4_XS/Q4_K/Q6_K. I «150 µs × 7» della roofline non compaiono in nessuna istanza. **Leva chiusa**: va tolta dalla
classifica (−0,6/−0,7 ms che non ci sono).

## 3. Testa Q5_K a 4-5 colonne con meno registri: `STRATA_B6_HEAD_MINB=6` (opt-in)

`b6_mmvq_rows_kernel<Q5K, 4, 2>` usa 89 registri (traccia nsys: 1 235 µs a T=4 contro 874 a T=2). Il kernel prende
ora `MINB` come parametro di `__launch_bounds__`. Con la variabile, la testa (n_out ≥ 65 536) a 4-5 colonne chiede 6
blocchi per SM: stessa aritmetica nello stesso ordine, quindi stessi bit.

`b6_mmvq_bench 300 0 1` sulla 2080 Ti, ms a chiamata (B = righe del B6), tutti i confronti `ok ok ok`:

| colonne | MINB 1 (oggi) | MINB 6 | MINB 8 |
|---|---|---|---|
| 3 | 1,075 | 1,076 | 1,362 |
| **4** | 1,356 | **1,232 (−9%)** | 1,257 |
| **5** | 1,614 | **1,458 (−10%)** | 1,856 |
| 8 | 2,370 | 6,199 (spill) | 24,4 |

Da qui solo 4-5 colonne. Peso: −0,12/−0,16 ms nelle finestre a T=4-5 dello stadio 1, ~−0,05 ms in media al T medio di
3,2-3,4.

## 4. Verifica nel motore (cache fissa)

Le corse brevi DET-x7 con `--expert-cache auto --elastic` non sono deterministiche quando il monitor 8K sta sulla
3060 (SPEED2 §5). Si usano quindi la cache fissa e niente elastica (`det_check_fixed.py`: `--set-opt=--expert-cache=1500
--drop-opt=--elastic`, cache della 3060 a 2336 slot). Build `build-speed4-311b6fda` (strata.exe sha256 `79fbc4d0…`),
L02 a 64 token:

| braccio | `L02.txt` |
|---|---|
| off | `5fefd82e` (lo stesso dell'off di speed2 a cache fissa) |
| `STRATA_TOPK_GRID=1` | `5fefd82e` |
| `STRATA_B6_HEAD_MINB=6` | `5fefd82e` |
| entrambe | `5fefd82e` |
| **braccio B**: `STRATA_NODECUT=1 STRATA_QFUSE_HC=0 STRATA_TOPK_GRID=1 STRATA_B6_HEAD_MINB=6` | `5fefd82e` |

Riga `strata topk grid: CUDA0 8 CTAs x 512 threads per query, 32772 B of keys per CTA` (capacità 262 144:
32 KB di chiavi per CTA, una CTA per SM sulla 2080 Ti, margine fino a 7 query).

## 5. Per il B-A-B notturno

- Build: `build-speed4-311b6fda` (commit `311b6fda`, sha256 `79fbc4d09a16d14ba38c777c6fcc2e2940280ea83342cea5bc77fa714daf517e`).
  Contiene NODECUT, TOPK_GRID e HEAD_MINB; EXPDEC_WT/MAGIC sono spenti di default; DENSE_WR non c'è (archiviata).
- Braccio B: `STRATA_NODECUT=1 STRATA_QFUSE_HC=0 STRATA_TOPK_GRID=1 STRATA_B6_HEAD_MINB=6` sopra l'env attuale.
  **`STRATA_QFUSE_HC=0` è obbligatorio**: con NODECUT=1 da solo il testo cambia (SPEED2 §5).
- Atteso: −0,5/−0,6 ms a finestra dal top-k, −0,05 dalla testa, più quanto valgono FUSE_WAIT e FUSE_HITS
  (−3 nodi per strato). Totale ~−0,7/−1,0 ms, cioè +3/+4% t/s.

## 6. Aperto

- **QFUSE_HC non è bitwise su CUDA**, e il motivo non è ancora noto. Con `STRATA_QFUSE_HC=1` il L02 a cache fissa
  cambia in modo riproducibile (`1fb98380` ×3 contro `5fefd82e`). Ho escluso a lettura due cause: i quantizzatori
  (`gr_up_q32_kernel`, `gr_q8_tail` e `native_quantize_q8_1` hanno la stessa riduzione xor 16→1 e gli stessi
  `q8_1_finite`/`q8_1_ds`) e l'aritmetica per riga (`gr_up_q32_kernel` è quella di `gr_up_multi_kernel`). Prossimo
  passo: un test che confronti `fused_gr_read_multi` con e senza `q8_mixed`, da T=1 a 5 (mixed, R_out, inject e byte
  q8), più le immagini q8 che legge il motore (`xq_` della metà attenzione, `nat_xq_` della metà FFN).
- Top-k a più di 5 query (batch multi-chat): la griglia si rifiuta dove non sta con margine. Non è misurata.

## Stato

- 12/10 pomeriggio: compilato sì, parità sì (434/0 su entrambe le schede), bench sì, motore a cache fissa sì (5/5
  identici). Velocità nel motore: provato no (B-A-B notturno).
