# QSAROWS: le T righe di una finestra nei kernel QSA del decode

Branch `lab/qsarows` del fork `imanu86/Strata-2080Ti`, da `lab/ports1` (`295d6e7f`: `lab/upsync` = upstream v0.1.42 più
le patch del Daily, `STRATA_CP_*` di `docs/CRITPATH.md`, `STRATA_CHAIN_FUSE2` di `docs/CHAINFUSE.md`, port #1437/#1889).
Configurazione del Daily: 3060 = stadio 0 (layer 0-15), 2080 Ti = stadio 1 (layer 16-47, testa, drafter MTP),
`--spec 4 --kv int8 --kv-resident 32768 --max-context 262144 --mtp rt_it48k`, chat singola a 131k.

Stato: vedi «Stato della build» in fondo. Micro-bench sì (`qsa_rows_bench`); una cattura nsys del motore (build
`743b2d07`, §3) sotto `bench.lock`.

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
L'nsys per T del §3 lo conferma nel motore: per riga lo stadio 1 paga +3,26 ms, di cui esperti +1,31, MMVQ densi (B6)
+0,63, hyper-connection +0,48, attese host +0,36, QSA +0,30; lo stadio 0 +2,45, di cui esperti +1,04, MMVQ +0,59, hc
+0,41, QSA +0,24. Portato qui (§4) solo il pezzo QSA che conviene, bit per bit: `STRATA_QSA_ROWS=1`, i punteggi
dell'indicizzatore (−0,06 / −0,12 ms per riga). Il kernel multi-riga dell'attenzione no (§4).

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

## 3. Da dove viene davvero il costo per riga: nsys per T, per famiglia e per stadio

Cattura `DET-qsarows-nsys2` (12/10, build `build-ports1-743b2d07`, banco forzato L02 a 131k: `STRATA_FORCE_IDS` +
`STRATA_FORCE_WINDOWS=ref_def1_trace.txt`, 286 finestre di L02 dopo una L01 di 16 token che scalda il prefisso, opzioni
`--layer-split 16 --trim-stage-weights --pipeline-windows 2 --vram-reserve-later-mib 384 --host-core last`, env
`STRATA_VRAM_FREE=device STRATA_ROUTE_TAIL_SKIP=0 STRATA_ATTN_DAILY=1 STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1
STRATA_MTP_SHARED_BRANCH=0 STRATA_SIDE_MIXER=2`, `CUDA_VISIBLE_DEVICES=1,0`; `nsys profile --trace=cuda
--cuda-graph-trace=node`, 1,26 M kernel; cartella `20261004_routed_keep\DET-qsarows-nsys2\` con `nsys_rows.txt`,
`run_nsys.bat`; script `nsys_rows.py` e `run_nsys.py` in `20261004_routed_keep\`). Nessun crash a 286 finestre; il
report `.nsys-rep`/`.sqlite` in `D:\ds4_work\strata\nsys\`.

Metodo: i kernel di un lancio di grafo condividono il `correlationId` del `cudaGraphLaunch`; il grafo della finestra è
uno per T (`exec_[T]`) e il T si legge da `gridZ` del kernel dei chunk dell'attenzione; device 0 = 3060 = stadio 0
(712 nodi), device 1 = 2080 Ti = stadio 1 (1397 nodi). Per (stadio, T) la mediana sulle finestre del tempo kernel per
famiglia, il muro del grafo (primo inizio → ultima fine) e i buchi (muro − unione dei kernel); il costo marginale per
riga è la pendenza ai minimi quadrati sulle mediane T = 1..4. Finestre: stadio 0 n = 218/114/66/81, stadio 1
110/81/45/58. In ms per finestra:

**Stadio 0 (3060, layer 0-15)**

| famiglia | T=1 | T=2 | T=3 | T=4 | **per riga** | kernel che la fanno |
|---|---|---|---|---|---|---|
| esperti VRAM | 1,93 | 2,97 | 4,12 | 5,00 | **+1,04** | `native_gu_multi` +0,51, `native_down_multi` +0,37, router `bf16_f32_mmvf(_multi)` +0,07, `route(_multi)` |
| MMVQ densi | 2,52 | 3,17 | 3,64 | 4,32 | **+0,59** | a T=1 i kernel per formato (`q6_k` 0,60, `iq4_xs` 0,74, `q4_k` 0,33, `q5_k` 0,15, `small` 0,09 = 1,96); da T=2 `b6_mmvq_rows` 2,09 → 2,42 → 3,04 (+0,47 per riga) + `native_mmvq_multi` 0,18-0,26; `mmvq_multi` +0,07 |
| hyper-connection | 1,58 | 2,05 | 2,50 | 2,80 | **+0,41** | `gr_up_multi` +0,21 (0,74 → 1,39), `gr_down_staged` +0,12 (0,73 → 1,09); `gr_norm_split`, `gr_write_norm_rs` piatti |
| QSA | 0,68 | 0,89 | 1,15 | 1,40 | **+0,24** | `block_scores_multi` +0,14 (0,22 → 0,64), `attn_chunk_pre75` +0,09 (0,12 → 0,37), `block_topk_wide` piatto 0,30-0,32 |
| GDN | 0,24 | 0,28 | 0,33 | 0,38 | +0,05 | `gdn_step_norm_multi` +0,03, `gdn_conv_l2_multi`, `gdn_ab_multi` |
| attese host (`wait_flag_ge`) | 2,07 | 2,04 | 0,84 | 2,20 | ~0 | **2 ms piatti**: la GPU aspetta i flag dell'host per layer |
| altro (copie, resolve, doorbell) | 0,26 | 0,31 | 0,38 | 0,60 | +0,11 | `copy_kernel`, `copy_rows_from_mapped`, `doorbell_publish` |
| buchi fra i nodi | 0,01 | 0,01 | 0,01 | 0,01 | 0 | |
| **muro del grafo** | **9,31** | **11,69** | **13,39** | **16,91** | **+2,45** | |

**Stadio 1 (2080 Ti, layer 16-47, testa)**

| famiglia | T=1 | T=2 | T=3 | T=4 | **per riga** | kernel che la fanno |
|---|---|---|---|---|---|---|
| esperti VRAM | 3,04 | 4,43 | 5,68 | 7,00 | **+1,31** | `native_gu_multi` +0,88 (1,56 → 4,21), `native_down_multi` +0,36 (0,73 → 1,82), router +0,08 |
| MMVQ densi | 5,13 | 5,55 | 6,11 | 7,06 | **+0,63** | T=1 per formato (`q6_k` 1,52, `q5_k` 1,06, `q4_k` 0,56, `iq4_xs` 0,55, `small` 0,23 = 3,92); da T=2 `b6_mmvq_rows` 4,04 → 4,46 → 5,19 (+0,58 per riga, 134 chiamate) + `native_mmvq_multi` 0,25-0,31; `mmvq_multi` +0,11 |
| hyper-connection | 2,42 | 2,71 | 3,22 | 3,85 | **+0,48** | `gr_up_multi` +0,27 (1,10 → 1,91), `gr_down_staged` +0,21 (1,04 → 1,66) |
| attese host (`wait_flag_ge`) | 1,46 | 1,95 | 2,05 | 2,62 | **+0,36** | 96 attese per finestra: il servizio degli esperti per layer cresce con le righe |
| QSA | 1,24 | 1,48 | 1,80 | 2,14 | **+0,30** | `block_scores_multi` +0,16 (0,34 → 0,82), `attn_chunk_pre75` +0,12 (0,18 → 0,55), `block_topk_wide` piatto 0,65-0,69 |
| GDN | 0,39 | 0,45 | 0,54 | 0,65 | +0,09 | `gdn_step_norm_multi` +0,04, `gdn_ab_multi` +0,04 |
| altro | 0,52 | 0,57 | 0,65 | 0,77 | +0,08 | copie, `resolve`, doorbell |
| PLE | 0,08 | 0,08 | 0,08 | 0,08 | 0 | |
| buchi fra i nodi | 3,49 | 3,57 | 3,57 | 3,64 | +0,05 | **3,5 ms piatti**: ~2,5 µs fra i 1397 nodi del grafo |
| **muro del grafo** | **16,68** | **19,83** | **22,54** | **26,62** | **+3,26** | |

(La catena del drafter sulla 2080 Ti: 1904 grafi di 54 kernel, muro 0,89 ms, di cui QSA 0,30 e MMVQ 0,36: il passo
di `docs/CHAINFUSE.md`, invariato.) Le pendenze dei muri (+2,45 e +3,26 ms per riga) coincidono con le F0−L0 / F1−L1
della traccia della pipeline (`DET-cp-clean`: +2,8 e +3,0 ms). Il QSA ne è il **10%** (stadio 0) e il **9%** (stadio 1),
come il micro-bench del §2 prevedeva.

### Le leve, in ms recuperabili per riga

1. **Esperti (`native_gu_multi_kernel` / `native_down_multi_kernel`, `iq_kernels.cu`): +1,04 / +1,31 ms per riga.**
   I pesi di un esperto sono già letti una volta per finestra (`GRP_NC = 4` righe per passata); il costo per riga è
   quello degli esperti **distinti** in più (~9 per layer e riga, `VRAM hits` 24 → 35 per layer-window fra T 2 e 3,5) e
   della loro decodifica IQ3_XXS: a T = 1 1,56 ms per 32 layer × 10 esperti × ~1,2 MB di gate/up = 12 MB per layer →
   ~250 GB/s, il **40% della banda** della 2080 Ti; a T = 4 ~55%. Il kernel è legato alla decodifica dei codebook
   IQ3_XXS e al prodotto per entrata, non alla banda. Recuperabile: portarlo al 70-80% della banda vale ~**0,5 ms per
   riga sullo stadio 1 e ~0,4 sullo stadio 0** (più ~1 ms piatto a T = 1). Cosa serve: profilo del kernel (ncu: issue
   slot, pipe della decodifica), poi una riscrittura della decodifica/lettura con le stesse somme per entrata (stesso
   albero per esperto → stessi bit) o, fuori parità, un formato più leggero per gli esperti caldi. Lavoro grande.
2. **MMVQ densi con B6 (`b6_mmvq_rows_kernel`, `native_mmvq.cu`): +0,47 / +0,58 ms per riga.** I pesi sono letti una
   volta ma il kernel paga ~0,3 ms per riga e stadio di prodotti Q8_1 (il micro-bench B6 dava 0,28 + 0,26); la 2080 Ti
   ha dal `lab/ports1` la tabella MMVQ_IL sm_75 (`STRATA_MMVQ_IL=1`, PR #1437, bit per bit): B-A-B pronto, atteso
   **0,2-0,4 ms per riga** se IL batte B6 sulle forme calde (Q6_K 2560×6144 / 6144×2560, IQ4_XS). Sulla 3060 la
   tabella sm_80+ è di default in 0.1.42 e qui era spenta dall'env (`STRATA_MMVQ_IL=0` riproduce il Daily): stesso B-A-B.
3. **Hyper-connection (`gr_up_multi_kernel`, `gr_down_staged_kernel`, `fused_gr.cu`): +0,41 / +0,48 ms per riga.**
   65 chiamate per stadio di kernel piccoli (17 → 29 µs l'uno da T = 1 a 4): il lavoro per riga è dentro il blocco,
   non su più blocchi. Recuperabile ~**0,3 ms per riga e stadio** con righe su blocchi distinti (stesse somme per riga →
   stessi bit); serve leggere i due kernel e la loro geometria (`GrMulti`). Lavoro medio.
4. **Attese host (`wait_flag_ge_kernel`): 2 ms piatti sullo stadio 0, 1,5 → 2,6 ms sullo stadio 1 (+0,36 per riga).**
   È il thread host che serve gli esperti layer per layer (`CRITPATH`, `per-layer host`): fronte host, non kernel.
5. **Buchi fra i nodi dello stadio 1: 3,5 ms piatti per finestra** (13% di una finestra T = 2): fusioni di nodi nel
   grafo della verifica come in `CHAINFUSE` (−2,5 µs a nodo tolto); non è per riga ma è la voce piatta più grossa.
6. **QSA: +0,24 / +0,30 ms per riga.** Metà sono i punteggi (`block_scores_multi`): fatto qui, §4 (`STRATA_QSA_ROWS=1`,
   −0,06 / −0,12 ms per riga); l'altra metà è l'ondata in più del kernel dei chunk (§2), non conviene.

## 4. `STRATA_QSA_ROWS=1`: i punteggi dell'indicizzatore con meno pressione per riga (bit per bit)

Dall'nsys (§3) il kernel QSA che cresce di più con le righe è `block_scores_multi_kernel` (+0,14 / +0,16 ms per riga su
4 / 8 layer: ~20 µs per chiamata e riga su entrambe le schede). Le chiavi sono già lette una volta per tutte le query;
il costo per riga sta nelle **pipe di shared e shuffle**: per (blocco, query) 4 letture `LDS.128` delle fette di q e 5
shuffle per testa (20 per blocco e query) — 32 769 blocchi × (4 LDS + 20 SHFL) per query ≈ 15-20 µs di sola emissione.
`block_scores_rows_kernel<NQ>` (`qsa_select.cu`, NQ = 1..4): (1) le fette di q delle NQ query (4 teste × 4 float a lane)
in **registri** per tutto il kernel (44 / 63 / 78 / 94 registri, STACK 0, `cuobjdump -res-usage`); (2) i prodotti di
lane delle 4 teste **ridotti a scatter** con gli accoppiamenti di `warp_sum` (xor 16 e 8 tengono una testa per lane,
poi 4, 2, 1: sei shuffle per le quattro teste invece di venti) e raccolti sulla lane 0 in ordine di testa (tre shuffle).
Ogni somma addiziona gli stessi due operandi dello stesso livello di `warp_sum` (la somma float commuta), l'espressione
del prodotto è lo stesso testo nello stesso file, i relu si sommano nell'ordine delle teste: **stessi bit** del kernel
multi e del kernel per query. Blocco di coda e +1e9 come prima. Opt-in: `STRATA_QSA_ROWS=1` esatto, in
`qsa_block_scores` per le chiamate senza `active_blocks` fino a 4 query (le finestre del decode); oltre, o senza la
variabile, il kernel multi di prima.

Parità e µs (`qsa_rows_bench --rows`, 12/10, `docs/qsa_rows_bench_rows_2026-10-12.txt`): punteggi **e** id del top-k
identici (memcmp) a T = 1..4 a 131k e a contesto corto (celle 100-103, blocco di coda pieno), 16 casi su due schede, con e
senza `STRATA_ATTN_DAILY`:

| `qsa_block_scores`, µs a chiamata | T=1 | T=2 | T=3 | T=4 | per riga |
|---|---|---|---|---|---|
| 2080 Ti, multi (riferimento) | 47,2 | 56,9 | 78,2 | 101,2 | +18,0 |
| 2080 Ti, `STRATA_QSA_ROWS=1` | **36,2** | **38,0** | **49,2** | **54,6** | **+6,1** |
| 3060, multi | 73,4 | 109,4 | 151,9 | 193,5 | +40,0 |
| 3060, `STRATA_QSA_ROWS=1` | **63,3** | **75,5** | **98,0** | **130,2** | **+22,3** |

Nel motore: stadio 1 (8 layer) −0,09 ms a T = 1 e −0,37 a T = 4 per finestra; stadio 0 (4 layer) −0,04 e −0,25. Piccolo
(1-1,5% di una finestra), sicuro, bit per bit. Il T = 4 sulla 3060 resta alto perché a 94 registri il kernel scende a
2 CTA per SM: una versione a due passate di 2 query terrebbe 63 registri (da provare se la 3060 conta).

### Il kernel multi-riga dell'attenzione: disegno e perché non è qui

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
riga sola: niente da guadagnare lì. `STRATA_QSA_ROWS` qui accende solo il kernel dei punteggi; se un giorno la verifica attende su T > 4 o su window densi, il disegno sopra è quello da portare, con
`qsa_rows_bench --rows` come parità (memcmp per riga contro `qsa_decode_attn_batch`).

## 5. Come misurarlo nel motore (banco forzato B-A-B)

Stesso exe e argomenti del §3, `STRATA_DECODE_TIMING=1`, `STRATA_FORCE_IDS` + `STRATA_FORCE_WINDOWS` (righe per finestra
forzate, uguali fra i bracci). Tre corse alternate B (`STRATA_QSA_ROWS=1`), A (senza), B:

- identità: `strata work: … accepted X of Y drafts … target picks replaced` uguale (gli id della selezione sono gli
  stessi bit, quindi anche l'attenzione e i token);
- tempo: `strata decode stage 1: GPU-reach wait` (−0,1/−0,4 ms per finestra secondo T), `strata pipeline: … ms/window`,
  t/s; sotto il rumore (±5%) serve la traccia nsys o `STRATA_PIPELINE_TRACE` + `pipe_rows.py` (F1−L1 per T).

Per le leve del §3: `STRATA_MMVQ_IL=1` contro `STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1` (leva 2, pronta); nsys come
`DET-qsarows-nsys2` (`run_nsys.bat NAME 500`, poi `nsys export --type=sqlite` e `nsys_rows.py`) dopo ogni intervento
sugli esperti o sulle hyper-connection: la tabella per famiglia e T dice se la pendenza è scesa.

## 6. Rischi e limiti

- La cattura nsys è una richiesta (L02, 286 finestre) con tracciamento acceso: i µs assoluti sono gonfiati di poco
  (`--cuda-graph-trace=node`), le pendenze per T no (stesso rumore a ogni T); la classe fresca/speculativa non è
  distinta (stesso grafo); la L01 di 16 token è servita solo a scaldare il prefisso.
- `STRATA_PIPELINE_TRACE` non ha scritto il file in questa corsa (serve anche `STRATA_PIPELINE_DEBUG=1`, come in
  `DET-cp-nsys5`): il confronto con F0−L0 / F1−L1 usa `DET-cp-clean`.
- Il micro-bench è sintetico (selezioni da punteggi casuali, unione 75-85%); le pendenze nel motore (§3) lo confermano.
- `STRATA_QSA_ROWS=1` a T = 4 usa 94 registri (2 CTA per SM): sulla 3060 il guadagno a T = 4 è un terzo, non la metà.
- Le stime di recupero delle leve 1 e 3 sono dall'efficienza misurata (banda, µs per chiamata), non da un prototipo.

## Stato della build

Compilato sì (`build_lock.py`, CUDA 12.6, sm_75+86, test spenti): `strata.exe` dal codice di `4d8ba8bb`
(`STRATA_QSA_ROWS=1` opt-in; senza la variabile i lanci sono quelli di `lab/ports1` `295d6e7f`), sha256 vedi
`build.json`, copiato con `qsa_rows_bench.exe`, i log dei bench e `build.json` in
`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\build-qsarows-<sha>\`.
Installato no, provato no (il motore sotto nsys è stato avviato sulla build `743b2d07`, non su questa); micro-bench sì
sulle due schede sotto `bench.lock` (parità bit per bit e µs del §2 e del §4). La build precedente (`c6fd0440`, solo
bench e documento) è in `build-qsarows-c6fd0440\` con sha256 `68c6413615cd06de854cb472e6a4bad51c949851d571e382d9b535a892e64b22`.

## File toccati

- `src/kernels/cuda/qsa_select.cu`, `include/strata/kernels/qsa_select.hpp`: `block_scores_rows_kernel<NQ>`,
  `qsa_block_scores_rows`, `STRATA_QSA_ROWS` in `qsa_block_scores`.
- `src/kernels/qsa_rows_bench.cpp`, `CMakeLists.txt` (target `qsa_rows_bench`, `EXCLUDE_FROM_ALL`; `--rows` = parità e µs).
- `docs/QSAROWS.md`, `docs/qsa_rows_bench_2026-10-12.txt`, `docs/qsa_rows_bench_rows_2026-10-12.txt` (uscite grezze).
- Fuori dal repo (lab, `moe-aggressive-commit\...\20261004_routed_keep\`): `run_nsys.py` (run_long + `--cases`,
  `--first-tokens`, variabili di iniezione nsys), `nsys_rows.py`, `pipe_rows.py`, `DET-qsarows-nsys2\`.
