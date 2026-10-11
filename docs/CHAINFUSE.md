# CHAINFUSE: fusioni vere nel passo della catena MTP (2080 Ti, sm_75)

Branch `lab/chainfuse` del fork `imanu86/Strata-2080Ti`, da `lab/critpath2` (`8b4fd975`: `lab/upsync` = upstream v0.1.42
più le patch del Daily, più le leve `STRATA_CP_*` di `docs/CRITPATH.md`). Configurazione del Daily: 3060 = stadio 0,
2080 Ti = stadio 1 con testa e drafter MTP, `--spec 4`, `--kv int8`, `--kv-resident 32768`, `--mtp rt_it48k`.

Obiettivo: un passo della catena del drafter (54 kernel, 0,93 ms) a ~0,5 ms **con le stesse bozze bit per bit**.
Tutto opt-in: `STRATA_CHAIN_FUSE2=1`. Senza la variabile il motore registra gli stessi grafi di `lab/critpath2`.

Stato: compilato (vedi in fondo), installato no, provato no. Nessuna GPU, nessun motore, nessun test eseguito.

## 1. La mappa del passo (nsys, `DET-cp-nsys5`, 2011 passi, mediana)

Un passo è un grafo di 54 kernel sullo stream del drafter (step graph). Tempo mediano **911 µs**: **877 µs di kernel,
30 µs di buchi fra kernel dipendenti, più una bolla di 33 µs** (sotto). Il passo **non** è limitato dalla latenza di
lancio: i ~25 kernel sotto i 3 µs valgono in tutto ~55 µs; il grosso sono pochi kernel grandi.

| # | kernel | griglia | µs | cosa |
|---|---|---|---|---|
| 0 | `copy_from_mapped_multi` | 10×7 | 2,3 | righe step/pos dal mapped (con `STRATA_CP_CHAIN_FUSE=1` sparisce) |
| 1 | `embed_rows` | 10 | 4,5 | embedding del token |
| 2-3 | `norm` + `native_quantize_q8_1` | 1 / 10 | 3,2 + 1,6 | RMS dell'embedding e q8_1 |
| 4 | `native_small_mmvq` | 2560 | 16,0 | `fc_embedding` 2560×2560 Q8_0 |
| 5-6 | `norm` + `quantize` | 1 / 40 | 6,0 + 1,7 | RMS sui 4 stream (10240) e q8_1 |
| 7 | `native_mmvq_multi` | 1280 | 20,7 | `fc_hidden` sui 4 stream |
| 8 | `add_streams_broadcast` | 40 | 1,9 | R = h + e |
| 9-11 | `gr_norm_split`, `gr_down_staged`, `gr_up_multi` | 1×4 / 41 / 160 | 4,4 + 15,9 + 16,4 | hyper-connection dell'attenzione |
| 12-14 | `quantize`, `small_mmvq` ×2 | 10 / 512 / 512 | 1,7 + 4,7 + 4,7 | q8_1 di `mixed`, `k_proj`, `v_proj` |
| 15-16 | `norm_rope`, `kv_append_q8` | 2 / 2×4×2 | 2,5 + 2,6 | K norm+rope, append int8 |
| 17 | `native_small_mmvq` | 12288 | 62,9 | `q_proj` 2560→12288 (33 MB: alla banda) |
| 18-19 | `norm_rope`, `window_ids` | 24 / 8 | 2,5 + 2,0 | q norm+rope, finestra di 32k celle |
| **20** | **`attn_chunk_kernel` (pre75)** | **512×2** | **240,5** | **attenzione densa su 32768 celle, 1024 blocchi, 5 ondate da ~48 µs** |
| **21** | **`attn_merge_v2_kernel`** | **24** | **50,3** | **merge dei 512 chunk per testa, 24 blocchi** |
| 22-23 | `gate`, `quantize` | 24 / 24 | 1,8 + 1,7 | gate sigmoide e q8_1 |
| 24 | `native_small_mmvq` | 2560 | 32,5 | `o_proj` 6144→2560 |
| 25-27 | `gr_*` ×3 | | 5,1 + 15,5 + 17,0 | hyper-connection del MLP (scrittura dell'attenzione già fusa) |
| 28-41 | router, esperti, shared expert | | ~100 | `bf16_f32_mmvf` 7,1, `route` 5,3, `group_resident` 4,2, `gu_grouped_t` 24,4, `down_grouped_t` 14,8, shared expert ~28 su un ramo laterale (sovrapposto) |
| 42-43 | `combine_k10_vec4`, `post` | 5 / 40 | 2,3 + 2,0 | combine e scrittura del residuo |
| 44-46 | `gr_*` ×3 (mixer) | | 4,3 + 16,0 + 16,6 | **bolla di 33 µs fra `gr_down_staged` e `gr_up_multi`** (99,5% dei passi) |
| 47 | `quantize` | 10 | 1,7 | q8_1 di `sample` |
| **48** | **`native_q5_k_mmvq`** | **48542** | **166,6** | **testa Q5_K sul sotto-vocabolario (68 MB: 67% della banda)** |
| 49-53 | `sampler_greedy`, `row_top_prob_split`, `map_ids`, `mtp_select`, `force_token` | 1 / 8 / 1 / 1 / 1 | 17,0 + 4,5 + 1,7 + 7,0 + 3,6 | argmax a un blocco, probabilità, mappa, selezione, forzatura |

Tre voci fanno il 60%: attenzione 291 µs (33%), testa 167 µs (19%), le tre hyper-connection 111 µs (13%). Il pavimento di
banda del passo (q 33 MB + o 17 + fc 14 + testa 68 + K/V 33 + parziali 25 + esperti ~20 = ~210 MB a 616 GB/s) è ~350 µs:
sotto non si va a parità di formati.

**Perché l'attenzione costa 240 µs.** `attn_chunk_kernel_pre75` (il kernel del Daily, `STRATA_ATTN_DAILY=1`): un blocco
per (chunk di 64 celle, testa K/V), 78 registri, 3 blocchi per SM, 1024 blocchi = 5 ondate da 48 µs. Dentro il blocco la
catena è di round-trip di memoria **dipendenti**: i punteggi 2 celle per volta per warp (4 viaggi), i valori 4 celle per
volta per thread (16 viaggi), più q e page table. Per 64 celle il lavoro aritmetico è di pochi µs: il blocco aspetta la
memoria. Il merge v2 fa la stessa cosa: 24 blocchi, per ogni dimensione 512 chunk in sequenza a batch di 16 load
dipendenti (32 viaggi).

**La bolla di 33 µs.** Sta sempre nello stesso punto (il mixer, `gr_down_staged` → `gr_up_multi`), non nelle altre due
chiamate identiche, e i grafi dei round la hanno nello stesso kernel (indice 46 o 54 a seconda di T). Non è uno split
periodico del grafo (i grafi da 1397 nodi hanno ~4 bolle, non 30). È legata alla struttura: è il primo tratto dopo il
join del ramo laterale dello shared expert (`sh_fork_`/`sh_join_`), e il join stesso costa 6 µs (`sigmoid_scale` →
`combine`). Ipotesi: un arco fra stream dentro il grafo paga una sincronizzazione del driver (WDDM). Prova a costo zero:
`STRATA_MTP_SHARED_BRANCH=0` (shared expert sullo stream principale: +28 µs di kernel in serie, −39 µs di bolle se
l'ipotesi regge). Il micro-bench ha `--graph` che misura un grafo di 50 kernel con e senza fork/join.

## 1b. Prima misura (coordinatore, `e6b33daa`, `chain_step_bench` sulla 2080 Ti, parità PASS su tutto)

| segmento | default | fuso | rapporto |
|---|---|---|---|
| chunk pass, primo kernel `mlp` (VB 8 / 16 / 32) | 244,8 / 244,2 / 245,1 µs | 406,7 / 362,5 / 626,5 | **1,66 / 1,48 / 2,56: più lento** |
| merge v3 contro v2 | 45,1 | 32,9 | 0,73 |
| norma+q8_1 (2560 / 10240) | 13,5 / 15,4 | 8,7 / 11,4 | 0,64 / 0,74 |
| gate+q8_1 | 12,0 | 6,5 | 0,54 |
| coda della testa | 85,0 | 12,4 | 0,15 |
| combine+post | 12,4 | 6,9 | 0,56 |

Sulla 3060 lo stesso quadro. Nel motore (B-A-B, banco forzato, 131k): tutto acceso (compreso il chunk `mlp`) −0,9%;
senza il chunk (`MASK=62`) più `STRATA_MTP_SHARED_BRANCH=0`: **+1,6%** (+2,9 e +0,3). Il kernel dei chunk `mlp` era
sbagliato: `cuobjdump -res-usage` sull'oggetto compilato dà **80 registri con spill** (STACK 40 B a VB 8/16, 224 B a
VB 32) contro 73 e zero del pre75, a parità di shared (15,9 KB) e di occupazione (3 blocchi/SM); in più il loop a una
cella per volta raddoppiava le letture di `q` dalla shared (il pre75 le condivide fra due celle: ~30 µs di LDS su tutto
il kernel, 60 nel `mlp`). Tolto; al suo posto `attn_chunk_kernel_stage` (§2).

## 2. Cosa fonde `STRATA_CHAIN_FUSE2=1`

Regola: ogni fusione mantiene **le stesse espressioni nello stesso ordine** del kernel che sostituisce (stesso albero di
somme per riga, stesse funzioni intrinseche, stesso file e stessi flag di compilazione: `native_qsa.cu`,
`native_mmvq.cu`, `native_moe.cu` e `native_gr_postops.cu` compilano con `--use_fast_math`, quindi le fusioni che li
toccano vivono in quei file). Cambiano solo il numero di lanci e il numero di load in volo.

Due variabili distinte, per misurare a parte: `STRATA_CHAIN_FUSE2=1` con `STRATA_CHAIN_FUSE2_MASK` (default **62**:
tutto tranne il kernel dei chunk) accende le fusioni dei kernel piccoli e il merge; `STRATA_CHAIN_FUSE2_ATTN=1..4`
(richiede `FUSE2=1`; 0 o assente = pre75) sceglie la variante del kernel dei chunk. `STRATA_MTP_SHARED_BRANCH=0` è la
terza leva, già esistente (§3c). Le fusioni una per volta:

| bit | fusione | kernel prima → dopo | atteso |
|---|---|---|---|
| `ATTN=n` | `attn_chunk_kernel_stage<M, KPF, EARLY>` (`qsa_decode_attn.cu`) | `attn_chunk_kernel_pre75` con la stessa struttura (coppie di chiavi NC=2 che condividono le letture di `q`, stesse espressioni, stessi registri ±6) e i **valori in scena in shared 16 celle per volta con un solo load da 16 B per thread** (4 KB + 128 B di scale, la scena successiva caricata in registri mentre la corrente si piega): 4 viaggi di memoria invece di 16, +4,2 KB di shared (20 KB/blocco, ancora 3/SM). n=1 solo scena; n=2 + la coppia di chiavi successiva precaricata (6 registri); n=3 + la prima scena presa prima dei punteggi; n=4 entrambe. Stesse pieghe in ordine di cella, celle mascherate saltate come nel pre75 | 240 → ~130-160 µs se la catena di latenza era il limite; il resto (46 MB di DRAM ≈ 75 µs, LDS di `q` ~30 µs, 5 ondate) non si muove. Registri da leggere con `cuobjdump -res-usage` (sotto) |
| 2 | `attn_merge_v3_kernel` | `attn_merge_v2_kernel` → m_c, l_c e w_c = `__expf(m_c − M)` di ogni chunk in shared una volta per blocco (uguali per tutte le 256 dimensioni), 64 load di accumulatori in volo, piega in ordine di chunk; M con `fmaxf` (esatto) | 50 → ~8-10 µs |
| 4 | `norm_q8_1` (`native_qsa.cu`) | `norm` + `native_quantize_q8_1` → il kernel della norma con in coda il quantizzatore sulle 32 colonne consecutive del warp (lane l del warp w = colonna 32w+l+1024k: lo stesso blocco e le stesse lane del quantizzatore, stesse `warp_max`/`warp_sum`, `q8_1_finite`, `roundf(x/d)`); due volte per passo (embedding, 4 stream); `en_`/`hn_` non più scritti (nessun altro li legge) | −2 lanci, −4 µs |
| 8 | `gate_q8_1` | `gate` + `quantize` → gate sigmoide e quantizzatore nello stesso thread (thread i = elemento i in entrambi) | −1 lancio, −2 µs |
| 16 | `mtp_head_tail` (`verify_kernels.cu`) | `sampler_greedy` (1 blocco su 48542 logit) + `row_top_prob_split` + `map_ids` + `mtp_select` + `force_token` → 12 blocchi da 1024: argmax a fette con la regola di `argmax_rows` (indice più basso del massimo sopra −inf, NaN mai, 0 se nessuno: esatta per costruzione), l'ultimo blocco ricompone, calcola la probabilità con **lo stesso albero** di `row_top_prob_split` (thread virtuale vt somma i ≡ vt mod 1024 con `__expf(l−m)`, xor-sum di warp, 32 somme in ordine, `1/tot`), mappa l'id e fa le parole di `mtp_select`; la copia della riga R è spalmata su tutti i blocchi | 34 → ~8 µs |
| 32 | `combine_k10_post` (`native_gr_postops.cu`) | `combine_k10_vec4` + `post` → combine (primo prodotto arrotondato, un `fmaf` per esperto in ordine, poi lo shared) e scrittura del residuo (`__fmaf_rn(y, 2σ(inj/hc), R)`) nello stesso thread; solo T = 1, k = 10, puntatori allineati (le condizioni del kernel vec4 originale) | −1 lancio, −2,5 µs |

Registri delle quattro varianti int8 su sm_75 (`cuobjdump -res-usage`, build `2780342f`), pre75 = 73 / STACK 0 /
shared 15872: senza il vincolo di occupazione erano 81-88 registri (l'allocazione arrotonda a 8: da 81 in su restano
**2** blocchi per SM); con `__launch_bounds__(256, 3)` tutte a 80 e shared 20096 (3 blocchi/SM), ma con spill: variante
1 **STACK 0**, 2 e 3 STACK 8 B, 4 STACK 40 B (come il `mlp` scartato). Da provare nell'ordine 1, 3, 2; la 4 è attesa
peggiore. Default del bench `--attn 1`; nel motore `STRATA_CHAIN_FUSE2_ATTN=1`.

Non toccati: le proiezioni MMVQ (alla banda o quasi), le tre hyper-connection (`fused_gr.cu`, alberi di somme
propri), router+esperti (già sovrapposti al ramo laterale), `embed_rows`, `window_ids`, `norm_rope`, `kv_append`.
La testa Q5_K (167 µs, 68 MB) scende sotto i ~110 µs solo cambiando formato o kernel di somma: non a parità di bit.

Dove entra: `MtpDrafter::record_front` (norma+q8_1 solo con T = 1: i passi e i round con T = 1), `record_rest`
(attenzione, gate+q8_1, combine+post; con la coda fusa salta argmax/prob/map), `capture_round`/`capture_step` (la coda
fusa al posto di `mtp_select`+`force_token`). Solo i grafi argmax (il campionamento accoppiato tiene i suoi). Solo CUDA.

**Onestà sul bersaglio.** Le fusioni dei kernel piccoli (bit 4-32) valgono ~40-45 µs su 930. Il passo scende verso
0,5 ms solo se l'attenzione risponde (`ATTN` e bit 2: da 291 a ~160-190 µs) **e** la bolla di 33 µs sparisce. Il pavimento
di banda (~350 µs) e la testa (167 µs) restano. Stima a fusioni tutte accese, se l'attenzione rende come previsto:
930 − (200÷230) − 45 ≈ **650-690 µs**; con la bolla tolta ~620-650. Per 0,5 ms servirebbe anche la testa in un
formato più leggero (cambia le bozze: fuori dal perimetro) o metà testa sulla 3060 (lavoro grande, altra storia).

## 3. Come misurare

### 3a. Micro-bench `chain_step_bench` (prima: parità e µs di ogni segmento)

Target CMake fuori dal default (solo CUDA):

```
python ...\20261006_due_schede\build_lock.py C:\Users\imanu\source\repos\Strata-wt\chainfuse --target chain_step_bench
C:\Users\imanu\source\repos\Strata-wt\chainfuse\build-sm7586\chain_step_bench.exe --device 0 --iters 200 --attn 4 --graph
```

(`--device`: l'ordinale CUDA della 2080 Ti nel processo; senza `CUDA_VISIBLE_DEVICES` è 0.) Dati sintetici alle forme
reali del drafter (2560, hc 4, 24 teste × 256, 2 teste K/V, 32768 celle int8 e fp16, testa 48542). Per ogni segmento:
sequenza di default contro fusa, **memcmp** di ogni uscita che il passo poi legge (parziali e uscita dell'attenzione su
10 larghezze × 2 page table × 4 varianti, byte q8_1, R/tok/out/prob della coda con pareggi, NaN, tutti −inf, ultimo indice,
tutti uguali), poi i µs in B-A-B (fuso, default, fuso) con eventi CUDA. Uscita 0 = tutto bit per bit; 77 = niente GPU.
Riga finale: µs di default e fusi dei soli segmenti toccati e il risparmio per passo. `--attn 1..4` sceglie la variante
del segmento «attention»; la tabella «chunk pass» le misura tutte e quattro: la migliore va in `STRATA_CHAIN_FUSE2_ATTN`.
`--graph`: costo di un fork/join dentro un grafo di 50 kernel da un blocco (la bolla del §1).

Da leggere: `chunk pass stage variant n vs pre75` (atteso 0,55-0,65; sopra 1 la variante va scartata), `merge v3 vs v2`
(misurato 0,73), `head tail` (misurato 0,15),
`attention segment`. Se un segmento stampa `FAIL`, quella fusione non va accesa (maschera) finché non si capisce.

### 3b. Nel motore: B-A-B col banco forzato

Stesso exe e stessi argomenti del Daily (`docs/UPSYNC.md`, «Riprodurre il Daily»), `STRATA_DECODE_TIMING=1`,
`STRATA_FORCE_IDS=<liste>`, `STRATA_FORCE_WINDOWS=<traccia di riferimento>`; con `STRATA_CP_CHAIN_PRE`/`CHAIN_FUSE` come
nel riferimento. Tre corse alternate: B (`STRATA_CHAIN_FUSE2=1`), A (senza), B.

- Identità: `strata work: … accepted X of Y drafts, … target picks replaced on verified rows Z` uguale nei due bracci
  (stesse bozze); senza forzatura il testo greedy identico.
- Tempo: `the chain from its launch to B: fresh …` (la catena sul percorso critico delle finestre fresche: −0,3/−0,45 ms
  attesi a 3-4 passi), classe `speculative` del `rollback cost` (la catena accanto allo stadio 1: −0,2/−0,3 ms per
  passo in meno di GPU contesa), `strata pipeline: … ms/window`.
- Per attribuire: `STRATA_CHAIN_FUSE2_ATTN=n` con `MASK=0` (solo il chunk), `MASK=2` (solo il merge), `=16` (solo coda),
  `=44` (norm, gate, post), poi tutte; `STRATA_MTP_SHARED_BRANCH=0` da solo e insieme.
- nsys (come `DET-cp-nsys5`): la mediana del grafo del passo e la lista dei kernel con lo script
  `chain_step_kernels.py` (scratchpad della sessione; ricostruibile: raggruppa i kernel per `correlationId` delle
  `cudaGraphLaunch`, i grafi di 54 → ora 45 kernel sul device 1).

### 3c. La bolla

B-A-B di `STRATA_MTP_SHARED_BRANCH=0` da solo (già esistente, nessuna fusione): se il passo perde ~10 µs netti e la
bolla sparisce dalla traccia, la strada è togliere il ramo laterale e fondere lo shared expert nel percorso principale
(un kernel: gate+up+SiLU+q8 e down, ~28 µs → ~15).

## 4. Rischi

- **Parità.** Garantita per costruzione (stesse espressioni, stesso ordine) e controllata dal micro-bench su dati
  sintetici; sul motore il confronto è `accepted X of Y` / `target picks replaced` identici fra i bracci. Punti delicati:
  la contrazione in `fma` delle espressioni dei punteggi (`k[0]*q.x + …`) dipende dal compilatore, ma è la stessa
  espressione nello stesso file con gli stessi flag del kernel pre75 (il pre75 stesso si affida a questo); `__expf` e
  `rsqrtf`/divisioni sotto `--use_fast_math` restano nei file originali.
- **Registri.** La lezione del primo kernel: +7 registri e spill bastano a perdere il 50%. `attn_chunk_kernel_stage` ha
  `__launch_bounds__(256)` come il pre75; dopo ogni build leggere `cuobjdump -res-usage build-sm7586\CMakeFiles\
  strata_kernels.dir\src\kernels\cuda\qsa_decode_attn.cu.obj` (sola lettura, niente GPU): REG ≤ 85 e STACK 0 per le
  quattro varianti, altrimenti la variante è da scartare. `attn_merge_v3` tiene 64 accumulatori in volo per thread: 24 blocchi,
  nessun problema di occupazione.
- **Code 1024 thread.** `mtp_head_tail` usa il contatore «ultimo blocco» di `argmax_rows` (già in produzione) e uno
  scratch zero nell'arena del drafter (memset all'allocazione, rimesso a zero a ogni lancio).
- **Perimetro.** Solo T = 1 per norma+q8_1 e combine+post (i round con T > 1 tengono il percorso vecchio: i loro kernel
  sono gli stessi di prima); solo grafi argmax; solo CUDA; pool q4_0 → kernel di prima per l'attenzione. `--mtp-hnorm
  stream` tiene la norma raggruppata (non fusa).
- **Bolla di 33 µs** non toccata dalle fusioni (è fra due kernel non fusi): se è un arco fra stream, la toglie solo il
  ramo laterale spento (§3c).

## Stato della build

Compilato sì (`build_lock.py`, CUDA 12.6, sm_75+86, test spenti): `strata.exe` dal codice del motore con
`attn_chunk_kernel_stage`, sha256 `2780342f8792d047bb5f690cb70cbb6d920163e3c04e3828125352d93a333be5`, e
`chain_step_bench.exe` (target a parte), entrambi in `C:\Users\imanu\source\repos\Strata-wt\chainfuse\build-sm7586\`.
Installato no, provato no da me: nessuna GPU usata, nessun motore avviato; i µs delle varianti `stage` sono stime da
verificare col bench (§3a) prima del B-A-B. La prima build (`e6b33daa`, sha `040843a5…`) è quella misurata nel §1b.
Senza `STRATA_CHAIN_FUSE2` l'exe deve comportarsi come `lab/critpath2` `8b4fd975`.

## File toccati

- `src/kernels/cuda/qsa_decode_attn.cu`, `include/strata/kernels/qsa_decode_attn.hpp`: `attn_chunk_kernel_stage`,
  `attn_merge_v3_kernel`, `qsa_decode_attn_batch_fuse2`, `qsa_decode_attn_fuse2_built`; `chunks_only` 2..5 e
  `merge_only` 2 per il bench.
- `src/kernels/cuda/native_qsa.cu`, `include/strata/kernels/native_qsa.hpp`: `native_qsa_rms_norm_q8_1`,
  `native_qsa_gate_q8_1`.
- `src/kernels/cuda/verify_kernels.cu`, `include/strata/kernels/verify_kernels.hpp`: `mtp_head_tail`,
  `mtp_head_tail_scratch_bytes`.
- `src/kernels/cuda/native_gr_postops.cu`, `include/strata/kernels/native_gr_postops.hpp`: `native_moe_combine_k10_post`.
- `src/core/mtp.cpp`, `include/strata/core/mtp.hpp`: `chain_fuse2_mask`/`chain_fuse2_attn`, il cablaggio, `tail_scratch_`.
- `src/kernels/chain_step_bench.cpp`, `CMakeLists.txt`: il micro-bench (`EXCLUDE_FROM_ALL`).
