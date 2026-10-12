# lab/speed2 — proiezioni dense sullo sm_75 con lo schema di EXPDEC

Base: `lab/speed1` (0.1.43 + pila del fork + tagli di nodo CUDA). Leva 2 della classifica di `docs/SPEED1.md`.

## 1. Dove va il tempo (2080 Ti, stadio 1, traccia nsys DET-x3-hostgap-nsysA, finestra mediana)

Finestra a T=1: 14,1 ms di kernel; a T=2: 16,1 ms. Le proiezioni dense (Q4_K / Q5_K / Q6_K / IQ4_XS: qkv, gate,
out, q, k/v, esperto condiviso) degli strati 16-47 leggono **~1,3 GB a finestra** (tabella GGUF, `dense_bytes.py`);
a 616 GB/s sarebbero 2,1 ms. Misurate:

| kernel (T=2) | n | µs/lancio | MB/lancio | GB/s | % picco |
|---|---|---|---|---|---|
| `b6_mmvq_rows` qkv 10240 righe | 24 | 43,1 | 17,4 (media dei 4 formati) | 404 | 66 |
| `b6_mmvq_rows` out 2560 righe | 30 | 31,4 | 12,9 Q6_K / 8,9 Q4_K | ~380 | ~62 |
| `b6_mmvq_rows` gate 6144 righe | 19 | 25,4 | 9,6 | 378 | 61 |
| `b6_mmvq_rows` q 12288 righe | 7 | 54,5 | 22,3 | 409 | 66 |
| testa Q5_K 248320 righe | 1 | 874 | 437 | 500 | 81 |

A T=1 i kernel a colonna singola stanno al 44-76% (IQ4_XS i peggiori: 2560 righe 272 GB/s). I kernel HC (bf16,
`gr_up_multi` 19,7 µs e `gr_down_staged` 17,6 µs per 6,55 MB) stanno al 54-67%. Totale denso ~4,1 ms (T=2) + HC
2,4 ms: portare tutto al 75% del picco vale ~−0,9 ms a finestra; all'85% ~−2 ms.

Perché il B6 resta al 60-66%: ogni blocco (128 thread) fa 2 righe × 10 blocchi di 256 = 2,7 KB di pesi (IQ4_XS) in
una sola iterazione, poi parziali in shared, `__syncthreads`, riduzione delle 2×T somme da parte del solo warp 0.
Con 5120 blocchi da ~4 µs il costo fisso per blocco domina: una «ondata» di 68×16 blocchi muove 2,9 MB in ~9 µs.
Le attivazioni (9 LDG per colonna per blocco, da L1) e il `kqs` scalare fanno il resto.

## 2. Cosa fa questo ramo: `STRATA_DENSE_WR=1` (opt-in, CUDA)

`dense_wr_kernel<F, NCOLS, R, STAGE, MINB>` in `src/kernels/cuda/native_mmvq.cu`, formati Q4_K / Q5_K / Q6_K /
IQ4_XS, 1 ≤ ncols ≤ 4.

- **Un warp per R righe intere.** Il layout esatto di default divide i blocchi di una riga fra i 4 warp del blocco
  (thread `tid = 32·y + x` → blocchi `tid/T, +BPI, …`), parcheggia i parziali dei warp 1-3 in shared e li fa sommare al
  warp 0 nell'ordine 1, 2, 3, poi butterfly. Qui la lane x esegue da sé le quattro catene (`vy = 0..3`, stessi blocchi
  nello stesso ordine) in quattro accumulatori, le somma nello stesso ordine e fa lo stesso butterfly: **stessi float
  nello stesso ordine → bitwise il layout esatto**, quindi bitwise una chiamata a colonna singola. Niente parziali in
  shared né `__syncthreads` per blocco; i caricamenti dei pesi delle R righe per posizione di blocco sono indipendenti
  e viaggiano insieme.
- **Attivazioni in shared, SoA** (`STAGE`): i blocchi q8_1 di ogni colonna copiati una volta per blocco (8 int di `qs`
  per blocco, poi le scale basse), letti con LDS (IQ4_XS: due LDS.128) al posto delle LDG per thread. `xload_s` in
  ogni `B6X` produce gli stessi int e float di `xload`. Sopra 48 KB (ncols × n_in/32 × 36 B) si torna alle letture
  globali (`STAGE = false`), come il B6.
- Griglia `⌈n_out / 4R⌉` blocchi da 128 thread, `__launch_bounds__(128, MINB)`.

| variabile | default | significato |
|---|---|---|
| `STRATA_DENSE_WR` | 0 | 1 accende |
| `STRATA_DENSE_WR_R` | 2 | righe per warp: 1, 2, 4 |
| `STRATA_DENSE_WR_STAGE` | 1 | attivazioni in shared |
| `STRATA_DENSE_WR_MINB` | 0 | blocchi minimi per SM: 2 o 4; 0 = 4 se ncols·R ≤ 8, altrimenti 2 |
| `STRATA_DENSE_WR_MINBLK` | 256 | sotto tanti blocchi di lancio restano i kernel di default (k/v 512 righe, esperto condiviso 640) |
| `STRATA_DENSE_WR_T1` | 1 | anche a ncols = 1 (al posto dei kernel a colonna singola) |

Gancio: `STRATA_DENSE_WR_TRY(Traits)` nei quattro wrapper `native_*_mmvq`, prima del ramo `ncols > 1`; non con
`STRATA_TSUM=1` né col layout upstream (`native_mmvq_set_multi_exact(false)`). Riga di log `strata dense_wr:` al primo
lancio. HIP: invariato (il blocco è `#if !defined(__HIPCC__)`).

API per il banco: `NativeDenseWrParams`, `native_mmvq_set_dense_wr` / `native_mmvq_dense_wr` /
`native_mmvq_dense_wr_takes` in `include/strata/kernels/native_mmvq.hpp`.

## 3. Micro-bench: `dense_bench` (target `EXCLUDE_FROM_ALL`)

    build-sm7586/dense_bench [reps=300] [device=-1] [shapes=model] [choices=def,b6,wr,wr:r1,wr:r4,wr:s0] [T=1,2,3,4]

Le forme vere degli strati 16-47 (qkv 2560→10240 Q6_K/IQ4_XS/Q4_K/Q5_K, gate 2560→6144, out 6144→2560, q
2560→12288, kv 2560→512, shexp 2560→640), pesi sintetici con scale fp16 sane che ruotano su ≥ 96 MB (letti freddi).
Per cella: memcmp con `def` (colonna singola a T=1, `STRATA_B6_MMVQ_ROWS=1` sopra, come la config) e con T chiamate a
colonna singola; mediana di 5 giri di `reps` chiamate; GB/s e % del picco della scheda. Exit 1 se una cella differisce.

## 4. Verifica e passi successivi

1. `dense_bench` su CUDA1 (2080 Ti) e CUDA0 (3060): tutte le celle `identical 1col=ok`; scegliere R / STAGE / MINB
   dai µs per forma; se una forma non guadagna, alzare `MINBLK` o lasciarla ai default.
2. Run breve dell'engine con `STRATA_DENSE_WR=1`: `L02.txt` e primi logit identici a `=0` (`speed1_check.py` con
   l'arm `wr`).
3. B-A-B del coordinatore: A = config attuale, B = `STRATA_NODECUT=1 STRATA_DENSE_WR=1` (+ i knob scelti).
4. Dopo: i kernel HC (`gr_up_multi` / `gr_down_staged`, 2,4 ms a finestra al 54-67%): `STRATA_GR_FAST=1` esiste già
   (`gr_up_fast_kernel`: le 8 righe del warp caricate prima dei dot, bitwise) e va misurato con `fused_gr_bench`;
   poi la stessa cura al `gr_down_staged` (41 blocchi soli: prefetch dei pesi più profondo, meno righe per blocco).

## Stato

- 12/10 10:00: kernel, ganci, knob, bench e doc scritti; **compilato no / provato no** (build in coda dietro
  `bench.lock`, poi `dense_bench` sotto lo stesso lock).
