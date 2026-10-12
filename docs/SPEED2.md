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

## 5. Risultati

### Passata 1 (commit 0107c261, build `build-speed2-0107c261`, 12/10 10:20)

`dense_bench 300` su entrambe le schede: **384 celle per scheda, tutte `identical 1col=ok`** (def, b6, wr, wr:r1,
wr:r4, wr:s0 × 16 forme × T=1..4). Run engine breve (`DET-x7-speed2-wr`, L02 64 token, `STRATA_NODECUT=1
STRATA_DENSE_WR=1`): `L02.txt` identico a off; la riga `strata dense_wr:` c'è.

Tempi sulla 2080 Ti (µs per chiamata, def → miglior wr):

| forma | T=1 | T=2 | T=3 | T=4 |
|---|---|---|---|---|
| qkv IQ4_XS | 31,5 → 30,9 (+2%) | 35,2 → 32,9 (+7%) | 44,3 → 37,7 (+15%) | 53,0 → 46,2 (+13%) |
| qkv Q4_K | 37,7 → 32,2 (+15%) | 38,4 → 34,7 (+10%) | 44,6 → 38,8 (+13%) | 53,3 → 47,7 (+11%) |
| qkv Q5_K | 39,7 → 37,4 (+6%) | 42,7 → 39,9 (+7%) | 48,5 → 43,6 (+10%) | 60,4 → 53,4 (+12%) |
| qkv Q6_K | 56,7 → 47,3 (+17%) | 51,1 → 52,1 (−2%) | 56,1 → 57,9 (−3%) | 61,4 → 62,6 (−2%) |
| q IQ4_XS / Q4_K | +4% / +16% | +8% / +10% | +14% / +11% | +13% / +11% |
| gate Q4_K / Q5_K | +12% / +5% | +7% / +5% | +8% / +8% | +10% / +12% |
| out Q4_K / Q6_K (6144 ingressi) | −8% / −12% | +1% / 0% | −8% / −8% | −25% / −8% |

- Il vincitore è quasi sempre **`wr:s0`**: le attivazioni in shared (SoA) **perdono** su Turing e su Ampere (la copia e la
  barriera costano più delle LDG che vanno in L1). Default cambiato: `STRATA_DENSE_WR_STAGE=0`.
- Q6_K (T=32: ogni blocco su tutto il warp) e le righe da 6144 ingressi peggiorano: esclusi dai default
  (`STRATA_DENSE_WR_TYPES=12,13,23`, `STRATA_DENSE_WR_MAXIN=2560`; il bench li misura con `a1:k0`).
- Peso sulla finestra: con i formati e le forme che guadagnano (qkv/gate/q in IQ4_XS, Q4_K, Q5_K: ~45% dei byte densi)
  il risparmio è **~50-60 µs a T=2, ~100-150 µs a T=3-4**, cioè 0,3-0,8% della finestra: molto meno dei −0,7/−1,0 ms
  stimati. I kernel di default stavano già al 60-74% del picco e il WR li porta al 65-78%: il vincolo bitwise fissa la
  mappa lane → elementi, quindi i byte in volo per warp restano R × 2,7 KB.

### Passata 2 (commit 49ddc63f, build `build-speed2-49ddc63f`, 12/10 10:47)

Caricamenti dei pesi delle 4 catene × R righe tutti prima dell'applicazione (indirizzi clampati). **Peggio**: sulla
2080 Ti il guadagno complessivo sulle celle prese scende dal 6,6% al 2,9% e il `wr` di default perde su qkv Q4_K T=2
(−7%) e Q5_K T=2 (−11%) dove la passata 1 guadagnava +10/+7%: pressione sui registri, ptxas sovrappone meglio da sé
nella forma con la guardia. Bench: 576 celle per scheda tutte `identical 1col=ok`. Il run engine `DET-x7-speed2b-wr`
ha però dato un **`L02.txt` diverso** dai 5 run precedenti (tutti identici fra loro); il run `off` gemello si è
bloccato (deadline di generazione, log fermo a «everything loaded»): in corso un secondo giro off/wr sullo stesso build
(`DET-x7-speed2c-*`) per distinguere un bug del kernel da un run non deterministico.

### Passata 3 (commit 5a357afb, build `build-speed2-5a357afb`, sha256 `547a5ce4…`, 12/10 12:30)

Ciclo dei blocchi riportato alla forma della passata 1 (guardia per catena, pesi caricati dentro la guardia); restano
i default nuovi (`STAGE=0`, tipi 12/13/23, `MAXIN=2560`).

`dense_bench 300` su entrambe le schede: **576 celle per scheda, tutte `identical 1col=ok`**. Il `wr` di default
(R = 2, attivazioni da globale) sulle forme che prende (qkv/gate/q in IQ4_XS, Q4_K, Q5_K), somma dei µs di una
chiamata per forma:

| scheda | T=1 | T=2 | T=3 | T=4 |
|---|---|---|---|---|
| 2080 Ti (`device 0` del bench) | 262 → 241 (−8%) | 280 → 263 (−6%) | 332 → 299 (−10%) | 400 → 358 (−10%) |
| 3060 (`device 1`, rumorosa: OktoCGControl in primo piano) | 432 → 431 (0%) | 505 → 489 (−3%) | 610 → 530 (−13%) | 738 → 608 (−18%) |

Per forma, sulla 2080 Ti a T=2: qkv IQ4_XS 35,1 → 33,1, Q4_K 38,1 → 35,0, Q5_K 42,6 → 40,1; gate IQ4_XS 22,9 → 22,8,
Q4_K 25,6 → 24,3, Q5_K 28,2 → 27,3 µs. Peso sulla finestra dello stadio 1: le chiamate prese sono 16 gate + 14 qkv
+ 3 q per gruppo (tipi dal GGUF, strati 16-47) → **~0,06 ms a T=2, ~0,13 a T=3, ~0,16 a T=4**, cioè lo 0,4-1% dello
stadio. È lo stesso ordine della passata 1.

**Verdetto: bitwise sì (bench), guadagno marginale. Leva archiviata**: il codice resta opt-in (`STRATA_DENSE_WR=1`),
fuori dalle build da banco. Il vincolo bitwise fissa la mappa lane → elementi del layout esatto, quindi i byte in volo
per warp restano quelli del B6: per salire davvero serve un altro albero di somma, che non è bitwise.

### Il L02 diverso della passata 2: non è un bug del kernel, il banco DET-x7 non è deterministico

Corse brevi (L02, 64 token, giorno) sulle build di speed1/speed2/speed3, `md5` di `L02.txt`:

| braccio | corse | uguali al riferimento `d48bfa50` | diverse |
|---|---|---|---|
| `off` (NODECUT=0, DENSE_WR=0, o le sole leve EXPDEC_WT/MAGIC/DUMMY_NODES) | 9 | 9 | 0 |
| `wr` passata 1 (0107c261) | 1 | 1 | 0 |
| `wr` passata 2 (49ddc63f) | 4 | 1 | 3 (`e020c9d8` ×2, `d770b592`) |
| `wr` passata 3 (5a357afb) | 2 | 1 | 1 (`d556a925`) |
| `nodecut` (NODECUT=1, DENSE_WR=0) | 4 | 3 | 1 (`07617392`) |
| `wronly` (NODECUT=0, DENSE_WR=1) | 3 | 2 | 1 (`4d097cff`) |

Le corse `wr` della passata 2 divergono **in punti diversi** (carattere 38 e 213): un difetto deterministico del
kernel darebbe lo stesso testo sbagliato a ogni corsa. Divergono anche la passata 3, NODECUT da solo e DENSE_WR
da solo, e quasi tutte allo stesso token (carattere 38, «explain» contro «Explain»: un quasi pareggio che anche la build `cand2`
risolve con «Explain» 3 volte su 3). I kernel sono bitwise nei bench (576/576 celle per scheda).

La causa che varia da corsa a corsa è la **cache esperti della 3060**. Il banco usa `--expert-cache auto --elastic`
con `STRATA_VRAM_FREE=device`: la cache si dimensiona dalla VRAM libera dell'intera scheda letta con NVML, e la 3060
pilota il monitor 8K e OktoCGControl. Il nucleo fisso va da **1611 a 2949 slot** a seconda della corsa. Nelle tre
corse più lontane (`e020c9d8` della 2g, `07617392`, `4d097cff`) la coda elastica si restringe anche durante la corsa
(`strata elastic: -707/-710/-711 slots`): hit rate 78-79% contro 87-90%, 461-540 esperti letti via PCIe contro
25-160. Un esperto calcolato dalla CPU o via PCIe non dà gli stessi bit di uno in VRAM (SPEED1 §2), quindi il testo
greedy cambia con la residenza (e la coda elastica ruota secondo i tempi: anche a pari nucleo, 2070 slot, una corsa su
cinque è uscita diversa). I 9 `off` uguali sono stati fortuna: nucleo di 2070-2754 slot e nessun restringimento.

Conseguenza: con questa configurazione un L02 identico è un indizio, non una prova di bitwise.

### Oracolo a cache fissa (12/10 pomeriggio)

Stesse corse con `--set-opt=--expert-cache=1500 --drop-opt=--elastic` (cache della 3060 fissa a 2336 slot, niente
coda elastica; script `det_check_fixed.py`), build `build-speed2-5a357afb`:

| braccio | `L02.txt` | lookup / hit / PCIe |
|---|---|---|
| `off` ×2 | `5fefd82e` ×2 | 42643 / 34560 / 397 (uguali al numero) |
| `wronly` (DENSE_WR=1) | `5fefd82e` | 42643 / 34560 / 397 |
| `nodecut` ×2 (NODECUT=1) | `1fb98380` ×2 | 46621 / 38239-38229 / 419 |
| `STRATA_QFUSE_HC=1` da solo | `1fb98380` | 46621 / 38229 / 419 |
| `STRATA_FUSE_WAIT=1` da solo | `5fefd82e` | 42643 / 34560 / 397 |
| `STRATA_FUSE_HITS=1` da solo | `5fefd82e` | 42643 / 34560 / 397 |

A cache fissa la corsa è deterministica (stessi lookup al numero) e risponde:

- **DENSE_WR è bitwise anche nel motore.**
- **NODECUT non è bitwise, per colpa di `QFUSE_HC`**: cambia il testo in modo riproducibile. `FUSE_WAIT` e
  `FUSE_HITS` sono bitwise. Per un B-A-B di NODECUT bitwise: `STRATA_NODECUT=1 STRATA_QFUSE_HC=0` (la variabile
  propria vince), sulla build `build-speed1-127de4d6` così com'è. Dove QFUSE_HC si separa dal percorso di default è
  da indagare (`lab/speed4`, SPEED4.md).
- Per i controlli DET-x7 di giorno usare la cache fissa: con `--expert-cache auto --elastic` e il monitor sulla 3060
  il testo dipende dalla VRAM libera.

## Stato

- 12/10 11:15: passata 3 scritta.
- 12/10 12:30-16:30: compilata (`build-speed2-5a357afb`), bench sulle due schede (576/576 bitwise), run engine.
  **Bitwise sì (bench e motore a cache fissa); guadagno 0,06-0,16 ms a finestra sullo stadio 1: leva archiviata**,
  `STRATA_DENSE_WR` resta opt-in e fuori dalle build da banco. Il L02 diverso era il banco, non il kernel (§5).
