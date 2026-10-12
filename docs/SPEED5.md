# lab/speed5 — la lettura delle righe PLE che crolla, e le leve rimaste

Base: `lab/speed4`.

## 1. Perché la gather PLE crolla (B-A-B DET-x14-s4, braccio B1)

**Sintomo.** B1 ha fatto 31,8/27,5 t/s contro ~66/72. La gather PLE del decode è salita a 13,5 ms per finestra
(A e B2: 0,04), con 15,9 s di attesa del prefill al chunk 121856.

**Cosa dicono i log (`ple io:` di fine corsa, i tre bracci):**

| braccio | hit row cache | letture SSD | latenza media | p99 | letture > 100 ms | bloccato |
|---|---|---|---|---|---|---|
| A | 53,4% | 585 904 (2,45 GB) | 2,7 ms | 6,4 ms | 0 | 6,8 s |
| **B1** | **54,6%** | 589 543 (2,47 GB) | **14,3 ms** | **259 ms** | **27 147** | **46,0 s** |
| B2 | 53,3% | 589 121 (2,46 GB) | 2,9 ms | 6,8 ms | 0 | 6,8 s |

- **La cache non si svuota.** La hit rate è la stessa nei tre bracci, e anche il numero di letture. Cambia solo la
  latenza di ogni lettura: in B1 è 5 volte più alta, con code oltre 250 ms. Il collo è il disco, non la cache.
- **Il disco non è saturo di banda.** In B1, durante il decode, C: legge 2-35 MB/s (`ambiente.json`), eppure la
  latenza è di 14 ms. L'unità è lenta per singola operazione. Nello stesso B-A-B la sonda di A ha misurato 999
  righe/s, e A ha letto ferma a ~60 MB/s per 120 s prima di risalire a 1 GB/s: l'unità rallenta a tratti.
- **L'unità.** C: è una Lexar EQ790 NVMe senza DRAM, piena al 93% (75-84 GB liberi su 931): è anche il disco di
  sistema su cui lavora l'owner (OneDrive ed Edge compaiono nei campioni di B1). Senza privilegi di amministratore
  non leggo la temperatura, quindi lo strozzamento termico o il GC dell'unità restano ipotesi; la conseguenza è
  misurata.

**Le due cache, e cosa le svuota.**

- **La row cache è memoria nostra**: `PleReader`, set-associativa a 8 vie, 90 B a riga, nell'heap del processo.
  Con `--ple-row-cache 16777216` alloca ~1,6 GB all'avvio e ne usa ~55 MB in una corsa L01+L02 (~600K righe).
  Windows non la scarta; muore con il processo, quindi **ogni avvio riparte da zero**: le ~570K letture SSD
  (2,4 GB in pagine da 4 KiB per 54 MB di righe) si rifanno a ogni braccio e a ogni riavvio del Daily.
- **La page cache di Windows (standby list) le righe PLE non le vede mai**: in modalità `direct` (il default) il
  file si apre con `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS`
  (`src/platform/direct_file.cpp`). Le pagine dello shard 00002 che si vedono in standby (i «~5 GB di tabella in
  RAM») vengono da altri lettori dello stesso file, come i pesi densi di `--native-dense-gguf ...00002` all'avvio o
  le corse con `--ple-io mmap/ram`. La lettura diretta non le usa. Sono pagine pulite di file: Windows le ricicla per
  prime sotto pressione (allocazioni del motore stesso, copie grandi, applicazioni dell'owner).

**Nel Daily vero.** Sì: `D:\ds4_work\strata\sota\strata_sota.log` (PLE da D:, SATA) registra 16 attese del primo
chunk del prefill, con mediana 1,8 s, p90 3,5 s e massimo 4,7 s di TTFT in più. Il primo chunk dopo un periodo di
inattività trova l'unità «addormentata» o lenta.

## 2. La correzione: `--ple-io ram` (esiste già, opt-in) e `STRATA_PLE_RAM_PREFETCH=1` (nuova)

Con `--ple-io ram` il motore mappa la tabella e la blocca in RAM (`VirtualLock`, minimo del working set alzato):
26,8 GiB, una volta all'avvio, poi zero letture dal disco. Le righe sono gli stessi byte del file, quindi i bit non
cambiano.

Misura (build `build-speed4-311b6fda`, L02 a cache fissa, `DET-x7-ple-pleram`, PLE da C:):

| | `direct` (A/B2) | `--ple-io ram` |
|---|---|---|
| avvio della tabella | — | **31,7 s** (bloccata) |
| gather PLE decode | 0,040-0,060 ms/finestra | **0,017 ms/finestra** |
| prefill L01 | 78-80 s | 78,5 s |
| `L02.txt` | `5fefd82e` | **`5fefd82e`** |

RAM: 26,8 GiB bloccati. Con il motore spento la macchina ha 80,8 GB liberi; il Daily usa ~33 GB, quindi ne
restano ~18 all'owner. La row cache in modalità ram non si alloca (il lettore diretto non si apre), quindi
`--ple-row-cache` non costa.

`STRATA_PLE_RAM_PREFETCH=1` (questo ramo, `src/kernels/ngram.cpp`): prima del lock, `PrefetchVirtualMemory` a
blocchi da 256 MiB e 8 thread che toccano le pagine. Serve a non far pagare a `VirtualLock` il caricamento da un
solo thread a un cluster alla volta, e conta soprattutto da D: (SATA).

Misura da D: (§4).

## 3. Leve rimaste della roofline (stato)

- **QFUSE_HC** (−1 nodo a strato, parte di NODECUT): non è bitwise su CUDA (SPEED2 §5, SPEED4 §6). I quantizzatori
  (`gr_up_q32_kernel`, `gr_q8_tail`, `native_quantize_q8_1`, `q8_1_store`) sono identici; i kernel `up` hanno la
  stessa aritmetica. La differenza va cercata nel modo in cui il verify consuma `xq_`/`nat_xq_` (rami laterali,
  ordine). Corse mirate in §4.
- **Copie pageable → pinned** (cache della 2080 Ti): fuori dal percorso critico, su stream propri, e il driver fa
  già lo staging: −0,2 ms al massimo. Non fatta.
- **Seconda finestra B sulla 3060, thread host per stadio, fusioni nuove (norma+quantize, swiglu+quantize,
  doorbell+route)**: cambi strutturali da più giorni ciascuno, da pianificare a parte.

## 4. Misure dopo il B-A-B DET-x15 (12/10 sera, PC dell'owner libero)

### 4a. PLE da D: (SATA), build `build-speed5-1c08017a`, L02 a cache fissa

| braccio | avvio tabella | attesa prefill | gather decode | `L02.txt` |
|---|---|---|---|---|
| `direct` (default) | — | chunk 0: 2 037 ms | 0,131 ms/finestra | `5fefd82e` |
| `--ple-io ram` | **70,9 s** (bloccata, a freddo) | nessuna | **0,017 ms/finestra** | `5fefd82e` |
| `--ple-io ram` + `STRATA_PLE_RAM_PREFETCH=1` | 3,0 + 0,6 s (**a caldo**: pagine ancora in standby dalla corsa prima) | nessuna | 0,017 ms/finestra | `5fefd82e` |

Bitwise in tutti e tre. Il prefetch a freddo non è misurato: senza amministratore non si può svuotare la standby list.
Il limite fisico da SATA è ~50 s (26,8 GiB a ~550 MB/s), contro i 70,9 s del caricamento a un thread. Nel B-A-B
DET-x15 (PLE da D:) la lettura diretta ha avuto latenza media di 12-13 ms, gather di 0,19-0,25 ms a finestra e 4 608
letture oltre 100 ms in B1. Con `--ple-io ram` tutto questo sparisce, a costo di 26,8 GiB bloccati e dell'avvio.

### 4b. QFUSE_HC: dove nasce la differenza (build `build-speed4-311b6fda`, cache fissa)

| braccio | `L02.txt` |
|---|---|
| off, off senza side mixer, off senza DF_BRANCH | `5fefd82e` |
| QFUSE_HC=1 con il vecchio kernel `up` (`STRATA_QFUSE_HC_LOCAL=0`) | `1fb98380` |
| QFUSE_HC=1 senza side mixer (`STRATA_SIDE_MIXER=0`) | `1fb98380` |
| QFUSE_HC=1 senza DF_BRANCH (`STRATA_DF_BRANCH=0`) | `1fb98380` |

Non dipende dal kernel `up` (q32 o multi con coda atomica), né dai rami laterali. Cambia il testo il solo fatto di
usare l'immagine q8 scritta dalla lettura HC al posto di quella quantizzata a parte. Il prossimo passo è un
interruttore per metà (attenzione `xq_` o FFN `nat_xq_`) per capire quale delle due immagini differisce. Priorità
bassa: vale −1 nodo a strato.

### 4c. Il percorso critico: perché speed4 è neutro

Dal B-A-B DET-x15 (`strata pipeline classes`), A contro B2:

| | finestre fresche | speculative |
|---|---|---|
| L01 | 29,20 → 29,04 ms | 19,89 → 19,81 ms |
| L02 | 27,54 → 27,18 ms | 18,69 → 18,48 ms |

Le speculative sono il solo stadio 1, eppure scendono di appena 0,1-0,2 ms contro i −0,7/−1,0 ms tolti ai kernel.
Lo stadio 1 quindi non è legato ai propri kernel ma alle attese: `GPU-reach wait` vale circa 14 ms a finestra, e
nella traccia nsys i `wait_flag_ge` della 2080 Ti sono il 37% del suo tempo kernel (piano ed esperti CPU dell'host,
hand-off dallo stadio 0). Le fresche sono il 70-73% delle finestre: catena MTP (3,4-4,1 ms dal lancio a B, con la
3060 ferma) → stadio 0 → stadio 1, tutto in serie.

Stadio 0 nella traccia (`DET-x3-hostgap-nsysB2`, 148 passi della 3060): 14,2 ms di kernel a passo, di cui densi 3,9,
esperti 2,7, hc 2,4, attese host 2,2, QSA 1,1, protocollo 0,9, GDN 0,6. Sono ~0,75 ms a layer contro ~0,55 sulla
2080 Ti. NODECUT, TOPK_GRID ed EXPDEC sono già attivi anche sullo stadio 0: le righe `strata topk grid: CUDA0` e
`strata expdec: CUDA0` sono della 3060, perché con `CUDA_VISIBLE_DEVICES=1,0` CUDA0 è la 3060. Il loro guadagno lì
è piccolo (top-k: 4 layer QSA × ~45 µs).

### 4d. Ribilanciare lo split (config reale, PLE da D:, 256 token di L02, una corsa per braccio)

| `--layer-split` | 3060: slot cache | 2080 Ti: slot cache | ms/finestra | fresche | speculative | tok/finestra | t/s |
|---|---|---|---|---|---|---|---|
| 16 (oggi) | 4 506 (nucleo 3 454) | 9 148 | 27,24 | 28,99 | 19,99 | 1,79 | 65,7 |
| 15 | 4 386 (3 329) | 9 106 | 26,03 | 28,51 | 19,25 | 1,64 | 63,1 |
| 14 | 4 436 (3 370) | 9 128 | 25,57 | 28,48 | 19,36 | 1,64 | 64,2 |

- Sta in VRAM su entrambe le schede.
- Le finestre si accorciano: fresche −0,5 ms, media −1,2/−1,7 ms. È coerente con un layer spostato che costa
  ~0,75 ms sulla 3060 e ~0,55 sulla 2080 Ti: in una finestra in serie si guadagna la differenza, ~0,2 ms a layer.
- I token per finestra cambiano col testo (la residenza cambia i bit), quindi i t/s di una corsa singola non
  decidono. Serve il B-A-B a finestre forzate: A = split 16, B = split 14.

## 5. Per il B-A-B

| prova | build | cosa cambia nel braccio B |
|---|---|---|
| split | qualunque (es. `build-speed4-311b6fda`) | `--layer-split 14` (o 15) al posto di 16 |
| PLE in RAM | `build-speed5-1c08017a` (sha256 `e0edb91e…`) | `--ple-io ram` + `STRATA_PLE_RAM_PREFETCH=1`; PLE da D: |

## Stato

- 12/10 sera: prefetch compilato (`build-speed5-1c08017a`) e provato a caldo, bitwise; split provato una corsa per
  braccio. B-A-B: provato no.
