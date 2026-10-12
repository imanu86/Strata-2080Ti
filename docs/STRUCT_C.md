# lab/struct-c — meno attesa della GPU sul lavoro dell'host (leva C)

Base: `lab/speed4` (`63a94046`: v0.1.43 + pila del fork + leve speed, tutte opt-in). Tutto il ramo è opt-in: senza
variabili `STRATA_SC_*` il motore lancia gli stessi grafi nello stesso ordine dallo stesso thread.

Convenzione: **M** = misurato (log del motore, corse brevi di questo ramo), **S** = stimato.

## 0. Conclusione

- **La «GPU-reach wait» non è la GPU che aspetta l'host**: è l'host che aspetta la GPU (§1). I ~14 ms su 26 della
  diagnosi sono calcolo della GPU, non un margine dell'host.
- **Quanto aspetta davvero la GPU**, misurato sulla GPU con `STRATA_SC_WAITSTAT=1` (§2; corse da 1024 token, cache
  fissa a 1500, 83% di hit, 3,1 esperti CPU per layer-finestra): la **2080 Ti aspetta il piano 2,5 ms a finestra** (7,4
  attese > 0,1 ms) e le righe CPU 1,0; la **3060** il piano 0,2 e le righe CPU 2,9 (M). Il lancio dei grafi fuori da
  nsys dura 0,09-0,13 ms (stadio 1) e 0,01 ms (stadio 0): non è la causa (M).
- **La causa dell'attesa sulla 2080 Ti è il pool CPU unico**: lo stadio 1 suona mentre il pool calcola un layer della B
  speculativa. Prova: trattenere i layer di B nei buchi dello stadio 1 (`STRATA_SC_BGAP_US=150`) porta l'attesa del
  piano della 2080 Ti da 3,09 a 1,82 ms e il suo muro da 17,5 a 16,1 ms; ma la 3060 (B) passa da 0,2 a 0,95 ms di
  attesa e le finestre speculative calano (M, 1024 token). Il guadagno sulla 2080 Ti si paga sulla B.
- **Thread dedicato ai doorbell** (`STRATA_SC_SERVE=1`, §3): bitwise, ma **neutro o peggiore**: serve la B con più
  prontezza, quindi il pool è più spesso occupato quando suona lo stadio 1 (attesa del piano 2,54 → 3,09 ms) e il tempo
  CPU del pool sale; ms/finestra +0,0/+1,7 (M). Con BGAP: 30,22 contro 29,15 della base (M, 1024 token). **Non
  proporlo al B-A-B.**
- **Snapshot GDN in linea** (`STRATA_SC_SNAP_INLINE=1`, §4): bitwise, una copia da 39 MB per B lanciata invece di due
  per finestra su uno stream laterale accanto allo stadio 0. Corse brevi (256 token, singole, rumore ±1,5 ms):
  31,77 ms/finestra di media su 3 contro 32,01 su 2 (adattivo acceso), 33,55 contro 34,45 (adattivo spento); la prima
  coppia pulita diceva −1,4 ms (M). **Stima −0,3/−1,0 ms a finestra (−1/−4%)**, da confermare col B-A-B: è la sola
  leva del ramo da provare.
- Scartate coi numeri: prefetch dalle bozze (un esperto non si calcola prima che la GPU arrivi al layer: il suo
  ingresso è l'attivazione del layer), `STRATA_ADAPT_OVERLAP` (solo ciclo seriale a una scheda), copie di promozione
  pinned (fuori dal percorso critico, ≤ 0,2 ms S), meno esperti CPU (cambia i bit).
- **Leva strutturale che resta** (non fatta qui): un pool CPU separato per la B speculativa (2-3 thread sui gemelli
  HT), così lo stadio 1 non aspetta mai la B: tetto ≈ l'attesa del piano della 2080 Ti, **−1/−2,5 ms nelle finestre
  con B accanto** (S), bitwise se il riparto per righe resta per esperto (`STRATA_IQ_MT_MIN=1`). Richiede un secondo
  `ExpertDispatch` (piano, job, contatori d'uso da sommare per il tier adattivo): 2-3 giorni.

## 1. Dove la GPU aspetta davvero l'host

**La «GPU-reach wait» del `decode timing` non è un'attesa della GPU.** In `Verifier::service` (`verify.cpp`) vale
`ms_wait += now_ms() - fl_since_ms_` quando l'host vede suonare il doorbell: è il tempo in cui **l'host aspetta la
GPU** (calcolo dei layer), dal servizio precedente allo squillo. I ~14 ms su 26 della diagnosi sono quindi lavoro
della GPU, non un margine dell'host.

L'attesa vera della GPU sull'host è dentro il grafo, in tre punti per layer (`post()` di `record_window`):

| punto | cosa aspetta la GPU | chi lo alza |
|---|---|---|
| waitA (`wait_flag_ge(m_flagA_)`, con NODECUT fuso nella copia del piano) | il piano del layer: quali esperti in VRAM | l'host, appena vede il doorbell: `expert_pool_dispatch_multi` → `publish_plan` |
| waitB (`m_flagB_`) | la quota PCIe in staging | alzato insieme ad A quando nulla va via PCIe (il nostro caso) |
| righe CPU (`copy_rows_from_mapped_after`, fuso) | gli esperti calcolati dal pool CPU | l'host a fine pool (`h_flag_`) |

waitA = latenza dell'host nel vedere lo squillo + il piano (~0,13 ms a finestra in tutto, M); le righe CPU = waitA +
calcolo CPU − quanto la GPU impiega per i suoi esperti in VRAM. L'host che serve è **un thread solo** (il ciclo della
pipeline) e **un pool solo** (7 worker + il thread chiamante) per entrambi gli stadi e per la B speculativa.

**Le opzioni del mandato, coi numeri:**

| opzione | esito | perché |
|---|---|---|
| prefetch guidato dalle bozze MTP | **impossibile come calcolo anticipato** | l'ingresso di un esperto è l'attivazione del layer, nota solo quando la GPU arriva al doorbell; neppure gli id sono noti prima del router. Resta «promuovere in VRAM gli esperti previsti»: cambia i bit (GPU invece di CPU) ed è ciò che il tier adattivo fa già |
| `STRATA_ADAPT_OVERLAP` | **non applicabile** | solo il ciclo seriale a una scheda con arena in RAM (`generate.cpp`, «the single-GPU, plain-RAM-arena decode loop only»); in pipeline il tier ha già il suo thread e i suoi fence |
| copie di promozione pinned | scartata, ≤ 0,2 ms (S) | stream propri, fuori dal percorso critico; il costo è sul thread del tier, non sul ciclo |
| thread host per stadio | fatto come **un thread dedicato a tutti i doorbell** (§3) | due thread col pool sarebbero due pool: il pool non è rientrante (piano, sink, contatori nel Drive) e dividere 8 core raddoppia il tempo CPU di ogni stadio |
| job CPU prima | coperto da §3 | il doorbell suona subito dopo il router: prima non c'è nulla da calcolare; si accorcia solo la latenza dell'host |
| meno esperti CPU | scartata | sposta esperti fra CPU e GPU: cambia i bit |
| snapshot GDN più leggeri | fatto (§4) | fp16 cambierebbe i bit; si tagliano le copie inutili e la loro sovrapposizione allo stadio 0 |

**Lancio dei grafi.** Fuori da nsys (`STRATA_SC_WAITSTAT`, §2) `cudaGraphLaunch` dura **0,10 ms** per lo stadio 1 e
**0,01 ms** per lo stadio 0 (M). I «0,5 ms × 11,8 lanci a finestra» della traccia erano gonfiati dal profiler (la
stessa lezione di HOSTGAP §5b): il ciclo occupato nei lanci non è la causa delle attese lunghe.

## 2. Misura: `STRATA_SC_WAITSTAT=1`

Le briciole #649 (`trace_m_`, timestamp `globaltimer` scritti in memoria mappata dai nodi `gpu_stamp`) si allocano
anche senza `STRATA_VERIFY_TRACE`; con la sola WAITSTAT si stampano 6 punti a layer (0, 18, 19, 20, 23, 24) invece di
~25. A ogni `pl_finish` (`sc_wait_note`) si sommano per stadio: waitA = 18 → 19, esperti VRAM = 19 → 20, righe CPU +
combine = 23 → 24, il muro del grafo (primo → ultimo timestamp), le attese > 0,1 ms, e il tempo host di
`cudaGraphLaunch`. Riga a fine richiesta, con `STRATA_DECODE_TIMING=1`:

`strata struct-c waits, stage s: N windows, GPU wall X ms/window; per window: waitA a (n > 0.1 ms: m ms), VRAM
experts v, waitB b, CPU rows+combine c (n > 0.1 ms: m ms); L layer-groups/window; graph launch on the host x ms × k`

Solo misura: sono nodi in più nel grafo (~200 a finestra sullo stadio 1). Non usarla nei bracci di velocità.

## 3. Thread dedicato ai doorbell: `STRATA_SC_SERVE=1`

**Cosa fa.** Nella pipeline a due stadi (`--pipeline-windows 2`, due schede) i doorbell di tutte le finestre in volo
li serve un thread a parte (`sc_thr`, `generate.cpp`): piano, esperti CPU, flag. Il ciclo lancia, chiude
(`done`/`pl_finish`), legge la catena e decide come prima, ma non chiama più `service`. Passaggio di mano per
verificatore (`[stadio][parità]`): 1 = consegnato al server al lancio (`sc_arm`), 2 = tutti i layer serviti (il ciclo
chiude la finestra), −1 = errore. Ordine del server: prima il percorso verificato (A sullo stadio 0, A sullo stadio
1), poi la B un layer per passata, come `pl_yield`; la finestra condannata (`D`) riceve piani vuoti come prima.

**Concorrenza.** Il pool, il `Drive` e i piani restano su un thread solo (il server). Ciò che il ciclo cambia e il pool
legge (la tabella di residenza `host_res`, la sorgente degli esperti, i passi dei tier adattivo ed elastico) lo cambia
sotto `sc_mu` (`sc_hold`: `pl_adapt`, le pubblicazioni di `el_gap`, `a_gap`, `adapt_tick`); il server prende `sc_mu`
per ogni chiamata e cede quando il ciclo lo chiede (`sc_wants`). La tabella PLE (slot di prefetch) è toccata da
`pl_stage` (ciclo) e da `gather_batch` (server): un mutex suo (`sc_ple_lock`, `verify.cpp`), spento di default.
In caso di errore (`die`) e prima del drain il server si ferma e si unisce (`sc_halt`).

**Bit per bit.** Stessi piani, stesso lavoro CPU e GPU per layer: cambia solo il momento del servizio. Il tier
adattivo applica gli scambi al verdetto come prima.

**Opzioni.** `STRATA_SC_SERVE_CPU` / `STRATA_SC_LOOP_CPU`: processori logici del server e del ciclo (default: il server
eredita il processore del ciclo se il ciclo è pinnato, il ciclo va sul gemello HT; nel nostro motore il thread host
non è pinnato, quindi di default nessuno dei due lo è). `STRATA_SC_BGAP_US=g` (0 = spento): mentre lo stadio 1 di A
è in volo, il layer suonato di B aspetta un buco dello stadio 1 — nessuno squillo dello stadio 1 in attesa e un layer
dello stadio 1 servito da meno di g µs — o al massimo `STRATA_SC_BMAX_US` (1000).

Riga a fine richiesta: `strata struct-c serve: N windows served on the server thread, … service calls, … passes held
for the loop; BGAP: … passes held B, … B layers past STRATA_SC_BMAX_US`.

## 4. Snapshot GDN in linea: `STRATA_SC_SNAP_INLINE=1`

**Cosa fa.** Con `--pipeline-windows 2` lo stato GDN dello stadio 0 (12 layer, 39,2 MB) si fotografa per poter
annullare una B sbagliata. Di default (`pl_snap_overlap`) la copia si fa **a ogni lancio** di finestra sullo stadio 0
(A e B: due copie D2D a finestra), su uno stream laterale accanto allo stadio 0, con eventi fra gli stream. La foto
serve solo se dietro la finestra parte una B (il commit speculativo); e la finestra non scrive lo stato (lo scrivono
solo i commit), quindi la stessa foto si può prendere **in linea subito prima del commit speculativo**, una sola per B
lanciata. È il modo `STRATA_PIPELINE_SNAP_OVERLAP=0` che il motore ha già (un buffer, `cudaMemcpyAsync` sullo stream
dello stadio prima del commit, ripristino dallo stesso buffer), oggi raggiungibile solo con `STRATA_PIPELINE_DEBUG=1`:
`STRATA_SC_SNAP_INLINE=1` lo accende senza il cancello di debug. Riga d'avvio: `--pipeline-windows 2: … (decode too;
the GDN snapshot on the critical path)`.

**Bit per bit.** Gli stessi byte copiati (lo stato prima del commit speculativo) e ripristinati. La copia in linea
precede la B, che ha margine (lo stadio 0 di B gira accanto allo stadio 1 di A). Libera anche 39 MB sulla 3060 (un
buffer invece di due).

## 5. Misure

Corse brevi L02 a 131k, cache fissa (`--expert-cache 1500`, senza `--elastic`), configurazione di
`det_check_fixed.py` + `STRATA_NODECUT=1 STRATA_QFUSE_HC=0 STRATA_TOPK_GRID=1 STRATA_B6_HEAD_MINB=6` (braccio B di
speed4); prefisso `r` = `--ple-io ram` (toglie il rumore del disco: nelle prime corse senza, la gather PLE è salita a
1,6 ms a finestra e il prefill a 228-304 s); prefisso `a` = anche `--adapt-every 0`. Cartelle `DET-x17-sc*` e
`DET-x19-sc*` in `corse_2080ti\20261004_routed_keep`. Una corsa per braccio: valgono il testo e le attese, non le
differenze di velocità sotto ±1,5 ms (lo stesso braccio ha dato 33,06 e 35,84 ms/finestra a distanza di un'ora).

### 5a. Testo

- **Col tier adattivo acceso il testo non è deterministico fra corse identiche** anche a cache fissa: a 256 token la
  base ha dato `b6119545` (roff, roff2, off), `4455b5ea` (offws) e `75a97dd1` (roff3). Gli scambi del tier si
  applicano al verdetto solo se le copie sono atterrate (`cudaEventQuery`): dipende dai tempi. Per la verifica bitwise
  serve `--adapt-every 0`.
- **Con `--adapt-every 0` tutti i bracci danno lo stesso testo** (`dbef6b7e`, 256 token): aoff, aoff2, asrvp (SERVE),
  ainl (SNAP_INLINE), asrvpb (SERVE + BGAP 150), asrvpinl (SERVE + SNAP_INLINE). **Bitwise.**
- Con l'adattivo acceso: SERVE e SNAP_INLINE hanno dato lo stesso testo della base nelle stesse condizioni (`b6119545` a
  256 token: srv, snap0, both, rsrv, rsnap0, rboth, rinl, rinl2; `b216fd52` a 1024: roff = rsrv).
- `STRATA_SC_WAITSTAT` è solo misura: nodi in più nel grafo; i suoi testi variano come quelli della base.

### 5b. Attese della GPU (`STRATA_SC_WAITSTAT=1`, ms a finestra)

| corsa (token) | stadio | muro | waitA (n > 0,1 ms) | esperti VRAM | righe CPU + combine | lancio host |
|---|---|---|---|---|---|---|
| offws (256) | 0 · 3060 | 13,23 | 0,49 (0,62) | 1,80 | 2,67 | 0,011 |
| | 1 · 2080 Ti | 17,38 | **2,66** (7,32) | 3,36 | 1,29 | 0,098 |
| srvws (256), SERVE | 0 | 13,43 | 0,22 (0,30) | 1,74 | 3,24 | 0,010 |
| | 1 | 17,53 | 2,56 (6,78) | 3,44 | 1,40 | 0,095 |
| roffws (1024) | 0 | 12,32 | 0,23 (0,21) | 1,54 | 2,87 | 0,009 |
| | 1 | 16,58 | **2,54** (7,40) | 3,21 | 1,01 | 0,093 |
| rsrvpws (1024), SERVE pinnato 14/15 | 0 | 13,35 | 0,21 (0,24) | 1,66 | 3,38 | 0,011 |
| | 1 | 17,49 | 3,09 (7,84) | 3,33 | 1,12 | 0,125 |
| rsrvpbws (1024), SERVE + BGAP 150 | 0 | 14,03 | 0,95 (2,95) | 1,55 | 3,50 | 0,011 |
| | 1 | 16,13 | **1,82** (4,09) | 3,29 | 1,15 | 0,125 |

waitB = 0 ovunque (nessuna quota PCIe). Lettura: (1) il ciclo occupato costa solo sulla 3060 (waitA 0,49 → 0,22
col server); (2) sulla 2080 Ti l'attesa del piano è il pool occupato dalla B: trattenere la B la toglie per metà;
(3) le righe CPU della 3060 (2,7-3,5 ms) sono calcolo CPU puro sul percorso critico delle finestre fresche.

Limite: queste corse sono a 83% di hit (3,1-3,5 esperti CPU per layer-finestra); il banco del Daily sta a 96-97,5%
(0,5-0,9). Con meno esperti CPU il pool è meno occupato e l'attesa dello stadio 1 sulla B si accorcia: la quota
recuperabile al banco è probabilmente minore (S).

### 5c. ms a finestra (`decode timing`, una corsa per braccio)

| braccio | 256 token, adattivo acceso | 256, `--adapt-every 0` | 1024 token |
|---|---|---|---|
| base | 32,20 · 31,82 (· 33,68 testo diverso) | 35,84 · 33,06 | 28,60 · (ws 29,15) |
| SNAP_INLINE | 30,78 · 33,19 · 31,35 (senza ram 31,08) | 33,55 | — |
| SERVE | 32,32 · 32,04 · 31,64 (pinnato) | 34,72 (pinnato) | 29,99 · (ws, pinnato, 30,90) |
| SERVE + SNAP_INLINE | 32,62 | 34,66 | — |
| SERVE + BGAP 150 | — | 36,18 (prefill lento) | (ws 30,22) |

Classi della prima coppia pulita (roff → rsnap0): fresche 36,30 → 34,48, speculative 21,00 → 20,66; `rollback cost`
fresh+Bbeside 43,77 → 38,65, fresh+afterRB+Bbeside 34,87 → 33,44.

## Stato

- Codice: `94d57bea` (WAITSTAT), `5994f240` (SERVE), `badc4bc2` (BGAP), `4f3cdee8` (SNAP_INLINE). File toccati:
  `src/program/generate.cpp` (ciclo della pipeline: server, `sc_hold`, riga `struct-c waits`, SNAP_INLINE),
  `src/core/verify.cpp` e `include/strata/core/verify.hpp` (WAITSTAT, `sc_ple_lock`, `sc_rang`/`sc_layers_served`).
- Build: `build-structc-4f3cdee8` (sha256 `fb4bf2b1bd96603255027c3ab76ab53d06db95ff8bfc214ec053a70805f95fa8`) in
  `corse_2080ti\20261004_routed_keep\` con `build.json`; prima `build-structc-5994f240` (`1a9cacc1…`).
- **Per il B-A-B** (lo fa il coordinatore): `build-structc-4f3cdee8`, A = configurazione corrente, **B = A +
  `STRATA_SC_SNAP_INLINE=1`**. Nel log di B: `--pipeline-windows 2: … (decode too; the GDN snapshot on the critical
  path)`. Da leggere `strata pipeline classes` (fresche) e `rollback cost`. SERVE/BGAP: non proporli.
- **Compilato sì, provato sì (corse brevi a cache fissa, bitwise con `--adapt-every 0`), B-A-B no, installato no.**
  SERVE con `--elastic` e con `--adapt-async`: compilato, mai eseguito (tutte le corse senza `--elastic`).
