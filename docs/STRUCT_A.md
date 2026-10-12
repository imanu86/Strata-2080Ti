# STRUCT-A: la 3060 al lavoro mentre gira la catena MTP

Branch `lab/struct-a` del fork `imanu86/Strata-2080Ti`, da `lab/speed4` (`63a94046`: v0.1.43 + la pila + le leve
speed). Configurazione: 3060 = stadio 0 (CUDA0 con `CUDA_VISIBLE_DEVICES=1,0`, layer 0-15), 2080 Ti = stadio 1 (layer
16-47, testa, catena MTP), `--pipeline-windows 2 --spec 4 --spec-min-p 0.5`. Tutto opt-in (`STRATA_SA_*`); senza
variabili il motore lancia gli stessi grafi nello stesso ordine di `lab/speed4`.

Convenzione: **M** = misurato (traccia della pipeline o log del motore), **S** = stimato.

## 1. Quanto pesa la catena sul percorso critico

Fonti: `STRATA_PIPELINE_TRACE` della corsa pulita `DET-cp-clean` (1834 verdetti, banco forzato, L01+L02 a 131k; build
`persist-954f24c`, stessa struttura della pipeline di oggi), script `structa_trace.py` (scratchpad della sessione,
ricostruibile: per ogni verdetto `V` legge `CL`/`CE`/`L0`/`F0`/`L1`/`F1` della finestra dopo); classi e catena dai log di
`DET-x16-split-a` (build `speed4-311b6fda`, la base di oggi). Tempi puliti, non nsys: la traccia nsys gonfia di 2-3 volte
l'host (HOSTGAP), per questo i «~4,8 ms fra sampler e stadio 0» della traccia nsys qui diventano ~2,3.

**Finestre fresche (73%).** Dal verdetto al lancio dello stadio 0 della finestra dopo (M, mediane):

| tratto | ms | cosa |
|---|---|---|
| `V`→`CL` | 0,36 | verdetto, commit, lancio della catena (host) |
| `CL`→`CE`, T=1 (42% delle fresche) | 0,85 | round + 1 uscita: la prima bozza sotto `spec-min-p` decide T=1 |
| `CL`→`CE`, T=2 (28%) | 1,79 | 2 uscite |
| `CL`→`CE`, T≥3 (30%) | 2,72 | 3 uscite (un passo = 0,93 ms) |
| `CE`→`L0` | 0,07 | lancio dello stadio 0 |
| **`V`→`L0`, media** | **2,20** | la 3060 è ferma |

Poi stadio 0 (T=1 7,93 · T=2 10,0 · T=3 12,2 · T=4 14,4 ms), `F0`→`L1` 0,17, stadio 1 (T=1 13,1 · T=2 15,5 · T=3 18,2 ·
T=4 21,7), `F1`→`V` 0,12 (M). Sulla media delle finestre: 2,2 × 0,73 = **1,6 ms a finestra** di catena + host prima
dello stadio 0 (S da M).

**Finestre speculative (27%).** La catena «forzata» (tipo 1) parte al verdetto sulla stessa 2080 Ti dello stadio 1. Lo
stadio 1 di una speculativa dura più di quello di una fresca allo stesso T (M): **+2,5 / +3,1 / +3,8 / +4,8 ms** a T = 1
/ 2 / 3 / 4, cioè ~70% della lunghezza della catena (round + `B.T + 2` passi da 0,93 ms): la catena ruba la scheda allo
stadio 1. Sulla media: ~3,1 × 0,27 = **0,85 ms a finestra** (S da M).

**Totale catena sul percorso critico: ~2,4 ms a finestra (~9%).** È il tetto di tutto quello che si fa sulla catena; la
parte che la 3060 può prendere è più piccola (§2).

## 2. Le tre strade proposte, e perché due non vanno

**a. Stadio 0 a pezzi sulle bozze man mano che escono (finestra più corta o allungabile).** No. Lo stadio 0 legge i pesi
una volta per finestra qualunque sia T (layer della 3060: ~0,50 ms a T=1, +0,13 ms a riga, M): passare le righe in due
volte rilegge i pesi (T=1 + T=3 = 20 ms contro 14,4 per T=4). Accorciare A e passare le righe in più a B costa invece un
**passaggio in più sullo stadio 1** (~13 ms fissi): da un caso reale (T=3 naturale) 33,8 ms per tre token diventano
27,1 + ~18,5. Le righe non si spezzano: la finestra deve partire con tutte le sue righe.

**b. MTP sulla 3060 (prova del 7/10, `docs/sm75/mtp-device-20261007.json`, −9/−12%).** Il json lo dice in una riga:
«stage1/speculative windows get faster while stage0/fresh windows get slower; the latter dominate». Il perché, coi
numeri del §1:
- nelle **fresche** (73%) la catena è in serie con lo stadio 0 in ogni caso; sulla 3060 un passo costa ~1,5-1,7× (360
  contro 616 GB/s) e i residui devono passare dalla 2080 Ti per la memoria pinned: il tratto `V`→`L0` si allunga;
- sempre nelle fresche, il resto della catena (le bozze di B, ~4 ms) oggi gira sulla 2080 Ti **ferma** (aspetta lo
  stadio 0): sulla 3060 gira **dentro** lo stadio 0 di A e lo allunga. Tempo gratis trasformato in tempo critico;
- la 3060 ospita i pesi e il K/V del drafter (gli esperti del layer MTP sono ~0,9 GB): ~500-700 slot di cache in meno
  allo stadio 0, più esperti sulla CPU in ogni finestra;
- nelle speculative guadagna i ~3 ms del §1, ma solo sul 27% delle finestre, e la B che ne esce è in ritardo.
Un drafter doppio (catena forzata sulla 3060, catena fresca sulla 2080 Ti) terrebbe solo il guadagno, ma serve un secondo
K/V del drafter allineato a ogni catena e ~1 GB di 3060: non lo faccio.

**c. Lanciare lo stadio 0 su un'ipotesi prima che la catena finisca, scartando il lavoro se è sbagliata.** Sì, ma solo
per la **testa** della finestra. Una finestra intera sbagliata non si annulla (un grafo lanciato va finito: 5-8 ms di
3060) e scrive K/V e indicizzatore QSA. I primi layer dello stadio 0 (0-2) sono invece GDN: una finestra della pipeline
non tocca lo stato GDN (lo fa il commit: `set_always_publish`), non scrive K/V; l'unico stato che muove è la storia PLE
del layer 1 (`ss.ple.hist`, avanzata in place e poi riscritta dal commit), che si salva e si ripristina con una copia di
pochi KB. Quindi la testa (layer 0..c-1, c ≤ 2 di default) si può lanciare **in scommessa** e buttare a costo ~zero.

## 3. Il disegno

### 3a. `STRATA_SA_BET=1`: la scommessa sulla testa dello stadio 0 (la 3060 al lavoro durante la catena)

- Lo stadio 0 ha, per T = 1..`STRATA_SA_BET_MAXT` (2), due grafi in più catturati da `record_window` su un intervallo di
  layer: **testa** = ingressi + layer `[0, c)`, **coda** = layer `[c, 16)` + hand-off. Stessi kernel, stesso stream,
  stesso ordine dei doorbell: testa + coda = la finestra di sempre (`STRATA_SA_BET_LAYERS` = c, default 2, limitato ai
  layer GDN prima del primo QSA dello stadio).
- **Scommessa 1**: al verdetto che apre una finestra fresca (la catena tipo 2 appena lanciata, eventuale undo di D fatto),
  la 3060 parte subito con la testa di A' = `[x]` (T=1: x è la correzione del target, nota al verdetto). L'host serve i c
  layer della testa come una finestra normale (esperti CPU compresi) e si ferma lì.
- **Scommessa 2**: quando la prima uscita della catena ha probabilità ≥ `spec-min-p` (quindi T ≥ 2) la testa 1 è persa:
  finita, si butta; parte la testa di `[x, d1]` (T=2).
- **Verifica** nel punto dove oggi parte lo stadio 0 di A (A pronto, dopo `force_window` del banco): se la scommessa ha
  lo stesso T, la stessa posizione e gli stessi token di A, si lancia solo la **coda** (la finestra continua dal layer
  c); altrimenti si aspetta la fine della testa (servita a piani vuoti), si ripristina la storia PLE, si stornano dal
  contatore d'uso del tier gli id contati dalla testa, e A parte normale.
- Bit per bit: una finestra confermata fa gli stessi kernel sugli stessi dati nello stesso ordine; una testa buttata non
  lascia stato (GDN intatto, nessun K/V, storia PLE ripristinata, `drive.d.usage` stornato esattamente: incrementi di
  1.0 in float). Il testo greedy deve restare identico.

Stima (S, dalla traccia del §1; testa di 2 layer, scommessa lanciata a `V`+0,45):

| T di A' (quota delle fresche) | oggi: fine stadio 0 | con la scommessa | guadagno |
|---|---|---|---|
| 1 (42%) | V+1,29+7,93 = V+9,2 | testa da V+0,45, coda alla decisione (V+1,28) dietro la testa: ~V+8,4 | **−0,8 ms** |
| 2 (28%) | V+2,21+10,0 = V+12,2 | testa 1 buttata a V+1,45, testa 2 fino a V+2,75, coda: ~V+11,5 | **−0,7 ms** |
| ≥3 (30%) | — | la testa 2 è finita prima della decisione (V+3,1): nessun ritardo | 0 |

**−0,53 ms per finestra fresca, −0,39 ms a finestra in media (~1,5%)**. La 3060 lavora in ~1,1 ms dei 2,2 di attesa.

### 3b. `STRATA_SA_CHAIN=1`: la catena forzata a passi, senza passi buttati (meno catena dentro lo stadio 1)

`chain_launch` oggi accoda **tutti** i passi (fino a 7) al lancio, anche quelli che servono solo se le bozze restano
sopra `spec-min-p`. Nella catena forzata (tipo 1, finestre speculative) le uscite dopo la decisione di B non le legge
nessuno, ma girano sulla 2080 Ti dentro lo stadio 1. Con la leva la catena tipo 1 lancia il round e un passo alla volta:
il ciclo aggiunge il passo successivo quando vede l'uscita prima, e smette appena B è deciso. La catena tipo 2 (fresca,
latenza critica, gira sulla 2080 Ti ferma) resta accodata tutta come oggi.

Stima (S): uscite buttate per catena forzata = 3 − B.T per B.T ≤ 2 → ~1,1 passi (distribuzione dei T della traccia) =
~1,0 ms di catena, ~0,7 ms di stadio 1 per finestra speculativa → **−0,2 ms a finestra (~0,8%)**. Costo: la catena forzata
arriva a B ~0,1-0,3 ms più tardi (latenza dell'host fra un passo e l'altro), dentro il margine di B (stadio 0 di B +
catena ~14 ms contro ~16-19 di stadio 1). Bit per bit: B è deciso sulle stesse uscite; le uscite tolte scrivevano solo
celle K/V del drafter oltre la bozza, che il round successivo riscrive prima di leggerle.

### 3c. Somma attesa

−0,6 ms a finestra (~2,3%) se le due si sommano: L02 ~68 → ~69,5, L01 ~75 → ~76,5 t/s (S). Non porta a 80 da sola: la
catena vale ~2,4 ms a finestra in tutto, e la parte che la 3060 può fare al suo posto è la testa della finestra.

## 4. Misura

- Correttezza: L02 corto (64 token) a cache fissa (`--expert-cache 1500`, senza `--elastic`), `L02.txt` identico fra
  spento, `STRATA_SA_BET=1`, `STRATA_SA_CHAIN=1` ed entrambe.
- Per finestra: `strata pipeline classes` (fresh deve scendere di ~0,5 ms con BET; spec di ~0,7 con CHAIN), `the chain
  from its launch to B: forced` (sale di poco con CHAIN), e la riga nuova `strata struct-a:` (scommesse, confermate per T,
  buttate, passi di catena risparmiati).
- Traccia della pipeline: eventi nuovi `SH` (testa lanciata, x0 = T), `ST` (coda lanciata), `SX` (testa buttata).

## 5. Cosa ha detto la prima misura: la 3060 dopo il verdetto non è ferma

Build `build-structa-7c1a7aca` (scommesse da T=1), L02 256 token a cache fissa, una corsa per braccio, PC carico
(prefill 78-436 s: i tempi assoluti valgono poco, le differenze dentro la stessa classe sì). Script `structa_undo.py`
(scratchpad): finestre fresche divise per «il verdetto prima ha fatto l'undo di una B sbagliata» (evento `U`).

| finestre fresche T=1, `V`→fine stadio 0 (M, mediane) | `off` | `STRATA_SA_BET=1` (coda confermata) |
|---|---|---|
| senza undo (~10% delle fresche) | 12,16 ms (n 6) | **10,24 ms (n 5)** |
| con undo (~90%) | 10,76 ms (n 33) | 11,02 ms (n 39) |

- **Dove la 3060 è davvero ferma, la testa rende** (−1,9 ms, campione piccolo).
- **Dopo un rollback non lo è.** L'undo della B sbagliata (ripristino dello snapshot GDN da 39 MB + commit di A
  rigiocato, `U` a V+0,45) occupa lo stadio 0 fino a ~V+1,2 (dedotto: coda lanciata a V+1,17, fine stadio 0 a V+11,0 =
  uno stadio 0 intero dopo ~V+1,2). È proprio quando la catena decide T=1: la testa T=1 non ha spazio.
- **La testa T=1 persa costa**: occupa lo stadio 0 quando servirebbe la testa T=2 e ritarda le finestre T=2 (`V`→`L0`
  2,70 contro 2,17 ms, mediane), e su 12 scommesse T=2 solo 3 sono partite in tempo.

Correzione (`build-structa-<sha>` successiva): **niente scommesse T=1 di default** (`STRATA_SA_BET_MINT=2`; `=1` le
riaccende). La scommessa T=2 (`[x, d1]`) parte alla prima uscita della catena con `d1` ≥ `spec-min-p` (~V+1,2, quando
l'undo ha finito) e guadagna fino alla decisione di T=2 (~V+2,15): ~0,9 ms sulle fresche T=2 (S); persa (T ≥ 3) finisce
prima della decisione di T ≥ 3 (~V+3,1) e non ritarda nulla. Stima rivista: ~−0,27 ms per finestra fresca, **~−0,2 ms a
finestra (~0,8%)**, sotto la risoluzione del B-A-B (±2%).

## 6. Misure della build finale (`build-structa-075ce875`)

L02 256 token, cache fissa 1500 senza `--elastic`, **residenza statica** (`--adapt-every 0`), una corsa per braccio, PC
in uso (prefill 79-295 s). Serie `DET-x18-sa6-*` (+ `DET-x18-sa4-soff/soff2` della build prima, stesso testo).

**Testo.** `L02.txt` identico a `soff` in tutti i bracci: `sbet`, `sboth`, `schain`, `sprio`, `schainprio`, `sall`
(1199 caratteri; e `sa4`: `soff2`, `sbet` con le scommesse T=1, `sboth`). Con le teste confermate: 30 code T=2 nella
build finale, 36 T=1 + 20 T=2 nella precedente.

Perché a residenza statica: col tier adattivo acceso (il default, anche a cache fissa) la residenza segue i tempi della
corsa. A 256 token due `off` della stessa build differiscono fra loro dal carattere 352 (`DET-x18-sa3-off` contro
`-off2`, 644 contro 679 scambi del tier), e `bet`/`both` coincidono con `off2`. A 64 token il tier non arriva a
cambiare nulla: `DET-x18-sa2-*` identici a `off` (`5fefd82e`), compreso `both`.

**Per finestra (M, mediane).**

| | `off` (3 corse) | `STRATA_SA_BET=1` | note |
|---|---|---|---|
| fresche T=2 dopo un undo, `V`→fine stadio 0 | 15,39 · 14,34 · 15,95 | **13,54 · 13,97** (`sbet` · `sboth`, n 24) | −0,9/−1,9 ms, dove la stima (§5) diceva ~−0,9 |
| fresche T=3 / T=4 (testa T=2 persa) | 17,9-25,7 / 23,0-27,1 | 17,8-18,8 / 22,4-23,0 | nessun ritardo visibile |
| scommesse T=2 | — | 71-72 piazzate, 30 confermate | testa a V+1,26, coda a V+2,36 |

| classe speculativa, ms | bracci senza catena a passi | bracci con `STRATA_SA_CHAIN=1` |
|---|---|---|
| `DET-x18-sa6` + `sa4` | 22,36 · 22,25 · 22,35 (`off`) · 22,15 (`sbet`) · 22,29 (`sprio`) | 21,21 · 21,69 · 21,77 · 21,32 (sa6) · 21,79 (sa4) |

- **Catena a passi**: −0,6/−1,1 ms per finestra speculativa, coerente in cinque bracci; 46 uscite non eseguite su 31
  catene forzate. La catena arriva a B di poco più tardi (3,80 contro 3,70 ms).
- **Priorità della catena** (segnalazione di struct-b: `STRATA_MTP_PRIORITY=0`, stream del drafter a priorità di
  default invece che massima; esiste già): classe speculativa 22,29, nessun guadagno, e la catena forzata arriva a B
  più tardi (4,30 contro 3,70 ms). Con la catena a passi la contesa che resta è piccola; la priorità non va toccata.
- **Classe fresca e ms/finestra**: nel rumore. Due `off` identici differiscono di ~1 ms sulla classe fresca
  (37,37 / 36,40 / 38,65); i bracci con la scommessa stanno a 36,1-36,3.

Valore atteso sulla media (S, pesando le misure sulle quote di finestre): scommessa −1,3 ms × 21% di fresche T=2 ×
~78% di fresche = **~−0,2 ms a finestra**; catena a passi −0,7 ms × 22-27% di speculative = **~−0,15/−0,2 ms a
finestra**. Insieme **~−0,4 ms a
finestra (~1,5%)**: sotto la risoluzione di un B-A-B a tre bracci (±2%), da leggere con `strata pipeline classes` e
`strata struct-a:` per finestra più che dai t/s.

## 7. Per il B-A-B del coordinatore

- Build `build-structa-075ce875`, `strata.exe` sha256 `9ef62e54337515c8f1c7d40f3adcebaec3f3ce23d2094d0f9f6140d04c8c00b0`
  (codice `075ce875`; i doc dopo non toccano il codice).
- Braccio A: env attuale (speed4 braccio B compreso). Braccio B: A + `STRATA_SA_BET=1 STRATA_SA_CHAIN=1`. Non
  aggiungere `STRATA_MTP_PRIORITY=0`.
- Nel log di B: `strata struct-a: STRATA_SA_BET: heads of 2 layers on CUDA0, bets T=2..2` all'avvio e, per richiesta,
  `strata struct-a: head bets (placed/tail/dropped) …; forced chains fed a step at a time N, outputs not run M`.
- Righe da leggere: `strata pipeline classes` (speculative −0,6/−1 ms; fresche −0,2/−0,4), `the chain from its launch
  to B: forced` (pari o +0,1).
- Manopole: `STRATA_SA_BET_MINT=1` riaccende le scommesse T=1 (peggio, §5), `STRATA_SA_BET_MAXT=3` la T=3 (stimata a
  somma zero), `STRATA_SA_BET_LAYERS` (2), `STRATA_SA_CHAIN_AHEAD` (0: un passo alla volta).
- Effetti collaterali nei log: le teste perse passano dal pool (esperti CPU dei loro layer), quindi `GPU-reach wait` e
  `CPU experts` del `decode timing` contano un po' di lavoro in più; il tier non le conta (`ExpertDispatch::no_usage`).

## Stato

Disegno e stima: §1-4. Prima misura e correzione: §5. Misure finali: §6. Compilato sì (`build-structa-075ce875`),
provato sì (corse brevi a cache fissa, testo identico), B-A-B no, installato no. Codice: `3e309ed7` (leve), `7c1a7aca` (servizio della testa sul suo
evento: senza, «layer 1 never rang» se l'host passava prima che la testa suonasse; niente secondo snapshot GDN).
