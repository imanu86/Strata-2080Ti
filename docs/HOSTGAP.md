# HOSTGAP: i tempi morti della finestra (host fra le finestre, buchi nel grafo, attese dei flag)

Branch `lab/hostgap` del fork `imanu86/Strata-2080Ti`, da `lab/expertdec` (`4892f482`: candidato Daily + EXPDEC).
Configurazione del candidato (`docs/CANDIDATE.md`), `STRATA_EXPDEC=1` acceso (B-A-B +1,1%). Tutto opt-in
(`STRATA_HG_*`); senza variabili il motore lancia gli stessi grafi nello stesso ordine.

Dati: le catture nsys `DET-x2-expdec-nsys{A,B}` (build `e4885f1d`, banco forzato L02 96 finestre, `--cuda-graph-trace
=node`) e `DET-qsarows-nsys2` (`build-ports1-743b2d07`, 295 finestre), lette con gli script `hostgap_analysis.py`,
`launch_bubble.py`, `dev_idle.py`, `bubbles.py` (scratchpad della sessione, ricostruibili: raggruppano i kernel per
`correlationId` del `cudaGraphLaunch`, incrociano la tabella `CUPTI_ACTIVITY_KIND_RUNTIME` per i tempi host delle API,
fanno l'unione degli intervalli dei kernel per device). Sotto nsys l'host è 2-3 volte più lento (il `cudaGraphLaunch`
dei 1397 nodi 2,4-3,8 ms contro ~1-1,2 sul banco pulito): valgono le proporzioni e le cause, gli assoluti del banco
pulito vengono da `strata decode timing` / `strata pipeline classes` (26,06 ms/finestra; fresh 27,9 ms, speculative
18,8) e da QSAROWS §3 (muro del grafo dello stadio 1 ~19,8 ms a T = 2).

## 1. Dove va il tempo fuori dai kernel (2080 Ti = stadio 1, il percorso critico)

Unione dei kernel sulla 2080 Ti nelle 96 finestre di L02 (cattura B): occupata il 49%; per finestra (nsys) 13,0 ms
prima dell'inizio di un grafo di finestra, **3,2 ms dentro il grafo prima di `gr_up_multi`**, 2,9 ms prima dei
`copy16` della catena MTP (la catena che aspetta l'host fra un passo e l'altro), 1,8 ms di buchi sotto i 20 µs fra i
nodi, 0,2 ms prima di `sampler_greedy`. Le voci, con la causa nel codice:

### 1a. Il lancio del grafo dello stadio 1 ferma la GPU per tutta la sua durata host

Il grafo di una finestra dello stadio 1 ha 1397 nodi. `Verifier::pl_launch` (`src/core/verify.cpp`, ora
`pl_launch_impl`, il `cudaGraphLaunch(exec_[T], cs_)`) dura sull'host **2,4-3,8 ms sotto nsys, ~1-1,2 ms sul banco
pulito** (FABLE_PLAN: «~1,2 ms host del lancio dei 1397 nodi»). Nel frattempo la GPU esegue i primi 8 nodi (le copie
dei token/posizioni dal mapped, `gr_norm_split`, `gr_down_staged` del primo layer) e poi **aspetta che il lancio
finisca**: la bolla fra `gr_down_staged` e `gr_up_multi` (nodo 8) termina 0,03-0,05 ms dopo il ritorno dell'API in
tutte le finestre delle tre catture (`launch_bubble.py`: «bubble end − API end» mediana 0,026 / 0,051 ms). È il
driver WDDM che sottomette il primo pacchetto di comandi mentre costruisce il resto. QSAROWS l'aveva contata come
«buchi fra i 1397 nodi, 3,5 ms piatti»: i buchi veri fra i nodi sono ~1,3 ms (sotto), il resto è questa bolla.

### 1b. Da fine stadio 0 a inizio stadio 1

`pump0` (`src/program/generate.cpp` ~12072) vede la fine dello stadio 0 interrogando `Verifier::done` (evento) a ogni
passata del ciclo, poi `launch_s1` → `V1(A).pl_launch` (~12408). Stadio 0 finito → inizio dell'API di lancio:
**1,3-2,3 ms** (nsys); poi il lancio (1a). Sul banco pulito la traccia della pipeline dava F0→L1 ≈ 1,2 ms.

### 1c. Il giro dell'host fra due finestre fresche

Fine del grafo dello stadio 1 → inizio dell'API di lancio dello stadio 0 della finestra dopo: **3,4-3,8 ms** (nsys):
verdetto (`V`, ~12752), lettura/lancio della catena MTP e costruzione di B, `pl_stage` + `cudaGraphLaunch` dello
stadio 0 (712 nodi: 1,2-1,7 ms host sotto nsys, 0,35-0,7 pulito, con la stessa bolla sulla 3060). Qui vivono le leve
già provate e risultate neutre: `STRATA_CP_L1FIRST` (ordine delle chiamate), `STRATA_CP_CHAIN_PRE` (catena dal
device), `STRATA_FP_EARLY_L1` di `lab/fable-plan` (stadio 1 dietro l'**evento** dello stadio 0 con
`cudaStreamWaitEvent` fra device: sotto WDDM l'attesa di un evento di un'altra scheda non sottomette il grafo prima che
l'evento sia compiuto, quindi la bolla 1a e il polling 1b restavano; qui si usa invece una **parola mappata e uno spin
sul device**, come i doorbell).

### 1d. Finestre fresche: lo stadio 0 è seriale

Per il 67% delle finestre (fresh, 27,9 ms) la 2080 Ti è ferma mentre la 3060 fa lo stadio 0 (10-13 ms): è la struttura
della pipeline (B speculativa sul percorso nel 33% dei casi). Non è un tempo morto dell'host: non toccato qui.

## 2. I buchi fra i nodi del grafo

Nella finestra dello stadio 1: ~836 buchi da 0-2 µs (mediana 0,7 µs) e ~128 da 2-4 µs (2,5-2,8 µs dopo
`doorbell_publish`, `add_hits`, `gdn_conv_l2`), **~1,3-1,8 ms per finestra**: la latenza di dispatch fra nodi
dipendenti, proporzionale al numero di nodi. Si riduce solo con meno nodi (fusioni come in `CHAINFUSE` per la catena:
nel grafo della finestra i candidati sono i ~30 kernel da 2-4 µs per layer: `native_quantize_q8_1` ×3,
`swiglu_q8_1_entries` ×2, `combine`+`sigmoid_scale`, `rebase_ptrs`/`fetch_blobs`, i tre `wait_flag_ge`). Più 0,2 ms
prima di `sampler_greedy` (fine grafo). Non fatto qui: lavoro grande, ~1 ms di tetto, da fare con parità bit per bit
kernel per kernel.

## 3. Le attese `wait_flag_ge`

96 per finestra (3 per layer: piano A, righe PCIe/CPU B, flag della PLE). Il 66% dura < 5 µs (il flag è già alto),
p90 107 µs, p99 812 µs: la somma (1,4-1,7 ms a T = 1-2, 5-6 a T = 3-4 sotto nsys) viene da ~10 attese lunghe per
finestra. La **prima** (layer 16, mediana 42 µs, > 100 µs nel 44% delle finestre) è l'host ancora dentro il
`cudaGraphLaunch` quando la GPU suona il primo doorbell: la toglie il prelancio (§4). Le altre, sparse sui layer
(posizioni 17, 23, 38, 50, 59, 65, 83, 86, 92), sono il flag B dei layer con un esperto sulla CPU (`CPU experts
0,5-0,9 per layer-finestra` nel `decode timing`: la GPU aspetta il calcolo CPU, `verify.cpp` ~1729
`wait_flag_ge(m_flagB_)`) e la latenza del thread unico che serve i due stadi e la B (`pump0`). Spostare esperti
dalla CPU alla GPU **non è bit per bit** (la GPU arrotonda diversamente: il motore lo dichiara all'avvio): il lato CPU
non si tocca qui. `STRATA_CP_RESPLAN` (piano sul device per i layer tutti residenti) e `S1FIRST` erano neutri sul
banco (handoff 11/10).

## 4. Leva implementata: `STRATA_HG_PRELAUNCH=1` (lancio sul thread del ciclo) / `=2` (su un thread a parte, §4b)

**Cosa fa.** Alla partenza dello stadio 0 di A (`L0`) il ciclo lancia subito anche il grafo dello stadio 1 di A
(`Verifier::pl_prelaunch`), nella variante `exec_hg_[T]` catturata in `capture_all` (`hg_prepare`) con un
`wait_flag_ge` sulla parola mappata `m_hg_` **subito prima delle copie dell'hand-off** (`record_window`, `lb_ > 0`):
tutto ciò che precede (copie di token/step/pos dal mapped, che il ciclo ha già scritto con `pl_stage`) gira subito, le
copie dell'hand-off e il resto dei 1397 nodi aspettano. Quando `pump0` vede lo stadio 0 finito, invece di lanciare
chiama `pl_release` (scrive 1 nella parola, come un doorbell) e `L1` nella traccia. Così il `cudaGraphLaunch` (~1-1,2
ms) e il polling della fine dello stadio 0 si sovrappongono allo stadio 0; la 2080 Ti riparte dall'hand-off entro i µs
della scrittura. Solo le A (vanno sempre allo stadio 1); le B speculative no (un grafo prelanciato non si annulla: se B
fosse rifiutata girerebbe a vuoto 20 ms). Contatore a fine richiesta: `strata hostgap: STRATA_HG_PRELAUNCH: N stage-1
windows prelaunched at their stage-0 launch, M released at stage 0's end`; traccia `P1` (prelancio) e `L1` (rilascio).
Spento accanto a `STRATA_CP_RESPLAN` (un set di grafi alla volta).

**Bit per bit.** Stessi kernel nello stesso ordine sugli stessi dati: cambia solo l'istante in cui il grafo parte. Le
copie dell'hand-off leggono il mapped dopo che l'host ha visto lo stadio 0 finito (stessa condizione di prima); la
parola è portata a 0 dall'host prima del lancio e a 1 al rilascio, con `_mm_sfence`. Il testo greedy e
`target picks replaced` devono restare identici fra i bracci.

**Atteso.** Finestre fresche −(1b + 1a) ≈ **−1,5/−2,5 ms** sul banco pulito (meno la latenza del rilascio, ~0,1 ms
di polling), cioè ~**−1/−1,5 ms per finestra** media (−4/−6%); le speculative invariate; la prima attesa del flag
(§3) scompare. Nel `decode timing` la «GPU-reach wait» dello stadio 1 si conta dal rilascio (`fl_launch_ms_`).

**Rischi.** (1) Lo spin (un blocco, un thread) occupa uno slot dello stadio 1 per ~10 ms mentre la 3060 lavora: la
catena MTP (stream ad alta priorità) e il commit della finestra prima non ne risentono; (2) se lo stadio 0 non finisce
(errore) il grafo resta a girare: il ciclo muore comunque con l'errore (`die`) e il processo esce; (3) WDDM: due grafi
in coda sulla stessa scheda (il prelanciato e il commit/la B precedente) sono già il caso di oggi.

**Misura.** nsys A/B sullo stesso build (`DET-x3-hostgap-nsys{A,B}`, A = `STRATA_EXPDEC=1`, B = + `STRATA_HG_PRELAUNCH=1`):
stadio 0 finito → primo kernel utile dello stadio 1 (dopo lo spin), la bolla al nodo 8 sparita, il muro del grafo e
`decode timing`; poi il B-A-B dell'owner col banco forzato (`strata pipeline classes: fresh … ms` è la riga che deve
scendere).

### 4b. `STRATA_HG_PRELAUNCH=2`: il lancio su un altro thread

Prima misura del modo 1 (`DET-x3-hostgap-nsys{A,B}`, build `f1e6c639`, 96 finestre di L02 sotto nsys; A =
`STRATA_EXPDEC=1`, B = + `=1`): il meccanismo funziona (72 prelanci / 72 rilasci; da fine stadio 0 al primo kernel
utile dello stadio 1 **4,93 → 0,01 ms**; la bolla al nodo 8 sparita: idle dentro il grafo prima di `gr_up_multi`
2,15 → 0,10 ms per finestra) ma la finestra **non scende** (L02 36,37 → 36,82 ms/finestra, fresh 40,47 → 40,15,
speculative 26,84 → 26,76): il muro dello stadio 0 della stessa finestra sale **13,15 → 14,93 ms** e la durata utile
dello stadio 1 18,55 → 19,69. Causa: il `cudaGraphLaunch` dei 1397 nodi (2,8 ms sotto nsys, ~1,2 pulito) ora occupa
il thread del ciclo **subito dopo il lancio dello stadio 0**, proprio quando la 3060 suona i doorbell dei primi layer:
lo stadio 0 aspetta il piano dell'host per tutta la durata del lancio. Il tempo si sposta, non sparisce. Quindi il
modo 2: `pl_prelaunch_arm` sul thread del ciclo (staging, parola a 0), `pl_prelaunch_fire` (il `cudaGraphLaunch` e
l'evento) su un thread a parte (`std::async`); il ciclo marca lo stadio 1 in volo quando il future è pronto, e il
rilascio può precedere il lancio (il grafo trova la parola già a 1 e corre). `target picks replaced` varia già fra
corse identiche (3/3/4/3/5; 39/36 nel B-A-B di EXPDEC: dipende da quali esperti finiscono sulla CPU), non è un segno
della leva.

## 5. Esito (nsys, build `72fc813a`, `DET-x3-hostgap-nsys{A,B,B2}`: 96 finestre di L02 sotto nsys, stesso banco forzato)

| | A (`STRATA_EXPDEC=1`) | B (`+ HG_PRELAUNCH=1`) | B2 (`+ HG_PRELAUNCH=2`) |
|---|---|---|---|
| `decode timing` L02, ms/finestra | 36,37 | 36,82 | **35,19 (−3,2%)** |
| classe fresh / speculative, ms | 40,47 / 26,84 | 40,15 / 26,76 | **38,97 / 24,97** |
| fine stadio 0 → primo kernel utile dello stadio 1 | 4,93 | 0,01 | 0,01 |
| idle dentro il grafo prima di `gr_up_multi` (la bolla del lancio) | 2,15 | 0,10 | ~0 |
| muro del grafo dello stadio 0 (3060) | 12,84 | 13,20 | 12,96 |
| durata utile dello stadio 1 (dal rilascio all'ultimo kernel) | 18,55 | 19,69 | 18,94 |
| periodo (fine stadio 1 → fine stadio 1 successivo), mediana | 35,39 | 35,56 | **34,40** |
| prelanci / rilasci | - | 72 / 72 | 70 / 70 |
| `accepted … drafts`, token | 104/154, 200 | 104/154, 200 | 104/154, 200 |

Sotto nsys il lancio dei 1397 nodi costa 2,8 ms: il modo 2 ne toglie ~1,2 ms per finestra (−3%); sul banco pulito il
lancio è ~1-1,2 ms e il polling di F0 ~0,3-1, quindi l'attesa è **−0,5/−1 ms per finestra (−2/−4%)**, concentrata
sulle finestre fresche. Il modo 1 non va usato (sposta il costo sullo stadio 0). `target picks replaced` 3 / 5 / 4:
varia fra corse identiche (quali esperti finiscono sulla CPU), non è la leva; i draft accettati sono gli stessi.

### 5b. B-A-B dell'owner (`DET-x5-hostgap-{a,b1,b2}`, build `72fc813a`, EXPDEC acceso ovunque, B = `STRATA_HG_PRELAUNCH=2`): **neutro**

| braccio | L01 t/s | L02 t/s | L02 ms/finestra | fresh (n, ms) | speculative (n, ms) | per-layer host | stage | PLE gather + staging ms/finestra |
|---|---|---|---|---|---|---|---|---|
| A | 72,0 | 66,4 | 25,71 | 429 · 28,39 | 171 · 18,98 | 1,15 | 0,09 | 0,01 + 0,07 |
| B1 | 65,8 | 64,1 | 26,64 | 435 · 29,33 | 165 · 19,57 | **2,12** | **1,03** | **0,48 + 0,55** (L01: 1,33 + 1,24) |
| B2 | 71,9 | 66,5 | 25,66 | 432 · 28,25 | 168 · 18,98 | 1,19 | 0,08 | 0,01 + 0,07 |

Guadagno di B su A −6,4% / +0,1%, media −3,1%; 521/600 finestre identiche. Lettura:

1. **B1 è un outlier dell'host, non della leva**: nel suo log `strata decode PLE primary: gather 691 ms (1,327 ms/
   finestra), other staging 1,242 ms/finestra` in L01 e 0,478 / 0,550 in L02, contro 0,008-0,014 / 0,07-0,08 in A e B2;
   di qui `stage 1,03` e `per-layer host 2,12`. Sono le righe PLE lette dall'SSD / row cache a ogni finestra
   (`pl_stage` → `prefetch_rows`, `gather_batch`): I/O lento in quella corsa (anche L01 −8,6%, dove il prelancio è
   identico a B2). CPU di altri processi e disco (`ambiente.json`) uguali fra B1 e B2.
2. **B2 ≡ A**: fresh −0,14 ms (−0,5%), speculative identiche, L02 +0,1%. Il prelancio funziona (432/432 rilasci) ma
   sul banco pulito il tratto che toglie — polling di F0 + `cudaGraphLaunch` dei 1397 nodi — vale ~0,3-0,5 ms per
   finestra fresca, non i 4,9 ms visti sotto nsys: il profiler gonfia di 2-3 volte il costo host del lancio (2,8 ms
   contro ~1) e i buchi fra i nodi (è lui la «bolla» da 3,5 ms). Il −3,2% di `-nsysB2` era in gran parte un artefatto
   della misura.
3. **Lo spin di un blocco sulla 2080 Ti durante lo stadio 0 non costa nulla di misurabile**: con la leva accesa (B2)
   speculative, hand-off e `GPU-reach wait` (14,64 contro 15,10) sono pari o migliori di A; gli esperti su CPU
   (`CPU experts 0,52 per layer-finestra`, `CPU 2,00 ms`) identici.

**Chiusura: `STRATA_HG_PRELAUNCH` resta opt-in, default spento, e si chiude come neutro.** Nessuna correzione credibile:
il costo che la leva toglie è già sotto la risoluzione del B-A-B (±2%); per vederlo servirebbe un banco con lo stadio 1
spesso in attesa dell'host, che oggi non c'è. I tempi morti veri della finestra restano quelli strutturali del §1d
(lo stadio 0 seriale nelle finestre fresche: 67% delle finestre, ~10 ms di 2080 Ti ferma) e i buchi fra i nodi (§2).

**Lezione di metodo.** I buchi e le latenze host letti da una traccia nsys vanno divisi per 2-3 prima di stimare il
guadagno sul banco pulito: il profiler aggiunge ~1-2 µs a nodo di grafo e allunga le API di lancio; il `decode timing`
del motore (`GPU-reach wait`, `per-layer host`, `stage`, `PLE gather`) è la misura buona per il lato host.

**Per il B-A-B** (lo lancia l'owner): `build-hostgap-72fc813a\strata.exe`, config del candidato + `STRATA_EXPDEC=1` in
entrambi i bracci, B con `STRATA_HG_PRELAUNCH=2`; nel log di B `strata hostgap: STRATA_HG_PRELAUNCH: N stage-1 windows
prelaunched at their stage-0 launch, N released at stage 0's end`; da leggere `strata pipeline classes: fresh … ms`.

**Dopo.** (a) La parola dell'hand-off scritta dal **device** dello stadio 0 (ultimo nodo del suo grafo, come i
doorbell) invece che dall'host a F0: toglie anche il polling di F0 (~0,1-0,3 ms) e il giro del ciclo; (b) lo stesso
prelancio per lo stadio 0 della finestra successiva non si può fare (i token arrivano dal verdetto), ma il suo lancio
(712 nodi, 0,35-0,7 ms) può andare su un thread mentre il ciclo legge la catena; (c) i ~1,3 ms di buchi fra i nodi e
i 0,2 ms prima del sampler restano per le fusioni (§2).

## Stato

Build di `72fc813a` (codice; questa pagina è stata aggiornata dopo): compilata con `build_lock.py` (Release, CUDA
75;86, test spenti), rc 0. `strata.exe` sha256 `61aca041fa290eea04b5f8a31ca08870e65952de0e82b80f4b8f1dbf7bbf04c6`,
copiata con `build.json` in `moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\
build-hostgap-72fc813a\` (accanto `build-hostgap-f1e6c639`, il modo 1 solo), più `strata.pdb` dello stesso build per
leggere i dump dei crash d'avvio. **Compilato sì, nsys nel motore sì, B-A-B sì (`DET-x5-hostgap`, §5b: neutro),
installato no.** Nessuna nuova build.

Crash d'avvio del banco (indagine 12/10, sola lettura): ~1 avvio su 8 muore dopo `CUDA1 prompt path borrows …` dentro
l'init del prompt path, su ogni build dal 6/10 (Daily compreso); WER: `nvcuda64.dll` 32.0.15.9621, offset `0x46c025`,
`0xC0000409` dati 2 (stack cookie del driver), quattro volte dal 7/10, dump in `%LOCALAPPDATA%\CrashDumps`. Non è
EXPDEC né hostgap (nessuno dei due è ancora stato toccato a quel punto). L'harness (`run_long.py`, `run_nsys.py`,
commit `9707e10` in moe-aggressive-commit) ora registra l'exit code (`start_failures` nel result.json) e riprova
l'avvio una volta sullo stesso braccio («retry avvio»).
