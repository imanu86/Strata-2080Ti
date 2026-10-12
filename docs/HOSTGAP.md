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

## 4. Leva implementata: `STRATA_HG_PRELAUNCH=1`

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

## 5. Esito

In corso.
