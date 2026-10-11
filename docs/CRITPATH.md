# CRITPATH: leve sul percorso critico della chat singola a 131k (due schede)

Branch `lab/critpath2` del fork `imanu86/Strata-2080Ti`, da `lab/upsync` (`431e28a7`: upstream v0.1.42 più le patch del
Daily, vedi `docs/UPSYNC.md`). Configurazione: 3060 = stadio 0 (layer 0-15), 2080 Ti = stadio 1 (layer 16-47, testa,
MTP), `--layer-split 16 --pipeline-windows 2`, cache elastica, `--spec 4`.

Obiettivo: da ~68 a 80 t/s, finestra media da ~26 a ~22-23 ms. Tutte le leve sono **opt-in** (variabili
`STRATA_CP_*`); senza variabili il motore lancia gli stessi grafi e fa lo stesso ciclo di `lab/upsync`. Nessuna leva
cambia un'aritmetica: l'output greedy resta identico bit per bit; le bozze MTP hanno gli stessi input negli stessi
grafi (cambiano l'ordine e il momento dei lanci). Le finestre possono differire solo dove il motore dipende già dai tempi
(policy del lookup, tier adattivo): col banco forzato `STRATA_FORCE_WINDOWS` sono le stesse.

Stato: compilato sì (vedi in fondo), installato no, provato no. Nessuna GPU, nessun motore, nessun test.

## Percorso critico di partenza (nsys + traccia della pipeline, Daily)

- Finestre fresche (73%): verdetto → lancio catena MTP sull'host 0,37 ms + catena fino alla taglia di A 1,24 ms +
  stadio 0 6,9 ms + passaggio 0→1 0,63 ms + stadio 1 16,8 ms + verdetto 0,12 ms.
- Finestre speculative (27%): solo lo stadio 1, allungato di 2,5-3,6 ms dalla catena MTP sulla stessa scheda.
- Stadio 1: kernel 82%, GPU ferma sui flag dell'host per layer 11% (22% nelle speculative), buchi fra i 1397 nodi ~6%.
- Con la B speculativa sullo stadio 0 l'attesa dei flag dello stadio 1 sale da 1,16 a 2,05 ms: lo stesso thread serve B.

## Misura comune (banco forzato, B-A-B)

Stesso exe, stessi argomenti del Daily (`docs/UPSYNC.md`, «Riprodurre il Daily»), `STRATA_DECODE_TIMING=1`,
`STRATA_FORCE_IDS=<liste>` e `STRATA_FORCE_WINDOWS=<traccia di riferimento>` (le finestre del riferimento rigiocate:
stesso lavoro nei due bracci), `STRATA_PIPELINE_TRACE=<file>` facoltativo. Tre corse alternate: B (leva accesa), A
(nessuna `STRATA_CP_*`), B. Una leva alla volta.

Da leggere per richiesta:
- `strata pipeline: N windows in X ms (Y ms/window)` e `strata pipeline classes: fresh … ms | speculative … ms`;
- `strata pipeline rollback cost` (fresh/spec, +Bbeside);
- `strata critpath: …` (una riga per richiesta quando una `STRATA_CP_*` è accesa: i contatori di ogni leva);
- identità: `strata work: … accepted X of Y drafts, … target picks replaced on verified rows Z` deve dare la stessa
  Z nei due bracci (le scelte del target sono le stesse); senza forzatura, il testo greedy identico.

Contatori nella traccia della pipeline: `L1` (lancio dello stadio 1), `CP` (catena accodata, x0 = passi), `CL` (catena
lanciata o adottata).

## 1. Servizio dello stadio 1 prima della B — `STRATA_CP_S1FIRST=1`

**Cosa fa.** Con `pl_yield` (profilo `STRATA_PIPELINE_ELASTIC`, di default con due schede) il ciclo serve A sullo
stadio 0, poi lo stadio 1 di A, poi la B speculativa un layer per passata. Il buco: la chiamata al pool per un layer di
B (CPU) può partire un attimo prima che lo stadio 1 suoni, e lo stadio 1 aspetta tutta la chiamata. Con la leva, mentre
lo stadio 1 di A è in volo, un layer di B **già suonato** viene servito solo se:
- lo stadio 1 non ha un layer suonato in attesa (altrimenti passa prima lui, alla passata successiva), e
- il layer di B sta prima del prossimo squillo previsto dello stadio 1 (ultimo servizio + intervallo GPU fra uno
  squillo servito e il successivo, EMA; costo di un layer di B, EMA), **oppure** siamo appena dopo un servizio dello
  stadio 1 (inizio del suo intervallo: il punto in cui B ritarda meno lo stadio 1 se il suo layer non ci sta).
- `STRATA_CP_S1FIRST_MAX_US` (1500) limita quanto un layer di B può aspettare.

B ha margine: lo stadio 1 dura ~17 ms, lo stadio 0 di B 7-11 ms. Un thread host per stadio è stato scartato: il pool
(`ExpertDispatch`) non è rientrante (piano, sink e contatori condivisi) e i due stadi si dividerebbero gli stessi core.

**Contatori.** `S1FIRST B held N passes (X ms), M B layers in a stage-1 gap, K past the bound; stage-1 ring gap G ms,
B layer C ms`.

**Misura.** B-A-B; atteso: classe `speculative` e `fresh+Bbeside` del rollback cost giù di ~0,5-1 ms (obiettivo
~1 ms), finestre `fresh` senza B invariate. Con `STRATA_VERIFY_PROFILE` non usare (blocca la pipeline, vedi handoff).

**Rischi.** Se B è spesso sul percorso e il suo stadio 0 si allunga oltre lo stadio 1 di A, la finestra dopo il
verdetto parte più tardi: guardare `K past the bound` e la classe `spec`. Solo con `pl_yield` acceso.

## 2. Meno flag host per layer — `STRATA_CP_RESPLAN=1` (`STRATA_CP_RESPLAN_MIN`, default 1.0)

**Perché DEVPLAN (E-6 in pipeline, `STRATA_FP_DEVPLAN` di `lab/expert-persist`) peggiorava del 6%.** Letto il
codice di E-6 sotto `always_publish_`:
1. **L'host serviva lo stesso ogni layer pianificato dal device**, e lo serviva tardi: il doorbell incrementa il
   contatore e pubblica righe e id nello stesso buffer per ogni layer; la GPU, che non aspetta più l'host su quei layer,
   va avanti e sovrascrive il buffer. L'host calcolava allora sulla CPU gli esperti «mancanti» con le righe di un altro
   layer: lavoro CPU buttato sul thread che serve anche gli altri stadi e **conteggi d'uso inquinati** (le righe del
   layer l+2 contate sul layer l) → scambi peggiori del tier adattivo → meno hit.
2. `copy_or_zero_from_mapped` copiava **tutte** le n·k righe dalla memoria mappata quando il gruppo non era pianificato
   sul device (T=4: 40 righe × 2048 float ≈ 320 KB su PCIe per layer), invece delle sole righe CPU come il percorso
   normale (`copy_rows_from_mapped`): ~25 µs per layer, ~0,8 ms per finestra sullo stadio 1.
3. `resident_plan` e le attese `_or` su **ogni** layer, anche dove quasi mai tutti gli instradamenti sono in VRAM.
4. Tabella caricata con `cudaMemcpy` sincrono prima di ogni lancio in pipeline (quando cambia) e la stretta di mano col
   fence del tier.

**Cosa fa questa variante.**
- Solo i layer dello stadio con residenza ≥ `STRATA_CP_RESPLAN_MIN` al momento della cattura (default 1.0: tutti i 512
  esperti in VRAM, «i soli layer tutti residenti») entrano nella maschera; il primo layer dello stadio 0 resta fuori
  (il suo flag sblocca le righe PLE). Gli altri layer hanno il grafo di sempre, senza kernel in più.
- Secondo set di grafi (`exec_cp_`, catturati in `capture_all`): nei layer della maschera `resident_plan` prova il
  piano sul device; se tutti gli instradamenti del gruppo sono in VRAM il doorbell (`doorbell_publish_cp`) pubblica
  **solo gli id** in un registro per squillo e scrive `map[squillo] = 1`, le attese tornano subito e si copiano solo le
  righe CPU (zero, senza leggere PCIe). Altrimenti pubblica tutto e `map = 2`: servizio normale.
- Ogni squillo è **memorizzato** (valore), non incrementato: l'host legge `map[squillo]` e su `1` non chiama il pool,
  conta solo l'uso degli id (`drive_note_split`: stessi incrementi del pool, ogni voce un hit) e passa oltre.
- Coerenza tabella device/host: prima di ogni lancio (`cp_sync_res`) le righe dello stadio e gli offset degli slot (un
  arena elastica li sposta) vanno sul device se l'host è cambiato, in asincrono sullo stream dello stadio da memoria
  pinned. Un esperto sfrattato è `-1` sull'host prima che il suo slot venga riusato: una finestra lanciata dopo il marchio
  ha la tabella nuova; una lanciata prima è dietro il fence del tier. Per il tier bloccante (thread a parte) il fence
  aspetta che il ciclo passi un punto fra due lanci dopo i marchi (`cp_fence_req/ack`), così nessun lancio cade fra il
  confronto e gli eventi del fence. Un layer tutto residente non ha candidati per il tier, quindi di solito non cambia.

**Contatori.** All'avvio: `strata critpath: STRATA_CP_RESPLAN: layers [a, b) on CUDAx: N at >= M% VRAM residency …:
<layer>`. Per richiesta: `RESPLAN (all requests): stage k: N layers, P planned / H host, U uploads`.

**Misura.** Prima guardare la riga d'avvio: se nessun layer è al 100% (probabile sulla 3060, possibile sulla 2080 Ti),
provare `STRATA_CP_RESPLAN_MIN=0.95` / `0.9` (più layer nella maschera, più `resident_plan` che falliscono: il rapporto
P/H dice quanto rende). B-A-B; atteso: `ms_wait`/waitA dello stadio 1 giù per i layer pianificati (~0,05-0,2 ms per
layer-finestra), CPU host libera per l'altro stadio; `U` pochi per richiesta (uno per giro del tier, non uno per
finestra).

**Rischi.** Correttezza dipende dalla tabella device allineata (sopra); un errore lì produce output sbagliato, non un
blocco: confrontare `target picks replaced` e il testo. L'uso dei layer pianificati è contato dagli id (giusto), ma il
lookahead CS-T e il trace di routing (`--dump-routing`) non li vedono. Maschera decisa una volta (prima cattura): un
layer che diventa tutto residente dopo non entra. Due set di grafi: più tempo di cattura e un po' di memoria.
`STRATA_VERIFY_PROFILE` non mostra lo stadio waitA dei layer pianificati.

## 3. Catena MTP — `STRATA_CP_CHAIN_PRE=n` e `STRATA_CP_CHAIN_FUSE=1`

### 3a. Round e primi passi in coda allo stadio 1 — `STRATA_CP_CHAIN_PRE=n` (n = 1 .. S_mtp-1; 1 = «round e primo passo»)

**Cosa fa.** Appena lanciato lo stadio 1 di A (e finita la catena che ha fatto B, e deciso se B parte), il ciclo
accoda sullo stream della catena, **dietro l'evento di fine del grafo dello stadio 1**: copia delle righe R di A,
kernel `mtp_chain_verdict` (dagli argmax di A sul device, più le sostituzioni del banco forzato: calcola a, «sul
percorso», e scrive token, righe step/pos, riga e forzature come `chain_launch` sull'host), il round e i primi n passi
(`MtpDrafter::chain_prelaunch`). La catena parte quando finisce lo stadio 1, senza giro dall'host (verdetto 0,12 ms +
lancio 0,37 ms + latenza di polling). Al verdetto `chain_adopt` confronta quello che il device ha derivato con quello
che l'host lancerebbe (stesse righe mappate, stessa B): se coincide lancia solo i passi restanti, altrimenti lascia
finire la catena accodata e lancia come prima. Solo greedy senza campionamento sull'host (`host_sampling()`), MTP sulla
stessa scheda dello stadio 1.

**Contatori.** `CHAIN_PRE Q queued, A adopted, R redone`; traccia `CP`.

**Misura.** B-A-B con `n=1`, poi `n=2`/`3`. Atteso: finestre `fresh` −0,3/−0,6 ms (`the chain from its launch to B:
fresh` scende), `A adopted` vicino a `Q queued`. Le bozze sono le stesse di prima (stessi input, stessi grafi):
`accepted X of Y drafts` uguale fra i bracci.

**Rischi.** La catena accodata gira anche quando non serve (fine richiesta) o va rifatta (B cambiata dopo
l'accodamento: `redone`): GPU della 2080 Ti occupata ~0,5-1,5 ms in più in quei casi. Mai con richieste campionate.

### 3b. Meno lanci per passo — `STRATA_CP_CHAIN_FUSE=1`

**Cosa fa.** Il round copia in una volta le righe step/pos di tutti i passi (l'host le prepara tutte prima del round,
in `chain_launch` e `draft()`; la variante 3a le scrive tutte dal device), quindi un passo non copia più la sua; e
`force_token` è dentro `mtp_select`. Due kernel in meno per passo (54 → 52), uno in meno nel round, parole identiche.

**Onestà sul bersaglio 0,93 → 0,5 ms.** Questa leva toglie ~2 lanci su 54: ~0,02-0,04 ms per passo, non 0,4. Il passo è
fatto di kernel piccoli limitati dalla latenza, ma i grandi (proiezioni, attenzione su 32k celle, testa sul
sotto-vocabolario, esperti) restano; i grafi per passo esistono già (e il grafo unico della coda della catena,
`STRATA_MTP_CHAIN_TAIL_GRAPH`, handoff del 7/10 §9, non ha dato beneficio ripetibile). Per dimezzare il passo serve una fusione vera dei kernel (norm+quantize,
gate+quantize, argmax+top-prob, un kernel persistente per il layer MTP), da fare **con la lista per kernel del passo da
nsys** e una prova di parità bit per bit per ogni fusione: non fatta qui alla cieca. Leva complementare già pronta:
`STRATA_ATTN_DAILY=1` (kernel dell'attenzione QSA del Daily, ~2× sul kernel dei chunk, vale anche per il drafter).

**Misura.** B-A-B; `the chain from its launch to B` e classe `spec` (la catena gira accanto allo stadio 1).

**Rischi.** Bassi: stesse parole nelle stesse posizioni; HIP non cambia la copia per passo (solo `mtp_select`).

## 4. Ordine di lancio — `STRATA_CP_L1FIRST=1`

**Cosa fa.** Lo stadio 1 di A viene lanciato nella stessa passata in cui il ciclo vede A finire lo stadio 0, subito
dopo i pump: prima della lettura della catena, del gap elastico e del lancio speculativo di B (commit di A sullo stadio
0 con l'eventuale attesa del commit precedente, snapshot, staging). Prima lo stadio 1 aspettava tutto questo
(~0,45 ms nelle finestre fresche). Non è `EARLY_L1` di FABLE (lancio anticipato dietro l'evento dello stadio 0): qui
cambia solo l'ordine delle chiamate host.

**Contatori.** `L1FIRST N early stage-1 launches`.

**Misura.** B-A-B; atteso: finestre `fresh` −0,3/−0,45 ms; nella traccia l'intervallo `F0`→`L1` scende a pochi µs.

**Rischi.** Nessuno noto: lo stadio 1 dipende solo dall'hand-off di A, già scritto; `--adapt-async` riceve `a_gap(1)`
come prima. Il contatore `chain late` può cambiare di poco (conta B non pronta al lancio dello stadio 1).

## Ordine consigliato delle prove

Una variabile alla volta, B-A-B ciascuna, testo/`target picks replaced` identici:
1. `STRATA_CP_L1FIRST=1`
2. `STRATA_CP_S1FIRST=1`
3. `STRATA_CP_CHAIN_PRE=1` (poi `2`, `3`)
4. `STRATA_CP_CHAIN_FUSE=1`
5. `STRATA_CP_RESPLAN=1` (leggere la riga d'avvio; se 0 layer, `STRATA_CP_RESPLAN_MIN=0.95`)

Poi le promosse insieme contro A.

## Stato della build

Compilato sì (`build_lock.py`, CUDA 75;86, test spenti), installato no, provato no: nessuna GPU, nessun motore, nessun
test. `strata.exe` dal codice di `7a267888`: sha256 `7c73ef0fe9530cfa3095500f2be88d2f679c0ec1831873adf6ba6f32b514e87f`,
copiato con `build.json` in `moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\
build-critpath2-<sha>\`. Senza `STRATA_CP_*` deve comportarsi come `lab/upsync` `431e28a7`.

## File toccati

- `src/program/generate.cpp`: ciclo della pipeline (leve 1, 3a, 4, stretta di mano del fence, riga `strata critpath`),
  `drive_note_split`, `cp_env_int`.
- `src/core/verify.cpp`, `include/strata/core/verify.hpp`: leva 2 (`cp_prepare`, `cp_sync_res`, grafi `exec_cp_`,
  servizio), accessori per 1 e 3a.
- `src/core/mtp.cpp`, `include/strata/core/mtp.hpp`: `chain_prelaunch`, `chain_adopt`, FUSE.
- `src/kernels/cuda/elementwise.cu`, `verify_kernels.cu` (+ header): `doorbell_publish_cp`, `mtp_chain_verdict`,
  `mtp_select` con forzatura.
