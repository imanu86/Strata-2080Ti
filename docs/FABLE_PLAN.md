# FABLE_PLAN — decode a due schede: da 28,7 a ~24,6 ms per finestra (lab, tutto opt-in)

Branch `lab/fable-plan` su `main` (ca43719). Scritto senza compilare né eseguire: **compilato no, installato no,
provato no**. Ogni voce è dietro una sua variabile d'ambiente, default spenta: lo stesso binario serve il braccio A
(tutto spento) e il braccio B (una variabile accesa) di ogni confronto. Il piano di misura è in fondo.

Profilo di partenza (macchina dell'owner, 131k di contesto, 3060 = stadio 0 strati 0-15, 2080 Ti = stadio 1 strati
16-47 + testa + drafter, `--pipeline-windows 2`): finestra FRESH ~32,6 ms profilata (verdetto→catena 0,46; catena fino
alla taglia di A 2,2; `cudaGraphLaunch` stadio 0 0,35; stadio 0 10,4; F0→L1 1,2; stadio 1 18,1); finestra SPECULATIVA
on-path 24,2 ms (V→L1 1,6 + stadio 1 22,4, gonfiato di 0,73-0,84 ms per passo della catena ad alta priorità).

## Convenzioni di misura

- Bench deterministico dell'owner, decode a 131k, `STRATA_DECODE_TIMING=1` e `STRATA_PIPELINE_TRACE=<file>` accesi in
  entrambi i bracci: le righe `strata pipeline: … ms/window`, `strata pipeline classes: fresh … | speculative …`,
  `strata decode stage k: … graph launch … ms/window` e la traccia (eventi `L0 L1 F0 F1 V CL CE CD`, più i nuovi
  `CT` = catena potata, `EA` = copie elastiche asincrone accodate).
- Testo: con una sola variabile accesa alla volta il testo greedy deve restare identico al braccio A, tranne dove è
  scritto il contrario (la sonda `STRATA_FP_PROBE_WAITA` lo rompe di proposito).
- Un confronto = B A B (tre run alternati) per voce, poi tutte le voci promosse insieme contro il braccio A.

## 1. Catena MTP a bassa priorità — `STRATA_MTP_PRIORITY`

- Commit: `85234937 FABLE-MTP-Aggiungi la priorità minima del flusso della catena`.
- Variabile: `STRATA_MTP_PRIORITY=1` (default: massima, com'era), `0` = priorità del device, **`-1` = minima** (nuovo).
  Letta in `MtpDrafter::load` (`src/core/mtp.cpp`); con `-1` stampa `strata mtp: STRATA_MTP_PRIORITY=-1: …`.
- Cosa misurare: le finestre speculative on-path (`speculative … ms`), dove la catena gira sulla 2080 Ti accanto allo
  stadio 1; e la latenza della catena (`the chain from its launch to B: forced … ms`), che con la priorità minima può
  crescere e ritardare B (`chain late` nella riga `classes`).
- Attesa: fino a −0,7/−0,8 ms per passo di catena sovrapposto allo stadio 1 (le 24,2 ms della speculativa verso ~22,4),
  se la catena non diventa il collo di bottiglia di B.
- Non verificato: che `cudaDeviceGetStreamPriorityRange` dia una priorità "minima" distinta dal default su WDDM (se
  coincidono, `-1` ≡ `0`).

## 2. Lancio anticipato dello stadio 1 nelle finestre fresche — `STRATA_FP_EARLY_L1`

- Commit: `1ab9a37b FABLE-SPLIT-Misura il tempo host di pl_launch per stadio e accetta un evento di attesa`,
  `8c7cd800 FABLE-SPLIT-Lancia lo stadio 1 dietro l'evento dello stadio 0 nelle finestre fresche`.
- Variabile: `STRATA_FP_EARLY_L1=1`. Il timer è sempre attivo: `Verifier::pl_launch` accumula `ms_launch` (riga
  `strata decode stage k: … graph launch X ms/window`, anche per lo stadio 1 e per le finestre dispari via
  `absorb_stats`).
- Meccanica: alla `L0` di una A fresca il loop lancia subito anche `V1(A).pl_launch(…, V0(A).done_event())`: lo stream
  della 2080 Ti fa `cudaStreamWaitEvent` sull'evento che lo stadio 0 registra dopo il suo grafo (l'hand-off è scritto
  dentro il grafo), così i ~1,2 ms host del lancio dei 1397 nodi si sovrappongono ai ~10 ms dello stadio 0. A va
  sempre allo stadio 1, niente lavoro sprecato; B speculativa non cambia (il suo stadio 1 parte dopo il verdetto).
  Contatore: `STRATA_FP_EARLY_L1: N stage-1 windows launched behind stage 0's event`.
- Cosa misurare: nella traccia l'intervallo F0→L1 delle finestre fresche deve sparire (L1 subito dopo L0); la classe
  `fresh … ms`.
- Attesa: −1,0/−1,2 ms sulle finestre fresche (~70%), cioè ~−0,8 ms medi.
- Non verificato: l'attesa cross-device di un evento su WDDM (dovrebbe funzionare: eventi `cudaEventDisableTiming`,
  stream non bloccanti); che il grafo dello stadio 1 in coda dietro l'evento non ritardi la catena MTP sullo stesso
  device (altra stream, non dovrebbe). Lo "GPU-reach wait" dello stadio 1 nel timing crescerà di ~10 ms per finestra:
  è contabile, non costo.

## 3. Piano sul device in pipeline (waitA → 0) — `STRATA_FP_DEVPLAN`; sonda `STRATA_FP_PROBE_WAITA`

- Commit: `e39c5e62 FABLE-SPLIT-Riattiva il piano sul device in pipeline con tabella sincronizzata ai gap … e sonda
  waitA`.
- Variabili: `STRATA_FP_DEVPLAN=1` (produzione candidata); `STRATA_FP_PROBE_WAITA=1` (solo lab: **testo sbagliato**).
- Meccanica DEVPLAN: `Verifier::init` accende `device_plan_` (E-6) anche con `always_publish_` (gli stadi in
  pipeline): ogni gruppo i cui esperti sono tutti residenti nella tabella del device è pianificato dal kernel
  `resident_plan` e la GPU salta `wait_flag_ge(flagA)`, la copia del piano, `flagB` e la copia delle righe CPU
  (`skip_`); l'host continua a pubblicare righe e piano (il suo lavoro resta, l'attesa della GPU sparisce). La
  correttezza regge perché il serve loop tiene la tabella di ogni scheda allineata a `host_res`: `fp_res_sync(card)`
  confronta le righe della scheda con l'ultima copia caricata (`fp_res_shadow`) e ricarica (32-64 KB, sincrono)
  prima di **ogni** lancio in pipeline su quella scheda — i lanci avvengono ai gap della scheda — e nella lettura
  prompt in pipeline; il fence del tier adattivo bloccante (`adapt_fence`) prima di registrare gli eventi aspetta che il
  loop abbia caricato i marchi di eviction (`fp_fence_req/ack`, serviti a ogni iterazione e prima dei join), così una
  finestra lanciata dopo i marchi ha la tabella nuova e una lanciata prima è coperta dal fence. Il tier asincrono
  marca sul loop thread e accoda le copie negli slot ad `a_gap`, che precede il lancio e quindi la sync.
  Contatore: `STRATA_FP_DEVPLAN: N residency table uploads`.
- Meccanica PROBE: in `Verifier::service`, prima dei job CPU, costruisce nel blocco mappato un piano "tutto residente"
  (ogni entry un hit VRAM dallo slot di `host_res`, slot 0 se non residente), alza flagA e flagB, poi lascia correre il
  pool con il sink puntato su un blocco scratch che la GPU non legge. Misura il limite superiore del togliere waitA
  (la GPU fa comunque il lavoro degli esperti VRAM); il testo è sbagliato per ogni esperto non residente.
- Cosa misurare: prima la sonda (B A B, solo ms/window: dice quanto vale al massimo waitA); poi DEVPLAN con testo
  identico al braccio A e il profilo GPU (`STRATA_VERIFY_PROFILE=1`, stadio `waitA` per strato). Con la cache elastica
  che cresce, contare gli upload (devono essere pochi per richiesta: una crescita o uno shrink, non uno a finestra).
- Attesa: dipende da quanti gruppi sono tutti residenti (sulla 2080 Ti con 32 strati pochi all'inizio, di più col
  tier adattivo); la sonda lo quantifica. Ordine di grandezza: 0,05-0,2 ms per strato pianificato dal device.
- Non verificato: `resident_plan` con `skip_` dentro i grafi con `always_publish_` (il codice E-6 esiste, non era mai
  acceso con la pipeline); la finestra di gara tier bloccante ↔ loop (ragionata sopra, non provata); il costo del
  `cudaMemcpy` sincrono ai gap con l'altra scheda in volo (lo stesso che fa già `pl_apply`/`res_upload`).

## 4. Crescita elastica asincrona — `STRATA_FP_EL_ASYNC`

- Commit: `755b8d96 FABLE-CPU-Rimbalza i blob della crescita elastica su memoria pinned da un thread e copia al gap
  successivo`.
- Variabile: `STRATA_FP_EL_ASYNC=1` (richiede `STRATA_PIPELINE_ELASTIC=1`, il profilo P3 già in uso).
- Meccanica: `el_grow(pipe_gap)` mappa e azzera gli slot come prima; i blob non pinned non vengono più copiati con
  `cudaMemcpyAsync` da memoria pageable (che blocca il thread per tutta la copia: ~88 ms per richiesta misurati) ma
  letti da un thread worker (`ExpertSource::copy_blob`, come il job del tier asincrono) in un bounce pinned
  (`cudaHostAlloc`, dimensione = `room` della crescita, ≤ `STRATA_PIPELINE_ELASTIC_STEP_MIB`); al gap successivo di
  stadio 0 (`el_gap`) il loop accoda le copie pinned→device su `adapt_stream`, registra `el_fill_ev` e la pubblicazione
  segue com'era. Una crescita in volo alla volta; `el_hold_tier` resta alto fino alla pubblicazione; drain di fine
  richiesta e `die` raccolgono il worker. Traccia: `EA`. Contatore: `STRATA_FP_EL_ASYNC: N growths bounced …`.
- Cosa misurare: `elastic tail beside the windows: … ms on the loop's thread` (deve scendere da ~88 a pochi ms) e
  `the N windows after them … ms/window` contro `the run`; il bench a richieste multiple con la cache che cresce.
- Attesa: −0,3/−0,4 ms medi per finestra (le 6 finestre lente per crescita spariscono), testo identico.
- Non verificato: `copy_blob` concorrente al pool per la sorgente in uso dall'owner (il tier asincrono lo fa già);
  il bounce non viene liberato all'uscita (una sola allocazione per processo).

## 5. Catena più economica — `STRATA_FP_CHAIN_TRIM`, `--mtp-window`

- Commit: `1e084f0e FABLE-MTP-Lancia i passi di coda della catena mentre escono gli output e potali alla taglia decisa
  di B`, `d37ab1c0 FABLE-MTP-Documenta --mtp-window …` (`docs/MULTI_GPU.md`).
- Variabili: `STRATA_FP_CHAIN_TRIM=1` (+ `STRATA_FP_CHAIN_TRIM_AHEAD=N`, default 2 passi in coda oltre gli output
  arrivati). Attivo solo se la richiesta ha `spec_min_p > 0` (senza la regola B prende sempre S−1 draft e ogni passo
  serve: la catena resta intera, la riga finale lo dice).
- Meccanica (`MtpDrafter::set_chain_ahead / chain_extend / chain_close`): `chain_launch` accoda round + passi forzati +
  `ahead` passi; il loop, man mano che gli output arrivano (`chain_outputs_ready`), estende di un passo finché la
  taglia di B non è decisa, poi `chain_close`: i passi mai lanciati (`CT` nella traccia, `N chain steps never
  launched`) non costano nulla allo stadio 1. Finché la catena è "aperta" `chain_poll` non la dichiara finita, così un
  host lento non tronca mai B; `chain_close` precede ogni attesa della fine.
- Cosa misurare: `speculative … ms` e `fresh … ms`, la latenza `chain … to B` (può crescere di qualche decimo: ogni
  estensione è un round trip host), `tok` per finestra invariati, testo identico.
- Attesa: 1-3 passi potati per catena forzata (0,7-0,8 ms l'uno sulla 2080 Ti) quando B si ferma a un draft sotto
  min_p; da combinare con `STRATA_MTP_PRIORITY=-1`. `--mtp-window 8192/16384`: catena più corta su 131k, da pesare
  contro l'accettazione (documentato).
- Non verificato: la latenza aggiunta delle estensioni su WDDM (`cudaStreamQuery` dopo ogni lancio); con `AHEAD=1` il
  GPU può restare ozioso tra un passo e l'altro.

## 6. Efficienza dei kernel (micro-bench bitwise, varianti opt-in)

Tre bench standalone (CMake, `STRATA_ENABLE_CUDA`): `fp_expert_bench`, `fp_hc_bench`, `fp_shexp_bench`. Ognuno
confronta bit per bit (memcmp) la variante contro il kernel attuale su input sintetici e ne misura il tempo con eventi
CUDA su entrambe le schede (`device -1`), exit code 1 se qualcosa differisce. **Il bitwise è un'argomentazione sul
codice (stesso ordine di accumulo, stesse operazioni) finché il bench non lo conferma**: la contrazione FMA del
compilatore può smentirla; una variante `DIFFERENT` non si usa.

### 6a. Esperti VRAM gate/up + down — `STRATA_FP_EXPERT_V`

- Commit: `f33473da FABLE-KERNEL-Varianti opt-in dei kernel gate/up e down degli esperti VRAM con micro-bench bitwise`,
  `cf84a08f FABLE-KERNEL-Aggiungi la variante persistente dei kernel esperti con prefetch in registri …`.
- Variabile: `STRATA_FP_EXPERT_V=0..4` (setter `native_expert_set_fp_variant`), solo per la coppia IQ3_S/IQ4_NL a
  2560/640: 1 = codebook IQ3_S in shared memory (gate/up); 2 = due righe per sub-warp con i load di entrambe prima dei
  dp4a (gate/up e down); 3 = 1+2; 4 = kernel persistente (griglia = k×SM, `STRATA_FP_EXPERT_PERSIST_K`, codebook
  caricato una volta per blocco, doppio buffer in registri; cp.async solo su sm_86).
- Bench: `fp_expert_bench [reps] [device] [groups]` — T 1..4, ~20 gruppi, pool di blob > L2, righe
  `dev T v us GB/s %peak identical|DIFFERENT`. Esiste anche `native_expert_bench` su righe GGUF reali.
- Cosa misurare: la variante bitwise più veloce su ciascuna scheda (obiettivo ≥ 65% del picco contro 41-43%), poi
  nel decode `STRATA_FP_EXPERT_V=<n>` sui ms/window e sul profilo GPU (`VRAM hits`).
- Attesa: fino a ~3 ms per finestra a 70% del picco; realisticamente 1-2 ms se una variante regge.
- Non verificato: compilazione (template con `if constexpr` su array di dimensione dipendente, `S26IQ3S::load<STG>`),
  pressione dei registri delle varianti a due righe (`__launch_bounds__(256)`), bitwise.

### 6b. Lettura hc BF16 — `STRATA_FP_HC_FUSE_NORM`

- Commit: `12188f2e FABLE-KERNEL-Fondi la norm nella proiezione down della lettura hc con micro-bench bitwise`,
  `cc49098e FABLE-KERNEL-Aggiungi il livello 2 della norm fusa su 81 blocchi …`.
- Variabile: `STRATA_FP_HC_FUSE_NORM=1|2` (setter `fused_gr_set_fp_fuse_norm`), solo CUDA e solo dove la scheda usa
  la lettura `staged` (la default): il lancio della norm sparisce, ogni blocco della down ricalcola rs e i tile di xn
  con l'aritmetica e l'ordine della norm (xn e rs restano scritti, un blocco per tile); 2 = stessa cosa su 80+1
  blocchi con 4 warp di righe (copre i 68 SM della 2080 Ti). `fused_gr_check` la verifica bit per bit all'avvio e
  ricade sulla lettura normale se differisce (riga `strata hc: CUDAk: …`).
- Bench: `fp_hc_bench [iters] [Tmin] [Tmax] [device]` — default vs livello 1 vs livello 2, T 1..6, apply×inject,
  `%peak`.
- Cosa misurare: il bench (il calo con T a 38%/33% indica shared memory/ALU, non DRAM: il livello 2 è la risposta
  "più SM"); nel decode il profilo GPU `hc read`/`norm` per strato.
- Attesa: −0,2/−0,5 ms per finestra (48 lanci di norm in meno, 4-7% di efficienza → inglobati). Passi successivi NON
  bitwise, non implementati: split-K della down, hc in Q8_0 (`STRATA_HC_Q8`, path esistente, metà dei byte).
- Non verificato: compilazione (lambda generica con `cudaFuncGetAttributes`, extern shared), registri/spill a T ≥ 5.

### 6c. Esperto condiviso — `STRATA_FP_SHEXP_FUSE`

- Commit: `2bee311d FABLE-KERNEL-Riduci i lanci dell'esperto condiviso con sigmoide e SwiGLU nel kernel down …`.
- Variabile: `STRATA_FP_SHEXP_FUSE=1|2` (setter `shared_expert_set_fp_fuse`), solo sul path con gate BF16 nativo non
  differito e down IQ4_NL: 1 = il gate scalare prima della down e `sigmoid(g[t])` nell'epilogo del kernel down (un
  lancio in meno); 2 = anche SwiGLU + quantizzazione q8_1 nel prologo dello stesso kernel (due lanci in meno, più
  lavoro ridondante per blocco). Nuovi entry `native_iq4_nl_mmvq_scaled` / `native_iq4_nl_swiglu_mmvq_scaled`.
- Bench: `fp_shexp_bench [iters] [device]` — geometria del modello (gate Q4_K, up IQ3_S, down IQ4_NL), T 1..4.
- Cosa misurare: il bench; nel decode poco (l'esperto condiviso gira su una stream laterale e `STRATA_LFUSE=1`,
  esistente, fonde già gate+up e sposta il gate nel router: confrontare anche contro quello).
- Attesa: ≤ 0,1-0,2 ms per finestra; voce a bassa priorità.
- Non verificato: compilazione (`if constexpr` con `__shared__` in template, 33,8 KB di shared al livello 2 NCOLS=8),
  bitwise del livello 2.

## Piano di test (B A B per voce, poi tutto insieme)

Ordine consigliato, dal rendimento atteso e dal rischio più basso:

1. `STRATA_MTP_PRIORITY=-1` (già in prova dall'owner) — B A B.
2. `STRATA_FP_EARLY_L1=1` — B A B; verificare nella traccia F0→L1 ≈ 0 sulle fresche.
3. `STRATA_FP_CHAIN_TRIM=1` — B A B; poi insieme a 1.
4. `STRATA_FP_EL_ASYNC=1` — B A B su un bench con più richieste e crescita della cache.
5. `STRATA_FP_PROBE_WAITA=1` — B A B solo per i ms (testo rotto); se il delta è > 0,5 ms, `STRATA_FP_DEVPLAN=1` B A B con
   testo identico.
6. Kernel: prima i tre bench su entrambe le schede (bitwise + tempi), poi nel decode la sola variante vincente per
   famiglia (`STRATA_FP_EXPERT_V=n`, `STRATA_FP_HC_FUSE_NORM=n`, `STRATA_FP_SHEXP_FUSE=n`), B A B ciascuna.
7. Tutte le voci promosse insieme contro il braccio A: obiettivo ≤ 24,6 ms/finestra (80 t/s), testo identico.

Se una build fallisce su una voce, la voce è isolata nel suo commit: `git revert` del singolo commit tiene il resto.
