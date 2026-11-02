# PIPELINE3_PIANO: il parallelo vero, tappe di esecuzione

Ramo `lab/upsync-v0.2.9`, 2/11 sera. Segue `PIPELINE3.md` (conti) e `PIPELINE2.md` §2a (salto fra le schede). Macchina:
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, 0 = 3080 Ti 12 GB sm_86 (Oculink x4), 1 = 3060 (schermo, mai usata), 2 = 2080 Ti 22 GB
sm_75 (x16); niente P2P. Modello Qwen3.8-Flash-Next IQ3_XXS, 48 strati, k = 10.

Convenzione: **M** = misurato, **S** = stimato da misure. Tutto il lavoro sta dietro **`STRATA_EP=1`**, spenta di
default: spenta, nessuna riga di esecuzione cambia e il testo è identico (verifica a cache fissa in ogni tappa).

## 0. Conclusione

- **Si procede.** Il banco `ep_bench` non era ancora girato alla stesura (§1: la coda F25 teneva il banco), quindi
  nessun numero vero dice che i 115 t/s non si raggiungono; la stima resta S e la **tappa 1 è il primo numero vero
  nel motore** (2-3 giorni, non 10-20): se lì l'esposto per strato supera i 125 µs si cambia forma, se supera i 200 µs
  anche col kernel unico ci si ferma.
- Finestra rifatta (§1, S da M): 15,4 ms di pavimento (11,7 della padrona con i 9 hit locali, M, + 1,5 miss CPU + 1,7
  bozze + 0,5 emissione e stadio) + 48 × esposto: a 80 µs 19,2 ms (~125 t/s), a 100 20,2 (~119), a 125 21,4
  (~112-115), a 150 22,6 (~106). **115 t/s ⇔ esposto ≤ ~120 µs** a 2,4 token a finestra (M: 2,3-2,6).
- La lettura dei prompt lunghi (vincolo: entro il −10% di 900 / 1850 / 2210 t/s a 4K/32K/128K, M) si tiene con la
  **divisione a strati nel prompt ed expert-parallel nella scrittura** (§3a): i densi degli strati 20-47 stanno su
  entrambe le schede, la 2080 Ti legge il prompt come stadio 1 come oggi, e dopo il prompt il K/V degli strati 20-47
  passa sulla 3080 Ti (0,3 s a 128K, S: < 1% del tempo di lettura). È la tappa 4; le tappe 1-3 misurano e chiudono la
  scrittura con la lettura di `Vpeer` (−33/−42% sui prompt lunghi, M), che per misurare la scrittura basta.
- Cache elastica: resta spenta con il peer (`generate.cpp:3699`, già così). Nucleo RAM: fermo nelle tappe 1-2, torna
  nella tappa 3 come sorgente dei miss per il server. Adattamento: resta, fra le finestre, con la tabella di residenza
  del server rinfrescata insieme a quella della padrona. Catena (`--pipeline-windows`): **spenta** (vale 0, M; con
  `--peer-device` è già rifiutata, `generate.cpp:5221`).

## 1. Il banco `ep_bench`: non ancora misurato, e come si legge quando c'è

Alla stesura (2/11, 17:43) `D:\ds4_work\strata\ep_bench\ep_bench_run1.txt` **non esiste**: la coda F25 era ancora
sull'ultimo braccio (`Wbet02`, `strata.exe` acceso, `bench.lock` suo) e la sessione principale lancia il banco dopo.
Le colonne di `PIPELINE3` §7 (`floor`, `persist`, `host`, `host-ns`, `exp.*`, `wait@rows p50/p90`) vanno lette **prima
di iniziare la tappa 1** e scritte qui; fino ad allora i numeri di questo piano sono S:

| lettura di `exp.persist` a «T=3 as Vpeer» | esposto atteso nel motore (grafo, S) | finestra (S) | t/s (S) | che cosa fare |
|---|---:|---:|---:|---|
| ≤ 100 µs | 80-120 | 19-21 | 115-125 | tappa 1 così com'è |
| 100-150 | 120-170 | 21-23,5 | 102-115 | tappa 1, poi il kernel unico (1b) prima della tappa 2 |
| 150-200 | 170-220 | 23,5-26 | 92-102 | solo il kernel unico può salvarla: 1b subito, con righe bf16 e caldi alla padrona |
| > 200 | — | ≥ 25 | ≤ 96 | **stop**: il tetto non regge (`procede = false`) |

Come si passa dal banco al motore: il grafo del server (§3c) aggiunge ~15-20 µs di nodi dipendenti a strato rispetto
al `persist` del banco; il banco sottostima la copertura della padrona (una lettura di banda da ~90 µs contro i
245 µs misurati a strato), e questo toglie fino a 40-50 µs; `wait@rows p50` dice quanto del server resta scoperto
davvero. Controllo di coerenza: `exp.host` deve stare a 400-500 µs (i 430-530 di `Vpeer`, M); se non ci sta, il banco
non rappresenta il motore e la tappa 1 è l'unico numero che conta.

Finestra = 11,7 (M, `Vpeer`: 48 strati della 3080 Ti con ~9 hit locali) + 48 × esposto + 1-2 miss CPU in più (M:
+0,4-1,0 voci a strato-finestra rispetto alla serie) + bozze 1,2-2,1 (M) + emissione e stadio 0,5 (M). Pavimento
15,4 ms (~155 t/s asintotici a 2,4 token a finestra); 25,3 ms oggi (~104 t/s, M).

## 2. Il codice reale (letto il 2/11): che cosa c'è già, che cosa manca

Percorso di una finestra di scrittura, padrona = CUDA0 (`verify.cpp`, `expert_source.cpp`, `peer_experts.cpp`):

1. **Il grafo della padrona pubblica già x e gli id sul device**: `pre(l)` lancia `doorbell_publish_res`
   (`elementwise.cu:329`): copia nelle pagine mappate `m_ids_` (T×k int) e, se almeno un esperto non è residente sulla
   padrona, `m_x_` (T×2560 fp32), poi `__threadfence_system` e `m_seq_ += 1` (ring `(l-lb)*G+grp+1`, azzerato a
   ogni finestra, `verify.cpp:2929`). Il filo del pool aspetta `seq`, pianifica (`expert_source.cpp:4905-5047`:
   `kind[i]` = 0 VRAM padrona, 1 PCIe, 2 peer, −1 CPU), pubblica il piano (flag A), e la padrona lo copia con
   `wait_copy_i32_from_mapped` (`verify.cpp:2291`).
2. **Il peer di oggi è guidato dall'host a ogni strato**: subito dopo il piano il pool chiama `PeerExperts::launch`
   (`peer_experts.cpp:260`: `On(device_)`, 2 `cudaMemcpyAsync`, `hadamard_rows`, `quantize_q8_1_rows`,
   `native_expert_grouped`, `scatter_rows_f32` verso le righe mappate `out`), fa i suoi miss, e in
   `expert_source.cpp:5277` chiama `finish`, che gira su `cudaStreamQuery` (`peer_experts.cpp:339`); solo dopo il pool
   torna e `verify.cpp` alza `m_flag_` (le righe CPU). Il grafo copia le righe con `copy_rows_from_mapped`
   (`verify.cpp:2334-2340`, salta le righe hit) e fa il combine. Ogni strato paga 0,43-0,53 ms di host (M, `Vpeer`).
3. **Il piano sul device esiste già per la padrona**: `resident_plan` (`verify_kernels.cu:1514`) costruisce il piano
   dagli id e dalla tabella di residenza `hits_.d_res` senza host (E-6, `device_plan_`), e `native_expert_grouped`
   (`iq_kernels.cu:6246`) prende i conteggi **sul device** (`n_groups` puntatore, `cap_groups` per la griglia): è la
   forma che il grafo della padrona usa già (`verify.cpp:2273`). Quindi il server può pianificare e calcolare senza
   che l'host conti niente.
4. **Le attese con flag in memoria mappata sono il pane del grafo**: `wait_flag_ge_kernel` (`verify_kernels.cu:1224`)
   gira su una parola mappata con `strata_spin_pause`; la copia con attesa fusa (`copy_rows_from_mapped_after`,
   `elementwise.cu:289`). Un nodo dipendente nel grafo costa ~2,4 µs (`verify.cpp:396`). I buffer mappati sono
   `Portable` solo con `--peer-device` (`set_peer_portable`, `generate.cpp:3710`): la 2080 Ti li può mappare.
5. **Il prompt col peer** (`prefill.cpp:2269`, `Prefill::set_peer`): senza P2P prende la via dell'host
   (`STRATA_PF_PEER_HOST`), righe compatte, somme in f16, una quota `ps_frac 0,35` di esperti letti in streaming dalla
   RAM. Con `--layer-split` ogni stadio ha il proprio `Prefill` (`generate.cpp:10160`, `set_stage`) e lo stadio
   successivo presta la scheda come aiuto (`help_pp`). `--peer-device` oggi **rifiuta** `--layer-split`
   (`generate.cpp:4174`) e `--prompt-cache-file` (`generate.cpp:3760`).
6. **Cancelli di oggi che il peer chiude**: elastica → fissa (`3699`), catena (`5221`), prefisso su disco (`3760`),
   sessioni su file (`14730`), salto di coda `STRATA_ROUTE_TAIL_SKIP` (`expert_source.cpp:4938`, spento con peer).
   `STRATA_ARENA_PIN_GIB=8` nella configurazione `PEER` della coda: solo 8 GiB dell'arena sono bloccati (l'arena intera
   esauriva la memoria bloccabile e il flusso CUDA della 2080 Ti non si apriva).
7. **Copia del K/V fra schede**: `conversation_snapshot.cpp` fa le copie sulla scheda dello stato (`QsaState::dev`,
   d14aa61e) per fasce di posizioni: è la base per lo specchio del K/V della tappa 4, non si scrive da zero.

Che cosa manca, in ordine: (a) un server sulla 2080 Ti che aspetti `m_seq_` **dal device**, pianifichi sul device e
scriva le righe con un flag suo; (b) il grafo della padrona che aspetti anche quel flag; (c) il pool che con `STRATA_EP`
non lanci e non aspetti il peer; (d) le tabelle di residenza (padrona e server) sul device del server, rinfrescate fra
le finestre; (e) il prompt a stadi con la scrittura EP; (f) i miss dalla RAM bloccata letti dal server.

## 3. Decisioni

### 3a. Vincolo lettura: divisione a strati nel prompt, expert-parallel nella scrittura

Perché `Vpeer` perde il 33-42% (M): a 4K fa 892 t/s contro 860-934 (pari), a 32K/128K crolla. Non sono gli esperti
(il peer del prompt li divide già): è l'**attenzione a contesto lungo di 48 strati su una scheda sola** (12 strati QSA
tutti sulla 3080 Ti invece di 5/7 divisi) più i densi di 48 strati invece di 20. Nessuna ripartizione di esperti la
recupera: serve che i densi e il K/V degli strati 20-47 stiano anche sulla 2080 Ti nel prompt.

Disegno (tappa 4): con `STRATA_EP=1`, `--layer-split 20 --peer-device 2` è ammesso. Il prompt gira **come oggi** (stadio
0 su CUDA0 strati 0-19, stadio 1 sulla 2080 Ti strati 20-47, `help_pp` e il peer del prompt per gli esperti del chunk).
La scrittura gira sul verificatore di CUDA0 esteso a 0-47 (i densi di 20-47 anche su CUDA0: in `Vpeer` ci stanno già,
700 MiB liberi, M) col server EP. Fra prompt e scrittura: **specchio del K/V** degli strati 20-47 per le posizioni
nuove, 2080 Ti → host → 3080 Ti (12 strati QSA, 7 nello stadio 1: ~14 KB a token, 1,8 GB a 128K, 0,3 s sul percorso
12 → 6,7 GB/s, S: +0,5% del tempo di lettura a 128K, +0,2% a 4K) e gli stati GDN (piccoli); fra una risposta e il
prompt seguente, il contrario per i token generati (256 × 14 KB = 3,6 MB). Il prefisso su disco (`--prompt-cache-file`)
resta rifiutato con il peer finché lo specchio non è provato: è un pezzo della tappa 4.

Costo in VRAM (S): la 2080 Ti perde i densi di 28 strati (~1,75 GiB) e il K/V di 7 strati a contesto pieno (1,8 GB a
128K): ~−2000 slot sui 10972 di `Vpeer` (−18%), cioè +0,3-0,5 voci CPU a strato-finestra (S da PIPELINE2 §2b), che la
tappa 3 (miss sul server) ricompra.

Scartate: la doppia modalità con pesi ricaricati (`PIPELINE2` §9c: 1-2,5 s a cambio, +30-50% di TTFT a 4K); il prompt
tutto EP con più esperti alla 2080 Ti (non tocca l'attenzione, che è il collo).

### 3b. Convivenza

| cosa | oggi con `--peer-device` | con `STRATA_EP=1` |
|---|---|---|
| cache elastica | forzata fissa (`3699`) | uguale: fissa. Il prestito del prompt non tocca le tabelle in scrittura |
| nucleo RAM (`STRATA_RAM_CORE`) | spento nella configurazione `PEER` | tappe 1-2 spento; tappa 3: la fetta bloccata (8 GiB) è la sorgente dei miss del server |
| adattamento (swap fra le finestre) | `adapt` + `apply_pending` del peer (`generate.cpp:11897, 12192`) | uguale, più l'invio delle righe cambiate delle due tabelle di residenza al device del server prima del lancio della finestra (≤ 32 swap: < 20 µs, S) |
| catena (`--pipeline-windows`) | rifiutata (`5221`) | spenta: vale 0 (M, `Eb0*`) |
| piano sul device E-6 (`device_plan_`) | solo con tutto residente sulla padrona | tappa 2: esteso a «residente sulla padrona o sul server» |
| `STRATA_ROUTE_TAIL_SKIP` | spento col peer | spento (cambia il testo; non è di questo lavoro) |
| bozze (drafter MTP) | sulla 3080 Ti (`Vpeer`: 1,2-2,1 ms, M) | uguale |

### 3c. Forma del server: prima a grafo, poi (solo se i numeri lo chiedono) a kernel unico

Il «kernel persistente» di `PIPELINE3` §3a fonde `hadamard` + `quantize` + `native_expert_grouped` (3-5 kernel) +
`scatter` in un lancio per finestra: 6-10 giorni, perché i kernel degli esperti vanno riscritti dentro un ciclo sugli
strati. La **tappa 1 non lo fa**: mette sulla 2080 Ti un **grafo per finestra** di 48 × ~8 nodi (attesa di `m_seq_` sul
device, copia di x e id dalle pagine mappate, piano sul device, hadamard, quantize, esperti con i conteggi sul device,
scatter nelle righe mappate + flag), lanciato una volta a finestra prima del grafo della padrona. L'host è fuori dal
percorso dello strato esattamente come col kernel unico; il server paga ~2,4 µs a nodo fra i suoi kernel (~15-20 µs a
strato, S), che il banco §1 isola (colonna `persist` = kernel unico; il grafo sta fra `persist` e `host-ns`). Se la
tappa 1 misura un esposto a strato entro 30 µs dal `persist` del banco, il kernel unico non serve mai.

## 4. Tappe

Ogni tappa compila da sola (`build_lock_k.py`, sm_75+86), si prova da sola con un braccio `FEPn` della coda
(`coda_notte.py`, §5) e si chiude con una misura. Commit piccoli `FABLE-STRATA-…`, il ramo compila a ogni commit.

### Tappa 1 — il server a grafo sulla 2080 Ti, e l'esposto per strato misurato nel motore (2-3 giorni)

- `src/core/peer_experts.{hpp,cpp}`: `PeerExperts::ep_open(Verify&)` (solo con `STRATA_EP=1`): puntatori device
  (sul device del server) delle pagine mappate `m_seq_`, `m_ids_`, `m_x_`, `m_ymiss_` della padrona; una parola mappata
  nuova `m_flagS_` (Portable); copia su device delle due tabelle di residenza (48 × 512 int32 ciascuna); un grafo per
  (T, G) con, per strato e gruppo: `wait_flag_ge(m_seq_, ring)` → copia id + x → `resident_plan` nella variante a due
  tabelle (mio = sul server e non sulla padrona) → `hadamard_rows` + `quantize_q8_1_rows` → `native_expert_grouped`
  con conteggi device → `scatter_rows_f32` nelle righe mappate `out` → kernel che alza `m_flagS_ = ring` dopo
  `__threadfence_system`. `ep_launch(T, G)` lancia il grafo; `ep_wait()` a fine finestra. Attesa limitata (200 ms come
  `ep_bench`) con parola d'errore, così un errore della padrona non lascia un kernel a girare fino al watchdog.
- `src/core/verify.cpp` `post()`: con EP acceso, `wait_flag_ge(m_flagS_, ring)` prima della copia delle righe
  (`2328-2340`; le varianti fuse restano spente sotto EP in questa tappa). `run()`: lancio del server prima del grafo
  della finestra, dopo l'azzeramento dei flag (`2929`).
- `src/core/expert_source.cpp`: con EP, `kind == 2` resta (il pool non azzera quelle righe: `5165`, come
  `launched_direct`), `launch`/`finish` non fanno lavoro GPU (contano solo).
- `src/program/generate.cpp`: `STRATA_EP` letta una volta; la riga `strata decode peer:` stampa anche i µs esposti a
  strato = (`GPU-reach wait` − 11,7 ms di `Vpeer`) / 48 e la mediana dell'attesa a `m_flagS_` (stamp sul device come
  `wait@rows` del banco).
- **Criterio**: braccio `FEP1` (§5). Testo identico a `Vpeer` a cache fissa (stesso prompt, `STRATA_EP=0` contro `=1`,
  stessa build). Esposto per strato nel motore ≤ **125 µs** e `ms/window` ≤ 22 → 115 t/s raggiungibili, si continua;
  > 150 µs con `wait@rows` p50 vicino all'esposto → la tappa 1b è il kernel unico (6-10 giorni) prima di tutto il
  resto; > 200 µs anche nel kernel unico → **stop**, `procede = false`.

### Tappa 2 — il piano sul device anche in EP: l'host pianifica solo i miss (1-2 giorni)

- `verify_kernels.cu`: `resident_plan` a due tabelle (padrona + server) e `skip_` quando il gruppo non ha miss;
  `verify.cpp`: `device_plan_` acceso con EP quando le tabelle sono allineate (oggi solo con tutto residente); il pool
  salta il piano dei gruppi senza miss (`wait_flag_ge_or(m_flag_, ring, skip_)` già c'è, `2319`).
- **Criterio**: `per-layer host [plan]` ≤ 0,1 ms a finestra; `ms/window` −0,5/−1 rispetto alla tappa 1 (S); testo
  identico.

### Tappa 3 — i miss letti dal server dalla RAM bloccata (3-4 giorni)

- Prima la misura: `tools/ep_bench.cu` con un blob «host» (1,7 MB in pagine bloccate, letto con `__ldcv` a 16 byte da
  64+ blocchi della 2080 Ti): GB/s e µs a blob. Se < 6 GB/s (> 0,3 ms a blob) la tappa si ferma qui e lo dice.
- Poi: il piano a due tabelle manda al server anche le voci con blob nella fetta bloccata dell'arena (il nucleo RAM:
  `STRATA_RAM_CORE` riacceso, i blob caldi non residenti nella fetta di 8 GiB; puntatore host nel piano, come
  `kind == 1` fa con `d.src->pinned`); il server li calcola leggendo il blob in streaming dalle pagine mappate (variante
  di `native_expert_grouped` con sorgente mappata, o copia nel suo scratch prima del kernel: la misura decide). Il CPU
  tiene solo i miss fuori dalla fetta.
- **Criterio**: `CPU experts` a strato-finestra → < 0,3 voci; attese CPU a finestra (`jobs` + `CPU`) → < 0,5 ms;
  `ms/window` −1/−2 (S); scrittura stabile con 6 core occupati (`carico_cpu.py`: oggi −13%, M).

### Tappa 4 — il prompt a stadi con la scrittura EP (4-6 giorni)

- `generate.cpp`: con EP, `--layer-split N --peer-device D` ammesso (`4174`); il verificatore di CUDA0 costruito su
  0-47 (densi di tutti gli strati su CUDA0, come già fa `--peer-device` da solo), lo stadio 1 tiene `Prefill` e stato
  K/V per 20-47 ma **non** verifica in scrittura; lo specchio del K/V (`conversation_snapshot.cpp`) copia le posizioni
  nuove 2080 Ti → CUDA0 dopo ogni prompt e CUDA0 → 2080 Ti prima del prompt seguente; `--prompt-cache-file` riammesso
  con EP quando lo specchio è provato (le immagini radice si fanno dallo stato di CUDA0).
- **Criterio**: lettura ≥ 90% di `Vrif1` a 32K e 128K (≥ 1660 / ≥ 1990 t/s) e pari a 4K; testo identico a cache fissa
  con e senza specchio; a più turni il prefisso riusato (tempo del secondo turno come oggi, M in `corse_turni`).

### Tappa 5 — il banco completo e le leve fini (2-3 giorni)

- Caldi alla padrona (`STRATA_PEER_HOT=0`, già c'è) e righe bf16 al ritorno (cambia i bit: separato, opt-in dentro EP),
  solo se la tappa 1 ha lasciato > 20 µs sul tavolo. Poi il banco completo 4K/32K/128K, 3 corse, contro `Vrif1`.
- **Criterio**: scrittura ≥ **115 t/s** sul banco (oggi ~104, M), lettura entro il −10% sui tre bersagli, testo
  identico a `Vpeer` (o a `--peer-device` + bf16 quando acceso), nessun `strata.exe` rimasto acceso.

Giorni in tutto (S): 12-18 con il server a grafo; +6-10 se serve il kernel unico.

## 5. Come si prova

- Build: `build_lock_k.py <worktree> --target strata --jobs 6` (in background, attende il banco e prende il lucchetto),
  copia in `corse_2080ti\20261004_routed_keep\build-up029pila-<sha corto>\` (strata.exe, strata_kq.dll,
  strata-vision.exe copiato, cartella `source`, `build.json` con la nota «compilato sì, installato no, provato …»).
- Coda: in `coda_notte.py` una costante `BEPn = … / 'build-up029pila-<sha>'` e una fase `FEPn` con bracci `_lungo`
  (`banco_lungo.py`, 3-12 minuti di caricamento + 90 s), configurazione `PEER` (`--peer-device`, riserva 3000,
  `STRATA_ARENA_PIN_GIB=8`, catena e prefisso tolti) più `STRATA_EP`: per ogni tappa una coppia `Xep{n}` (`STRATA_EP=1`)
  e `Xpeer{n}` (`STRATA_EP=0`, stessa build: il riferimento e la verifica del testo), più `Xrif{n}` (serie, divisione
  20) quando serve il confronto di lettura. `python -u coda_notte.py FEPn > fepn.out`, in background; esiti in
  `corse/<nome>/lungo.json`, `sommario_lungo.json`, `engine.log`. Niente visione sulla 3060, `--ple-io direct` lo
  mette la coda. Dopo ogni braccio: nessun `strata.exe` acceso.
- Se un motore non parte o si pianta: `engine.log`; tre tentativi sulla stessa cosa, poi ci si ferma e si dice.

## 6. Rischi

- **Watchdog WDDM**: il server gira su una parola mappata per tutta la finestra; le attese della padrona lo fanno già
  (`wait_flag_ge`), ma un errore a metà finestra lascerebbe il server appeso: attesa limitata + parola d'errore (tappa
  1), e `ep_wait()` sempre prima di chiudere o cambiare grafo.
- **Allineamento dei piani**: la CPU (`kind`) e il server (piano sul device) devono dire la stessa cosa su ogni voce: le
  due tabelle cambiano solo fra le finestre (`apply_pending`), e la tappa 1 verifica il testo contro `Vpeer`.
- **VRAM della 3080 Ti**: 700 MiB liberi in `Vpeer` (sotto la soglia di guardia): la tappa 4 aggiunge il K/V di 20-47 a
  contesto pieno (già previsto dal verificatore su 0-47) e può chiedere `--vram-reserve-mib` più alta e meno slot.
- **Memoria bloccabile**: 8 GiB di arena + i buffer mappati; la tappa 3 non ne chiede altra.
- **Il banco sottostima la padrona** (`PIPELINE3` §7): i 245 µs misurati per strato coprono il server più di quanto
  il banco mostri, quindi l'esposto nel motore può essere più basso del `persist`; mai più alto per quella ragione.
