# PIPELINE3_PIANO: il parallelo vero, tappe di esecuzione

Ramo `lab/upsync-v0.2.9`, 2/11 sera, **seconda stesura** dopo il banco `ep_bench` (girato alle 17:44) e la revisione del
piano (sei bloccanti, nove correzioni, tutti verificati sul codice il 2/11 sera: le righe citate sono quelle lette). Segue
`PIPELINE3.md` (conti) e `PIPELINE2.md` §2a (salto fra le schede). Macchina: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, 0 = 3080 Ti
12 GB sm_86 (Oculink x4), 1 = 3060 (schermo, mai usata), 2 = 2080 Ti 22 GB sm_75 (x16); niente P2P. Modello
Qwen3.8-Flash-Next IQ3_XXS, 48 strati, k = 10.

Convenzione: **M** = misurato, **S** = stimato da misure. Tutto il lavoro sta dietro **`STRATA_EP=1`**, spenta di
default: spenta, nessuna riga di esecuzione cambia e il testo è identico (verifica a cache fissa in ogni tappa).

## 0. Conclusione

- **Tappa 0 fatta (2/11 sera, M): span GPU del server a strato p50 ~300 µs, ≥ 250 → `procede = false`.** Esito e
  numeri sotto la tappa 0 (§4) e in `RISULTATI.md` della cartella delle corse; il resto del piano resta come scritto,
  fermo finché una misura dentro lo span (copie e scatter contro kernel esperti) non lo riapre.
- **Si procede, ma la prima tappa è una misura di un'ora, non il server.** Il banco (§1, M) dice che a T=3 il server
  persistente espone **113 µs** a strato e quello guidato dall'host **118**: togliere l'host dal percorso, nel banco, vale
  **4,7 µs**. Nel motore (`Vpeer`, M) l'host costa invece 446-542 µs a strato. Quindi il banco **non rappresenta il
  motore**, e la tesi di `PIPELINE3` §0/§2 («il resto è latenza di sottomissione e polling») non è provata: tolti i
  ~135 µs che il banco spiega e i 3,3-5,7 ms di CPU vera a finestra (S), restano **200-285 µs a strato** (9,6-13,7 ms a
  finestra, S) che possono essere i kernel veri del server (`hadamard`, `quantize`, 3-5 kernel esperti, `scatter_rows`).
  Se è così, un server a grafo non li toglie e la finestra resta sopra i 30 ms. **La tappa 0 lo misura con due stamp
  `__globaltimer` intorno ai kernel di `PeerExperts::launch`**: sotto 150 µs di span GPU a strato si scrive il server,
  fra 150 e 250 solo con le leve di §1, sopra 250 ci si ferma (`procede = false`).
- **Il join dell'adattamento non è nel conto e da solo annulla il guadagno**: 3,88-6,89 ms a finestra in `Vpeer` (M,
  8,5-16,7 swap), 0,00 in `Vrif1` (M). Sul percorso del peer le due vie non bloccanti sono rifiutate (overlap
  `generate.cpp:11828`, asincrono `9371`). Va tolto o sovrapposto **prima** di qualunque conto (tappa 0b, §3b).
- Finestra rifatta (§1, S da M): 15,4 ms di pavimento + 48 × esposto, con l'esposto nel motore **125-140 µs** (S: il
  `persist` del banco più i nodi del grafo, **senza** sottrarre i «40-50 µs di copertura» che non esistono: 11,7 ms non è
  il lavoro della padrona, §1). **Senza join 21,5-22 ms**, cioè 109-121 t/s secondo i token a finestra (2,4-2,6, M): a
  cavallo dei 115. **Con il join 25,4-29 ms**: nessun guadagno. Sotto i 115 si arriva solo con join fuori più una fra
  caldi alla padrona (98 µs nel banco, M), righe bf16 (90, M: cambia i bit) e miss sul server (−1/−2 ms, S).
- La lettura dei prompt lunghi (vincolo: entro il −10% di 900 / 1850 / 2210 t/s a 4K/32K/128K, M) resta a **divisione
  a strati nel prompt ed expert-parallel nella scrittura** (§3a), ma il prompt **non gira «come oggi»**: `host_res` è
  una tabella sola, `help_pp` non tiene esperti e vale solo su prompt a un chunk, e la residenza in EP non è quella di
  `Vrif1`. La tappa 4 è riscritta e preceduta da una corsa in serie con le cache ridotte agli slot che avranno in EP.
- Cache elastica: fissa con il peer (`generate.cpp:3699`). Nucleo RAM: fermo (non fa quello che la vecchia tappa 3
  gli chiedeva, §3f). Catena (`--pipeline-windows`): spenta (vale 0, M; rifiutata col peer, `5205-5221`).

## 1. Il banco `ep_bench` (M, `D:\ds4_work\strata\ep_bench\ep_bench_run1.txt`, 2/11 17:44, 40 finestre × 48 strati)

| configurazione | floor | all | persist | host | host-ns | exp.persist | exp.host | wait@rows p50/p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T=2 come Vpeer, righe fp32 | 89,9 | 108,0 | 181,0 | 186,8 | 186,4 | **91,0** | 96,9 | 64,5 / 79,9 |
| **T=3 come Vpeer, righe fp32** | 93,4 | 120,9 | **206,8** | **211,5** | 211,4 | **113,4** | **118,1** | 78,8 / 89,1 |
| T=4 come Vpeer, righe fp32 | 98,2 | 134,1 | 228,1 | 241,2 | 240,9 | 129,9 | 143,0 | 86,0 / 91,1 |
| T=3 caldi alla padrona | 99,8 | 123,3 | 198,1 | 200,6 | 200,3 | **98,3** | 100,8 | 67,6 / 78,8 |
| T=3 righe bf16 | 93,2 | 120,4 | 183,4 | 191,5 | 191,8 | 90,2 | 98,3 | 70,7 / 75,8 |
| T=3 righe sommate per token | 93,4 | 120,9 | 171,4 | 182,0 | 183,6 | 78,0 | 88,6 | 67,6 / 72,7 |
| T=3 soli flag (niente esperti) | 76,7 | 76,7 | 98,7 | 129,6 | 129,2 | 22,0 | 53,0 | 17,4 / 19,5 |

µs a strato, mediane; nessun timeout (`err flag 0`). Che cosa dice e che cosa non dice:

1. **Persist ≈ host.** Nel banco il server guidato dall'host (spin sul flag, 2 copie H2D, 4 lanci, spin su
   `cudaStreamQuery`, cambio di scheda) costa 4,7 µs più del kernel persistente a T=3 (0,3 a T=2, 13 a T=4). Il kernel
   unico sposta molto solo con i soli flag (22 contro 53 µs): con i dati 113 contro 118. **I 6-10 giorni del kernel
   unico non si decidono sull'esposto: si decidono sulla tappa 0.** La riga «100-150 → tappa 1 poi 1b» della prima
   stesura è tolta.
2. **Il banco non rappresenta il motore.** Il controllo di coerenza della prima stesura chiedeva `exp.host` a 400-500 µs
   (i 430-530 di `Vpeer`): ne fa 118. `Vpeer` paga 21,4-26,0 ms di host a finestra a 4K/32K (M: `per-layer host`
   21,40-26,02), cioè **446-542 µs a strato**. Il banco spiega ~135 µs di span (211,5 − 76,7 del pavimento a soli flag);
   la CPU vera (il run del pool per i miss, 0,8-1,9 voci a strato-finestra) vale 3,3-5,7 ms a finestra (S, dai `CPU`
   1,96-6,41 di `Vrif1`). Restano **200-285 µs a strato non spiegati**, 9,6-13,7 ms a finestra (S). Il banco calcola gli
   esperti come letture di banda di 1,7 MB (`expert_items`): se i 200-285 µs sono i kernel veri del server
   (`hadamard_rows` + `quantize_q8_1_rows` + `native_expert_grouped` in 3-5 lanci + `scatter_rows_kernel` a 128 thread
   per riga, `peer_experts.cpp:318-330`) con i loro vuoti fra lancio e lancio sotto WDDM, **il server a grafo non li
   toglie**. Lo misura la tappa 0.
3. **Il banco non verifica nessun dato.** `k_server` accumula un `acc` che finisce in `sink` solo se vale 0xFFFFFFFF
   (`ep_bench.cu:205`); nessun numero di sequenza in x né nelle righe; `err flag 0` dice solo che nessuna attesa ha
   superato i 200 ms (`WAIT_NS`, riga 48). **L'ordine flag/payload fra due schede senza P2P non è provato.** La tappa 1
   mette un tag finestra+strato nell'ultima parola di x e di ogni riga, controllato da chi legge (§3e).
4. **x nel banco è q8 da 8,6 KB; nel motore `m_x_` è fp32 da 30,7 KB a T=3** (`verify.cpp:878`: `T * N * 4`), scritto
   a 4 byte da `doorbell_publish_res_kernel`: lato server ~5 µs in più a 4,3 GB/s (S). Si può quantizzare prima di
   pubblicare solo cambiando il doorbell: non in tappa 1.

**Conversione banco → motore (corretta).** `GPU-reach wait` = 11,7 ms (M) è `ms_wait`, cioè **solo l'attesa dell'host
su `seq`** (`verify.cpp:3332`: da prima dello spin a quando lo strato suona). Fra due attese l'host serve lo strato
(`ms_pool`), e in quel tempo il grafo fa la quantizzazione, la copia del piano e gli hit locali della padrona: **stanno
fuori dagli 11,7 ms**. Quindi i «−40/−50 µs di copertura» della prima stesura non esistono, non si sottraggono i ~40 µs
di hit di `PIPELINE3` §3a, e «mai più alto per quella ragione» è tolto. Esposto atteso nel motore = `exp.persist` 113
(M) + nodi del grafo 30-50 (S, §3c) − quel che la padrona copre davvero (non misurato: 0 nel conto) = **125-140 µs**
(S), con `wait@rows p50` 79 µs che dice quanto del server resta scoperto nel banco.

Finestra = 15,4 ms di pavimento (11,7 M + 1,5 miss CPU in più, M: +0,4-1,0 voci a strato-finestra + 1,7 bozze, M +
0,5 emissione e stadio, M) + 48 × esposto + join:

| esposto a strato (µs) | finestra senza join (ms) | t/s a 2,4-2,6 token a finestra | con join 3,9-6,9 (ms) | che cosa serve |
|---:|---:|---:|---:|---|
| 100 (caldi alla padrona: 98 nel banco, M) | 20,2 | 119-129 | 24,1-27,1 | tappa 0b riuscita + `STRATA_PEER_HOT=0` |
| **125-140 (S, dal banco)** | **21,4-22,1** | **109-121** | 25,3-29,0 | tappa 0b riuscita |
| 150 | 22,6 | 106-115 | 26,5-29,5 | + miss sul server (−1/−2 ms, S) |
| 200 | 25,0 | 96-104 | ≥ 28,9 | niente: oggi fa 25,3 (M) |
| 250 o più (span GPU del server, tappa 0) | ≥ 27,4 | ≤ 95 | — | **stop** |

**115 t/s ⇔ finestra ≤ 20,9-22,6 ms** secondo i token a finestra: al centro della stima si sta a cavallo, con il join
dentro si sta sotto a tutte le esposizioni.

## 2. Il codice reale (letto il 2/11): che cosa c'è già, che cosa manca

Percorso di una finestra di scrittura, padrona = CUDA0 (`verify.cpp`, `expert_source.cpp`, `peer_experts.cpp`):

1. **Il grafo della padrona pubblica già x e gli id sul device**: `pre(l)` lancia `doorbell_publish_res`
   (`elementwise.cu:329`): copia nelle pagine mappate `m_ids_` (T×k int) e `m_x_` (T×2560 fp32), poi
   `__threadfence_system` e `*seq = *seq + 1` (`elementwise.cu:211`; ring `(l-lb)*G+grp+1`, azzerato a ogni finestra,
   `verify.cpp:2926`). Il filo del pool aspetta `seq` (`verify.cpp:3242`), pianifica (`expert_source.cpp:4905-5047`:
   `kind[i]` = 0 VRAM padrona, 1 PCIe, 2 peer, −1 CPU) contando `d.usage` per l'adattamento (`4902`), pubblica il piano
   (flag A), e la padrona lo copia con `wait_copy_i32_from_mapped` (`verify.cpp:2291`).
2. **Il peer di oggi è guidato dall'host a ogni strato**: subito dopo il piano il pool chiama `PeerExperts::launch`
   (`peer_experts.cpp:260`: `On(device_)`, 2 `cudaMemcpyAsync`, `hadamard_rows`, `quantize_q8_1_rows`,
   `native_expert_grouped` con griglia esatta `groups, rows` (`322`), `scatter_rows_f32` verso le righe mappate `out`),
   fa i suoi miss, e in `expert_source.cpp:5279` chiama `finish`, che gira su `cudaStreamQuery` (`peer_experts.cpp:345`);
   solo dopo il pool torna e `verify.cpp:3330` alza `*flag`. Il grafo copia le righe con `copy_rows_from_mapped`
   (`verify.cpp:2337`, salta le righe hit) e fa il combine. Il pool **non azzera** le righe `kind == 2` (`5165`): se
   nessuno le scrive, nel combine entra la riga della finestra prima.
3. **Il piano sul device esiste già per la padrona**: `resident_plan` (`verify_kernels.cu:1519`) e la variante parallela
   `resident_plan_par_kernel` (`1434`), che però mette `s_bad` se **una sola** voce non è residente (`1448`) e allora
   `*skip = 0` e nessun piano; quando pianifica mette `counts[1] = n` (`1423`, `1488`: tutte le voci sue). Vuole la
   tabella `slot_off` (`1519`). `native_expert_grouped` prende i conteggi sul device (`n_groups` puntatore, `cap_groups`
   per la griglia).
4. **Le attese con flag in memoria mappata sono il pane del grafo**: `wait_flag_ge_kernel` (`verify_kernels.cu:1224`)
   gira **senza limite** su una parola mappata; `wait_handoff_kernel` (`1245`) è la stessa attesa **con limite e parola
   d'errore** (`__globaltimer`), già nel motore per il passaggio fra stadi. `handoff_publish_kernel` (`1259`) è la forma
   del motore per «dati poi flag» fra schede: `__threadfence_system` per thread, `__syncthreads`, contatore atomico per
   blocco, l'ultimo blocco alza il flag. `scatter_rows_kernel` (`elementwise.cu:263`) invece fa store semplici senza
   fence: oggi la visibilità gliela dà la fine dello stream in `finish`.
5. **Tenuta (#267)**: l'host aspetta uno strato al massimo **20 s** (`verify.cpp:3261`) e poi `release_gpu_waits`
   (`589`), che alza a `UINT32_MAX` solo `h_flag_`, `h_flagA_`, `h_flagB_`, `hflag_in_h_` (`595`) e drena solo `cs_` e
   `copy_` (`601`); se non drena, `RELEASE-NOT-DRAINED` (`605`) e il motore va riavviato (`3058`). Il seq del doorbell
   non si può «alzare oltre tutto»: fa `+1` e torna a 0.
6. **Flush WDDM**: il filo dell'host fa `cudaStreamQuery(cs_)` ogni 2 ms mentre aspetta (`3246-3249`); sotto WDDM il
   lavoro di uno stream resta nel buffer del driver finché qualcosa lo spinge (#2188, commento a `generate.cpp:11848`:
   «copies for layer 45 never landed»). Nel banco il server parte perché c'è un `cudaStreamQuery` subito dopo il lancio
   (`ep_bench.cu:371`), per **un** kernel.
7. **Adattamento con il peer**: `adapt` (`peer_experts.cpp:363`) segna la vittima non residente **subito** (`401`) e
   accoda la copia del nuovo blob nello stesso slot su `refill_` (`398`); `apply_pending` (`408`) aspetta l'evento e
   aggiorna `res_`. Nel motore `apply_pending(wait)` chiama prima quello del peer (`generate.cpp:11897`) e poi scrive
   `host_res` (`11910`). Overlap: «not with --peer-device» (`11828`); con `device_plan_` acceso `set_adapt_gate` rifiuta
   (`11847`); asincrono: «not with --peer-device» (`9371`).
8. **Residenza**: `host_res` è una tabella sola, «under a split host_res holds slot numbers of whichever cache owns the
   layer» (`generate.cpp:12215`); la usa anche lo stadio 1 del prompt (`st.sp.init(..., host_res.data(), ...)`,
   `10170`). Cambia all'adattamento, al prestito del prompt e al rientro (`19619-19631`: «refill the lent slots … give
   them back»; `Vpeer` presta 2119 dei 2561 slot, M), a `res_put`. Il peer ha la sua `res_` privata. Pin dell'arena:
   un prefisso (`ArenaExpertSource::pinned`, `expert_source.cpp:6423`: `blob_offset + blob_bytes <= pinned_bytes_`): con
   `STRATA_ARENA_PIN_GIB=8` «10 slices pinned (7 GiB)» (M, log `Vpeer`), cioè gli strati 0-9 e nulla di 10-47.
   `STRATA_PIN_BY_NEED` (`6901-6918`) pinna per bisogno ma è «not with a layer split or a peer GPU» (`6915`).
9. **Il prompt col peer** (`prefill.cpp:2271`, `Prefill::set_peer`, chiamato solo su `sp`: `generate.cpp:10263`):
   righe compatte via host, una quota di esperti in streaming dalla RAM. `help_pp` (`2170`) è «stream-only: it holds no
   experts», esclude `m.pp` (`2208`) e serve solo i prompt a **un chunk** (`3061-3082`). `--peer-device` rifiuta
   `--layer-split` (`4174`) e `--prompt-cache-file` (`3760`).
10. **K/V**: la configurazione gira con `--kv-resident 32768`: «32768 of 262144 cells per QSA layer in VRAM, the K/V in
    3,09 GiB of pinned RAM» (M, log `Vpeer`); le celle oltre le 32768 stanno nella copia host. Le copie dello snapshot
    sulla scheda dello stato (`conversation_snapshot.cpp`, d14aa61e) sono la base dello specchio.
11. **Griglie e nodi**: i ~2,4 µs a nodo dipendente sono misurati su gfx1201 (`verify.cpp:396-397`), non su Turing. Con
    `STRATA_EXP_MODE=8` (nella configurazione `TUTTE` della coda) le varianti lanciano **una riga di blocchi per ogni
    gruppo possibile** (`iq_kernels.cu:6332-6334`: V100, 0 gruppi 11,3 contro 6,5 µs, 1 gruppo 29,6 contro 21,3), e un
    grafo con i conteggi sul device deve lanciare `cap, cap` (30 righe contro le ~13 di oggi).

Che cosa manca, in ordine: (a) la misura dello span GPU del server di oggi (tappa 0); (b) il join fuori dal percorso
(0b); (c) un server sulla 2080 Ti che aspetti `m_seq_` dal device, pianifichi sul device e scriva righe + flag nella forma
di `handoff_publish_kernel`, con limite, abort e flush; (d) il grafo della padrona che aspetti anche quel flag; (e) il
pool che con `STRATA_EP` non lanci e non aspetti il peer ma conti le voci; (f) le tabelle di residenza sul device del
server, rinfrescate dentro `ep_launch` da `host_res` e `res_` vive; (g) i miss dalla RAM bloccata per bisogno; (h) il
prompt a stadi con la scrittura EP e due tabelle di residenza.

## 3. Decisioni

### 3a. Vincolo lettura: divisione a strati nel prompt, expert-parallel nella scrittura (riscritta)

Perché `Vpeer` perde il 33-42% (M): a 4K fa 892 t/s contro 860-934 (pari), a 32K/128K crolla. La prima stesura lo
attribuiva all'attenzione di 48 strati su una scheda sola: è dedotto da `Vpeer`, **nessun dato lo prova su questa
disposizione** (in `Vpeer` la 3080 Ti ha 2561 slot e ne presta 2119 al prompt: anche gli esperti sono quasi tutti fuori).
Resta la decisione (i densi e il K/V di 20-47 anche sulla 2080 Ti nel prompt), ma il prompt **non gira come oggi**:

- (a) **`host_res` non basta**: con EP gli strati 20-47 di `host_res` devono restare slot della padrona (il
  verificatore legge lì), mentre i blob dello stadio 1 stanno in `PeerExperts::cache_` con la sua `res_`. Lo stadio 1 del
  prompt (`st.sp.init(..., host_res.data())`) leggerebbe slot sbagliati: serve una **seconda tabella di residenza per il
  prompt** (quella del server), passata allo stadio 1 al posto di `host_res`.
- (b) **`help_pp` non è il peer del prompt**: non tiene esperti e vale a un chunk. Gli esperti del chunk dello stadio 0
  chiesti alla 2080 Ti passano da `set_peer` di `sp` (il percorso di `Vpeer`, via host, righe compatte), **mentre la
  2080 Ti fa lo stadio 1**: in concorrenza nei chunk pipelinati.
- (c) **La residenza non è quella di `Vrif1`**: 3080 Ti 5477 slot sui soli strati 0-19 (M) contro ~1070 in EP (S:
  2561 − 1491 di densi e K/V dei 28 strati nuovi), con il prestito del prompt che ne prende la maggioranza (2119 su
  2561 in `Vpeer`, M); 2080 Ti 9090 sui 20-47 (M) contro ~6400 in EP (S: 10972 − ~1750 MiB di densi − K/V residente −
  prestito). Lo stadio 0 chiederebbe la maggioranza dei suoi esperti alla 2080 Ti.

Quindi **prima di scrivere la tappa 4 si misura** (tappa 4a): una corsa in serie (`--layer-split 20`, niente peer) con
le due cache **ridotte** agli slot che avranno in EP (`--expert-cache` e `--expert-cache-device1` forzati a ~1070 /
~6400) sui tre bersagli: dice quanto della lettura si perde per i soli esperti, senza scrivere nulla. Se la perdita è
già oltre il −10%, i densi su entrambe le schede non bastano e il vincolo si tiene solo con la doppia modalità
(`PIPELINE2` §9c: 1-2,5 s a cambio) o con meno K/V residente.

Specchio del K/V (tappa 4b): la configurazione ha `--kv-resident 32768`, quindi **non** si copiano 1,8 GB VRAM → VRAM a
128K: oltre le 32768 celle la verità sta nella copia host (3,09 GiB bloccati, M). Lo specchio stabilisce **di chi è la
copia host** con due stati (prompt: 2080 Ti per 20-47; scrittura: 3080 Ti per tutto) e copia solo le celle residenti
nuove: per il prompt 7 strati QSA × 32768 celle al massimo (S: ≤ 0,45 GB, ~70 ms sul percorso x16 → 6,7 GB/s), per la
risposta i 256 token generati. Vanno specchiati anche `idx_pooled`, `idx_tail`, `idx_block_pos` e le cache di prefisso in
RAM (`--prompt-cache` della coda, conversation cache a stadi, `generate.cpp:10764`), non solo il prefisso su disco. I
«−1,8 GB sulla 2080 Ti» e i «0,3 s a 128K» della prima stesura sono tolti.

Scartate: il prompt tutto EP (non dimostrato che non siano gli esperti, e la residenza EP della 3080 Ti è troppo
piccola); i pesi ricaricati restano la riserva se 4a fallisce.

### 3b. Convivenza

| cosa | oggi con `--peer-device` | con `STRATA_EP=1` |
|---|---|---|
| cache elastica | forzata fissa (`3699`) | uguale: fissa |
| nucleo RAM (`STRATA_RAM_CORE`) | spento nella configurazione `PEER` | spento: è un nucleo fisso di slot VRAM la cui copia RAM viene **restituita** (`docs/RAM_CORE.md`), il contrario di quel che la tappa 3 vuole |
| adattamento: join | **3,88-6,89 ms a finestra** (M, 8,5-16,7 swap); `Vrif1` 0,00 (M) con 5,7-15,3 swap | **tappa 0b, tre scelte in ordine**: (1) flush della coda `refill_` del peer subito dopo `adapt` + `apply_pending` a polling fra gli strati (S: il join di `Vpeer` è la copia WDDM non spinta fino a `cudaEventSynchronize`, #2188; `Vrif1` non lo paga perché il filo degli stadi fa flush ogni 2 ms); (2) meno swap (`--adapt-swaps 4`, `--adapt-every` più lungo) pagando miss CPU; (3) overlap ammesso sotto EP con tabelle congelate al flip, e allora **niente piano sul device** (`set_adapt_gate` lo rifiuta, `11847`): la tappa 2 sacrificata |
| adattamento: tabelle | `adapt` + `apply_pending` del peer (`11897`) | uguale, più l'invio delle righe cambiate **dentro `ep_launch`** da copia ombra contro `host_res` e `res_` vive (§3f) |
| catena (`--pipeline-windows`) | rifiutata (`5205-5221`) | spenta: vale 0 (M, `Eb0*`) |
| piano sul device (`device_plan_`) | solo con tutto residente sulla padrona | tappa 2, riscritta (§3f): solo se 0b sceglie (1) o (2) |
| `STRATA_ROUTE_TAIL_SKIP` | spento col peer | spento |
| bozze (drafter MTP) | sulla 3080 Ti (`Vpeer`: 1,2-2,1 ms, M) | uguale |
| limite d'attesa | host 20 s (`3261`), GPU senza limite | host 20 s; **le attese del server a 20 s** con parola di abort (§3d), non i 200 ms del banco |

### 3c. Forma del server: grafo in tappa 1, kernel unico deciso dalla tappa 0 (non dall'esposto)

Il «kernel persistente» di `PIPELINE3` §3a fonde `hadamard` + `quantize` + `native_expert_grouped` + `scatter` in un
lancio per finestra: 6-10 giorni. Il banco dice che sposterebbe poco rispetto al grafo **se** i kernel veri costano come
le letture di banda (persist ≈ host). La tappa 1 mette un **grafo per finestra** di 48 × 11-13 nodi (attesa di `m_seq_`,
copia id, copia x, piano, hadamard se `had_gu`, quantize, 3-5 kernel esperti, scatter + flag, conteggio), lanciato una
volta a finestra prima del grafo della padrona. Costo dei nodi **ricalcolato**: i 2,4 µs sono di gfx1201; su Turing con
`STRATA_EXP_MODE=8` le varianti lanciano una riga di blocchi per gruppo possibile e il grafo deve usare `cap, cap`
(`iq_kernels.cu:6332`: +4,8 µs a vuoto, +8,3 con un gruppo, V100): **+30-50 µs a strato** (S), non 15-20. Il criterio
«entro 30 µs dal persist» è tolto: la tappa 1 confronta il suo esposto con lo **span GPU della tappa 0**, non col banco.
Il kernel unico si fa solo se la tappa 0 misura uno span fra 150 e 250 µs **e** la tappa 1 non lo abbassa sotto 150.

### 3d. Tenuta su finestra: limite, abort, flush (nuova; bloccante 3 e 4)

- **Nessuna attesa senza limite sul server.** Le attese della 2080 Ti usano la forma di `wait_handoff_kernel`
  (`__globaltimer`, parola d'errore mappata) con **20 s**, come il limite dell'host: fra due doorbell ci sono miss CPU,
  `prefetch` da disco con `--ple-io direct`, `gather_batch` del PLE a k == 0 (`verify.cpp:3322`) e paging a 700 MiB
  liberi; i 200 ms del banco sarebbero falsi positivi che uccidono la finestra e dopo `released_` il motore va riavviato.
- **Parola di abort** `m_abortS_` (mappata, Portable) letta da **ogni** attesa del server; `release_gpu_waits` la alza
  insieme a `h_flagS_` (aggiunta alla lista di `verify.cpp:595`) e **drena anche lo stream del server** (`On(device
  del server)`, `cudaStreamQuery` fino al drain, dentro i 5 s); `release_live_verifiers` (`538`) passa dal verificatore
  che conosce il suo `PeerExperts`. La padrona, se il server si ferma, dopo 20 s esce con `verify: timed out` e la
  release la sblocca davvero: niente `RELEASE-NOT-DRAINED` con la GPU persa.
- **Flush del server a ogni strato dal filo del pool**: sotto WDDM il grafo del server non parte finché qualcosa lo
  spinge, e un `cudaStreamQuery` solo dopo il lancio è provato per un kernel (`ep_bench.cu:371`), non per ~530 nodi;
  `PIPELINE2` Tappa 1 ha misurato sulla 2080 Ti un buco di ~2,0 ms in ~90 finestre su ~110 dentro il grafo (M il buco,
  S la causa). Oggi quel buco copre un'attesa; in EP è sul percorso critico. Il filo del pool, quando serve lo strato,
  fa `cudaStreamQuery(stream del server)` **senza cambio di scheda** (`STRATA_P2_FLUSH_NOSWITCH` provato, 5,15 M poll;
  se il driver rifiuta, con `On`): è una riga nel ciclo di `run` accanto al flush di `cs_`.

### 3e. Ordine di memoria del ritorno e tag di finestra (nuova; correzioni 2 e 3)

Il piano precedente metteva `__threadfence_system` in un kernel separato dopo `scatter_rows_f32`: una fence di un altro
kernel **non** è la forma che il motore usa, e la 3080 Ti potrebbe vedere il flag prima dell'ultima riga, con testo
diverso solo a volte. Il server scrive le righe con un **`scatter_rows_flag`** nella forma di `handoff_publish_kernel`:
store (`__stwt` a 16 byte), `__threadfence_system` per thread, `__syncthreads`, contatore atomico per blocco, l'ultimo
blocco azzera il contatore, fence, alza `m_flagS_ = ring`, fence. Il banco `k_server` fa lo stesso (`ep_bench.cu:198-202`).

Tag: la padrona scrive `ring` nell'ultima parola di `m_x_` (una parola fp32 in più per riga di token, fuori dal
conto) e il server la confronta con quella attesa prima di usare x; il server scrive `ring` nell'ultima parola di ogni
riga di ritorno e il `copy_rows_from_mapped` della padrona (variante EP) lo confronta: una discordanza alza la parola
d'errore e la finestra fallisce con un messaggio, invece di produrre un testo diverso. Il tag si prova prima nel banco
(`ep_bench.cu`: `k_wait_copy` e `k_server`), che oggi non verifica nulla.

### 3f. Tabelle sul device, piano del server e divergenza (nuova; correzioni 4, 5, 6)

- **Invio delle tabelle dentro `ep_launch`**, non «agganciato ad `adapt`»: `host_res` cambia anche al prestito del
  prompt e al rientro (`19619`), a `res_put`, ad `apply_pending`; `res_` del peer cambia in `adapt` (eviction
  immediata, `401`) mentre `refill_` sta già scrivendo nello slot (`398`): una tabella in ritardo legge un blob a metà.
  `ep_launch` confronta una **copia ombra** con `host_res` e `res_` vive e manda le righe diverse su `stream_` prima del
  grafo (≤ 48 righe da 2 KB: < 50 µs, S). La fonte è `host_res`, non `d_res`, che può restare indietro (`11926-11929`).
  `ep_open` carica anche `slot_off` del peer (`resident_plan` la richiede, `1519`).
- **Piano del server**: `resident_plan_par_kernel` va riscritto in variante a due tabelle che **emette solo le voci del
  server** (res del server ≥ 0 **e** res della padrona < 0, per non calcolare due volte un esperto doppio) e non mette
  `s_bad` sulle voci altrui; `counts[1]` = le sue voci, non `n`.
- **Divergenza silenziosa**: con EP il pool non azzera le righe `kind == 2` (`5165`): una voce che nessuno calcola
  porta nel combine la riga della finestra prima, e il testo può non vederla. A fine strato il server scrive **voci e
  gruppi** in una parola mappata (`m_cntS_[ring]`) e `ep_wait` le confronta con il conteggio `kind == 2` del pool
  (come `STRATA_ADAPT_CHECKRES`): una discordanza è un errore, non un testo diverso. Deciso in tappa 1.
- **Tappa 2 corretta**: «`skip_` quando il gruppo non ha miss» non va: con `device_plan_` il ramo è
  `copy_or_zero_from_mapped` (`verify.cpp:2318-2321`), che con `skip == ring` scrive zeri su **tutto** `parts_out` e,
  quando non salta, copia tutte le n×K righe sull'Oculink invece delle ~18 del server. Il combine EP usa sempre
  `copy_rows_from_mapped` con `p_dst` (le righe hit della padrona saltate) e aspetta `m_flagS_` più `m_flag_` solo se ci
  sono miss CPU. Il criterio «[plan] ≤ 0,1 ms» non è raggiungibile: `Vrif1` fa 0,15-0,25 (M) e il pool non si salta
  perché conta `d.usage` (`4902`); il criterio diventa «plan ≤ `Vrif1`» e la tappa 2 vale solo i µs di piano della
  padrona che il device-plan risparmia (S: 0,5 ms a finestra).

## 4. Tappe

Ogni tappa compila da sola (`build_lock_k.py`, sm_75+86), si prova da sola con un braccio `FEPn` della coda
(`coda_notte.py`, §5) e si chiude con una misura. Commit piccoli `FABLE-STRATA-…`, il ramo compila a ogni commit.
Nessuna tappa dopo la 0 inizia prima che la 0 e la 0b abbiano i loro numeri.

### Tappa 0 — lo span GPU del server di oggi (un'ora di codice, 30-50 minuti di coda)

- `src/core/peer_experts.{hpp,cpp}`: con `STRATA_PEER_STAMP=1` (opt-in, dentro `STRATA_DECODE_TIMING`) due stamp
  `__globaltimer` su `stream_` (il kernel di `gpu_stamp` di `verify_kernels.cu`, esportato) in una pagina mappata per
  strato: prima della prima `cudaMemcpyAsync` di `launch` (`310`) e dopo `scatter_rows_f32` (`330`). Niente altro
  cambia: il lavoro GPU è identico.
- `src/program/generate.cpp`: la riga `strata decode peer:` (c'è da 6bbd336e: `18973`) stampa anche **span GPU a strato
  p50/p90** e la somma a finestra, accanto ai ms di `launch` e `finish`.
- Braccio `FEP0`: `Vst` = `PEER` + `TEMPI` + `STRATA_PEER_STAMP=1`, build nuova (`build-up029pila-<sha>`, la `B3FF`
  non ha la riga), `_lungo`, 3 corse; `Vns` = lo stesso con `STRATA_PEER_NOSWITCH=1` (compilato, mai provato).
- **Criterio (decide tutto)**: span GPU del server a strato (p50) **< 150 µs** → tappa 1 a grafo; **150-250** → tappa
  1 solo con caldi alla padrona e righe bf16 fin dall'inizio, e il kernel unico resta in lista; **≥ 250 µs** → `procede =
  false`, si scrive il perché in `RISULTATI.md` e ci si ferma. In più: `host(launch + finish)` − span = quanto l'host
  aggiunge davvero (M), e testo identico a `Vpeer` (stessa build, `STRATA_PEER_STAMP=0`).

**Esito** (2/11 sera, M, `FEP0` in `corse_2080ti\20261101_tre_schede`, build `8ad3bef3`, 3 coppie `Vst`/`Vns` a contesto
128K, 42 richieste): span GPU del server a strato p50 **297-315 µs** per braccio (242-360 sulle richieste), p90 399-461;
somma 14,2-15,1 ms a finestra contro `launch + finish` dell'host 11,4-12,1 (launch 2,1-3,1, finish 5,7-11,1). Il flusso
parte prima che `launch` finisca (la somma degli span supera l'attesa in `finish`), quindi lo span può contenere fino a
45-65 µs di vuoti di sottomissione; il limite inferiore del lavoro GPU puro è l'attesa in `finish`, 120-230 µs a strato.
Tutti e due i limiti stanno sopra i 150 della tappa 1 a grafo e il centro sopra i 250: **`procede = false`**, motivo in
`RISULTATI.md`. `STRATA_PEER_NOSWITCH`: nessuna differenza, mai rifiutata. Testo: uguale fra i bracci a parità di
adattamento, non misurato contro `STRATA_PEER_STAMP=0` a cache fissa. Per riaprire: due stamp in più (dopo le copie,
prima dello scatter) che dicano se i ~300 µs sono copie e scatter sulle righe mappate o i kernel esperti stessi.

### Tappa 0b — il join dell'adattamento fuori dal percorso (mezza giornata)

- Prima la misura senza codice: braccio `Vsw4` = `PEER` + `--adapt-swaps 4` e `Vsw0` = `--adapt-swaps 0` (join e miss
  CPU a confronto con `Vpeer`: M).
- Poi la scelta (1) di §3b: `PeerExperts::adapt` fa `cudaStreamQuery(refill_)` dopo l'ultima copia accodata (il flush
  WDDM, #2188) e `apply_pending(false)` è chiamata dal filo del pool fra gli strati (già a polling con
  `STRATA_PEER_NOSWITCH`); `STRATA_PEER_REFILL_FLUSH=1`, opt-in dentro EP.
- **Criterio**: `adapt join` ≤ **0,5 ms** a finestra con gli swap di `Vpeer` (8-17) e `CPU experts` a strato-finestra
  non peggiore di `Vpeer`; testo identico a cache fissa. Se (1) non basta, (2) con il costo in miss scritto; (3) solo
  dichiarando la tappa 2 persa.

### Tappa 1 — il server a grafo sulla 2080 Ti, con tenuta, tag e conteggio (3-4 giorni)

- `src/core/peer_experts.{hpp,cpp}`: `ep_open(Verify&)` (solo con `STRATA_EP=1`): puntatori device delle pagine mappate
  `m_seq_`, `m_ids_`, `m_x_`, `m_ymiss_` della padrona; parole mappate nuove `m_flagS_`, `m_abortS_`, `m_errS_`,
  `m_cntS_[ring]` (Portable); copie ombra e device delle due tabelle di residenza (48 × 512 int32) e di `slot_off`; un
  grafo per (T, G) con, per strato e gruppo: attesa limitata (20 s, abort) di `m_seq_ ≥ ring` + controllo del tag di x →
  copia id + x → piano a due tabelle (§3f) → `hadamard_rows` + `quantize_q8_1_rows` → `native_expert_grouped` con
  conteggi device e griglia `cap, cap` → **`scatter_rows_flag`** (§3e: righe + tag + fence + contatore + `m_flagS_ =
  ring`) → scrittura di voci e gruppi in `m_cntS_`. `ep_launch(T, G)`: confronto ombra/tabelle vive, invio righe
  diverse, lancio del grafo, `cudaStreamQuery` subito dopo; `ep_flush()` dal pool a ogni strato (§3d); `ep_wait()` a
  fine finestra con il confronto dei conteggi; `ep_abort()` dalla release.
- `src/core/verify.cpp`: `post()` con EP: `wait_flag_ge(m_flagS_, ring)` + `m_flag_` solo se il pool ha miss CPU, poi
  `copy_rows_from_mapped` con `p_dst` e controllo del tag; `run()`: `ep_launch` dopo l'azzeramento dei flag (`2926`),
  `ep_flush` nel ciclo accanto al flush di `cs_` (`3246`), `ep_wait` in coda; `release_gpu_waits`: `h_flagS_` e
  `h_abortS_` nella lista (`595`) e drain dello stream del server (`601`).
- `src/core/expert_source.cpp`: con EP `kind == 2` resta, il pool non lancia e non aspetta il peer, conta le voci per
  il confronto di `ep_wait`.
- `src/program/generate.cpp`: `STRATA_EP` letta una volta; la riga `strata decode peer:` stampa l'esposto a strato =
  (`GPU-reach wait` − 11,7 di `Vpeer`) / 48, lo span GPU del server (gli stamp della tappa 0 dentro il grafo) e la
  mediana dell'attesa a `m_flagS_`.
- `tools/ep_bench.cu`: tag di finestra+strato in x e nelle righe, controllati in `k_wait_copy` e `k_server`; un giro
  del banco prima del motore (gli errori di ordine si vedono lì a costo zero).
- **Criterio**: braccio `FEP1`. Testo identico a `Vpeer` a cache fissa (`STRATA_EP=0` contro `=1`, stessa build);
  zero discordanze di conteggio e di tag su 3 corse; esposto a strato nel motore ≤ **150 µs** e `ms/window` ≤ 22 con il
  join della 0b → si continua; 150-200 → caldi alla padrona + bf16 (tappa 5 anticipata) e si rimisura; > 200 →
  kernel unico (1b, 6-10 giorni) **solo se** la tappa 0 aveva dato span < 150 (altrimenti lo stop era già lì).

### Tappa 2 — il piano della padrona sul device anche in EP (1-2 giorni; solo se 0b ha scelto (1) o (2))

- `verify_kernels.cu`: `resident_plan_par_kernel` a due tabelle per la padrona (le sue voci; le altrui non sono
  `s_bad`); `verify.cpp`: `device_plan_` acceso con EP; il combine resta `copy_rows_from_mapped` con `p_dst` (§3f), mai
  `copy_or_zero_from_mapped`; il pool pianifica comunque (conta `d.usage`) ma la padrona non aspetta il suo piano.
- **Criterio**: `per-layer host [plan]` ≤ quello di `Vrif1` (0,15-0,25 ms, M); `ms/window` −0,3/−0,5 rispetto alla
  tappa 1 (S); testo identico.

### Tappa 3 — i miss letti dal server dalla RAM bloccata per bisogno (3-4 giorni)

- Prima la misura: `tools/ep_bench.cu` con un blob «host» (1,7 MB in pagine bloccate, letto con `__ldcv` a 16 byte da
  64+ blocchi della 2080 Ti) **e lo stesso blob dentro `native_expert_grouped` con sorgente mappata** (il kernel vero, non
  solo la lettura di banda). Se < 6 GB/s (> 0,3 ms a blob) la tappa si ferma qui e lo dice.
- Poi: `STRATA_PIN_BY_NEED` esteso al peer (oggi «not with a layer split or a peer GPU», `6915`; cambia il layout
  dell'arena, `padded_`): la fetta di 8 GiB tiene i blob **caldi non residenti su nessuna delle due schede**, in ordine di
  profilo, invece del prefisso degli strati 0-9. Stima rifatta: 8 GiB coprono ~4800 delle ~11000 coppie fuori dalle GPU
  (S), non «tutti i miss». Il piano a due tabelle manda al server le voci con blob nella fetta (puntatore host nel piano,
  come `kind == 1` fa con `d.src->pinned`); la CPU tiene i miss fuori dalla fetta. Il nucleo RAM resta spento.
- **Criterio**: `CPU experts` a strato-finestra → < 0,5 voci (da 0,8-1,9, M); attese CPU a finestra → < 1 ms; `ms/window`
  −1/−2 (S); scrittura stabile con 6 core occupati (`carico_cpu.py`: oggi −13%, M).

### Tappa 4 — il prompt a stadi con la scrittura EP (misura prima, poi 6-9 giorni)

- **4a, misura senza codice** (1 giorno di coda): corsa in serie `--layer-split 20` con le cache forzate agli slot che
  avranno in EP (~1070 / ~6400, S) sui tre bersagli, contro `Vrif1`. Dice se il vincolo si tiene con i soli esperti
  ridotti; se la lettura perde già > 10% a 32K/128K, la tappa 4 passa alla doppia modalità (`PIPELINE2` §9c) e il
  conto dei giorni si rifà.
- **4b, codice**: `generate.cpp`: con EP, `--layer-split N --peer-device D` ammesso (`4174`); il verificatore di CUDA0
  costruito su 0-47 (densi di tutti gli strati su CUDA0, come già fa `--peer-device` da solo); lo stadio 1 tiene
  `Prefill` e stato K/V per 20-47 con **la tabella di residenza del server** al posto di `host_res` (`10170`) e gli
  esperti del chunk dello stadio 0 chiesti via `set_peer` di `sp` (`10263`), ma **non** verifica in scrittura. Specchio
  del K/V (`conversation_snapshot.cpp`): due stati per la copia host (§3a), copia delle sole celle residenti nuove, più
  `idx_pooled`, `idx_tail`, `idx_block_pos` e le cache di prefisso in RAM; `--prompt-cache-file` riammesso con EP quando
  lo specchio è provato.
- **Criterio**: lettura ≥ 90% di `Vrif1` a 32K e 128K (≥ 1660 / ≥ 1990 t/s) e pari a 4K; testo identico a cache fissa
  con e senza specchio; a più turni il prefisso riusato (tempo del secondo turno come oggi, M in `corse_turni`).

### Tappa 5 — il banco completo e le leve fini (2-3 giorni)

- Caldi alla padrona (`STRATA_PEER_HOT=0`, c'è: 98 µs nel banco, M) e righe bf16 al ritorno (90 µs, M; cambia i bit:
  opt-in dentro EP), se non già anticipate dalla tappa 1. Poi il banco completo 4K/32K/128K, 3 corse, contro `Vrif1`.
- **Criterio**: scrittura ≥ **115 t/s** sul banco (oggi ~104, M), lettura entro il −10% sui tre bersagli, testo
  identico a `Vpeer` (o a `--peer-device` + bf16 quando acceso), nessun `strata.exe` rimasto acceso.

Giorni in tutto (S): 1 ora + mezza giornata per decidere (0, 0b); poi 14-22 con il server a grafo; +6-10 se serve il
kernel unico; la tappa 4 può passare alla doppia modalità dopo 4a.

## 5. Come si prova

- Build: `build_lock_k.py <worktree> --target strata --jobs 6` (in background, attende il banco e prende il lucchetto),
  copia in `corse_2080ti\20261004_routed_keep\build-up029pila-<sha corto>\` (strata.exe, strata_kq.dll,
  strata-vision.exe copiato, cartella `source`, `build.json` con la nota «compilato sì, installato no, provato …»).
- Coda: in `coda_notte.py` una costante `BEPn = … / 'build-up029pila-<sha>'` e una fase `FEPn` con bracci `_lungo`
  (`banco_lungo.py`, 3-12 minuti di caricamento + 90 s), configurazione `PEER` (`--peer-device 1 --peer-reserve-mib
  3000`, `STRATA_ARENA_PIN_GIB=8`, catena, divisione, prefisso ed elastica tolti) più `TEMPI` e la variabile della tappa:
  per ogni tappa una coppia `Xep{n}` (`STRATA_EP=1`) e `Xpeer{n}` (`STRATA_EP=0`, stessa build: riferimento e verifica
  del testo), più `Xrif{n}` (serie, divisione 20) quando serve il confronto di lettura. `python -u coda_notte.py FEPn >
  fepn.out`, in background; esiti in `corse/<nome>/lungo.json`, `sommario_lungo.json`, `engine.log`. Niente visione
  sulla 3060, `--ple-io direct` lo mette la coda. Dopo ogni braccio: nessun `strata.exe` acceso.
- Se un motore non parte o si pianta: `engine.log`; tre tentativi sulla stessa cosa, poi ci si ferma e si dice.

## 6. Rischi

- **Lo span del server è già troppo alto** (tappa 0 ≥ 250 µs): è il rischio principale e per questo si misura prima di
  scrivere; se succede, il lavoro finito è una riga di diagnostica opt-in e un banco con i tag.
- **Watchdog WDDM e GPU perse**: attese del server limitate a 20 s con parola di abort letta da ogni attesa; `h_flagS_`
  e l'abort nella release; drain dello stream del server dentro la release; `ep_wait()` sempre prima di chiudere o
  cambiare grafo. Un falso positivo sotto i 20 s uccide la finestra e chiede il riavvio: per questo non 200 ms.
- **Flush WDDM del server**: senza il flush dal filo del pool il grafo della 2080 Ti può partire in ritardo o non
  partire; il flush è nel disegno della tappa 1, misurato con gli stamp della tappa 0 dentro il grafo.
- **Ordine flag/payload senza P2P**: non provato dal banco; `scatter_rows_flag` nella forma di `handoff_publish_kernel`
  più i tag di finestra+strato lo rendono un errore visibile, non un testo diverso a volte.
- **Divergenza fra `kind` e piano del server**: le righe `kind == 2` non azzerate; il conteggio voci/gruppi a fine
  strato confrontato in `ep_wait`.
- **Tabelle di residenza in ritardo**: invio dentro `ep_launch` da `host_res` e `res_` vive, mai da `d_res`.
- **VRAM della 3080 Ti**: 700 MiB liberi in `Vpeer` (sotto la soglia di guardia): la tappa 4 può chiedere
  `--vram-reserve-mib` più alta e meno slot; la residenza EP della 3080 Ti (~1070 slot, S) è il motivo della corsa 4a.
- **Memoria bloccabile**: 8 GiB di arena + i buffer mappati; il pin per bisogno della tappa 3 non ne chiede altra.
- **Il banco sottostima il lavoro del server, non quello della padrona**: le letture di banda non sono i kernel veri;
  l'unica copertura certa della padrona sono gli hit locali che girano mentre il pool serve lo strato, e non si
  sottraggono finché non sono misurati nel motore.
