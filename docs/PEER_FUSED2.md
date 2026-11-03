# PEER_FUSED2: due server di esperti (2080 Ti + 3060), un kernel persistente ciascuno

Ramo `lab/upsync-v0.2.9`, 3/11. Estende `PEER_FUSED.md` (un server, la 2080 Ti) a **due schede server**, ognuna con
il proprio mega-kernel per finestra (`STRATA_PEER_FUSED=1`). La seconda è la RTX 3060 12 GB (sm_86), che fa anche da
scheda video del desktop e sta su un riser PCIe x1 Gen2 (~0,5 GB/s, latenza alta): i suoi esperti sono **fissi**,
caricati una volta all'apertura, e per token passano solo attivazioni e righe di risultato. L'idea: le coppie che la
2080 Ti non riesce a tenere le recupera la 3060, e la padrona ha meno righe CPU e meno quota PCIe.

Tutto dietro la lista `--peer-device A,B`. **Con un solo device nulla cambia**: `peer2` non viene mai aperto,
`drive.d.peer2` è nullo, i parametri nuovi di `PeerExperts` (`before`, `ordinal`, `mine`, `other`) stanno al loro
default e il codice percorso è quello di prima, voce per voce; il solo costo aggiunto sul percorso a un server è un
confronto con un puntatore nullo per strato nel dispatch e nell'adapt (`d.peer2 != nullptr`, `peer2.valid()`).

Scritto e **non compilato, non installato, non provato** (§9).

## 0. In una frase

`generate.cpp` apre due `PeerExperts`: `peer` (server 1, come oggi) e `peer2` (server 2, `ordinal` 1). Il secondo si
riempie con le coppie di `ranked` che né la padrona né il primo hanno. Il pool del verificatore marca ogni voce con un
**proprietario**: `kind` 0 VRAM della padrona, 1 quota PCIe, **2 server 1**, **3 server 2**, −1 CPU; ogni server
lancia (o, con fused, conta) le voci con il proprio codice e scrive le righe nelle stesse righe mappate `m_ymiss_`. Con
`STRATA_PEER_FUSED=1` ogni server ha il suo kernel persistente, la sua `PeerFusedPage`, le sue tabelle; il
verificatore ha **due hook** (`set_peer_fused`, `set_peer_fused2`) e il grafo della padrona aspetta **due flag** a
passo, prima quello del server 1 poi quello del server 2, poi le righe CPU come oggi.

## 1. Opzioni e variabili

| opzione / variabile | default | che cosa fa |
|---|---|---|
| `--peer-device A,B` | un solo valore = un server, come oggi | A = server 1, B = server 2; al massimo due; B ≠ A, entrambi ≥ 1 (la padrona è il device 0); rifiutato altrimenti con un messaggio |
| `--peer-reserve-mib A[,B]` | 600; B = A | VRAM lasciata libera su ciascuna scheda prima di riempire gli slot |
| `--peer-slots A[,B]` | 0 (quanti ne entrano); B = A | tetto di slot per server |
| `STRATA_PEER2_ADAPT` | 0 | 1: anche il server 2 adatta (§3) |
| `STRATA_PEER2_ADAPT_SWAPS` | 2 | scambi massimi per giro del server 2, se adatta |
| `STRATA_PEER2_FUSED` | 1 (segue `STRATA_PEER_FUSED`) | 0: il server 2 resta sul percorso per-strato anche con `STRATA_PEER_FUSED=1` (il server 1 tiene il suo kernel) |
| `STRATA_PEER2_WAIT_MS` | 1500 | limite di **ogni** attesa dentro il kernel del server 2 (doorbell, `lready`, barriere): sotto i 2 s del TDR di Windows (§5). Il server 1 resta a `STRATA_PEER_FUSED_WAIT_MS` (20 000) |
| `STRATA_PEER2_FUSED_BLOCKS` | = `STRATA_PEER_FUSED_BLOCKS` | tetto di blocchi residenti per SM del kernel del server 2: con 1 la 3060 lascia posto al desktop (§5) |

`--peer-adapt-swaps`, `--peer-prefill-rows`, `STRATA_PEER_HOT`, `STRATA_PEER_*` restano del server 1. Le variabili
`STRATA_PEER2_*` le legge solo il `PeerExperts` aperto con `ordinal` 1 (`peer_fused.hpp`: `kEnvOn2`, `kEnvWaitMs2`,
`kWaitMs2Default`; le due dell'adapt sono lette in `generate.cpp`, una volta).

Ordine dei device nelle prove: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0,1,2` → 0 = 3080 Ti (padrona),
1 = 3060, 2 = 2080 Ti. Quindi `--peer-device 2,1`: server 1 la 2080 Ti, server 2 la 3060.

## 2. Riempimento e tabella dei proprietari

- `PeerExperts::open(..., before, ordinal)`: il ciclo su `ranked` salta le coppie che la padrona ha (`primary.slot_of`,
  come oggi) **e** quelle che `before->has(l, e)` (il server 1, aperto prima, fermo). Il server 1 prende le prime
  coppie del ranking che la padrona non ha; il server 2 le **successive**, fino al suo budget. Una coppia sta su una
  scheda sola per costruzione. `STRATA_PEER_HOT` sposta coppie calde dietro il rango `at` per il server 1: il server 2
  prende quello che resta dopo, invariato.
- Se `peer2.open` fallisce (device non visibile, allocazione, nessuna coppia rimasta): **una riga** su stderr
  (`second expert server on GPU B not opened (...): continuing with GPU A alone`) e il motore va avanti con un server.
  Non è un errore fatale, a differenza del server 1.
- Proprietario di una voce, nel pool (`expert_source.cpp`, `expert_pool_dispatch_multi`), nell'ordine in cui si
  chiede: slot della padrona (`host_res`) → `peer->has` (2) → `peer2->has` (3) → helper → Foresight → quota PCIe (1) →
  CPU (−1). La precedenza vale anche se, per un errore, una coppia stesse su due schede: la computerebbe il server 1.
  `nmiss` (i miss del piano, da cui la quota PCIe) esclude le coppie di entrambi i server: meno righe CPU e meno
  quota PCIe sulla padrona, che era il punto.
- `res_table()` / `cache_mut()` / `lend_guard()` (tappa 4b) e `set_peer` del prompt restano del **server 1**: lo stadio
  EP, il prestito della coda e la quota del prompt sul peer non vedono il server 2. Il percorso del prompt tratta le
  coppie del server 2 come non residenti (le legge dall'arena host): corretto, e voluto (riser lento). Le tappe 4/4b
  funzionano come prima; la nuova modalità non le richiede.
- Candidati dell'adapt della **padrona** (`generate.cpp`, `cand`): esclusi anche quelli che `peer2.has(l, e)`, come
  già quelli del server 1 (uno slot della 3060 non va sprecato per una coppia che la padrona poi si riprende).

## 3. Adattamento

- Server 1: `peer.adapt(usage, r0, swaps, err, other = &peer2)`: i candidati che il server 2 **ha o sta portando
  dentro** (`PeerExperts::claims`: `res_ ≥ 0` oppure in `pending_`) sono esclusi. Il resto come oggi.
- Server 2: **spento di default**. Con `STRATA_PEER2_ADAPT=1`, dopo l'adapt del server 1 **sullo stesso filo**:
  `peer2.adapt(usage, r0, STRATA_PEER2_ADAPT_SWAPS, err, other = &peer)`. Ordine deciso: il server 1 segna la vittima
  −1 subito (`res_[out] = -1`) e accoda la nuova coppia in `pending_`; il server 2 che viene dopo vede la vittima come
  candidata (è l'idea del proprietario: quello che la 2080 Ti perde lo recupera la 3060) e **non** vede la nuova
  coppia (è in `pending_` del server 1 → `claims` dice sì). Uno scambio non porta mai una coppia su due schede.
- `apply_pending` / `poll_pending` per server (`generate.cpp` `apply_pending`, `expert_source.cpp` dopo il `finish`).
- `claims` legge `pending_` dell'altro server dal filo dell'adapt, fra due finestre: lo stesso filo che lo scrive
  (`adapt`); `apply_pending` lo svuota al join del decode o sul filo del pool **dentro** una finestra, mai accanto a
  un adapt. Nessun lucchetto, come per `res_` oggi.

## 4. Mega-kernel: due hook, due flag

| pezzo | server 1 | server 2 |
|---|---|---|
| pagina mappata (`PeerFusedPage`, 8 KiB) | la sua | la sua (`h_page_` di `peer2`) |
| tabelle in VRAM (`d_own`, `d_slot_ptr`, `d_plan[2]`, `d_sync`, `d_lay`) | le sue | le sue, sulla 3060 |
| tabella di finestra `own_win_` | `res_ ≥ 0 && host_res < 0` | `res_ ≥ 0 && host_res < 0 && res1 < 0` (`fused_set_other_res(peer.res_table())`): la precedenza del dispatch, così i due kernel non pianificano mai la stessa voce |
| hook nel verificatore | `set_peer_fused` | `set_peer_fused2` |
| `begin` | prima | dopo, sullo **stesso** `WindowHost` (stesso doorbell `m_seq_`, stesse `m_x_`, `m_ids_`, `m_ymiss_`) |
| nodo nel grafo (per passo) | `wait_flag_ge(flag1, ring)` | `wait_flag_ge(flag2, ring)` subito dopo, poi l'attesa di `m_flag_` (righe CPU) |
| `flush` nel ciclo d'attesa | sì | sì |
| `end` dopo la sync | prima | dopo (un `end` del server 1 fallito ritorna; la guardia `PeerWindow` aborta il 2) |
| `release_gpu_waits` | flag a `UINT32_MAX`, `abort`, flag di nuovo a `UINT32_MAX` | lo stesso, dopo |
| `launch()` dentro la finestra | conta le voci `kind == 2` | conta le voci `kind == 3` (`mine`) e le confronta con `cnt[s]` del suo kernel a `end` |
| limite delle attese | 20 s | 1,5 s (`STRATA_PEER2_WAIT_MS`) |

Le due `fused_begin` corrono una dietro l'altra sul filo del verificatore prima del lancio del grafo: `own_win_` del
server 2 legge `res_` del server 1 nello stesso istante in cui il server 1 legge la propria, quindi le due tabelle di
finestra concordano, e `has()` del pool (che con la finestra aperta risponde da `own_win_`) concorda con entrambe.
`poll_pending` che muove `res_` a metà finestra non tocca le tabelle di finestra, come oggi.

**Misto**: un server può restare sul percorso per-strato (pack rifiutato da `peer_fused_shape` su quella scheda, o
`STRATA_PEER2_FUSED=0`) mentre l'altro ha il kernel: il verificatore tratta i due hook in modo indipendente, il pool
lancia per-strato chi non è fused. Nessuna combinazione è vietata.

**Istanze per architettura**: il kernel `peer_fused_kernel<2>` sta dentro `iq_kernels.cu` e viene compilato per ogni
architettura di `CMAKE_CUDA_ARCHITECTURES` (fat binary): la build che serve la 3080 Ti (sm_86) e la 2080 Ti (sm_75)
contiene già l'istanza sm_86 per la 3060. `peer_fused_shape` gira sotto il device del server (`On(device_)` in
`open`) e calcola occupazione, SM e memoria condivisa **di quella scheda** (la 3060 ha 28 SM e 100 KiB per SM: più
blocchi per SM a parità di MINB; il tetto `kDynSharedMax` 48 KiB resta). **CMake non va toccato** se la build include
già sm_86; se la build fosse solo sm_75+sm_120 la 3060 non avrebbe codice e `open` fallirebbe con una riga (§2).
`exp_mode()` deve essere 8 anche sulla 3060: con `STRATA_EXP_MODE=8` (come nelle corse `Qfus*`) lo è su ogni device;
senza la variabile sm_86 tiene il layout CUDA e `peer_fused_shape` rifiuta il kernel sulla 3060 (riga su stderr,
percorso per-strato).

## 5. La 3060 ha il desktop

- **TDR**: Windows resetta la scheda video se un pacchetto GPU dura più di 2 s. Il kernel persistente dura quanto la
  finestra (parte a `begin`, esce dopo l'ultimo passo); una finestra normale sta nelle decine di ms. Il rischio è una
  finestra che si blocca (la padrona non suona più, una barriera che non arriva): per questo **ogni** attesa del
  kernel del server 2 è limitata a 1,5 s (`STRATA_PEER2_WAIT_MS`), così il kernel esce da solo con `kErrTimeoutSeq` /
  `kErrTimeoutBarrier` prima del TDR, alza `flag_out = UINT32_MAX`, la finestra fallisce a `end` con il messaggio di
  oggi e il motore resta vivo. Il limite è per attesa, non per finestra: una finestra più lunga di 2 s senza stalli
  (48 strati × un passo lento) supererebbe comunque il TDR. **Non provato**; se succede, `STRATA_PEER2_FUSED=0` tiene
  la 3060 sul percorso per-strato (kernel brevi, nessun rischio) e il server 1 tiene il suo kernel.
- **Niente attese attive fuori dalle finestre**: fra `end` e il `begin` successivo sulla 3060 non gira nulla; i
  `cudaStreamQuery` del flush sono chiamate host. Le attese di `finish()` (HOSTWAIT) sono spin dell'host.
- **Apertura**: `peer2.open` che fallisce non è fatale (§2). Un'allocazione che fallisce dentro `fused_open` chiude il
  tier (`close()`), `open` ritorna falso, stessa riga e un server solo.
- **Traffico sul riser** (stima, da verificare con gli stamp): per strato il kernel legge `x` (T × 2560 × 4 B: 40 KB a
  T = 4) dalla pagina mappata attraverso il link e scrive le righe (10 KB a voce) nelle righe mappate; con ~5 voci a
  strato sul server 2 sono ~90 KB a strato, ~180 µs a 0,5 GB/s, nell'ordine dello span di oggi della 2080 Ti (290-330
  µs). Per finestra ~9 ms, sovrapposti alle righe CPU. Le letture di `x` sono per-warp a 128 B: la latenza del riser
  conta più della banda; la fase C del kernel le emette in parallelo su tutti i blocchi. Il percorso per-strato
  (`STRATA_PEER2_FUSED=0`) muove gli stessi byte con `cudaMemcpyAsync` (una copia di piano + righe per strato) e lo
  scatter diretto.

## 6. Bit-identità: che cosa cambia e che cosa no

- **Ordine di somma**: non cambia. La riga di una voce `j` la scrive un solo attore (padrona, server 1, server 2,
  PCIe o CPU) nella **sua** riga mappata; la somma pesata sulle `k` voci di un token la fa il grafo della padrona
  (`combine`) nell'ordine delle voci, uguale a oggi. "Prima server 1 poi server 2" è l'ordine delle **attese** nel
  grafo e delle chiamate `launch`/`finish`/`begin`/`end`, non di una somma.
- **Insieme di righe GPU ↔ CPU**: cambia. Una voce che oggi è una riga CPU (kernel CPU, altra aritmetica) diventa una
  riga del server 2 (q8_1 + `native_expert_grouped`): quei bit cambiano rispetto a un server, come già cambiano fra
  una corsa con e senza `--peer-device`. **Detto, non nascosto**: la risposta con due server non è bit per bit quella
  con uno. È la stessa differenza di una tier più grande sul server 1.
- **Righe del server 2 contro righe del server 1**: lo stesso kernel, gli stessi interi (`dp4a`), le stesse somme in
  float nello stesso ordine; il claim di `peer_experts.hpp` («an expert gives bit-identical rows on either card») vale
  già fra 3080 Ti (sm_86) e 2080 Ti (sm_75). Il banco con `--second` lo verifica fra due schede qualunque (§8): il
  riferimento delle voci del server 2 gira sulla **prima** scheda sugli stessi byte.
- **Senza `--peer-device A,B`**: bit identici a oggi (nessuna voce cambia proprietario, nessun nodo in più).

## 7. Log

- Apertura, una riga per server: `strata generate: peer GPU A: N experts, G GiB (filled in s)...` (come oggi) e
  `strata generate: peer GPU B (second server): N experts, G GiB (filled in s, reserve R MiB); with the primary's P
  and GPU A's Q that is T of M on the GPUs`. Con fused, `fused_open` stampa `the peer (second server) on GPU B serves
  each window with one persistent kernel: ... waits bounded at 1500 ms`.
- Tempi di finestra: alla riga `strata decode peer:` del server 1 (invariata) segue `strata decode peer2 (GPU B): per
  window: … ms host in launch + … ms waiting in finish over … layers with a share; per layer-window: … entries, …
  experts; adapt … ms host, … swaps/window (STRATA_PEER_FUSED: n windows fused, … steps/window); GPU span per step
  p50 … us, p90 … us, … ms/window` — l'attesa è **per server** (`ms_wait` di ciascun `PeerExperts`; con fused e senza
  HOSTWAIT è ~0, e contano gli span degli stamp del kernel).

## 8. Banco: `peer_fused_test --second D2`

`tools/peer_fused_test.cu` (stesso bersaglio CMake `peer_fused_test`, nessuna riga di CMake toccata): con `--second
D2` un secondo kernel persistente su `D2` accanto a quello su `device`, come nel motore: pagina propria, tabella di
proprietà **disgiunta** (~40 % delle coppie al server 1, ~30 % del resto al server 2: `kind` 2 e 3), arena di slot
propria con **gli stessi blob sintetici slot per slot** delle due schede, piano/sync/stream propri, lo stesso doorbell
e le stesse pagine `x`/`ids`/`ymiss`. L'host suona una volta, aspetta **entrambi** i flag (con `--primary D` il
kernel-padrona accoda due attese per passo, nell'ordine del grafo), fotografa le righe e, a kernel usciti, ricalcola
il riferimento **sulla prima scheda** per il piano del server 1 e per quello del server 2 (gli indirizzi dell'arena
della prima scheda, gli stessi byte) e confronta `memcmp` ogni voce: del server 1 → riferimento, del server 2 →
riferimento, di nessuno → sentinella. `cnt[s]` e stamp su entrambe le pagine; i passi forzati vuoti sono vuoti su
entrambi i kernel; le due prove di tenuta abortiscono e fanno scadere entrambi (il limite del server 2 è
`STRATA_PEER2_WAIT_MS` come nel motore, 200 ms nella prova (b)). Senza `--second` il banco è quello di prima.

```
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0,1,2
set STRATA_EXP_MODE=8
peer_fused_test 2 --second 1 --primary 0        (2080 Ti server 1, 3060 server 2, 3080 Ti padrona finta)
peer_fused_test 2 --second 1                    (senza padrona: doorbell dall'host)
```

## 9. Non provato e rischi aperti

Compilato no, installato no, provato no. Dove la compilazione può fermarsi: la lambda `refuse` di `fused_open` ora
cattura `this`; `claims()` inline nell'header usa `pending_` dichiarato più sotto nella classe (lecito in C++ per i
corpi inline, ma da vedere); `verify.cpp` itera `{peer_hook_, peer_hook2_}` in un `initializer_list` di puntatori
`const Hook*`; nel banco, le lambde `begin_window(w, w2)` e `check_step` con il ciclo `which`.

Rischi:

1. **TDR** sulla 3060 (§5): il limite per attesa non copre una finestra lunga senza stalli. Prima prova con
   `STRATA_PEER2_FUSED=0` (per-strato sulla 3060, kernel sulla 2080 Ti), poi con il kernel anche sulla 3060.
2. **Riser x1**: la pagina mappata letta per-warp dal kernel attraverso un link a latenza alta potrebbe costare più
   della stima di §5; gli stamp (`GPU span per step` della riga `peer2`) lo dicono. Se la 3060 rallenta la finestra
   (il grafo aspetta il flag 2 prima delle righe CPU), meno slot sulla 3060 (`--peer-slots A,B`) o spegnerla.
3. **Memoria mappata**: ogni server chiama `cudaHostGetDevicePointer` sulle stesse pagine Portable del verificatore
   sotto il proprio device; con tre device nel processo il flag Portable è già acceso (`set_peer_portable`).
4. **P2P**: `open` prova `cudaDeviceEnablePeerAccess` padrona↔3060 e la copia di prova (#2666); sul riser l'esito è
   ignoto, ma `p2p_` serve solo al percorso del prompt, che il server 2 non fa.
5. **Pool**: `kind == 3` è nuovo in `expert_pool_dispatch_multi`; ogni ramo che tratta `kind >= 0` (azzeramento righe,
   contatori) è stato esteso, ma un ramo dimenticato darebbe una riga azzerata sopra quella del server 2 (il banco
   non lo vede: è nel motore). Da controllare con `STRATA_PEER_DIRECT=0` (finish copia a posteriori) come prova
   incrociata.
6. **Desktop e prelazione**: il kernel persistente occupa tutti gli SM della 3060 per la durata della finestra; il
   compositore di Windows condivide la scheda a fette di tempo (prelazione a livello di istruzione da Pascal in su),
   quindi il desktop può scattare durante il decode e le attese del kernel possono allungarsi di una fetta.
   `STRATA_PEER2_FUSED_BLOCKS=1` lascia SM liberi; `STRATA_PEER_FUSED_COOP=0` evita il lancio cooperativo (che
   pretende la co-residenza di tutti i blocchi) se il driver lo rifiuta con la grafica attiva. Non provato.

## 10. Comando di prova suggerito (prima la chat principale)

```
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0,1,2
set STRATA_EXP_MODE=8
set STRATA_PEER_FUSED=1
set STRATA_PEER2_FUSED=0            (prima corsa: kernel solo sulla 2080 Ti)
strata ... --peer-device 2,1 --peer-reserve-mib 3000,1500 ...
```

Attesi: due righe di apertura (§7), `strata decode peer:` e `strata decode peer2 (GPU 1):`, risposta sensata. Poi
`STRATA_PEER2_FUSED=1` e la riga `the peer (second server) on GPU 1 serves each window with one persistent kernel`.
Confronto: la stessa corsa con `--peer-device 2` sola (bit diversi, §6: va misurato con KL / testo, non con `memcmp`).
