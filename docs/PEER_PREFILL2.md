# PEER_PREFILL2: anche il secondo server di esperti legge il prompt, con un modello di costo

Ramo `lab/upsync-v0.2.9`, 3/11 sera, scritto su `020c723e`. Estende `PEER_FUSED2.md` (due server di esperti: 2080 Ti +
3060, mega-kernel in scrittura) alla **lettura** del prompt. Tutto dietro **`STRATA_PEER2_PREFILL=1`**, default 0:
spenta, nessun oggetto, nessuna allocazione, nessuna riga cambia (`m.pp2` nullo, `set_peer2` non chiamata, il piano
dello stream e il layout delle righe quelli di prima, voce per voce).

Scritto e **non compilato, non installato, non provato** (§7). Convenzione come negli altri documenti: **M** misurato,
**S** stimato da misure.

## 0. In una frase

Oggi in lettura lavora solo il primo server: `strata prefill: peer GPU A computes its experts' rows of each prompt
chunk (…)` (`Prefill::set_peer`, `PeerPrefill`). Le 5747 coppie fisse della 3060 (9,4 GiB, M) in lettura non le usa
nessuno: la padrona le riporta dalla RAM a ogni pezzo come ogni coppia non residente. Con la variabile accesa il
secondo server riceve, per ogni pezzo di prompt e per ogni strato, **solo le righe instradate** ai suoi esperti che un
**modello di costo** decide di mandargli (già quantizzate q8_1, tagliate dalla `Xq` della padrona), le calcola con lo
stesso percorso MMQ del primo server e rimanda le righe FP32 nel blocco di `Dm` della padrona, che le somma con la sua
`combine` nel suo ordine. Il modello sceglie per esperto e per pezzo fra (a) mandare le righe e (b) far portare il blob
dalla RAM, con le bande dei collegamenti misurate all'apertura (o date), e tiene il lavoro del server dentro il tempo
che la padrona spende comunque in streaming su quello strato. Nessuna costante del riser: cambiando il link (Oculink)
cambiano le misure, non il codice.

## 1. Da dove partiamo (righe lette su `020c723e`)

| cosa | dove | oggi |
|---|---|---|
| quota del primo server nel prompt | `prefill.cpp` `Prefill::set_peer`, `struct PeerPrefill` | tutte le attivazioni del pezzo (T × N, fp16 sulla via host) al server, somme pesate per token di ritorno (`sums`), più il 35 % degli esperti in streaming della padrona attraverso un anello suo (`ps_frac`) |
| piano dello stream | `run`, «step 3» (`seq`, `pseq`) | per strato, ogni esperto che né `host_res` né `peer->has` → `seq` (anello H2D della padrona), **prima** che il routing dello strato sia noto |
| layout delle righe di uno strato | `on_peer`, `m.off` | `[locali \| peer-held (1) \| peer-streamed (2) \| CPU]`; il blocco del peer parte a `rows_local` (`P.back_at`); la `moe_combine_peer` (sums) salta le righe da `rows_local` in su |
| blocco per strato del peer | `peer_now`, `PeerPrefill& P = *m.pp` | ~290 righe scritte per **un** server |
| secondo server | `generate.cpp` `peer2.open(…, &peer, 1)` | solo decode (`kind 3`); «the prompt path's peer share (set_peer) and the tier loan (tappa 4b) stay the FIRST server's» |
| `Xq` della padrona | `mmq::quantize_scatter(xsrc, slot_dev, src_dev, m.Xq, …, T, K)` | **tutte** le T·K righe del pezzo, in ordine di esperto, q8_1: anche quelle dei blocchi dei server; il prodotto gate/up della padrona la legge intera (`total_rows = T·K`, bounds assoluti) |
| formato q8 di MMQ | `moe_mmq.cu` `q8_bytes`, `swiglu_quant_kernel` (`y + k_block * rows + r`), `Context::run` (`total_rows` → `ne11`) | **a blocchi di colonna**, non per riga: un buffer quantizzato con `rows` righe tiene il blocco `(kb, r)` — i 128 valori `[kb·128, +128)` della riga `r`, `block_q8_1_mmq`, 144 B — all'indice `kb × rows + r`, `kb < pad512(N)/128 = 20`; `q8_bytes(rows, N) = rows × 2 880 + 18 432` (coda fissa di 128 blocchi per l'over-read dei tile: `q8_bytes(1) = 21 312`, `q8_bytes(2) = 24 192`). Il percorso compatto del primo server quantizza **per gruppo** (`rows = nr` del gruppo, `total_rows = nr`, bounds relativi) |

Due fatti decidono il disegno:

1. **Le attivazioni intere non passano dal riser.** A T = 7680 la via del primo server muove 39 MB (fp16) in andata e
   39 di somme in ritorno per strato: a 0,5 GB/s sono ~160 ms a strato, 7,5 s a pezzo su 48 strati, quanto l'intero
   pezzo (M: 5,5 s a 1400 t/s). Il secondo server può servire solo se riceve **le righe sue** e nient'altro.
2. **La decisione va presa prima del routing.** `seq` si costruisce prima che `m.cnt` dello strato esista, e l'anello
   emette le copie in anticipo: un esperto tolto dallo stream dopo è già in volo. Quindi il modello decide **al piano
   del pezzo**, con una stima delle righe instradate (il conteggio dello stesso `(strato, esperto)` nel **pezzo
   precedente**: il routing di un prompt è stabile da pezzo a pezzo; al primo pezzo la priorità uniforme T·K/n_expert).
3. **Il blocco q8 del server non si taglia per riga.** Col formato a blocchi di colonna un intervallo di righe della
   `Xq` della padrona (passo T·K) sono 20 strisce, una per blocco di colonna, e un gruppo del server (che il
   prodotto legge con `total_rows = nr`) le vuole con passo `nr`: **una `cudaMemcpy2DAsync` per gruppo** (sorgente
   passo `T·K × 144`, destinazione passo `nr × 144`, larghezza `nr × 144`, altezza 20) nella pagina mappata a
   `r0 × 2 880`, poi una H2D contigua del blocco. Gli stessi byte di prima, nel layout che il server avrebbe scritto
   quantizzando da sé. La prima stesura tagliava a `r × 2 880` ed è caduta al banco (`q8 rows are not a fixed
   stride (21312, 24192)`): il controllo in `set_peer2` ora verifica `q8_bytes(rows) = rows × rb + coda` e il
   taglio è quello di sopra. Ricalcolare la q8 dal FP16/FP32 sul server costerebbe 2-3,5× i byte sul link: no.

## 2. Modello di costo (il centro)

Per ogni pezzo di T token, per ogni strato `l` e per ogni esperto `e` che il secondo server tiene (e che né la cache
della padrona né il primo server hanno: la precedenza del dispatch), con `c` = righe instradate stimate:

| scelta | byte sul link della padrona | tempo del server |
|---|---|---|
| **(a)** mandare le righe | `c × (xq_rb + N × 4)` = `c × 13 120` B (q8_1 MMQ di 2560 valori = 20 blocchi × 144 B = 2 880, §1 punto 3; FP32 di ritorno 10 240) | `c × 13 120 / bw_server + c × row_us` (+ `layer_us` una volta per strato con almeno una voce) |
| **(b)** streaming del blob | `blob(l)` = 1,2-1,7 MiB | 0 (la padrona: `blob / bw_padrona`) |

- **Candidato (a)**: `c × 13 120 < blob(l)`, cioè `c` sotto ~90-130 righe (1,6 MiB / 13 120 ≈ 128). Un esperto con 0
  righe nel pezzo prima è candidato a costo 0.
- **Ordine**: per rapporto `risparmio / tempo del server` decrescente (`risparmio = blob − c × 13 120` byte sul link
  della padrona): prima i più freddi.
- **Budget per strato** (il server lento non finisce mai dopo la padrona): la somma dei tempi (a) presi, più
  `layer_us`, deve stare in `budget_frac × (n_nonres − presi) × blob / bw_padrona`, il tempo che la padrona passa
  comunque a portare dalla RAM gli esperti **che restano** in streaming su quello strato (il suo pavimento di tempo nei
  pezzi grandi, dove lo streaming domina). Greedy: ogni esperto preso riduce il budget di un blob.
- **Stima di `c`**: `cnt_prev[l][e]` del pezzo precedente (aggiornato a ogni strato, piano o no); −1 al primo pezzo →
  `T·K/n_expert` (a T = 7680, K = 10, 512 esperti: 150 righe → nessun (a) al primo pezzo; a T = 4096: 80 → tutti i
  candidati, dentro il budget). Un pezzo sotto `stream_all_min` (1024) non ha piano: la camminata a stadi non passa di
  qui.
- **Bande**: `bw_server` e `bw_padrona` misurate a `set_peer2` con una copia bloccata H2D da 32 MiB cronometrata (la
  migliore di 3 dopo una di riscaldamento) su ciascuna scheda; oppure date: `STRATA_PEER_PREFILL_BW_GBS=A,B` (per
  ordinale, come `--peer-device`) e `STRATA_PREFILL_BW_GBS` per la padrona. Senza misura e senza valore la quota è
  rifiutata con una riga (un modello senza misura tirerebbe a indovinare).

### 2.2 Anello chiuso (quarta stesura, dopo la prova nel motore del 3/11 notte)

La prova con la terza stesura (build `4512b587`, `--peer-device 1,2`, 2080 Ti server 1 su x16, 3060 server 2 sul
riser) ha **dimezzato la lettura**: 446/778/929 contro 966/1378/1466 (M), con `combine 12152 ms (78,7 %)` a 4K: la
padrona aspettava la 3060. Tre cause, tutte nel modello aperto:

1. **La banda della padrona misurata era falsa** (0,85 GB/s; la 3060 0,38): una copia sola da 32 MiB cronometrata
   dall'host con `cudaStreamSynchronize`, e sotto WDDM la latenza di invio del driver (decine di ms) era quasi tutto
   il tempo. Ora `measure_h2d_gbs` fa **8 copie in fila da memoria bloccata** (la stessa natura dell'arena) e misura
   con **eventi sullo stream** (il tempo in cui la scheda è occupata, non la vista dell'host), meglio di 3 giri dopo
   uno di riscaldamento, con `cudaStreamQuery` per spingere il lotto.
2. **Il tetto per strato non limitava nulla**: con `bw0` = 0,85 un blob «costava» 1,9 ms e il budget di uno strato
   (`n_nonres × t_blob` ≈ 400 × 1,9 ms ≈ 0,8 s) accoglieva ogni candidato; per questo `ROW_US` non entrava nel piano
   (identico con 1 e con 6) e il «risparmio stimato» era 424 s su un prompt da 134 s. Con la banda vera (~5-6 GB/s) il
   budget torna a ~100 ms per strato e il costo per riga decide.
3. **Nessuna retroazione**: il modello non vedeva quanto la padrona aspettava davvero. Ora, per ogni pezzo:
   - sulla padrona, due eventi per strato intorno all'attesa delle righe del server (`ev_w0` prima di
     `cudaStreamWaitEvent(ev_done2)`, `ev_w1` dopo la `copy_f32_wide`): lo **stallo** della padrona per il server,
     meno il tempo nominale della copia delle righe;
   - sul server, due eventi per strato (`ev_s0` prima della H2D del blocco, `ev_s1` dopo l'ultima D2H delle righe): lo
     **span del server**;
   - al piano del pezzo successivo (`p2_fold`, prima di `p2_plan_layer`; l'ultimo evento del pezzo prima è di norma
     già completo, altrimenti lo si aspetta): il **costo per riga misurato** `row_us_eff = (span − strati ×
     layer_us) / righe` (EWMA 0,5) entra nel piano al posto di `row_us` quando è maggiore; e il **fattore** del budget
     si muove dallo stallo contro il risparmio stimato del pezzo: stallo > 25 % del risparmio → ×0,5; > 5 % → ×0,8;
     < 1 % e sotto 1 ms → ×1,25; limiti [0,02, 4]. Il budget per strato è `fattore × STRATA_PEER2_PREFILL_BUDGET ×
     (n_nonres − presi) × t_blob0`.
   - una riga per pezzo: `strata prefill: second server GPU B (stage on CUDAx), chunk c: sent P pairs (R rows) over L
     layers; server S ms, the primary stalled W ms for it, estimated saving E ms; row cost measured m us (model m0: link
     + compute); budget factor f0 -> f1`, e nel riepilogo di fine prompt le somme (`closed loop over n chunks read:
     server … ms, the primary stalled … ms, row cost measured … us, budget factor now …`).

Che cosa aspettarsi sul riser: al primo pezzo (priorità uniforme, fattore 1, costo per riga del modello ≈ 34 µs
con la 3060 a 0,38 GB/s) il server prende meno di prima (il budget è ~100 ms per strato, non 800); la riga del pezzo 0
dirà uno stallo ancora positivo (il costo vero per riga della 3060 sui gruppi piccoli è ~60-70 µs: 254 ms per strato
a 3458 righe nella prova) e `row cost measured ~60-70 us`, fattore 1 → 0,5 o 0,8; dal pezzo 1 il piano usa il costo
misurato e il fattore, lo stallo scende verso lo zero e il fattore si assesta dove la 3060 finisce appena prima della
padrona. Un prompt da un pezzo solo (4K) non ha retroazione: paga il primo piano. Sull'Oculink la stessa riga mostrerà
costo per riga ~5 µs e fattore che sale fino a 4 finché la soglia dei byte (non il budget) limita i candidati.

### 2.3 Il risultato vero: ON/OFF sul tempo del pezzo (quinta stesura, dopo la prova con `076e7d90`)

Con l'anello del §2.2 le bande sono vere (3060 0,38, padrona 6,72 GB/s), il fattore scende a 0,03-0,04 con stallo di
2-3 ms a pezzo — **e la lettura resta sotto la versione spenta** (878/1274/1297 contro 963/1376/1464, M): +255 ms a
pezzo, ~5 ms a strato, con il server che a regime riceve 20-30 righe a strato. Tre cose che il §2.2 non vedeva:

1. **Il costo fisso per strato del server** (~6 ms sul riser: decine di chiamate CUDA, copie piccole, due prodotti su
   gruppi minuscoli) finiva nel «costo per riga» (240 µs «a riga» con 26 righe a strato). Ora `layer_us_eff` è l'EWMA
   dello **span più piccolo** fra gli strati del pezzo (quelli con meno righe ≈ il solo costo fisso) e `row_us_eff`
   è ciò che resta sopra di esso; il piano parte da `max(layer_us, layer_us_eff)`.
2. **Il risparmio era teorico, il costo della padrona reale.** I blob mandati al server non sono più in `seq` (il
   `continue` del piano precede `seq.push_back`: non vengono portati), ma la padrona paga per strato il **tempo host
   per emettere il lavoro del server** (`peer_layer`: ~40 chiamate CUDA su un altro device sotto WDDM, con due
   `cudaSetDevice`) più copie 2D, eventi e un'attesa fra device: quando l'host è il collo del prompt, è tempo di
   pezzo. Ora si misura (`host_ms_chunk` → `host_us_layer`, EWMA) e **uno strato è mandato solo se il suo risparmio
   stimato copre due volte quel costo** (`layers_dropped` nella riga): sul riser uno strato da 20-30 righe non
   conviene mai.
3. **L'anello guarda il tempo vero, a coppie di pezzi adiacenti.** La prima versione (quinta stesura, `4d38da47`)
   confrontava due EWMA del pezzo intero per token, con e senza server, e sceglieva ON a torto (0,762 contro 0,807
   ms/token, prompt il 12 % più lento): il costo per token cambia con T (il primo pezzo ha 4089 token, gli altri di
   più) e con la **posizione** (l'attenzione cresce col contesto), e le due EWMA erano fatte di pezzi in posizioni
   diverse. Ora: due eventi sulla padrona per ogni strato MoE (`ev_m0` all'inizio del MoE, `ev_m1` dopo la
   `combine`), sempre; la misura del pezzo è la **somma degli span MoE** (niente attenzione → niente posizione;
   scala con le righe del pezzo). Si confrontano solo **coppie di pezzi adiacenti con lo stesso T**: un pezzo ON
   seguito da uno OFF, `guadagno = (MoE_off − MoE_on) / MoE_off`, EWMA 0,5 (`gain_ewma`); la modalità è ON solo con
   `gain_ewma > 2 %` (`kGainOn`). **Si parte spenti**: da OFF una coppia di prova (ON, poi OFF) ogni 16 pezzi
   (`kTrialEvery`), **mai sul pezzo 0 di un prompt** (un prompt da un pezzo non prova mai; il primo pezzo può anche
   essere più corto e la coppia sarebbe nulla); da ON una coppia di controllo (il pezzo ON in corso, poi uno OFF)
   ogni 8 (`kCheckEvery`). Una coppia non attraversa due prompt (`run` azzera il mezzo pezzo pendente). Modalità,
   guadagno e costi **persistono fra i prompt** (vivono nel `PeerPrefill`).

Riga per pezzo (sempre, una per pezzo): `strata prefill: second server GPU B (stage on CUDAx), chunk c (k of its
prompt) ON|OFF [(pair, first|second)]: sent P pairs (R rows) over L layers (D dropped: too few rows for the host's
cost); server S ms, the primary stalled W ms for it, estimated saving E ms; primary MoE M ms over L layers[; pair: ON
a ms vs OFF b ms at T tokens = gain g% (EWMA G% over n pairs)]; server fixed F us + r us a row (model m0), host in
its issue h us a layer; budget factor f0 -> f1; mode x -> y`. A fine prompt le somme e `pair gain EWMA`.

**Sul riser, atteso**: il banco manda per primo il 4K (un pezzo): OFF, nessuna prova → lettura = versione spenta.
32K/128K: pezzo 0 OFF; pezzo 1 ON (coppia, prima metà) con `host … us a layer` ~1-2 ms e stallo; pezzo 2 OFF
(seconda metà): `pair: ON a vs OFF b = gain −5…−10 % (EWMA …) ; mode off -> off`; poi OFF fino al pezzo 17, altra
coppia. Costo delle prove: ~1 pezzo ON ogni 16 → lettura entro l'1-2 % della versione spenta a 128K, uguale a 4K.
**Sull'Oculink, atteso**: la prima coppia dà guadagno positivo (blob tolti a una padrona davvero in streaming,
costo fisso del server ~0,5-1 ms), `mode off -> on`, poi una coppia di controllo ogni 8 e il fattore che sale.

### 2.1 Esempio numerico: riser x1 Gen2 contro Oculink x4

Numeri: blob 1,6 MiB; padrona 3080 Ti sull'Oculink a 6,7 GB/s (M) → un blob le costa **250 µs**; per strato lo stadio 0
porta ~325-460 esperti dalla RAM (S, `EP_SPLIT_PREFILL.md` §8/§13) → **~100 ms a strato di streaming**, il budget;
la 3060 tiene ~120 coppie a strato (5747/48), le più fredde del ranking.

| | riser x1 Gen2 (~0,5 GB/s, M) | Oculink x4 (~6 GB/s, S) |
|---|---:|---:|
| tempo del server per riga (13 120 B + 1 µs) | 27 µs | 3,2 µs |
| righe che stanno nel budget di 100 ms | ~3 700 | ~31 000 |
| esperti presi a strato (candidati da ~40 righe medie) | ~90 (il budget li ferma) | tutti i ~120 che passano la soglia (la soglia li ferma) |
| streaming tolto alla padrona per strato | ~90 × 250 µs ≈ **22 ms (−20 %)** | ~120 × 250 µs ≈ **30 ms (−30 %)** |
| byte sul link della padrona per quei esperti | 90 × 40 × 13 120 ≈ 47 MB **invece di** 90 × 1,6 MiB ≈ 151 MB | 63 MB invece di 201 MB |

È una stima (S): il guadagno vero dipende da quanto lo streaming della padrona oggi si sovrappone al calcolo e da
quanti candidati stanno davvero sotto la soglia. Lo dicono la riga `strata prefill timing (second server GPU B): …
planned P of C candidate pairs over L layers, estimated X ms of the primary's streaming saved` e il confronto di
lettura con/senza la variabile. Con l'Oculink il modello cambia da solo (bande misurate), e il limite passa dal budget
alla soglia: lì converrebbe alzare la soglia mandando anche esperti più caldi, ma solo se il link della padrona ne ha
il margine — il criterio dei byte sul link della padrona resta quello giusto.

## 3. Disegno nel codice

- **`PeerPrefill`** (`prefill.cpp`): campi nuovi, tutti inerti sul primo server: `rows_in`, `host_xq` (mappata,
  Portable), `xq_blk` (VRAM del server, + `MMQ_TAIL`), `xq_rb`, `bw_gbs`, `bw0_gbs`, `row_us`, `layer_us`,
  `budget_frac`, `cnt_prev`, `plan`, contatori.
- **`Prefill::set_peer2(peer, err)`** (`prefill.hpp/.cpp`): apre il `PeerPrefill` del secondo server in modo `rows_in`:
  via host, compatto, **niente `sums`** (righe FP32 di ritorno), niente `mixed`, niente `src`; buffer per gruppo
  (`GU_g`, `H_g`, `Hq_g`, `Dm_b[2]`, `grp_gu/grp_d`, `ident`, `bounds`) e il blocco `xq_blk` da `T_max × K` righe.
  Rifiuta: MMQ assente, pezzo sotto `stream_all_min`, pack non nativo, NVFP4, attivazioni FP4, stessa scheda dello
  stadio; APR declinato con una riga. Misura (o legge) le bande, stampa la riga di §4.
- **Piano** (`run`, «step 3»): `p2_plan_layer(l)` prima del ciclo sugli esperti dello strato: azzera il piano dello
  strato, salta gli strati fuori dalla grouping MMQ o `wmma_only`, raccoglie i candidati e applica §2; nel ciclo
  l'esperto con `plan[l][e]` è `continue` (non va in `seq` né in `pseq`). Il piano intero è azzerato a ogni pezzo
  (uno strato saltato dalla KVA non ne eredita uno).
- **Layout delle righe** (per strato, dopo il blocco del primo server): `on_peer[e] = 3` per le coppie del piano con
  `cnt > 0` (solo con `stream_all` e MMQ); `cnt_prev[l]` aggiornato sempre; ordine dei blocchi **`3, 1, 2`**:
  `[locali | server 2 | server 1 held | server 1 streamed | CPU]`. Il blocco del secondo sta **prima** di quello del
  primo perché `moe_combine_peer` (il primo in modo `sums`) salta ogni riga da `back_at` del primo in su: le righe FP32
  del secondo stanno sotto, in `Dm`, e la combine le legge come locali. `order_peer2` raccoglie le sue voci.
- **Un corpo per la lista di server**: il blocco per strato è ora la lambda `peer_layer(P, ord, r_at, nr_p, timed)`,
  chiamata per `m.pp2` (`order_peer2`, `rows_local`, `rows_peer2`) e poi per `m.pp` (`order_peer`, `rows_local +
  rows_peer2`, `rows_peer`). Dentro, i gruppi (`P.groups`) si fanno **prima** delle copie; il solo ramo nuovo è
  `rows_in`: per ogni gruppo la padrona copia D2H con una `cudaMemcpy2DAsync` le sue `nr` righe (20 strisce da
  `m.Xq + (r_at + r0) × 144`, passo `T·K × 144`) nella pagina mappata a `r0 × 2 880` con passo `nr × 144` (§1 punto
  3; sullo stream delle copie, come la via host del primo), il server copia H2D il blocco intero in `xq_blk`,
  nessuna tabella `src`, e il prodotto gate/up del gruppo legge `xq_blk + r0 × xq_rb` con `total_rows = nr` al posto
  di `quantize(P.mixed, …)`. Tutto il resto (gather, prodotti, swiglu, quantize di H, down, righe D2H in `host_rows`
  per gruppo su `s_out`, `ev_done`) è il codice di prima. `pe.mark` solo con `timed` (gli eventi del `PeTimer`
  vivono sulla scheda del primo). Misure del formato in `PeerPrefill`: `xq_blk_b` 144, `xq_kb` 20, `xq_rb` 2 880,
  da `q8_bytes(0/1/2/3, N)`.
- **Dopo i prodotti**: `peer2_now` → `cudaStreamWaitEvent(ev_done2)` + `copy_f32_wide` delle righe nel blocco di `Dm`,
  prima del ramo del primo; la `combine` non cambia.
- **Interdizioni coerenti**: `cpu_maybe` anche `!m.pp2` (la quota CPU vuole il suo blocco in coda); `lm_prepare`
  declina con `m.pp2` come con `m.pp`.
- **`generate.cpp`**: dopo `sp.set_peer`, con `peer2.valid()` e la variabile: `sp.set_peer2(&peer2)` e, con la tappa 4b,
  `ep_stage->sp.set_peer2(&peer2)` sotto `OnDevice(ep.dev)` — **due** `PeerPrefill` sulla 3060, uno per stadio, stream
  e buffer propri (gli stadi corrono in pipeline e la 3060 serve entrambi). Con `STRATA_EP_SPLIT_LM=1` non si chiama
  (una riga). `peer2.quiesce` accanto a `peer.quiesce` nel `lend` e prima del passaggio di stato: con
  `STRATA_PEER2_ADAPT=1` una copia dell'adapt del server 2 non deve essere in volo mentre gli stadi leggono i suoi
  slot (senza adapt non fa nulla).

## 4. Variabili e default

| variabile | default | che cosa fa |
|---|---|---|
| `STRATA_PEER2_PREFILL` | **0** | 1: il secondo server legge il prompt col modello di costo (§2). 0: come oggi, le sue coppie vengono dalla RAM |
| `STRATA_PEER_PREFILL_BW_GBS` | — | `A[,B]`: banda H2D dei server in GB/s per ordinale; assente o 0 = misurata a `set_peer2` |
| `STRATA_PREFILL_BW_GBS` | — | la banda H2D della padrona (il costo di un blob in streaming); assente = misurata |
| `STRATA_PEER2_PREFILL_ROW_US` | 1 | µs di calcolo del server per riga instradata, sommati al tempo di trasferimento |
| `STRATA_PEER2_PREFILL_LAYER_US` | 300 | µs fissi per strato in cui il server ha almeno una voce (eventi, lanci, latenza del link) |
| `STRATA_PEER2_PREFILL_BUDGET` | 1,0 | frazione (0..4) del tempo di streaming della padrona sullo strato che il server può usare, moltiplicata dal fattore dell'anello chiuso (§2.2); sotto 1 più prudente, sopra 1 per misurare se la padrona non era davvero il collo |
| `STRATA_PEER2_PREFILL_BITS` | 0 | 1: il piano prende solo gli strati con `MmqPlan::group_sort` (entrambi i prodotti MMQ a tile interi su **ogni** scheda visibile: `mmq::whole_tiles`): lì le righe del server 2 sono quelle della padrona bit per bit per costruzione (§5). 0: tutti gli strati MMQ |

Riga all'apertura: `strata prefill: peer GPU B (second server) computes the rows of its experts that the cost model
sends it (q8_1 rows in, FP32 rows out through mapped host memory, up to R rows per layer; its link x.xx GB/s
(measured|given), the primary's y.yy GB/s (…): an expert goes there under ~N routed rows of a chunk, within the
primary's streaming time of the layer); M MiB left free on it`. A fine prompt (con `STRATA_PREFILL_TIMING`): `strata
prefill timing (second server GPU B): MoE layers with rows …, rows/layer …, experts/layer …; the cost model planned P
of C candidate pairs over L layers, estimated X ms of the primary's streaming saved (links: server …, primary …)`.

VRAM sulla 3060 per ogni `PeerPrefill` a T = 7680 (S): `xq_blk` 221 MB + `Dm_b` 157 + `GU_g` 39 + `H_g` 20 + `Hq_g`
5,5 + gruppi ~26 ≈ **470 MB**; con la tappa 4b sono due → ~0,95 GB: `--peer-reserve-mib A,B` con **B ≥ 2000** (il
desktop sta lì). RAM bloccata: `host_xq` 221 MB + `host_rows` 315 MB per istanza.

## 5. Bit

- **Spenta**: identici a oggi (nessuna voce cambia proprietario; il layout a tre blocchi con `rows_peer2 = 0` è quello
  di prima).
- **Accesa**: l'esperto mandato al server 2 non è più calcolato dalla padrona dal suo anello, ma dal server 2 con gli
  **stessi byte q8_1** (tagliati dalla `Xq` della padrona: `quantize_scatter` li ha scritti una volta per tutte le
  righe), gli stessi blob, gli stessi kernel MMQ, la stessa sequenza di chiamate. Le righe FP32 tornano nelle
  **stesse righe di `Dm`** e la `combine` le somma nello stesso ordine `k`: **la somma per token è la stessa se le
  righe sono le stesse.**
- **Dove le righe possono differire: il regime di lancio di MMQ** (`mmq.cuh` `launch_mul_mat_q`, letto il 3/11
  notte). Il lancio ha `tile = (1280/I) × ntx × n_esperti` tile di uscita (`ntx = ⌈max_rows/J⌉`). Se le tile riempiono
  le onde di SM della scheda almeno al 90 % (`100·tile / (nsm·⌈tile/nsm⌉)`), **tiling**: un blocco per tile, K in
  sequenza, e i bit di una riga dipendono solo dalla riga e dall'esperto. Sotto il 90 %, **stream-k**: `nsm` blocchi
  si dividono lo spazio `tile × K` in parti contigue, una tile tagliata fra due blocchi dà due somme parziali più un
  `fixup`: **un altro ordine di somma in float**, che dipende da `nsm`, da `ntx` e da `n_esperti`. Il banco della
  prima stesura lo ha mostrato: sulla stessa 3080 Ti (80 SM, `1280/I = 10` tile per esperto) i gruppi del server da
  7, 5 e 2 esperti (140/100/40 tile: 87, 62, 50 %) differivano dalla seconda tile di colonna in poi (`J = 128`: «first
  differing row 128»), il gruppo da 11 (220 tile, 91,7 %) era identico; le differenze «1e3–2e7» sono ultimi bit su
  valori da 1e9–1e11 (le attivazioni a 1e4 del generatore; il banco ora stampa anche il massimo relativo).
- **Rimedio: il lancio del server è imbottito a 16 esperti** (`gu.n = dn.n = MMQ_GROUP` con `rows_in`; i bounds
  relativi ripetono già l'ultimo fino all'indice 16, quindi gli esperti in più hanno `col_diff = 0`: tile saltate
  prima di ogni lettura, `fixup` che esce subito, slot dei pesi mai letto; `ident` con 1024 voci di margine per le
  letture degli id del `fixup`). Così il lancio ha le tile di un **gruppo pieno**, lo stesso regime dei gruppi pieni
  della padrona: sulla 3080 Ti `160·ntx` tile su 80 SM riempiono sempre le onde (100 %), sulla 3060 (28 SM) al 95 %,
  **tiling su entrambe** → K in sequenza → **gli stessi bit** di un gruppo pieno della padrona, a parità di
  configurazione MMQ (3080 Ti e 3060 sono entrambe sm_86: stesse `I`, stessi tile K, stessa `sram_layout`).
- **Che cosa resta diverso, per costruzione**: (1) la **2080 Ti** (sm_75: altra configurazione `I`/K, `dp4a`) somma in
  un altro ordine da riga 0, su ogni tipo: col primo server sulla 2080 Ti le righe del prompt già oggi non sono
  quelle della padrona; (2) il **gruppo parziale** della padrona (l'ultimo, sotto 16 esperti): se le sue tile non
  riempiono le onde è in stream-k e i suoi bit dipendono dai compagni di gruppo — spostando esperti al server 2 i
  gruppi della padrona si ridistribuiscono, quindi «testo identico alla corsa senza la variabile» non vale per
  costruzione, esattamente come oggi con `STRATA_MMQ_GROUP_SORT` («exact for the full groups only»). Vale invece
  «righe del server 2 = righe che la padrona darebbe in un gruppo pieno», che è ciò che il banco verifica.
- `mmq::whole_tiles(tipo, righe, 16)` (`MmqPlan::group_sort[l]`) chiede la stessa cosa **a ogni scheda visibile**:
  su questa postazione dice no per la 2080 Ti (68 SM: 160 tile = 3 onde al 78 %) anche dove 3080 Ti e 3060
  concordano. **`STRATA_PEER2_PREFILL_BITS=1`** resta il filtro prudente (solo strati `group_sort`): qui non ne
  accende nessuno finché la 2080 Ti è visibile; il banco è l'arbitro sulle due sm_86.
- **Prima pezzo contro i successivi**: la stima di `c` cambia il piano, non i bit di un esperto dato chi lo calcola;
  ma **chi calcola cosa** dipende dal pezzo precedente, quindi due corse della stessa richiesta con la stessa cache
  danno lo stesso piano (deterministico: stessi conteggi), mentre un routing diverso a monte cambia il piano. Detto,
  non nascosto.

## 6. Banco: `peer_prefill2_test`

`tools/peer_prefill2_test.cpp` (bersaglio CMake `peer_prefill2_test`, `EXCLUDE_FROM_ALL`, linka `strata_prefill` per
MMQ). Due device qualunque (anche lo stesso due volte). Modello sintetico misto (le dieci coppie di tipi del pack,
generatore di `peer_fused_test` copiato: scale fp16 finite e piccole, codici casuali), `n_experts` blob per strato
**uguali slot per slot** sulle due schede, T token con K esperti distinti e pesi, il layout di righe della padrona
(ordine di id, `off`, `src`, `slot`). Per pezzo e per strato: la padrona quantizza tutte le righe (`mmq::quantize`
con `src`: gli stessi byte di `quantize_scatter`) e calcola il riferimento di tutti gli esperti **come fa il motore**
(buffer intero, `total_rows = T·K`, bounds assoluti); un intervallo contiguo di esperti `[e0, e1)` è «del secondo
server»: le sue righe q8_1 vanno D2H nella pagina mappata con una copia 2D per gruppo (§1 punto 3) e H2D sul secondo
device, gli stessi gruppi girano lì con la sua arena (gruppo per gruppo, `total_rows = nr`, bounds relativi), le
righe FP32 tornano D2H in `host_rows`; stampa all'avvio le misure del formato (`q8 layout: 144 B a block, 20 column
blocks a row, 2880 B a row, tail 18432 B`) e cade subito se `q8_bytes` non è `rows × rb + coda`;
`memcmp` riga per riga contro il riferimento, e le **somme pesate per token** in ordine `k` ricalcolate sull'host da
entrambi gli insiemi di righe, confrontate con `memcmp`. Per ogni coppia di tipi stampa `whole_tiles` dei due
prodotti (§5). Per strato stampa il tempo del server (eventi sul suo stream: H2D del blocco, prodotti, D2H delle
righe) e i MiB mossi: una misura della via per il modello di costo.

```
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0,1,2
peer_prefill2_test 0 0                 (stessa scheda: la via per host da sola; deve passare per primo)
peer_prefill2_test 0 1                 (padrona 3080 Ti, secondo server 3060; T 512, 32 esperti, 4 pezzi)
peer_prefill2_test 0 1 2048 64 4       (pezzo più grande, più esperti: multipli di 16)
peer_prefill2_test 0 2                 (contro la 2080 Ti: atteso FAIL con relativo ~1e-7, sm_75: §5)
```

Esito: `PASS: 0 chunk-layers failed` o `FAIL` con la prima riga diversa, `max |diff|` e massimo relativo per strato;
77 senza MMQ o senza due device. Un FAIL con relativo ~1e-7 («last bits, another float order») è un regime di
lancio o un'architettura diversa (§5), non la via; un FAIL con relativo grande è un errore della via (il taglio del
blocco, i bounds relativi, le code). `n_experts` non multiplo di 16: la nota del banco avverte che l'ultimo gruppo
della padrona è parziale e può differire.

## 7. Non provato, rischi, dove la compilazione può fermarsi

Compilato no, installato no, provato no.

Punti da guardare per primi in compilazione:
1. `prefill.cpp` `peer_layer`: lambda con cattura `[&]` che usa `err`, `m`, `T`, `l`, `lay`, `f`-less (la `f` è
   dichiarata dentro), `grp_mapped`, `src_h`, `slot_h`, `pe`, `p_release_to`, `p_issue_until`, `had_d`, `had_block`,
   `mmq_gt/dt/gub/db`; dentro al ramo `sums && gather` esiste una `size_t at` locale (per questo il parametro si
   chiama `r_at`); il `return false` dentro restituisce dalla lambda e il chiamante fa `return false`.
2. `for (int kind : {3, 1, 2})` con la guardia `!on_peer.empty()` nella condizione del ciclo interno.
3. `p2_plan_layer`: `struct Cand` locale alla lambda, `std::sort` con lambda, `ptrdiff_t` nei `fill`.
4. `set_peer2`: `measure_h2d_gbs` in namespace anonimo dopo `set_peer` (usa `core::OnDevice`); `env_bw_gbs`,
   `env_double`; `peer_had_signs` è nel namespace anonimo precedente (visibile). `Prefill::peer2_prefill_env` statica.
5. `generate.cpp`: `ep_stage` (unique_ptr) e `primary_device` nello scope del `set_peer`; `strata::core::OnDevice`.
6. Il banco: `mmq::Product` con `xq` `const void*` da `(const uint8_t*) xq + …`; `strata::kernels::iq_row_bytes`,
   `native_expert_layout`; `MMQ_GROUP`/`MMQ_TAIL` ridefiniti localmente (quelli di `prefill.cpp` sono file-local).
7. `cudaMemcpy2DAsync` D2H verso memoria mappata (pinned): lecita; se il driver la rifiutasse su una scheda, il
   ripiego è `xq_kb` copie 1D per gruppo (stessi byte, più chiamate).

Storia: la prima stesura (3/11 notte) **compilava pulita** (motore e banco, sha256 `1da6f485`) ma il banco cadeva
in tutti i casi su `q8 rows are not a fixed stride (21312, 24192)`: il formato è a blocchi di colonna (§1 punto 3)
e la coda fissa di 18 432 B faceva fallire il controllo `q8_bytes(2) = 2 × q8_bytes(1)`. Seconda stesura (build
`60ab8dc3`, compilata pulita): layout riconosciuto, ma righe diverse dalla 128 in poi sulla stessa scheda (30 FAIL su
40 su `0 0`, 10 su 40 sulla 3060, 40 su 40 sulla 2080 Ti dalla riga 0): il regime di lancio stream-k dei gruppi
piccoli (§5). Terza stesura: lancio del server imbottito a 16 esperti, riferimento del banco a gruppi pieni
(`n_experts` 32), massimo relativo stampato. Di nuovo **non compilata e non provata**; il banco deve passare prima
su `0 0`, poi su `0 1`; su `0 2` ci si aspetta FAIL con relativo ~1e-7 (sm_75).

Misura utile arrivata con la seconda stesura (M): sul riser la 3060 fa **0,37–0,38 GB/s** compresi i prodotti
(113 MiB in ~295 ms per strato a 9 040 righe); la padrona su se stessa 5,5 GB/s. Quarta stesura (dopo la prova nel
motore con `4512b587`, banco tutto PASS): misura di banda con eventi su 8 copie, anello chiuso per pezzo (§2.2),
costo per riga misurato nel piano. Prova con `076e7d90` (M): bande vere, fattore a 0,03-0,04, stallo 2-3 ms, ma
lettura 878/1274/1297 contro 963/1376/1464 spenta. Quinta stesura (`4d38da47`, §2.3): costo fisso per strato separato,
tempo host dell'emissione misurato e filtro per strato, ON/OFF su due EWMA per token del pezzo intero — sceglieva ON
a torto (803/1182/1290). Sesta stesura: misura della sola parte MoE, coppie di pezzi adiacenti a T uguale, si parte
spenti, prove ogni 16 da spenti e controlli ogni 8 da accesi, mai sul pezzo 0 di un prompt. **Non compilata, non
provata.** `STRATA_PEER2_PREFILL_ROW_US` resta a 1: il costo vero lo misura l'anello.

Rischi funzionali:
- **Bit fra schede** (§5): se il banco dice che le righe della 3060 differiscono negli ultimi bit per i tipi senza
  `whole_tiles`, la lettura con la variabile non è bit per bit quella senza. È un fatto da mettere nel confronto
  (KL / testo), non da nascondere.
- **La stima di `c`** sbaglia al primo pezzo e dopo un cambio di argomento: un esperto stimato freddo e poi caldo
  costa al server `c_vero × 27 µs` sul riser (a 1000 righe: 27 ms su uno strato il cui budget era ~100 ms: assorbito;
  a 5000 righe: 135 ms, il server finisce dopo la padrona per quello strato). Il budget con `cnt_prev` lo limita in
  media, non nel caso singolo. `STRATA_PEER2_PREFILL_BUDGET=0.5` per stare larghi.
- **Il link della padrona è condiviso**: i byte (a) stanno sullo stesso Oculink dei blob; il modello li conta, ma la
  D2H del blocco e la lettura mappata delle righe sono in concorrenza con l'anello: se l'anello non era il collo, il
  guadagno è minore della stima.
- **VRAM della 3060**: ~470 MB per istanza, due con la tappa 4b (§4); con `--peer-reserve-mib A,B` troppo basso
  `set_peer2` fallisce con una riga e lo stadio legge come prima.
- **`ps_frac` del primo server**: il 35 % degli esperti in streaming della padrona che il primo server si prende passa
  **dopo** il piano del secondo (il `continue` del piano viene prima): il primo non vede più gli esperti del secondo.
  Voluto.
- **Due `PeerPrefill` sulla stessa scheda** (tappa 4b): stream separati, buffer separati, nessun dato condiviso;
  il modello di ciascuno non sa del traffico dell'altro sul riser (vedi §9).

## 8. Tappa 4b (`STRATA_EP_SPLIT_PREFILL=1`) con due server: che cosa ho controllato

Letto `generate.cpp` su `020c723e` con `peer2` presente, senza e con la variabile nuova:

- **Apertura**: `peer2.open` viene dopo `peer.open` e dopo lo stadio EP (che sta sulla scheda del **primo** server:
  `ep.dev = o.peer_device`). Con `--peer-device 1,2` lo stadio EP va sulla GPU 1: se è la 3060 sul riser, lo stadio
  `[K, 48)` gira sulla scheda lenta — legale, lento. **Mettere prima la scheda veloce** (`--peer-device 2,1`). Il
  `reserve2` è indipendente da `ep_reserve_add` (giusto: lo stadio non tocca la seconda scheda).
- **Prestito/`refill`**: `pf_parts` ha la sola parte del primo server (`peer.cache_mut()`, `peer.res_table_mut()`);
  `lend`/`refill_issue` non toccano `peer2`. Le coppie di `peer2` non entrano in nessuna coda prestata: non possono
  essere sovrascritte da un `relayout`. Nessun conteggio di slot del secondo server è coinvolto. Corretto senza
  modifiche.
- **`res_table` e finestre**: `peer2.fused_set_other_res(peer.res_table())` — il puntatore non viene mai riallocato
  (`res_` dimensionato a `open`); durante il prestito le voci prestate del primo sono `kNotResident`, ma nessuna
  finestra è aperta durante un prompt; a `fused_begin` (dopo il `refill`) la tabella del primo è tornata intera, e
  `own_win_` del secondo la legge allora. Corretto.
- **`quiesce`**: oggi solo `peer`; ho aggiunto `peer2.quiesce` negli stessi due punti (`lend` della parte EP e prima
  del passaggio di stato): con `STRATA_PEER2_ADAPT=1` e la variabile nuova gli stadi leggono gli slot del secondo
  (`slot_ptr`) e una copia dell'adapt non deve essere in volo. Senza adapt e senza la variabile è un `apply_pending`
  vuoto e due `cudaStreamSynchronize` su stream fermi.
- **Rifiuti**: `ep_split::admit` non sa di `peer2` e non deve: nessuna combinazione nuova è insensata. L'unica
  combinazione che rifiuto con una riga è `STRATA_PEER2_PREFILL=1` + `STRATA_EP_SPLIT_LM=1` (nessuna forma
  layer-major con una quota su un server). `--batch`, `--prefill-head-alias`, `--layer-split` sono già rifiutati a
  monte.
- **Esito della chat principale** su `020c723e` con `--peer-device 1,2`: non arrivato mentre scrivevo; se riporta un
  rifiuto o una caduta con `peer2` e la tappa 4b, il punto da guardare per primo è l'ordine dei device (sopra) e la
  riga `second expert server on GPU B not opened` (VRAM della seconda scheda dopo il desktop).

## 9. Multichat (non fatto, non chiuso)

Due conversazioni che leggono in parallelo avrebbero due `Prefill` e quindi due `PeerPrefill` per server, ognuno con
stream, buffer e pagine proprie: nulla in questo disegno è un singleton (le bande misurate e `cnt_prev` sono per
istanza). Due cose da aggiungere allora, non ora: (1) un contatore condiviso del carico del link di ciascun server
(oggi ogni istanza pianifica come se fosse sola: due istanze sul riser si prenderebbero ciascuna il budget intero);
(2) la VRAM del server per istanza (~470 MB a T = 7680), da contare in `--peer-reserve-mib B`. La precedenza di
proprietà (padrona → server 1 → server 2) e il taglio per righe non cambiano con più conversazioni.

## 10. Comandi di prova nel motore (chat principale)

```
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=0,1,2
set STRATA_EXP_MODE=8
set STRATA_PEER_FUSED=1
set STRATA_PEER2_PREFILL=1
set STRATA_PREFILL_TIMING=1
strata ... --peer-device 2,1 --peer-reserve-mib 3000,2000 ...
```

Attesi: la riga `peer GPU 1 (second server) computes the rows of its experts that the cost model sends it (… its link
0.4x GB/s (measured), the primary's 6.x GB/s (measured) …)`; a fine prompt la riga `second server GPU 1: … planned P of
C candidate pairs over L layers`; la lettura (t/s a 4K/32K/128K) contro la stessa corsa con `STRATA_PEER2_PREFILL=0`
(riferimenti del 3/11: due server in lettura 931-954 / 1298-1375 / 1425-1464); il testo confrontato con KL dove il
banco ha detto che i bit differiscono, con `memcmp` dove ha detto che sono uguali. Poi con `STRATA_EP_SPLIT_PREFILL=1`
(tappa 4b): due righe di apertura del secondo server (CUDA0 e lo stadio EP), riserva B a 2000. Variabili per le
misure: `STRATA_PEER2_PREFILL_BUDGET=0.5` e `=2`, `STRATA_PEER_PREFILL_BW_GBS=,6` per simulare l'Oculink sulla 3060
prima di averlo (il modello manderà più esperti; il tempo vero del riser dirà quanto il budget contava).
