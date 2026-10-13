# PIPELINE2: le due schede davvero in parallelo — analisi e disegno

Branch `lab/pipeline2` del fork `imanu86/Strata-2080Ti`, da `lab/struct-a` (`d21b2870` = codice `075ce875` + doc).
Configurazione di riferimento: 3060 = stadio 0 (CUDA0 con `CUDA_VISIBLE_DEVICES=1,0`, layer 0-15, Oculink PCIe 4.0 x4,
pilota il monitor 8K), 2080 Ti = stadio 1 (layer 16-47, testa, catena MTP, x16), `--pipeline-windows 2 --spec 4
--spec-min-p 0.5`, Daily 0.1.43 (`STRATA_SA_BET/SA_CHAIN` e leve speed).

Convenzione: **M** = misurato (micro-bench, log del motore, tracce), **S** = stimato (byte, banda, proporzioni). I ms
«a finestra» sono sulla media L02 del banco forzato a 131k; le classi fresche/speculative vengono da `STRUCT_B.md` §2b
(28,3 / 19,8 ms, 72% / 28%, media ~25,9 sul banco del 7/10; oggi con struct-a ~24,4 ms a 70 t/s).

## Stato

| fase | stato |
|---|---|
| 1. analisi e disegno (questo file) | **fatto** il 13/10; micro-bench del salto fra le schede **misurato** (§2a); modello di residenza dal trace (§2b); P-state: log al secondo letti, campionamento fitto **da fare** (§2c, comando pronto) |
| 2. report all'owner | consegnato il 13/10; **via del coordinatore** il 13/10 notte per le tappe 0a → 4 del §5 (opt-in, bit per bit). Restano fuori expert-parallel, tensor-parallel, stadi invertiti, B ad albero |
| 3. implementazione a tappe | ripresa il 13/10 alle 03:25 (crediti API). 0b e 2a **compilate e verificate a testo identico** (build `build-pipeline2-cd3bd31b`); 0a e 1 in corso. Diario e ripartenza nel §8 |
| 4. stima TP bivalente (richiesta dell'owner) | **fatta**, §9: TP senza MTP ~50 t/s, TP con MTP ~76/66 (L01/L02) = pari a oggi → **no** |

**Build pronta per il B-A-B**: `corse_2080ti\20261004_routed_keep\build-pipeline2-cd3bd31b\` (codice `e76d96dc`,
`strata.exe` sha256 `3b884217…605e1e`, `build.json`). Compilato sì (build_lock, CUDA 12.6 sm_75+86, 13/10 04:05).
Provato sì, solo a testo (§8, tappa 0b): L02 da 256 token identico a spento in tutti i bracci. Installato no. B-A-B no.
Braccio A = env della Daily 0.1.43; braccio B = A + `STRATA_P2_SMEM=1`; a parte A + `STRATA_P2_FLUSH_NOSWITCH=1`.

### Ripartenza (per un agente senza contesto)

1. `cd C:\Users\imanu\source\repos\Strata-wt\pipeline2` (branch `lab/pipeline2`): `git status`, `git pull --ff-only`.
2. *(fatto il 13/10: `build-pipeline2-cd3bd31b`)* Build: `python <corse_2080ti>\20261006_due_schede\build_lock.py C:\Users\imanu\source\repos\Strata-wt\pipeline2`
   (la cartella `build-sm7586` è già configurata: è incrementale). Se compila, copiare `build-sm7586\strata.exe`
   (+ `strata-vision.exe` da `build-structa-075ce875`, `source\serve`, `source\tools`, come le build precedenti) in
   `<corse_2080ti>\20261004_routed_keep\build-pipeline2-<sha>\` con `build.json` (`status`, `executable_sha256`, `note`).
3. *(fatto il 13/10: testo identico, §8)* Verifica del testo: `python D:\ds4_work\strata\pipeline2\p2_check.py build-pipeline2-<sha> DET-p2-chk1 256
   off,stat,nosw,smem,both` (L02 a cache fissa 1500, senza `--elastic`, `--adapt-every 0`, env della Daily 0.1.43; ogni
   braccio prende da sé il `bench.lock`). Tutti i bracci devono dare `L02.txt` e primi logit IDENTICI a `off`. Leggere le
   righe `strata pipeline2 switches:` (bracci `stat`/`nosw`) e `strata pipeline2: STRATA_P2_SMEM` (bracci `smem`/`both`).
4. Tappa 0a: `bash D:\ds4_work\strata\pipeline2\p2_pstate.sh DET-p2-pstate3` (il primo tentativo `DET-p2-pstate1` è
   morto per «Startup deadline» a 300 s dopo la notte multi-chat: riprovare a PC fermo).
5. Tappa 1: `D:\ds4_work\strata\pipeline2\nsys_base.bat DET-p2-nsys-base` (base) e la stessa con
   `build-pipeline2-<sha>` e `--env STRATA_P2_SMEM=1` come 3° argomento e seguenti (cattura con la leva); poi
   `python tools\graph_census.py D:\ds4_work\strata\nsys\<nome>.sqlite`. Decide la tappa 2a: lo spigolo
   `gr_down_staged → gr_up_multi` della 2080 Ti deve sparire o calare.
6. Al coordinatore per il B-A-B: braccio A = env della Daily 0.1.43; braccio B = A + `STRATA_P2_SMEM=1` (e, a parte,
   A + `STRATA_P2_FLUSH_NOSWITCH=1`). Solo dopo la verifica del testo.
7. Poi le tappe 2-3 del §5 (meno nodi nel `post()`), scelte sul censimento del punto 5.

## 0. Conclusione

- **Il salto fra le due schede costa poco in latenza, molto in byte.** Senza P2P (WDDM, Ampere ↔ Turing: `peer access
  0` in entrambe le direzioni, M) ogni salto passa dalla RAM mappata. Andata e ritorno con il solo flag: **7-8 µs** fra
  due kernel persistenti, **11,5 µs** come nodi di grafo (M). Con i dati di un layer expert-parallel a T=2 (6 KB di
  ingresso q8_1, 80 KB di righe fp32 al ritorno): **28 µs** fra kernel persistenti, **36-46 µs** come nodi di grafo; a
  T=4 (200 KB al ritorno) **70 µs** esposti per layer (M). Le copie via copy engine con eventi fra schede costano
  **90-130 µs** (M): non servono.
- **Gli esperti non si dividono senza togliere VRAM a qualcuno.** La VRAM delle due schede è piena di esperti e la
  coda della residenza vale molto: per dare alla 3060 il 30% del traffico degli esperti dello stadio 1 servono ~1,5 GB
  di esperti caldi, che tolgono **10 punti di hit** allo stadio 0 (S, dal trace di routing, §2b). Con quote piccole
  (0,4 GB, 11% del traffico) il guadagno sullo stadio 1 (−0,45 ms) è pareggiato dai miss in più sullo stadio 0.
- **Il resto non è banda.** In una finestra fresca ~18 ms sono kernel che leggono pesi (divisibili fra le schede) e
  **~10 ms sono latenze in serie** che nessuna divisione del lavoro tocca: attese dei flag dell'host 3,8, catena prima
  dello stadio 0 2,1, QSA (latenza) 1,9, buchi fra i nodi 1,3, bolla del lancio ~0,8 (M, §1). Le finestre speculative
  (19,8 ms) sono già il «max dei due stadi»: la pipeline B dà il parallelo nel 28% delle finestre.
- **Stime delle alternative** (ms a finestra sulla media, §3-4): expert-parallel pieno #1798 **−0,5/−2**, ibrido
  #1461 **−0,1/−0,5**, micro-finestre **+5/+7 (peggio)**, tensor-parallel **−1/−3** con 4-6 settimane, ordine degli
  stadi invertito **−0,5/−1**, B ad albero **−0,3/−1** (un fork l'ha già provata: nessun guadagno). Nessuna
  alternativa «parallela» arriva a −3 ms (80 t/s su L02) in giorni.
- **Raccomandazione** (§5): non riscrivere il motore per l'expert-parallel. Lavorare sulla **serie**, che pesa su tutte
  le finestre: tappa 0 a costo quasi zero (P-state, flush senza `cudaSetDevice`), poi il **`post()` in un lancio
  persistente per layer** (meno nodi, attese dei flag dentro il kernel) su entrambe le schede, misurabile a tappe e bit
  per bit. Atteso **−0,7/−1,5 ms a finestra** (S), più la tappa 0. L'expert-parallel torna conveniente solo se cambia la VRAM (una scheda
  da 16-24 GB al posto della 3060): §6.

## 1. Da dove partiamo: chi è ferma, e quanto

Finestra fresca (72%, M, `STRUCT_B.md` §2b, `ROOFLINE` §1d): verdetto → A decisa (catena) **2,10** · stadio 0 di A
**10,07** · F0 → L1 **0,17** · stadio 1 di A **15,86** = **28,3 ms**. Finestra speculativa (28%): verdetto → stadio 1
0,49 · stadio 1 **19,21** (catena forzata sulla stessa scheda) = **19,8 ms**.

| | 3060 | 2080 Ti |
|---|---|---|
| fresca: cosa fa | ferma 2,1 · stadio 0 di A 10,1 · stadio 0 di B ~10 dentro lo stadio 1 di A · ferma ~5,8 | catena 2,1 (critica) · resto della catena ~4 · **ferma ~6** · stadio 1 15,9 |
| speculativa | stadio 0 di B ~10 · ferma ~10 | stadio 1 + catena forzata 19,2 |
| GPU in calcolo (nsys, M) | 49% | 62% |
| entrambe in calcolo (M) | **24%** del tempo | |

Composizione della fresca (S da M, tempi puliti; ROOFLINE §1b-1d scalato dai muri nsys):

| voce | ms | divisibile fra le schede? |
|---|---|---|
| kernel che leggono pesi: stadio 0 (densi, hc, esperti VRAM) | ~7,3 | sì, in linea di principio |
| kernel che leggono pesi: stadio 1 (densi, hc, esperti VRAM, testa) | ~10,8 | sì, in linea di principio |
| attese della GPU sui flag dell'host (piano, righe CPU) | ~3,8 (2,0 + 1,8) | no: dipendono dal pool CPU |
| catena prima dello stadio 0 (verdetto, catena, lancio) | 2,1 | no |
| QSA (top-k a un CTA per riga, attenzione a catena) | ~1,9 | poco: latenza, non banda |
| buchi fra i 1397 nodi + bolla del lancio | ~2,1 | no (sono nodi, non byte) |
| F0 → L1 | 0,17 | — |

Il tetto di qualunque parallelismo è dimezzare i ~18 ms della prima metà: ~9-10 ms, in pratica ~12 (kernel piccoli con
pavimento di latenza: hc da 17-29 µs contro 10,6 di minimo). Le ~10 ms della seconda metà restano.

## 2. Misure nuove

### 2a. Il salto fra le schede (`tools/hop_bench.cu`, M)

Build: `nvcc -O2 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86`, eseguibile in
`D:\ds4_work\strata\pipeline2\hop_bench.exe`; corsa sotto `bench.lock` il 13/10, PC in uso, monitor 8K acceso
(`hop_bench_run2.txt`; la `run1` aveva copie a 4 byte per accesso: 2-5× più lente, scartata). Memoria host
`cudaHostAlloc(Portable|Mapped)`; scritture `__stwt` da 16 byte + `__threadfence_system` + flag; letture `__ldcv` da 16
byte (come `copy_from_mapped` del motore).

`cudaDeviceCanAccessPeer`: **0** in entrambe le direzioni (M): niente P2P, come atteso su WDDM con due architetture.

**1. Andata e ritorno fra due kernel persistenti** (un blocco da 512 thread per lato, 2000 giri; p50/p99 in µs):

| payload | avvia la 2080 Ti | avvia la 3060 |
|---|---|---|
| solo flag | 8,2 / 9,6 (7,0 con `__nanosleep`) | 7,2 / 7,2 |
| 6 KB in andata (ingresso MoE q8_1, T=2) | 9,8 / 10,3 | 8,2 / 9,2 |
| 80 KB al ritorno (8 righe esperto fp32) | 26,7 / 28,7 | 25,6 / 26,6 |
| EP T=2: 6 KB + 80 KB | **28,7 / 30,5** | 27,6 / 28,7 |
| EP T=4: 12 KB + 200 KB | 58,3 / 59,4 | 56,3 / 57,3 |
| TP T=2: 20 KB per verso | 16,6 / 18,2 | 16,4 / 18,4 |
| hand-off T=2: 100 KB per verso | 54,1 / 55,3 | 54,3 / 55,3 |

Un salto col solo flag vale ~3,5-4 µs; 80 KB aggiungono ~19 µs (≈4,3 GB/s effettivi sul percorso 3060 → RAM →
2080 Ti con un blocco solo). Nessuna differenza di rilievo fra le due direzioni: con accessi da 16 byte l'Oculink non è
il collo, lo è la latenza delle letture mappate.

**2. Un layer expert-parallel come nodi di grafo** (la forma del motore: grafo da 48 layer su ciascuna scheda, epoca
sul device, attesa fusa nella copia come `FUSE_WAIT`; µs per layer; «esposto» = scambio − stessa scheda da sola):

| densi (2080 Ti) | esperti propri | esperti della 3060 | ritorno | baseline | con scambio | **esposto** |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 80 KB (attesa e copia separate) | 2,8 | 38,7 | 36,0 |
| 0 | 0 | 0 | 80 KB (fuse) | 2,6 | 48,1 | **45,5** |
| 0 | 0 | 0 | solo flag | 2,6 | 14,1 | **11,5** |
| 0 | 0 | 0 | 20 KB (righe sommate sulla 3060) | 2,6 | 22,9 | 20,3 |
| 200 | 50 | 0 | 80 KB | 252,8 | 293,7 | 41,0 |
| 200 | 50 | 80 | 80 KB | 332,7 (50+80 sulla 2080 Ti) | 349,1 | **+16,4** |
| 200 | 80 | 80 | 80 KB | 362,6 | 352,3 | −10,3 |
| 200 | 50 | 80 | 20 KB | 332,6 | 321,0 | −11,6 |
| 200 | 50 | 80 | 40 KB (righe bf16) | 332,6 | 327,6 | −5,0 |
| 200 | 50 | 80 | T=4: 200 KB | 332,7 | 403,1 | **+70,4** |
| 200 | 50 | 80 | T=4: 40 KB | 332,6 | 334,8 | +2,2 |

Lettura: **la catena di uno scambio con righe fp32 vale ~36-46 µs** (≈ i 26 µs di #1798 su PCIe 5 + Oculink e copie
in più); si nasconde solo se la scheda proprietaria ha ≥ 40-50 µs di lavoro suo da fare intanto. Le righe fp32 (quelle
che tengono il combine bit per bit) a T=4 non si nascondono più (+70 µs × 48 layer = 3,4 ms). Righe sommate o bf16
dimezzano il conto ma cambiano i bit.

**3. Copy engine** (D2H sulla prima, evento fra schede, H2D sulla seconda e ritorno): 90-130 µs a giro (M). Fuori
discussione per uno scambio per layer.

### 2b. Quanto traffico di esperti costa quanta VRAM (`tools/ep_residency.py`, S da trace)

Trace `D:\ds4_work\strata\f0\routing_addestra.bin` (`--dump-routing`, 514 848 record = ~10 700 token × 48 layer);
residenza greedy per frequenza dentro lo stadio, con gli slot di oggi (3754 sulla 3060, 9149 sulla 2080 Ti, blob medi
1,50 / 1,87 MB). In-sample, quindi hit ottimistici in assoluto (84% / 94% contro il 97% del motore col tier adattivo):
contano le differenze fra gli scenari. Output in `D:\ds4_work\strata\pipeline2\ep_residency_addestra.txt`.

- Il routing è **piatto**: i 10% di esperti più caldi di un layer portano il 43% del traffico, il 50% ne porta l'89%.
  Nello stadio 1 il 10% del traffico sta in 172 coppie (0,31 GB), il 20% in 456 (0,83 GB), il 30% in 833 (1,52 GB).
- **(b1) la 3060 tiene gli esperti caldi dello stadio 1** e scarta la coda dei suoi (stessi byte); la 2080 Ti usa gli
  slot liberati per la coda dello stadio 1:

| coppie spostate | GB | quota di traffico dello stadio 1 sulla 3060 | hit stadio 0 | hit stadio 1 | CPU in più per layer-finestra, stadio 0 / 1 |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 83,6% | 94,2% | — |
| 200 | 0,37 | 11% | 81,4% | 94,6% | +0,45 / −0,09 |
| 400 | 0,73 | 18% | 78,9% | 95,0% | +0,93 / −0,17 |
| 800 | 1,46 | 29% | 73,4% | 95,8% | +2,04 / −0,32 |

- **(b2) il contrario** (la 2080 Ti tiene gli esperti caldi dello stadio 0): 0,29 GB → 16% del traffico dello stadio 0,
  hit 83,6 → 85,3% / 94,2 → 93,8%.

### 2c. P-state durante il decode (voce 6 del coordinatore, #1748)

- Dai log del 11/10 (`D:\ds4_work\strata\lab_runs\gpulog-upsync*`, un campione al secondo, M): durante il decode la
  3060 è in **P0, memoria a 7501 MHz** nel 94-97% dei campioni e in P3 (5001 MHz) nel 3-6%; la 2080 Ti sempre **P0,
  7000-7250 MHz**, SM a 1800-1875 MHz. Nessun P5 a 810 MHz con il motore al lavoro (2 campioni su ~290, ai bordi).
  Da ferma la 3060 sta in P5/810 MHz e il link scende a PCIe gen 2 (x4).
- Campionamento fitto (`nvidia-smi -lms 10`) durante una richiesta: **non ancora fatto**. Il 13/10 notte il
  `bench.lock` era della coda multi-chat del coordinatore (`DET-mc2-*`, ~14 min a braccio) e la sonda è scaduta in
  attesa senza girare. Comando pronto (~4 min, sotto `bench.lock`):

  `cd D:\ds4_work\strata\pipeline2 && python <corse_2080ti>\20261006_due_schede\bench_lock.py pipeline2-pstate --
  python pstate_probe.py <corse_2080ti>\20261004_routed_keep\build-structa-075ce875 P2-pstate1 700`

  Uscite `P2-pstate1_smi.csv` (pstate, clock SM/memoria, utilizzo, link ogni ~10 ms) e `P2-pstate1_result.json`
  (istanti della richiesta). Da leggere: quota dei campioni sotto P0 o con la memoria < 7000 MHz fra «request» e
  «done», per scheda.

## 3. Le alternative

Per ognuna: finestra minima stimata, trasferimenti (dal §2a), VRAM, bit, giorni. Base: fresca 28,3 / speculativa 19,8
/ media 25,9 ms (M).

### (a) Expert-parallel pieno, #1798 portato su CUDA

**Disegno.** La 2080 Ti esegue tutti i 48 layer (densi, router, combine, KV, GDN), la testa e la catena MTP; la 3060
diventa un server di esperti: per ogni layer riceve id e ingresso q8_1 (6-12 KB), calcola gli esperti che tiene e
scrive le righe fp32 nei `parts_` della 2080 Ti in memoria mappata, con un flag nello stesso kernel. Partizione per
traffico a coppie alternate (come #1798), ma pesata per velocità: la 3060 tiene la coda fredda (più byte, meno
traffico), la 2080 Ti gli esperti caldi. Niente stadi, niente pipeline B.

**Tempo (S, T≈2, tempi puliti).** Per layer: densi sulla 2080 Ti ~0,215 ms (dallo stadio 1: (12 − 4,1 esperti − 1,0
testa)/32) + esperti. La 3060 tiene ~44% dei byte degli esperti residenti ma, con la coda fredda, ~18-25% del traffico:
proprietaria ~95 µs, 3060 40 + 1,22 × 25 ≈ 70 µs → **la 2080 Ti resta il collo**, lo scambio è nascosto. 48 layer ×
0,31 = 15,0 + testa 0,9 + buchi (48 layer + nodi EP, ~2,2) + bolla ~1,2 + attese CPU (ora tutte su un percorso: ~3) +
catena 2,1 + verdetto 0,1 ≈ **24,5 ms per tutte le finestre**. Contro 25,9 di media oggi: **−0,5/−2 ms** (−2/−8%).
- Perché meno di #1798: là tutti gli esperti stavano in VRAM e gli esperti erano il 22% della finestra, la scheda 0
  era identica alla N; qui si perde la pipeline B (finestre speculative a 19,8 ms) e i densi dello stadio 0 passano
  sulla 2080 Ti senza che la 3060 faccia altro che esperti (49% → ~25% di occupazione).
- Trasferimenti: 48 scambi a finestra × 6 KB + 60-80 KB = ~4 MB a finestra; esposti ~0 se la proprietaria ha più
  lavoro, +0,5/+3 ms se il bilanciamento sbaglia o a T=4 (§2a: +70 µs per layer con righe fp32).
- VRAM: la 2080 Ti prende i densi dello stadio 0 (1,13 GB), K/V e indicizzatore dei 4 QSA (~0,2 GB) e gli stati GDN;
  la 3060 libera altrettanto per gli esperti. Somma degli esperti residenti invariata (S). #1798 spegne gli scambi
  adattivi e i prestiti al prompt; da noi la cache elastica della 3060 (monitor 8K) andrebbe ripensata.
- Bit: gli esperti calcolati sulla 3060 (sm_86) invece che sulla 2080 Ti cambiano i bit se il kernel ha rami per
  architettura; le righe fp32 nel combine di sempre tengono l'ordine. Deterministico, non uguale a oggi.
- Prefill: oggi a layer split; con tutti i densi sulla 2080 Ti gli esperti dello stadio 0 non sono più sulla scheda
  che fa il prefill di quei layer (#1798 li legge dal GGUF o da 18 GB di RAM pinned): TTFT da rifare.
- **Giorni: 15-25.** Rischio alto. **Sconsigliata.**

### (b) Ibrido, #1461: layer split + una quota di esperti dell'altro stadio

**(b1) La 3060 calcola una quota degli esperti dello stadio 1.** Dentro lo stadio 1 di A la 3060 gira già lo stadio 0
di B (5,8 ms di margine, M): la quota va su uno stream ad alta priorità della 3060, accanto a B. Con la quota a f del
traffico: proprietaria (1−f)·E, 3060 c + 1,22·f·E (E = 128 µs per layer a T=2 con EXPDEC, M da `EXPERTDEC` §7; c =
36-46 µs, M; 1,22 = rapporto µs/MB degli esperti 3060/2080 Ti, M). Con la VRAM del §2b:
- 0,37 GB (f = 11%): stadio 1 −14 µs × 32 = **−0,45 ms** (tutte le finestre) e −2,8 esperti CPU; stadio 0 +7 esperti
  CPU a finestra (**+0,4/+0,6 ms** nelle fresche, ×0,72). Saldo **~−0,1/−0,3 ms**.
- 0,73 GB (f = 18%): −0,74 −0,3 +0,7 ≈ **−0,35 ms**.
- Oltre, lo stadio 0 crolla (−10 punti di hit a 1,5 GB).
- Trasferimenti: come (a) ma solo sui 32 layer dello stadio 1. Contesa con B sulla 3060: B più lenta di ~1-2 ms (ci
  sta nel margine di 5,8 ms, p10 1,9 ms).
- Bit: come (a). **Giorni 8-12** (grafo helper per T sulla 3060, piano a ruoli in `resident_plan`, tabelle di
  residenza a tre stati, tier adattivo ed elastico). **Sconsigliata: −0,1/−0,5 ms.**

**(b2) La 2080 Ti calcola una quota degli esperti dello stadio 0 delle fresche** (è ferma ~6 ms): 0,29 GB → 16% del
traffico dello stadio 0; −19 µs × 16 layer = −0,3 ms e −5 esperti CPU (−0,35 ms) per fresca, +2,4 esperti CPU sullo
stadio 1; ma lo stadio 0 di B (dentro lo stadio 1 di A) ha bisogno degli stessi esperti → helper a bassa priorità sulla
2080 Ti critica (+~0,3 ms nell'89% delle finestre). Saldo **~0/−0,2 ms**. Sconsigliata.

Il §2b dice però una cosa utile a parte: spostare 0,3-0,6 GB di slot **dalla 2080 Ti allo stadio 0** alza l'hit dello
stadio 0 molto più di quanto lo abbassi nello stadio 1. È la stessa leva dello split (14 era −0,6%, M), ma per slot:
da guardare con i numeri veri del motore (hit per stadio), non con il trace.

### (c) Micro-finestre e riordino

- **(c1) Finestra spezzata in righe che scorrono fra gli stadi.** No (M, `STRUCT_A` §2a): un passaggio per pezzo
  rilegge i pesi (stadio 0: T=1 + T=3 = 20 ms contro 14,4 di T=4) e ogni pezzo in più costa un passaggio dello stadio
  1 (~13 ms fissi). **+5/+7 ms.**
- **(c2) Ordine degli stadi invertito** (2080 Ti = layer 0-31, 3060 = layer 32-47; testa e catena restano sulla 2080 Ti
  dopo un salto di ritorno da 10 KB). Fresche invariate (~28 ms: stessa somma); speculative limitate dallo stadio 0 di B
  sulla 2080 Ti (~14,4 + catena) invece che dallo stadio 1 di A (19,2): **−2,4 ms per speculativa, ~−0,7 a finestra**
  (S). Contro: la 3060 tiene i layer con gli esperti più grossi (down IQ4_NL, 1,87 MB: ~3000 slot invece di 3754, hit
  più basso), gli snapshot GDN di B passano a 24 layer (75 MB) sulla 2080 Ti, e il motore assume testa e catena
  sull'ultimo stadio. **−0,5/−1 ms, 8-12 giorni.** Non prioritaria.
- **(c3) Stadio 1 di A e di B nello stesso passaggio** (B dentro A): è una finestra più lunga, e la pendenza per riga
  dello stadio 1 è 3,26 ms (M). Lo decide già `--spec-min-p`. No.

### (d) Altre idee (mie e della ricognizione del coordinatore, `RICOGNIZIONE_PR_FORK_2026-10-13.md`)

- **(d1) Tensor-parallel per righe d'uscita** (ogni matrice divisa 37/63 fra le schede, uscite riunite con
  all-gather; i K/V per testa). Il parallelo vero su ogni finestra. Stima (S): lavoro sui pesi ~16 ms-2080 Ti a
  finestra / 1,7 schede-equivalenti = 9,4 ms ideali, ~12 coi pavimenti dei kernel piccoli; + le ~10 ms di latenze in
  serie (§1) + 100-200 scambi da ~10 µs (§2a: 1-2 ms) ≈ **23-25 ms per tutte le finestre**, senza pipeline B: **−1/−3
  ms**. QSA ha 2 teste K/V (50/50, la 3060 diventa il collo dell'attenzione); il prefill va rifatto sulla stessa
  ripartizione (all-reduce di 42 MB a layer per pezzo da 4K sull'Oculink: +10-40 s su 131k); bit diversi
  (all-reduce). **4-6 settimane.** Sconsigliata ora.
- **(d2) B ad albero: un secondo candidato per la riga 0 di B** (il 69% delle B sbaglia proprio lì, `STRUCT_B`). Il
  ramo in più sta nel margine della 3060 (+0,13 ms per riga e layer), la sua catena nella 2080 Ti ferma delle fresche.
  Serve lo stato GDN a rami e una cella K/V per ramo nei 4 QSA dello stadio 0. Il fork `unimatrix099/bonus-branch` l'ha
  provata: seconda bozza giusta nel 31% dei casi, **nessun guadagno**; `STRUCT_B` stimava −0,2/−0,35 per la variante
  senza bozze. **−0,3/−1 ms, 8-12 giorni**, evidenza contraria. No.
- **(d3) Meno nodi per layer: `post()` in un lancio persistente** (fork unimatrix099, HIP: +3,5% su una scheda, +9% su
  due in pipeline; fronte FFN da 23 a 11 kernel). Da noi i buchi fra i 1397 nodi valgono 1,3-1,8 ms e la bolla del
  lancio ~1 ms sullo stadio 1 (M, `HOSTGAP`), più le attese dei flag spezzate in nodi a parte. Vale per **tutte** le
  finestre e per entrambe le schede: **−0,7/−1,5 ms a finestra** (S; `NODECUT` ha già tolto una parte dei nodi: il censimento della tappa 1 dice quanto resta). Bit per bit se ogni kernel fuso fa la stessa
  aritmetica nello stesso ordine. **6-10 giorni**, a tappe misurabili. **Raccomandata** (§5).
- **(d4) Hand-off senza sincronizzazione dell'host, #1120.** La patch del 6/10 si spegne con `--pipeline-windows`; da
  noi l'equivalente (`STRATA_HG_PRELAUNCH=2`: grafo dello stadio 1 lanciato in anticipo che aspetta una parola mappata)
  è già stato misurato **neutro** (B-A-B `DET-x5-hostgap`, `HOSTGAP.md`), e F0 → L1 vale 0,17 ms puliti (M). Il salto
  sul device costa ~4 µs (§2a). **≤ −0,2 ms.** Non serve come tappa a sé; il flag sul device è un mattone di (d3).
- **(d5) Polling della pipeline senza `cudaSetDevice`** (fork NTRYourWaifu v7/v8, `STRATA_FLUSH_NO_SWITCH`, +52
  righe; 88,2 → 90,6 t/s dichiarati). Il nostro `Verifier::service` (`verify.cpp:3481`) ha già la scorciatoia senza
  chiamate CUDA quando nessun doorbell ha suonato (upstream #1741), ma ogni 2 ms fa `OnDevice` (un `cudaGetDevice` e
  due `cudaSetDevice`) + `cudaEventQuery` + `cudaStreamQuery`; lo stesso nei polling della catena in `mtp.cpp`. Se una
  di queste chiamate blocca il thread mentre suona un layer, la GPU aspetta. Non è dimostrato che sia la causa delle
  nostre attese (`STRUCT_C`: la causa principale è il pool unico conteso con B), ma il porto è piccolo e bit per bit.
  **−0/−0,5 ms, 1 giorno.** Tappa 0.
- **(d6) Clock bloccati / `keep_warm.cu`** (#1748). I log al secondo dicono P0 e memoria piena durante il decode (§2c):
  il caso #1748 (attese di 12-22 ms fra stadi su TCP) non è il nostro (buchi di ~2-10 ms). Il campionamento fitto
  decide; se mostra cali sotto P0 dentro le finestre, B-A-B con `nvidia-smi -lgc/-lmc` (azione admin dell'owner) o un
  porto di `keep_warm.cu`.
- **(d7) Sweep dello split 13-18** (#2083, #2048). Lo split 14 era −0,6% (M) e il modello del §1 dice perché: ogni
  layer spostato sulla 3060 accorcia le speculative e allunga le fresche (~0,62 contro ~0,45 ms per layer). Utile come
  base del B-A-B del coordinatore, non come leva.
- **(d8) Catena MTP sulla 3060 / con una sola attesa per giro, #492.** La catena sulla 3060 è stata misurata il 7/10:
  **−9/−12%** (`docs/sm75/mtp-device-20261007.json`, analisi in `STRUCT_A` §2b). La catena fresca (tipo 2) è già
  accodata tutta al lancio, quella forzata va a passi (`STRATA_SA_CHAIN`, misurata −0,6/−1,1 ms per speculativa);
  CHAINFUSE (kernel fusi) +0,4%. No.

## 4. Tabella riassuntiva

| | finestra stimata (media) | Δ ms a finestra | trasferimenti per finestra | VRAM | bit | giorni |
|---|---|---|---|---|---|---|
| oggi (M) | 25,9 (fresca 28,3 · spec 19,8) | — | hand-off 51 KB × T | — | — | — |
| (a) EP pieno #1798 | ~24,5 per tutte | −0,5/−2 | 48 × (6-12 KB + 60-200 KB) | densi stadio 0 → 2080 Ti | deterministico, non uguale | 15-25 |
| (b1) 3060 aiuta lo stadio 1 | ~25,6 | −0,1/−0,5 | 32 × (6 KB + 30-80 KB) | 0,4-0,7 GB di esperti stadio 0 persi | come (a) | 8-12 |
| (b2) 2080 Ti aiuta lo stadio 0 | ~25,8 | 0/−0,2 | 16 × (6 KB + 80 KB) | 0,3 GB di esperti stadio 1 persi | come (a) | 8-12 |
| (c1) micro-finestre | ~31-33 | +5/+7 | — | — | — | — |
| (c2) stadi invertiti | ~25,2 | −0,5/−1 | hand-off + 10 KB di ritorno | esperti più grossi sulla 3060 | uguale | 8-12 |
| (d1) tensor-parallel | ~23-25 per tutte | −1/−3 | 100-200 × 10-20 KB | ripartita | diverso | 20-30 |
| (d2) B ad albero | ~25,3 | −0,3/−1 (fork: 0) | — | stato GDN a rami | uguale (testo verificato) | 8-12 |
| **(d3) post() persistente** | ~24,4-25,2 | **−0,7/−1,5** | — | — | **uguale** | 6-10 |
| (d4) hand-off sul device | — | ≤ −0,2 (prelancio già neutro) | — | — | uguale | 2-3 |
| **(d5) polling senza cudaSetDevice** | — | 0/−0,5 | — | — | **uguale** | **1** |
| (d6) P-state | — | da misurare | — | — | uguale | 0 |

## 5. Raccomandazione e tappe

**Strada raccomandata: accorciare la serie, non dividere i pesi.** Le misure dicono che lo scambio per layer costa
36-46 µs con righe fp32 e che la VRAM per dare esperti all'altra scheda non c'è; i ~10 ms di latenze in serie
restano in ogni variante parallela. La leva che tocca le latenze su **entrambe** le schede e su **tutte** le finestre è
meno nodi e meno attese spezzate per layer.

Tappe (tutte opt-in, `STRATA_P2_*`; verifica del testo a cache fissa `--expert-cache 1500`, senza `--elastic`,
`--adapt-every 0`, L02 corto identico a spento; misure per finestra con `strata decode timing` / `pipeline classes`;
niente B-A-B di velocità: li fa il coordinatore):

| tappa | cosa | misura che la chiude | atteso |
|---|---|---|---|
| **0a** | P-state a campioni fitti durante il decode (§2c) | quota del tempo sotto P0 dentro le finestre | decide se serve (d6) |
| **0b** | `STRATA_P2_FLUSH_NOSWITCH=1`: polling di `service`, `done()` e della catena senza `OnDevice` (porto di NTRYourWaifu v7/v8, ripiego col cambio di scheda se la query è rifiutata) + contatore del tempo host nei flush | tempo nei flush per finestra; attese dei flag (`STRATA_SC_WAITSTAT`) | 0/−0,5 ms |
| **1** | censimento dei nodi per layer dello stadio 1 e dello stadio 0 dalla cattura (`nsys` con `--cuda-graph-trace=node`, già nei `.sqlite` di `D:\ds4_work\strata\nsys`): quali kernel sono contigui e fondibili senza cambiare aritmetica | elenco con µs di buco per confine | — |
| **2** | `post()` fuso per layer sulla 2080 Ti: attesa del piano + copia + esperti VRAM + attesa righe CPU + combine in un lancio persistente (cooperativo per gruppi di blocchi), sul percorso con doorbell | nodi per finestra, buchi, muro dello stadio 1 | −0,6/−1 ms |
| **3** | lo stesso sullo stadio 0 (3060) e sul fronte FFN (`pre()` dal router al quantize) | muro dello stadio 0 delle fresche | −0,4/−1 ms |
| **4** | solo se le tappe 2-3 rendono: lancio anticipato dello stadio 1 con flag sul device (#1120) dentro il grafo fuso | F0 → L1 e bolla | −0,1/−0,3 ms |

Cosa **non** faccio senza un nuovo via: expert-parallel (a/b), tensor-parallel (d1), stadi invertiti (c2), B ad
albero (d2).

## 6. Che cosa cambierebbe il verdetto

- **Più VRAM sulla seconda scheda.** L'expert-parallel di #1798 rende quando tutti gli esperti stanno in VRAM: con
  una 16-24 GB al posto della 3060 (o tutti gli esperti dello stadio 1 residenti) (a) diventa −2/−4 ms e la quota della
  seconda scheda non costa hit. Il micro-bench del §2a vale anche lì (stesso percorso via RAM se manca il P2P).
- **Righe di ritorno più leggere a bit uguali.** Se il combine accettasse le somme parziali della 3060 in un ordine
  fisso (oggi somma le 10 righe in ordine di k), il ritorno scenderebbe da 80 a 20 KB per layer (§2a: esposto
  −11 µs invece di +16): non è bit per bit con oggi, ma è deterministico. Con l'owner che accetta bit diversi, (b1)
  migliora di ~0,2 ms; non basta.
- **Pendenza per riga più bassa** (oggi 3,26 ms per riga sullo stadio 1, di cui solo ~0,5 di byte): finestre più larghe
  costerebbero poco e la pipeline B conterebbe meno. È lavoro sui kernel (EXPDEC lato pesi, MMVQ e hc a righe), non di
  struttura.

## 7. File

- `tools/hop_bench.cu`: micro-bench del salto (§2a). Uscite in `D:\ds4_work\strata\pipeline2\hop_bench_run{1,2}.txt`.
- `tools/ep_residency.py`: modello di residenza da un trace di routing (§2b).
- `D:\ds4_work\strata\pipeline2\pstate_probe.py`: richiesta di prova col motore del laboratorio e `nvidia-smi -lms 10`
  (§2c); usa la config della Daily 0.1.43 in sola lettura, porta 8102, senza file del prefisso né visione.

## 8. Diario delle tappe

Regole: ogni leva è opt-in (`STRATA_P2_*`); ogni tappa chiude con una build in
`corse_2080ti\20261004_routed_keep\build-pipeline2-<sha>\` (`build.json`), l'env per il braccio B del B-A-B e la
verifica del testo a cache fissa (`--expert-cache 1500`, senza `--elastic`, `--adapt-every 0`, L02 corto identico al
braccio spento). I B-A-B li fa il coordinatore.

### Tappa 0a — P-state a ~10 ms (da rifare: primo tentativo morto all'avvio)

`D:\ds4_work\strata\pipeline2\p2_pstate.sh`: corsa forzata breve `DET-p2-pstate1` (L01+L02 131k, 256 token, env della
Daily 0.1.43 sulla `build-structa-075ce875`), `nvidia-smi -lms 10` solo mentre il `bench.lock` porta il suo nome.
Lanciata il 13/10 alle 02:20, in coda dietro la notte multi-chat del coordinatore (`DET-mc2-*`).

### Verifica del testo di 0b e 2a (13/10 04:10-04:30, M)

`p2_check.py build-pipeline2-cd3bd31b DET-p2-chk1 256 off,stat,nosw,smem,both` (L02 131k, greedy libero, cache fissa
1500, senza `--elastic`, `--adapt-every 0`, env della Daily 0.1.43; uscita `D:\ds4_work\strata\pipeline2\p2_chk1.txt`):

- **`L02.txt` identico a `off` (1199 caratteri) in `stat`, `nosw`, `smem`, `both`**; bozze accettate identiche in tutti
  i bracci (`[113, 187]`), stesse 143 finestre, stesse classi (112 fresche, 31 speculative, 10 disaccordi della catena).
- I **primi logit non sono confrontati**: `STRATA_DUMP_FIRST_LOGITS` scrive solo nel ciclo di serve non in pipeline
  (`generate.cpp` ~13369); con `--pipeline-windows 2` non scatta (nessun file, nessun errore). La prova è il testo.
- `STRATA_P2_SMEM`: «CUDA0/CUDA1 prefers shared memory (ok)», due volte per scheda (un `Verifier::init` per stadio e
  per la seconda finestra; innocuo).
- Le velocità di queste corse (53,6-55,0 t/s, 32,5-33,4 ms a finestra) sono **una corsa per braccio senza
  riscaldamento né testo forzato**: non sono un confronto di velocità (servono i B-A-B del coordinatore).

### Tappa 0b — polling senza `cudaSetDevice` (compilata, testo identico; B-A-B da fare)

Misura dei cambi di scheda (M, corse sopra, 143 finestre):

| braccio | cambi di scheda per finestra | ms host dentro i cambi per finestra | più lungo | poll senza cambio |
|---|---|---|---|---|
| `stat` (solo conteggio) | **8088** | **3,37** | 0,076 ms (nessuno > 0,1 ms) | 0 |
| `nosw` (+ `STRATA_P2_FLUSH_NOSWITCH=1`) | **37** | **0,05** | 0,015 ms | 5,15 milioni, nessun rifiuto |

Lettura: il polling senza cambio funziona sul driver dell'owner (nessuna query rifiutata) e toglie il 99,5% dei cambi.
I 3,4 ms host a finestra sono del filo che fa il polling, in gran parte **sovrapposti** al lavoro GPU: quanto ne stava
sul percorso critico lo dice solo il B-A-B (S: 0/−0,5 ms a finestra, come nel §5). Nessun cambio > 0,1 ms in questa
corsa: lo stallo dentro `cudaSetDevice` visto su 0.1.41 qui non si è presentato.

- `STRATA_P2_FLUSH_NOSWITCH=1`: `Verifier::service` (flush ogni 2 ms), `Verifier::done`, `MtpDrafter::chain_outputs_ready`
  e `chain_poll` interrogano eventi e stream senza `OnDevice` (porto di NTRYourWaifu `two-pc-042` v7/v8, che lì è acceso
  di default). Se una query rifiuta la scheda «straniera» il motore lo dice una volta e torna al cambio di scheda.
- `STRATA_P2_SWITCH_STAT=1` (solo misura): conta i cambi di scheda di `OnDevice` e il tempo host dentro
  `cudaGetDevice/cudaSetDevice`; riga a fine richiesta con `STRATA_DECODE_TIMING`: `strata pipeline2 switches: N device
  switches (x per window), y ms host in them per window, longest z ms, k over 0.1 ms; polls without a switch …`.
- Bit: nessun lavoro GPU cambia.

### Tappa 1 — censimento dei nodi (strumento pronto, cattura nuova da fare)

`tools/graph_census.py <cattura.sqlite>`: per scheda, la finestra mediana (nodi, somma dei kernel, tempo in cui nessun
kernel della finestra gira), per kernel nodi/µs/idle prima, e gli «spigoli» di idle (kernel finito per ultimo → kernel
dopo). Cattura nuova della base in coda: `D:\ds4_work\strata\pipeline2\nsys_base.bat DET-p2-nsys-base` (L01 16 + L02 200
token, env della Daily 0.1.43, `--cuda-graph-trace=node`).

Prima lettura sulla cattura vecchia `DET-x3-hostgap-nsysB2` (build `72fc813a`, senza NODECUT, M sotto nsys):
- 2080 Ti: ~1398 nodi a finestra, idle dentro il grafo **1,17 ms**, di cui **0,74 ms su un solo spigolo:
  `gr_down_staged` → `gr_up_multi`, 11,4 µs ciascuno, 65 a finestra**. Gli altri spigoli valgono 0,03-0,13 ms.
- 3060: idle dentro il grafo 0,01 ms; lo stesso spigolo vale 0,10 µs.
- Causa probabile (dalla cattura: `sharedMemoryExecuted`): sulla 2080 Ti `gr_down_staged` chiede carveout 100 (64 KiB di
  shared) e quasi tutti gli altri kernel girano a 32 KiB: lo SM cambia configurazione L1/shared due volte per lettura
  hc, e su Turing il cambio scarica lo SM. Sulla 3060 (Ampere) il cambio non costa.
- **Tappa 2a nuova, piccola e bit per bit** (codice in `Verifier::init`, compilato in `build-pipeline2-cd3bd31b`, testo identico): `STRATA_P2_SMEM=1` → `cudaDeviceSetCacheConfig(cudaFuncCachePreferShared)` su
  ogni scheda prima delle catture. Atteso: −0,3/−0,6 ms a finestra sullo stadio 1 (S, dagli 0,74 ms nsys gonfiati
  ~1,5-2×). Da confermare con la cattura nuova e una cattura con la leva accesa.

## 9. Appendice: tensor-parallel «bivalente» (richiesta dell'owner, 13/10 notte; solo analisi, S)

Proposta: prefill come oggi (layer split), decode in tensor-parallel (TP); variante senza MTP e variante con MTP (finestre
di verifica a T righe in TP, catena di bozze in serie, perdita della sola pipeline B). Ogni numero qui è **stimato** (S)
dalle misure del §1-2 (tempi puliti per stadio di `STRUCT_A`/`STRUCT_B`, famiglie di kernel di `ROOFLINE` §1c, salto del
§2a); nessun codice TP esiste.

### 9a. Il passaggio di un token in TP (T=1)

Riferimento in serie oggi a T=1, senza MTP (M): stadio 0 7,93 + stadio 1 13,1 + F0→L1 0,17 + host ~0,4 = **~21,6 ms**
(46 t/s). Il TP ne cambia così le voci:

| voce | oggi in serie, T=1 | TP, T=1 | perché |
|---|---|---|---|
| kernel che leggono pesi (5,09 GB a T=1) | 5,4 (3060) + 8,5 (2080 Ti) = 13,9 | **8,7** | 5,09 GB / (280 + 389 GB/s effettivi oggi) = 7,6 ms, ×1,15 per griglie dimezzate |
| kernel piccoli (< 8 µs: norme, quantize, swiglu, route, doorbell, flag) | 1,6 + 2,2 = 3,8 | **~4,3** | non si dividono: ognuna delle due schede li esegue per tutti i 48 layer (la 3060 più lenta) |
| buchi fra i nodi | ~1,5 | ~2,2 | ~48 × 45 nodi per scheda invece di 32 × 40 |
| scambi (all-gather di 10 KB a T=1) | 0 | **~2,0** | 4-5 scambi per layer (hc lato attenzione, uscita attenzione, hc lato FFN, uscita MoE, router) × 48 × ~9 µs del §2a, in serie: il consumatore aspetta sempre |
| QSA con le 2 teste K/V divise | (dentro i kernel) | +0,4 | una testa K/V e 12 teste q per scheda; i punteggi dell'indicizzatore (4 teste) vanno sommati fra le schede prima del top-k: 12 all-reduce da 128 KB (~30 µs) |
| attese degli esperti CPU | ~3,5 | ~3,0 | stesso lavoro CPU, un solo percorso |
| host, lancio, campionamento | ~0,6 | ~0,3 | il token successivo può partire dal device |
| **totale** | **~21,6** | **~19-21** (forchetta 17-23) | |

### 9b. Le due varianti, sul banco a 131k

Oggi (M): ~77 / ~70 t/s su L01 / L02 con MTP, 1,97 / 1,71 token a finestra (media 1,85), ~25,6 / ~24,4 ms a finestra.

| variante | ms a finestra | token a finestra | L01 | L02 |
|---|---|---|---|---|
| **TP senza MTP** | 17-23 (centro ~20) | 1 | **43-59 (~50)** | **43-59 (~50)** |
| **TP con MTP** (catena in serie sulla 2080 Ti) | catena prima della finestra ~2,1 (3060 ferma) + host 0,3 + passaggio a T≈2 ~23,4 (T=1 ~20 + ~3,4 per la riga in più: la pendenza di oggi, 5,7 ms per riga in serie, divisa fra le schede) = **~25,8** | 1,97 / 1,71 | **~76** (forchetta 70-87) | **~66** (forchetta 61-76) |
| TP con MTP, catena anch'essa in TP (passo 0,93 → ~0,65 ms) | ~25,2 | | ~78 | ~68 |

Perché il TP con MTP non guadagna: rispetto a oggi accorcia la finestra fresca (28,3 → ~25,8: −2,5 ms nel 72% delle
finestre) ma allunga la speculativa (19,8 → ~25,8: +6 ms nel 28%), perché senza la pipeline B ogni finestra paga catena
+ tutti i layer. Netto sulla media: **−0,3/+0,5 ms a finestra**, cioè pari. La catena in serie costa ~2,1 ms a finestra
(~8%) con la 3060 ferma; in TP la si accorcia di ~0,6 ms, non di più (latenza dei passi, non banda).

Soglia: per 80 t/s su L02 (1,71 token) serve 21,4 ms a finestra → un passaggio a T≈2 ≤ ~19 ms → **≤ 0,38 ms per layer a
T=2** in TP (testa compresa a parte). La stima è 0,46-0,49 ms per layer. Su L01 la soglia è 0,45 ms per layer: al bordo.

### 9c. Il cambio di modalità (pesi ricaricati dalla RAM, correzione dell'owner)

I pesi non passano fra le schede: ogni scheda ricarica dalla RAM (arena col nucleo RAM, `dense.bin`) la sua fetta, le due
copie in parallelo, 3060 sull'Oculink ~6,7 GB/s, 2080 Ti in x16 ~12,9 GB/s.

| | 3060 | 2080 Ti |
|---|---|---|
| densi + testa nel layout TP (fetta per banda, ~41% / 59% di 3,83 GB) | 1,57 GB | 2,26 GB |
| cache degli esperti da riempire (fette di riga degli esperti residenti, §9d) | ~4,9 GB | ~10,7 GB |
| totale | **~6,5 GB → ~0,97 s** | **~13 GB → ~1,0 s** |
| stato: K/V QSA da ridistribuire per testa (1 KB per token e layer, metà per scheda; la parte residente in VRAM, 32 768 celle × 12 layer: metà = ~0,2 GB; la parte oltre `--kv-resident` sta in RAM bloccata e non si muove) + chiavi dell'indicizzatore (16,8 MB per layer QSA a 131k, metà: ~0,1 GB) + stato GDN per testa (36 × 3,1 MB, metà: ~56 MB) | ~0,2 GB in entrata | ~0,1 GB in entrata |

Un cambio costa **~1,0-1,2 s** se l'arena e `dense.bin` sono in RAM e le copie sono pinned; **1,5-2,5 s** con la parte
pageable dell'arena (oggi pinned solo 8 GiB su 29,6) o `dense.bin` da leggere dal disco. Il ritorno al layer split
(prefill lungo successivo) costa uguale. Due layout dei pesi insieme non stanno: la 3060 perderebbe 1,57 GB di esperti
(~1 050 slot, −10 punti di hit sullo stadio 0 dal §2b), la 2080 Ti 2,26 GB (~1 200 slot).

### 9d. Residenza degli esperti in TP

Ogni esperto residente è diviso per righe fra le due schede (gate/up per `n_ff`, down per righe d'uscita o d'ingresso),
quindi **le due schede hanno lo stesso insieme residente e lo stesso hit**. Con la fetta degli esperti al ~31% sulla 3060
(4,9 / 15,6 GB, diversa dal 41% dei densi: si bilancia per classe di kernel) l'insieme vale ~15,6 GB, come la somma di
oggi, ma scelto per frequenza su **tutti** i 48 layer invece che a quote fisse per stadio: dal §2b oggi lo stadio 0 è
sotto-servito, quindi l'hit globale sale di poco (S: +0,5/+1 punto). Gli esperti CPU restano interi sulla CPU; ogni
scheda legge la sua fetta delle righe. La cache elastica della 3060 (monitor 8K) diventa più rigida: togliere una fetta
dalla 3060 toglie l'esperto a entrambe.

### 9e. Prefill in TP (variante mista)

- **Prefill lungo a freddo**: in layer split come oggi, poi un cambio (~1-2,5 s): **+1,5-3%** su un 131k da ~75-80 s.
- **Prefill incrementale in TP** (pochi k token): a 2k token un all-gather/all-reduce porta ~21 MB per scambio; con 1
  scambio per layer (~1 GB, l'ipotesi del coordinatore) la 3060 riceve e manda ~1 GB sull'Oculink full duplex: **~150
  ms**; con 3 scambi per layer, realistici col hc e il MoE, **~0,3-0,45 s**. Oggi 2k token di prefill costano ~1,2 s
  (131k in ~75 s): **+12/+38%** se gli scambi non si sovrappongono al calcolo, meno con i chunk a pipeline. In più la 3060
  riceve dalla RAM il 41% (contro ~33% di oggi) degli esperti non residenti che il prefill legge: +~25% sul suo link.
- **Prefill tutto in TP**: 131k = 32 chunk da 4K × 48 layer × 3 scambi da 42 MB ≈ 190 GB sull'Oculink → **+25-40 s**
  sul 131k (oggi il layer split già fa lavorare le due schede insieme sui chunk diversi: il TP non accelera il calcolo).

### 9f. Verdetto

- **TP senza MTP: no.** ~50 t/s (43-59) contro 77/70: la catena MTP vale ~1,85 token a finestra e il TP non lo recupera.
- **TP con MTP: non sposta.** ~76 / ~66 t/s al centro (forchette 70-87 / 61-76) contro 77 / 70: pari o peggio, perché si
  perde la pipeline B. TTFT: +1-2,5 s dopo ogni prefill lungo a freddo, +0,15-0,45 s sui turni incrementali da 2k.
  Lavoro: TP del decode con tutte le famiglie di kernel a fette (densi, hc, router, esperti, QSA a teste, indicizzatore,
  GDN, PLE, testa), scambi nel grafo, cambio di modalità, tier adattivo ed elastico a fette, checkpoint delle
  conversazioni divisi: **4-6 settimane**; un prototipo minimo (solo decode, cache fissa, `--adapt-every 0`, greedy, niente
  batch) 8-12 giorni.
- **Prototipo di 1-2 giorni**: un TP vero non sta in 1-2 giorni. Ha senso solo come **banco di un layer**: i kernel veri
  (`b6_mmvq_rows`, `gr_*`, `expdec_gu`, `native_down_multi`, QSA) su fette 41/59 sulle due schede, con gli scambi del §2a,
  in un grafo da 48 copie. Misura l'incertezza che decide (pavimenti dei kernel a fette + scambi): **se il layer a T=2 sta
  sotto ~0,38 ms il TP con MTP passa gli 80 su L02; la stima è 0,46-0,49**. Lo consiglio solo se l'owner vuole chiudere la
  domanda con un numero; non lo metto davanti alle tappe.
- **Contro il kernel persistente / meno nodi** (tappe 2-3): −0,7/−1,5 ms a finestra (+3/+6%) su tutte le finestre, bit
  per bit, giorni invece di settimane, nessun costo sul TTFT, la pipeline B resta. Resta la strada; il TP no.

