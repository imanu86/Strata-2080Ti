# PIPELINE3: le due schede davvero insieme su ogni finestra — conti rifatti per 3080 Ti + 2080 Ti

Ramo `lab/upsync-v0.2.9`, 2/11. Rifà i §1, §3 e §4 di `PIPELINE2.md` (che era fatto con la 3060 come stadio 0) con le
misure del 2/11 sulla macchina di oggi: **3080 Ti 12 GB** (sm_86, 912 GB/s, Oculink PCIe 4.0 x4, sonda 6,7 GB/s) +
**2080 Ti 22 GB** (sm_75, 616 GB/s, x16 terza generazione, 13 GB/s), 3060 solo schermo. Niente P2P (WDDM, Ampere ↔
Turing). Modello Qwen3.8-Flash-Next IQ3_XXS, 48 strati, k = 10, blob medio 1,7 MB, n_embd 2560.

Convenzione: **M** = misurato (registri `engine.log` delle corse in
`corse_2080ti\20261101_tre_schede\corse\`, `RISULTATI.md`, micro-banco), **S** = stimato da misure. I ms sono per
finestra di scrittura (banco della comunità 4K/32K/128K, 256 token, T medio 2,9-3,2 righe, 2,3-2,6 token a finestra).

## 0. Conclusione

- **Oggi in serie (divisione 20, `Pstadi20`, M)**: la 3080 Ti fa 20 strati in 6,5 ms di GPU + 1,6 di attese CPU, la
  2080 Ti 28 strati in 12,6 + 1,1: verifica 22,4 ms, finestra 25,3 (bozze 2,5, emissione 0,5). Ogni scheda è ferma
  mentre lavora l'altra: la 3080 Ti il 68% della finestra, la 2080 Ti il 46%. La catena (pipeline B) non recupera
  nulla di misurabile (`Eb0*`: fresche 29,6-30,1 ms, speculative 18,1, ma solo 34-39 speculative su 90 restano sul
  percorso, media 26,0 contro 25,3 senza catena; M).
- **Il `--peer-device` di oggi (`Vpeer`, M) è expert-parallel con l'host nel mezzo**: 48 strati sulla 3080 Ti in
  **11,7 ms** di GPU (0,245 ms a strato, con ~9 esperti locali per strato-finestra), ma ogni strato paga
  **0,43-0,53 ms di host** (plan 4,3-7,2 + CPU 15,4-23,3 ms a finestra, dove «CPU» è il run del pool *più* lo spin su
  `cudaStreamQuery` del peer: `expert_source.cpp:5277`): finestra 40-53 ms, 46-60 t/s. Il salto GPU→GPU con flag fra
  kernel persistenti costa 8 µs, con i dati di uno strato 28-70 µs (`PIPELINE2` §2a, M): **il disegno a `launch` +
  `finish` sul filo dell'host costa 8-15 volte il possibile dell'hardware**, non mille, ma basta a uccidere la strada.
- **Expert-parallel vero** (3080 Ti padrona di tutti gli strati, 2080 Ti server di esperti con kernel persistente e flag
  sul device, host fuori dal percorso): verifica **16-19 ms** (S da M, §3a), finestra **18,5-21,5** contro 25,3:
  **−15/−27%**, cioè **~120-140 t/s** sul banco dove oggi ne fa ~104. Tetto onesto 140; sotto i 115 non vale il lavoro.
  Nessuna scommessa: ogni finestra è deterministica, la classe «speculativa» sparisce. Costo: **10-20 giorni**, e la
  lettura dei prompt lunghi perde il 33-42% finché i densi non stanno anche sulla 2080 Ti (§3e).
- **Idea nuova (§5): il server prende anche i miss.** In EP la 2080 Ti è in calcolo ~30% del tempo e siede sul link
  x16: gli esperti non residenti li legge lei dalla RAM bloccata (1,7 MB in 0,13-0,2 ms) invece della CPU (0,4-0,9 ms
  a esperto su AVX-2). Toglie le attese CPU dal percorso (2,7 ms a finestra oggi, M) e la sensibilità al carico CPU
  (−7% con OktoCGControl, −13% con 6 core occupati, M). Vale **−1/−2 ms** (S) in più di EP e non dipende dalle bozze.
- **Che cosa non riempie il tempo fermo**: dentro una sola risposta il decode è una catena di 48 strati; senza dividere
  i pesi (EP, TP) l'unica altra cosa che una scheda ferma può fare è un'altra richiesta (multichat) o una scommessa.
  Il TP sui densi varrebbe altri −2,4 ms (S) in 4-6 settimane: no.
- **Prototipo** (§7): `tools/ep_bench.cu` misura su queste due schede, senza motore, i µs esposti per strato nelle
  quattro forme (server persistente con flag, server guidato dall'host come oggi, con e senza cambio di scheda,
  tutto sulla 3080 Ti). Nel motore: `STRATA_PEER_NOSWITCH=1` (d5 sul percorso peer, opt-in) e la riga
  `strata decode peer:` che separa il tempo di `launch` da quello di `finish`. Compilato: vedi §7. Provato no.

## 1. Da dove partiamo: chi è ferma, e quanto (M)

Serie, divisione 20, senza catena, tutte le leve (`Pstadi20`, `STRATA_SPLIT_TIMING=1`, ~950 finestre):

| | 3080 Ti (strati 0-19) | 2080 Ti (strati 20-47) | finestra |
|---|---:|---:|---:|
| GPU in corsa | 6,5 (0,325 ms/strato) | 12,6 (0,45 ms/strato) | 19,1 |
| attese delle righe CPU | 1,6 | 1,1 | 2,7 |
| altro (piano, stage, salto fra stadi) | ~0,3 | ~0,3 | 0,6 |
| **verifica** | 8,4 | 14,0 | **22,4** |
| bozze (catena MTP, sull'ultimo stadio) + emissione | — | 2,5 | 3,0 |
| **finestra** | | | **25,3** |
| ferma | ~17 ms (68%) | ~11,5 ms (46%) | |

Composizione della finestra per chi può dividerla:

| voce | ms | divisibile fra le schede? |
|---|---:|---|
| strati della 3080 Ti: densi, attenzione, hc, router, esperti VRAM | 6,5 | sì (EP: solo gli esperti; TP: tutto) |
| strati della 2080 Ti: idem | 12,6 | sì |
| attese delle righe CPU | 2,7 | no, ma spostabili sul server (§5) |
| bozze (catena di 4 passi, 0,5-0,6 ms a passo) | 2,5 | no: latenza, non banda |
| emissione, piano, salti fra stadi, adapt join | ~1,1 | no |

In EP il pavimento è la **serie dei 48 strati sulla 3080 Ti senza i suoi esperti**: 0,245 − ~0,04 (i 9 hit, S) ≈
0,21 ms × 48 ≈ **10 ms** (S da M), più bozze ed emissione 3,0: **~13-14 ms è il limite che nessun EP tocca**
(a 2,4 token a finestra: ~170 t/s asintotici). La 2080 Ti in serie oggi fa 0,45 ms a strato: la stessa cosa fatta
da lei costerebbe 21,6 ms, per questo la padrona è la 3080 Ti.

## 2. Il peer di oggi, misurato (`Vpeer`, 2/11 16:07, build `835f73b2`, M)

`--peer-device 1 --peer-reserve-mib 3000` (con la riserva di default 600 il peer muore di `stream: out of memory`),
`STRATA_PEER_HOT 0.45` (3915 delle prime 8700 coppie spostate dietro la 12615ª). Confronto con `Vrif1` (divisione 20,
stessa sessione, stessi prompt).

| | `Vrif1` (serie) | `Vpeer` (peer di oggi) |
|---|---:|---:|
| slot esperti 3080 Ti / 2080 Ti | 5477 (7,72 GiB) / 9090 (16,25 GiB) = 14567 | 2561 (4,17 GiB) / 10972 (17,80 GiB) = 13533 |
| VRAM libera a modello caricato | — | 3080 Ti 700 MiB (sotto la soglia di guardia), 2080 Ti 2190 MiB |
| scrittura 4K / 32K / 128K t/s | 92-106 / 94-103 / 98-100 | 46-60 / 56 / 46-57 |
| ms a finestra | 21,3-24,7 | 39,8-53,0 |
| GPU-reach wait (3080 Ti) | 8,4-10,2 (20 strati, con la catena) | **11,7-11,9** a 4K/32K, 13,3-14,9 a 128K (48 strati) |
| per-layer host: plan / CPU | 0,15-0,25 / 2,0-6,4 | **4,3-7,2 / 15,4-23,3** |
| adapt join | 0 | 3,9-6,9 (10-17 swap a finestra) |
| esperti per strato-finestra: VRAM 3080 Ti / peer / CPU | 33-38 / — / 0,5-0,9 | 9,2-10,6 / ~18 (62-64% delle 29 voci) / 1,0-1,9 |
| lettura 4K / 32K / 128K t/s | 860-934 / 1798-1858 / 2148-2227 | 892 / 1158-1258 (**−33%**) / 1247-1294 (**−42%**) |

Per strato il percorso del peer vale (plan + CPU) / 48 = **0,43-0,53 ms di host in serie** (M): spin sul flag del
doorbell, `cudaMemcpyAsync` × 2, `hadamard` + `quantize` + `native_expert_grouped` + `scatter_rows_f32` lanciati su un
altro device sotto WDDM, spin su `cudaStreamQuery` (`peer_experts.cpp:245-350`). Il calcolo vero degli esperti sul
server è ~0,1 ms (S, §3a): il resto è latenza di sottomissione e di polling dell'host.

## 3. Conti rifatti

### 3a. Expert-parallel vero: 3080 Ti padrona, 2080 Ti server con kernel persistente (S da M)

Disegno: la 3080 Ti esegue tutti i 48 strati (densi, QSA, GDN, hc, router, i suoi esperti, combine, testa, catena);
scrive x in q8_1 + id nella memoria mappata e alza un flag **dal grafo**; sulla 2080 Ti un kernel persistente per
finestra (tutti i blocchi residenti) aspetta il flag, copia x in VRAM, calcola gli esperti che tiene (tabella di
residenza sul device), scrive le righe fp32 nelle stesse righe mappate che oggi riceve dalla CPU e alza il flag;
il `wait_copy` della 3080 Ti aspetta il flag della CPU **e** quello del server. Host fuori dal percorso: pianifica
solo i miss CPU, come oggi, e le rotazioni fra finestre.

Per strato, a T ≈ 3 (29 voci: 9 locali, 18 sul server, 1,4 CPU come in `Vpeer`):

| voce | µs | fonte |
|---|---:|---|
| 3080 Ti, tutto il suo lavoro con i 9 hit locali (in serie) | **245** | M (`Vpeer`: 11,7 ms / 48) |
| di cui i 9 hit locali (~7 blob, 12 MB) | ~40 | S (250-350 GB/s effettivi sugli esperti) |
| server: 18 voci ≈ 13 blob ≈ 22,6 MB | 75-150 | S (la 2080 Ti in serie fa ~22 blob in ~0,15 ms: 260 GB/s; a 150 GB/s come EXPDEC 2080 Ti vecchio: 150) |
| salto andata (8-12 KB) + ritorno (18 righe fp32 = 184 KB) + flag | 45-60 | S da M (§2a: 8 µs flag, +19 µs per 80 KB) |
| **esposto** = salto + server − hit locali | **80-170** | |
| con ripartizione «caldi alla padrona» (STRATA_PEER_HOT=0: 43% locale, 52% server, §2b di PIPELINE2) | 55-120 | S |
| con righe bf16 al ritorno (92 KB; bit diversi) | −20 | S |

Finestra: 11,7 (M) + 48 × 0,08-0,17 = **3,8-8,2** (S) + miss CPU in più esposti (+0,4-1,0 voci a strato-finestra
rispetto alla serie, M; ~0,05 ms per voce-strato, S: **1-2**) + stage 0,3 = **verifica 17-22 ms**; con i caldi alla
padrona **16-19**. Finestra = verifica + bozze 1,2-2,1 (M, drafter sulla 3080 Ti in `Vpeer`) + emissione 0,2 + adapt
join tenuto fuori dal giro (0,5) = **18,5-21,5 ms** contro 25,3 (M): **−15/−27%**, 2,4 token a finestra → **~115-130
t/s** al centro, **tetto 140** se server e salto stanno ai minimi. Il «tutte le finestre uguali» vale anche sulle
fresche di oggi (29,6-30,1 ms con la catena): lì il guadagno è −30/−38%.

Conti al contorno:
- **VRAM** (M): slot 14567 → 13533 (−7%): la 3080 Ti perde 2916 slot (5,0 GiB) per 28 strati di densi (~62 MiB a
  strato: 1781 MiB di proiezioni + ~1200 di pack per 48 strati), K/V residente dei QSA, stati GDN, buffer di verifica
  e la riserva; la 2080 Ti ne guadagna 1882 con la riserva a 3000 MiB (2190 MiB restano al ring del prompt). I miss
  CPU salgono di +0,4-1,0 voci a strato-finestra (M, `Vpeer` contro `Vrif1`).
- **Traffico** (S da M): 48 × (12 KB + 184 KB) ≈ 9,4 MB a finestra; a 50 finestre/s ~0,5 GB/s, da aggiungere ai
  0,74 / 0,88 GB/s di oggi in ricezione (M, `passaggio.csv`). Il link non è il collo: lo è la latenza delle letture
  mappate a 16 byte (§2a), per questo le righe bf16 o sommate per token (30 KB) valgono 20-40 µs a strato.
- **48 scambi**: tutti sul percorso critico, nessuno si sovrappone al successivo (lo strato L+1 dipende da L).
- **Pavimento delle latenze in serie**: 10 ms di densi della 3080 Ti + 2,5 bozze + 0,5 emissione + ~1 salti e piano
  = **~14 ms**; EP ci arriva a 4-7 ms.
- **Finestre speculative**: non esistono più; la pipeline B oggi vale 0 (M), non si perde nulla.
- **Lettura del prompt** (M): −33/−42% a 32K/128K con il peer di oggi, perché il prefill dei densi gira solo sulla
  3080 Ti; §3e.
- **Bit**: gli esperti sul server sono già calcolati con gli stessi kernel (`peer_experts.hpp:11`) e le righe fp32
  entrano nel combine nell'ordine di sempre: identici al `--peer-device` di oggi, non alla serie (sm_86 contro
  sm_75 per gli esperti che cambiano scheda).

### 3b. Ibrido: layer split per i densi + esperti divisi per frequenza su tutti gli strati (S)

Ogni strato ha una scheda padrona dei densi (0-19 → 3080 Ti, 20-47 → 2080 Ti) e gli esperti stanno dove li manda il
profilo globale; la non-padrona fa i suoi esperti con lo stesso scambio di §3a. Il percorso critico per strato è lo
stesso di §3a ma 28 strati hanno la padrona lenta (0,45 contro ~0,30 a strato per i densi, M): **+3,5 ms** rispetto
a EP puro, nessun guadagno in VRAM (i densi stanno comunque su una sola scheda per strato). Vale solo se la 3080 Ti
non ha VRAM per tutti i densi: ce l'ha. **No.**

### 3c. Micro-finestre, stadi invertiti, B ad albero

Come in `PIPELINE2` §3c-d: la micro-finestra rilegge i pesi (+5/+7 ms), gli stadi invertiti spostano solo la classe
speculativa che oggi non rende (M: ~0), la B ad albero è stata misurata a 0 da un fork. **Nessuna cambia con
l'hardware nuovo.**

### 3d. Tensor-parallel sui densi (S)

Sulla serie EP di 10 ms di densi, dividere le matrici 60/40 fra 912 e 616 GB/s porta la 3080 Ti a ~6,5 ms ma aggiunge
≥ 2 sincronizzazioni a strato (dopo l'attenzione, dopo il MoE: 96 salti da 16-20 µs con i dati, §2a) = +1,7 ms, più
i K/V dei QSA per testa e l'indicizzatore sommato fra le schede: **−2,4 ms** (S) in 4-6 settimane. Non prima di EP.

### 3e. La lettura del prompt in EP

Con tutti i densi sulla 3080 Ti il prefill perde il layer split (M: −33/−42% sui prompt lunghi con il peer di oggi,
che fa già calcolare al server i suoi esperti del chunk). Due uscite:
- **densi anche sulla 2080 Ti** (~2,9 GiB: −1700 slot sul server, S: +0,3 voci CPU a strato-finestra) e prefill a
  layer split come oggi con gli esperti del chunk chiesti alla scheda che li tiene (il percorso del peer nel prompt
  esiste già, va reso simmetrico). È il prezzo giusto: la lettura è il 70% del tempo nelle prove a più turni (M).
- oppure **doppia modalità** (§9c di PIPELINE2): layer split nel prefill, EP nel decode, 1-2,5 s a cambio. Solo
  per prompt lunghi; peggiora il TTFT.

## 4. Tabella riassuntiva (finestra di scrittura media, oggi 25,3 ms / ~104 t/s, M)

| | finestra (ms) | Δ | scambi a finestra | VRAM | bit | lettura | giorni |
|---|---:|---:|---|---|---|---|---:|
| oggi, serie divisione 20 (M) | 25,3 | — | 1 salto da 51 KB × T | 14567 slot | — | 2068-2210 a 128K | — |
| oggi + catena (M) | 26,0 | +0,7 | idem | −253 MiB | uguali | idem | — |
| `--peer-device` di oggi (M) | 40-53 | **+15/+28** | 48 × host | 13533 | peer | −33/−42% | — |
| **(a) EP vero, server persistente, flag sul device** | **18,5-21,5** | **−4/−7** | 48 × (12 KB + 184 KB) sul device | 13533 | come il peer | −33/−42% finché §3e | 10-20 |
| (a) + caldi alla padrona + righe bf16 | 17,5-20 | −5/−8 | idem, 92 KB | idem | diversi (bf16) | idem | +1 |
| **(a) + miss sul server via x16 (§5)** | **17-20** | **−5/−8** | + i blob dei miss (1,7 MB ciascuno) | idem | come il peer | idem | +3-5 |
| (a) + densi anche sulla 2080 Ti | +0,3 | — | — | −1700 slot server | — | **pari a oggi** | +5-8 |
| (b) ibrido layer split + esperti divisi | 22-25 | −0,5/−3 | idem | idem | | pari | 10-15 |
| (d1) TP sui densi sopra EP | 16-18 | −2,4 in più | 96 salti in più | ripartita | diversi | da rifare | 20-30 |
| micro-finestre / stadi invertiti / B ad albero | ≥ 25 | 0/+7 | | | | | |

## 5. L'idea nuova: il server prende anche i miss, e la testa si divide

In EP la 2080 Ti lavora ~0,1 ms su 0,35 a strato: è ferma il 70% del tempo, sul link più veloce della macchina.
Oggi gli esperti non residenti li calcola la CPU: 7 worker AVX-2, 0,4-0,9 ms a esperto (M: `CPU 24,95` ms a
finestra per 7,2 job a strato nella finestra di riscaldamento), e in serie ne restano 1,6 + 1,1 = 2,7 ms a finestra
(M). Con EP i miss salgono (+0,4-1,0 voci a strato-finestra, M). Il kernel persistente del server può invece
**leggere il blob del miss dalla RAM bloccata** (l'arena è già registrata a fette: 48 fette, 39 GiB, M) con letture
mappate a 16 byte: 1,7 MB a 8-13 GB/s = **0,13-0,2 ms**, in parallelo ai suoi esperti e ai densi della padrona.
- Guadagno: le attese CPU escono dal percorso critico (**−1/−2 ms a finestra**, S), il pool resta libero per il filo
  dell'host e per la catena, e la scrittura smette di dipendere dal carico CPU (M: −7% con OktoCGControl aperto,
  −13% con 6 core occupati, `Kspin`).
- Costo: il server deve sapere dove sta il blob (puntatore host nel piano dello strato, che oggi la CPU calcola
  comunque: `kind[i] == -1`), e la lettura mappata del blob va fatta da molti blocchi (§2a: 4,3 GB/s con un blocco
  solo). La 3080 Ti non può farlo: sull'Oculink la sonda dà `pcie_frac 0.18` (M).
- Da misurare prima: la banda delle letture mappate da 64+ blocchi sulla 2080 Ti (il banco §7 la misura con
  `dense_mib` a 0 e un blob «host»: da aggiungere se il primo giro conferma il resto).

Due cose piccole nello stesso spirito: la **testa Q5_K (437 MB, ~0,5-0,9 ms)** divisa per righe di vocabolario fra le
schede con uno scambio di top-k (−0,3 ms, S); il **primo passo della catena** calcolato per tutte le T posizioni
candidate sul server mentre la padrona fa la testa (−0,5 ms, S). Dopo EP, non prima.

## 6. La strada col guadagno massimo plausibile e il suo tetto

**EP vero con server persistente (§3a) + miss sul server (§5)**: finestra 25,3 → **17-20 ms** (S), cioè **+25/+45%**
sulla scrittura, **tetto onesto ~140 t/s** sul banco (oggi ~104), centro 120-130. Sotto i 115 (esposto > 200 µs a
strato nel banco §7) non vale 2-3 settimane.

Che cosa lo decide, in ordine:
1. **l'esposto per strato con flag sul device** (§7, `exp.persist` a «T=3 as Vpeer»): ≤ 100 µs → tetto 140;
   150-200 → 120; > 200 → lasciar perdere;
2. la banda delle letture mappate da molti blocchi sulla 2080 Ti (per §5);
3. la lettura dei prompt: senza §3e si perde il 33-42% a 32K/128K. Va messa nel conto prima di scrivere il server.

Lavoro (S): server persistente che fonde `hadamard_rows` + `quantize_q8_1_rows` + `native_expert_grouped` +
`scatter_rows_f32` in un lancio per finestra con tabella di residenza sul device e rotazioni applicate fra le
finestre (6-10 giorni); `wait_copy` della padrona su due flag e il piano CPU senza il peer (2 giorni); il prompt
(§3e, 5-8 giorni); i miss sul server (3-5 giorni). Tutto opt-in, verifica a testo identico al `--peer-device` di
oggi a cache fissa.

## 7. Prototipo: `tools/ep_bench.cu`, e le due leve nel motore

**`tools/ep_bench.cu`** (compilato, non eseguito: il banco della coda F24/F25 aveva le schede). Due kernel persistenti
e grafi nella forma del motore, 48 strati a finestra, 40 finestre, su **3080 Ti padrona + 2080 Ti server**, memoria
mappata portabile, flag con `__threadfence_system`, letture `__ldcv` a 16 byte come `copy_from_mapped`. Per ogni
configurazione (T = 2/3/4 come `Vpeer`, caldi alla padrona, righe bf16, righe sommate per token, soli flag) misura la
mediana dei µs a strato in cinque forme:

| forma | che cosa |
|---|---|
| `floor` | la padrona da sola con i soli esperti suoi (il lavoro del server sparisce): il limite |
| `all` | la padrona da sola con tutti gli esperti (una scheda sola) |
| `persist` | server persistente, flag sul device nei due versi (il disegno di §3a) |
| `host` | server guidato dall'host a ogni strato come oggi: spin sul flag, 2 copie H2D, 4 lanci, spin su `cudaStreamQuery`, poi il flag, con i cambi di scheda di `On` |
| `host-ns` | come `host` senza il cambio di scheda intorno allo spin (`STRATA_PEER_NOSWITCH`) |

`exposed = forma − floor`; più la distribuzione (p50/p90) dell'attesa al flag delle righe. I blob sono letture di
1,7 MB per intero (i kernel degli esperti sono limitati dalla banda), i densi 62 MiB a strato (il pack: 2987 MiB /
48). Attenzione: il banco **sottostima** il lavoro della padrona (una lettura di banda, non 60 kernel piccoli:
~90 µs contro 210 misurati), quindi l'esposto in `persist` è un **tetto superiore** del vero (nel motore gli hit
locali e i densi coprono di più il server).

Costruzione ed esecuzione (sotto `bench_lock.py`, ~30 s con le due schede libere, OktoCGControl può restare aperto):

    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc" -O2 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -o D:\ds4_work\strata\ep_bench\ep_bench.exe tools\ep_bench.cu
    python <corse_2080ti>\20261006_due_schede\bench_lock.py ep-bench1 -- D:\ds4_work\strata\ep_bench\ep_bench.exe 40 62 1728 > D:\ds4_work\strata\ep_bench\ep_bench_run1.txt

Lettura attesa (S): `host` ≈ 400-500 µs (come i 430-530 di `Vpeer`), `host-ns` qualche decina in meno, `persist`
80-170; la colonna `wait@rows` dice quanto del server resta scoperto.

**Nel motore** (ramo `lab/upsync-v0.2.9`, opt-in, nessun lavoro GPU cambia):
- `STRATA_PEER_NOSWITCH=1`: `PeerExperts::finish` e `apply_pending` interrogano stream ed evento del peer senza
  `cudaGetDevice`/`cudaSetDevice` intorno (`peer_experts.cpp`); se il driver rifiuta una query il motore lo dice una
  volta e torna al cambio di scheda. È la (d5) di PIPELINE2 sul percorso del peer (la 0b del verificatore,
  `STRATA_P2_FLUSH_NOSWITCH`, è già in questo ramo: `verify.cpp:4975`).
- `strata decode peer:` con `STRATA_DECODE_TIMING=1` e `--peer-device`: ms di host a finestra dentro `launch()` e
  dentro l'attesa di `finish()`, strati con una quota sul peer, voci ed esperti per strato-finestra. Serve a separare
  nel «CPU» della riga dei tempi la parte del pool da quella del peer (oggi indistinguibili, §2).

## 8. Che cosa non è provato

- `ep_bench.exe`: compilato, **mai eseguito**: i numeri di §3a restano S finché non gira.
- `STRATA_PEER_NOSWITCH` e la riga `decode peer`: compilati, **non provati**, testo non verificato (la leva non tocca
  il lavoro GPU; la verifica giusta è `--peer-device 1 --peer-reserve-mib 3000` a cache fissa, con e senza la
  variabile, testo identico).
- Tutte le stime di §3a dipendono da un numero che il motore non ha mai misurato su questa coppia: i µs del
  server persistente con il kernel vero degli esperti. Il banco li approssima con letture di banda.
- La lettura del prompt in EP con i densi su entrambe le schede (§3e) è solo disegnata.
