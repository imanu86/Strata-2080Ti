# lab/struct-b — il lavoro speculativo della pipeline (leva B)

Base: `lab/speed4` (v0.1.43 + pila del fork + leve speed, tutte opt-in). Obiettivo: più lavoro utile della RTX 3060
(stadio 0, ferma ~2/3 del tempo) nella pipeline a due finestre. Convenzione: **M** = misurato (log del motore, traccia
`STRATA_PIPELINE_TRACE`, `force_trace`), **S** = stimato.

## 0. Conclusione

- **B c'è quasi sempre, sbaglia quasi sempre.** Sul banco pulito (`DET-x9-u43-a`, L02, 600 finestre) B parte in 536
  finestre (**89%**) e 63 restano sotto il gate (theta 0,10); il 61% citato viene dalla cattura nsys. Il punto debole
  è un altro: B è sul percorso solo in 167 casi su 536 (**31%**). A è accettata intera nell'80% delle finestre
  (`force_trace`), quindi a sbagliare è quasi sempre la **riga 0 di B**, cioè la congettura sul token bonus di A (M).
- **Questo dipende dalla regola, non dal caso.** La regola di `--spec-min-p` chiude A alla prima bozza sotto 0,5.
  La riga 0 di B è proprio quella bozza scartata, quindi B scommette sempre sulla bozza meno sicura: `p_on` cade fra
  0,1 e 0,4 e la calibrazione la conferma (decile 0,1: 17% sul percorso; decile 0,4: 44%).
- **La 3060 non ha posto per una seconda finestra.** B finisce lo stadio 0 **5,8 ms prima** del verdetto di A
  (mediana; p10 1,9 ms). Una finestra T=1 sulla 3060 costa ~7-9 ms (M). Restano fuori quindi due strade:
  - `--pipeline-windows 3`: lo stadio 1 è il collo, e B è già pronta in anticipo;
  - una seconda B in serie a B: sfora il verdetto e richiede un terzo verificatore, un hand-off copiato e la
    riparazione della cella K/V che B2 sovrascrive (§3).
  La stima della seconda B è di −0,2/−0,35 ms a finestra (S), troppo poco per questa complessità.
- **La leva scelta: dimensionare la finestra fresca sulla pipeline.** Una finestra fresca paga i due stadi in serie
  qualunque sia T: T=1 costa ~26,6 ms per **un** token, e da T=1 a T=2 si spendono solo ~1,3 ms (M, traccia).
  `STRATA_SB_FRESH=<tmax>` fa prendere alla finestra fresca anche la prima bozza scartata dalla regola, quando restano
  ≤ tmax righe e la bozza ha probabilità ≥ `STRATA_SB_FLOOR` (0,10). Stima: −0,5/−0,75 ms a finestra su L02 (S, §4).
  `STRATA_SB_THETA` apre il gate di B senza `STRATA_PIPELINE_DEBUG`; `STRATA_SB_B` estende anche B (default spento).

## 1. Come la pipeline decide B oggi (`src/program/generate.cpp`, ciclo `--pipeline-windows 2`)

- **A** è la finestra più vecchia non verificata, **B** la successiva, **D** una B sbagliata ancora in volo.
- **Catena del drafter.** Al verdetto di A parte una catena MTP (`mtp.chain_launch`, grafo per passo, stream ad alta
  priorità sulla 2080 Ti):
  - *kind 2* (fresca, dopo un verdetto fuori percorso): 7 uscite. La nuova A prende le sue bozze con la regola di
    min-p (`t_rule`) appena la decide la prima bozza sotto soglia (`CE`). B si fa dalle uscite seguenti: la sua riga
    0 è la bozza `A.T-1`, cioè quella che ha chiuso A.
  - *kind 1* (dopo un verdetto sul percorso): catena sopra A verificata, forzata lungo le bozze della B appena
    promossa, poi bonus e bozze nuove: `B.T + S_mtp - 1` uscite.
- **Lancio di B sullo stadio 0** (`make_b`, blocco «B, speculatively»): quando A ha finito lo stadio 0 e
  `p_on = Π prob(bozze di A) · prob(riga 0 di B) ≥ theta`. Prima del lancio A si committa in modo speculativo sullo
  stadio 0 (snapshot GDN prima).
- **Verdetto.** B è sul percorso se A è accettata intera e la riga 0 di B coincide col bonus vero. Altrimenti B
  diventa D: riceve piani vuoti (`DOOM_SKIP`), finisce, e lo stadio 0 torna al commit reale di A (snapshot + replay).
- **«chain late»** (165 su 167) conta le volte in cui, al lancio dello stadio 1 di una A promossa, la catena kind 1
  non ha ancora fatto B. Succede sempre (la catena parte in quell'istante) e non pesa: B finisce comunque prima.
- **«forced-chain disagreements»** (2-5 su ~170) conta le volte in cui la catena kind 1 non riproduce le bozze di A:
  allora `p_on = 0`. Sono rare, quindi la catena ancorata al residuo vero ripete quasi sempre la catena precedente.

## 2. Numeri

### 2a. Esiti di B (M, `DET-x9-u43-a`, `force_trace.txt`)

| | L01 (521 finestre) | L02 (600) |
|---|---|---|
| B lanciate / sotto il gate / sul percorso | 493 / 27 / 171 | 536 / 63 / 167 |
| A accettata intera | 410 (79%) | 481 (80%) |
| A con T=1 (prima bozza < min-p) | 191 (37%) | 254 (42%) |
| fresche: ms · token | 30,52 · 2,01 | 28,69 · 1,71 |
| speculative: ms · token | 20,41 · 1,87 | 18,85 · 1,72 |
| rollback: fresca dopo un RB contro fresca senza RB (B accanto) | 30,13 contro 31,98 | 27,99 contro 29,17 |

Una B sbagliata non costa nulla di misurabile: la fresca che segue un rollback non è più lenta.

### 2b. La finestra nel tempo (M, `STRATA_PIPELINE_TRACE` di `DET-cp-clean`, L02, 600 finestre, build del 7/10)

| fresca (435) | ms | | speculativa (164) | ms |
|---|---|---|---|---|
| verdetto → A decisa (catena) | 2,10 | | verdetto → stadio 1 | 0,49 |
| stadio 0 di A | 10,07 | | stadio 1 di A | 19,21 |
| F0 → L1 | 0,17 | | | |
| stadio 1 di A | 15,86 | | | |
| **finestra** | **28,31** | | **finestra** | **19,82** |
| B: fine stadio 0 prima del verdetto | 5,82 (p10 1,9) | | | 5,31 |

Per T (finestra · stadio 1): fresca 26,6 · 15,0 (T=1), 27,9 · 15,6 (T=2), 30,5 · 16,9 (T=3), 35,3 · 19,6 (T=4);
speculativa 16,1 · 15,6, 19,1 · 18,6, 22,8 · 22,2, 28,5 · 27,2. A parità di T lo stadio 1 speculativo è più lento di
+0,6/+3,0/+5,4/+7,5 ms: lì la catena kind 1 (`B.T + 3` passi da ~0,94 ms) gira sulla 2080 Ti **insieme** allo stadio 1,
con lo stream a priorità massima. Nelle fresche la catena gira durante lo stadio 0, con la 2080 Ti ferma.

## 3. Opzioni valutate (ms a finestra su L02, ~25,9 ms)

| opzione | stima | esito |
|---|---|---|
| theta 0 (lanciare anche le B sotto 0,10: 5 su 63 sarebbero sul percorso) | −0,08 (S) | tenuta come `STRATA_SB_THETA`, utile con la leva sotto |
| `--pipeline-windows 3` (C dietro B sullo stadio 0) | 0 | scartata: lo stadio 1 è il collo e B è già pronta 5,8 ms prima |
| far partire B prima | 0 | scartata, per la stessa ragione |
| **B2** = secondo candidato del bonus (runner-up del drafter) alla stessa posizione, T=1, dopo B | −0,2/−0,35 (S) | scartata, vedi sotto |
| **finestra fresca +1 bozza** (`STRATA_SB_FRESH`) | **−0,5/−0,75** (S) | **implementata** |
| catena kind 1 a priorità bassa / fermata alla decisione di B (contesa con lo stadio 1 nelle speculative) | −0,3/−0,5 (S) | fuori leva (catena MTP): segnalata al coordinatore |

**B2 nei dettagli.**
- Copertura: B2 vince quando A è intera, la riga 0 di B è sbagliata e il runner-up è giusto: 0,8 × 0,62 × ~0,3 ≈
  **15%** delle finestre in cui gira (S; il runner-up a ~0,3 non è misurato).
- Guadagno per vittoria: senza bozze B2 rende un token. Il guadagno viene solo dalla B successiva, che scommette sulla
  prima bozza (~0,58 contro 0,31): **~2,4 ms** per vittoria (S, modello a due stati con g = 15,2 ms/token e
  fresca−speculativa = 9,8 ms).
- Tempo: con B2 a T=1 la 3060 finisce ~1 ms dopo il verdetto nelle fresche e ~3-4 ms dopo nelle speculative.
- Costi strutturali:
  - un terzo verificatore sulla 3060 (~160 MiB di cache esperti in meno);
  - una copia dell'hand-off, perché l'hand-off è per parità;
  - la riparazione della cella K/V `B.p` dei 4 layer QSA dello stadio 0, che B2 sovrascrive quando poi vince B
    (28% delle finestre);
  - un top-2 sul device nella catena.

## 4. La leva: finestra fresca + 1 bozza (`STRATA_SB_FRESH`, opt-in)

**Perché conviene.** Siano `q` la probabilità della bozza esclusa, `r` quella della bozza dopo, `g` ~15,2 ms per
token (costo medio) e Δ ~9,8 ms il valore di una finestra speculativa rispetto a una fresca (§2a). Il guadagno di una
riga in più in una fresca a T=1 vale

`−c_riga + q·g − q·(1−r)·Δ`

con questi termini:
- `c_riga`: il costo della riga, ~1,3 ms da T=1 a T=2 (M, compresa la catena);
- `q·g`: i token attesi in più;
- `q·(1−r)·Δ`: la perdita sulla B successiva, che ora scommette su `q·r` invece che su `q`.

Con `q` = 0,3 e `r` = 0,5 il guadagno è +1,8 ms; il pareggio è a `q` ≈ 0,13, da cui il floor di 0,10-0,15. Le
finestre interessate sono le fresche a T=1, il ~40% di tutte. Stima: **−0,5/−0,75 ms a finestra** (S). Da T=2 a T=3
`c_riga` sale a ~1,6-2,6 ms e il margine si assottiglia: tmax 2 di default, 3 da provare.

Nelle speculative una riga in più costa ~3 ms di stadio 1 (contesa con la catena) e il conto torna ~0. Per questo
`STRATA_SB_B` resta spento.

**Cosa cambia nel codice.**
- `sb_extend` in `generate.cpp`, nel blocco `lab/struct-b`, applicato nei due punti in cui si decide la T di una A
  fresca e in `make_b`.
- La bozza aggiunta è già calcolata, perché è quella che ha deciso la chiusura di A. A quindi parte allo stesso
  istante di prima.
- B prende come riga 0 la bozza successiva, con `p_on` = prodotto vero.

Env:
- `STRATA_SB_FRESH=<tmax>`: 0 o assente = spento;
- `STRATA_SB_FLOOR=<p>` (default 0,10);
- `STRATA_SB_B=<tmax>` (default spento);
- `STRATA_SB_THETA=<f>`: gate di B senza `STRATA_PIPELINE_DEBUG`.

Riga di log `strata struct-b:` per richiesta, con `STRATA_DECODE_TIMING`.

**Correttezza.** Ogni riga resta verificata dallo stadio 1 e cambia solo la composizione delle finestre. Il testo
greedy è lo stesso, a meno di quasi-pareggi numerici: gli esperti CPU usano un kernel diverso a T=1, e con
`STRATA_IQ_MT_MIN=1` le righe sono bit per bit indipendenti da T (nota in `native_expert.cpp`). Le verifiche sono nel §5.

## 5. Verifiche e misure

(in corso)
