# DRAFTWIN: quante e quali righe verificare, nella pipeline a due schede

Branch `lab/draftwin` del fork `imanu86/Strata-2080Ti`, da `lab/chainfuse` (`50c04dc0`: `lab/upsync` = upstream v0.1.42 più
le patch del Daily, più `STRATA_CP_*` di `docs/CRITPATH.md` e `STRATA_CHAIN_FUSE2` di `docs/CHAINFUSE.md`). Configurazione
del Daily: 3060 = stadio 0, 2080 Ti = stadio 1 con testa e drafter MTP, `--spec 4 --spec-min-p 0.5 --mtp rt_it48k`,
`--layer-split 16 --pipeline-windows 2`, chat singola a 131k.

Obiettivo del fronte: più token accettati per finestra di verifica (oggi 1,97 su L01 e 1,71 su L02) senza cambiare le
scelte del target. Tutto opt-in: `STRATA_DW=1`. Senza la variabile il ciclo dimensiona A e B con `--spec-min-p` come prima.

Stato: compilato (vedi in fondo), installato no, provato no. Nessuna GPU, nessun motore, nessun test eseguito.

## 1. Dove si perdono i token: le misure

Dati: `strata work`, `pipeline classes`, `pipeline calibration` dei log `DET-*/engine.log` del banco forzato (B-A-B,
`STRATA_FORCE_IDS` + `STRATA_FORCE_WINDOWS`), e le tracce `pipe_trace.txt` di `DET-cp-clean` (tre richieste L02 con lo
stesso codice di base, B6 MMVQ acceso) lette con lo script `pipe_rows.py` (verdetto→verdetto per classe e T, stadio 1
`F1−L1`, catena `CL→CE`).

**Finestre e righe (L02, 600 finestre, 1024 token).** `rows/window 1:254 2:175 3:75 4:95`, `accepted 426 of 614`
(69%). Il 42% delle finestre ha una riga sola: il primo token della catena aveva p < 0,5. Il drafter è calibrato
(B6_PIANO, profondità 0: p 0,3 → 36-43% accettate, 0,5 → 56-61%, 0,7 → 68-84%); la calibrazione dei decili di p_on della
pipeline lo conferma (0,2 → 27%, 0,3 → 36%, 0,4 → 45%).

**Costo di una riga, misurato nel codice attuale (DET-cp-clean, mediane di tre richieste L02):**

| classe | T=1 | T=2 | T=3 | T=4 | per riga |
|---|---|---|---|---|---|
| fresca (verdetto→verdetto) | 23,2 | 28,7 | 34,9 | 40,5 | **~5,7 ms** |
| speculativa (verdetto→verdetto) | 16,2 | 19,2 | 22,9 | 27,4 | **~3,7 ms** |
| stadio 1 fresca (`F1−L1`) | 13,1 | 15,5 | 18,3 | 21,9 | ~2,9 ms |
| stadio 1 speculativa | 15,6 | 18,6 | 22,1 | 26,4 | ~3,6 ms (+2,5 ms a ogni T: la catena sulla stessa 2080 Ti) |
| catena fresca `CL→CE` (taglia di A decisa) | 0,94 | 1,8 | 2,65 | 2,6 | **~0,9 ms per passo**, sul percorso critico |

Accettate medie per T: T=2 0,76 · T=3 1,38 · T=4 1,81. La terza bozza aggiunge 0,43 token: a 15,5 ms per token (1,71
token / 26,45 ms) vale 6,7 ms, e costa 5,6-6,6 ms (riga + passo di catena). **Pareggio.** La premessa «una riga costa
~0,3 ms» vale solo per i kernel densi del micro-bench B6 (0,28 + 0,26 ms sulle due schede): sul percorso critico una
riga fresca costa 5,7 ms (passo di catena 0,9 + stadio 0 ~2,5 + stadio 1 ~2,5-2,9: esperti, lavoro host per layer,
proiezioni) e una speculativa 3,7.

**La penalità fresca.** Una finestra fresca costa 7,0 / 9,5 / 12,0 / 13,1 ms in più di una speculativa a T = 1..4 (lo
stadio 0 di B è nascosto dietro lo stadio 1 di A, la catena di A non è sul percorso). Media pesata sulle taglie viste
~8-9 ms. Le fresche sono il 73%: 164 B su 543 lanciate sono sul percorso (30%), e **la riga 0 di B è per costruzione il
token che il drafter non voleva** (quello sotto min-p, `B.tok[0] = oc[A.T−1]`): p_on = P(A intera) × p(quel token),
quasi sempre sotto 0,4.

### Il modello, con questi numeri

Riga in più in A (bozza con p, cumulata q delle bozze prima): vale q·cal(p) token ≈ q·cal(p)·15,5 ms; costa 5,7 ms
(fresca) o 3,7 (speculativa) **più** la penalità fresca persa: la finestra dopo è speculativa solo se A è intera E la
bozza trattenuta è giusta, quindi ogni riga aggiunta moltiplica P_on per cal(della bozza successiva) (~0,4-0,6).

- T=1 → T=2 con p=0,6 (cal 0,66): +0,66 token (10,2 ms) contro 5,7 + 8·0,66·(1−0,5) = 8,3 ms → **si prende** (oggi sì).
- p=0,5 (cal 0,57): 8,8 ms contro 8,0: pareggio → min-p 0,5 è il punto giusto per la prima riga (come B6 aveva trovato).
- **Riga di coda** (il token sotto min-p, già calcolato dal passo che ha fermato la catena, quindi senza costo di
  catena): a T=1 con p=0,4 (cal 0,5): +0,5 token (7,75 ms) contro 5,7 + 8·0,5·0,5 = 7,7 → pareggio; con p=0,3 perde.
  Oggi quel token è la testa della B speculativa, che lo verifica gratis (è il bonus di A) e se è giusto fa risparmiare
  la penalità fresca: non è un token «gratis».
- Terza bozza (T=3 → T=4) con q≈0,42 e p=0,6: +0,25 token (3,9 ms) contro 5,7 + 8·0,25·0,5 = 6,7 → **perde**: oggi
  si prende (p ≥ 0,5) e costa.
- **Albero / top-2 alla prima posizione incerta**: +P(rifiuto)·P(top2 giusto) ≤ 0,3·0,4 = 0,12 token per riga in più
  (1,9 ms) contro 5,7 ms: **in perdita di 3 ms per finestra**; in più i layer GDN sono ricorrenti, un ramo chiede uno
  stato forkato o kernel con maschera ad albero su sm_75 e sm_86 (B6_PIANO, confermato leggendo `verify.cpp`: la finestra
  è una sequenza di T posizioni consecutive, K/V e stato GDN in ordine). Scartato.
- **Lookup/suffix + MTP**: già nel motore (`--suffix-draft 3`, `pick_lookup`, `lookup_next`, `--lookup-chain`): su L01/L02
  lavora su 2-19 finestre di 450-660 (B6_PIANO). Sul banco non è una leva; su codice o tool call lo è già, ma non si
  misura con L01/L02.
- **Calibrazione di p_on**: il decile 0,0 è sul percorso il 16% delle volte; il gate theta 0,10 trattiene 56-77 B per
  richiesta: lanciarle varrebbe 0,16·8 ≈ 1,3 ms a fronte di ~1,2 ms di contesa host (`fresh+Bbeside` contro `fresh`).
  Pareggio: il gate è già dove deve stare.
- Tetto di qualunque selezione di righe con questo drafter e questi costi: **oracolo +12-16%** (B6, replay). Regole
  realistiche: +1-3%. Per gli 80 t/s servono il costo per riga (5,7 → ~2 ms: esperti e host per layer) e la penalità
  fresca (8 ms × 73%): sono fronti kernel/host (`CRITPATH`, `CHAINFUSE`, thread host per stadio), non di politica.

## 2. La leva: `STRATA_DW=1`, taglia di A e B per token attesi al millisecondo, consapevole della pipeline

Scelta perché è l'unica delle candidate che il modello non dà in perdita, perché unifica in una regola i casi che la
soglia fissa tratta male (terza bozza a p 0,5-0,6, coda a p 0,4-0,5 quando la penalità fresca è bassa, B più lunga perché
le sue righe costano 3,7 e non 5,7), e perché **misura da sola** quello che oggi i log non danno (costo per classe × T,
calibrazione per profondità, penalità fresca, probabilità che la bozza trattenuta fosse giusta). Guadagno atteso onesto:
**0 / +3%** su L01-L02 (B6 aveva misurato +0,2 / +1,6% nel replay della regola senza il termine della pipeline; il termine
vale «+10 punti di percorso = −0,7 ms/finestra»), cioè +0,0-0,05 token per finestra o −0,3-0,8 ms per finestra. Non è la
strada per gli 80: è il modo di chiudere il fronte «righe» con una misura invece di un'altra soglia.

### La regola

Per A (classe fresca, sul percorso critico) e per B (classe speculativa, dietro A), fra le taglie T = 1 .. min(S_mtp,
1 + bozze disponibili) si prende quella che massimizza

    V(T) = E(T) − R · [ C(classe, T) + F · (1 − P_on(T)) ]

- E(T) = 1 + Σ prodotti cumulati delle accettazioni calibrate delle T−1 bozze (token emessi in media);
- C(classe, T) = tempo verdetto→verdetto misurato di una finestra di quella classe e taglia: EMA per classe × T con
  priore lineare (fresca 17,5 + 5,7·T, speculativa 12,4 + 3,8·T: i numeri sopra), peso del priore `STRATA_DW_COST_PRIOR_N` (10);
- F = penalità fresca misurata (differenza fra le due classi, pesata sulle taglie viste; priore 8 ms);
- P_on(T) = prodotto delle accettazioni delle T−1 bozze × accettazione della prima bozza trattenuta (la sua p è nota se
  la catena l'ha già prodotta, altrimenti la media corrente delle trattenute, priore 0,4);
- R = token/ms correnti (EMA; priore `STRATA_DW_R0` 0,065), scalati da `STRATA_DW_R_SCALE`.

Accettazione cal(profondità, p): tabella appresa online per profondità (0..3) × decile di p, priore = p con peso
`STRATA_DW_PRIOR_N` (8), decadimento 0,998. Impara da ogni riga verificata fino alla prima rifiutata e dal **verdetto
controfattuale** della bozza trattenuta: quando A è intera, la riga bonus dice se `B.tok[0]` (la trattenuta) era giusta.

**Quando si decide.** A oggi parte appena un output è sotto min-p. Con DW parte appena nessuna bozza in più, nemmeno
una certa (p = 1), potrebbe battere la migliore finestra sugli output già arrivati, contando il passo di catena che si
aspetterebbe (`STRATA_DW_STEP_MS`, 0,9 ms, sul percorso fresco). B decide allo stesso modo senza costo del passo (la sua
attesa è fuori dal percorso critico; la catena ha comunque tutti i passi in coda). Il tetto S_mtp resta: nessuna finestra
oltre `--spec 4`.

Con `STRATA_DW_TAIL=0` nessuna riga sotto `--spec-min-p` (la regola può solo tagliare). Con `STRATA_DW_FPEN=0` cade il
termine della pipeline: resta la regola di Dinkelbach di B6, ma sul ciclo pipelined.

Perimetro: solo il ciclo pipelined (`--pipeline-windows 2`); il ciclo seriale è invariato. Le finestre di lookup
(`pick_lookup`, `lookup_next`) non passano dalla regola e non insegnano alla tabella (le loro p sono 1,0). Il gate theta
di B e la stima p_on sono invariati. Nessun kernel toccato, nessuna aritmetica: ogni token emesso è quello del target.

### Variabili

| variabile | default | cosa |
|---|---|---|
| `STRATA_DW` | 0 | 1 accende la regola |
| `STRATA_DW_R_SCALE` | 1,0 | scala R (> 1 più severa: meno righe) |
| `STRATA_DW_TAIL` | 1 | 0: mai una riga con p < `--spec-min-p` |
| `STRATA_DW_FPEN` | 1,0 | scala la penalità fresca F (0 = Dinkelbach puro) |
| `STRATA_DW_PRIOR_N` | 8 | peso del priore p nella calibrazione |
| `STRATA_DW_COST_FRESH` / `_SPEC` | `17.5,5.7` / `12.4,3.8` | priore del costo: ms = a + b·T |
| `STRATA_DW_COST_PRIOR_N` | 10 | peso del priore nel costo (osservazioni equivalenti) |
| `STRATA_DW_STEP_MS` | 0,9 | costo di aspettare un passo di catena per A |
| `STRATA_DW_R0` | 0,065 | R prima della prima misura (token/ms) |

### Righe di log (per richiesta, con `STRATA_DW=1`)

```
strata draftwin: A N sized (n shorter, n longer than --spec-min-p's, n with a tail below it, n waited past its stop), B N sized (n shorter, n longer, n tail); R 0.0650 tok/ms, fresh penalty 8.3 ms, held draft right 0.41
strata draftwin cost (T: ms/n, prior-blended): fresh 1:23.2/187 2:28.7/124 3:34.9/61 4:40.5/73 | spec 1:16.2/66 ...
strata draftwin acceptance depth d (decile: rate/n): 0.2:0.29/41 0.3:0.38/55 ...
```

Traccia della pipeline (`STRATA_PIPELINE_TRACE`): evento `DW seq T_minp T_dw` ogni volta che la regola sceglie una taglia
diversa da quella di `--spec-min-p` (A e B).

## 3. Come misurarlo col banco

**Attenzione a `STRATA_FORCE_WINDOWS`**: impone le righe della traccia di riferimento (`force_window` sovrascrive T e le
righe di A e B dopo la scelta), quindi con la forzatura delle finestre DW non ha effetto e un B-A-B misurerebbe il nulla.
Per una leva che cambia le righe serve il banco con **`STRATA_FORCE_IDS` senza `STRATA_FORCE_WINDOWS`**: il testo resta
quello della lista (`target picks replaced on verified rows` lo garantisce), le finestre sono libere, e si confrontano:

- `strata work: … rows/window … accepted X of Y drafts` → token per finestra (1024 / finestre): è il numero primo;
- `strata pipeline: N windows in X ms (Y ms/window)` e `pipeline classes` (quota di speculative: il termine F deve
  alzarla, −0,7 ms/finestra per +10 punti);
- `strata draftwin: …` (quante taglie cambiate e in che verso; `waited` deve restare basso: ogni attesa è 0,9 ms sul
  percorso fresco);
- t/s della richiesta (`generated in … ms`).

B-A-B: stesso exe, stessi argomenti del Daily (`docs/UPSYNC.md`, «Riprodurre il Daily»), `STRATA_DECODE_TIMING=1`,
`STRATA_FORCE_IDS=<liste>`, niente `STRATA_FORCE_WINDOWS`; B = `STRATA_DW=1`, A = niente. Tre corse alternate; il
rumore fra corse della stessa configurazione è ±5% in t/s e ±0,15 bozze accettate per finestra (B6), quindi un effetto
sotto il 3% non si vede con una corsa per braccio: contare i token per finestra, che è deterministico a parità di scelte.

Per attribuire: `STRATA_DW=1 STRATA_DW_FPEN=0` (solo Dinkelbach), `STRATA_DW_TAIL=0` (solo tagli), `STRATA_DW_R_SCALE=1.2`
e `0.8` (severità). Le tabelle di `strata draftwin cost` e `acceptance` della corsa A (con `STRATA_DW=1
STRATA_DW_R_SCALE=100`, che taglia tutto a T=1: **non** è una misura neutra) non servono: le tabelle si leggono dalla
corsa B stessa, dove la regola sceglie.

## 4. Rischi

- **Latenza della catena.** Se la regola aspetta un output in più dove min-p avrebbe già deciso (`waited`), la finestra
  fresca paga 0,9 ms; il criterio «nessuna bozza certa può battere la migliore» con il passo contato lo limita. Guardare
  `waited` e `the chain from its launch to B: fresh`.
- **Partenza fredda.** Prime finestre con i priori: costi della L02 a 131k. Su contesti corti o altre macchine i priori
  sono sbagliati finché l'EMA non li copre (`STRATA_DW_COST_PRIOR_N` osservazioni); passare `STRATA_DW_COST_*` misurati.
- **Deriva.** Costi e calibrazione sono EMA: il banco forzato è deterministico negli id ma non nei tempi, quindi due
  corse B possono scegliere taglie leggermente diverse. Il confronto è sui token per finestra e sui ms per finestra, non
  sull'identità delle finestre.
- **Il termine F presume il gate.** P_on conta la B come lanciata; sotto theta non lo è e la penalità fresca è certa. Il
  modello sovrastima un po' il valore di tenere P_on alto quando P_on è già sotto 0,1.
- **B più lunga = più lavoro host accanto allo stadio 1.** Le righe di B costano 3,7 ms solo se B va sul percorso, ma la
  B scartata (70%) ha comunque fatto servire i suoi layer dal thread host unico (`spec+Bbeside`). Non è nel modello.
- Senza `STRATA_DW` il codice è identico a `lab/chainfuse`: `t_rule`, `make_b`, i criteri `decided` e gli stessi grafi.

## 5. Prossimo candidato strutturale (fuori da questa leva, serve la GPU)

Nelle finestre speculative lo stadio 1 dura 2,5 ms in più a ogni T: è la catena «forzata» (kind 1, `B.T + S_mtp − 1`
passi, fino a 7 × 0,9 ms) sulla stessa 2080 Ti. I passi forzati attraverso le bozze di B potrebbero essere un **round**
con T righe (un grafo, le stesse parole) invece di B.T−1 passi sequenziali, e i passi oltre la taglia decisa della
prossima B non partire. Vale −1/−2 ms per finestra speculativa (−0,3/−0,5 ms medi, +1,5%): cambia i grafi del drafter,
da provare con parità sulle bozze (`accepted X of Y` uguale).

## Stato della build

Compilato: vedi il commit `FABLE-MTP-DRAFTWIN.md: stato della build` (sha dell'exe e cartella `build-draftwin-<sha>`).
Installato no, provato no: nessuna GPU usata, nessun motore avviato. Senza `STRATA_DW` l'exe deve comportarsi come
`lab/chainfuse` `50c04dc0`.

## File toccati

- `src/program/generate.cpp`: `struct DraftWin` (regola, tabelle, report), `PW::pb0`, `dw_size_a`, `make_b`, i criteri
  `decided` di A e B nel ciclo pipelined, l'apprendimento al verdetto, la riga `strata draftwin`.
- `docs/DRAFTWIN.md`.
