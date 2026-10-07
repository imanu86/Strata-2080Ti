# Seme della cache esperti dalla massa della prefill (`STRATA_PREFILL_SEED`)

Branch `lab/prefill-seed`. Opt-in: senza `STRATA_PREFILL_SEED` nell'ambiente il motore è invariato (nessun buffer,
nessun kernel, nessuna riga di log). **Compilato no, provato no**: il codice è stato scritto a banco occupato; la
prima cosa da fare è la build e il banco qui sotto.

## Cosa fa

Oggi la residenza degli esperti in VRAM all'inizio del decode è quella statica del profilo (`--expert-profile`) più
quello che il tier adattivo (`--adapt-every 4 --adapt-swaps 96 --adapt-decay 0.7`) ha imparato dalle richieste
precedenti. La prefill non lo alimenta: `drive.d.usage` conta le scelte solo nelle finestre di decode, e un
prompt nuovo paga un adattamento lento (al banco ~4.000 cambi di residenza per richiesta, e la seconda richiesta
dello stesso testo va ~2% più veloce della prima).

Il seme porta su Strata quello che l'owner aveva fatto su DeepSeek (`ds4-port/src/cuda/prefill_observer.{h,cu}`,
`cuda_moe_prefill_vram_seed`): **a fine prefill le statistiche di routing del prompt mettono in VRAM gli esperti
giusti prima del decode**.

1. **Accumulo** (`src/prefill/prefill.cpp`, subito dopo `route(...)`): per ogni (layer, esperto) la **massa** (somma
   dei pesi di gate normalizzati, `m.w`) e il **conteggio** (scelte), kernel `router_mass_count`
   (`src/kernels/cuda/native_router.cu`) in un buffer `[n_layers × n_expert]` sulla scheda che calcola il layer:
   CUDA0 per lo stadio 0, ogni stadio dello split per i suoi layer. Il buffer vive sullo stream del prompt, quindi è
   ordinato con i chunk; viene azzerato a inizio richiesta e letto (D2H) dopo il refill di fine prefill.
2. **Piano** (`include/strata/core/prefill_seed.hpp`, `prefill_seed_plan`, puro e testato): per layer i non
   residenti ordinati per massa decrescente (parità: id crescente) e i residenti sfrattabili per massa crescente; il
   candidato i-esimo prende il posto della vittima i-esima finché `massa_cand ≥ GAIN × massa_vittima` e
   strettamente maggiore (un esperto mai scelto dal prompt non entra mai). Poi il budget: `FLOOR` scambi garantiti per
   layer, il resto a budget globale `SWAPS` per guadagno (`massa_cand − massa_vittima`) decrescente. **Gli scambi
   restano dentro lo stesso layer**, nella cache della scheda che lo possiede.
3. **Esecuzione** (`src/program/generate.cpp`, lambda `prefill_seed`, chiamata dopo `refill(err)` e dopo
   `closed_refill`): prima `apply_pending(true)` e `adapt_tick(true)` (nulla del tier adattivo in volo), poi per lotti
   pari alla capacità degli scambi (`exchange_capacity()` in modalità RAM residente, altrimenti 96):
   `resident_stage_swaps` (copia indietro nella copia RAM, sulla scheda proprietaria via `seed_home_of`, identica a
   `home_of` di `adapt`), `adapt_copy_h2d` sul refill stream della scheda (o `copy_blobs` batched per CUDA0 con
   `STRATA_DMA_BATCH`), sync, `commit_exchanges`, aggiornamento di `host_res`, `release(in)`, `prefetch(out)`,
   infine `res_upload()` (tabella di residenza su ogni scheda). Sincrono e misurato, come `closed_refill`.
4. **Prior** su `drive.d.usage` (il tier adattivo): senza prior un esperto appena seminato avrebbe usage 0 e sarebbe
   la prima vittima del giro successivo. DeepSeek inizializza la media mobile con `massa/conteggio` = una sola
   osservazione sintetica; qui `usage` è un conteggio di scelte decaduto (×0,7 ogni 4 finestre), e `massa/conteggio`
   (≈0,1) sarebbe sotto la soglia d'ingresso 2 e quindi irrilevante. L'equivalente con decay è **un orizzonte di
   scelte sintetiche**: `prior[i] = PRIOR × massa[i] × (Σconteggi/Σmassa) / token` — le scelte che l'esperto
   raccoglierebbe in `PRIOR` token se il decode instradasse come il prompt, pesate per massa.
   `usage[i] = max(usage[i], prior[i])`, mai abbassato, su tutti gli esperti visti (non solo i seminati): i candidati
   rimasti fuori budget entrano al primo giro di `adapt` se il decode li conferma. Con `PRIOR=32` il seme pesa quanto
   ~2-3 giri di adattamento e svanisce col decay: non resta appiccicato.

Regole rispettate: vittime come in `adapt` (slot `< el_core` del nucleo elastico mai sfrattati salvo
`STRATA_ELASTIC_CORE_ADAPT`; con `STRATA_PIPELINE_ELASTIC` e split gli stadi successivi sono tutti sfrattabili);
candidati mai detenuti da peer (`--peer-device`), remote o helper. Il routing non cambia: **l'output è identico bit
per bit al default**, cambia solo dove stanno i pesi (come per gli scambi adattivi).

## Variabili (ambiente, default a riposo)

| Variabile | Default | Significato |
|---|---:|---|
| `STRATA_PREFILL_SEED` | non impostata | Impostata (anche `=0`): buffer allocati e accumulo attivo («armato»), il seme si accende per richiesta. `=1`: il seme gira dopo ogni prompt. |
| `STRATA_PREFILL_SEED_SWAPS` | 512 | Tetto totale di scambi per richiesta su tutti i layer (0 = nessun tetto). ~1-2,5 MiB a blob: 512 scambi ≈ 0,5-1,2 GiB ≈ 50-100 ms di PCIe. |
| `STRATA_PREFILL_SEED_FLOOR` | 0 | Scambi garantiti a ogni layer prima del budget globale. |
| `STRATA_PREFILL_SEED_GAIN` | 1.5 | Soglia moltiplicativa: il candidato entra se la sua massa è almeno GAIN volte quella della vittima (minimo 1). |
| `STRATA_PREFILL_SEED_MIN_TOKENS` | 256 | Il seme gira solo se la prefill ha letto almeno N token **nuovi** (sotto: riga `saltato`). |
| `STRATA_PREFILL_SEED_PRIOR` | 32 | Orizzonte in token del prior su `usage` (0 = non toccare `usage`). |
| `STRATA_PREFILL_SEED_CARRY` | 1 | Riporto della massa dei prompt precedenti della conversazione viva (vedi prompt cache). |

Prerequisiti: `--serve`, `--expert-profile` (tabella di residenza) e una sorgente esperti; altrimenti
`strata prefill seed: needs --expert-profile ... off`. In modalità RAM residente il seme riserva 96 buffer di scambio
(`reserve_exchanges(96)`, idempotente: la riserva del tier adattivo, se maggiore, resta).

### Leva a caldo per richiesta: `lab=`

Sulla riga della richiesta (`GEN <max_new> lab=PREFILL_SEED:1,PREFILL_SEED_SWAPS:256 <id,...>`) la chiave `lab=`
accetta `NOME:VALORE,...` con i nomi delle variabili senza `STRATA_` (maiuscole o minuscole). Serve il motore armato
(`STRATA_PREFILL_SEED` impostata all'avvio, anche `=0`): così l'A/B è due client alternati sullo stesso processo, uno
con `lab=PREFILL_SEED:1` e uno con `lab=PREFILL_SEED:0`, senza riavvio e con la prompt cache calda. Il parser
(`prefill_seed_lab_apply`) prende **solo** le leve `PREFILL_SEED*` e lascia stare gli altri nomi: su
`origin/lab/fable-plan` la stessa chiave è di `fp_lab_parse` (`include/strata/core/fp_lab.hpp`), che oggi rifiuta i
nomi sconosciuti — al merge basta far saltare a `fp_lab_parse` i nomi che iniziano con `PREFILL_SEED` e passare la
stessa stringa a entrambi i parser. Un elemento malformato lascia la richiesta ai default dell'ambiente (riga
`strata serve: lab: ...`).

## Prompt cache: cosa vede la prefill

Con `--prompt-cache` la prefill legge **solo il suffisso** oltre il prefisso ripreso (`n − resume` token). La massa
accumulata è quindi quella del suffisso, e `MIN_TOKENS` si confronta con i token nuovi: un turno breve di una
conversazione lunga non semina (il decode si è già adattato a quella conversazione).

`CARRY=1` (default) tiene sul host la massa e il conteggio dell'ultima prefill della conversazione viva
(`seed_carry_*`): se la richiesta riprende dalla sessione viva (`from_live`), la massa del suffisso si somma a quella
dei prompt precedenti e il piano vede la conversazione intera; se riprende da un checkpoint parcheggiato/su disco o
riparte da zero, il riporto riparte dal suffisso corrente. I token generati fra una prefill e l'altra non sono nella
massa (solo i token del prompt). Usare «anche gli ultimi token» del prefisso ripreso richiederebbe salvare la massa
per checkpoint accanto alla prompt cache: non fatto, il riporto copre il caso che conta (la conversazione che
continua).

Nota sul banco «due richieste uguali»: alla seconda richiesta il prefisso è in cache, la prefill vede 0-pochi token
nuovi e il seme **non gira** (`saltato`). È voluto: il confronto è prima richiesta con seme vs prima richiesta senza
(vedi sotto), non seconda vs seconda.

## Come misurarlo

Banco deterministico dell'owner (`STRATA_FORCE_IDS` + `STRATA_FORCE_WINDOWS`, `STRATA_DECODE_TIMING=1`), due schede
con `--layer-split`, `--pipeline-windows 2` e cache elastica di default. Riferimento: due richieste **uguali** di
seguito senza seme — la prima paga l'adattamento, la seconda no (≈ +2%). Il seme deve avvicinare la prima alla
seconda.

1. Braccio A: `STRATA_PREFILL_SEED=0` (armato, spento) — richiesta 1 poi richiesta 2 dello stesso testo. Annotare
   t/s di entrambe e, con `STRATA_EXPERT_USAGE=<prefisso>`, la riga `strata expert usage: ... residency flips seen` e
   la quota «AT THE CALL ... of the gate mass were resident» per richiesta.
2. Braccio B: `STRATA_PREFILL_SEED=1` (o `lab=PREFILL_SEED:1` sulla richiesta 1) — stesso testo, stessa sequenza.
   Attesi: la riga `strata prefill seed: N scambi, M esperti, copertura massa X% -> Y%, ms (pianificati, candidati,
   token, MiB copiati, prior su K celle)` dopo il prompt della richiesta 1; t/s della richiesta 1 più vicino a quello
   della richiesta 2 di A; meno residency flips nella richiesta 1; quota di massa residente alla chiamata più alta
   dall'inizio del decode.
3. Costo del seme: i `ms` della riga (D2H + piano + copie) si pagano una volta, prima della prima finestra; vanno
   sottratti o inclusi coerentemente nel confronto end-to-end (`REUSED` → primo token).
4. Testo: con `STRATA_FORCE_IDS` il testo è forzato; senza, il testo greedy deve restare **identico** fra A e B
   (cambia solo la residenza). Se differisce c'è un bug (copia in uno slot sbagliato), non una variante accettabile.
5. B A B alternati per ogni voce, come nelle convenzioni di `FABLE_PLAN.md`. Leve da esplorare in ordine: `SWAPS`
   (256/512/1024), `GAIN` (1.2/1.5/2), `PRIOR` (0/16/32/64), `FLOOR` (0/2/4).

Test host-only (nessuna GPU): `ctest -R prefill_seed_test` con `STRATA_BUILD_TESTS=ON` — piano, soglie, tetto,
floor, regole di esclusione, prior, chiave `lab=`.

## Rischi e punti da verificare per primi

- **Non compilato**: `generate.cpp` è una funzione gigante; i nomi catturati (`el_core`, `core_adapt`, `stage_of`,
  `helper_holds`, `remote_opt`, `peer`, `adapt_tick`, `req_pseed`, `seed_bufs`) sono quelli che `adapt` e
  `closed_refill` usano nello stesso ambito, ma la prima build può sollevare warning di ombreggiatura (`bytes`, `lay`
  dentro la lambda) o una firma da sistemare.
- **Residenza coerente**: dopo il seme `host_res` è l'unica verità; `res_upload()` la carica su tutte le schede. Se
  una copia fallisce il motore esce con `ERR prefill seed: ...` (come per gli scambi adattivi): niente stati a metà.
- **Modalità RAM residente** (`complement_ready`): gli scambi passano dai buffer di scambio e `commit_exchanges`
  dopo ogni lotto; quelli oltre la capacità vengono scartati dal lotto da `resident_stage_swaps` (conteggiati come
  non fatti: `N scambi` < `pianificati`). Se `N` è sempre molto sotto il piano, alzare la riserva.
- **Cache elastica**: il seme gira fra il refill di fine prompt e la prima finestra, dove `elastic_step` non
  interviene; il nucleo non è mai vittima (salvo `STRATA_ELASTIC_CORE_ADAPT`). Con `STRATA_PIPELINE_ELASTIC` i
  fill di una crescita (`el_fill`) sono pubblicati a fine richiesta precedente, quindi non in volo qui.
- **Pipeline**: nessuna finestra in volo fra prefill e decode (invariante di `pl_drain`), quindi niente `adapt_fence`;
  se in futuro il prompt si sovrapponesse al decode di un altro slot (`--batch`), il seme andrà recintato come
  `adapt`.
- **Costo**: 512 scambi × ~1,5 MiB ≈ 0,75 GiB per richiesta lunga; su prompt molto frequenti e corti conviene
  `MIN_TOKENS` alto o `SWAPS` basso. Il kernel di accumulo costa T×10 atomiche per layer e chunk: trascurabile.
- **Prior troppo forte**: con `PRIOR` alto il tier adattivo impiega più giri a correggere un prompt che non predice
  il decode (es. prompt lungo di contesto e risposta su altro). Se la seconda metà del decode peggiora, abbassare
  `PRIOR` o `SWAPS`.
- **Prompt cache**: con `CARRY=0` ogni turno vede solo il suffisso; con `CARRY=1` un cambio di argomento dentro la
  stessa conversazione porta con sé la massa vecchia (nessun decay sul riporto: da valutare un fattore).

## Riferimenti

- DeepSeek: `moe-aggressive-commit/ds4-port/src/cuda/prefill_observer.{h,cu}` (485e268, cd45e77),
  `ds4-p4a/ds4_cuda.cu` `cuda_moe_prefill_vram_seed` (621fbe9, ce806c6). Ablazione: massa +7,3% contro frequenza
  −9,2%.
- Strata: `STRATA_CLOSED_ROUTING` (`closed_router_count`, `closed_refill`) come modello di conteggio in prefill e di
  scambi a lotti; `adapt` per le regole di vittime/candidati e i percorsi di copia; `STRATA_EXPERT_USAGE` per la
  misura di chiamate, massa e flip per esperto nel decode.
