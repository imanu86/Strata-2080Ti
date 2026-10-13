# Multi-chat su 3060 + 2080 Ti (lab/multichat43)

Stato al 13/10, notte. Obiettivo dell'owner: più chat (Claude Code / DSH, anche sotto-agenti della stessa chat madre)
in parallelo, con un aggregato vicino al doppio della chat singola (~77/70 t/s) e la chat singola che non peggiora.

- Base: `lab/struct-a` d21b2870; la build provata è `build-structa-075ce875` (sha256 9ef62e54..., **lo stesso exe
  della 0.1.43 Daily**). Il branch si chiama `lab/multichat43` perché `lab/multichat` esiste già su origin (lavoro del
  7/10 su base 0.1.39/40, superato da upstream: batch-mtp sullo split e gruppi sono in 0.1.42/43); spostarlo avrebbe
  richiesto un force-push.
- Harness: `moe-aggressive-commit/docs/porto/strata_adattivo/corse_2080ti/20261013_multichat/mc2.py` (copia in sola
  lettura la config del Daily `strata-0143-daily.json`, senza `--prompt-cache-file` e `--vision`; le opzioni date in
  `--extra` sostituiscono quelle del Daily). Code di prova `coda_mc2_notte*.sh`, tabella `tabella.py`.
- Legenda: **(M)** misurato, **(S)** stimato.

## Scenari

| scenario | cosa |
|---|---|
| `solo:S` | una chat alla volta, contesto 8k in cache: decode medio della chat singola con quella config |
| `pf:S` | una lettura alla volta di una coda nuova di 128/512/2048/8192 token su un prefisso 8k in cache (TTFT, s) |
| `same:N:S` | N chat insieme sullo stesso contesto 8k, domande diverse, 384 token ciascuna |
| `diff:N:S` | N chat insieme su contesti 8k diversi (scaldati all'avvio: se la cache non li tiene, si rileggono) |
| `cold:4:S` | 3 chat in decode e, 3 s dopo, una con un contesto nuovo da 16k (prefill accanto al decode) |
| `agents:N:P` | N sotto-agenti con lo stesso prefisso da 32k, 3 turni ciascuno: ogni turno aggiunge un tool result (~1,5k token) e la risposta precedente, 200 token per turno, partenze sfalsate di 0,7 s |
| `*:G` | come sopra con contesti da 64k |

Aggregato = token generati / wall dal primo invio all'ultima fine (TTFT e letture comprese).

## Prove di configurazione (M, 13/10 01:46-03:00, una corsa per braccio)

`CC` = `--conversation-cache-mib 20480 --conversation-cache-slots 8` (il Daily ha 6144/2).

| scenario | Daily | `--batch 4 --batch-mtp` g1 +CC | **`--batch 4 --batch-groups 2` +CC** | `--batch 8 --batch-mtp` g1 +CC | `--batch 8 --batch-groups 2` +CC |
|---|---|---|---|---|---|
| solo:S (decode, 2 richieste) | 55,8 / 75,2 | 66,6 / 70,0 | 66,4 / 68,3 | 62,7 / 62,5 | 65,4 / 67,5 |
| pf:S TTFT 128/512/2k/8k (s) | 1,19/1,92/4,45/9,97 | 1,34/2,09/4,78/9,22 | 1,34/2,17/4,77/10,47 | 1,38/2,20/4,86/10,59 | 1,39/2,20/4,91/9,61 |
| same:2 | 61,4 | 63,6 | 60,6 | 60,5 | 57,6 |
| same:4 | 70,8 | 73,6 | **84,7** | 69,7 | 76,0 |
| same:8 | 70,6 | 69,7 | 70,0 | 60,8 | 71,8 |
| diff:2 | 35,6 | 31,3 | 35,0 | 27,3 | 35,9 |
| diff:4 | 37,4 | 34,1 | 40,2 | 28,4 | 40,8 |
| diff:8 | 24,7 | 33,6 | 39,3 | 28,9 | 40,6 |
| cold:4 | 23,8 | 34,0 | **47,9** | 30,2 | 45,4 |
| agents:4:P | 18,4 | 20,2 | **23,3** | 19,0 | 19,9 |
| agents:8:P | 27,5 | 26,0 | **33,3** | 27,2 | 32,1 |

Letture:
- **La raccomandazione di upstream 0.1.43.1 (a0116b27, #2038) vale anche qui**: gruppi in pipeline senza
  `--batch-mtp` è il braccio migliore ovunque (same:4 +20%, cold:4 +101%, diff:8 +59%, agents:8 +21% sul Daily).
  `--batch-mtp` a un gruppo pareggia il Daily. Il porto di a0116b27 è solo il default (con `--batch-groups 2` esplicito
  non serve codice).
- **Siamo lontani dal 2x**: il massimo è 84,7 t/s (same:4), e sul caso reale (agenti) 33 t/s contro 27,5.
- 8 slot non rendono più di 4: ogni slot toglie VRAM agli esperti (sotto).
- La chat singola con `--batch` configurato: 66-68 t/s contro 75 del Daily (−10%, due sole richieste: vedi coda 3).

## Diagnosi

### 1. Perché `--batch` spegne la pipeline della chat singola (M dal codice, peso M)

`generate.cpp` (decisione di `--pipeline-windows`): `o.batch != 0 ? "not with --batch slots"`. Non è un'incompatibilità
dimostrata: il commit d'origine (bfa532e9) dice «off for anything it does not support: --batch slots ...; nothing is in
flight outside the two pipelined loops». Con `--batch` configurato la richiesta sola gira comunque sul percorso solo
(GEN con MTP), ma senza le due finestre in volo. In più con `--batch-groups > 1` si spegne la cache elastica («not
validated with --batch-groups; using fixed cache») e gli slot tolgono VRAM agli esperti: con `--batch 4 --batch-groups 2`
la cache passa da 4246 a 3382 slot sulla 3060 (−20%) e da 9148 a 7739 sulla 2080 Ti (−15%).
Peso sulla chat singola: −10% (M, 2 richieste per braccio; la coda 3 lo rimisura con 8). Il 47-49 t/s del 12/10 era
con `--batch 6` (2,18 GiB di slot sulla 3060).

### 2. Perché la finestra batch costa 75-94 ms (M)

`--batch 4 --batch-mtp`, same:4 (4 chat, 6,9 righe a finestra, 77 ms): stadio 0 = GPU 18,9 + esperti CPU 6,7 ms;
stadio 1 = GPU 26,1 + esperti CPU 9,8 ms; commit 3,3 + emissione 3,4 + adattivo 1,5 ms; il resto (~7 ms) passaggio e
staging. **Gli stadi corrono uno dopo l'altro** (`run_slot_rows` sincronizza lo stadio 0 e poi chiama lo stadio 1):
ogni scheda calcola il 25-34% della finestra e aspetta il resto. È il «lavorano un quarto del tempo» dell'owner.

Profilo GPU (`STRATA_VERIFY_PROFILE=1`, M, finestre miste solo+batch, i fusi spenti dal profilo): pesi densi e
hyper-connection ~35-40% del tempo GPU dello stadio, esperti in VRAM ~22%, attenzione QSA ~12% (ogni chat ha il suo
K/V: in batch non si condivide). Un riga in più nella stessa chat costa ~2,2/2,9 ms (stadio 0/1, da STRUCT_A); in batch
fra chat diverse ~3,7 ms a stadio (S, da 6,9 righe in 77 ms), perché ogni chat sceglie esperti suoi.

### 3. Esperti su CPU, attese, prefill che blocca (M)

- Esperti CPU: 6,7 + 9,8 = 16,5 ms su 77 (21%) a 4 chat; a 8 slot (`--batch 8 --batch-mtp`) salgono a 17 + 29 ms
  per finestra (meno VRAM, più chat) e la finestra arriva a 98 ms: per questo 8 slot rendono meno di 4.
- Attese GPU dell'host (wait): 45 ms su 77 (58%) = calcolo GPU, ma fatto da una scheda alla volta.
- **Prefill**: è il costo dominante nei casi reali.
  - Ogni lettura costa ~1 s fisso più ~1,0-1,8 ms a token: 128 token 1,2 s, 512 1,9 s, 2k 4,5 s, 8k 10 s con
    prefisso in cache (M, Daily). A 33k da zero 1452 tok/s (0,69 ms/token).
  - In agents:8 la lettura dei prompt è il **64% del wall** sia sul Daily (104 s su 161) sia coi gruppi (85 s su 134);
    il decode è la parte minore.
  - Coi gruppi in pipeline (`piped`) una lettura non lascia decodificare nessuno finché non finisce
    (`batch_read_piece` torna 0 se `piped`): il prompt ferma tutti (è il #2050 di upstream). Con un gruppo solo la
    lettura va a chunk e fra un chunk e l'altro gli slot decodificano metà del tempo del chunk: una lettura da 8k beside
    decode è durata 14,6 s (580 t/s, 2 parti) contro 11,1 s da sola, e gli slot hanno decodificato 2,6 s.
  - Ammissione: dopo la lettura la sessione viene **copiata** nello slot (`copy_to_slot`: K/V in RAM pinned salvato in
    un'immagine e ricaricato, su ogni stadio): ~13 ms per 1k token (8k 0,10 s; 131k 0,7-1,5 s, M 12/10), e la stessa
    copia all'indietro quando il turno dopo riparte dallo slot. Nel frattempo nessuno slot decodifica.
- **Cache**: col Daily (6144 MiB / 2 conversazioni) 8 contesti diversi non restano in cache: diff:4 e diff:8
  rileggono 8k a richiesta (11 s l'una). Con 8 posti si rileggono ancora i contesti più vecchi, perché ogni
  conversazione (contesto + domanda + risposta) occupa un posto. Il prefisso comune dei sotto-agenti invece funziona già:
  un posto parcheggiato che contiene il prefisso da 33380 token serve tutti (letti solo i tool result), ma ogni agente
  che perde la sua storia la rilegge (59k token nuovi letti sul Daily contro 35k coi gruppi e 8 posti).

### 4. Cosa fa diversamente blange48 (#2058: 4x 5080, 144/268/392 t/s a 1/4/8 client)

- 4 stadi: la sua chat singola (finestre seriali) usa una scheda su quattro alla volta; i gruppi in pipeline (4 gruppi
  da 2) le riempiono tutte. Il guadagno massimo dei gruppi è il numero di stadi (4x per lui, 2x per noi), e la nostra
  chat singola ha già la pipeline a 2 finestre, quindi il nostro punto di partenza è più alto.
- Esperti quasi tutti in VRAM (64 GB con `--trim-stage-weights`: 84-100% per scheda nella misura upstream): una
  riga in più costa solo GPU. Da noi il 45% degli esperti è in RAM e il batch fra chat diverse moltiplica gli esperti
  CPU.
- `--batch-groups 4` senza `--batch-mtp` (i gruppi non fanno bozze): esattamente a0116b27. Il suo «spec 4» vale per
  la sola chat singola.
- `--kv-resident 32768` come noi.

### 5. `--batch-groups 2` con 2 stadi fa lo stesso? (M)

Sì, è lo stesso meccanismo: 2 gruppi da 2 slot, ognuno sul suo stadio (`strata batch (pipelined, 2 groups of 2)`).
same:4: 1531 righe in 17,7 s = 86 righe/s; con `--batch-mtp` a un gruppo 6,95 token ogni 77 ms = 90 t/s di finestra
ma più tempo in ammissioni. Il guadagno dei gruppi è la sovrapposizione degli stadi; quello che perdono è l'MTP (1
token a riga invece di 1,74), l'adattivo e la cache elastica.

## Riferimenti esterni (dalla ricognizione 86ce57e)

- **a0116b27 / #2038** (0.1.43.1): solo il default; misurato sopra.
- **#1524**: finestre larghe condivise fra 2-4 stream su una scheda, 1,3-2,1x. Su una scheda: da noi il limite è la
  finestra a 8 righe (`kVerifyMaxT = 8`, 4 slot con MTP) e gli stadi seriali; idea utile per le finestre larghe.
- **ScrambledEggsOnToast/spec208** (`--batch-mtp --batch-mtp-pipeline --batch-groups 2`): bozze MTP dentro i gruppi in
  pipeline, a ondate di 2 slot, con un secondo verificatore per gruppo. Misura del fork (3 schede, tutti gli esperti
  in VRAM): +14% (prompt corti) e +6,6% (9,5k) sui gruppi senza MTP. Il codice tocca `generate.cpp` (+300 righe),
  `verify.cpp`, `resident_slots.cpp` su una base che ha 957 commit propri: da riscrivere sulla nostra, non da
  cherry-pick. Nota del fork: un gruppo finito aspetta che gli altri stadi abbiano servito gli esperti CPU prima del
  commit, quindi con esperti in RAM (noi) la sovrapposizione si riduce.

## Seconda coda (M, 13/10 02:54-03:19; interrotta per fine budget)

**Cache a 16 posti** (`--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16`):

| scenario | Daily | g2 + 8 posti | **g2 + 16 posti** |
|---|---|---|---|
| solo:S (decode) | 55,8 / 75,2 | 66,4 / 68,3 | 66,6 / 71,2 (media 68,9) |
| same:4 | 70,8 | 84,7 | 80,3 |
| diff:4 | 37,4 | 40,2 | **85,9** (nessuna rilettura) |
| diff:8 | 24,7 | 39,3 | **70,3** |
| cold:4 | 23,8 | 47,9 | 48,1 |
| agents:4:P | 18,4 | 23,3 | **26,6** |
| agents:8:P | 27,5 | 33,3 | 33,5 |

La cache più grande raddoppia le chat su contesti diversi (diff:4 2,3x il Daily). È config pura: RAM 96 GB, un
contesto da 8k parcheggiato pesa ~0,48 GB, da 33k ~1,1 GB, da 131k ~3,5-4 GB (S); 28 GiB bastano per ~8 agenti a 64k.

**Da dove viene il secondo fisso di ogni lettura** (`STRATA_PREFILL_TIMING=1`, Daily, prefisso 8k in cache, M):

| coda nuova | stadio 0 (GPU) | stadio 1 (GPU) | totale richiesta | dequant (st.0/st.1) |
|---|---|---|---|---|
| 139 token | 483 ms | 510 ms | 1,36 s | 46% / 43% |
| 529 token | 910 ms | 970 ms | 2,27 s | 39% / 39% |
| 2026 token | 2344 ms | 2245 ms | 5,31 s | 26% / 22% |
| 8192+399 token | 4653+1298 ms | 6989 ms | 13,1 s | 14% |

- Un pezzo di prompt tocca quasi tutti gli esperti dei suoi layer già da ~140 token (140 x 10 scelte su 512 esperti):
  **dequantizzare e caricare tutti gli esperti costa ~0,2-0,6 s a stadio, qualunque sia la lunghezza**. È questo il
  «secondo fisso», non la cache (il ripristino del checkpoint è 44 ms).
- **Con un pezzo solo gli stadi vanno in serie** (stadio 0 poi stadio 1): la pipeline fra stadi c'è solo fra i chunk
  di una lettura lunga. Un tool result da 2k tiene ferma ogni scheda metà del tempo.
- Il drafter dopo ogni chunk costa 89-365 ms sullo stadio 1 (`STRATA_SPLIT_MTP_BATCH=1` lo mette in batch: non provato).

Non misurati (coda fermata): `STRATA_PREFILL_HELP=1`, `STRATA_BATCH_DECODE_SHARE=2`, contesti 64k (`:G`),
`--kv-resident 16384`, la chat singola a 8 richieste (coda 3). Gli script sono pronti: `coda_mc2_notte2.sh` dalla riga
`b4g2-cc16-help`, `coda_mc2_notte3.sh`.

## Configurazione proposta oggi (M)

`--batch 4 --batch-groups 2 --conversation-cache-mib 28672 --conversation-cache-slots 16` sopra la config del Daily.
Aggregato: 2 chat ~60 (come il Daily), 4 chat 80-86, 8 chat 70, agenti 27-34 t/s. **Costo: chat singola −8/−10%**
(66-71 contro 75 del Daily, campione piccolo) perché `--batch` spegne `--pipeline-windows 2` e la cache elastica e
toglie ~15-20% di slot esperti. Quindi **non** va nel Daily finché la leva L1 non riporta la chat singola al livello del
Daily: oggi è una config da usare solo nelle sessioni con più agenti.

## Disegno: una sola lettura dei pesi per più sessioni (direzione dell'owner)

### Cosa dicono i numeri

- Finestra di decode batch (4 chat, 7 righe): ~77 ms, cioè ~11 ms a riga; una riga in più nella stessa chat ~5 ms
  (2,2 + 2,9 ms dei due stadi).
- Un pezzo di prefill paga comunque tutti gli esperti (dequant + caricamento, ~0,5 s a stadio): una riga di decode
  agganciata costa solo il suo calcolo e la sua attenzione, ~0,57 ms a token a stadio al ritmo di un chunk da 8k,
  **~1,1-1,5 ms a riga sui due stadi contro ~11 ms in finestra (S)**.
- Ma un passo misto dura quanto il chunk (0,5-5 s): la chat in decode avanza di 1,7 token a passo. Agganciare righe di
  decode a un singolo prefill risparmia solo una finestra da 77 ms per chunk (−2…−15% del tempo di prefill):
  **il batching misto puro, da solo, rende poco qui**, perché il nostro prefill non è limitato dalla lettura dei pesi
  residenti (il caso di Sarathi/vLLM), ma dal toccare tutti gli esperti in ogni chunk.
- La forma che rende è il **prefill impacchettato di più sessioni**: le code di più agenti (tool result di 1-2k) nello
  stesso chunk. Quattro letture da 2k costano oggi 4 x 2,3 s a stadio = 9,2 s; un chunk da 8k ne costa 4,65 (M per i
  pezzi, S per l'impacchettamento): **−50% del tempo di prefill**, che negli agenti è il 64% del wall. Le righe di
  decode delle altre chat salgono sullo stesso chunk quasi gratis.
- E i chunk impacchettati consecutivi vanno in **pipeline fra i due stadi** (pacco k sullo stadio 1, pacco k+1 sullo
  stadio 0), come già fa la lettura lunga.

### Cosa serve nel motore

1. Un passo di prefill con **segmenti**: ogni riga ha la sua sessione (`SessionState` dello slot). Pesi densi, HC,
   esperto condiviso e esperti instradati sono per riga: si concatenano. Per sessione: GDN (scan a chunk con stato
   iniziale per segmento, oggi uno solo), attenzione QSA di prompt sul K/V del segmento (causale) e append K/V, PLE,
   indexer. Le righe di decode sono segmenti lunghi 1-2 (riga + bozza MTP) che usano l'attenzione di decode
   (`qsa_decode_attn_batch` esiste già).
2. Commit per segmento: accetta/rifiuta la bozza (`commit_slot_prefixes`), bozze MTP degli slot dopo il passo
   (come `batch_step`).
3. Scheduler (server + motore): a ogni passo prende tutte le letture in attesa (fino a un tetto di token) e le righe
   delle chat in decode; quando nessuno legge, finestre di decode (gruppi in pipeline). Con le sessioni in fasi diverse
   (una legge, una genera, una aspetta un tool) ogni passata si riempie con quello che c'è.
4. Ammissione senza copia: la lettura scrive direttamente nella sessione dello slot (oggi legge nella sessione
   principale e poi `copy_to_slot` copia tutto il K/V: ~13 ms per 1k token, 1-1,5 s a 131k, a ogni turno).

### Prefisso condiviso dei sotto-agenti

Funziona già, senza codice (M): la lettura dei sotto-agenti riprende dal checkpoint del turno parcheggiato che contiene
il prefisso comune (in agents:8 `cached 33380` su 35k: si leggono solo le code). Serve solo una cache con abbastanza
posti, uno per agente più il prefisso (16 posti sopra). Il K/V e lo stato GDN del prefisso però si **copiano** a ogni
ammissione (nessuna condivisione fisica): oltre 64k la copia pesa (0,7-1,5 s a 131k); la condivisione vera (pagine K/V
copy-on-write fra slot) è un passo successivo, utile solo coi contesti lunghi.

### Stima (S) degli aggregati, t/s

| | 2 chat | 4 chat | 8 chat |
|---|---|---|---|
| Daily oggi, chat in decode (same/diff) (M) | 61 / 36 | 71 / 37 | 71 / 25 |
| g2 + 16 posti oggi (M) | 61 / 35 | 80 / 86 | 70 / 70 |
| + L1 L2 (pipeline per il solo, gruppi con MTP) | 65-70 | 105-120 | 100-120 |
| Agenti, Daily oggi (M) | ~29 (S) | 18 | 28 |
| Agenti, g2 + 16 posti oggi (M) | — | 27 | 34 |
| + L3 prefill impacchettato e righe agganciate | 35-40 | 45-55 | 55-65 |
| + L2 | 40-45 | 50-60 | 60-75 |

Il 2x della chat singola (~150) non è raggiungibile con queste leve su due stadi con il 45% degli esperti in RAM:
il massimo realistico è ~1,5x (S) sulle chat in decode e ~2x il Daily attuale sugli agenti.

## L3 — prefill impacchettato: `STRATA_PACK_PREFILL` (scritto il 13/10, opt-in)

### Cosa dice il log degli agenti (M, `mc2_b4g2-cc16`, agents:8)

Ogni turno di un agente costa ~4,6 s di control lines: lettura della coda (1,6-2,1k token su 35k in cache) **~4,1 s**
(390-480 tok/s), park della conversazione uscente 65-180 ms, ripristino di quella entrante 70-110 ms, copia nello slot
210-250 ms. Il server dà le control lines a **una richiesta alla volta** (`_take_control`): il motore non vede mai due
prompt insieme, quindi l'impacchettamento ha bisogno anche del server.

### Disegno

1. **Server** (`_pack_hint`): una richiesta che sta per aspettare le control lines (lock preso da un'altra) manda
   avanti il suo prompt, `BHINT <id,...>`, senza risposta. Solo testo, solo se l'INFO del motore dice `pack_hints=N`.
2. **Motore, slot di pacco**: con `STRATA_PACK_PREFILL=N` (1-8) il motore crea N sessioni per stadio, come gli slot del
   batch ma **fuori dalle finestre** (coi gruppi in pipeline uno slot inattivo viene sporcato dalle righe di
   riempimento: non può fare da deposito). Costano VRAM come uno slot (0,36 GiB su CUDA0 e 0,66 GiB su CUDA1 ciascuna,
   M dal log; RAM pinned del K/V in streaming come uno slot).
3. **Lettura impacchettata** (`pk_read`, nella `read_part` della richiesta che ha le control lines, solo per una
   lettura in un pezzo, cioè coi gruppi in pipeline): per ogni prompt mandato avanti, fino a N, il motore prende il
   prefisso in cache più lungo (una conversazione parcheggiata, letta sul posto con `ConversationCache::peek`, o un
   checkpoint del prompt corrente sotto la sua posizione), lo ripristina nello slot di pacco su ogni stadio, e legge
   la coda fino all'**ultimo confine di turno** (lo stesso `turn_at` che la richiesta userebbe) **negli stessi chunk**
   della lettura corrente: `Prefill::run_packed`. Poi salva lì il checkpoint di confine.
4. **`Prefill::run_packed`**: le righe dei segmenti concatenate e lette a chunk; hyper-connection, proiezioni ed
   esperti sul chunk intero (una lettura e un dequant degli esperti per tutti), per segmento sulla sua sessione e alle
   sue posizioni: storia PLE, conv e ricorrenza GDN, append K/V (con lo staging per segmento se il K/V è in streaming),
   indexer, selezione, attenzione. `on_chunk`/`on_stage_chunk` vedono solo le righe della sessione propria (draft MTP,
   checkpoint periodici). Il percorso di default è un ramo `pk == nullptr`: stesso codice di prima.
5. **Ripresa**: quando arriva la `BGEN` della richiesta mandata avanti, lo slot di pacco è una sorgente come uno slot
   (`copy_from_slot`), più lunga del parcheggio; il checkpoint di confine passa nella catena principale (il turno
   dopo riparte da lì), la richiesta legge solo l'intestazione del turno. Lo slot di pacco si libera.
6. Opzioni: `STRATA_PACK_MIN` (default 128: code più corte non si impacchettano), `STRATA_PACK_CHUNK` (default 0 = un
   chunk come una lettura sola fino a 8192; N = chunk di ~N token, così un pacco grande va in pipeline fra i due stadi).

Limiti: niente immagini, niente `--batch-mtp`, solo con `--prompt-cache`; un `BHINT` di una richiesta poi annullata
legge per niente (lo slot si ricicla dal più vecchio); la richiesta che ha già le control lines non aspetta nessuno.

### Stima prima della misura (S)

Dalla tabella `STRATA_PREFILL_TIMING`: 4 code da 2k costano 4 x (2,34 + 2,25) = 18,4 s di GPU in serie, una da 8,6k
5,95 + 6,99 = 12,9 s (−30%); con `STRATA_PACK_CHUNK=4096` i due stadi si sovrappongono (S −40…−50%). Per ogni richiesta
impacchettata restano copia dallo slot di pacco (~0,2 s), intestazione, copia nello slot e park (~0,6 s). Sugli agenti
a 8: S 40-50 t/s contro 33,5 (la stima di 55-65 sopra contava anche le righe di decode agganciate, non fatte).

### Misure di costo e parità (M, 13/10 05:22-08:45, build `build-multichat-b6d8df2d`)

Scenario `packdet:4:P` di `mc2.py`: 4 sotto-agenti sul prefisso da 33k, turno 0 uno alla volta, poi il turno 1 (coda
~1,6k token a testa) mandato **insieme** con `max_tokens 1` (il wall di questa fase è il costo delle letture), poi il
turno 1 di nuovo uno alla volta da 200 token (decodificato da solo dal checkpoint di confine che la lettura concorrente
ha lasciato). Config: g2 + 16 posti, `--expert-cache 1500 --adapt-every 0` (cache fissa). Il braccio base ha gli stessi
slot di pacco allocati e mai usati (`STRATA_PACK_MIN=999999`), quindi la stessa VRAM.

| braccio | letture concorrenti di 4 code | TTFT a0/a1/a2/a3 (s) | testo t0 | testo t1 a0 / a1 a2 a3 |
|---|---|---|---|---|
| base (con MTP) | 18,80 s | 4,8 / 9,4 / 14,0 / 18,6 | = | = / = = = |
| `STRATA_PACK_PREFILL=3` | **11,12 s (−41%)** | 9,6 / 10,1 / 10,6 / 11,0 | identico | identico / diverso a 590, 47, 128 caratteri |
| `PACK=3 STRATA_PACK_CHUNK=4096` | 11,08 s | 9,3 / 10,4 / 11,0 / 9,7 | identico | identico / diverso a 207, 47, 128 |
| base senza MTP | 19,30 s | 5,1 / 9,7 / 14,4 / 19,2 | = | = / = = = |
| `PACK=3` senza MTP | **12,33 s (−36%)** | 10,7 / 12,3 / 11,1 / 11,6 | identico | identico / diverso a 381, 231, 135 |

- Il pacco legge 1569 + 3 x ~1610 token in **8,5 s** (un passo da 6,4k), contro ~4,6 s a coda letta da sola.
  La prima richiesta (quella che ha le control lines) aspetta il pacco: il suo TTFT sale da ~5 a ~10 s, le altre
  scendono; l'ultima da 18,6 a 11 s.
- `STRATA_PACK_CHUNK=4096` (due chunk in pipeline fra gli stadi) non cambia il wall: con 6,4k token il pacco è già
  dominato dal calcolo, non dal fisso degli esperti.
- **Parità**: il testo della richiesta che legge (a0) è identico in tutte le corse, con le sue righe dentro il pacco. Le
  tre richieste impacchettate divergono a metà testo su un token quasi pari (testo coerente, stesso contenuto: per
  esempio `-hf, -mu, -dr` contro `-hf, -dr`). **Non è il drafter MTP**: senza `--mtp` divergono lo stesso. È la
  numerica della geometria del chunk: lo stesso pacco tagliato diversamente (`STRATA_PACK_CHUNK=4096`) sposta già la
  divergenza di a1 da 590 a 207 caratteri. Tra due corse base con VRAM diversa (`base` senza slot di pacco contro
  `base3`) divergono perfino i turni 0. Il testo identico bit a bit è possibile solo se ogni segmento si legge con la
  sua geometria da solo, cioè senza pacco. Conclusione: parità **numerica** (stesso contenuto, divergenza solo sui quasi
  pari), non bitwise.
- Il drafter MTP di una richiesta che riparte da uno slot di pacco ha il K/V di un'altra conversazione: le bozze
  rendono meno nel primo decode da solo, ma il testo non dipende da questo (la verifica è esatta). È così anche oggi
  per chi riparte da uno slot batch senza `--batch-mtp`. Una ricostruzione del K/V del drafter (righe finali del
  pacco ripassate nel drafter alla ripresa) è abbozzata in
  `corse_2080ti/20261013_multichat/wip_pack_drafter_rows_b6d8df2d.patch`, non finita.

## Leve di codice, in ordine

1. **L3 — prefill impacchettato di più sessioni con le righe di decode agganciate** (la direzione dell'owner, nella
   forma che rende qui). **Scritto il 13/10 senza le righe di decode** (sopra, `STRATA_PACK_PREFILL`). Grande: `Prefill::run` a segmenti (GDN e attenzione per segmento), scheduler. Primo passo
   misurabile: impacchettare solo le letture (senza righe di decode), 2 sessioni, parità di testo contro le letture
   separate.
2. **L1 — `--pipeline-windows 2` per la richiesta sola anche con `--batch`**: togliere il cancello «not with --batch
   slots» e scaricare la pipeline prima di promuovere la richiesta in uno slot; insieme, la cache elastica con i gruppi.
   Media. Condizione per mettere il batch nel Daily. **Si tocca con `lab/pipeline2`** (l'altro agente ridisegna la
   pipeline della chat singola): concordare prima chi tocca il cancello.
3. **L2 — gruppi in pipeline con le bozze MTP** (idea di spec208, da riscrivere sulla nostra base: il fork ha 957
   commit propri): S +30-40% sulle chat in decode (1,74 token a riga contro 1).
4. **#2050** di upstream: con i gruppi una lettura non ferma più gli altri (`batch_read_piece` torna 0 con `piped`).
   Media, cold:4 e agenti.
5. Ammissione senza copia (scrivere nello slot), utile coi contesti lunghi.
6. Config da provare (pronte negli script): `STRATA_PREFILL_HELP=1`, `STRATA_SPLIT_MTP_BATCH=1`,
   `STRATA_BATCH_DECODE_SHARE`, `--kv-resident 16384`, contesti 64k.

## Prossimo passo esatto

1. Lanciare `coda_mc2_notte3.sh` (chat singola Daily contro g2+16 posti, 8 richieste ciascuno; ~20 min) e le righe
   rimaste di `coda_mc2_notte2.sh` (da `b4g2-cc16-help`), sempre con `bench_lock.py`.
2. Decidere con l'owner il nome del branch (`lab/multichat43` qui) e con l'agente di `lab/pipeline2` chi fa L1.
3. Cominciare L3 dal prefill impacchettato di 2 sessioni (senza decode), parità di testo e tempo contro due letture.
