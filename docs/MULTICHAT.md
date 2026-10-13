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

## Leve di codice (prima stima; nessuna scritta)

In testa l'indicazione dell'owner: **una sola lettura dei pesi per più sessioni, prefill e decode nella stessa
passata** (batching misto alla Sarathi-Serve).

Vedi il report intermedio per le stime per 2/4/8 chat; le misure della seconda coda (costo fisso della lettura, cache a
16 posti, quota decode/prefill, contesti 64k, kv-resident 16k) e della terza (chat singola) vanno qui sotto.
