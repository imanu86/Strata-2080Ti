# TTFT43: tempo al primo token sulla base 0.1.43

Branch `lab/ttft43` del fork `imanu86/Strata-2080Ti`, worktree `Strata-wt\ttft43`, da `origin/lab/struct-a` d21b2870
(= codice del Daily «Strata 0.1.43 Daily» 075ce875 + un commit di soli doc). Campo: TTFT e riuso del prefisso e delle
conversazioni **nel tempo** (disco, riavvii, client che si alternano). Il prefisso condiviso fra sessioni
**simultanee** è dell'agente multi-chat (`lab/multichat*`); la decodifica dell'agente pipeline (`lab/pipeline2`).
Diagnosi e piano di partenza: `docs/TTFT.md` (12/10, base 0.1.42).

Stato: **misurato** = numeri da una corsa del banco; **stimato** = calcolo o dichiarazione di altri, non misurato qui.

## Commit

| sha | cosa | variabile | default |
|---|---|---|---|
| c00a8463 | port di a9ead5f3: file del prefisso scritto su un thread; un file per prefisso con LRU | `STRATA_PREFIX_DISK_SYNC=1` torna in linea; `STRATA_PREFIX_DISK_MULTI=<GiB>` | thread acceso; multi spento |
| ea73f49e | port di 48ce5f27: righe PLE del primo chunk della parte dopo lette in anticipo | `STRATA_PLE_LOOKAHEAD=0` la spegne | acceso |
| c02b6a19 | port di cfd00e3d: `docs/TTFT.md` | | |
| abd685c5 | conversazioni su disco al confine di turno, a motore fermo | `STRATA_CONV_DISK=<s>`, `STRATA_CONV_DISK_MIN` (4096) | spento |
| a2111f8b | #1852 (Guillaume Putier): il server ritokenizza solo dopo l'ultimo speciale condiviso | `STRATA_PROMPT_ENCODER=1` nell'ambiente del **server** | spento |

I tre commit del 12/10 si sono applicati puliti sulla 0.1.43 (generate.cpp e prefill.cpp in auto-merge, nessun
conflitto). Tutte le variabili nuove sono fuori dall'identità del file del prefisso: accenderle non invalida i file.

### STRATA_CONV_DISK (leva 3 del piano)

- **Salvataggio**: dopo una richiesta il motore ricorda il suo confine di turno B (il checkpoint `turn_at`, quello da
  cui riparte il turno dopo). Se resta fermo `STRATA_CONV_DISK` secondi, salva un'immagine a B in un file con chiave
  come quello del system turn (`<--prompt-cache-file>.<16 hex>`, stesso formato v2, stesso LRU di
  `STRATA_PREFIX_DISK_MULTI`). L'immagine: celle K/V [0, B) delle sessioni (le celle oltre B sono scritte dopo e non
  si prendono) + stato in corsa (GDN, PLE, code dell'indicizzatore) del checkpoint a B, stadio per stadio. È la stessa
  cosa da cui riparte un turno in RAM. Cattura sul thread principale (una richiesta che arriva aspetta la cattura),
  SHA-256 e scrittura sul thread del prefisso.
- **Lettura**: una richiesta calcola in un passaggio le chiavi di tutti i suoi confini di turno oltre quanto le danno
  RAM e system turn (`prefix_cache_keyed_paths`), elenca una volta la cartella e legge il file più lungo per la
  stessa strada del system turn (controllo token per token, validazione per stadio, upload).
- La chiave FNV dei file con chiave ora mette gli id prima della lunghezza (serve al passaggio unico); i file di
  `lab/ttft` non sono mai entrati nel Daily.
- Limiti: salva solo la conversazione **viva** al momento della pausa (una conversazione parcheggiata in RAM da
  un'altra non va su disco finché non torna viva e il motore si ferma); niente immagini; niente `--batch`; il K/V
  del draft MTP fuori dalla sua finestra resta quello che c'era (pesa solo sull'accettazione delle bozze, mai sul testo:
  le bozze sono verificate).

## Banco TTFT

`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261006_due_schede\ttft_bench.py`, coda
`coda_ttft43.sh`, build copiate con `copia_build.py`. Un server sulla porta 8102 con la config del Daily
(`strata-0143-daily.json`, sola lettura) meno la visione, `--prompt-cache-file` in una cartella nuova per braccio
(`D:\ds4_work\strata\lab_prefix\ttft43\<tag>\`). Prompt sintetici da testo pubblico upstream v0.1.43 (docs e sorgenti),
istruzioni generiche scritte nel banco, nessun dato personale. TTFT = invio → primo delta di contenuto in streaming
(template e tokenizzazione del server compresi).

| richiesta | caso | cosa |
|---|---|---|
| cc1_t1 | freddo | Claude Code, sessione nuova, ~40k (system ~24k + 12 tool ~4k + primo messaggio ~12,5k), niente su disco |
| cc1_t2 | c | stessa conversazione + un turno (tool_use + tool_result ~2,5k) |
| side1 | e | richiesta laterale (titolo), ~150 token, altro system |
| cc1_t3 | e | il turno dopo la laterale |
| dsh1_t1 | b | DSH, ~32k (system ~29k), a freddo |
| cc2_t1 | b | Claude Code sessione nuova dopo DSH (stesso system e tool) |
| dsh1_t2 | b | DSH + un turno dopo Claude Code; poi 12 s di pausa |
| cc1_t4 | b | ritorno alla conversazione cc1 dopo due altre |
| — | — | 25 s di pausa, `/unload`, riavvio completo di server e motore sulla stessa cartella |
| cc3_t1 | a | Claude Code sessione nuova, prefisso già su disco |
| dsh2_t1 | a | DSH sessione nuova, prefisso già su disco |
| cc1_t5 | d | ripresa di cc1 dopo il riavvio |
| dsh1_t3 | d | ripresa di dsh1 dopo il riavvio |

Bracci: fA / fB0 / fBL a cache fissa (`--expert-cache 1500`, senza `--elastic`, `--adapt-every 0`) per il testo (Daily,
ttft43 coi default, ttft43 con le leve); A / B / C con la config del Daily per i tempi (Daily; ttft43 con
`STRATA_PREFIX_DISK_MULTI=24 STRATA_CONV_DISK=10` + `STRATA_PROMPT_ENCODER=1`; C = B + `STRATA_PREFILL_HELP=1`).

## Misure

(in corso: coda `coda_ttft43.sh build-ttft43-a2111f8b "fA fB0 A B fBL C"`, 13/10 dalle 03:10)

## Penalità di ripetizione (controllo del 13/10, sola lettura)

- `--pipeline-windows 2` decodifica in serie una richiesta con `penalty_last_n` > 0 e lo scrive una volta per motivo:
  `strata serve: --pipeline-windows: a request with repetition penalties (penalty_last_n) decodes serially`. Il server
  manda `penalty_last_n=64` appena una fra `repetition_penalty` ≠ 1, `frequency_penalty` ≠ 0, `presence_penalty` ≠ 0
  è nella richiesta o nel `sampling` della config.
- **Misurato nei log**: la riga non compare mai. `strata_sota.log` (Daily be4cf1c con `--pipeline-windows 2`, 194
  richieste reali di Claude Code e DSH dal 7 all'11/10), `strata_0143_daily.log`, i `runs_cc\*\engine.log` di Claude
  Code (cc-split-0710b: 26 righe `pipeline classes`, nessuna seriale). La stringa c'è negli exe be4cf1c e 0.1.43.
- Config: nessuna chiave `sampling` in `strata-0143-daily.json`, `strata-0142-daily.json`, `strata-2080ti-sota.json`.
  Gateway (`qwen_gateway\*.py`): nessun riferimento alle penalità; i suoi log (`log_gateway\*.jsonl`) non tengono i
  corpi, solo metadati.
- Unico caso possibile: il recupero del ciclo di ragionamento del server alza `presence_penalty` a 1,5 per il seguito
  della risposta (raro; mai visto nei log sopra).
- Costo se capitasse: la decodifica perde la sovrapposizione delle due schede per quella richiesta (stimato: il
  guadagno di `--pipeline-windows 2`, non misurato qui). Oggi: **nessun costo**.

## Build

`a2111f8b` (codice): `build_lock.py`, rc 0, 3485 s di cui 3212 in attesa del lock. `strata.exe` sha256
`22a2eb857ed36b36ab25c9089af251740bd6c398ba79d062de759b6fc41e5e29`, copiata con `copia_build.py` in
`corse_2080ti\20261004_routed_keep\build-ttft43-a2111f8b\` (build.json, `source\serve` e `source\tools` di a2111f8b,
`strata-vision.exe` del Daily). Test del server: test_prompt_encoder 7 OK, test_strata_tokenizer 11 OK, test_server
304 OK. `prefix_cache_file_test` (chiavi in un passaggio) scritto, non eseguito: è sotto
`STRATA_BUILD_CONVERSATION_TESTS`, che `build_lock.py` non accende. Compilato sì, installato no.
