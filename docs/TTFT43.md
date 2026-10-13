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

## Misure (13/10, banco TTFT, build a2111f8b)

Coda `coda_ttft43.sh` dalle 03:20 alle 04:05 (fA fB0 A B fBL C), poi B2 B3 (05:08-05:22, `coda_ttft43_avvio.sh`) e B4 A2
B5 con `monitor_avvio.py` sulla vita 2 (05:24-06:17, `coda_ttft43_avvio2.sh`). Prompt e richieste come nella tabella
sopra; TTFT in secondi, **misurato**.

| richiesta | caso | A Daily r1 | A2 = Daily r2 (MULTI, senza CONV_DISK) | B (leve) | B2 | B3 | B4 | B5 | C (B + HELP) |
|---|---|---|---|---|---|---|---|---|---|
| cc1_t1 | freddo | 39,38 | | 35,31 | 45,80 | 36,75 | | | 34,75 |
| cc1_t2 | c | 10,11 | | 8,94 | 11,44 | 8,89 | | | 8,95 |
| cc2_t1 | b | 14,94 | | 14,50 | 14,75 | 16,08 | | | 14,67 |
| cc1_t4 | b | 16,33 | | 16,81 | 16,91 | 16,26 | | | 16,33 |
| cc3_t1 | a | 38,17 | 16,81 | 16,23 | 15,22 | 16,31 | 15,05 | 16,48 | 15,06 |
| dsh2_t1 | a | 32,53 | 4,88 | 4,91 | 4,95 | 4,86 | 5,06 | 4,89 | 4,89 |
| cc1_t5 | d | 20,16 | 19,69 | 6,39 | 6,45 | 6,36 | 6,34 | 6,36 | 6,33 |
| dsh1_t3 | d | 10,33 | 10,36 | 5,14 | 5,30 | 5,06 | 5,19 | 5,26 | 5,09 |
| **avvio vita 1 / vita 2 (s)** | | 67 / 66 | 67 / 68 | 67 / **277** | 67 / 70 | 68 / **206** | 67 / 67 | 70 / 68 | 67 / 67 |
| arena vita 2 (GiB/s) | | 0,92 | 0,91 | **0,19** | 0,91 | **0,22** | 0,91 | 0,91 | 0,91 |

- Caso d (ripresa dopo il riavvio): 20,2 / 10,3 s → **6,3-6,5 / 5,1-5,3 s** in tutte e sei le corse con
  `STRATA_CONV_DISK=10` (`conversation disk cache: hit 48188` e `hit 33881` in ogni `engine2.log`, lettura e
  validazione del file 0,7-0,9 s). Caso a (client alternati, file del prefisso): 38,2 / 32,5 → 15-16,8 / 4,9-5,1 s.
- Testo a cache fissa: fB0 e fBL uguali a fA su **12/12** richieste, casi d compresi (fBL ha
  `STRATA_PREFIX_DISK_MULTI=24 STRATA_CONV_DISK=10` + `STRATA_PROMPT_ENCODER=1`). I bracci con la config del Daily
  (`--elastic`, adattamento) differiscono da A in modo variabile da corsa a corsa (B 11/12, B2 6/12, B3 5/12): è la
  cache elastica, non le leve.
- C (`STRATA_PREFILL_HELP=1`): nessun guadagno sui turni corti c/e (8,95 contro 8,94; 5,34 contro 5,31).
- A2 è girato col `build-0143-daily` già promosso alle 05:06 a r2 (= a2111f8b, `STRATA_PREFIX_DISK_MULTI=24` nel JSON,
  `STRATA_CONV_DISK` spento): il braccio «A» della coda legge exe e config del Daily al momento della corsa. Senza
  `STRATA_PROMPT_ENCODER` (sta nel bat, che il banco non legge).

### Il riavvio da 277 s non viene dalle conversazioni su disco

Richiesta del 13/10: con `STRATA_CONV_DISK=10` il riavvio passa da ~67 a ~277 s «perché tutte le conversazioni vengono
ricaricate all'avvio»; serviva un caricamento pigro. **Il caricamento è già pigro**, e il riavvio lento ha un'altra
causa:

1. **Codice** (`src/program/generate.cpp`, avvio del server): con `STRATA_PREFIX_DISK_MULTI` e `STRATA_CONV_DISK`
   l'avvio legge due variabili e scrive due righe, senza aprire né elencare file. La cartella si elenca e il file si
   legge solo quando una richiesta ha un confine di turno che la RAM non copre (`prefix_cache_keyed_paths`, poi la
   stessa lettura del system turn). Nei log della vita 2 la prima riga `conversation disk cache: hit` arriva su
   cc1_t5, la terza richiesta, non all'avvio.
2. **Dove si perde il tempo**: nelle vite lente è il caricamento dell'arena degli esperti (39,97 GiB) a scendere da
   0,91-0,94 a 0,19 / 0,22 GiB/s. Il resto dell'avvio è normale (B3: embedding 0,4 s, proiezioni 1,1 s). Il monitor
   delle vite veloci mostra che l'arena legge da **PhysicalDrive1 = C:, NVMe Lexar EQ790** (il GGUF in
   `C:\Users\imanu\Downloads`) a ~1 GB/s con la CPU quasi ferma. I file delle conversazioni stanno su
   `D:\ds4_work\strata\lab_prefix`, che è il **SATA EMTEC X160**: un altro disco.
3. **Frequenza nel lab**: su **1038** avvii con l'arena da 39,97 GiB nei log di `corse_2080ti\` (dal 1/10), **67 (6%)**
   hanno caricato sotto 0,5 GiB/s, e 65 di questi **senza** `STRATA_CONV_DISK`. Uno è di stanotte, multi-chat
   `mc2_m43b-pack3b-det` alle 05:41: 0,22 GiB/s, senza la leva. Con la leva: 2 lente su 7 vite 2; senza (fA fB0 A A2):
   0 su 4. Sono numeri troppo piccoli per legare la leva al rallentamento, e il codice non tocca né C: né l'avvio.
4. **Causa del caricamento lento: non identificata.** Le vite lente erano il 9° (B, 03:47) e il 4° (B3, 05:18) avvio
   da 40 GB in pochi minuti. Ipotesi da verificare: l'NVMe C: che rallenta per temperatura (lettura SMART solo da
   amministratore: `Get-StorageReliabilityCounter`), oppure il blocco in RAM di 33 GB (VirtualLock + working-set
   minimum) che rientra nel tasso misurato. Non è Windows Search: `lab_prefix` non è indicizzata.

Codice del motore **non modificato**: un secondo meccanismo pigro sopra quello esistente non cambierebbe nulla.
La proposta di riaccendere `STRATA_CONV_DISK=10` è stata applicata al Daily r2 alle 08:37 del 13/10: il controllo
successivo in sola lettura trova `STRATA_PREFIX_DISK_MULTI=24` e `STRATA_CONV_DISK=10` nell'env di
`D:\ds4_work\strata\Strata\strata-0143-daily.json`. Il lanciatore ha `STRATA_PROMPT_ENCODER=1`; il banco lo deve
impostare esplicitamente perché non usa il lanciatore. Il riavvio lento (6% degli avvii storici) resta da capire.

Il controllo senza elevazione del 13/10 trova entrambi i dischi `Healthy`, ma `Get-StorageReliabilityCounter`
fallisce con «Accesso a una risorsa CIM non disponibile per il client». Non ci sono nuove misure della temperatura
e non si attribuisce il rallentamento alla temperatura o a `VirtualLock` senza una misura.

## Prossimo passo (esatto)

1. Sotto `bench.lock`, B-A-B con lo stesso Daily r2: `--ple-io direct`, `--ple-io ram`, `--ple-io direct`. Tutte le
   altre opzioni e gli env uguali (MULTI 24, CONV 10, prompt encoder 1). Il PLE in RAM richiede circa 26,8 GiB
   aggiuntivi: i circa 78 GiB liberi a macchina ferma non sono una misura del margine dopo il caricamento degli
   esperti. Registrare RAM libera e working set dopo READY, costo di caricamento PLE, avvio e TTFT di entrambe le
   vite. Il braccio RAM è valido solo con la riga `PLE table locked in RAM (--ple-io ram)`; altrimenti fermarlo.
2. Per chiudere questo confronto usare `tools/ttft43_bench.py`, che importa il banco originale in sola lettura e
   copia la fixture pubblica già esistente (SHA-256
   `0cd2059a3e705c99544e0f78e7eda40f76be45c034f794bbfdabc8054df9a1be`). `--output-root` va nel workspace Codex,
   `--prefix-root D:\ds4_work\strata\lab_prefix\codex_completion_20261013` tiene le cache sullo stesso disco del
   banco precedente; ogni tag crea una cartella nuova e non cancella le precedenti. Piano `short`: cc1_t1, cc1_t2,
   pausa 12 s, dsh1_t1, pausa 25 s e riavvio; cc3_t1, dsh2_t1, cc1_t3, dsh1_t2. Sono sette richieste con ripresa di
   entrambe le conversazioni e due client nuovi col prefisso su disco. `--require-prefix-hits 2` e
   `--require-conversation-hits 2` verificano il riuso; `--require-ram-locked` per il braccio RAM. Timeout o risposta
   vuota fermano la corsa, lasciano log e risultati parziali e restituiscono un codice diverso da zero. Impostare
   la modalità con `--args="--ple-io direct"`/`--args="--ple-io ram"`: la config Daily non contiene `--ple-io`, quindi
   il vecchio `--set-opt` non può sostituirla. Le nuove misure GPU sono ancora **da eseguire**.
3. Se le attese PLE restano un costo materiale, valutare la cache delle sole righe usate salvata su disco; oggi non
   è implementata. Prima misurare la modalità RAM già esistente. Il rallentamento sporadico dell'arena esperti è
   separato: una diagnosi termica richiede accesso ai contatori, senza riavvii ripetuti arbitrariamente.

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
304 OK. L'eseguibile è installato come Daily r2; config e lanciatore non sono stati modificati nel completamento.

Misure del 13/10 (sezione Misure) tutte con questa build: nessuna build nuova, il codice non è cambiato.

### Completamento CPU del 13/10

La verifica rimasta aperta era `prefix_cache_file_test`, sotto `STRATA_BUILD_CONVERSATION_TESTS`, che
`build_lock.py` non accende. Build separata `build-ttft-cpu-completion`, MSVC 19.44, Visual Studio 2022 x64 Release,
CUDA/HIP spenti, `STRATA_BUILD_TESTS=OFF`, `STRATA_BUILD_CONVERSATION_TESTS=ON`, `STRATA_NATIVE_EXPERTS=OFF`:

```powershell
cmake -S . -B build-ttft-cpu-completion -G "Visual Studio 17 2022" -A x64 `
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_NATIVE_EXPERTS=OFF `
  -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=ON
cmake --build build-ttft-cpu-completion --config Release --parallel 2 --target `
  prefix_cache_file_test prefix_cache_portable_test conversation_cache_test `
  conversation_split_failure_test conversation_memory_test conversation_file_test `
  session_save_reclaim_test message_boundary_test conv_cache_test
ctest --test-dir build-ttft-cpu-completion -C Release --verbose --output-on-failure --timeout 60 `
  -R '^(prefix_cache_file_test|prefix_cache_portable_test|conversation_cache_test|conversation_split_failure_test|conversation_memory_test|conversation_file_test|session_save_reclaim_test|message_boundary_test|conv_cache_test)$'
python -B tools/test_ttft43_bench.py
```

**Misurato: 9/9 CTest passati, 7,32 s**. Native e portable SHA-256: 252 controlli ciascuno; cache conversazioni
4225; memoria 25; file 183; reclaim 119; confini messaggi 21. Fallimenti di allocazione sullo split: preservati
8/12/16 casi con 1/2/3 stadi. Politica di ritenzione: 17 casi passati. Comprende roundtrip v2 multi-stadio e 131k,
file corrotti, atomicità della scrittura, chiavi in un passaggio dei confini di turno e LRU. File temporanei nel
build e `TEMP`/`TMP` impostate a `build-ttft-cpu-completion/test-scratch`; nessun modello o GPU usati. Log:
`build-ttft-cpu-completion/ttft-cpu-ctest.log`.

**6/6 test CPU del wrapper passati**: rifiuto del PLE non bloccato in RAM e di hit mancanti, pulizia del processo
proprio se l'avvio fallisce (nessuna pulizia quando la porta è occupata prima del lancio), sette richieste e due
vite, timeout fermato alla prima richiesta con risultato parziale conservato. Import della fixture reale verificato
con `--only-fixture`, senza motore. Il codice del motore/server non cambia: non si ripete la suite server da 304
test già passata, né si ricompila CUDA per un aggiornamento del banco e della documentazione.
