# UPSYNC_029 - la pila del fork sopra upstream v0.2.9 (1/11/2026)

Branch `lab/upsync-v0.2.9` da `v0.2.9` (c62d5e3a, 239 commit oltre v0.2.7.1) + 18 patch (cherry-pick da `lab/upsync-v0.2.7.1`, 19 meno la 07) + `stack/`.
**Stato: patch portate si (18), compilato no, installato no, provato no.** Solo controlli leggeri: `git apply`, `check_anchors.py --check` (18 su 18 `clean`), `--verify-docs`, `py_compile` e `serve/test_model_card_sampling.py` (6 test, ok). Nessuna build, nessun motore avviato.

## Esito per patch

| patch | esito | note |
|---|---|---|
| 01 elastic-cache | applicata | pulita; `expert_cache.*` non e' cambiato fra 0.2.7.1 e 0.2.9 (upstream ha solo `--vram-elastic`, altra cosa) |
| 02 split-prompt-borrow | applicata | pulita |
| 03 forced-bench | applicata | pulita |
| 04 elastic-split-pipeline | applicata | pulita |
| 05 prefix-disk | applicata | pulita (in 0.2.7.1 serviva un merge a mano su `mtp.hpp`, qui no) |
| 06 vram-free-nvml | applicata, ridotta di fatto | `STRATA_VRAM_FREE=device` e `device_free_bytes()` sono gia' in upstream; resta solo il `min` con NVML in `el_free` della cache elastica |
| 07 park-no-foreign-kv | scartata (gia' in 0.2.7.1) | PR #1503 entrata upstream, gia' fra le ritirate di `stack/manifest.json` |
| 08 pipeline-architectds | applicata | pulita |
| 09 df-defaults | applicata | pulita; DF_BRANCH e DF_PLE restano opt-in upstream, il fork li accende |
| 10 ple-diag | applicata | pulita |
| 11 prefill-sm75 | applicata | pulita |
| 12 ram-core | applicata | pulita (tocca `expert_source.cpp`, dove upstream ha messo l'area condivisa su Windows: merge automatico, da rileggere) |
| 13 sa-bet | adattata | 1 conflitto in `generate.cpp`: upstream scrive `kVerifyNarrowT - 1` (finestre fino a 16 righe, la pipeline a due finestre resta a 8) dove la patch aveva `kVerifyMaxT - 1`; tenuto `kVerifyNarrowT` piu' le righe `sa_now`/`sa_chains` della patch |
| 14 prefix-disk-multi | applicata | pulita |
| 15 ple-lookahead | applicata | pulita (upstream ha toccato il PLE multi-prompt, a8fba1f5, solo in `prefill.cpp`: nessuna sovrapposizione) |
| 16 conv-disk | applicata | pulita |
| 17 p2-flush-noswitch | applicata | pulita; `exec_` e' `[2][kWindowMaxT + 1]` upstream, la patch non lo tocca |
| 18 hc16-sm75 | adattata | 2 conflitti in `prefill.cpp`: upstream ha aggiunto i segni `pt.mark(kPfHcUp / kPfHcInj / kPfHcMix)` (e1e7e326) fra le proiezioni down/up/inject; tenuti i rami hc16 della patch e i tre segni dove li ha messi upstream |
| 19 greedy-temp0 | applicata (obbligatoria) | pulita; a temperatura <= 0 la scheda modello non porta piu' top_p/top_k/penalty, il motore resta su argmax |

Nessuna patch e' entrata upstream fra 0.2.7.1 e 0.2.9 (cercati: `--elastic`, `--prompt-cache-file`, `STRATA_PLE_LOOKAHEAD`, `STRATA_SA_BET`, `STRATA_SA_CHAIN`, `STRATA_PIPELINE_ELASTIC`, `STRATA_HC16_SM75`, RAM core, conversazione su disco).

## Cosa e' cambiato sotto la pila

- **Finestre fino a 16 righe** (9d3c8c36, `window_limits.hpp`): `kVerifyNarrowT = 8`, `kWindowMaxT` = 8 o 16 da `-DSTRATA_WINDOW_MAX_T` (auto: HIP 16, **CUDA 8**). Su CUDA di default tutto resta a 8. Le nostre patch non aggiungono costanti legate alla finestra tranne `exec_head_/exec_tail_[9]` (13, bet con T <= 3, `STRATA_SA_BET_MAXT` limitato a 3) e `PW::tok[8]` (upstream). Con `-DSTRATA_WINDOW_MAX_T=16` non e' provato niente.
- Lettura hyper-connection v3 (21f950b4, 42aee7a3, 1f05fdc2): vedi sotto, sul nostro hardware e' spenta.
- Area condivisa degli esperti su Windows (7aafac42): opt-in, vedi sotto.

## Rischi e punti da rileggere

1. **Non compilato.** Il controllo e' sintattico per costruzione dei cherry-pick, non semantico: il test di identificatori (nessun nome tolto da upstream fra 0.2.7.1 e 0.2.9 compare nelle righe aggiunte dalla pila) e' pulito, ma le firme non sono verificate. Compilare sm_75 + sm_86.
2. **13**, `generate.cpp` intorno alla riga 17988 (catena su A, `chain_n`) e `chain_launch(..., err, sa_now)`: il nuovo punto di chiamata del batch con MTP a catena (riga ~13993) usa il default `n_now = 0`, come deve.
3. **18**, `prefill.cpp` intorno a 3826-3866: l'ordine dei segni di timing e dei rami hc16; con `STRATA_PREFILL_TIMING` la riga "hc read" deve ancora sommare.
4. **12** e l'area condivisa su Windows: entrambe toccano `ArenaExpertSource::open` / `expert_source.cpp`. Merge automatico; rileggere se si usano insieme repliche con area condivisa e nucleo RAM.
5. **06**: se `STRATA_VRAM_FREE=device` non e' acceso, il `min` non cambia nulla (NVML e' letto solo con quella variabile).
6. File da ricompilare (tutto il motore, per gli header toccati: `verify.hpp`, `mtp.hpp`, `expert_cache.hpp`, `expert_source.hpp`, `prefill.hpp`, `prefill/kernels.hpp`): `CMakeLists.txt` (riconfigurare: link `CUDA::cuda_driver`, nuovi test), `src/program/generate.cpp`, `src/core/{verify,mtp,expert_cache,expert_source,prefix_cache_file}.cpp`, `src/prefill/{prefill.cpp,kernels.cu}`, `src/kernels/{ngram.cpp,cuda/qsa_select.cu}`, `src/ngram/ple_reader.cpp`, test `src/core/prefix_cache_file_test.cpp`, `tests/core/expert_cache_elastic_test.cpp`. `sycl/src/prefill/kernels.dp.cpp` (18) non si compila qui.

## Variabili di upstream 0.2.9 chieste (default su Windows CUDA)

| nome | in 0.2.9 | default Windows CUDA |
|---|---|---|
| `STRATA_COALESCE_PREFILL=N` | si, solo `serve/server.py` (+ righe `BCO`/`BCOK` del motore, `generate.cpp:12570`); "research, PAR-H"; con `STRATA_COALESCE_TOKENS` (16384), `STRATA_COALESCE_WAIT_MS` (0), adattivo `STRATA_COALESCE_MIN_REQS` / `STRATA_COALESCE_BURST_MS` | **0 = spento** |
| `STRATA_BATCH_CONCURRENT_READ` | si, `generate.cpp:16378`: il prompt si legge su un thread mentre le finestre dei slot girano; serve `--batch` con `--no-prefill-borrow`, una GPU, niente pipeline/immagini/kv-grow | **spento** (opt-in: acceso solo con valore diverso da 0) |
| `STRATA_PF_W16_CACHE_MIB=N` | si, `gemm.cu:1397` / `gemm.hpp:96`, solo CUDA: tiene la dequantizzazione FP16 di fino a N MiB di pesi sulla scheda (stessi bit); usa la scheda solo se restano > 2 GiB liberi | **0 = spento** |
| `STRATA_GDN_TC` | si, `kernels.cu:3000`; ricorrenza GDN del prompt su tensor core, prompt >= 128 token, serve sm_80+ e ~122 KB di memoria condivisa (sm_86/89/120: variante "split" da 98 KB) | motore: **spento**. Il server lo imposta a 1 (con `STRATA_GDN_CONVL2=2`) solo su **Linux**, motore Qwen `qwen4exp`, non HIP/SYCL/Metal/Vulkan (`gdn_stack_env`, "Windows: nothing"). Sulla 2080 Ti (sm_75) non puo' comunque partire |
| `STRATA_GDN_CONVL2` | si, `kernels.cu:2864` (0 / 1 / 2: 2 = conv, SiLU e norma q/k in un passaggio) | motore CUDA: **0**. Tabella AMD gfx12 e gfx1151: 1. Il server lo mette a 2 solo su Linux CUDA |
| area condivisa su Windows (7aafac42) | si: sezione con nome (`CreateFileMapping`, `win:Local\strata-replicas-<pid>`) per `--shared-expert-arena`, stessa intestazione e parola di stato, elezione di chi legge via pid (`arena_this_pid`/`arena_pid_alive`); `shared_arena_possible` ora vale per `win`; test `pinned_shared_win_test` | **spenta**: solo con `"replica_shared_arena": true` (o `STRATA_REPLICA_SHARED_ARENA=1`) e almeno 2 repliche |
| lettura hyper-connection v3 (21f950b4, 42aee7a3, 1f05fdc2) | si, `src/kernels/cuda/fused_gr.cu`: kernel esatti per 5..8 token con due righe per warp (`STRATA_GR_R2`, default 2; 0 = kernel generici, 1/2/4), somme dei warp trasposte per 5..8 token; stessi bit, decode c=8 +5% / +2% | la v3 di per se' e' `STRATA_GR_V3`: **spenta** salvo scheda compute capability 12.0 discreta con > 13 GiB (`gr_v3_default`); `=1` la forza. Sulla 2080 Ti + 3060 non gira, quindi `STRATA_GR_R2` non ha effetto |

## Leva nuova (2/11): `STRATA_MTP_DEVICE=N`, il drafter su una scheda senza stadi

Tre commit sopra `b6ba3fa6` (`d14aa61e`, `e9ed88ed`, `2b3bccf0`), tutti spenti di default: senza la variabile nessuna riga cambia.

- **Cosa fa.** `STRATA_MTP_DEVICE=1` con `CUDA_VISIBLE_DEVICES=0,1,2 --split-device 2 --layer-split 20 --pipeline-windows 2`: il drafter, le sue catene e il suo K/V stanno sulla 3060 (CUDA1), che non esegue nessuno stadio. La riga della finestra arriva una volta per catena (`stage_source_R`, come con `STRATA_MTP_STAGE0=1`), le righe del sottoinsieme della testa vengono copiate dalla 2080 Ti una volta al bind, l'embedding legge la tabella nativa in memoria host mappata (serve: file IQ, non `STRATA_EMBED_VRAM=1`). Vale alle condizioni di `STRATA_MTP_STAGE0` (`--pipeline-windows 2`, `draft_vocab.bin`, niente `--batch`), piu' niente `--prefill-head-alias`; `STRATA_MTP_STAGE0=1` vince se sono chiesti entrambi; una scheda non visibile o che esegue uno stadio viene rifiutata con una riga.
- **Come e' fatto.** `QsaState::dev` (`layer.hpp`): le copie dello snapshot K/V (`conversation_snapshot.cpp`: save/restore/restore_prefix/verify/copy/source) passano sulla scheda dello stato e non usano stadio ne' slot VRAM del parcheggio. `MtpDrafter::set_apart` (`mtp.hpp/.cpp`): tabelle rope proprie invece di quelle della sessione (che sta su un'altra scheda), `st_.dev = device_`, rilascio delle tabelle alla fine. In `generate.cpp` i punti dove conta solo «non sull'ultima scheda» (buffer delle righe, bind, gruppo dello snapshot, K/V del prompt in batch) usano `mtp_off_last = mtp_first || mtp_apart`.
- **Riga di avvio.** `strata generate: STRATA_MTP_DEVICE: the draft layer, its chain and its K/V on CUDA1, which runs no stage (... MiB there, ... GiB left free)`. A fine richiesta, con `STRATA_DECODE_TIMING`, la riga `strata pipeline: drafter on CUDA1: ... ms of host time per chain launch (... ms of it staging the source rows, N times so far)` dice quanto costa il passaggio delle righe sul riser x1.
- **Stato: compilato si' (`build-up029pila-<sha>` in `corse_2080ti\20261004_routed_keep`, sm_75+86, incrementale, 151 s), installato no, provato no.** Testo a cache fissa da verificare identico a spento prima di qualunque B-A-B.
- **Rischi.** (1) Il riser x1 della 3060 (~600 MB/s): ogni catena porta T righe da 32 KB dalla 2080 Ti via RAM (D2D senza P2P): 0,2-0,5 ms a catena sul percorso critico delle fresche, contro i ~2,1 ms di catena di oggi; lo dice la riga sopra. (2) La 3060 tiene lo schermo 8K e il compositore (GPU al 90%): la catena divide la scheda con il DWM, la varianza della latenza puo' mangiare il guadagno. (3) Parcheggio/ripristino delle conversazioni con il K/V del drafter su una terza scheda: percorso nuovo (`QsaState::dev`), non provato; da passare almeno un cambio di conversazione e un ripristino da disco. (4) Non toccato: la catena in batch (`--batch`, esclusa), `--prefill-head-alias` (escluso), HIP/Metal (il codice e' comune ma non compilato qui).
- **Nota su `STRATA_MTP_STAGE0=1` (preesistente, non corretta qui).** I percorsi del prefisso su disco e della conversazione su disco (`generate.cpp` ~14428, ~15443, ~17159: `k + 1 == n_st`) salvano e ripristinano il K/V del drafter con l'immagine dell'ULTIMO stadio, sotto la sua scheda, anche quando il drafter sta sulla prima (`STRATA_MTP_STAGE0=1`): il parcheggio in RAM invece lo mette con l'immagine della radice. Con l'anello del drafter (`kv_mode 2`) il ripristino lancia `kv_ring_restore` sulla scheda corrente con puntatori di un'altra scheda. Rimedio a una riga, se si vuole: `st_.dev = device_` anche per `mtp_first` (oggi `set_apart` lo fa solo per `STRATA_MTP_DEVICE`), cosi' le copie passano sulla scheda giusta qualunque immagine le porti.

## Esperti sulla 3060 con catena accesa: valutazione (2/11, non conviene ora)

Domanda: le cache ausiliarie (`--expert-cache-device1 N`, `RemoteExperts`) possono convivere con `--elastic`, `--pipeline-windows 2` e il nucleo RAM, per assorbire sulla 3060 gli esperti che a divisione alta finirebbero sulla CPU (a 24 strati la 3080 Ti ne scarica 5,7 ms a finestra)?

Cosa le spegne oggi, nel codice:
- `generate.cpp:3699-3703`: con una cache ausiliaria o `--peer-device` la cache elastica torna fissa («elastic cache does not support peer or remote expert tiers»).
- `generate.cpp:5212-5223`: `--pipeline-windows` si spegne con le cache ausiliarie («not with the helper caches»); `verify.cpp:4858`: le finestre in pipeline non catturano `--remote-expert-opt` (le righe ridotte entrano nel grafo: `verify.cpp:2331`, `2374`).
- `generate.cpp:9791`: il nucleo RAM non si arma con «remote experts are on»; `generate.cpp:9420-9427`: gli esperti di un'ausiliaria restano fuori dalla copia in RAM; sotto WDDM con ausiliarie si registrano solo 8 GiB dell'arena (`docs/SECOND_GPU.md`, `STRATA_ARENA_PIN_GIB`).
- `generate.cpp:11827-11831`: l'overlap adattivo e' escluso dalla divisione a strati e dalla pipeline comunque.

Come lavora un'ausiliaria (`remote_experts.cpp:265-392`, `expert_source.cpp:5058-5066`, `5270-5276`): per ogni strato con almeno un esperto suo, dentro la chiamata del pool (`drive_pool_multi_impl`, sul filo principale) `begin` copia l'ingresso in host pinned, lancia i kernel sulla scheda e la copia di ritorno, poi `finish` fa `cudaStreamSynchronize` **bloccante sul filo principale** e ricopia le righe nelle righe CPU mappate. La tabella e' statica (riempita dal profilo all'avvio; solo `--remote-expert-opt` la fa adattare, ed e' escluso dalla pipeline). Con la divisione a strati `split_drive_b = split_drive` condivide lo stesso `Drive` (`generate.cpp:10638`), quindi lo stato dell'ausiliaria non e' per parita': regge solo perche' `begin`/`finish` stanno nella stessa chiamata e il servizio degli strati e' seriale sul filo principale.

Perche' non conviene ora:
1. **Il guadagno ha un tetto basso e incerto.** Con l'elastica a 99,3% di hit l'ausiliaria servirebbe lo 0,2% delle richieste: il valore sta solo nel permettere la divisione a 24 (piu' strati sulla 3080 Ti), dove i 5,7 ms di CPU a finestra sono la coda fredda degli strati 0-23 (~24 x 512 x 1,5 MB = 18 GB meno i 7-8 GB sulla 3080 Ti: ~10 GB, piu' dei ~9 GB liberi della 3060) e il routing e' piatto (`PIPELINE2.md` 2b: il 50% degli esperti porta l'89% del traffico). Stima: -2/-3 ms sulle fresche (72%) se la 3060 tenesse meta' della coda, cioe' -1,5/-2 ms di media, PRIMA dei costi sotto.
2. **I costi stanno sul percorso critico della pipeline.** `finish` blocca il filo principale che serve gli strati di entrambe le finestre; il riser x1 (~600 MB/s) porta 32 KB di ingresso + 8 KB per riga di ritorno per strato con hit; la 3060 e' al 90% per il compositore 8K, quindi l'attesa e' a varianza alta. La tabella statica non si rinnova (un rinnovo costa 1,5 MB a 600 MB/s = 2,5 ms a esperto): la coda fredda che manca al profilo resta sulla CPU.
3. **Tre cancelli da togliere e rivalidare, non uno.** Pipeline (riga di politica, `5222`), elastica (`3699`: la residenza di CUDA0 cambia nel tempo e l'ausiliaria la esclude per chiamata via `primary_res`, `remote_experts.cpp:284`, quindi tecnicamente regge, ma non e' validato) e nucleo RAM (`9791`: budget di memoria pinned sotto WDDM da ripartire fra nucleo, arena e buffer dell'ausiliaria). Ognuno e' un B-A-B a se'; il drop a 30-80 t/s visto dal proprietario e' la somma di questi tre spenti, non il costo dell'ausiliaria in se'.
4. **Il punto 1 di questa stessa consegna gia' usa la 3060** (drafter, ~1 GB + K/V): esperti e drafter sulla stessa scheda si contendono VRAM, banda del riser e la scheda stessa.

Se si vuole comunque una misura a costo minimo: togliere il solo cancello della pipeline con una variabile (`STRATA_PIPELINE_HELPERS=1` su `generate.cpp:5222`, senza `--remote-expert-opt`), cache fissa e nucleo RAM spento come oggi con le ausiliarie, e leggere nel B-A-B quanto pesano le attese di `finish` (`ms_wait` nella riga per richiesta delle ausiliarie, `generate.cpp:19431-19436`). Se il peso e' sotto 1 ms a finestra, ha senso aprire il cancello dell'elastica; altrimenti la strada e' chiusa. Non fatto.
