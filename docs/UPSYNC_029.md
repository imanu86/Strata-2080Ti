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
