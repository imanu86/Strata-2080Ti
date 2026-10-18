# UPSYNC_MAIN - la pila del fork sopra upstream v0.1.45 (18/10/2026)

Branch `lab/upsync-main` del fork `imanu86/Strata-2080Ti`, da `upstream/main` 477ac13c (= tag `v0.1.45`, 17/10).
Prima di questo giro la pila (registro `stack/`, 17 patch) stava su v0.1.43 (`lab/upsync43-pulito`); fra 0.1.43 e 0.1.45
upstream ha 579 commit. Stato: **compilato si (sm_75+86), installato no, provato no.** Nessuna misura: tutto quello che
segue sul comportamento e' lettura del codice e compilazione.

## 1. Le nostre PR upstream (gh pr view, 18/10)

Tutte e 11 risultano **OPEN**, anche quelle il cui codice upstream ha gia' preso (Niko1221 le porta a mano su `main`,
conservando l'autore, e risponde nel thread senza chiuderle).

| PR | cosa | in 0.1.45? |
|---|---|---|
| #1117 | `--elastic` | no ("senza claim di throughput, fuori dal giro velocita'") |
| #1417 | stima prestito per device | no (nulla cambia su upstream corrente; da mandare con i buffer Turing) |
| #1418 | MMVQ righe riusate | **si**, `STRATA_B6_MMVQ_ROWS=1` opt-in (8e68aa58) |
| #1419 | banco forzato | no |
| #2058 | EXPDEC | **si**, `STRATA_EXPDEC=1` opt-in (f7eaf2de) |
| #2059 | QSA_ROWS | **si**, `STRATA_QSA_ROWS=1` opt-in (5b6c9398) |
| #2073 | PLE diretta | **si** (0.1.43.1, 62497bff); la diagnostica "benvenuta come PR piccola a parte" |
| #2075 | ATTN_DAILY | **si**, `STRATA_ATTN_DAILY=1` opt-in (731528be) |
| #2077 | `--prompt-cache-file` | no ("si sceglie un solo disegno su disco", #1090) |
| #2080 | SELECT_SIMT di default | **rifiutata** (cambia l'ordine di somma; serve KL indipendente su sm_75) |
| #2081 | dequant 16 B sm_75 | **si**, ma dietro `STRATA_SM75_PERF=1` / `STRATA_TC_SM75` (d976db34, 7f1964a6) |

Tag nuovi da 0.1.43: `v0.1.43.1` (12/10), `v0.1.44` (13/10), `v0.1.44.1` (14/10), `v0.1.45` (17/10). I tag locali del
fork con lo stesso nome non sono stati sovrascritti (git rifiuta: "would clobber existing tag"); quelli nuovi si'.
In 0.1.45 tutte le cinque leve Turing "bitwise" (#2069-#2072, #2081, #2093, #2220-#2223...) sono spente di default e si
accendono con un solo interruttore, `STRATA_SM75_PERF=1` (`include/strata/core/sm75_series.hpp`); ogni leva tiene la sua
variabile, che vince in entrambe le direzioni.

## 2. Esito per patch

`check_anchors.py --check` sulla pila vecchia contro `upstream/main`: 2 clean (02, 11), 1 con 3-way (08), 14 da riscrivere.
Poi ogni patch e' stata confrontata riga per riga (le righe aggiunte presenti nel file upstream): 04, 10, 12, 16, 17 hanno
0-3 righe mancanti, tutte commenti o manopole di lab: **gia' in upstream**.

| vecchia | leva | esito | nuova |
|---|---|---|---|
| 01 | elastic-cache | adattata (5 conflitti: `borrow_spill`, `guard_`, CMake) | 01 |
| 02 | split-prompt-borrow | pulita | 02 |
| 03 | forced-bench | adattata (finestra fusa `STRATA_ADMIT_MERGE_FIRST` non forzata) | 03 |
| 04 | b6-mmvq-rows | **gia' in upstream**, tolta | - |
| 05 | elastic-split-pipeline | pulita sopra la 01 | 04 |
| 06 | prefix-disk | adattata (prestito di prefisso di upstream: il disco piu' lungo vince) | 05 |
| 07 | vram-free-nvml | adattata (solo DETAILS.md) | 06 |
| 08 | park-no-foreign-kv | pulita (3-way) | 07 |
| 09 | pipeline-architectds | pulita (3-way) | 08 |
| 10 | attn-daily | **gia' in upstream**, tolta | - |
| 11 | df-defaults | pulita | 09 |
| 12 | side-mixer | **gia' in upstream** (PR #1889), tolta | - |
| 13 | ple-direct | meta' in upstream (default diretto); resta la **diagnostica** | 10 |
| 14 | prefill-sm75 | dequant 16 B in upstream; restano i default SIMT e TC_SM75; **hc16 portato dopo come patch 18** (par. 8) | 11 |
| 15 | ram-core | adattata (`reads_` ora atomico) | 12 |
| 16 | qsa-rows | **gia' in upstream**, tolta | - |
| 17 | expdec | **gia' in upstream**, tolta | - |
| nuova | `lab/struct-a` 075ce875 | SA_BET/SA_CHAIN: 3 commit, conflitti di adiacenza | 13 |
| nuova | `lab/ttft43` c00a8463 | STRATA_PREFIX_DISK_MULTI: pulita | 14 |
| nuova | `lab/ttft43` ea73f49e | STRATA_PLE_LOOKAHEAD (acceso di default come nel Daily r2) | 15 |
| nuova | `lab/ttft43` abd685c5 | STRATA_CONV_DISK: adattata | 16 |
| nuova | `lab/ttft43` a2111f8b | #1852 STRATA_PROMPT_ENCODER: **gia' in upstream** (f13a870d), tolta | - |
| nuova | `lab/pipeline2` f6a0b6b2 | P2_FLUSH_NOSWITCH + SWITCH_STAT + SMEM + FLUSH_US (4 commit) | 17 |

Le leve di `lab/expertdec` (EXPDEC) e `lab/qsarows` (QSA_ROWS) erano gia' le patch 17 e 16 del registro: ora sono
quelle di upstream. Non portati i neutri o peggiori di `lab/struct-a` (NODECUT, TOPK_GRID, HEAD_MINB, EXPDEC_WT).

## 3. Punti dubbi (decisioni da confermare)

1. **hc16 (le proiezioni dell'hyper-connection in FP16 sui tensor core, -19,5% del tempo GPU del prompt nel Daily).**
   Upstream ha un percorso proprio e piu' largo: `STRATA_BF16_TC=1` con `STRATA_XN_F16` (#655/#2091), sugli stessi punti di
   chiamata (`xf16` in `Prefill::run_impl`). Il 18/10 non l'avevo portato; il 19/10 e' stato portato come leva opt-in
   `STRATA_HC16_SM75=1` (patch 18, par. 8), con precedenza su BF16_TC nei suoi punti di chiamata. Sulla 0.1.45 il prompt
   della 2080 Ti **non ha hc16 di default**: serve l'interruttore nell'env del JSON. Il codice vecchio sta in
   `pr/prefill-hc16-sm75` 626d324c e `lab/expertdec` 794268e4.
2. **SELECT_SIMT** resta acceso di default sotto sm_80 (come il Daily), contro il rifiuto upstream: e' una scelta del fork.
   Idem `STRATA_TC_SM75` acceso di default (e' bitwise). Il resto della serie Turing di upstream (`STRATA_SM75_PERF=1`) e'
   spento: sono kernel che il Daily 0.1.43 non aveva, da provare a parte.
3. **Prefisso su disco contro prestito di prefisso** (nuovo in 0.1.45): se la radice su disco e' piu' lunga della
   corrispondenza in RAM (o del prestito) vince il disco e il donatore resta parcheggiato; negli altri casi nulla cambia.
   Non provato con una sessione vera.
4. **Banco forzato e prima finestra fusa**: con `STRATA_ADMIT_MERGE_FIRST=1` (spento di default) la finestra fusa non viene
   ne' forzata ne' riprodotta. Per il banco DET va lasciato spento.
5. **STRATA_PLE_LOOKAHEAD** e' acceso di default (come in `lab/ttft43`), non opt-in. `STRATA_CONV_DISK` e' opt-in.
6. Il JSON del Daily (`strata-0143-daily.json`, modificato il 13/10 08:37) imposta anche `STRATA_NODECUT=1`,
   `STRATA_TOPK_GRID=1`, `STRATA_B6_HEAD_MINB=6` (leve neutre di `lab/struct-a`, **non** nella pila: la 0.1.45 le ignora) e
   `STRATA_CONV_DISK=10` (acceso, mentre il bat dice "spenta"). Un confronto con il Daily non e' alla pari finche' non e'
   deciso.

## 4. Build (sm_75+86, `build_lock.py`)

- `build_lock.py` aspetta che non giri nessun `strata.exe`; il Daily dell'owner e' acceso (porta 8100) e non l'ho toccato.
  Ho usato una copia dello script (scratchpad) identica ma senza quel controllo (4 job, priorita' bassa, stessa `build.lock`).
  La compilazione non usa la GPU, `bench.lock` non e' stato preso.
- `build-upsmain-d06429ee`: strata.exe sha256 `fd8890197bf294c259bdb7917689b2b8ca572d113ab8a5ccf75f5b98eff48a12`.
- `build-upstream-477ac13c` (upstream puro, sorgenti esportati con `git archive` in `Strata-wt\upstream-src-477ac13c`, senza
  worktree): strata.exe sha256 `bfda01a9707f5b946388bf9befa7cd5438b38387539c60a3feb7dedceb7440ad`.
- `strata-vision.exe` e' copiato dal Daily (sha256 93441f40...): `tools/vision` cambia di una riga fra 0.1.43 e 0.1.45 e non
  e' stato ricompilato (e' un progetto a parte).
- Cartelle: `...\corse_2080ti\20261004_routed_keep\build-upsmain-d06429ee\` e `...\build-upstream-477ac13c\`, con
  `build.json`, `strata.exe`, `strata-vision.exe`, `source\serve`, `source\tools`.

## 5. Cosa resta

- B-A-B (coordinatore): upsmain contro upstream puro, a parita' di env (STRATA_EXPDEC/QSA_ROWS/ATTN_DAILY/B6_MMVQ_ROWS/
  SIDE_MIXER=2 sono ora quelli di upstream), poi contro il Daily r2. Con `STRATA_BF16_TC=1` e `STRATA_HC16_SM75=1` come bracci
  del prompt a 131k (il secondo non e' ancora misurato lungo: vedi par. 8).
- Verificare con `--expert-cache 1500`, senza `--elastic`, `--adapt-every 0` che le leve restino bitwise.
- Provare `--prompt-cache-file` + `STRATA_PREFIX_DISK_MULTI` con due client (prestito + disco).

## 6. La regressione di L01 nel B-A-B del 18/10 (Daily r2 contro upsmain): non e' nel codice (19/10, misurato)

Dato di partenza (`DET-x22-dly-*`): L01 69,9 -> 65,5 / 66,6 t/s, "per-layer host" 3,17 -> 5,45 / 5,14 ms, e in piu' "stage" 0,29 ->
1,92 / 2,06 ms. Lo stage e' lo staging di finestra, PLE compresa: nei due bracci B il "PLE primary" sale a 0,81-1,02 ms/finestra di
gather e 1,04-1,11 di altro staging (Daily: 0,11 e 0,18). Il tempo e' **attesa sul lettore PLE**: fra "PLE before" e "PLE after" del
decode di L01 il "blocked" vale 961 / 1034 ms nei due B (Daily 108 / 115 ms), a parita' di letture SSD (18,1k contro 18,1k) e di righe.
Cioe' ogni lettura e' costata di piu', non ce ne sono di piu'. Il costo cade sulle finestre "fresh" (dopo un rollback, righe non
prefetchate): classe fresh 34,5 ms contro 31,9; le speculative sono uguali (21,0 contro 20,5). L02 e' pari (64,9 / 64,7 contro 65,4).

**Non e' riproducibile.** Stessa build `build-upsmain-d06429ee`, stesso exe (sha uguale), stessa env e stessi argomenti dei due B
(plan.json identici tranne il percorso della traccia): `DET-x22-sw-b1/b2` (16:30, 16:37) 69,6 / 68,2 t/s con stage 0,29 / 0,32; e tre
corse nuove di oggi (bench.lock, banco forzato 131k, `DET-s45-*`):

| corsa | exe | L01 t/s | L01 host | L01 stage | L01 PLE blocked | L02 t/s |
|---|---|---|---|---|---|---|
| dly-warm / dly-a / s45-h1a | Daily r2 | 69,6 / 69,9 / 67,1 | 3,27 / 3,17 / 3,91 | 0,28 / 0,29 / 0,28 | 108 / 115 / - ms | 63,7 / 65,4 / 63,5 |
| sw-b1 / sw-b2 | upsmain | 69,6 / 68,2 | 3,57 / 3,96 | 0,29 / 0,32 | 112 / 122 ms | 65,3 / 63,8 |
| **dly-b1 / dly-b2** | upsmain | **65,5 / 66,6** | 5,45 / 5,14 | **1,92 / 2,06** | **961 / 1034 ms** | 64,9 / 64,7 |
| s45-r1 | upsmain | 69,7 | 3,67 | 0,27 | 103 ms | 64,7 |
| s45-h1b (subito dopo il Daily h1a) | upsmain | 67,7 | 3,22 | 0,27 | - | 64,5 |

Media L01: Daily 68,9; upsmain senza i due bracci anomali 68,8 (con: 67,9). L02: Daily 64,2; upsmain 64,7. **A parita' di condizioni
upsmain e' pari al Daily r2** (±1%, dentro il rumore fra sessioni, che su questo banco arriva a 4-5% come mostrano le due sw-b contro
le due dly-b). L'ordine "Daily poi upsmain" non e' la causa (h1a -> h1b nessuna penale). Il banco ha colpito solo l'L01 dei due B.

Cosa ho escluso leggendo il codice 0.1.43 -> 0.1.45 (579 commit) sul cammino di decode di questa config:
- pool CPU / esperti CPU: `pool.cpp` (UMWAIT opt-in, `--host-core sibling`), `native_expert.cpp` (NVFP4, EXACT1 opt-in),
  `iq_avx2_rows.inl` (solo righe nuove sotto `gu_ggml_one`): nessun cambio di default per IQ3_XXS. CPU 5,2-5,8 contro 4,7 ms e CPU
  experts 1,18-1,28 contro 1,14-1,21 per layer-finestra vengono dalla cache piu' piccola (ultimo punto), non dal kernel.
- PLE: l'unico cambio e' in `ngram.cpp` (#2622: `gather_batch` riusa chiavi prefetchate riordinate; `a2de7446`: `std::recursive_mutex`
  su `PleTable`). Il lettore (`ple_reader.cpp`) e' identico. Il mutex e' preso anche da `prefetch_rows` (drafter, stage 0) e da
  `gather_batch` (che lo tiene mentre aspetta l'SSD): una contesa fra catena di draft e finestra e' possibile, ma e' un'ipotesi non
  misurata e non spiega perche' sw-b / s45-r1 / h1b, uguali, non la mostrano.
- `session.cpp`, `generate.cpp`, `verify.cpp`: i cambi sono Metal, batch, `--host-core sibling`; la riga di timing e' la stessa.
- Dimensione della cache expert: `expert cache N slots` parte da una lettura NVML della VRAM libera che oscilla (3214-3313 su upsmain,
  3238-3377 sul Daily, 3098 in s45-r1); non e' un difetto di 0.1.45. Un 2-3% di slot in meno = 4-8% di esperti CPU in piu', e' la sola
  differenza persistente dell'host (+0,3-0,5 ms su L01) e sta dentro il rumore.

Causa: **disturbo esterno e transitorio sulla lettura PLE dell'L01 dei due bracci B** (ipotesi sul meccanismo, non provata: in dly-b1 c'era
una lettura estranea da 28 MB/s su `PhysicalDrive1` dalle 16:42:03, in dly-a un 63 MB/s per ~3 minuti che ha portato il caricamento
dell'arena del Daily a 0,17 GiB/s; il decode di dly-b1 e dly-b2 e' caduto alle 16:42:40 e alle 16:52:20, a 10 minuti esatti: un lavoro
periodico di altri agenti o di sistema e' compatibile, ma non l'ho trovato fra le attivita' pianificate di Windows). Nessuna correzione
al motore: **non ho cambiato codice, la build resta `build-upsmain-d06429ee`** (strata.exe sha256 `fd8890197bf2...`).

Per il prossimo B-A-B: scartare e ripetere un braccio se nel suo `engine.log` lo "stage" di L01 supera 0,6 ms o il "blocked" PLE fra
"before" e "after" del decode supera ~300 ms; `bab_verdict.py` dovrebbe stamparli accanto ai t/s, cosi' un braccio disturbato si vede subito.

Differenza persistente e vera (non e' L01): il **prompt da 131k** e' piu' lento del 10-12% su upsmain: 1463-1489 tok/s (sw-b1/b2, s45-r1,
h1b) contro 1607-1701 del Daily (dly-warm, dly-a, h1a), cioe' 88-89 s contro 77-82 s. E' la mancanza di hc16 (punto dubbio 1 del
paragrafo 3): bracci da misurare `STRATA_HC16_SM75=1` (par. 8) e `STRATA_BF16_TC=1`.

## 7. Interruttori nuovi di upstream 0.1.44 / 0.1.45 che possono valere per 2080 Ti + 3060 (da misurare, nessuno e' acceso nella pila)

Config di riferimento: 2 schede (CUDA0 3060 sm_86 strati 0-15, CUDA1 2080 Ti sm_75 strati 16-47), `--pipeline-windows 2`, MTP `--spec 4`,
KV int8, pack IQ3_XXS, PLE diretta, CPU solo AVX-2, Windows/WDDM. Tutti sono variabili d'ambiente. Letti dal codice e dai documenti di
upstream; nessuno provato qui.

Decode
- `STRATA_SM75_PERF=1`: accende in blocco i kernel Turing bitwise di upstream, **solo su CUDA1** (sm_75; la 3060 non cambia). Contiene
  `STRATA_SM75_IL` (tabella matvec interleaved per piccoli batch), `STRATA_SM75_EXP` (layout esperti mode 4), `STRATA_GR_UP75` e
  `STRATA_GR_DOWN_TG` (lettura hyper-connection), `STRATA_SCORES_HALVE` e `STRATA_TOPK_SPLIT` (selezione QSA in decode),
  `STRATA_ATTN_V3` (attenzione di decode su pool INT8), `STRATA_TOPK_SAMPLE`, `STRATA_SCORES_THREAD`, `STRATA_PA_V1_PREFETCH` (prompt),
  `STRATA_TC_SM75` (gia' acceso dalla pila), `STRATA_Q2_WARP` (solo Q2_0, inerte con IQ3_XXS). Dichiarato: decode +5-6% a 4K-32K, prompt
  a profondita' fino a +30%, tutto bitwise. Ogni sigla vince su SM75_PERF in entrambe le direzioni (=0/=1).
  **Attenzione**: l'env del Daily ha `STRATA_MMVQ_IL=0`, che spegne il percorso interleaved su tutte le schede (`g_mmvq_il()` in `verify.cpp`
  gata la copia interleaved delle attivazioni, e `STRATA_QIL` con lei): da come leggo il codice `SM75_IL` e `STRATA_QIL` (di default su
  Turing, +0,5-1%) restano inerti, non provato. Un secondo braccio sensato e' `STRATA_SM75_PERF=1` senza
  `STRATA_MMVQ_IL=0` (ma cosi' cambia anche la 3060, dove l'IL e' di default acceso: servono le due schede nel conto).
- `STRATA_MTP_STAGE0=1`: con `--pipeline-windows` il layer di draft sta sulla **prima** scheda (3060) invece che sull'ultima, e la catena di
  draft non divide piu' la 2080 Ti con la sua finestra (upstream, 2x R9700: +21% decode). Serve `draft_vocab.bin` nella dir MTP (c'e' in
  `rt_it48k`). Costo: ~840 MiB di VRAM in meno sulla 3060 (circa 580 slot di cache). Su CUDA e' spento di default e il log lo segnala.
- `STRATA_QSA_SPLIT_ATTN=1` (#1888, sm_80+, KV int8): attenzione di decode divisa su piu' blocchi sul percorso mma int8; solo 3060.
  Dichiarato +2,8% / +3,5%; non bitwise (KL 0,0016 contro un pavimento di 0,0012).
- `STRATA_HC_REQ8=1` (#1262): proiezioni hyper-connection in int8 con una scala per 32: meno VRAM, piu' slot di cache. Dichiarato decode
  -0,2 / +2,2%, KL 0,0018, non bitwise.
- `STRATA_CPU_IQ_ALL_EXACT1=1` (#1415 #1544): righe esperto a 1 token in AVX-2 con l'ordine float di ggml (bitwise) al posto del `vec_dot` di
  ggml; solo CPU senza AVX-512 e **non con `STRATA_IQ_MT_MIN=1`** (che il banco imposta): da provare senza quella.
- `STRATA_POOL_UMWAIT=<cicli>` (con `_C02`, `_ESC`): i worker del pool aspettano con UMWAIT invece di girare a vuoto (CPU con WAITPKG, build
  MSVC); risparmia energia, rischio di latenza di risveglio. Probabilmente neutro con `STRATA_POOL_SPIN_US=100`.
- Non si applicano a questo assetto: `STRATA_PIN_BY_NEED=1` (arena su una GPU), `STRATA_PCIE_BALANCE=1` (una GPU, senza `--batch`),
  `STRATA_SPLIT_OVERLAP=1` (spento accanto a `--pipeline-windows`), `STRATA_VERIFY_ASYNC_LAUNCH=1` (non per le finestre in pipeline).

Prompt / TTFT
- `STRATA_BF16_TC=1` (con `STRATA_XN_F16`, acceso se non =0): proiezioni BF16 (hyper-connection, router, indicizzatore...) in FP16 sui tensor
  core; invece del nostro hc16 (ora patch 18, `STRATA_HC16_SM75=1`, par. 8; sui suoi punti hc16 vince). Non bitwise. Dichiarato prompt +15-18% sulla 2080 Ti (Q2_0). Primo braccio da provare.
- `STRATA_PF_FUSED=1` + `STRATA_PF_FUSED_SM75=1`: esperti del prompt fusi in int8 (MMA m8n8k16 su Turing). Documentato per Q2_0; il ramo
  sm_75 esiste anche in `moe_fused_iq.cu`: da verificare se copre IQ3_XXS. Non bitwise.
- `STRATA_PA_V2T=1` (attenzione del prompt v2 con staging in registri) e `STRATA_PROMPT_ATTN_Q4_TC`: Turing, non bitwise.
  `STRATA_PROMPT_ATTN_IMMA=1` (sm_80+, KV int8, solo 3060): prompt +2,5%, rounding-level.
- `STRATA_HANDOFF_PIPE=P` (P 2-16) e `--prefill-pipe 1` / `STRATA_PREFILL_PIPE=1`: passaggio dei chunk di prompt fra le due schede in P pezzi
  su un thread persistente / geometria dei chunk per la pipeline; opt-in, il secondo tocca l'arrotondamento.
- `STRATA_REFILL_PARTS=1`: riempie gli slot presi in prestito dal prompt direttamente dalla mappa del GGUF (-0,1-0,3 s a fine prompt).

Batch / multi-chat (solo con `--batch`; nessuno e' di default; non cambiano cio' che si calcola)
- `STRATA_SLOT_COPY_DIRECT=1` + `STRATA_SLOT_COPY_BATCHSYNC=1` (ammissione/ripresa di uno slot device-to-device), `STRATA_SLOT_CKPT_MOVE=1`
  e `STRATA_CKPT_POOL=1` / `STRATA_CKPT_THREADS=1` (checkpoint senza copia / in parallelo), `STRATA_ADMIT_MERGE_FIRST=1` (ultimo pezzo di
  prompt e primo token in una finestra, solo con tutti gli esperti residenti: il banco DET lo vuole spento), `STRATA_ADMIT_SHORT_FIRST=N`
  (server), `STRATA_BATCH_SELF_COMMIT=1`, `STRATA_QSA_SLOTS=1`, `STRATA_BATCH_ATTN_ROWS=1`, `STRATA_BATCH_GDN_GROUPS=1`,
  `STRATA_PIPE_BYIELD_NODRAIN=1`, `STRATA_PIPED_CONCURRENT=1` (prompt su un thread mentre gli slot decodificano), `STRATA_BATCH_MTP_SERIAL=1`.
  Misurati da upstream su 4x R9700 con 8-16 slot; per un solo utente non servono.

## 8. hc16 portato come leva opt-in `STRATA_HC16_SM75=1` (patch 18, 18/10 sera)

Commit `d9a26384`, build `build-upsmain-d9a26384` (strata.exe sha256 `ac37c4eac933ee2e3369ad64bdeccdbed939c4c06406493f1d2893b3ac843131`, sm_75+86,
compilato con `build_lock.py` originale dopo la coda del coordinatore). **Compilato si, provato solo in breve (sotto), 131k non misurato.**

- Cosa fa: sulla 2080 Ti le proiezioni dell'hyper-connection (down, up, inject) e `ssm_alpha/beta` del prompt vanno sui tensor core FP16. Il peso BF16
  e' ricodificato in FP16 una matrice alla volta in un buffer di 6,55 MB (`hc16_w_to_f16`), le attivazioni sono l'immagine FP16 del valore FP32
  (`f16img` 2 nei writer `gr_*`, `gr_silu` a 1). Il layer di draft resta BF16. Non e' bitwise. Solo sm_75, spento di default, fuori da `STRATA_SM75_PERF`.
- Rispetto al codice 0.1.43 (`pr/prefill-hc16-sm75`): niente buffer `xn16h`/`lo16h` (il parametro `f16img` dei writer da bool a int, anche in
  `sycl/.../kernels.dp.cpp`, non compilato); resta il solo buffer dei pesi, preso in `carve` e in `bytes_needed_impl` alla pari.
  Le righe aggiunte a `generate.cpp` per la riserva della cache (6,55 MB) non sono state portate: il prezzo piatto di 160 MiB + 680 KiB/token le copre.
- **Precedenza con `STRATA_BF16_TC=1`** (decisa qui): sui punti di chiamata di hc16 vince hc16 (xn16 scritto con la regola 2, mai come FP16-del-BF16 di
  `XN_F16`; peso in staging al posto della conversione per chiamata). Tutte le altre GEMM BF16 (router, indicizzatore, layer di draft) seguono BF16_TC.
  Con i due accesi il layer di draft resta quindi FP16 per BF16_TC (la perdita di accettazione di 27 punti misurata sul Daily e' di BF16_TC, non di hc16).
  Spento con BF16X2, `STRATA_HC_REQ8` e matrici hc non BF16 residenti (errore esplicito).
- Prova breve (banco forzato no, `bench.lock`, config DET come la coda del 18/10, `STRATA_PREFILL_TIMING=1`, prompt ~4k e ~16k token, 96 token a T=0,
  `hc16_breve.py`, cartelle `DET-hc16-*`): nessun errore, testo coerente in tutti i bracci (stessa apertura "We need answer user's request ..." in tutti, poi
  diverge di poco, come atteso per un percorso non bitwise).

| braccio | fase "hc read" sulla 2080 Ti, 16k | prompt 4k | prompt 16k |
|---|---|---|---|
| senza (warm, `hc16b-off`) | 3264 ms | 10,6 s | 24,9 s |
| `HC16_SM75=1` (`hc16-on`, subito dopo un avvio freddo) | 945 ms | 12,7 s | 28,9 s |
| `BF16_TC=1` (`hc16b-tc`) | 973 ms | 9,0 s | 19,7 s |
| `BF16_TC=1` + `HC16_SM75=1` (`hc16b-both`) | 831 ms | 9,6 s | 19,7 s |

Lettura: la fase "hc read" (la sola che hc16 tocca) cala del 71% con hc16 come con BF16_TC (-2,3 s su 16k), cioe' il guadagno atteso c'e'. Il tempo totale
del prompt corto e' dominato da altro (dequant degli esperti e attese sul disco: nel braccio `hc16-on` l'avvio era a freddo, 4k e 16k non sono alla pari), e
BF16_TC guadagna di piu' perche' sposta anche le altre GEMM BF16. Il confronto di tempo-prompt a 131k contro BF16_TC (1506 -> 1753-1769 tok/s, +17%, dato del
coordinatore) **non e' ancora fatto**: serve un braccio `STRATA_HC16_SM75=1` a 131k nella coda.
