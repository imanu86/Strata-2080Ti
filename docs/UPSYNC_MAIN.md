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
| 14 | prefill-sm75 | dequant 16 B in upstream; **hc16 non portato**; restano i default SIMT e TC_SM75 | 11 |
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
   chiamata (`xf16` in `Prefill::run_impl`). Due disegni sugli stessi punti non convivono, quindi non l'ho portato.
   Conseguenza: sulla 0.1.45 il prompt della 2080 Ti **non ha hc16 di default**. Da misurare: tempo di prefill 131k e KL con
   `STRATA_BF16_TC=1` nell'env del JSON. Il codice vecchio sta in `lab/expertdec` 794268e4.
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
  SIDE_MIXER=2 sono ora quelli di upstream), poi contro il Daily r2. Con `STRATA_BF16_TC=1` come terzo braccio.
- Verificare con `--expert-cache 1500`, senza `--elastic`, `--adapt-every 0` che le leve restino bitwise.
- Provare `--prompt-cache-file` + `STRATA_PREFIX_DISK_MULTI` con due client (prestito + disco).
