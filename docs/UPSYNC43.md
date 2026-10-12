# UPSYNC43: la pila del Daily su upstream 0.1.43

Branch `lab/upsync43` del fork `imanu86/Strata-2080Ti`, worktree `Strata-wt\upsync43`. Base `upstream/main` =
`7695859a` = tag **v0.1.43** (11/10/2026). Sopra, leva per leva, la pila di `lab/candidate2` `ac6065dc` («Strata 0.1.42
Daily r2»: upstream v0.1.42 `61b3fb5d` + 70 commit). Diario di lavoro in italiano; il **registro della pila** leggibile
anche da altri utenti del motore è `stack/` (inglese): `INDEX.md`, `BLOCKS.md`, una `.patch` per leva, `manifest.json`,
`check_anchors.py`.

## 1. Cosa cambia da 0.1.42 a 0.1.43 (72 commit upstream) che ci riguarda

| Upstream 0.1.43 | Effetto su di noi |
|---|---|
| `STRATA_ADAPT_OVERLAP` (copie del tier adattivo mentre gira la finestra dopo): acceso su Linux, **opt-in su Windows** (`7c035d72`, −2,6% su una 5070) | nessuno di default; riordina gli scambi in `generate.cpp` (conflitto di adiacenza col nucleo RAM, risolto) |
| riparto PCIe misurato per il pack Q2_0 canonico: opt-in su Windows (`STRATA_PCIE_AUTO_CANON=1`) | nessuno (IQ3_XXS, layer split) |
| tail skip spento da solo con ≥ 90% di esperti in cache (#1884, `STRATA_TAIL_SKIP_MAX_CACHED`) | la config tiene comunque `STRATA_ROUTE_TAIL_SKIP=0` |
| guardia NaN del serve accesa (#879/#1892, `STRATA_NAN_GUARD=0` la spegne; bit per bit, < 0,2%) | nuova, lasciata accesa |
| `#1712`: step record e id dei token del chunk di prefill da un buffer pinned (`tok_up`) | conflitto con l'elastica in `prefill.cpp` (risolto: `tok_up` dimensionato su `cap_T`) |
| sm_86: Q3_K sul percorso IL e una colonna (`STRATA_MMVQ_IL_Q3K/T1`), `mmvq_il_parity` con le forme vere; sm_60: tabella Pascal | il dispatch IL è riscritto: **#1437 (tabella sm_75) non si applica più** |
| gfx12/HIP: `STRATA_QFUSE_HC`, `STRATA_FUSE_HITS`, `STRATA_FUSE_CONVAB`, pipeline a 2 finestre di default sui split 3+ stadi | solo adiacenze in `verify.cpp` (SIDE_MIXER, CP_RESPLAN), risolte |
| `#1737` route-resident one warp per token (opt-in), `#1892`, SYCL/B70, benchmark `BENCH.bat`, NUMA, GGUF unmap | nessuno |

Delle nostre cose **nulla** è entrato in 0.1.43: `--elastic`, `--prompt-cache-file`, `STRATA_FORCE_*`, `STRATA_VRAM_FREE`,
`STRATA_B6_MMVQ_ROWS`, `STRATA_SIDE_MIXER`, `STRATA_ATTN_DAILY`, `STRATA_RAM_CORE`, `STRATA_PIPELINE_ELASTIC`,
`STRATA_TC_SM75`, `STRATA_PLE_AUTO_MMAP`, `STRATA_EXPDEC`, `STRATA_QSA_ROWS` non esistono su 0.1.43 puro. Le PR #1117,
#1417, #1418, #1419 (nostre), #1911, #1503, #1889, #1437 (di altri) sono tutte ancora **aperte** (controllato il 12/10).

## 2. Inventario della pila (70 commit di `lab/candidate2` → 19 leve + 1 commit di soli doc)

| # | Leva (`stack/`) | Interruttore (default) | Origine | Upstream | Su 0.1.43 | Valore misurato |
|---|---|---|---|---|---|---|
| 01 | elastic-cache | `--elastic` (off) | nostra, PR #1117 | aperta | adattata (`tok_up`/`cap_T`) | default del Daily dal 3/10 |
| 02 | split-prompt-borrow | sempre con `--layer-split` | nostra, PR #1417 | aperta | pulita | correttezza |
| 03 | forced-bench | `STRATA_FORCE_IDS/WINDOWS/TRACE` (off) | nostra, PR #1419 | aperta | pulita | il banco di tutti i B-A-B |
| 04 | b6-mmvq-rows | `STRATA_B6_MMVQ_ROWS=1` (off) | nostra, PR #1418 | aperta (upstream fa tabelle IL) | adiacenza CMake | kernel −17/−33%; IL upstream −3,0%, tabella sm_75 0,0% contro B6 |
| 05 | elastic-split-pipeline | `STRATA_PIPELINE_ELASTIC` (on con 2 schede) | nostra | mai proposta | pulita | config del Daily (83,7 t/s reali 7/10) |
| 06 | prefix-disk | `--prompt-cache-file` (off) | nostra | mai proposta | pulita | TTFT Claude Code ~42 → 16-20 s |
| 07 | vram-free-nvml | `STRATA_VRAM_FREE=device` (off) | PR #1911 (Xiaomi2008) + collante | aperta | pulita | toglie i crolli 7-36 t/s sotto WDDM |
| 08 | park-no-foreign-kv | sempre | PR #1503 (sanastasiou) | aperta | pulita | correttezza |
| 09 | pipeline-architectds | `STRATA_PIPELINE_ELASTIC` (on) | nostra | mai proposta | pulita | senza: −4,6/−7,3/−7,8%; con (+11): +0,7% = pari |
| 10 | attn-daily | `STRATA_ATTN_DAILY=1` (off) | nostra (corpo #540, merge Hardin22) | chunk pre75 solo sm_60; merge v2 no | pulita | `attn_chunk_kernel` 99,8 → 46,4 µs (3060) |
| 11 | df-defaults | `STRATA_DF_BRANCH/PLE` on (upstream opt-in) | nostra (default di #904) | #904 chiuso/merged opt-in | pulita | +2,4% / +3,3% nel Daily |
| 12 | critpath-levers | `STRATA_CP_*` (off) | nostra | mai proposta | adattata (`hits_in_combine`) | rumore (−0,4/−2,5/+0,1/−0,1%) |
| 13 | chain-fuse | `STRATA_CHAIN_FUSE2` (off) | nostra | mai proposta | pulita | +0,4% rumore |
| 14 | side-mixer | `STRATA_SIDE_MIXER=1/2` (off) | PR #1889 (KarlGrier) | aperta | adattata (senza #1437, `FUSE_CONVAB`) | `=2` +0,9% |
| 15 | ple-direct | diretto; `STRATA_PLE_AUTO_MMAP=1` per mmap | nostra | mai proposta | pulita | prefill 131k 260 → 75 s, decode +3,9% |
| 16 | prefill-sm75 | `STRATA_SELECT_SIMT=0`, `STRATA_TC_SM75=0` spengono | nostra (SIMT da #1091) | #1091 merged opt-in; il resto mai | pulita | prefill 100 → 75 s (con 15) |
| 17 | ram-core | `STRATA_RAM_CORE=1`, `_SLOTS_DEV1=5500` (off) | nostra | mai proposta | adiacenza (`ADAPT_OVERLAP`) | −10,35 GiB, +0,1% |
| 18 | qsa-rows | `STRATA_QSA_ROWS=1` (off) | nostra | branch `pr/qsa-rows-scores` pronto | pulita | kernel 101 → 55 µs; +3,2/+0,1% |
| 19 | expdec | `STRATA_EXPDEC=1` (off) | nostra | branch `pr/verify-expdec` pronto | pulita | gate/up −11/−23%; +1,1% |
| — | docs UPSYNC.md/CANDIDATE.md | — | — | — | pulita | solo documentazione, fuori da `stack/` |

Già di upstream (niente da portare): `--layer-split`, `--trim-stage-weights`, `--pipeline-windows 2`,
`--vram-reserve-later-mib`, `--host-core`, `--prompt-cache-tail/-every`, `STRATA_MTP_SHARED_BRANCH` (esiste in `mtp.cpp`
di upstream, `=0` nella config), `fit_max_tokens` (è del server Python `serve/`, identico fra fork e upstream),
`strata-vision.exe` (`tools/vision` identico fra v0.1.42 e v0.1.43: l'exe `93441f40…` di `build-upsync-133e3a69` vale
tale e quale).

**Lasciata fuori**: #1437 (`2f84fa7f`, `4ad73342`, `b0fbff3f`, tabella MMVQ_IL per forma su sm_75): 0.1.43 ha
riscritto `native_mmvq_il_rows_for`/`native_mmvq_il_supported` (una colonna, Q3_K, tabella sm_86, `STRATA_MMVQ_IL_T1/Q3K`)
e #1437 cambia la stessa firma (`n_in`); per noi vale 0,0% (il Daily gira `STRATA_MMVQ_IL=0` → B6) ed è la PR aperta di
un altro autore, che la ribaserà lui. Con la config del Daily il comportamento è identico. SIDE_MIXER, che nel fork
usava la firma di #1437, qui usa quella di 0.1.43 (interleave solo con `n >= 2`, come la lambda `mm`).

Non erano in `lab/candidate2` e restano sui loro branch: `lab/hostgap` (`STRATA_HG_PRELAUNCH`, neutro, chiuso),
`lab/ttft` (`STRATA_PREFIX_DISK_MULTI`, `STRATA_PLE_LOOKAHEAD`, non provato).

## 3. Conflitti risolti (5 su 19 leve)

| Leva | File | Conflitto | Soluzione |
|---|---|---|---|
| 01 | `src/prefill/prefill.cpp` | #1712 aggiunge `m.tok_up.init(chunk)` dove l'elastica dimensiona `tok_host` su `cap_T` | tutte e due su `cap_T` (il buffer pinned copre il chunk massimo come quello host) |
| 04 | `CMakeLists.txt` | `router_chain_bench` (0.1.43) e `b6_mmvq_bench` nello stesso punto | tenuti entrambi |
| 12 | `src/core/verify.cpp` | `bool hits_in_combine = false;` (STRATA_FUSE_HITS, gfx12) prima di `if (ar_on())`, dove CP_RESPLAN inserisce il suo ramo | dichiarazione prima, poi `if (cpl) {...} else if (ar_on())`; il ramo chiama `moe_hit_add` da sé |
| 14 | `src/core/verify.cpp` | `g_qfuse_hc` accanto a `g_side_mixer`; `fuse_ab` (STRATA_FUSE_CONVAB, HIP) e `side_proj` sulla stessa riga; `if (!fuse_ab)` contro `if (!sidem)`; chiamate IL con `n_in` | entrambe le funzioni; `fuse_ab` anche `&& !sidem`; `if (!fuse_ab && !sidem)`; firma a 3 argomenti e interleave solo con `n >= 2` |
| 17 | `src/program/generate.cpp` | riordino degli scambi di `STRATA_ADAPT_OVERLAP` e `rc_swaps.fetch_add` adiacenti | tenuti entrambi, nell'ordine upstream → nostro |

Le altre 14 leve si sono applicate pulite (`git cherry-pick -X patience`). Ogni leva è un commit:
`git log upstream/main..lab/upsync43`. Il codice è stato controllato uguale prima e dopo l'autosquash che ha tolto i
percorsi del lab dai doc (`git diff 4bc562e7 HEAD -- . ':!docs'` vuoto).

## 4. Il registro `stack/`

- `stack/NN-nome.patch`: `git format-patch` di ogni leva (19), applicabili in ordine con `git am stack/*.patch`.
- `stack/manifest.json`: metadati scritti a mano (scopo, interruttore, origine, stato upstream, valore, note) + le
  **ancore** calcolate dalle patch: per ogni hunk, file, simbolo (quello che git scrive dopo `@@`) e 2-3 righe
  dell'originale **non patchato**, uniche nel file (allargate fino a 6 righe se servono). 346 punti di innesto, 343
  ancore uniche, 3 ambigue (segnate), 0 senza ancora.
- `stack/INDEX.md` (corto, da leggere prima di ogni upsync) e `stack/BLOCKS.md` (i blocchi innestati, ancora in
  testa) sono **generati** da `check_anchors.py --build`; `--verify-docs` controlla che siano allineati.
- `check_anchors.py --check <repo> --rev <tag>`: worktree temporaneo del tag, per ogni ancora presente / ambigua /
  spostata / sparita, per ogni patch pulita / 3-way / da riscrivere (in sequenza: una patch da riscrivere viene saltata
  e le seguenti provate senza).

Esito del 12/10:

| Tag | Pulite | 3-way | Da riscrivere | Ancore sparite |
|---|---|---|---|---|
| v0.1.43 (la base) | 19 | 0 | 0 | 0 (3 ambigue) |
| v0.1.42 (controprova all'indietro) | 7 (02, 03, 10, 11, 15, 16, 19) | 1 (08) | 11 (01 `prefill.cpp`; 04, 06, 18 `CMakeLists.txt`; 05, 07, 09, 17 `generate.cpp`; 12 `verify.cpp`+`generate.cpp`; 13 `mtp.cpp`; 14 `verify.cpp`) | 27 su 346 |

Sul v0.1.42 i «da riscrivere» sono gli stessi file dei conflitti veri del port (più i punti dove 0.1.43 ha aggiunto
righe che le patch ora danno per scontate: `tok_up`, `hits_in_combine`, `g_qfuse_hc`, `aov`): lo strumento misura quel
che serve.

Nel registro, per ogni leva, è annotato quale default upstream va forzato per riprodurre il nostro comportamento
(tabella «Upstream defaults this fork forces» in `INDEX.md`): `SELECT_SIMT` (patch 16, acceso), `DF_BRANCH/DF_PLE`
(patch 11, accesi), PLE diretta (patch 15), `NAN_GUARD` (lasciato acceso), `ROUTE_TAIL_SKIP=0` e `MMVQ_IL=0 +
B6_MMVQ_ROWS=1` (dalla config), `PIPELINE_ELASTIC` (patch 09).

## 5. Build

`build_lock.py` (Release, CUDA 75;86, test spenti, llama.cpp del Daily), rc 0, 312 s, 140 passi, solo warning C4996 e
`#221-D` di sempre. `strata.exe` sha256 **`b9e597672a37a710e7f633dad3dd0a73862efe4c266637fd24bf5b0b651a177a`**, copiata
con `build.json` (status `completed`), `strata-vision.exe` (`93441f40…`, da `build-candidate2-ac6065dc`), `source\serve`
e `source\tools` del worktree in `corse_2080ti\20261004_routed_keep\build-upsync43-9676fdda\`. Il codice di `9676fdda` è
identico a quello compilato (l'autosquash ha toccato solo tre doc).

## 6. Verifica

Vedi §9.

## 7. Piano per smettere di portarci dietro la pila

Criterio: proporre upstream ciò che è generale, opt-in e bit per bit; tenere nostro ciò che è legato alle due schede
sotto WDDM o che dipende da PR nostre ferme; ritirare ciò che non vale niente. **Nessuna PR aperta né commento in
questa sessione**: solo la lista.

Da proporre, in ordine (prima le piccole, autonome, bit per bit, con micro-bench):

1. **qsa-rows** (18): kernel solo, bit per bit, opt-in, +1-1,5%; branch `pr/qsa-rows-scores` già su v0.1.43.
2. **expdec** (19): kernel solo, bit per bit, opt-in, +1,1%; branch `pr/verify-expdec` già su v0.1.43.
3. **ple-direct** (15): piccola, generale (chi ha la PLE su un SATA lento vede 260 → 75 s); da proporre come opt-out
   della sonda (`STRATA_PLE_PROBE_MMAP=0`, o la sonda che avvisa e non cambia), con le misure del log.
4. **attn-daily** (10): compilare `attn_chunk_kernel_pre75` su ogni build CUDA e il merge v2 dietro un opt-in; bit per
   bit per costruzione, ma **prima** va fatto girare `qsa_merge_daily_parity` (test spenti nelle nostre build: mai
   eseguito) e un B-A-B pulito del solo `STRATA_ATTN_DAILY`.
5. **prefill-sm75** (16): i tre pezzi separati: SIMT di default sotto sm_80 (upstream lo ha voluto opt-in: proporre
   solo con i dati −16% prompt 2080 Ti), dequant 16 B (bit per bit, sm_75), hc16 opt-in (`STRATA_TC_SM75=1`, non bit
   per bit: dirlo).
6. **ram-core** (17): generale per chi usa l'arena residente (−10 GiB), opt-in; tocca `generate.cpp` e
   `expert_source.cpp`: PR media, dopo le quattro sopra.
7. **prefix-disk** (06): TTFT dopo un riavvio, opt-in, file nuovi + `generate.cpp`; grande, per ultima.
8. **df-defaults** (11): è un cambio di default che upstream ha rifiutato per #905 su Linux; proporre al più «acceso di
   default su Windows» con i dati; altrimenti resta config.

Già aperte da altri, da sostenere senza rifarle: #1911 (NVML), #1503, #1889 (SIDE_MIXER); le nostre #1117, #1417, #1418,
#1419 restano in coda.

Da tenere solo nostre: **05 elastic-split-pipeline** e **09 pipeline-architectds** (la topologia 3060+2080 Ti con due
finestre; dipendono da #1117, ferma da 0.1.40); **03 forced-bench** finché #1419 non entra; **04 B6** finché upstream
preferisce le tabelle IL (per noi vince B6).

Da **ritirare** al prossimo upsync se nessun B-A-B li riscatta: **12 critpath-levers** e **13 chain-fuse** (71 punti di
innesto, la superficie di conflitto più grande — `generate.cpp`, `verify.cpp`, `mtp.cpp` — per un guadagno nel rumore).
Senza di loro la pila scende a 17 leve e 275 innesti.

## 8. Da fare (coordinatore)

- B-A-B di velocità `build-upsync43-9676fdda` contro `build-candidate2-ac6065dc` (stessa config r2).
- Decidere su 12/13 (ritiro) e sul `stack/` nel branch di default del fork (`main`, oggi al vecchio lignaggio
  `e305fe09`: il commit di `stack/` lì andrebbe da solo, senza toccare il resto).
- Compilare una volta con `--tests` e far girare `qsa_merge_daily_parity`, `mmvq_il_parity`, `qsa_rows_bench --rows`,
  `expdec_bench` sulla build 0.1.43.

## 9. Esito della verifica (12/10, 07:55-07:59, sotto `bench.lock`)

Compilato sì, installato no, **provato sì** (due prove brevi, nessun B-A-B):

1. **Avvio** con una copia di `strata-0142-daily.json` (stessi argomenti ed env del Daily r2; exe `build-upsync43-9676fdda`,
   `--prompt-cache-file` e log propri, porta 8101): `/health` `loaded: true` dopo **70 s**, `images: true`, modello
   `qwen3.8-flash-next-iq3_xxs` a 262144 di contesto; una richiesta di 61 token servita (24 generati, 61 t/s a freddo).
   Nel log: `STRATA_VRAM_FREE=device` (NVML 7357 MiB contro 9679 di CUDA), sonda PLE 91 220 righe/s («fine», diretta),
   elastica a due zone, `strata ram core` dev1 5500 slot senza copia RAM, privata 77,54 → 67,37 GiB, working set 48,70 →
   33,63 GiB, riletti 0, finestre 1-6 catturate. Nessun errore.
2. **Banco forzato** `DET-x8-upsync43-b` (`det_force.sh` con gli argomenti di `coda_cand2.sh`, braccio B: `STRATA_EXPDEC=1
   STRATA_QSA_ROWS=1`): **L01.txt e L02.txt identici byte per byte** a quelli del Daily r2 (`DET-x7-cand2-b1`, sha
   `fd37c80c…` / `45d230d2…`); 521 e 600 finestre come il riferimento, `theta 0.10`, speculative 493/537, riletti 0,
   fuori posto 0, prefill 131 248 token in 72,9 s (1800 tok/s).

Velocità di quella singola corsa (non è un B-A-B, nessun braccio A, subito dopo la cattura MTP di un altro agente):
L01 69,7 t/s, L02 63,1 t/s contro 73,7/68,0 di `DET-x7-cand2-b1` (−5,5/−7,2%). **Da misurare col B-A-B del
coordinatore** prima di trarne conclusioni; se la differenza regge, i sospetti sono le novità di 0.1.43 attive anche con
la nostra config (guardia NaN `STRATA_NAN_GUARD=0` per escluderla, buffer pinned #1712, riordino degli scambi) o lo stato
delle schede dopo la corsa precedente.
