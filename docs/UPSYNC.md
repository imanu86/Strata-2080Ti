# UPSYNC: base pulita su upstream 0.1.42 per le misure a due schede

Branch `lab/upsync` del fork `imanu86/Strata-2080Ti`. Obiettivo dell'owner: ripartire da upstream e portare sopra solo
quello che serve alla configurazione del Daily (3060 12 GB stadio 0 col monitor + 2080 Ti 22 GB stadio 1, Windows/WDDM),
così da misurare su una base pulita. Prima i confronti erano falsati dalla VRAM della 3060 che varia sotto WDDM.

## Base

- `upstream/main` = `61b3fb5d` = tag **v0.1.42** (10/10/2026). Dopo `82f46a8` (v0.1.40.1, base del Daily `be4cf1c`):
  628 commit, 194 in prima linea (0.1.40.2, .3, .4, 0.1.41, 0.1.42).
- Delle nostre cose upstream NON ha incorporato nulla dei quattro PR aperti: #1117 (elastica), #1417 (prestito del
  prompt), #1418 (B6), #1419 (banco forzato). Niente `--elastic`, niente `--prompt-cache-file`, niente
  `STRATA_FORCE_*`, niente `STRATA_PIPELINE_ELASTIC`.
- Upstream ha invece ormai: pipeline a più stadi con il tier adattivo asincrono (#1122, `STRATA_PIPELINE_ADAPT_ASYNC`),
  parcheggio delle conversazioni con `--layer-split` (immagini per blocco), `--batch-groups auto` di default con split e
  `--batch` (0.1.41), batch-MTP sul layer split (#1636), il fix dello stallo del layer split su Windows (`6b392641`,
  nessuna chiamata CUDA nel poll finché il campanello non suona), tipo della testa del draft nei drafter di `--batch-mtp`
  (#1110), punteggi QSA FP32 a tile sotto sm_80 (#1091), rami del grafo/PDL di #904 (`STRATA_DF_*`), pin degli stage
  opt-in (`STRATA_STAGE_PIN=1`), MMVQ_IL per T 2-4 di default (bit per bit), route tail skip di default (vedi sotto).

## Patch portate (un commit ciascuna, in ordine)

| lab/upsync | Origine nel fork | Cosa |
|---|---|---|
| `836adb09` | `c7f2b447` (PR #1117) | cache elastica con nucleo fisso, `--elastic` |
| `abf69469` | `4f9f1489` (PR #1417) | prestito del prompt stimato sul device del proprio blocco |
| `71ece11b` | `398bc08a` `7954908b` `4780a0cd` (PR #1419) | banco forzato `STRATA_FORCE_IDS` / `STRATA_FORCE_WINDOWS` / `STRATA_FORCE_TRACE`, adattato alla pipeline a più stadi (rigioco al lancio sullo stadio 0) |
| `566d0e40` | `8cae814e` (PR #1418) | B6 MMVQ multi-colonna, **opt-in** `STRATA_B6_MMVQ_ROWS=1` |
| `5c8d075c` | `2c820e6b` `81f49716` | elastica solo sul blocco di CUDA0 con `--layer-split`; il blocco del prompt non cresce oltre quello iniziale; regola `--batch-groups` ripetuta sul numero risolto |
| `f4f93254` | `dd8eba52` `1541e340` `12a6f553`(parte) `83c27bbd`(parte) `a0ebf745` `8b9c342b` | elastica di CUDA0 accanto a `--pipeline-windows`, accesa di default con due schede: la coda si muove solo ai vuoti dello stadio 0 (`el_gap`), crescita pubblicata quando i riempimenti sono arrivati, coda 1,5 GiB, passo 128 MiB, il prompt prende in prestito anche il nucleo |
| `f3f75bdd` | `f01c0843` `96e2cdc9`(parte) `434ddca4`(parte) | prefisso su disco `--prompt-cache-file` con due schede (formato v2, un'immagine per blocco) |
| `865d8203` | upstream PR #1911 (aperta) | `STRATA_VRAM_FREE=device`: VRAM libera della scheda intera via NVML |
| `b033a625` | nuovo | la cache elastica legge la VRAM libera con `device_free_bytes()` (NVML con `STRATA_VRAM_FREE=device`) |
| `03da168d` | upstream PR #1503 (aperta) | il parcheggio non usa mai il K/V di un'altra conversazione |
| `bb816e18` | pezzo di `0cc421e7` | il drafter MTP usa `first_needed()` anche in `mtp.cpp`, così `set_full_prefix` vale davvero |
| `341542b7` | nuovo | il prefisso su disco usa `available_host_bytes()` (0.1.42 ha tolto `conversation_available_memory`) |
| `da277800` | `5140210d` + gating di `83c27bbd` | architectds: finestra verificata servita per prima (`Verifier::service(..., max_layers)`; stadio frontale di A, poi l'ultimo stadio di A, poi D e B, B un layer per passata mentre gira l'ultimo stadio di A), lookup oltre una finestra (`lookup_next`, anche dopo una finestra della catena con match >= 12 e tasso >= 0,85), **theta 0,10**. Acceso dal profilo `STRATA_PIPELINE_ELASTIC` (di default con due schede e `--pipeline-windows`, come nel Daily) |
| `856bd674` | `ae6a4561` | log: costo dei rollback per classe (`strata pipeline rollback cost`) |
| `ab54c5e1` | `12a6f553` (parte rimasta) | log: latenza della catena dal lancio al B pronto, forzata e fresca |
| `5d3b59cb` | `f0071e71` (kernel dell'attenzione del decode del Daily) | `STRATA_ATTN_DAILY=1`: il kernel dei chunk dell'attenzione QSA del decode del Daily su ogni scheda CUDA (vedi sotto). **Opt-in** |
| `d0b9121d` | `3effa9d5` (via `lab/wip-0809` `dd249e45`) | `STRATA_ATTN_DAILY=1` accende anche il merge `attn_merge_v2_kernel` del Daily; harness `qsa_merge_daily_parity`. **Opt-in** |
| `133e3a69` | default del Daily per il PR #904 (`be4cf1c`, `verify.cpp`) | `STRATA_DF_BRANCH` acceso di default con CUDA su Windows, `STRATA_DF_PLE` acceso di default con CUDA, come nel Daily; `=0` li spegne |

`host-core`, `--layer-split`, `--trim-stage-weights`, `--pipeline-windows`, `--vram-reserve-later-mib`,
`--prompt-cache-tail/-every` sono già di upstream.

### NVML con due schede

`nvml_device_free` cerca la scheda NVML dal PCI bus id del device CUDA **corrente** (`dominio:bus:device.0` da
`cudaGetDeviceProperties`), quindi l'abbinamento non dipende dall'ordine di `CUDA_VISIBLE_DEVICES=1,0`: la cache elastica
legge su `OnDevice(0)` = la 3060, e NVML risponde per la 3060. Il dimensionamento all'avvio di upstream usa già
`device_free_bytes()` su ogni scheda. Non provato sulle schede (compilato sì, provato no).

## Patch lasciate fuori

| Fork | Perché |
|---|---|
| `12a6f553`, solo `STRATA_PL_EARLY_CHAIN` (#910 di Hardin22) | opt-in, spento nel Daily (`strata-dueschede.json` non lo imposta) e assente in upstream 0.1.42: non cambia il comportamento da riprodurre |
| `434ddca4` (`STRATA_SPLIT_COMMIT_ASYNC`, tempi per blocco) | sperimentale / diagnostica |
| `d2f0d52a` `d868840b` `93af2041` `3634cad2` `3d702c72` | misure diagnostiche (lancio del grafo, budget WDDM per fase, PLE) |
| `dd8eba52` (punto quieto `el_plan`, `pl_el_held`), `1541e340` (`pl_el_watch`) | `dd8eba52` era già sostituito nel Daily da `1541e340` (che toglie `el_plan`/`pl_el_runs`/`pl_el_held`); di `1541e340` manca solo il log delle finestre dopo gli eventi elastici |
| `be4cf1c0` (B6 di default) | B6 resta opt-in; su 0.1.42 MMVQ_IL viene scelto prima (vedi sotto) |
| `15fd2eac` `3f436aab` `c88c2192` (`STRATA_PREFILL_EXACT_SMALL`, `STRATA_QSA_PREFILL_MULTI`, percorsi sm_75 `STRATA_TC_SM75`) | solo prefill; fuori dal decode che si misura |
| `b8a51dbc`..`df973585` | già dentro `c7f2b447` (tranne `8b9c342b`, portato col commit della pipeline) |
| `bdaf85d6` (#1122) `0cc421e7` (#1110) `dab24920` (#1091) `0f3ffb3f` (#904) `a32b5e1c` (#931), commit HIP/gfx11 di Claude/Strata W | già in upstream (a mano da Niko) |
| `fde196d9` (#1033) `5be83b69` (#1064) `f4a2590c` (#949) `65a1a138` (#965) | in upstream secondo i sorgenti (stringhe e funzioni presenti) |
| `a521c6a2` (#1038 `STRATA_PCIE_BALANCE`) `862c18aa` (#930) `c3fc8f34` `185b5791` `6f565c6b` | porting di laboratorio o kernel opt-in non usati dal Daily (`3effa9d5` e `f0071e71` erano invece di default nel Daily: portati opt-in in `5d3b59cb`/`d0b9121d`) |
| `2d47c09c`..`664d12a2`, `7aa91e1f`..`fa83c2e3`, `f968ef9c` `66b80a2e` `b9f5b9eb` | sonde e banchi (ROUTED_KEEP, neuroni, esperti raggruppati) |
| commit di `main` dell'8/10 (GPT6.1: `1c6da3ac`, `e305fe09`, `0c9862c4`, `b7b7903a`, `4fe26e35`, `93367637`, ...) | sperimentali, guadagno nullo (B A B +0,3%); il batch-MTP sul split è ora di upstream |

## Attenzione QSA del decode (`STRATA_ATTN_DAILY=1`)

Dopo `ab54c5e1` la pipeline è a posto (speculative 172 contro 164, theta 0,10) ma resta ~1,1 ms di GPU in più per
finestra (banco forzato 131k, B-A-B, L02 600 finestre: GPU-reach wait 15,25 contro 14,16 ms, fresche 30,19 contro 28,36,
speculative 20,67 contro 19,57). nsys sulla 3060: `attn_chunk_kernel` 99,77 µs su upstream contro 46,42 nel Daily.
La causa principale sono però i `STRATA_DF_*` spenti (vedi «Opzioni che su 0.1.42 cambiano significato»), riaccesi
in `133e3a69`; l'attenzione è il secondo pezzo.

Perché: il Daily (`f0071e71`, dal fork Encapsulate) aveva messo nel kernel principale `attn_chunk_kernel` il corpo di
PR #540 (sskver): due celle per passo con `reduce12` (16 shuffle invece di 12 × 5 `warp_sum` per cella), probabilità
cell-major (`spt`, tre float4 per cella nel ciclo dei V), V caricati a gruppi di quattro. In più il merge
`attn_merge_v2_kernel` (`3effa9d5`, di Hardin22), di default su NVIDIA. Upstream 0.1.42 ha lo stesso corpo come
`attn_chunk_kernel_pre75`, ma su CUDA lo compila solo con `STRATA_EXPERIMENTAL_SM60` e lo sceglie solo sotto sm_75:
la 3060 (sm_86) e la 2080 Ti (sm_75) girano il kernel di base. Il merge v2 non c'è.

Port (`5d3b59cb`, `d0b9121d`): `attn_chunk_kernel_pre75` si compila in ogni build CUDA; `STRATA_ATTN_DAILY=1` lo sceglie
su ogni scheda (decode a una query e batch: verifica, drafter MTP) e accende il merge v2.
`attn_chunk_kernel_pre75<M, 2, false>` è il kernel del Daily riga per riga; unica differenza: `spt` condivide la memoria
di `sq` (15 872 B di shared come il kernel di base, il Daily ne usava 3 KB in più), nessuna differenza aritmetica.
Registri: 68-78 contro 37-41 del kernel di base (`cuobjdump -res-usage` dell'obj), da 256 thread per blocco.

Bit per bit: identico per costruzione. `reduce12` somma in ogni livello le stesse due coppie del butterfly di `warp_sum`
(xor 16, 8, 4, 2, 1; la somma float è commutativa), i V si piegano nello stesso ordine di celle con gli stessi `fmaf`,
le celle mascherate restano escluse; il merge v2 fa gli stessi `fmaf` nello stesso ordine e salta esattamente i chunk
che il kernel dei chunk marca vuoti. Quindi Daily, upstream e `STRATA_ATTN_DAILY=1` danno lo stesso output.
Verifica sulla GPU (da fare): `qsa_merge_daily_parity` (con `--tests`; ctest con `STRATA_ATTN_DAILY=0` e `=1`) confronta
con memcmp il kernel dei chunk del Daily e il merge v2 col riferimento upstream su quattro formati KV, larghezze ai
bordi e pagine non residenti; `--bench` misura i due kernel dei chunk in B-A-B. Resta **opt-in** finché non gira.

Variabili: `STRATA_ATTN_DAILY=1` (tutto il percorso del Daily), `STRATA_ATTN_MERGE_V2=0/1` (forza solo il merge),
`STRATA_ATTN_PRE75=0` (spegne il kernel dei chunk del Daily). HIP invariato.

B-A-B sul banco: stesso exe e stesse opzioni del confronto col Daily, `STRATA_ATTN_DAILY=1` / niente / `=1`. Atteso:
~50 µs in meno per chiamata di `attn_chunk_kernel` sulla 3060 (nsys); quanta parte del ~1,1 ms per finestra copre
è da misurare.

## Riprodurre il Daily

Stessi argomenti del JSON `strata-dueschede.json` del Daily:

```
--pack ... --native <shard1> --ple-gguf <shard2> --native-dense-gguf <shard1> --native-dense-gguf <shard2>
--ple-row-cache 16777216 --expert-profile ...\expert-profile.bin --expert-cache auto --prefill auto
--spec 4 --spec-min-p 0.5 --mtp ...\rt_it48k --max-context 262144 --kv int8 --kv-resident 32768
--vram-reserve-mib 1393 --elastic --prompt-cache-tail --prompt-cache-every 16384
--prompt-cache-file <UN ALTRO FILE, non quello del Daily>
--layer-split 16 --trim-stage-weights --pipeline-windows 2 --vram-reserve-later-mib 384 --host-core last
```

Ambiente: `CUDA_VISIBLE_DEVICES=1,0`, `STRATA_POOL_SPIN_US=100`, `STRATA_WATCHDOG_S=240`.
Per confronti col Daily aggiungere `STRATA_ROUTE_TAIL_SKIP=0`.

### Opzioni che su 0.1.42 cambiano significato

- **`STRATA_ROUTE_TAIL_SKIP=7` è acceso di default** (CUDA serve, anche col layer split, ogni stadio per sé): gli esperti
  mancati che tutti i token della finestra mettono al rango 7 o oltre non vengono né copiati né calcolati. Risposte
  leggermente diverse e **meno lavoro sugli esperti**: anche col banco forzato il carico non è quello del Daily.
  `STRATA_ROUTE_TAIL_SKIP=0` torna al comportamento del Daily.
- **MMVQ_IL** (T 2-4, bit per bit) in 0.1.42 è di default **solo sulla 3060**: `il_arch_ok()` vuole sm_80+ (o
  sm_70 con `STRATA_SM70_TABLE=1`), quindi la 2080 Ti (sm_75) resta sui kernel multi-colonna. Correzione del
  `lab/ports1`: qui prima c'era scritto che sm_75 usava la tabella sm_80+, ma nel codice di 0.1.42 non succede.
  Dal `lab/ports1` (PR #1437) la 2080 Ti ha la sua tabella per forma con `STRATA_MMVQ_IL=1` (vedi sotto).
  `STRATA_MMVQ_IL=0` rimette i kernel multi-colonna di 0.1.40 su tutte e due. B6 (`STRATA_B6_MMVQ_ROWS=1`, nel Daily
  acceso) scatta solo dove IL non prende la forma: per riprodurre il Daily `STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1`.
- **`STRATA_PIPELINE_ELASTIC`**: di default acceso con due schede e `--pipeline-windows` come nel Daily, col profilo
  architectds dal `da277800` (theta 0,10, verificata prima, lookup oltre una finestra). `STRATA_PIPELINE_ELASTIC=0`
  torna a upstream (theta 0,20, ordine D-A-B). Il riepilogo scrive `theta 0.10` e `yield on`.
- **`--vram-reserve-mib 1393`** era tarato per compensare la cecità di WDDM. Con `STRATA_VRAM_FREE=device` la riserva
  si confronta con la VRAM libera dell'intera 3060 (desktop 8K compreso): la stessa cifra lascia più margine reale;
  va ritarata (probabile 700-1000) prima di confrontare.
- **`--prompt-cache-file`**: l'identità del file include exe, argomenti e variabili `STRATA_*`/`CUDA_*`: il file del
  Daily (`prompt-prefix-dueschede-v1.bin`) non corrisponde e verrebbe **sostituito**. Usare un percorso diverso.
- `STRATA_PREFILL_EXACT_SMALL=1`, `STRATA_QSA_PREFILL_MULTI=1`: qui non esistono (solo prefill del fork): nessun effetto.
  `STRATA_PARALLEL_INTERMEDIATE_QUANT=0` era già il default del fork e qui non esiste: nessun effetto.
- Il riparto PCIe automatico (0.1.42) e la quota di prefill su CPU (0.1.41) valgono con una GPU sola: col layer split
  restano spenti, come nel Daily.
- `--batch` con layer split ora pipelina un gruppo per stadio di default: la cache elastica si spegne da sola
  (`--batch-groups 1` la tiene).
- **`STRATA_DF_*` (PR #904, `verify.cpp`)**: nel Daily accesi di default su CUDA, in 0.1.42 opt-in (`e != nullptr`).
  `STRATA_DF_BRANCH` (rami laterali del mixer nel grafo della finestra, +2,4% a 131k nel Daily; upstream lo ha spento
  per uno stallo su Linux, #905, mai visto su Windows) e `STRATA_DF_PLE` (post-proiezioni PLE della finestra in una
  passata, bit per bit, +3,3%): **dal `133e3a69` di nuovo accesi di default** (BRANCH con CUDA su Windows, PLE con
  CUDA), `=0` li spegne. Erano la causa principale del ~1,1 ms di GPU per finestra in più di `ab54c5e1`.
  `STRATA_DF_PDL` (solo sm_90+) e `STRATA_DF_QB_Q4` (solo KV q4_0) cambiano default anche loro ma non toccano la 3060,
  la 2080 Ti né `--kv int8`: lasciati come upstream.
- **`STRATA_SELECT_SIMT`** (#1091, blocchi QSA FP32 a tile sotto sm_80): nel Daily acceso di default, in 0.1.42 opt-in
  (`=1`). Riguarda solo il prefill; per un confronto di prefill col Daily `STRATA_SELECT_SIMT=1`.

### Kernel del decode cambiati da 0.1.40.1 a 0.1.42 (con le opzioni del confronto)

Con `STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1 STRATA_ROUTE_TAIL_SKIP=0` HEAD lancia sostanzialmente gli stessi kernel
del Daily, tolti l'attenzione QSA (sopra) e i `STRATA_DF_*`.

| Kernel | Cambio | Effetto atteso |
|---|---|---|
| attenzione QSA (`qsa_decode_attn.cu`) | il corpo veloce del Daily solo sotto sm_75; niente merge v2 | ~2x sul kernel dei chunk; ripreso con `STRATA_ATTN_DAILY=1` |
| GDN (`gdn*.cu`), `gr.cu`, `native_gr_postops.cu`, `native_ple_postops.cu` | identici al Daily | nessuno |
| esperti (`s2_expert_grouped.cu`, `shared_expert.cu`) | identici | nessuno |
| HC (`fused_gr.cu`, `ccd657a2`) | prodotti della norma HC tenuti nello scratch esistente | bit per bit, neutro |
| MMVQ (`native_mmvq.cu`, `q8_1_il.cuh`) | MMVQ_IL per T 2-4 di default sulla 3060; la 2080 Ti (sm_75) resta sui kernel multi-colonna (`il_arch_ok()`: sm_80+), tabella sm_75 opt-in dal `lab/ports1` | spento con `STRATA_MMVQ_IL=0`; B6 opt-in qui, di default nel Daily |
| router (`native_router.cu`, `route_prior.cu`) | `STRATA_ROUTE_RESIDENT`, `STRATA_ROUTE_PRIOR` opt-in; route tail skip di default | nessuno con `STRATA_ROUTE_TAIL_SKIP=0` |
| `iq_kernels.cu`, `verify_kernels.cu` | clamp delle scale q8_1 (#606, #1448), Volta mode 8 opt-in | trascurabile |
| `dequant_bf16.cu`, `native_bf16.cu`, `sampler.cu`, `qsa_prompt_attn.cu` | Q8_0 coalescato, PDL sotto sm_70, sampler opt-in, Volta | nessuno sul decode a greedy |
| `qsa_select.cu` | SIMT #1091 opt-in (di default nel Daily) | solo prefill |

## lab/ports1: PR #1437 (MMVQ_IL su sm_75) e #1889 (`STRATA_SIDE_MIXER`)

Branch `lab/ports1` da `lab/chainfuse` (`50c04dc0`). Due PR upstream ancora aperte, portate con `gh pr diff`:

| Commit | PR | Cosa |
|---|---|---|
| `2f84fa7f` | #1437 (porting su 0.1.42: il PR è su un `main` precedente alle tabelle per architettura) | tabella `kIlRowsSm75` per forma esatta (n_in x n_out), solo con `STRATA_MMVQ_IL=1` esatto; `native_mmvq_il_supported(..., n_in)`, `native_mmvq_il_rows_for(cc, ..., n_in)`, `native_mmvq_il_last_rows()` |
| `4ad73342` | #1437 | `mmvq_il_parity`: forme sm_75, controllo del percorso lanciato, opt-in rigoroso, `--sm75-bench` a coppie |
| `ae719a8e` | #1889 (applicato com'è) | `STRATA_SIDE_MIXER=1/2`, più le chiamate IL del ramo con `n_in` ed esclusione di `STRATA_DF_BRANCH` in init |

### MMVQ_IL su sm_75 e B6: chi vince

La tabella sm_75 vale solo con `STRATA_MMVQ_IL=1` **esatto** (`true`, `yes` o non impostato la lasciano spenta; `=0`
spegne IL su tutte e due le schede). La 3060 (sm_86) non cambia: `=1` o non impostato le danno la stessa tabella
condivisa sm_80+. `STRATA_MMVQ_IL_ROWS` sovrascrive anche la tabella sm_75, ma solo con l'opt-in.

Con `STRATA_MMVQ_IL=1` e `STRATA_B6_MMVQ_ROWS=1` insieme la scelta si fa per chiamata, in `verify.cpp` (`mm`):

1. se la forma esatta e T (2-4) hanno una cella non nulla nella tabella della scheda, **vince IL** (kernel interleaved):
   B6 non vede quella chiamata;
2. altrimenti la chiamata va a `native_mmvq` e lì **vince B6** (Q4_K, Q5_K, Q6_K, IQ4_XS, layout esatto, niente
   S26 tsum). Sulla 2080 Ti questo vuol dire: celle a 0 della tabella (es. IQ4_XS 2560x640 a T 2-3, Q6_K 2560x6144 e
   6144x2560 a T 2), forme non elencate (le chiavi 2560x512 ecc.), finestre T 5+, e tutto quello che non passa dal
   gate IL (draft, prompt).

Con `STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1` B6 prende tutto, come nel Daily. Non provato sulle schede: le
celle in cui IL batte B6 sulla 2080 Ti vanno misurate (`mmvq_il_parity --sm75-bench` confronta IL con i kernel
multi-colonna, non con B6; con `STRATA_B6_MMVQ_ROWS=1` nell'ambiente il lato «multi» del banco è B6).

### `STRATA_SIDE_MIXER` e `STRATA_DF_BRANCH`: si sovrappongono

Stessa idea (lavoro del mixer che legge solo l'input del layer su rami del grafo della finestra), passi in parte uguali:

| | GDN | QSA | altro |
|---|---|---|---|
| `STRATA_DF_BRANCH` (qui acceso di default con CUDA su Windows) | a/b, z | catena di q, query dell'indicizzatore (solo `qb`) | copia di step/pos della finestra sul ramo; solo G = 1 (niente `--spec-split`), niente batch, niente profiler |
| `STRATA_SIDE_MIXER=1` | a/b | tutta la catena dell'indicizzatore: chiave, snapshot della coda, append, query, punteggi, top-k | niente batch |
| `STRATA_SIDE_MIXER=2` | a/b, z (anche col kernel IL) | come `=1` più la catena di q (anche col kernel IL) | niente batch |

`=2` contiene tutto `STRATA_DF_BRANCH` tranne la copia degli input. Per questo non si accendono mai insieme: con
`STRATA_SIDE_MIXER` a 1 o 2 `STRATA_DF_BRANCH` si spegne già in `Verifier::init` (i suoi stream non vengono creati)
e in `record_window` (`br` falso). Il default Windows cede in silenzio; se `STRATA_DF_BRANCH` è chiesto in modo
esplicito, stderr lo dice una volta. `STRATA_SIDE_MIXER=0` o non impostato: il codice di prima, stessi lanci nello
stesso ordine. Dichiarato dal PR: bit per bit (exact7 7/7), +1,3% / +4,2% (tail skip acceso / spento) su una 5080
Linux, `=2` 1,9% per giro meno di `STRATA_DF_BRANCH`. **Non provato** col layer split né su Windows; il rischio di
stallo di #905 (fork/join nel grafo) è lo stesso di `STRATA_DF_BRANCH`, che da noi su Windows non si è mai visto.

Da provare sul banco (B-A-B, stesse opzioni del confronto):
- 2080 Ti: `STRATA_MMVQ_IL=1` contro non impostato, con e senza `STRATA_B6_MMVQ_ROWS=1`;
- mixer: default (`STRATA_DF_BRANCH` acceso) contro `STRATA_SIDE_MIXER=2` (e `=1`).

## Stato

Compilato sì (`133e3a69`, `build_lock.py`, CUDA 75;86, test spenti): `strata.exe` sha256
`811d43f8250dde1926ae98e40f9fb11d3f25f0479152eed30047da42f7247c1d`, copiato con `build.json` in
`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\build-upsync-133e3a69\`.
Installato no, provato no: nessuna GPU, nessun test, nessun motore avviato; `qsa_merge_daily_parity` non compilato
(test spenti). Rispetto a `ab54c5e1` cambiano di default `STRATA_DF_BRANCH` e `STRATA_DF_PLE` (accesi, come nel
Daily); l'attenzione del Daily resta dietro `STRATA_ATTN_DAILY=1`.

Da misurare sul banco, stesse opzioni del confronto col Daily:
1. `133e3a69` contro il Daily: il ~1,1 ms per finestra dovrebbe sparire in buona parte (DF_*);
2. B-A-B `STRATA_ATTN_DAILY=1` / niente / `=1` per il resto; se il guadagno c'è e la parità passa, accenderlo di default.

Build `d0b9121d` (attenzione opt-in, DF_* come upstream): sha256
`c416e9d1536cc42b512be466df4532c1d051c8dbe89e2a93627f0ecfbc5610fd` (`build-upsync-d0b9121d\`).

Build `ab54c5e1` (senza l'attenzione del Daily): sha256 `397e065d0064493e9b25a2309d1e5964430bdb833354581aa6038163e8878558`
(`build-upsync-ab54c5e1\`).

Perché: sul banco forzato a 131k (L01/L02) `341542b7` era più lento del Daily del 4,6-7,3% anche con le opzioni
del Daily; su L02 164 finestre speculative su 600 nel Daily contro 133, stage host 0,50 → 1,69 ms, per-layer host
1,09 → 2,36 ms. Mancava il profilo architectds: con theta 0,20 meno B passano il gate, e con l'ordine D-A-B di
upstream il ciclo serve tutto B (e D) prima del layer di A su cui la scheda aspetta. Da verificare sul banco: il
riepilogo deve dire `theta 0.10`, `yield on`, e le speculative tornare verso 164/600.

Build precedente: `341542b7`, sha256 `e25c9e2f5e1d3857aba37a49e5d072d4907f3eb95cfd6aff08057ebd42bc4e64`
(`build-upsync-341542b7\`), senza il profilo architectds.
