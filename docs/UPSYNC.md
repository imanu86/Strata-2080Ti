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
| (vedi log) | nuovo | il drafter MTP usa `first_needed()` anche in `mtp.cpp`, così `set_full_prefix` vale davvero (pezzo di `0cc421e7`) |

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
| `5140210d` (architectds: verificata servita prima, lookup oltre una finestra, theta 0,10) | prestazione, non serve per far girare; nel Daily era acceso dal profilo `STRATA_PIPELINE_ELASTIC`. Qui theta resta 0,20 di upstream |
| `12a6f553` (`STRATA_PL_EARLY_CHAIN`, latenza della catena) | sperimentale / diagnostica |
| `434ddca4` (`STRATA_SPLIT_COMMIT_ASYNC`, tempi per blocco) | sperimentale / diagnostica |
| `ae6a4561` `d2f0d52a` `d868840b` `93af2041` `3634cad2` `3d702c72` | misure diagnostiche (rollback per classe, lancio del grafo, budget WDDM per fase, PLE) |
| `be4cf1c0` (B6 di default) | B6 resta opt-in; su 0.1.42 MMVQ_IL viene scelto prima (vedi sotto) |
| `15fd2eac` `3f436aab` `c88c2192` (`STRATA_PREFILL_EXACT_SMALL`, `STRATA_QSA_PREFILL_MULTI`, percorsi sm_75 `STRATA_TC_SM75`) | solo prefill; fuori dal decode che si misura |
| `b8a51dbc`..`df973585` | già dentro `c7f2b447` (tranne `8b9c342b`, portato col commit della pipeline) |
| `bdaf85d6` (#1122) `0cc421e7` (#1110) `dab24920` (#1091) `0f3ffb3f` (#904) `a32b5e1c` (#931), commit HIP/gfx11 di Claude/Strata W | già in upstream (a mano da Niko) |
| `fde196d9` (#1033) `5be83b69` (#1064) `f4a2590c` (#949) `65a1a138` (#965) | in upstream secondo i sorgenti (stringhe e funzioni presenti) |
| `a521c6a2` (#1038 `STRATA_PCIE_BALANCE`) `862c18aa` (#930) `3effa9d5` (attn merge v2) `f0071e71` `c3fc8f34` `185b5791` `6f565c6b` | porting di laboratorio o kernel opt-in non usati dal Daily |
| `2d47c09c`..`664d12a2`, `7aa91e1f`..`fa83c2e3`, `f968ef9c` `66b80a2e` `b9f5b9eb` | sonde e banchi (ROUTED_KEEP, neuroni, esperti raggruppati) |
| commit di `main` dell'8/10 (GPT6.1: `1c6da3ac`, `e305fe09`, `0c9862c4`, `b7b7903a`, `4fe26e35`, `93367637`, ...) | sperimentali, guadagno nullo (B A B +0,3%); il batch-MTP sul split è ora di upstream |

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
- **MMVQ_IL** (T 2-4, bit per bit) è di default su entrambe le schede: cambia solo la velocità. `STRATA_MMVQ_IL=0`
  rimette i kernel multi-colonna di 0.1.40. B6 (`STRATA_B6_MMVQ_ROWS=1`, nel Daily acceso) scatta solo dove IL non
  prende la forma: per riprodurre il Daily `STRATA_MMVQ_IL=0 STRATA_B6_MMVQ_ROWS=1`.
- **`STRATA_PIPELINE_ELASTIC`**: di default acceso con due schede e `--pipeline-windows` come nel Daily, ma senza il
  profilo architectds (theta 0,20 invece di 0,10, niente "verificata prima" né lookup oltre una finestra).
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

## Stato

Compilato: vedi il `build.json` della build `build-upsync-<sha>` in
`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\`. Installato no, provato no:
nessuna GPU, nessun test, nessun motore avviato.
