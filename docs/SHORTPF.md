# SHORTPF - il costo fisso delle letture piccole (18/10/2026)

Branch `lab/shortpf` da `lab/upsync-main` d9a26384 (upstream v0.1.45 + pila). Base misurata: `build-upsmain-d06429ee`
(sha256 fd889019...). Config: Daily 0.1.43 (JSON letto, non toccato) con `--elastic`, `--expert-cache auto`, due schede
(2080 Ti = CUDA0, 3060 = CUDA1), `--layer-split 16`, `--pipeline-windows 2`, MTP. Stato: **compilato la base si, la leva no,
provato solo la base.** Dati parziali (18/10, 20:35): la leva non e' ancora scritta.

## 1. Come si misura

`bench/shortpf/shortpf_bench.py`: conversazione `/v1/chat/completions` con ~12k token di sistema gia' in cache, poi turni con un
estratto di codice da ~100 / 300 / 1000 / 3000 / 8000 token nuovi (la risposta del motore e' rimandata come messaggio assistant:
riprende la conversation cache, si leggono solo il messaggio nuovo + ~25 token di template). TTFT = invio -> primo delta.
`--short-read` e' cambiato **per richiesta** nella stessa vita del motore (`STRATA_PIPELINE_DEBUG=1` +
`STRATA_PIPELINE_SWITCH`), cosi' finestre e percorso del prompt si confrontano alternati, sullo stesso contesto.
Motore a priorita' alta: la macchina e' condivisa e la CPU e' al 100% per i `node` di Codex (altri agenti); la prima corsa a
priorita' normale ha dato 131 tok/s a freddo (invece di ~600) ed e' stata scartata. **Rumore residuo alto** (stessa richiesta,
corse diverse: 8000 token 15-24 s): le differenze sotto ~10% non sono significative.

## 2. TTFT per dimensione (mediana, s; n = ripetizioni)

| nuovi token | percorso del prompt (`--short-read 0`) | finestre di decode (`--short-read 1024`) | col `--short-read 64` di oggi |
|---|---|---|---|
| ~120 | 1,74 (n=2; 1,72-1,77) | **1,06** (0,84-1,28) | 1,06 se N <= 64 altrimenti 1,74 (vedi sotto) |
| ~320 | 2,55 (2,50-2,59) | 2,79 (2,64-2,94) | percorso del prompt |
| ~1030 | **4,50** | 6,61 (6,42-6,80) | percorso del prompt |
| ~3050 | 7,73 | non provato (stima ~18 s a 6 ms a token) | percorso del prompt |
| ~8100 | 15-24 (rumore) | - | percorso del prompt |

- **Incrocio: ~250-300 token nuovi** (compresi ~25 di template e la risposta precedente). Sotto, le finestre vincono (~6 ms a
  token, ~1 s fisso); sopra, vince il percorso del prompt (fisso ~1,4-1,5 s, poi ~1,5-2,9 ms a token fino a ~1k).
- Con `--short-read 64` (oggi) una lettura da 100-250 token va sul percorso del prompt e paga 1,7-2,4 s dove le finestre ne
  pagherebbero 0,8-2,4. **Valore consigliato: `--short-read 256`** (tra 64 e 256 le finestre guadagnano 0,3-0,9 s; a 256 le
  due strade si pareggiano). 64 e 256 come vite separate del motore non sono state misurate: il confronto e' sulle due strade,
  che e' tutto cio' che l'opzione cambia (soglia sui token nuovi della parte).
- Mediane del tuo log (237 richieste >32k a 92 s, 477 <500 a 1,5 s) tornano con queste: il fisso e' ~1-1,5 s.

## 3. Scomposizione `STRATA_PREFILL_TIMING=1` (percorso del prompt, `--short-read 0`)

Tempo GPU di una corsa per stadio (ms) e quota delle fasi maggiori. Lo stadio 0 e' la 2080 Ti (16 layer), lo stadio 1 la 3060
(32 layer); **gli stadi vanno in serie** su un pezzo solo.

| token | stadio 0 | fasi stadio 0 | stadio 1 | fasi stadio 1 |
|---|---|---|---|---|
| 113 | 395 | dequant 43%, gemm gate/up 17%, down 7%, combine 7% | 685 | hc read 31%, dequant 26%, gather 8%, gemm 17% |
| 305 | 764 | dequant 48%, gemm 25% | 1357 | dequant 37%, hc read 17%, gemm 20%, gather 7,5% |
| 1027 | 1788 | dequant 46%, gemm 20%, combine 8% | 2600 | dequant 41%, hc read 13%, gemm 21% |
| 3067 | 3322 | combine 33%, dequant 23,5%, gemm 16% | 3786 | dequant 25,5%, hc read 18%, gemm 22% |
| 8148 | 6554 | combine 35%, dequant 15% | 6717 | hc read 22%, dequant 14,5%, gemm 20% |
| 6 (coda) | 108 | - | 352 | hc read 54% |

- "dequant" = da dopo l'attesa della copia fino al GEMM: il gather degli esperti nel layout MMQ **e** l'attesa dell'host che
  alimenta l'anello (arena non bloccata: solo 8 GiB sono pinned su ~40). E' il 40-48% fino a ~1k token: e' lo streaming PCIe degli
  esperti non residenti (~50% degli esperti per scheda), non una vera dequantizzazione.
- **Stadio 1: ~190-230 ms di "hc read" fissi, anche con 6 token** (stadio 0: 7 ms). Causa non ancora trovata.
- La coda da 6 token (intestazione dell'assistente) costa 460 ms sul percorso del prompt; con `--short-read >= 6` va in finestre.
- Dopo il pezzo: draft MTP 5-60 ms fino a 1k token, 183 ms a 3k, 505 ms a 8k (poco).

## 4. Leve a costo zero provate (solo env, percorso del prompt `--short-read 0`, build upsmain, config Daily con `--elastic`)

TTFT mediano in s (4 ripetizioni da ~100/300/1000, 2 da 3000, 1 da 8000; macchina meno carica che in §2: controllo rifatto).
Tra parentesi la differenza dal controllo `x0`.

| braccio | ~120 | ~320 | ~1030 | ~3050 | ~8100 |
|---|---|---|---|---|---|
| controllo x0 | 1,87 | 2,66 | 4,26 | 7,63 | 14,47 |
| `STRATA_PREFILL_CPU_SHARE=auto` | 1,58 (-15%) | 2,12 (-20%) | 3,48 (-18%) | 7,03 (-8%) | 14,39 |
| `STRATA_PREFILL_CPU_SHARE=0.6` | 1,35 (-28%) | 1,77 (-33%) | 2,98 (-30%) | 6,09 (-20%) | 12,66 |
| `STRATA_PREFILL_CPU_SHARE=0.8` | **1,23 (-34%)** | **1,67 (-37%)** | **2,95 (-31%)** | **6,07 (-20%)** | 12,67 (-12%) |
| `STRATA_PREFILL_STREAM_MIN=256` | 1,95 | 3,72 (+40%) | 4,56 | 7,66 | 14,44 |
| `HELP=1` + `STREAM_MIN=256` | 1,96 | 3,45 (+30%) | 4,34 | 8,01 | 15,77 |

- **La quota CPU e' la leva.** E' spenta di default sulla divisione a strati (`arm_cpu_share` la arma solo senza stadi): con
  `STRATA_PREFILL_CPU_SHARE=<f>` la CPU (inattiva durante il prompt) calcola gli esperti non residenti a pochi token invece di
  farli passare dal PCIe. Il `auto` misurato sta troppo basso (calibra una quota ~0,3-0,4); fissa 0,8 e' meglio di 0,6; con
  `TIMING` + share lo stadio 0 scende da 395 a 296 ms (113 token) e da 764 a 421 ms (306 token); lo stadio 1 da 685 a 523 ms e
  da 1357 a 771 ms. Il limite alto del percorso (3072 token) viene da `STRATA_PREFILL_CPU_SHARE_MAX`.
- `STREAM_MIN` piu' basso peggiora (lo streaming a ordine fisso non conviene sotto 1024); `HELP` non aiuta nei pezzi piccoli.
- **Non e' bitwise.** A cache fissa (`--expert-cache 1500`, senza `--elastic`, `--adapt-every 0`, `--short-read` via finestre o
  percorso) due corse del controllo hanno lo stesso testo in 13 risposte su 17 (le 4 diverse sono tutte nella seconda
  conversazione); controllo contro quota CPU: 4 su 17 uguali. Le risposte sono riassunti liberi da 32 token a temperatura 0:
  una variazione numerica di ~0,004 di KL (la nota di upstream sulla quota CPU) basta a cambiare le parole. La KL sul testo
  lungo non e' stata misurata.
- Residuo nello stadio 1: "hc read" ~160-170 ms fissi anche con 6 token (con la quota CPU e' il 32-59% della corsa).

## 5. Prossimi passi

1. Build `build-shortpf-<sha>` con `STRATA_PREFILL_LAYER_TIMES=1` (tempo host per layer) per trovare i ~200 ms di "hc read" dello
   stadio 1 e il costo per layer del percorso a pezzi piccoli.
2. Arm alternati: `STREAM_MIN` 128/256 (+HELP), CPU share, MTP batch; poi la leva con env proprio.
3. Verifica del testo a cache fissa (`--expert-cache 1500`, senza `--elastic`, `--adapt-every 0`) prima/dopo.
