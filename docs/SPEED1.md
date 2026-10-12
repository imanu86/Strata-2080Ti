# lab/speed1 — tagli di nodo nel grafo di verifica su CUDA

Base: `lab/upsync43-pulito` (Strata 0.1.43 + la pila del fork). Obiettivo della serie: 80+ t/s in chat singola a 131k
sulla coppia RTX 2080 Ti (stadio 1, strati 16-47) + RTX 3060 (stadio 0, strati 0-15), `--pipeline-windows 2`.

## 1. Da dove vengono i numeri

Traccia nsys della finestra a T=2 sullo stadio 1 (`--cuda-graph-trace=node`, finestra mediana): 1397 nodi, kernel
15,3 ms + intervalli fra nodi 3,9 ms. Sotto nsys il costo di lancio dell'host è gonfiato 2-3×: gli intervalli veri
fra nodi sono ~1,3-1,8 ms a finestra (analisi in `lab/hostgap`, `docs/HOSTGAP.md`). Per strato: 40 nodi, 403 µs di
kernel, 27 µs di intervalli. I nodi piccoli della finestra (mediana, somma a finestra):

| kernel | n | µs | per nodo |
|---|---|---|---|
| `wait_flag_ge` | 96 | 354 | 3,7 |
| `gr_norm_split` | 65 | 281 | 4,3 |
| `doorbell_publish` | 32 | 176 | 5,5 |
| `quantize_q8_1` | 97 | 172 | 1,8 |
| `swiglu_quantize` | 32 | 167 | 5,2 |
| `copy_i32_from_mapped` | 35 | 160 | 4,6 |
| `route` | 32 | 150 | 4,7 |
| `swiglu_q8_1_entries` | 63 | 143 | 2,3 |
| `add_hits` | 32 | ~143 | 4,5 |

Un nodo in meno vale il suo kernel più l'intervallo che lo segue (1-3 µs). `STRATA_DUMMY_NODES=k` (già in
`verify.cpp`) aggiunge k nodi vuoti per strato: misura diretta del costo di un nodo sulla 2080 Ti.

## 2. Classifica delle leve rimaste (ms a finestra, rischio bitwise, giorni)

| # | leva | ms a finestra | bitwise | giorni |
|---|---|---|---|---|
| 1 | **Tagli di nodo nel grafo** — (a) porta su CUDA i tagli gfx12 della 0.1.43 (`FUSE_WAIT` 2 nodi/strato, `FUSE_HITS` 1, `QFUSE_HC` 1): 128 nodi su 1397, kernel −0,44 ms + intervalli −0,13/−0,36 ms; (b) fusioni nuove (norma+quantize, swiglu+quantize, doorbell+route, wait nel `fetch_blobs`): altri 150-200 nodi | (a) −0,5/−0,8 · (b) −0,8/−1,5 · tot −1,3/−2,3 | (a) nessuno: stessi kernel e stesso ordine delle somme, già validati bitwise su gfx12; (b) basso-medio, le somme vanno tenute nello stesso ordine | (a) 0,5 (fatto, da verificare) · (b) 3-5 |
| 2 | **Proiezioni dense sullo sm_75** — staging SoA delle attivazioni e righe multiple per lane (lo schema di EXPDEC) in `gr_down_staged`/`gr_up_multi`/`mmvq_multi`/`b6_mmvq_rows`, che pesano ~⅓ dei 403 µs dello strato (~4,5 ms a finestra); −15/−20% su quei kernel | −0,7/−1,0 | basso se si tiene l'albero delle somme per riga (come EXPDEC, bitwise) | 2-3 |
| 3 | **Meno attese degli esperti su CPU** — `STRATA_ADAPT_OVERLAP=1` (0.1.43, opt-in su Windows) in B-A-B a costo zero; poi prefetch guidato dalle bozze MTP. Vale solo nelle finestre dove il pool arriva tardi (es. gather PLE 1,3 ms a finestra nel braccio B1 di hostgap) | −0,5/−1,5 quando c'è attesa, 0 altrimenti | **cambia i bit**: dove gira un esperto (CPU o GPU) cambia il risultato come un hit di cache | 0,5 (B-A-B) · 3-5 (prefetch) |

Scartate: hand-off scritto dal device (`lab/hostgap`: neutro in B-A-B); catena MTP (pilota a parte dell'owner).

## 3. Cosa fa questo ramo (leva 1a)

I kernel fusi della 143-amdk compilano su CUDA da sempre; su CUDA erano fermi solo i quattro gate in `verify.cpp`
(`#if defined(STRATA_USE_HIP) … #else return false`). Ora un build CUDA li accende per env, opt-in; i build HIP non
cambiano.

| variabile | taglio | nodi/strato |
|---|---|---|
| `STRATA_FUSE_WAIT=1` | `wait_flag_ge` + `copy_i32_from_mapped` → `wait_copy_i32_from_mapped`; `wait_flag_ge` + `copy_rows_from_mapped` → `copy_rows_from_mapped_after` | −2 |
| `STRATA_FUSE_HITS=1` | `add_hits` dentro `combine_k10_vec4_hits` (stessa somma `(p+h)·w`) | −1 |
| `STRATA_QFUSE_HC=1` | la lettura dell'hyper-connection scrive da sé l'immagine q8_1 di `mixed`: salta il `quantize_q8_1` degli esperti | −1 |
| `STRATA_FUSE_CONVAB=1` | conv GDN + proiezioni a/b in un lancio (su gfx1201 −0,3%: fuori dall'insieme) | −1 |
| `STRATA_NODECUT=1` | i primi tre insieme (la variabile propria vince: `STRATA_FUSE_HITS=0 STRATA_NODECUT=1` lascia fuori HITS) | −4 |

Riga di log al primo grafo registrato: `strata nodecut: CUDA fuse_wait 1 fuse_hits 1 qfuse_hc 1 fuse_convab 0 (…)`.
Condizioni già nel codice: niente fusione con il profiler (`prof_on_`), con il piano sul device (`device_plan_`), con
`remote_opt_`; HITS solo con `dec_batch`, n ≤ 8, k = 10, puntatori allineati a 16.

## 4. Verifica (sotto `bench.lock`, dopo le ~11:50)

1. Stessi token: due run brevi del banco (`run_long.py --tokens 256`, temperatura 0) con `STRATA_NODECUT=0` e `=1`,
   `generated_ids` uguali e `STRATA_DUMP_FIRST_LOGITS` uguale byte a byte.
2. Costo di un nodo: `STRATA_DUMMY_NODES=0/4/8` → µs per nodo sulla 2080 Ti, per fissare i ms della leva 1.
3. nsys di una finestra con `STRATA_NODECUT=1`: nodi 1397 → ~1269, intervalli e kernel piccoli.
4. B-A-B del coordinatore: A = config attuale, B = `STRATA_NODECUT=1`.

## Stato

- 12/10 09:15: gate portati su CUDA (opt-in). Il build (`build_lock.py`) aspetta che il pilota MTP rilasci
  `bench.lock` (~11:50); le verifiche del §4 vengono dopo: **compilato no / provato no**.
