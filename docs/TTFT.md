# TTFT: tempo al primo token nel nostro uso

Branch `lab/ttft` del fork `imanu86/Strata-2080Ti`, da `origin/lab/candidate` (`443ce1c9`). Uso reale: Claude Code e
DSH su due schede (3060 stadio 0, layer 0-15; 2080 Ti stadio 1, layer 16-47), `--layer-split 16 --pipeline-windows 2
--prompt-cache-file --prompt-cache-every 16384 --prompt-cache-tail`, contesti 30-130k.

Priorità dell'owner (12/10): i turni successivi vanno già bene; lento è il **primo** TTFT (prima richiesta dopo
l'avvio, primo turno di una sessione nuova, ripresa di una conversazione dopo un riavvio).

## Dati

| Log | Cosa dice |
|---|---|
| `D:\ds4_work\swe_casa\runs_cc\cc-split-0710b\engine.log` (Daily `be4cf1c`, due schede, Claude Code) | 1ª richiesta 39 795 token da 0: **41,8 s** (952 tok/s). File del prefisso assente; `waited 3791 ms` e `waited 5666 ms` per le righe PLE (il primo chunk di ognuna delle due parti, sistema e primo messaggio); salvataggio 26 395 token in 760 ms dentro la richiesta |
| `D:\ds4_work\swe_casa\runs_cc\cc4-batchmtp\engine.log` | prefisso su disco trovato: 26 786 ripristinati (lettura e controllo 634 ms, upload 81 ms) + 16 945 letti in **19,8 s** (856 tok/s) |
| `D:\ds4_work\strata\sota\strata_sota.log` (una scheda) | 8 volte `miss (prefix length changed)`: un altro client aveva sovrascritto l'unico file. Con il prefisso di 31 190: 60 522 token in 22,3 s; ripresa di conversazioni da 109-126k: **66-68 s** |
| `lab_runs\overnight-abab-20261009-1943\bench-ssd\runs\*\engine.log` | 113 090 token dal disco: lettura e controllo 2,0 s + upload 0,25 s; poi 18 158 letti in 22 s |
| `D:\ds4_work\strata\sota\strata_candidate.log`, `DET-*\engine.log` | candidato: 25 token in 301 ms; banco L02: 167 token dopo il checkpoint a 131 072 in 1,5-1,9 s (con parcheggio e ripristino del checkpoint) |

## Primo TTFT: dove va il tempo

| Caso | Oggi | Pezzi |
|---|---|---|
| Claude Code ~40k, sessione nuova, file del suo prefisso presente | 16-20 s | lettura e controllo del file 0,6-0,8 s + upload 0,1 s; 13-17k token del primo messaggio a freddo (~850 tok/s), di cui ~5 s di attesa per le righe PLE del primo chunk; prima finestra con cattura dei grafi ~0,3 s |
| Claude Code ~40k, file sovrascritto da un altro client (o assente) | ~42 s | 40k token da 0; 9,5 s di attese PLE (primo chunk di ogni parte); 0,76 s di salvataggio del prefisso dentro la richiesta |
| DSH, prefisso ~31k | 22 s se il file è il suo, altrimenti come sopra | stesso schema |
| ripresa di una conversazione da 130k dopo un riavvio | 65-80 s | solo il prefisso (26-31k) dal disco, il resto (~100k) riletto |

Perché la prima richiesta è lenta oltre la prefill:

1. **Un solo file del prefisso.** Il file tiene un prefisso; il primo client diverso lo sovrascrive e l'altro riparte da
   zero (42 s invece di 18).
2. **PLE a freddo.** La cache delle righe è vuota dopo l'avvio e la lettura diretta salta la cache del sistema. Il
   primo chunk di ogni `sp.run` aspetta tutte le sue righe (la lettura anticipata copre solo i chunk dopo): 3,8 s per
   la parte del sistema, 5,7 s per la parte del primo messaggio.
3. **Salvataggio del prefisso sincrono**: SHA-256 e scrittura di 500 MB dentro il tempo del prompt.
4. **Il prefisso su disco si ferma al primo confine di turno** (fine del system, ~26k su ~40k): il resto è il primo
   messaggio, che in Claude Code contiene CLAUDE.md, memoria e stato git e cambia da una sessione all'altra. Un
   salvataggio più lungo serve a riprendere *la stessa* conversazione (punto 3 del piano), non alle sessioni nuove.
5. Costi a freddo minori: cattura dei grafi delle finestre alla prima decodifica (~0,3 s), caricamento pigro dei
   moduli CUDA, cache elastica che cresce dopo 5 s di spazio (pesa sulla decodifica, non sul TTFT).

## Turno tipico (append di 100-3 000 token su 40-130k), per riferimento

Regressione sui 25 turni del log SWE: `ms ≈ 490 + 1,8·N + 10,5·(contesto in migliaia)`, cioè 0,9-1,2 s fissi per
pochi token e ~550 tok/s marginali (la prefill da 131k fa ~1 750 tok/s).

| Pezzo | Costo | Causa |
|---|---|---|
| server: template e tokenizzazione | 20-60 ms (60 ms a 140k) | `encode_prompt` ricodifica tutto: #1852 lo porta a ~2 ms |
| ricerca del prefisso | ~0 | il turno riparte dalla sessione viva (`from_live`): niente parcheggio né ripristino |
| parte fino all'ultimo `<\|im_start\|>` | da `--short-read` 64 in giù: finestre da 6 token, 45-60 ms l'una (8-10 ms/token); sopra: percorso a chunk | sotto 1 024 token il chunk è «staged»: anello di 8 slot, una sincronizzazione per layer MoE, esperti non residenti per PCIe (la 3060 è a 6,7 GB/s); un solo chunk = i due stadi in serie (lo stadio 1 aspetta la consegna dello stadio 0); `STRATA_PREFILL_CPU_SHARE` è spento col layer split |
| dentro il chunk piccolo (`STRATA_PREFILL_TIMING`, 2080 Ti sola, `.codex_runs\20261003_chat_bottleneck\baseline_okto_profile`) | 211 token: GPU 1 885 ms, di cui attesa delle copie degli esperti 38%, dequant 21%, GEMM 15%, `kv stage` 125 ms; 1 007 token: 3 685 ms, copie 27%, dequant 18%, GEMM 20%, `qsa select` 14% | il costo dei chunk piccoli sono gli esperti non residenti (PCIe) e la loro dequantizzazione, non l'attenzione |
| termine col contesto | ~0,5-0,7 s a 50-70k | le K/V oltre le 32 768 celle residenti stanno in RAM e la prefill le ricopia per layer QSA; `STRATA_KV_PREFETCH=1` (opt-in, stessi dati) le sovrappone al calcolo |
| checkpoint al confine di turno | uno per turno | `checkpoint_at(turn_at)` per il turno dopo |
| intestazione dell'assistente | 1 finestra | percorso a finestre |
| prima finestra di decodifica + MTP | ~40 ms | fuori da `prompt_ms` |

## Implementato in `lab/ttft` (output identico)

1. **File del prefisso scritto su un thread** (di default; `STRATA_PREFIX_DISK_SYNC=1` torna in linea). La richiesta
   aspetta solo la copia della sessione sull'host; SHA-256 e scrittura vanno su un thread. Una lettura del file aspetta
   la scrittura in corso; l'uscita del motore pure. Log: `saved ... (on a thread; the request waited X ms for the
   capture)`.
2. **Un file per prefisso** (`STRATA_PREFIX_DISK_MULTI=<GiB>`, opt-in): `<--prompt-cache-file>.<16 hex>`, chiave FNV-1a
   degli id del prefisso e del token di frontiera. Il file resta verificato token per token, quindi una collisione è
   solo un miss. Oltre il tetto escono i file usati meno di recente: un ripristino aggiorna la data del file. Senza un
   file con chiave si legge ancora il file unico. Test in `prefix_cache_file_test` (chiave, elenco, LRU, il file appena
   scritto non esce mai).
3. **Lettura anticipata delle righe PLE della parte dopo** (di default; `STRATA_PLE_LOOKAHEAD=0` la spegne):
   - una parte a chunk seguita da un'altra parte a chunk (system, poi primo messaggio) legge le righe del primo chunk
     della seconda appena la prima ha le righe del suo ultimo chunk (`Prefill::set_ple_lookahead`);
   - con `STRATA_PREFIX_DISK_MULTI`, quando esiste il file con chiave, le righe della parte dopo il prefisso si leggono
     mentre si legge il file (`Prefill::ple_warm_now`).
   Le righe vanno nella cache delle righe (`PleTable::warm_rows`, solo lettura diretta col thread del lettore, che è
   thread-safe), e il primo gather della run le trova lì: stessi byte, prima. La run aspetta la lettura in corso prima
   del suo primo gather, sul thread del gather. Le statistiche `ple io` contano anche le righe anticipate.

Le tre variabili sono escluse dall'identità del file del prefisso: accenderle o spegnerle non invalida i file salvati.

Stima (da misurare): Claude Code sessione nuova con file presente 16-20 s → 15-19 s; file di un altro client: 42 s →
16-20 s con `STRATA_PREFIX_DISK_MULTI` (dalla seconda volta); primo avvio assoluto di un client: 42 → ~36 s.

## Piano, per guadagno e rischio

| # | Cosa | Guadagno sul primo TTFT | Rischio | Stato |
|---|---|---|---|---|
| 1 | un file per prefisso, LRU con tetto | 42 → 16-20 s quando i client si alternano | basso | fatto, opt-in |
| 2 | scrittura del file su un thread; righe PLE della parte dopo in anticipo | −0,5 s; −4/−5 s al primo avvio di un client; −0,7 s col file | basso | fatto, di default |
| 3 | conversazioni su disco fra un riavvio e l'altro: a motore fermo da ~20 s si salva la sessione viva al suo ultimo confine di turno, con chiave come al punto 1; alla richiesta si cerca il file più lungo fra i confini di turno della richiesta (hash progressivo) | ripresa di 130k: 65-80 s → 3-6 s (113k: 2,3 s dal disco) | medio: 2 GB per 100k in int8, serve un tetto | da fare, al posto di #1937 (che non applica: dipende da #879, duplica `disk_read_ms`, secondo archivio accanto al nostro v2) |
| 4 | cache delle righe PLE salvata e ricaricata all'avvio (righe usate, ~60 MB per 600k righe, controllo a campione contro il file) | −3/−5 s sul primo messaggio (CLAUDE.md e memoria si ripetono) | basso-medio | da fare |
| 5 | riscaldamento all'avvio: ripristino in RAM del file di prefisso più recente prima di READY; una richiesta da 1 token per catturare i grafi | −0,7 s e −0,2/−0,5 s; avvio +1-2 s | basso | da misurare |
| 6 | #1852 (tokenizzazione dal prefisso comune, Python, id identici) | −55 ms a turno a 125k | basso, applica pulito | pronto da portare |
| 7 | turno tipico: A/B di `STRATA_KV_PREFETCH=1` (stessi dati), `STRATA_SPLIT_SMALL_OWN` (niente prestito per i prompt piccoli), `--short-read` più alto; poi il percorso degli esperti dei chunk < 1 024 (copie PCIe + dequant = 45-60% della GPU) | da misurare | basso | banco |
| 8 | `STRATA_PREFILL_CPU_SHARE=auto` col layer split; #1537 | da misurare | non bit per bit (KL ~0,004) / caso raro | dopo |

Non portare: #1455 (alternativo a #1852, si ferma a 64k), #1810 (spento col layer split), #1511 (non tocca il TTFT).

## Build

`48ce5f27` (codice; questo file aggiunge solo documentazione): `build_lock.py --tests`, rc 0. `strata.exe` sha256
`7fc01e09077570c42b2174b45c67f1573a41d5efd2d0a7414982d84afc03789e`, copiata con `build.json` in
`moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\build-ttft-48ce5f27\` (accanto
`source\serve`, `source\tools` e `strata-vision.exe` del candidato). `prefix_cache_file_test` non eseguito: è sotto
`STRATA_BUILD_CONVERSATION_TESTS`, che `build_lock.py` non accende. Compilato sì, installato no, provato no.

## Come misurare sul banco

- B-A-B con lo stesso exe: `STRATA_PREFIX_DISK_SYNC=1 STRATA_PLE_LOOKAHEAD=0` (come il candidato) contro i default.
- Prima richiesta di Claude Code da motore appena avviato, con `--prompt-cache-file` in una cartella vuota: righe
  `waited ... PLE rows`, `prefix disk cache: saved`, `prompt ... read in`.
- `STRATA_PREFIX_DISK_MULTI=8`: alternare due system prompt diversi; atteso `restored` a ogni richiesta dopo la prima di
  ciascuno.
