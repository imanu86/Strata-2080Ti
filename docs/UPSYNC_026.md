# UPSYNC_026 - la pila del fork sopra upstream v0.2.6 (25/10/2026)

Branch `lab/upsync-026` del fork `imanu86/Strata-2080Ti`, da `v0.2.6` (commit 4b9e05a2, motore 0.1.51 in `CMakeLists.txt`) + 18 commit di patch
(cherry-pick da `lab/upsync-020`, uno per leva) + `stack/`. Metodo e comandi di build/banco: `docs/UPSYNC_020.md`.
Il braccio "pura" sta nel worktree `Strata-wt/upsync-026-pure` (branch `lab/upsync-026-pure`, = v0.2.6 senza modifiche).

**Stato: patch portate si (18 di 18, pushate), compilato si (sm_75+86, entrambe), installato no, provato no, misurato si (par. 4).**

## 1. Esito per patch (nessuna ritirata)

Nessun codice delle 18 e' arrivato in upstream fra 0.2.0-beta e 0.2.6 (cercati i nomi delle variabili e delle funzioni in `src/`, `include/`, `serve/`).
`--vram-elastic` resta una cosa diversa dal nostro `--elastic`; `STRATA_SELECT_SIMT` e `STRATA_TC_SM75` restano opt-in come in 0.2.0.
`check_anchors.py --check` contro v0.2.6 dava "rewrite" per 9 patch, ma a cascata (la 05 aggiunge file che servono a 14 e 16); con `cherry-pick`
(3-way) i conflitti veri sono **tre**:

| patch | esito su 0.2.6 |
|---|---|
| 01, 02, 03, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18 | applicate senza conflitti (3-way automatico dove l'ancora e' mossa) |
| 04 elastic-split-pipeline | **adattata**: il gate di `pl_adapt` usa ora `o.adapt_iv(rounds)` (intervallo di riscaldamento per scheda) al posto di `o.adapt_every`; tenuto `adapt_iv` piu' `!el_hold_tier` |
| 05 prefix-disk | **adattata**: upstream (#3493) dimensiona da solo la cache conversazioni dalla RAM libera (`conversation_cache_mib = -1`, helper `auto_conversation_cache_mib`); tenuti il blocco upstream e il nostro, `--prompt-cache-file` accanto a `--conversation-cache-mib` |
| 06 vram-free-nvml | codice pulito; **conflitto solo in `docs/DETAILS.md`** (riscritta): presa la versione upstream, il paragrafo su `STRATA_VRAM_FREE` resta solo nel commit della leva e in `stack/06-vram-free-nvml.patch` (riletto: non e' nel DETAILS.md nuovo) |

Da riguardare riga per riga alla prova (applicate in 3-way senza conflitto, nessun cambio di significato verificato): 12 (`expert_source`), 13, 17, 18.

## 2. Novita' di 0.2.1-0.2.6 che toccano il banco

- Cache conversazioni **accesa di default** in `--serve` (dimensionata dalla RAM libera): la config A (pura, di default) la usa; la Daily ha `--conversation-cache-mib 6144`.
- `nv_defaults.cpp`: nuova `apply_spec_adapt_default` solo per sm_86 e solo con `STRATA_SPEC_ADAPT_SM86=1` (opt-in): con 3060 + 2080 Ti non scatta.
- La tabella sm_86 di 0.2.0 resta ferma ("tutte le schede uguali"): le leve del Daily (B6_MMVQ_ROWS, ATTN_DAILY, QSA_ROWS, SIDE_MIXER) in A non ci sono.
- Opzioni della config 0.1.43: tutte presenti in 0.2.6 salvo `--elastic` e `--prompt-cache-file` (nostre, solo con la pila). Nessuna rinominata o rimossa.

## 3. Build e banco

- Build (`build_lock_noeng.py`, 4 job, priorita' bassa, 14-15 minuti l'una; nessuna GPU): A `build-up026-4b9e05a2` (sha256 ca941d04...), B `build-up026pila-ffe66174` (sha256 5d70573d...), in
  `moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261004_routed_keep\`. `strata-vision.exe` copiato dalla Daily 0.1.43 (non ricompilato), `source\` con serve, tools, strata_control, strata_cli.
- Config in `corse_2080ti\20261025_upsync026\`: `config-A-pura-default.json` (Daily senza `--elastic`, senza `--prompt-cache-file`, senza nessuna `STRATA_*`; resta `CUDA_VISIBLE_DEVICES=1,0`),
  `config-B-pila-daily.json` (Daily, senza le env inerti NODECUT/TOPK_GRID/B6_HEAD_MINB/QFUSE_HC, con `STRATA_HC16_SM75=1` al posto dell'hc16 che nella Daily 0.1.43 era di serie). `run_all.sh` fa Daily, A, B, Daily.
- Banco eseguito il 25/10 (18:00-18:53, bench.lock di un agente morto rimosso dal coordinatore), Jump Desktop e OktoCGControl aperti, `run_all.sh`.

## 4. Misure (mediane prompt tok/s | decode tok/s, 4K / 32K / 128K)

| config | 4K | 32K | 128K |
|---|---|---|---|
| Daily 0.1.43, corsa 1 | 494 / 39,2 | 1038 / 40,6 | 1172 / 40,1 |
| A: 0.2.6 pura, default due schede | 540 / 33,4 | 630 / 39,5 | 681 / 21,8 |
| B: 0.2.6 + pila, config Daily + `STRATA_HC16_SM75=1` | 507 / 34,9 | 998 / 33,0 | 1236 / 34,6 |
| Daily 0.1.43, corsa 2 | 481 / 37,9 | 966 / 36,7 | 1171 / 39,6 |

Il PC era circa a meta' della velocita' del 19/10 (Daily r2: 75,9/70,4/75,3 decode): VRAM gia' occupata dal desktop (B parte con 31 MiB liberi su CUDA1, A con 62, il Daily con 1,3 GB), paging WDDM.
Valgono solo i confronti fra righe di oggi. A e' sotto il Daily (prompt 32K/128K -40%, decode 128K instabile 18-50); B pareggia il prompt a 128K ma il decode e' -8/-15%. Da ripetere a VRAM libera; token e accettazione bozze non confrontati.
