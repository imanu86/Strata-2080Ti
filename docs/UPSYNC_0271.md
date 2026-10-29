# UPSYNC_0271 - la pila del fork sopra upstream v0.2.7.1 (29/10/2026)

Branch `lab/upsync-v0.2.7.1` da `v0.2.7.1` (46c152a1) + 18 patch (cherry-pick da `lab/upsync-026`, 19 meno la 07) + `stack/`.
**Stato: patch portate si (18), compilato si (sm_75+86, build_lock.py, 628 s, sha256 c1f6c27a...), installato no, provato solo al banco comunita' (3080 Ti).**

- Patch 07 (park-no-foreign-kv, PR #1503) gia' in upstream: saltata, spostata fra le ritirate di `stack/manifest.json`.
- Conflitti risolti a mano: 01 (expert_cache.hpp, due blocchi adiacenti), 05 (mtp.hpp `first_needed` con `cell_floor_` KVA + `full_prefix_`), 06 (NVML gia' in upstream `device_free_bytes`, tenuta la versione upstream e il `min` in `el_free`), 13 (`exec_` ora `[sl][T]`; aggiunti i distruttori STRUCT-A), 15 (`gather_raw_iq4nl` + `warm_rows`), 17 (`exec_[sl][T]`), 18 (gate KVA + hc16 su alpha/beta).
- Novita' di 0.2.7 "tipo REAP": `STRATA_ROUTE_PRIOR=1.5` di default su NVIDIA discrete <= 13 GiB (il router preferisce gli esperti gia' sulla scheda; non pota, cambia la scelta). Spegnere: `STRATA_ROUTE_PRIOR=0`.
- Misure: `moe-aggressive-commit/.../20261019_banco_comunita/RISULTATI.md` (sezione 29/10).
