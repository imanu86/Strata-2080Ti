#!/bin/bash
# Elastic-cache PR on upstream 0.1.39: (1) with --elastic off the PR exe is bit-identical to pure upstream
# (fixed-placement teacher forcing, code + explain); (2) 131k L01/L02 in the Daily configuration: B = PR exe with
# --elastic, then A = pure upstream (fixed cache).  --prompt-cache-tail (#614, not in upstream) is dropped everywhere.
H=/c/Users/imanu/source/repos/moe-aggressive-commit/docs/porto/strata_adattivo/corse_2080ti/20261005_handoff/script
D=/c/Users/imanu/source/repos/moe-aggressive-commit/docs/porto/strata_adattivo/corse_2080ti/20261004_routed_keep
bash "$H/tf_fixed.sh" "TFX-up0139|--build build-up0139 --drop-opt=--prompt-cache-tail" \
                      "TFX-epr-off|--build build-elasticpr --drop-opt=--prompt-cache-tail"
cd "$D" || exit 1
for t in code explain; do
  cmp -s TFX-epr-off-$t/logpos.tsv TFX-up0139-$t/logpos.tsv && echo "$t: PR senza --elastic identica bit per bit a upstream 0.1.39" \
                                                            || echo "$t: PR senza --elastic DIVERSA da upstream 0.1.39"
done | tee tf_fixed_elasticpr.txt
for spec in "LE-elastic|--build build-elasticpr --drop-opt=--prompt-cache-tail" \
            "LE-up0139|--build build-up0139 --drop-opt=--prompt-cache-tail --drop-opt=--elastic"; do
  n=${spec%%|*}; args=${spec#*|}
  python -c "
from run_keep import snapshot; import json,time
for _ in range(40):
    s=snapshot()
    if not s['competing'] and s['commit_free_gib']>=58: break
    time.sleep(15)
print(json.dumps(s)[:110])"
  timeout 3000 python -u run_long.py --name "$n" $args --env STRATA_ARENA_PIN_GIB=64 > "$n.runlog" 2>&1
  echo "$n rc=$?"
  grep -a "strata elastic\|expert cache [0-9]* slots" "$n/engine.log" | head -4 | cut -c1-200
  grep -a "serve: prompt [0-9]" "$n/engine.log" | cut -c1-200
done
