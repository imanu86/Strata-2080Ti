#!/bin/bash
# several arms under ONE bench.lock:  bash coda_bundle.sh <build dir> <prefix> <arm...>
D=/c/Users/imanu/source/repos/moe-aggressive-commit/docs/porto/strata_adattivo/corse_2080ti/20261006_due_schede
S=$(dirname "$(readlink -f "$0")")
P=$2
python $D/bench_lock.py DET-shortpf-$P-bundle -- env NOLOCK=1 bash $S/coda_shortpf.sh "$@"
