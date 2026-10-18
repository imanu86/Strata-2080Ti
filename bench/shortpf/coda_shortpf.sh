#!/bin/bash
# SHORTPF: one arm per engine life, each under bench.lock (bench_lock.py).
#   bash coda_shortpf.sh <build dir in 20261004_routed_keep> <prefix tag> <arm...>
# arms: sr64 (the build's default) sr0 sr256 sr1024 sr0t (sr0 + STRATA_PREFILL_TIMING=1) sr0cpu (sr0 + CPU share auto)
#       fixed-sr64 / fixed-sr0 (cache 1500, no --elastic, --adapt-every 0, for the text check)
D=/c/Users/imanu/source/repos/moe-aggressive-commit/docs/porto/strata_adattivo/corse_2080ti/20261006_due_schede
K=$D/../20261004_routed_keep
PY=/d/ds4_work/strata/Strata/.venv/Scripts/python.exe
B=$K/$1; P=$2; shift 2
# NOLOCK=1: the caller already holds bench.lock (coda_bundle.sh)
if [ -n "$NOLOCK" ]; then LOCKCMD="bash $D/nolock.sh"; else LOCKCMD="python $D/bench_lock.py"; fi
S=$(dirname "$(readlink -f "$0")")
run() {
  local tag=$1; shift
  while [ -e $D/shortpf_$P-$tag.runlog ]; do tag=${tag}b; done
  echo "== $P-$tag $(date +%H:%M)"
  $LOCKCMD DET-shortpf-$P-$tag -- $PY -B $S/shortpf_bench.py --tag $P-$tag --exe $B/strata.exe --serve $B/source/serve "$@" > $D/shortpf_$P-$tag.runlog 2>&1
  echo "$P-$tag rc=$? $(date +%H:%M)"
  grep -a -E "^c[0-9]_|same text|priority" $D/shortpf_$P-$tag.runlog | cut -c1-200
}
for arm in "$@"; do
  case $arm in
    xbase) run xbase ;;                                                            # short_read switched per request: 0 / 1024
    xt)    run xt --force-sr 0 --env "STRATA_PREFILL_TIMING=1" ;;                  # the prompt path only, with the timing lines
    xcpu)  run xcpu --force-sr 0 --env "STRATA_PREFILL_CPU_SHARE=auto" ;;
    xmtp)  run xmtp --force-sr 0 --env "STRATA_SPLIT_MTP_BATCH=1" ;;
    xsm256) run xsm256 --force-sr 0 --env "STRATA_PREFILL_STREAM_MIN=256" ;;
    xsm128) run xsm128 --force-sr 0 --env "STRATA_PREFILL_STREAM_MIN=128" ;;
    xcpusm) run xcpusm --force-sr 0 --env "STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_STREAM_MIN=256" ;;
    xlt)   run xlt --force-sr 0 --env "STRATA_PREFILL_LAYER_TIMES=1 STRATA_PREFILL_TIMING=1" ;;
    xcpu06) run xcpu06 --force-sr 0 --env "STRATA_PREFILL_CPU_SHARE=0.6" ;;
    xcpu08) run xcpu08 --force-sr 0 --env "STRATA_PREFILL_CPU_SHARE=0.8" ;;
    xtcpu) run xtcpu --force-sr 0 --env "STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_TIMING=1 STRATA_DBG_CPU_GATE=1" ;;
    fx0)   run fx0 --force-sr 0 --fixed ;;
    fxcpu) run fxcpu --force-sr 0 --fixed --env "STRATA_PREFILL_CPU_SHARE=auto" ;;
    z0)    run z0 --force-sr 0 ;;
    zs)    run zs --force-sr 0 --env "STRATA_SHORTPF=1" ;;
    zs10)  run zs10 --force-sr 0 --env "STRATA_SHORTPF=1 STRATA_SHORTPF_SHARE=1.0" ;;
    zlt)   run zlt --force-sr 0 --env "STRATA_SHORTPF=1 STRATA_PREFILL_LAYER_TIMES=1 STRATA_PREFILL_TIMING=1" ;;
    fz0)   run fz0 --force-sr 0 --fixed ;;
    fzs)   run fzs --force-sr 0 --fixed --env "STRATA_SHORTPF=1" ;;
    xhelp) run xhelp --force-sr 0 --env "STRATA_PREFILL_HELP=1" ;;
    xhelpsm) run xhelpsm --force-sr 0 --env "STRATA_PREFILL_HELP=1 STRATA_PREFILL_STREAM_MIN=256" ;;
    x0)    run x0 --force-sr 0 ;;                                                  # the prompt path, plain
    x*)    echo "arm $arm: add it to the case" ;;
    fixed-sr64) run fixed-sr64 --plan sizes --fixed ;;
    *) echo "unknown arm $arm" ;;
  esac
done
echo "== fine $P $(date +%H:%M)"
