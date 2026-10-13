"""lab/pipeline2 tappa 1: per graph exec of a capture made with --cuda-graph-trace=graph (no per-node instrumentation):
how long each graph runs on its card, how long the cudaGraphLaunch call takes and where the graph starts in it.
    python graph_wall.py <capture.sqlite> [min graphs per exec, default 20]
"""
import sqlite3, sys, collections, statistics as st
c = sqlite3.connect(f'file:{sys.argv[1]}?mode=ro', uri=True)
R = {r[0]: (r[1], r[2]) for r in c.execute(
    "select r.correlationId, r.start, r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId "
    "where s.value like 'cudaGraphLaunch%'")}
by = collections.defaultdict(list)
for s0, e0, dev, corr, ex in c.execute(
        "select start, end, deviceId, correlationId, graphExecId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
    if corr in R:
        a, b = R[corr]
        by[(dev, ex)].append(((e0 - s0) / 1e3, (b - a) / 1e3, (s0 - a) / 1e3, (b - s0) / 1e3))
med = lambda v: st.median(v) if v else 0.0
print('device exec   n   graph run us (p10/med/p90)   launch call us (med)   graph starts after call start us (med)')
for (dev, ex), v in sorted(by.items(), key=lambda kv: (kv[0][0], -med([x[0] for x in kv[1]]))):
    if len(v) < int(sys.argv[2] if len(sys.argv) > 2 else 20):
        continue
    run = sorted(x[0] for x in v)
    print(f'{dev:6} {ex:5} {len(v):4}   {run[len(run)//10]:8.0f} {med(run):8.0f} {run[9*len(run)//10]:8.0f}   '
          f'{med([x[1] for x in v]):10.0f}   {med([x[2] for x in v]):10.0f}')
# the window graphs pooled per card (graphs running over 5 ms: the stage windows, not the MTP chain or the commits)
for dev in sorted({k[0] for k in by}):
    v = [x for (d, _), xs in by.items() if d == dev for x in xs if x[0] > 5000]
    if not v:
        continue
    run = sorted(x[0] for x in v)
    print(f'device {dev}: {len(v)} window graphs, run median {med(run) / 1e3:.2f} ms (p10 {run[len(run)//10] / 1e3:.2f}, '
          f'p90 {run[9*len(run)//10] / 1e3:.2f}), cudaGraphLaunch median {med([x[1] for x in v]):.0f} us')
