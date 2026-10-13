"""lab/pipeline2: host time inside cudaGraphLaunch per graph (by card and node count) of an nsys capture.
    python launch_cost.py <capture.sqlite>
"""
import sqlite3, sys, collections, statistics
c = sqlite3.connect(f'file:{sys.argv[1]}?mode=ro', uri=True)
R = {r[0]: (r[1], r[2]) for r in c.execute(
    "select r.correlationId, r.start, r.end from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId "
    "where s.value like 'cudaGraphLaunch%'")}
per = collections.defaultdict(lambda: [0, 0, None])   # corr -> kernels, device, first kernel start
for corr, dev, st in c.execute("select correlationId, deviceId, start from CUPTI_ACTIVITY_KIND_KERNEL"):
    if corr in R:
        p = per[corr]; p[0] += 1; p[1] = dev; p[2] = st if p[2] is None else min(p[2], st)
by = collections.defaultdict(list)
for corr, (n, dev, st) in per.items():
    a, b = R[corr]
    bucket = 'big' if n >= 300 else 'small'
    by[(dev, bucket)].append(((b - a) / 1000, n, (b - st) / 1000))
for (dev, bucket), v in sorted(by.items()):
    d = [x[0] for x in v]; n = [x[1] for x in v]; run = [x[2] for x in v]
    print(f'device {dev} {bucket:5} graphs {len(v):5}: launch call median {statistics.median(d):7.0f} us '
          f'(p10 {sorted(d)[len(d)//10]:.0f}, p90 {sorted(d)[9*len(d)//10]:.0f}), kernels median {statistics.median(n):.0f}, '
          f'{statistics.median(d)/max(1,statistics.median(n)):.2f} us per kernel node; first kernel starts '
          f'{statistics.median(run):.0f} us before the call returns')
