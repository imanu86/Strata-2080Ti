"""lab/pipeline2 tappa 1 (docs/PIPELINE2.md): node census of the verify windows from an nsys capture.

    python tools/graph_census.py <capture.sqlite> [--min-nodes 300] [--top 40]

Needs `nsys profile --trace=cuda --cuda-graph-trace=node`: every graph node is a kernel row whose correlationId is
the cudaGraphLaunch's.  A launch with >= min-nodes kernels on one device is a stage window.  Per device (stage) it
prints the median window: nodes, kernel time, idle time (no kernel of the launch running: the union of the launch's
kernel intervals against its wall), and per kernel name the nodes per window, the median us per node and the idle
that precedes the node (the gap the node's dispatch costs on the critical path).  Under nsys the gaps are inflated
2-3x (HOSTGAP): read the proportions.
"""
import argparse, sqlite3, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('db')
ap.add_argument('--min-nodes', type=int, default=300)
ap.add_argument('--top', type=int, default=40)
a = ap.parse_args()
c = sqlite3.connect(f'file:{a.db}?mode=ro', uri=True)
gpus = {r[0]: r[1] for r in c.execute('select id, name from TARGET_INFO_GPU')}
names = {r[0]: r[1] for r in c.execute('select id, value from StringIds')}
launch_ids = {r[0] for r in c.execute(
    "select r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id = r.nameId "
    "where s.value like 'cudaGraphLaunch%'")}
rows = defaultdict(list)
for st, en, dev, corr, nm in c.execute(
        'select start, end, deviceId, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null'):
    if corr in launch_ids:
        rows[(dev, corr)].append((st, en, names.get(nm, str(nm))))

per_dev = defaultdict(list)
for (dev, corr), ks in rows.items():
    if len(ks) >= a.min_nodes:
        per_dev[dev].append(sorted(ks))

for dev, wins in sorted(per_dev.items()):
    sizes = statistics.median(len(w) for w in wins)
    # windows of the dominant size class (+-15%): the same T and graph
    wins = [w for w in wins if abs(len(w) - sizes) <= 0.15 * sizes]
    walls, kern, idle = [], [], []
    stat = defaultdict(lambda: [0, [], []])   # name -> [nodes, durations, preceding idle]
    for w in wins:
        t0, t1 = w[0][0], max(k[1] for k in w)
        busy_end = t0
        tot_idle = 0
        cnt = defaultdict(int)
        for st, en, nm in w:
            gap = max(0, st - busy_end)
            tot_idle += gap
            busy_end = max(busy_end, en)
            s = stat[nm]
            s[1].append(en - st)
            s[2].append(gap)
            cnt[nm] += 1
        for nm, n in cnt.items():
            stat[nm][0] += n
        walls.append(t1 - t0)
        kern.append(sum(en - st for st, en, _ in w))
        idle.append(tot_idle)
    nw = len(wins)
    print(f'\n== {gpus.get(dev, dev)} (device {dev}): {nw} windows of ~{int(sizes)} nodes; median wall '
          f'{statistics.median(walls) / 1e6:.2f} ms, kernel sum {statistics.median(kern) / 1e6:.2f} ms, idle '
          f'{statistics.median(idle) / 1e6:.2f} ms')
    print(f'{"kernel":<44} {"nodes/win":>9} {"us/node":>8} {"ms/win":>7} {"idle before, us/node":>21} {"idle ms/win":>11}')
    order = sorted(stat.items(), key=lambda kv: -(sum(kv[1][1]) + sum(kv[1][2])))
    for nm, (n, d, g) in order[:a.top]:
        print(f'{nm[:44]:<44} {n / nw:>9.1f} {statistics.median(d) / 1e3:>8.1f} {sum(d) / nw / 1e6:>7.3f} '
              f'{statistics.mean(g) / 1e3:>21.2f} {sum(g) / nw / 1e6:>11.3f}')
    small = [(nm, v) for nm, v in stat.items() if statistics.median(v[1]) < 8000]
    print(f'kernels under 8 us: {sum(v[0] for _, v in small) / nw:.0f} nodes/window, '
          f'{sum(sum(v[1]) for _, v in small) / nw / 1e6:.2f} ms kernel + {sum(sum(v[2]) for _, v in small) / nw / 1e6:.2f} ms idle before them')
