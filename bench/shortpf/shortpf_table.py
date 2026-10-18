"""Median TTFT per target size and short_read from shortpf_<tag>/results.json.
   python shortpf_table.py tag [tag...]   (tags of the xover plan: rows keyed by target and short_read)"""
import json, statistics, sys
from pathlib import Path
H = Path(r'C:\Users\imanu\source\repos\moe-aggressive-commit\docs\porto\strata_adattivo\corse_2080ti\20261006_due_schede')
rows = {}
for tag in sys.argv[1:]:
    r = json.loads((H / f'shortpf_{tag}' / 'results.json').read_text(encoding='utf-8'))
    for q in r['requests']:
        if q['target'] is None or not q['ok']:
            continue
        rows.setdefault((q['target'], q['short_read']), {}).setdefault(tag, []).append((q['ttft'], q['new_tokens']))
print(f'{"target":>6} {"sr":>5} ' + ' '.join(f'{t:>22}' for t in sys.argv[1:]))
for (n, sr), d in sorted(rows.items()):
    cells = []
    for t in sys.argv[1:]:
        v = d.get(t)
        cells.append('-' if not v else f'{statistics.median(x for x, _ in v):.2f}s n={len(v)} ({min(x for x,_ in v):.2f}-{max(x for x,_ in v):.2f})')
    print(f'{n:>6} {sr:>5} ' + ' '.join(f'{c:>22}' for c in cells))
