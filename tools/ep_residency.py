"""lab/pipeline2 (docs/PIPELINE2.md). Expert-parallel residency model from a routing trace (--dump-routing format: int layer, int k, k ids, k weights).

How much stage-1 expert traffic can the 3060 take for a given number of its VRAM bytes, and what does stage 0 lose?
usage: python tools/ep_residency.py <routing trace>.  Greedy by frequency within a stage (in-sample: hit rates are optimistic; what counts is the difference between scenarios).
"""
import struct, sys
from collections import Counter
import numpy as np

path = sys.argv[1]
blob = open(path, 'rb').read()
NL, NE = 48, 512
freq = np.zeros((NL, NE), np.int64)
off = 0
n_rec = 0
while off + 8 <= len(blob):
    layer, k = struct.unpack_from('<ii', blob, off)
    off += 8
    ids = struct.unpack_from('<%di' % k, blob, off)
    off += 8 * k
    for e in ids:
        freq[layer, e] += 1
    n_rec += 1
print(f'records {n_rec}, tokens ~{n_rec // NL}')

S0 = range(0, 16)
S1 = range(16, 48)
B0, B1 = 1.50, 1.87          # MB per expert blob, stage 0 / stage 1 (ROOFLINE §1a)
SL0, SL1 = 3754, 9149        # slots today (ROOFLINE §3)

# global greedy within a stage: the stage's slots go to the most frequent (layer, expert) pairs
def ranked(layers):
    pairs = [(freq[l, e], l, e) for l in layers for e in range(NE)]
    pairs.sort(reverse=True)
    return pairs

r0, r1 = ranked(S0), ranked(S1)
tot0 = sum(p[0] for p in r0)
tot1 = sum(p[0] for p in r1)
c0 = np.cumsum([p[0] for p in r0]) / tot0
c1 = np.cumsum([p[0] for p in r1]) / tot1
print(f'today: stage 0 hit {c0[SL0 - 1]:.4f} with {SL0} slots; stage 1 hit {c1[SL1 - 1]:.4f} with {SL1} slots')
for share in (0.1, 0.2, 0.3, 0.4):
    n = int(np.searchsorted(c1, share)) + 1
    print(f'  stage 1: top {n} pairs ({n * B1 / 1024:.2f} GB) carry {share:.0%} of its traffic')

print('\n(b1) the 3060 holds the stage-1 top-P pairs (helper); it drops its stage-0 tail by the same bytes;')
print('    the 2080 Ti frees those P pairs and takes the next stage-1 pairs')
print(f'{"P":>6} {"GB":>6} {"helper share f":>15} {"S0 hit":>8} {"S1 hit":>8} {"dCPU/lw S0":>11} {"dCPU/lw S1":>11}')
for P in (0, 100, 200, 400, 600, 800, 1200, 1600):
    gb = P * B1
    drop0 = int(round(gb / B0))
    h0 = c0[SL0 - 1 - drop0] if SL0 - 1 - drop0 >= 0 else 0.0
    f = c1[P - 1] if P > 0 else 0.0
    h1 = c1[SL1 - 1 + P] if SL1 - 1 + P < len(c1) else 1.0
    # CPU experts per layer-window: misses x ~15.5 distinct experts (T=2)... use lookups: 10 x T=2 per layer-window
    lw = 20
    d0 = (c0[SL0 - 1] - h0) * lw
    d1 = (c1[SL1 - 1] - h1) * lw
    print(f'{P:>6} {gb / 1024:>6.2f} {f:>15.3f} {h0:>8.4f} {h1:>8.4f} {d0:>+11.3f} {d1:>+11.3f}')

print('\n(b2) the 2080 Ti holds the stage-0 top-Q pairs (helper for stage 0); it drops its stage-1 tail by the same bytes')
print(f'{"Q":>6} {"GB":>6} {"helper share f":>15} {"S0 hit":>8} {"S1 hit":>8}')
for Q in (0, 100, 200, 400, 600, 800):
    gb = Q * B0
    drop1 = int(round(gb / B1))
    f = c0[Q - 1] if Q > 0 else 0.0
    h0 = c0[SL0 - 1 + Q] if SL0 - 1 + Q < len(c0) else 1.0
    h1 = c1[SL1 - 1 - drop1]
    print(f'{Q:>6} {gb / 1024:>6.2f} {f:>15.3f} {h0:>8.4f} {h1:>8.4f}')

# per-layer skew: traffic share of the top 10% / 20% experts of a layer
tops = []
for l in range(NL):
    s = np.sort(freq[l])[::-1]
    t = s.sum()
    tops.append((s[:51].sum() / t, s[:102].sum() / t, s[:256].sum() / t))
tops = np.array(tops)
m = tops.mean(0)
print(f'\nper layer, share of traffic of the top 10% / 20% / 50% experts: mean {m[0]:.3f} / {m[1]:.3f} / {m[2]:.3f}')
