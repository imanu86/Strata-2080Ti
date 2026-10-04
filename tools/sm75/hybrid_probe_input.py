"""Convert one SNTv0001 trace into bounded, single-token hybrid-probe cases.

Requires numpy. Reads accepted positions only; no engine/GPU is started.
The three-full-expert comparison receives the largest current routing weights.
This is a warm kernel working-set comparison, not an online cache policy.
"""
import argparse
import struct
from pathlib import Path

import numpy as np


def convert(source, first, positions):
    if first < 0 or not 1 <= positions <= 30:
        raise ValueError('expected nonnegative first position and 1..30 positions')
    if source.stat().st_size > 512 * 1024 ** 2:
        raise ValueError('trace exceeds the 512 MiB input bound')
    data = source.read_bytes()
    if len(data) < 40 or data[:8] != b'SNTv0001' or struct.unpack_from('<4I4i', data, 8) != (2560, 640, 10, 4, 0, 1, 35, 36):
        raise ValueError('unsupported trace header')
    at = 40
    cases = {}
    windows = 0
    while at < len(data):
        if at + 24 > len(data):
            raise ValueError('truncated trace record')
        t, keep, pos, serial = struct.unpack_from('<IIqQ', data, at)
        at += 24
        windows += 1
        if not 1 <= keep <= t <= 8 or pos < 0 or serial == 0 or windows > 64:
            raise ValueError('invalid trace record')
        need = 4*t + 4*t*(2560*4 + 10*12 + 10*2560*4)
        if at + need > len(data):
            raise ValueError('truncated trace arrays')
        tokens = np.frombuffer(data, '<i4', t, at)
        at += 4*t
        for layer in (0, 1, 35, 36):
            x = np.frombuffer(data, '<f4', t*2560, at).reshape(t, 2560)
            at += t*2560*4
            ids = np.frombuffer(data, '<i4', t*10, at).reshape(t, 10)
            at += t*10*4
            weights = np.frombuffer(data, '<f4', t*10, at).reshape(t, 10)
            at += t*10*4
            at += t*10*2560*4  # Original expert output, not needed by this probe.
            tiers = np.frombuffer(data, '<i4', t*10, at).reshape(t, 10)
            at += t*10*4
            for row in range(keep):
                if not first <= pos + row < first + positions:
                    continue
                if (not np.isfinite(x[row]).all() or not np.isfinite(weights[row]).all()
                        or np.any(weights[row] < 0) or tokens[row] < 0
                        or len(set(ids[row])) != 10 or np.any((ids[row] < 0) | (ids[row] >= 512))
                        or np.any((tiers[row] < 0) | (tiers[row] > 2))):
                    raise ValueError('invalid selected activation or routing')
                key = (pos + row, layer)
                if key in cases:
                    raise ValueError('duplicate accepted position')
                order = np.argsort(-weights[row], kind='stable')
                cases[key] = (struct.pack('<3i', layer, pos+row, int(tokens[row])) + x[row].tobytes()
                              + ids[row, order].tobytes() + weights[row, order].tobytes() + tiers[row, order].tobytes())
    if len(cases) != 4*positions:
        raise ValueError('requested accepted positions are not all present')
    return b'SHBv0001' + struct.pack('<4I', len(cases), 2560, 10, 640) + b''.join(cases[key] for key in sorted(cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--first-position', type=int, required=True)
    parser.add_argument('--positions', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = convert(args.trace, args.first_position, args.positions)
    with args.output.open('xb') as stream:
        stream.write(data)
    print(f'SHB_INPUT_OK {args.positions * 4} cases')


if __name__ == '__main__':
    main()
