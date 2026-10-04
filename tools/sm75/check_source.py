"""Check tested source bytes without CUDA, downloads or model loading."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Explicit manifest, relative to the repository root')
    args = parser.parse_args()
    current = (ROOT / args.manifest) if args.manifest else next(
        ROOT / 'docs/sm75' / name for name in (
            'neuron-trace-20261004.json', 'release-20261004.json', 'source-manifest.json')
        if (ROOT / 'docs/sm75' / name).is_file())
    manifest = json.loads(current.read_text(encoding='utf-8'))
    for name, expected in manifest.get('historical_manifests', {}).items():
        if hashlib.sha256((ROOT/'docs/sm75'/name).read_bytes()).hexdigest() != expected:
            raise SystemExit('Historical source manifest changed: '+name)
    if 'previous_manifest_sha256' in manifest:
        previous = ROOT/'docs/sm75'/manifest['previous_manifest']
        if hashlib.sha256(previous.read_bytes()).hexdigest() != manifest['previous_manifest_sha256']:
            raise SystemExit('Historical source manifest changed')
    overrides = set(manifest.get('publication_overrides', []))
    bad, checked = [], 0
    for name, expected in manifest['files'].items():
        if name in overrides:
            continue
        path = ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            bad.append(name)
        checked += 1
    for name, marker in [('src/kernels/cuda/qsa_prompt_attn.cu', 'STRATA_QSA_CH'),
                         ('src/kernels/cuda/qsa_select.cu', 'STRATA_SCORE_BPW')]:
        if marker in (ROOT/name).read_text(encoding='utf-8'):
            bad.append(name+' unexpected experiment')
    if bad:
        raise SystemExit('Source mismatch: '+', '.join(bad))
    print(f"SOURCE_OK {checked} files; current {manifest['snapshot_version']}; original provenance preserved")


if __name__ == '__main__':
    main()
