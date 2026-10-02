"""Check tested source bytes without CUDA, downloads or model loading."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    manifest = json.loads((ROOT/'docs/sm75/source-manifest.json').read_text(encoding='utf-8'))
    overrides = set(manifest['publication_overrides'])
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
    print(f'SOURCE_OK {checked} files; tested daily snapshot preserved')


if __name__ == '__main__':
    main()
