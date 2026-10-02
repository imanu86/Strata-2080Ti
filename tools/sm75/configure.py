"""Generate local SM75 config from explicit paths; never read credentials."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exe', type=Path, default=ROOT/'engine/strata.exe')
    ap.add_argument('--pack', type=Path, required=True)
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--ple-gguf', type=Path, required=True)
    ap.add_argument('--mtp', type=Path, required=True)
    ap.add_argument('--expert-profile', type=Path, default=ROOT/'data/expert-profile.bin')
    ap.add_argument('--context', type=int, default=262144)
    ap.add_argument('--kv-resident', type=int, default=32768)
    ap.add_argument('--reserve-mib', type=int, default=1393)
    ap.add_argument('--port', type=int, default=8100)
    ap.add_argument('--out', type=Path, default=ROOT/'profiles/local-sm75.json')
    a = ap.parse_args()
    if not 8192 <= a.context <= 262144 or not 0 < a.kv_resident <= a.context:
        ap.error('context8192..262144; resident cells must fit context')
    if a.reserve_mib < 0 or not 1 <= a.port <= 65535:
        ap.error('nonnegative reserve, port1..65535')
    paths = {}
    for key in ['exe', 'pack', 'native', 'ple_gguf', 'mtp', 'expert_profile']:
        path = getattr(a, key).resolve(strict=True)
        is_dir = key in ('pack', 'mtp')
        if (is_dir and not path.is_dir()) or (not is_dir and not path.is_file()):
            ap.error(key+' has wrong file/directory type')
        paths[key] = path
    tokenizer = paths['pack']/'tokenizer'
    for name in ['vocab.json', 'merges.txt', 'token_type.json', 'chat_template.jinja']:
        if not (tokenizer/name).is_file():
            ap.error('Missing tokenizer: '+name)
    out = a.out.resolve()
    if out.exists():
        ap.error('Output exists; choose another path to preserve your config')
    cfg = {'exe': str(paths['exe']), 'args': [
        '--pack', str(paths['pack']), '--native', str(paths['native']),
        '--ple-gguf', str(paths['ple_gguf']), '--expert-profile', str(paths['expert_profile']),
        '--expert-cache', 'auto', '--prefill', 'auto', '--spec', '4', '--spec-min-p', '0.5',
        '--mtp', str(paths['mtp']), '--max-context', str(a.context), '--kv', 'int8',
        '--kv-resident', str(a.kv_resident), '--vram-reserve-mib', str(a.reserve_mib), '--elastic'],
        'cwd': str(ROOT), 'tokenizer': str(tokenizer), 'model_name': 'qwen3.8-flash-next-iq3_xxs',
        'log': str(ROOT/'logs/strata-sm75.log'), 'port': a.port, 'host': '127.0.0.1',
        'env': {'STRATA_WATCHDOG_S': '240'}}
    (ROOT/'logs').mkdir(exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2)+'\n', encoding='utf-8')
    print('CONFIG_WRITTEN', out)
    print('Loopback API; elastic cache; KV capacity', a.context, '; engine not started')


if __name__ == '__main__':
    main()
