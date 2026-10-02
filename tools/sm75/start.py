"""Start the fork server using the selected local configuration and port."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def command(config):
    config = config.resolve(strict=True)
    cfg = json.loads(config.read_text(encoding='utf-8-sig'))
    port = int(cfg.get('port', 8100))
    if not 1 <= port <= 65535:
        raise ValueError('Port must be between 1 and 65535')
    host = cfg.get('host', '127.0.0.1')
    if host != '127.0.0.1':
        raise ValueError('This launcher requires loopback; use the server CLI for other addresses')
    return [sys.executable, str(ROOT/'serve/server.py'), '--engine', 'strata',
            '--config', str(config), '--host', host, '--port', str(port)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=ROOT/'profiles/local-sm75.json')
    args = command(ap.parse_args().config)
    os.chdir(ROOT)
    os.execv(sys.executable, args)


if __name__ == '__main__':
    main()
