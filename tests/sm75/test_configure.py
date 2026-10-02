"""Portable config/launcher integration checks; no GPU or real weights."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('sm75_start', ROOT/'tools/sm75/start.py')
start = importlib.util.module_from_spec(spec)
spec.loader.exec_module(start)


class ConfigureTests(unittest.TestCase):
    def test_paths_port_and_preservation(self):
        with tempfile.TemporaryDirectory(prefix='strata paths ') as tmp:
            base = Path(tmp)
            pack = base/'pack with spaces'
            tokenizer = pack/'tokenizer'
            tokenizer.mkdir(parents=True)
            for name in ['vocab.json', 'merges.txt', 'token_type.json', 'chat_template.jinja']:
                (tokenizer/name).write_text('{}')
            mtp = base/'mtp'
            mtp.mkdir()
            for name in ['strata.exe', 'first.gguf', 'second.gguf', 'profile.bin']:
                (base/name).write_bytes(b'fixture only')
            out = base/'local config.json'
            args = [sys.executable, str(ROOT/'tools/sm75/configure.py'),
                    '--exe', str(base/'strata.exe'), '--pack', str(pack),
                    '--native', str(base/'first.gguf'), '--ple-gguf', str(base/'second.gguf'),
                    '--mtp', str(mtp), '--expert-profile', str(base/'profile.bin'),
                    '--port', '8123', '--out', str(out)]
            completed = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            original = out.read_bytes()
            cfg = json.loads(original)
            self.assertEqual(cfg['env'], {'STRATA_WATCHDOG_S': '240'})
            self.assertIn('--elastic', cfg['args'])
            self.assertEqual(cfg['args'][cfg['args'].index('--max-context')+1], '262144')
            command = start.command(out)
            self.assertEqual(command[command.index('--port')+1], '8123')
            self.assertEqual(command[command.index('--config')+1], str(out.resolve()))
            again = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(out.read_bytes(), original)
            bad = subprocess.run(args+['--context', '262145'], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            cfg['host'] = '0.0.0.0'
            out.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                start.command(out)


if __name__ == '__main__':
    unittest.main()
