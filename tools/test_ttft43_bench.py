"""CPU checks for TTFT failure reporting and engine cleanup, without launching an engine."""
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import ttft43_bench as bench


class BenchTest(unittest.TestCase):
    def test_ram_arm_requires_actual_lock(self):
        log = "strata generate: PLE table loaded (not locked) (--ple-io ram) in 4.0 s"
        with self.assertRaisesRegex(RuntimeError, "arm is invalid"):
            bench.ple_startup(log, True)
        self.assertFalse(bench.ple_startup(log, False)["locked"])
        log = "strata generate: PLE table locked in RAM (--ple-io ram) in 26.0 s"
        self.assertTrue(bench.ple_startup(log, True)["locked"])

    def test_restart_requires_requested_hits(self):
        log = "prefix disk cache: restored 2000\nconversation disk cache: hit 4000\n"
        self.assertEqual(bench.require_cache_hits(log, 1, 1), {"prefix": 1, "conversation": 1})
        with self.assertRaisesRegex(RuntimeError, "conversation 1/2"):
            bench.require_cache_hits(log, 1, 2)

    def test_startup_failure_cleans_up_started_server(self):
        class Server:
            p = object()
            stopped = False
            def __enter__(self):
                raise TimeoutError("startup")
            def __exit__(self, *args):
                self.stopped = True
        server = bench.managed_server_class(types.SimpleNamespace(Server=Server))()
        with self.assertRaises(TimeoutError):
            with server:
                self.fail("unreachable")
        self.assertTrue(server.stopped)

    def test_startup_failure_before_spawn_leaves_other_servers_alone(self):
        class Server:
            p = None
            def __enter__(self):
                raise OSError("port busy")
            def __exit__(self, *args):
                raise AssertionError("no process owned")
        server = bench.managed_server_class(types.SimpleNamespace(Server=Server))()
        with self.assertRaisesRegex(OSError, "port busy"):
            server.__enter__()

    def make_legacy(self, root, failure=None):
        legacy = types.SimpleNamespace(H=root, OUT=root / "ttft_test", PREFIX_ROOT=root / "cache")
        legacy.a = types.SimpleNamespace(tag="test", exe=str(root / "fake.exe"), serve="fake",
            env="", args="", set_opt=[], drop_opt=[], server_env="", maxt=24, idle=25, compare=None)
        (root / "ttft_fixture.json").write_text("{}", encoding="utf-8")
        legacy.sha = lambda path: "fixture-hash"
        legacy.cc_body = legacy.dsh_body = lambda *args: {"stream": True}
        legacy.side_body = lambda *args: {"stream": True}
        calls = []
        legacy.calls = calls
        def stream(*args, **kwargs):
            calls.append(kwargs["timeout"])
            if failure:
                raise failure
            return 1.0, 2.0, "reply", {"prompt_tokens": 40}
        legacy.stream = stream
        legacy.log_lines = lambda life, pos: ([], pos)
        legacy.stops = []
        class Server:
            def __init__(self, life, prefix_file):
                self.life, self.p, self.startup_s = life, None, 3.0
                self.cfg = {"env": {}, "args": []}
                (legacy.OUT / f"engine{life}.log").write_text(
                    "prefix disk cache: restored 100\nprefix disk cache: restored 200\n"
                    "conversation disk cache: hit 300\nconversation disk cache: hit 400\n",
                    encoding="utf-8")
                prefix_file.write_text("fake", encoding="utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                legacy.stops.append(self.life)
        legacy.Server = Server
        return legacy

    def run_legacy(self, legacy):
        options = types.SimpleNamespace(plan="short", request_timeout=0.25, require_ram_locked=False,
            require_prefix_hits=2, require_conversation_hits=2, require_identical=False)
        with patch.object(bench.time, "sleep"), patch.object(bench, "memory_sample", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bench.run(legacy, options)

    def test_success_runs_seven_requests_and_cleans_up_both_lives(self):
        with tempfile.TemporaryDirectory() as temp:
            legacy = self.make_legacy(Path(temp))
            self.assertEqual(self.run_legacy(legacy), 0)
            result = json.loads((legacy.OUT / "results.json").read_text())
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["requests"]), 7)
            self.assertEqual(legacy.calls, [0.25] * 7)
            self.assertEqual(legacy.stops, [1, 2])

    def test_timeout_fails_fast_and_preserves_result(self):
        with tempfile.TemporaryDirectory() as temp:
            legacy = self.make_legacy(Path(temp), TimeoutError("request timed out"))
            self.assertEqual(self.run_legacy(legacy), 1)
            result = json.loads((legacy.OUT / "results.json").read_text())
            self.assertFalse(result["ok"])
            self.assertIn("request timed out", result["error"])
            self.assertEqual(len(result["requests"]), 1)
            self.assertEqual(legacy.stops, [1])


if __name__ == "__main__":
    unittest.main()
