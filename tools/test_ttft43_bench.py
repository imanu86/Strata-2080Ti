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


def sse(events):
    return b"".join(b"data: " + (event.encode() if isinstance(event, str) else
                    json.dumps(event).encode()) + b"\n\n" for event in events)


def messages(payload="reply", kind="text_delta", call_id="toolu_123"):
    block = {"type": "text", "text": ""} if kind == "text_delta" else {
        "type": "tool_use", "id": call_id, "name": "Read", "input": {}}
    field = "text" if kind == "text_delta" else "partial_json"
    return [{"type": "message_start", "message": {"usage": {"input_tokens": 40}}},
            {"type": "content_block_start", "index": 0, "content_block": block},
            {"type": "content_block_delta", "index": 0, "delta": {"type": kind, field: payload}},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 24}},
            {"type": "message_stop"}]


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
            env="", args="", set_opt=[], drop_opt=[], server_env="", maxt=24, idle=25, compare=None, port=8102)
        (root / "ttft_fixture.json").write_text("{}", encoding="utf-8")
        legacy.sha = lambda path: "fixture-hash"
        legacy.cc_body = legacy.dsh_body = lambda *args: {"stream": True}
        legacy.side_body = lambda *args: {"stream": True}
        calls = []
        legacy.calls = calls
        def urlopen(request, timeout):
            calls.append(timeout)
            if failure:
                raise failure
            events = (messages() if request.full_url.endswith("/v1/messages") else
                      [{"choices": [{"delta": {"content": "reply"}}]}, "[DONE]"])
            return io.BytesIO(sse(events))
        legacy.urlopen = urlopen
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
                patch.object(bench.urllib.request, "urlopen", side_effect=legacy.urlopen), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bench.run(legacy, options)

    def test_success_runs_eight_requests_and_cleans_up_both_lives(self):
        with tempfile.TemporaryDirectory() as temp:
            legacy = self.make_legacy(Path(temp))
            self.assertEqual(self.run_legacy(legacy), 0)
            result = json.loads((legacy.OUT / "results.json").read_text())
            self.assertTrue(result["ok"])
            self.assertEqual(len(result["requests"]), 8)
            self.assertEqual(legacy.calls, [0.25] * 8)
            self.assertEqual(legacy.stops, [1, 2])

    def test_short_plan_saves_eligible_dsh_turn_before_restart(self):
        dsh = [row for row in bench.SHORT_PLAN if row[4] == "dsh1"]
        self.assertEqual([(row[0], row[5]) for row in dsh], [(1, 1), (1, 2), (2, 3)])
        # Check eligibility using boundaries measured with the unchanged public fixture.
        root, first, second, minimum = 29296, 32340, 33881, 4096
        self.assertLess(first - root, minimum)
        self.assertGreaterEqual(second - root, minimum)

    def test_timeout_fails_fast_and_preserves_result(self):
        with tempfile.TemporaryDirectory() as temp:
            legacy = self.make_legacy(Path(temp), TimeoutError("request timed out"))
            self.assertEqual(self.run_legacy(legacy), 1)
            result = json.loads((legacy.OUT / "results.json").read_text())
            self.assertFalse(result["ok"])
            self.assertIn("request timed out", result["error"])
            self.assertEqual(len(result["requests"]), 1)
            self.assertEqual(legacy.stops, [1])

    def stream_events(self, root, events, path="/v1/messages"):
        legacy = types.SimpleNamespace(OUT=root, a=types.SimpleNamespace(port=8102))
        raw = sse(events)
        with patch.object(bench.urllib.request, "urlopen", return_value=io.BytesIO(raw)):
            result = bench.stream_output(legacy, path, {"stream": True}, 1, "response")
        self.assertEqual((root / "response.sse").read_bytes(), raw)
        return result

    def test_adapter_retains_text_usage_and_first_delta(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.stream_events(Path(temp), messages())
            self.assertEqual(result["text"], "reply")
            self.assertIsNotNone(result["ttft"])
            self.assertEqual(result["first_delta_type"], "text_delta")
            self.assertEqual(result["usage"], {"input_tokens": 40, "output_tokens": 24})

    def test_adapter_retains_tool_only_and_truncated_arguments(self):
        with tempfile.TemporaryDirectory() as temp:
            partial = '{"file_path":"/work/strata/src'
            result = self.stream_events(Path(temp), messages(partial, "input_json_delta"))
            self.assertEqual(result["text"], "")
            self.assertIsNotNone(result["ttft"])
            self.assertEqual(result["first_delta_type"], "input_json_delta")
            tool = bench.content_for_comparison(result)["tools"][0]
            self.assertEqual(tool, {"name": "Read", "input": {}, "partial_json": partial})

    def test_empty_start_stop_has_no_ttft(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.stream_events(Path(temp), [messages()[0], {"type": "message_stop"}])
            self.assertIsNone(result["ttft"])
            self.assertEqual(bench.content_for_comparison(result), {"text": "", "thinking": "", "tools": []})

    def test_empty_response_fails_benchmark_and_stops_server(self):
        with tempfile.TemporaryDirectory() as temp:
            legacy = self.make_legacy(Path(temp))
            legacy.urlopen = lambda *args, **kwargs: io.BytesIO(sse([messages()[0], {"type": "message_stop"}]))
            self.assertEqual(self.run_legacy(legacy), 1)
            result = json.loads((legacy.OUT / "results.json").read_text())
            self.assertIn("no output payload", result["error"])
            self.assertEqual(len(result["requests"]), 1)
            self.assertEqual(legacy.stops, [1])

    def test_sse_error_keeps_raw_and_error_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "SSE error"):
                self.stream_events(root, [{"type": "error", "error": {"message": "engine failed"}}])
            self.assertIn(b"engine failed", (root / "response.sse").read_bytes())
            result = json.loads((root / "response.output.json").read_text())
            self.assertIn("engine failed", result["error"])

    def test_interrupted_stream_does_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "completion marker"):
                self.stream_events(Path(temp), messages()[:-1])

    def test_openai_text_preserves_same_first_payload_definition(self):
        with tempfile.TemporaryDirectory() as temp:
            result = self.stream_events(Path(temp), [
                {"choices": [{"delta": {"content": "reply"}}]},
                {"choices": [], "usage": {"completion_tokens": 24}}, "[DONE]"], "/v1/chat/completions")
            self.assertEqual(result["text"], "reply")
            self.assertEqual(result["first_delta_type"], "content")
            self.assertIsNotNone(result["ttft"])

    def test_tool_comparison_ignores_ids_and_chunk_boundaries_but_checks_payload(self):
        first = {"text": "", "events": messages('{"x":1}', "input_json_delta", "first")}
        second_events = messages('{"x":', "input_json_delta", "second")
        second_events.insert(3, {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "input_json_delta", "partial_json": "1}"}})
        second = {"text": "", "events": second_events}
        self.assertEqual(bench.content_for_comparison(first), bench.content_for_comparison(second))
        second_events[3]["delta"]["partial_json"] = "2}"
        self.assertNotEqual(bench.content_for_comparison(first), bench.content_for_comparison(second))


if __name__ == "__main__":
    unittest.main()
