"""Run the existing TTFT fixture with explicit output/cache directories and real failures.

The original lab script supplies the public prompts, HTTP bodies and server configuration.
It is imported read-only; no output is written next to it. Run under the workstation's
benchmark lock, with one engine at a time. Use --plan short for seven requests and a restart.

Arguments after this wrapper's options are passed to ttft_bench.py, for example:
    --tag direct1 --exe PATH --server-env STRATA_PROMPT_ENCODER=1
    --args="--ple-io direct" --env "STRATA_PREFIX_DISK_MULTI=24 STRATA_CONV_DISK=10"
"""
import argparse
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path


SHORT_PLAN = [
    (1, "cc1_t1", "freddo", "cc", "cc1", 1),
    (1, "cc1_t2", "c", "cc", "cc1", 2),
    (1, "idle12", "pausa", "idle", "", 12),
    (1, "dsh1_t1", "b-freddo", "dsh", "dsh1", 1),
    (2, "cc3_t1", "a", "cc", "cc3", 1),
    (2, "dsh2_t1", "a", "dsh", "dsh2", 1),
    (2, "cc1_t3", "d", "cc", "cc1", 3),
    (2, "dsh1_t2", "d", "dsh", "dsh1", 2),
]


def load_legacy(path, arguments):
    spec = importlib.util.spec_from_file_location("strata_ttft_legacy", path)
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    try:
        sys.argv = [str(path), *arguments]
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    return module


def managed_server_class(legacy):
    class ManagedServer(legacy.Server):
        def __enter__(self):
            try:
                return super().__enter__()
            except BaseException:
                # A failed __enter__ does not receive the context manager's __exit__.
                # Only clean up a server started by this object, including its engine.
                if self.p is not None:
                    self.__exit__(*sys.exc_info())
                raise
    return ManagedServer


def memory_sample(server=None):
    import psutil
    vm = psutil.virtual_memory()
    sample = {"available_bytes": vm.available, "total_bytes": vm.total, "processes": []}
    if server is not None and server.p is not None:
        try:
            parent = psutil.Process(server.p.pid)
            for process in [parent, *parent.children(recursive=True)]:
                try:
                    info = process.memory_info()
                    sample["processes"].append({"pid": process.pid, "name": process.name(),
                        "working_set_bytes": info.rss, "virtual_bytes": info.vms,
                        "peak_working_set_bytes": getattr(info, "peak_wset", None)})
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
    return sample


def engine_text(output, life):
    return (output / f"engine{life}.log").read_text(encoding="utf-8", errors="replace")


def ple_startup(log, require_locked):
    lines = [line for line in log.splitlines() if "PLE table " in line and "(--ple-io ram)" in line]
    locked = any("PLE table locked in RAM (--ple-io ram)" in line for line in lines)
    if require_locked and not locked:
        raise RuntimeError("--ple-io ram did not report a table locked in RAM; arm is invalid")
    return {"locked": locked, "lines": lines}


def require_cache_hits(log, prefix_hits, conversation_hits):
    prefix = sum("prefix disk cache: restored" in line for line in log.splitlines())
    conversation = sum("conversation disk cache: hit" in line for line in log.splitlines())
    if prefix < prefix_hits or conversation < conversation_hits:
        raise RuntimeError(f"restart cache hits: prefix {prefix}/{prefix_hits}, "
                           f"conversation {conversation}/{conversation_hits}")
    return {"prefix": prefix, "conversation": conversation}


def run(legacy, options):
    a, out = legacy.a, legacy.OUT
    out.mkdir(exist_ok=False)
    prefix_dir = legacy.PREFIX_ROOT / a.tag
    prefix_dir.mkdir(parents=True, exist_ok=False)
    prefix_file = prefix_dir / "prefix.bin"
    fixture_path = legacy.H / "ttft_fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    plan = SHORT_PLAN if options.plan == "short" else legacy.PLAN
    build_file = Path(a.exe).parent / "build.json"
    result = {"tag": a.tag, "exe": a.exe, "exe_sha256": legacy.sha(a.exe),
        "build": json.loads(build_file.read_text()) if build_file.exists() else None,
        "serve": a.serve, "env": a.env, "args": a.args, "set_opt": a.set_opt,
        "drop_opt": a.drop_opt, "server_env": a.server_env, "plan": options.plan,
        "fixture_sha256": legacy.sha(fixture_path), "maxt": a.maxt, "idle": a.idle,
        "prefix_file": str(prefix_file), "request_timeout": options.request_timeout,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "lives": {}, "requests": []}
    server_class = managed_server_class(legacy)
    error = None
    try:
        for life in (1, 2):
            result["lives"][life] = {"memory_before_start": memory_sample()}
            with server_class(life, prefix_file) as server:
                current = result["lives"][life]
                current.update(startup_s=round(server.startup_s, 2), env=server.cfg.get("env"),
                               args=server.cfg["args"], memory_after_ready=memory_sample(server),
                               ple=ple_startup(engine_text(out, life), options.require_ram_locked))
                position = legacy.log_lines(life, 0)[1]
                for lf, name, case, client, session, turn in plan:
                    if lf != life:
                        continue
                    if client == "idle":
                        time.sleep(turn)
                        lines, position = legacy.log_lines(life, position)
                        current.setdefault("pauses", {})[name] = lines
                        continue
                    path = "/v1/messages" if client in ("cc", "side") else "/v1/chat/completions"
                    body = (legacy.cc_body(fixture, session, turn) if client == "cc" else
                            legacy.dsh_body(fixture, session, turn) if client == "dsh" else
                            legacy.side_body(fixture))
                    entry = {"life": life, "name": name, "case": case, "ok": False}
                    result["requests"].append(entry)
                    try:
                        ttft, wall, text, usage = legacy.stream(path, body, timeout=options.request_timeout)
                        if ttft is None or not text:
                            raise RuntimeError("response contained no text token")
                    except Exception as exc:
                        entry["error"] = str(exc)
                        raise RuntimeError(f"{name} failed: {exc}") from exc
                    time.sleep(1)
                    lines, position = legacy.log_lines(life, position)
                    (out / f"{name}.txt").write_text(text, encoding="utf-8")
                    entry.update(ok=True, ttft=round(ttft, 2), wall=round(wall, 2), usage=usage,
                                 engine=lines, memory=memory_sample(server))
                    print(f"{name}: TTFT {ttft:.2f} s, wall {wall:.2f} s", flush=True)
                    for line in lines:
                        if "waited" in line or "disk cache" in line:
                            print("  " + line, flush=True)
                    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                if life == 1:
                    time.sleep(a.idle)
                    current["idle_lines"] = legacy.log_lines(life, position)[0]
            result["lives"][life]["memory_after_stop"] = memory_sample()
        result["restart_cache_hits"] = require_cache_hits(engine_text(out, 2),
            options.require_prefix_hits, options.require_conversation_hits)
        if a.compare:
            other = legacy.H / f"ttft_{a.compare}"
            result["same_text_as"] = {entry["name"]:
                (other / f"{entry['name']}.txt").is_file() and
                (other / f"{entry['name']}.txt").read_text(encoding="utf-8") ==
                (out / f"{entry['name']}.txt").read_text(encoding="utf-8")
                for entry in result["requests"]}
            if options.require_identical and not all(result["same_text_as"].values()):
                raise RuntimeError("reply text differs from the comparison arm")
    except Exception as exc:
        error = str(exc)
    finally:
        result.update(ended=time.strftime("%Y-%m-%d %H:%M:%S"), ok=error is None, error=error)
        result["prefix_files"] = sorted((path.name, path.stat().st_size) for path in prefix_dir.iterdir())
        (out / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        rows = [f"{entry['name']}: TTFT {entry.get('ttft')} s, wall {entry.get('wall')} s, "
                f"ok {entry['ok']}" for entry in result["requests"]]
        (out / "summary.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    if error:
        print("TTFT benchmark failed: " + error, file=sys.stderr)
    return 1 if error else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--legacy-bench", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--prefix-root", required=True, type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--plan", choices=("full", "short"), default="short")
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--require-prefix-hits", type=int, default=0)
    parser.add_argument("--require-conversation-hits", type=int, default=0)
    parser.add_argument("--require-ram-locked", action="store_true")
    parser.add_argument("--require-identical", action="store_true")
    options, remaining = parser.parse_known_args(argv)
    if options.request_timeout <= 0 or min(options.require_prefix_hits, options.require_conversation_hits) < 0:
        parser.error("timeouts must be positive and minimum hit counts non-negative")
    legacy_path = options.legacy_bench.resolve(strict=True)
    legacy = load_legacy(legacy_path, remaining)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", legacy.a.tag):
        parser.error("--tag must be a plain directory name")
    if legacy.a.compare and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", legacy.a.compare):
        parser.error("--compare must be a plain directory name")
    if options.require_identical and not legacy.a.compare:
        parser.error("--require-identical needs --compare")
    output_root = options.output_root.resolve()
    if output_root == legacy_path.parent or legacy_path.parent in output_root.parents:
        parser.error("--output-root must be outside the original lab")
    output_root.mkdir(parents=True, exist_ok=True)
    fixture_source = (options.fixture or legacy_path.parent / "ttft_fixture.json").resolve(strict=True)
    fixture_target = output_root / "ttft_fixture.json"
    if fixture_target.exists():
        if legacy.sha(fixture_source) != legacy.sha(fixture_target):
            parser.error("existing output fixture differs from the requested fixture")
    else:
        shutil.copyfile(fixture_source, fixture_target)
    legacy.H, legacy.OUT = output_root, output_root / f"ttft_{legacy.a.tag}"
    legacy.PREFIX_ROOT = options.prefix_root.resolve()
    if legacy.a.only_fixture:
        print("fixture sha256 " + legacy.sha(fixture_target))
        return 0
    return run(legacy, options)


if __name__ == "__main__":
    sys.exit(main())
