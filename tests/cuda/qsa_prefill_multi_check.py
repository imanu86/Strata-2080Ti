#!/usr/bin/env python3
"""Public, bounded SM75 QSA multi-query parity check; dry-run unless --run is supplied."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

CASES = [
    {"name": "ctx32_nq9_random", "ctx": 32, "nq": 9, "pattern": 0, "active": 1},
    {"name": "ctx20001_nq7_random", "ctx": 20001, "nq": 7, "pattern": 0, "active": 1},
    {"name": "ctx20001_nq8_random", "ctx": 20001, "nq": 8, "pattern": 0, "active": 1},
    {"name": "ctx20001_nq9_random", "ctx": 20001, "nq": 9, "pattern": 0, "active": 1},
    {"name": "ctx130380_nq1007_random", "ctx": 130380, "nq": 1007, "pattern": 0, "active": 1},
    {"name": "ctx249322_nq1007_random", "ctx": 249322, "nq": 1007, "pattern": 0, "active": 1},
    {"name": "ctx130380_nq256_random_perf", "ctx": 130380, "nq": 256, "pattern": 0, "active": 1, "timing": True},
    {"name": "ctx249322_nq256_random_perf", "ctx": 249322, "nq": 256, "pattern": 0, "active": 1, "timing": True},
    {"name": "ctx32768_nq8192_random_cap32768", "ctx": 32768, "nq": 8192, "cap": 32768, "pattern": 0, "active": 1},
    {"name": "ctx130380_nq9_ties", "ctx": 130380, "nq": 9, "pattern": 1, "active": 1},
    {"name": "ctx249322_nq9_naninf", "ctx": 249322, "nq": 9, "pattern": 2, "active": 1},
    {"name": "ctx130380_nq9_active0", "ctx": 130380, "nq": 9, "pattern": 0, "active": 0},
]
REPS = 5
DEFAULT_PROCESS_TIMEOUT = 90
DEFAULT_DEADLINE = 600
DRIVER_LINE = re.compile(
    r"^context=(\d+) queries=(\d+) pattern=(\d+) active=(-?\d+) "
    r"scores_ms=([0-9]+(?:\.[0-9]+)?) topk_ms=([0-9]+(?:\.[0-9]+)?) "
    r"score_values=(\d+) id_values=(\d+)$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_driver_stdout(stdout: str) -> dict:
    for line in stdout.splitlines():
        match = DRIVER_LINE.fullmatch(line.strip())
        if match:
            keys = ("context", "queries", "pattern", "active", "scores_ms", "topk_ms",
                    "score_values", "id_values")
            values = dict(zip(keys, match.groups()))
            return {key: float(value) if key.endswith("_ms") else int(value)
                    for key, value in values.items()}
    raise ValueError("driver emitted no recognized metrics line")


def compare_files(left: Path, right: Path, chunk_size: int = 1 << 20,
                  deadline_at: float | None = None) -> dict:
    """Compare full binary dumps in bounded memory, hashing and counting differing bytes."""
    h_left, h_right = hashlib.sha256(), hashlib.sha256()
    size_left = size_right = offset = differing = 0
    first_difference = None
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            if deadline_at is not None and time.monotonic() >= deadline_at:
                raise TimeoutError("overall deadline exceeded during dump comparison")
            xa, xb = a.read(chunk_size), b.read(chunk_size)
            if not xa and not xb:
                break
            h_left.update(xa); h_right.update(xb)
            size_left += len(xa); size_right += len(xb)
            common = min(len(xa), len(xb))
            if xa[:common] != xb[:common]:
                for index, (va, vb) in enumerate(zip(xa[:common], xb[:common])):
                    if va != vb:
                        differing += 1
                        if first_difference is None:
                            first_difference = offset + index
            if len(xa) != len(xb):
                differing += abs(len(xa) - len(xb))
                if first_difference is None:
                    first_difference = offset + common
            offset += max(len(xa), len(xb))
    if deadline_at is not None and time.monotonic() >= deadline_at:
        raise TimeoutError("overall deadline exceeded after dump comparison")
    return {"sha256_left": h_left.hexdigest(), "sha256_right": h_right.hexdigest(),
            "bytes_left": size_left, "bytes_right": size_right,
            "differing_byte_count": differing, "first_differing_byte": first_difference,
            "byte_identical": size_left == size_right and differing == 0}


def case_arms(case: dict) -> tuple[str, ...]:
    return ("A1", "B", "A2") if case.get("timing") else ("A", "B")


def finalize_report(report: dict, deadline_at: float | None = None) -> dict:
    observed_cases = report.get("cases", [])
    shape_ok = (len(observed_cases) == len(CASES) and
                [entry.get("case", {}).get("name") for entry in observed_cases] ==
                [case["name"] for case in CASES])
    if shape_ok:
        for fixture, observed in zip(CASES, observed_cases):
            expected_arms = case_arms(fixture)
            expected_pairs = (("A1", "B"), ("B", "A2")) if fixture.get("timing") else (("A", "B"),)
            actual_pairs = tuple((item.get("left"), item.get("right"))
                                 for item in observed.get("comparisons", []))
            if (tuple(item.get("arm") for item in observed.get("runs", [])) != expected_arms
                    or actual_pairs != expected_pairs):
                shape_ok = False
                break
    comparisons = [entry for case in observed_cases for entry in case.get("comparisons", [])]
    runs = [entry for case in observed_cases for entry in case.get("runs", [])]
    deadline_ok = deadline_at is None or time.monotonic() < deadline_at
    ok = ("error" not in report and shape_ok and bool(comparisons)
          and all(run.get("exit_code") == 0 and run.get("timed_out") is False for run in runs)
          and all(item.get("byte_identical") is True for item in comparisons)
          and report.get("exe_unchanged") is True and deadline_ok)
    report["pass_all"] = ok
    report["status"] = "passed" if ok else "failed"
    return report


def cpu_contract_check() -> dict:
    stdout = ("context=130380 queries=1007 pattern=0 active=32596 scores_ms=0.123456 "
              "topk_ms=0.234567 score_values=32775000 id_values=263000000\n")
    parsed = parse_driver_stdout(stdout)
    if parsed["context"] != 130380 or parsed["queries"] != 1007 or parsed["scores_ms"] != 0.123456:
        raise AssertionError("driver stdout parser fixture failed")
    with tempfile.TemporaryDirectory(prefix="qsa-public-cpu-") as temp:
        root = Path(temp)
        first, equal, different, shorter = (root / name for name in ("a.bin", "b.bin", "c.bin", "d.bin"))
        first.write_bytes(b"public-fixture-" * 2048)
        equal.write_bytes(first.read_bytes())
        altered = bytearray(first.read_bytes()); altered[777] ^= 1; different.write_bytes(altered)
        shorter.write_bytes(b"short")
        same = compare_files(first, equal)
        diff = compare_files(first, different)
        size_diff = compare_files(first, shorter)
        if not same["byte_identical"] or diff["first_differing_byte"] != 777 or size_diff["byte_identical"]:
            raise AssertionError("streaming byte comparator fixture failed")
        def synthetic_report(last_comparisons=None):
            cases = []
            for index, fixture in enumerate(CASES):
                arms = case_arms(fixture)
                pairs = (("A1", "B"), ("B", "A2")) if fixture.get("timing") else (("A", "B"),)
                comparisons = [dict(left=left, right=right, **same) for left, right in pairs]
                if index == len(CASES) - 1 and last_comparisons is not None:
                    comparisons = last_comparisons
                cases.append({"case": fixture,
                              "runs": [{"arm": arm, "exit_code": 0, "timed_out": False} for arm in arms],
                              "comparisons": comparisons})
            return {"cases": cases, "exe_unchanged": True}

        good = finalize_report(synthetic_report())
        bad = finalize_report(synthetic_report([{"left": "A", "right": "B", **diff}]))
        missing_last_compare = finalize_report(synthetic_report([]))
        partial_last_run = synthetic_report()
        partial_last_run["cases"][-1]["runs"].pop()
        partial_last_run = finalize_report(partial_last_run)
        errored_last = synthetic_report()
        errored_last["error"] = "injected last-case error"
        errored_last = finalize_report(errored_last)
        if (good["status"] != "passed" or bad["status"] != "failed"
                or missing_last_compare["status"] != "failed"
                or partial_last_run["status"] != "failed" or errored_last["status"] != "failed"):
            raise AssertionError("machine-readable PASS/FAIL fixture failed")
    return {"stdout_parser": "passed", "equal_dump": True, "mismatch_offset": 777,
            "size_mismatch": True, "report_status_pass_fail": True}


def make_plan(exe: Path) -> list[dict]:
    plan = []
    for case in CASES:
        cap = case.get("cap", 262144)
        if case["ctx"] > cap or case["nq"] > case["ctx"]:
            raise ValueError(f"invalid bounded test dimensions for {case['name']}")
        for arm in case_arms(case):
            plan.append({"case": case["name"], "arm": arm,
                         "env_STRATA_QSA_PREFILL_MULTI": "1" if arm == "B" else "0",
                         "argv_template": [str(exe), str(case["ctx"]), str(case["nq"]), str(REPS), str(cap),
                                           "<temporary-dump.bin>", str(case["pattern"]), str(case["active"])]})
    return plan


def write_json(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory for plan/logs/results")
    parser.add_argument("--run", action="store_true", help="explicitly run the SM75 GPU driver")
    parser.add_argument("--process-timeout", type=int, default=DEFAULT_PROCESS_TIMEOUT)
    parser.add_argument("--deadline", type=int, default=DEFAULT_DEADLINE)
    args = parser.parse_args()
    if args.process_timeout < 1 or args.deadline < args.process_timeout:
        parser.error("require process-timeout >= 1 and deadline >= process-timeout")
    exe, out = args.exe.resolve(), args.out.resolve()
    plan = make_plan(exe)
    cpu_check = cpu_contract_check()
    print(json.dumps({"mode": "RUN" if args.run else "DRY_RUN", "executable": str(exe),
                      "output": str(out), "case_count": len(CASES), "process_count": len(plan),
                      "process_timeout_seconds": args.process_timeout, "deadline_seconds": args.deadline,
                      "cpu_contract_check": cpu_check, "plan": plan}, indent=2))
    if not args.run:
        return 0
    if not exe.is_file():
        raise SystemExit("--run requires an existing executable")
    if out.exists() or not out.parent.is_dir():
        raise SystemExit("--out must name a new directory whose parent already exists")
    out.mkdir()
    exe_hash_before = sha256_file(exe)
    report = {"status": "running", "pass_all": None, "started_at": time.time(),
              "executable": str(exe), "exe_sha256_before": exe_hash_before,
              "process_timeout_seconds": args.process_timeout, "deadline_seconds": args.deadline,
              "repetitions": REPS, "cases": []}
    write_json(out / "plan.json", {"cases": CASES, "plan": plan, "cpu_contract_check": cpu_check})
    deadline_at = time.monotonic() + args.deadline
    owned_dumps = set()
    try:
        for case in CASES:
            case_report = {"case": case, "runs": [], "comparisons": []}
            dumps = {}
            for arm in case_arms(case):
                remaining = deadline_at - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("overall deadline exceeded")
                env = {key: value for key, value in os.environ.items() if not key.upper().startswith("STRATA_")}
                env["STRATA_QSA_PREFILL_MULTI"] = "1" if arm == "B" else "0"
                started = time.monotonic()
                with tempfile.TemporaryDirectory(prefix="qsa-dump-", dir=out) as temp_dir:
                    temp_root = Path(temp_dir).resolve()
                    if temp_root.parent != out:
                        raise RuntimeError("temporary dump directory escaped output directory")
                    dump = temp_root / "output.bin"
                    argv = [str(exe), str(case["ctx"]), str(case["nq"]), str(REPS),
                            str(case.get("cap", 262144)), str(dump), str(case["pattern"]), str(case["active"])]
                    try:
                        proc = subprocess.run(argv, env=env, capture_output=True, text=True,
                                              timeout=min(args.process_timeout, remaining), check=False)
                        stdout, stderr, code, timed_out = proc.stdout, proc.stderr, proc.returncode, False
                    except subprocess.TimeoutExpired as exc:
                        stdout, stderr = exc.stdout or "", exc.stderr or ""
                        if isinstance(stdout, bytes): stdout = stdout.decode(errors="replace")
                        if isinstance(stderr, bytes): stderr = stderr.decode(errors="replace")
                        code, timed_out = None, True
                    elapsed = time.monotonic() - started
                    (out / f"{case['name']}.{arm}.stdout.txt").write_text(stdout, encoding="utf-8")
                    (out / f"{case['name']}.{arm}.stderr.txt").write_text(stderr, encoding="utf-8")
                    run = {"arm": arm, "exit_code": code, "timed_out": timed_out,
                           "elapsed_seconds": elapsed, "stdout": stdout, "stderr": stderr}
                    if dump.is_file():
                        run["dump_sha256"] = sha256_file(dump)
                    case_report["runs"].append(run)
                    if timed_out or code != 0:
                        report["cases"].append(case_report)
                        raise RuntimeError(f"driver run failed for {case['name']} arm {arm}")
                    metrics = parse_driver_stdout(stdout)
                    expected = {"context": case["ctx"], "queries": case["nq"], "pattern": case["pattern"]}
                    if any(metrics[key] != value for key, value in expected.items()):
                        report["cases"].append(case_report)
                        raise RuntimeError(f"driver metrics do not match fixture: {metrics}")
                    run["metrics"] = metrics
                    # Move the completed dump into another temp directory only within out
                    # until the corresponding pair has been compared.
                    keep = out / f".{case['name']}.{arm}.compare.tmp"
                    if keep.exists():
                        raise RuntimeError("unexpected temporary dump collision")
                    dump.replace(keep)
                    dumps[arm] = keep
                    owned_dumps.add(keep)
                if time.monotonic() >= deadline_at:
                    raise TimeoutError("overall deadline exceeded")
            pairs = [("A1", "B"), ("B", "A2")] if case.get("timing") else [("A", "B")]
            for left, right in pairs:
                if time.monotonic() >= deadline_at:
                    raise TimeoutError("overall deadline exceeded before dump comparison")
                case_report["comparisons"].append({"left": left, "right": right,
                                                    **compare_files(dumps[left], dumps[right], deadline_at=deadline_at)})
                if time.monotonic() >= deadline_at:
                    raise TimeoutError("overall deadline exceeded after dump comparison")
            for dump in dumps.values():
                if dump.parent != out or not dump.name.endswith(".compare.tmp"):
                    raise RuntimeError("refusing to remove dump outside this run's output directory")
                dump.unlink()
                owned_dumps.discard(dump)
            report["cases"].append(case_report)
            write_json(out / "progress.json", report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["status"] = "failed"
        report["pass_all"] = False
    finally:
        cleanup_errors = []
        for dump in tuple(owned_dumps):
            if dump.parent != out or not dump.name.startswith(".") or not dump.name.endswith(".compare.tmp"):
                cleanup_errors.append(f"refused unsafe cleanup path: {dump}")
                continue
            try:
                dump.unlink(missing_ok=True)
                owned_dumps.discard(dump)
            except OSError as exc:
                cleanup_errors.append(f"{dump}: {exc}")
        if cleanup_errors:
            report["cleanup_errors"] = cleanup_errors
            report["error"] = report.get("error", "") + " temporary dump cleanup failed"
    try:
        report["exe_sha256_after"] = sha256_file(exe)
        report["exe_unchanged"] = report["exe_sha256_after"] == exe_hash_before
    except OSError as exc:
        report["exe_sha256_after_error"] = str(exc)
        report["exe_unchanged"] = False
    report["finished_at"] = time.time()
    if time.monotonic() >= deadline_at:
        report["error"] = report.get("error", "") + " overall deadline exceeded before final PASS check"
    finalize_report(report, deadline_at=deadline_at)
    write_json(out / "results.json", report)
    return 0 if report["pass_all"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
