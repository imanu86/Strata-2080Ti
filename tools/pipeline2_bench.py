"""Run the existing 131k lab harness with its results outside the source repos.

Use bench_lock.py around EACH run of this script.  It imports run_nsys.main,
not run_nsys's lock-taking __main__, so the lab lock is acquired only once.
--dry-run prints the exact options without importing the GPU harness.

The same binary, fixed cache and Daily 0.1.43 environment serve both arms;
only STRATA_P2_FLUSH_NOSWITCH changes.  Free greedy runs check correctness;
forced IDs/windows are for timing comparisons, not numerical correctness.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

DAILY = (
    "STRATA_POOL_SPIN_US=100", "STRATA_VRAM_FREE=device", "STRATA_ROUTE_TAIL_SKIP=0",
    "STRATA_ATTN_DAILY=1", "STRATA_MMVQ_IL=0", "STRATA_B6_MMVQ_ROWS=1",
    "STRATA_MTP_SHARED_BRANCH=0", "STRATA_SIDE_MIXER=2", "STRATA_RAM_CORE=1",
    "STRATA_RAM_CORE_SLOTS_DEV1=5500", "STRATA_EXPDEC=1", "STRATA_QSA_ROWS=1",
    "STRATA_NODECUT=1", "STRATA_QFUSE_HC=0", "STRATA_TOPK_GRID=1",
    "STRATA_B6_HEAD_MINB=6", "STRATA_SA_BET=1", "STRATA_SA_CHAIN=1",
    "CUDA_VISIBLE_DEVICES=1,0",
)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def options(a):
    cases = a.cases.split(",")
    if not cases or any(c not in ("L01", "L02") for c in cases) or len(set(cases)) != len(cases):
        raise ValueError("cases must be L01, L02, or L01,L02")
    if not re.fullmatch(r"DET-[A-Za-z0-9_-]+", a.name):
        raise ValueError("use a new DET- name without path separators")
    if a.tokens < 1:
        raise ValueError("tokens must be positive")
    if a.force_ids and len(a.force_ids) != len(cases):
        raise ValueError("supply one --force-ids file per case, in request order")
    if a.force_windows and not a.force_ids:
        raise ValueError("--force-windows requires --force-ids")
    for p in [a.harness, a.build / "build.json", a.build / "strata.exe", *a.force_ids,
              *([a.force_windows] if a.force_windows else [])]:
        if not p.is_file():
            raise ValueError(f"missing input: {p}")
    lab_root = next((p for p in a.harness.parents if (p / ".git").exists()), a.harness.parent)
    source_root = Path(__file__).resolve().parents[1]
    if any(a.output_root == p or p in a.output_root.parents for p in (lab_root, source_root)):
        raise ValueError("output root must be outside the lab repository")
    if (a.output_root / a.name).exists():
        raise ValueError(f"output already exists: {a.output_root / a.name}")
    opts = ["--name", a.name, "--build", str(a.build), "--cases", a.cases,
            "--tokens", str(a.tokens), "--set-opt=--expert-cache=1500", "--drop-opt=--elastic",
            "--add-opt=--layer-split=16", "--add-opt=--trim-stage-weights",
            "--add-opt=--pipeline-windows=2", "--add-opt=--vram-reserve-later-mib=384",
            "--add-opt=--host-core=last", "--add-opt=--adapt-every=0"]
    env = [*DAILY, f"STRATA_P2_FLUSH_NOSWITCH={int(a.arm == 'B')}",
           f"STRATA_P2_SWITCH_STAT={int(a.switch_stat)}", "STRATA_P2_SMEM=0", "STRATA_P2_FLUSH_US=2000"]
    if a.force_ids:
        env += ["STRATA_FORCE_IDS=" + ";".join(str(p) for p in a.force_ids)]
    if a.force_windows:
        env += [f"STRATA_FORCE_WINDOWS={a.force_windows}"]
    # Trace both free and forced generations.  Unlike forced IDs, this changes no tokens.
    env += [f"STRATA_FORCE_TRACE={a.output_root / a.name / 'force_trace.txt'}"]
    for kv in env:
        opts.extend(("--env", kv))
    return opts


def run(a):
    opts = options(a)
    pin = json.loads((a.build / "build.json").read_text(encoding="utf-8"))
    if pin.get("status") != "completed" or digest(a.build / "strata.exe") != pin.get("executable_sha256"):
        raise ValueError("build metadata/hash does not match the executable")
    print(json.dumps(dict(output=str(a.output_root / a.name), executable_sha256=pin["executable_sha256"],
                          harness=str(a.harness), options=opts), indent=2), flush=True)
    if a.dry_run:
        return
    lock = a.locks / "bench.lock"
    if not lock.is_file() or lock.read_text(encoding="utf-8").strip() != a.name:
        raise ValueError("run this command under bench_lock.py using the SAME --name")
    # No bytecode may be created in the shared/dirty lab repo while loading its existing helpers.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("pipeline2_existing_harness", a.harness)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    original_root = harness.L
    a.output_root.mkdir(parents=True, exist_ok=True)
    baseline = a.output_root / "state-baseline.json"
    if not baseline.exists():
        baseline.write_bytes((original_root / "state-baseline.json").read_bytes())
    harness.L = a.output_root
    sys.argv = [str(a.harness), *opts]
    harness.main()


def load_arm(path):
    res = json.loads((path / "result.json").read_text(encoding="utf-8"))
    plan = json.loads((path / "plan.json").read_text(encoding="utf-8"))
    if res.get("status") != "completed" or res.get("daily_unchanged") is not True:
        raise ValueError(f"incomplete run or changed Daily: {path}")
    if res.get("close_error"):
        raise ValueError(f"engine cleanup failed: {path}: {res['close_error']}")
    rows = res.get("requests", [])
    cases = {r["case"]: r for r in rows}
    if len(cases) != len(rows) or set(cases) != set(plan["prompt_tokens"]):
        raise ValueError(f"missing/duplicate cases: {path}")
    for c, r in cases.items():
        last = r["last"]
        if len(r["generated_ids"]) != plan["tokens"] or last.get("generated") != plan["tokens"]:
            raise ValueError(f"missing tokens: {path}/{c}")
        if not last.get("decode_ms") or last["decode_ms"] <= 0:
            raise ValueError(f"no decode timing: {path}/{c}")
    log = (path / "engine.log").read_text(encoding="utf-8", errors="replace")
    if "poll without the device switch was refused" in log:
        raise ValueError(f"NOSWITCH fell back to device switching: {path}")
    trace = (path / "force_trace.txt").read_bytes()
    if not trace:
        raise ValueError(f"missing window trace: {path}")
    return res, plan, cases, trace, log


def compare(a):
    ref = load_arm(a.reference)
    report = []
    fixed_windows = bool(ref[1]["env"].get("STRATA_FORCE_WINDOWS"))
    def window_summary(trace):
        rows = collections.defaultdict(lambda: dict(windows=0, rows_histogram=collections.Counter(), accepted=0))
        for line in trace.decode("utf-8").splitlines():
            fields = line.split()
            if not fields or fields[0] != "W":
                continue
            req, width, accepted = int(fields[1]), int(fields[3]), int(fields[4])
            if width <= 0 or len(fields) != 5 + width:
                raise ValueError("invalid window trace")
            row = rows[req]
            row["windows"] += 1
            row["rows_histogram"][width] += 1
            row["accepted"] += accepted
        return dict(rows)
    # The output/trace paths and the one NOSWITCH setting are the only arm differences.
    def stable_env(plan):
        return {k: v for k, v in plan["env"].items()
                if k not in ("STRATA_FORCE_TRACE", "STRATA_P2_FLUSH_NOSWITCH")}
    for p in a.candidates:
        cur = load_arm(p)
        for key in ("executable_sha256", "config_sha256", "args", "prompt_tokens", "tokens"):
            if ref[1][key] != cur[1][key]:
                raise ValueError(f"different {key}: {p}")
        if stable_env(ref[1]) != stable_env(cur[1]):
            raise ValueError(f"other environment settings differ: {p}")
        same_ids = {c: ref[2][c]["generated_ids"] == cur[2][c]["generated_ids"] for c in ref[2]}
        same_trace = ref[3] == cur[3]
        # Greedy correctness concerns emitted IDs. Scheduling can change draft/window work without changing them.
        # A forced-window timing comparison must also repeat the reference's exact work.
        if not all(same_ids.values()) or (fixed_windows and not same_trace):
            raise ValueError(f"tokens/windows differ: {p}: ids {same_ids}, trace {same_trace}")
        timing = {}
        for c in ref[2]:
            ar, br = ref[2][c], cur[2][c]
            ams, bms = ar["last"]["decode_ms"], br["last"]["decode_ms"]
            timing[c] = dict(reference_tps=ar["decode_tps"], candidate_tps=br["decode_tps"],
                             decode_time_gain_percent=100 * (ams - bms) / ams)
        report.append(dict(candidate=str(p), ids_identical=same_ids, windows_identical=same_trace,
                           windows=window_summary(cur[3]), timing=timing,
                           pipeline_classes=[line for line in cur[4].splitlines() if line.startswith("strata pipeline classes")]))
    print(json.dumps(dict(reference=str(a.reference), comparisons=report,
                          forced=bool(ref[1]["env"].get("STRATA_FORCE_IDS")),
                          fixed_windows_required=fixed_windows, reference_windows=window_summary(ref[3]),
                          reference_pipeline_classes=[line for line in ref[4].splitlines() if line.startswith("strata pipeline classes")],
                          note="Forced output verifies equal work, not numerical correctness." if ref[1]["env"].get("STRATA_FORCE_IDS")
                               else "Free greedy IDs must match; scheduling/window differences are recorded. No float-logit parity claim."), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    rp = sub.add_parser("run")
    for key in ("harness", "build", "output-root"):
        rp.add_argument("--" + key, type=lambda s: Path(s).resolve(), required=True)
    rp.add_argument("--name", required=True)
    rp.add_argument("--arm", choices=("A", "B"), required=True)
    rp.add_argument("--cases", default="L01,L02")
    rp.add_argument("--tokens", type=int, default=1024)
    rp.add_argument("--force-ids", type=lambda s: Path(s).resolve(), action="append", default=[])
    rp.add_argument("--force-windows", type=lambda s: Path(s).resolve())
    rp.add_argument("--switch-stat", action="store_true", help="diagnostic only; leave off for B-A-B timing")
    rp.add_argument("--locks", type=Path, default=Path(r"D:\ds4_work\strata\lab_locks"))
    rp.add_argument("--dry-run", action="store_true")
    cp = sub.add_parser("compare")
    cp.add_argument("reference", type=Path)
    cp.add_argument("candidates", type=Path, nargs="+")
    a = ap.parse_args()
    try:
        run(a) if a.action == "run" else compare(a)
    except (ValueError, OSError, KeyError) as exc:
        ap.exit(1, f"pipeline2 bench: {exc}\n")


if __name__ == "__main__":
    main()
