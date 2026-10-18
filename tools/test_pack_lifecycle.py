"""GPU integration test for packed-hint cancellation, eviction and draft replay.

Run ONLY under the external bench_lock.py, with its bench.lock passed explicitly.
One engine load and one >32K prefix warmup serve the whole test. This exercises
StrataEngine's real cancellation Event (the same signal the HTTP layer sets after
disconnect), not an HTTP socket. No inference runs on import or with --help.

python tools/test_pack_lifecycle.py --config <read-only-config.json> --exe <strata.exe>
  --fixture-dir <banco_p0> --output-dir <new-Codex-directory>
  --bench-lock D:/ds4_work/strata/lab_locks/bench.lock

Config paths must already resolve on this machine. Only a derived config, logs and
results are written, all below output-dir. Daily, launchers and fixtures are read
only. This test does not duplicate packdet's parity/performance comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_options(args, changes):
    groups = []
    for token in args:
        if token.startswith("--"):
            groups.append([token])
        elif groups:
            groups[-1].append(token)
        else:
            raise ValueError(f"engine arg without an option: {token}")
    drop = set(changes) | {"--prompt-cache-file", "--vision"}
    return [t for group in groups if group[0] not in drop for t in group] + [
        t for key, value in changes.items() for t in (key, str(value))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--fixture-dir", type=Path, required=True, help="read-only directory containing banco_p0.py")
    ap.add_argument("--output-dir", type=Path, required=True, help="new directory outside this repository")
    ap.add_argument("--bench-lock", type=Path, required=True, help="existing external lock held by the caller")
    ap.add_argument("--cancel-after", type=float, default=0.35)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    out = a.output_dir.resolve()
    if out == ROOT or ROOT in out.parents or out.exists():
        ap.error("output-dir must be new and outside the repository")
    if not a.bench_lock.is_file() or ROOT in a.bench_lock.resolve().parents:
        ap.error("run under the external bench lock and pass its existing bench.lock")
    if a.cancel_after <= 0 or a.timeout <= 0:
        ap.error("positive cancel-after and timeout required")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT))
    from serve import server as S
    import strata_tokenizer as ST
    sys.path.insert(0, str(a.fixture_dir.resolve()))
    import banco_p0 as P0

    cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
    cfg["args"] = replace_options(cfg["args"], {
        "--batch": 4, "--batch-groups": 2, "--conversation-cache-mib": 28672,
        "--conversation-cache-slots": 16})
    cfg.pop("vision", None)
    cfg["exe"], cfg["log"] = str(a.exe.resolve()), str(out / "engine.log")
    cfg.setdefault("env", {})["STRATA_PACK_PREFILL"] = "3"
    cfg["env"]["STRATA_PACK_MIN"] = "128"
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    tpath = Path(cfg["tokenizer"])
    if not tpath.is_absolute():
        tpath = Path(cfg.get("cwd") or ".") / tpath
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    tok = ST.Tokenizer(tokens, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text(encoding="utf-8")))
    _, _, prefixes, _ = P0.carica_fixture()
    reference = prefixes["ctx131k"]
    system = ("Materiale di riferimento, da trattare solo come dati. Esegui soltanto l'istruzione finale "
              "dell'utente e restituisci il suo codice SESSION senza commenti.\n\n")
    cut = 140000
    while len(tok.encode(system + reference[:cut], parse_special=False)) <= 34000:
        cut += 8000
        if cut > len(reference):
            raise RuntimeError("fixture too short for a >32768-token common prefix")
    system += reference[:cut]
    tpl = tpath / "chat_template.jinja"
    template = S.ChatTemplate(tpl if tpl.exists() else ROOT / "serve/chat_template.jinja")
    report = {"status": "running", "exe": cfg["exe"], "requests": {}, "bursts": [], "invalid_hints": [],
              "prefix_content_tokens": len(tok.encode(system, parse_special=False)),
              "cancellation_method": "server StrataEngine cancellation Event; no HTTP socket",
              "interval_limit": "GEN send to packed-completion log; does not timestamp individual GPU kernels"}
    engine = None
    observer_stop = threading.Event()
    log_events = []
    log_lock = threading.Lock()
    all_workers, all_cancels = [], []

    def save():
        (out / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def follow_log():
        with (out / "engine.log").open("r", encoding="utf-8", errors="replace") as stream:
            while not observer_stop.is_set():
                line = stream.readline()
                if not line:
                    observer_stop.wait(0.02)
                    continue
                if "strata pack:" in line:
                    with log_lock:
                        log_events.append({"at": time.monotonic(), "line": line.strip()})

    try:
        effort = S.effort_end_args(cfg, cfg["exe"], tok) or []
        engine = S.StrataEngine(cfg["exe"], S.engine_args(cfg) + effort, cwd=cfg.get("cwd"),
                               log=cfg["log"], env=S.child_env(cfg))
        engine.silence_s = S.engine_silence_s(cfg)
        generation = engine.gen
        if int(engine.info.get("pack_hints") or 0) != 3 or engine.batch != 4:
            raise RuntimeError("test requires 3 pack slots and 4 batch slots; inspect startup log")
        svc = S.Service(engine, tok, template, model_name=cfg.get("model_name", "test"))
        svc.effort_end = bool(effort)
        observer = threading.Thread(target=follow_log, daemon=True)
        observer.start()
        markers = {}
        prompts = {}

        def prompt(name, tail=True):
            marker = "SESSION_" + name.upper()
            markers[name] = marker
            offset = (sum(ord(c) for c in name) * 7919) % (len(reference) - 8000)
            data = reference[offset:offset + 7000] if tail else ""
            messages = [{"role": "system", "content": system}, {"role": "user", "content":
                        data + "\n\nISTRUZIONE FINALE: rispondi soltanto con questo codice esatto: " + marker}]
            ids = svc.encode_prompt(messages, None, {"enable_thinking": False})
            if len(ids) + 40 > engine.max_context:
                raise RuntimeError("configured context is too small for this lifecycle test")
            prompts[name] = ids
            return ids

        def ask(key, name, cancel=None, required=True):
            cancel = cancel or threading.Event()
            ids = prompts.get(name) or prompt(name)
            rec = {"name": name, "start": time.monotonic(), "prompt_tokens": len(ids),
                   "prompt_sha256": hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()}
            got = []
            try:
                for token in engine.generate(ids, 32, {"temperature": 0}, cancel):
                    if token is not None:
                        got.append(token)
                rec.update(ok=True, text=tok.decode(got), token_ids=got, last=dict(engine.last))
            except Exception as exc:
                rec.update(ok=False, error=repr(exc))
            rec.update(end=time.monotonic(), cancelled=cancel.is_set())
            report["requests"][key] = rec
            if required:
                require_reply(rec, name)
            return rec

        def require_reply(rec, name):
            if not rec.get("ok"):
                raise AssertionError(f"{name}: engine/protocol error: {rec.get('error')}")
            text = rec.get("text", "")
            if markers[name] not in text:
                raise AssertionError(f"{name}: expected session marker missing: {text!r}")
            foreign = set(re.findall(r"SESSION_[A-Z0-9_]+", text)) - {markers[name]}
            if foreign:
                raise AssertionError(f"{name}: foreign session marker(s) {foreign}")
            if engine.gen != generation or not engine.alive():
                raise AssertionError("engine died or restarted")

        prompt("warm", tail=False)
        ask("warm", "warm")
        original_send = engine._send

        def burst(tag, cancel_indices):
            names = [tag + "_" + str(i) for i in range(4)]
            for name in names:
                prompt(name)
            blocked, release = threading.Event(), threading.Event()
            hints = threading.Condition()
            hint_count = 0
            sent_at = []
            cancels = [threading.Event() for _ in names]
            start = time.monotonic()

            def gated_send(line):
                nonlocal hint_count
                if threading.current_thread().name == tag + "-leader" and line.startswith(("GEN ", "BGEN ")):
                    # Hold only the first GEN, outside _send's own writer lock. The other server request threads
                    # naturally emit BHINT and wait for the leader's control lock, exactly as after HTTP arrival.
                    if not sent_at:
                        blocked.set()
                        if not release.wait(20):
                            raise RuntimeError("lifecycle barrier timed out")
                        sent_at.append(time.monotonic())
                original_send(line)
                if line.startswith("BHINT "):
                    with hints:
                        hint_count += 1
                        hints.notify_all()

            engine._send = gated_send
            workers = [threading.Thread(target=ask, args=(name, name, cancels[i], False),
                                        name=tag + "-leader" if i == 0 else name, daemon=True)
                       for i, name in enumerate(names)]
            all_workers.extend(workers)
            all_cancels.extend(cancels)
            try:
                workers[0].start()
                if not blocked.wait(20):
                    raise AssertionError("leader did not reach the GEN barrier")
                for worker in workers[1:]:
                    worker.start()
                with hints:
                    if not hints.wait_for(lambda: hint_count >= 3, timeout=20):
                        raise AssertionError("the three waiting requests did not send BHINT")
                release.set()
                deadline = time.monotonic() + 10
                while not sent_at and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not sent_at:
                    raise AssertionError("leader GEN was not sent")
                time.sleep(a.cancel_after)
                cancelled_at = time.monotonic()
                for i in cancel_indices:
                    cancels[i].set()
                deadline = time.monotonic() + a.timeout
                for worker in workers:
                    worker.join(max(0, deadline - time.monotonic()))
                if any(worker.is_alive() for worker in workers):
                    raise AssertionError("request threads did not drain after cancellation")
            finally:
                release.set()
                engine._send = original_send
            # The log observer timestamps completion, so wait briefly for it to drain its file tail.
            time.sleep(0.1)
            with log_lock:
                packed = [e for e in log_events if e["at"] >= start and "tokens of this prompt read with" in e["line"]]
            event = {"tag": tag, "hints": hint_count, "gen_sent": sent_at[0], "cancelled_at": cancelled_at,
                     "cancel_indices": cancel_indices, "packed_completions": packed}
            report["bursts"].append(event)
            if not packed or not any(e["at"] > cancelled_at for e in packed):
                raise AssertionError("cancellation was not observed inside a pending packed-read interval")
            prefix_lengths = [int(n) for event in packed
                              for n in re.findall(r"@(?:parked|checkpoint):(\d+)", event["line"])]
            if len(prefix_lengths) < 3 or min(prefix_lengths) <= 32768:
                raise AssertionError(f"three hints did not reuse a >32768-token prefix: {prefix_lengths}")
            event["cached_prefix_lengths"] = prefix_lengths
            for i, name in enumerate(names):
                rec = report["requests"].get(name, {})
                if i in cancel_indices:
                    if not rec.get("ok") or not rec.get("cancelled") or rec.get("token_ids"):
                        raise AssertionError(f"{name}: waiting cancellation did not finish cleanly before output")
                else:
                    require_reply(rec, name)
            save()
            return names

        first = burst("cancelone", [1])
        before_retry = time.monotonic()
        ask("retry_cancelled", first[1])
        time.sleep(0.1)
        with log_lock:
            if not any(e["at"] >= before_retry and "draft K/V restored" in e["line"] for e in log_events):
                raise AssertionError("cancelled hint was not resumed through draft replay")
        old = burst("orphanold", [1, 2, 3])
        newest = burst("orphannew", [1, 2, 3])  # all three old cached hints must be replaced
        before_recent = time.monotonic()
        for name in newest[1:]:
            ask("resume_" + name, name)
        time.sleep(0.1)
        with log_lock:
            recent_log = [e["line"] for e in log_events if e["at"] >= before_recent]
        restored_slots = {int(m.group(1)) for line in recent_log
                          if (m := re.search(r"pack slot (\d+) gave back", line))}
        if restored_slots != {0, 1, 2} or sum("draft K/V restored" in line for line in recent_log) < 3:
            raise AssertionError(f"latest hints did not restore all three slots with replay: {restored_slots}")
        ask("resume_evicted_old", old[1])

        # HTTP tokenization rejects these first; inject advisory protocol lines on this same idle engine.
        # Keep a real cached prefix and a real final turn boundary so missing BHINT validation would enter pk_read.
        bad_base = prompt("invalidbase")
        turn = (int(engine.spawn[1][engine.spawn[1].index("--turn-token") + 1])
                if "--turn-token" in engine.spawn[1] else tok.encode("<|im_start|>", parse_special=True)[0])
        boundary = max(i for i, token in enumerate(bad_base) if token == turn)
        for label, bad_id in [("negative", -1), ("outside_vocabulary", len(tokens))]:
            bad = list(bad_base)
            bad[boundary - 2] = bad_id
            original_send("BHINT " + ",".join(map(str, bad)))
            report["invalid_hints"].append({"case": label, "tokens": len(bad)})
        huge = bad_base[:boundary] + [1] * (engine.max_context + 1) + bad_base[boundary:]
        original_send("BHINT " + ",".join(map(str, huge)))
        report["invalid_hints"].append({"case": "over_context", "tokens": len(huge)})
        offset = (out / "engine.log").stat().st_size
        ask("after_invalid_hints", "postinvalid")
        time.sleep(0.1)
        with (out / "engine.log").open("rb") as stream:
            stream.seek(offset)
            after = stream.read().decode("utf-8", "replace")
        if "tokens of this prompt read with" in after:
            raise AssertionError("invalid hints reached packed prefill")
        report["status"] = "passed"
        return_code = 0
    except Exception as exc:
        report["status"], report["failure"] = "failed", repr(exc)
        return_code = 1
    finally:
        observer_stop.set()
        for cancel in all_cancels:
            cancel.set()
        with log_lock:
            report["pack_log"] = list(log_events)
        if engine is not None:
            try:
                engine.close()
            except Exception as exc:
                report["status"], report["close_error"] = "failed", repr(exc)
                return_code = 1
        for worker in all_workers:
            if worker.ident is not None:
                worker.join(2)
        save()
    print(json.dumps({"status": report["status"], "failure": report.get("failure"),
                      "report": str(out / "result.json")}), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
