"""SHORTPF bench: the time to the first token of a SHORT READ (a tool result of 100 ... 8000 new tokens on a context
already in cache), on the owner's Daily config (read only), one engine life per arm.

Three conversations (/v1/chat/completions), each with its own ~12k-token system text, are driven turn by turn the way a
client does: the reply the engine gave is sent back as the assistant message, so the conversation cache resumes the
context and only the new user message (plus the previous reply and the template, ~40 tokens) is read.
  A  sizes ascending      100 300 1000 3000 8000
  B  sizes descending     8000 3000 1000 300 100
  C  sizes ascending      (a third sample per size)
Per request: TTFT (send -> first content delta, server template and tokenization included), wall, prompt/cached tokens
from the usage, and the engine's own lines for it ("strata serve: prompt ... read in ...", PREFILL_TIMING lines).
Text check: every reply is stored; --compare <tag> says whether the replies are identical to another arm's.

    python bench_lock.py DET-shortpf-<tag> -- python shortpf_bench.py --tag <tag> --exe <strata.exe> --serve <serve dir>
        [--env "K=V K2=V2"] [--args "--short-read 256"] [--fixed] [--compare <tag>]
--fixed: the text-check config (--expert-cache 1500, no --elastic, --adapt-every 0).
Outputs (data): shortpf_<tag>/  results.json, summary.txt, engine.log, server.log, *.txt
"""
import argparse, hashlib, json, os, shlex, socket, subprocess, sys, time, urllib.request
from pathlib import Path

H = Path(os.environ.get('SHORTPF_OUT', r'C:\Users\imanu\source\repos\moe-aggressive-commit\docs\porto\strata_adattivo'
                        r'\corse_2080ti\20261006_due_schede'))
D = Path(r'D:\ds4_work\strata\Strata')
REPO = Path(r'C:\Users\imanu\source\repos\Strata-2080Ti')
SHA = '7695859a'   # upstream v0.1.43: public text only
DAILY_CFG = D / 'strata-0143-daily.json'
PREFIX_ROOT = Path(r'D:\ds4_work\strata\lab_prefix\shortpf')
SIZES = [100, 300, 1000, 3000, 8000]
FIXTURE = Path(__file__).resolve().parent / ('shortpf_fixture.json' if '--plan' in sys.argv and sys.argv[sys.argv.index('--plan') + 1] == 'sizes' else 'shortpf_fixture2.json')

ap = argparse.ArgumentParser()
ap.add_argument('--tag', required=True)
ap.add_argument('--exe', required=True)
ap.add_argument('--serve', required=True)
ap.add_argument('--env', default='', help='extra engine env, "K=V K2=V2"')
ap.add_argument('--args', default='', help='extra engine args appended to the Daily ones')
ap.add_argument('--fixed', action='store_true', help='--expert-cache 1500, no --elastic, --adapt-every 0')
ap.add_argument('--server-env', default='')
ap.add_argument('--port', type=int, default=8103)
ap.add_argument('--maxt', type=int, default=32)
ap.add_argument('--reps', type=int, default=3, help='conversations: A asc, B desc, C asc (1-3)')
ap.add_argument('--compare', default=None)
ap.add_argument('--plan', default='xover', choices=['xover', 'sizes'],
                help='xover: two conversations, short_read switched per request (needs the pipeline switch); '
                     'sizes: three conversations, the engines own --short-read')
ap.add_argument('--force-sr', type=int, default=None, help='xover: every request with this short_read')
ap.add_argument('--no-prio', action='store_true', help='leave the engine at normal priority')
ap.add_argument('--only-fixture', action='store_true')
a = ap.parse_args()
OUT = H / f'shortpf_{a.tag}'


# ---------------------------------------------------------------- fixture
def git_text(path):
    return subprocess.run(['git', '-C', str(REPO), 'show', f'{SHA}:{path}'], capture_output=True,
                          check=True).stdout.decode('utf-8', 'replace').replace('\r\n', '\n')


def tokenizer():
    sys.path.insert(0, str(Path(a.serve).parent / 'tools'))
    import strata_tokenizer as ST
    cfg = json.loads(DAILY_CFG.read_text(encoding='utf-8-sig'))
    tp = Path(cfg['tokenizer'])
    vocab = json.loads((tp / 'vocab.json').read_text(encoding='utf-8'))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tp / 'merges.txt').read_text(encoding='utf-8').split('\n'),
                        json.loads((tp / 'token_type.json').read_text()))


TOK = None


def ntok(s):
    return len(TOK.encode(s))


def cut(text, want):
    probe = text[:40000]
    ratio = max(ntok(probe), 1) / max(len(probe), 1)
    n = min(len(text), int(want / ratio))
    for _ in range(4):
        got = ntok(text[:n])
        if abs(got - want) <= want * 0.02 or n >= len(text):
            break
        n = min(len(text), int(n * want / max(got, 1)))
    k = text.rfind('\n', 0, n)
    return text[:k if k > n * 0.9 else n]


INSTR = """Sei l'assistente operativo di un gruppo di sviluppo. Rispondi in italiano, in modo breve e concreto.
Lavori su un motore di inferenza per modelli a esperti su schede grafiche di fascia consumer. Non inventare numeri.

# Documentazione di riferimento
"""


def build_fixture():
    global TOK
    TOK = tokenizer()
    docs = ['docs/INTEL.md', 'docs/AMD_HIP.md', 'docs/MULTI_GPU.md']
    systems = [INSTR + cut(git_text(p), 12000) for p in docs]
    pool_files = ['src/core/expert_source.cpp', 'src/core/pinned.cu', 'src/core/session.cpp', 'src/core/weights.cpp',
                  'src/kernels/cpu/pool.cpp', 'src/core/conversation_file.cpp']
    pool = '\n'.join(git_text(p) for p in pool_files)
    users = []   # per conversation, per size: the piece
    off = 0
    for rep in range(3):
        row = {}
        for n in SIZES:
            piece = cut(pool[off:], n)
            off += len(piece)
            row[str(n)] = piece
        users.append(row)
    return dict(sha=SHA, systems=systems, users=users, first='Presentati in una frase.',
                sizes={'systems': [ntok(s) for s in systems], 'users': [{k: ntok(v) for k, v in r.items()} for r in users]})


XPLAN = [[(100, 0), (100, 1024), (300, 1024), (300, 0), (1000, 0), (1000, 1024), (3000, 0), (8000, 0)],
         [(1000, 1024), (1000, 0), (300, 0), (300, 1024), (100, 1024), (100, 0), (3000, 0)]]


def build_fixture2():
    global TOK
    TOK = tokenizer()
    f1 = Path(__file__).resolve().parent / 'shortpf_fixture.json'
    base = json.loads(f1.read_text(encoding='utf-8')) if f1.exists() else build_fixture()
    pool_files = ['src/core/expert_source.cpp', 'src/core/pinned.cu', 'src/core/session.cpp', 'src/core/weights.cpp',
                  'src/kernels/cpu/pool.cpp', 'src/core/conversation_file.cpp', 'src/core/expert_cache.cpp']
    pool = '\n'.join(git_text(p) for p in pool_files)
    off = len(pool) // 3   # not the slices of the sizes plan
    pieces = []
    for conv, steps in enumerate(XPLAN):
        row = []
        for n, _ in steps:
            piece = cut(pool[off:], n)
            off += len(piece)
            row.append(piece)
        pieces.append(row)
    return dict(sha=SHA, systems=base['systems'][:2], pieces=pieces, first=base['first'],
                sizes={'systems': [ntok(x) for x in base['systems'][:2]], 'pieces': [[ntok(x) for x in r] for r in pieces]})


ASK = '\n\nRispondi con una sola frase breve su cosa fa questo estratto.'


def body(fx, conv, history, user_text):
    msgs = [{'role': 'system', 'content': fx['systems'][conv]}]
    for u, r in history:
        msgs += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    msgs.append({'role': 'user', 'content': user_text})
    return {'model': 'x', 'max_tokens': a.maxt, 'temperature': 0, 'stream': True,
            'chat_template_kwargs': {'enable_thinking': False}, 'messages': msgs}


def stream(path, bd, timeout=900):
    req = urllib.request.Request(f'http://127.0.0.1:{a.port}{path}', data=json.dumps(bd).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0 = time.monotonic()
    first, parts, usage = None, [], {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            c = json.loads(data)
            if c.get('usage'):
                usage.update(c['usage'])
                usage['timings'] = c.get('timings')
            for ch in c.get('choices') or []:
                d = ch.get('delta') or {}
                piece = d.get('content') or d.get('reasoning_content') or ''
                if piece and first is None:
                    first = time.monotonic()
                if d.get('content'):
                    parts.append(d['content'])
    end = time.monotonic()
    return (first or end) - t0, end - t0, ''.join(parts), usage


def http(path, payload=None, timeout=10):
    req = urllib.request.Request(f'http://127.0.0.1:{a.port}' + path,
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'}, method='GET' if payload is None else 'POST')
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Server:
    def __init__(self, prefix_file):
        cfg = json.loads(DAILY_CFG.read_text(encoding='utf-8-sig'))   # read only
        args = list(cfg['args'])
        k = args.index('--prompt-cache-file')
        args[k + 1] = str(prefix_file)
        if '--vision' in args:
            args.remove('--vision')
        cfg.pop('vision', None)
        if a.fixed:
            args[args.index('--expert-cache') + 1] = '1500'
            i = args.index('--elastic')
            del args[i]
            args += ['--adapt-every', '0']
        cfg['args'] = args + shlex.split(a.args)
        cfg['exe'], cfg['log'] = a.exe, str(OUT / 'engine.log')
        cfg['port'] = a.port
        for kv in shlex.split(a.env):
            k2, v2 = kv.split('=', 1)
            cfg.setdefault('env', {})[k2] = v2
        if a.plan == 'xover':
            cfg.setdefault('env', {})['STRATA_PIPELINE_DEBUG'] = '1'
            cfg['env']['STRATA_PIPELINE_SWITCH'] = str(OUT / 'switch.txt')
            (OUT / 'switch.txt').write_text('short_read=64', encoding='utf-8')
        self.cfg = cfg
        self.cfg_path = OUT / 'config.json'
        self.cfg_path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        self.p = None

    def __enter__(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', a.port))
        env = {k: v for k, v in os.environ.items() if k != 'STRATA_API_KEY' and not k.upper().startswith('STRATA_')}
        for kv in shlex.split(a.server_env):
            k2, v2 = kv.split('=', 1)
            env[k2] = v2
        t0 = time.monotonic()
        self.p = subprocess.Popen([str(D / '.venv/Scripts/python.exe'), '-B', str(Path(a.serve) / 'server.py'),
                                   '--engine', 'strata', '--config', str(self.cfg_path), '--host', '127.0.0.1',
                                   '--port', str(a.port)],
                                  cwd=D, env=env, stdout=(OUT / 'server.log').open('w', encoding='utf-8'),
                                  stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        while time.monotonic() - t0 < 900:
            assert self.p.poll() is None, 'server exited'
            try:
                h = http('/health')
                if h.get('status') == 'ok' and h.get('loaded'):
                    self.startup_s = time.monotonic() - t0
                    print(f'engine up in {self.startup_s:.0f} s', flush=True)
                    if not a.no_prio:   # the box is shared with other agents' CPU load: the engine gets the CPU first
                        import psutil
                        for kd in psutil.Process(self.p.pid).children(recursive=True):
                            if kd.name().lower() == 'strata.exe':
                                kd.nice(psutil.HIGH_PRIORITY_CLASS)
                                print('engine priority HIGH, pid', kd.pid, flush=True)
                    return self
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise TimeoutError('server startup')

    def __exit__(self, *exc):
        import psutil
        try:
            print('unload:', http('/unload', {}, timeout=120), flush=True)
        except Exception as e:   # noqa: BLE001
            print('unload failed:', repr(e)[:200], flush=True)
        kids = psutil.Process(self.p.pid).children(recursive=True)
        self.p.terminate()
        try:
            self.p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.p.kill()
        for kd in kids:
            try:
                kd.terminate()
            except psutil.Error:
                pass
        psutil.wait_procs(kids, timeout=30)


def log_lines(start):
    p = OUT / 'engine.log'
    if not p.exists():
        return [], start
    b = p.read_bytes()
    keys = ('strata serve: prompt ', 'prefill timing', 'conversation cache: restored', 'waited', 'PP ', 'short')
    lines = [l for l in b[start:].decode('utf-8', 'replace').splitlines() if any(k in l for k in keys)]
    return [l[:400] for l in lines], len(b)


def main():
    global TOK
    if FIXTURE.exists():
        fx = json.loads(FIXTURE.read_text(encoding='utf-8'))
    else:
        fx = build_fixture() if a.plan == 'sizes' else build_fixture2()
        FIXTURE.write_text(json.dumps(fx, ensure_ascii=False), encoding='utf-8')
    print('fixture sizes', json.dumps(fx['sizes']), 'sha256', sha(FIXTURE)[:16], flush=True)
    if a.only_fixture:
        return
    OUT.mkdir(exist_ok=False)
    pdir = PREFIX_ROOT / a.tag
    pdir.mkdir(parents=True, exist_ok=False)
    build = Path(a.exe).parent / 'build.json'
    res = dict(tag=a.tag, exe=a.exe, exe_sha256=sha(a.exe), build=json.loads(build.read_text()) if build.exists() else None,
               serve=a.serve, env=a.env, args=a.args, fixed=a.fixed, fixture_sha256=sha(FIXTURE),
               started=time.strftime('%Y-%m-%d %H:%M:%S'), requests=[])
    order = {0: SIZES, 1: SIZES[::-1], 2: SIZES}
    xo = a.plan == 'xover'
    with Server(pdir / 'prefix.bin') as srv:
        res['startup_s'] = round(srv.startup_s, 1)
        res['engine_env'] = srv.cfg.get('env')
        res['engine_args'] = srv.cfg['args']
        pos = log_lines(0)[1]
        for conv in range(2 if xo else a.reps):
            hist = []
            if xo:
                plan = [('cold', None, None)] + [(f'n{n}_sr{sr}_{k}', n, sr) for k, (n, sr) in enumerate(XPLAN[conv])]
            else:
                plan = [('cold', None, None)] + [(f'n{n}', n, None) for n in order[conv]]
            for k, (name, n, sr) in enumerate(plan):
                if n is None:
                    user = fx['first']
                elif xo:
                    user = fx['pieces'][conv][k - 1] + ASK
                else:
                    user = fx['users'][conv][str(n)] + ASK
                if xo:
                    srv_sr = 64 if sr is None else (a.force_sr if a.force_sr is not None else sr)
                    (OUT / 'switch.txt').write_text(f'short_read={srv_sr}', encoding='utf-8')
                nm = f'c{conv}_{name}'
                try:
                    ttft, wall, txt, usage = stream('/v1/chat/completions', body(fx, conv, hist, user))
                    ok, err = True, None
                except Exception as e:   # noqa: BLE001
                    ttft = wall = None; txt, usage, ok, err = '', {}, False, repr(e)[:300]
                time.sleep(0.8)
                lines, pos = log_lines(pos)
                (OUT / f'{nm}.txt').write_text(txt, encoding='utf-8')
                pt = usage.get('prompt_tokens') or usage.get('input_tokens')
                ct = (usage.get('prompt_tokens_details') or {}).get('cached_tokens') or usage.get('cache_read_input_tokens')
                r = dict(name=nm, conv=conv, target=n, short_read=(srv_sr if xo else None), ok=ok, err=err, ttft=ttft and round(ttft, 3),
                         wall=wall and round(wall, 3), prompt_tokens=pt, cached_tokens=ct,
                         new_tokens=(pt - (ct or 0)) if pt else None, text_head=txt[:80].replace('\n', ' '), engine=lines)
                res['requests'].append(r)
                if ok:
                    hist.append((user, txt))
                pl = next((l for l in lines if 'strata serve: prompt ' in l), '')
                print(f'{nm:18s} sr {srv_sr if xo else "-"} ttft {r["ttft"]} s new {r["new_tokens"]}  {pl[13:150]}', flush=True)
                for l in lines:
                    if 'prefill timing' in l:
                        print('      ', l[:300], flush=True)
    res['ended'] = time.strftime('%Y-%m-%d %H:%M:%S')
    if a.compare:
        other = H / f'shortpf_{a.compare}'
        res['same_text_as'] = {r['name']: ((other / f'{r["name"]}.txt').exists() and
                                           (other / f'{r["name"]}.txt').read_text(encoding='utf-8') ==
                                           (OUT / f'{r["name"]}.txt').read_text(encoding='utf-8'))
                               for r in res['requests']}
    (OUT / 'results.json').write_text(json.dumps(res, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    rows = [f'{r["name"]:16s} target {r["target"]} sr {r["short_read"]} ttft {r["ttft"]} s new {r["new_tokens"]} cached {r["cached_tokens"]}'
            for r in res['requests']]
    (OUT / 'summary.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    if a.compare:
        print('same text as', a.compare, json.dumps(res['same_text_as']), flush=True)


if __name__ == '__main__':
    main()
