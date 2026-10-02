"""Paired native-IQ model validation. Dry-run by default; no downloads.

Uses the fork's first-window logits diagnostic on growing, teacher-forced
prefixes. The predicted token is never appended to the reference text.
Raw logits stay in the private output directory, outside Git.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
POLICY = {
    'description': 'Local paired regression gate; not an upstream universal threshold',
    'block_tokens': 32,
    'nll_allowance_nats': 0.01,
    'median_kl_floor': 0.001,
    'p99_kl_floor': 0.01,
    'baseline_kl_multiplier': 3.0,
    'top1_allowance_percentage_points': 1.0,
}
TEXTS = {
    'code': (512, '''Review this Python module. Explain the ordering invariant and the test cases.
def stable_unique(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

def test_stable_unique():
    assert stable_unique([7, 2, 7, 1, 2, 9]) == [7, 2, 1, 9]
    assert stable_unique([]) == []
    assert stable_unique([4, 4, 4]) == [4]

The set records membership, while the list preserves first occurrence order.
The empty input returns an empty list. Repeated values never change the order.
An unordered set alone does not satisfy the public API. Complexity is linear
on average, assuming constant-time hashing. Objects must be hashable.
'''),
    'doc': (1024, '''Harbor archive operating notes.
The archive contains dated records with immutable identifiers. A revision
creates a new record and cites the previous identifier; it never overwrites
the previous text. The index stores both the creation date and the subject.
Readers can retrieve older revisions even after the subject receives a new
entry. The shipping register measures mass in kilograms and volume in liters.
Binary memory quantities use MiB and GiB: 1 GiB equals 1024 MiB. Decimal
network quantities use MB and GB. Conversion must name the unit explicitly.
Inspection happens before loading. A damaged crate is recorded, separated,
and reviewed by an operator. A label without an identifier is not accepted.
The morning report lists arrivals; the evening report reconciles departures.
The checksum verifies the contents of an exported archive. Keeping the
manifest with the export lets the next reader reproduce the inventory.
'''),
    'chat': (1024, '''User: I need to plan a small migration without losing local edits.
Assistant: First inspect the current version and save a recovery point. Then
apply the change in a controlled workspace and compare the results.
User: How do I distinguish capacity from the data actually used?
Assistant: Record them separately. A capacity of 262144 tokens does not mean
that a request contains 262144 tokens. Count the input and generated output.
User: What if two outputs differ slightly?
Assistant: Compare predictions on a fixed reference text, and check complete
answers to problems with known solutions. Small rounding differences can
change a nearly tied token, so textual equality is not the only metric.
User: What belongs in the report?
Assistant: The exact version, settings, input identifiers, measured results,
and the limits of the validation. Keep raw evidence so it can be inspected.
'''),
}
ANSWERS = [
    ('binary_units', 'Return only JSON with keys a_gib and b_gib. Compute '
     '48 * 384 MiB in GiB, and 48 * 1.75 * 192 MiB in GiB. 1 GiB = 1024 MiB.',
     {'a_gib': 18, 'b_gib': 15.75}),
    ('stable_unique', 'Return only JSON with key result. Apply stable_unique to '
     '[7,2,7,1,2,9]: keep the first occurrence of each value in input order.',
     {'result': [7, 2, 1, 9]}),
    ('ledger', 'Return only JSON with key balance. The initial balance is 1250. '
     'Add 375, subtract 480, then subtract 95. What is the final balance?', {'balance': 1050}),
    ('dependency_order', 'Return only JSON with key order. Tasks A, B, C, D: '
     'B requires A; C requires B; D requires C. Give their only valid order.',
     {'order': ['A', 'B', 'C', 'D']}),
    ('read_record', 'Return only JSON with key code. Records: north=N731, '
     'south=S482, west=W916. What is the code of south?', {'code': 'S482'}),
    ('weighted_sum', 'Return only JSON with key total. Three items cost 17 each, '
     'two items cost 23 each, and shipping costs 8. Compute the total.', {'total': 105}),
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def set_arg(args, key, value):
    args = list(args)
    while key in args:
        i = args.index(key)
        del args[i:i + 2]
    return args + [key, str(value)]


def check_dump(path, rows, vocab):
    import numpy as np
    header = np.fromfile(path, dtype='<i4', count=2)
    require(header.tolist() == [vocab, rows], 'Logits header differs from expected dimensions')
    require(Path(path).stat().st_size == 8 + rows * vocab * 4, 'Incomplete or oversized logits dump')
    return np.memmap(path, dtype='<f4', mode='r', offset=8, shape=(rows, vocab))


def logsoftmax(row):
    import numpy as np
    row = np.asarray(row, dtype=np.float64)
    require(bool(np.isfinite(row).all()), 'Non-finite logits')
    m = float(row.max())
    return row - (m + math.log(float(np.exp(row - m).sum())))


def paired_stats(a, b, ids):
    import numpy as np
    require(a.shape == b.shape and a.shape[0] == len(ids), 'Unpaired logits/text dimensions')
    kl, delta, same = [], [], []
    nll_a, nll_b = [], []
    for i in range(len(ids)):
        pa, pb = logsoftmax(a[i]), logsoftmax(b[i])
        # Final input logits have no known next token and are excluded from all metrics.
        if i + 1 == len(ids):
            continue
        target = ids[i + 1]
        require(0 <= target < a.shape[1], 'Reference target outside vocabulary')
        na, nb = -float(pa[target]), -float(pb[target])
        nll_a.append(na); nll_b.append(nb); delta.append(nb - na)
        kl.append(max(0.0, float((np.exp(pa) * (pa - pb)).sum())))
        same.append(bool(a[i].argmax() == b[i].argmax()))
    require(bool(delta), 'No scored positions')
    return {'positions': len(delta), 'top1_agreement_pct': 100 * float(np.mean(same)),
            'kl_mean': float(np.mean(kl)), 'kl_median': float(np.median(kl)),
            'kl_p99': float(np.percentile(kl, 99)), 'kl_max': float(np.max(kl)),
            'ppl_reference': math.exp(float(np.mean(nll_a))),
            'ppl_candidate': math.exp(float(np.mean(nll_b))),
            'nll_delta_mean': float(np.mean(delta)), 'nll_deltas': delta,
            'kl_values': kl, 'same_top1': same}


def block_interval(values, block=32):
    import numpy as np
    means = np.array([np.mean(values[i:i+block]) for i in range(0, len(values), block)])
    require(len(means) > 1, 'Not enough blocks for an uncertainty estimate')
    return float(np.mean(values)), 1.96 * float(means.std(ddof=1)) / math.sqrt(len(means))


def regression_gate(candidate, repeat, policy=POLICY):
    cand_mean, cand_ci = block_interval(candidate['nll_deltas'], policy['block_tokens'])
    rep_mean, rep_ci = block_interval(repeat['nll_deltas'], policy['block_tokens'])
    noise = abs(rep_mean) + rep_ci
    limits = {'nll_upper_nats': policy['nll_allowance_nats'] + noise,
              'median_kl': max(policy['median_kl_floor'], policy['baseline_kl_multiplier'] * repeat['kl_median']),
              'p99_kl': max(policy['p99_kl_floor'], policy['baseline_kl_multiplier'] * repeat['kl_p99']),
              'top1_min_pct': repeat['top1_agreement_pct'] - policy['top1_allowance_percentage_points']}
    checks = {'nll': cand_mean + cand_ci <= limits['nll_upper_nats'],
              'median_kl': candidate['kl_median'] <= limits['median_kl'],
              'p99_kl': candidate['kl_p99'] <= limits['p99_kl'],
              'top1': candidate['top1_agreement_pct'] >= limits['top1_min_pct']}
    return {'pass': all(checks.values()), 'checks': checks, 'limits': limits,
            'candidate_delta_nats': cand_mean, 'candidate_block_half_interval': cand_ci,
            'reference_repeat_noise_bound': noise,
            'uncertainty_note': 'Approximate paired 32-token block interval; a limited regression corpus'}


def public_stats(stats):
    return {k: v for k, v in stats.items() if k not in ('nll_deltas', 'kl_values', 'same_top1')}


def evaluate_answer(text, finish, expected):
    try:
        answer = json.loads(text.strip())
    except (ValueError, TypeError):
        return False
    return finish == 'stop' and answer == expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cache-slots', type=int, default=4600, help='Requested fixed lab budget; actual slots recorded')
    ap.add_argument('--deep-tokens', type=int, default=131072)
    ap.add_argument('--deadline-seconds', type=int, default=1800)
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    require(a.cache_slots > 0 and 32768 <= a.deep_tokens <= 250000 and a.deadline_seconds > 0, 'Invalid test bounds')
    require(not a.out.exists(), 'Refusing an existing output directory')
    cfg = json.loads(a.config.read_text(encoding='utf-8-sig'))
    args = [x for x in cfg['args'] if x != '--elastic']
    for key, value in [('--expert-cache', a.cache_slots), ('--prompt-cache', 6),
                       ('--conversation-cache-mib', 0), ('--adapt-swaps', 0),
                       ('--pcie-frac', 0), ('--suffix-draft', 0), ('--mtp-max-t', 4)]:
        args = set_arg(args, key, value)
    require(args[args.index('--max-context')+1] == '262144', 'Keep KV capacity 262144')
    require(args[args.index('--kv')+1] == 'int8' and args[args.index('--kv-resident')+1] == '32768', 'Unexpected KV configuration')
    plan = {'protocol': 'native-IQ A1(reference)/B(candidate)/A2(reference)',
            'config_sha256': sha(a.config), 'args': args, 'policy': POLICY,
            'texts': {k: {'tokens': v[0], 'text_sha256': hashlib.sha256(v[1].encode()).hexdigest()} for k,v in TEXTS.items()},
            'oracle_cases': [x[0] for x in ANSWERS] + ['deep_three_records'],
            'deep_prompt_tokens': a.deep_tokens, 'teacher_forcing': 'growing prefixes; max_new=1; prediction discarded',
            'prefill_scope': 'teacher prefixes exercise decode; separate deep oracle exercises batched prefill',
            'daily_configuration_modified': False, 'artificial_load': False,
            'gate_declared_before_runs': True}
    profile = Path(args[args.index('--expert-profile')+1])
    plan['expert_profile_sha256'] = sha(profile)
    for label, path in [('reference', a.reference), ('candidate', a.candidate)]:
        require(path.is_file(), 'Missing executable: '+str(path))
        plan[label] = {'exe': str(path.resolve()), 'sha256': sha(path)}
    if not a.run:
        print(json.dumps(plan, indent=2)); return

    import numpy as np
    import psutil
    sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
    from serve.server import StrataEngine
    from serve.frontend import ChatTemplate
    from tools.strata_tokenizer import Tokenizer
    require(not any((p.info['name'] or '').lower() in ('strata.exe', 'llama-server.exe')
                    for p in psutil.process_iter(['name'])), 'Another engine is active; no process was stopped')
    require(psutil.virtual_memory().available > 16 * 2**30, 'Insufficient available RAM for loading the model')
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'plan.json', plan)
    tp = Path(cfg['tokenizer'])
    vocab = json.loads((tp/'vocab.json').read_text(encoding='utf-8'))
    tokens = [None]*len(vocab)
    for token, index in vocab.items(): tokens[index] = token
    tok = Tokenizer(tokens, (tp/'merges.txt').read_text(encoding='utf-8').split('\n'),
                    json.loads((tp/'token_type.json').read_text(encoding='utf-8')))
    tpl = ChatTemplate(tp/'chat_template.jinja')
    texts = {}
    for name,(count,text) in TEXTS.items():
        ids = tok.encode(text * (count // 64 + 2), parse_special=False)[:count]
        require(len(ids) == count, 'Corpus too short')
        texts[name] = ids
        write_json(a.out/(name+'.ids.json'), ids)
        (a.out/(name+'.txt')).write_text(tok.decode(ids), encoding='utf-8')
    plan['tokenizer_sha256'] = {name: sha(tp/name) for name in ['vocab.json','merges.txt','token_type.json','chat_template.jinja']}
    plan['token_ids_sha256'] = {name: sha(a.out/(name+'.ids.json')) for name in texts}
    write_json(a.out/'plan.json', plan)

    def prompt(text):
        return tok.encode(tpl.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
    gold = {'north': 'ARCO-731', 'south': 'LUNA-482', 'west': 'PORTO-916'}
    template = tpl.render([{'role': 'user', 'content':
        'Read the archive below. AUTH_RECORD lines are authoritative. '
        'Return only JSON with the codes for north, south, west.\n__ARCHIVE__\n'
        'Give all three codes, using keys north, south, west.'}], enable_thinking=False)
    left, right = template.split('__ARCHIVE__')
    left, right = tok.encode(left, parse_special=True), tok.encode(right, parse_special=True)
    records = [tok.encode('\nAUTH_RECORD '+k+'='+v+'\n', parse_special=False) for k,v in gold.items()]
    fill = texts['doc']
    remaining = a.deep_tokens-len(left)-len(right)-sum(map(len,records))
    lengths = [remaining//10, remaining*4//10, remaining*4//10]
    lengths.append(remaining-sum(lengths))
    deep = list(left)
    for i, size in enumerate(lengths):
        deep += (fill*(size//len(fill)+1))[:size]
        if i < 3: deep += records[i]
    deep += right
    require(len(deep) == a.deep_tokens, 'Deep fixture size differs')
    write_json(a.out/'deep.ids.json', deep)
    write_json(a.out/'oracles.json', [{'name': n, 'prompt': p, 'expected': e} for n,p,e in ANSWERS]+
               [{'name': 'deep_three_records', 'expected': gold, 'prompt_tokens': len(deep), 'ids_sha256': sha(a.out/'deep.ids.json')}])

    def resource_check():
        require(psutil.virtual_memory().available > 8 * 2**30, 'Available RAM fell below lab floor')
    results = {'plan': plan, 'arms': {}, 'status': 'running'}
    timer = threading.Timer(a.deadline_seconds, lambda: os._exit(124))
    timer.daemon = True; timer.start()
    try:
        for label, exe in [('A1',a.reference),('B',a.candidate),('A2',a.reference)]:
            require(sha(profile) == plan['expert_profile_sha256'], 'Expert profile changed during validation')
            folder = a.out/label; folder.mkdir()
            env = dict(os.environ); env.pop('STRATA_API_KEY', None)
            env.update(cfg.get('env', {}))
            env.update(STRATA_IQ_MT_MIN='1', STRATA_PARALLEL_INTERMEDIATE_QUANT='0',
                       STRATA_DUMP_FIRST_LOGITS=str(folder/'first'), STRATA_WATCHDOG_S='240')
            env['PATH'] = os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
            print('START', label, exe.name, flush=True)
            engine = StrataEngine(str(exe.resolve()), args, cwd=str(folder), log=str(folder/'engine.log'), env=env, lazy=True)
            arm = {'exe_sha256': sha(exe), 'teacher': {}, 'answers': []}
            results['arms'][label] = arm
            try:
                engine.restart()
                arm['info'] = dict(engine.info)
                require(engine.max_context == 262144, 'Runtime KV capacity differs')
                print('READY', label, json.dumps(arm['info']), flush=True)
                require(arm['info'].get('vram_free_mib', 0) >= 1393, 'Fixed lab budget leaves insufficient VRAM margin')
                write_json(a.out/'results.json', results)
                for name, ids in texts.items():
                    path = folder/(name+'.logits.bin')
                    with path.open('wb') as out:
                        out.write(np.array([len(tokens),len(ids)], dtype='<i4').tobytes())
                        for pos in range(len(ids)):
                            raw = folder/('first.pos'+str(pos)+'.f32')
                            if raw.exists(): raw.unlink()
                            generated = [v for v in engine.generate(ids[:pos+1], 1, {'temperature':0}, threading.Event()) if v is not None]
                            require(len(generated) == 1 and engine.last['prompt_tokens'] == pos+1, 'Incomplete teacher-forced request')
                            if pos:
                                require(engine.last.get('reused', 0) >= pos-1, 'Growing reference prefix was not reused')
                            require(raw.is_file() and raw.stat().st_size == len(tokens)*4, 'Missing first-window logits row')
                            row = np.fromfile(raw, dtype='<f4')
                            require(bool(np.isfinite(row).all()), 'Non-finite native logits')
                            out.write(row.tobytes()); raw.unlink()
                            if (pos+1) % 128 == 0:
                                resource_check()
                                print('TEACHER',label,name,pos+1,'/',len(ids),flush=True)
                    check_dump(path, len(ids), len(tokens))
                    arm['teacher'][name] = {'rows':len(ids),'scored_positions':len(ids)-1,'sha256':sha(path)}
                    write_json(a.out/'results.json',results)
                for name,text,expected in ANSWERS + [('deep_three_records',None,gold)]:
                    ids = deep if text is None else prompt(text)
                    generated = [v for v in engine.generate(ids, 256, {'temperature':0}, threading.Event()) if v is not None]
                    body = tok.decode(generated)
                    for stop in ('<|im_end|>','<|endoftext|>'): body = body.removesuffix(stop)
                    passed = evaluate_answer(body,engine.last['finish'],expected)
                    item = {'name':name,'expected':expected,'output':body,'pass':passed,**engine.last}
                    arm['answers'].append(item)
                    write_json(a.out/'results.json',results)
                    print('ANSWER',label,name,'PASS' if passed else 'FAIL',engine.last['finish'],flush=True)
                    resource_check()
            finally:
                engine.close()
            print('STOPPED',label,flush=True)
        settings = ('expert_slots','kv','kv_resident','context','spec','mtp_max','lookup')
        infos = [results['arms'][x]['info'] for x in ('A1','B','A2')]
        for key in settings:
            require(key in infos[0] and all(v.get(key) == infos[0][key] for v in infos), 'Unpaired engine setting: '+key)
        report = {'texts': {}, 'policy': POLICY}
        merged_candidate = {'nll_deltas': [], 'kl_values': [], 'same_top1': []}
        merged_repeat = {'nll_deltas': [], 'kl_values': [], 'same_top1': []}
        for name, ids in texts.items():
            matrices = [check_dump(a.out/x/(name+'.logits.bin'),len(ids),len(tokens)) for x in ('A1','B','A2')]
            candidate = paired_stats(matrices[0],matrices[1],ids)
            repeat = paired_stats(matrices[0],matrices[2],ids)
            report['texts'][name] = {'A1_vs_B':public_stats(candidate), 'A1_vs_A2':public_stats(repeat),
                                    'gate':regression_gate(candidate,repeat)}
            for key in merged_candidate:
                merged_candidate[key] += candidate[key]; merged_repeat[key] += repeat[key]
        for obj in (merged_candidate,merged_repeat):
            obj.update(kl_median=float(np.median(obj['kl_values'])),kl_p99=float(np.percentile(obj['kl_values'],99)),
                       top1_agreement_pct=100*float(np.mean(obj['same_top1'])))
        report['aggregate_gate'] = regression_gate(merged_candidate,merged_repeat)
        report['oracle_pass'] = all(x['pass'] for arm in results['arms'].values() for x in arm['answers'])
        report['pass'] = report['aggregate_gate']['pass'] and report['oracle_pass'] and all(x['gate']['pass'] for x in report['texts'].values())
        report['limits'] = 'Local fixed-residency corpus and 131k retrieval; not a general quality score, elastic-cache test, or isolated upstream PR benchmark.'
        results.update(status='completed', report=report)
        write_json(a.out/'comparison.json',report); write_json(a.out/'results.json',results)
        print('VALIDATION', 'PASS' if report['pass'] else 'FAIL', json.dumps(report['aggregate_gate']),flush=True)
        return 0 if report['pass'] else 1
    except BaseException as error:
        results.update(status='failed',error=str(error)); write_json(a.out/'results.json',results)
        raise
    finally:
        timer.cancel()


if __name__ == '__main__':
    sys.exit(main() or 0)
