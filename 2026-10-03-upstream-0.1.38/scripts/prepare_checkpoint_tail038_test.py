from pathlib import Path
BASE=Path(__file__).resolve().parent
s=(BASE/'test_elastic038_parity.py').read_text(encoding='utf-8')
pairs=[
 ("R=BASE/'elastic038_20261003'", "R=BASE/'checkpoint-tail038_20261003'"),
 ("V.set_arg(old['plan']['args'],'--prefill',2048)", "V.set_arg(old['plan']['args'],'--prefill',6144)"),
 ("'arms':['A1','Boff','Bon','A2']", "'arms':['A1','B','A2']"),
 ("'elastic_on_mode':'VMM arena only; resize disabled by long real policy intervals; whole initial cache lendable'", "'checkpoint_candidate':'one extra existing chunk boundary near prompt tail; no alignment cuts'"),
 ("'elastic_flags':['--elastic','--elastic-tail-mib','32768','--elastic-period-ms','2147483647','--elastic-stable-ms','2147483647','--elastic-reserve-mib','0']", "'candidate_flags':['--prompt-cache-tail']"),
 ("'prefill_chunk':2048", "'prefill_chunk':6144"),
 ("qa=json.loads((R/'qa-build.json').read_text())", "qa=json.loads((R/'build.json').read_text())"),
 ("strata-qa-elastic.exe", "strata-qa-tail.exe"),
 ("qa['sha256']", "qa['qa_sha256']"),
 ("result['plan']['elastic_flags'] if label=='Bon'", "result['plan']['candidate_flags'] if label=='B'"),
 ("for mode in ('Boff','Bon'):", "for mode in ('B',):"),
 ("'controlled arm resized'", "'tail-only arm unexpectedly contains elastic code'"),
]
for a,b in pairs:
    assert a in s,a;s=s.replace(a,b)
s=s.replace('"""Isolated elastic port: pristine A, port disabled, VMM stable, pristine repeat.', '"""Tail checkpoint alternative: pristine A, optional tail checkpoint B, pristine repeat.')
s=s.replace("'tests_are_local_regression_not_upstream_universal_standard':True", "'tests_are_local_regression_not_upstream_universal_standard':True,'original_grid_gate':'FAIL preserved; this is a separate replacement design' ")
(BASE/'test_checkpoint_tail038_parity.py').write_text(s,encoding='utf-8',newline='\n')
print('Tail A/B/A complete frozen-corpus runner prepared')
