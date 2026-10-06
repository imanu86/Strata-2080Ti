from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent;s=(BASE/'test_elastic038_operational.py').read_text(encoding='utf-8')
pairs=[("R=BASE/'elastic038_20261003'","R=BASE/'elastic-tail038_20261003'"),("parity=json.loads((R/'parity-r1/results.json').read_text())","parity=json.loads((BASE/'elastic038_20261003/parity-r1/results.json').read_text())"),("strata-qa-elastic.exe","strata-qa-combined.exe"),("strata-elastic.exe","strata-combined.exe"),("CUDA_CACHE_PATH=str(R/'parity-r1/cuda-cache')","CUDA_CACHE_PATH=str(BASE/'elastic038_20261003/parity-r1/cuda-cache')"),("args=V.set_arg(cfg['args'],'--prompt-cache-every',16384)","args=V.set_arg(cfg['args'],'--prompt-cache-every',16384)+['--prompt-cache-tail']"),("'quality_gate_prerequisite':'parity-r1'","'quality_gate_prerequisites':['elastic038/parity-r1','checkpoint-tail038/parity-r1']")]
for a,b in pairs:assert a in s,a;s=s.replace(a,b)
needle="OUT=R/'operational-r1';OUT.mkdir(exist_ok=False)"
s=s.replace(needle,"tail=json.loads((BASE/'checkpoint-tail038_20261003/parity-r1/results.json').read_text());assert tail['status']=='completed' and tail['gate_pass'] and tail['oracle_pass']\n"+needle)
s=s.replace('Natural single-GPU elastic serve smoke','Combined elastic + tail-checkpoint single-GPU serve smoke')
if '--final' in sys.argv:s=s.replace("R=BASE/'elastic-tail038_20261003'", "R=BASE/'elastic-tail038_final_20261003'")
(BASE/('test_elastic_tail038_final_operational.py' if '--final' in sys.argv else 'test_elastic_tail038_operational.py')).write_text(s,encoding='utf-8',newline='\n')
print('Combined operational runner prepared; both isolated quality gates required')
