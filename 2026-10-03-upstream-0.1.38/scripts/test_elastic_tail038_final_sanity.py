"""Final CLI-polished integration binary, tail flag off, versus measured VMM arm."""
from pathlib import Path
import os,sys,json,threading,time
BASE=Path(__file__).resolve().parent;pure='--pure' in sys.argv;R=BASE/('elastic038_20261003' if pure else 'elastic-tail038_final_20261003');Q=BASE/'elastic038_20261003/parity-r1'
sys.path[:0]=[str(R/'source'),r'D:\ds4_work\strata\Strata\.venv\Lib\site-packages',r'C:\Users\imanu\source\repos\Strata-2080Ti\tools\sm75']
import numpy as np,psutil,validate_model as V
from serve.server import StrataEngine
from serve.frontend import ChatTemplate,OutputParser
from tools.strata_tokenizer import Tokenizer
assert json.loads((R/'cli-tests.json').read_text())['status']=='completed'
quality=json.loads((Q/'results.json').read_text());assert quality['gate_pass'] and quality['oracle_pass']
OUT=R/'final-sanity';OUT.mkdir(exist_ok=False)
cfg=json.loads(Path(r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json').read_text(encoding='utf-8-sig'))
tp=Path(cfg['tokenizer']);vocab=json.loads((tp/'vocab.json').read_text(encoding='utf-8'));tokens=[None]*len(vocab)
for text,index in vocab.items():tokens[index]=text
tok=Tokenizer(tokens,(tp/'merges.txt').read_text(encoding='utf-8').split('\n'),json.loads((tp/'token_type.json').read_text(encoding='utf-8')))
warm=tok.encode(ChatTemplate(tp/'chat_template.jinja').render([{'role':'user','content':'Return only JSON with key ready and value true.'}],enable_thinking=False),parse_special=True)
fixtures={n:json.loads((BASE/'checkpoint038_20261003/model-tests-r3'/f'{n}.ids.json').read_text()) for n in ('fresh','follow')}
args=quality['arms']['Bon']['args'];reference=V.check_dump(Q/'Bon/code.logits.bin',512,len(tokens));ids=json.loads((Q/'code.ids.json').read_text())
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(PYTHONDONTWRITEBYTECODE='1',STRATA_WATCHDOG_S='240',STRATA_PARALLEL_INTERMEDIATE_QUANT='0',STRATA_TEST_FIRST_LOGITS=str(OUT/'first.f32'),CUDA_CACHE_PATH=str(Q/'cuda-cache'));env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
exe=R/('strata-qa-elastic-final.exe' if pure else 'strata-qa-combined.exe');e=StrataEngine(str(exe),args,cwd=str(OUT),log=str(OUT/'engine.log'),env=env,lazy=True)
result={'status':'running','exe_sha256':V.sha(exe),'args':args,'method':'tail flag disabled; same stable VMM arm as complete quality gate; exact first rows and64 additional teacher rows checked after nonnumerical CLI polish','requests':[],'teacher_matches':[]};start=time.monotonic();cancel=threading.Event()
def save():(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
def row():
    x=np.fromfile(OUT/'first.f32',dtype='<f4');assert x.size==len(tokens) and np.isfinite(x).all();return x
assert not any('strata' in (p.info['name'] or '').lower() and (p.info['name'] or '').lower().endswith('.exe') for p in psutil.process_iter(['name']))
save()
try:
    e.restart();result['info']=dict(e.info);assert int(e.info['expert_slots'])==5357
    list(e.generate(warm,32,{'temperature':0},cancel))
    for name,prompt in fixtures.items():
        (OUT/'first.f32').unlink(missing_ok=True);generated=[t for t in e.generate(prompt,128,{'temperature':0},cancel) if t is not None]
        current=row();match=bool(np.array_equal(current,np.fromfile(Q/'Bon'/f'{name}-first.f32',dtype='<f4')));current.tofile(OUT/f'{name}-first.f32')
        text=tok.decode(generated)
        for tag in ('<|im_end|>','<|endoftext|>'):text=text.removesuffix(tag)
        p=OutputParser(thinking=False);content=''.join(x.text for x in p.feed(text)+p.finish() if x.kind=='content');passed=V.evaluate_answer(content,e.last['finish'],quality['plan']['expected'])
        result['requests'].append({'name':name,'bitwise':match,'oracle_pass':passed,'content':content,**e.last});save();assert match and passed
        print('FINAL SANITY',name,'bitwise',match,flush=True)
    for pos in range(64):
        assert psutil.virtual_memory().available>8*2**30 and time.monotonic()-start<900
        (OUT/'first.f32').unlink(missing_ok=True);list(e.generate(fixtures['follow']+ids[:pos+1],1,{'temperature':0},cancel))
        match=bool(np.array_equal(row(),reference[pos]));result['teacher_matches'].append(match);assert match,pos
    result.update(status='completed',elapsed_s=time.monotonic()-start,bitwise=True)
except BaseException as exc:result.update(status='failed',error=repr(exc),elapsed_s=time.monotonic()-start);raise
finally:e.close();save();print('FINAL SANITY STATUS',result['status'],flush=True)
