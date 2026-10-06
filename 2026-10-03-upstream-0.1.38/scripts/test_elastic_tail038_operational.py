"""Combined elastic + tail-checkpoint single-GPU serve smoke at20k/131k/250k. No artificial load.

This tests operational resizing and recall, not a paired speed or quality gate.
The complete controlled quality gate is a prerequisite. Daily is untouched.
"""
from pathlib import Path
import os,sys,json,time,threading,datetime,subprocess
BASE=Path(__file__).resolve().parent;R=BASE/'elastic-tail038_20261003'
sys.path[:0]=[str(R/'source'),r'D:\ds4_work\strata\Strata\.venv\Lib\site-packages',r'C:\Users\imanu\source\repos\Strata-2080Ti\tools\sm75']
import numpy as np,psutil,validate_model as V
from serve.server import StrataEngine
from serve.frontend import ChatTemplate,OutputParser
from tools.strata_tokenizer import Tokenizer
parity=json.loads((BASE/'elastic038_20261003/parity-r1/results.json').read_text());assert parity['status']=='completed' and parity['gate_pass'] and parity['oracle_pass']
tail=json.loads((BASE/'checkpoint-tail038_20261003/parity-r1/results.json').read_text());assert tail['status']=='completed' and tail['gate_pass'] and tail['oracle_pass']
OUT=R/'operational-r1';OUT.mkdir(exist_ok=False)
cfg=json.loads(Path(r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json').read_text(encoding='utf-8-sig'))
tp=Path(cfg['tokenizer']);vocab=json.loads((tp/'vocab.json').read_text(encoding='utf-8'));tokens=[None]*len(vocab)
for text,index in vocab.items():tokens[index]=text
tok=Tokenizer(tokens,(tp/'merges.txt').read_text(encoding='utf-8').split('\n'),json.loads((tp/'token_type.json').read_text()));tpl=ChatTemplate(tp/'chat_template.jinja')
def ids(text):return tok.encode(tpl.render([{'role':'user','content':text}],enable_thinking=False),parse_special=True)
def decode(generated):
    text=tok.decode(generated)
    for tag in ('<|im_end|>','<|endoftext|>'):text=text.removesuffix(tag)
    p=OutputParser(thinking=False);return ''.join(x.text for x in p.feed(text)+p.finish() if x.kind=='content')
def write(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
gold=parity['plan']['expected'];args=V.set_arg(cfg['args'],'--prompt-cache-every',16384)+['--prompt-cache-tail']
plan={'exe_sha256':V.sha(R/'strata-qa-combined.exe'),'shipping_exe_sha256':V.sha(R/'strata-combined.exe'),'args':args,'contexts':[20000,131072,250000],'max_context':262144,'artificial_pressure':False,'quality_gate_prerequisites':['elastic038/parity-r1','checkpoint-tail038/parity-r1'],'method':'one process; cold recall, long response1024, branch recall, next turn1000-token appendix at each depth','maximum_seconds':2400,'this_is_operational_smoke_not_sota_or_full_quality_attestation':True}
result={'status':'running','plan':plan,'requests':[]};start=time.monotonic();cancel=threading.Event();stop=threading.Event();phase='start'
def save():write(OUT/'results.json',result)
def guard():assert time.monotonic()-start<2400 and psutil.virtual_memory().available>8*2**30
def monitor():
    with (OUT/'resources.jsonl').open('w',encoding='utf-8') as f:
        while not stop.is_set():
            row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'ram_available_gib':psutil.virtual_memory().available/2**30,'cpu_percent':psutil.cpu_percent(None)}
            try:row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW).strip()
            except Exception as exc:row['gpu_error']=repr(exc)
            if row['ram_available_gib']<8:cancel.set()
            f.write(json.dumps(row)+'\n');f.flush();stop.wait(10)
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(cfg.get('env',{}))
env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(BASE/'elastic038_20261003/parity-r1/cuda-cache'),STRATA_ELASTIC_TRACE='1',STRATA_TEST_FIRST_LOGITS=str(OUT/'first.f32'))
env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
assert not any('strata' in (p.info['name'] or '').lower() and (p.info['name'] or '').lower().endswith('.exe') for p in psutil.process_iter(['name']))
e=StrataEngine(str(R/'strata-qa-combined.exe'),args,cwd=str(OUT),log=str(OUT/'engine.log'),env=env,lazy=True)
def request(name,prompt,max_new,expected=None):
    global phase
    phase=name;guard();assert len(prompt)+max_new+8<262144
    (OUT/'first.f32').unlink(missing_ok=True);t0=time.perf_counter();first=None;generated=[]
    for t in e.generate(prompt,max_new,{'temperature':0},cancel):
        if t is not None:
            if first is None:first=time.perf_counter()-t0
            generated.append(t)
    row=np.fromfile(OUT/'first.f32',dtype='<f4');assert row.size==len(tokens) and np.isfinite(row).all();row.tofile(OUT/f'{name}-first.f32')
    text=decode(generated);item={'name':name,'input_tokens':len(prompt),'max_new':max_new,'ttft_s':first,'wall_s':time.perf_counter()-t0,'output_ids':generated,'content':text,'first_logits_finite':True,**e.last}
    if expected is not None:
        clean=text.strip()
        if clean.startswith('```json\n') and clean.endswith('\n```'):clean=clean[8:-4].strip()
        item['oracle_pass']=V.evaluate_answer(clean,e.last['finish'],expected)
    result['requests'].append(item);save();print('ELASTIC REQUEST',name,'reuse',e.last.get('reused'),'oracle',item.get('oracle_pass'),'TTFT',round(first or 0,3),flush=True)
    return generated
fill=tok.encode(V.TEXTS['doc'][1],parse_special=False)
rendered=tpl.render([{'role':'user','content':'Read the archive. Only AUTH_RECORD lines define the requested codes.\n__ARCHIVE__\nReturn only JSON with keys north, south, west containing the codes.'}],enable_thinking=False)
left,right=rendered.split('__ARCHIVE__');left=tok.encode(left,parse_special=True);right=tok.encode(right,parse_special=True)
records=[tok.encode('\nAUTH_RECORD '+k+'='+v+'\n',parse_special=False) for k,v in gold.items()]
n=20000-len(left)-len(right)-sum(map(len,records));prefix=left[:]
for i,c in enumerate((n//10,n*4//10,n*4//10,n-(n//10+n*4//10+n*4//10))):
    prefix+=(fill*(c//len(fill)+1))[:c]
    if i<3:prefix+=records[i]
fixtures={20000:{'fresh':prefix+right,'branch':prefix+tok.encode('\nSECOND appendix.\n',parse_special=False)+right}}
for d in (131072,250000):fixtures[d]={n:json.loads((BASE/'checkpoint_daily_20261003/paired-r2'/f'{d}-{n}.ids.json').read_text()) for n in ('fresh','branch')}
guide=tok.encode('\n'+tpl.render([{'role':'user','content':'Write a long numbered guide for operating this archive safely. Give at least twenty detailed steps with examples.'}],enable_thinking=False),parse_special=True)
next_ids=(fill*(1000//len(fill)+1))[:1000]
follow=tok.encode('\n'+tpl.render([{'role':'user','content':tok.decode(next_ids)+'\nUsing only earlier AUTH_RECORD entries, return only JSON with keys north, south, west and their codes.'}],enable_thinking=False),parse_special=True)
save();thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
    e.restart();result['startup_info']=dict(e.info);save();assert e.max_context==262144
    list(e.generate(ids('Return only JSON with key ready and value true.'),32,{'temperature':0},cancel))
    for depth in plan['contexts']:
        f=fixtures[depth];assert len(f['fresh'])==depth
        for name,p in f.items():write(OUT/f'{depth}-{name}.ids.json',p)
        request(f'{depth}-fresh',f['fresh'],128,gold)
        request(f'{depth}-long',f['fresh']+guide,1024)
        generated=request(f'{depth}-branch',f['branch'],128,gold)
        request(f'{depth}-next',f['branch']+generated+follow,128,gold)
    log=(OUT/'engine.log').read_text(encoding='utf-8',errors='replace')
    result['elastic_events']=[l for l in log.splitlines() if 'strata elastic:' in l]
    result['chunk_events']=[l for l in log.splitlines() if 'prompt chunk' in l or 'prefill auto' in l or 'prompt path borrows' in l]
    result['oracle_pass']=all(x.get('oracle_pass',True) for x in result['requests'])
    result.update(status='completed',elapsed_s=time.monotonic()-start)
except BaseException as exc:result.update(status='failed',error=repr(exc),elapsed_s=time.monotonic()-start);raise
finally:e.close();stop.set();thread.join(timeout=12);save();print('ELASTIC STATUS',result['status'],flush=True)
