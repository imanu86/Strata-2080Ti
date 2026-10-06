"""Proper assistant turn closure, distinct next-turn oracles and deep long decode.

The previous operational smoke's deep guide requests omitted the assistant
answer/end marker. Its48-token outputs are retained and not called long decode.
"""
from pathlib import Path
import os,sys,json,time,threading,datetime,subprocess
BASE=Path(__file__).resolve().parent;R=BASE/'elastic-tail038_final_20261003'
sys.path[:0]=[str(R/'source'),r'D:\ds4_work\strata\Strata\.venv\Lib\site-packages',r'C:\Users\imanu\source\repos\Strata-2080Ti\tools\sm75']
import numpy as np,psutil,validate_model as V
from serve.server import StrataEngine
from serve.frontend import ChatTemplate,OutputParser
from tools.strata_tokenizer import Tokenizer
prior=json.loads((R/'operational-r1/results.json').read_text());assert prior['status']=='completed' and prior['oracle_pass']
OUT=R/'dialogue-r1';OUT.mkdir(exist_ok=False)
cfg=json.loads(Path(r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json').read_text(encoding='utf-8-sig'))
tp=Path(cfg['tokenizer']);vocab=json.loads((tp/'vocab.json').read_text(encoding='utf-8'));tokens=[None]*len(vocab)
for text,index in vocab.items():tokens[index]=text
tok=Tokenizer(tokens,(tp/'merges.txt').read_text(encoding='utf-8').split('\n'),json.loads((tp/'token_type.json').read_text(encoding='utf-8')));tpl=ChatTemplate(tp/'chat_template.jinja');eot=tok.encode('<|im_end|>',parse_special=True);assert len(eot)==1
def next_user(text):return tok.encode('\n'+tpl.render([{'role':'user','content':text}],enable_thinking=False),parse_special=True)
def close(prompt,generated):return prompt+generated+([] if generated and generated[-1]==eot[0] else eot)
def decode(generated):
    text=tok.decode(generated)
    for tag in ('<|im_end|>','<|endoftext|>'):text=text.removesuffix(tag)
    p=OutputParser(thinking=False);return ''.join(x.text for x in p.feed(text)+p.finish() if x.kind=='content')
args=prior['plan']['args'];result={'status':'running','plan':{'args':args,'exe_sha256':V.sha(R/'strata-qa-combined.exe'),'contexts':[131072,250000],'method':'append actual generated assistant answer, close im_end, append new user; distinct marker oracle before and after long decode','max_new_guide':1024,'artificial_pressure':False,'prior48_token_guides':'kept as short instruction-following smoke, not a long-decode pass'},'requests':[]};start=time.monotonic();cancel=threading.Event();stop=threading.Event();phase='start'
def save():(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
def monitor():
    with (OUT/'resources.jsonl').open('w',encoding='utf-8') as f:
        while not stop.is_set():
            row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'ram_available_gib':psutil.virtual_memory().available/2**30,'cpu_percent':psutil.cpu_percent(None)}
            try:row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW).strip()
            except Exception as exc:row['gpu_error']=repr(exc)
            if row['ram_available_gib']<8:cancel.set()
            f.write(json.dumps(row)+'\n');f.flush();stop.wait(10)
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(cfg.get('env',{}));env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(BASE/'elastic038_20261003/parity-r1/cuda-cache'),STRATA_ELASTIC_TRACE='1',STRATA_TEST_FIRST_LOGITS=str(OUT/'first.f32'));env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
e=StrataEngine(str(R/'strata-qa-combined.exe'),args,cwd=str(OUT),log=str(OUT/'engine.log'),env=env,lazy=True)
def request(name,prompt,max_new,expected=None):
    global phase
    phase=name;assert time.monotonic()-start<2400 and psutil.virtual_memory().available>8*2**30 and len(prompt)+max_new+8<262144
    (OUT/'first.f32').unlink(missing_ok=True);t0=time.perf_counter();first=None;generated=[]
    for t in e.generate(prompt,max_new,{'temperature':0},cancel):
        if t is not None:
            if first is None:first=time.perf_counter()-t0
            generated.append(t)
    row=np.fromfile(OUT/'first.f32',dtype='<f4');assert row.size==len(tokens) and np.isfinite(row).all();row.tofile(OUT/f'{name}-first.f32');text=decode(generated)
    item={'name':name,'input_tokens':len(prompt),'output_tokens':len(generated),'output_ids':generated,'content':text,'ttft_s':first,'wall_s':time.perf_counter()-t0,**e.last}
    if expected is not None:
        clean=text.strip()
        if clean.startswith('```json\n') and clean.endswith('\n```'):clean=clean[8:-4].strip()
        item['oracle_pass']=V.evaluate_answer(clean,e.last['finish'],expected)
    result['requests'].append(item);save();print('DIALOGUE',name,'out',len(generated),'oracle',item.get('oracle_pass'),'finish',e.last['finish'],flush=True)
    if expected is not None:assert item['oracle_pass'],name
    return generated
assert not any('strata' in (p.info['name'] or '').lower() and (p.info['name'] or '').lower().endswith('.exe') for p in psutil.process_iter(['name']))
save();thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
    e.restart();result['startup_info']=dict(e.info);save()
    list(e.generate(tok.encode(tpl.render([{'role':'user','content':'Return only JSON with key ready and value true.'}],enable_thinking=False),parse_special=True),32,{'temperature':0},cancel))
    gold=json.loads((BASE/'elastic038_20261003/parity-r1/results.json').read_text())['plan']['expected']
    for depth in (131072,250000):
        fresh=json.loads((BASE/'checkpoint_daily_20261003/paired-r2'/f'{depth}-fresh.ids.json').read_text())
        generated=request(f'{depth}-fresh',fresh,128,gold);history=close(fresh,generated)
        marker='TAIL-CHECK-'+str(depth);prompt=history+next_user('For this new turn ignore the archive question. Return only JSON with key marker and literal value '+marker+'.')
        generated=request(f'{depth}-marker-before',prompt,128,{'marker':marker});history=close(prompt,generated)
        guide=history+next_user('For this new turn write a detailed operating manual, not JSON. Give eighty numbered steps about safe archive operations, each with four explanatory sentences and an example. Begin with step1. Continue through all80steps; target more than3000words.')
        generated=request(f'{depth}-long',guide,1024);history=close(guide,generated)
        final='AFTER-DECODE-'+str(depth);prompt=history+next_user('For this new turn return only JSON with key marker and literal value '+final+'.')
        request(f'{depth}-marker-after',prompt,128,{'marker':final})
    result['long_decode_pass']=all(x['output_tokens']==1024 and x['finish']=='length' for x in result['requests'] if x['name'].endswith('-long'))
    result['oracle_pass']=all(x.get('oracle_pass',True) for x in result['requests']);result.update(status='completed',elapsed_s=time.monotonic()-start)
except BaseException as exc:result.update(status='failed',error=repr(exc),elapsed_s=time.monotonic()-start);raise
finally:e.close();stop.set();thread.join(timeout=12);save();print('DIALOGUE STATUS',result['status'],flush=True)
