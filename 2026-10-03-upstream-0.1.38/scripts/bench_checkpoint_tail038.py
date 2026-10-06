"""Three interleaved shipping-executable pairs, frozen33.8k branch fixture."""
from pathlib import Path
import os,sys,json,time,threading,statistics,subprocess,datetime
BASE=Path(__file__).resolve().parent;R=BASE/'checkpoint-tail038_20261003';OLD=BASE/'checkpoint038_20261003'
sys.path[:0]=[str(R/'source'),r'D:\ds4_work\strata\Strata\.venv\Lib\site-packages',r'C:\Users\imanu\source\repos\Strata-2080Ti\tools\sm75']
import psutil,validate_model as V
from serve.server import StrataEngine
from serve.frontend import OutputParser
from tools.strata_tokenizer import Tokenizer
quality=json.loads((R/'parity-r1/results.json').read_text());assert quality['status']=='completed' and quality['gate_pass'] and quality['oracle_pass']
OUT=R/'timing-r1';OUT.mkdir(exist_ok=False)
cfg=json.loads(Path(r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json').read_text(encoding='utf-8-sig'))
tp=Path(cfg['tokenizer']);vocab=json.loads((tp/'vocab.json').read_text(encoding='utf-8'));tokens=[None]*len(vocab)
for text,index in vocab.items():tokens[index]=text
tok=Tokenizer(tokens,(tp/'merges.txt').read_text(encoding='utf-8').split('\n'),json.loads((tp/'token_type.json').read_text(encoding='utf-8')))
from serve.frontend import ChatTemplate
warm=tok.encode(ChatTemplate(tp/'chat_template.jinja').render([{'role':'user','content':'Return only JSON with key ready and value true.'}],enable_thinking=False),parse_special=True)
fixtures={n:json.loads((OLD/'model-tests-r3'/f'{n}.ids.json').read_text()) for n in ('fresh','follow')}
plan={'args':quality['plan']['args'],'candidate_flags':['--prompt-cache-tail'],'pairs':3,'order':['A1','B1','B2','A2','A3','B3'],'fresh_tokens':len(fixtures['fresh']),'follow_tokens':len(fixtures['follow']),'expected':quality['plan']['expected'],'diagnostic_readout':False,'artificial_pressure':False,'kv_capacity':262144,'maximum_seconds':1800,'paired_branch_fixture_not_general_sota':True}
result={'status':'running','plan':plan,'arms':{}};started=time.monotonic();stop=threading.Event();cancel=threading.Event();phase='start'
def save():(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
def guard():assert time.monotonic()-started<1800 and psutil.virtual_memory().available>8*2**30
def monitor():
    with (OUT/'resources.jsonl').open('w',encoding='utf-8') as f:
        while not stop.is_set():
            row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'ram_available_gib':psutil.virtual_memory().available/2**30,'cpu_percent':psutil.cpu_percent(None)}
            try:row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW).strip()
            except Exception as exc:row['gpu_error']=repr(exc)
            f.write(json.dumps(row)+'\n');f.flush();stop.wait(10)
assert not any('strata' in (p.info['name'] or '').lower() and (p.info['name'] or '').lower().endswith('.exe') for p in psutil.process_iter(['name']))
thread=threading.Thread(target=monitor,daemon=True);thread.start();save()
try:
    for label in plan['order']:
        guard();phase=label;mode=label[0];folder=OUT/label;folder.mkdir()
        exe=OLD/'strata-baseline.exe' if mode=='A' else R/'strata-tail.exe';args=plan['args']+(plan['candidate_flags'] if mode=='B' else [])
        env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_CACHE_PATH=str(R/'parity-r1/cuda-cache'),STRATA_WATCHDOG_S='240',STRATA_PARALLEL_INTERMEDIATE_QUANT='0');env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
        e=StrataEngine(str(exe),args,cwd=str(folder),log=str(folder/'engine.log'),env=env,lazy=True)
        arm={'mode':mode,'pair':int(label[1]),'exe_sha256':V.sha(exe),'args':args,'requests':[]};result['arms'][label]=arm;save();print('TIMING START',label,flush=True)
        try:
            e.restart();arm['info']=dict(e.info);assert e.max_context==262144 and int(e.info['expert_slots'])==5357
            list(e.generate(warm,32,{'temperature':0},cancel))
            for name,ids in fixtures.items():
                guard();t0=time.perf_counter();first=None;generated=[]
                for token in e.generate(ids,128,{'temperature':0},cancel):
                    if token is not None:
                        if first is None:first=time.perf_counter()-t0
                        generated.append(token)
                text=tok.decode(generated)
                for tag in ('<|im_end|>','<|endoftext|>'):text=text.removesuffix(tag)
                p=OutputParser(thinking=False);content=''.join(x.text for x in p.feed(text)+p.finish() if x.kind=='content')
                passed=V.evaluate_answer(content,e.last['finish'],plan['expected']);arm['requests'].append({'name':name,'ttft_s':first,'wall_s':time.perf_counter()-t0,'output_ids':generated,'content':content,'oracle_pass':passed,**e.last});save();assert passed
                print('TIMING RESULT',label,name,'reuse',e.last['reused'],'TTFT',round(first or 0,3),flush=True)
            arm['completed']=True;save()
        finally:e.close()
    summary={}
    for name in ('fresh','follow'):
        values={m:[next(x for x in a['requests'] if x['name']==name)['ttft_s'] for a in result['arms'].values() if a['mode']==m] for m in ('A','B')}
        a,b=[statistics.median(values[m]) for m in ('A','B')];summary[name]={'values_s':values,'median_A_s':a,'median_B_s':b,'candidate_delta_pct':100*(b/a-1),'reference_spread_pct':100*(max(values['A'])-min(values['A']))/a}
    result.update(status='completed',elapsed_s=time.monotonic()-started,summary=summary,oracle_pass=True)
except BaseException as exc:result.update(status='failed',error=repr(exc),elapsed_s=time.monotonic()-started);raise
finally:stop.set();thread.join(timeout=12);save();print('TIMING STATUS',result['status'],flush=True)
