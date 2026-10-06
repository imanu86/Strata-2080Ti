"""Tail checkpoint alternative: pristine A, optional tail checkpoint B, pristine repeat.

Preserves checkpoint cadence, expert placement, chunk geometry and speculative
settings. Complete frozen corpora and complete recall answers, real model only.
"""
from pathlib import Path
import os,sys,json,threading,time,hashlib,datetime,subprocess
BASE=Path(__file__).resolve().parent; R=BASE/'checkpoint-tail038_20261003'; OLD=BASE/'checkpoint038_20261003'
sys.path[:0]=[str(R/'source'),r'D:\ds4_work\strata\Strata\.venv\Lib\site-packages',r'C:\Users\imanu\source\repos\Strata-2080Ti\tools\sm75']
import numpy as np,psutil,validate_model as V
from serve.server import StrataEngine
from serve.frontend import ChatTemplate,OutputParser
from tools.strata_tokenizer import Tokenizer
OUT=R/'parity-r1';OUT.mkdir(exist_ok=False)
cfg=json.loads(Path(r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json').read_text(encoding='utf-8-sig'))
old=json.loads((OLD/'model-tests-r3/results.json').read_text());args=V.set_arg(old['plan']['args'],'--prefill',6144)
tp=Path(cfg['tokenizer']);vocab=json.loads((tp/'vocab.json').read_text(encoding='utf-8'));tokens=[None]*len(vocab)
for text,index in vocab.items():tokens[index]=text
tok=Tokenizer(tokens,(tp/'merges.txt').read_text(encoding='utf-8').split('\n'),json.loads((tp/'token_type.json').read_text()))
tpl=ChatTemplate(tp/'chat_template.jinja')
warm=tok.encode(tpl.render([{'role':'user','content':'Return only JSON with key ready and value true.'}],enable_thinking=False),parse_special=True)
fixtures={n:json.loads((OLD/'model-tests-r3'/f'{n}.ids.json').read_text()) for n in ('fresh','follow')}
texts={n:tok.encode(text*(count//64+2),parse_special=False)[:count] for n,(count,text) in V.TEXTS.items()}
gold=old['plan']['expected'];start=time.monotonic();cancel=threading.Event();stop=threading.Event();phase='start'
result={'status':'running','plan':{'upstream':'99f3dbd0b21d1401b3769e0c0d963913607f380b','args':args,'arms':['A1','B','A2'],'checkpoint_candidate':'one extra existing chunk boundary near prompt tail; no alignment cuts','candidate_flags':['--prompt-cache-tail'],'policy':V.POLICY,'expected':gold,'artificial_pressure':False,'maximum_seconds':10800,'scored_positions_per_arm':2557,'kv_capacity':262144,'checkpoint_every':16384,'prefill_chunk':6144,'tests_are_local_regression_not_upstream_universal_standard':True,'original_grid_gate':'FAIL preserved; this is a separate replacement design' },'arms':{}}
def save():(OUT/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
def guard():
    assert time.monotonic()-start<result['plan']['maximum_seconds'],'deadline'
    assert psutil.virtual_memory().available>8*2**30,'RAM reserve'
def decode(ids):
    text=tok.decode(ids)
    for tag in ('<|im_end|>','<|endoftext|>'):text=text.removesuffix(tag)
    parser=OutputParser(thinking=False)
    return ''.join(x.text for x in parser.feed(text)+parser.finish() if x.kind=='content')
def read(folder):
    row=np.fromfile(folder/'first.f32',dtype='<f4');assert row.size==len(tokens) and np.isfinite(row).all();return row
def monitor():
    with (OUT/'resources.jsonl').open('w',encoding='utf-8') as f:
        while not stop.is_set():
            row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,'ram_available_gib':psutil.virtual_memory().available/2**30,'cpu_percent':psutil.cpu_percent(None)}
            try:row['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True,timeout=8,creationflags=subprocess.CREATE_NO_WINDOW).strip()
            except Exception as exc:row['gpu_error']=repr(exc)
            f.write(json.dumps(row)+'\n');f.flush();stop.wait(10)
assert not any('strata' in (p.info['name'] or '').lower() and (p.info['name'] or '').lower().endswith('.exe') for p in psutil.process_iter(['name']))
qa=json.loads((R/'build.json').read_text());assert V.sha(OLD/'strata-qa-A.exe')=='cb6ba7618f71ed509322d2339c57eafdc1c57ea4a44549dc641cc2ee2b678bed'
assert V.sha(R/'strata-qa-tail.exe')==qa['qa_sha256']
for n,ids in texts.items():(OUT/f'{n}.ids.json').write_text(json.dumps(ids))
save();thread=threading.Thread(target=monitor,daemon=True);thread.start()
try:
    for label in result['plan']['arms']:
        guard();phase=label;folder=OUT/label;folder.mkdir()
        exe=OLD/'strata-qa-A.exe' if label in ('A1','A2') else R/'strata-qa-tail.exe'
        current=args+(result['plan']['candidate_flags'] if label=='B' else [])
        env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')}
        env.update(PYTHONDONTWRITEBYTECODE='1',STRATA_WATCHDOG_S='240',STRATA_PARALLEL_INTERMEDIATE_QUANT='0',STRATA_TEST_FIRST_LOGITS=str(folder/'first.f32'),CUDA_CACHE_PATH=str(OUT/'cuda-cache'))
        env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
        engine=StrataEngine(str(exe),current,cwd=str(folder),log=str(folder/'engine.log'),env=env,lazy=True)
        arm={'exe_sha256':V.sha(exe),'args':current,'requests':[],'teacher':{}};result['arms'][label]=arm;save();print('PARITY START',label,flush=True)
        try:
            engine.restart();arm['info']=dict(engine.info);save()
            assert engine.max_context==262144 and int(engine.info['expert_slots'])==5357
            list(engine.generate(warm,32,{'temperature':0},cancel))
            for name,ids in fixtures.items():
                guard();(folder/'first.f32').unlink(missing_ok=True);t0=time.perf_counter();first=None;generated=[]
                for token in engine.generate(ids,128,{'temperature':0},cancel):
                    if token is not None:
                        if first is None:first=time.perf_counter()-t0
                        generated.append(token)
                read(folder).tofile(folder/f'{name}-first.f32');text=decode(generated)
                passed=V.evaluate_answer(text,engine.last['finish'],gold)
                arm['requests'].append({'name':name,'ttft_s':first,'wall_s':time.perf_counter()-t0,'output_ids':generated,'content':text,'oracle_pass':passed,**engine.last});save()
                print('PARITY RECALL',label,name,'pass',passed,'reuse',engine.last.get('reused'),flush=True)
            for corpus,ids in texts.items():
                phase=label+'/'+corpus;raw=folder/f'{corpus}.logits.bin';reused=[]
                with raw.open('wb') as f:
                    f.write(np.asarray([len(tokens),len(ids)],dtype='<i4').tobytes())
                    for pos in range(len(ids)):
                        guard();(folder/'first.f32').unlink(missing_ok=True)
                        generated=[t for t in engine.generate(fixtures['follow']+ids[:pos+1],1,{'temperature':0},cancel) if t is not None]
                        assert len(generated)==1;f.write(read(folder).tobytes());reused.append(engine.last['reused'])
                        if (pos+1)%128==0:print('PARITY ROWS',label,corpus,pos+1,flush=True)
                V.check_dump(raw,len(ids),len(tokens));arm['teacher'][corpus]={'rows':len(ids),'sha256':V.sha(raw),'reused_min':min(reused),'reused_max':max(reused)};save()
            log=(folder/'engine.log').read_text(encoding='utf-8',errors='replace')
            arm['elastic_events']=[l for l in log.splitlines() if 'strata elastic:' in l]
            assert not any('strata elastic: +' in l or 'strata elastic: -' in l for l in log.splitlines()),'tail-only arm unexpectedly contains elastic code'
            arm['completed']=True;save()
        finally:engine.close()
        print('PARITY STOPPED',label,flush=True)
    def compare(left,right):
        stats={}
        for corpus,ids in texts.items():
            a=V.check_dump(OUT/left/f'{corpus}.logits.bin',len(ids),len(tokens));b=V.check_dump(OUT/right/f'{corpus}.logits.bin',len(ids),len(tokens))
            stats[corpus]=V.paired_stats(a,b,ids);stats[corpus]['bitwise_identical']=bool(np.array_equal(a,b));del a,b
        return stats
    repeat=compare('A1','A2');result['reference_repeat']={n:V.public_stats(s) for n,s in repeat.items()}
    result['comparisons']={}
    for mode in ('B',):
        stats=compare('A1',mode);result['comparisons'][mode]={n:{'stats':V.public_stats(s),'gate':V.regression_gate(s,repeat[n])} for n,s in stats.items()}
    result['gate_pass']=all(x['gate']['pass'] for d in result['comparisons'].values() for x in d.values())
    result['oracle_pass']=all(r['oracle_pass'] for a in result['arms'].values() for r in a['requests'])
    result['status']='completed';result['elapsed_s']=time.monotonic()-start
except BaseException as exc:result.update(status='failed',error=repr(exc),elapsed_s=time.monotonic()-start);raise
finally:stop.set();thread.join(timeout=12);save();print('PARITY STATUS',result['status'],flush=True)
