from pathlib import Path
import sys,json,hashlib,subprocess,datetime
sys.path.insert(0,r'D:\ds4_work\strata\Strata\.venv\Lib\site-packages')
import psutil
BASE=Path(__file__).resolve().parent
expected={
r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json':'ccffa537cdb264f99317948c1c6f41ffbc1e5ce370cafb1ca8c0ca717e77e112',
r'D:\ds4_work\strata\Strata\build-0136-daily\strata.exe':'c35505ca79b790bd37ed7073fcadd399dd30d1b88e85bb99a6eb24a52d945d0a',
r'D:\ds4_work\strata\Strata\build-0136-daily\source\serve\server.py':'73f90f417e9207811fdc47e5e6240f69bbde0f61ff027b9e0d8cb7b7bb2f4d81',
r'C:\Users\imanu\Desktop\SERVER IA LOCALE\Strata 2080 Ti - SOTA 30-9 - porta 8100.bat':'1103f2ffb7727aafe1754f964c1969a03c0838a659edfc5984b1f4afbd2bd2c4',
r'D:\ds4_work\strata\Strata\data\expert-profile.bin':'8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf'}
actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in expected}
assert actual==expected
engines=[p.info for p in psutil.process_iter(['name','pid']) if 'strata' in (p.info['name'] or '').lower() and (p.info['name'] or '').endswith('.exe')]
assert not engines,engines
repos={}
for p in (r'C:\Users\imanu\source\repos\moe-aggressive-commit',r'C:\Users\imanu\source\repos\Strata-2080Ti',r'D:\ds4_work\strata\Strata'):
    def git(*args):return subprocess.check_output(['git',*args],cwd=p,text=True).strip()
    branch=git('branch','--show-current');status=git('status','--short','--branch')
    if 'moe-aggressive' not in p:assert branch=='main'
    repos[p]={'branch':branch,'status':status,'head':git('rev-parse','HEAD')}
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
out={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'daily_hashes':actual,'daily_unchanged':True,'repositories':repos,'strata_engines':engines,'available_ram_gib':psutil.virtual_memory().available/2**30,'gpu_csv':gpu,'artificial_pressure':False}
(BASE/'elastic-tail-final-state.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'daily_unchanged':True,'strata_engines':engines,'available_ram_gib':out['available_ram_gib'],'gpu':gpu}))
