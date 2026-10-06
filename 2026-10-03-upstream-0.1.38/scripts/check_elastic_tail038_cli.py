from pathlib import Path
import os,json,subprocess,hashlib,sys
BASE=Path(__file__).resolve().parent;pure='--pure' in sys.argv;R=BASE/('elastic038_20261003' if pure else 'elastic-tail038_final_20261003')
cfg=json.loads(Path(r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json').read_text(encoding='utf-8-sig'))
env={k:v for k,v in os.environ.items() if not k.startswith('STRATA_')};env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
exe=R/('strata-elastic-final.exe' if pure else 'strata-combined.exe');cases=[]
flags=['--elastic-reserve-mib','--elastic-chunk-mib','--elastic-period-ms','--elastic-stable-ms','--elastic-tail-mib','--elastic-shrink-step-mib']
for flag in flags:
    bad=['x','1junk','2147483648','-2' if flag.endswith('tail-mib') else '-1']
    if flag in ('--elastic-chunk-mib','--elastic-shrink-step-mib'):bad+=['0']
    for value in bad:
        p=subprocess.run([str(exe),flag,value,'--help'],capture_output=True,text=True,env=env,cwd=R,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
        assert p.returncode==2 and p.stderr.startswith(flag+' needs an integer'),(flag,value,p.returncode,p.stderr[:200])
        cases.append({'flag':flag,'value':value,'expected':2,'actual':p.returncode})
    good=['-1','0','1'] if flag.endswith('tail-mib') else (['1','64'] if flag in ('--elastic-chunk-mib','--elastic-shrink-step-mib') else ['0','1','2147483647'])
    for value in good:
        p=subprocess.run([str(exe),flag,value,'--help'],capture_output=True,text=True,env=env,cwd=R,timeout=15,creationflags=subprocess.CREATE_NO_WINDOW)
        assert p.returncode==0 and '--elastic-shrink-step-mib' in p.stdout+p.stderr,(flag,value,p.returncode,p.stderr)
        cases.append({'flag':flag,'value':value,'expected':0,'actual':p.returncode})
result={'status':'completed','exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'checks':len(cases),'cases':cases,'method':'parser then --help; no model or pressure allocation','inference_body':'unchanged by CLI polish'}
(R/'cli-tests.json').write_text(json.dumps(result,indent=2));print('CLI validation',len(cases),'checks passed')
