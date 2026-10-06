"""Rebuild both Prefill-dependent TUs; reuse unchanged pinned CUDA libraries.

All outputs and dependency copies live in this new plain source laboratory.
The old checkpoint lab is read-only. No checkout, branch or worktree is created.
"""
from pathlib import Path
import os,subprocess,sys,shutil,json,hashlib,difflib
BASE=Path(__file__).resolve().parent
OLD=BASE/('elastic038_20261003' if '--combined' in sys.argv else 'checkpoint038_20261003')
R=BASE/('elastic-tail038_final_20261003' if '--final' in sys.argv else ('elastic-tail038_20261003' if '--combined' in sys.argv else 'checkpoint-tail038_20261003'))
B=R/'build';B.mkdir(exist_ok=False)
VC=Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC');CM=Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake')
env=dict(os.environ);env.pop('STRATA_API_KEY',None)
p=subprocess.run(f'cmd.exe /d /s /c "call "{VC/"Auxiliary/Build/vcvars64.bat"}" >nul && set"',capture_output=True,text=True,encoding='cp1252',errors='replace',env=env,creationflags=subprocess.CREATE_NO_WINDOW);assert p.returncode==0
for l in p.stdout.splitlines():
    if '=' in l and not l.startswith('='):k,v=l.split('=',1);env[k]=v
env.pop('STRATA_API_KEY',None)
deps=['strata_engine.lib','strata_kernels.lib','strata_core.lib','strata_kernels_cpu.lib','strata_spec.lib','strata_mmq.lib','ggml/src/ggml-cpu.lib','ggml/src/ggml-base.lib','CMakeFiles/strata_prefill.dir/src/prefill/gemm.cu.obj','CMakeFiles/strata_prefill.dir/src/prefill/kernels.cu.obj']
meta={'method':'original Ninja compile/link commands; both Prefill-dependent C++ TUs rebuilt; unchanged CUDA/engine dependencies copied read-only','dependency_hashes':{},'commands':[]}
for rel in deps:
    target=B/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OLD/'build'/rel,target)
    meta['dependency_hashes'][rel]=hashlib.sha256(target.read_bytes()).hexdigest()
(B/'CMakeFiles/strata.dir/src/program').mkdir(parents=True,exist_ok=True)
cmds=subprocess.check_output([str(CM/'Ninja/ninja.exe'),'-C',str(OLD/'build'),'-t','commands','strata'],text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW).splitlines()
selected=[next(c for c in cmds if '/FoCMakeFiles\\strata_prefill.dir\\src\\prefill\\prefill.cpp.obj' in c),next(c for c in cmds if '/FoCMakeFiles\\strata.dir\\src\\program\\generate.cpp.obj' in c),next(c for c in cmds if '/out:strata_prefill.lib' in c),next(c for c in cmds if '/out:strata.exe' in c)]
selected=[c.replace(str(OLD/'source'),str(R/'source')) for c in selected];meta['commands']=selected
def run(c,name):
    with (R/(name+'.log')).open('w',encoding='utf-8') as f:p=subprocess.run(c,stdout=f,stderr=subprocess.STDOUT,cwd=B,env=env,timeout=600,creationflags=subprocess.CREATE_NO_WINDOW)
    if p.returncode:print((R/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-5000:])
    assert p.returncode==0,name
print('Rebuilding both Prefill-dependent C++ files',flush=True)
for i,c in enumerate(selected):run(c,f'build-{i}')
shipping=R/('strata-combined.exe' if '--combined' in sys.argv else 'strata-tail.exe');shutil.copy2(B/'strata.exe',shipping);meta['shipping_sha256']=hashlib.sha256(shipping.read_bytes()).hexdigest()
p=R/'source/src/program/generate.cpp';before=p.read_text(encoding='utf-8')
patch=(BASE/'checkpoint038_20261003/qa-readout-A.patch').read_text(encoding='utf-8');hook='\n'.join(l[1:] for l in patch.splitlines() if l.startswith('+') and not l.startswith('+++'))+'\n'
needle='                first_window = false;\n                bool eos = false;';assert before.count(needle)==1
after=before.replace(needle,hook+needle);(R/'qa-readout.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/src/program/generate.cpp',tofile='b/src/program/generate.cpp')),encoding='utf-8',newline='\n')
try:
    p.write_text(after,encoding='utf-8',newline='\n');run(selected[1],'build-qa-generate');run(selected[3],'build-qa-link')
    qa=R/('strata-qa-combined.exe' if '--combined' in sys.argv else 'strata-qa-tail.exe');shutil.copy2(B/'strata.exe',qa);meta['qa_sha256']=hashlib.sha256(qa.read_bytes()).hexdigest()
finally:p.write_text(before,encoding='utf-8',newline='\n');shutil.copy2(shipping,B/'strata.exe')
(R/'build.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('Tail shipping and common QA readout built',flush=True)
