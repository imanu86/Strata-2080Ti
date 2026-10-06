from pathlib import Path
import os,sys,subprocess,json,hashlib,shutil,difflib
BASE=Path(__file__).resolve().parent;R=BASE/'elastic038_20261003';S=R/'source'
final='--final' in sys.argv
VC=Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC')
CM=Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake')
env=dict(os.environ);env.pop('STRATA_API_KEY',None)
init=subprocess.run(f'cmd.exe /d /s /c "call "{VC / "Auxiliary/Build/vcvars64.bat"}" >nul && set"',capture_output=True,text=True,encoding='cp1252',errors='replace',env=env,creationflags=subprocess.CREATE_NO_WINDOW);assert init.returncode==0
for line in init.stdout.splitlines():
    if '=' in line and not line.startswith('='):
        k,v=line.split('=',1);env[k]=v
env.pop('STRATA_API_KEY',None);env['PYTHONDONTWRITEBYTECODE']='1'
def run(args,name):
    print(name,flush=True)
    with (R/(name+'.log')).open('w',encoding='utf-8') as f:p=subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,env=env,cwd=R,timeout=1800,creationflags=subprocess.CREATE_NO_WINDOW)
    if p.returncode:print((R/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-6000:])
    assert p.returncode==0,name
run([CM/'CMake/bin/cmake.exe','--build',R/'build','--parallel','2','--target','strata','expert_cache_elastic_test','expert_cache_per_layer_test','expert_profile_save_test'],'build-shipping-tests-final' if final else 'build-shipping-tests')
exe=R/('strata-elastic-final.exe' if final else 'strata-elastic.exe');assert not exe.exists();shutil.copy2(R/'build/strata.exe',exe)
cfg=json.loads(Path(r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json').read_text(encoding='utf-8-sig'))
env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
run([CM/'CMake/bin/ctest.exe','--test-dir',R/'build','--output-on-failure','-R','^(expert_cache_elastic_test|expert_cache_per_layer_test|expert_profile_save_test)$'],'ctest-elastic-final-all' if final else 'ctest-elastic')
p=S/'src/program/generate.cpp';before=p.read_text(encoding='utf-8')
oldpatch=(BASE/'checkpoint038_20261003/qa-readout-A.patch').read_text(encoding='utf-8')
hook='\n'.join(l[1:] for l in oldpatch.splitlines() if l.startswith('+') and not l.startswith('+++'))+'\n'
assert hashlib.sha256(('\n'.join('+'+l for l in hook.splitlines())).encode()).hexdigest()=='55f32dc38f80de208a1f1f0c6c89c43908d12b815126f0ae3a3c03738972b7ef'
needle='                first_window = false;\n                bool eos = false;';assert before.count(needle)==1
after=before.replace(needle,hook+needle);diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/src/program/generate.cpp',tofile='b/src/program/generate.cpp'))
(R/('qa-readout-final.patch' if final else 'qa-readout.patch')).write_text(diff,encoding='utf-8',newline='\n')
try:
    p.write_text(after,encoding='utf-8',newline='\n')
    run([CM/'CMake/bin/cmake.exe','--build',R/'build','--parallel','2','--target','strata'],'build-qa-elastic-final' if final else 'build-qa-elastic')
    qa=R/('strata-qa-elastic-final.exe' if final else 'strata-qa-elastic.exe');assert not qa.exists();shutil.copy2(R/'build/strata.exe',qa)
    (R/('qa-build-final.json' if final else 'qa-build.json')).write_text(json.dumps({'sha256':hashlib.sha256(qa.read_bytes()).hexdigest(),'shipping_exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'shipping_patch_separate':True,'readout_added_lines_sha256':'55f32dc38f80de208a1f1f0c6c89c43908d12b815126f0ae3a3c03738972b7ef','same_readout_as_pristine_A':True},indent=2))
finally:
    p.write_text(before,encoding='utf-8',newline='\n')
    shutil.copy2(exe,R/'build/strata.exe')
print('Shipping, functional tests and QA ready',flush=True)
