from pathlib import Path
import os,subprocess,sys
BASE=Path(__file__).resolve().parent; R=BASE/'elastic038_20261003'
VC=Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC')
CUDA=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6')
CM=Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake')
env=dict(os.environ); env.pop('STRATA_API_KEY',None)
init=subprocess.run(f'cmd.exe /d /s /c "call "{VC / "Auxiliary/Build/vcvars64.bat"}" >nul && set"',capture_output=True,text=True,encoding='cp1252',errors='replace',env=env,creationflags=subprocess.CREATE_NO_WINDOW)
assert init.returncode==0
for line in init.stdout.splitlines():
    if '=' in line and not line.startswith('='):
        k,v=line.split('=',1); env[k]=v
env.pop('STRATA_API_KEY',None);env['PYTHONDONTWRITEBYTECODE']='1'
def run(args,name):
    print(name,flush=True)
    with (R/(name+'.log')).open('w',encoding='utf-8') as log:
        p=subprocess.run([str(x) for x in args],stdout=log,stderr=subprocess.STDOUT,env=env,cwd=R,timeout=1800,creationflags=subprocess.CREATE_NO_WINDOW)
    print(name,p.returncode,flush=True)
    if p.returncode:
        print((R/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-7000:]);sys.exit(p.returncode)
if '--configure' in sys.argv:
    run([CM/'CMake/bin/cmake.exe','-S',R/'source','-B',R/'build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_MAKE_PROGRAM={CM/"Ninja/ninja.exe"}',f'-DCMAKE_CUDA_COMPILER={CUDA/"bin/nvcc.exe"}','-DCMAKE_CUDA_ARCHITECTURES=75','-DCMAKE_CUDA_FLAGS=-D_WINDOWS -Xcompiler=/EHsc','-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared','-DSTRATA_ENABLE_CUDA=ON','-DSTRATA_ENABLE_HIP=OFF','-DSTRATA_BUILD_TESTS=ON','-DSTRATA_PARITY_PROMPT_ATTN=ON','-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_GGML_DIR=D:/ds4_work/strata/Strata/third_party/llama.cpp','-DFETCHCONTENT_FULLY_DISCONNECTED=ON',f'-DPython3_EXECUTABLE={sys.executable}'],'configure')
run([CM/'CMake/bin/cmake.exe','--build',R/'build','--parallel','2','--target','expert_cache_elastic_test' if '--unit-only' in sys.argv else 'strata'],'build-unit-final' if '--unit-only' in sys.argv else ('build-baseline' if '--baseline' in sys.argv else 'build-candidate'))
if '--unit-only' in sys.argv:
    import json
    cfg=json.loads(Path(r'D:\ds4_work\strata\Strata\strata-2080ti-sota.json').read_text(encoding='utf-8-sig'))
    env['PATH']=os.pathsep.join(cfg.get('lib_dirs',[])+[env.get('PATH','')])
    run([CM/'CMake/bin/ctest.exe','--test-dir',R/'build','--output-on-failure','-V','-R','^expert_cache_elastic_test$'],'ctest-elastic-final')
if '--baseline' in sys.argv:
    import shutil,hashlib,json
    exe=R/'strata-baseline.exe'; assert not exe.exists(); shutil.copy2(R/'build/strata.exe',exe)
    (R/'baseline-build.json').write_text(json.dumps({'exe':str(exe),'sha256':hashlib.sha256(exe.read_bytes()).hexdigest()},indent=2))

