"""Elastic-cache PR (fork branch contrib/elastic-core-0139, worktree C:\\Users\\imanu\\source\\repos\\Strata-elastic-pr):
build the pure upstream 0.1.39 baseline (6f32ec0) and the PR head in the same build folder, run the cache CTests on
the PR head, and stage both executables for the benches (20261004_routed_keep/build-up0139, build-elasticpr).

Same configuration as the Daily's shipping builds (Release, sm_75, CUDA 12.6 shared runtime, native experts, tests on,
llama.cpp 3cf03257).
"""
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

B = Path(__file__).resolve().parent
W = Path(r'C:\Users\imanu\source\repos\Strata-elastic-pr')
BUILD = B / 'build'
K = B.parent / '20261004_routed_keep'
D = Path(r'D:\ds4_work\strata\Strata')
VC = Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC')
CM = Path(r'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake')
CUDA = Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6')
BRANCH = 'contrib/elastic-core-0139'
UPSTREAM = '6f32ec070f23ced9f50e704d854d775da52591ab'

env = {k: v for k, v in os.environ.items() if not k.upper().startswith('STRATA_')}
p = subprocess.run(f'cmd.exe /d /s /c "call "{VC / "Auxiliary/Build/vcvars64.bat"}" >nul && set"', capture_output=True,
                   text=True, encoding='cp1252', errors='replace', env=env, creationflags=subprocess.CREATE_NO_WINDOW)
assert p.returncode == 0
for line in p.stdout.splitlines():
    if '=' in line and not line.startswith('='):
        k, v = line.split('=', 1)
        env[k] = v
env = {k: v for k, v in env.items() if not k.upper().startswith('STRATA_')}
env['PATH'] = str(CUDA / 'bin') + os.pathsep + env.get('PATH', '')
steps = {}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=W, text=True, encoding='utf-8', errors='replace').strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(cmd, name, timeout=5400):
    print(time.strftime('%H:%M:%S'), name, flush=True)
    t0 = time.monotonic()
    with (B / (name + '.log')).open('w', encoding='utf-8') as f:
        q = subprocess.run([str(x) for x in cmd], cwd=B, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                           creationflags=subprocess.CREATE_NO_WINDOW)
    steps[name] = dict(rc=q.returncode, seconds=round(time.monotonic() - t0, 1))
    if q.returncode:
        print((B / (name + '.log')).read_text(encoding='utf-8', errors='replace')[-6000:])
    assert q.returncode == 0, name


def stage(name, note):
    dst = K / name
    dst.mkdir(exist_ok=True)
    shutil.copy2(BUILD / 'strata.exe', dst / 'strata.exe')
    (dst / 'build.json').write_text(json.dumps(dict(status='completed', executable_sha256=sha(dst / 'strata.exe'), note=note),
                                               indent=2) + '\n', encoding='utf-8')
    return sha(dst / 'strata.exe')


assert not git('status', '--porcelain'), 'worktree not clean'
head = git('rev-parse', BRANCH)
tests = ['expert_cache_elastic_test', 'expert_cache_per_layer_test', 'expert_profile_save_test', 'expert_cache_segmented_test']
run([CM / 'CMake/bin/cmake.exe', '-S', W, '-B', BUILD, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
     f'-DCMAKE_MAKE_PROGRAM={CM / "Ninja/ninja.exe"}', f'-DCMAKE_CUDA_COMPILER={CUDA / "bin/nvcc.exe"}',
     '-DCMAKE_CUDA_ARCHITECTURES=75', '-DCMAKE_CUDA_FLAGS=-D_WINDOWS -Xcompiler=/EHsc', '-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared',
     '-DSTRATA_ENABLE_CUDA=ON', '-DSTRATA_ENABLE_HIP=OFF', '-DSTRATA_BUILD_TESTS=ON', '-DSTRATA_NATIVE_EXPERTS=ON',
     f'-DSTRATA_GGML_DIR={(D / "third_party/llama.cpp").as_posix()}', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
     f'-DPython3_EXECUTABLE={sys.executable}'], 'configure')
# 1. pure upstream 0.1.39 (the worktree detached at 6f32ec0 for the build only, then back on the branch)
git('checkout', '-q', '--detach', UPSTREAM)
try:
    run([CM / 'CMake/bin/cmake.exe', '--build', BUILD, '--parallel', '8', '--target', 'strata'], 'build-upstream')
    up_sha = stage('build-up0139', f'pure upstream 0.1.39 {UPSTREAM[:8]}, build of 20261006_elastic_pr')
finally:
    git('checkout', '-q', BRANCH)
# 2. the PR head, its tests
run([CM / 'CMake/bin/cmake.exe', '-S', W, '-B', BUILD], 'reconfigure')
run([CM / 'CMake/bin/cmake.exe', '--build', BUILD, '--parallel', '8', '--target', 'strata', *tests], 'build-pr')
pr_sha = stage('build-elasticpr', f'{BRANCH} {head[:8]} (upstream 0.1.39 + --elastic), build of 20261006_elastic_pr')
run([CM / 'CMake/bin/ctest.exe', '--test-dir', BUILD, '--output-on-failure', '-R', '^(' + '|'.join(tests) + ')$'], 'ctest')
(B / 'build.json').write_text(json.dumps(dict(pr_head=head, upstream=UPSTREAM, upstream_exe_sha256=up_sha, pr_exe_sha256=pr_sha,
                                              steps=steps), indent=2) + '\n', encoding='utf-8')
print('READY', head[:8], 'upstream', up_sha[:12], 'pr', pr_sha[:12], flush=True)
