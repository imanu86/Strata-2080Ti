"""Export pinned upstream and transplant only the Daily elastic cache delta."""
from pathlib import Path
import subprocess,re,json,zipfile,hashlib,difflib
BASE=Path(__file__).resolve().parent
OLD=BASE/'checkpoint038_20261003';R=BASE/'elastic038_20261003'
FORK=Path(r'C:\Users\imanu\source\repos\Strata-2080Ti')
PIN='99f3dbd0b21d1401b3769e0c0d963913607f380b'
assert subprocess.check_output(['git','branch','--show-current'],cwd=FORK,text=True).strip()=='main'
print(subprocess.check_output(['git','status','--short','--branch'],cwd=FORK,text=True))
R.mkdir(exist_ok=False);S=R/'source';S.mkdir()
z=zipfile.ZipFile(OLD/'upstream.zip');prefix=z.namelist()[0].split('/')[0]+'/'
original={}
for name in z.namelist():
    if name.endswith('/') or not name.startswith(prefix):continue
    rel=name[len(prefix):];p=Path(rel)
    assert not p.is_absolute() and '..' not in p.parts
    if rel.startswith('.github/workflows/'):continue
    data=z.read(name);original[rel]=data
    target=S/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
files=['include/strata/core/expert_cache.hpp','src/core/expert_cache.cpp','include/strata/prefill/prefill.hpp','src/prefill/prefill.cpp','src/program/generate.cpp']
patch='';selected={}
for rel in files:
    diff=subprocess.check_output(['git','diff','-U3','36fa455e579b23a9c909c2c6fe1bddd9e51cb8ca','--',rel],cwd=FORK).decode('utf-8')
    blocks=re.split(r'(?=^@@ )',diff,flags=re.M)
    keep=[]
    for h in blocks[1:]:
        n=int(re.match(r'@@ -(\d+)',h)[1])
        changes='\n'.join(l for l in h.splitlines() if l.startswith(('+','-')))
        use=(rel.endswith('expert_cache.hpp') or rel.endswith('expert_cache.cpp') or
             (rel.endswith('prefill.hpp') and 'chunk_max' in changes) or
             (rel.endswith('prefill.cpp') and 576<=n<=685) or
             (rel.endswith('generate.cpp') and any(t in changes for t in ['elastic','el_core','sp_chunk_max','fills still landing in the tail'])))
        if use:keep.append(h)
    selected[rel]=len(keep);patch+=blocks[0]+''.join(keep)
(R/'daily-elastic-selected.patch').write_text(patch,encoding='utf-8',newline='\n')
# Keep rejected hunks for explicit adaptation; never silently drop a port conflict.
p=subprocess.run(['git','apply','--reject','--whitespace=nowarn',str(R/'daily-elastic-selected.patch')],cwd=S,capture_output=True,text=True)
(R/'transplant.log').write_text(p.stdout+p.stderr,encoding='utf-8')
meta={'upstream':PIN,'source_export':True,'fork_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=FORK,text=True).strip(),'selected_hunks':selected,'transplant_exit':p.returncode,'rejects':[str(x.relative_to(S)) for x in S.rglob('*.rej')],'checkpoint_excluded':True,'sm75_kernel_changes_excluded':True,'daily_unchanged':True}
(R/'preparation.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2));print(p.stderr)
