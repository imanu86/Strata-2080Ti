from pathlib import Path
import re,json,zipfile,difflib,hashlib
BASE=Path(__file__).resolve().parent;R=BASE/'elastic038_20261003';S=R/'source'
p=S/'src/program/generate.cpp';s=p.read_text(encoding='utf-8')
rej=p.with_suffix('.cpp.rej').read_text(encoding='utf-8')
blocks=re.split(r'(?=^@@ )',rej,flags=re.M)[1:]
for i,needle in [(0,'    int vram_reserve_mib = 700;'),(1,'        else if (a == "--vram-reserve-mib") {'),(2,'        // #477: write the learned profile (between requests and at QUIT:')]:
    add='\n'.join(l[1:] for l in blocks[i].splitlines() if l.startswith('+'))+'\n'
    at=s.index(needle)
    if i<2:at=s.index('\n',at)+1
    s=s[:at]+add+s[at:]
# Upstream changed the serve candidate predicate; the context-only port landed
# the old victim hunk in CLI. Put it in serve and leave CLI unmodified.
changed='else if (r[e] >= el_core) vict.emplace_back(u[e], e);   // fork: the core never leaves'
assert s.count(changed)==1
s=s.replace(changed,'else vict.emplace_back(u[e], e);')
pos=s.index('        auto adapt = [&]() -> bool {')
vict=s.index('else vict.emplace_back(u[e], e);',pos)
s=s[:vict]+s[vict:].replace('else vict.emplace_back(u[e], e);',changed,1)
# CUDA VMM is opt-in for serve, one local GPU. Existing other modes keep fixed allocation.
s=s.replace('if (!sized_slots.empty() && o.elastic) {','if (!sized_slots.empty() && o.elastic && o.serve && !multi_gpu) {')
p.write_text(s,encoding='utf-8',newline='\n')
p=S/'src/core/expert_cache.cpp';s=p.read_text(encoding='utf-8')
s=s.replace('#include <cuda.h>','#if !defined(STRATA_USE_HIP)\n#include <cuda.h>\n#endif')
begin=s.index('namespace {\nbool drv_ok(');end=s.index('bool ExpertCache::grow(',begin)
stub='''#else
bool ExpertCache::map_to(uint64_t, std::string& err) { err = "elastic cache requires CUDA VMM"; return false; }
void ExpertCache::unmap_above(uint64_t) {}
bool ExpertCache::open_sized_elastic(const std::vector<int64_t>&, int64_t, int64_t, uint64_t, uint64_t, std::string& err) {
    close(); err = "elastic cache requires CUDA VMM; using fixed cache"; return false;
}
#endif

'''
s=s[:begin]+'#if !defined(STRATA_USE_HIP)\n'+s[begin:end]+stub+s[end:]
s=s.replace('        if (va_ != 0) cuMemAddressFree((CUdeviceptr) va_, va_bytes_);','#if !defined(STRATA_USE_HIP)\n        if (va_ != 0) cuMemAddressFree((CUdeviceptr) va_, va_bytes_);\n#endif')
p.write_text(s,encoding='utf-8',newline='\n')
p=S/'CMakeLists.txt';s=p.read_text(encoding='utf-8');needle='  target_link_libraries(strata_engine PUBLIC strata_core strata_kernels strata_kernels_cpu ${_strata_gpu_runtime_target})'
assert s.count(needle)==1
s=s.replace(needle,needle+'\n  if(STRATA_ENABLE_CUDA)\n    target_link_libraries(strata_engine PUBLIC CUDA::cuda_driver)\n  endif()')
p.write_text(s,encoding='utf-8',newline='\n')
z=zipfile.ZipFile(BASE/'checkpoint038_20261003/upstream.zip');prefix=z.namelist()[0].split('/')[0]+'/'
files=json.loads((R/'preparation.json').read_text())['selected_hunks'];files=list(files)+['CMakeLists.txt']
patch='';hashes={}
for rel in files:
    a=z.read(prefix+rel).decode('utf-8');b=(S/rel).read_text(encoding='utf-8')
    patch+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
    hashes[rel]=hashlib.sha256((S/rel).read_bytes()).hexdigest()
(R/'elastic.patch').write_text(patch,encoding='utf-8',newline='\n')
(R/'adaptation.json').write_text(json.dumps({'files':hashes,'scope':'Daily elastic cache only; CUDA guard/link and upstream038 context adaptation; no checkpoint/topk/sm75 kernel port','upstream':'99f3dbd0b21d1401b3769e0c0d963913607f380b','original_author':'Claude Opus 5.5','porter':'Codex','rejected_original_hunks_retained':True},indent=2),encoding='utf-8')
assert 'align_every' not in (S/'include/strata/prefill/prefill.hpp').read_text()
assert '+                if (const char* fl = std::getenv("STRATA_DUMP_FIRST_LOGITS"))' not in patch
print('Elastic-only patch prepared',len(patch),'bytes',len(files),'files')
