from pathlib import Path
import json,zipfile,difflib,hashlib,sys
BASE=Path(__file__).resolve().parent; R=BASE/'elastic038_20261003'; S=R/'source'
if '--adapt-token-buffer' in sys.argv:
    p=S/'src/prefill/prefill.cpp'; s=p.read_text(encoding='utf-8')
    # 0.1.38 batched embeddings introduced independent token buffers. They must
    # cover the largest future relayout, just like the other one-time buffers.
    assert s.count('(size_t) chunk * sizeof(int32_t)')==1
    s=s.replace('(size_t) chunk * sizeof(int32_t)', '(size_t) cap_T * sizeof(int32_t)')
    s=s.replace('m.tok_host.resize((size_t) chunk);','m.tok_host.resize((size_t) cap_T);')
    p.write_text(s,encoding='utf-8',newline='\n')
    p=S/'src/program/generate.cpp';s=p.read_text(encoding='utf-8')
    needle='if (!sized_slots.empty() && o.elastic && o.serve && !multi_gpu) {'
    assert s.count(needle)==1
    s=s.replace(needle,needle.replace('!multi_gpu','!multi_gpu && !o.resident_cpu_experts'))
    needle='        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL")'
    at=s.index(needle)
    s=s[:at]+'''        if (o.elastic && (!o.serve || multi_gpu || o.resident_cpu_experts))
            std::fprintf(stderr, "strata: elastic cache requires single-GPU serve without resident CPU complement; using fixed cache\\n");
'''+s[at:]
    p.write_text(s,encoding='utf-8',newline='\n')
z=zipfile.ZipFile(BASE/'checkpoint038_20261003/upstream.zip');prefix=z.namelist()[0].split('/')[0]+'/'
meta=json.loads((R/'adaptation.json').read_text()); files=list(meta['files'])
if 'tests/core/expert_cache_elastic_test.cpp' not in files: files.append('tests/core/expert_cache_elastic_test.cpp')
if (S/'docs/ELASTIC_CACHE.md').exists() and 'docs/ELASTIC_CACHE.md' not in files: files.append('docs/ELASTIC_CACHE.md')
patch=''
if '--normalize-lf' in sys.argv: meta['pre_lf_file_hashes']=dict(meta['files'])
for rel in files:
    a=z.read(prefix+rel).decode('utf-8') if prefix+rel in z.namelist() else '';b=(S/rel).read_text(encoding='utf-8')
    if '--normalize-lf' in sys.argv: (S/rel).write_text(b,encoding='utf-8',newline='\n')
    patch+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+rel if prefix+rel in z.namelist() else '/dev/null',tofile='b/'+rel))
    meta['files'][rel]=hashlib.sha256((S/rel).read_bytes()).hexdigest()
assert 'align_every' not in (S/'include/strata/prefill/prefill.hpp').read_text()
assert not any('STRATA_DUMP_FIRST_LOGITS' in l for l in patch.splitlines() if l.startswith('+'))
meta['upstream_adaptations']=['HIP compile guard (not hardware tested)','CUDA driver link','resident CPU complement falls back to fixed cache','batched-embedding token buffers cover chunk_max']
meta['original_elastic_commits']=['b8a51dbc413d99f1c04c151f1bb16e238c84af10','50a37fa0bcf73c2dcdbe4fbe0f639fa50b23ce59','16755a67abfa84b5d2b76501b7260526db62078a','8966cafb1287598d4f63028b5345f66c85387dd6']
meta['original_coauthor']='Claude Opus 5.5 <noreply@anthropic.com>'
(R/'elastic.patch').write_text(patch,encoding='utf-8',newline='\n')
meta['patch_sha256']=hashlib.sha256((R/'elastic.patch').read_bytes()).hexdigest()
(R/'adaptation.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Elastic-only shipping patch:',len(patch),'bytes;',len(files),'files')
