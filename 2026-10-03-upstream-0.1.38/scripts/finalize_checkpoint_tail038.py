from pathlib import Path
import zipfile,json,difflib,hashlib,subprocess
BASE=Path(__file__).resolve().parent;R=BASE/'checkpoint-tail038_20261003';S=R/'source'
z=zipfile.ZipFile(BASE/'checkpoint038_20261003/upstream.zip');prefix=z.namelist()[0].split('/')[0]+'/'
patch='';files={};check=R/'apply-check';check.mkdir(exist_ok=False)
for rel in ['src/program/generate.cpp','docs/PROMPT_CACHE_TAIL.md']:
    present=prefix+rel in z.namelist();a=z.read(prefix+rel).decode('utf-8') if present else '';b=(S/rel).read_text(encoding='utf-8')
    patch+=''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='a/'+rel if present else '/dev/null',tofile='b/'+rel))
    if present:
        p=check/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(a,encoding='utf-8',newline='\n')
    files[rel]=hashlib.sha256((S/rel).read_bytes()).hexdigest()
(R/'checkpoint-tail.patch').write_text(patch,encoding='utf-8',newline='\n')
p=subprocess.run(['git','-c','core.autocrlf=false','apply',str(R/'checkpoint-tail.patch')],cwd=check,capture_output=True,text=True);assert p.returncode==0,p.stderr
for rel,want in files.items():assert hashlib.sha256((check/rel).read_bytes()).hexdigest()==want,rel
meta=json.loads((R/'preparation.json').read_text());meta.update(patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),files=files,pristine_apply_check=True)
(R/'preparation.json').write_text(json.dumps(meta,indent=2))
print('Tail shipping patch applies to pristine upstream; resulting hashes match')
